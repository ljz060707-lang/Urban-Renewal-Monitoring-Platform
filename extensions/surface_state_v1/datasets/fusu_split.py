"""Deterministic FUSU sample split with an explicit weak-isolation fallback."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw

COORDINATE = re.compile(r"^(?:(?:city|tile)_)?(?P<x>\d+)_(?P<y>\d+)$", re.I)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_fusu_sample(
    manifest: str | Path,
    output: str | Path,
    *,
    validation_fraction: float = 0.20,
    seed: int = 20260729,
    fold: int = 0,
    num_folds: int = 5,
) -> dict:
    rows = list(csv.DictReader(Path(manifest).open(encoding="utf-8-sig")))
    if not rows:
        raise ValueError("FUSU manifest is empty")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    parsed = []
    for row in rows:
        match = COORDINATE.match(row["sample_id"])
        x = int(match.group("x")) if match else None
        y = int(match.group("y")) if match else None
        parsed.append({**row, "grid_x": x, "grid_y": y, "coordinate_source": "filename_two_integer_tokens" if match else "unavailable"})
    parse_rate = sum(row["grid_x"] is not None for row in parsed) / len(parsed)

    # The first token is constant in this 500-pair package, so x_y cannot be
    # established as a 2-D grid. Reuse the pre-audited filename buckets as
    # groups and label the isolation strength honestly.
    usable_2d = parse_rate >= 0.95 and len({row["grid_x"] for row in parsed}) > 1
    split_mode = "spatial_block" if usable_2d else "grouped_sample_split"
    groups: dict[str, list[dict]] = {}
    for row in parsed:
        group = (
            f"{row['grid_x'] // 5}_{row['grid_y'] // 5}"
            if usable_2d
            else row.get("scene_id") or row["sample_id"].split("_", 1)[0]
        )
        row["split_group"] = group
        groups.setdefault(group, []).append(row)

    assignments: dict[str, int] = {}
    for group in sorted(groups):
        assignments[group] = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % num_folds
    validation_groups = {group for group, assigned in assignments.items() if assigned == fold}
    if not validation_groups:
        validation_groups = {min(groups, key=lambda group: abs(len(groups[group]) - len(rows) * validation_fraction))}

    train, validation, excluded = [], [], []
    for group, members in groups.items():
        target = validation if group in validation_groups else train
        for row in members:
            row["split"] = "validation" if target is validation else "train"
            target.append(row)

    # With weak grouped isolation, no geometric buffer can be justified.
    fieldnames = list(parsed[0])
    _write_csv(output / "FUSU_SAMPLE_TRAIN_MANIFEST.csv", train, fieldnames)
    _write_csv(output / "FUSU_SAMPLE_VALIDATION_MANIFEST.csv", validation, fieldnames)
    _write_csv(output / "FUSU_SAMPLE_BUFFER_EXCLUDED.csv", excluded, fieldnames)
    fold_rows = [{"split_group": group, "fold": assigned, "sample_count": len(groups[group])} for group, assigned in sorted(assignments.items())]
    _write_csv(output / "FUSU_SAMPLE_5FOLD_ASSIGNMENT.csv", fold_rows, ["split_group", "fold", "sample_count"])

    train_hashes = {_sha256(row["image_t1"]) for row in train}
    val_hashes = {_sha256(row["image_t1"]) for row in validation}
    train_coords = {(row["grid_x"], row["grid_y"]) for row in train}
    val_coords = {(row["grid_x"], row["grid_y"]) for row in validation}
    minimum_distance = None
    if usable_2d and train_coords and val_coords:
        minimum_distance = min(math.dist(left, right) for left in train_coords for right in val_coords)

    canvas = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "FUSU 500-pair sample grouped holdout (not full FUSU validation)", fill="black")
    x = 20
    for group in sorted(groups):
        width = max(10, int(820 * len(groups[group]) / len(rows)))
        color = "#e76f51" if group in validation_groups else "#2a9d8f"
        draw.rectangle((x, 70, min(880, x + width), 150), fill=color)
        draw.text((x, 155), group.replace("filename_bucket_", "b"), fill="black")
        x += width + 2
    canvas.save(output / "FUSU_SAMPLE_SPATIAL_SPLIT.png")

    audit = {
        "dataset_status": "sample_or_partial",
        "training_scope": "current_500_pair_sample",
        "validation_scope": "sample_spatial_holdout",
        "split_mode": split_mode,
        "spatial_isolation_strength": "strong" if usable_2d else "weak",
        "total_pairs": len(rows),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "buffer_excluded_pairs": len(excluded),
        "coordinate_parse_rate": parse_rate,
        "coordinate_interpretation": "two_integer_tokens_not_verified_as_2d_grid" if not usable_2d else "filename_grid_inferred",
        "train_validation_hash_overlap": len(train_hashes & val_hashes),
        "train_validation_coordinate_overlap": len(train_coords & val_coords),
        "minimum_grid_distance": minimum_distance,
        "split_seed": seed,
        "fold": fold,
        "formal_full_dataset_validation": False,
        "sample_spatial_validation": bool(train and validation and not (train_hashes & val_hashes)),
        "full_fusu_validated": False,
        "national_generalization_verified": False,
        "test_read_count": 0,
    }
    (output / "FUSU_SAMPLE_SPATIAL_SPLIT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
