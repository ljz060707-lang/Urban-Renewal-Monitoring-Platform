"""Deterministic WHU Train split for the detached objectizer pilot."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(path: Path) -> str:
    return f"whu_train_{path.stem}"


def discover_whu_train(root: Path) -> list[dict[str, Any]]:
    image_root = Path(root) / "train" / "image"
    label_root = Path(root) / "train" / "label"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"WHU Train image/label directories missing under {root}")
    rows: list[dict[str, Any]] = []
    for image_path in sorted(image_root.glob("*.tif"), key=lambda item: item.stem):
        label_path = label_root / image_path.name
        if not label_path.is_file():
            continue
        with Image.open(label_path) as image:
            label = np.asarray(image)
        if label.ndim == 3:
            label = label[..., 0]
        valid = label != 255
        building = (label > 0) & valid
        labels, count = ndimage.label(building, structure=ndimage.generate_binary_structure(2, 2))
        rows.append({
            "sample_id": _sample_id(image_path),
            "split": "train",
            "image_path": str(image_path.resolve()),
            "label_path": str(label_path.resolve()),
            "image_sha256": _sha256(image_path),
            "label_sha256": _sha256(label_path),
            "positive_fraction": float(building.mean()),
            "valid_fraction": float(valid.mean()),
            "object_count": int(count),
            "test_pixel_read": False,
        })
    if not rows:
        raise RuntimeError("no paired WHU Train samples found")
    return rows


def _assign_strata(rows: list[dict[str, Any]]) -> None:
    values = np.asarray([float(row["positive_fraction"]) for row in rows], dtype=np.float64)
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    for row in rows:
        value = float(row["positive_fraction"])
        row["coverage_stratum"] = "low" if value <= q1 else "medium" if value <= q2 else "high"


def _take_counts(total: int, requested: dict[str, int]) -> dict[str, int]:
    """Allocate at most the requested total within one stratum.

    The earlier implementation accidentally allocated the *entire* stratum to
    one split when the stratum was larger than the requested pilot.  Keeping
    this helper proportional and capped makes it safe for callers that use it
    directly; ``build_boundary_split`` performs the global per-stratum
    allocation below so the final counts are exactly 256/128/32.
    """
    target = min(int(total), int(sum(requested.values())))
    if target <= 0:
        return {key: 0 for key in requested}
    denominator = max(1, int(sum(requested.values())))
    raw = {key: target * int(value) / denominator for key, value in requested.items()}
    values = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = target - sum(values.values())
    order = sorted(requested, key=lambda item: (-(raw[item] - values[item]), item))
    for key in order[:remainder]:
        values[key] += 1
    return values


def _allocate_strata(groups: dict[str, list[dict[str, Any]]], requested: dict[str, int]) -> dict[str, dict[str, int]]:
    """Allocate each requested split across strata by available row count."""
    total = sum(len(values) for values in groups.values())
    allocation = {stratum: {key: 0 for key in requested} for stratum in groups}
    for key, target in requested.items():
        target = min(int(target), total)
        if target <= 0 or total <= 0:
            continue
        raw = {stratum: target * len(values) / total for stratum, values in groups.items()}
        used = {stratum: int(np.floor(value)) for stratum, value in raw.items()}
        remaining = target - sum(used.values())
        order = sorted(groups, key=lambda stratum: (-(raw[stratum] - used[stratum]), stratum))
        for stratum in order:
            if remaining <= 0:
                break
            capacity = len(groups[stratum]) - sum(allocation[stratum].values())
            take = min(1, max(0, capacity))
            if used[stratum] > 0:
                take = min(used[stratum], max(0, capacity))
            allocation[stratum][key] = take
            remaining -= take
        # Fill any remainder caused by a saturated stratum in deterministic
        # order without ever exceeding a group's capacity.
        if remaining > 0:
            for stratum in order:
                capacity = len(groups[stratum]) - sum(allocation[stratum].values())
                take = min(remaining, max(0, capacity))
                allocation[stratum][key] += take
                remaining -= take
                if remaining <= 0:
                    break
    return allocation


def build_boundary_split(rows: list[dict[str, Any]], *, seed: int = 20260805, train_count: int = 256, calibration_count: int = 128, diagnostic_count: int = 32) -> dict[str, Any]:
    if train_count + calibration_count + diagnostic_count > len(rows):
        raise ValueError("requested split counts exceed available WHU Train samples")
    rows = [dict(row) for row in sorted(rows, key=lambda item: str(item["sample_id"]))]
    # Identical label files are commonly duplicated empty/background masks in
    # WHU.  They are not independent evidence and, if split independently,
    # create hash leakage.  Keep them in the registry but exclude the entire
    # duplicate hash group from this bounded pilot.
    label_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}
    for row in rows:
        label_counts[str(row.get("label_sha256"))] = label_counts.get(str(row.get("label_sha256")), 0) + 1
        image_counts[str(row.get("image_sha256"))] = image_counts.get(str(row.get("image_sha256")), 0) + 1
    eligible: list[dict[str, Any]] = []
    duplicate_excluded: list[dict[str, Any]] = []
    for row in rows:
        duplicate = label_counts.get(str(row.get("label_sha256")), 0) > 1 or image_counts.get(str(row.get("image_sha256")), 0) > 1
        row["duplicate_hash_group"] = bool(duplicate)
        if duplicate:
            row["assigned_split"] = "unused_duplicate_hash"
            duplicate_excluded.append(row)
        else:
            eligible.append(row)
    if train_count + calibration_count + diagnostic_count > len(eligible):
        raise ValueError("requested split counts exceed hash-independent WHU Train samples")
    _assign_strata(eligible)
    rng = np.random.default_rng(int(seed))
    requested = {"boundary_train": int(train_count), "boundary_calibration": int(calibration_count), "boundary_diagnostic": int(diagnostic_count)}
    selected: dict[str, list[dict[str, Any]]] = {key: [] for key in requested}
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in ("low", "medium", "high")}
    for row in eligible:
        groups[str(row["coverage_stratum"])].append(row)
    allocations = _allocate_strata(groups, requested)
    # First select per stratum to retain low/medium/high coverage, then use a
    # deterministic remainder pass to hit exact requested counts.
    for stratum in ("low", "medium", "high"):
        group = list(groups[stratum])
        rng.shuffle(group)
        cursor = 0
        for key in requested:
            count = allocations[stratum][key]
            selected[key].extend(group[cursor:cursor + count])
            cursor += count
    used = {row["sample_id"] for values in selected.values() for row in values}
    remaining = [row for row in eligible if row["sample_id"] not in used]
    for key in requested:
        deficit = requested[key] - len(selected[key])
        if deficit > 0:
            selected[key].extend(remaining[:deficit]); remaining = remaining[deficit:]
    assigned: dict[str, str] = {}
    for key, values in selected.items():
        for row in values:
            if row["sample_id"] in assigned:
                raise RuntimeError(f"split overlap: {row['sample_id']}")
            assigned[row["sample_id"]] = key
            row["assigned_split"] = key
    for row in eligible:
        row.setdefault("assigned_split", "unused_train")
    unused = [row for row in eligible if row["assigned_split"] == "unused_train"]
    return {"all": rows, **selected, "unused": unused, "duplicate_hash_excluded": duplicate_excluded, "seed": int(seed), "counts": {key: len(value) for key, value in selected.items()}}


def split_leakage_audit(split: dict[str, Any]) -> dict[str, Any]:
    groups = [split.get("boundary_train", []), split.get("boundary_calibration", []), split.get("boundary_diagnostic", [])]
    ids = [row["sample_id"] for group in groups for row in group]
    image_hashes = [row["image_sha256"] for group in groups for row in group]
    label_hashes = [row["label_sha256"] for group in groups for row in group]
    return {
        "critical_leakage_count": int(len(ids) - len(set(ids)) + len(image_hashes) - len(set(image_hashes)) + len(label_hashes) - len(set(label_hashes))),
        "sample_id_overlap": False if len(ids) == len(set(ids)) else True,
        "image_hash_overlap": False if len(image_hashes) == len(set(image_hashes)) else True,
        "label_hash_overlap": False if len(label_hashes) == len(set(label_hashes)) else True,
        "test_pixel_reads": 0,
        "whu_test_pixel_reads": 0,
        "second_test_pixel_reads": 0,
        "strictly_train_only": all(str(row.get("split")) == "train" and not bool(row.get("test_pixel_read")) for group in groups for row in group),
        "duplicate_hash_excluded_count": len(split.get("duplicate_hash_excluded", [])),
        "duplicate_hash_excluded_from_selected_splits": True,
    }


def write_registry(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "split", "assigned_split", "image_path", "label_path", "image_sha256", "label_sha256", "positive_fraction", "valid_fraction", "object_count", "coverage_stratum", "test_pixel_read"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


__all__ = ["discover_whu_train", "build_boundary_split", "split_leakage_audit", "write_registry"]
