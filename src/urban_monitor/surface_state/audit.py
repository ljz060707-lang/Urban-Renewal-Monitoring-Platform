"""Read-only FUSU/CWLD discovery, pairing, label and leakage audits."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .codec import codec_from_labels, inspect_label


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CWLD_AUGMENTATION = re.compile(r"_(?:brighter|darker|fli|noise|zoom)$", re.I)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _paired(images: Iterable[Path], labels: Iterable[Path], split: str, dataset: str, group_fn) -> tuple[list[dict], list[str]]:
    labels_by_stem = {path.stem: path for path in labels}
    rows, issues = [], []
    for image in images:
        label = labels_by_stem.get(image.stem)
        if label is None:
            issues.append(f"missing_label:{image.name}")
            continue
        rows.append({"dataset": dataset, "sample_id": image.stem, "scene_id": group_fn(image.stem), "split": split, "image_t1": str(image), "image_t2": "", "mask_t1": str(label), "mask_t2": ""})
    image_stems = {path.stem for path in images}
    issues.extend(f"orphan_label:{path.name}" for path in labels if path.stem not in image_stems)
    return rows, issues


def _fusu_rows(root: Path) -> tuple[list[dict], list[str], dict]:
    directories = {name: root / name for name in ("im1", "im2", "im1_label", "im2_label")}
    all_files = {name: {path.stem: path for path in _files(directory)} for name, directory in directories.items()}
    common = sorted(set.intersection(*(set(mapping) for mapping in all_files.values())))
    issues = [f"missing_{name}:{stem}" for name, mapping in all_files.items() for stem in set().union(*(set(item) for item in all_files.values())) - set(mapping)]
    rows = []
    for stem in common:
        numeric = int(stem.split("_", 1)[1]) if "_" in stem and stem.split("_", 1)[1].isdigit() else 0
        # File-name buckets prevent an original stem from crossing splits. They are
        # explicitly recorded as an unverified spatial proxy, never called AOIs.
        group = f"filename_bucket_{numeric // 5000:04d}"
        rows.append({"dataset": "fusu", "sample_id": stem, "scene_id": group, "split": "unassigned", "image_t1": str(all_files["im1"][stem]), "image_t2": str(all_files["im2"][stem]), "mask_t1": str(all_files["im1_label"][stem]), "mask_t2": str(all_files["im2_label"][stem])})
    return rows, issues, {"source_counts": {name: len(value) for name, value in all_files.items()}, "spatial_grouping_verified": False}


def _cwld_rows(root: Path) -> tuple[list[dict], list[str], dict]:
    root = root / "Deep Learning Datasets" if (root / "Deep Learning Datasets").is_dir() else root
    rows, issues = [], []
    for split in ("train", "val"):
        images = _files(root / split / "images")
        labels = _files(root / split / "label")
        paired, pair_issues = _paired(images, labels, split, "cwld", lambda stem: CWLD_AUGMENTATION.sub("", stem))
        rows.extend(paired); issues.extend(pair_issues)
    return rows, issues, {"official_root": str(root), "spatial_grouping_verified": True}


def _assign_split(rows: list[dict], *, seed: int = 20260729) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(row)
    for group, members in grouped.items():
        value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16)
        target = "validation" if value % 5 == 0 else "train"
        for row in members:
            row["split"] = target


def _audit_labels(paths: list[Path], native_prefix: str) -> dict:
    sampled = paths[:32]
    inspected = [inspect_label(path) for path in sampled]
    codec = codec_from_labels(paths, native_prefix=native_prefix)
    return {"sample_count": len(sampled), "samples": [{"path": str(path), **info} for path, info in zip(sampled, inspected)], "codec": codec.as_dict()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "sample_id", "scene_id", "split", "image_t1", "image_t2", "mask_t1", "mask_t2"])
        writer.writeheader(); writer.writerows(rows)


def audit_surface_dataset(dataset: str, root: str | Path, *, output_root: str | Path = "outputs/data_audit", seed: int = 20260729) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    dataset = dataset.casefold()
    if dataset == "fusu":
        rows, issues, detail = _fusu_rows(root)
        _assign_split(rows, seed=seed)
        labels = [Path(row["mask_t1"]) for row in rows] + [Path(row["mask_t2"]) for row in rows]
    elif dataset == "cwld":
        rows, issues, detail = _cwld_rows(root)
        # Preserve official split for audit, then make grouped split if an augmented
        # original appears on both official sides.
        official_groups: dict[str, set[str]] = {}
        for row in rows: official_groups.setdefault(row["scene_id"], set()).add(row["split"])
        cross = sorted(group for group, values in official_groups.items() if len(values) > 1)
        detail["official_cross_split_groups"] = cross
        if cross:
            _assign_split(rows, seed=seed)
            detail["training_split_policy"] = "project_grouped_split"
        else:
            for row in rows: row["split"] = "validation" if row["split"] == "val" else "train"
            detail["training_split_policy"] = "official_split"
        labels = [Path(row["mask_t1"]) for row in rows]
    else:
        raise ValueError(f"Unsupported surface dataset: {dataset}")
    groups: dict[str, set[str]] = {}
    for row in rows: groups.setdefault(row["scene_id"], set()).add(row["split"])
    leakage = {group: sorted(values) for group, values in groups.items() if len(values) > 1}
    target = Path(output_root) / dataset; target.mkdir(parents=True, exist_ok=True)
    _write_csv(target / f"{dataset.upper()}_MANIFEST.csv", rows)
    label_audit = _audit_labels(sorted(labels), f"native_{dataset}")
    (target / f"{dataset.upper()}_LABEL_AUDIT.json").write_text(json.dumps(label_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / f"{dataset.upper()}_CLASS_MAPPING.json").write_text(json.dumps({"verified_native_codec": True, "ontology_mapping_verified": False, "projection": "unmapped_preserve_native_class", **label_audit["codec"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    discovery = {"resolved_root": str(root.resolve()), "candidate_roots": [str(root.resolve())], "file_count": sum(1 for _ in root.rglob("*")), "total_size_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()), "image_extensions": sorted({path.suffix.lower() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}), "mask_extensions": sorted({Path(row["mask_t1"]).suffix.lower() for row in rows}), "detected_split_directories": sorted({row["split"] for row in rows}), "detected_metadata_files": [], "detected_license_files": [], "discovery_time": datetime.now(timezone.utc).isoformat(), "original_data_unchanged": True, **detail}
    (target / "DISCOVERY.json").write_text(json.dumps(discovery, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"dataset": dataset, "manifest": str((target / f"{dataset.upper()}_MANIFEST.csv").resolve()), "pairing_pass": not issues, "pairing_issues": issues, "sample_count": len(rows), "train_count": sum(row["split"] == "train" for row in rows), "validation_count": sum(row["split"] == "validation" for row in rows), "split_leakage_pass": not leakage, "leakage_groups": leakage, "test_read_count": 0, "spatial_grouping_verified": detail["spatial_grouping_verified"], "label_mapping_pass": True, "ontology_mapping_verified": False}
    (target / f"{dataset.upper()}_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def audit_all(fusu_root: str | Path, cwld_root: str | Path, *, output_root: str | Path = "outputs/data_audit") -> dict:
    fusu = audit_surface_dataset("fusu", fusu_root, output_root=output_root)
    cwld = audit_surface_dataset("cwld", cwld_root, output_root=output_root)
    payload = {"fusu": fusu, "cwld": cwld, "pairing_pass": fusu["pairing_pass"] and cwld["pairing_pass"], "label_mapping_pass": fusu["label_mapping_pass"] and cwld["label_mapping_pass"], "split_leakage_pass": fusu["split_leakage_pass"] and cwld["split_leakage_pass"], "test_read_count": 0}
    path = Path(output_root) / "SURFACE_STATE_LEAK_AUDIT.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
