from __future__ import annotations

"""Formal, read-only readiness gate for the UCS-386 available-image subset.

The gate deliberately treats the existing train/validation/locked-test manifests
as release artifacts.  It copies them byte-for-byte into the new audit directory,
but never rewrites, reorders, or re-splits them.  Training/validation pixels may
be inspected for data-quality metrics.  Locked Test is metadata-only: this module
does not call ``read`` on a Test raster and does not load a Test label array.
"""

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .json_io import dump_json, load_json


EXPECTED_COUNTS = {"train": 296, "validation": 44, "test": 46}
EXPECTED_CLASS_NAMES = (
    "ucs_background",
    "ucs_exposed_area",
    "ucs_dustproof_net",
    "ucs_building",
)
VALID_LABEL_VALUES = frozenset({0, 1, 2, 3})
IGNORE_INDEX = 255
DEFAULT_FULL_RECORD_TOTAL = 600
DEFAULT_SPLIT_DIRNAME = "construction_activity_20260802_111842"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    dump_json(path, payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_key(row: Mapping[str, str]) -> str:
    scene = (row.get("scene_id") or "").strip()
    if scene:
        return scene
    crs = (row.get("crs") or "").strip()
    grid_x = (row.get("grid_x") or "").strip()
    grid_y = (row.get("grid_y") or "").strip()
    if crs or grid_x or grid_y:
        return f"{crs}:{grid_x}:{grid_y}"
    return (row.get("sample_id") or "").strip()


def _augmentation_group_key(row: Mapping[str, str]) -> str:
    """Use the stable source stem as a no-augmentation grouping key.

    The current manifests contain no augmentation records.  Keeping this key in
    the audit makes the absence of cross-split augmentation groups explicit and
    prevents a future dataset adapter from silently changing the split contract.
    """

    image_path = Path(row.get("image_path", ""))
    return f"source_stem:{image_path.stem}"


def _resolve_manifest_paths(row: Mapping[str, str]) -> tuple[Path, Path]:
    return Path(row["image_path"]), Path(row["label_path"])


def _manifest_integrity(rows_by_split: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    sample_ids: dict[str, set[str]] = {}
    image_hashes: dict[str, set[str]] = {}
    label_hashes: dict[str, set[str]] = {}
    scene_groups: dict[str, set[str]] = {}
    augmentation_groups: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        sample_ids[split] = {str(row.get("sample_id", "")) for row in rows}
        image_hashes[split] = {str(row.get("image_sha256", "")) for row in rows if row.get("image_sha256")}
        label_hashes[split] = {str(row.get("label_sha256", "")) for row in rows if row.get("label_sha256")}
        scene_groups[split] = {_group_key(row) for row in rows}
        augmentation_groups[split] = {_augmentation_group_key(row) for row in rows}

    def pairwise_overlap(values: Mapping[str, set[str]]) -> dict[str, list[str]]:
        output: dict[str, list[str]] = {}
        names = list(values)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = sorted(values[left] & values[right])
                output[f"{left}__{right}"] = overlap
        return output

    return {
        "sample_id_overlap": pairwise_overlap(sample_ids),
        "image_sha256_overlap": pairwise_overlap(image_hashes),
        "label_sha256_overlap": pairwise_overlap(label_hashes),
        "spatial_group_overlap": pairwise_overlap(scene_groups),
        "augmentation_group_overlap": pairwise_overlap(augmentation_groups),
        "sample_id_unique": len(set().union(*sample_ids.values())) == sum(len(values) for values in sample_ids.values()),
        "image_sha256_unique_across_splits": not any(pairwise_overlap(image_hashes).values()),
        "spatial_groups_disjoint": not any(pairwise_overlap(scene_groups).values()),
        "augmentation_groups_disjoint": not any(pairwise_overlap(augmentation_groups).values()),
    }


def _empty_class_counts() -> dict[str, int]:
    return {str(index): 0 for index in range(4)}


def _empty_image_stats() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "finite_pixel_count": 0,
        "nonzero_pixel_count": 0,
        "read_failures": [],
        "band_stats": {
            str(band): {
                "min": None,
                "max": None,
                "dynamic_range_min_max": None,
                "p02_min": None,
                "p98_max": None,
                "non_constant_sample_count": 0,
            }
            for band in range(1, 4)
        },
    }


def _update_band_stats(stats: dict[str, Any], band: int, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    entry = stats["band_stats"][str(band)]
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    p02 = float(np.percentile(finite, 2.0))
    p98 = float(np.percentile(finite, 98.0))
    entry["min"] = minimum if entry["min"] is None else min(float(entry["min"]), minimum)
    entry["max"] = maximum if entry["max"] is None else max(float(entry["max"]), maximum)
    entry["dynamic_range_min_max"] = float(entry["max"] - entry["min"])
    entry["p02_min"] = p02 if entry["p02_min"] is None else min(float(entry["p02_min"]), p02)
    entry["p98_max"] = p98 if entry["p98_max"] is None else max(float(entry["p98_max"]), p98)
    if maximum > minimum:
        entry["non_constant_sample_count"] += 1


def _audit_train_validation_pixels(
    rows_by_split: Mapping[str, Sequence[Mapping[str, str]]],
    rgb_mapping: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Read only Train/Validation pixels and return coverage plus size audit."""

    import rasterio

    split_stats: dict[str, Any] = {}
    class_coverage: dict[str, Any] = {}
    read_failures = 0
    for split, rows in rows_by_split.items():
        if split == "test":
            continue
        image_stats = _empty_image_stats()
        class_counts = _empty_class_counts()
        raw_value_counts: Counter[str] = Counter()
        label_modes: Counter[str] = Counter()
        palette_count = 0
        blank_label_count = 0
        all_background_count = 0
        size_mismatch_count = 0
        sample_audits: list[dict[str, Any]] = []
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            image_path, label_path = _resolve_manifest_paths(row)
            sample_failure: str | None = None
            try:
                with rasterio.open(image_path) as dataset:
                    if max(rgb_mapping) > dataset.count or min(rgb_mapping) < 1:
                        raise ValueError(f"rgb_mapping={tuple(rgb_mapping)} outside raster band count={dataset.count}")
                    raw = dataset.read(tuple(int(band) for band in rgb_mapping)).astype(np.float32)
                    image_shape = [int(dataset.height), int(dataset.width)]
                    image_stats["sample_count"] += 1
                    finite = np.isfinite(raw)
                    image_stats["finite_pixel_count"] += int(finite.sum())
                    image_stats["nonzero_pixel_count"] += int(np.count_nonzero(np.where(finite, raw, 0)))
                    for offset, values in enumerate(raw, start=1):
                        _update_band_stats(image_stats, offset, values)
                    if not finite.any():
                        raise ValueError("selected RGB bands contain no finite pixels")
            except Exception as exc:  # pragma: no cover - exercised by malformed-data fixtures
                sample_failure = f"image: {type(exc).__name__}: {exc}"
                image_shape = None
            try:
                with Image.open(label_path) as label_image:
                    mode = str(label_image.mode)
                    size = [int(label_image.height), int(label_image.width)]
                    label_modes[mode] += 1
                    if mode == "P" and label_image.getpalette() is not None:
                        palette_count += 1
                    label = np.asarray(label_image)
                    if label.ndim != 2:
                        raise ValueError(f"label mode={mode} is not indexed/grayscale")
                    raw_values, raw_counts = np.unique(label, return_counts=True)
                    for value, count in zip(raw_values.tolist(), raw_counts.tolist()):
                        raw_value_counts[str(int(value))] += int(count)
                    valid = label != IGNORE_INDEX
                    effective = label[valid]
                    for value in sorted(VALID_LABEL_VALUES):
                        class_counts[str(value)] += int(np.count_nonzero(effective == value))
                    if effective.size == 0:
                        blank_label_count += 1
                    elif np.all(effective == 0):
                        all_background_count += 1
                    if image_shape is not None and image_shape != size:
                        size_mismatch_count += 1
                        raise ValueError(f"image_size={image_shape} label_size={size}")
            except Exception as exc:  # pragma: no cover - exercised by malformed-data fixtures
                if sample_failure is None:
                    sample_failure = f"label: {type(exc).__name__}: {exc}"
                else:
                    sample_failure += f"; label: {type(exc).__name__}: {exc}"
                size = None
            if sample_failure is not None:
                read_failures += 1
                image_stats["read_failures"].append({"sample_id": sample_id, "error": sample_failure})
            sample_audits.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "image_size": image_shape,
                    "label_size": size,
                    "read_ok": sample_failure is None,
                }
            )

        effective_pixels = int(sum(class_counts.values()))
        split_stats[split] = {
            "sample_count": len(rows),
            "image": image_stats,
            "label": {
                "mode_counts": dict(label_modes),
                "palette_indexed_count": palette_count,
                "class_counts": class_counts,
                "raw_value_counts": dict(raw_value_counts),
                "effective_pixel_count": effective_pixels,
                "ignore_pixel_count": int(raw_value_counts.get(str(IGNORE_INDEX), 0)),
                "blank_label_count": blank_label_count,
                "all_background_label_count": all_background_count,
                "size_mismatch_count": size_mismatch_count,
            },
            "sample_audits": sample_audits,
        }
        class_coverage[split] = {
            "sample_count": len(rows),
            "effective_pixel_count": effective_pixels,
            "class_counts": class_counts,
            "class_pixel_share": {
                EXPECTED_CLASS_NAMES[index]: (count / effective_pixels if effective_pixels else None)
                for index, count in ((int(key), value) for key, value in class_counts.items())
            },
            "classes_present": [
                EXPECTED_CLASS_NAMES[index] for index, count in ((int(key), value) for key, value in class_counts.items()) if count > 0
            ],
            "non_background_effective_pixel_count": int(sum(class_counts[str(index)] for index in range(1, 4))),
            "blank_label_count": blank_label_count,
            "all_background_label_count": all_background_count,
        }

    return split_stats, class_coverage, read_failures


def _audit_test_metadata(rows: Sequence[Mapping[str, str]], rgb_mapping: Sequence[int]) -> dict[str, Any]:
    """Inspect only headers/stat metadata for Test; never read pixel arrays."""

    import rasterio

    rows_out: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        image_path, label_path = _resolve_manifest_paths(row)
        item: dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": row.get("scene_id"),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "image_sha256_from_manifest": row.get("image_sha256"),
            "label_sha256_from_manifest": row.get("label_sha256"),
            "image_exists": image_path.exists(),
            "label_exists": label_path.exists(),
            "image_size_bytes": image_path.stat().st_size if image_path.exists() else None,
            "label_size_bytes": label_path.stat().st_size if label_path.exists() else None,
            "image_metadata": None,
            "label_metadata": None,
            "pixel_read": False,
        }
        try:
            with rasterio.open(image_path) as dataset:
                item["image_metadata"] = {
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "count": int(dataset.count),
                    "dtypes": list(dataset.dtypes),
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "transform": [float(value) for value in dataset.transform],
                    "selected_rgb_bands_available": min(rgb_mapping) >= 1 and max(rgb_mapping) <= dataset.count,
                }
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": f"image metadata: {type(exc).__name__}: {exc}"})
        try:
            with Image.open(label_path) as label_image:
                item["label_metadata"] = {
                    "width": int(label_image.width),
                    "height": int(label_image.height),
                    "mode": str(label_image.mode),
                    "indexed_palette": str(label_image.mode) == "P",
                }
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": f"label metadata: {type(exc).__name__}: {exc}"})
        rows_out.append(item)
    return {
        "sample_count": len(rows),
        "rows": rows_out,
        "metadata_failures": failures,
        "test_pixel_reads": 0,
        "image_hashes_verified": False,
        "hash_policy": "manifest_hashes_recorded_only; no Test file-byte/pixel read performed",
    }


def _copy_manifest(source: Path, output: Path) -> dict[str, Any]:
    source_sha = _sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    output_sha = _sha256(output)
    return {
        "source": str(source.resolve()),
        "derived_copy": str(output.resolve()),
        "source_sha256": source_sha,
        "derived_copy_sha256": output_sha,
        "byte_for_byte_identical": source_sha == output_sha and source.stat().st_size == output.stat().st_size,
    }


def _load_recovery_status(project_root: Path) -> dict[str, Any] | None:
    candidates = sorted((project_root / "outputs" / "urrs" / "audits").glob("ucs_missing_image_resolution_*/UCS_MISSING_IMAGE_FINAL_STATUS.json"))
    if not candidates:
        return None
    try:
        return load_json(candidates[-1])
    except Exception:
        return {"status": "unreadable_recovery_audit", "path": str(candidates[-1])}


def _write_scope_limitations(output_dir: Path, gate: Mapping[str, Any]) -> None:
    limitations = gate["scope_limitations"]
    lines = [
        "# UCS-386 scope limitations",
        "",
        "This release is an available-image subset gate, not a full UCS release.",
        "",
        f"- dataset_id: `{gate['dataset_id']}`",
        f"- available records: `{limitations['available_records']}/{limitations['full_record_total']}`",
        f"- missing records: `{limitations['missing_records']}`",
        f"- full_ucs_training: `{limitations['full_ucs_training']}`",
        f"- subset_training: `{limitations['subset_training']}`",
        f"- missingness_bias_not_ruled_out: `{limitations['missingness_bias_not_ruled_out']}`",
        f"- national_generalization_verified: `{limitations['national_generalization_verified']}`",
        "",
        "The 214 missing UCS images are recorded as true local absence after the",
        "cross-dataset numeric-stem collision audit. WHU/DSIFN assets are not UCS",
        "images and were not used as recovery inputs. No file under the external",
        "raw-data roots was modified by this gate.",
        "",
        "The existing Train/Validation/Locked Test manifests are preserved exactly",
        "and are copied byte-for-byte into the audit directory. Locked Test is",
        "metadata-only at this stage; no Test pixels were read and no Test metric",
        "was used for selection, early stopping, or threshold tuning.",
        "",
        "The resulting candidate, if trained and frozen, remains a research task",
        "model limited to this UCS-386 available-image subset. It must not be",
        "described as a full UCS, national, production, or ground-truth model.",
    ]
    (output_dir / "UCS386_SCOPE_LIMITATIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_ucs386_subset_gate(
    project_root: str | Path,
    ucs_root: str | Path,
    *,
    split_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    rgb_mapping: Sequence[int] = (3, 2, 1),
    license_status: str = "confirmed_training_allowed",
    license_source: str = "explicit_user_confirmation",
    full_record_total: int = DEFAULT_FULL_RECORD_TOTAL,
) -> dict[str, Any]:
    """Build the formal UCS-386 gate and all required audit receipts."""

    project_root = Path(project_root)
    ucs_root = Path(ucs_root)
    split_dir = Path(split_dir) if split_dir is not None else project_root / "outputs" / "urrs" / "audits" / DEFAULT_SPLIT_DIRNAME
    if output_dir is None:
        output_dir = project_root / "outputs" / "urrs" / "audits" / f"ucs386_subset_readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_mapping = tuple(int(value) for value in rgb_mapping)
    if len(rgb_mapping) != 3 or len(set(rgb_mapping)) != 3 or min(rgb_mapping) < 1:
        raise ValueError(f"RGB mapping must be three distinct one-based bands: {rgb_mapping}")

    manifest_names = {
        "train": "UCS_TRAIN_MANIFEST.csv",
        "validation": "UCS_VALIDATION_MANIFEST.csv",
        "test": "UCS_TEST_LOCKED_MANIFEST.csv",
    }
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    manifest_audit: dict[str, Any] = {}
    manifest_failures: list[str] = []
    for split, filename in manifest_names.items():
        source = split_dir / filename
        if not source.exists():
            manifest_failures.append(f"missing manifest: {source}")
            rows_by_split[split] = []
            continue
        rows_by_split[split] = _rows(source)
        manifest_audit[split] = _copy_manifest(source, output_dir / filename)
        if len(rows_by_split[split]) != EXPECTED_COUNTS[split]:
            manifest_failures.append(f"{split} count={len(rows_by_split[split])}, expected={EXPECTED_COUNTS[split]}")
        if any(str(row.get("split", "")).casefold() != split for row in rows_by_split[split]):
            manifest_failures.append(f"{split} manifest has unexpected split values")

    rows_all = [row for rows in rows_by_split.values() for row in rows]
    path_failures: list[str] = []
    for row in rows_all:
        image_path, label_path = _resolve_manifest_paths(row)
        if not image_path.exists():
            path_failures.append(f"missing image: {image_path}")
        if not label_path.exists():
            path_failures.append(f"missing label: {label_path}")
        try:
            if image_path.resolve().is_relative_to(ucs_root.resolve()) is False:
                path_failures.append(f"image outside UCS root: {image_path}")
            if label_path.resolve().is_relative_to(ucs_root.resolve()) is False:
                path_failures.append(f"label outside UCS root: {label_path}")
        except AttributeError:  # pragma: no cover - Python 3.8 compatibility fallback
            pass

    integrity = _manifest_integrity(rows_by_split)
    split_audit = {
        "dataset_id": "UCS-386-available-image-subset",
        "split_preserved": True,
        "expected_counts": EXPECTED_COUNTS,
        "observed_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "manifest_files": manifest_audit,
        "integrity": integrity,
        "path_failures": path_failures,
        "manifest_failures": manifest_failures,
        "test_manifest_locked": True,
        "test_pixel_reads": 0,
        "spatial_group_definition": "scene_id, falling back to crs:grid_x:grid_y",
        "augmentation_group_definition": "source image stem; no augmentation rows are present",
    }
    _write_json(output_dir / "UCS386_SPLIT_AUDIT.json", split_audit)

    train_val_stats, class_coverage, train_val_read_failures = _audit_train_validation_pixels(rows_by_split, rgb_mapping)
    test_metadata = _audit_test_metadata(rows_by_split.get("test", []), rgb_mapping)

    combined_class_counts = _empty_class_counts()
    for split in ("train", "validation"):
        for key, value in class_coverage.get(split, {}).get("class_counts", {}).items():
            combined_class_counts[key] += int(value)
    class_coverage_payload = {
        "dataset_id": "UCS-386-available-image-subset",
        "class_names": list(EXPECTED_CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "valid_label_values": sorted(VALID_LABEL_VALUES),
        "train_validation_only": True,
        "per_split": class_coverage,
        "combined_train_validation": {
            "effective_pixel_count": int(sum(combined_class_counts.values())),
            "class_counts": combined_class_counts,
            "classes_present": [EXPECTED_CLASS_NAMES[index] for index, count in ((int(key), value) for key, value in combined_class_counts.items()) if count > 0],
        },
        "raw_label_values": {
            split: train_val_stats.get(split, {}).get("label", {}).get("raw_value_counts", {})
            for split in ("train", "validation")
        },
        "locked_test_pixel_reads": 0,
    }
    _write_json(output_dir / "UCS386_CLASS_COVERAGE.json", class_coverage_payload)

    recovery_status = _load_recovery_status(project_root)
    recovery_ok = bool(recovery_status and recovery_status.get("valid_ucs_recovery_candidates") == 0 and recovery_status.get("remaining_missing_ucs_images") == 214)
    required_checks = {
        "recovery_exclusion_complete": recovery_ok,
        "manifest_counts_exact": not manifest_failures,
        "paths_exist_and_are_under_ucs_root": not path_failures,
        "sample_ids_unique": bool(integrity["sample_id_unique"]),
        "image_hashes_disjoint": bool(integrity["image_sha256_unique_across_splits"]),
        "spatial_groups_disjoint": bool(integrity["spatial_groups_disjoint"]),
        "augmentation_groups_disjoint": bool(integrity["augmentation_groups_disjoint"]),
        "train_validation_read_failures_zero": train_val_read_failures == 0,
        "train_validation_image_label_sizes_match": all(
            train_val_stats.get(split, {}).get("label", {}).get("size_mismatch_count", 1) == 0 for split in ("train", "validation")
        ),
        "indexed_palette_labels": all(
            train_val_stats.get(split, {}).get("label", {}).get("mode_counts", {}).get("P", 0) == len(rows_by_split.get(split, []))
            and train_val_stats.get(split, {}).get("label", {}).get("palette_indexed_count", 0) == len(rows_by_split.get(split, []))
            for split in ("train", "validation")
        ),
        "label_values_within_ontology": all(
            set(train_val_stats.get(split, {}).get("label", {}).get("raw_value_counts", {})).issubset({"0", "1", "2", "3", str(IGNORE_INDEX)})
            for split in ("train", "validation")
        ),
        "all_required_classes_present_in_train": all(
            int(class_coverage.get("train", {}).get("class_counts", {}).get(str(index), 0)) > 0 for index in range(4)
        ),
        "non_background_classes_present_in_validation": all(
            int(class_coverage.get("validation", {}).get("class_counts", {}).get(str(index), 0)) > 0 for index in range(1, 4)
        ),
        "locked_test_metadata_failures_zero": not test_metadata["metadata_failures"],
        "locked_test_pixel_reads_zero": test_metadata["test_pixel_reads"] == 0,
        "rgb_mapping_3_2_1_confirmed": list(rgb_mapping) == [3, 2, 1],
        "license_confirmed_training_allowed": license_status == "confirmed_training_allowed" and license_source == "explicit_user_confirmation",
    }
    gate_status = "FORMAL_SUBSET_TRAINING_ALLOWED" if all(required_checks.values()) else "FORMAL_SUBSET_TRAINING_BLOCKED"

    gate_payload = {
        "status": gate_status,
        "dataset_id": "UCS-386-available-image-subset",
        "scope": {
            "available_image_pairs": 386,
            "train": 296,
            "validation": 44,
            "locked_test": 46,
            "missing_full_record": 214,
        },
        "rgb_mapping": {
            "one_based_bands": list(rgb_mapping),
            "semantic_order": "R,G,B",
            "verified": list(rgb_mapping) == [3, 2, 1],
            "verification_source": "explicit_user_confirmation",
        },
        "ontology": {
            "class_names": list(EXPECTED_CLASS_NAMES),
            "valid_label_values": sorted(VALID_LABEL_VALUES),
            "ignore_index": IGNORE_INDEX,
            "label_encoding": "indexed_palette_png",
        },
        "required_checks": required_checks,
        "train_validation_pixel_audit": train_val_stats,
        "locked_test_inventory_metadata": test_metadata,
        "recovery_exclusion": recovery_status,
        "test_pixel_reads": 0,
        "source_data_modified": 0,
        "scope_limitations": {
            "available_records": 386,
            "full_record_total": int(full_record_total),
            "missing_records": 214,
            "full_record_coverage": f"386/{int(full_record_total)}",
            "full_ucs_training": False,
            "subset_training": True,
            "missingness_bias_not_ruled_out": True,
            "national_generalization_verified": False,
        },
        "license": {
            "status": license_status,
            "source": license_source,
            "training_allowed_for_subset": license_status == "confirmed_training_allowed",
        },
        "preservation": {
            "existing_split_manifests_rewritten": False,
            "derived_manifests_byte_for_byte_copies": True,
            "raw_ucs_inventory_modified": False,
            "whu_dsifn_recovery_used": False,
            "locked_test_used_for_selection": False,
        },
        "audit_directory": str(output_dir.resolve()),
    }
    _write_json(output_dir / "UCS386_SUBSET_GATE.json", gate_payload)

    receipt = {
        "status": "APPROVED_FOR_SUBSET_TRAINING" if gate_status == "FORMAL_SUBSET_TRAINING_ALLOWED" else "NOT_APPROVED",
        "gate_status": gate_status,
        "dataset_id": "UCS-386-available-image-subset",
        "full_ucs_training": False,
        "subset_training": True,
        "full_record_coverage": f"386/{int(full_record_total)}",
        "missingness_bias_not_ruled_out": True,
        "national_generalization_verified": False,
        "license": {
            "status": license_status,
            "source": license_source,
            "explicit_user_confirmation": license_source == "explicit_user_confirmation",
        },
        "split": {"train": 296, "validation": 44, "locked_test": 46},
        "rgb_mapping": [3, 2, 1],
        "ignore_index": IGNORE_INDEX,
        "locked_test_pixel_reads": 0,
        "locked_test_used_for_selection": False,
        "source_data_modified": 0,
        "approved_checks": required_checks,
        "audit_directory": str(output_dir.resolve()),
    }
    _write_json(output_dir / "UCS386_TRAINING_APPROVAL_RECEIPT.json", receipt)

    gate_payload["artifacts"] = {
        "gate": str((output_dir / "UCS386_SUBSET_GATE.json").resolve()),
        "scope_limitations": str((output_dir / "UCS386_SCOPE_LIMITATIONS.md").resolve()),
        "class_coverage": str((output_dir / "UCS386_CLASS_COVERAGE.json").resolve()),
        "split_audit": str((output_dir / "UCS386_SPLIT_AUDIT.json").resolve()),
        "training_approval_receipt": str((output_dir / "UCS386_TRAINING_APPROVAL_RECEIPT.json").resolve()),
    }
    _write_json(output_dir / "UCS386_SUBSET_GATE.json", gate_payload)
    _write_scope_limitations(output_dir, gate_payload)
    return gate_payload
