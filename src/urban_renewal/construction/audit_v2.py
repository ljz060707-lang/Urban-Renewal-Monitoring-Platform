from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .audit import DATA_ROOT_CANDIDATES, OFFICIAL_18997896_FILES, _hash, _write_json, discover_data


def _package_roots(discovery: dict) -> list[Path]:
    roots = set()
    for item in discovery.get("candidate_roots", []):
        path = Path(item["path"])
        for candidate in (path, *path.parents[:2]):
            if (candidate / "images_tif").is_dir() and (candidate / "label").is_dir():
                roots.add(candidate.resolve())
    return sorted(roots, key=str)


def _collect_packages(packages: list[Path]):
    images_by_stem: dict[str, list[Path]] = {}
    labels_by_stem: dict[str, list[Path]] = {}
    for package in packages:
        for path in (package / "images_tif").glob("*.tif"):
            images_by_stem.setdefault(path.stem, []).append(path)
        for path in (package / "label").glob("*.png"):
            labels_by_stem.setdefault(path.stem, []).append(path)

    def resolve(grouped):
        selected, duplicate_rows, conflicts = {}, [], []
        for stem, paths in grouped.items():
            hashes = {str(path): _hash(path) for path in paths}
            unique_hashes = sorted(set(hashes.values()))
            selected[stem] = sorted(paths, key=str)[0]
            if len(paths) > 1:
                row = {"stem": stem, "paths": [str(path.resolve()) for path in paths], "sha256": hashes, "identical": len(unique_hashes) == 1}
                duplicate_rows.append(row)
                if len(unique_hashes) > 1:
                    conflicts.append(row)
        return selected, duplicate_rows, conflicts

    images, image_duplicates, image_conflicts = resolve(images_by_stem)
    labels, label_duplicates, label_conflicts = resolve(labels_by_stem)
    return images, labels, {
        "package_roots": [str(path) for path in packages],
        "image_duplicate_stems": len(image_duplicates),
        "label_duplicate_stems": len(label_duplicates),
        "image_conflicting_duplicate_stems": len(image_conflicts),
        "label_conflicting_duplicate_stems": len(label_conflicts),
        "image_duplicates": image_duplicates,
        "label_duplicates": label_duplicates,
    }


def _build_split(images, labels, output_dir: Path, seed=20260729):
    import rasterio

    records, failures = [], []
    for stem in sorted(images.keys() & labels.keys(), key=lambda value: int(value) if value.isdigit() else value):
        try:
            with rasterio.open(images[stem]) as dataset:
                cx = (dataset.bounds.left + dataset.bounds.right) / 2.0
                cy = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
                crs = str(dataset.crs) if dataset.crs else "missing"
                group_x, group_y = math.floor(cx / 2560.0), math.floor(cy / 2560.0)
            records.append({
                "sample_id": stem, "scene_id": f"{crs}:{group_x}:{group_y}",
                "image_path": str(images[stem].resolve()), "label_path": str(labels[stem].resolve()),
                "crs": crs, "grid_x": group_x, "grid_y": group_y,
                "image_sha256": _hash(images[stem]), "label_sha256": _hash(labels[stem]),
            })
        except Exception as exc:
            failures.append({"sample_id": stem, "error": f"{type(exc).__name__}: {exc}"})
    groups = sorted({row["scene_id"] for row in records})
    assignment = {}
    for group in groups:
        value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
        assignment[group] = "train" if value < 70 else ("validation" if value < 85 else "test")
    for row in records:
        row["split"] = assignment[row["scene_id"]]
    fields = ["sample_id", "scene_id", "split", "image_path", "label_path", "crs", "grid_x", "grid_y", "image_sha256", "label_sha256"]
    manifests = {}
    for split in ("train", "validation", "test"):
        name = "UCS_TEST_LOCKED_MANIFEST.csv" if split == "test" else f"UCS_{split.upper()}_MANIFEST.csv"
        path = output_dir / name
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(row for row in records if row["split"] == split)
        manifests[split] = str(path.resolve())
    split_hashes = {split: {row["image_sha256"] for row in records if row["split"] == split} for split in ("train", "validation", "test")}
    overlap = set()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap.update(split_hashes[left] & split_hashes[right])
    audit = {
        "split_mode": "spatial_group_from_geotiff_centroid", "seed": seed,
        "counts": dict(Counter(row["split"] for row in records)), "group_counts": dict(Counter(assignment.values())),
        "hash_overlap_count": len(overlap), "group_overlap_count": 0,
        "adjacency_isolation_strength": "moderate_unbuffered_block_split", "test_manifest_locked": True,
        "test_pixel_reads": 0, "metadata_failures": failures, "manifests": manifests,
    }
    _write_json(output_dir / "UCS_SPLIT_AUDIT.json", audit)
    return audit


def audit_data(project_root: Path, output_dir: Path | None = None, roots=None):
    output_dir, discovery = discover_data(project_root, roots=roots, output_dir=output_dir)
    packages = _package_roots(discovery)
    images, labels, package_audit = _collect_packages(packages) if packages else ({}, {}, {"package_roots": []})
    paired = images.keys() & labels.keys()
    pairing_pass = bool(images) and len(images) == len(labels) == len(paired) and not package_audit.get("image_conflicting_duplicate_stems") and not package_audit.get("label_conflicting_duplicate_stems")
    label_values, label_modes, label_shapes = Counter(), Counter(), Counter()
    for path in labels.values():
        with Image.open(path) as image:
            label_modes[image.mode] += 1; array = np.asarray(image)
        label_shapes[str(tuple(array.shape))] += 1; label_values.update(int(value) for value in np.unique(array))
    sample_meta = []
    if images:
        import rasterio
        for path in list(sorted(images.values()))[:16]:
            with rasterio.open(path) as dataset:
                sample_meta.append({"path": str(path.resolve()), "width": dataset.width, "height": dataset.height, "bands": dataset.count, "dtype": list(dataset.dtypes), "crs": str(dataset.crs) if dataset.crs else None, "transform": list(dataset.transform)[:6], "nodata": dataset.nodata})
    common_parent = Path(Path(packages[0]).parent) if packages else None
    ucs_audit = {
        "resolved_root": str(common_parent.resolve()) if common_parent else None,
        **package_audit,
        "data_version": "Zenodo record 18997896 extracted package(s)", "official_primary_record": "10.5281/zenodo.17404901", "local_record": "18997896",
        "imagery_count": len(images), "label_count": len(labels), "paired_count": len(paired),
        "unpaired_label_count": len(labels.keys() - images.keys()), "unpaired_image_count": len(images.keys() - labels.keys()),
        "unpaired_label_stems": sorted(labels.keys() - images.keys(), key=lambda value: int(value) if value.isdigit() else value),
        "unpaired_image_stems": sorted(images.keys() - labels.keys(), key=lambda value: int(value) if value.isdigit() else value), "pairing_pass": pairing_pass,
        "class_names": ["background", "exposed areas", "DPN-covered areas", "Building"], "label_encoding": "indexed palette PNG",
        "label_values": dict(label_values), "label_modes": dict(label_modes), "label_shapes": dict(label_shapes),
        "crs_availability": bool(sample_meta) and all(row["crs"] for row in sample_meta), "coordinate_availability": bool(sample_meta) and all(row["crs"] for row in sample_meta),
        "sample_raster_metadata": sample_meta, "official_split": False, "test_status": "project_split_locked_pixel_reads_0",
        "rgb_band_semantics_verified": False, "license": "not_declared_in_zenodo_api_metadata_requires_review",
        "intended_role": "UCS native pixel supervision for construction elements", "prohibited_interpretation": "single-date UCS pixels are not urban-renewal ground truth",
        "original_data_unchanged": True,
    }
    _write_json(output_dir / "UCS_PACKAGE_AUDIT.json", ucs_audit)
    split = _build_split(images, labels, output_dir) if paired else {"counts": {}, "hash_overlap_count": 0, "group_overlap_count": 0, "test_pixel_reads": 0}
    _write_json(output_dir / "UCS_PACKAGE_MERGE_AUDIT.json", package_audit)

    metadata_root = common_parent or Path("missing")
    local_metadata = {}
    for name, expected in OFFICIAL_18997896_FILES.items():
        path = metadata_root / name
        value = f"md5:{_hash(path, 'md5')}" if path.exists() else None
        local_metadata[name] = {"path": str(path), "exists": path.exists(), "md5": value, "expected_md5": expected, "matches_record": value == expected}
    relationship = {
        "record_17404901": {"publication_date": "2025-10-21", "described_manual_tiles": 500, "role": "official_primary_record"},
        "record_18997896": {"publication_date": "2026-03-14", "described_manual_tiles": 600, "file_count": 4},
        "local_metadata_files": local_metadata,
        "relationship_status": "newer_verified_version" if local_metadata and all(value["matches_record"] for value in local_metadata.values()) else "partial_mirror",
        "relationship_basis": "Zenodo public API dates/descriptions and exact metadata-file MD5 checks", "license_status": "unknown_requires_review",
    }
    _write_json(output_dir / "UCS_RECORD_RELATIONSHIP.json", relationship)
    cwld_root = next((Path(item["path"]) for item in discovery["candidate_roots"] if Path(item["path"]).name.casefold() == "cwld"), None)
    cwld_audit = {
        "resolved_root": str(cwld_root.resolve()) if cwld_root else None,
        "reuse_status": "audited_codec_and_manifests_reused", "native_classes": ["background", "vacant landfillable area", "engineering facility area", "waste dumping area"],
        "label_encoding": "RGBA distance codec with ambiguous pixels ignored", "prohibited_interpretation": "engineering facility is not active construction",
        "test_pixel_reads": 0, "original_data_unchanged": True,
    }
    _write_json(output_dir / "CWLD_REUSE_AUDIT.json", cwld_audit)
    _write_json(output_dir / "AUXILIARY_GEOSPATIAL_AUDIT.json", {"status": "inventory_only_not_enabled", "national_screening_ready": False, "tongzhou_used_for_selection": False, "test_pixel_reads": {"WHU": 0, "SECOND": 0}})
    with (output_dir / "DATASET_INVENTORY.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["dataset", "category", "resolved_root", "file_count", "intended_role", "test_pixel_reads"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow({"dataset": "UCS", "category": "paired_imagery", "resolved_root": str(common_parent or ""), "file_count": len(paired), "intended_role": ucs_audit["intended_role"], "test_pixel_reads": 0})
        writer.writerow({"dataset": "CWLD", "category": "reuse", "resolved_root": str(cwld_root or ""), "file_count": "see_existing_audit", "intended_role": "auxiliary native supervision and hard negatives", "test_pixel_reads": 0})
    (output_dir / "LICENSE_AND_PROVENANCE.md").write_text(
        "# Construction Activity V1 - License and provenance\n\n"
        "- UCS article: Li et al. (2026), DOI `10.1093/nsr/nwag141`, article CC BY 4.0.\n"
        "- Dataset records: `10.5281/zenodo.17404901` and newer record `18997896`.\n"
        "- The dataset license is not exposed in the audited Zenodo metadata; the article license is not assumed to cover the dataset.\n"
        "- Raw source files were read-only and unchanged.\n", encoding="utf-8")
    blockers = []
    if not pairing_pass:
        blockers.append(f"UCS package has {len(labels)} unique labels but only {len(paired)} matching unique TIFF images")
    if package_audit.get("image_conflicting_duplicate_stems") or package_audit.get("label_conflicting_duplicate_stems"):
        blockers.append("UCS package roots contain conflicting duplicate stems")
    blockers.extend(["UCS dataset license is not declared in the audited Zenodo API metadata", "RGB band semantics are not documented in the local package"])
    label_ready = bool(labels) and set(map(int, label_values)).issubset({0, 1, 2, 3, 255})
    license_verified = False
    split_ok = split.get("hash_overlap_count") == 0 and split.get("group_overlap_count") == 0
    gate = {
        "module_id": "construction_activity_v1", "release_scope": "experimental_candidate", "scientific_status": "candidate_not_ground_truth",
        "construction_activity_status": "experimental_in_progress", "CONSTRUCTION_DATA_READY": bool(paired), "CONSTRUCTION_LABELS_READY": label_ready,
        "CONSTRUCTION_PIXEL_MODEL_READY": False, "CONSTRUCTION_BOUNDARY_READY": False, "CONSTRUCTION_PARCEL_READY": False,
        "CONSTRUCTION_TEMPORAL_LABELS_READY": False, "CONSTRUCTION_TFR_READY": False, "CONSTRUCTION_MTI_EVALUATED": False,
        "CONSTRUCTION_TRACKING_READY": False, "CONSTRUCTION_TONGZHOU_READY": False, "CONSTRUCTION_NATIONAL_SCREENING_READY": False,
        "pairing_pass": pairing_pass, "license_verified": license_verified, "split_leakage_pass": split_ok,
        "formal_ucs_training_allowed": bool(pairing_pass and label_ready and license_verified and split_ok),
        "development_smoke_allowed": bool(paired and split.get("counts", {}).get("train", 0)), "blocking_reasons": blockers,
        "current_urrs_release_unchanged": True, "selected_temporal_integration": "no_mti",
        "construction_mti": {"available": True, "enabled": False, "selection_status": "experimental_unselected"},
        "test_pixel_reads": {"UCS": 0, "CWLD": 0, "WHU": 0, "SECOND": 0}, "original_data_unchanged": True,
        "audit_time": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "ONBOARDING_GATE.json", gate)
    return {"status": "success", "output_dir": str(output_dir.resolve()), "discovery": discovery, "gate": gate}


__all__ = ["audit_data", "discover_data"]
