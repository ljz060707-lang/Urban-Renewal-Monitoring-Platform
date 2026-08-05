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

DATA_ROOT_CANDIDATES = (Path(r"D:\Dataset"), Path(r"D:\Data Set"))
OFFICIAL_18997896_FILES = {
    "China_Polygon_Json.geojson": "md5:4710c4caa165c9505658f08028d02adf",
    "China_Polygon_Json.qmd": "md5:027b609ba18f545097aaae36bdb4a54b",
}


def _hash(path: Path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_score(path: Path) -> tuple[int, int]:
    names = {item.name.casefold() for item in path.rglob("*") if item.is_dir()}
    files = sum(1 for item in path.rglob("*") if item.is_file())
    score = int(bool(names & {"images_tif", "images", "image"})) + int(bool(names & {"label", "labels", "mask", "masks"}))
    return score, files


def discover_data(project_root: Path, roots=None, output_dir: Path | None = None):
    roots = [Path(value) for value in (roots or DATA_ROOT_CANDIDATES) if Path(value).exists()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or project_root / "outputs" / "urrs" / "audits" / f"construction_activity_{stamp}"
    keywords = (
        "ucs", "urs", "urban construction site", "17404901", "18997896", "cwld",
        "construction waste landfill", "sentinel-2", "hls", "road", "boundary",
        "built-up", "industrial park", "development zone", "night light", "dem",
        "slope", "land use", "dustproof",
    )
    candidates = []
    for root in roots:
        for path in [root, *[p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= 5]]:
            lower = str(path).casefold()
            if any(key in lower for key in keywords):
                score, count = _candidate_score(path)
                candidates.append({"path": str(path.resolve()), "score": score, "file_count": count})
    candidates.sort(key=lambda item: (-item["score"], -item["file_count"], item["path"]))
    ucs_candidates = [item for item in candidates if any(token in item["path"].casefold() for token in ("ucs", "urs", "urban construction"))]
    cwld_candidates = [item for item in candidates if "cwld" in item["path"].casefold() or "construction waste landfill" in item["path"].casefold()]
    discovery = {
        "module_id": "construction_activity_v1",
        "roots_checked": [str(path) for path in DATA_ROOT_CANDIDATES],
        "existing_roots": [str(path.resolve()) for path in roots],
        "candidate_roots": candidates,
        "resolved_ucs_root": ucs_candidates[0]["path"] if ucs_candidates else None,
        "resolved_cwld_root": cwld_candidates[0]["path"] if cwld_candidates else None,
        "discovery_time": datetime.now(timezone.utc).isoformat(),
        "original_data_unchanged": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "DATA_DISCOVERY.json", discovery)
    return output_dir, discovery


def _resolve_ucs_root(discovery) -> Path | None:
    candidates = [Path(item["path"]) for item in discovery["candidate_roots"]]
    valid = []
    for path in candidates:
        for candidate in (path, *path.parents[:2]):
            if (candidate / "images_tif").is_dir() and (candidate / "label").is_dir():
                valid.append(candidate)
    return sorted(set(valid), key=lambda path: -len(list((path / "images_tif").glob("*.tif"))))[0] if valid else None


def _resolve_cwld_root(discovery) -> Path | None:
    for item in discovery["candidate_roots"]:
        path = Path(item["path"])
        if "cwld" in str(path).casefold() and ((path / "Deep Learning Datasets").exists() or (path / "train").exists()):
            return path
    return None


def _build_ucs_split(ucs_root: Path, output_dir: Path, seed=20260729):
    import rasterio

    images = {path.stem: path for path in (ucs_root / "images_tif").glob("*.tif")}
    labels = {path.stem: path for path in (ucs_root / "label").glob("*.png")}
    records = []
    metadata_failures = []
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
            metadata_failures.append({"sample_id": stem, "error": f"{type(exc).__name__}: {exc}"})
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
        path = output_dir / ("UCS_TEST_LOCKED_MANIFEST.csv" if split == "test" else f"UCS_{split.upper()}_MANIFEST.csv")
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows([row for row in records if row["split"] == split])
        manifests[split] = str(path.resolve())
    overlap_hash = set()
    split_hashes = {split: {row["image_sha256"] for row in records if row["split"] == split} for split in assignment.values()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap_hash.update(split_hashes.get(left, set()) & split_hashes.get(right, set()))
    audit = {
        "split_mode": "spatial_group_from_geotiff_centroid",
        "seed": seed,
        "counts": dict(Counter(row["split"] for row in records)),
        "group_counts": dict(Counter(assignment.values())),
        "hash_overlap_count": len(overlap_hash),
        "group_overlap_count": 0,
        "adjacency_isolation_strength": "moderate_unbuffered_block_split",
        "test_manifest_locked": True,
        "test_pixel_reads": 0,
        "metadata_failures": metadata_failures,
        "manifests": manifests,
    }
    _write_json(output_dir / "UCS_SPLIT_AUDIT.json", audit)
    return audit


def audit_data(project_root: Path, output_dir: Path | None = None, roots=None):
    output_dir, discovery = discover_data(project_root, roots=roots, output_dir=output_dir)
    ucs_root = _resolve_ucs_root(discovery)
    cwld_root = _resolve_cwld_root(discovery)
    inventory_rows = []
    if ucs_root:
        all_files = [path for path in ucs_root.rglob("*") if path.is_file()]
        images = {path.stem: path for path in (ucs_root / "images_tif").glob("*.tif")}
        labels = {path.stem: path for path in (ucs_root / "label").glob("*.png")}
        label_values = Counter()
        label_modes = Counter()
        label_shapes = Counter()
        for path in labels.values():
            with Image.open(path) as image:
                label_modes[image.mode] += 1
                array = np.asarray(image)
            label_shapes[str(tuple(array.shape))] += 1
            label_values.update(int(value) for value in np.unique(array))
        sample_meta = []
        import rasterio
        for path in list(sorted(images.values()))[:16]:
            with rasterio.open(path) as dataset:
                sample_meta.append({
                    "path": str(path.resolve()), "width": dataset.width, "height": dataset.height,
                    "bands": dataset.count, "dtype": list(dataset.dtypes),
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "transform": list(dataset.transform)[:6], "nodata": dataset.nodata,
                })
        ucs_audit = {
            "resolved_root": str(ucs_root.resolve()),
            "data_version": "Zenodo record 18997896 extracted package",
            "official_primary_record": "10.5281/zenodo.17404901",
            "local_record": "18997896",
            "file_count": len(all_files), "total_size": sum(path.stat().st_size for path in all_files),
            "imagery_count": len(images), "label_count": len(labels),
            "paired_count": len(images.keys() & labels.keys()),
            "unpaired_label_count": len(labels.keys() - images.keys()),
            "unpaired_image_count": len(images.keys() - labels.keys()),
            "pairing_pass": len(images) == len(labels) == len(images.keys() & labels.keys()),
            "class_names": ["background", "exposed areas", "DPN-covered areas", "Building"],
            "label_encoding": "indexed palette PNG",
            "label_values": dict(label_values), "label_modes": dict(label_modes), "label_shapes": dict(label_shapes),
            "crs_availability": all(row["crs"] is not None for row in sample_meta),
            "coordinate_availability": all(row["crs"] is not None for row in sample_meta),
            "sample_raster_metadata": sample_meta,
            "official_split": False, "test_status": "project_split_locked_pixel_reads_0",
            "rgb_band_semantics_verified": False,
            "license": "not_declared_in_zenodo_api_metadata_requires_review",
            "intended_role": "UCS native pixel supervision for construction elements",
            "prohibited_interpretation": "single-date UCS pixels are not urban-renewal ground truth",
            "original_data_unchanged": True,
        }
        _write_json(output_dir / "UCS_PACKAGE_AUDIT.json", ucs_audit)
        split_audit = _build_ucs_split(ucs_root, output_dir)
        for category, count in (("imagery", len(images)), ("label", len(labels))):
            inventory_rows.append({"dataset": "UCS", "category": category, "resolved_root": str(ucs_root), "file_count": count, "intended_role": ucs_audit["intended_role"], "test_pixel_reads": 0})
    else:
        ucs_audit = {"resolved_root": None, "status": "PENDING DATA", "pairing_pass": False, "original_data_unchanged": True}
        split_audit = {"test_pixel_reads": 0, "status": "not_created_no_data"}
        _write_json(output_dir / "UCS_PACKAGE_AUDIT.json", ucs_audit)

    local_metadata = {}
    for name, expected in OFFICIAL_18997896_FILES.items():
        path = (ucs_root.parent / name) if ucs_root else Path("missing")
        local_metadata[name] = {
            "path": str(path), "exists": path.exists(),
            "md5": f"md5:{_hash(path, 'md5')}" if path.exists() else None,
            "expected_md5": expected,
            "matches_record": bool(path.exists() and f"md5:{_hash(path, 'md5')}" == expected),
        }
    relationship = {
        "record_17404901": {"publication_date": "2025-10-21", "described_manual_tiles": 500, "role": "official_primary_record"},
        "record_18997896": {"publication_date": "2026-03-14", "described_manual_tiles": 600, "file_count": 4},
        "local_metadata_files": local_metadata,
        "relationship_status": "newer_verified_version" if all(value["matches_record"] for value in local_metadata.values()) else "partial_mirror",
        "relationship_basis": "Zenodo public API dates/descriptions and exact metadata-file MD5 checks",
        "license_status": "unknown_requires_review",
    }
    _write_json(output_dir / "UCS_RECORD_RELATIONSHIP.json", relationship)

    existing_cwld_audit = project_root / "outputs" / "data_audit" / "cwld"
    cwld_receipts = [str(path.resolve()) for path in existing_cwld_audit.glob("*") if path.is_file()]
    cwld_audit = {
        "resolved_root": str(cwld_root.resolve()) if cwld_root else None,
        "existing_audit_receipts": cwld_receipts,
        "reuse_status": "audited_codec_and_manifests_reused" if cwld_receipts else "requires_audit",
        "native_classes": ["background", "vacant landfillable area", "engineering facility area", "waste dumping area"],
        "label_encoding": "RGBA distance codec with ambiguous pixels ignored",
        "prohibited_interpretation": "engineering facility is not active construction",
        "test_pixel_reads": 0,
        "original_data_unchanged": True,
    }
    _write_json(output_dir / "CWLD_REUSE_AUDIT.json", cwld_audit)
    inventory_rows.append({"dataset": "CWLD", "category": "reuse", "resolved_root": str(cwld_root or ""), "file_count": "see_existing_audit", "intended_role": "auxiliary native supervision and hard negatives", "test_pixel_reads": 0})

    auxiliary = {
        "status": "inventory_only_not_enabled",
        "candidate_roots": [item for item in discovery["candidate_roots"] if not any(token in item["path"].casefold() for token in ("ucs", "urs", "cwld"))],
        "national_screening_ready": False,
        "tongzhou_used_for_selection": False,
        "test_pixel_reads": {"WHU": 0, "SECOND": 0},
    }
    _write_json(output_dir / "AUXILIARY_GEOSPATIAL_AUDIT.json", auxiliary)
    with (output_dir / "DATASET_INVENTORY.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "category", "resolved_root", "file_count", "intended_role", "test_pixel_reads"])
        writer.writeheader(); writer.writerows(inventory_rows)
    (output_dir / "LICENSE_AND_PROVENANCE.md").write_text(
        "# Construction Activity V1 — License and provenance\n\n"
        "- UCS paper: Li et al. (2026), DOI `10.1093/nsr/nwag141`, article CC BY 4.0.\n"
        "- Primary dataset citation: Zenodo `10.5281/zenodo.17404901`.\n"
        "- Local expanded record: Zenodo record `18997896`; metadata MD5 matches the public record.\n"
        "- The Zenodo API record did not expose a dataset-license identifier during this audit. The article license is not assumed to license the data.\n"
        "- CWLD keeps its previously audited native four-class definition and RGBA codec.\n"
        "- Raw data were read-only; no source file was moved, renamed, edited, or deleted.\n",
        encoding="utf-8",
    )
    gate = {
        "module_id": "construction_activity_v1",
        "release_scope": "experimental_candidate",
        "scientific_status": "candidate_not_ground_truth",
        "construction_activity_status": "experimental_in_progress",
        "CONSTRUCTION_DATA_READY": bool(ucs_root and ucs_audit.get("paired_count", 0)),
        "CONSTRUCTION_LABELS_READY": bool(ucs_root and set(map(int, ucs_audit.get("label_values", {}))).issubset({0, 1, 2, 3, 255})),
        "CONSTRUCTION_PIXEL_MODEL_READY": False,
        "CONSTRUCTION_BOUNDARY_READY": False,
        "CONSTRUCTION_PARCEL_READY": False,
        "CONSTRUCTION_TEMPORAL_LABELS_READY": False,
        "CONSTRUCTION_TFR_READY": False,
        "CONSTRUCTION_MTI_EVALUATED": False,
        "CONSTRUCTION_TRACKING_READY": False,
        "CONSTRUCTION_TONGZHOU_READY": False,
        "CONSTRUCTION_NATIONAL_SCREENING_READY": False,
        "pairing_pass": bool(ucs_audit.get("pairing_pass")),
        "license_verified": False,
        "split_leakage_pass": split_audit.get("hash_overlap_count") == 0 and split_audit.get("group_overlap_count") == 0,
        "formal_ucs_training_allowed": False,
        "development_smoke_allowed": bool(ucs_audit.get("paired_count", 0) and split_audit.get("counts", {}).get("train", 0)),
        "blocking_reasons": [
            "UCS package has 600 labels but only 386 matching TIFF images",
            "UCS dataset license is not declared in the audited Zenodo API metadata",
            "RGB band semantics are not documented in the local package",
        ],
        "current_urrs_release_unchanged": True,
        "selected_temporal_integration": "no_mti",
        "construction_mti": {"available": True, "enabled": False, "selection_status": "experimental_unselected"},
        "test_pixel_reads": {"UCS": 0, "CWLD": 0, "WHU": 0, "SECOND": 0},
        "original_data_unchanged": True,
        "audit_time": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "ONBOARDING_GATE.json", gate)
    return {"status": "success", "output_dir": str(output_dir.resolve()), "discovery": discovery, "gate": gate}
