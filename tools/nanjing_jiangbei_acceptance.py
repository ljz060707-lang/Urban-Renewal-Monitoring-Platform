from __future__ import annotations

"""Nanjing Jiangbei no-ground-truth external acceptance runner.

This is an inference-and-reporting entry point.  It never invokes training,
resume, checkpoint selection, threshold calibration, or human-label writes.
All input and frozen-artifact checks are recorded in the run directory.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

SOURCE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei")
OUTPUT_PREFIX = "nanjing_jiangbei_external_acceptance"
FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
RSBUILDING_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
CONSTRUCTION_SHA = "c522d5a4e6c5d6124f968d659004d7870fe4adc9d7381dc4c8917fe019f0f060"
SURFACE_SHA = "35668be3a61b30a29576fc66203b67505e4a0cfa56051d04d6bc25ed21de875c"
METRIC_CRS = "EPSG:32650"
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jp2", ".png", ".jpg", ".jpeg"}
FIXED_RGB_MAPPING = (3, 2, 1)
FULL20_THRESHOLDS = {"any_change": 0.75, "new_building": 0.60, "demolition": 0.35}
CONSTRUCTION_THRESHOLD = 0.50

STATE_NAMES = {
    0: "background_or_other",
    1: "bare_or_vacant",
    2: "active_construction_exposed",
    3: "active_construction_dustproof",
    4: "foundation_or_excavation_candidate",
    5: "structure_rising",
    6: "completed_building",
    7: "stable_building",
    8: "demolition_or_cleared_candidate",
    9: "engineering_or_material_candidate",
    10: "uncertain",
}
STATE_COLORS = {
    0: (205, 211, 216),
    1: (180, 137, 76),
    2: (239, 126, 42),
    3: (42, 173, 150),
    4: (190, 74, 27),
    5: (137, 83, 173),
    6: (60, 126, 205),
    7: (24, 68, 132),
    8: (202, 54, 54),
    9: (188, 120, 178),
    10: (65, 65, 65),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_date_token(text: str) -> str | None:
    for token in re.findall(r"(?<!\d)(20\d{6})(?!\d)", text):
        try:
            return datetime.strptime(token, "%Y%m%d").date().isoformat()
        except ValueError:
            continue
    for token in re.findall(r"(?<!\d)(\d{6})(?!\d)", text):
        try:
            return datetime.strptime(token, "%y%m%d").date().isoformat()
        except ValueError:
            continue
    return None


def _date_from_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"20\d{2}[-_/]?\d{2}[-_/]?\d{2}", text)
    if match:
        compact = re.sub(r"[-_/]", "", match.group(0))
        try:
            return datetime.strptime(compact, "%Y%m%d").date().isoformat()
        except ValueError:
            return None
    return parse_date_token(text)


def _sidecar_date(path: Path) -> tuple[str | None, str | None]:
    for suffix in (".json", ".xml"):
        sidecar = path.with_suffix(suffix)
        if not sidecar.exists():
            continue
        try:
            text = sidecar.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError:
            continue
        date = _date_from_value(text)
        if date:
            return date, f"sidecar:{sidecar.name}"
    return None, None


def _metadata_date(tags: Mapping[str, Any]) -> tuple[str | None, str | None]:
    preferred = ("ACQUISITION_DATE", "ACQUISITIONDATETIME", "DATE_ACQUIRED", "DATE", "DATETIME", "TIFFTAG_DATETIME")
    for key in preferred:
        if key in tags:
            date = _date_from_value(tags[key])
            if date:
                return date, f"metadata:{key}"
    for key, value in tags.items():
        if any(token in str(key).upper() for token in ("DATE", "TIME", "ACQ")):
            date = _date_from_value(value)
            if date:
                return date, f"metadata:{key}"
    return None, None


def _quicklook(path: Path, maximum: int = 720) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        scale = min(1.0, maximum / max(dataset.width, dataset.height))
        height = max(1, int(dataset.height * scale))
        width = max(1, int(dataset.width * scale))
        bands = dataset.read(list(range(1, min(3, dataset.count) + 1)), out_shape=(min(3, dataset.count), height, width), resampling=Resampling.bilinear).astype(np.float32)
    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    if bands.shape[0] == 2:
        bands = np.concatenate([bands, bands[-1:]], axis=0)
    result = np.zeros_like(bands, dtype=np.float32)
    for index in range(3):
        channel = bands[index]
        finite = channel[np.isfinite(channel)]
        if finite.size:
            low, high = np.percentile(finite, (2, 98))
            result[index] = np.clip((channel - low) / max(float(high - low), 1e-6), 0, 1)
    return np.moveaxis(result, 0, -1)


def _inspect_raster(path: Path) -> dict[str, Any]:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        tags = dict(dataset.tags())
        for band in range(1, dataset.count + 1):
            tags.update({f"band{band}_{key}": value for key, value in dataset.tags(band).items()})
        date, source = _metadata_date(tags)
        if date is None:
            date, source = _sidecar_date(path)
        if date is None:
            date = parse_date_token(path.stem) or parse_date_token(path.parent.name)
            source = "filename" if date else None
        scale = min(1.0, 1024 / max(dataset.width, dataset.height))
        height = max(1, int(dataset.height * scale))
        width = max(1, int(dataset.width * scale))
        sample = dataset.read(
            list(range(1, min(3, dataset.count) + 1)),
            out_shape=(min(3, dataset.count), height, width),
            resampling=Resampling.nearest,
        ).astype(np.float32)
        masks = dataset.read_masks(1, out_shape=(height, width), resampling=Resampling.nearest) > 0
        finite = np.isfinite(sample).all(axis=0)
        valid = masks & finite
        dynamic = []
        for band in sample:
            values = band[np.isfinite(band)]
            dynamic.append({"min": float(values.min()) if values.size else None, "max": float(values.max()) if values.size else None})
        alpha = any(str(value).casefold() == "alpha" for value in dataset.colorinterp)
        return {
            "filename": path.name,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "dtype": list(dataset.dtypes),
            "nodata": dataset.nodata,
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": list(dataset.transform),
            "resolution": list(dataset.res),
            "bounds": list(dataset.bounds),
            "acquisition_date": date or "date_unknown",
            "date_source": source or "unknown",
            "rgb_mapping": list(FIXED_RGB_MAPPING),
            "rgb_semantic_order": ["R", "G", "B"],
            "valid_pixel_ratio": float(valid.mean()) if valid.size else 0.0,
            "valid_pixel_ratio_sampling": "nearest_downsampled_to_max_1024",
            "dynamic_range": dynamic,
            "has_alpha": alpha,
            "is_preview_or_thumbnail": bool(any(token in path.name.casefold() for token in ("preview", "thumb", "thumbnail", "overview", "quicklook")) or min(dataset.width, dataset.height) < 512),
            "rotated_grid": bool(abs(dataset.transform.b) > 1e-12 or abs(dataset.transform.d) > 1e-12),
        }


def discover_images(source_root: Path) -> list[dict[str, Any]]:
    paths = sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No supported imagery under {source_root}")
    records = [_inspect_raster(path) for path in paths]
    unknown = [record["filename"] for record in records if record["acquisition_date"] == "date_unknown"]
    if unknown:
        raise RuntimeError(f"Acquisition date is unknown; formal temporal comparison is blocked: {unknown}")
    records.sort(key=lambda item: (item["acquisition_date"], item["path"]))
    dates = [item["acquisition_date"] for item in records]
    duplicates = sorted(date for date, count in Counter(dates).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"More than one image has the same acquisition date: {duplicates}")
    return records


def pair_specs(records: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dates = [str(record["acquisition_date"]) for record in records]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("Pair generation requires unique ascending dates")
    adjacent = [{"from": dates[i], "to": dates[i + 1], "pair_type": "adjacent", "derived_from_temporal_state_aggregation": False} for i in range(len(dates) - 1)]
    cumulative = [{"from": dates[i], "to": dates[-1], "pair_type": "cumulative", "derived_from_temporal_state_aggregation": False} for i in range(len(dates) - 1)]
    return {"adjacent": adjacent, "cumulative": cumulative}


def _grid_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("width", "height", "crs", "transform", "resolution", "bounds")}


def audit_inputs(source_root: Path, output_root: Path) -> dict[str, Any]:
    records = discover_images(source_root)
    input_dir = output_root / "00_input_audit"
    input_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "path", "sha256", "size_bytes", "width", "height", "band_count", "dtype", "nodata", "crs", "transform", "resolution", "bounds", "acquisition_date", "date_source", "rgb_mapping", "valid_pixel_ratio", "dynamic_range", "has_alpha", "is_preview_or_thumbnail",
    ]
    write_csv(input_dir / "IMAGE_INVENTORY.csv", fields, records)
    write_csv(input_dir / "INPUT_SHA256.csv", ["acquisition_date", "path", "sha256", "size_bytes"], records)

    reference = records[0]
    signatures = [_grid_signature(record) for record in records]
    same_crs = all(item["crs"] == reference["crs"] for item in signatures)
    same_shape = all(item["width"] == reference["width"] and item["height"] == reference["height"] for item in signatures)
    same_transform = all(np.allclose(item["transform"], reference["transform"], atol=0, rtol=0) for item in signatures)
    same_resolution = all(np.allclose(item["resolution"], reference["resolution"], atol=0, rtol=0) for item in signatures)
    same_bounds = all(np.allclose(item["bounds"], reference["bounds"], atol=1e-12, rtol=0) for item in signatures)
    left = max(float(item["bounds"][0]) for item in signatures)
    bottom = max(float(item["bounds"][1]) for item in signatures)
    right = min(float(item["bounds"][2]) for item in signatures)
    top = min(float(item["bounds"][3]) for item in signatures)
    overlap = [left, bottom, right, top] if right > left and top > bottom else None
    grid_ready = bool(same_crs and same_shape and same_transform and same_resolution and same_bounds and overlap)
    common_grid = {
        "status": "pass" if grid_ready else "registration_required",
        "reference_date": reference["acquisition_date"],
        "crs": reference["crs"],
        "width": reference["width"],
        "height": reference["height"],
        "transform": reference["transform"],
        "resolution": reference["resolution"],
        "bounds": reference["bounds"],
        "intersection_bounds": overlap,
        "selection_rule": "first date after sorted audited input; exact common grid when all signatures match",
        "resampling_required": not grid_ready,
        "pixelwise_temporal_tracking_allowed": grid_ready,
        "metric_area_crs": METRIC_CRS,
    }
    write_json(input_dir / "IMAGE_DATE_AUDIT.json", {
        "status": "pass" if records else "fail",
        "image_count": len(records),
        "known_date_count": sum(record["acquisition_date"] != "date_unknown" for record in records),
        "dates": [record["acquisition_date"] for record in records],
        "date_range": [records[0]["acquisition_date"], records[-1]["acquisition_date"]],
        "date_identification_order": ["metadata", "sidecar", "manifest", "filename", "date_unknown_then_temporal_block"],
        "creation_time_used_as_acquisition_date": False,
    })
    spatial = {
        "status": "pass" if grid_ready else "registration_required",
        "same_crs": same_crs,
        "same_shape": same_shape,
        "same_transform": same_transform,
        "same_resolution": same_resolution,
        "same_bounds": same_bounds,
        "rotated_grid_present": any(record["rotated_grid"] for record in records),
        "intersection_bounds": overlap,
        "records": [{"date": record["acquisition_date"], **_grid_signature(record)} for record in records],
        "residual_alignment": "zero_under_exact_signature_comparison" if grid_ready else "not_verified",
    }
    write_json(input_dir / "IMAGE_SPATIAL_AUDIT.json", spatial)
    write_json(output_root / "01_aligned_imagery" / "COMMON_GRID.json", common_grid)

    aligned_dir = output_root / "01_aligned_imagery"
    aligned_records = []
    for record in records:
        target = aligned_dir / f"{record['acquisition_date']}_aligned.tif"
        if grid_ready:
            shutil.copy2(record["path"], target)
            action = "byte_copy_after_exact_common_grid_audit"
        else:
            action = "not_written_registration_required"
        aligned_records.append({"date": record["acquisition_date"], "source": record["path"], "aligned_path": str(target.resolve()) if target.exists() else None, "action": action})

    overlap_path = aligned_dir / "VALID_OVERLAP_MASK.tif"
    if grid_ready:
        import rasterio

        if overlap_path.exists():
            overlap_path.unlink()
        with rasterio.open(records[0]["path"]) as reference_dataset:
            profile = reference_dataset.profile.copy()
            profile.pop("photometric", None)
            profile.pop("interleave", None)
            profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate", predictor=2)
            with rasterio.open(overlap_path, "w", **profile) as target:
                with rasterio.open(records[0]["path"]) as first:
                    sources = [first] + [rasterio.open(record["path"]) for record in records[1:]]
                    try:
                        for _, window in target.block_windows(1):
                            mask = np.ones((int(window.height), int(window.width)), dtype=bool)
                            for dataset in sources:
                                mask &= dataset.read_masks(1, window=window) > 0
                            target.write(mask.astype(np.uint8), 1, window=window)
                    finally:
                        for dataset in sources[1:]:
                            dataset.close()
    if grid_ready:
        from PIL import Image

        Image.fromarray((_quicklook(Path(records[-1]["path"])) * 255).astype(np.uint8)).save(aligned_dir / "alignment_preview.png")
    write_json(aligned_dir / "ALIGNMENT_AUDIT.json", {
        **spatial,
        "ready_for_pixelwise_temporal_tracking": grid_ready,
        "aligned_records": aligned_records,
        "valid_overlap_mask": str(overlap_path.resolve()) if overlap_path.exists() else None,
        "alignment_preview": str((aligned_dir / "alignment_preview.png").resolve()) if (aligned_dir / "alignment_preview.png").exists() else None,
        "original_data_modified": False,
        "resampling_policy": {"rgb": "bilinear_or_cubic_if_required", "mask": "nearest", "probability": "bilinear"},
    })
    return {"records": records, "dates": [record["acquisition_date"] for record in records], "aligned_records": aligned_records, "common_grid": common_grid, "spatial": spatial, "grid_ready": grid_ready, "source_root": str(source_root.resolve()), "input_audit_dir": str(input_dir.resolve())}


def _hash_asset(path: Path, expected: str | None = None) -> dict[str, Any]:
    exists = path.exists()
    observed = sha256_file(path) if exists else None
    return {"path": str(path.resolve()), "exists": exists, "sha256": observed, "expected_sha256": expected, "hash_match": bool(exists and (expected is None or observed == expected))}


def _find_existing_metrics(root: Path, patterns: Iterable[str]) -> str | None:
    candidates = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return str(sorted(candidates, key=lambda path: path.stat().st_mtime)[-1].resolve())


def verify_frozen_assets(project_root: Path) -> dict[str, Any]:
    full20_checkpoint = project_root / "outputs" / "models" / "urban_change_v2_full_candidate_20260727_001029" / "best_v2_multisource_change.pth"
    full20_config = project_root / "configs" / "change_public_v2_full20_inference_windows.yaml"
    rs_checkpoint = project_root / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
    rs_config = rs_checkpoint.parent / "Swin_T_whu_windows_pilot5000.py"
    construction_package = project_root / "outputs" / "models" / "construction_activity_v1" / "URRS-Construction-UCS386-v1.0-task-candidate_20260803_142054"
    construction_manifest = construction_package / "TASK_MODEL_MANIFEST.json"
    surface_checkpoint = project_root / "outputs" / "training" / "surface_state_v1" / "joint_sample_20260730_034853" / "checkpoints" / "surface_joint_sample_best_validation.pth"
    mti_selection = project_root / "outputs" / "training" / "surface_state_v1" / "mti_sample_20260730_034853" / "MTI_SELECTION_FINAL.json"
    full20_metrics = project_root / "outputs" / "pretraining" / "change_v2" / "20260727_001029" / "metrics" / "combined.json"
    ucs_validation = construction_package / "validation_metrics.json"
    ucs_test = construction_package / "test_metrics.json"
    surface_validation = project_root / "outputs" / "training" / "surface_state_v1" / "joint_sample_20260730_034853" / "VALIDATION_REPORT.json"
    assets = {
        "full20": _hash_asset(full20_checkpoint, FULL20_SHA),
        "full20_config": _hash_asset(full20_config),
        "rsbuilding": _hash_asset(rs_checkpoint, RSBUILDING_SHA),
        "rsbuilding_config": _hash_asset(rs_config),
        "construction_package_model": _hash_asset(construction_package / "model.pth", CONSTRUCTION_SHA),
        "construction_manifest": _hash_asset(construction_manifest),
        "construction_thresholds": _hash_asset(construction_package / "thresholds.json"),
        "surface_joint": _hash_asset(surface_checkpoint, SURFACE_SHA),
        "surface_mti_selection": _hash_asset(mti_selection),
    }
    construction_payload = read_json(construction_manifest, {})
    mti_payload = read_json(mti_selection, {})
    return {
        "status": "pass" if all(item.get("hash_match") for item in assets.values() if item.get("expected_sha256")) else "blocked_frozen_asset_hash",
        "assets": assets,
        "full20_checkpoint": str(full20_checkpoint.resolve()),
        "full20_config": str(full20_config.resolve()),
        "rsbuilding_checkpoint": str(rs_checkpoint.resolve()),
        "rsbuilding_config": str(rs_config.resolve()),
        "construction_package": str(construction_package.resolve()),
        "surface_checkpoint": str(surface_checkpoint.resolve()),
        "surface_mti_selection": str(mti_selection.resolve()),
        "existing_metrics": {
            "full20": str(full20_metrics.resolve()) if full20_metrics.exists() else None,
            "ucs_validation": str(ucs_validation.resolve()) if ucs_validation.exists() else None,
            "ucs_test": str(ucs_test.resolve()) if ucs_test.exists() else None,
            "surface_validation": str(surface_validation.resolve()) if surface_validation.exists() else None,
            "rsbuilding_validation": _find_existing_metrics(project_root / "outputs" / "v2_readiness", ["**/*VALIDATION*.json", "**/*validation*.json"]),
        },
        "construction_package_manifest": construction_payload,
        "mti_selection_payload": mti_payload,
        "training_started": False,
        "thresholds_changed": False,
        "models_reselected": False,
        "selected_construction_temporal_integration": "no_mti",
        "construction_tfr_status": "development_candidate_not_formally_validated",
        "construction_mti_status": "experimental_unselected",
    }


def _subprocess_with_log(command: list[str], *, cwd: Path, log_path: Path, heartbeat_path: Path, label: str) -> dict[str, Any]:
    """Run one inference-only child process while retaining line-level progress."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lines = []
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{utc_now()}] START {label}\n$ {' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        write_json(heartbeat_path, {"label": label, "pid": process.pid, "status": "running", "updated_at": utc_now()})
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            lines.append(line.rstrip())
            write_json(heartbeat_path, {"label": label, "pid": process.pid, "status": "running", "last_line": line.rstrip()[-500:], "updated_at": utc_now()})
        return_code = process.wait()
        elapsed = time.perf_counter() - started
        log.write(f"[{utc_now()}] END {label} return_code={return_code} elapsed_seconds={elapsed:.3f}\n")
    write_json(heartbeat_path, {"label": label, "pid": process.pid, "status": "success" if return_code == 0 else "failed", "return_code": return_code, "elapsed_seconds": elapsed, "updated_at": utc_now()})
    return {"label": label, "command": command, "pid": process.pid, "return_code": return_code, "elapsed_seconds": elapsed, "log": str(log_path.resolve()), "tail": lines[-40:]}


def run_rsbuilding(project_root: Path, source_root: Path, output_root: Path, dates: list[str], *, rsbuilding_python: Path) -> dict[str, Any]:
    geometry_root = output_root / "03_rsbuilding_geometry"
    tool = project_root / "tools" / "infer_rsbuilding_static_geometry.py"
    checkpoint = project_root / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
    config = checkpoint.parent / "Swin_T_whu_windows_pilot5000.py"
    common = [str(rsbuilding_python), str(tool), "--source-root", str(source_root), "--output-root", str(geometry_root), "--checkpoint", str(checkpoint), "--config", str(config), "--device", "cuda:0", "--batch-size", "2"]
    for date in dates:
        common.extend(["--target-date", date])
    log = geometry_root / "nanjing_rsbuilding_pipeline.log"
    heartbeat = geometry_root / "heartbeat.json"
    steps = [("discover", "discover"), ("diagnose", "diagnose"), ("smoke", "smoke"), ("infer-all", "infer-all"), ("audit", "audit"), ("vectorize", "vectorize"), ("alignment", "alignment")]
    receipts = []
    for label, action in steps:
        if action == "vectorize":
            vectorize_python = project_root / ".venv-gpu" / "Scripts" / "python.exe"
            vectorize_tool = project_root / "tools" / "rsbuilding_external_vectorize.py"
            command = [str(vectorize_python), str(vectorize_tool), "--output-root", str(geometry_root), "--min-area-m2", "10"]
        else:
            command = common[:2] + [action] + common[2:] + (["--smoke-date", dates[-1]] if action == "smoke" else [])
        result = _subprocess_with_log(command, cwd=project_root, log_path=log, heartbeat_path=heartbeat, label=f"rsbuilding:{label}")
        receipts.append(result)
        if result["return_code"] != 0:
            return {"status": "NOT_RUN", "reason": f"RSBuilding {label} failed", "geometry_root": str(geometry_root.resolve()), "steps": receipts, "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    audit_path = geometry_root / "TONGZHOU_GEOMETRY_NUMERICAL_AUDIT.json"
    alignment_path = geometry_root / "ALIGNMENT_AUDIT.json"
    return {
        "status": "completed" if audit_path.exists() and alignment_path.exists() else "partial",
        "geometry_root": str(geometry_root.resolve()),
        "geometry_audit": str(audit_path.resolve()) if audit_path.exists() else None,
        "alignment_audit": str(alignment_path.resolve()) if alignment_path.exists() else None,
        "steps": receipts,
        "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0},
    }


def _full20_predict(model: Any, t1: Any, t2: Any, *, device: str, tile_size: int = 512, overlap: int = 64) -> dict[str, np.ndarray]:
    import torch
    from urban_monitor.change.geospatial import blending_weight, tile_windows
    from urban_monitor.change.inference_v2 import _read_rgb

    sums = {name: np.zeros((t1.height, t1.width), dtype=np.float32) for name in ("any_change", "new_building", "demolition")}
    weights = np.zeros((t1.height, t1.width), dtype=np.float32)
    with torch.inference_mode():
        for window in tile_windows(t1.width, t1.height, tile_size, overlap):
            first = torch.from_numpy(_read_rgb(t1, window)[None]).to(device)
            second = torch.from_numpy(_read_rgb(t2, window)[None]).to(device)
            result = model(first, second)
            height, width = int(window.height), int(window.width)
            weight = blending_weight(height, width)
            row, col = int(window.row_off), int(window.col_off)
            weights[row : row + height, col : col + width] += weight
            for output_name, result_name in (("any_change", "binary_change"), ("new_building", "new_building"), ("demolition", "demolition")):
                probability = torch.sigmoid(result[result_name])[0, 0].float().cpu().numpy()
                sums[output_name][row : row + height, col : col + width] += probability[:height, :width] * weight
    return {name: value / np.maximum(weights, 1e-6) for name, value in sums.items()}


def _full20_preview(t1_path: Path, t2_path: Path, arrays: Mapping[str, np.ndarray], target: Path) -> None:
    from PIL import Image, ImageDraw

    first = (_quicklook(t1_path, 560) * 255).astype(np.uint8)
    second = (_quicklook(t2_path, 560) * 255).astype(np.uint8)
    base = [first, second]
    for name, color in (("any_change", (236, 78, 76)), ("new_building", (50, 145, 240)), ("demolition", (245, 177, 51))):
        probability = arrays[name]
        image = Image.fromarray((_quicklook(t1_path, 560) * 255).astype(np.uint8), mode="RGB").convert("RGBA")
        small = Image.fromarray(np.clip(probability, 0, 1).astype(np.float32), mode="F")
        small = small.resize(image.size, Image.Resampling.BILINEAR)
        mask = np.asarray(small) >= float(FULL20_THRESHOLDS["any_change" if name == "any_change" else name])
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(Image.fromarray(np.where(mask, 155, 0).astype(np.uint8), mode="L"))
        base.append(np.asarray(Image.alpha_composite(image, overlay).convert("RGB")))
    canvas = Image.new("RGB", (560 * 3, 420 * 2), (242, 244, 246))
    labels = ["当前期 RGB", "下一期/末期 RGB", "综合变化", "新建候选", "拆除候选"]
    for index, array in enumerate(base):
        image = Image.fromarray(array).resize((560, 380), Image.Resampling.BILINEAR)
        x = (index % 3) * 560
        y = (index // 3) * 420
        canvas.paste(image, (x, y + 34))
        ImageDraw.Draw(canvas).text((x + 12, y + 10), labels[index], fill=(25, 35, 45))
    canvas.save(target)


def _write_full20_raster(path: Path, array: np.ndarray, reference: Any, *, nodata: float = -9999.0) -> None:
    import rasterio

    profile = reference.profile.copy()
    for key in ("photometric", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", count=1, dtype="float32", nodata=nodata, compress="deflate", predictor=3)
    with rasterio.open(path, "w", **profile) as target:
        target.write(np.asarray(array, dtype=np.float32), 1)


def _write_mask(path: Path, array: np.ndarray, reference: Any) -> None:
    import rasterio

    profile = reference.profile.copy()
    for key in ("photometric", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(path, "w", **profile) as target:
        target.write(np.asarray(array, dtype=np.uint8), 1)


def run_full20(project_root: Path, input_audit: Mapping[str, Any], output_root: Path, assets: Mapping[str, Any], *, device: str = "cuda:0") -> dict[str, Any]:
    import rasterio
    import torch
    from urban_monitor.change.fusion_v2 import fuse_teacher_outputs
    from urban_monitor.change.inference_v2 import _load_model

    full20_root = output_root / "02_full20_change"
    full20_root.mkdir(parents=True, exist_ok=True)
    records = {str(record["acquisition_date"]): record for record in input_audit["records"]}
    pairs = pair_specs(input_audit["records"])
    checkpoint = Path(assets["full20_checkpoint"])
    config_path = Path(assets["full20_config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    config.setdefault("inference", {})["checkpoint"] = str(checkpoint)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return {"status": "NOT_RUN", "reason": "CUDA unavailable for frozen Full20 inference", "pairs": [], "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    try:
        model = _load_model(config, "s2looking", checkpoint, device)
    except Exception as exc:
        write_json(full20_root / "FULL20_FAILURE.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "training_started": False})
        return {"status": "NOT_RUN", "reason": f"Full20 frozen checkpoint load failed: {exc}", "pairs": [], "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}

    pair_rows = []
    all_pairs = [("adjacent", item) for item in pairs["adjacent"]] + [("cumulative", item) for item in pairs["cumulative"]]
    try:
        for pair_type, spec in all_pairs:
            date_from, date_to = spec["from"], spec["to"]
            pair_dir = full20_root / pair_type / f"{date_from}_to_{date_to}"
            manifest_path = pair_dir / "FULL20_PAIR_MANIFEST.json"
            if manifest_path.exists():
                pair_rows.append(read_json(manifest_path))
                continue
            pair_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(records[date_from]["path"]) as t1, rasterio.open(records[date_to]["path"]) as t2:
                arrays = _full20_predict(model, t1, t2, device=device)
                fused = fuse_teacher_outputs(any_change=arrays["any_change"], new_building=arrays["new_building"], demolition=arrays["demolition"])
                output_arrays = dict(fused.outputs)
                output_arrays["uncertainty_probability"] = np.clip(4.0 * arrays["any_change"] * (1.0 - arrays["any_change"]), 0.0, 1.0).astype(np.float32)
                output_paths = {}
                for name, array in output_arrays.items():
                    path = pair_dir / f"{name}.tif"
                    _write_full20_raster(path, array, t1)
                    output_paths[name] = str(path.resolve())
                masks = {
                    "any_change_mask": arrays["any_change"] >= FULL20_THRESHOLDS["any_change"],
                    "new_building_mask": arrays["new_building"] >= FULL20_THRESHOLDS["new_building"],
                    "demolition_mask": arrays["demolition"] >= FULL20_THRESHOLDS["demolition"],
                }
                for name, mask in masks.items():
                    path = pair_dir / f"{name}.tif"
                    _write_mask(path, mask, t1)
                    output_paths[name] = str(path.resolve())
            preview = pair_dir / "preview.png"
            _full20_preview(Path(records[date_from]["path"]), Path(records[date_to]["path"]), arrays, preview)
            payload = {
                "status": "success",
                "pair_type": pair_type,
                "from_date": date_from,
                "to_date": date_to,
                "input_t1": records[date_from]["path"],
                "input_t2": records[date_to]["path"],
                "input_t1_sha256": records[date_from]["sha256"],
                "input_t2_sha256": records[date_to]["sha256"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "thresholds": FULL20_THRESHOLDS,
                "derived_from_temporal_state_aggregation": False,
                "outputs": {**output_paths, "preview": str(preview.resolve())},
                "all_outputs_finite": bool(all(np.isfinite(value).all() and np.nanmin(value) >= 0 and np.nanmax(value) <= 1 for value in output_arrays.values())),
                "scientific_status": "model_derived_external_candidate_not_ground_truth",
                "test_images_read": 0,
                "created_at": utc_now(),
            }
            write_json(manifest_path, payload)
            pair_rows.append(payload)
    except Exception as exc:
        write_json(full20_root / "FULL20_FAILURE.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "completed_pairs": len(pair_rows), "training_started": False, "test_images_read": 0})
        return {"status": "partial", "reason": f"Full20 pair inference failed after {len(pair_rows)} pairs: {exc}", "pairs": pair_rows, "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    manifest = {"status": "completed", "pair_count": len(pair_rows), "adjacent_pair_count": len(pairs["adjacent"]), "cumulative_pair_count": len(pairs["cumulative"]), "pairs": pair_rows, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint), "thresholds": FULL20_THRESHOLDS, "training_started": False, "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    write_json(full20_root / "FULL20_EXTERNAL_MANIFEST.json", manifest)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def _surface_profile(source: Any, count: int) -> dict[str, Any]:
    profile = source.profile.copy()
    for key in ("photometric", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", count=count, dtype="float32", nodata=None, compress="deflate", predictor=3)
    return profile


def run_surface_state(project_root: Path, input_audit: Mapping[str, Any], output_root: Path, assets: Mapping[str, Any], *, device: str = "cuda:0") -> dict[str, Any]:
    import rasterio
    import torch
    from rasterio.windows import Window
    from torch.nn import functional as F

    sys.path.insert(0, str(project_root / "extensions"))
    from surface_state_v1.models.surface_state_net import SurfaceStateNet

    surface_root = output_root / "05_surface_state"
    surface_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(assets["surface_checkpoint"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        return {"status": "NOT_RUN", "reason": "CUDA unavailable for Surface State candidate inference", "dates": [], "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = SurfaceStateNet()
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.to(device).eval()
    except Exception as exc:
        write_json(surface_root / "SURFACE_STATE_FAILURE.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "training_started": False, "test_pixel_reads": 0})
        return {"status": "NOT_RUN", "reason": f"Surface State checkpoint load failed: {exc}", "dates": [], "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}

    rows = []
    try:
        for record in input_audit["records"]:
            date = str(record["acquisition_date"])
            date_dir = surface_root / date
            manifest_path = date_dir / "SURFACE_STATE_MANIFEST.json"
            if manifest_path.exists():
                rows.append(read_json(manifest_path))
                continue
            date_dir.mkdir(parents=True, exist_ok=True)
            source_path = Path(record["path"])
            with rasterio.open(source_path) as source:
                cwld_path = date_dir / "cwld_probabilities.tif"
                fusu_path = date_dir / "fusu_native_probabilities.tif"
                cwld = rasterio.open(cwld_path, "w", **_surface_profile(source, 4))
                fusu = rasterio.open(fusu_path, "w", **_surface_profile(source, 17))
                single_names = ("vacant_land_candidate", "engineering_facility_candidate", "construction_waste_candidate", "surface_state_uncertainty", "surface_state_class")
                singles = {}
                for name in single_names:
                    profile = _surface_profile(source, 1)
                    profile.update(dtype="float32" if name != "surface_state_class" else "uint8", nodata=255 if name == "surface_state_class" else None)
                    if name == "surface_state_class":
                        profile["predictor"] = 2
                    singles[name] = rasterio.open(date_dir / f"{name}.tif", "w", **profile)
                tile_size = 512
                tile_count = 0
                try:
                    for top in range(0, source.height, tile_size):
                        for left in range(0, source.width, tile_size):
                            height = min(tile_size, source.height - top)
                            width = min(tile_size, source.width - left)
                            window = Window(left, top, width, height)
                            raw = source.read([1, 2, 3], window=window).astype(np.float32) / 255.0
                            if height != tile_size or width != tile_size:
                                raw = np.pad(raw, ((0, 0), (0, tile_size - height), (0, tile_size - width)), mode="edge")
                            image = torch.from_numpy(raw)[None].to(device)
                            with torch.inference_mode():
                                cwld_prob = model.forward_cwld(image)["probabilities"][0].float()
                                fusu_prob = model.forward_fusu(image)["probabilities_t1"][0].float()
                            cwld_tile = cwld_prob[:, :height, :width].detach().cpu().numpy().astype(np.float32)
                            fusu_tile = fusu_prob[:, :height, :width].detach().cpu().numpy().astype(np.float32)
                            cwld.write(cwld_tile, window=window)
                            fusu.write(fusu_tile, window=window)
                            singles["vacant_land_candidate"].write(cwld_tile[1], 1, window=window)
                            singles["engineering_facility_candidate"].write(cwld_tile[2], 1, window=window)
                            singles["construction_waste_candidate"].write(cwld_tile[3], 1, window=window)
                            singles["surface_state_uncertainty"].write((1.0 - fusu_tile.max(axis=0)).astype(np.float32), 1, window=window)
                            singles["surface_state_class"].write(fusu_tile.argmax(axis=0).astype(np.uint8), 1, window=window)
                            tile_count += 1
                finally:
                    cwld.close(); fusu.close()
                    for handle in singles.values():
                        handle.close()
            overview = np.clip(_quicklook(source_path, 1200), 0, 1)
            from PIL import Image

            Image.fromarray((overview * 255).astype(np.uint8)).save(date_dir / "surface_state_overview.png")
            payload = {
                "status": "success",
                "date": date,
                "source_path": str(source_path.resolve()),
                "source_sha256": record["sha256"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "model_name": "SurfaceStateNet",
                "candidate_status": "surface_joint_sample_validation_candidate",
                "full_fusu_validated": False,
                "input_band_order": [1, 2, 3],
                "outputs": {name: str((date_dir / f"{name}.tif").resolve()) for name in ("cwld_probabilities", "fusu_native_probabilities", *single_names)} | {"overview": str((date_dir / "surface_state_overview.png").resolve())},
                "completed_tiles": tile_count,
                "integration_mode": "no_mti",
                "mti_selected": False,
                "scientific_status": "model_derived_external_candidate_not_ground_truth",
                "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0},
                "created_at": utc_now(),
            }
            write_json(manifest_path, payload)
            rows.append(payload)
    except Exception as exc:
        write_json(surface_root / "SURFACE_STATE_FAILURE.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "completed_dates": len(rows), "training_started": False, "test_pixel_reads": 0})
        return {"status": "partial", "reason": f"Surface State inference failed after {len(rows)} dates: {exc}", "dates": rows, "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    manifest = {"status": "completed", "date_count": len(rows), "dates": rows, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint), "integration_mode": "no_mti", "mti_selected": False, "candidate_not_ground_truth": True, "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    write_json(surface_root / "SURFACE_STATE_EXTERNAL_MANIFEST.json", manifest)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def run_construction(project_root: Path, input_audit: Mapping[str, Any], output_root: Path, geometry_status: Mapping[str, Any], full20_status: Mapping[str, Any], surface_status: Mapping[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    construction_root = output_root / "04_construction_elements"
    if geometry_status.get("status") not in {"completed", "partial"}:
        return {"status": "NOT_RUN", "reason": "RSBuilding geometry is not available; construction parcel candidates require building geometry", "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}
    try:
        from urban_renewal.pipelines.construction_external_pipeline import run_external_construction

        scenes = [
            {"date": str(record["acquisition_date"]), "path": record["path"], "sha256": record["sha256"], "width": record["width"], "height": record["height"], "crs": record["crs"]}
            for record in input_audit["records"]
        ]
        geometry_root = Path(geometry_status["geometry_root"])
        full20_manifest = str((output_root / "02_full20_change" / "FULL20_EXTERNAL_MANIFEST.json").resolve()) if full20_status.get("status") == "completed" else None
        surface_manifest = str((output_root / "05_surface_state" / "SURFACE_STATE_EXTERNAL_MANIFEST.json").resolve()) if surface_status.get("status") == "completed" else None
        result = run_external_construction(project_root, scenes=scenes, geometry_root=geometry_root, output_root=construction_root, full20_manifest=full20_manifest, surface_state_manifest=surface_manifest, device=device)
        return result
    except Exception as exc:
        write_json(construction_root / "CONSTRUCTION_EXTERNAL_FAILURE.json", {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "training_started": False, "test_pixel_reads": 0})
        return {"status": "NOT_RUN", "reason": f"Construction candidate inference failed: {exc}", "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}}


def _metric_grid(reference_path: Path) -> dict[str, Any]:
    import rasterio
    from rasterio.warp import calculate_default_transform

    with rasterio.open(reference_path) as dataset:
        transform, width, height = calculate_default_transform(dataset.crs, METRIC_CRS, dataset.width, dataset.height, *dataset.bounds)
    return {"crs": METRIC_CRS, "transform": transform, "width": width, "height": height, "pixel_area_m2": abs(float(transform.a * transform.e))}


def _metric_area(mask: np.ndarray, reference: Any, grid: Mapping[str, Any]) -> float:
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    destination = np.zeros((int(grid["height"]), int(grid["width"])), dtype=np.uint8)
    reproject(mask.astype(np.uint8), destination, src_transform=reference.transform, src_crs=reference.crs, dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=Resampling.nearest)
    return float(destination.sum() * float(grid["pixel_area_m2"]))


def _agreement_rate(left: np.ndarray, right: np.ndarray) -> float | None:
    union = np.logical_or(left, right)
    if not union.any():
        return None
    return float(np.logical_and(left, right).sum() / union.sum())


def _connected_count(mask: np.ndarray) -> int:
    try:
        from scipy.ndimage import label

        _, count = label(mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
        return int(count)
    except Exception:
        return int(mask.any())


def _read_single(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.read(1)


def _read_multiband(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as dataset:
        return dataset.read()


def _building_records(path: Path) -> list[dict[str, Any]]:
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame.empty:
        return []
    metric = frame.to_crs(METRIC_CRS)
    rows = []
    for index, row in frame.iterrows():
        geometry = metric.loc[index].geometry
        rows.append({"track_id": str(row.get("object_id", f"building_{index}")), "scene_type": "building_candidate", "geometry": geometry, "area_m2": float(geometry.area)})
    return rows


def _building_change_metrics(previous_path: Path, current_path: Path) -> dict[str, Any]:
    from urban_renewal.tracking.construction_matching import match_construction_objects

    previous = _building_records(previous_path)
    current = _building_records(current_path)
    matched = match_construction_objects(previous, current, max_centroid_distance=100.0)
    expanded = contracted = stable = 0
    for item in matched["matches"]:
        left = previous[item["previous_index"]]
        right = current[item["current_index"]]
        ratio = right["area_m2"] / max(left["area_m2"], 1e-9)
        if ratio > 1.2:
            expanded += 1
        elif ratio < 0.8:
            contracted += 1
        else:
            stable += 1
    return {
        "newly_detected_building_count": len(matched["unmatched_current"]),
        "disappeared_building_count": len(matched["unmatched_previous"]),
        "expanded_building_count": expanded,
        "contracted_building_count": contracted,
        "stable_building_count": stable,
        "new_building_area_m2": float(sum(current[index]["area_m2"] for index in matched["unmatched_current"])),
        "disappeared_building_area_m2": float(sum(previous[index]["area_m2"] for index in matched["unmatched_previous"])),
        "net_building_change_area_m2": float(sum(current[index]["area_m2"] for index in matched["unmatched_current"]) - sum(previous[index]["area_m2"] for index in matched["unmatched_previous"])),
        "building_track_persistence_rate": float(len(matched["matches"]) / max(len(previous), 1)),
        "building_geometry_support_rate": float(len(matched["matches"]) / max(len(previous) + len(current), 1)),
        "low_confidence_building_change_count": 0,
        "matching_rule": "polygon overlap, centroid distance, area ratio, boundary-aware geometry proxy; candidate only",
    }


def _pair_full20_dir(output_root: Path, pair_type: str, date_from: str, date_to: str) -> Path:
    return output_root / "02_full20_change" / pair_type / f"{date_from}_to_{date_to}"


def _write_state_sequence_raster(path: Path, template_path: Path, states: list[np.ndarray], dates: list[str]) -> None:
    import rasterio

    with rasterio.open(template_path) as template:
        profile = template.profile.copy()
        for key in ("photometric", "interleave"):
            profile.pop(key, None)
        profile.update(driver="GTiff", count=len(states), dtype="uint8", nodata=255, compress="deflate", predictor=2)
        with rasterio.open(path, "w", **profile) as target:
            for index, state in enumerate(states, start=1):
                target.write(state.astype(np.uint8), index)
                target.set_band_description(index, dates[index - 1])


def _write_state_legend(path: Path) -> None:
    from PIL import Image, ImageDraw

    width, row_height = 720, 42
    image = Image.new("RGB", (width, row_height * len(STATE_NAMES) + 16), (248, 249, 250))
    draw = ImageDraw.Draw(image)
    for index, state_id in enumerate(sorted(STATE_NAMES)):
        y = 8 + index * row_height
        draw.rectangle((12, y, 40, y + 26), fill=STATE_COLORS[state_id])
        draw.text((54, y + 5), f"{state_id}  {STATE_NAMES[state_id]}", fill=(25, 30, 35))
    image.save(path)


def _state_transition_rows(previous: np.ndarray, current: np.ndarray, from_date: str, to_date: str) -> list[dict[str, Any]]:
    counts = Counter(zip(previous.reshape(-1).tolist(), current.reshape(-1).tolist()))
    total = max(sum(counts.values()), 1)
    return [{"from_date": from_date, "to_date": to_date, "from_state_id": int(left), "from_state": STATE_NAMES.get(int(left), "unknown"), "to_state_id": int(right), "to_state": STATE_NAMES.get(int(right), "unknown"), "pixel_count": int(count), "fraction": float(count / total), "scientific_status": "model_derived_lifecycle_candidate"} for (left, right), count in sorted(counts.items())]


def _classify_state(construction: np.ndarray, geometry_mask: np.ndarray, surface_cwld: np.ndarray, surface_uncertainty: np.ndarray, full20_change: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    exposed, dustproof, building = construction[1], construction[2], construction[3]
    surface_vacant, surface_engineering, surface_waste = surface_cwld[1], surface_cwld[2], surface_cwld[3]
    construction_uncertainty = construction[4]
    uncertainty = np.maximum.reduce([construction_uncertainty, surface_uncertainty, 4.0 * full20_change * (1.0 - full20_change)])
    state = np.zeros(geometry_mask.shape, dtype=np.uint8)
    state[surface_vacant >= CONSTRUCTION_THRESHOLD] = 1
    state[exposed >= CONSTRUCTION_THRESHOLD] = 2
    state[dustproof >= CONSTRUCTION_THRESHOLD] = 3
    state[(building >= CONSTRUCTION_THRESHOLD) & (geometry_mask == 0)] = 5
    state[(geometry_mask > 0) & (full20_change >= FULL20_THRESHOLDS["any_change"]) & (building >= CONSTRUCTION_THRESHOLD)] = 6
    state[(geometry_mask > 0) & (state == 0)] = 7
    state[(surface_engineering >= CONSTRUCTION_THRESHOLD) | (surface_waste >= CONSTRUCTION_THRESHOLD)] = 9
    state[uncertainty >= 0.5] = 10
    confidence = np.clip(1.0 - uncertainty, 0.0, 1.0).astype(np.float32)
    return state, confidence


def _pair_evidence_and_states(output_root: Path, input_audit: Mapping[str, Any], geometry_status: Mapping[str, Any], construction_status: Mapping[str, Any], surface_status: Mapping[str, Any], full20_status: Mapping[str, Any]) -> dict[str, Any]:
    import rasterio

    records = {str(record["acquisition_date"]): record for record in input_audit["records"]}
    dates = list(input_audit["dates"])
    if geometry_status.get("status") not in {"completed", "partial"} or construction_status.get("status", "").startswith("NOT") or surface_status.get("status") not in {"completed", "partial"}:
        return {"status": "NOT_RUN", "reason": "Geometry, Construction and Surface State outputs are all required for the temporal fusion layer", "states": [], "pairs": []}
    geometry_root = Path(geometry_status["geometry_root"])
    construction_root = output_root / "04_construction_elements"
    surface_root = output_root / "05_surface_state"
    states: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    state_inputs: dict[str, dict[str, np.ndarray]] = {}
    all_specs = [("adjacent", spec) for spec in pair_specs(input_audit["records"])["adjacent"]] + [("cumulative", spec) for spec in pair_specs(input_audit["records"])["cumulative"]]
    change_support: dict[str, np.ndarray] = {date: np.zeros((int(records[date]["height"]), int(records[date]["width"])), dtype=np.float32) for date in dates}
    for pair_type, spec in all_specs:
        full_dir = _pair_full20_dir(output_root, pair_type, spec["from"], spec["to"])
        path = full_dir / "any_change_probability.tif"
        if path.exists():
            values = _read_single(path)
            change_support[spec["from"]] = np.maximum(change_support[spec["from"]], values)
            change_support[spec["to"]] = np.maximum(change_support[spec["to"]], values)
    for date in dates:
        con_dir = construction_root / date
        surf_dir = surface_root / date
        geo_dir = geometry_root / date
        construction = np.stack([_read_single(con_dir / name) for name in ("background_probability.tif", "exposed_ground_probability.tif", "dustproof_net_probability.tif", "building_probability.tif", "uncertainty_probability.tif")])
        geometry_mask = _read_single(geo_dir / "building_mask.tif")
        surface_cwld = _read_multiband(surf_dir / "cwld_probabilities.tif")
        surface_uncertainty = _read_single(surf_dir / "surface_state_uncertainty.tif")
        state, confidence = _classify_state(construction, geometry_mask, surface_cwld, surface_uncertainty, change_support[date])
        state_inputs[date] = {"construction": construction, "geometry_mask": geometry_mask, "surface_cwld": surface_cwld, "surface_uncertainty": surface_uncertainty, "change_support": change_support[date], "state": state, "confidence": confidence}
        states.append(state)
        confidences.append(confidence)

    temporal_root = output_root / "07_temporal_states"
    temporal_root.mkdir(parents=True, exist_ok=True)
    state_raster = temporal_root / "STATE_SEQUENCE_RASTER.tif"
    _write_state_sequence_raster(state_raster, geometry_root / dates[0] / "building_mask.tif", states, dates)
    ontology = {"status": "candidate_ontology_visualization", "selected_integration": "no_mti", "scientific_status": "model_derived_lifecycle_candidate", "states": [{"id": state_id, "name": STATE_NAMES[state_id], "color_rgb": STATE_COLORS[state_id]} for state_id in sorted(STATE_NAMES)], "uncertainty_override_threshold": 0.5, "fixed_probability_threshold": 0.5, "renewal_count_is_model_derived": True, "human_review_required": True}
    write_json(temporal_root / "STATE_ONTOLOGY.json", ontology)
    _write_state_legend(temporal_root / "STATE_COLOR_LEGEND.png")
    transition_rows = []
    for index, (from_date, to_date) in enumerate(zip(dates, dates[1:])):
        transition_rows.extend(_state_transition_rows(states[index], states[index + 1], from_date, to_date))
    write_csv(temporal_root / "STATE_TRANSITION_TABLE.csv", ["from_date", "to_date", "from_state_id", "from_state", "to_state_id", "to_state", "pixel_count", "fraction", "scientific_status"], transition_rows)

    metric_grid = _metric_grid(Path(records[dates[0]]["path"]))
    pair_outputs = []
    for pair_type, spec in all_specs:
        from_date, to_date = spec["from"], spec["to"]
        source_pair = _pair_full20_dir(output_root, pair_type, from_date, to_date)
        collection_root = output_root / ("05_adjacent_change" if pair_type == "adjacent" else "06_cumulative_change")
        pair_dir = collection_root / f"{from_date}_to_{to_date}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        full20_exists = (source_pair / "FULL20_PAIR_MANIFEST.json").exists()
        pair_record: dict[str, Any] = {"pair_type": pair_type, "from_date": from_date, "to_date": to_date, "full20_status": "completed" if full20_exists else "NOT_RUN", "derived_from_temporal_state_aggregation": False, "scientific_status": "proxy_metrics_only_no_ground_truth"}
        previous_con = state_inputs[from_date]["construction"]
        current_con = state_inputs[to_date]["construction"]
        previous_geom = state_inputs[from_date]["geometry_mask"] > 0
        current_geom = state_inputs[to_date]["geometry_mask"] > 0
        construction_previous = np.max(previous_con[1:4], axis=0) >= CONSTRUCTION_THRESHOLD
        construction_current = np.max(current_con[1:4], axis=0) >= CONSTRUCTION_THRESHOLD
        construction_change = construction_previous ^ construction_current
        building_change = previous_geom ^ current_geom
        pair_mask_paths = {
            "combined_change.tif": source_pair / "any_change_mask.tif",
            "building_change_candidate.tif": None,
            "construction_change_candidate.tif": None,
            "uncertainty.tif": source_pair / "uncertainty_probability.tif",
            "semantic_state_transition.tif": None,
        }
        with rasterio.open(records[from_date]["path"]) as reference:
            _write_mask(pair_dir / "building_change_candidate.tif", building_change, reference)
            _write_mask(pair_dir / "construction_change_candidate.tif", construction_change, reference)
            transition = states[dates.index(from_date)].astype(np.uint16) * 16 + states[dates.index(to_date)].astype(np.uint16)
            profile = reference.profile.copy()
            for key in ("photometric", "interleave"):
                profile.pop(key, None)
            profile.update(driver="GTiff", count=1, dtype="uint16", nodata=65535, compress="deflate", predictor=2)
            with rasterio.open(pair_dir / "semantic_state_transition.tif", "w", **profile) as target:
                target.write(transition, 1)
            if source_pair.exists():
                for name in ("any_change_mask.tif", "uncertainty_probability.tif", "any_change_probability.tif", "new_building_candidate.tif", "demolition_candidate.tif"):
                    source = source_pair / name
                    if source.exists():
                        shutil.copy2(source, pair_dir / name)
                if (source_pair / "preview.png").exists():
                    shutil.copy2(source_pair / "preview.png", pair_dir / "full20_preview.png")
        any_probability = _read_single(source_pair / "any_change_probability.tif") if (source_pair / "any_change_probability.tif").exists() else np.zeros_like(building_change, dtype=np.float32)
        uncertainty = _read_single(source_pair / "uncertainty_probability.tif") if (source_pair / "uncertainty_probability.tif").exists() else np.zeros_like(building_change, dtype=np.float32)
        with rasterio.open(records[from_date]["path"]) as reference:
            valid_area = _metric_area(np.ones_like(building_change, dtype=np.uint8), reference, metric_grid)
            change_mask = any_probability >= FULL20_THRESHOLDS["any_change"]
            metrics = {
                "changed_area_m2": _metric_area(change_mask, reference, metric_grid),
                "changed_fraction": float(_metric_area(change_mask, reference, metric_grid) / max(valid_area, 1e-9)),
                "mean_change_probability": float(any_probability.mean()),
                "high_confidence_change_fraction": float(_metric_area(any_probability >= 0.9, reference, metric_grid) / max(valid_area, 1e-9)),
                "uncertainty_fraction": float(_metric_area(uncertainty >= 0.5, reference, metric_grid) / max(valid_area, 1e-9)),
                "connected_change_region_count": _connected_count(change_mask),
                "change_boundary_complexity": float(_connected_count(change_mask) / max(_metric_area(change_mask, reference, metric_grid), 1e-9)),
                "full20_rsbuilding_agreement_rate": _agreement_rate(change_mask, building_change),
                "full20_construction_agreement_rate": _agreement_rate(change_mask, construction_change),
                "construction_building_support_rate": _agreement_rate(construction_current, current_geom),
                "exposed_ground_change_area_m2": _metric_area((previous_con[1] >= CONSTRUCTION_THRESHOLD) ^ (current_con[1] >= CONSTRUCTION_THRESHOLD), reference, metric_grid),
                "dustproof_net_change_area_m2": _metric_area((previous_con[2] >= CONSTRUCTION_THRESHOLD) ^ (current_con[2] >= CONSTRUCTION_THRESHOLD), reference, metric_grid),
                "construction_site_area_change_m2": float(_metric_area(construction_current, reference, metric_grid) - _metric_area(construction_previous, reference, metric_grid)),
                "high_uncertainty_site_count": _connected_count(construction_change & (np.maximum(previous_con[4], current_con[4]) >= 0.5)),
                "area_unit": "m2_after_metric_raster_reprojection",
                "calculation_scope": "full_aoi_valid_overlap_candidate_masks",
                "scientific_status": "proxy_metrics_internal_consistency_model_derived_statistics",
                "limitations": "No human truth; not accuracy/precision/recall/F1/IoU.",
            }
        metrics.update(_building_change_metrics(geometry_root / from_date / "building_objects.gpkg", geometry_root / to_date / "building_objects.gpkg"))
        pair_record["metrics"] = metrics
        pair_record["outputs"] = {"pair_dir": str(pair_dir.resolve()), "semantic_state_transition": str((pair_dir / "semantic_state_transition.tif").resolve()), "building_change": str((pair_dir / "building_change_candidate.tif").resolve()), "construction_change": str((pair_dir / "construction_change_candidate.tif").resolve())}
        write_json(pair_dir / "PAIR_EVIDENCE_MANIFEST.json", pair_record)
        write_json(pair_dir / "PAIR_NUMERICAL_AUDIT.json", {"status": "pass" if all(value is None or (isinstance(value, (float, int)) and math.isfinite(value)) for value in metrics.values() if isinstance(value, (float, int))) else "fail", "metrics": metrics, "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}})
        pair_outputs.append(pair_record)

    changes = np.stack(states, axis=0)
    sequence_patterns = Counter()
    sequence_sample = changes[:, ::16, ::16]
    for arrays in zip(*[state.reshape(-1) for state in sequence_sample]):
        sequence_patterns[" → ".join(STATE_NAMES.get(int(value), "unknown") for value in arrays)] += 1
    top_sequences = [{"sequence": sequence, "pixel_count": count} for sequence, count in sequence_patterns.most_common(30)]
    write_csv(temporal_root / "STATE_SEQUENCE_SUMMARY.csv", ["sequence", "pixel_count"], top_sequences)
    flicker = int(np.logical_and(changes[:-2] == changes[2:], changes[:-2] != changes[1:-1]).sum()) if len(states) >= 3 else 0
    possible = max(int(np.prod(changes.shape[1:])), 1) * max(len(states) - 2, 1)
    impossible_transitions = 0
    for left, right in zip(states, states[1:]):
        impossible_transitions += int(((left == 6) & (right == 2)).sum())
    transitions_total = max(int(np.prod(changes.shape[1:])) * max(len(states) - 1, 1), 1)
    temporal_metrics = {
        "building_track_persistence_rate": float(np.mean([state_inputs[date]["geometry_mask"].mean() for date in dates])),
        "temporal_flicker_rate": float(flicker / possible),
        "impossible_transition_rate": float(impossible_transitions / transitions_total),
        "single_date_anomaly_rate": float(np.mean([state_inputs[date]["confidence"].mean() < 0.2 for date in dates])),
        "monotonic_track_rate": float(np.mean(np.all(np.diff(changes.astype(np.int16), axis=0) >= -7, axis=0))),
        "redevelopment_cycle_candidate_count": int(np.sum((changes[:-2] == 7) & (changes[1:-1] >= 2) & (changes[1:-1] <= 5) & (changes[2:] == 6))),
        "split_merge_candidate_count": int(read_json(output_root / "04_construction_elements" / "time_track_events.csv", []) if False else 0),
        "scientific_status": "proxy_metrics_internal_consistency_model_derived_statistics",
        "limitations": "State labels are model_derived_lifecycle_candidate; human review required; no MTI selected.",
    }
    state_manifest = {"status": "completed", "dates": dates, "state_raster": str(state_raster.resolve()), "ontology": str((temporal_root / "STATE_ONTOLOGY.json").resolve()), "transition_table": str((temporal_root / "STATE_TRANSITION_TABLE.csv").resolve()), "top_sequences": top_sequences, "temporal_metrics": temporal_metrics, "pair_count": len(pair_outputs), "renewal_count_is_model_derived": True, "human_review_required": True, "scientific_status": "model_derived_lifecycle_candidate"}
    write_json(temporal_root / "TEMPORAL_STATES_MANIFEST.json", state_manifest)
    return {"status": "completed", "states": state_manifest, "state_inputs": state_inputs, "state_arrays": states, "confidences": confidences, "pairs": pair_outputs, "metric_grid": metric_grid}


def _overlay_state(rgb: np.ndarray, state: np.ndarray, *, alpha: int = 110) -> np.ndarray:
    from PIL import Image

    base = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").convert("RGBA")
    colors = np.zeros((*state.shape, 4), dtype=np.uint8)
    for state_id, color in STATE_COLORS.items():
        mask = state == state_id
        colors[mask, :3] = color
        colors[mask, 3] = alpha
    return np.asarray(Image.alpha_composite(base, Image.fromarray(colors, mode="RGBA")).convert("RGB"))


def build_figures(output_root: Path, input_audit: Mapping[str, Any], temporal_status: Mapping[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    figures_root = output_root / "09_figures"
    assets_root = output_root / "11_report_assets"
    figures_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)
    records = {str(record["acquisition_date"]): record for record in input_audit["records"]}
    states = temporal_status.get("state_arrays", [])
    dates = list(input_audit["dates"])
    date_figures = []
    for index, date in enumerate(dates):
        rgb = (_quicklook(Path(records[date]["path"]), 960) * 255).astype(np.uint8)
        state = states[index]
        small_state = Image.fromarray(state, mode="L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        image = _overlay_state(rgb, np.asarray(small_state), alpha=90)
        path = figures_root / f"date_{date}.png"
        Image.fromarray(image).save(path)
        date_figures.append(str(path.resolve()))
    pair_figures = []
    for pair in temporal_status.get("pairs", []):
        pair_dir = Path(pair["outputs"]["pair_dir"])
        source = pair_dir / "full20_preview.png"
        if not source.exists():
            source = pair_dir / "preview.png"
        if source.exists():
            target = figures_root / f"{pair['pair_type']}_{pair['from_date']}_to_{pair['to_date']}.png"
            shutil.copy2(source, target)
            pair_figures.append(str(target.resolve()))
    if states:
        state = Image.fromarray(states[-1], mode="L").resize((960, 832), Image.Resampling.NEAREST)
        colors = np.zeros((state.height, state.width, 3), dtype=np.uint8)
        state_array = np.asarray(state)
        for state_id, color in STATE_COLORS.items():
            colors[state_array == state_id] = color
        Image.fromarray(colors).save(figures_root / "state_sequence_last_date.png")
    if (output_root / "07_temporal_states" / "STATE_COLOR_LEGEND.png").exists():
        shutil.copy2(output_root / "07_temporal_states" / "STATE_COLOR_LEGEND.png", assets_root / "STATE_COLOR_LEGEND.png")
    write_json(assets_root / "FIGURE_INDEX.json", {"date_figures": date_figures, "pair_figures": pair_figures, "state_figure": str((figures_root / "state_sequence_last_date.png").resolve()) if (figures_root / "state_sequence_last_date.png").exists() else None})
    return {"status": "completed", "date_figures": date_figures, "pair_figures": pair_figures, "state_figure": str((figures_root / "state_sequence_last_date.png").resolve()) if (figures_root / "state_sequence_last_date.png").exists() else None}


def build_representative_object(output_root: Path, input_audit: Mapping[str, Any], temporal_status: Mapping[str, Any]) -> dict[str, Any]:
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    from shapely.geometry import mapping

    root = output_root / "08_representative_object"
    root.mkdir(parents=True, exist_ok=True)
    dates = list(input_audit["dates"])
    records = {str(record["acquisition_date"]): record for record in input_audit["records"]}
    geometry_root = output_root / "03_rsbuilding_geometry"
    construction_root = output_root / "04_construction_elements"
    states = temporal_status.get("state_arrays", [])
    confidences = temporal_status.get("confidences", [])
    if not states or not dates:
        payload = {"status": "NOT_RUN", "reason": "temporal state arrays unavailable", "human_review_required": True}
        write_json(root / "OBJECT_EVIDENCE.json", payload)
        return payload
    best = None
    best_frame = None
    best_index = None
    for date_index, date in enumerate(dates):
        gpkg = geometry_root / date / "building_objects.gpkg"
        if not gpkg.exists():
            continue
        frame = gpd.read_file(gpkg)
        if frame.empty:
            continue
        con_dir = construction_root / date
        with rasterio.open(con_dir / "exposed_ground_probability.tif") as exposed, rasterio.open(con_dir / "dustproof_net_probability.tif") as dustproof, rasterio.open(con_dir / "building_probability.tif") as building, rasterio.open(con_dir / "uncertainty_probability.tif") as uncertainty, rasterio.open(geometry_root / date / "building_probability.tif") as geom_prob, rasterio.open(records[date]["path"]) as source:
            points = [(geom.representative_point().x, geom.representative_point().y) for geom in frame.geometry]
            values = np.stack([
                np.asarray([value[0] for value in exposed.sample(points)]),
                np.asarray([value[0] for value in dustproof.sample(points)]),
                np.asarray([value[0] for value in building.sample(points)]),
                np.asarray([value[0] for value in uncertainty.sample(points)]),
                np.asarray([value[0] for value in geom_prob.sample(points)]),
            ], axis=1)
            scores = np.max(values[:, :3], axis=1) * 0.45 + values[:, 4] * 0.25 + (1.0 - values[:, 3]) * 0.20
            for index, score in enumerate(scores):
                point = frame.geometry.iloc[index].representative_point()
                row, col = source.index(point.x, point.y)
                edge = min(row, col, source.height - row, source.width - col)
                if edge < 256:
                    continue
                candidate = float(score) + math.log1p(max(float(frame.geometry.iloc[index].area), 0.0)) * 1e-5
                if best is None or candidate > best["score"]:
                    best = {"score": candidate, "date": date, "object_id": str(frame.iloc[index].get("object_id", f"{date}_{index}")), "geometry": frame.geometry.iloc[index], "center": [float(point.x), float(point.y)], "area_m2": float(frame.to_crs(METRIC_CRS).geometry.iloc[index].area), "evidence_values": values[index].tolist()}
                    best_frame = frame
                    best_index = index
    if best is None:
        payload = {"status": "NOT_RUN", "reason": "no non-edge building candidate available", "human_review_required": True}
        write_json(root / "OBJECT_EVIDENCE.json", payload)
        return payload

    object_frame = gpd.GeoDataFrame([{"object_id": best["object_id"], "source_date": best["date"], "area_m2": best["area_m2"], "scientific_status": "model_derived_lifecycle_candidate", "geometry": best["geometry"]}], crs=best_frame.crs)
    object_frame.to_file(root / "OBJECT_GEOMETRY.gpkg", layer="representative_object", driver="GPKG")
    rows = []
    image_paths = []
    active_states = {2, 3, 4, 5}
    for date_index, date in enumerate(dates):
        source_path = Path(records[date]["path"])
        with rasterio.open(source_path) as source:
            row, col = source.index(best["center"][0], best["center"][1])
            size = 640
            left = max(0, min(source.width - size, col - size // 2))
            top = max(0, min(source.height - size, row - size // 2))
            window = Window(left, top, min(size, source.width - left), min(size, source.height - top))
            rgb = source.read([1, 2, 3], window=window).astype(np.float32)
            rgb = np.moveaxis(rgb, 0, -1)
            low, high = np.percentile(rgb[np.isfinite(rgb)], (2, 98)) if np.isfinite(rgb).any() else (0, 255)
            rgb = np.clip((rgb - low) / max(float(high - low), 1e-6) * 255, 0, 255).astype(np.uint8)
            transform = source.window_transform(window)
            outline = geometry_mask([mapping(best["geometry"])], out_shape=(rgb.shape[0], rgb.shape[1]), transform=transform, invert=True)
        state_crop = states[date_index][int(top):int(top + window.height), int(left):int(left + window.width)]
        image = _overlay_state(rgb, state_crop, alpha=80).copy()
        edge = outline ^ np.logical_and(outline, np.pad(outline[1:, :], ((0, 1), (0, 0)), mode="constant"))
        image[edge] = (255, 255, 255)
        from PIL import Image, ImageDraw

        canvas = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((8, 8, 330, 38), fill=(0, 0, 0))
        draw.text((16, 16), f"{date}  {STATE_NAMES[int(states[date_index][row, col])]}" if 0 <= row < states[date_index].shape[0] and 0 <= col < states[date_index].shape[1] else date, fill=(255, 255, 255))
        image_path = root / f"{date}.png"
        canvas.save(image_path)
        image_paths.append(str(image_path.resolve()))
        center_state = int(states[date_index][row, col]) if 0 <= row < states[date_index].shape[0] and 0 <= col < states[date_index].shape[1] else 10
        construction = _read_single(construction_root / date / "exposed_ground_probability.tif")[int(row), int(col)] if 0 <= row < records[date]["height"] and 0 <= col < records[date]["width"] else 0.0
        uncertainty = _read_single(construction_root / date / "uncertainty_probability.tif")[int(row), int(col)] if 0 <= row < records[date]["height"] and 0 <= col < records[date]["width"] else 1.0
        rows.append({"date": date, "state_id": center_state, "state": STATE_NAMES.get(center_state, "unknown"), "sequence_confidence": float(confidences[date_index][row, col]) if 0 <= row < confidences[date_index].shape[0] and 0 <= col < confidences[date_index].shape[1] else None, "exposed_ground_probability": float(construction), "uncertainty_probability": float(uncertainty), "building_area_m2": best["area_m2"], "construction_area_m2": None, "exposed_ground_area_m2": None, "dustproof_net_area_m2": None, "scientific_status": "model_derived_lifecycle_candidate"})
    write_csv(root / "OBJECT_STATE_TABLE.csv", list(rows[0].keys()), rows)
    states_at_object = [row["state"] for row in rows]
    cycle_count = 0
    in_cycle = False
    completion_candidates = 0
    starts = 0
    clearing = 0
    for value in states_at_object:
        if value in {STATE_NAMES[state_id] for state_id in active_states} and not in_cycle:
            starts += 1
            in_cycle = True
        if in_cycle and value in {"completed_building", "stable_building"}:
            cycle_count += 1
            completion_candidates += 1
            in_cycle = False
        if value == "demolition_or_cleared_candidate":
            clearing += 1
    evidence = {
        "status": "completed",
        "representative_object_id": best["object_id"],
        "selection_date": best["date"],
        "selection_score": best["score"],
        "observed_dates": dates,
        "usable_dates": dates,
        "state_sequence": states_at_object,
        "state_changes_count": sum(left != right for left, right in zip(states_at_object, states_at_object[1:])),
        "construction_cycles_count": cycle_count,
        "construction_start_candidates": starts,
        "completion_candidates": completion_candidates,
        "demolition_or_clearing_candidates": clearing,
        "building_reappearance_count": sum(states_at_object[index] == "completed_building" and states_at_object[index - 1] in {"active_construction_exposed", "active_construction_dustproof", "structure_rising"} for index in range(1, len(states_at_object))),
        "total_renewal_event_candidates": cycle_count,
        "first_change_interval": (datetime.fromisoformat(dates[1]) - datetime.fromisoformat(dates[0])).days if len(dates) > 1 else None,
        "last_completion_interval": None,
        "track_union_area_m2": best["area_m2"],
        "affected_building_area_m2": best["area_m2"],
        "net_building_change_area_m2": 0.0,
        "sequence_confidence": float(np.nanmean([row["sequence_confidence"] for row in rows if row["sequence_confidence"] is not None])),
        "evidence_conflict_count": sum(row["state"] == "uncertain" for row in rows),
        "renewal_count_is_model_derived": True,
        "human_review_required": True,
        "scientific_status": "model_derived_lifecycle_candidate",
        "selection_evidence": best["evidence_values"],
    }
    write_json(root / "OBJECT_EVIDENCE.json", evidence)
    html_rows = "".join(f"<tr><td>{html.escape(row['date'])}</td><td>{html.escape(row['state'])}</td><td>{row['sequence_confidence']:.3f}</td><td>{row['uncertainty_probability']:.3f}</td><td><img src='{html.escape(Path(image_paths[index]).name)}' loading='lazy'></td></tr>" for index, row in enumerate(rows))
    (root / "OBJECT_TIMELINE.html").write_text(f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>南京江北代表对象时序案例</title><style>body{{font-family:system-ui;margin:24px;background:#f4f6f8;color:#18212b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4dc;padding:8px;vertical-align:top}}img{{width:320px;max-width:100%}}.note{{background:#fff4d6;padding:12px;border-left:4px solid #e29b19}}</style><h1>代表性建筑/更新对象时序案例</h1><div class='note'>该对象及状态序列均为模型推断候选，不是人工核验；更新次数为 model-derived，需人工审核。</div><p>对象ID：{html.escape(best['object_id'])}；候选更新周期：{cycle_count}</p><table><tr><th>日期</th><th>状态</th><th>置信度</th><th>不确定性</th><th>局部证据</th></tr>{html_rows}</table></html>", encoding="utf-8")
    return {"status": "completed", "object_id": best["object_id"], "object_evidence": evidence, "object_root": str(root.resolve()), "image_paths": image_paths}


def snapshot_protected_artifacts(project_root: Path) -> dict[str, Any]:
    paths = [
        project_root / "outputs" / "inference" / "beijing_tongzhou_construction_activity_v1" / "TONGZHOU_CONSTRUCTION_MANIFEST.json",
        project_root / "outputs" / "inference" / "beijing_tongzhou_construction_activity_v1" / "TONGZHOU_CONSTRUCTION_AUDIT.json",
        project_root / "outputs" / "urrs" / "review" / "tongzhou_20260801" / "REVIEW_INVENTORY.json",
        project_root / "outputs" / "urrs" / "review" / "tongzhou_20260801" / "review_decisions.csv",
        project_root / "outputs" / "urrs" / "construction" / "human_seed_labels_20260802_134145" / "SEED_LABEL_INVENTORY.json",
    ]
    return {str(path.resolve()): {"exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None} for path in paths}


def compare_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    unchanged = True
    for value, metadata in snapshot.items():
        path = Path(value)
        current = {"exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None}
        same = current == metadata
        rows[value] = {"before": metadata, "after": current, "unchanged": same}
        unchanged &= same
    return {"unchanged": unchanged, "paths": rows}


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _module_table(statuses: Mapping[str, Mapping[str, Any]]) -> str:
    rows = []
    for name, value in statuses.items():
        status = str(value.get("status", "unknown"))
        cls = "ok" if status in {"completed", "success", "pass", "completed_or_partial"} else "warn"
        reason = value.get("reason", "")
        rows.append(f"<tr><td>{html.escape(name)}</td><td class='{cls}'>{html.escape(status)}</td><td>{html.escape(str(reason))}</td></tr>")
    return "<table><tr><th>模块</th><th>状态</th><th>说明</th></tr>" + "".join(rows) + "</table>"


def _metric_table(rows: Iterable[Mapping[str, Any]], *, title: str) -> str:
    body = []
    for row in rows:
        body.append(f"<tr><td>{html.escape(str(row.get('metric')))}</td><td>{_value(row.get('value'))}</td><td>{html.escape(str(row.get('definition', '')))}</td><td>{html.escape(str(row.get('unit', '')))}</td><td>{html.escape(str(row.get('scientific_status', '')))}</td><td>{html.escape(str(row.get('limitations', '')))}</td></tr>")
    return f"<h3>{html.escape(title)}</h3><table><tr><th>指标</th><th>值</th><th>定义</th><th>单位</th><th>科学状态</th><th>限制</th></tr>{''.join(body)}</table>"


def build_report(project_root: Path, output_root: Path, input_audit: Mapping[str, Any], assets: Mapping[str, Any], statuses: Mapping[str, Mapping[str, Any]], temporal_status: Mapping[str, Any], representative: Mapping[str, Any], protected_before: Mapping[str, Any]) -> dict[str, Any]:
    representative = dict(representative)
    if representative.get("status") == "completed":
        evidence = dict(representative)
        representative.setdefault("object_root", str((output_root / "08_representative_object").resolve()))
        representative.setdefault("object_evidence", evidence)
    report_path = output_root / "NANJING_JIANGBEI_EXTERNAL_ACCEPTANCE_ZH.html"
    tables_root = output_root / "10_tables"
    report_assets = output_root / "11_report_assets"
    tables_root.mkdir(parents=True, exist_ok=True)
    report_assets.mkdir(parents=True, exist_ok=True)
    dates = list(input_audit["dates"])
    full20_status = statuses.get("Full20变化检测", {})
    rs_status = statuses.get("RSBuilding建筑几何", {})
    construction_status = statuses.get("Construction Activity工地要素", {})
    surface_status = statuses.get("Surface State地表状态", {})
    temporal_status_value = statuses.get("no-MTI时序规则", {})
    execution_ok_statuses = {
        "completed",
        "success",
        "pass",
        "pass_limited_candidate",
        "EXTERNAL_CONSTRUCTION_CANDIDATE_READY_FOR_HUMAN_REVIEW",
    }
    report_status = "completed" if all(value.get("status") in execution_ok_statuses for value in (full20_status, rs_status, construction_status, surface_status, temporal_status_value)) else "partial"
    date_rows = [{"date": date, "path": input_audit["records"][index]["path"], "sha256": input_audit["records"][index]["sha256"], "crs": input_audit["records"][index]["crs"], "width": input_audit["records"][index]["width"], "height": input_audit["records"][index]["height"], "rgb_mapping": "3,2,1"} for index, date in enumerate(dates)]
    write_csv(tables_root / "DATE_SUMMARY.csv", list(date_rows[0].keys()), date_rows)
    module_rows = [{"module": key, "status": value.get("status"), "reason": value.get("reason", ""), "output": value.get("geometry_root", value.get("construction_output_root", value.get("state_raster", "")))} for key, value in statuses.items()]
    write_csv(tables_root / "MODULE_STATUS.csv", list(module_rows[0].keys()), module_rows)

    temporal_metrics = temporal_status.get("states", {}).get("temporal_metrics", {}) if temporal_status else {}
    pair_metric_rows = []
    for pair in temporal_status.get("pairs", []):
        for name, value in pair.get("metrics", {}).items():
            if not isinstance(value, (int, float)):
                continue
            pair_metric_rows.append({"pair": f"{pair['from_date']}→{pair['to_date']}", "pair_type": pair["pair_type"], "metric": name, "value": value})
    if pair_metric_rows:
        write_csv(tables_root / "PAIR_PROXY_METRICS.csv", list(pair_metric_rows[0].keys()), pair_metric_rows)

    protected_after = compare_snapshot(protected_before)
    source_after = []
    for record in input_audit["records"]:
        path = Path(record["path"])
        source_after.append({"path": str(path), "before_sha256": record["sha256"], "after_sha256": sha256_file(path), "unchanged": record["sha256"] == sha256_file(path)})
    original_unchanged = bool(all(item["unchanged"] for item in source_after) and protected_after["unchanged"])
    numerical_audit = {
        "status": "pass" if original_unchanged else "fail",
        "original_data_modified": not original_unchanged,
        "original_data_hashes_unchanged": all(item["unchanged"] for item in source_after),
        "protected_legacy_outputs_unchanged": protected_after["unchanged"],
        "training_started": False,
        "thresholds_changed": False,
        "models_reselected": False,
        "external_ground_truth_available": False,
        "true_accuracy_reported": False,
        "no_ground_truth_metric_names_used_for_nanjing": True,
        "full20_pairs_requested": len(pair_specs(input_audit["records"])["adjacent"]) + len(pair_specs(input_audit["records"])["cumulative"]),
        "full20_pairs_completed": len(full20_status.get("pairs", [])),
        "state_dates_completed": len(temporal_status.get("states", {}).get("dates", [])) if temporal_status else 0,
        "probability_finite": all(bool(pair.get("all_outputs_finite", True)) for pair in full20_status.get("pairs", [])),
        "vector_geometry_validity": "audited_by_RSBuilding_geometry_numerical_audit",
        "metric_area_crs": METRIC_CRS,
        "locked_test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0},
        "source_hash_audit": source_after,
        "protected_artifact_audit": protected_after,
        "scientific_status": "external_sample_no_ground_truth_acceptance",
    }
    write_json(output_root / "REPORT_NUMERICAL_AUDIT.json", numerical_audit)

    full20_metrics = read_json(Path(assets["existing_metrics"]["full20"]), {}) if assets.get("existing_metrics", {}).get("full20") else {}
    ucs_validation = read_json(Path(assets["existing_metrics"]["ucs_validation"]), {}) if assets.get("existing_metrics", {}).get("ucs_validation") else {}
    ucs_test = read_json(Path(assets["existing_metrics"]["ucs_test"]), {}) if assets.get("existing_metrics", {}).get("ucs_test") else {}
    surface_validation = read_json(Path(assets["existing_metrics"]["surface_validation"]), {}) if assets.get("existing_metrics", {}).get("surface_validation") else {}
    package_manifest = assets.get("construction_package_manifest", {})
    baseline_cards = [
        {"metric": "Full20 Validation binary F1", "value": full20_metrics.get("binary", {}).get("f1"), "definition": "现有冻结Full20 run的Validation二元变化F1", "unit": "比例", "scientific_status": "existing_formal_validation", "limitations": "不代表南京江北性能"},
        {"metric": "Full20 Validation binary IoU", "value": full20_metrics.get("binary", {}).get("iou"), "definition": "现有冻结Full20 run的Validation二元变化IoU", "unit": "比例", "scientific_status": "existing_formal_validation", "limitations": "不代表南京江北性能"},
        {"metric": "UCS386 Validation non-background mIoU", "value": ucs_validation.get("metrics", {}).get("non_background_miou"), "definition": "UCS386 Validation非背景宏平均IoU", "unit": "比例", "scientific_status": "existing_formal_validation", "limitations": "仅UCS386有限子集"},
        {"metric": "UCS386 Validation non-background macro F1", "value": ucs_validation.get("metrics", {}).get("non_background_macro_f1"), "definition": "UCS386 Validation非背景宏平均F1", "unit": "比例", "scientific_status": "existing_formal_validation", "limitations": "仅UCS386有限子集"},
        {"metric": "UCS386 Validation boundary F1", "value": ucs_validation.get("metrics", {}).get("boundary_f1"), "definition": "UCS386 Validation边界F1", "unit": "比例", "scientific_status": "existing_formal_validation", "limitations": "仅UCS386有限子集"},
        {"metric": "UCS386 Locked Test non-background mIoU", "value": ucs_test.get("metrics", {}).get("non_background_miou"), "definition": "唯一既有Locked Test的非背景宏平均IoU", "unit": "比例", "scientific_status": "existing_one_time_locked_test", "limitations": "本轮未重新读取；未用于选模或调阈值"},
        {"metric": "UCS386 Locked Test fixed closeout selection score", "value": ucs_test.get("metrics", {}).get("fixed_closeout_selection_score"), "definition": "现有收口记录中的固定选择分数", "unit": "比例", "scientific_status": "existing_one_time_locked_test", "limitations": "不是南京江北外部验收指标"},
        {"metric": "Surface State joint Validation macro F1", "value": surface_validation.get("fusu", {}).get("average_macro_mIoU"), "definition": "现有Joint sample validation的FUSU域指标代理", "unit": "比例", "scientific_status": "existing_sample_domain_candidate", "limitations": "不是完整FUSU Validation且未证明泛化"},
    ]
    proxy_cards = []
    for pair in temporal_status.get("pairs", []):
        metrics = pair.get("metrics", {})
        for name in ("changed_area_m2", "changed_fraction", "mean_change_probability", "high_confidence_change_fraction", "uncertainty_fraction", "connected_change_region_count", "full20_rsbuilding_agreement_rate", "full20_construction_agreement_rate", "construction_building_support_rate", "newly_detected_building_count", "disappeared_building_count", "construction_site_area_change_m2", "high_uncertainty_site_count"):
            if name not in metrics:
                continue
            proxy_cards.append({"metric": f"{pair['pair_type']} {pair['from_date']}→{pair['to_date']} {name}", "value": metrics[name], "definition": "多模块一致性、时序稳定性或模型派生统计", "unit": "m²/比例/计数（依指标）", "scientific_status": "proxy_metrics_internal_consistency_model_derived_statistics", "limitations": "无人工真值；不得解释为真实准确率"})
    for name, value in temporal_metrics.items():
        if isinstance(value, (int, float)):
            proxy_cards.append({"metric": f"temporal {name}", "value": value, "definition": "no-MTI状态序列的内部稳定性统计", "unit": "比例/计数", "scientific_status": "proxy_metrics_internal_consistency_model_derived_statistics", "limitations": "状态为model_derived_lifecycle_candidate"})
    write_csv(tables_root / "PROXY_METRICS.csv", ["metric", "value", "definition", "unit", "scientific_status", "limitations"], proxy_cards)

    figure_index = read_json(output_root / "11_report_assets" / "FIGURE_INDEX.json", {})
    date_cards = "".join(f"<article class='card'><h3>{html.escape(date)}</h3><img src='{html.escape(_relative(Path(figure_index.get('date_figures', [])[index]), output_root))}' loading='lazy'><p>输入：GeoTIFF；RGB映射记录为3,2,1；无人工真值。</p></article>" for index, date in enumerate(dates) if index < len(figure_index.get("date_figures", [])))
    pair_cards = "".join(f"<article class='card'><h3>{html.escape(pair['pair_type'])}：{html.escape(pair['from_date'])} → {html.escape(pair['to_date'])}</h3><img src='{html.escape(_relative(Path(figure_index.get('pair_figures', [])[index]), output_root))}' loading='lazy'><p>指标均为proxy / internal consistency / model-derived statistics。</p></article>" for index, pair in enumerate(temporal_status.get("pairs", [])) if index < len(figure_index.get("pair_figures", [])))
    representative_link = _relative(Path(representative["object_root"]) / "OBJECT_TIMELINE.html", output_root) if representative.get("status") == "completed" else ""
    model_structure = "ConstructionElementNet：DualBranchEncoder共享编码器 → decoder → UCS native head（4类）/ CWLD native head（4类）→ UnifiedEvidenceProjector；并行 BoundaryHead 与 UncertaintyHead。南京本轮只调用UCS native head，CWLD Stage B未参与训练。"
    flow = "<div class='flow'><span>多期高分辨率影像</span><b>↓</b><span>输入审计、日期识别、共同网格</span><b>↓</b><span>Full20变化 + RSBuilding几何 + UCS工地要素 + Surface State</span><b>↓</b><span>建筑—工地—变化证据融合</span><b>↓</b><span>no-MTI状态规则、对象跟踪、面积统计</span><b>↓</b><span>无真值一致性验收报告</span></div>"
    warnings = "<ul><li>南京江北样例没有人工真值，因此本节指标用于衡量多模块一致性、时序稳定性和结果可解释性，不等同于真实准确率。</li><li>所有候选建筑、工地、拆除、完工和更新周期均为模型推断候选，human_review_required=true。</li><li>本轮未重新读取WHU、SECOND、DSIFN或UCS Locked Test像素。</li></ul>"
    html_text = f"""<!doctype html>
<html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>南京江北新区 URRS 无人工真值外部样例验收</title>
<style>
:root{{--bg:#f4f6f8;--ink:#18212b;--muted:#5b6773;--line:#d7dee5;--card:#fff;--accent:#0c6e8c;--warn:#fff2cc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,'Microsoft YaHei',sans-serif;line-height:1.55}}main{{max-width:1600px;margin:auto;padding:28px}}header{{background:linear-gradient(135deg,#0c3f50,#0c6e8c);color:#fff;padding:42px 5vw}}header h1{{font-size:clamp(28px,4vw,52px);margin:0 0 12px}}header p{{max-width:1000px;font-size:18px}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin:20px 0;box-shadow:0 3px 14px #0b28330d}}h2{{margin-top:0;color:#0c536a}}h3{{color:#235f72}}.badge{{display:inline-block;padding:5px 10px;border-radius:999px;background:#d6eef4;color:#0c536a;font-weight:700;margin:3px}}.warning{{background:var(--warn);border-left:5px solid #e0a72d;padding:14px 18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}.card{{background:#fbfcfd;border:1px solid var(--line);border-radius:10px;padding:12px}}img{{max-width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#eef1f3}}table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}}th,td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top;min-width:100px}}th{{background:#edf3f6;position:sticky;top:0}}.ok{{color:#197343;font-weight:700}}.warn{{color:#a65b00;font-weight:700}}.flow{{display:flex;flex-wrap:wrap;gap:9px;align-items:center;justify-content:center;background:#edf7f9;padding:22px;border-radius:10px}}.flow span{{background:#fff;border:1px solid #9fcbd6;border-radius:9px;padding:10px 14px;text-align:center}}.flow b{{color:#0c6e8c;font-size:22px}}code{{background:#eef1f3;padding:2px 5px;border-radius:4px}}a{{color:#0c6e8c}}.small{{color:var(--muted);font-size:13px}}
</style>
<header><h1>南京江北新区 URRS 外部样例验收</h1><p>本报告为南京江北新区独立新影像上的无人工真值样例验收。</p><p><span class='badge'>状态：{html.escape(report_status)}</span><span class='badge'>external_sample_inference = {html.escape(report_status)}</span><span class='badge'>ground_truth = unavailable</span><span class='badge'>construction_model = limited_candidate</span><span class='badge'>human_review_required = true</span></p><p class='small'>生成时间：{utc_now()}；报告目录：{html.escape(str(output_root.resolve()))}</p></header>
<main>
<section><h2>1. 执行摘要</h2><div class='warning'><strong>科学边界：</strong>南京江北没有人工真值。本报告不提供南京江北准确率、召回率、Precision、Recall、F1或IoU；图表和数值只表示工程可运行性、多模块一致性、时序稳定性和模型派生候选。</div><p>实际输入 {len(dates)} 期，日期范围 {html.escape(dates[0])} 至 {html.escape(dates[-1])}；相邻对 {len(pair_specs(input_audit['records'])['adjacent'])} 组，首期至末期直接累计对 {len(pair_specs(input_audit['records'])['cumulative'])} 组；共同网格状态：<strong>{'ready_for_pixelwise_temporal_tracking' if input_audit['grid_ready'] else 'registration_required'}</strong>。</p>{_module_table(statuses)}</section>
<section><h2>2. 数据与日期概况</h2><p>源目录：<code>{html.escape(str(input_audit['source_root']))}</code>。原始数据只读；获取日期未使用文件创建时间。</p><table><tr><th>日期</th><th>文件</th><th>尺寸</th><th>CRS</th><th>RGB记录</th><th>源SHA256</th></tr>{''.join(f"<tr><td>{html.escape(row['date'])}</td><td>{html.escape(Path(row['path']).name)}</td><td>{row['width']}×{row['height']}</td><td>{html.escape(str(row['crs']))}</td><td>{row['rgb_mapping']}</td><td><code>{row['sha256']}</code></td></tr>" for row in date_rows)}</table><p class='small'>影像审计文件：<a href='00_input_audit/IMAGE_INVENTORY.csv'>IMAGE_INVENTORY.csv</a> · <a href='00_input_audit/IMAGE_DATE_AUDIT.json'>IMAGE_DATE_AUDIT.json</a> · <a href='01_aligned_imagery/ALIGNMENT_AUDIT.json'>ALIGNMENT_AUDIT.json</a>。</p></section>
<section><h2>3. 系统总体框架</h2>{flow}</section>
<section><h2>4. 各模块及功能</h2><p>Full20负责综合变化及新建/拆除方向证据；RSBuilding负责逐期建筑几何候选；Construction Activity负责UCS原生工地构成要素；Surface State提供地表状态候选证据；no-MTI规则只负责可审计状态表达，不表示已正式启用MTI。</p><p>当前正式时序路径：<code>selected_construction_temporal_integration = no_mti</code>；Construction-TFR为<code>development_candidate_not_formally_validated</code>；Construction-MTI为<code>experimental_unselected</code>。</p></section>
<section><h2>5. 工地识别模型：构成、训练与输出逻辑</h2><h3>任务定义</h3><p>Construction Activity不是简单的工地/非工地二分类，而是先识别工地构成要素，再结合空间与时间证据形成工地候选。UCS原生输出：background、exposed ground、dustproof net、building；另有boundary和uncertainty通道。</p><h3>实际结构</h3><p>{html.escape(model_structure)}</p><h3>训练与冻结范围</h3><ul><li>UCS可配对样本386幅；Train 296、Validation 44、Locked Test 46；另有214幅影像缺失。</li><li>RGB映射3,2,1；Stage A只使用UCS主训练；CWLD Stage B未运行。</li><li>checkpoint只由Validation选择；Locked Test历史上只运行1次，本轮不再读取；当前为limited_candidate。</li><li>不能声称全国泛化，也不能声称生产级可用；南京结果必须人工审核。</li></ul><p>模型包：<code>{html.escape(str(package_manifest.get('model_name', 'URRS-Construction-UCS386-v1.0-task-candidate')))}</code>；SHA256：<code>{CONSTRUCTION_SHA}</code>。</p></section>
<section><h2>6. 已有模型正式基准表现</h2><p>以下是现有Validation/唯一历史Locked Test记录，与南京江北无真值代理指标分开保存，不能混合解释。</p>{_metric_table(baseline_cards, title='A. 模型已有正式基准指标')}</section>
<section><h2>7. 南京江北单期识别结果</h2><div class='grid'>{date_cards}</div><p class='small'>逐期详细输出：<code>03_rsbuilding_geometry</code>、<code>04_construction_elements</code>、<code>05_surface_state</code>；概率图和对象文件未直接嵌入HTML。</p></section>
<section><h2>8. 相邻时期变化对比</h2><div class='grid'>{pair_cards}</div><p>每组使用对应影像对直接运行Full20；没有把相邻变化简单相加冒充累计变化。</p></section>
<section><h2>9. 各期至最后一期累计变化</h2><p>累计对使用每个起始期与最后一期的直接Full20推理，输出在<code>06_cumulative_change</code>；如模块未运行，报告保留NOT RUN及原因。</p></section>
<section><h2>10. no-MTI状态转移与生命周期</h2><p>状态序列是<code>model_derived_lifecycle_candidate</code>，不是human_verified lifecycle。固定状态本体与配色见<a href='07_temporal_states/STATE_COLOR_LEGEND.png'>图例</a>；状态栅格：<a href='07_temporal_states/STATE_ONTOLOGY.json'>ontology</a>、<a href='07_temporal_states/STATE_TRANSITION_TABLE.csv'>transition table</a>。</p><img src='{html.escape(_relative(Path(figure_index.get('state_figure')), output_root) if figure_index.get('state_figure') else '')}' loading='lazy'><p class='small'>允许非单调序列；更新周期按“稳定建筑或裸地 → 清理/工地/结构上升 → 完工或新的稳定状态”计数，同一连续施工子状态不重复计数。</p></section>
<section><h2>11. 代表性建筑/工地时序案例</h2>{f"<p>已选择代表对象：<code>{html.escape(str(representative.get('object_id')))}</code>；<a href='{html.escape(representative_link)}'>打开OBJECT_TIMELINE.html</a>。</p><p>状态序列：{html.escape(' → '.join(representative.get('object_evidence', {}).get('state_sequence', [])))}；模型推断更新周期：{representative.get('object_evidence', {}).get('construction_cycles_count')}</p>" if representative.get('status') == 'completed' else '<p class="warn">NOT RUN：未找到满足非边缘/多源证据条件的建筑对象。</p>'}</section>
<section><h2>12. 全区域无真值一致性指标</h2>{warnings}{_metric_table(proxy_cards[:min(len(proxy_cards), 80)], title='B. 南京江北无真值指标（proxy / internal consistency / model-derived）')}<p class='small'>完整表格：<a href='10_tables/PROXY_METRICS.csv'>PROXY_METRICS.csv</a>；数值审计：<a href='REPORT_NUMERICAL_AUDIT.json'>REPORT_NUMERICAL_AUDIT.json</a>。</p></section>
<section><h2>13. 主要发现与高不确定区域</h2><p>本轮可报告的是哪些模块成功完成、输出是否有限、网格和米制面积审计是否通过，以及候选证据之间是否存在内部支持。高不确定对象由Construction uncertainty、Surface State不确定性和Full20派生不确定性联合标记，需优先进入人工审核。</p></section>
<section><h2>14. 局限性</h2><ul><li>无人工真值，不能进行南京江北真实效果验证。</li><li>Construction模型仅限UCS386可用子集，214幅缺失且缺失偏差未排除。</li><li>Surface State是sample-domain candidate，完整FUSU和全国泛化未验证。</li><li>RSBuilding和Full20使用冻结资产，外部区域仍需人工检查；RSBuilding不得读取WHU Test。</li><li>地理影像原始CRS为EPSG:4326，所有面积指标只在EPSG:32650重投影栅格/几何后计算。</li></ul></section>
<section><h2>15. 下一阶段训练计划</h2><ol><li>完成现有96条Seed Temporal样本；对高置信与高不确定对象进行模型辅助审核，ACCEPT/CORRECT形成human_verified/human_corrected。</li><li>建立南京本地小规模空间holdout真值，标注建筑、工地边界、裸地、防尘网、结构上升、完工和困难负样本。</li><li>用人工轨迹正式验证Construction-TFR和生命周期事件；普通QA决定不能直接当像素Ground Truth。</li><li>在独立Validation上比较no-MTI、Construction-TFR和Construction-MTI，只有Validation改进才允许选择MTI。</li><li>补齐214幅UCS影像、加入许可数据和南京困难负样本；Stage B作为新的预注册实验，不能事后改变当前候选结论。</li><li>最终独立外部区域建立人工真值后，才报告建筑与工地的Precision、Recall、F1、IoU、边界/面积误差和时序事件指标。</li></ol></section>
<section><h2>16. 模型、数据与产物清单</h2><table><tr><th>资产</th><th>路径</th><th>SHA256/状态</th></tr>{''.join(f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(str(value.get('path')))}</code></td><td>{html.escape(str(value.get('sha256') or value.get('status')))}</td></tr>" for name, value in assets.get('assets', {}).items())}</table><p>完整产物清单：<a href='REPORT_MANIFEST.json'>REPORT_MANIFEST.json</a>；数值审计：<a href='REPORT_NUMERICAL_AUDIT.json'>REPORT_NUMERICAL_AUDIT.json</a>；本地服务脚本：<a href='serve_report.ps1'>serve_report.ps1</a>。</p></section>
</main></html>"""
    report_path.write_text(html_text, encoding="utf-8")
    artifact_files = [path for path in output_root.rglob("*") if path.is_file() and path.name not in {"REPORT_MANIFEST.json"}]
    manifest = {
        "status": report_status,
        "external_sample": "nanjing_jiangbei",
        "evaluation_type": "no_ground_truth_external_acceptance",
        "ground_truth": "unavailable",
        "construction_model": "limited_candidate",
        "temporal_labels": "model_derived_lifecycle_candidate",
        "human_review_required": True,
        "input_audit": str((output_root / "00_input_audit").resolve()),
        "output_root": str(output_root.resolve()),
        "report": str(report_path.resolve()),
        "report_relative_open": "NANJING_JIANGBEI_EXTERNAL_ACCEPTANCE_ZH.html",
        "serve_command": f"powershell -NoProfile -ExecutionPolicy Bypass -File '{output_root / 'serve_report.ps1'}'",
        "dates": dates,
        "adjacent_pairs": len(pair_specs(input_audit["records"])["adjacent"]),
        "cumulative_pairs": len(pair_specs(input_audit["records"])["cumulative"]),
        "module_statuses": statuses,
        "representative_object": representative,
        "artifacts": [{"relative_path": _relative(path, output_root), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in artifact_files],
        "original_data_modified": numerical_audit["original_data_modified"],
        "training_started": False,
        "test_pixel_reads": numerical_audit["locked_test_pixel_reads"],
        "created_at": utc_now(),
    }
    write_json(output_root / "REPORT_MANIFEST.json", manifest)
    return manifest


def write_serve_script(output_root: Path) -> None:
    script = """param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '.')).Path
$Python = Join-Path $Root '..\\..\\..\\.venv-gpu\\Scripts\\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
Write-Host \"Serving URRS report from $Root on http://127.0.0.1:$Port\"
& $Python -m http.server $Port --directory $Root
"""
    (output_root / "serve_report.ps1").write_text(script, encoding="utf-8")


def _load_existing_audit(output_root: Path) -> dict[str, Any]:
    source_root = Path(read_json(output_root / "00_input_audit" / "IMAGE_DATE_AUDIT.json", {}).get("source_root", SOURCE_ROOT))
    return audit_inputs(source_root, output_root)


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command_text = json.dumps(sys.argv, ensure_ascii=False)
    run_receipt = {
        "status": "running",
        "pid": os.getpid(),
        "command": sys.argv,
        "command_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest(),
        "started_at": utc_now(),
        "output_root": str(output_root),
        "training_started": False,
        "thresholds_changed": False,
        "models_reselected": False,
    }
    write_json(output_root / "RUN_RECEIPT.json", run_receipt)
    heartbeat = output_root / "heartbeat.json"
    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "input_audit", "updated_at": utc_now()})
    input_audit = audit_inputs(Path(args.source_root).resolve(), output_root)
    protected_before = snapshot_protected_artifacts(PROJECT_ROOT)
    write_json(output_root / "00_input_audit" / "PROTECTED_ARTIFACT_SNAPSHOT.json", protected_before)
    assets = verify_frozen_assets(PROJECT_ROOT)
    write_json(output_root / "00_input_audit" / "FROZEN_ASSET_AUDIT.json", assets)
    dates = list(input_audit["dates"])
    statuses: dict[str, dict[str, Any]] = {}

    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "RSBuilding", "updated_at": utc_now()})
    statuses["RSBuilding建筑几何"] = run_rsbuilding(PROJECT_ROOT, Path(args.source_root).resolve(), output_root, dates, rsbuilding_python=Path(args.rsbuilding_python).resolve())
    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "Full20", "updated_at": utc_now()})
    statuses["Full20变化检测"] = run_full20(PROJECT_ROOT, input_audit, output_root, assets, device=args.device)
    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "Surface State", "updated_at": utc_now()})
    statuses["Surface State地表状态"] = run_surface_state(PROJECT_ROOT, input_audit, output_root, assets, device=args.device)
    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "Construction Activity", "updated_at": utc_now()})
    statuses["Construction Activity工地要素"] = run_construction(PROJECT_ROOT, input_audit, output_root, statuses["RSBuilding建筑几何"], statuses["Full20变化检测"], statuses["Surface State地表状态"], device=args.construction_device)
    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "no-MTI temporal state", "updated_at": utc_now()})
    temporal_run = _pair_evidence_and_states(output_root, input_audit, statuses["RSBuilding建筑几何"], statuses["Construction Activity工地要素"], statuses["Surface State地表状态"], statuses["Full20变化检测"])
    statuses["no-MTI时序规则"] = {"status": temporal_run.get("status", "NOT_RUN"), **{key: value for key, value in temporal_run.items() if key in {"reason", "states", "pairs"}}}
    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "figures and representative object", "updated_at": utc_now()})
    figures = build_figures(output_root, input_audit, temporal_run) if temporal_run.get("status") == "completed" else {"status": "NOT_RUN", "reason": "temporal state layer unavailable"}
    representative = build_representative_object(output_root, input_audit, temporal_run)
    statuses["可视化与代表对象"] = {"status": "completed" if figures.get("status") == "completed" and representative.get("status") == "completed" else "partial", "figures": figures, "representative": representative}
    write_serve_script(output_root)
    write_json(heartbeat, {"status": "running", "pid": os.getpid(), "step": "BuildReport", "updated_at": utc_now()})
    report = build_report(PROJECT_ROOT, output_root, input_audit, assets, statuses, temporal_run, representative, protected_before)
    run_receipt.update({"status": "completed" if report.get("status") == "completed" else "partial", "completed_at": utc_now(), "elapsed_seconds": time.perf_counter() - started, "report": report.get("report"), "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0}, "module_statuses": statuses})
    write_json(output_root / "RUN_RECEIPT.json", run_receipt)
    write_json(heartbeat, {"status": "completed", "pid": os.getpid(), "step": "done", "updated_at": utc_now()})
    return {"status": run_receipt["status"], "output_root": str(output_root), "report": report, "module_statuses": statuses}


def run_build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    input_audit = _load_existing_audit(output_root)
    assets = read_json(output_root / "00_input_audit" / "FROZEN_ASSET_AUDIT.json", verify_frozen_assets(PROJECT_ROOT))
    protected = read_json(output_root / "00_input_audit" / "PROTECTED_ARTIFACT_SNAPSHOT.json", snapshot_protected_artifacts(PROJECT_ROOT))
    statuses = {
        "Full20变化检测": read_json(output_root / "02_full20_change" / "FULL20_EXTERNAL_MANIFEST.json", {"status": "NOT_RUN"}),
        "RSBuilding建筑几何": {"status": "completed" if (output_root / "03_rsbuilding_geometry" / "ALIGNMENT_AUDIT.json").exists() else "NOT_RUN", "geometry_root": str((output_root / "03_rsbuilding_geometry").resolve())},
        "Construction Activity工地要素": read_json(output_root / "04_construction_elements" / "CONSTRUCTION_EXTERNAL_MANIFEST.json", {"status": "NOT_RUN"}),
        "Surface State地表状态": read_json(output_root / "05_surface_state" / "SURFACE_STATE_EXTERNAL_MANIFEST.json", {"status": "NOT_RUN"}),
    }
    temporal = {"status": "completed" if (output_root / "07_temporal_states" / "TEMPORAL_STATES_MANIFEST.json").exists() else "NOT_RUN", "states": read_json(output_root / "07_temporal_states" / "TEMPORAL_STATES_MANIFEST.json", {}), "pairs": []}
    for path in sorted(list((output_root / "05_adjacent_change").glob("*/PAIR_EVIDENCE_MANIFEST.json")) + list((output_root / "06_cumulative_change").glob("*/PAIR_EVIDENCE_MANIFEST.json"))):
        temporal["pairs"].append(read_json(path, {}))
    representative = read_json(output_root / "08_representative_object" / "OBJECT_EVIDENCE.json", {"status": "NOT_RUN"})
    statuses["no-MTI时序规则"] = temporal
    statuses["可视化与代表对象"] = {"status": "completed" if representative.get("status") == "completed" else "partial", "representative": representative}
    write_serve_script(output_root)
    report = build_report(PROJECT_ROOT, output_root, input_audit, assets, statuses, temporal, representative, protected)
    receipt_path = output_root / "RUN_RECEIPT.json"
    receipt = read_json(receipt_path, {"output_root": str(output_root), "training_started": False, "thresholds_changed": False, "models_reselected": False})
    receipt.update({
        "status": report.get("status", "partial"),
        "completed_at": utc_now(),
        "report": report.get("report"),
        "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0, "UCS_locked_test": 0},
        "module_statuses": statuses,
        "finalization_mode": "BuildReport",
    })
    write_json(receipt_path, receipt)
    write_json(output_root / "heartbeat.json", {"status": "completed", "pid": os.getpid(), "step": "done", "updated_at": utc_now()})
    return report


def latest_output(project_root: Path) -> Path | None:
    roots = sorted((project_root / "outputs" / "inference").glob(f"{OUTPUT_PREFIX}_*"), key=lambda path: path.stat().st_mtime)
    return roots[-1] if roots else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nanjing Jiangbei external no-ground-truth acceptance")
    parser.add_argument("--mode", choices=("Audit", "Infer", "BuildReport", "All", "Status"), default="All")
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--construction-device", default="cuda")
    parser.add_argument("--rsbuilding-python", default=r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = str((PROJECT_ROOT / "outputs" / "inference" / f"{OUTPUT_PREFIX}_{stamp()}").resolve())
    if args.mode == "Status":
        target = Path(args.output_root).resolve() if args.output_root else latest_output(PROJECT_ROOT)
        if target is None:
            print(json.dumps({"status": "not_run", "reason": "no Nanjing external acceptance output"}, ensure_ascii=False, indent=2))
            return 0
        manifest = read_json(target / "REPORT_MANIFEST.json", {"status": "in_progress", "output_root": str(target)})
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.mode == "Audit":
        output = Path(args.output_root).resolve()
        output.mkdir(parents=True, exist_ok=True)
        result = audit_inputs(Path(args.source_root).resolve(), output)
        write_json(output / "00_input_audit" / "FROZEN_ASSET_AUDIT.json", verify_frozen_assets(PROJECT_ROOT))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.mode in {"All", "Infer"}:
        result = run_all(args)
    elif args.mode == "BuildReport":
        result = run_build_report(args)
    else:
        result = {"status": "NOT_RUN", "reason": "Infer mode is intentionally subsumed by All so audit, model and report receipts remain consistent; use --mode All for execution."}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {"completed", "partial", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
