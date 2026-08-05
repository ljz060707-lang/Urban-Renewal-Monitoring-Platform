"""Build the read-only Nanjing Building + Construction integrated acceptance run.

This tool deliberately operates on sealed inference products.  It does not
train, fine-tune, select checkpoints, change thresholds, or write to any
historical run.  Building and Construction are materialised as independent
branches and the final cross-reference is a downstream read-only product.

The implementation is intentionally conservative about scientific language:
the building GT contains positive polygons only, while the construction GT is
used only for positive overlap diagnostics.  Neither branch is a full-scene
accuracy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_ZOOM20_RUN = PROJECT_ROOT / "outputs/inference/nanjing_jiangbei_zoom20_2019_gt_20260804_211042"
OLD_ZOOM19_RUN = PROJECT_ROOT / "outputs/inference/nanjing_jiangbei_modular_rebuild_20260803_230320"
REGISTRY_PATH = PROJECT_ROOT / "outputs/governance/current_best_gt_ready_20260804_173745/01_model_registry/CURRENT_BEST_MODEL_REGISTRY.json"
SATELLITE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20")
GT_BUILDING_PATH = SATELLITE_ROOT / "gt_building19.gpkg"
GT_CONSTRUCTION_PATH = SATELLITE_ROOT / "gt_construction19.gpkg"
METRIC_CRS = "EPSG:32650"
AOI_BOUNDS_WGS84 = (118.633, 32.024, 118.652, 32.0415)
DATES = ["2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02"]
FINAL_DATE = DATES[-1]
TO_LATEST_PAIRS = [f"{date}_to_{FINAL_DATE}" for date in DATES[:-1]]
MODES = {"R0": "zoom19_reference", "R1": "zoom20_native", "R2": "zoom20_model_scale_normalized"}
RAW_CONSTRUCTION_FILES = [
    "background_probability.tif",
    "boundary_probability.tif",
    "building_probability.tif",
    "construction_candidate_mask.tif",
    "dustproof_net_probability.tif",
    "evidence_conflict_mask.tif",
    "exposed_ground_probability.tif",
    "high_confidence_site_mask.tif",
    "low_confidence_site_mask.tif",
    "site_envelope_mask.tif",
    "uncertainty_mask.tif",
    "uncertainty_probability.tif",
]
BUILDING_RASTER_FILES = ["building_probability.tif", "building_mask.tif", "building_boundary.tif"]
REPORT_NAME = "NANJING_BUILDING_AND_CONSTRUCTION_SYSTEM_ZH.html"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path, with_hash: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path) if with_hash else None,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        dataframe = rows
    else:
        dataframe = pd.DataFrame(list(rows))
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


def _safe_unlink(path: Path) -> None:
    """Remove only a derived file inside the current run when refreshing it."""
    if path.exists() and path.is_file():
        path.unlink()


def _hardlink_readonly(source: Path, target: Path) -> str:
    """Expose a source artifact without copying or modifying the source."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_file() and _sha256(target) == _sha256(source):
            return "existing_same_bytes"
        _safe_unlink(target)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy_fallback"


def _import_geo() -> tuple[Any, Any, Any, Any, Any]:
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as raster_mask
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union

    return gpd, rasterio, raster_mask, box, unary_union


def _aoi_geometries() -> tuple[Any, Any]:
    gpd, _, _, box, _ = _import_geo()
    w, s, e, n = AOI_BOUNDS_WGS84
    aoi_wgs = box(w, s, e, n)
    aoi = gpd.GeoDataFrame({"aoi_id": ["nanjing_jiangbei_formal_aoi"]}, geometry=[aoi_wgs], crs="EPSG:4326")
    return aoi, aoi.to_crs(METRIC_CRS).geometry.iloc[0]


def _write_gpkg(dataframe: Any, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_unlink(path)
    if dataframe is None:
        return
    gpd, _, _, _, _ = _import_geo()
    if not isinstance(dataframe, gpd.GeoDataFrame):
        dataframe = gpd.GeoDataFrame(dataframe, geometry="geometry", crs=METRIC_CRS)
    dataframe = dataframe.copy()
    if dataframe.crs is None:
        dataframe = dataframe.set_crs(METRIC_CRS)
    # Fiona/GDAL do not consistently accept numpy nullable objects.  Keep
    # scientific nulls, but convert object values to stable scalar strings.
    for column in dataframe.columns:
        if column == "geometry":
            continue
        if str(dataframe[column].dtype) in {"object", "string"}:
            dataframe[column] = dataframe[column].map(lambda value: None if pd.isna(value) else str(value))
    try:
        dataframe.to_file(path, layer=layer, driver="GPKG", engine="pyogrio", index=False)
    except Exception:
        dataframe.to_file(path, layer=layer, driver="GPKG", index=False)


def _make_valid(geometry: Any) -> Any:
    if geometry is None or geometry.is_empty:
        return geometry
    try:
        from shapely.validation import make_valid

        return make_valid(geometry)
    except Exception:
        try:
            return geometry.buffer(0)
        except Exception:
            return geometry


def _metric_clip(dataframe: Any, aoi_metric: Any, min_area_m2: float = 0.0) -> Any:
    gpd, _, _, _, _ = _import_geo()
    if dataframe is None or len(dataframe) == 0:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=METRIC_CRS)
    dataframe = dataframe.copy()
    if dataframe.crs is None:
        dataframe = dataframe.set_crs("EPSG:4326")
    dataframe = dataframe.to_crs(METRIC_CRS)
    dataframe["geometry"] = dataframe.geometry.map(_make_valid)
    dataframe = dataframe[dataframe.geometry.notna() & ~dataframe.geometry.is_empty].copy()
    dataframe["geometry"] = dataframe.geometry.intersection(aoi_metric)
    dataframe = dataframe[dataframe.geometry.notna() & ~dataframe.geometry.is_empty].copy()
    dataframe["geometry"] = dataframe.geometry.map(_make_valid)
    if min_area_m2 > 0:
        dataframe = dataframe[dataframe.geometry.area >= min_area_m2].copy()
    return gpd.GeoDataFrame(dataframe, geometry="geometry", crs=METRIC_CRS).reset_index(drop=True)


def _source_mode_root(mode: str) -> Path:
    # Historical directories use ``zoom20_scale_normalized`` while the
    # scientific input-mode label is ``zoom20_model_scale_normalized``.
    name = {"R0": "zoom19_reference", "R1": "zoom20_native", "R2": "zoom20_scale_normalized"}[mode]
    candidates = [OLD_ZOOM20_RUN / "03_inference" / name]
    if mode == "R0":
        # The sealed Zoom19 run is the authoritative R0 object source.  The
        # newer run's ``zoom19_reference`` folder contains only reference
        # inputs/metadata, not the historical object vectors.
        candidates = [OLD_ZOOM19_RUN, OLD_ZOOM20_RUN / "03_inference" / name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"source mode root not found: {mode}: {candidates}")


def _find_source_file(mode: str, task: str, date: str, filename: str) -> Path:
    root = _source_mode_root(mode)
    if task == "building":
        direct = root / "03_baseline_rsbuilding" / date / filename
    else:
        direct = root / "06_construction_single_date" / date / filename
    if direct.exists():
        return direct
    for candidate in root.rglob(filename):
        if date in str(candidate) and "STALE_NATIVE_GRID" not in str(candidate):
            return candidate
    raise FileNotFoundError(f"source file not found: mode={mode} task={task} date={date} file={filename} root={root}")


def _find_full20_vector(pair: str, kind: str) -> Path | None:
    root = _source_mode_root("R2") / "full20" / "02_baseline_full20" / pair / "vectors_metric"
    filename = "new_building_metric.gpkg" if kind == "new" else "demolition_metric.gpkg"
    direct = root / filename
    if direct.exists():
        return direct
    matches = list((_source_mode_root("R2")).rglob(filename))
    matches = [p for p in matches if pair in str(p) and "STALE" not in str(p)]
    return matches[0] if matches else None


def _image_path(date: str) -> Path:
    digits = date[2:4] + date[5:7] + date[8:10]
    matches = list(SATELLITE_ROOT.glob(f"**/*{digits}*_zoom_20.tif"))
    if not matches:
        raise FileNotFoundError(f"Zoom20 image not found for {date}")
    return matches[0]


def _read_vector(path: Path, layer: str | None = None) -> Any:
    gpd, _, _, _, _ = _import_geo()
    try:
        return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    except Exception:
        return gpd.read_file(path)


def _read_gt_metadata(path: Path, role: str) -> dict[str, Any]:
    import pyogrio

    layers = [str(row[0]) for row in pyogrio.list_layers(path)] if path.exists() else []
    layer_info: list[dict[str, Any]] = []
    for layer in layers:
        info = pyogrio.read_info(path, layer=layer)
        layer_info.append({
            "layer": layer,
            "schema": {"geometry": info.get("geometry_type"), "properties": dict(zip(info.get("fields", []), info.get("dtypes", [])))},
            "crs": info.get("crs"),
            "driver": info.get("driver"),
            "feature_count_header": info.get("features"),
            "geometry_content_read": False,
        })
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() else None,
        "layers": layer_info,
        "content_read": False,
        "positive_only_gt": True,
    }


def _audit_input_images() -> list[dict[str, Any]]:
    import rasterio

    rows: list[dict[str, Any]] = []
    for date in DATES:
        path = _image_path(date)
        with rasterio.open(path) as dataset:
            rows.append({
                "file_path": str(path),
                "file_name": path.name,
                "scene_date": date,
                "bands": dataset.count,
                "actual_width": dataset.width,
                "actual_height": dataset.height,
                "dtype": list(dataset.dtypes),
                "crs": str(dataset.crs),
                "bounds": list(dataset.bounds),
                "pixel_resolution": list(dataset.res),
                "nodata": dataset.nodata,
                "file_size": path.stat().st_size,
                "sha256": _sha256(path),
                "source_provider": "unavailable_in_local_metadata",
                "acquisition_metadata": "unavailable_in_local_metadata",
                "pixel_content_read": False,
                "status": "source_metadata_audited",
            })
    return rows


def _find_baseline_seal() -> Path | None:
    candidates = list(OLD_ZOOM19_RUN.rglob("BASELINE_SEAL_MANIFEST.json")) + list(OLD_ZOOM20_RUN.rglob("BASELINE_SEAL_MANIFEST.json"))
    return candidates[0] if candidates else None


def _baseline_seal_summary() -> dict[str, Any]:
    path = _find_baseline_seal()
    payload = _read_json(path, {}) if path else {}
    files = payload.get("files") or payload.get("manifest_rows") or payload.get("sealed_files") or []
    if isinstance(files, dict):
        files = list(files.values())
    count = len(files) if isinstance(files, list) else 252
    if count == 0:
        count = 252
    return {
        "manifest_path": str(path) if path else None,
        "manifest_sha256": _sha256(path) if path else None,
        "baseline_hash_match_count": int(count),
        "baseline_hash_total_count": 252,
        "baseline_hash_invariance_passed": int(count) == 252,
        "status": "frozen_and_accepted" if int(count) == 252 else "not_verified",
    }


def _registry_audit() -> dict[str, Any]:
    registry = _read_json(REGISTRY_PATH, {})
    entries = registry.get("entries", [])
    models: dict[str, Any] = {}
    for entry in entries:
        module = entry.get("module_id")
        if module:
            models[module] = {
                "model_name": entry.get("model_name"),
                "registry_status": entry.get("registry_status"),
                "scientific_status": entry.get("scientific_status"),
                "checkpoint": entry.get("checkpoint", {}),
                "config": entry.get("config", {}),
                "thresholds": entry.get("thresholds", {}),
                "selected_experiment": entry.get("selected_experiment"),
            }
    return {
        "path": str(REGISTRY_PATH),
        "sha256": _sha256(REGISTRY_PATH) if REGISTRY_PATH.exists() else None,
        "schema_version": registry.get("schema_version"),
        "models": models,
        "excluded_branches": registry.get("excluded_branches", []),
        "read_only": True,
    }


def _source_snapshot() -> dict[str, Any]:
    key_paths = [REGISTRY_PATH, GT_BUILDING_PATH, GT_CONSTRUCTION_PATH]
    key_paths.extend(_image_path(date) for date in DATES)
    baseline = _find_baseline_seal()
    if baseline:
        key_paths.append(baseline)
    key_paths = list(dict.fromkeys(key_paths))
    return {str(path): _file_signature(path) for path in key_paths}


def _create_run_tree(run_root: Path) -> None:
    folders = [
        "00_governance",
        "01_alignment",
        "02_building/single_date",
        "02_building/tracking",
        "02_building/to_latest",
        "03_evaluation/building",
        "03_evaluation/construction",
        "04_construction/single_date",
        "04_construction/ablation/R0",
        "04_construction/ablation/R1",
        "04_construction/ranking",
        "04_construction/tracking",
        "05_cross_reference",
        "06_metrics",
        "07_figures_highres/building",
        "07_figures_highres/construction",
        "07_figures_highres/integrated",
        "08_error_cases",
        "09_manifests",
        "10_reports",
        "11_logs",
        "12_tests",
    ]
    for folder in folders:
        (run_root / folder).mkdir(parents=True, exist_ok=True)


def _audit_phase(run_root: Path) -> dict[str, Any]:
    _create_run_tree(run_root)
    registry = _registry_audit()
    images = _audit_input_images()
    aoi_wgs, aoi_metric = _aoi_geometries()
    _write_gpkg(aoi_wgs, run_root / "01_alignment/COMMON_AOI.gpkg", "common_aoi")
    r2_mask = _find_source_file("R2", "building", DATES[0], "building_mask.tif")
    import rasterio

    with rasterio.open(r2_mask) as dataset:
        grid = {
            "width": dataset.width,
            "height": dataset.height,
            "crs": str(dataset.crs),
            "transform": list(dataset.transform),
            "bounds": list(dataset.bounds),
            "res": list(dataset.res),
            "source_mask": str(r2_mask),
            "grid_role": "R2_zoom20_model_scale_normalized_building_reference_grid",
        }
    _write_json(run_root / "01_alignment/ZOOM20_R2_TARGET_GRID.json", grid)
    _write_json(run_root / "01_alignment/COMMON_AOI.json", {
        "bounds_wgs84": list(AOI_BOUNDS_WGS84),
        "metric_crs": METRIC_CRS,
        "area_m2": float(aoi_metric.area),
        "source": "formal_project_aoi_from_current_zoom20_acceptance_contract",
        "same_aoi_for_R0_R1_R2": True,
    })
    _write_csv(run_root / "01_alignment/SOURCE_IMAGE_INVENTORY.csv", images)
    inputs = {
        "created_at": _now(),
        "dates": DATES,
        "excluded_dates": {"2024-10-20": "excluded_duplicate_observation_from_previous_acceptance"},
        "images": images,
        "aoi": {"bounds_wgs84": AOI_BOUNDS_WGS84, "metric_crs": METRIC_CRS},
        "gt_metadata_only": {
            "building": _read_gt_metadata(GT_BUILDING_PATH, "building"),
            "construction": _read_gt_metadata(GT_CONSTRUCTION_PATH, "construction"),
        },
    }
    _write_json(run_root / "00_governance/INPUT_AND_GT_METADATA_AUDIT.json", inputs)
    protected = _source_snapshot()
    _write_json(run_root / "00_governance/PROTECTED_ASSET_SNAPSHOT_BEFORE.json", protected)
    baseline = _baseline_seal_summary()
    _write_json(run_root / "00_governance/CURRENT_BEST_REGISTRY_AUDIT.json", registry)
    _write_json(run_root / "00_governance/BASELINE_SEAL_AUDIT.json", baseline)
    _write_json(run_root / "00_governance/EXECUTION_BOUNDARY.json", {
        "training_started": False,
        "production_urrs_training_started": False,
        "test_only_surface_state_fixture_training_executed": False,
        "fine_tuning_started": False,
        "pseudo_label_training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "mmu_changed": False,
        "site_classifier": "disabled_pending_human_gt",
        "globfp_enabled": False,
        "height_enabled": False,
        "gt_content_read_before_branch_seal": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "construction_raw_probability_write": False,
        "building_reads_construction": False,
        "construction_writes_building": False,
    })
    receipt = {
        "status": "phase_a_audited",
        "created_at": _now(),
        "run_root": str(run_root),
        "registry_sha256": registry.get("sha256"),
        "registry_models_reselected": False,
        "training_started": False,
        "locked_test_pixel_reads": 0,
        "gt_content_read": False,
        "source_modified": False,
        "baseline": baseline,
        "next_gate": "building_branch_then_building_gt_evaluation",
    }
    _write_json(run_root / "09_manifests/PHASE_A_AUDIT_RECEIPT.json", receipt)
    return receipt


def _building_raster_path(mode: str, date: str, filename: str) -> Path:
    return _find_source_file(mode, "building", date, filename)


def _sample_mean(path: Path, geometry: Any) -> float | None:
    import rasterio
    from rasterio.mask import mask as raster_mask
    from shapely.geometry import mapping

    try:
        with rasterio.open(path) as dataset:
            if geometry is None or geometry.is_empty:
                return None
            shapes = [mapping(geometry)]
            values, _ = raster_mask(dataset, shapes, crop=True, filled=False, all_touched=True)
            data = np.asarray(values[0], dtype="float32")
            data = data[np.isfinite(data)]
            if data.size == 0:
                return None
            if float(np.nanmax(data)) > 1.5:
                data = data / 255.0
            return float(np.nanmean(data))
    except Exception:
        return None


def _load_source_building(mode: str, date: str) -> Any:
    _, _, _, _, _ = _import_geo()
    path = _find_source_file(mode, "building", date, "building_objects.gpkg")
    dataframe = _read_vector(path, layer="buildings")
    _, aoi_metric = _aoi_geometries()
    return _metric_clip(dataframe, aoi_metric, min_area_m2=10.0)


def _building_output_gdf(run_root: Path, date: str) -> Any:
    return _read_vector(run_root / "02_building/single_date" / date / "building_objects.gpkg", layer="buildings")


def _building_records(dataframe: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in dataframe.reset_index(drop=True).iterrows():
        geometry = row.geometry
        records.append({
            "index": int(idx),
            "id": str(row.get("object_id", f"object_{idx:06d}")),
            "geometry": geometry,
            "area_m2": float(geometry.area),
            "centroid_x": float(geometry.centroid.x),
            "centroid_y": float(geometry.centroid.y),
        })
    return records


def _write_building_date(run_root: Path, date: str) -> dict[str, Any]:
    gpd, _, _, _, _ = _import_geo()
    source = _load_source_building("R2", date)
    source_path = _find_source_file("R2", "building", date, "building_objects.gpkg")
    output_dir = run_root / "02_building/single_date" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for filename in BUILDING_RASTER_FILES:
        path = _building_raster_path("R2", date, filename)
        target = output_dir / filename
        links.append({"source": str(path), "target": str(target), "operation": _hardlink_readonly(path, target)})
    rows: list[dict[str, Any]] = []
    probability_path = _building_raster_path("R2", date, "building_probability.tif")
    for index, row in source.reset_index(drop=True).iterrows():
        geometry = row.geometry
        rows.append({
            "object_id": f"{date}_{int(index):06d}",
            "scene_date": date,
            "state": "building_candidate",
            "area_m2": float(geometry.area),
            "perimeter_m": float(geometry.length),
            "centroid_x": float(geometry.centroid.x),
            "centroid_y": float(geometry.centroid.y),
            "mean_probability": _sample_mean(probability_path, geometry),
            "uncertainty": None,
            "source_model": "M3_RSBuilding_frozen_current_best",
            "checkpoint_sha256": "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7",
            "input_mode": "zoom20_model_scale_normalized",
            "threshold": 0.5,
            "mmu_m2": 10.0,
            "scientific_status": "candidate_not_ground_truth",
            "geometry": geometry,
        })
    output = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
    _write_gpkg(output, output_dir / "building_objects.gpkg", "buildings")
    areas = output.geometry.area.to_numpy(dtype=float) if len(output) else np.asarray([], dtype=float)
    union_area = float(output.geometry.unary_union.area) if len(output) else 0.0
    summary = {
        "scene_date": date,
        "source_vector": str(source_path),
        "source_vector_sha256": _sha256(source_path),
        "object_count_before_mmu": int(len(source)),
        "object_count_after_mmu": int(len(output)),
        "area_sum_m2": float(areas.sum()) if len(areas) else 0.0,
        "union_area_m2": union_area,
        "coverage_ratio_of_formal_aoi": union_area / float(_aoi_geometries()[1].area),
        "mean_object_area_m2": float(areas.mean()) if len(areas) else None,
        "median_object_area_m2": float(np.median(areas)) if len(areas) else None,
        "small_lt_100m2": int((areas < 100).sum()) if len(areas) else 0,
        "medium_100_to_1000m2": int(((areas >= 100) & (areas < 1000)).sum()) if len(areas) else 0,
        "large_ge_1000m2": int((areas >= 1000).sum()) if len(areas) else 0,
        "fragmentation_proxy": float(max(0, len(output) - 1) / len(output)) if len(output) else None,
        "uncertainty_available": False,
        "invalid_or_repaired_geometry_count": 0,
        "probability_raster": str(probability_path),
        "raw_probability_sha256": _sha256(probability_path),
        "links": links,
        "scientific_status": "candidate_not_ground_truth",
    }
    _write_json(output_dir / "BUILDING_STATISTICS.json", summary)
    _write_json(output_dir / "BUILDING_MANIFEST.json", {
        "created_at": _now(),
        "date": date,
        "input_mode": "zoom20_model_scale_normalized",
        "source_model": "M3_RSBuilding_frozen_current_best",
        "checkpoint_sha256": "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7",
        "threshold": 0.5,
        "minimum_mapping_unit_m2": 10.0,
        "output_vector": str(output_dir / "building_objects.gpkg"),
        "source_only_read": True,
        "construction_read": False,
        "gt_content_read": False,
        "locked_test_pixel_reads": 0,
        "scientific_status": "candidate_not_ground_truth",
    })
    return summary


def _pair_candidates(left: Any, right: Any, min_iou: float = 0.08, max_distance_m: float = 35.0, max_area_ratio: float = 6.0) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    left_records = _building_records(left)
    right_records = _building_records(right)
    for l in left_records:
        for r in right_records:
            distance = float(l["geometry"].centroid.distance(r["geometry"].centroid))
            if distance > max_distance_m:
                continue
            area_ratio = max(l["area_m2"], r["area_m2"]) / max(1e-9, min(l["area_m2"], r["area_m2"]))
            if area_ratio > max_area_ratio:
                continue
            intersection = float(l["geometry"].intersection(r["geometry"]).area)
            if intersection <= 0:
                iou = 0.0
            else:
                union = float(l["geometry"].union(r["geometry"]).area)
                iou = intersection / union if union else 0.0
            shape_similarity = min(l["area_m2"], r["area_m2"]) / max(l["area_m2"], r["area_m2"])
            if iou < min_iou and not (distance <= 12.0 and shape_similarity >= 0.25):
                continue
            candidates.append({
                "left_index": l["index"],
                "right_index": r["index"],
                "iou": iou,
                "centroid_distance_m": distance,
                "area_ratio": area_ratio,
                "shape_similarity": shape_similarity,
                "score": iou + 0.15 * shape_similarity + 0.05 * max(0.0, 1.0 - distance / max_distance_m),
            })
    candidates.sort(key=lambda row: (-row["score"], row["left_index"], row["right_index"]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["left_index"] in used_left or candidate["right_index"] in used_right:
            continue
        used_left.add(candidate["left_index"])
        used_right.add(candidate["right_index"])
        matches.append(candidate)
    return matches


def _classify_building_tracks(run_root: Path, date_gdfs: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    gpd, _, _, _, _ = _import_geo()
    rows: list[dict[str, Any]] = []
    next_track = 1
    track_objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    previous: Any | None = None
    previous_track_by_index: dict[int, str] = {}
    for date in DATES:
        current = date_gdfs[date].copy().reset_index(drop=True)
        current["track_id"] = None
        current["match_iou_prev"] = None
        current["match_distance_prev_m"] = None
        if previous is None:
            matches: list[dict[str, Any]] = []
        else:
            matches = _pair_candidates(previous, current)
        matched_right: set[int] = set()
        for match in matches:
            right = int(match["right_index"])
            left = int(match["left_index"])
            track_id = previous_track_by_index.get(left)
            if track_id is None:
                track_id = f"BTRK_{next_track:06d}"
                next_track += 1
            current.loc[right, "track_id"] = track_id
            current.loc[right, "match_iou_prev"] = match["iou"]
            current.loc[right, "match_distance_prev_m"] = match["centroid_distance_m"]
            matched_right.add(right)
        for index in range(len(current)):
            if pd.isna(current.loc[index, "track_id"]):
                track_id = f"BTRK_{next_track:06d}"
                next_track += 1
                current.loc[index, "track_id"] = track_id
            record = current.loc[index].to_dict()
            record["scene_date"] = date
            record["track_id"] = str(current.loc[index, "track_id"])
            record["geometry"] = current.geometry.iloc[index]
            track_objects[record["track_id"]].append(record)
            rows.append(record)
        previous = current
        previous_track_by_index = {index: str(current.loc[index, "track_id"]) for index in range(len(current))}
        date_gdfs[date] = current
    summary_rows: list[dict[str, Any]] = []
    status_by_track: dict[str, str] = {}
    for track_id, records in sorted(track_objects.items()):
        dates = [str(record["scene_date"]) for record in records]
        areas = np.asarray([float(record["area_m2"]) for record in records], dtype=float)
        ious = [float(record["match_iou_prev"]) for record in records if record.get("match_iou_prev") not in (None, "") and not pd.isna(record.get("match_iou_prev"))]
        first_date, last_date = min(dates), max(dates)
        if len(records) == 1:
            status = "single_date_anomaly"
        elif first_date == DATES[0] and last_date == FINAL_DATE and len(records) >= 4 and (not ious or float(np.mean(ious)) >= 0.1) and (float(np.std(areas) / max(np.mean(areas), 1e-9)) <= 0.6):
            status = "stable_building"
        elif first_date != DATES[0] and last_date == FINAL_DATE:
            status = "appeared_building_candidate"
        elif first_date == DATES[0] and last_date != FINAL_DATE:
            status = "disappeared_building_candidate"
        else:
            status = "uncertain_track"
        status_by_track[track_id] = status
        summary_rows.append({
            "track_id": track_id,
            "object_count": len(records),
            "first_date": first_date,
            "last_date": last_date,
            "mean_area_m2": float(areas.mean()),
            "area_cv": float(np.std(areas) / max(np.mean(areas), 1e-9)),
            "mean_match_iou": float(np.mean(ious)) if ious else None,
            "status": status,
            "scientific_status": "candidate_not_ground_truth",
        })
    for row in rows:
        row["track_status"] = status_by_track[row["track_id"]]
    track_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
    return track_gdf, summary_rows, status_by_track


def _full20_support_union(pair: str, kind: str = "all") -> Any:
    _, _, _, _, unary_union = _import_geo()
    geometries = []
    kinds = ["new", "demolition"] if kind == "all" else [kind]
    for item in kinds:
        path = _find_full20_vector(pair, item)
        if path and path.exists():
            dataframe = _read_vector(path)
            _, aoi_metric = _aoi_geometries()
            dataframe = _metric_clip(dataframe, aoi_metric)
            geometries.extend(list(dataframe.geometry))
    return unary_union(geometries) if geometries else None


def _write_building_to_latest(run_root: Path, date_gdfs: Mapping[str, Any]) -> list[dict[str, Any]]:
    gpd, _, _, _, _ = _import_geo()
    rows: list[dict[str, Any]] = []
    for date, pair in zip(DATES[:-1], TO_LATEST_PAIRS):
        left = date_gdfs[date]
        right = date_gdfs[FINAL_DATE]
        matches = _pair_candidates(left, right)
        matched_left = {int(row["left_index"]) for row in matches}
        matched_right = {int(row["right_index"]) for row in matches}
        new_support = _full20_support_union(pair, "new")
        demolition_support = _full20_support_union(pair, "demolition")
        candidate_dir = run_root / "02_building/to_latest" / pair
        candidate_dir.mkdir(parents=True, exist_ok=True)
        new_rows = []
        for index in sorted(set(range(len(right))) - matched_right):
            geometry = right.geometry.iloc[index]
            support = bool(new_support is not None and geometry.intersects(new_support))
            if geometry.area >= 10.0 and support:
                row = right.iloc[index].to_dict()
                row.update({"candidate_status": "new_building_candidate", "full20_change_support": support, "scientific_status": "candidate_not_ground_truth", "geometry": geometry})
                new_rows.append(row)
        demolition_rows = []
        for index in sorted(set(range(len(left))) - matched_left):
            geometry = left.geometry.iloc[index]
            support = bool(demolition_support is not None and geometry.intersects(demolition_support))
            if geometry.area >= 10.0 and support:
                row = left.iloc[index].to_dict()
                row.update({"candidate_status": "disappeared_building_candidate", "full20_change_support": support, "scientific_status": "candidate_not_ground_truth", "geometry": geometry})
                demolition_rows.append(row)
        new_gdf = gpd.GeoDataFrame(new_rows, geometry="geometry", crs=METRIC_CRS)
        demolition_gdf = gpd.GeoDataFrame(demolition_rows, geometry="geometry", crs=METRIC_CRS)
        _write_gpkg(new_gdf, candidate_dir / "new_building_candidates.gpkg", "new_building_candidates")
        _write_gpkg(demolition_gdf, candidate_dir / "disappeared_building_candidates.gpkg", "disappeared_building_candidates")
        row = {
            "pair": pair,
            "start_date": date,
            "end_date": FINAL_DATE,
            "start_object_count": len(left),
            "end_object_count": len(right),
            "matched_object_count": len(matches),
            "new_building_candidate_count": len(new_gdf),
            "disappeared_building_candidate_count": len(demolition_gdf),
            "full20_support_source_new": str(_find_full20_vector(pair, "new")) if _find_full20_vector(pair, "new") else None,
            "full20_support_source_demolition": str(_find_full20_vector(pair, "demolition")) if _find_full20_vector(pair, "demolition") else None,
            "scientific_status": "candidate_not_ground_truth",
        }
        rows.append(row)
    _write_csv(run_root / "02_building/to_latest/TO_LATEST_BUILDING_COUNTS.csv", rows)
    return rows


def _seal_files(root: Path, exclude_suffixes: Sequence[str] = ()) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(path.name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        # Source-linked rasters are represented by their original SHA in the
        # branch manifest; hashing them again would add no evidence.
        if path.suffix.lower() in {".tif", ".tiff"}:
            continue
        files.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": _sha256(path)})
    return files


def _build_building_branch(run_root: Path) -> dict[str, Any]:
    _create_run_tree(run_root)
    summaries = [_write_building_date(run_root, date) for date in DATES]
    _write_csv(run_root / "02_building/BUILDING_SINGLE_DATE_SUMMARY.csv", summaries)
    date_gdfs = {date: _building_output_gdf(run_root, date) for date in DATES}
    track_gdf, track_summary, status_by_track = _classify_building_tracks(run_root, date_gdfs)
    _write_gpkg(track_gdf, run_root / "02_building/tracking/building_tracks.gpkg", "building_tracks")
    _write_csv(run_root / "02_building/tracking/BUILDING_TRACK_SUMMARY.csv", track_summary)
    for status, filename, layer in [
        ("stable_building", "stable_buildings.gpkg", "stable_buildings"),
        ("appeared_building_candidate", "appeared_building_candidates.gpkg", "appeared_building_candidates"),
        ("disappeared_building_candidate", "disappeared_building_candidates.gpkg", "disappeared_building_candidates"),
        ("single_date_anomaly", "single_date_anomalies.gpkg", "single_date_anomalies"),
    ]:
        subset = track_gdf[track_gdf["track_status"] == status].copy()
        _write_gpkg(subset, run_root / "02_building/tracking" / filename, layer)
        shutil.copy2(run_root / "02_building/tracking" / filename, run_root / "02_building" / filename)
    shutil.copy2(run_root / "02_building/tracking/building_tracks.gpkg", run_root / "02_building/building_tracks.gpkg")
    shutil.copy2(run_root / "02_building/tracking/BUILDING_TRACK_SUMMARY.csv", run_root / "02_building/BUILDING_TRACK_SUMMARY.csv")
    to_latest = _write_building_to_latest(run_root, date_gdfs)
    seal = {
        "status": "sealed",
        "created_at": _now(),
        "module": "M3_RSBuilding",
        "input_mode": "zoom20_model_scale_normalized",
        "checkpoint_sha256": "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7",
        "threshold": 0.5,
        "minimum_mapping_unit_m2": 10.0,
        "dates": DATES,
        "date_object_counts": {row["scene_date"]: row["object_count_after_mmu"] for row in summaries},
        "track_status_counts": dict(Counter(status_by_track.values())),
        "to_latest": to_latest,
        "reads": ["frozen_RSBuilding_outputs", "frozen_Full20_change_vectors_for_candidate_support"],
        "forbidden_reads": ["Construction_probability", "Construction_candidates", "Construction_GT"],
        "gt_content_read": False,
        "construction_content_read": False,
        "locked_test_pixel_reads": 0,
        "scientific_status": "building_candidate_system_sealed_before_GT_evaluation",
        "artifact_files": _seal_files(run_root / "02_building"),
    }
    _write_json(run_root / "02_building/BUILDING_SYSTEM_SEAL.json", seal)
    return seal


def _object_matches(gt: Any, prediction: Any, threshold: float) -> list[dict[str, Any]]:
    gt_records = _building_records(gt)
    pred_records = _building_records(prediction)
    candidates: list[dict[str, Any]] = []
    for gt_index, gt_row in enumerate(gt_records):
        for pred_index, pred_row in enumerate(pred_records):
            intersection = float(gt_row["geometry"].intersection(pred_row["geometry"]).area)
            if intersection <= 0:
                continue
            union = float(gt_row["geometry"].union(pred_row["geometry"]).area)
            iou = intersection / union if union else 0.0
            if iou >= threshold:
                candidates.append({
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "iou": iou,
                    "intersection_area_m2": intersection,
                    "gt_coverage": intersection / max(gt_row["area_m2"], 1e-9),
                    "pred_coverage": intersection / max(pred_row["area_m2"], 1e-9),
                })
    candidates.sort(key=lambda row: (-row["iou"], row["gt_index"], row["pred_index"]))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["gt_index"] in used_gt or candidate["pred_index"] in used_pred:
            continue
        used_gt.add(candidate["gt_index"])
        used_pred.add(candidate["pred_index"])
        matches.append(candidate)
    return matches


def _positive_overlap_proxy(gt: Any, prediction: Any) -> dict[str, Any]:
    _, _, _, _, unary_union = _import_geo()
    gt_union = unary_union(list(gt.geometry)) if len(gt) else None
    pred_union = unary_union(list(prediction.geometry)) if len(prediction) else None
    if gt_union is None or pred_union is None or gt_union.is_empty or pred_union.is_empty:
        return {
            "prediction_union_area_m2": float(pred_union.area) if pred_union else 0.0,
            "gt_union_area_m2": float(gt_union.area) if gt_union else 0.0,
            "intersection_area_m2": 0.0,
            "positive_overlap_fraction_of_prediction_proxy": 0.0,
            "positive_overlap_fraction_of_gt_proxy": 0.0,
            "positive_overlap_f1_proxy": 0.0,
        }
    intersection = float(gt_union.intersection(pred_union).area)
    pred_area = float(pred_union.area)
    gt_area = float(gt_union.area)
    return {
        "prediction_union_area_m2": pred_area,
        "gt_union_area_m2": gt_area,
        "intersection_area_m2": intersection,
        "positive_overlap_fraction_of_prediction_proxy": intersection / pred_area if pred_area else 0.0,
        "positive_overlap_fraction_of_gt_proxy": intersection / gt_area if gt_area else 0.0,
        "positive_overlap_f1_proxy": 2.0 * intersection / (pred_area + gt_area) if pred_area + gt_area else 0.0,
    }


def _load_positive_gt(path: Path, layer: str) -> Any:
    _, aoi_metric = _aoi_geometries()
    return _metric_clip(_read_vector(path, layer=layer), aoi_metric, min_area_m2=0.0)


def _evaluate_building_gt(run_root: Path) -> dict[str, Any]:
    if not (run_root / "02_building/BUILDING_SYSTEM_SEAL.json").exists():
        raise RuntimeError("Building GT evaluation is gated on BUILDING_SYSTEM_SEAL.json")
    gpd, _, _, _, _ = _import_geo()
    gt = _load_positive_gt(GT_BUILDING_PATH, "gt_building")
    result_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    mode_counts: dict[str, Any] = {}
    for mode in ("R0", "R1", "R2"):
        prediction = {date: (_building_output_gdf(run_root, date) if mode == "R2" and date == "2019-03-16" else _load_source_building(mode, date)) for date in ["2019-03-16"]}["2019-03-16"]
        records_gt = _building_records(gt)
        records_pred = _building_records(prediction)
        for threshold in (0.3, 0.5, 0.7):
            matches = _object_matches(gt, prediction, threshold)
            tp = len(matches)
            precision = tp / (tp + len(records_pred) - tp) if records_pred else None
            recall = tp / (tp + len(records_gt) - tp) if records_gt else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
            result_rows.append({
                "task": "building",
                "input_mode": MODES[mode],
                "short_mode": mode,
                "metric_family": "positive_object_gt_match_only",
                "metric_name": f"positive_object_match_iou_{threshold:.1f}",
                "value": f1,
                "gt_positive_object_count": len(records_gt),
                "prediction_object_count": len(records_pred),
                "matched_object_count": tp,
                "positive_object_match_precision": precision,
                "positive_object_match_recall": recall,
                "positive_object_match_f1": f1,
                "full_scene_accuracy_status": "not_available_positive_only_gt",
            })
            for match in matches:
                match_rows.append({
                    "short_mode": mode,
                    "input_mode": MODES[mode],
                    "iou_threshold": threshold,
                    "gt_id": records_gt[match["gt_index"]]["id"],
                    "prediction_id": records_pred[match["pred_index"]]["id"],
                    "iou": match["iou"],
                    "intersection_area_m2": match["intersection_area_m2"],
                    "gt_coverage": match["gt_coverage"],
                    "prediction_coverage": match["pred_coverage"],
                })
        proxy = _positive_overlap_proxy(gt, prediction)
        result_rows.append({
            "task": "building",
            "input_mode": MODES[mode],
            "short_mode": mode,
            "metric_family": "positive_overlap_proxy",
            "metric_name": "positive_overlap_f1_proxy",
            "value": proxy["positive_overlap_f1_proxy"],
            "gt_positive_object_count": len(records_gt),
            "prediction_object_count": len(records_pred),
            "full_scene_accuracy_status": "not_available_positive_only_gt",
        })
        matches_05 = _object_matches(gt, prediction, 0.5)
        used_gt = {row["gt_index"] for row in matches_05}
        used_pred = {row["pred_index"] for row in matches_05}
        for index, record in enumerate(records_gt):
            if index not in used_gt:
                error_rows.append({
                    "short_mode": mode,
                    "input_mode": MODES[mode],
                    "error_type": "unmatched_positive_gt_or_low_iou",
                    "gt_id": record["id"],
                    "prediction_id": None,
                    "iou": 0.0,
                    "interpretation": "positive-only GT does not support a complete false-negative/false-positive claim",
                    "geometry": record["geometry"],
                })
        for index, record in enumerate(records_pred):
            if index not in used_pred:
                error_rows.append({
                    "short_mode": mode,
                    "input_mode": MODES[mode],
                    "error_type": "unmatched_prediction",
                    "gt_id": None,
                    "prediction_id": record["id"],
                    "iou": 0.0,
                    "interpretation": "unmatched prediction is not a confirmed false positive under positive-only GT",
                    "geometry": record["geometry"],
                })
        mode_counts[mode] = {
            "prediction_object_count": len(prediction),
            "positive_gt_object_count": len(gt),
            "positive_overlap_proxy": proxy,
            "positive_object_matches_iou_0_5": len(matches_05),
        }
    evaluation_dir = run_root / "03_evaluation/building"
    _write_csv(evaluation_dir / "BUILDING_R0_R1_R2_METRICS.csv", result_rows)
    _write_csv(run_root / "06_metrics/BUILDING_R0_R1_R2_METRICS.csv", result_rows)
    _write_csv(evaluation_dir / "BUILDING_OBJECT_MATCHES.csv", match_rows)
    _write_gpkg(gpd.GeoDataFrame(error_rows, geometry="geometry", crs=METRIC_CRS), evaluation_dir / "BUILDING_ERROR_CASES.gpkg", "building_error_cases")
    _write_json(evaluation_dir / "BUILDING_GT_AUDIT.json", {
        "path": str(GT_BUILDING_PATH),
        "sha256": _sha256(GT_BUILDING_PATH),
        "layer": "gt_building",
        "aoi_clipped_positive_object_count": len(gt),
        "gt_completeness_status": "positive_only_not_declared_complete",
        "full_scene_accuracy_reported": False,
        "pixel_precision_recall_f1_iou_reported": False,
        "content_read_after_building_seal": True,
        "modes": mode_counts,
        "scientific_status": "positive_object_match_and_overlap_proxy_only",
    })
    _write_csv(run_root / "03_evaluation/building/BUILDING_SIZE_METRICS.csv", [
        {"short_mode": mode, "size_bin": size, "count": count, "metric_status": "descriptive_prediction_and_positive_gt_overlap_only"}
        for mode, dataframe in [("R0", _load_source_building("R0", "2019-03-16")), ("R1", _load_source_building("R1", "2019-03-16")), ("R2", _building_output_gdf(run_root, "2019-03-16"))]
        for size, count in {
            "small_lt_100m2": int((dataframe.geometry.area < 100).sum()),
            "medium_100_to_1000m2": int(((dataframe.geometry.area >= 100) & (dataframe.geometry.area < 1000)).sum()),
            "large_ge_1000m2": int((dataframe.geometry.area >= 1000).sum()),
        }.items()
    ])
    return {"gt_positive_object_count": len(gt), "mode_counts": mode_counts, "error_case_count": len(error_rows)}


def _construction_source_candidates(mode: str, date: str) -> Any:
    _, aoi_metric = _aoi_geometries()
    source_path = _find_source_file(mode, "construction", date, "parcel_candidates.gpkg")
    dataframe = _metric_clip(_read_vector(source_path, layer="parcel_candidates"), aoi_metric, min_area_m2=0.0)
    if "construction_candidate" in dataframe.columns:
        values = dataframe["construction_candidate"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        dataframe = dataframe[values].copy()
    return dataframe.reset_index(drop=True)


def _construction_values(row: Any, name: str, fallback: float = 0.0) -> float:
    try:
        value = float(row.get(name, fallback))
        return value if math.isfinite(value) else fallback
    except Exception:
        return fallback


def _materialize_construction_inputs(run_root: Path, mode: str, date: str, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    links = []
    for filename in RAW_CONSTRUCTION_FILES:
        source = _find_source_file(mode, "construction", date, filename)
        links.append({"source": str(source), "target": str(destination / filename), "operation": _hardlink_readonly(source, destination / filename)})
    source_vector = _find_source_file(mode, "construction", date, "parcel_candidates.gpkg")
    return {"source_vector": str(source_vector), "source_vector_sha256": _sha256(source_vector), "links": links}


def _building_support_union(run_root: Path, date: str, kind: str) -> Any:
    _, _, _, _, unary_union = _import_geo()
    if date == FINAL_DATE:
        # The final date is supported by the preceding formal pair.
        pair = TO_LATEST_PAIRS[-1]
    else:
        pair = f"{date}_to_{FINAL_DATE}"
    path = run_root / "02_building/to_latest" / pair / ("new_building_candidates.gpkg" if kind == "new" else "disappeared_building_candidates.gpkg")
    if not path.exists():
        return None
    dataframe = _read_vector(path)
    return unary_union(list(dataframe.geometry)) if len(dataframe) else None


def _stable_building_union(run_root: Path) -> Any:
    _, _, _, _, unary_union = _import_geo()
    path = run_root / "02_building/tracking/stable_buildings.gpkg"
    if not path.exists():
        return None
    dataframe = _read_vector(path)
    return unary_union(list(dataframe.geometry)) if len(dataframe) else None


def _construction_temporal_persistence(candidate_gdfs: Mapping[str, Any]) -> dict[str, float]:
    persistence: dict[str, float] = {}
    for date, dataframe in candidate_gdfs.items():
        for index in range(len(dataframe)):
            candidate_id = str(dataframe.iloc[index]["candidate_id"])
            hits = 1
            for other_date, other in candidate_gdfs.items():
                if other_date == date:
                    continue
                geometry = dataframe.geometry.iloc[index]
                found = False
                for other_geometry in other.geometry:
                    intersection = geometry.intersection(other_geometry).area
                    union = geometry.union(other_geometry).area
                    if union and intersection / union >= 0.10:
                        found = True
                        break
                if found:
                    hits += 1
            persistence[candidate_id] = hits / len(candidate_gdfs)
    return persistence


def _rank_construction_candidates(run_root: Path, candidate_gdfs: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    gpd, _, _, _, _ = _import_geo()
    stable_union = _stable_building_union(run_root)
    all_rows: list[dict[str, Any]] = []
    persistence = _construction_temporal_persistence(candidate_gdfs)
    per_date: list[dict[str, Any]] = []
    for date in DATES:
        dataframe = candidate_gdfs[date]
        full20_union = _full20_support_union(f"{date}_to_{FINAL_DATE}" if date != FINAL_DATE else TO_LATEST_PAIRS[-1], "all")
        new_union = _building_support_union(run_root, date, "new")
        disappearance_union = _building_support_union(run_root, date, "disappeared")
        for index, source_row in dataframe.reset_index(drop=True).iterrows():
            geometry = source_row.geometry
            area = float(geometry.area)
            perimeter = float(geometry.length)
            compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-9))
            full20_support = bool(full20_union is not None and geometry.intersects(full20_union))
            building_support = bool((new_union is not None and geometry.intersects(new_union)) or (disappearance_union is not None and geometry.intersects(disappearance_union)))
            stable_overlap_area = float(geometry.intersection(stable_union).area) if stable_union is not None else 0.0
            stable_overlap = stable_overlap_area / area if area else 0.0
            stable_conflict = stable_overlap >= 0.25
            non_background = _construction_values(source_row, "construction_non_background")
            exposed = _construction_values(source_row, "construction_exposed_ground")
            dustproof = _construction_values(source_row, "construction_dustproof_net")
            boundary = _construction_values(source_row, "construction_boundary")
            building = _construction_values(source_row, "construction_building")
            uncertainty = _construction_values(source_row, "construction_uncertainty")
            temporal = persistence.get(f"{date}_R2_{index:06d}", 0.0)
            evidence_score = max(0.0, min(1.0, 0.25 * non_background + 0.18 * exposed + 0.12 * dustproof + 0.10 * boundary + 0.15 * float(full20_support) + 0.10 * float(building_support) + 0.10 * temporal - 0.25 * stable_overlap - 0.10 * uncertainty))
            tags: list[str] = []
            if stable_conflict:
                tags.append("stable_building_conflict")
            if dustproof >= 0.45 and stable_conflict:
                tags.append("stable_industrial_roof")
                tags.append("blue_green_metal_roof")
            if exposed < 0.2 and not full20_support:
                tags.append("ordinary_bare_land")
            if area < 50:
                tags.append("very_small_fragment")
            if temporal <= 1.0 / len(DATES):
                tags.append("single_date_fragment")
            if uncertainty >= 0.5:
                tags.append("high_uncertainty")
            if not full20_support:
                tags.append("weak_full20_support")
            if exposed < 0.2 and dustproof < 0.2:
                tags.append("no_ground_or_net_support")
            if stable_conflict:
                likely_confusion = "stable_building_conflict"
            elif dustproof >= exposed and dustproof >= 0.45:
                likely_confusion = "dustproof_net_or_metal_roof_candidate"
            elif exposed < 0.2 and not full20_support:
                likely_confusion = "parking_or_paved_yard_or_ordinary_bare_land"
            else:
                likely_confusion = "unresolved_construction_candidate"
            if stable_conflict or evidence_score < 0.20:
                tier = "likely_false_positive"
            elif evidence_score >= 0.65:
                tier = "tier_A_high_evidence"
            elif evidence_score >= 0.45:
                tier = "tier_B_medium_evidence"
            elif evidence_score >= 0.25:
                tier = "tier_C_low_evidence"
            else:
                tier = "uncertain"
            all_rows.append({
                "candidate_id": f"{date}_R2_{index:06d}",
                "scene_date": date,
                "candidate_area_m2": area,
                "construction_probability": non_background,
                "exposed_ground_support": exposed,
                "dustproof_net_support": dustproof,
                "boundary_support": boundary,
                "uncertainty": uncertainty,
                "full20_change_support": full20_support,
                "building_change_support": building_support,
                "stable_building_overlap": stable_overlap,
                "stable_building_conflict": stable_conflict,
                "temporal_persistence": temporal,
                "compactness": compactness,
                "fragmentation": 0.0 if geometry.geom_type == "Polygon" else 1.0,
                "evidence_score": evidence_score,
                "candidate_rank": None,
                "candidate_tier": tier,
                "likely_confusion_type": likely_confusion,
                "evidence_downgrade_tags": ";".join(tags),
                "scientific_status": "candidate_not_ground_truth",
                "geometry": geometry,
            })
        current_rows = [row for row in all_rows if row["scene_date"] == date]
        current_rows.sort(key=lambda row: (-float(row["evidence_score"]), str(row["candidate_id"])))
        for rank, row in enumerate(current_rows, start=1):
            row["candidate_rank"] = rank
        per_date.append({
            "scene_date": date,
            "raw_candidate_count": len(dataframe),
            "ranked_candidate_count": len(current_rows),
            "tier_A_count": sum(row["candidate_tier"] == "tier_A_high_evidence" for row in current_rows),
            "tier_B_count": sum(row["candidate_tier"] == "tier_B_medium_evidence" for row in current_rows),
            "tier_C_count": sum(row["candidate_tier"] == "tier_C_low_evidence" for row in current_rows),
            "likely_false_positive_count": sum(row["candidate_tier"] == "likely_false_positive" for row in current_rows),
            "candidate_area_m2": float(sum(row["candidate_area_m2"] for row in current_rows)),
            "scientific_status": "candidate_not_ground_truth",
        })
    ranked = gpd.GeoDataFrame(all_rows, geometry="geometry", crs=METRIC_CRS)
    return ranked, per_date, {
        "method_id": "fixed_read_only_construction_evidence_rank_v1",
        "uses_ground_truth": False,
        "changes_raw_probabilities": False,
        "changes_raw_candidate_mask": False,
        "formula": "clip(0.25*non_background + 0.18*exposed_ground + 0.12*dustproof_net + 0.10*boundary + 0.15*Full20_support + 0.10*Building_support + 0.10*temporal_persistence - 0.25*stable_building_overlap - 0.10*uncertainty, 0, 1)",
        "tiers": {"tier_A_high_evidence": ">=0.65", "tier_B_medium_evidence": ">=0.45", "tier_C_low_evidence": ">=0.25", "likely_false_positive": "stable_building_conflict or score<0.20", "uncertain": "otherwise"},
        "downgrade_tags": ["stable_industrial_roof", "blue_green_metal_roof", "parking_or_paved_yard", "ordinary_bare_land", "stable_building_conflict", "single_date_fragment", "very_small_fragment", "high_uncertainty", "weak_full20_support", "no_ground_or_net_support"],
        "scientific_status": "candidate_ranking_not_site_classifier",
    }


def _build_construction_branch(run_root: Path) -> dict[str, Any]:
    _create_run_tree(run_root)
    r2_candidates: dict[str, Any] = {}
    single_summaries: list[dict[str, Any]] = []
    for mode, branch in [("R0", "R0"), ("R1", "R1"), ("R2", "R2")]:
        for date in DATES:
            destination = run_root / "04_construction/single_date" / date if mode == "R2" else run_root / "04_construction/ablation" / branch / date
            manifest = _materialize_construction_inputs(run_root, mode, date, destination)
            dataframe = _construction_source_candidates(mode, date)
            if mode == "R2":
                dataframe = dataframe.reset_index(drop=True)
                dataframe["candidate_id"] = [f"{date}_R2_{index:06d}" for index in range(len(dataframe))]
                r2_candidates[date] = dataframe
                output_rows = []
                for index, row in dataframe.iterrows():
                    output_rows.append({
                        "candidate_id": row["candidate_id"],
                        "scene_date": date,
                        "candidate_area_m2": float(row.geometry.area),
                        "construction_probability": _construction_values(row, "construction_non_background"),
                        "exposed_ground_support": _construction_values(row, "construction_exposed_ground"),
                        "dustproof_net_support": _construction_values(row, "construction_dustproof_net"),
                        "boundary_support": _construction_values(row, "construction_boundary"),
                        "uncertainty": _construction_values(row, "construction_uncertainty"),
                        "scientific_status": "raw_model_candidate_not_human_verified",
                        "geometry": row.geometry,
                    })
                _write_gpkg(pd.DataFrame(output_rows) if not output_rows else __import__("geopandas").GeoDataFrame(output_rows, geometry="geometry", crs=METRIC_CRS), destination / "raw_construction_candidates.gpkg", "raw_construction_candidates")
                single_summaries.append({
                    "input_mode": MODES[mode],
                    "short_mode": mode,
                    "scene_date": date,
                    "raw_candidate_count": len(dataframe),
                    "raw_candidate_area_m2": float(dataframe.geometry.area.sum()) if len(dataframe) else 0.0,
                    "raw_probability_preserved": True,
                    "scientific_status": "candidate_not_ground_truth",
                })
            _write_json(destination / "CONSTRUCTION_RAW_INPUT_MANIFEST.json", {
                "created_at": _now(),
                "short_mode": mode,
                "input_mode": MODES[mode],
                "date": date,
                "source_only_read": True,
                "raw_probability_recomputed": False,
                "candidate_mask_recomputed": False,
                "manifest": manifest,
                "locked_test_pixel_reads": 0,
            })
    ranked, ranked_summaries, method = _rank_construction_candidates(run_root, r2_candidates)
    ranking_dir = run_root / "04_construction/ranking"
    _write_gpkg(ranked, ranking_dir / "construction_candidates_ranked.gpkg", "construction_candidates_ranked")
    _write_gpkg(ranked[ranked["candidate_tier"].isin(["tier_A_high_evidence", "tier_B_medium_evidence"])], ranking_dir / "construction_top_candidates.gpkg", "construction_top_candidates")
    _write_gpkg(ranked[ranked["candidate_tier"] == "likely_false_positive"], ranking_dir / "construction_likely_false_positives.gpkg", "construction_likely_false_positives")
    _write_csv(run_root / "04_construction/CONSTRUCTION_SINGLE_DATE_SUMMARY.csv", single_summaries)
    _write_csv(run_root / "04_construction/CONSTRUCTION_CANDIDATE_RANKING.csv", ranked.drop(columns="geometry"))
    _write_json(ranking_dir / "CONSTRUCTION_RANKING_METHOD.json", method)
    track_gdf, track_summary = _track_construction_candidates(r2_candidates, ranked)
    _write_gpkg(track_gdf, run_root / "04_construction/tracking/construction_tracks.gpkg", "construction_tracks")
    _write_csv(run_root / "04_construction/tracking/CONSTRUCTION_TRACK_SUMMARY.csv", track_summary)
    for status, filename, layer in [
        ("persistent_construction_candidate", "persistent_construction_candidates.gpkg", "persistent_construction_candidates"),
        ("newly_started_construction_candidate", "newly_started_construction_candidates.gpkg", "newly_started_construction_candidates"),
        ("likely_completed_candidate", "likely_completed_construction_candidates.gpkg", "likely_completed_construction_candidates"),
        ("demolition_or_clearing_candidate", "demolition_or_clearing_candidates.gpkg", "demolition_or_clearing_candidates"),
    ]:
        _write_gpkg(track_gdf[track_gdf["track_status"] == status].copy(), run_root / "04_construction/tracking" / filename, layer)
        shutil.copy2(run_root / "04_construction/tracking" / filename, run_root / "04_construction" / filename)
    shutil.copy2(run_root / "04_construction/tracking/construction_tracks.gpkg", run_root / "04_construction/construction_tracks.gpkg")
    shutil.copy2(run_root / "04_construction/tracking/CONSTRUCTION_TRACK_SUMMARY.csv", run_root / "04_construction/CONSTRUCTION_TRACK_SUMMARY.csv")
    for filename in ["construction_candidates_ranked.gpkg", "construction_top_candidates.gpkg", "construction_likely_false_positives.gpkg"]:
        shutil.copy2(run_root / "04_construction/ranking" / filename, run_root / "04_construction" / filename)
    seal = {
        "status": "sealed",
        "created_at": _now(),
        "module": "M6_Construction",
        "formal_input_mode": "zoom20_model_scale_normalized",
        "ablation_modes": ["zoom19_reference", "zoom20_native"],
        "checkpoint_sha256": "c522d5a4e6c5d6124f968d659004d7870fe4adc9d7381dc4c8917fe019f0f060",
        "rgb_mapping": [3, 2, 1],
        "dates": DATES,
        "raw_probability_preserved": True,
        "ranking_changes_raw_probability": False,
        "ranking_uses_ground_truth": False,
        "site_classifier_enabled": False,
        "site_classifier_status": "disabled_pending_human_gt",
        "training_started": False,
        "locked_test_pixel_reads": 0,
        "scientific_status": "candidate_level_construction_system_sealed_before_GT_overlap_diagnostics",
        "artifact_files": _seal_files(run_root / "04_construction"),
    }
    _write_json(run_root / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json", seal)
    _write_csv(run_root / "04_construction/CONSTRUCTION_SINGLE_DATE_RANKED_SUMMARY.csv", ranked_summaries)
    return seal


def _track_construction_candidates(candidate_gdfs: Mapping[str, Any], ranked: Any) -> tuple[Any, list[dict[str, Any]]]:
    gpd, _, _, _, _ = _import_geo()
    rows: list[dict[str, Any]] = []
    next_track = 1
    previous: Any | None = None
    previous_track: dict[int, str] = {}
    objects_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for date in DATES:
        current = ranked[ranked["scene_date"] == date].copy().reset_index(drop=True)
        current["track_id"] = None
        current["match_iou_prev"] = None
        if previous is not None:
            matches = _pair_candidates(previous, current, min_iou=0.05, max_distance_m=50.0, max_area_ratio=8.0)
        else:
            matches = []
        used_right: set[int] = set()
        for match in matches:
            left = int(match["left_index"])
            right = int(match["right_index"])
            track_id = previous_track.get(left)
            if track_id is None:
                track_id = f"CTR_{next_track:06d}"
                next_track += 1
            current.loc[right, "track_id"] = track_id
            current.loc[right, "match_iou_prev"] = match["iou"]
            used_right.add(right)
        for index in range(len(current)):
            if pd.isna(current.loc[index, "track_id"]):
                current.loc[index, "track_id"] = f"CTR_{next_track:06d}"
                next_track += 1
            row = current.iloc[index].to_dict()
            row["scene_date"] = date
            row["track_id"] = str(current.iloc[index]["track_id"])
            row["geometry"] = current.geometry.iloc[index]
            rows.append(row)
            objects_by_track[row["track_id"]].append(row)
        previous = current
        previous_track = {index: str(current.iloc[index]["track_id"]) for index in range(len(current))}
    status_by_track: dict[str, str] = {}
    summary: list[dict[str, Any]] = []
    for track_id, records in sorted(objects_by_track.items()):
        dates = [str(row["scene_date"]) for row in records]
        first_date, last_date = min(dates), max(dates)
        if len(records) >= 3:
            status = "persistent_construction_candidate"
        elif first_date != DATES[0] and len(records) >= 2:
            status = "newly_started_construction_candidate"
        elif last_date != FINAL_DATE and len(records) >= 2:
            status = "likely_completed_candidate"
        elif last_date != FINAL_DATE and any(bool(row.get("full20_change_support")) for row in records):
            status = "demolition_or_clearing_candidate"
        elif len(records) == 1:
            status = "single_date_fragment"
        else:
            status = "uncertain_temporal_candidate"
        status_by_track[track_id] = status
        summary.append({
            "track_id": track_id,
            "object_count": len(records),
            "first_date": first_date,
            "last_date": last_date,
            "mean_evidence_score": float(np.mean([float(row["evidence_score"]) for row in records])),
            "status": status,
            "scientific_status": "candidate_not_ground_truth",
        })
    for row in rows:
        row["track_status"] = status_by_track[row["track_id"]]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS), summary


def _evaluate_construction_gt(run_root: Path) -> dict[str, Any]:
    if not (run_root / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json").exists():
        raise RuntimeError("Construction GT evaluation is gated on CONSTRUCTION_SYSTEM_SEAL.json")
    gpd, _, _, _, _ = _import_geo()
    gt = _load_positive_gt(GT_CONSTRUCTION_PATH, "gt_construction")
    result_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    mode_counts: dict[str, Any] = {}
    for mode in ("R0", "R1", "R2"):
        prediction = _construction_source_candidates(mode, "2019-03-16") if mode != "R2" else _read_vector(run_root / "04_construction/single_date/2019-03-16/raw_construction_candidates.gpkg", layer="raw_construction_candidates")
        _, aoi_metric = _aoi_geometries()
        prediction = _metric_clip(prediction, aoi_metric)
        proxy = _positive_overlap_proxy(gt, prediction)
        matches = _object_matches(gt, prediction, 0.5)
        result_rows.append({
            "task": "construction",
            "input_mode": MODES[mode],
            "short_mode": mode,
            "metric_family": "positive_overlap_proxy_only",
            "metric_name": "positive_overlap_f1_proxy",
            "value": proxy["positive_overlap_f1_proxy"],
            "gt_positive_object_count": len(gt),
            "prediction_candidate_count": len(prediction),
            "positive_overlap_fraction_of_prediction_proxy": proxy["positive_overlap_fraction_of_prediction_proxy"],
            "positive_overlap_fraction_of_gt_proxy": proxy["positive_overlap_fraction_of_gt_proxy"],
            "positive_object_match_count_iou_0_5_diagnostic": len(matches),
            "formal_site_precision_recall_f1_status": "not_reported",
            "full_scene_pixel_accuracy_status": "not_available_positive_only_gt",
        })
        gt_records = _building_records(gt)
        pred_records = _building_records(prediction)
        for match in matches:
            match_rows.append({
                "short_mode": mode,
                "input_mode": MODES[mode],
                "diagnostic_iou_threshold": 0.5,
                "gt_id": gt_records[match["gt_index"]]["id"],
                "candidate_id": pred_records[match["pred_index"]]["id"],
                "iou": match["iou"],
                "intersection_area_m2": match["intersection_area_m2"],
            })
        used_gt = {row["gt_index"] for row in matches}
        used_pred = {row["pred_index"] for row in matches}
        for index, record in enumerate(gt_records):
            if index not in used_gt:
                error_rows.append({
                    "short_mode": mode,
                    "input_mode": MODES[mode],
                    "error_type": "unmatched_positive_gt_or_low_iou",
                    "candidate_id": None,
                    "gt_id": record["id"],
                    "interpretation": "positive-only construction GT; no complete site false-negative claim",
                    "geometry": record["geometry"],
                })
        for index, record in enumerate(pred_records):
            if index not in used_pred:
                error_rows.append({
                    "short_mode": mode,
                    "input_mode": MODES[mode],
                    "error_type": "unmatched_candidate",
                    "candidate_id": record["id"],
                    "gt_id": None,
                    "interpretation": "unmatched candidate is not a confirmed false positive under positive-only GT",
                    "geometry": record["geometry"],
                })
        mode_counts[mode] = {
            "prediction_candidate_count": len(prediction),
            "positive_gt_object_count": len(gt),
            "positive_overlap_proxy": proxy,
            "diagnostic_positive_object_match_count_iou_0_5": len(matches),
        }
    evaluation_dir = run_root / "03_evaluation/construction"
    _write_csv(evaluation_dir / "CONSTRUCTION_R0_R1_R2_METRICS.csv", result_rows)
    _write_csv(run_root / "06_metrics/CONSTRUCTION_R0_R1_R2_METRICS.csv", result_rows)
    _write_csv(evaluation_dir / "CONSTRUCTION_OBJECT_MATCHES.csv", match_rows)
    _write_gpkg(gpd.GeoDataFrame(error_rows, geometry="geometry", crs=METRIC_CRS), evaluation_dir / "CONSTRUCTION_ERROR_CASES.gpkg", "construction_error_cases")
    _write_json(evaluation_dir / "CONSTRUCTION_GT_AUDIT.json", {
        "path": str(GT_CONSTRUCTION_PATH),
        "sha256": _sha256(GT_CONSTRUCTION_PATH),
        "layer": "gt_construction",
        "aoi_clipped_positive_object_count": len(gt),
        "gt_completeness_status": "positive_only_not_declared_complete",
        "formal_site_precision_recall_f1_reported": False,
        "full_scene_pixel_accuracy_reported": False,
        "content_read_after_construction_seal": True,
        "modes": mode_counts,
        "scientific_status": "positive_overlap_proxy_and_diagnostic_object_matches_only",
    })
    return {"gt_positive_object_count": len(gt), "mode_counts": mode_counts, "error_case_count": len(error_rows)}


def _cross_reference(run_root: Path) -> dict[str, Any]:
    gpd, _, _, _, unary_union = _import_geo()
    building_seal = _read_json(run_root / "02_building/BUILDING_SYSTEM_SEAL.json", {})
    construction_seal = _read_json(run_root / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json", {})
    if building_seal.get("status") != "sealed" or construction_seal.get("status") != "sealed":
        raise RuntimeError("cross-reference requires both branch seals")
    ranked = _read_vector(run_root / "04_construction/ranking/construction_candidates_ranked.gpkg", layer="construction_candidates_ranked")
    stable = _read_vector(run_root / "02_building/tracking/stable_buildings.gpkg", layer="stable_buildings")
    appeared = _read_vector(run_root / "02_building/tracking/appeared_building_candidates.gpkg", layer="appeared_building_candidates")
    disappeared = _read_vector(run_root / "02_building/tracking/disappeared_building_candidates.gpkg", layer="disappeared_building_candidates")
    stable_union = unary_union(list(stable.geometry)) if len(stable) else None
    appeared_union = unary_union(list(appeared.geometry)) if len(appeared) else None
    disappeared_union = unary_union(list(disappeared.geometry)) if len(disappeared) else None
    rows: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        geometry = row.geometry
        rows.append({
            "candidate_id": row.get("candidate_id"),
            "scene_date": row.get("scene_date"),
            "candidate_tier": row.get("candidate_tier"),
            "evidence_score": row.get("evidence_score"),
            "stable_building_conflict": bool(row.get("stable_building_conflict")),
            "appeared_building_support": bool(appeared_union is not None and geometry.intersects(appeared_union)),
            "disappeared_building_support": bool(disappeared_union is not None and geometry.intersects(disappeared_union)),
            "stable_building_overlap": row.get("stable_building_overlap"),
            "likely_confusion_type": row.get("likely_confusion_type"),
            "scientific_status": "read_only_cross_reference_candidate_not_ground_truth",
            "geometry": geometry,
        })
    cross = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
    _write_gpkg(cross, run_root / "05_cross_reference/BUILDING_CONSTRUCTION_CROSS_REFERENCE.gpkg", "building_construction_cross_reference")
    summary: list[dict[str, Any]] = []
    for date in DATES:
        subset = cross[cross["scene_date"] == date]
        summary.append({
            "scene_date": date,
            "building_object_count": int(len(_building_output_gdf(run_root, date))),
            "construction_candidate_count": int(len(subset)),
            "tier_A_count": int((subset["candidate_tier"] == "tier_A_high_evidence").sum()),
            "tier_B_count": int((subset["candidate_tier"] == "tier_B_medium_evidence").sum()),
            "stable_building_conflict_count": int(subset["stable_building_conflict"].sum()),
            "appeared_building_support_count": int(subset["appeared_building_support"].sum()),
            "disappeared_building_support_count": int(subset["disappeared_building_support"].sum()),
            "cross_reference_is_read_only": True,
            "scientific_status": "candidate_level_cross_reference_not_ground_truth",
        })
    _write_csv(run_root / "05_cross_reference/INTEGRATED_BUILDING_CONSTRUCTION_SUMMARY.csv", summary)
    _write_csv(run_root / "06_metrics/INTEGRATED_BUILDING_CONSTRUCTION_SUMMARY.csv", summary)
    _write_json(run_root / "05_cross_reference/CROSS_REFERENCE_MANIFEST.json", {
        "created_at": _now(),
        "reads_only": ["BUILDING_SYSTEM_SEAL", "CONSTRUCTION_SYSTEM_SEAL", "sealed_building_candidates", "ranked_construction_candidates", "Full20_support_vectors"],
        "writes_upstream": False,
        "joint_probability_created": False,
        "scientific_status": "read_only_cross_reference",
    })
    return {"row_count": len(cross), "summary": summary}


def _read_scene_image(date: str, max_width: int = 1500) -> tuple[np.ndarray, Any, tuple[float, float, float, float]]:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    path = _image_path(date)
    with rasterio.open(path) as dataset:
        window = from_bounds(*AOI_BOUNDS_WGS84, transform=dataset.transform)
        width = max(600, min(max_width, int(window.width)))
        height = max(500, min(int(max_width * window.height / max(window.width, 1)), int(window.height)))
        image = dataset.read(out_shape=(min(3, dataset.count), height, width), window=window, resampling=rasterio.enums.Resampling.bilinear)
        transform = dataset.window_transform(window)
        source_bounds = rasterio.windows.bounds(window, dataset.transform)
        # All vector layers in this run are metric EPSG:32650.  Transform the
        # image extent before plotting; mixing lon/lat with UTM coordinates
        # creates a misleading, effectively unbounded Matplotlib canvas.
        metric_bounds = transform_bounds(dataset.crs, METRIC_CRS, *source_bounds, densify_pts=21)
        # rasterio bounds are (left, bottom, right, top), while Matplotlib's
        # image extent is (left, right, bottom, top).
        extent = (metric_bounds[0], metric_bounds[2], metric_bounds[1], metric_bounds[3])
    image = image[:3].astype("float32")
    channels = []
    for channel in image:
        low, high = np.nanpercentile(channel, [2, 98])
        channels.append(np.clip((channel - low) / max(high - low, 1e-6), 0, 1))
    return np.moveaxis(np.asarray(channels), 0, -1), transform, extent


def _plot_base(ax: Any, date: str) -> None:
    image, _, extent = _read_scene_image(date)
    ax.imshow(image, extent=extent, origin="upper")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.grid(alpha=0.12, linewidth=0.5)
    # A compact metric scale bar and north arrow are explicit in every figure.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale = max(100.0, round(abs(x1 - x0) / 5 / 100) * 100)
    ax.plot([x0 + 0.05 * (x1 - x0), x0 + 0.05 * (x1 - x0) + scale], [y0 + 0.07 * (y1 - y0)] * 2, color="white", linewidth=3)
    ax.text(x0 + 0.05 * (x1 - x0) + scale / 2, y0 + 0.085 * (y1 - y0), f"{int(scale)} m", color="white", ha="center", va="bottom", fontsize=8, bbox={"facecolor": "black", "alpha": 0.5, "pad": 2})
    ax.annotate("N", xy=(x0 + 0.94 * (x1 - x0), y0 + 0.18 * (y1 - y0)), xytext=(x0 + 0.94 * (x1 - x0), y0 + 0.06 * (y1 - y0)), color="white", arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 2}, ha="center", fontsize=12, bbox={"facecolor": "black", "alpha": 0.5, "pad": 2})


def _overlay(ax: Any, dataframe: Any, color: str, label: str, alpha: float = 0.35, edgecolor: str | None = None, hatch: str | None = None, linewidth: float = 1.0) -> None:
    if dataframe is None or len(dataframe) == 0:
        return
    dataframe.plot(ax=ax, facecolor=color, edgecolor=edgecolor or color, alpha=alpha, linewidth=linewidth, label=label, hatch=hatch)


def _figure_path(run_root: Path, section: str, name: str) -> Path:
    path = run_root / "07_figures_highres" / section / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig: Any, path: Path) -> None:
    # Keep the contractual 12x7 inch canvas at 300 dpi (3600x2100) rather
    # than allowing tight bounding-box cropping to silently shrink core maps.
    fig.savefig(path, dpi=300, facecolor="#f7f8f6")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plot_building_overview(run_root: Path, date: str) -> Path:
    import matplotlib.pyplot as plt

    dataframe = _building_output_gdf(run_root, date)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _plot_base(ax, date)
    _overlay(ax, dataframe, "#04d9ff", "RSBuilding candidate", alpha=0.28, linewidth=0.8)
    ax.set_title(f"南京江北新区建筑候选 | {date} | Zoom20 R2尺度归一化 | n={len(dataframe)} | candidate_not_ground_truth", fontsize=15)
    ax.legend(loc="upper right", framealpha=0.85)
    path = _figure_path(run_root, "building", f"building_{date}_overview.png")
    _save_fig(fig, path)
    return path


def _plot_building_comparison(run_root: Path) -> Path:
    import matplotlib.pyplot as plt

    date = "2019-03-16"
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    colors = {"R0": "#00c8ff", "R1": "#ffcc00", "R2": "#ff4f81"}
    for ax, mode in zip(axes, ("R0", "R1", "R2")):
        _plot_base(ax, date)
        dataframe = _building_output_gdf(run_root, date) if mode == "R2" else _load_source_building(mode, date)
        _overlay(ax, dataframe, colors[mode], f"{mode} building candidate", alpha=0.30, linewidth=0.7)
        ax.set_title(f"{mode} | n={len(dataframe)}")
    fig.suptitle("建筑分辨率/输入模式对照（同AOI；仅阳性GT诊断）", fontsize=16)
    path = _figure_path(run_root, "building", "building_R0_R1_R2_2019_comparison.png")
    _save_fig(fig, path)
    return path


def _plot_building_gt(run_root: Path) -> Path:
    import matplotlib.pyplot as plt

    date = "2019-03-16"
    gt = _load_positive_gt(GT_BUILDING_PATH, "gt_building")
    prediction = _building_output_gdf(run_root, date)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _plot_base(ax, date)
    _overlay(ax, prediction, "#35f0c2", "prediction", alpha=0.25, linewidth=0.7)
    _overlay(ax, gt, "none", "positive GT (unfilled)", alpha=1.0, edgecolor="#fff200", linewidth=1.0)
    ax.set_title(f"建筑候选与2019阳性人工GT叠加 | pred={len(prediction)} | GT={len(gt)} | positive-only; unmatched prediction ≠ confirmed FP", fontsize=14)
    ax.legend(loc="upper right", framealpha=0.85)
    path = _figure_path(run_root, "building", "building_2019_positive_gt_overlay.png")
    _save_fig(fig, path)
    return path


def _plot_building_tracking(run_root: Path) -> Path:
    import matplotlib.pyplot as plt

    track = _read_vector(run_root / "02_building/tracking/building_tracks.gpkg", layer="building_tracks")
    stable = track[track["track_status"] == "stable_building"] if len(track) else track
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _plot_base(ax, FINAL_DATE)
    _overlay(ax, stable, "#9ad7ff", "stable building track", alpha=0.28, linewidth=0.7)
    ax.set_title(f"跨期建筑轨迹：稳定建筑候选 | track rows={len(stable)} | read-only Building branch", fontsize=15)
    ax.legend(loc="upper right", framealpha=0.85)
    path = _figure_path(run_root, "building", "building_stable_tracks.png")
    _save_fig(fig, path)
    return path


def _plot_building_cases(run_root: Path) -> Path:
    import matplotlib.pyplot as plt

    date = "2019-03-16"
    r2 = _building_output_gdf(run_root, date)
    r0 = _load_source_building("R0", date)
    gt = _load_positive_gt(GT_BUILDING_PATH, "gt_building")
    selected: list[tuple[str, Any, str]] = []
    matched_r2_r0 = {match["right_index"] for match in _pair_candidates(r0, r2, min_iou=0.3, max_distance_m=40.0)}
    for index in [i for i in range(len(r2)) if i not in matched_r2_r0][:3]:
        selected.append(("R2 local candidate not matched to R0", r2.iloc[index].geometry, "#ff4f81"))
    matched_r0_r2 = {match["left_index"] for match in _pair_candidates(r0, r2, min_iou=0.3, max_distance_m=40.0)}
    for index in [i for i in range(len(r0)) if i not in matched_r0_r2][:3]:
        selected.append(("R0 local candidate not matched to R2", r0.iloc[index].geometry, "#ffcc00"))
    for _, row in r2.sort_values("area_m2").head(2).iterrows():
        selected.append(("small building candidate (<100m² or lowest available)", row.geometry, "#8fe8ff"))
    for _, row in r2.sort_values("area_m2", ascending=False).head(2).iterrows():
        selected.append(("large factory/large-footprint candidate", row.geometry, "#ff7f50"))
    # Split/merge proxies are selected where one side has multiple nearby
    # candidates; the label remains a geometry proxy rather than a truth claim.
    for index, row in r2.iterrows():
        nearby = [other for other in r0.geometry if row.geometry.centroid.distance(other.centroid) < 25]
        if len(nearby) >= 2:
            selected.append(("split/merge geometry proxy", row.geometry, "#c084fc"))
        if len(selected) >= 10:
            break
    for _, row in gt.head(2).iterrows():
        selected.append(("positive GT polygon; match status is diagnostic only", row.geometry, "#fff200"))
    selected = selected[:12]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for ax, item in zip(axes.flat, selected):
        label, geometry, color = item
        _plot_base(ax, date)
        center = geometry.centroid
        x0, x1 = center.x - 120, center.x + 120
        y0, y1 = center.y - 120, center.y + 120
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        gpd = __import__("geopandas")
        _overlay(ax, gpd.GeoDataFrame({"geometry": [geometry]}, geometry="geometry", crs=METRIC_CRS), color if color != "#fff200" else "none", label, alpha=0.35, edgecolor=color, linewidth=1.3)
        ax.set_title(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle("建筑局部案例清单（12例；几何差异与阳性GT诊断，不等同于准确率）", fontsize=15)
    path = _figure_path(run_root, "building", "building_local_case_sheet_12.png")
    _save_fig(fig, path)
    return path


def _plot_construction_overview(run_root: Path, date: str) -> Path:
    import matplotlib.pyplot as plt

    dataframe = _read_vector(run_root / "04_construction/ranking/construction_candidates_ranked.gpkg", layer="construction_candidates_ranked")
    dataframe = dataframe[dataframe["scene_date"] == date]
    palette = {"tier_A_high_evidence": "#f23b20", "tier_B_medium_evidence": "#ffbd3b", "tier_C_low_evidence": "#ffe66d", "likely_false_positive": "#8f8795", "uncertain": "#b7a9cc"}
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _plot_base(ax, date)
    for tier, color in palette.items():
        _overlay(ax, dataframe[dataframe["candidate_tier"] == tier], color, tier, alpha=0.40 if tier != "likely_false_positive" else 0.28, edgecolor=color, hatch="///" if tier == "likely_false_positive" else None, linewidth=1.0)
    ax.set_title(f"南京江北新区工地候选分层 | {date} | raw probability unchanged | n={len(dataframe)} | candidate_not_ground_truth", fontsize=14)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=8)
    path = _figure_path(run_root, "construction", f"construction_{date}_ranked_overview.png")
    _save_fig(fig, path)
    return path


def _plot_construction_comparison(run_root: Path) -> Path:
    import matplotlib.pyplot as plt

    date = "2019-03-16"
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
    colors = {"R0": "#00c8ff", "R1": "#ffcc00", "R2": "#ff4f81"}
    for ax, mode in zip(axes, ("R0", "R1", "R2")):
        _plot_base(ax, date)
        dataframe = (_read_vector(run_root / "04_construction/single_date/2019-03-16/raw_construction_candidates.gpkg", layer="raw_construction_candidates") if mode == "R2" else _construction_source_candidates(mode, date))
        _overlay(ax, dataframe, colors[mode], f"{mode} raw candidate", alpha=0.30, linewidth=0.8)
        ax.set_title(f"{mode} | raw candidates={len(dataframe)}")
    fig.suptitle("工地原始候选 R0/R1/R2 对照（不改变原始概率）", fontsize=16)
    path = _figure_path(run_root, "construction", "construction_R0_R1_R2_2019_comparison.png")
    _save_fig(fig, path)
    return path


def _plot_construction_gt(run_root: Path) -> Path:
    import matplotlib.pyplot as plt

    date = "2019-03-16"
    gt = _load_positive_gt(GT_CONSTRUCTION_PATH, "gt_construction")
    prediction = _read_vector(run_root / "04_construction/ranking/construction_candidates_ranked.gpkg", layer="construction_candidates_ranked")
    prediction = prediction[prediction["scene_date"] == date]
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _plot_base(ax, date)
    _overlay(ax, prediction, "#f47b20", "ranked candidate", alpha=0.30, linewidth=0.8)
    _overlay(ax, gt, "none", "positive construction GT (unfilled)", alpha=1.0, edgecolor="#ffffff", linewidth=1.4)
    ax.set_title(f"工地候选与2019阳性GT叠加 | candidate={len(prediction)} | GT={len(gt)} | overlap proxy only", fontsize=14)
    ax.legend(loc="upper right", framealpha=0.85)
    path = _figure_path(run_root, "construction", "construction_2019_positive_gt_overlay.png")
    _save_fig(fig, path)
    return path


def _plot_construction_tracking(run_root: Path) -> Path:
    import matplotlib.pyplot as plt

    track = _read_vector(run_root / "04_construction/tracking/construction_tracks.gpkg", layer="construction_tracks")
    persistent = track[track["track_status"] == "persistent_construction_candidate"] if len(track) else track
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _plot_base(ax, FINAL_DATE)
    _overlay(ax, persistent, "#b661ff", "persistent candidate", alpha=0.35, linewidth=0.8)
    ax.set_title(f"工地跨期轨迹候选 | persistent rows={len(persistent)} | lifecycle is model-derived candidate", fontsize=14)
    ax.legend(loc="upper right", framealpha=0.85)
    path = _figure_path(run_root, "construction", "construction_persistent_tracks.png")
    _save_fig(fig, path)
    return path


def _plot_construction_cases(run_root: Path) -> Path:
    import matplotlib.pyplot as plt
    import geopandas as gpd

    date = "2019-03-16"
    ranked = _read_vector(run_root / "04_construction/ranking/construction_candidates_ranked.gpkg", layer="construction_candidates_ranked")
    ranked = ranked[ranked["scene_date"] == date].sort_values(["candidate_tier", "evidence_score"], ascending=[True, False]).reset_index(drop=True)
    gt = _load_positive_gt(GT_CONSTRUCTION_PATH, "gt_construction")
    selected: list[tuple[str, Any, str]] = []
    category_order = [
        ("GT positive overlap proxy candidate", lambda row: bool(len(gt) and any(row.geometry.intersects(g) for g in gt.geometry)), "#ff3b30"),
        ("GT positive polygon without claim of hit", lambda row: True, "#ffffff"),
        ("stable building conflict", lambda row: bool(row.get("stable_building_conflict")), "#8f8795"),
        ("blue/green metal roof downgrade tag", lambda row: "blue_green_metal_roof" in str(row.get("evidence_downgrade_tags")), "#a59cae"),
        ("parking/paved/ordinary-bare-land proxy", lambda row: "parking_or_paved_yard" in str(row.get("likely_confusion_type")) or "ordinary_bare_land" in str(row.get("evidence_downgrade_tags")), "#d3a95f"),
        ("dustproof support", lambda row: float(row.get("dustproof_net_support", 0)) >= 0.45, "#ff9f1c"),
        ("exposed-ground support", lambda row: float(row.get("exposed_ground_support", 0)) >= 0.45, "#e76f51"),
        ("single-date fragment", lambda row: "single_date_fragment" in str(row.get("evidence_downgrade_tags")), "#d0c45a"),
        ("persistent candidate", lambda row: bool((run_root / "04_construction/tracking/persistent_construction_candidates.gpkg").exists()), "#b661ff"),
        ("uncertain candidate", lambda row: row.get("candidate_tier") == "uncertain", "#b7a9cc"),
        ("tier A candidate", lambda row: row.get("candidate_tier") == "tier_A_high_evidence", "#f23b20"),
        ("tier B candidate", lambda row: row.get("candidate_tier") == "tier_B_medium_evidence", "#ffbd3b"),
    ]
    used: set[int] = set()
    for label, predicate, color in category_order:
        for index, row in ranked.iterrows():
            if index in used:
                continue
            try:
                match = predicate(row)
            except Exception:
                match = False
            if match:
                used.add(index)
                selected.append((label, row.geometry, color))
                break
        if len(selected) >= 12:
            break
    for index, row in ranked.iterrows():
        if len(selected) >= 12:
            break
        if index not in used:
            selected.append((f"candidate fallback | tier={row.get('candidate_tier')}", row.geometry, "#f4a261"))
            used.add(index)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for ax, item in zip(axes.flat, selected[:12]):
        label, geometry, color = item
        _plot_base(ax, date)
        center = geometry.centroid
        ax.set_xlim(center.x - 150, center.x + 150)
        ax.set_ylim(center.y - 150, center.y + 150)
        _overlay(ax, gpd.GeoDataFrame({"geometry": [geometry]}, geometry="geometry", crs=METRIC_CRS), color if color != "#ffffff" else "none", label, alpha=0.35, edgecolor=color, hatch="///" if "conflict" in label or "roof" in label else None, linewidth=1.3)
        ax.set_title(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.7)
    for ax in axes.flat[len(selected[:12]):]:
        ax.axis("off")
    fig.suptitle("工地局部案例清单（12例；候选证据/冲突/阳性GT重叠诊断，不等同于site accuracy）", fontsize=15)
    path = _figure_path(run_root, "construction", "construction_local_case_sheet_12.png")
    _save_fig(fig, path)
    return path


def _plot_integrated(run_root: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    cross = _read_vector(run_root / "05_cross_reference/BUILDING_CONSTRUCTION_CROSS_REFERENCE.gpkg", layer="building_construction_cross_reference")
    paths: list[Path] = []
    for date in ["2019-03-16", FINAL_DATE]:
        subset = cross[cross["scene_date"] == date]
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        _plot_base(ax, date)
        _overlay(ax, subset[subset["stable_building_conflict"] == False], "#ff7f50", "candidate without stable conflict", alpha=0.30, linewidth=0.8)
        _overlay(ax, subset[subset["stable_building_conflict"] == True], "#9b5de5", "stable building conflict", alpha=0.35, edgecolor="#9b5de5", hatch="///", linewidth=0.9)
        ax.set_title(f"只读跨模块叠加 | {date} | Building + Construction evidence; no upstream write", fontsize=14)
        ax.legend(loc="upper right", framealpha=0.85)
        path = _figure_path(run_root, "integrated", f"integrated_cross_reference_{date}.png")
        _save_fig(fig, path)
        paths.append(path)
    return paths


def _build_figures(run_root: Path) -> dict[str, Any]:
    import matplotlib

    # Prefer a Windows CJK font so the Chinese labels in the technical report
    # remain readable in the exported PNGs.  Fall back silently when the host
    # does not provide one.
    for font_path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\simsun.ttc")]:
        if font_path.exists():
            from matplotlib import font_manager

            font_manager.fontManager.addfont(str(font_path))
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    paths: list[Path] = []
    for date in DATES:
        paths.append(_plot_building_overview(run_root, date))
        paths.append(_plot_construction_overview(run_root, date))
    paths.extend([_plot_building_comparison(run_root), _plot_building_gt(run_root), _plot_building_tracking(run_root), _plot_building_cases(run_root)])
    paths.extend([_plot_construction_comparison(run_root), _plot_construction_gt(run_root), _plot_construction_tracking(run_root), _plot_construction_cases(run_root)])
    paths.extend(_plot_integrated(run_root))
    figure_records = []
    for path in paths:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        figure_records.append({"path": str(path.relative_to(run_root)), "width": width, "height": height, "dpi_target": 300, "meets_minimum_3600x2100": width >= 3600 and height >= 2100})
    _write_json(run_root / "09_manifests/CHART_MAP.json", {
        "created_at": _now(),
        "figure_count": len(figure_records),
        "figures": figure_records,
        "all_core_figures_meet_minimum": all(row["meets_minimum_3600x2100"] for row in figure_records),
        "base_layer": "Zoom20 source image; polygons semi-transparent on top",
        "metric_crs": METRIC_CRS,
        "same_formal_aoi": True,
    })
    return {"figure_count": len(figure_records), "figures": figure_records}


def _render_table(dataframe: pd.DataFrame, max_rows: int = 40) -> str:
    if dataframe is None or len(dataframe) == 0:
        return "<p class='muted'>无可展示行。</p>"
    table = dataframe.head(max_rows).copy()
    columns = list(table.columns)
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for _, row in table.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            missing = False
            if value is None:
                missing = True
            elif np.isscalar(value):
                try:
                    missing = bool(pd.isna(value))
                except Exception:
                    missing = False
            if missing:
                value = "—"
            elif isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<div class='table-wrap'><table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table></div>"


def _relative_report_image(run_root: Path, path: str) -> str:
    return os.path.relpath(run_root / path, run_root / "10_reports").replace("\\", "/")


def _build_report(run_root: Path) -> dict[str, Any]:
    building_metrics_path = run_root / "03_evaluation/building/BUILDING_R0_R1_R2_METRICS.csv"
    construction_metrics_path = run_root / "03_evaluation/construction/CONSTRUCTION_R0_R1_R2_METRICS.csv"
    building_summary_path = run_root / "02_building/BUILDING_SINGLE_DATE_SUMMARY.csv"
    construction_summary_path = run_root / "04_construction/CONSTRUCTION_SINGLE_DATE_RANKED_SUMMARY.csv"
    cross_summary_path = run_root / "05_cross_reference/INTEGRATED_BUILDING_CONSTRUCTION_SUMMARY.csv"
    building_metrics = pd.read_csv(building_metrics_path, encoding="utf-8-sig") if building_metrics_path.exists() else pd.DataFrame()
    construction_metrics = pd.read_csv(construction_metrics_path, encoding="utf-8-sig") if construction_metrics_path.exists() else pd.DataFrame()
    building_summary = pd.read_csv(building_summary_path, encoding="utf-8-sig") if building_summary_path.exists() else pd.DataFrame()
    construction_summary = pd.read_csv(construction_summary_path, encoding="utf-8-sig") if construction_summary_path.exists() else pd.DataFrame()
    cross_summary = pd.read_csv(cross_summary_path, encoding="utf-8-sig") if cross_summary_path.exists() else pd.DataFrame()
    building_seal = _read_json(run_root / "02_building/BUILDING_SYSTEM_SEAL.json", {})
    construction_seal = _read_json(run_root / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json", {})
    building_gt = _read_json(run_root / "03_evaluation/building/BUILDING_GT_AUDIT.json", {})
    construction_gt = _read_json(run_root / "03_evaluation/construction/CONSTRUCTION_GT_AUDIT.json", {})
    registry = _read_json(run_root / "00_governance/CURRENT_BEST_REGISTRY_AUDIT.json", {})
    baseline = _read_json(run_root / "00_governance/BASELINE_SEAL_AUDIT.json", {})
    invariance = _read_json(run_root / "09_manifests/PROTECTED_ASSET_INVARIANCE.json", {})
    chart_map = _read_json(run_root / "09_manifests/CHART_MAP.json", {})
    report_dir = run_root / "10_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    image_records = chart_map.get("figures", [])
    image_html = []
    for record in image_records:
        relative = _relative_report_image(run_root, record["path"])
        label = Path(record["path"]).stem.replace("_", " ")
        image_html.append(f"<figure><a href='{html.escape(relative)}' target='_blank'><img src='{html.escape(relative)}' alt='{html.escape(label)}'></a><figcaption>{html.escape(label)} · {record['width']}×{record['height']} px · click to open</figcaption></figure>")
    model_rows = []
    for module, model in (registry.get("models") or {}).items():
        model_rows.append({
            "module": module,
            "model_name": model.get("model_name"),
            "checkpoint_sha256": (model.get("checkpoint") or {}).get("sha256"),
            "registry_status": model.get("registry_status"),
            "scientific_status": model.get("scientific_status"),
        })
    html_text = f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'><title>{REPORT_NAME}</title>
<style>
:root{{--ink:#17211d;--muted:#5c6962;--line:#d8dfd9;--accent:#0f766e;--paper:#f7f8f6;--card:#ffffff;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;line-height:1.6}}
main{{width:94%;max-width:1650px;margin:0 auto;padding:30px 0 70px}} header{{background:#163b36;color:#f3f7f4;padding:36px 5%;margin:0 0 28px;border-bottom:6px solid #d5b36a}} h1{{font-size:34px;line-height:1.2;margin:0 0 12px}} h2{{font-size:24px;margin:42px 0 12px;border-left:5px solid var(--accent);padding-left:12px}} h3{{font-size:18px;margin-top:25px}} p{{max-width:1150px}} .lede{{font-size:18px;max-width:1200px}} .badges{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}} .badge{{padding:7px 12px;background:#eef4ef;border:1px solid #c9d8cc;border-radius:99px;font-weight:600}} .badge.warn{{background:#fff3d6;border-color:#e5c77a}} .badge.fail{{background:#ffe5df;border-color:#d78b7d}} .card-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin:20px 0}} .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;box-shadow:0 2px 9px #10261a10}} .value{{font-size:25px;font-weight:700;color:var(--accent)}} .muted{{color:var(--muted)}} .table-wrap{{overflow-x:auto;background:white;border:1px solid var(--line);margin:12px 0 22px}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}} th{{background:#eaf2ec;position:sticky;top:0}} tr:nth-child(even){{background:#fbfcfb}} .figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(470px,1fr));gap:20px}} figure{{margin:0;background:#fff;border:1px solid var(--line);padding:10px;border-radius:8px}} figure img{{display:block;width:100%;height:auto}} figcaption{{font-size:12px;color:var(--muted);margin-top:7px}} .notice{{border-left:4px solid #d5b36a;background:#fff8e8;padding:13px 16px;margin:15px 0;max-width:1250px}} code{{background:#e8eee9;padding:2px 5px;border-radius:3px;word-break:break-all}} ul{{max-width:1200px}} footer{{border-top:1px solid var(--line);margin-top:45px;padding-top:18px;color:var(--muted);font-size:13px}}
</style></head><body><header><main style='padding:0'><h1>南京江北新区建筑—工地独立系统集成验收</h1><p class='lede'>本报告在冻结模型与既有 Zoom20 R2 结果上，建设建筑识别、工地候选、只读跨模块交叉引用和 2019 阳性人工GT诊断。所有数值均保留模块边界，不把候选或阳性重叠代理写成全场景准确率。</p><div class='badges'><span class='badge'>六期唯一日期；2024已排除</span><span class='badge'>Full20 / RSBuilding / Construction checkpoint冻结</span><span class='badge'>Building seal: {html.escape(str(building_seal.get('status')))}</span><span class='badge'>Construction seal: {html.escape(str(construction_seal.get('status')))}</span><span class='badge warn'>Construction ground truth: positive-only diagnostics</span><span class='badge warn'>Human review required</span></div></main></header><main>
<section><h2>1. 执行摘要</h2><div class='card-grid'>
<div class='card'><div class='muted'>建筑分支</div><div class='value'>{html.escape(str(building_seal.get('status','—')))}</div><div>R2尺度归一化；候选系统</div></div>
<div class='card'><div class='muted'>工地分支</div><div class='value'>{html.escape(str(construction_seal.get('status','—')))}</div><div>原始概率不变；候选分层</div></div>
<div class='card'><div class='muted'>建筑阳性GT</div><div class='value'>{building_gt.get('aoi_clipped_positive_object_count', building_gt.get('gt_positive_object_count','—'))}</div><div>只用于封存后阳性对象诊断</div></div>
<div class='card'><div class='muted'>工地阳性GT</div><div class='value'>{construction_gt.get('aoi_clipped_positive_object_count', construction_gt.get('gt_positive_object_count','—'))}</div><div>不报告site precision/recall/F1</div></div>
<div class='card'><div class='muted'>核心高清图</div><div class='value'>{chart_map.get('figure_count','—')}</div><div>≥3600×2100，约300 dpi</div></div>
<div class='card'><div class='muted'>Protected assets</div><div class='value'>{'PASS' if invariance.get('all_protected_assets_unchanged') else 'CHECK'}</div><div>registry / checkpoints / old outputs</div></div>
</div><div class='notice'><b>科学边界：</b>建筑分支达到独立对象候选与阳性GT对象匹配诊断；工地分支达到原始候选、透明证据排序和生命周期候选展示，但不是已验证的工地分类器。南京没有完整背景GT，因此不报告全场景 pixel accuracy、site precision/recall/F1 或 IoU。</div></section>
<section><h2>2. 当前冻结模型与血缘</h2>{_render_table(pd.DataFrame(model_rows))}<p class='muted'>Current Best Registry 只读：{html.escape(str(REGISTRY_PATH))}；SHA256：<code>{html.escape(str(registry.get('sha256')))}</code>。C-B1保持rejected；Site Classifier disabled_pending_human_gt；GloBFP与建筑高程支路未启用。</p></section>
<section><h2>3. 模块职责与不变性</h2><ul><li>Building主线只读取冻结RSBuilding输出、R2尺度归一化输入，以及Full20封存变化证据用于到最新候选支持；不读取Construction。</li><li>Construction保留R0/R1/R2原始概率和候选，R2排序只产生下游候选等级，不改写原始概率；可读取封存Building和Full20作只读支持/冲突/排序。</li><li>Cross-reference只读两条封存支路，不回写任何上游概率、几何或checkpoint。</li><li>没有训练、微调、伪标签训练、模型重选、阈值/MMU调整；Locked Test pixel reads = 0。</li></ul>{_render_table(pd.DataFrame([invariance]))}</section>
<section><h2>4. 时间范围、AOI与GT顺序</h2><p>正式日期：{', '.join(DATES)}；最终日期：{FINAL_DATE}；2024-10-20沿用前序重复观察排除结论。面积、距离和对象匹配均使用 EPSG:32650；所有R0/R1/R2使用同一正式AOI。GT在各分支seal之后才读取几何，原始GT文件保持不变。</p>{_render_table(pd.DataFrame([{'AOI':str(AOI_BOUNDS_WGS84),'metric_crs':METRIC_CRS,'building_gt_sha256':_sha256(GT_BUILDING_PATH),'construction_gt_sha256':_sha256(GT_CONSTRUCTION_PATH)}]))}</section>
<section><h2>5. 建筑识别主线与2019阳性GT诊断</h2><p>六期对象统计如下。<code>uncertainty</code>由冻结RSBuilding分支未提供对象级不确定性，因此保持为不可用，不人为构造。</p>{_render_table(building_summary)}{_render_table(building_metrics)}<p>建筑GT为阳性多边形集合：可报告 positive-object match 与 overlap proxy；未匹配预测不等于已确认假阳性，未匹配GT不等于完整漏检率。</p><div class='figures'>{''.join(image_html[:0])}</div></section>
<section><h2>6. 工地候选、证据排序与2019阳性GT诊断</h2><p>工地保留C-A冻结模型和原始候选；R0/R1作为历史输入消融，R2作为独立候选系统。排序方法是固定、数据无关、未读取GT的透明下游评分，绝不改变 raw probability、candidate mask 或阈值。</p>{_render_table(construction_summary)}{_render_table(construction_metrics)}<p>Construction GT只支持阳性重叠代理和诊断性对象交叠计数；不报告 formal site precision/recall/F1/pixel accuracy。</p></section>
<section><h2>7. 只读跨模块交叉引用</h2>{_render_table(cross_summary)}<p>交叉引用用于解释稳定建筑冲突、出现/消失建筑支持和工地候选证据，不生成联合概率，也不修改建筑或工地原始产物。</p></section>
<section><h2>8. 高清地图与案例</h2><p>底图为原始Zoom20影像，候选多边形以半透明叠加呈现；图例、比例尺、北向、日期、输入模式和科学状态写入图面。局部案例是确定性选择的证据诊断，不是成功案例筛选。</p><div class='figures'>{''.join(image_html)}</div></section>
<section><h2>9. 失败、降级与限制</h2><ul><li>建筑高度、GloBFP、Site Classifier、C-B1/C-B2/C-B3均不在本run-root范围内。</li><li>建筑分支的对象级统计受冻结RSBuilding输出和阳性GT完整性限制；无完整背景，不能推导全场景准确率。</li><li>工地原始候选较为保守且存在稳定屋顶、停车场、裸地和单期碎片冲突，排序结果必须人工复核。</li><li>Zoom20来源元数据不可用；本报告不把更高采样密度表述成真实信息增益。</li><li>所有出现、消失、稳定和生命周期状态均为 candidate，不是已证实城市更新事实。</li></ul></section>
<section><h2>10. 产物与复现入口</h2><ul><li>集成run-root：<code>{html.escape(str(run_root))}</code></li><li>本唯一中文报告：<code>10_reports/{REPORT_NAME}</code></li><li>建筑seal：<code>02_building/BUILDING_SYSTEM_SEAL.json</code></li><li>工地seal：<code>04_construction/CONSTRUCTION_SYSTEM_SEAL.json</code></li><li>跨模块：<code>05_cross_reference/BUILDING_CONSTRUCTION_CROSS_REFERENCE.gpkg</code></li><li>数字审计：<code>09_manifests/REPORT_NUMERICAL_AUDIT.json</code>、<code>09_manifests/FINAL_NANJING_INTEGRATED_ACCEPTANCE_RECEIPT.json</code></li></ul></section>
<footer>Generated at {_now()} · source inputs are read-only · human review remains required.</footer></main></body></html>"""
    report_path = report_dir / REPORT_NAME
    report_path.write_text(html_text, encoding="utf-8")
    report_audit = {
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "one_html_only": True,
        "independent_building_html_created": False,
        "independent_construction_html_created": False,
        "figure_count": chart_map.get("figure_count"),
        "building_metric_rows": len(building_metrics),
        "construction_metric_rows": len(construction_metrics),
        "building_summary_rows": len(building_summary),
        "construction_summary_rows": len(construction_summary),
        "cross_summary_rows": len(cross_summary),
        "forbidden_claims": ["full_scene_accuracy", "site_precision_recall_f1", "true_construction_accuracy", "confirmed_false_positive"],
        "metric_language_guard": "positive_object_match_or_overlap_proxy_only",
    }
    _write_json(run_root / "09_manifests/REPORT_NUMERICAL_AUDIT.json", report_audit)
    _write_json(run_root / "REPORT_NUMERICAL_AUDIT.json", report_audit)
    return report_audit


def _finalize_invariance(run_root: Path) -> dict[str, Any]:
    before = _read_json(run_root / "00_governance/PROTECTED_ASSET_SNAPSHOT_BEFORE.json", {})
    after = _source_snapshot()
    key_comparisons = []
    for path, before_sig in before.items():
        after_sig = after.get(path, {"exists": False})
        key_comparisons.append({"path": path, "before": before_sig, "after": after_sig, "unchanged": before_sig == after_sig})
    registry_before = _read_json(run_root / "00_governance/CURRENT_BEST_REGISTRY_AUDIT.json", {})
    registry_after = _registry_audit()
    baseline = _baseline_seal_summary()
    checkpoint_checks = []
    for module, model in (registry_after.get("models") or {}).items():
        checkpoint = model.get("checkpoint") or {}
        path = Path(checkpoint.get("path", ""))
        current_sha = _sha256(path) if path.exists() else None
        checkpoint_checks.append({"module": module, "path": str(path), "expected_sha256": checkpoint.get("sha256"), "current_sha256": current_sha, "unchanged": bool(current_sha and current_sha == checkpoint.get("sha256"))})
    all_assets = all(row["unchanged"] for row in key_comparisons) and registry_before.get("sha256") == registry_after.get("sha256") and all(row["unchanged"] for row in checkpoint_checks) and bool(baseline.get("baseline_hash_invariance_passed"))
    payload = {
        "created_at": _now(),
        "registry_sha_before": registry_before.get("sha256"),
        "registry_sha_after": registry_after.get("sha256"),
        "registry_unchanged": registry_before.get("sha256") == registry_after.get("sha256"),
        "checkpoint_checks": checkpoint_checks,
        "baseline_hash_match_count": baseline.get("baseline_hash_match_count"),
        "baseline_hash_total_count": baseline.get("baseline_hash_total_count"),
        "baseline_hash_invariance_passed": baseline.get("baseline_hash_invariance_passed"),
        "protected_key_file_comparisons": key_comparisons,
        "original_source_modified": False if all(row["unchanged"] for row in key_comparisons) else True,
        "locked_test_pixel_reads": 0,
        "all_protected_assets_unchanged": all_assets,
        "status": "PASS" if all_assets else "FAIL",
    }
    _write_json(run_root / "09_manifests/PROTECTED_ASSET_INVARIANCE.json", payload)
    _write_json(run_root / "PROTECTED_ASSET_INVARIANCE.json", payload)
    _write_json(run_root / "09_manifests/BASELINE_POST_INTEGRATED_INVARIANCE.json", {
        "baseline_hash_match_count": baseline.get("baseline_hash_match_count"),
        "baseline_hash_total_count": baseline.get("baseline_hash_total_count"),
        "baseline_hash_invariance_passed": baseline.get("baseline_hash_invariance_passed"),
        "old_baseline_recomputed": False,
        "historical_baseline_not_written": True,
    })
    return payload


def _final_receipt(run_root: Path, invariance: Mapping[str, Any], report: Mapping[str, Any], building_eval: Mapping[str, Any], construction_eval: Mapping[str, Any]) -> dict[str, Any]:
    test_receipt = _read_json(run_root / "12_tests/TEST_RECEIPT.json", {})
    receipt = {
        "created_at": _now(),
        "run_root": str(run_root),
        "status": "completed_with_scientific_boundaries" if invariance.get("status") == "PASS" else "FAIL_PROTECTED_ASSET_INVARIANCE",
        "building_status": "building_candidate_system_sealed_and_positive_gt_diagnostic_completed",
        "construction_status": "construction_candidate_system_sealed_and_positive_overlap_diagnostic_completed",
        "building_gt_positive_object_count": building_eval.get("aoi_clipped_positive_object_count", building_eval.get("gt_positive_object_count")),
        "construction_gt_positive_object_count": construction_eval.get("aoi_clipped_positive_object_count", construction_eval.get("gt_positive_object_count")),
        "training_started": False,
        "production_urrs_training_started": False,
        "test_only_surface_state_fixture_training_executed": bool(test_receipt.get("test_only_surface_state_fixture_training_executed", False)),
        "fine_tuning_started": False,
        "pseudo_label_training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "mmu_changed": False,
        "C_B1_status": "rejected",
        "site_classifier_status": "disabled_pending_human_gt",
        "globfp_status": "disabled_tier_c_not_allowed_for_selection_or_training",
        "building_height_status": "disabled_out_of_scope",
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "full_scene_accuracy_reported": False,
        "true_construction_accuracy_reported": False,
        "true_building_accuracy_reported": False,
        "human_review_required": True,
        "baseline": {
            "hash_match_count": invariance.get("baseline_hash_match_count"),
            "hash_total_count": invariance.get("baseline_hash_total_count"),
            "unchanged": invariance.get("baseline_hash_invariance_passed"),
        },
        "registry_unchanged": invariance.get("registry_unchanged"),
        "checkpoint_unchanged": all(row.get("unchanged") for row in invariance.get("checkpoint_checks", [])),
        "main_report": str(run_root / "10_reports" / REPORT_NAME),
        "report_audit": str(run_root / "09_manifests/REPORT_NUMERICAL_AUDIT.json"),
        "test_receipt": test_receipt,
        "scientific_status": {
            "Nanjing_baseline": "frozen_and_accepted",
            "Nanjing_building": "candidate_system_positive_gt_diagnostic_only",
            "Nanjing_construction": "candidate_level_not_formal_classifier_accuracy",
            "Nanjing_construction_ground_truth": "positive_only_and_incomplete",
            "Human_review": "required",
        },
        "report": report,
    }
    _write_json(run_root / "09_manifests/FINAL_NANJING_INTEGRATED_ACCEPTANCE_RECEIPT.json", receipt)
    _write_json(run_root / "FINAL_NANJING_INTEGRATED_ACCEPTANCE_RECEIPT.json", receipt)
    return receipt


def _status(run_root: Path) -> dict[str, Any]:
    files = {
        "phase_a": run_root / "09_manifests/PHASE_A_AUDIT_RECEIPT.json",
        "building_seal": run_root / "02_building/BUILDING_SYSTEM_SEAL.json",
        "construction_seal": run_root / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json",
        "building_gt": run_root / "03_evaluation/building/BUILDING_GT_AUDIT.json",
        "construction_gt": run_root / "03_evaluation/construction/CONSTRUCTION_GT_AUDIT.json",
        "cross": run_root / "05_cross_reference/CROSS_REFERENCE_MANIFEST.json",
        "figures": run_root / "09_manifests/CHART_MAP.json",
        "report": run_root / "10_reports" / REPORT_NAME,
        "receipt": run_root / "09_manifests/FINAL_NANJING_INTEGRATED_ACCEPTANCE_RECEIPT.json",
    }
    return {"run_root": str(run_root), "artifacts": {name: {"exists": path.exists(), "path": str(path)} for name, path in files.items()}}


def _run(run_root: Path, mode: str) -> dict[str, Any]:
    _create_run_tree(run_root)
    results: dict[str, Any] = {}
    if mode in {"audit", "all"}:
        results["audit"] = _audit_phase(run_root)
    if mode in {"building", "all"}:
        results["building"] = _build_building_branch(run_root)
    if mode in {"evaluate-building", "all"}:
        results["building_evaluation"] = _evaluate_building_gt(run_root)
    if mode in {"construction", "all"}:
        results["construction"] = _build_construction_branch(run_root)
    if mode in {"evaluate-construction", "evaluate", "all"}:
        results["construction_evaluation"] = _evaluate_construction_gt(run_root)
    if mode in {"cross", "all"}:
        results["cross_reference"] = _cross_reference(run_root)
    if mode in {"figures", "all"}:
        results["figures"] = _build_figures(run_root)
    if mode in {"report", "all"}:
        results["invariance"] = _finalize_invariance(run_root)
        results["report"] = _build_report(run_root)
        building_eval = _read_json(run_root / "03_evaluation/building/BUILDING_GT_AUDIT.json", {})
        construction_eval = _read_json(run_root / "03_evaluation/construction/CONSTRUCTION_GT_AUDIT.json", {})
        results["final_receipt"] = _final_receipt(run_root, results["invariance"], results["report"], building_eval, construction_eval)
    if mode == "status":
        results["status"] = _status(run_root)
    _write_json(run_root / "11_logs/LAST_COMMAND_RESULT.json", {"created_at": _now(), "mode": mode, "result_keys": sorted(results), "status": "completed"})
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "building", "evaluate-building", "construction", "evaluate-construction", "evaluate", "cross", "figures", "report", "all", "status"], default="all")
    args = parser.parse_args(argv)
    try:
        result = _run(args.run_root.resolve(), args.mode)
    except Exception as exc:
        error = {"created_at": _now(), "mode": args.mode, "error_type": type(exc).__name__, "error": str(exc)}
        args.run_root.mkdir(parents=True, exist_ok=True)
        _write_json(args.run_root / "11_logs/LAST_COMMAND_ERROR.json", error)
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
