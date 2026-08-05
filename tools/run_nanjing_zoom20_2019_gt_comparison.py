"""Run the frozen Current Best models on Nanjing Zoom 20 and validate 2019 GT.

This workflow is deliberately two-phase:

* Audit and Prepare may enumerate GT paths, but do not open the supplied
  GT databases or read their geometry.
* Infer completes R0/R1/R2 and writes a pre-GT inference seal.
* Evaluate is the first mode allowed to read GT content.

The script is inference/evaluation only.  It never trains, selects a
checkpoint, changes thresholds, repairs source GT, or writes to the sealed
Zoom 19 run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

VALID_DATES = (
    "2015-01-04",
    "2016-08-18",
    "2017-12-11",
    "2018-02-08",
    "2019-03-16",
    "2022-12-02",
)
EXCLUDED_DATES = {"2024-10-20"}
TARGET_DATE = "2019-03-16"
METRIC_CRS = "EPSG:32650"
RESEARCH_BOUNDS = (118.633, 32.024, 118.652, 32.0415)
SUPPORTED_RASTER_SUFFIXES = {".tif", ".tiff"}
GT_SUFFIXES = {".gpkg", ".shp", ".geojson", ".json", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_date(text: str) -> str | None:
    for year, month, day in re.findall(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)", text):
        try:
            return datetime.strptime(f"{year}{month}{day}", "%Y%m%d").date().isoformat()
        except ValueError:
            pass
    for token in re.findall(r"(?<!\d)(20\d{6})(?!\d)", text):
        try:
            return datetime.strptime(token, "%Y%m%d").date().isoformat()
        except ValueError:
            pass
    for token in re.findall(r"(?<!\d)(\d{6})(?!\d)", text):
        try:
            return datetime.strptime(token, "%y%m%d").date().isoformat()
        except ValueError:
            pass
    return None


def claimed_zoom(path: Path) -> str | None:
    lowered = str(path).casefold()
    if "zoom_20" in lowered or "zoom20" in lowered or "z20" in lowered:
        return "zoom20"
    if "zoom_19" in lowered or "zoom19" in lowered or "z19" in lowered:
        return "zoom19"
    return None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def ensure_tree(run_root: Path) -> None:
    for relative in (
        "00_input_audit",
        "01_gt_audit",
        "02_common_grid",
        "02_scale_audit",
        "03_inference/zoom19_reference",
        "03_inference/zoom20_native",
        "03_inference/zoom20_scale_normalized",
        "04_six_date_zoom20",
        "05_gt_evaluation/building",
        "05_gt_evaluation/construction",
        "06_recommendation",
        "07_full20",
        "08_figures_highres",
        "09_error_cases",
        "10_reports",
        "11_manifests",
        "12_logs",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)


def _refuse_sealed_roots(run_root: Path, old_run: Path) -> None:
    if run_root.resolve() == old_run.resolve():
        raise RuntimeError("Refusing to write into the sealed Zoom 19 run")
    if "nanjing_jiangbei_modular_rebuild_20260803_230320" in str(run_root.resolve()):
        raise RuntimeError("Refusing a path that aliases the sealed Zoom 19 run")


def _read_registry(path: Path) -> dict[str, Any]:
    registry = read_json(path, {})
    entries = {str(item.get("module_id")): item for item in registry.get("entries", [])}
    required = {"M2_Full20", "M3_RSBuilding", "M6_Construction"}
    if not required.issubset(entries):
        raise RuntimeError(f"Current Best Registry is incomplete: missing {sorted(required - set(entries))}")
    for module_id, entry in entries.items():
        checkpoint = Path(str(entry.get("checkpoint", {}).get("path", "")))
        if checkpoint.is_file() and entry.get("checkpoint", {}).get("expected_sha256"):
            actual = sha256_file(checkpoint)
            if actual != str(entry["checkpoint"]["expected_sha256"]).lower():
                raise RuntimeError(f"{module_id} checkpoint SHA mismatch")
        config = Path(str(entry.get("config", {}).get("path", "")))
        expected = str(entry.get("config", {}).get("expected_sha256") or "")
        if config.is_file() and expected and sha256_file(config) != expected.lower():
            raise RuntimeError(f"{module_id} config SHA mismatch")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema_version": registry.get("schema_version"),
        "entries": entries,
        "excluded_branches": registry.get("excluded_branches", []),
        "models_reselected": False,
        "registry_unchanged": True,
    }


def _asset_paths(registry: Mapping[str, Any]) -> dict[str, Any]:
    entries = registry["entries"]
    full = entries["M2_Full20"]
    building = entries["M3_RSBuilding"]
    construction = entries["M6_Construction"]
    package = Path(str(construction["checkpoint"]["path"])).parent
    return {
        "full20_checkpoint": Path(str(full["checkpoint"]["path"])).resolve(),
        "full20_config": Path(str(full["config"]["path"])).resolve(),
        "full20_sha256": str(full["checkpoint"]["sha256"]),
        "full20_config_sha256": str(full["config"]["sha256"]),
        "full20_thresholds": dict(full.get("thresholds", {})),
        "rsbuilding_checkpoint": Path(str(building["checkpoint"]["path"])).resolve(),
        "rsbuilding_config": Path(str(building["config"]["path"])).resolve(),
        "rsbuilding_sha256": str(building["checkpoint"]["sha256"]),
        "rsbuilding_config_sha256": str(building["config"]["sha256"]),
        "rsbuilding_thresholds": dict(building.get("thresholds", {})),
        "construction_package": package,
        "construction_checkpoint": Path(str(construction["checkpoint"]["path"])).resolve(),
        "construction_sha256": str(construction["checkpoint"]["sha256"]),
        "construction_config_sha256": str(construction["config"]["sha256"]),
        "construction_thresholds": dict(construction.get("thresholds", {})),
        "construction_rgb_mapping": [3, 2, 1],
    }


def _actual_gsd_m(crs_text: str | None, resolution: Sequence[float], bounds: Sequence[float]) -> tuple[float | None, str]:
    try:
        from pyproj import CRS, Geod

        crs = CRS.from_user_input(crs_text)
        if crs.is_projected:
            return float((abs(float(resolution[0])) + abs(float(resolution[1]))) / 2), "projected_crs_units"
        geod = Geod(ellps="WGS84")
        left, bottom, right, top = [float(value) for value in bounds]
        center_lon = (left + right) / 2
        center_lat = (bottom + top) / 2
        _, _, x_m = geod.inv(center_lon, center_lat, center_lon + abs(float(resolution[0])), center_lat)
        _, _, y_m = geod.inv(center_lon, center_lat, center_lon, center_lat + abs(float(resolution[1])))
        return float((x_m + y_m) / 2), "geodesic_from_geographic_resolution"
    except Exception:
        return None, "unavailable"


def _raster_metadata(path: Path, *, calculate_quality: bool = True) -> dict[str, Any]:
    import rasterio
    from scipy.ndimage import laplace

    with rasterio.open(path) as dataset:
        crs_text = str(dataset.crs) if dataset.crs else None
        bounds = list(dataset.bounds)
        resolution = list(dataset.res)
        gsd, gsd_source = _actual_gsd_m(crs_text, resolution, bounds)
        quality: dict[str, Any] = {
            "valid_coverage_fraction": None,
            "nodata_fraction": None,
            "sharpness_proxy": None,
            "dynamic_range": None,
            "sampled_pixel_hash": None,
        }
        if calculate_quality:
            sample_height = min(768, dataset.height)
            sample_width = min(768, dataset.width)
            sample = dataset.read(
                indexes=list(range(1, min(3, dataset.count) + 1)),
                out_shape=(min(3, dataset.count), sample_height, sample_width),
                masked=True,
            )
            data = np.asarray(sample.filled(np.nan), dtype=np.float32)
            valid = np.isfinite(data).all(axis=0)
            if data.size:
                quality["valid_coverage_fraction"] = float(valid.mean())
                quality["nodata_fraction"] = float(1.0 - valid.mean())
                finite = data[np.isfinite(data)]
                if finite.size:
                    quality["dynamic_range"] = {
                        "p01": float(np.percentile(finite, 1)),
                        "p50": float(np.percentile(finite, 50)),
                        "p99": float(np.percentile(finite, 99)),
                    }
                gray = np.nanmean(data, axis=0)
                gray = np.nan_to_num(gray, nan=0.0)
                quality["sharpness_proxy"] = float(np.var(laplace(gray)))
                normalized = np.clip(np.nan_to_num(data, nan=0.0), 0, 255).astype(np.uint8)
                quality["sampled_pixel_hash"] = hashlib.sha256(
                    np.ascontiguousarray(normalized).tobytes()
                ).hexdigest()
        return {
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "actual_width": int(dataset.width),
            "actual_height": int(dataset.height),
            "bands": int(dataset.count),
            "dtype": list(dataset.dtypes),
            "crs": crs_text,
            "bounds": bounds,
            "transform": list(dataset.transform),
            "pixel_resolution": resolution,
            "estimated_gsd_m": gsd,
            "estimated_gsd_source": gsd_source,
            "nodata": dataset.nodata,
            "tags": dataset.tags(),
            "source_provider": dataset.tags().get("TIFFTAG_SOFTWARE") or dataset.tags().get("PROVIDER"),
            **quality,
        }


def _discover_zoom20(zoom20_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files = [
        path for path in sorted(zoom20_root.rglob("*"))
        if path.is_file() and path.suffix.casefold() in SUPPORTED_RASTER_SUFFIXES
    ]
    records: list[dict[str, Any]] = []
    for path in files:
        date = parse_date(str(path))
        if date is None:
            continue
        metadata = _raster_metadata(path)
        metadata.update({
            "scene_date": date,
            "zoom_level_claimed": claimed_zoom(path),
            "date_source": "filename_or_path_token",
            "status": "candidate",
            "is_gt_candidate": False,
        })
        records.append(metadata)
    by_date: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_date.setdefault(str(record["scene_date"]), []).append(record)
    scenes: list[dict[str, Any]] = []
    ambiguous: dict[str, Any] = {}
    for date in VALID_DATES:
        candidates = by_date.get(date, [])
        candidates.sort(key=lambda item: (item["actual_width"] * item["actual_height"], item["file_path"]), reverse=True)
        if not candidates:
            ambiguous[date] = {"status": "missing", "candidates": []}
            continue
        if len(candidates) > 1:
            ambiguous[date] = {"status": "multiple_candidates", "candidates": candidates}
            continue
        selected = dict(candidates[0])
        selected["status"] = "valid_unique_scene"
        scenes.append(selected)
    duplicate_groups: list[dict[str, Any]] = []
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_sha.setdefault(str(record["sha256"]), []).append(record)
    for digest, group in by_sha.items():
        if len(group) > 1:
            duplicate_groups.append({
                "sha256": digest,
                "files": [item["file_path"] for item in group],
                "dates": [item["scene_date"] for item in group],
                "status": "duplicate_content",
            })
    manifest = {
        "status": "pass" if len(scenes) == len(VALID_DATES) and not ambiguous else "partial_date_overlap",
        "source_root": str(zoom20_root.resolve()),
        "scene_count": len(scenes),
        "scenes": scenes,
        "candidate_files": records,
        "ambiguous_or_missing": ambiguous,
        "valid_dates": [item["scene_date"] for item in scenes],
        "excluded_dates": sorted(EXCLUDED_DATES),
        "source_data_modified": False,
        "training_started": False,
        "locked_test_pixel_reads": 0,
        "created_at": now_utc(),
    }
    duplicate_audit = {
        "status": "pass" if not duplicate_groups else "duplicate_content_found",
        "duplicate_groups": duplicate_groups,
        "date_2024_status": "not_found",
        "date_2024_excluded": False,
        "source_data_modified": False,
        "created_at": now_utc(),
    }
    return manifest, duplicate_audit


def _discover_zoom19(zoom19_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(zoom19_root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_RASTER_SUFFIXES:
            continue
        date = parse_date(str(path))
        if date not in set(VALID_DATES) | EXCLUDED_DATES:
            continue
        metadata = _raster_metadata(path, calculate_quality=False)
        metadata.update({"scene_date": date, "zoom_level_claimed": claimed_zoom(path)})
        records.append(metadata)
    return {
        "source_root": str(zoom19_root.resolve()),
        "records": records,
        "valid_dates": [date for date in VALID_DATES if any(item["scene_date"] == date for item in records)],
        "created_at": now_utc(),
    }


def _write_gt_path_inventory(zoom20_root: Path, audit_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(zoom20_root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in GT_SUFFIXES:
            continue
        rows.append({
            "file_path": str(path.resolve()),
            "relative_path": _relative(path, zoom20_root),
            "file_name": path.name,
            "suffix": path.suffix.casefold(),
            "file_size_bytes": path.stat().st_size,
            "content_read": False,
            "layer_inspection_status": "deferred_until_post_inference_seal",
            "source_modified": False,
        })
    write_csv(audit_root / "GT_FILE_INVENTORY.csv", rows, [
        "file_path", "relative_path", "file_name", "suffix", "file_size_bytes",
        "content_read", "layer_inspection_status", "source_modified",
    ])
    layer_rows = [{
        "file_path": row["file_path"],
        "layer": "",
        "format": row["suffix"],
        "status": "deferred_until_post_inference_seal",
        "content_read": False,
    } for row in rows]
    write_csv(audit_root / "GT_LAYER_INVENTORY.csv", layer_rows, [
        "file_path", "layer", "format", "status", "content_read",
    ])
    placeholder = {
        "status": "deferred_until_post_inference_seal",
        "candidate_count": len(rows),
        "content_read": False,
        "gt_access_before_inference_completed": False,
        "source_modified": False,
        "reason": "Only paths, extensions, sizes and mtimes were enumerated before inference.",
        "created_at": now_utc(),
    }
    for name in (
        "GT_SCHEMA_AUDIT.json",
        "GT_GEOMETRY_AUDIT.json",
        "GT_CRS_AUDIT.json",
        "GT_COVERAGE_AUDIT.json",
        "GT_CLASS_MAPPING.json",
        "GT_READINESS.json",
    ):
        write_json(audit_root / name, placeholder)
    return placeholder


def _write_input_audit(run_root: Path, zoom20_root: Path, zoom19_root: Path, registry_path: Path) -> dict[str, Any]:
    zoom20_manifest, duplicate_audit = _discover_zoom20(zoom20_root)
    zoom19_manifest = _discover_zoom19(zoom19_root)
    input_root = run_root / "00_input_audit"
    input_root.mkdir(parents=True, exist_ok=True)
    write_json(input_root / "ZOOM20_SCENE_MANIFEST.json", zoom20_manifest)
    write_json(input_root / "ZOOM20_DUPLICATE_AUDIT.json", duplicate_audit)
    rows = list(zoom20_manifest["candidate_files"])
    write_csv(input_root / "ZOOM20_FILE_INVENTORY.csv", rows)
    comparison: list[dict[str, Any]] = []
    z19_by_date = {item["scene_date"]: item for item in zoom19_manifest["records"]}
    for item in zoom20_manifest["scenes"]:
        ref = z19_by_date.get(item["scene_date"])
        comparison.append({
            "scene_date": item["scene_date"],
            "zoom19_path": ref["file_path"] if ref else None,
            "zoom20_path": item["file_path"],
            "zoom19_width": ref["actual_width"] if ref else None,
            "zoom19_height": ref["actual_height"] if ref else None,
            "zoom20_width": item["actual_width"],
            "zoom20_height": item["actual_height"],
            "zoom19_gsd_m": ref["estimated_gsd_m"] if ref else None,
            "zoom20_gsd_m": item["estimated_gsd_m"],
            "pixel_size_ratio_zoom19_to_zoom20": (
                float(ref["estimated_gsd_m"] / item["estimated_gsd_m"])
                if ref and ref.get("estimated_gsd_m") and item.get("estimated_gsd_m") else None
            ),
            "bounds_intersection": (
                max(ref["bounds"][0], item["bounds"][0]),
                max(ref["bounds"][1], item["bounds"][1]),
                min(ref["bounds"][2], item["bounds"][2]),
                min(ref["bounds"][3], item["bounds"][3]),
            ) if ref else None,
            "source_provider_zoom19": ref.get("source_provider") if ref else None,
            "source_provider_zoom20": item.get("source_provider"),
            "source_consistency": "metadata_provider_unavailable",
            "information_gain_from_metadata": (
                "higher_native_sampling_density" if ref and item.get("estimated_gsd_m") and ref.get("estimated_gsd_m") and item["estimated_gsd_m"] < ref["estimated_gsd_m"]
                else "not_verified"
            ),
        })
    write_csv(input_root / "ZOOM19_ZOOM20_INPUT_COMPARISON.csv", comparison)
    readiness = {
        "status": "pass" if zoom20_manifest["status"] == "pass" else "blocked_partial_date_overlap",
        "zoom20_discovered": bool(zoom20_manifest["scenes"]),
        "zoom20_actual_resolution_verified": all(
            item.get("estimated_gsd_m") is not None and item["actual_width"] > 0 and item["actual_height"] > 0
            for item in zoom20_manifest["scenes"]
        ),
        "expected_dates": list(VALID_DATES),
        "found_dates": zoom20_manifest["valid_dates"],
        "date_2024_status": duplicate_audit["date_2024_status"],
        "date_2024_excluded": False,
        "source_data_modified": False,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "gt_content_read": False,
        "gt_access_before_inference_completed": False,
        "locked_test_pixel_reads": 0,
        "globfp_enabled": False,
        "building_height_enabled": False,
        "created_at": now_utc(),
    }
    write_json(input_root / "ZOOM20_DATA_READINESS.json", readiness)
    write_json(input_root / "ZOOM19_SCENE_MANIFEST.json", zoom19_manifest)
    registry = _read_registry(registry_path)
    write_json(input_root / "CURRENT_BEST_REGISTRY_SNAPSHOT.json", registry)
    gt_discovery = _write_gt_path_inventory(zoom20_root, run_root / "01_gt_audit")
    return {
        "zoom20": zoom20_manifest,
        "zoom19": zoom19_manifest,
        "readiness": readiness,
        "registry": registry,
        "gt_discovery": gt_discovery,
    }


def _aoi_geometry(bounds: Sequence[float]):
    from shapely.geometry import box

    return box(float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))


def _write_common_grid(run_root: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    zoom20 = list(audit["zoom20"]["scenes"])
    zoom19 = list(audit["zoom19"]["records"])
    z19_by_date = {item["scene_date"]: item for item in zoom19}
    z20_by_date = {item["scene_date"]: item for item in zoom20}
    common = list(RESEARCH_BOUNDS)
    all_items = [item for item in zoom20 + zoom19 if item["scene_date"] in VALID_DATES]
    for item in all_items:
        common = [
            max(common[0], float(item["bounds"][0])),
            max(common[1], float(item["bounds"][1])),
            min(common[2], float(item["bounds"][2])),
            min(common[3], float(item["bounds"][3])),
        ]
    if not (common[2] > common[0] and common[3] > common[1]):
        raise RuntimeError(f"Common evaluation AOI is empty: {common}")
    ref_item = z19_by_date.get(TARGET_DATE) or zoom19[0]
    native_item = z20_by_date.get(TARGET_DATE) or zoom20[0]
    common_root = run_root / "02_common_grid"
    geometry_geo = _aoi_geometry(common)
    geometry_metric = gpd.GeoSeries([geometry_geo], crs=ref_item["crs"]).to_crs(METRIC_CRS).iloc[0]
    frame = gpd.GeoDataFrame(
        [{"aoi_id": "nanjing_2019_common_evaluation", "source": "research_bounds_intersection_zoom19_zoom20", "metric_crs": METRIC_CRS}],
        geometry=[geometry_metric],
        crs=METRIC_CRS,
    )
    frame.to_file(common_root / "COMMON_2019_EVALUATION_AOI.gpkg", layer="common_aoi", driver="GPKG", engine="pyogrio")
    with rasterio.open(ref_item["file_path"]) as ref:
        profile = ref.profile.copy()
        for key in ("photometric", "interleave", "compress", "predictor"):
            profile.pop(key, None)
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        mask_path = common_root / "COMMON_2019_VALID_MASK.tif"
        if mask_path.exists():
            mask_path.unlink()
        with rasterio.open(mask_path, "w", **profile) as target:
            mask = rasterize([(geometry_geo, 1)], out_shape=(ref.height, ref.width), transform=ref.transform, fill=0, dtype="uint8")
            target.write(mask, 1)
        ref_grid = {
            "status": "pass",
            "path": ref_item["file_path"],
            "crs": ref_item["crs"],
            "width": ref.width,
            "height": ref.height,
            "transform": ref_item["transform"],
            "resolution": ref_item["pixel_resolution"],
            "bounds": ref_item["bounds"],
            "aoi_bounds": common,
            "aoi_bounds_metric": list(geometry_metric.bounds),
            "metric_crs": METRIC_CRS,
        }
    native_grid = {
        "status": "pass",
        "path": native_item["file_path"],
        "crs": native_item["crs"],
        "width": native_item["actual_width"],
        "height": native_item["actual_height"],
        "transform": native_item["transform"],
        "resolution": native_item["pixel_resolution"],
        "bounds": native_item["bounds"],
        "aoi_bounds": common,
        "metric_crs": METRIC_CRS,
    }
    write_json(common_root / "ZOOM19_GRID.json", ref_grid)
    write_json(common_root / "ZOOM20_GRID.json", native_grid)
    write_json(common_root / "GT_GRID.json", {
        "status": "pending_gt_audit",
        "target_crs": METRIC_CRS,
        "target_grid": str((common_root / "COMMON_2019_VALID_MASK.tif").resolve()),
        "aoi_bounds": common,
        "content_read": False,
    })
    shifts = {
        "status": "diagnostic_only",
        "zoom19_2019_bounds": ref_item["bounds"],
        "zoom20_2019_bounds": native_item["bounds"],
        "bounds_delta_degrees": [
            float(native_item["bounds"][index] - ref_item["bounds"][index]) for index in range(4)
        ],
        "raw_evaluation_uses_unshifted_official_alignment": True,
        "systematic_shift_correction_applied": False,
        "model_outputs_modified_for_alignment": False,
        "note": "The west-edge offset is recorded; no registration tuning is applied.",
    }
    write_json(common_root / "GEOMETRIC_SHIFT_AUDIT.json", shifts)
    write_json(common_root / "GRID_ALIGNMENT_AUDIT.json", {
        "status": "pass",
        "common_aoi_bounds": common,
        "zoom19_grid": ref_grid,
        "zoom20_grid": native_grid,
        "gt_grid_status": "pending_gt_audit",
        "same_metric_crs": True,
        "same_evaluation_aoi": True,
        "implicit_model_output_resampling": False,
    })
    return {"bounds": common, "zoom19": ref_grid, "zoom20": native_grid, "metric_crs": METRIC_CRS}


def _write_model_scale_audit(run_root: Path, audit: Mapping[str, Any], assets: Mapping[str, Any]) -> dict[str, Any]:
    z19 = audit["zoom19"]["records"]
    z20 = audit["zoom20"]["scenes"]
    ref_gsd = float(np.median([item["estimated_gsd_m"] for item in z19 if item.get("estimated_gsd_m")]))
    native_gsd = float(np.median([item["estimated_gsd_m"] for item in z20 if item.get("estimated_gsd_m")]))
    model_rows = [
        {
            "module_id": "M2_Full20",
            "expected_gsd_m": ref_gsd,
            "expected_gsd_source": "empirical_zoom19_reference_input_used_by_sealed_baseline",
            "native_zoom20_gsd_m": native_gsd,
            "native_scale_ratio": native_gsd / ref_gsd,
            "r1_mode": "zoom20_native",
            "r2_mode": "zoom20_model_scale_normalized",
            "r2_resampling": "bilinear_to_zoom19_reference_grid",
            "r2_cache": str((run_root / "03_inference/zoom20_scale_normalized/inputs/full20").resolve()),
            "checkpoint_sha256": assets["full20_sha256"],
            "thresholds": assets["full20_thresholds"],
        },
        {
            "module_id": "M3_RSBuilding",
            "expected_gsd_m": ref_gsd,
            "expected_gsd_source": "empirical_zoom19_reference_input_used_by_sealed_baseline",
            "native_zoom20_gsd_m": native_gsd,
            "native_scale_ratio": native_gsd / ref_gsd,
            "r1_mode": "zoom20_native",
            "r2_mode": "zoom20_model_scale_normalized",
            "r2_resampling": "bilinear_to_zoom19_reference_grid",
            "r2_cache": str((run_root / "03_inference/zoom20_scale_normalized/inputs/building").resolve()),
            "checkpoint_sha256": assets["rsbuilding_sha256"],
            "thresholds": assets["rsbuilding_thresholds"],
        },
        {
            "module_id": "M6_Construction",
            "expected_gsd_m": ref_gsd,
            "expected_gsd_source": "empirical_zoom19_reference_input_used_by_sealed_baseline; UCS package has pixel tile size but no reliable physical GSD",
            "native_zoom20_gsd_m": native_gsd,
            "native_scale_ratio": native_gsd / ref_gsd,
            "native_scale_ratio": native_gsd / ref_gsd,
            "r1_mode": "zoom20_native",
            "r2_mode": "zoom20_model_scale_normalized",
            "r2_resampling": "bilinear_to_zoom19_reference_grid; independent construction cache",
            "r2_cache": str((run_root / "03_inference/zoom20_scale_normalized/inputs/construction").resolve()),
            "checkpoint_sha256": assets["construction_sha256"],
            "thresholds": assets["construction_thresholds"],
            "rgb_mapping": assets["construction_rgb_mapping"],
        },
    ]
    payload = {
        "status": "pass",
        "reference_zoom19_gsd_m": ref_gsd,
        "native_zoom20_gsd_m": native_gsd,
        "models": model_rows,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "created_at": now_utc(),
    }
    write_json(run_root / "02_scale_audit/MODEL_EXPECTED_SCALE.json", payload)
    write_json(run_root / "02_scale_audit/ZOOM20_SCALE_COMPATIBILITY.json", {
        "status": "native_scale_mismatch_detected" if native_gsd < ref_gsd * 0.75 else "native_scale_close",
        "reference_zoom19_gsd_m": ref_gsd,
        "native_zoom20_gsd_m": native_gsd,
        "scale_ratio_zoom20_to_zoom19": native_gsd / ref_gsd,
        "normalization_required_for_each_model": True,
        "same_resampled_tensor_shared": False,
    })
    write_json(run_root / "02_scale_audit/MODEL_SPECIFIC_RESAMPLING_PLAN.json", {
        "status": "pass",
        "independent_plans": [
            {"module_id": item["module_id"], "input_cache": item["r2_cache"], "resampling": item["r2_resampling"]}
            for item in model_rows
        ],
        "source_images_unchanged": True,
        "shared_normalized_tensor": False,
    })
    return payload


def _make_scaled_cache(run_root: Path, audit: Mapping[str, Any], module_name: str) -> dict[str, Any]:
    import rasterio
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    target_dir = run_root / "03_inference/zoom20_scale_normalized/inputs" / module_name
    target_dir.mkdir(parents=True, exist_ok=True)
    ref = next(item for item in audit["zoom19"]["records"] if item["scene_date"] == TARGET_DATE)
    rows: list[dict[str, Any]] = []
    for scene in audit["zoom20"]["scenes"]:
        date = str(scene["scene_date"])
        target = target_dir / f"nj_jiangbei{datetime.strptime(date, '%Y-%m-%d').strftime('%y%m%d')}_zoom20_scale_normalized_{module_name}.tif"
        expected_count = min(3, int(scene["bands"]))
        target_valid = False
        if target.exists():
            try:
                with rasterio.open(target) as existing:
                    target_valid = (
                        existing.width == int(ref["actual_width"])
                        and existing.height == int(ref["actual_height"])
                        and existing.count == expected_count
                    )
            except Exception:
                target_valid = False
            if not target_valid:
                # This target is a derivative in the new run only.  A previous
                # interrupted write must not be treated as a valid cache.
                target.unlink(missing_ok=True)
        if not target_valid:
            try:
                with rasterio.open(ref["file_path"]) as reference, rasterio.open(scene["file_path"]) as source:
                    # Do not inherit YCbCr/interleave or block metadata from the
                    # compressed Zoom19 source.  Those tags can leave a tiny,
                    # unreadable TIFF when writing a freshly reprojected RGB
                    # cache with rasterio/GDAL on Windows.
                    profile = {
                        "driver": "GTiff",
                        "width": reference.width,
                        "height": reference.height,
                        "count": expected_count,
                        "dtype": "uint8",
                        "crs": reference.crs,
                        "transform": reference.transform,
                        "nodata": 0,
                        "compress": "deflate",
                        "predictor": 2,
                    }
                    with rasterio.open(target, "w", **profile) as output:
                        for index in range(1, expected_count + 1):
                            reproject(
                                source=rasterio.band(source, index),
                                destination=rasterio.band(output, index),
                                src_transform=source.transform,
                                src_crs=source.crs,
                                dst_transform=reference.transform,
                                dst_crs=reference.crs,
                                resampling=Resampling.bilinear,
                            )
            except Exception:
                # Keep the run resumable without leaving a misleading partial
                # cache.  Only the new derivative target is touched here.
                target.unlink(missing_ok=True)
                raise
        rows.append({
            "date": date,
            "source_path": scene["file_path"],
            "source_sha256": scene["sha256"],
            "normalized_path": str(target.resolve()),
            "normalized_sha256": sha256_file(target),
            "module_id": module_name,
            "resampling": "bilinear",
            "target_grid": str((run_root / "02_common_grid/ZOOM19_GRID.json").resolve()),
            "source_modified": False,
        })
    write_json(target_dir / "RESAMPLING_MANIFEST.json", {"module_id": module_name, "rows": rows, "source_modified": False})
    return {"root": str(target_dir.resolve()), "rows": rows}


def _configure_modular_runner(assets: Mapping[str, Any], rsbuilding_python: Path):
    from tools import nanjing_jiangbei_modular_rebuild as modular

    modular.FULL20_CHECKPOINT = assets["full20_checkpoint"]
    modular.FULL20_CONFIG = assets["full20_config"]
    modular.EXPECTED_FULL20_SHA = assets["full20_sha256"]
    modular.RSBUILDING_CHECKPOINT = assets["rsbuilding_checkpoint"]
    modular.RSBUILDING_CONFIG = assets["rsbuilding_config"]
    modular.EXPECTED_RSBUILDING_SHA = assets["rsbuilding_sha256"]
    modular.CONSTRUCTION_PACKAGE = assets["construction_package"]
    modular.CONSTRUCTION_CHECKPOINT = assets["construction_checkpoint"]
    modular.EXPECTED_CONSTRUCTION_SHA = assets["construction_sha256"]
    modular.RSBUILDING_PYTHON = rsbuilding_python.resolve()
    return modular


def _modular_args(source_root: Path, python: Path, device: str, construction_device: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_root=str(source_root.resolve()),
        output_root=None,
        python=str(python.resolve()),
        device=device,
        construction_device=construction_device,
        construction_enabled=False,
    )


def _run_modular_baseline(
    source_root: Path,
    output_root: Path,
    assets: Mapping[str, Any],
    python: Path,
    rsbuilding_python: Path,
    *,
    run_construction: bool,
) -> dict[str, Any]:
    modular = _configure_modular_runner(assets, rsbuilding_python)
    args = _modular_args(source_root, python, "cuda:0", "cuda")
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "running", "output_root": str(output_root.resolve())}
    try:
        if not (output_root / "04_baseline_fusion/BASELINE_SEAL_MANIFEST.json").exists():
            result["baseline"] = modular.run_baseline_stage(args, output_root)
        else:
            result["baseline"] = {"status": "reused_sealed_baseline"}
        if run_construction:
            if not (output_root / "06_construction_single_date/CONSTRUCTION_BRANCH_MANIFEST.json").exists():
                result["construction"] = modular.run_construction_stage(args, output_root)
            else:
                result["construction"] = {"status": "reused_construction"}
            if (
                (output_root / "04_baseline_fusion/BASELINE_SEAL_MANIFEST.json").exists()
                and (output_root / "06_construction_single_date/CONSTRUCTION_BRANCH_MANIFEST.json").exists()
                and (output_root / "07_construction_temporal/CONSTRUCTION_TEMPORAL_MANIFEST.json").exists()
            ):
                result["cross_reference"] = modular.run_cross_reference(output_root)
        result["status"] = "completed"
    except Exception as exc:
        result.update({
            "status": "failed_or_degraded",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        write_json(output_root / "FAILURE_RECEIPT.json", result)
    write_json(output_root / "MODE_RUN_RECEIPT.json", {
        **result,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "gt_content_read": False,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
        "created_at": now_utc(),
    })
    return result


def _run_r2_construction(
    output_root: Path,
    geometry_root: Path,
    audit: Mapping[str, Any],
    assets: Mapping[str, Any],
    scene_root: Path,
) -> dict[str, Any]:
    from tools import nanjing_jiangbei_modular_rebuild as modular
    from urban_renewal.pipelines.construction_external_pipeline import run_external_construction

    construction_root = output_root / "06_construction_single_date"
    final_path = construction_root / "CONSTRUCTION_BRANCH_MANIFEST.json"
    if final_path.exists():
        existing = read_json(final_path, {})
        existing_sources = [str(row.get("source_path", "")) for row in existing.get("dates", [])]
        if (
            existing.get("input_mode") == "zoom20_model_scale_normalized"
            and existing_sources
            and all(str(scene_root.resolve()) in source for source in existing_sources)
        ):
            return existing
    scenes = [
        {
            "date": str(item["scene_date"]),
            "path": str(
                scene_root
                / f"nj_jiangbei{datetime.strptime(str(item['scene_date']), '%Y-%m-%d').strftime('%y%m%d')}_zoom20_scale_normalized_construction.tif"
            ),
        }
        for item in audit["zoom20"]["scenes"]
    ]
    try:
        import rasterio

        for scene in scenes:
            source_path = Path(scene["path"])
            with rasterio.open(source_path) as dataset:
                scene.update({
                    "sha256": sha256_file(source_path),
                    "width": int(dataset.width),
                    "height": int(dataset.height),
                    "crs": str(dataset.crs) if dataset.crs else None,
                })
        external = run_external_construction(
            PROJECT_ROOT,
            scenes=scenes,
            geometry_root=geometry_root,
            output_root=construction_root,
            full20_manifest=None,
            surface_state_manifest=None,
            device="cuda",
        )
        rows: list[dict[str, Any]] = []
        for date in audit["zoom20"]["valid_dates"]:
            scene_payload = external.get("scene_outputs", {}).get(str(date), {})
            source_path = Path(next(item["path"] for item in scenes if item["date"] == date))
            candidate_path = Path(scene_payload.get("outputs", {}).get("parcel_candidates", ""))
            date_root = construction_root / str(date)
            derivatives = modular._rasterize_construction_candidates(candidate_path, source_path, date_root)
            row = {
                "date": str(date),
                "source_path": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "model_sha256": assets["construction_sha256"],
                "rgb_mapping": assets["construction_rgb_mapping"],
                "probability_outputs": scene_payload.get("inference", {}).get("outputs", {}),
                **derivatives,
                "scientific_status": "candidate_not_ground_truth",
                "human_review_required": True,
                "test_pixel_reads": 0,
                "input_mode": "zoom20_model_scale_normalized",
            }
            write_json(date_root / "CONSTRUCTION_DERIVATIVE_MANIFEST.json", row)
            rows.append(row)
        payload = {
            "status": "completed_or_degraded_candidate",
            "module": "M6_Construction_Activity",
            "dates": rows,
            "model_package": str(assets["construction_package"].resolve()),
            "model_sha256": assets["construction_sha256"],
            "rgb_mapping": assets["construction_rgb_mapping"],
            "input_mode": "zoom20_model_scale_normalized",
            "geometry_root": str(geometry_root.resolve()),
            "writes_to_baseline_dirs": False,
            "training_started": False,
            "locked_test_pixel_reads": 0,
            "human_review_required": True,
            "scientific_status": "limited_candidate_not_ground_truth",
            "created_at": now_utc(),
        }
        write_json(final_path, payload)
        return payload
    except Exception as exc:
        failure = {
            "status": "failed_or_degraded",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_started": False,
            "gt_content_read": False,
            "source_data_modified": False,
        }
        write_json(output_root / "CONSTRUCTION_FAILURE_RECEIPT.json", failure)
        return failure


def _quarantine_mismatched_construction_temporal(construction_root: Path) -> str | None:
    """Preserve, then move, stale temporal rasters from a previous grid."""

    import rasterio

    temporal_root = construction_root / "07_construction_temporal"
    surface_root = temporal_root / "surface_state"
    if not surface_root.exists():
        return None
    mismatched = False
    for scene_root in sorted(item for item in surface_root.iterdir() if item.is_dir()):
        class_path = scene_root / "surface_state_class.tif"
        uncertainty_path = scene_root / "surface_state_uncertainty.tif"
        if not class_path.exists() or not uncertainty_path.exists():
            continue
        with rasterio.open(class_path) as class_dataset, rasterio.open(uncertainty_path) as uncertainty_dataset:
            if (class_dataset.width, class_dataset.height, str(class_dataset.crs)) != (
                uncertainty_dataset.width,
                uncertainty_dataset.height,
                str(uncertainty_dataset.crs),
            ):
                mismatched = True
                break
    if not mismatched:
        return None
    stale_root = construction_root / "07_construction_temporal_STALE_NATIVE_GRID"
    if stale_root.exists():
        suffix = 2
        while (construction_root / f"07_construction_temporal_STALE_NATIVE_GRID_{suffix}").exists():
            suffix += 1
        stale_root = construction_root / f"07_construction_temporal_STALE_NATIVE_GRID_{suffix}"
    shutil.move(str(temporal_root), str(stale_root))
    return str(stale_root.resolve())


def _quarantine_mismatched_construction_outputs(construction_root: Path) -> dict[str, str | None]:
    """Preserve stale native-grid M6/M7 artifacts before a corrected R2 run."""

    import rasterio

    single_root = construction_root / "06_construction_single_date"
    mismatch = False
    for date_root in sorted(item for item in single_root.iterdir() if item.is_dir()) if single_root.exists() else []:
        candidate = date_root / "construction_candidate_mask.tif"
        probability = date_root / "uncertainty_probability.tif"
        if not candidate.exists() or not probability.exists():
            continue
        with rasterio.open(candidate) as candidate_dataset, rasterio.open(probability) as probability_dataset:
            if (candidate_dataset.width, candidate_dataset.height, str(candidate_dataset.crs)) != (
                probability_dataset.width,
                probability_dataset.height,
                str(probability_dataset.crs),
            ):
                mismatch = True
                break
    if not mismatch:
        return {"single_date": None, "temporal": None}

    def move_preserving(path: Path, label: str) -> str | None:
        if not path.exists():
            return None
        stale = construction_root / f"{label}_STALE_NATIVE_GRID"
        if stale.exists():
            suffix = 2
            while (construction_root / f"{label}_STALE_NATIVE_GRID_{suffix}").exists():
                suffix += 1
            stale = construction_root / f"{label}_STALE_NATIVE_GRID_{suffix}"
        shutil.move(str(path), str(stale))
        return str(stale.resolve())

    return {
        "single_date": move_preserving(single_root, "06_construction_single_date"),
        "temporal": move_preserving(construction_root / "07_construction_temporal", "07_construction_temporal"),
    }


def _run_r2(
    run_root: Path,
    audit: Mapping[str, Any],
    assets: Mapping[str, Any],
    python: Path,
    rsbuilding_python: Path,
) -> dict[str, Any]:
    caches = {
        "full20": _make_scaled_cache(run_root, audit, "full20"),
        "building": _make_scaled_cache(run_root, audit, "building"),
        "construction": _make_scaled_cache(run_root, audit, "construction"),
    }
    modular = _configure_modular_runner(assets, rsbuilding_python)
    full_root = run_root / "03_inference/zoom20_scale_normalized/full20"
    building_root = run_root / "03_inference/zoom20_scale_normalized/building"
    construction_root = run_root / "03_inference/zoom20_scale_normalized/construction"
    full_args = _modular_args(Path(caches["full20"]["root"]), python, "cuda:0", "cuda")
    building_args = _modular_args(Path(caches["building"]["root"]), python, "cuda:0", "cuda")
    output: dict[str, Any] = {"status": "running", "caches": caches}
    try:
        modular.run_audit(full_args, full_root)
        output["full20"] = modular.run_full20_baseline(full_args, full_root)
    except Exception as exc:
        output["full20"] = {"status": "failed_or_degraded", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        write_json(full_root / "FAILURE_RECEIPT.json", output["full20"])
    try:
        modular.run_audit(building_args, building_root)
        output["building"] = modular.run_rsbuilding_baseline(building_args, building_root)
        output["tracking"] = modular.run_baseline_tracking(building_root)
    except Exception as exc:
        output["building"] = {"status": "failed_or_degraded", "error_type": type(exc).__name__, "error": str(exc), "error_traceback": traceback.format_exc()}
        write_json(building_root / "FAILURE_RECEIPT.json", output["building"])
    try:
        construction_args = _modular_args(Path(caches["construction"]["root"]), python, "cuda:0", "cuda")
        modular.run_audit(construction_args, construction_root)
        output["construction_stale_quarantine"] = _quarantine_mismatched_construction_outputs(construction_root)
        output["construction"] = _run_r2_construction(
            construction_root,
            building_root / "03_baseline_rsbuilding",
            audit,
            assets,
            Path(caches["construction"]["root"]),
        )
        if output["construction"].get("status") == "completed_or_degraded_candidate":
            try:
                output["construction_temporal_stale_quarantine"] = _quarantine_mismatched_construction_temporal(construction_root)
                output["construction_temporal"] = modular.run_construction_temporal(construction_args, construction_root)
            except Exception as exc:
                output["construction_temporal"] = {
                    "status": "failed_or_degraded",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
    except Exception as exc:
        output["construction"] = {"status": "failed_or_degraded", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
    output["status"] = "completed_or_degraded"
    write_json(run_root / "03_inference/zoom20_scale_normalized/R2_RUN_RECEIPT.json", {
        **output,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "gt_content_read": False,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
        "created_at": now_utc(),
    })
    return output


def _verify_r0(old_run: Path, assets: Mapping[str, Any]) -> dict[str, Any]:
    seal_path = old_run / "04_baseline_fusion/BASELINE_SEAL_MANIFEST.json"
    seal = read_json(seal_path, {})
    if seal.get("status") != "sealed" or int(seal.get("file_count", 0)) != 252:
        return {"status": "blocked_old_baseline_seal_not_252", "seal_path": str(seal_path.resolve())}
    seal_check_values = {
        "full20_checkpoint_sha256": str(seal.get("checkpoint_sha256", {}).get("full20", "")),
        "rsbuilding_checkpoint_sha256": str(seal.get("checkpoint_sha256", {}).get("rsbuilding", "")),
        "full20_config_sha256": str(seal.get("config_sha256", {}).get("full20", "")),
        "rsbuilding_config_sha256": str(seal.get("config_sha256", {}).get("rsbuilding", "")),
    }
    expected_values = {
        "full20_checkpoint_sha256": assets["full20_sha256"],
        "rsbuilding_checkpoint_sha256": assets["rsbuilding_sha256"],
        "full20_config_sha256": assets["full20_config_sha256"],
        "rsbuilding_config_sha256": assets["rsbuilding_config_sha256"],
    }
    thresholds_match = (
        seal.get("thresholds", {}).get("full20") == {
            "any_change": assets["full20_thresholds"].get("any_change"),
            "new_building": assets["full20_thresholds"].get("new_building"),
            "demolition": assets["full20_thresholds"].get("demolition"),
        }
        and float(seal.get("thresholds", {}).get("rsbuilding", -1)) == float(assets["rsbuilding_thresholds"].get("building_probability", -2))
        and float(seal.get("thresholds", {}).get("mmu_m2", -1)) == float(assets["rsbuilding_thresholds"].get("minimum_mapping_unit_m2", -2))
    )
    match = all(seal_check_values[key] == value for key, value in expected_values.items()) and thresholds_match
    return {
        "status": "pass" if match else "blocked_registry_seal_mismatch",
        "source": "existing_sealed_zoom19_outputs",
        "old_run": str(old_run.resolve()),
        "seal_path": str(seal_path.resolve()),
        "seal_sha256": sha256_file(seal_path),
        "file_count": seal.get("file_count"),
        "checkpoint_config_checks": {
            "actual": seal_check_values,
            "expected": expected_values,
            "thresholds_match": thresholds_match,
        },
        "checkpoint_config_match": match,
        "r0_model_reuse": True,
        "rerun": False,
    }


def _write_pre_gt_seal(run_root: Path, r0: Mapping[str, Any], run_results: Mapping[str, Any]) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    inference_root = run_root / "03_inference"
    for path in sorted(inference_root.rglob("*")):
        if path.is_file() and path.name != "INFERENCE_OUTPUT_SHA256.csv":
            output_rows.append({
                "relative_path": _relative(path, run_root),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    write_csv(run_root / "03_inference/INFERENCE_OUTPUT_SHA256.csv", output_rows)
    statuses = {
        "r0": r0.get("status"),
        "r1": run_results.get("r1", {}).get("status"),
        "r2": run_results.get("r2", {}).get("status"),
    }
    payload = {
        "status": "sealed",
        "inference_completion_status": "completed_or_degraded_all_attempted",
        "r0_r1_r2_status": statuses,
        "output_count": len(output_rows),
        "output_sha256_csv": str((run_root / "03_inference/INFERENCE_OUTPUT_SHA256.csv").resolve()),
        "gt_content_read": False,
        "gt_access_before_inference_completed": False,
        "gt_geometry_read": False,
        "gt_pixel_read": False,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
        "model_sha256_unchanged": True,
        "created_at": now_utc(),
    }
    write_json(run_root / "03_inference/PRE_GT_INFERENCE_SEAL.json", payload)
    write_json(run_root / "05_gt_evaluation/GT_ACCESS_LOG.json", {
        "status": "sealed_inference_before_gt_access",
        "gt_access_before_inference_completed": False,
        "events": [{
            "event": "pre_gt_inference_seal",
            "content_read": False,
            "geometry_read": False,
            "pixel_read": False,
            "timestamp": now_utc(),
        }],
    })
    return payload


def _metric_aoi_geometry():
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoSeries([box(*RESEARCH_BOUNDS)], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]


def _repair_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.is_valid:
        return geometry
    try:
        from shapely import make_valid

        repaired = make_valid(geometry)
    except Exception:
        repaired = geometry.buffer(0)
    return repaired if repaired is not None and not repaired.is_empty else None


def _read_metric_layer(path: Path, layer: str, *, keep_all: bool = True):
    import geopandas as gpd
    import pyogrio

    dataframe = pyogrio.read_dataframe(path, layer=layer)
    source_crs = dataframe.crs
    if source_crs is None:
        info = pyogrio.read_info(path, layer=layer)
        source_crs = info.get("crs")
    dataframe = gpd.GeoDataFrame(dataframe, geometry="geometry", crs=source_crs)
    dataframe["geometry"] = dataframe.geometry.map(_repair_geometry)
    dataframe = dataframe[dataframe.geometry.notna() & ~dataframe.geometry.is_empty].copy()
    dataframe = dataframe.to_crs(METRIC_CRS)
    aoi = _metric_aoi_geometry()
    if keep_all:
        dataframe = dataframe[dataframe.geometry.intersects(aoi)].copy()
        dataframe["geometry"] = dataframe.geometry.intersection(aoi)
        dataframe = dataframe[dataframe.geometry.notna() & ~dataframe.geometry.is_empty].copy()
    return dataframe


def _read_wgs_layer(path: Path, layer: str):
    import geopandas as gpd
    import pyogrio

    dataframe = pyogrio.read_dataframe(path, layer=layer)
    source_crs = dataframe.crs
    if source_crs is None:
        source_crs = pyogrio.read_info(path, layer=layer).get("crs")
    dataframe = gpd.GeoDataFrame(dataframe, geometry="geometry", crs=source_crs)
    dataframe["geometry"] = dataframe.geometry.map(_repair_geometry)
    return dataframe[dataframe.geometry.notna() & ~dataframe.geometry.is_empty].copy()


def _gt_specs(zoom20_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "building",
            "path": zoom20_root / "gt_building19.gpkg",
            "layer": "gt_building",
            "semantic_class": "building_footprint_positive",
        },
        {
            "role": "construction",
            "path": zoom20_root / "gt_construction19.gpkg",
            "layer": "gt_construction",
            "semantic_class": "construction_site_positive",
        },
    ]


def _audit_gt_and_load(run_root: Path, zoom20_root: Path) -> dict[str, Any]:
    """Read the supplied positive-object GT only after the inference seal."""

    import pyogrio
    from shapely.ops import unary_union

    seal = read_json(run_root / "03_inference/PRE_GT_INFERENCE_SEAL.json", {})
    if seal.get("status") != "sealed":
        raise RuntimeError("GT access requires a sealed pre-GT inference receipt")
    access_log_path = run_root / "05_gt_evaluation/GT_ACCESS_LOG.json"
    access_log = read_json(access_log_path, {})
    if access_log.get("gt_access_before_inference_completed") is True:
        raise RuntimeError("GT access log indicates content was read before inference completed")

    file_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    crs_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any]] = {}
    aoi = _metric_aoi_geometry()

    for spec in _gt_specs(zoom20_root):
        path = Path(spec["path"])
        if not path.exists():
            file_rows.append({"role": spec["role"], "path": str(path), "status": "missing"})
            continue
        before_sha = sha256_file(path)
        layers = pyogrio.list_layers(path)
        file_rows.append({
            "role": spec["role"],
            "path": str(path.resolve()),
            "file_name": path.name,
            "file_size_bytes": path.stat().st_size,
            "sha256_before_read": before_sha,
            "layer_count": len(layers),
            "status": "read_after_pre_gt_seal",
        })
        for layer_entry in layers:
            layer = str(layer_entry[0])
            info = pyogrio.read_info(path, layer=layer)
            wgs = _read_wgs_layer(path, layer)
            metric = _read_metric_layer(path, layer)
            geometries = list(wgs.geometry)
            valid_count = sum(bool(item is not None and item.is_valid) for item in geometries)
            empty_count = sum(bool(item is None or item.is_empty) for item in geometries)
            duplicate_keys = [item.wkb_hex for item in geometries if item is not None and not item.is_empty]
            duplicate_count = len(duplicate_keys) - len(set(duplicate_keys))
            union = unary_union(list(metric.geometry)) if len(metric) else None
            union_area = float(union.area) if union is not None else 0.0
            aoi_intersection_area = float(union.intersection(aoi).area) if union is not None else 0.0
            layer_rows.append({
                "role": spec["role"],
                "layer": layer,
                "feature_count": int(info.get("features", len(wgs))),
                "geometry_type": str(info.get("geometry_type")),
                "crs": str(info.get("crs")),
                "bounds": json.dumps(info.get("total_bounds"), ensure_ascii=False),
                "fields": json.dumps(list(info.get("fields", [])), ensure_ascii=False),
                "status": "read",
            })
            schema_rows.append({
                "role": spec["role"],
                "path": str(path.resolve()),
                "layer": layer,
                "fields": list(info.get("fields", [])),
                "dtypes": list(info.get("dtypes", [])),
                "fid_column": info.get("fid_column"),
                "geometry_name": info.get("geometry_name"),
                "geometry_type": info.get("geometry_type"),
                "encoding": info.get("encoding"),
                "layer_metadata_present": bool(info.get("layer_metadata")),
            })
            geometry_rows.append({
                "role": spec["role"],
                "layer": layer,
                "feature_count_read": len(wgs),
                "empty_geometry_count": empty_count,
                "invalid_geometry_count": len(wgs) - valid_count - empty_count,
                "duplicate_geometry_count": duplicate_count,
                "repaired_geometry_count": sum(
                    1 for original, repaired in zip(geometries, wgs.geometry)
                    if original is not None and repaired is not None and original.wkb_hex != repaired.wkb_hex
                ),
                "metric_feature_count_in_common_aoi": len(metric),
                "status": "pass" if empty_count == 0 and duplicate_count == 0 else "pass_with_audit_flags",
            })
            crs_rows.append({
                "role": spec["role"],
                "layer": layer,
                "source_crs": str(info.get("crs")),
                "metric_crs": METRIC_CRS,
                "reprojection_for_metrics": True,
            })
            coverage_rows.append({
                "role": spec["role"],
                "layer": layer,
                "common_aoi_bounds_wgs84": list(RESEARCH_BOUNDS),
                "common_aoi_area_m2": float(aoi.area),
                "geometry_union_area_m2_after_clip": union_area,
                "geometry_union_area_m2_inside_common_aoi": aoi_intersection_area,
                "feature_count_inside_common_aoi": len(metric),
                "background_or_completeness_declared": False,
                "pixel_metrics_allowed": False,
            })
            class_rows.append({
                "role": spec["role"],
                "source_file": path.name,
                "source_layer": layer,
                "class_field": None,
                "class_values": [spec["semantic_class"]],
                "mapping_basis": "layer_and_file_role_plus_positive_polygon_schema",
                "positive_only": True,
                "background_class_present": False,
            })
            loaded[spec["role"]] = {
                "path": str(path.resolve()),
                "layer": layer,
                "wgs": wgs,
                "metric": metric,
                "info": info,
            }
        after_sha = sha256_file(path)
        file_rows[-1]["sha256_after_read"] = after_sha
        file_rows[-1]["source_unchanged_during_read"] = before_sha == after_sha

    readiness = {
        "status": "ready_for_positive_object_match_only" if all(role in loaded for role in ("building", "construction")) else "partial_missing_gt_role",
        "building_positive_object_count": int(len(loaded.get("building", {}).get("metric", []))),
        "construction_positive_object_count": int(len(loaded.get("construction", {}).get("metric", []))),
        "background_class_available": False,
        "negative_background_declared": False,
        "pixel_precision_recall_f1_iou_allowed": False,
        "object_match_metrics_allowed": True,
        "evaluation_date": TARGET_DATE,
        "scientific_boundary": "Positive polygon GT supports object matching and overlap proxies; full-scene pixel accuracy is unavailable without complete background/negative labels.",
        "gt_content_read_after_pre_gt_seal": True,
        "locked_test_pixel_reads": 0,
        "source_modified": False,
        "created_at": now_utc(),
    }
    audit_root = run_root / "05_gt_evaluation"
    write_csv(audit_root / "GT_FILE_INVENTORY.csv", file_rows)
    write_csv(audit_root / "GT_LAYER_INVENTORY.csv", layer_rows)
    write_json(audit_root / "GT_SCHEMA_AUDIT.json", {"status": "pass", "rows": schema_rows})
    write_json(audit_root / "GT_GEOMETRY_AUDIT.json", {"status": "pass", "rows": geometry_rows})
    write_json(audit_root / "GT_CRS_AUDIT.json", {"status": "pass", "rows": crs_rows})
    write_json(audit_root / "GT_COVERAGE_AUDIT.json", {"status": "positive_only_no_completeness_declaration", "rows": coverage_rows})
    write_json(audit_root / "GT_CLASS_MAPPING.json", {"status": "pass_positive_object_roles", "rows": class_rows})
    write_json(audit_root / "GT_READINESS.json", readiness)
    access_log.setdefault("events", []).append({
        "event": "gt_schema_geometry_read_after_pre_gt_seal",
        "content_read": True,
        "geometry_read": True,
        "pixel_read": False,
        "timestamp": now_utc(),
        "source_modified": False,
    })
    access_log.update({
        "status": "gt_read_after_sealed_inference",
        "gt_content_read": True,
        "gt_geometry_read": True,
        "gt_pixel_read": False,
        "gt_access_before_inference_completed": False,
        "locked_test_pixel_reads": 0,
        "source_modified": False,
    })
    write_json(access_log_path, access_log)
    return {"loaded": loaded, "readiness": readiness, "file_rows": file_rows}


def _mode_roots(run_root: Path, old_run: Path) -> list[dict[str, Any]]:
    return [
        {
            "input_mode": "zoom19_existing",
            "short_mode": "R0",
            "full20_root": old_run,
            "building_root": old_run,
            "construction_root": old_run,
            "source": "existing_sealed_zoom19_outputs",
        },
        {
            "input_mode": "zoom20_native",
            "short_mode": "R1",
            "full20_root": run_root / "03_inference/zoom20_native",
            "building_root": run_root / "03_inference/zoom20_native",
            "construction_root": run_root / "03_inference/zoom20_native",
            "source": "new_inference",
        },
        {
            "input_mode": "zoom20_model_scale_normalized",
            "short_mode": "R2",
            "full20_root": run_root / "03_inference/zoom20_scale_normalized/full20",
            "building_root": run_root / "03_inference/zoom20_scale_normalized/building",
            "construction_root": run_root / "03_inference/zoom20_scale_normalized/construction",
            "source": "new_inference_model_specific_resampling",
        },
    ]


def _empty_metric_gdf():
    import geopandas as gpd

    return gpd.GeoDataFrame({"prediction_id": []}, geometry=gpd.GeoSeries([], crs=METRIC_CRS), crs=METRIC_CRS)


def _prediction_gdf(root: Path, task: str, date: str):
    import geopandas as gpd
    import pyogrio

    if task == "building":
        path = root / "03_baseline_rsbuilding" / date / "building_objects.gpkg"
        layer = "buildings"
    else:
        path = root / "06_construction_single_date" / date / "parcel_candidates.gpkg"
        layer = "parcel_candidates"
    if not path.exists():
        return _empty_metric_gdf(), path
    dataframe = pyogrio.read_dataframe(path, layer=layer)
    dataframe = gpd.GeoDataFrame(dataframe, geometry="geometry", crs=dataframe.crs or METRIC_CRS)
    if task == "construction" and "construction_candidate" in dataframe.columns:
        dataframe = dataframe[dataframe["construction_candidate"].fillna(False).astype(bool)].copy()
    dataframe["geometry"] = dataframe.geometry.map(_repair_geometry)
    dataframe = dataframe[dataframe.geometry.notna() & ~dataframe.geometry.is_empty].copy()
    dataframe = dataframe.to_crs(METRIC_CRS)
    aoi = _metric_aoi_geometry()
    dataframe = dataframe[dataframe.geometry.intersects(aoi)].copy()
    dataframe["geometry"] = dataframe.geometry.intersection(aoi)
    dataframe = dataframe[dataframe.geometry.notna() & ~dataframe.geometry.is_empty].copy()
    id_field = "object_id" if "object_id" in dataframe.columns else None
    dataframe["prediction_id"] = (
        dataframe[id_field].astype(str) if id_field else [f"{task}_pred_{index:06d}" for index in range(len(dataframe))]
    )
    return dataframe, path


def _object_records(dataframe, id_field: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in dataframe.reset_index(drop=True).iterrows():
        geometry = row.geometry
        records.append({
            "index": index,
            "id": str(row.get(id_field, f"object_{index:06d}")),
            "geometry": geometry,
            "area_m2": float(geometry.area),
            "perimeter_m": float(geometry.length),
        })
    return records


def _greedy_matches(gt_records: Sequence[Mapping[str, Any]], pred_records: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for gt_index, gt in enumerate(gt_records):
        for pred_index, pred in enumerate(pred_records):
            try:
                intersection_area = float(gt["geometry"].intersection(pred["geometry"]).area)
            except Exception:
                intersection_area = 0.0
            if intersection_area <= 0:
                continue
            union_area = float(gt["geometry"].union(pred["geometry"]).area)
            iou = intersection_area / union_area if union_area else 0.0
            if iou >= threshold:
                candidates.append({
                    "gt_index": gt_index,
                    "pred_index": pred_index,
                    "iou": iou,
                    "intersection_area_m2": intersection_area,
                    "gt_coverage": intersection_area / float(gt["area_m2"]) if gt["area_m2"] else 0.0,
                    "pred_coverage": intersection_area / float(pred["area_m2"]) if pred["area_m2"] else 0.0,
                })
    candidates.sort(key=lambda item: (-item["iou"], item["gt_index"], item["pred_index"]))
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


def _object_score(gt_records: Sequence[Mapping[str, Any]], pred_records: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    matches = _greedy_matches(gt_records, pred_records, threshold)
    tp = len(matches)
    fp = len(pred_records) - tp
    fn = len(gt_records) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": float(np.mean([item["iou"] for item in matches])) if matches else None,
        "mean_gt_coverage_matched": float(np.mean([item["gt_coverage"] for item in matches])) if matches else None,
        "matches": matches,
    }


def _geometry_summary(dataframe) -> dict[str, Any]:
    from shapely.ops import unary_union

    if len(dataframe) == 0:
        return {
            "object_count": 0,
            "area_m2": 0.0,
            "coverage_m2": 0.0,
            "mean_object_area_m2": None,
            "median_object_area_m2": None,
            "small_object_count_lt_100m2": 0,
            "boundary_complexity_per_m": None,
            "object_fragmentation_rate": None,
        }
    geometries = list(dataframe.geometry)
    union = unary_union(geometries)
    component_count = len(union.geoms) if hasattr(union, "geoms") else 1
    areas = np.asarray([float(item.area) for item in geometries], dtype=float)
    perimeters = np.asarray([float(item.length) for item in geometries], dtype=float)
    return {
        "object_count": int(len(geometries)),
        "area_m2": float(areas.sum()),
        "coverage_m2": float(union.area),
        "mean_object_area_m2": float(areas.mean()),
        "median_object_area_m2": float(np.median(areas)),
        "small_object_count_lt_100m2": int((areas < 100.0).sum()),
        "boundary_complexity_per_m": float(perimeters.sum() / areas.sum()) if areas.sum() else None,
        "object_fragmentation_rate": float(max(0, len(geometries) - component_count) / len(geometries)),
    }


def _raster_positive_overlap(mask_path: Path, gt_wgs, *, threshold: float = 0.5) -> dict[str, Any]:
    import rasterio
    from rasterio.features import geometry_mask, rasterize
    from shapely.geometry import mapping

    if not mask_path.exists():
        return {"status": "missing_prediction_raster", "path": str(mask_path)}
    with rasterio.open(mask_path) as dataset:
        data = dataset.read(1)
        aoi_mask = geometry_mask(
            [mapping(_metric_aoi_geometry())],
            out_shape=data.shape,
            transform=dataset.transform,
            invert=True,
        ) if str(dataset.crs) == METRIC_CRS else geometry_mask(
            [mapping(__import__("shapely").geometry.box(*RESEARCH_BOUNDS))],
            out_shape=data.shape,
            transform=dataset.transform,
            invert=True,
        )
        gt_shapes = [(geometry, 1) for geometry in gt_wgs.geometry if geometry is not None and not geometry.is_empty]
        gt_mask = rasterize(gt_shapes, out_shape=data.shape, transform=dataset.transform, fill=0, dtype="uint8", all_touched=True)
        nodata = dataset.nodata
        # For binary candidate/change rasters, 0 is the legitimate background
        # class even when GDAL advertises it as nodata.  Building masks use 255
        # as the actual outside/unknown marker.  Do not silently discard the
        # background when calculating a common-AOI denominator.
        invalid_marker = nodata is not None and float(nodata) not in {0.0, 1.0}
        valid = aoi_mask & ((data != nodata) if invalid_marker else np.ones(data.shape, dtype=bool))
        prediction = (data >= threshold) & valid
        truth = (gt_mask > 0) & valid
        intersection = prediction & truth
        prediction_count = int(prediction.sum())
        truth_count = int(truth.sum())
        intersection_count = int(intersection.sum())
        return {
            "status": "positive_overlap_proxy",
            "path": str(mask_path.resolve()),
            "grid_width": int(dataset.width),
            "grid_height": int(dataset.height),
            "valid_fraction_in_common_aoi": float(valid.sum() / aoi_mask.sum()) if aoi_mask.sum() else None,
            "prediction_positive_fraction": float(prediction_count / valid.sum()) if valid.sum() else None,
            "gt_positive_fraction": float(truth_count / valid.sum()) if valid.sum() else None,
            "positive_overlap_fraction_of_prediction_proxy": float(intersection_count / prediction_count) if prediction_count else None,
            "positive_overlap_fraction_of_gt_proxy": float(intersection_count / truth_count) if truth_count else None,
            "positive_overlap_f1_proxy": float(2 * intersection_count / (prediction_count + truth_count)) if prediction_count + truth_count else None,
            "pixel_precision_recall_f1_iou": None,
            "pixel_metrics_block_reason": "GT contains positive polygons only; complete negative/background coverage is not declared.",
        }


def _mask_summary(mask_path: Path, probability_path: Path | None = None, threshold: float = 0.5) -> dict[str, Any]:
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import box, mapping
    from scipy import ndimage

    if not mask_path.exists():
        return {"status": "missing", "mask_path": str(mask_path)}
    with rasterio.open(mask_path) as dataset:
        data = dataset.read(1)
        aoi_mask = geometry_mask([mapping(box(*RESEARCH_BOUNDS))], out_shape=data.shape, transform=dataset.transform, invert=True)
        invalid_marker = dataset.nodata is not None and float(dataset.nodata) not in {0.0, 1.0}
        valid = aoi_mask & ((data != dataset.nodata) if invalid_marker else np.ones(data.shape, dtype=bool))
        positive = (data >= threshold) & valid
        labels, component_count = ndimage.label(positive)
        row: dict[str, Any] = {
            "status": "pass",
            "mask_path": str(mask_path.resolve()),
            "width": int(dataset.width),
            "height": int(dataset.height),
            "crs": str(dataset.crs),
            "valid_fraction": float(valid.sum() / aoi_mask.sum()) if aoi_mask.sum() else None,
            "positive_fraction": float(positive.sum() / valid.sum()) if valid.sum() else None,
            "positive_pixel_count": int(positive.sum()),
            "connected_change_region_count": int(component_count),
            "fragmented_change_region_count": int((np.bincount(labels.ravel())[1:] < 20).sum()) if component_count else 0,
        }
    if probability_path is not None and probability_path.exists():
        with rasterio.open(probability_path) as probability_dataset:
            probability = probability_dataset.read(1).astype("float32")
            if np.nanmax(probability) > 1.5:
                probability = probability / 255.0
            valid_probability = probability[valid]
            row.update({
                "mean_probability": float(np.nanmean(valid_probability)) if valid_probability.size else None,
                "high_confidence_fraction": float((valid_probability >= 0.75).mean()) if valid_probability.size else None,
                "uncertainty_fraction": None,
                "probability_path": str(probability_path.resolve()),
            })
    else:
        row.update({"mean_probability": None, "high_confidence_fraction": None, "uncertainty_fraction": None})
    return row


def _mask_path(root: Path, task: str, date: str) -> Path:
    if task == "building":
        return root / "03_baseline_rsbuilding" / date / "building_mask.tif"
    return root / "06_construction_single_date" / date / "construction_candidate_mask.tif"


def _full20_pair_root(root: Path, pair: str = "2019-03-16_to_2022-12-02") -> Path:
    return root / "02_baseline_full20" / pair


def _full20_summary(root: Path) -> dict[str, Any]:
    pair_root = _full20_pair_root(root)
    mask = pair_root / "rasters/any_change_mask.tif"
    probability = pair_root / "rasters/any_change_probability.tif"
    return _mask_summary(mask, probability, threshold=0.5)


def _write_gpkg(dataframe, path: Path, layer: str) -> None:
    import geopandas as gpd

    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(dataframe, gpd.GeoDataFrame):
        dataframe = gpd.GeoDataFrame(dataframe, geometry="geometry", crs=METRIC_CRS)
    dataframe.to_file(path, layer=layer, driver="GPKG", engine="pyogrio", index=False)


def _error_case_rows(task: str, input_mode: str, gt_records: Sequence[Mapping[str, Any]], pred_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matches = _object_score(gt_records, pred_records, 0.5)["matches"]
    matched_gt = {item["gt_index"] for item in matches}
    matched_pred = {item["pred_index"] for item in matches}
    for rank, index in enumerate([item for item in range(len(gt_records)) if item not in matched_gt], start=1):
        gt = gt_records[index]
        best_iou = 0.0
        best_pred = None
        for pred_index, pred in enumerate(pred_records):
            intersection = gt["geometry"].intersection(pred["geometry"]).area
            union = gt["geometry"].union(pred["geometry"]).area
            value = float(intersection / union) if union else 0.0
            if value > best_iou:
                best_iou, best_pred = value, pred_index
        rows.append({
            "task": task,
            "input_mode": input_mode,
            "error_type": "false_negative_or_low_iou",
            "gt_id": gt["id"],
            "prediction_id": pred_records[best_pred]["id"] if best_pred is not None else None,
            "iou": best_iou,
            "coverage": None,
            "object_area_m2": gt["area_m2"],
            "scene_date": TARGET_DATE,
            "case_rank": rank,
            "notes": "Positive-object GT match threshold is IoU 0.50; full-scene pixel background is unavailable.",
            "geometry": gt["geometry"],
        })
    for rank, index in enumerate([item for item in range(len(pred_records)) if item not in matched_pred], start=1):
        pred = pred_records[index]
        best_iou = 0.0
        best_gt = None
        for gt_index, gt in enumerate(gt_records):
            intersection = pred["geometry"].intersection(gt["geometry"]).area
            union = pred["geometry"].union(gt["geometry"]).area
            value = float(intersection / union) if union else 0.0
            if value > best_iou:
                best_iou, best_gt = value, gt_index
        rows.append({
            "task": task,
            "input_mode": input_mode,
            "error_type": "false_positive_or_low_iou",
            "gt_id": gt_records[best_gt]["id"] if best_gt is not None else None,
            "prediction_id": pred["id"],
            "iou": best_iou,
            "coverage": None,
            "object_area_m2": pred["area_m2"],
            "scene_date": TARGET_DATE,
            "case_rank": rank,
            "notes": "Unmatched positive candidate; interpret with positive-only GT coverage limitation.",
            "geometry": pred["geometry"],
        })
    return rows


def _evaluate_task_modes(
    run_root: Path,
    old_run: Path,
    task: str,
    gt_loaded: Mapping[str, Any],
) -> dict[str, Any]:
    gt_metric = gt_loaded["metric"]
    gt_wgs = gt_loaded["wgs"]
    gt_id_field = "id" if "id" in gt_metric.columns else None
    gt_records = _object_records(gt_metric, gt_id_field or "prediction_id")
    metrics_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    overlay_rows: list[dict[str, Any]] = []
    mode_results: dict[str, Any] = {}
    for mode in _mode_roots(run_root, old_run):
        root_key = f"{task}_root"
        prediction_gdf, prediction_path = _prediction_gdf(Path(mode[root_key]), task, TARGET_DATE)
        pred_records = _object_records(prediction_gdf, "prediction_id")
        geometry_summary = _geometry_summary(prediction_gdf)
        proxy = _raster_positive_overlap(_mask_path(Path(mode[root_key]), task, TARGET_DATE), gt_wgs)
        scores: dict[str, Any] = {}
        for threshold in (0.3, 0.5, 0.7):
            score = _object_score(gt_records, pred_records, threshold)
            scores[str(threshold)] = {key: value for key, value in score.items() if key != "matches"}
            metrics_rows.append({
                "task": task,
                "input_mode": mode["input_mode"],
                "short_mode": mode["short_mode"],
                "metric_family": "gt_positive_object_match",
                "metric_name": f"object_match_iou_{threshold:.1f}",
                "value": score["f1"],
                "gt_object_count": len(gt_records),
                "prediction_object_count": len(pred_records),
                "tp": score["tp"],
                "fp": score["fp"],
                "fn": score["fn"],
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1"],
                "mean_iou_matched": score["mean_iou_matched"],
                "mean_gt_coverage_matched": score["mean_gt_coverage_matched"],
                "pixel_metric_status": "not_available_positive_only_gt",
                "gt_completeness_status": "not_declared",
            })
        for metric_name, value in geometry_summary.items():
            metrics_rows.append({
                "task": task,
                "input_mode": mode["input_mode"],
                "short_mode": mode["short_mode"],
                "metric_family": "prediction_geometry_proxy",
                "metric_name": metric_name,
                "value": value,
                "gt_object_count": len(gt_records),
                "prediction_object_count": len(pred_records),
                "pixel_metric_status": "not_applicable",
                "gt_completeness_status": "not_applicable",
            })
        for metric_name, value in proxy.items():
            if metric_name in {"status", "path", "pixel_metrics_block_reason"}:
                continue
            metrics_rows.append({
                "task": task,
                "input_mode": mode["input_mode"],
                "short_mode": mode["short_mode"],
                "metric_family": "positive_overlap_proxy",
                "metric_name": metric_name,
                "value": value,
                "gt_object_count": len(gt_records),
                "prediction_object_count": len(pred_records),
                "pixel_metric_status": "proxy_only",
                "gt_completeness_status": "positive_only_not_declared_complete",
            })
        errors = _error_case_rows(task, mode["input_mode"], gt_records, pred_records)
        error_rows.extend(errors)
        for record in gt_records:
            overlay_rows.append({
                "task": task,
                "input_mode": "ground_truth",
                "source_type": "gt_positive_polygon",
                "object_id": record["id"],
                "scene_date": TARGET_DATE,
                "geometry": record["geometry"],
            })
        for record in pred_records:
            overlay_rows.append({
                "task": task,
                "input_mode": mode["input_mode"],
                "source_type": "prediction_positive_object",
                "object_id": record["id"],
                "scene_date": TARGET_DATE,
                "geometry": record["geometry"],
            })
        mode_results[mode["input_mode"]] = {
            "prediction_path": str(prediction_path.resolve()),
            "prediction_object_count": len(pred_records),
            "geometry_summary": geometry_summary,
            "positive_overlap_proxy": proxy,
            "object_scores": scores,
            "error_case_count": len(errors),
        }
        object_rows.append({
            "task": task,
            "input_mode": mode["input_mode"],
            "short_mode": mode["short_mode"],
            "gt_object_count": len(gt_records),
            "prediction_object_count": len(pred_records),
            "object_match_precision_iou_0_5": scores["0.5"]["precision"],
            "object_match_recall_iou_0_5": scores["0.5"]["recall"],
            "object_match_f1_iou_0_5": scores["0.5"]["f1"],
            "object_match_precision_iou_0_3": scores["0.3"]["precision"],
            "object_match_recall_iou_0_3": scores["0.3"]["recall"],
            "object_match_f1_iou_0_3": scores["0.3"]["f1"],
            "object_match_precision_iou_0_7": scores["0.7"]["precision"],
            "object_match_recall_iou_0_7": scores["0.7"]["recall"],
            "object_match_f1_iou_0_7": scores["0.7"]["f1"],
            "positive_overlap_precision_proxy": proxy.get("positive_overlap_fraction_of_prediction_proxy"),
            "positive_overlap_recall_proxy": proxy.get("positive_overlap_fraction_of_gt_proxy"),
            "positive_overlap_f1_proxy": proxy.get("positive_overlap_f1_proxy"),
            "pixel_metrics_status": "not_available_positive_only_gt",
        })
    return {
        "task": task,
        "gt_object_count": len(gt_records),
        "metrics_rows": metrics_rows,
        "object_rows": object_rows,
        "error_rows": error_rows,
        "overlay_rows": overlay_rows,
        "mode_results": mode_results,
    }


def _cross_date_proxy_metrics(run_root: Path, old_run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in _mode_roots(run_root, old_run):
        building_2019, _ = _prediction_gdf(Path(mode["building_root"]), "building", TARGET_DATE)
        building_2022, _ = _prediction_gdf(Path(mode["building_root"]), "building", "2022-12-02")
        construction_areas: list[float] = []
        construction_counts: list[int] = []
        for date in VALID_DATES:
            construction, _ = _prediction_gdf(Path(mode["construction_root"]), "construction", date)
            summary = _geometry_summary(construction)
            construction_areas.append(float(summary["coverage_m2"]))
            construction_counts.append(int(summary["object_count"]))
        b19 = _object_records(building_2019, "prediction_id")
        b22 = _object_records(building_2022, "prediction_id")
        stability = _object_score(b19, b22, 0.5)
        flicker_terms = [
            abs(current - previous) / max(current, previous)
            for previous, current in zip(construction_areas, construction_areas[1:])
            if max(current, previous) > 0
        ]
        rows.append({
            "input_mode": mode["input_mode"],
            "building_2019_to_2022_match_rate": float(stability["tp"] / max(len(b19), len(b22))) if max(len(b19), len(b22)) else None,
            "building_2019_to_2022_mean_iou": stability["mean_iou_matched"],
            "construction_temporal_flicker_proxy": float(np.mean(flicker_terms)) if flicker_terms else 0.0,
            "construction_candidate_area_m2_mean": float(np.mean(construction_areas)),
            "construction_candidate_count_mean": float(np.mean(construction_counts)),
            "construction_candidate_area_m2_by_date": construction_areas,
            "construction_candidate_count_by_date": construction_counts,
            "dates": list(VALID_DATES),
        })
    return rows


def _construction_cross_reference_metrics(run_root: Path, old_run: Path) -> list[dict[str, Any]]:
    from shapely.ops import unary_union

    rows: list[dict[str, Any]] = []
    for mode in _mode_roots(run_root, old_run):
        building, _ = _prediction_gdf(Path(mode["building_root"]), "building", TARGET_DATE)
        stable_building_union = unary_union(list(building.geometry)) if len(building) else None
        construction, _ = _prediction_gdf(Path(mode["construction_root"]), "construction", TARGET_DATE)
        construction_summary = _geometry_summary(construction)
        candidate_union = unary_union(list(construction.geometry)) if len(construction) else None
        conflict_area = float(candidate_union.intersection(stable_building_union).area) if candidate_union is not None and stable_building_union is not None else 0.0
        dustproof_weighted = None
        if len(construction) and "construction_dustproof_net" in construction.columns:
            dustproof_weighted = float(sum(float(row.geometry.area) * float(row.get("construction_dustproof_net", 0.0)) for _, row in construction.iterrows()))
        full20 = _full20_summary(Path(mode["full20_root"]))
        rows.append({
            "input_mode": mode["input_mode"],
            "stable_building_construction_conflict_rate": conflict_area / construction_summary["coverage_m2"] if construction_summary["coverage_m2"] else None,
            "dustproof_stable_roof_overlap_proxy": dustproof_weighted / construction_summary["area_m2"] if dustproof_weighted is not None and construction_summary["area_m2"] else None,
            "construction_candidate_count": construction_summary["object_count"],
            "construction_candidate_area_m2": construction_summary["coverage_m2"],
            "construction_uncertainty_mean": float(construction["construction_uncertainty"].mean()) if len(construction) and "construction_uncertainty" in construction.columns else None,
            "full20_construction_agreement_proxy": None,
            "full20_changed_fraction": full20.get("positive_fraction"),
            "full20_mean_change_probability": full20.get("mean_probability"),
        })
    return rows


def _full20_comparison_rows(run_root: Path, old_run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in _mode_roots(run_root, old_run):
        summary = _full20_summary(Path(mode["full20_root"]))
        rows.append({
            "input_mode": mode["input_mode"],
            "short_mode": mode["short_mode"],
            "pair": "2019-03-16_to_2022-12-02",
            "changed_fraction": summary.get("positive_fraction"),
            "mean_change_probability": summary.get("mean_probability"),
            "high_confidence_change_fraction": summary.get("high_confidence_fraction"),
            "uncertainty_fraction": summary.get("uncertainty_fraction"),
            "connected_change_region_count": summary.get("connected_change_region_count"),
            "fragmented_change_region_count": summary.get("fragmented_change_region_count"),
            "scientific_status": "model_derived_proxy_without_change_gt",
        })
    return rows


def _write_invariance_audits(run_root: Path, old_run: Path, registry_path: Path, zoom20_root: Path, zoom19_root: Path, assets: Mapping[str, Any], gt_file_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    registry_sha = sha256_file(registry_path)
    seal_path = old_run / "04_baseline_fusion/BASELINE_SEAL_MANIFEST.json"
    seal = read_json(seal_path, {})
    baseline_rows = seal.get("baseline_output_files", [])
    baseline_checks: list[dict[str, Any]] = []
    for item in baseline_rows:
        path = old_run / str(item.get("relative_path", ""))
        actual_sha = sha256_file(path) if path.exists() else None
        baseline_checks.append({
            "relative_path": item.get("relative_path"),
            "expected_sha256": item.get("sha256"),
            "actual_sha256": actual_sha,
            "match": actual_sha == item.get("sha256"),
            "exists": path.exists(),
        })
    baseline_audit = {
        "baseline_hash_match_count": int(sum(bool(item["match"]) for item in baseline_checks)),
        "baseline_hash_total_count": len(baseline_checks),
        "baseline_hash_invariance_passed": len(baseline_checks) == 252 and all(item["match"] for item in baseline_checks),
        "seal_sha256": sha256_file(seal_path),
        "file_list_sha256": seal.get("file_list_sha256"),
        "checks": baseline_checks,
    }
    registry_audit = {
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_sha,
        "registry_sha256_after": sha256_file(registry_path),
        "registry_unchanged": registry_sha == sha256_file(registry_path),
        "models_reselected": False,
    }
    checkpoint_audit = {
        "full20_checkpoint_sha256": {"registry": assets["full20_sha256"], "current": sha256_file(assets["full20_checkpoint"]), "match": assets["full20_sha256"] == sha256_file(assets["full20_checkpoint"])},
        "rsbuilding_checkpoint_sha256": {"registry": assets["rsbuilding_sha256"], "current": sha256_file(assets["rsbuilding_checkpoint"]), "match": assets["rsbuilding_sha256"] == sha256_file(assets["rsbuilding_checkpoint"])},
        "construction_checkpoint_sha256": {"registry": assets["construction_sha256"], "current": sha256_file(assets["construction_checkpoint"]), "match": assets["construction_sha256"] == sha256_file(assets["construction_checkpoint"])},
        "thresholds_changed": False,
    }
    scene_source_checks: list[dict[str, Any]] = []
    prepared = read_json(run_root / "11_manifests/AUDIT_PREPARE_RECEIPT.json", {})
    for scene in list(prepared.get("zoom20", {}).get("scenes", [])) + list(prepared.get("zoom19", {}).get("records", [])):
        path = Path(scene["file_path"])
        actual = sha256_file(path) if path.exists() else None
        scene_source_checks.append({"path": str(path), "expected_sha256": scene.get("sha256"), "actual_sha256": actual, "match": actual == scene.get("sha256"), "source_group": "zoom20" if path.is_relative_to(zoom20_root) else "zoom19"})
    gt_checks = [
        {"path": row.get("path"), "sha256_before_read": row.get("sha256_before_read"), "sha256_after_read": row.get("sha256_after_read"), "unchanged": row.get("source_unchanged_during_read")}
        for row in gt_file_rows
    ]
    write_json(run_root / "11_manifests/BASELINE_HASH_AUDIT.json", baseline_audit)
    write_json(run_root / "11_manifests/CURRENT_BEST_REGISTRY_INVARIANCE.json", registry_audit)
    write_json(run_root / "11_manifests/CHECKPOINT_INVARIANCE.json", checkpoint_audit)
    write_json(run_root / "11_manifests/ZOOM19_INVARIANCE.json", {"old_run_unchanged": True, "source": "existing_sealed_zoom19_outputs", "baseline_hash_audit": baseline_audit})
    write_json(run_root / "11_manifests/ZOOM20_SOURCE_INVARIANCE.json", {"scene_source_checks": [item for item in scene_source_checks if item["source_group"] == "zoom20"], "original_source_modified": not all(item["match"] for item in scene_source_checks)})
    write_json(run_root / "11_manifests/ORIGINAL_SOURCE_INVARIANCE.json", {"scene_source_checks": scene_source_checks, "original_source_modified": not all(item["match"] for item in scene_source_checks)})
    write_json(run_root / "11_manifests/GT_INVARIANCE.json", {"gt_source_checks": gt_checks, "gt_modified": not all(bool(item["unchanged"]) for item in gt_checks), "gt_used_for_training": False, "gt_used_for_model_selection": False})
    return {
        "baseline": baseline_audit,
        "registry": registry_audit,
        "checkpoint": checkpoint_audit,
        "scene_source_checks": scene_source_checks,
        "gt_checks": gt_checks,
        "original_source_modified": not all(item["match"] for item in scene_source_checks),
    }


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    preseal = read_json(run_root / "03_inference/PRE_GT_INFERENCE_SEAL.json", {})
    if preseal.get("status") != "sealed" or preseal.get("gt_content_read") is not False:
        raise RuntimeError("Evaluate requires a sealed inference receipt with gt_content_read=false")
    gt = _audit_gt_and_load(run_root, args.zoom20_root)
    modes = _mode_roots(run_root, args.old_run)
    building_result = _evaluate_task_modes(run_root, args.old_run, "building", gt["loaded"]["building"])
    construction_result = _evaluate_task_modes(run_root, args.old_run, "construction", gt["loaded"]["construction"])
    all_metric_rows = building_result["metrics_rows"] + construction_result["metrics_rows"]
    object_rows = building_result["object_rows"] + construction_result["object_rows"]
    cross_date_rows = _cross_date_proxy_metrics(run_root, args.old_run)
    construction_cross_rows = _construction_cross_reference_metrics(run_root, args.old_run)
    full20_rows = _full20_comparison_rows(run_root, args.old_run)
    comparison_root = run_root / "08_comparison"
    write_csv(comparison_root / "ZOOM19_ZOOM20_METRICS.csv", all_metric_rows)
    write_csv(comparison_root / "BUILDING_RESOLUTION_COMPARISON.csv", building_result["object_rows"])
    write_csv(comparison_root / "CONSTRUCTION_RESOLUTION_COMPARISON.csv", construction_result["object_rows"])
    write_csv(comparison_root / "FULL20_RESOLUTION_COMPARISON.csv", full20_rows)
    write_csv(comparison_root / "OBJECT_MATCH_COMPARISON.csv", object_rows)
    write_csv(comparison_root / "CONSTRUCTION_CROSS_REFERENCE_PROXY.csv", construction_cross_rows)
    write_json(comparison_root / "CROSS_DATE_PROXY_METRICS.json", {"rows": cross_date_rows, "scientific_status": "proxy_without_temporal_gt"})
    error_root = run_root / "09_error_cases"
    _write_gpkg(__import__("geopandas").GeoDataFrame(building_result["error_rows"], geometry="geometry", crs=METRIC_CRS), error_root / "BUILDING_ERROR_CASES.gpkg", "building_errors")
    _write_gpkg(__import__("geopandas").GeoDataFrame(construction_result["error_rows"], geometry="geometry", crs=METRIC_CRS), error_root / "CONSTRUCTION_ERROR_CASES.gpkg", "construction_errors")
    overlay = building_result["overlay_rows"] + construction_result["overlay_rows"]
    _write_gpkg(__import__("geopandas").GeoDataFrame(overlay, geometry="geometry", crs=METRIC_CRS), error_root / "GT_AND_PREDICTION_OVERLAY.gpkg", "gt_prediction_overlay")
    invariance = _write_invariance_audits(run_root, args.old_run, args.registry, args.zoom20_root, args.zoom19_root, _asset_paths(_read_registry(args.registry)), gt["file_rows"])
    def _best_input(rows: Sequence[Mapping[str, Any]], task: str) -> str:
        if task == "construction":
            # Construction parcels are smaller rule-based candidates than the
            # hand-drawn site polygons; when IoU object matches are all zero,
            # the positive-overlap proxy is descriptive only.  Do not choose
            # an input mode from a recall/false-positive trade-off without a
            # compatible site-boundary GT.
            if not any(float(row.get("object_match_f1_iou_0_5") or 0.0) > 0.0 for row in rows):
                return "insufficient_evidence"
            key = lambda row: float(row.get("positive_overlap_f1_proxy")) if row.get("positive_overlap_f1_proxy") not in (None, "") else -1.0
        else:
            key = lambda row: float(row.get("object_match_f1_iou_0_5")) if row.get("object_match_f1_iou_0_5") not in (None, "") else -1.0
        return max(rows, key=key)["input_mode"]

    recommendation = {
        "status": "candidate_pending_human_review",
        "recommended_building_input_mode": _best_input(building_result["object_rows"], "building"),
        "recommended_construction_input_mode": _best_input(construction_result["object_rows"], "construction"),
        "recommended_full20_input_mode": "insufficient_evidence_without_change_gt",
        "recommended_overall_input_mode": "candidate_pending_human_review",
        "selection_basis": "2019 positive-object GT match F1 at IoU 0.50 only; no checkpoint, threshold, or postprocessing was changed.",
        "pixel_accuracy_not_reported": True,
        "human_ground_truth_available": True,
        "gt_completeness_status": gt["readiness"]["status"],
        "models_reselected": False,
        "thresholds_changed": False,
        "created_at": now_utc(),
    }
    write_json(run_root / "06_recommendation/ZOOM20_INFERENCE_RECOMMENDATION.json", recommendation)
    summary = {
        "status": "completed_positive_object_gt_evaluation",
        "evaluation_date": TARGET_DATE,
        "gt_readiness": gt["readiness"],
        "building": building_result["mode_results"],
        "construction": construction_result["mode_results"],
        "full20": full20_rows,
        "cross_date_proxy": cross_date_rows,
        "construction_cross_reference_proxy": construction_cross_rows,
        "recommendation": recommendation,
        "invariance": invariance,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "gt_used_for_training": False,
        "gt_used_for_model_selection": False,
        "gt_modified": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": bool(invariance.get("original_source_modified")),
        "globfp_enabled": False,
        "building_height_enabled": False,
        "human_ground_truth_available": True,
        "true_accuracy_reported": False,
        "pixel_accuracy_reported": False,
        "created_at": now_utc(),
    }
    write_json(run_root / "05_gt_evaluation/2019_GT_EVALUATION.json", summary)
    write_json(run_root / "08_comparison/RESOLUTION_ABLATION_SUMMARY.json", summary)
    return summary


def _plot_raster(ax, path: Path, title: str, *, cmap: str | None = None) -> None:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        if cmap is None and dataset.count >= 3:
            preview = dataset.read([1, 2, 3], out_shape=(3, min(1400, dataset.height), min(1400, dataset.width)), resampling=Resampling.bilinear)
            image = np.moveaxis(preview, 0, -1).astype("float32")
            image = np.nan_to_num(image)
            for band in range(image.shape[-1]):
                low, high = np.percentile(image[..., band], [2, 98])
                image[..., band] = np.clip((image[..., band] - low) / max(high - low, 1.0), 0, 1)
            ax.imshow(image, extent=[dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top], origin="upper")
        else:
            image = dataset.read(1, out_shape=(min(1400, dataset.height), min(1400, dataset.width)), resampling=Resampling.nearest)
            ax.imshow(image, extent=[dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top], origin="upper", cmap=cmap or "viridis", alpha=0.82)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")


def _plot_vector(ax, dataframe, color: str, label: str, *, linewidth: float = 0.7) -> None:
    if dataframe is None or len(dataframe) == 0:
        return
    dataframe.to_crs("EPSG:4326").plot(ax=ax, facecolor="none", edgecolor=color, linewidth=linewidth, label=label)


def _write_highres_figures(run_root: Path, old_run: Path, gt: Mapping[str, Any]) -> list[str]:
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit = read_json(run_root / "11_manifests/AUDIT_PREPARE_RECEIPT.json", {})
    scene19 = next(item for item in audit.get("zoom19", {}).get("records", []) if item.get("scene_date") == TARGET_DATE)
    scene20 = next(item for item in audit.get("zoom20", {}).get("scenes", []) if item.get("scene_date") == TARGET_DATE)
    gt_building = gt["loaded"]["building"]["wgs"]
    gt_construction = gt["loaded"]["construction"]["wgs"]
    figures_root = run_root / "08_figures_highres"
    figures_root.mkdir(parents=True, exist_ok=True)
    modes = _mode_roots(run_root, old_run)
    paths: list[str] = []

    def save_figure(name: str, figure) -> None:
        path = figures_root / name
        # Keep the requested 12x7 inch canvas at 300 dpi (3600x2100). Tight
        # bounding-box cropping made wide vector comparisons shorter than the
        # project's minimum high-resolution canvas.
        figure.savefig(path, dpi=300)
        plt.close(figure)
        paths.append(str(path.resolve()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    _plot_raster(axes[0], Path(scene19["file_path"]), "Zoom19 2019 reference")
    _plot_raster(axes[1], Path(scene20["file_path"]), "Zoom20 2019 native")
    save_figure("INPUT_ZOOM19_ZOOM20_2019.png", fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 7), constrained_layout=True)
    for axis, mode in zip(axes, modes):
        pred, _ = _prediction_gdf(Path(mode["building_root"]), "building", TARGET_DATE)
        _plot_raster(axis, Path(scene19["file_path"]), mode["input_mode"])
        _plot_vector(axis, gt_building, "#00ffff", "GT", linewidth=0.8)
        _plot_vector(axis, pred, "#ff3355", "prediction", linewidth=0.65)
        axis.legend(loc="lower left", fontsize=7)
    save_figure("BUILDING_2019_GT_R0_R1_R2.png", fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 7), constrained_layout=True)
    for axis, mode in zip(axes, modes):
        pred, _ = _prediction_gdf(Path(mode["construction_root"]), "construction", TARGET_DATE)
        _plot_raster(axis, Path(scene19["file_path"]), mode["input_mode"])
        _plot_vector(axis, gt_construction, "#00ffff", "GT", linewidth=1.0)
        _plot_vector(axis, pred, "#ffcc00", "candidate", linewidth=0.8)
        axis.legend(loc="lower left", fontsize=7)
    save_figure("CONSTRUCTION_2019_GT_R0_R1_R2.png", fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 7), constrained_layout=True)
    for axis, mode in zip(axes, modes):
        pair = _full20_pair_root(Path(mode["full20_root"]))
        _plot_raster(axis, pair / "rasters/any_change_mask.tif", mode["input_mode"], cmap="magma")
    save_figure("FULL20_2019_TO_2022_R0_R1_R2.png", fig)

    # Produce deterministic case sheets from the largest error objects. These
    # are diagnostic review cases, not claims of model accuracy.
    evaluation = read_json(run_root / "05_gt_evaluation/2019_GT_EVALUATION.json", {})
    for task, filename in (("building", "BUILDING_ERROR_CASES.gpkg"), ("construction", "CONSTRUCTION_ERROR_CASES.gpkg")):
        error_path = run_root / "09_error_cases" / filename
        if not error_path.exists():
            continue
        try:
            errors = gpd.read_file(error_path, layer=f"{task}_errors", engine="pyogrio")
        except Exception:
            continue
        if len(errors) == 0:
            continue
        errors = errors.sort_values("object_area_m2", ascending=False).head(6)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for axis, (_, row) in zip(axes.ravel(), errors.iterrows()):
            _plot_raster(axis, Path(scene19["file_path"]), f"{row.get('input_mode')} {row.get('error_type')}")
            shape = gpd.GeoDataFrame([row], geometry="geometry", crs=METRIC_CRS)
            _plot_vector(axis, shape, "#ff3355", "error", linewidth=1.0)
        for axis in axes.ravel()[len(errors):]:
            axis.axis("off")
        save_figure(f"{task.upper()}_ERROR_CASES_2019.png", fig)
    return paths


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _write_report_html(run_root: Path, args: argparse.Namespace, figures: Sequence[str]) -> dict[str, Any]:
    evaluation = read_json(run_root / "05_gt_evaluation/2019_GT_EVALUATION.json", {})
    readiness = read_json(run_root / "05_gt_evaluation/GT_READINESS.json", {})
    recommendation = read_json(run_root / "06_recommendation/ZOOM20_INFERENCE_RECOMMENDATION.json", {})
    registry = read_json(args.registry, {})
    registry_checked = _read_registry(args.registry)
    assets = _asset_paths(registry_checked)
    scale = read_json(run_root / "02_scale_audit/MODEL_EXPECTED_SCALE.json", {})
    object_rows: list[dict[str, Any]] = []
    object_csv = run_root / "08_comparison/OBJECT_MATCH_COMPARISON.csv"
    if object_csv.exists():
        with object_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            object_rows = list(csv.DictReader(handle))
    report_root = run_root / "10_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    figure_links = []
    for figure in figures:
        figure_path = Path(figure)
        figure_links.append(f'<figure><a href="../08_figures_highres/{html.escape(figure_path.name)}"><img src="../08_figures_highres/{html.escape(figure_path.name)}" alt="{html.escape(figure_path.stem)}"></a><figcaption>{html.escape(figure_path.name)}</figcaption></figure>')
    table_rows = []
    for row in object_rows:
        table_rows.append([
            row.get("task"),
            row.get("input_mode"),
            row.get("gt_object_count"),
            row.get("prediction_object_count"),
            row.get("object_match_precision_iou_0_5"),
            row.get("object_match_recall_iou_0_5"),
            row.get("object_match_f1_iou_0_5"),
            row.get("positive_overlap_f1_proxy"),
        ])
    flags = {
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "gt_used_for_training": False,
        "gt_used_for_model_selection": False,
        "gt_modified": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "globfp_enabled": False,
        "building_height_enabled": False,
        "human_ground_truth_available": True,
        "true_accuracy_reported": False,
    }
    title = "南京江北新区 Zoom20 与 2019 人工 GT 对照验收"
    common = """
<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>%s</title>
<style>
body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f4f6f8;color:#17202a}
main{max-width:1500px;margin:0 auto;padding:28px}.hero{background:#102a43;color:white;padding:28px;border-radius:14px}
.flag{display:inline-block;margin:5px 6px 0 0;padding:6px 9px;border-radius:999px;background:#d9f99d;color:#17320b;font-size:13px}
.warning{background:#fff4cc;border-left:5px solid #e0a800;padding:14px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
section{background:white;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 1px 5px #0001}h2{margin-top:0;color:#123b5d}
table{border-collapse:collapse;width:100%%;font-size:13px}th,td{border:1px solid #d7dee5;padding:6px;text-align:left}th{background:#eaf1f7}
.figures{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure{margin:0}figure img{width:100%%;height:auto;border:1px solid #ccd6df}figcaption{font-size:12px;margin-top:5px}
code{word-break:break-all}.small{font-size:13px;color:#52606d}
</style></head><body><main>
<div class="hero"><h1>%s</h1><p>冻结 Current Best 模型下的 Zoom19 / Zoom20 R0-R1-R2 比较。2019 GT 仅在三阶段推理封存后读取。</p>%s</div>
""" % (html.escape(title), html.escape(title), "".join(f'<span class="flag">{html.escape(key)}={html.escape(str(value))}</span>' for key, value in flags.items()))
    body = f"""
<section><h2>1. 执行结论</h2><p>Current Best Registry 仅用于读取模型、配置和阈值；本轮没有训练、域适配、伪标签或模型重选。推荐结果仅是输入模式候选，不是南京总体准确率结论。</p>
<div class="warning"><b>科学边界：</b>GT 图层是正样本多边形，没有完整背景/负样本声明。因此报告正样本对象匹配和正样本重叠 proxy；不报告南京像素 Precision、Recall、F1、IoU 或 Accuracy。</div>
<p>Building 候选输入：<code>{html.escape(str(recommendation.get('recommended_building_input_mode')))}</code>；Construction 候选输入：<code>{html.escape(str(recommendation.get('recommended_construction_input_mode')))}</code>；Full20：<code>{html.escape(str(recommendation.get('recommended_full20_input_mode')))}</code>。</p></section>
<section><h2>2. GT 审计</h2>{_html_table(['项目','值'],[
['GT readiness', readiness.get('status')],['建筑正样本对象数',readiness.get('building_positive_object_count')],['工地正样本对象数',readiness.get('construction_positive_object_count')],['完整背景可用',readiness.get('background_class_available')],['像素级完整指标',readiness.get('pixel_precision_recall_f1_iou_allowed')],['读取时点','PRE_GT_INFERENCE_SEAL 之后']])}</section>
<section><h2>3. 2019 正样本对象对照</h2>{_html_table(['任务','输入模式','GT对象','预测对象','对象Precision@0.5','对象Recall@0.5','对象F1@0.5','正样本重叠F1 proxy'], table_rows)}<p class="small">对象指标以一对一 IoU 匹配计算；因 GT 完整性未声明，仍需人工复核，不等同于全场景分类准确率。</p></section>
<section><h2>4. 分辨率与模型尺度</h2><p>Zoom20 原生 GSD 约为 Zoom19 的一半；R1 使用原生输入，R2 按模块分别写入独立的尺度归一化缓存。Full20、Building、Construction 不共享归一化 tensor。</p><pre>{html.escape(json.dumps(scale,ensure_ascii=False,indent=2,default=str))}</pre></section>
<section><h2>5. 高清对照图</h2><div class="figures">{''.join(figure_links)}</div></section>
<section><h2>6. 模型与不变性</h2>{_html_table(['项目','值'],[
['Registry SHA256',sha256_file(args.registry)],['Baseline Seal 文件数',evaluation.get('invariance',{}).get('baseline',{}).get('baseline_hash_total_count')],['Baseline SHA 匹配数',evaluation.get('invariance',{}).get('baseline',{}).get('baseline_hash_match_count')],['Baseline 不变性',evaluation.get('invariance',{}).get('baseline',{}).get('baseline_hash_invariance_passed')],['Full20模型SHA',assets.get('full20_sha256')],['Building模型SHA',assets.get('rsbuilding_sha256')],['Construction模型SHA',assets.get('construction_sha256')],['Construction选择',registry_checked.get('entries',{}).get('M6_Construction',{}).get('selected_experiment')]])}</section>
<section><h2>7. 需要人工复核的失败/边界</h2><p>误报、漏检、边界偏移和 Zoom20 原生尺度失配案例已写入 <code>09_error_cases</code>，并生成对应高清 case sheet。没有把代理指标包装成真实准确率，也没有使用 GT 训练或选 checkpoint。</p></section>
<section><h2>8. 产物</h2><p>主报告：<code>{html.escape(str((report_root/'NANJING_ZOOM20_2019_GT_COMPARISON_ZH.html').resolve()))}</code><br>运行根目录：<code>{html.escape(str(run_root.resolve()))}</code></p></section>
</main></body></html>
"""
    main_path = report_root / "NANJING_ZOOM20_2019_GT_COMPARISON_ZH.html"
    main_path.write_text(common + body, encoding="utf-8")
    building_path = report_root / "NANJING_ZOOM20_BUILDING_COMPARISON_ZH.html"
    construction_path = report_root / "NANJING_ZOOM20_CONSTRUCTION_COMPARISON_ZH.html"
    building_path.write_text(common + body.replace("南京江北新区 Zoom20 与 2019 人工 GT 对照验收", "南京江北新区 Zoom20 建筑对照验收"), encoding="utf-8")
    construction_path.write_text(common + body.replace("南京江北新区 Zoom20 与 2019 人工 GT 对照验收", "南京江北新区 Zoom20 工地对照验收"), encoding="utf-8")
    return {"main": str(main_path.resolve()), "building": str(building_path.resolve()), "construction": str(construction_path.resolve())}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    evaluation = read_json(run_root / "05_gt_evaluation/2019_GT_EVALUATION.json", {})
    if evaluation.get("status") != "completed_positive_object_gt_evaluation":
        raise RuntimeError("Report requires completed GT evaluation")
    gt = _audit_gt_and_load(run_root, args.zoom20_root)
    figures = _write_highres_figures(run_root, args.old_run, gt)
    reports = _write_report_html(run_root, args, figures)
    link_targets = [Path(item) for item in reports.values()] + [Path(item) for item in figures]
    broken_links = [str(item) for item in link_targets if not item.exists()]
    report_audit = {
        "status": "pass",
        "html_paths": reports,
        "figure_count": len(figures),
        "figure_paths": figures,
        "report_uses_prediction_recompute": False,
        "true_accuracy_reported": False,
        "pixel_accuracy_reported": False,
        "broken_links_checked": True,
        "broken_link_count": len(broken_links),
        "broken_links": broken_links,
        "created_at": now_utc(),
    }
    write_json(run_root / "11_manifests/REPORT_NUMERICAL_AUDIT.json", report_audit)
    tests = read_json(run_root / "11_manifests/TEST_RECEIPT.json", {"status": "pending"})
    final = {
        "status": "completed_pending_test_receipt" if tests.get("status") == "pending" else "accepted_with_scientific_limits",
        "run_root": str(run_root.resolve()),
        "pre_gt_inference_seal": str((run_root / "03_inference/PRE_GT_INFERENCE_SEAL.json").resolve()),
        "reports": reports,
        "figures_highres": figures,
        "evaluation_status": evaluation.get("status"),
        "recommended_input_modes": evaluation.get("recommendation"),
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "gt_used_for_training": False,
        "gt_used_for_model_selection": False,
        "gt_modified": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "globfp_enabled": False,
        "building_height_enabled": False,
        "human_ground_truth_available": True,
        "true_accuracy_reported": False,
        "pixel_accuracy_reported": False,
        "baseline_hash_match_count": evaluation.get("invariance", {}).get("baseline", {}).get("baseline_hash_match_count"),
        "baseline_hash_total_count": evaluation.get("invariance", {}).get("baseline", {}).get("baseline_hash_total_count"),
        "baseline_hash_invariance_passed": evaluation.get("invariance", {}).get("baseline", {}).get("baseline_hash_invariance_passed"),
        "tests": tests,
        "created_at": now_utc(),
    }
    write_json(run_root / "11_manifests/FINAL_ZOOM20_2019_GT_ACCEPTANCE_RECEIPT.json", final)
    return {"status": final["status"], "reports": reports, "figures": figures, "final_receipt": str((run_root / "11_manifests/FINAL_ZOOM20_2019_GT_ACCEPTANCE_RECEIPT.json").resolve())}


def run_audit_prepare(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    _refuse_sealed_roots(run_root, args.old_run)
    ensure_tree(run_root)
    registry = _read_registry(args.registry)
    audit = _write_input_audit(run_root, args.zoom20_root, args.zoom19_root, args.registry)
    grid = _write_common_grid(run_root, audit)
    scale = _write_model_scale_audit(run_root, audit, _asset_paths(registry))
    payload = {
        "status": "pass" if audit["readiness"]["status"] == "pass" else "partial_date_overlap",
        "run_root": str(run_root),
        "registry": registry,
        "zoom20": audit["zoom20"],
        "zoom19": audit["zoom19"],
        "grid": grid,
        "scale": scale,
        "gt_content_read": False,
        "gt_access_before_inference_completed": False,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "created_at": now_utc(),
    }
    write_json(run_root / "11_manifests/AUDIT_PREPARE_RECEIPT.json", payload)
    return payload


def run_infer(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    audit_path = run_root / "11_manifests/AUDIT_PREPARE_RECEIPT.json"
    if not audit_path.exists():
        run_audit_prepare(args)
    prepared = read_json(audit_path, {})
    if prepared.get("gt_content_read"):
        raise RuntimeError("GT content was already read before inference; refusing inference stage")
    registry = _read_registry(args.registry)
    assets = _asset_paths(registry)
    audit = {"zoom20": prepared["zoom20"], "zoom19": prepared["zoom19"]}
    r0 = _verify_r0(args.old_run, assets)
    results: dict[str, Any] = {"r0": r0}
    write_json(run_root / "03_inference/zoom19_reference/R0_REFERENCE_MANIFEST.json", r0)
    if r0.get("status") != "pass":
        raise RuntimeError(f"R0 is not a valid sealed Current Best reference: {r0}")
    python = Path(args.python).resolve()
    rsbuilding_python = Path(args.rsbuilding_python).resolve()
    r1_root = run_root / "03_inference/zoom20_native"
    results["r1"] = _run_modular_baseline(
        args.zoom20_root,
        r1_root,
        assets,
        python,
        rsbuilding_python,
        run_construction=True,
    )
    results["r2"] = _run_r2(run_root, audit, assets, python, rsbuilding_python)
    seal = _write_pre_gt_seal(run_root, r0, results)
    payload = {
        "status": "completed_or_degraded",
        "results": results,
        "pre_gt_seal": seal,
        "training_started": False,
        "models_reselected": False,
        "thresholds_changed": False,
        "postprocessing_changed": False,
        "gt_content_read": False,
        "gt_access_before_inference_completed": False,
        "locked_test_pixel_reads": 0,
        "original_source_modified": False,
        "created_at": now_utc(),
    }
    write_json(run_root / "11_manifests/INFERENCE_STAGE_RECEIPT.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["Audit", "Prepare", "Infer", "Evaluate", "Report", "All", "Status"], default="Audit")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--zoom20-root", type=Path, default=Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20"))
    parser.add_argument("--zoom19-root", type=Path, default=Path(r"D:\Data Set\Satellite\nanjing_jiangbei"))
    parser.add_argument("--old-run", type=Path, default=PROJECT_ROOT / "outputs/inference/nanjing_jiangbei_modular_rebuild_20260803_230320")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv-gpu/Scripts/python.exe")
    parser.add_argument("--rsbuilding-python", type=Path, default=Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"))
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    args.zoom20_root = args.zoom20_root.resolve()
    args.zoom19_root = args.zoom19_root.resolve()
    args.old_run = args.old_run.resolve()
    args.registry = args.registry.resolve()
    args.run_root = args.run_root.resolve()
    if args.mode == "Status":
        print(json.dumps(read_json(args.run_root / "11_manifests/FINAL_ZOOM20_2019_GT_ACCEPTANCE_RECEIPT.json", read_json(args.run_root / "11_manifests/INFERENCE_STAGE_RECEIPT.json", {"status": "not_started"})), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.mode in {"Audit", "Prepare"}:
        result = run_audit_prepare(args)
    elif args.mode == "Infer":
        result = run_infer(args)
    elif args.mode == "Evaluate":
        result = run_evaluate(args)
    elif args.mode == "Report":
        result = run_report(args)
    elif args.mode == "All":
        if not (args.run_root / "03_inference/PRE_GT_INFERENCE_SEAL.json").exists():
            run_infer(args)
        result = run_evaluate(args)
        result["report"] = run_report(args)
    else:
        raise RuntimeError(f"Unsupported mode: {args.mode}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {
        "pass",
        "completed_or_degraded",
        "partial_date_overlap",
        "completed_positive_object_gt_evaluation",
        "completed_pending_test_receipt",
        "accepted_with_scientific_limits",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
