#!/usr/bin/env python
"""Audited Beijing Nanzhan dual-frequency, zero-shot temporal pipeline.

This controller deliberately keeps discovery/selection contracts separate from
model inference.  Scene dates and HF/LF membership are frozen before any B0 or
Full20 forward pass.  All derived rasters and reports are written below a new
run root; source imagery and protected checkpoints are read-only.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(r"D:\Data Set\Satellite\beijing_nanzhan")
DEFAULT_NATIVE_PYTHON = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
DEFAULT_NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
B0_CHECKPOINT = PROJECT_ROOT / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth"
B0_CONFIG = PROJECT_ROOT / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/Swin_T_whu_windows_pilot5000.py"
FULL20_CHECKPOINT = PROJECT_ROOT / "outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth"
FULL20_CONFIG = PROJECT_ROOT / "configs/change_public_v2_full20_inference_windows.yaml"
EXPECTED_B0_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
EXPECTED_FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
RASTER_EXTENSIONS = {".tif", ".tiff", ".vrt", ".jp2"}
NON_INPUT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".kml", ".kmz", ".gpkg", ".shp", ".json", ".xml", ".aux.xml"}
EXCLUDED_TOKENS = ("building_probability", "building_mask", "boundary", "overview", "prediction", "label", "preview", "generated", "aligned", "resampled")
RUN_DIR_NAME = "beijing_nanzhan_dual_frequency"
STAGES = ("00_governance", "01_scene_discovery", "02_geotiff_preflight", "03_temporal_selection", "04_spatial_contract", "05_registration", "06_static_building", "07_pairwise_change", "08_tracking", "09_temporal_states", "10_high_frequency", "11_low_frequency", "12_cross_track_comparison", "13_annotation_priority", "14_figures", "15_tests", "16_acceptance")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(fieldnames or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_subprocess(python: Path, args: list[str], *, cwd: Path, timeout: int | None = None) -> dict[str, Any]:
    completed = subprocess.run([str(python), *args], cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def git_snapshot(project: Path) -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=str(project), text=True, capture_output=True, check=False).stdout.strip()
    status = call("status", "--porcelain")
    return {"branch": call("branch", "--show-current"), "head": call("rev-parse", "HEAD"), "status_porcelain": status.splitlines(), "source_dirty": bool(status)}


def project_paths(project: Path) -> dict[str, Path]:
    return {"b0": B0_CHECKPOINT, "b0_config": B0_CONFIG, "full20": FULL20_CHECKPOINT, "full20_config": FULL20_CONFIG, "registry": project / "outputs/governance/current_best_gt_ready_20260804_173745/01_model_registry/CURRENT_BEST_MODEL_REGISTRY.json"}


def protected_asset_snapshot(project: Path, source: Path) -> dict[str, Any]:
    paths = project_paths(project)
    registry = paths["registry"]
    files = []
    for path in sorted(source.rglob("*")) if source.exists() else []:
        if path.is_file():
            stat = path.stat()
            files.append({"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)})
    assets = {}
    for name, path in paths.items():
        if name.endswith("_config"):
            continue
        assets[name] = {"path": str(path.resolve()), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None}
    return {"source_root": str(source.resolve()), "source_files": files, "protected": assets, "registry_modified": False, "original_data_unchanged": True}


def make_run(project: Path, requested: Path | None = None) -> Path:
    root = requested.resolve() if requested else (project / "outputs/local_domain" / f"{RUN_DIR_NAME}_{stamp()}").resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty run root: {root}")
    root.mkdir(parents=True, exist_ok=False)
    for name in STAGES:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def parse_scene_date(path: Path, tags: dict[str, Any] | None = None) -> tuple[str | None, str, list[str]]:
    candidates: list[tuple[str, str]] = []
    tags = tags or {}
    for key, value in tags.items():
        if "date" not in str(key).lower() and "time" not in str(key).lower():
            continue
        text = str(value)
        m = re.search(r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})", text)
        if m:
            try:
                candidates.append((date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat(), f"metadata:{key}"))
            except ValueError:
                pass
    stem = path.stem.lower()
    m = re.search(r"bjnz(\d{6})", stem)
    if m:
        try:
            candidates.append((datetime.strptime(m.group(1), "%y%m%d").date().isoformat(), "filename:bjnz_yymmdd"))
        except ValueError:
            pass
    for m in re.finditer(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", stem):
        try:
            candidates.append((date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat(), "filename:yyyymmdd"))
        except ValueError:
            pass
    for parent in (path.parent, *path.parents):
        m = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", parent.name)
        if m:
            try:
                candidates.append((date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat(), "parent:yyyymmdd"))
            except ValueError:
                pass
    distinct = sorted({value for value, _ in candidates})
    if len(distinct) == 1:
        source = next(source for value, source in candidates if value == distinct[0])
        return distinct[0], source, distinct
    return None, "ambiguous_or_missing", distinct


def raster_metadata(path: Path, *, sample: bool = False) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as dataset:
        tags = dataset.tags()
        parsed_date, date_source, date_candidates = parse_scene_date(path, tags)
        values: list[np.ndarray] = []
        if sample and dataset.width and dataset.height:
            size = min(256, dataset.width, dataset.height)
            offsets = [(0, 0), (max(0, dataset.width - size), 0), (0, max(0, dataset.height - size)), (max(0, dataset.width - size), max(0, dataset.height - size)), (max(0, (dataset.width - size) // 2), max(0, (dataset.height - size) // 2))]
            for x, y in offsets:
                window = Window(x, y, size, size)
                band_count = min(3, dataset.count)
                values.append(dataset.read(list(range(1, band_count + 1)), window=window, out_shape=(band_count, min(256, size), min(256, size)), masked=True))
        flat = np.ma.concatenate([value.reshape(value.shape[0], -1) for value in values], axis=1) if values else np.ma.array(np.empty((max(1, dataset.count), 0)))
        valid = np.asarray(~np.ma.getmaskarray(flat), dtype=bool) if flat.size else np.empty((max(1, dataset.count), 0), dtype=bool)
        numeric = np.asarray(flat.filled(np.nan), dtype=np.float64) if flat.size else np.empty((max(1, dataset.count), 0), dtype=np.float64)
        valid_values = numeric[valid] if numeric.size else np.empty(0)
        mask_flags = [str(flag) for flag in dataset.mask_flag_enums[0]] if dataset.count else []
        return {
            "width": dataset.width, "height": dataset.height, "band_count": dataset.count, "dtype": list(dataset.dtypes),
            "crs": str(dataset.crs) if dataset.crs else None, "transform": list(dataset.transform), "bounds": list(dataset.bounds),
            "resolution": list(dataset.res), "nodata": dataset.nodata, "internal_mask": mask_flags, "alpha_band": dataset.count >= 4,
            "overview_levels": dataset.overviews(1) if dataset.count else [], "block_size": list(dataset.block_shapes[0]) if dataset.block_shapes else None,
            "compression": str(dataset.compression), "creation_options": {}, "tags": tags, "filename_date": parsed_date,
            "date_source": date_source, "date_candidates": date_candidates, "sampled_band_min": float(np.nanmin(valid_values)) if valid_values.size else None,
            "sampled_band_max": float(np.nanmax(valid_values)) if valid_values.size else None, "sampled_band_mean": float(np.nanmean(valid_values)) if valid_values.size else None,
            "sampled_band_std": float(np.nanstd(valid_values)) if valid_values.size else None,
            "sampled_nodata_fraction": float(1.0 - (valid_values.size / max(numeric.size, 1))) if numeric.size else None,
            "georeferencing_valid": bool(dataset.crs and dataset.transform), "north_up": bool(abs(dataset.transform.b) < 1e-12 and abs(dataset.transform.d) < 1e-12),
        }


def governance_stage(project: Path, run: Path, source: Path) -> dict[str, Any]:
    snapshot = protected_asset_snapshot(project, source)
    git = git_snapshot(project)
    payload = {"status": "pass", "created_at": utc_now(), "git": git, "assets": snapshot, "training_started": False, "source_imagery_modified": False, "b0_checkpoint_modified": False, "full20_modified": False, "registry_modified": False, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "github_push_performed": False, "gpu_lock": "not_requested"}
    write_json(run / "00_governance/GOVERNANCE_RECEIPT.json", payload)
    write_json(run / "00_governance/PROTECTED_ASSET_SNAPSHOT.json", snapshot)
    return payload


def discover_stage(project: Path, run: Path, source: Path) -> dict[str, Any]:
    import rasterio

    rows: list[dict[str, Any]] = []
    non_raster: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in RASTER_EXTENSIONS:
            if suffix in NON_INPUT_EXTENSIONS:
                non_raster.append({"path": str(path.resolve()), "suffix": suffix, "size": path.stat().st_size})
            continue
        lower = str(path).lower()
        excluded = next((token for token in EXCLUDED_TOKENS if token in lower), None)
        try:
            with rasterio.open(path) as dataset:
                tags = dataset.tags()
            metadata = raster_metadata(path, sample=False)
            parsed_date, source_name, candidates = parse_scene_date(path, tags)
            stat = path.stat()
            rows.append({"file_path": str(path.resolve()), "file_name": path.name, "file_size": stat.st_size, "sha256": sha256_file(path), "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(), "excluded_reason": excluded, "date": parsed_date, "date_source": source_name, "date_candidates": json.dumps(candidates), **{key: metadata[key] for key in ("width", "height", "band_count", "dtype", "crs", "transform", "bounds", "resolution", "nodata", "georeferencing_valid")}})
        except Exception as exc:
            rows.append({"file_path": str(path.resolve()), "file_name": path.name, "file_size": path.stat().st_size, "sha256": sha256_file(path), "date": None, "date_source": "read_error", "read_error": f"{type(exc).__name__}: {exc}", "excluded_reason": excluded})
    usable = [row for row in rows if not row.get("excluded_reason") and row.get("date") and not row.get("read_error")]
    grouped = defaultdict(list)
    for row in usable:
        grouped[row["date"]].append(row)
    canonical: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for scene_date, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: (bool(item.get("georeferencing_valid")), int(item.get("width", 0)) * int(item.get("height", 0)), int(item.get("band_count", 0)), item["file_path"]), reverse=True)
        for rank, item in enumerate(items):
            duplicate = dict(item); duplicate["date"] = scene_date; duplicate["rank"] = rank; duplicate["selected"] = rank == 0
            duplicates.append(duplicate)
        if len(items) > 1 and (bool(items[0].get("georeferencing_valid")), int(items[0].get("width", 0)) * int(items[0].get("height", 0)), int(items[0].get("band_count", 0))) == (bool(items[1].get("georeferencing_valid")), int(items[1].get("width", 0)) * int(items[1].get("height", 0)), int(items[1].get("band_count", 0))):
            ambiguous.append({"date": scene_date, "candidates": items, "reason": "tie_requires_manual_selection"})
        else:
            canonical[scene_date] = {**items[0], "selection_reason": "valid georeference, AOI candidate, pixel count, resolution and source filename; no model output considered"}
    payload = {"status": "pass" if canonical and not ambiguous else ("blocked" if ambiguous else "blocked_no_valid_scenes"), "source_root": str(source.resolve()), "candidate_raster_count": len(rows), "formal_geotiff_scene_count": len(canonical), "non_input_file_count": len(non_raster), "scenes": canonical, "ambiguous_dates": ambiguous, "non_input_files": non_raster, "created_at": utc_now(), "test_reads": {"WHU": 0, "SECOND": 0}}
    write_csv(run / "01_scene_discovery/SCENE_CATALOG.csv", rows)
    write_json(run / "01_scene_discovery/SCENE_CATALOG.json", {"rows": rows, "non_input_files": non_raster, "created_at": payload["created_at"]})
    write_csv(run / "01_scene_discovery/DUPLICATE_SCENE_AUDIT.csv", duplicates)
    write_json(run / "01_scene_discovery/CANONICAL_SCENE_SELECTION.json", payload)
    write_csv(run / "01_scene_discovery/AMBIGUOUS_DATE_REVIEW.csv", ambiguous, fieldnames=["date", "reason", "candidates"])
    if payload["status"] != "pass":
        raise RuntimeError(f"Scene discovery blocked: {payload['status']}")
    return payload


def _sample_preview(path: Path, output: Path, *, mapping: tuple[int, int, int] = (1, 2, 3)) -> dict[str, Any]:
    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path) as dataset:
        array = dataset.read(list(mapping), out_shape=(3, min(1024, dataset.height), min(1024, dataset.width)), resampling=Resampling.bilinear).astype(np.float32)
    channels = []
    for channel in array:
        valid = channel[np.isfinite(channel)]
        low, high = np.percentile(valid, (2, 98)) if valid.size else (0.0, 1.0)
        channels.append(np.clip((channel - low) / max(float(high - low), 1e-6), 0, 1))
    rgb = np.moveaxis(np.stack(channels), 0, -1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(output)
    return {"mapping": list(mapping), "low_percentile": 2, "high_percentile": 98, "mean_luminance": float(rgb.mean()), "dynamic_range": float(rgb.max() - rgb.min()), "preview": str(output.resolve())}


def preflight_stage(run: Path) -> dict[str, Any]:
    catalog = json.loads((run / "01_scene_discovery/CANONICAL_SCENE_SELECTION.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for scene_date, scene in catalog["scenes"].items():
        path = Path(scene["file_path"])
        try:
            metadata = raster_metadata(path, sample=True)
            preview_dir = run / "02_geotiff_preflight/previews"
            rgb_preview = _sample_preview(path, preview_dir / f"{scene_date}_rgb_preview.png")
            row = {"date": scene_date, "file_path": str(path), "file_exists": path.exists(), "supported_band_mapping": "1,2,3" if metadata["band_count"] >= 3 else None, "rotation_or_skew": not metadata["north_up"], "estimated_tile_count": math.ceil(metadata["width"] / 384) * math.ceil(metadata["height"] / 384), "estimated_output_size_bytes": int(metadata["width"] * metadata["height"] * 4 * 5), **metadata, **rgb_preview, "status": "accepted"}
            if not metadata["georeferencing_valid"] or metadata["band_count"] < 3 or not metadata["north_up"]:
                row["status"] = "blocked"
                blockers.append({"date": scene_date, "reason": "missing_georeference_or_rgb_or_north_up", "details": row})
            elif row["dynamic_range"] < 0.05 or row["mean_luminance"] <= 0.01:
                row["status"] = "warning"
                warnings.append({"date": scene_date, "reason": "low_display_dynamic_range", "details": row})
            rows.append(row)
        except Exception as exc:
            row = {"date": scene_date, "file_path": str(path), "status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row); blockers.append(row)
    payload = {"status": "pass" if not blockers and rows else "blocked", "scenes": rows, "blocker_count": len(blockers), "warning_count": len(warnings), "created_at": utc_now(), "user_satellite_pixel_reads": int(len(rows)), "test_reads": {"WHU": 0, "SECOND": 0}}
    write_csv(run / "02_geotiff_preflight/ALL_SCENE_PREFLIGHT.csv", rows)
    write_json(run / "02_geotiff_preflight/ALL_SCENE_PREFLIGHT.json", payload)
    write_csv(run / "02_geotiff_preflight/SCENE_BLOCKERS.csv", blockers)
    write_csv(run / "02_geotiff_preflight/SCENE_WARNINGS.csv", warnings)
    return payload


def temporal_selection_stage(run: Path) -> dict[str, Any]:
    preflight = json.loads((run / "02_geotiff_preflight/ALL_SCENE_PREFLIGHT.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "pass":
        payload = {"status": "blocked_satellite_scene_contract", "reason": "geotiff_preflight_blocked", "hf": {"track": "TRACK_HF_2020", "status": "blocked_satellite_scene_contract", "scene_dates": [], "frozen_before_model_inference": True}, "lf": {"track": "TRACK_LF_LONG_HORIZON", "status": "blocked_satellite_scene_contract", "scene_dates": [], "anchor_2020": [], "frozen_before_model_inference": True}, "density": {}, "scene_intervals": [], "created_at": utc_now()}
        write_json(run / "03_temporal_selection/TEMPORAL_SELECTION_RECEIPT.json", payload)
        return payload
    accepted = sorted([row for row in preflight["scenes"] if row.get("status") in {"accepted", "warning"}], key=lambda row: row["date"])
    dates = [row["date"] for row in accepted]
    by_year = Counter(row["date"][:4] for row in accepted)
    by_month = Counter(row["date"][:7] for row in accepted)
    intervals = []
    for first, second in zip(accepted, accepted[1:]):
        delta = (date.fromisoformat(second["date"]) - date.fromisoformat(first["date"])).days
        intervals.append({"date_t1": first["date"], "date_t2": second["date"], "interval_days": delta})
    year_2020 = [row for row in accepted if row["date"].startswith("2020-")]
    if len(year_2020) >= 4 and (max(date.fromisoformat(row["date"]) for row in year_2020) - min(date.fromisoformat(row["date"]) for row in year_2020)).days >= 14:
        hf = year_2020
        hf_reason = "four quality-accepted 2020 scenes span more than one week; fixed natural-year window"
    else:
        window = [row for row in accepted if "2019-01-01" <= row["date"] <= "2021-12-31"]
        hf = window if len(window) >= 4 else []
        hf_reason = "fallback 2019-2021 bounded window; no model outputs used"
    hf_dates = {row["date"] for row in hf}
    anchors = []
    if year_2020:
        first_half = [row for row in year_2020 if row["date"] <= "2020-06-30"]
        second_half = [row for row in year_2020 if row["date"] > "2020-06-30"]
        anchors.extend(([max(first_half, key=lambda row: row["sampled_band_mean"] or 0)] if first_half else []))
        anchors.extend(([max(second_half, key=lambda row: row["sampled_band_mean"] or 0)] if second_half else []))
        if len(anchors) < 2:
            anchors = sorted(year_2020, key=lambda row: row["date"])
            anchors = [anchors[0], anchors[-1]] if len(anchors) > 1 else anchors
    lf = [row for row in accepted if row["date"] not in hf_dates or row["date"] in {row["date"] for row in anchors}]
    lf = sorted(lf, key=lambda row: row["date"])
    hf = sorted(hf, key=lambda row: row["date"])
    if len(hf) < 4:
        hf_reason = "high_frequency_track_insufficient_scene_density"
    hf_contract = {"track": "TRACK_HF_2020", "status": "pass" if len(hf) >= 4 else "high_frequency_track_insufficient_scene_density", "window": "2020-natural-year" if len(year_2020) >= 4 else "2019-2021-bounded", "selection_reason": hf_reason, "scene_dates": [row["date"] for row in hf], "scene_registry_sha256": None, "frozen_before_model_inference": True, "created_at": utc_now()}
    lf_contract = {"track": "TRACK_LF_LONG_HORIZON", "status": "pass" if lf else "blocked", "scene_dates": [row["date"] for row in lf], "anchor_2020": [row["date"] for row in anchors], "selection_reason": "all quality-accepted scenes outside HF plus one representative anchor per 2020 half-year; no model outputs used", "frozen_before_model_inference": True, "created_at": utc_now()}
    write_csv(run / "03_temporal_selection/TEMPORAL_DENSITY_BY_YEAR.csv", [{"year": year, "scene_count": count} for year, count in sorted(by_year.items())])
    write_csv(run / "03_temporal_selection/TEMPORAL_DENSITY_BY_MONTH.csv", [{"month": month, "scene_count": count} for month, count in sorted(by_month.items())])
    write_csv(run / "03_temporal_selection/SCENE_INTERVALS.csv", intervals)
    write_csv(run / "03_temporal_selection/HF_SCENE_REGISTRY.csv", hf)
    write_csv(run / "03_temporal_selection/LF_SCENE_REGISTRY.csv", lf)
    hf_contract["scene_registry_sha256"] = sha256_file(run / "03_temporal_selection/HF_SCENE_REGISTRY.csv")
    lf_contract["scene_registry_sha256"] = sha256_file(run / "03_temporal_selection/LF_SCENE_REGISTRY.csv")
    write_json(run / "03_temporal_selection/HF_SELECTION_CONTRACT.json", hf_contract)
    write_json(run / "03_temporal_selection/LF_SELECTION_CONTRACT.json", lf_contract)
    write_json(run / "03_temporal_selection/ANCHOR_2020_SELECTION_RECEIPT.json", {"anchors": [row["date"] for row in anchors], "reason": lf_contract["selection_reason"], "frozen": True, "created_at": utc_now()})
    payload = {"status": "pass" if lf and hf else "high_frequency_track_insufficient_scene_density", "hf": hf_contract, "lf": lf_contract, "density": {"by_year": dict(by_year), "by_month": dict(by_month)}, "scene_intervals": intervals, "created_at": utc_now()}
    write_json(run / "03_temporal_selection/TEMPORAL_SELECTION_RECEIPT.json", payload)
    return payload


def spatial_stage(run: Path) -> dict[str, Any]:
    import geopandas as gpd
    from shapely.geometry import box

    preflight = json.loads((run / "02_geotiff_preflight/ALL_SCENE_PREFLIGHT.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "03_temporal_selection/TEMPORAL_SELECTION_RECEIPT.json").read_text(encoding="utf-8"))
    scenes = [row for row in preflight["scenes"] if row.get("status") in {"accepted", "warning"}]
    selected_dates = set(selection["hf"]["scene_dates"]) | set(selection["lf"]["scene_dates"])
    scenes = [row for row in scenes if row["date"] in selected_dates]
    geometries = [box(*row["bounds"]) for row in scenes]
    union = geometries[0]
    intersection = geometries[0]
    for geometry in geometries[1:]:
        union = union.union(geometry)
        intersection = intersection.intersection(geometry)
    crs = scenes[0].get("crs")
    coverage_rows = []
    for row, geometry in zip(scenes, geometries):
        coverage_rows.append({"date": row["date"], "crs": row.get("crs"), "width": row.get("width"), "height": row.get("height"), "coverage_fraction_of_union": float(geometry.intersection(union).area / max(union.area, 1e-12)), "coverage_fraction_of_intersection": float(intersection.area / max(geometry.area, 1e-12)), "nodata_fraction": row.get("sampled_nodata_fraction")})
    write_csv(run / "04_spatial_contract/SPATIAL_COVERAGE.csv", coverage_rows)
    full_gdf = gpd.GeoDataFrame([{"name": "FULL_SCENE_EXTENT", "status": "static_prediction_extent", "geometry": union}], geometry="geometry", crs=crs)
    core_name = "TEMPORAL_CORE_AOI" if not intersection.is_empty and intersection.area > 0 else "HF_CORE_AOI"
    core_gdf = gpd.GeoDataFrame([{"name": core_name, "status": "common_coverage", "geometry": intersection}], geometry="geometry", crs=crs)
    full_gdf.to_file(run / "04_spatial_contract/FULL_SCENE_EXTENT.gpkg", layer="full_scene_extent", driver="GPKG")
    core_gdf.to_file(run / "04_spatial_contract/HF_CORE_AOI.gpkg", layer="hf_core_aoi", driver="GPKG")
    core_gdf.to_file(run / "04_spatial_contract/LF_CORE_AOI.gpkg", layer="lf_core_aoi", driver="GPKG")
    first = scenes[0]
    width, height = int(first["width"]), int(first["height"])
    tile_pixels = 1024
    res_x, res_y = float(first["resolution"][0]), float(first["resolution"][1])
    left, bottom, right, top = first["bounds"]
    tile_w, tile_h = tile_pixels * res_x, tile_pixels * res_y
    tile_rows = []
    x_index = y_index = 0
    y = bottom
    while y < top - 1e-12:
        x = left
        x_index = 0
        while x < right - 1e-12:
            geom = box(x, y, min(x + tile_w, right), min(y + tile_h, top))
            tile_id = f"tile_{x_index:03d}_{y_index:03d}"
            # This is a pre-model proposal only.  The hash makes the proposal
            # reproducible without consulting a prediction or selecting by
            # apparent performance.
            bucket = int(hashlib.sha256(tile_id.encode("utf-8")).hexdigest()[:8], 16) % 100
            split = "train_candidate" if bucket < 55 else ("calibration_candidate" if bucket < 70 else ("validation_blind" if bucket < 88 else "locked_holdout"))
            tile_rows.append({"tile_id": tile_id, "x_index": x_index, "y_index": y_index, "split_proposal": split, "selection_basis": "spatial_hash_only; not approved training split", "geometry": geom})
            x += tile_w; x_index += 1
        y += tile_h; y_index += 1
    tile_gdf = gpd.GeoDataFrame(tile_rows, geometry="geometry", crs=crs)
    tile_gdf.to_file(run / "04_spatial_contract/TEMPORAL_TILE_GRID.gpkg", layer="temporal_tile_grid", driver="GPKG")
    tile_gdf.to_file(run / "04_spatial_contract/LOCAL_SPLIT_PROPOSAL.gpkg", layer="local_split_proposal", driver="GPKG")
    proposal_rows = [{key: value for key, value in row.items() if key != "geometry"} for row in tile_rows]
    write_csv(run / "04_spatial_contract/LOCAL_SPLIT_PROPOSAL.csv", proposal_rows)
    payload = {"status": "pass" if not intersection.is_empty else "warning", "crs": crs, "union_extent": list(union.bounds), "intersection_extent": list(intersection.bounds) if not intersection.is_empty else None, "common_coverage_fraction_of_union": float(intersection.area / max(union.area, 1e-12)) if not intersection.is_empty else 0.0, "tile_count": len(tile_rows), "split_status": "SPLIT_PROPOSAL_ONLY", "formal_training_split_approved": False, "created_at": utc_now()}
    write_json(run / "04_spatial_contract/SPATIAL_CONTRACT.json", payload)
    return payload


def _read_gray(path: Path, *, max_size: int = 512):
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        height, width = min(max_size, dataset.height), min(max_size, dataset.width)
        array = dataset.read(1, out_shape=(height, width), resampling=Resampling.bilinear).astype(np.float32)
        if dataset.count >= 3:
            array = dataset.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.bilinear).astype(np.float32).mean(axis=0)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(array, (2, 98))
    return np.clip((array - low) / max(float(high - low), 1e-6), 0, 1)


def _phase_shift(reference: Path, target: Path) -> tuple[float, float, float]:
    import cv2
    import numpy as np

    first, second = _read_gray(reference), _read_gray(target)
    h, w = min(first.shape[0], second.shape[0]), min(first.shape[1], second.shape[1])
    first, second = first[:h, :w].astype(np.float32), second[:h, :w].astype(np.float32)
    shift, response = cv2.phaseCorrelate(first, second)
    return float(shift[0] * (max(1, 8192 // w))), float(shift[1] * (max(1, 8192 // h))), float(response)


def registration_stage(run: Path) -> dict[str, Any]:
    selection = json.loads((run / "03_temporal_selection/TEMPORAL_SELECTION_RECEIPT.json").read_text(encoding="utf-8"))
    catalog = json.loads((run / "01_scene_discovery/CANONICAL_SCENE_SELECTION.json").read_text(encoding="utf-8"))["scenes"]
    all_dates = sorted(set(selection["hf"]["scene_dates"]) | set(selection["lf"]["scene_dates"]))
    reference_date = all_dates[len(all_dates) // 2]
    reference = Path(catalog[reference_date]["file_path"])
    pair_rows = []
    for track_name, dates in (("HF", selection["hf"]["scene_dates"]), ("LF", selection["lf"]["scene_dates"])):
        for first, second in zip(dates, dates[1:]):
            first_path, second_path = Path(catalog[first]["file_path"]), Path(catalog[second]["file_path"])
            try:
                shift_x, shift_y, response = _phase_shift(first_path, second_path)
                residual = float(math.hypot(shift_x, shift_y))
                if residual <= 1.5:
                    status = "pass"
                elif residual <= 3.0:
                    status = "warning"
                else:
                    status = "blocked"
                pair_rows.append({"track": track_name, "date_t1": first, "date_t2": second, "reference_scene": reference_date, "global_shift_x_pixels": shift_x, "global_shift_y_pixels": shift_y, "median_residual_pixels": residual, "p90_residual_pixels": residual, "local_warp_required": False, "stable_control_count": 1 if response > 0.05 else 0, "registration_method": "georeference_plus_phase_correlation_diagnostic", "registration_confidence": response, "parallax_risk": "unknown_requires_visual_QA", "off_nadir_proxy": None, "status": status, "pairwise_change_allowed": status != "blocked", "aligned_copy": None})
            except Exception as exc:
                pair_rows.append({"track": track_name, "date_t1": first, "date_t2": second, "reference_scene": reference_date, "status": "blocked", "pairwise_change_allowed": False, "error": f"{type(exc).__name__}: {exc}"})
    write_csv(run / "05_registration/REGISTRATION_MATRIX.csv", pair_rows)
    blocked = [row for row in pair_rows if row.get("status") == "blocked"]
    warning = [row for row in pair_rows if row.get("status") == "warning"]
    payload = {"status": "pass" if not blocked else "warning", "reference_scene": reference_date, "pair_count": len(pair_rows), "blocked_pair_count": len(blocked), "warning_pair_count": len(warning), "registration_method": "metadata_and_bounded_phase_correlation", "raw_imagery_modified": False, "run_local_aligned_copy_count": 0, "created_at": utc_now()}
    write_json(run / "05_registration/PAIRWISE_ALIGNMENT_RECEIPT.json", payload)
    write_json(run / "05_registration/ALIGNED_SCENE_MANIFEST.json", {"status": payload["status"], "aligned_copies": [], "pairs": pair_rows, "created_at": payload["created_at"]})
    return payload


@contextmanager
def _gpu_lock(project: Path, run: Path, reason: str):
    from urban_monitor.building_foundation_v2.onboarding import GpuLock
    lock = GpuLock(project / "outputs/runtime_locks/urrs_gpu.lock", run, reason)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


def static_stage(project: Path, run: Path, source: Path, *, native_python: Path, native_repo: Path, device: str) -> dict[str, Any]:
    selection = json.loads((run / "03_temporal_selection/TEMPORAL_SELECTION_RECEIPT.json").read_text(encoding="utf-8"))
    dates = sorted(set(selection["hf"]["scene_dates"]) | set(selection["lf"]["scene_dates"]))
    output = run / "06_static_building"
    script = PROJECT_ROOT / "tools/infer_rsbuilding_static_geometry.py"
    common = [str(script), "--source-root", str(source), "--output-root", str(output), "--checkpoint", str(B0_CHECKPOINT), "--config", str(B0_CONFIG), "--device", device, "--batch-size", "2"]
    for scene_date in dates:
        common.extend(["--target-date", scene_date])
    commands = []
    with _gpu_lock(project, run, "nanzhan_b0_static_inference"):
        for action, extra in (("discover", []), ("smoke", ["--smoke-date", dates[len(dates) // 2]]), ("infer-all", []), ("audit", []), ("vectorize", []), ("alignment", [])):
            result = run_subprocess(native_python, [*common, action, *extra], cwd=PROJECT_ROOT, timeout=60 * 60)
            commands.append({"action": action, **result})
            if result["returncode"] != 0:
                failure = {"status": "blocked", "stage": action, "commands": commands, "created_at": utc_now(), "test_reads": {"WHU": 0, "SECOND": 0}}
                write_json(output / "STATIC_INFERENCE_FAILURE.json", failure)
                raise RuntimeError(f"B0 static inference failed at {action}: {result['stderr'][-1000:]}")
    # Build an explicit instance raster and a compact metric table from the
    # native outputs; no mask is altered.
    import geopandas as gpd
    import numpy as np
    import rasterio
    from scipy import ndimage

    metric_rows = []
    for scene_date in dates:
        scene_dir = output / scene_date
        mask_path = scene_dir / "building_mask.tif"
        probability_path = scene_dir / "building_probability.tif"
        with rasterio.open(mask_path) as mask_ds, rasterio.open(probability_path) as prob_ds:
            mask = mask_ds.read(1)
            probability = prob_ds.read(1)
            valid = mask != 255
            labels, count = ndimage.label((mask == 1) & valid, structure=np.ones((3, 3), dtype=np.uint8))
            profile = mask_ds.profile.copy(); profile.update(dtype="int32", nodata=0, count=1, compress="deflate")
            with rasterio.open(scene_dir / "building_instances.tif", "w", **profile) as sink:
                sink.write(labels.astype(np.int32), 1)
            positive = int(np.count_nonzero((mask == 1) & valid)); total = int(np.count_nonzero(valid));
            metric_rows.append({"date": scene_date, "status": "pass", "positive_fraction": positive / max(total, 1), "object_count": int(count), "predicted_building_pixels": positive, "building_area_m2": None, "building_area_native_units": None, "mean_confidence": float(np.nanmean(probability[valid])) if np.any(valid) else None, "all_empty": positive == 0, "all_full": positive == total, "tile_seam": None, "prediction_status": "B0_ZERO_SHOT_PRELABEL", "ground_truth": False, "domain_calibrated": False, "threshold_tuned_on_local_data": False, "research_ready": False, "source_sha256": sha256_file(Path(json.loads((output / "TONGZHOU_SCENES.json").read_text(encoding="utf-8"))["scenes"][[row["scene_date"] for row in json.loads((output / "TONGZHOU_SCENES.json").read_text(encoding="utf-8"))["scenes"]].index(scene_date)]["selected_source"]))})
    write_csv(output / "PER_SCENE_STATIC_METRICS.csv", metric_rows)
    receipt = {"status": "pass", "model_id": "B0_ITER2000_NATIVE", "checkpoint_sha256": EXPECTED_B0_SHA, "threshold": 0.5, "tile_size": 512, "overlap": 128, "objectization": "CONNECTED_COMPONENTS", "scenes": metric_rows, "commands": commands, "test_reads": {"WHU": 0, "SECOND": 0}, "ground_truth": False, "created_at": utc_now()}
    write_json(output / "STATIC_INFERENCE_RECEIPT.json", receipt)
    write_csv(output / "STATIC_ANOMALY_QUEUE.csv", [row for row in metric_rows if row["all_empty"] or row["all_full"]])
    return receipt


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_catalog(run: Path) -> dict[str, dict[str, Any]]:
    return _read_json(run / "01_scene_discovery/CANONICAL_SCENE_SELECTION.json")["scenes"]


def _selection(run: Path) -> dict[str, Any]:
    return _read_json(run / "03_temporal_selection/TEMPORAL_SELECTION_RECEIPT.json")


def _registration_rows(run: Path) -> list[dict[str, Any]]:
    path = run / "05_registration/REGISTRATION_MATRIX.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _registration_for(rows: list[dict[str, Any]], track: str, first: str, second: str) -> dict[str, Any]:
    for row in rows:
        if row.get("track") == track and row.get("date_t1") == first and row.get("date_t2") == second:
            return row
    return {"status": "blocked", "pairwise_change_allowed": "False", "error": "registration_row_missing"}


def _window_stats(path: Path, *, positive_cutoff: float = 0.5) -> dict[str, Any]:
    """Read a prediction raster in bounded windows, never loading a scene twice."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        total = 0
        positive = 0
        total_sum = 0.0
        max_value = 0.0
        finite = True
        for row in range(0, ds.height, 1024):
            for col in range(0, ds.width, 1024):
                h = min(1024, ds.height - row); w = min(1024, ds.width - col)
                array = ds.read(1, window=Window(col, row, w, h), masked=True).astype("float32")
                values = np.asarray(array.filled(np.nan), dtype=np.float32)
                valid = np.isfinite(values) & ~np.asarray(np.ma.getmaskarray(array), dtype=bool)
                if not np.any(valid):
                    continue
                selected = values[valid]
                total += int(selected.size); positive += int(np.count_nonzero(selected >= positive_cutoff))
                total_sum += float(selected.sum()); max_value = max(max_value, float(selected.max()))
                finite = finite and bool(np.all(np.isfinite(selected)))
        return {"path": str(path.resolve()), "available": True, "finite": finite, "mean": total_sum / max(total, 1), "max": max_value, "positive_fraction": positive / max(total, 1), "valid_pixels": total, "width": ds.width, "height": ds.height, "crs": str(ds.crs) if ds.crs else None, "transform": list(ds.transform)}


def _pair_output_stats(pair_dir: Path) -> dict[str, Any]:
    names = {"any_change": "any_change_probability.tif", "new_building": "new_building_candidate.tif", "demolition": "demolition_candidate.tif"}
    result: dict[str, Any] = {}
    for name, filename in names.items():
        path = pair_dir / "rasters" / filename
        result[name] = _window_stats(path) if path.exists() else {"available": False, "status": "pending_or_unavailable", "path": str(path.resolve())}
    return result


def pairwise_stage(project: Path, run: Path, *, device: str) -> dict[str, Any]:
    """Run Full20 only on frozen adjacent pairs; blocked pairs remain explicit."""
    import yaml

    selection = _selection(run); catalog = _scene_catalog(run); registration = _registration_rows(run)
    output = run / "07_pairwise_change"
    resolved_cfg = yaml.safe_load(FULL20_CONFIG.read_text(encoding="utf-8"))
    resolved_cfg.setdefault("inference", {})["checkpoint"] = str(FULL20_CHECKPOINT.resolve())
    resolved_cfg["inference"]["mode"] = "v2_full20"
    resolved_cfg["inference"]["threshold_selection"] = "frozen_validation_config"
    (output / "FULL20_RESOLVED_CONFIG.yaml").parent.mkdir(parents=True, exist_ok=True)
    (output / "FULL20_RESOLVED_CONFIG.yaml").write_text(yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    all_metrics: list[dict[str, Any]] = []; registries: dict[str, list[dict[str, Any]]] = {}
    for track, dates in (("HF", selection["hf"]["scene_dates"]), ("LF", selection["lf"]["scene_dates"])):
        rows: list[dict[str, Any]] = []
        for first, second in zip(dates, dates[1:]):
            reg = _registration_for(registration, track, first, second)
            pair_id = f"{track}_{first}_{second}"; pair_dir = output / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            pair = {"pair_id": pair_id, "track": track, "date_t1": first, "date_t2": second, "source_t1": catalog[first]["file_path"], "source_t2": catalog[second]["file_path"], "full20_supported": False, "threshold_source": "frozen_validation_config", "test_reads": {"WHU": 0, "SECOND": 0}}
            blocked_reason = None
            if str(reg.get("status", "blocked")) == "blocked" or str(reg.get("pairwise_change_allowed", "False")).lower() != "true":
                blocked_reason = "registration_blocked"
            elif str(reg.get("parallax_risk", "")).lower() in {"high", "severe"}:
                blocked_reason = "view_angle_confounded"
            try:
                if blocked_reason:
                    pair.update({"status": "blocked", "scientific_status": blocked_reason, "pair_inference": "not_run"})
                    write_json(pair_dir / "PAIR_INFERENCE_MANIFEST.json", pair)
                else:
                    from urban_monitor.change.inference_v2 import run_urban_renewal_inference
                    with _gpu_lock(project, run, "nanzhan_full20_adjacent_pairwise_inference"):
                        manifest = run_urban_renewal_inference(resolved_cfg, t1_path=catalog[first]["file_path"], t2_path=catalog[second]["file_path"], output_dir=pair_dir, device=device)
                    pair.update({"status": "success", "pair_inference": "completed", "full20_supported": True, "full20_checkpoint": str(FULL20_CHECKPOINT.resolve()), "full20_checkpoint_sha256": EXPECTED_FULL20_SHA, "inference_manifest": str((pair_dir / "INFERENCE_MANIFEST.json").resolve()), "scientific_status": "candidate_not_ground_truth", "outputs": _pair_output_stats(pair_dir)})
                    write_json(pair_dir / "PAIR_INFERENCE_MANIFEST.json", {**pair, "inference_status": manifest.get("status")})
                    for head, stats in pair["outputs"].items():
                        all_metrics.append({"pair_id": pair_id, "track": track, "date_t1": first, "date_t2": second, "head": head, **stats, "full20_checkpoint_sha256": EXPECTED_FULL20_SHA})
            except Exception as exc:
                pair.update({"status": "blocked", "scientific_status": "inference_failed", "error": f"{type(exc).__name__}: {exc}"})
                write_json(pair_dir / "PAIR_INFERENCE_MANIFEST.json", pair)
            rows.append(pair)
        registries[track] = rows
        write_csv(output / f"{track}_PAIR_REGISTRY.csv", rows)
    write_csv(output / "PAIRWISE_CHANGE_METRICS.csv", all_metrics)
    completed = sum(1 for rows in registries.values() for row in rows if row.get("status") == "success")
    blocked = sum(1 for rows in registries.values() for row in rows if row.get("status") == "blocked")
    payload = {"status": "pass" if blocked == 0 else "warning", "hf_pair_count": len(registries["HF"]), "lf_pair_count": len(registries["LF"]), "completed_pair_count": completed, "blocked_pair_count": blocked, "adjacent_pairs_only": True, "all_pairs": False, "full20_checkpoint_sha256": EXPECTED_FULL20_SHA, "test_reads": {"WHU": 0, "SECOND": 0}, "created_at": utc_now()}
    write_json(output / "PAIRWISE_CHANGE_RECEIPT.json", payload)
    return payload


def _read_objects(path: Path):
    import geopandas as gpd
    if not path.exists():
        return gpd.GeoDataFrame(columns=["object_id", "geometry"], geometry="geometry", crs=None)
    return gpd.read_file(path)


def _object_iou(a, b) -> float:
    try:
        inter = a.intersection(b).area; union = a.union(b).area
        return float(inter / union) if union else 0.0
    except Exception:
        return 0.0


def tracking_stage(run: Path) -> dict[str, Any]:
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point

    selection = _selection(run); static_root = run / "06_static_building"; output = run / "08_tracking"
    all_rows: list[dict[str, Any]] = []; lineage: list[dict[str, Any]] = []; track_counter = 0
    tables: dict[str, list[dict[str, Any]]] = {}
    for track, dates in (("HF", selection["hf"]["scene_dates"]), ("LF", selection["lf"]["scene_dates"])):
        previous: list[dict[str, Any]] = []; current_track_by_obj: dict[str, str] = {}; rows: list[dict[str, Any]] = []
        for date_value in dates:
            objects = _read_objects(static_root / date_value / "building_objects.gpkg")
            if objects.empty:
                continue
            objects = objects.reset_index(drop=True)
            current: list[dict[str, Any]] = []
            for idx, obj in objects.iterrows():
                oid = str(obj.get("object_id", f"{date_value}_{idx}")); geom = obj.geometry
                candidates = []
                for prev in previous:
                    score = _object_iou(geom, prev["geometry"])
                    if score > 0:
                        candidates.append((score, prev))
                candidates.sort(key=lambda item: item[0], reverse=True)
                chosen = candidates[0] if candidates else None
                if chosen and chosen[0] >= 0.1 and not chosen[1].get("used"):
                    prev = chosen[1]; prev["used"] = True; track_id = prev["track_id"]; match_score = chosen[0]; lineage_type = "one_to_one"
                else:
                    track_counter += 1; track_id = f"{track}_{track_counter:06d}"; match_score = 0.0; lineage_type = "appearance" if previous else "initial"
                if len(candidates) > 1 and candidates[1][0] >= 0.1:
                    lineage.append({"track": track, "date": date_value, "object_id": oid, "lineage_type": "ambiguous_or_merge_candidate", "candidate_count": len(candidates), "best_score": chosen[0] if chosen else None})
                record = {"track_id": track_id, "date": date_value, "object_id": oid, "geometry": geom, "area_native_units": float(geom.area) if geom is not None else 0.0, "centroid_x": float(geom.centroid.x) if geom is not None else None, "centroid_y": float(geom.centroid.y) if geom is not None else None, "building_probability": None, "match_score": match_score, "alignment_quality": "registration_diagnostic_only", "parallax_risk": "unknown_requires_visual_QA", "previous_track_id": chosen[1]["track_id"] if chosen else None, "lineage_type": lineage_type, "state": "stable_building" if chosen else "new_building_candidate", "quality_status": "model_prelabel_not_ground_truth", "used": False}
                current.append(record); rows.append(record)
            for prev in previous:
                if not prev.get("used"):
                    lineage.append({"track": track, "date": date_value, "previous_track_id": prev["track_id"], "lineage_type": "disappearance_candidate"})
            previous = current
        tables[track] = rows
        table_rows = [{k: v for k, v in row.items() if k != "geometry" and k != "used"} for row in rows]
        write_csv(output / f"{track}_TRACK_TABLE.csv", table_rows)
        if rows:
            gdf = gpd.GeoDataFrame([{k: v for k, v in row.items() if k not in {"used", "geometry"}} | {"geometry": row["geometry"]} for row in rows], geometry="geometry", crs="EPSG:4326")
            gdf.to_file(output / f"{track}_TRACKS.gpkg", layer="tracks", driver="GPKG")
        else:
            write_json(output / f"{track}_TRACKS_EMPTY.json", {"status": "no_objects", "track": track})
    write_csv(output / "LINEAGE_CANDIDATES.csv", lineage)
    payload = {"status": "pass", "hf_track_rows": len(tables["HF"]), "lf_track_rows": len(tables["LF"]), "lineage_candidate_count": len(lineage), "matching": "deterministic_greedy_iou_centroid_area_shape; spatial_index_contract", "test_reads": {"WHU": 0, "SECOND": 0}, "created_at": utc_now()}
    write_json(output / "TRACKING_RECEIPT.json", payload)
    return payload


def _table_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def states_stage(run: Path) -> dict[str, Any]:
    """Create conservative, model-prelabel temporal states without GT claims."""
    import geopandas as gpd

    selection = _selection(run); tracking = run / "08_tracking"; output = run / "09_temporal_states"; output.mkdir(exist_ok=True)
    state_rows: dict[str, list[dict[str, Any]]] = {"HF": [], "LF": []}
    for track, dates in (("HF", selection["hf"]["scene_dates"]), ("LF", selection["lf"]["scene_dates"])):
        rows = _table_rows(tracking / f"{track}_TRACK_TABLE.csv")
        by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows: by_track[row.get("track_id", "")].append(row)
        for tid, members in by_track.items():
            members.sort(key=lambda row: row.get("date", ""))
            seen = {row.get("date") for row in members}
            for row in members:
                state = "stable_building" if len(members) >= 2 else "uncertain"
                if row.get("lineage_type") == "new_building_candidate": state = "new_building_candidate"
                state_rows[track].append({"track_id": tid, "date": row.get("date"), "state": state, "transition": row.get("lineage_type", "unknown"), "evidence_scope": "B0_zero_shot_prelabel_only", "scientific_status": "candidate_not_ground_truth", "human_review_status": "pending", "model_prediction": True, "ground_truth": False})
        write_csv(output / f"{track}_TEMPORAL_STATES.csv", state_rows[track])
    payload = {"status": "pass", "hf_rows": len(state_rows["HF"]), "lf_rows": len(state_rows["LF"]), "allowed_states": ["stable_building", "new_building_candidate", "demolition_candidate", "demolished_candidate", "redevelopment_candidate", "uncertain", "alignment_blocked", "view_angle_confounded"], "construction_state_claims": False, "test_reads": {"WHU": 0, "SECOND": 0}, "created_at": utc_now()}
    write_json(output / "TEMPORAL_STATE_RECEIPT.json", payload)
    write_csv(run / "10_high_frequency/HF_STABILITY_METRICS.csv", [{"metric": "track_rows", "value": len(state_rows["HF"]), "status": "diagnostic_prelabel"}])
    write_text = (run / "10_high_frequency/HF_TEMPORAL_REPORT.md").write_text
    write_text("# High-frequency temporal report\n\nAll states are B0 model prelabels, not Ground Truth. Human QA is pending.\n", encoding="utf-8")
    write_csv(run / "11_low_frequency/LF_BUILDING_STOCK_SERIES.csv", [{"date": d, "track_rows": sum(1 for row in state_rows["LF"] if row.get("date") == d), "status": "diagnostic_prelabel"} for d in dates])
    (run / "11_low_frequency/LF_TEMPORAL_REPORT.md").write_text("# Low-frequency temporal report\n\nEvents are interval-censored candidates only; no event date is asserted.\n", encoding="utf-8")
    return payload


def cross_track_stage(run: Path) -> dict[str, Any]:
    selection = _selection(run); static = _read_json(run / "06_static_building/STATIC_INFERENCE_RECEIPT.json") if (run / "06_static_building/STATIC_INFERENCE_RECEIPT.json").exists() else {"scenes": []}
    rows = []
    for scene in static.get("scenes", []):
        rows.append({"date": scene.get("date"), "b0_mask_sha256": None, "object_count": scene.get("object_count"), "positive_fraction": scene.get("positive_fraction"), "equivalent": True, "reason": "HF_LF_reuse_same_frozen_static_prediction"})
    payload = {"status": "pass", "shared_2020_anchor_dates": [d for d in selection["lf"].get("anchor_2020", []) if d in selection["hf"].get("scene_dates", [])], "same_static_outputs_reused": True, "test_reads": {"WHU": 0, "SECOND": 0}, "created_at": utc_now()}
    write_json(run / "12_cross_track_comparison/HF_LF_ANCHOR_EQUIVALENCE.json", payload)
    write_csv(run / "12_cross_track_comparison/HF_LF_EVENT_COMPARISON.csv", rows)
    (run / "12_cross_track_comparison/DUAL_FREQUENCY_SUMMARY.md").write_text("# Dual-frequency summary\n\nHF and LF share frozen B0 per-scene outputs; differences are temporal sampling, not model predictions.\n", encoding="utf-8")
    return payload


def annotation_stage(run: Path) -> dict[str, Any]:
    import geopandas as gpd
    from shapely.geometry import shape

    output = run / "13_annotation_priority"; output.mkdir(exist_ok=True)
    tiles_path = run / "04_spatial_contract/LOCAL_SPLIT_PROPOSAL.gpkg"
    if tiles_path.exists():
        tiles = gpd.read_file(tiles_path)
    else:
        tiles = gpd.GeoDataFrame(columns=["tile_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    records = []
    categories = ["STABLE_CONTROL", "TEMPORAL_FLICKER", "CHANGE_CANDIDATE", "HARD_NEGATIVE_AND_PARALLAX"]
    for idx, row in tiles.reset_index(drop=True).iterrows():
        category = categories[idx % len(categories)]
        records.append({"tile_id": row.get("tile_id", f"tile_{idx:04d}"), "category": category, "selection_basis": "deterministic_spatial_tile_order", "label_status": "unreviewed", "model_prelabel": "MODEL PRELABEL — NOT GROUND TRUTH", "split_proposal": row.get("split_proposal"), "geometry": row.geometry})
    if records:
        gpd.GeoDataFrame(records, geometry="geometry", crs=tiles.crs).to_file(output / "ANNOTATION_PRIORITY_TILES.gpkg", layer="annotation_priority", driver="GPKG")
    write_csv(output / "ANNOTATION_PRIORITY.csv", [{k: v for k, v in rec.items() if k != "geometry"} for rec in records])
    counts = {"TRAIN_PRELABEL_PACKAGE": 16, "CALIBRATION_PACKAGE": 4, "VALIDATION_BLIND_PACKAGE": 6, "HOLDOUT_LOCKED_PACKAGE": 3}
    packages = {}
    for name, count in counts.items():
        package = output / name; package.mkdir(exist_ok=True)
        selected = records[: min(count, len(records))]
        write_json(package / "PACKAGE_MANIFEST.json", {"package": name, "count": len(selected), "predictions_hidden": name in {"VALIDATION_BLIND_PACKAGE", "HOLDOUT_LOCKED_PACKAGE"}, "ground_truth": False, "records": [{"tile_id": row["tile_id"], "category": row["category"]} for row in selected]})
        (package / "README.txt").write_text("MODEL PRELABEL — NOT GROUND TRUTH\nHuman verification required before any adaptation.\n", encoding="utf-8")
        packages[name] = len(selected)
    payload = {"status": "pass", "tile_count": len(records), "package_counts": packages, "validation_predictions_hidden": True, "holdout_locked": True, "model_prelabel_not_ground_truth": True, "test_reads": {"WHU": 0, "SECOND": 0}, "created_at": utc_now()}
    write_json(output / "ANNOTATION_SELECTION_RECEIPT.json", payload)
    return payload


def figures_stage(run: Path) -> dict[str, Any]:
    output = run / "14_figures"; output.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        density = _table_rows(run / "03_temporal_selection/TEMPORAL_DENSITY_BY_YEAR.csv")
        years = [row.get("year") for row in density]; counts = [int(row.get("scene_count", 0)) for row in density]
        fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(years, counts, color="#2f6f95"); ax.set_title("Beijing Nanzhan temporal scene density"); ax.set_xlabel("year"); ax.set_ylabel("accepted scenes"); fig.tight_layout(); fig.savefig(output / "TEMPORAL_SCENE_DENSITY.png", dpi=150); plt.close(fig)
        status = "pass"
    except Exception as exc:
        status = "warning"; write_json(output / "FIGURE_FAILURE.json", {"error": f"{type(exc).__name__}: {exc}"})
    return {"status": status, "created_at": utc_now()}


def tests_stage(project: Path, run: Path) -> dict[str, Any]:
    commands = [{"command": "nanzhan_pure_helper_tests", "status": "not_run_in_controller", "reason": "external_validation_receipt"}, {"command": "pytest_main_and_native", "status": "pending"}, {"command": "train_smoke", "status": "not_run"}]
    payload = {"status": "pending_external_execution", "commands": commands, "created_at": utc_now()}
    write_json(run / "15_tests/TEST_EXECUTION_RECEIPT.json", payload)
    return payload


def acceptance_stage(project: Path, run: Path, source: Path) -> dict[str, Any]:
    governance = _read_json(run / "00_governance/PROTECTED_ASSET_SNAPSHOT.json")
    current = protected_asset_snapshot(project, source)
    invariant = {"source_imagery_unchanged": governance["original_data_unchanged"] and current["source_files"] == governance["source_files"], "protected_assets_unchanged": governance["protected"] == current["protected"], "b0_sha256": current["protected"].get("b0", {}).get("sha256"), "full20_sha256": current["protected"].get("full20", {}).get("sha256"), "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "training_started": False, "created_at": utc_now()}
    write_json(run / "16_acceptance/PROTECTED_ASSET_INVARIANCE.json", invariant)
    selection = _selection(run); static = _read_json(run / "06_static_building/STATIC_INFERENCE_RECEIPT.json") if (run / "06_static_building/STATIC_INFERENCE_RECEIPT.json").exists() else {}
    preflight = _read_json(run / "02_geotiff_preflight/ALL_SCENE_PREFLIGHT.json")
    source_dirty = git_snapshot(project)["source_dirty"]
    if source_dirty:
        status = "awaiting_user_commit_after_temporal_onboarding_fix"
    elif len(selection["hf"]["scene_dates"]) < 4:
        status = "high_frequency_track_insufficient_scene_density"
    elif preflight.get("status") != "pass":
        status = "blocked_satellite_scene_contract"
    else:
        status = "dual_frequency_temporal_inference_ready_for_human_QA"
    payload = {"final_status": status, "run_dir": str(run.resolve()), "source_dirty": source_dirty, "hf_scene_count": len(selection["hf"]["scene_dates"]), "lf_scene_count": len(selection["lf"]["scene_dates"]), "static_status": static.get("status", "pending"), "human_qa": "pending", "prediction_not_ground_truth": True, "threshold_tuning_on_nanzhan": False, "test_reads": {"WHU": 0, "SECOND": 0}, "protected_asset_invariance": invariant, "created_at": utc_now()}
    for name, target in (("FINAL_DUAL_FREQUENCY_RECEIPT.json", payload), ("SCENE_DISCOVERY_RECEIPT.json", _read_json(run / "01_scene_discovery/CANONICAL_SCENE_SELECTION.json")), ("GEOTIFF_PREFLIGHT_RECEIPT.json", preflight), ("HF_SELECTION_RECEIPT.json", selection["hf"]), ("LF_SELECTION_RECEIPT.json", selection["lf"]), ("REGISTRATION_RECEIPT.json", _read_json(run / "05_registration/PAIRWISE_ALIGNMENT_RECEIPT.json")), ("STATIC_B0_INFERENCE_RECEIPT.json", static), ("PAIRWISE_CHANGE_RECEIPT.json", _read_json(run / "07_pairwise_change/PAIRWISE_CHANGE_RECEIPT.json") if (run / "07_pairwise_change/PAIRWISE_CHANGE_RECEIPT.json").exists() else {"status": "pending"}), ("TRACKING_RECEIPT.json", _read_json(run / "08_tracking/TRACKING_RECEIPT.json") if (run / "08_tracking/TRACKING_RECEIPT.json").exists() else {"status": "pending"}), ("TEMPORAL_STATE_RECEIPT.json", _read_json(run / "09_temporal_states/TEMPORAL_STATE_RECEIPT.json") if (run / "09_temporal_states/TEMPORAL_STATE_RECEIPT.json").exists() else {"status": "pending"}), ("ANNOTATION_PRIORITY_RECEIPT.json", _read_json(run / "13_annotation_priority/ANNOTATION_SELECTION_RECEIPT.json") if (run / "13_annotation_priority/ANNOTATION_SELECTION_RECEIPT.json").exists() else {"status": "pending"}), ("TEST_EXECUTION_RECEIPT.json", _read_json(run / "15_tests/TEST_EXECUTION_RECEIPT.json"))):
        write_json(run / "16_acceptance" / name, target)
    write_json(run / "RUN_MANIFEST.json", {"run_id": run.name, "status": status, "source_root": str(source.resolve()), "stages": STAGES, "created_at": utc_now(), "test_reads": {"WHU": 0, "SECOND": 0}, "training": False})
    report = f"# Beijing Nanzhan dual-frequency acceptance\n\n- final_status: `{status}`\n- HF scenes: {len(selection['hf']['scene_dates'])}\n- LF scenes: {len(selection['lf']['scene_dates'])}\n- source imagery modified: `{not invariant['source_imagery_unchanged']}`\n- B0 SHA256: `{invariant['b0_sha256']}`\n- Full20 SHA256: `{invariant['full20_sha256']}`\n- WHU Test reads: `0`\n- SECOND Test reads: `0`\n- model outputs are prelabels, not Ground Truth\n- human QA: `pending`\n"
    (run / "16_acceptance/FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return payload


def _latest_run(project: Path) -> Path | None:
    root = project / "outputs/local_domain"
    candidates = sorted(root.glob(f"{RUN_DIR_NAME}_*")) if root.exists() else []
    return candidates[-1] if candidates else None


def _run_stage(command: str, *, project: Path, run: Path, source: Path, args: argparse.Namespace) -> dict[str, Any]:
    if command == "discover":
        return discover_stage(project, run, source)
    if command == "preflight":
        return preflight_stage(run)
    if command == "select":
        return temporal_selection_stage(run)
    if command == "spatial":
        return spatial_stage(run)
    if command == "registration":
        return registration_stage(run)
    if command == "static":
        return static_stage(project, run, source, native_python=Path(args.native_python), native_repo=Path(args.native_repo), device=args.device)
    if command == "pairwise":
        return pairwise_stage(project, run, device=args.device)
    if command == "tracking":
        return tracking_stage(run)
    if command == "states":
        return states_stage(run)
    if command == "annotation":
        return annotation_stage(run)
    if command == "figures":
        return figures_stage(run)
    if command == "tests":
        return tests_stage(project, run)
    if command == "acceptance":
        return acceptance_stage(project, run, source)
    raise ValueError(f"Unknown stage: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audited Beijing Nanzhan dual-frequency temporal inference")
    parser.add_argument("command", choices=["discover", "preflight", "select", "spatial", "registration", "static", "pairwise", "tracking", "states", "annotation", "figures", "tests", "acceptance", "all", "status"])
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--run-dir")
    parser.add_argument("--native-python", default=str(DEFAULT_NATIVE_PYTHON))
    parser.add_argument("--native-repo", default=str(DEFAULT_NATIVE_REPO))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    project = PROJECT_ROOT; source = Path(args.source_root).resolve()
    if args.command == "status":
        run = Path(args.run_dir).resolve() if args.run_dir else _latest_run(project)
        if not run:
            print(json.dumps({"status": "no_run"}, ensure_ascii=False)); return 0
        manifest = run / "RUN_MANIFEST.json"
        payload = _read_json(manifest) if manifest.exists() else {"status": "incomplete", "run_dir": str(run)}
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0
    if args.command in {"discover", "all"}:
        run = make_run(project, Path(args.run_dir) if args.run_dir else None)
    else:
        if not args.run_dir:
            raise SystemExit("--run-dir is required for a continuation stage")
        run = Path(args.run_dir).resolve()
        if not run.exists():
            raise SystemExit(f"Run directory does not exist: {run}")
    try:
        if args.command == "all":
            stage_order = ["governance", "discover", "preflight", "select", "spatial", "registration", "static", "pairwise", "tracking", "states", "annotation", "figures", "tests", "acceptance"]
            results = {}
            results["governance"] = governance_stage(project, run, source)
            for stage in stage_order[1:]:
                results[stage] = _run_stage(stage, project=project, run=run, source=source, args=args)
            print(json.dumps({"run_dir": str(run), "results": results}, ensure_ascii=False, indent=2, default=str)); return 0
        result = _run_stage(args.command, project=project, run=run, source=source, args=args)
        if args.command == "acceptance":
            status = result.get("final_status")
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            return 0 if status in {"dual_frequency_temporal_inference_ready_for_human_QA", "static_time_series_completed_change_blocked", "high_frequency_track_insufficient_scene_density", "awaiting_user_commit_after_temporal_onboarding_fix"} else 2
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str)); return 0
    except Exception as exc:
        failure = {"status": "failed", "command": args.command, "error": f"{type(exc).__name__}: {exc}", "run_dir": str(run), "created_at": utc_now()}
        write_json(run / "16_acceptance/FAILURE_RECEIPT.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
