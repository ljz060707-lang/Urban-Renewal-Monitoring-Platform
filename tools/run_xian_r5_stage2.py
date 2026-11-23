#!/usr/bin/env python3
"""Build isolated Stage-2 actual-date evidence for the Xi'an R5 candidate.

The runner reads the R4 candidate and imagery as evidence, writes only beneath
F:\\xian\\XIAN_R5_MODEL_UPGRADE_20260923, and keeps ambiguous/missing inputs
explicit. It does not promote geometry or change any formal product.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUTPUT_ROOT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923")
STAGE2_ROOT = OUTPUT_ROOT / "17_stage2"
MAIN_ROOT = Path(r"F:\xian\xian")
SUPPLEMENT_ROOT = Path(r"F:\xian\XIAN补充")
DATE_AUDIT_ROOT = MAIN_ROOT / "date_audit"
R4_ROOT = Path(r"F:\xian\西安_当前最佳候选交付_20260917")
R4_CLOUD = R4_ROOT / "08_cloud_v2_candidate"
R4_GEOMETRY_RECEIPT = R4_ROOT / "00_handoff" / "GEOMETRY_SOURCE_PATHS.json"
R4_FIXED_CASES = R4_ROOT / "05_visual_qa_reference" / "FIXED_CASE_VISUAL_SCREEN.csv"
MAIN_TILE_DATES = DATE_AUDIT_ROOT / "tables" / "tile_actual_date.csv"
SUPPLEMENT_TILE_DATES = OUTPUT_ROOT / "00_audit" / "supplement_registry" / "supplement_tile_date_evidence.csv"
SAMPLE_CASE_IDS = [
    "B02-P2-24", "B03-TEX04", "B04-TQA07", "B04-TQA10", "B05-TQA03",
    "B05-P1-10", "B01-P2-26", "OBJ_0001", "OBJ_0008", "B05-TEX01",
    "B01-TEX02", "B05-P1-06",
]
SCENE_TO_EPOCH = {"xa180429": "2018", "xa190101": "2019", "xa201231": "2020"}
EPOCH_TO_SCENE = {value: key for key, value in SCENE_TO_EPOCH.items()}
EPOCH_INTERVALS = {
    "2018": ("2018_2019",),
    "2019": ("2018_2019", "2019_2020"),
    "2020": ("2019_2020",),
}
GRID_ORIGIN_X = 420534
GRID_ORIGIN_Y = 311895
GRID_ORIGIN_LON = 108.75778198242188
GRID_ORIGIN_LAT = 34.16130065917969
GRID_STEP_DEG = 0.0006866455078125


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_progress(text: str) -> None:
    path = STAGE2_ROOT / "13_receipts" / "RUN_PROGRESS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = path.read_text(encoding="utf-8") if path.is_file() else "# R5 Stage-2 progress\n"
    path.write_text(prior.rstrip() + f"\n- {utc_now()} — {text}\n", encoding="utf-8")


def update_status(**updates: Any) -> dict[str, Any]:
    path = STAGE2_ROOT / "13_receipts" / "RUN_STATUS.json"
    if path.is_file():
        status = json.loads(path.read_text(encoding="utf-8"))
    else:
        status = {
            "schema_version": "xian_r5.stage2.run_status.v1",
            "phase": "STAGE2_INITIALIZING",
            "actual_date_pair": None,
            "rbf_checkpoint_status": "NOT_CHECKED",
            "rbf_repro_status": "NOT_RUN",
            "stable_ground_n": 0,
            "parallax_n": 0,
            "sift_metrics": "NOT_RUN",
            "lightglue_metrics": "NOT_RUN",
            "fixed_cases_done": 0,
            "fixed_cases_pass": 0,
            "tbia_ab_cases": 0,
            "neighbor_switch_r4": None,
            "neighbor_switch_r5": None,
            "merge_cases_r4": None,
            "merge_cases_r5": None,
            "false_change_r4": None,
            "false_change_r5": None,
            "geometry_regression": "NOT_RUN",
            "current_gate": "INITIALIZING",
            "blocker": None,
            "next_action": "FREEZE_R4_AND_BUILD_ACTUAL_DATE_TILE_MAP",
        }
    status.update(updates)
    status["updated_at_utc"] = utc_now()
    status["active_process"] = os.getpid()
    write_json(path, status)
    return status


def _ensure_output_dirs() -> None:
    for name in (
        "00_date_harmonization", "01_checkpoint", "02_rbf_reproduction", "03_fixed_cases",
        "04_stable_ground", "05_parallax", "06_lightglue_ab", "07_tbia_ab",
        "08_merge_resolver", "09_false_change", "10_geometry_regression", "11_visual_qa",
        "12_reports", "13_receipts",
    ):
        (STAGE2_ROOT / name).mkdir(parents=True, exist_ok=True)


def freeze_r4() -> dict[str, Any]:
    """Hash the R4 candidate inputs before Stage-2 reads them."""
    receipt = json.loads(R4_GEOMETRY_RECEIPT.read_text(encoding="utf-8"))
    candidates = {
        "run_status": R4_CLOUD / "RUN_STATUS.json",
        "geometry_source_receipt": R4_GEOMETRY_RECEIPT,
        "fixed_case_visual_screen": R4_FIXED_CASES,
        "tracking_objects": R4_CLOUD / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv",
        "physical_tracks": R4_CLOUD / "FINAL_CANDIDATE_PHYSICAL_TRACKS_V2.csv",
        "presence": R4_CLOUD / "PRESENCE_V2_FROM_SCRATCH.csv",
        "state": R4_CLOUD / "STATE_V2_FROM_SCRATCH.csv",
    }
    for item in receipt.get("epochs", []):
        epoch = str(item.get("epoch"))
        candidates[f"geometry_{epoch}"] = Path(str(item.get("absolute_path") or ""))
    files: dict[str, Any] = {}
    for name, path in candidates.items():
        if not path.is_file():
            files[name] = {"path": str(path), "status": "MISSING"}
            continue
        stat = path.stat()
        files[name] = {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
            "status": "HASHED_READ_ONLY",
        }
    payload = {
        "schema_version": "xian_r5.stage2.r4_freeze.v1",
        "generated_at_utc": utc_now(),
        "candidate_root": str(R4_ROOT),
        "files": files,
        "declared_geometry_sha256": {str(item.get("epoch")): item.get("sha256") for item in receipt.get("epochs", [])},
        "r4_status": json.loads((R4_CLOUD / "RUN_STATUS.json").read_text(encoding="utf-8")),
        "production_modified": False,
        "formal_geometry_modified": False,
    }
    write_json(STAGE2_ROOT / "13_receipts" / "R4_FREEZE_STAGE2.json", payload)
    return payload


def _date_stratum(epoch: str, actual_date: str) -> str:
    explicit = {
        ("2018", "2018-04-29"): "2018_A",
        ("2019", "2018-11-01"): "2019_A",
        ("2019", "2018-11-08"): "2019_B",
        ("2020", "2019-12-10"): "2020_A",
    }
    return explicit.get((epoch, actual_date), f"{epoch}_OTHER_{actual_date}")


def _tile_bbox(tile_x: int, tile_y: int) -> dict[str, float]:
    xmin = GRID_ORIGIN_LON + (tile_x - GRID_ORIGIN_X) * GRID_STEP_DEG
    ymin = GRID_ORIGIN_LAT + (tile_y - GRID_ORIGIN_Y) * GRID_STEP_DEG
    return {"xmin": xmin, "ymin": ymin, "xmax": xmin + GRID_STEP_DEG, "ymax": ymin + GRID_STEP_DEG}


def _tile_area_km2_by_y(pyproj_geod: Any) -> dict[int, float]:
    rows: dict[int, float] = {}
    for tile_y in range(311850, 312350):
        bbox = _tile_bbox(GRID_ORIGIN_X, tile_y)
        area, _ = pyproj_geod.polygon_area_perimeter(
            [bbox["xmin"], bbox["xmax"], bbox["xmax"], bbox["xmin"]],
            [bbox["ymin"], bbox["ymin"], bbox["ymax"], bbox["ymax"]],
        )
        rows[tile_y] = abs(float(area)) / 1_000_000.0
    return rows


def build_actual_date_tile_map() -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyproj import Geod

    from xian_r5.alignment.ground_registration import index_tiles

    output_dir = STAGE2_ROOT / "00_date_harmonization"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ACTUAL_DATE_TILE_MAP.parquet"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage-2 date map: {output_path}")
    alias_path = OUTPUT_ROOT / "02_alignment" / "ACTUAL_DATE_TILE_MAP.parquet"
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    scene_index: dict[str, dict[str, Path]] = {}
    for scene in SCENE_TO_EPOCH:
        scene_index[SCENE_TO_EPOCH[scene]] = index_tiles(MAIN_ROOT / scene / "20")
    geod = Geod(ellps="WGS84")
    tile_area_by_y = _tile_area_km2_by_y(geod)

    bbox_type = pa.struct([
        pa.field("xmin", pa.float64()), pa.field("ymin", pa.float64()),
        pa.field("xmax", pa.float64()), pa.field("ymax", pa.float64()),
    ])
    schema = pa.schema([
        pa.field("tile_id", pa.string()),
        pa.field("epoch_label", pa.string()),
        pa.field("actual_date", pa.string()),
        pa.field("source_scene", pa.string()),
        pa.field("bbox", bbox_type),
        pa.field("pixel_available", pa.bool_()),
        pa.field("duplicate_group", pa.string()),
        pa.field("canonical_source", pa.string()),
        pa.field("date_confidence", pa.string()),
        pa.field("source_kind", pa.string()),
        pa.field("tile_x", pa.int32()),
        pa.field("tile_y", pa.int32()),
        pa.field("bbox_crs", pa.string()),
        pa.field("pixel_path", pa.string()),
    ])
    writer = pq.ParquetWriter(output_path, schema, compression="zstd", use_dictionary=True)
    batch: list[dict[str, Any]] = []
    main_date_tile_map: dict[str, dict[tuple[int, int], str]] = {epoch: {} for epoch in EPOCH_TO_SCENE}
    main_pixel_keys: dict[str, set[tuple[int, int]]] = {epoch: set() for epoch in EPOCH_TO_SCENE}
    counters: Counter[str] = Counter()
    date_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {
        "tile_count": 0, "pixel_tile_count": 0, "expected_area_km2": 0.0, "coverage_area_km2": 0.0,
    })
    input_hashes = {"main_tile_date_csv": sha256_file(MAIN_TILE_DATES), "supplement_tile_date_csv": sha256_file(SUPPLEMENT_TILE_DATES)}

    def flush() -> None:
        if not batch:
            return
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        batch.clear()

    with MAIN_TILE_DATES.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_num, row in enumerate(csv.DictReader(handle), 1):
            scene = row.get("task_name", "")
            epoch = SCENE_TO_EPOCH.get(scene)
            if not epoch:
                counters["main_unknown_scene_rows"] += 1
                continue
            tile_x, tile_y = int(row["tile_x"]), int(row["tile_y"])
            actual_date = row.get("actual_capture_date") or "DATE_UNKNOWN"
            key = (tile_x, tile_y)
            tile_path = scene_index[epoch].get(f"{tile_x}_{tile_y}")
            pixel_available = tile_path is not None
            bbox = _tile_bbox(tile_x, tile_y)
            area_km2 = tile_area_by_y.get(tile_y, 0.0)
            stat = date_stats[(epoch, actual_date)]
            stat["tile_count"] += 1
            stat["expected_area_km2"] += area_km2
            if pixel_available:
                stat["pixel_tile_count"] += 1
                stat["coverage_area_km2"] += area_km2
                main_pixel_keys[epoch].add(key)
            main_date_tile_map[epoch][key] = actual_date
            source = str(tile_path) if tile_path else "NO_CURRENT_PIXEL_TILE"
            batch.append({
                "tile_id": f"MAIN_Z20:{tile_x}:{tile_y}",
                "epoch_label": epoch,
                "actual_date": actual_date,
                "source_scene": scene,
                "bbox": bbox,
                "pixel_available": pixel_available,
                "duplicate_group": f"MAIN_Z20:{tile_x}:{tile_y}@{actual_date}" if actual_date != "DATE_UNKNOWN" else None,
                "canonical_source": source,
                "date_confidence": "VERIFIED_LOG_TILE" if row.get("date_source") == "log" and actual_date != "DATE_UNKNOWN" else "UNRESOLVED_NO_DATE_LOG",
                "source_kind": "FORMAL_MAIN_IMAGE_TILE",
                "tile_x": tile_x,
                "tile_y": tile_y,
                "bbox_crs": "EPSG:4326",
                "pixel_path": str(tile_path) if tile_path else None,
            })
            counters["main_rows"] += 1
            counters["main_pixel_rows"] += int(pixel_available)
            counters["main_date_unknown_rows"] += int(actual_date == "DATE_UNKNOWN")
            if len(batch) >= 30_000:
                flush()
                if row_num % 60_000 == 0:
                    update_status(phase="DATE_HARMONIZATION", current_gate="ACTUAL_DATE_TILE_MAP_RUNNING", blocker=None, next_action="FINISH_TILE_LEVEL_MAP")

    supplementary_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"tile_date_records": 0, "pixel_records": 0, "duplicate_records": 0})
    with SUPPLEMENT_TILE_DATES.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_num, row in enumerate(csv.DictReader(handle), 1):
            date = row.get("actual_acquisition_date") or "DATE_UNKNOWN"
            folder = row.get("folder_label") or row.get("source_asset_id") or "UNKNOWN_SUPPLEMENT_SOURCE"
            if date in {"2018-11-01", "2018-11-08"}:
                epoch_label = "SUPPLEMENT_SUPPORT_2019"
            elif date in {"2019-12-02", "2019-12-10", "2020-02-09", "2020-02-16", "2020-02-21", "2020-04-24", "2020-06-14", "2020-11-14"}:
                epoch_label = "SUPPLEMENT_SUPPORT_2020"
            else:
                epoch_label = "SUPPLEMENTAL_ONLY"
            qkey = row.get("tile_key") or "UNKNOWN_TILE_KEY"
            duplicate = str(row.get("same_acquisition_duplicate", "False")).strip().lower() == "true"
            # This cache key has no verified transform to the formal tile grid.
            # Its date is useful evidence, but bbox and pixel availability stay unknown/false.
            batch.append({
                "tile_id": f"SUPPLEMENT_QK20:{qkey}",
                "epoch_label": epoch_label,
                "actual_date": date,
                "source_scene": folder,
                "bbox": None,
                "pixel_available": False,
                "duplicate_group": f"SUPPLEMENT_QK20:{qkey}@{date}" if duplicate else None,
                "canonical_source": f"DATEVER_ONLY:{row.get('source_asset_id') or folder}",
                "date_confidence": "VERIFIED_PACKED_DATEVER_RECORD" if date != "DATE_UNKNOWN" else "UNRESOLVED_DATEVER_SENTINEL",
                "source_kind": "SUPPLEMENTAL_DATE_METADATA_ONLY",
                "tile_x": None,
                "tile_y": None,
                "bbox_crs": None,
                "pixel_path": None,
            })
            counters["supplement_rows"] += 1
            counters["supplement_duplicate_rows"] += int(duplicate)
            counters["supplement_pixel_rows"] += 0
            stat = supplementary_stats[date]
            stat["tile_date_records"] += 1
            stat["duplicate_records"] += int(duplicate)
            if len(batch) >= 30_000:
                flush()
                if row_num % 150_000 == 0:
                    update_status(phase="DATE_HARMONIZATION", current_gate="SUPPLEMENT_DATEVER_ROWS_RUNNING", blocker=None, next_action="FINISH_TILE_LEVEL_MAP")
    flush()
    writer.close()

    # The Stage-1 root has an explicit requested path. Hard-link the generated
    # map there when possible so both documented Stage-2 and Stage-1 locations
    # resolve to one immutable artifact without a second large copy.
    alias_status = "PREEXISTING_NOT_MODIFIED" if alias_path.exists() else "HARDLINKED"
    if not alias_path.exists():
        try:
            os.link(output_path, alias_path)
        except OSError as exc:
            alias_status = f"COPY_FALLBACK:{type(exc).__name__}"
            import shutil
            shutil.copy2(output_path, alias_path)

    coverage_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for (epoch, actual_date), values in date_stats.items():
        coverage_pairs[(epoch, actual_date)] = dict(values)
    receipt = {
        "schema_version": "xian_r5.actual_date_tile_map.v1",
        "generated_at_utc": utc_now(),
        "output": str(output_path),
        "requested_stage1_alias": str(alias_path),
        "alias_status": alias_status,
        "input_hashes": input_hashes,
        "counts": dict(counters),
        "main_tile_rows_by_epoch_actual_date": [
            {"epoch_label": epoch, "actual_date": date, **values}
            for (epoch, date), values in sorted(coverage_pairs.items())
        ],
        "supplemental_actual_date_records": dict(sorted(supplementary_stats.items())),
        "supplement_spatial_policy": "QUADKEY_BBOX_UNRESOLVED_NO_VERIFIED_TRANSFORM;PIXEL_AVAILABLE_FALSE_FOR_CACHE_DATEVER_ROWS",
        "date_policy": "nominal_epoch_labels_do_not_override_tile_actual_dates",
    }
    write_json(output_dir / "ACTUAL_DATE_TILE_MAP_RECEIPT.json", receipt)
    return {
        "receipt": receipt,
        "main_date_tile_map": main_date_tile_map,
        "main_pixel_keys": main_pixel_keys,
        "date_stats": date_stats,
        "scene_index": scene_index,
        "tile_area_by_y": tile_area_by_y,
    }


def _unknown_tracks_by_interval() -> dict[str, set[str]]:
    path = R4_CLOUD / "STATE_V2_FROM_SCRATCH.csv"
    result: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            state = str(row.get("state", "")).upper()
            if state.startswith("UNKNOWN") or state.startswith("UNRESOLVED"):
                result[row.get("interval", "")].add(row.get("physical_track_id", ""))
    result.pop("", None)
    return result


def load_fixed_case_track_map() -> tuple[dict[str, set[str]], dict[str, dict[str, list[dict[str, Any]]]], dict[str, set[int]], dict[str, set[str]]]:
    cases_by_track: dict[str, set[str]] = defaultdict(set)
    required_tracks: set[str] = set()
    with R4_FIXED_CASES.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = row.get("case_id", "")
            if case_id not in SAMPLE_CASE_IDS:
                continue
            for field in ("track_id", "building_id"):
                for value in str(row.get(field, "")).split(";"):
                    value = value.strip()
                    if value:
                        required_tracks.add(value)
                        cases_by_track[value].add(case_id)
    unknown_intervals = _unknown_tracks_by_interval()
    unknown_tracks_by_epoch: dict[str, set[str]] = {
        "2018": unknown_intervals.get("2018_2019", set()),
        "2019": unknown_intervals.get("2018_2019", set()) | unknown_intervals.get("2019_2020", set()),
        "2020": unknown_intervals.get("2019_2020", set()),
    }
    target_records: dict[str, dict[str, list[dict[str, Any]]]] = {epoch: defaultdict(list) for epoch in EPOCH_TO_SCENE}
    unknown_object_ids: dict[str, set[int]] = {epoch: set() for epoch in EPOCH_TO_SCENE}
    case_object_ids: dict[str, set[int]] = {epoch: set() for epoch in EPOCH_TO_SCENE}
    with (R4_CLOUD / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scene = row.get("scene_key", "")
            epoch = SCENE_TO_EPOCH.get(scene)
            if not epoch:
                continue
            uid = row.get("object_uid", "")
            match = re.search(r"::(\d+)$", uid)
            if not match:
                continue
            object_id = int(match.group(1))
            track_id = row.get("track_id", "")
            physical_track_id = row.get("physical_track_id", "")
            if track_id in required_tracks or row.get("building_id", "") in required_tracks:
                for case_id in cases_by_track.get(track_id, set()) | cases_by_track.get(row.get("building_id", ""), set()):
                    target_records[epoch][object_id].append({
                        "case_id": case_id,
                        "track_id": track_id or row.get("building_id", ""),
                        "physical_track_id": physical_track_id,
                        "object_uid": uid,
                        "scene_key": scene,
                    })
                    case_object_ids[epoch].add(object_id)
            if physical_track_id in unknown_tracks_by_epoch.get(epoch, set()):
                unknown_object_ids[epoch].add(object_id)
    return dict(cases_by_track), target_records, unknown_object_ids, case_object_ids


def _actual_tile_for_lonlat(epoch: str, lon: float, lat: float, tile_date_map: dict[str, dict[tuple[int, int], str]], pixel_keys: dict[str, set[tuple[int, int]]]) -> dict[str, Any]:
    tile_x = math.floor((lon - GRID_ORIGIN_LON) / GRID_STEP_DEG + GRID_ORIGIN_X)
    tile_y = math.floor((lat - GRID_ORIGIN_LAT) / GRID_STEP_DEG + GRID_ORIGIN_Y)
    actual_date = tile_date_map.get(epoch, {}).get((tile_x, tile_y), "NO_TILE_DATE_RECORD")
    pixel = (tile_x, tile_y) in pixel_keys.get(epoch, set())
    return {
        "tile_x": tile_x,
        "tile_y": tile_y,
        "tile_id": f"MAIN_Z20:{tile_x}:{tile_y}",
        "actual_date": actual_date,
        "pixel_available": pixel,
        "bbox": _tile_bbox(tile_x, tile_y),
        "pixel_path": str(next((p for p in []), "")) if False else None,
    }


def _centroid_lonlat(geometry: Any, transformer: Any) -> tuple[float, float]:
    centroid = geometry.centroid
    lon, lat = transformer.transform(centroid.x, centroid.y)
    return float(lon), float(lat)


def _image_bbox_px(geometry: Any, tile_bbox: dict[str, float], transformer: Any) -> list[int] | None:
    from shapely.ops import transform as geom_transform

    geographic = geom_transform(transformer.transform, geometry)
    xmin, ymin, xmax, ymax = geographic.bounds
    width = tile_bbox["xmax"] - tile_bbox["xmin"]
    height = tile_bbox["ymax"] - tile_bbox["ymin"]
    if width <= 0 or height <= 0:
        return None
    px0 = max(0, min(256, int(round((xmin - tile_bbox["xmin"]) / width * 256))))
    px1 = max(0, min(256, int(round((xmax - tile_bbox["xmin"]) / width * 256))))
    py0 = max(0, min(256, int(round((tile_bbox["ymax"] - ymax) / height * 256))))
    py1 = max(0, min(256, int(round((tile_bbox["ymax"] - ymin) / height * 256))))
    return [px0, py0, px1, py1]


def build_date_strata_and_fixed_map(tile_data: dict[str, Any]) -> dict[str, Any]:
    import fiona
    import numpy as np
    from pyproj import Transformer
    from shapely.geometry import mapping, shape

    tile_dates = tile_data["main_date_tile_map"]
    pixel_keys = tile_data["main_pixel_keys"]
    scene_index = tile_data["scene_index"]
    tile_area_by_y = tile_data["tile_area_by_y"]
    _, target_records, unknown_object_ids, case_object_ids = load_fixed_case_track_map()

    geometry_receipt = json.loads(R4_GEOMETRY_RECEIPT.read_text(encoding="utf-8"))
    geometry_paths = {str(item["epoch"]): Path(item["absolute_path"]) for item in geometry_receipt.get("epochs", [])}
    transformers = {epoch: Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True) for epoch in EPOCH_TO_SCENE}
    geometry_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {
        "r4_geometry_object_count": 0, "r4_geometry_area_m2": 0.0,
        "unknown_r4_object_count": 0, "unknown_r4_area_m2": 0.0,
        "fixed_case_observation_count": 0, "fixed_case_pixel_observation_count": 0,
    })
    total_geometry: Counter[str] = Counter()
    mapped_geometry: Counter[str] = Counter()
    target_geometries: dict[str, dict[int, dict[str, Any]]] = {epoch: {} for epoch in EPOCH_TO_SCENE}

    for epoch in ("2018", "2019", "2020"):
        path = geometry_paths[epoch]
        transformer = transformers[epoch]
        tile_map = tile_dates[epoch]
        wanted = target_records[epoch]
        with fiona.open(path, layer="buildings") as source:
            batch: list[tuple[int, float, float, float]] = []
            feature_geometries: dict[int, Any] = {}

            def process_batch() -> None:
                if not batch:
                    return
                ids = np.asarray([item[0] for item in batch], dtype=np.int64)
                xs = np.asarray([item[2] for item in batch], dtype=np.float64)
                ys = np.asarray([item[3] for item in batch], dtype=np.float64)
                lons, lats = transformer.transform(xs, ys)
                for idx, item in enumerate(batch):
                    object_id, area_m2 = item[0], item[1]
                    lon, lat = float(lons[idx]), float(lats[idx])
                    tile_x = math.floor((lon - GRID_ORIGIN_LON) / GRID_STEP_DEG + GRID_ORIGIN_X)
                    tile_y = math.floor((lat - GRID_ORIGIN_LAT) / GRID_STEP_DEG + GRID_ORIGIN_Y)
                    actual_date = tile_map.get((tile_x, tile_y), "NO_TILE_DATE_RECORD")
                    available = (tile_x, tile_y) in pixel_keys[epoch]
                    if actual_date in {"NO_TILE_DATE_RECORD", "DATE_UNKNOWN"}:
                        total_geometry[f"{epoch}:outside_or_unknown"] += 1
                    else:
                        total_geometry[f"{epoch}:date_mapped"] += 1
                    if available and actual_date not in {"NO_TILE_DATE_RECORD", "DATE_UNKNOWN"}:
                        key = (epoch, actual_date)
                        geometry_stats[key]["r4_geometry_object_count"] += 1
                        geometry_stats[key]["r4_geometry_area_m2"] += area_m2
                        mapped_geometry[epoch] += 1
                        if object_id in unknown_object_ids[epoch]:
                            geometry_stats[key]["unknown_r4_object_count"] += 1
                            geometry_stats[key]["unknown_r4_area_m2"] += area_m2
                        if object_id in case_object_ids[epoch]:
                            geometry_stats[key]["fixed_case_observation_count"] += 1
                            geometry_stats[key]["fixed_case_pixel_observation_count"] += 1
                    if object_id in wanted:
                        geom = feature_geometries[object_id]
                        props = {
                            "object_id": object_id,
                            "scene_key": EPOCH_TO_SCENE[epoch],
                            "area_m2": area_m2,
                        }
                        target_geometries[epoch][object_id] = {
                            "geometry": geom,
                            "properties": props,
                            "lon": lon,
                            "lat": lat,
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "actual_date": actual_date,
                            "pixel_available": available,
                            "pixel_path": str(scene_index[epoch].get(f"{tile_x}_{tile_y}")) if available else None,
                        }
                batch.clear()
                feature_geometries.clear()

            for feature in source:
                props = dict(feature.get("properties") or {})
                object_id = int(props.get("object_id"))
                area_m2 = float(props.get("area_m2") or 0.0)
                geom = shape(feature["geometry"]) if feature.get("geometry") else None
                if geom is None or geom.is_empty:
                    continue
                centroid = geom.centroid
                batch.append((object_id, area_m2, float(centroid.x), float(centroid.y)))
                if object_id in wanted:
                    feature_geometries[object_id] = geom
                total_geometry[f"{epoch}:read"] += 1
                if len(batch) >= 40_000:
                    process_batch()
                    update_status(phase="DATE_STRATA_COVERAGE", current_gate="R4_GEOMETRY_CENTROID_COVERAGE_RUNNING", blocker=None, next_action=f"FINISH_{epoch}_GEOMETRY_COVERAGE")
            process_batch()
        append_progress(f"R4 geometry coverage summarized for epoch {epoch}; read={total_geometry[f'{epoch}:read']}, date-mapped pixels={mapped_geometry[epoch]}.")

    coverage_rows: list[dict[str, Any]] = []
    for (epoch, actual_date), stat in sorted(tile_data["date_stats"].items()):
        geometry = geometry_stats.get((epoch, actual_date), {})
        area = float(stat["coverage_area_km2"])
        object_count = int(geometry.get("r4_geometry_object_count", 0))
        unknown_count = int(geometry.get("unknown_r4_object_count", 0))
        total_obj_area = float(geometry.get("r4_geometry_area_m2", 0.0))
        unknown_area = float(geometry.get("unknown_r4_area_m2", 0.0))
        coverage_rows.append({
            "epoch_label": epoch,
            "source_scene": EPOCH_TO_SCENE[epoch],
            "actual_date": actual_date,
            "date_stratum": _date_stratum(epoch, actual_date),
            "nominal_date_used_for_transition": False,
            "tile_count": int(stat["tile_count"]),
            "pixel_tile_count": int(stat["pixel_tile_count"]),
            "expected_tile_area_km2": round(float(stat["expected_area_km2"]), 6),
            "coverage_area_km2": round(area, 6),
            "r4_geometry_centroid_object_count": object_count,
            "r4_geometry_area_m2": round(total_obj_area, 3),
            "residual_unknown_object_count": unknown_count,
            "residual_unknown_area_m2": round(unknown_area, 3),
            "residual_unknown_object_pct": round(100.0 * unknown_count / object_count, 4) if object_count else None,
            "residual_unknown_area_pct": round(100.0 * unknown_area / total_obj_area, 4) if total_obj_area else None,
            "fixed_case_observation_count": int(geometry.get("fixed_case_observation_count", 0)),
            "fixed_case_pixel_observation_count": int(geometry.get("fixed_case_pixel_observation_count", 0)),
            "object_coverage_method": "R4_POLYGON_CENTROID_JOIN_TO_ACTUAL_DATE_TILE_AND_CURRENT_PIXEL_FILE",
            "candidate_geometry_is_ground_truth": False,
            "status": "DATE_AND_PIXEL_MAPPED" if actual_date != "DATE_UNKNOWN" else "DATE_UNKNOWN_RETAINED",
        })
    write_csv_path = STAGE2_ROOT / "00_date_harmonization" / "DATE_STRATA_COVERAGE.csv"
    write_csv(write_csv_path, coverage_rows)

    # Fixed-case target and neighbor geometries. These polygons remain a QA
    # artifact and are stored in an isolated GeoPackage under Stage-2.
    fixed_rows: list[dict[str, Any]] = []
    fixed_gpkg_path = STAGE2_ROOT / "03_fixed_cases" / "R5_FIXED_CASE_GEOMETRY.gpkg"
    fixed_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gpkg_schema = {
        "geometry": "Unknown",
        "properties": {
            "case_id": "str:32", "epoch_label": "str:8", "scene_key": "str:16",
            "object_id": "int", "object_uid": "str:64", "track_id": "str:64",
            "role": "str:16", "actual_date": "str:16", "pixel_available": "int",
            "distance_m": "float",
        },
    }
    if fixed_gpkg_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing fixed-case QA geometry: {fixed_gpkg_path}")
    case_tracks: dict[str, set[str]] = defaultdict(set)
    for track, case_ids in load_fixed_case_track_map()[0].items():
        for case_id in case_ids:
            case_tracks[case_id].add(track)

    with fiona.open(fixed_gpkg_path, "w", driver="GPKG", layer="fixed_case_polygons", schema=gpkg_schema, crs="EPSG:32649") as sink:
        for epoch in ("2018", "2019", "2020"):
            path = geometry_paths[epoch]
            transformer = transformers[epoch]
            scene_targets = target_geometries[epoch]
            case_epoch_objects: dict[tuple[str, str], set[int]] = defaultdict(set)
            for object_id, records in target_records[epoch].items():
                for record in records:
                    case_epoch_objects[(record["case_id"], record["track_id"])].add(object_id)

            for case_id in SAMPLE_CASE_IDS:
                objects_for_case = sorted({object_id for (cid, _), ids in case_epoch_objects.items() if cid == case_id for object_id in ids})
                neighbor_records: list[tuple[int, Any, float, str]] = []
                if objects_for_case:
                    geoms = [scene_targets[obj]["geometry"] for obj in objects_for_case if obj in scene_targets]
                    if geoms:
                        minx = min(g.bounds[0] for g in geoms) - 100.0
                        miny = min(g.bounds[1] for g in geoms) - 100.0
                        maxx = max(g.bounds[2] for g in geoms) + 100.0
                        maxy = max(g.bounds[3] for g in geoms) + 100.0
                        with fiona.open(path, layer="buildings") as source:
                            for feature in source.filter(bbox=(minx, miny, maxx, maxy)):
                                props = dict(feature.get("properties") or {})
                                neighbor_id = int(props.get("object_id"))
                                if neighbor_id in objects_for_case:
                                    continue
                                neighbor_geom = shape(feature["geometry"]) if feature.get("geometry") else None
                                if neighbor_geom is None or neighbor_geom.is_empty:
                                    continue
                                distance = min(neighbor_geom.distance(scene_targets[obj]["geometry"]) for obj in objects_for_case if obj in scene_targets)
                                if distance <= 100.0:
                                    neighbor_records.append((neighbor_id, neighbor_geom, float(distance), str(props.get("area_m2") or "")))
                    neighbor_records.sort(key=lambda item: item[2])
                    neighbor_records = neighbor_records[:20]

                neighbors_by_id = [item[0] for item in neighbor_records]
                for object_id in objects_for_case:
                    target = scene_targets.get(object_id)
                    source_records = [item for item in target_records[epoch].get(object_id, []) if item["case_id"] == case_id]
                    if target is None:
                        for record in source_records:
                            fixed_rows.append({
                                "case_id": case_id, "case_track_id": record["track_id"], "epoch_label": epoch,
                                "source_scene": EPOCH_TO_SCENE[epoch], "object_uid": record["object_uid"],
                                "r4_polygon_object_id": object_id, "r4_polygon_geometry_ref": None,
                                "actual_date": "NO_TILE_DATE_RECORD", "actual_date_tile_id": None,
                                "tile_x": None, "tile_y": None, "image_bbox_px": None,
                                "pixel_available": False, "pixel_tile_path": None,
                                "neighbor_object_ids": ";".join(map(str, neighbors_by_id)),
                                "neighbor_polygons_count": len(neighbor_records),
                                "status": "NO_PIXEL_EVIDENCE",
                            })
                        continue
                    bbox = list(target["geometry"].bounds)
                    image_bbox = _image_bbox_px(target["geometry"], _tile_bbox(target["tile_x"], target["tile_y"]), transformer)
                    for record in source_records:
                        fixed_rows.append({
                            "case_id": case_id,
                            "case_track_id": record["track_id"],
                            "physical_track_id": record["physical_track_id"],
                            "epoch_label": epoch,
                            "source_scene": EPOCH_TO_SCENE[epoch],
                            "object_uid": record["object_uid"],
                            "r4_polygon_object_id": object_id,
                            "r4_polygon_bbox_epsg32649": json.dumps(bbox, separators=(",", ":")),
                            "r4_polygon_geometry_ref": f"{fixed_gpkg_path}#fixed_case_polygons/{case_id}/{epoch}/{object_id}",
                            "actual_date": target["actual_date"],
                            "actual_date_tile_id": f"MAIN_Z20:{target['tile_x']}:{target['tile_y']}",
                            "tile_x": target["tile_x"],
                            "tile_y": target["tile_y"],
                            "image_bbox_px": json.dumps(image_bbox, separators=(",", ":")) if image_bbox else None,
                            "pixel_available": target["pixel_available"],
                            "pixel_tile_path": target["pixel_path"],
                            "neighbor_object_ids": ";".join(map(str, neighbors_by_id)),
                            "neighbor_polygons_count": len(neighbor_records),
                            "status": "PIXEL_AVAILABLE_ACTUAL_DATE_MAPPED" if target["pixel_available"] and target["actual_date"] != "DATE_UNKNOWN" else "NO_PIXEL_EVIDENCE",
                        })
                        sink.write({
                            "geometry": mapping(target["geometry"]),
                            "properties": {
                                "case_id": case_id, "epoch_label": epoch, "scene_key": EPOCH_TO_SCENE[epoch],
                                "object_id": object_id, "object_uid": record["object_uid"],
                                "track_id": record["track_id"], "role": "TARGET",
                                "actual_date": target["actual_date"], "pixel_available": int(target["pixel_available"]),
                                "distance_m": 0.0,
                            },
                        })
                for neighbor_id, neighbor_geom, distance, _ in neighbor_records:
                    sink.write({
                        "geometry": mapping(neighbor_geom),
                        "properties": {
                            "case_id": case_id, "epoch_label": epoch, "scene_key": EPOCH_TO_SCENE[epoch],
                            "object_id": neighbor_id, "object_uid": f"{EPOCH_TO_SCENE[epoch]}::{neighbor_id}",
                            "track_id": None, "role": "NEIGHBOR", "actual_date": None,
                            "pixel_available": 0, "distance_m": distance,
                        },
                    })
                if not objects_for_case:
                    fixed_rows.append({
                        "case_id": case_id, "case_track_id": ";".join(sorted(case_tracks.get(case_id, set()))),
                        "epoch_label": epoch, "source_scene": EPOCH_TO_SCENE[epoch],
                        "object_uid": None, "r4_polygon_object_id": None,
                        "actual_date": "NO_R4_OBJECT_FOR_TRACK_EPOCH", "actual_date_tile_id": None,
                        "tile_x": None, "tile_y": None, "image_bbox_px": None,
                        "pixel_available": False, "pixel_tile_path": None,
                        "neighbor_object_ids": ";".join(map(str, neighbors_by_id)),
                        "neighbor_polygons_count": len(neighbor_records), "status": "NO_PIXEL_EVIDENCE",
                    })

    write_csv(STAGE2_ROOT / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv", fixed_rows)
    write_json(STAGE2_ROOT / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP_RECEIPT.json", {
        "schema_version": "xian_r5.fixed_case_tile_map.v1",
        "generated_at_utc": utc_now(),
        "case_count": len(SAMPLE_CASE_IDS),
        "case_epoch_object_rows": len(fixed_rows),
        "pixel_available_rows": sum(str(row.get("pixel_available")).lower() == "true" for row in fixed_rows),
        "no_pixel_evidence_rows": sum(row.get("status") == "NO_PIXEL_EVIDENCE" for row in fixed_rows),
        "geometry_gpkg": str(fixed_gpkg_path),
        "geometry_gpkg_sha256": sha256_file(fixed_gpkg_path),
        "source_candidate_not_ground_truth": True,
    })

    # Create honest main/supplement strata rows. Supplemental cache dates are
    # date evidence only until a spatial transform and exact pixel tile exist.
    rows = coverage_rows.copy()
    for date, values in tile_data["receipt"].get("supplemental_actual_date_records", {}).items():
        if date == "DATE_UNKNOWN":
            stratum = "SUPPLEMENTAL_DATE_UNKNOWN"
        elif date in {"2018-11-01", "2018-11-08"}:
            stratum = "SUPPLEMENT_SUPPORT_2019"
        elif date in {"2019-12-02", "2019-12-10", "2020-02-09", "2020-02-16", "2020-02-21", "2020-04-24", "2020-06-14", "2020-11-14"}:
            stratum = "SUPPLEMENT_SUPPORT_2020"
        else:
            stratum = "SUPPLEMENTAL_ONLY"
        rows.append({
            "epoch_label": "SUPPLEMENTAL_ONLY", "source_scene": "F_XIAN_SUPPLEMENT",
            "actual_date": date, "date_stratum": stratum, "nominal_date_used_for_transition": False,
            "tile_count": values["tile_date_records"], "pixel_tile_count": 0,
            "expected_tile_area_km2": None, "coverage_area_km2": None,
            "r4_geometry_centroid_object_count": None, "r4_geometry_area_m2": None,
            "residual_unknown_object_count": None, "residual_unknown_area_m2": None,
            "residual_unknown_object_pct": None, "residual_unknown_area_pct": None,
            "fixed_case_observation_count": None, "fixed_case_pixel_observation_count": 0,
            "object_coverage_method": "NOT_SPATIALIZED_QUADKEY_BBOX_UNRESOLVED",
            "candidate_geometry_is_ground_truth": False,
            "status": "DATEVER_METADATA_ONLY_NO_PIXEL_TILE",
        })
    write_csv(STAGE2_ROOT / "00_date_harmonization" / "DATE_STRATA_COVERAGE.csv", rows)

    selected_pairs = select_actual_date_pairs(tile_data, target_geometries, geometry_stats, fixed_rows)
    write_json(STAGE2_ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION.json", selected_pairs)
    write_json(STAGE2_ROOT / "00_date_harmonization" / "R4_OBJECT_DATE_COVERAGE_RECEIPT.json", {
        "generated_at_utc": utc_now(),
        "geometry_files_read": {epoch: str(path) for epoch, path in geometry_paths.items()},
        "geometry_read_counts": dict(total_geometry),
        "geometry_pixel_date_mapped_counts": dict(mapped_geometry),
        "unknown_track_counts_by_interval": {key: len(value) for key, value in _unknown_tracks_by_interval().items()},
        "unknown_object_ids_by_epoch": {key: len(value) for key, value in unknown_object_ids.items()},
        "candidate_not_ground_truth": True,
        "method": "CENTROID_TO_VERIFIED_MAIN_TILE_DATE_AND_CURRENT_PIXEL_TILE;NOT_POLYGON_AREA_INTERSECTION",
    })
    return {
        "coverage_rows": rows,
        "fixed_case_rows": fixed_rows,
        "geometry_stats": geometry_stats,
        "target_geometries": target_geometries,
        "selected_pairs": selected_pairs,
    }


def select_actual_date_pairs(tile_data: dict[str, Any], targets: dict[str, dict[int, dict[str, Any]]], geometry_stats: dict[tuple[str, str], dict[str, float]], fixed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = [
        {"pair_id": "PAIR_A", "source_epoch": "2018", "source_date": "2018-04-29", "target_epoch": "2019", "target_date": "2018-11-01"},
        {"pair_id": "PAIR_B", "source_epoch": "2019", "source_date": "2018-11-01", "target_epoch": "2020", "target_date": "2019-12-10"},
    ]
    results: list[dict[str, Any]] = []
    for pair in definitions:
        left_epoch, right_epoch = pair["source_epoch"], pair["target_epoch"]
        left_date, right_date = pair["source_date"], pair["target_date"]
        left_keys = {key for key, date in tile_data["main_date_tile_map"][left_epoch].items() if date == left_date and key in tile_data["main_pixel_keys"][left_epoch]}
        right_keys = {key for key, date in tile_data["main_date_tile_map"][right_epoch].items() if date == right_date and key in tile_data["main_pixel_keys"][right_epoch]}
        common = left_keys & right_keys
        common_area = sum(tile_data["tile_area_by_y"].get(key[1], 0.0) for key in common)
        source_objs = geometry_stats.get((left_epoch, left_date), {}).get("r4_geometry_object_count", 0)
        target_objs = geometry_stats.get((right_epoch, right_date), {}).get("r4_geometry_object_count", 0)
        case_source = sum(row.get("actual_date") == left_date and row.get("epoch_label") == left_epoch and str(row.get("pixel_available")).lower() == "true" for row in fixed_rows)
        case_target = sum(row.get("actual_date") == right_date and row.get("epoch_label") == right_epoch and str(row.get("pixel_available")).lower() == "true" for row in fixed_rows)
        results.append({
            **pair,
            "actual_interval_days": (datetime.fromisoformat(right_date) - datetime.fromisoformat(left_date)).days,
            "source_date_tile_count_with_pixel": len(left_keys),
            "target_date_tile_count_with_pixel": len(right_keys),
            "same_spatial_tile_count": len(common),
            "same_spatial_coverage_area_km2": round(common_area, 6),
            "source_r4_object_centroid_count": int(source_objs),
            "target_r4_object_centroid_count": int(target_objs),
            "fixed_case_pixel_object_observations_source": int(case_source),
            "fixed_case_pixel_object_observations_target": int(case_target),
            "date_coherent": True,
            "mosaic_dates_combined": False,
            "status": "ELIGIBLE_FOR_BOUNDED_CANDIDATE_QA" if common else "BLOCKED_NO_COMMON_PIXEL_TILES",
        })
    chosen = next((row for row in results if row["pair_id"] == "PAIR_A"), None)
    return {
        "schema_version": "xian_r5.date_coherent_pair_selection.v1",
        "generated_at_utc": utc_now(),
        "pairs": results,
        "selection_rule": "use exact actual-date tile rows; no nominal epoch date substitution; choose both requested comparisons when they have real common pixels",
        "selected_for_bounded_stage2": [row["pair_id"] for row in results if row["status"] == "ELIGIBLE_FOR_BOUNDED_CANDIDATE_QA"],
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def record_checkpoint_access_blocker() -> dict[str, Any]:
    output_dir = STAGE2_ROOT / "01_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "xian_r5.rbf_sam_checkpoint_registry.v1",
        "generated_at_utc": utc_now(),
        "official_repository": "https://github.com/Xuaner624/RBF-SAM",
        "official_share_url": "https://pan.baidu.com/s/1vC_7cFpUh1JP9uim_QDJhA",
        "official_extraction_code": "r285",
        "official_readme_says_weights_available": True,
        "automatic_access_attempt": {
            "status": "BLOCKED_INTERACTIVE_SHARE_VERIFICATION",
            "http_status": 200,
            "returned_content": "Baidu Netdisk verification form, not a checkpoint file",
            "page_title": "百度网盘 请输入提取码",
            "requires_additional_verification_information": True,
            "checkpoint_downloaded": False,
            "source_was_official": True,
        },
        "checkpoint_path": None,
        "filename": None,
        "size_bytes": None,
        "sha256": None,
        "download_date": None,
        "config_compatibility": "NOT_EVALUATED_NO_CHECKPOINT",
        "status": "BLOCKED_DOWNLOAD_ACCESS",
    }
    write_json(output_dir / "CHECKPOINT_REGISTRY.json", payload)
    blocker = """# RBF-SAM checkpoint download blocker

The official RBF-SAM README says trained weights are available through the linked Baidu Netdisk share and gives extraction code `r285`. The official share URL returned an interactive Netdisk verification page rather than a checkpoint file. It asks for additional verification information, so no checkpoint was retrieved and no third-party mirror was used.

- Status: `BLOCKED_DOWNLOAD_ACCESS`
- Official source: https://github.com/Xuaner624/RBF-SAM
- Share page: https://pan.baidu.com/s/1vC_7cFpUh1JP9uim_QDJhA
- Checkpoint filename, size, SHA-256: unavailable because no file was delivered
- RBF-SAM inference: blocked; no weights or metrics were fabricated

All Stage-2 work that does not require RBF-SAM weights continues independently.
"""
    (output_dir / "RBF_SAM_CHECKPOINT_DOWNLOAD_BLOCKER.md").write_text(blocker, encoding="utf-8")
    return payload


def write_initial_environment_receipt() -> dict[str, Any]:
    output_dir = STAGE2_ROOT / "01_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "xian_r5.rbf_sam_environment_receipt.v1",
        "generated_at_utc": utc_now(),
        "requested_environment": {
            "python": "3.10.x preferred for PyTorch 1.13.1",
            "pytorch": "1.13.1+cu116",
            "cuda": "11.6",
            "mmcv_full": "1.7.0",
            "mmdetection": "2.x / repository-local customized code",
        },
        "existing_stage1_runtime": "Python 3.12 + isolated geospatial site-packages; not RBF-SAM compatible",
        "isolated_environment_root": str(STAGE2_ROOT / "01_checkpoint" / "rbf_sam_env"),
        "installation_status": "PENDING_ISOLATED_PYTHON_DISCOVERY",
        "production_environment_modified": False,
        "checkpoint_status": "BLOCKED_DOWNLOAD_ACCESS",
    }
    write_json(output_dir / "ENVIRONMENT_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    _ensure_output_dirs()
    update_status(phase="R4_FREEZE", current_gate="R4_FREEZE_RUNNING", blocker=None, next_action="HASH_R4_INPUTS")
    append_progress("Stage-2 initialized; R4 freeze hash and actual-date tile map started.")
    freeze = freeze_r4()
    append_progress(f"R4 freeze recorded for {len(freeze['files'])} candidate inputs; production and formal geometry are read-only.")
    checkpoint = record_checkpoint_access_blocker()
    environment = write_initial_environment_receipt()
    update_status(phase="ACTUAL_DATE_TILE_MAP", actual_date_pair=None, rbf_checkpoint_status=checkpoint["status"], rbf_repro_status="BLOCKED_DOWNLOAD_ACCESS", current_gate="ACTUAL_DATE_TILE_MAP_RUNNING", blocker="RBF_SAM_DOWNLOAD_REQUIRES_INTERACTIVE_VERIFICATION", next_action="BUILD_TILE_LEVEL_ACTUAL_DATE_MAP")
    tile_data = build_actual_date_tile_map()
    append_progress(f"Tile-level date map written: main rows={tile_data['receipt']['counts'].get('main_rows')}; supplement datever rows={tile_data['receipt']['counts'].get('supplement_rows')}.")
    update_status(phase="DATE_STRATA_AND_FIXED_CASE_MAPPING", current_gate="R4_GEOMETRY_COVERAGE_RUNNING", blocker="RBF_SAM_DOWNLOAD_REQUIRES_INTERACTIVE_VERIFICATION", next_action="MAP_R4_OBJECTS_AND_FIXED_CASES_TO_ACTUAL_DATE_TILES")
    evidence = build_date_strata_and_fixed_map(tile_data)
    update_status(
        phase="STAGE2_EVIDENCE_PREPARATION_COMPLETE",
        actual_date_pair=[{"source": "2018-04-29", "target": "2018-11-01"}, {"source": "2018-11-01", "target": "2019-12-10"}],
        rbf_checkpoint_status=checkpoint["status"],
        rbf_repro_status="BLOCKED_DOWNLOAD_ACCESS",
        fixed_cases_done=len(evidence["fixed_case_rows"]),
        fixed_cases_pass=0,
        current_gate="METHOD_GATES_NOT_RUN_OR_BLOCKED",
        blocker="RBF_SAM_DOWNLOAD_ACCESS;VISUAL_ALIGNMENT_STRATA_REVIEW_PENDING;IDENTITY_AB_GATE_PENDING",
        next_action="RUN_BOUNDED_ALIGNMENT_STRATA_AND_TBIA_AB_WHERE_EVIDENCE_IS_AVAILABLE",
    )
    append_progress("Actual-date coverage and fixed-case tile mapping completed. RBF-SAM remains blocked on official interactive share verification.")
    final = {
        "schema_version": "xian_r5.stage2.preparation_receipt.v1",
        "generated_at_utc": utc_now(),
        "r4_freeze_receipt": str(STAGE2_ROOT / "13_receipts" / "R4_FREEZE_STAGE2.json"),
        "actual_date_tile_map": tile_data["receipt"],
        "date_pair_selection": evidence["selected_pairs"],
        "fixed_case_rows": len(evidence["fixed_case_rows"]),
        "rbf_checkpoint_status": checkpoint["status"],
        "rbf_environment_initial_status": environment["installation_status"],
        "stage2_promotion": "NOT_READY_METHOD_GATES_NOT_COMPLETE",
        "production_modified": False,
    }
    write_json(STAGE2_ROOT / "13_receipts" / "STAGE2_PREPARATION_RECEIPT.json", final)
    print(json.dumps({"status": "STAGE2_PREPARATION_COMPLETE", "root": str(STAGE2_ROOT), "date_map_rows": tile_data["receipt"]["counts"], "fixed_case_rows": len(evidence["fixed_case_rows"]), "selected_pairs": evidence["selected_pairs"]["selected_for_bounded_stage2"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
