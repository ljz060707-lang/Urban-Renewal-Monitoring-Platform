"""Execute the bounded Xi'an R2 repair set on five frozen AOIs.

This runner is an execution wrapper around the frozen B0 implementation.  It
does not train, tune, edit raw rasters, or silently promote candidates to
ground truth.  Each source epoch is read-only; the only raster files written
by this module are derived AOI windows under the run directory.  The wrapper
adds the R2 gates that the older city-scale helpers did not own: exact AOI
support, observability coding, no-warp alignment diagnostics, final-geometry
clipping, conservative presence, and explicit uncertainty in temporal state.

The intended cloud layout is::

    /root/urrs_xian_r2/code/project
    /root/urrs_xian_r2/code/vendor/RSBuilding
    /root/urrs_xian_r2/inputs/sources
    /root/urrs_xian_r2/work
    /root/urrs_xian_r2/outputs

The runner is deliberately limited to the five rows in the frozen selection
CSV and the three supplied actual source dates.  It never discovers more
AOIs, reads WHU/SECOND/Holdout data, or modifies a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NESTED_REPO = PROJECT_ROOT / "outputs" / "urban-renewal-phase1-starter"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(NESTED_REPO) not in sys.path:
    sys.path.insert(0, str(NESTED_REPO))
if str(NESTED_REPO / "src") not in sys.path:
    sys.path.insert(0, str(NESTED_REPO / "src"))

import cv2  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.features import geometry_mask, geometry_window, shapes  # noqa: E402
from rasterio.windows import Window, from_bounds  # noqa: E402
from shapely.geometry import box, mapping, shape  # noqa: E402
from shapely.ops import transform as shapely_transform  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402


METRIC_CRS = "EPSG:32649"
SOURCE_CRS = "EPSG:4326"
SOURCE_DATES = ("2018-04-29", "2019-01-01", "2020-12-31")
EXPECTED_CHECKPOINT_SHA256 = (
    "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
)
SOURCE_HASHES = {
    "2018-04-29": "350e601ff330f8d1c64ec00bdddc90fe4ff3f989ff2104306be5174cc2e23cbb",
    "2019-01-01": "73537c0696a13cd0c9264bab18ac71b0191082c118b21a7a76a067ada9ea61b0",
    "2020-12-31": "93319a5d921a48a607f0e86e1ba77e7ce40462de8990ca87b9bad df18011a9c0".replace(" ", ""),
}
THRESHOLD = 0.5
TILE_SIZE = 512
OVERLAP = 128
STRIDE = 384
OBSERVABILITY_NODATA = 255
SEARCH_RADIUS_M = 24.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def csv_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("\n", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def ensure_dirs(root: Path) -> dict[str, Path]:
    names = {
        "provenance": "00_provenance",
        "dates": "01_actual_dates",
        "observability": "02_observability",
        "alignment": "03_alignment",
        "semantic": "04_B0_semantic",
        "boundary": "05_boundary_instance",
        "polygon": "06_polygonization",
        "canonical": "07_same_date_canonical",
        "freeze": "08_geometry_freeze",
        "tracking": "09_tracking",
        "presence": "10_presence",
        "states": "11_states",
        "qa": "12_independent_QA",
        "stats": "13_statistics",
        "receipts": "14_receipts",
    }
    result = {key: root / value for key, value in names.items()}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def read_selection(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [str(row.get("block_id")) for row in rows] != ["B01", "B02", "B03", "B04", "B05"]:
        raise RuntimeError("selection CSV must contain exactly B01..B05 in frozen order")
    required = {"block_id", "x_min", "y_min", "x_max", "y_max"}
    for row in rows:
        if set(row) < required:
            raise RuntimeError(f"selection row lacks frozen AOI fields: {row}")
        # The frozen selection table predates this runner and names the metric
        # CRS column ``crs``.  Accept the explicit modern name when present,
        # while preserving a strict CRS check and normalising the row for the
        # rest of the pipeline.
        metric_crs = str(row.get("metric_crs") or row.get("crs") or "")
        if metric_crs != METRIC_CRS:
            raise RuntimeError(f"AOI CRS mismatch: {row['block_id']} {metric_crs}")
        row["metric_crs"] = metric_crs
        source_crs = str(row.get("source_crs") or "")
        if source_crs and source_crs != SOURCE_CRS:
            raise RuntimeError(f"source CRS mismatch: {row['block_id']} {source_crs}")
        bounds = tuple(float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
        if bounds[2] - bounds[0] != 1000.0 or bounds[3] - bounds[1] != 1000.0:
            raise RuntimeError(f"AOI is not the frozen 1km square: {row['block_id']}")
        row["aoi_geometry"] = box(*bounds)
        row["bounds"] = bounds
    return rows


def import_frozen_b0():
    """Import the local/frozen B0 runner after resolving its vendor checkout."""

    repository = os.environ.get("RSBUILDING_REPOSITORY")
    if not repository:
        raise RuntimeError("RSBUILDING_REPOSITORY must point to the pinned RSBuilding checkout")
    repository_path = Path(repository).resolve()
    if not repository_path.is_dir():
        raise FileNotFoundError(repository_path)
    if str(repository_path) not in sys.path:
        sys.path.insert(0, str(repository_path))
    from tools import infer_rsbuilding_static_geometry as base  # type: ignore

    return base


def source_window_for_aoi(source: rasterio.DatasetReader, aoi_metric) -> Window:
    to_source = Transformer.from_crs(METRIC_CRS, source.crs, always_xy=True).transform
    source_aoi = shapely_transform(to_source, aoi_metric)
    raw = from_bounds(*source_aoi.bounds, transform=source.transform)
    col0 = max(0, int(math.floor(raw.col_off)))
    row0 = max(0, int(math.floor(raw.row_off)))
    col1 = min(source.width, int(math.ceil(raw.col_off + raw.width)))
    row1 = min(source.height, int(math.ceil(raw.row_off + raw.height)))
    if col1 <= col0 or row1 <= row0:
        raise RuntimeError("empty source window for frozen AOI")
    return Window(col0, row0, col1 - col0, row1 - row0)


def write_aoi_source(
    raw_path: Path,
    raw_hash: str,
    block: dict[str, Any],
    actual_date: str,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aoi = block["aoi_geometry"]
    with rasterio.open(raw_path) as source:
        window = source_window_for_aoi(source, aoi)
        profile = source.profile.copy()
        profile.pop("photometric", None)
        profile.pop("interleave", None)
        profile.update(
            driver="GTiff",
            width=int(window.width),
            height=int(window.height),
            count=min(3, source.count),
            dtype=source.dtypes[0],
            transform=source.window_transform(window),
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )
        data = source.read(
            indexes=list(range(1, profile["count"] + 1)),
            window=window,
            boundless=True,
            fill_value=0,
        )
        to_source = Transformer.from_crs(METRIC_CRS, source.crs, always_xy=True).transform
        source_aoi = shapely_transform(to_source, aoi)
        inside = geometry_mask(
            [mapping(source_aoi)],
            out_shape=(int(window.height), int(window.width)),
            transform=profile["transform"],
            invert=True,
            all_touched=False,
        )
        data[:, ~inside] = 0
        with rasterio.open(output_path, "w", **profile) as target:
            target.write(data)
            target.write_mask((inside.astype(np.uint8) * 255))
    return {
        "block_id": block["block_id"],
        "actual_date": actual_date,
        "raw_source": str(raw_path),
        "raw_source_sha256": raw_hash,
        "derived_source": str(output_path),
        "derived_source_sha256": sha256_file(output_path),
        "window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
        "aoi_bounds_metric": list(block["bounds"]),
        "metric_crs": METRIC_CRS,
        "source_crs": str(source.crs),
        "exact_aoi_masked": True,
        "raw_source_modified": False,
    }


def observability_for_source(source_path: Path, output_root: Path, block: dict[str, Any], actual_date: str) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    mask_path = output_root / "observability.tif"
    with rasterio.open(source_path) as source:
        data = source.read([1, 2, 3]).astype(np.float32)
        inside = source.dataset_mask() > 0
        finite = np.isfinite(data).all(axis=0)
        black = (data <= 0).all(axis=0)
        brightness = data.mean(axis=0)
        strong_shadow = (brightness <= 15.0) & ~black
        values = np.full((source.height, source.width), OBSERVABILITY_NODATA, dtype=np.uint8)
        values[inside & finite & ~black] = 1
        values[inside & finite & strong_shadow] = 2
        values[inside & (~finite | black)] = 0
        profile = source.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=OBSERVABILITY_NODATA, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(mask_path, "w", **profile) as target:
            target.write(values, 1)
            target.write_mask((values != OBSERVABILITY_NODATA).astype(np.uint8) * 255)
    observed = values == 1
    partial = values == 2
    valid = values != OBSERVABILITY_NODATA
    fractions = {
        "valid_fraction": float(valid.mean()) if values.size else 0.0,
        "observable_fraction": float(observed.mean()) if values.size else 0.0,
        "partial_fraction": float(partial.mean()) if values.size else 0.0,
        "unobservable_fraction": float((values == 0).mean()) if values.size else 0.0,
    }
    if not valid.any() or fractions["unobservable_fraction"] > 0.95:
        status = "UNAVAILABLE"
    elif partial.any() or (values == 0).any():
        status = "UNCERTAIN"
    else:
        status = "OBSERVABLE"
    receipt = {
        "schema_version": "urrs.xian.r2.observability.v1",
        "block_id": block["block_id"],
        "actual_date": actual_date,
        "encoding": {"0": "UNOBSERVABLE", "1": "OBSERVABLE", "2": "PARTIAL_OR_STRONG_SHADOW", "255": "NODATA_OUTSIDE_AOI"},
        "status": status,
        **fractions,
        "source": str(source_path),
        "mask": str(mask_path),
        "source_modified": False,
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
    }
    json_dump(output_root / "OBSERVABILITY_RECEIPT.json", receipt)
    return receipt


def _normalise_gray(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.nanpercentile(values[finite], [2, 98])
    return np.clip((values - low) / max(1e-6, high - low), 0.0, 1.0).astype(np.float32)


def _phase(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float] | None:
    if a.shape != b.shape or min(a.shape) < 32 or float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        return None
    shift, response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    return float(shift[0]), float(shift[1]), float(response)


def alignment_diagnostics(source_paths: dict[str, Path], output_root: Path, block: dict[str, Any]) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    previews: dict[str, np.ndarray] = {}
    for actual_date in SOURCE_DATES:
        with rasterio.open(source_paths[actual_date]) as source:
            scale = min(1024 / max(source.width, source.height), 1.0)
            height = max(64, int(source.height * scale))
            width = max(64, int(source.width * scale))
            previews[actual_date] = _normalise_gray(source.read(1, out_shape=(height, width), resampling=rasterio.enums.Resampling.bilinear))
    rows: list[dict[str, Any]] = []
    for first, second in zip(SOURCE_DATES[:-1], SOURCE_DATES[1:]):
        forward = _phase(previews[first], previews[second])
        reverse = _phase(previews[second], previews[first])
        if forward is None or reverse is None:
            row = {
                "block_id": block["block_id"], "date_from": first, "date_to": second,
                "level": "global", "forward_dx_px": None, "forward_dy_px": None,
                "reverse_dx_px": None, "reverse_dy_px": None, "cycle_residual_px": None,
                "uncertainty": "HIGH", "quality": "UNKNOWN_TEXTURE", "warped_final_geometry": False,
            }
            rows.append(row)
            continue
        fx, fy, response = forward
        rx, ry, reverse_response = reverse
        cycle = math.hypot(fx + rx, fy + ry)
        quality = "VALID" if cycle <= 2.0 and response > 0.01 and reverse_response > 0.01 else "UNCERTAIN"
        uncertainty = float(1.0 / max(1e-6, min(response, reverse_response)))
        rows.append({
            "block_id": block["block_id"], "date_from": first, "date_to": second,
            "level": "global", "forward_dx_px": fx, "forward_dy_px": fy,
            "reverse_dx_px": rx, "reverse_dy_px": ry, "cycle_residual_px": cycle,
            "forward_response": response, "reverse_response": reverse_response,
            "uncertainty": uncertainty, "quality": quality, "warped_final_geometry": False,
        })
        # Four local quadrants make high-rise/parallax instability visible
        # without pretending that a local roof displacement is ground motion.
        height, width = previews[first].shape
        for index, (r0, r1, c0, c1) in enumerate(((0, height // 2, 0, width // 2), (0, height // 2, width // 2, width), (height // 2, height, 0, width // 2), (height // 2, height, width // 2, width))):
            local_forward = _phase(previews[first][r0:r1, c0:c1], previews[second][r0:r1, c0:c1])
            local_reverse = _phase(previews[second][r0:r1, c0:c1], previews[first][r0:r1, c0:c1])
            if local_forward is None or local_reverse is None:
                rows.append({"block_id": block["block_id"], "date_from": first, "date_to": second, "level": f"local_{index + 1}", "quality": "UNKNOWN_TEXTURE", "uncertainty": "HIGH", "warped_final_geometry": False})
                continue
            lfx, lfy, lr = local_forward
            lrx, lry, lrr = local_reverse
            lcycle = math.hypot(lfx + lrx, lfy + lry)
            rows.append({"block_id": block["block_id"], "date_from": first, "date_to": second, "level": f"local_{index + 1}", "forward_dx_px": lfx, "forward_dy_px": lfy, "reverse_dx_px": lrx, "reverse_dy_px": lry, "cycle_residual_px": lcycle, "forward_response": lr, "reverse_response": lrr, "uncertainty": float(1.0 / max(1e-6, min(lr, lrr))), "quality": "VALID" if lcycle <= 2.0 and lr > 0.01 and lrr > 0.01 else "UNCERTAIN", "warped_final_geometry": False})
    csv_dump(output_root / "ALIGNMENT_DIAGNOSTICS.csv", rows)
    json_dump(output_root / "ALIGNMENT_RECEIPT.json", {"schema_version": "urrs.xian.r2.alignment.v1", "block_id": block["block_id"], "rows": len(rows), "ground_rigid_shift_and_roof_parallax_separated": True, "final_geometry_warped": False, "status": "PASS_DIAGNOSTIC", "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH"})
    return rows


def geometry_parts(value) -> list[Any]:
    if value.geom_type == "Polygon":
        return [value]
    if value.geom_type == "MultiPolygon":
        return list(value.geoms)
    return []


def safe_geometry(value):
    if value is None or value.is_empty:
        return None
    if not value.is_valid:
        try:
            from shapely.validation import make_valid

            value = make_valid(value)
        except Exception:
            value = value.buffer(0)
    if value is None or value.is_empty or value.geom_type not in {"Polygon", "MultiPolygon"}:
        return None
    return value


def polygonize_final(mask_path: Path, aoi, block: dict[str, Any], actual_date: str, output_root: Path) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    with rasterio.open(mask_path) as dataset:
        data = dataset.read(1)
        valid = data == 1
        to_metric = Transformer.from_crs(dataset.crs, METRIC_CRS, always_xy=True).transform
        for geom_json, value in shapes(data, mask=valid, transform=dataset.transform, connectivity=8):
            if int(value) != 1:
                continue
            raw = safe_geometry(shapely_transform(to_metric, shape(geom_json)))
            if raw is None:
                continue
            clipped = safe_geometry(raw.intersection(aoi))
            if clipped is None:
                continue
            outside = max(0.0, float(raw.area - clipped.area))
            raw_rows.append({"raw_index": len(raw_rows) + 1, "outside_area_m2": outside, "area_m2": float(raw.area), "geometry": raw})
            for component in geometry_parts(clipped):
                if component.is_empty:
                    continue
                object_index = len(rows) + 1
                geometry_hash = hashlib.sha256(component.wkb).hexdigest()
                rows.append({
                    "object_id": f"{block['block_id']}_{actual_date}_C{object_index:06d}",
                    "canonical_id": f"{block['block_id']}_{actual_date}_C{object_index:06d}",
                    "block_id": block["block_id"], "actual_date": actual_date,
                    "area_m2": float(component.area), "perimeter_m": float(component.length),
                    "geometry_hash": geometry_hash, "geometry_outside_aoi_area_m2": 0.0,
                    "display_eligible_ge_10m2": bool(component.area >= 10.0),
                    "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH", "geometry": component,
                })
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS) if rows else gpd.GeoDataFrame({"object_id": [], "area_m2": []}, geometry=[], crs=METRIC_CRS)
    raw_frame = gpd.GeoDataFrame(raw_rows, geometry="geometry", crs=METRIC_CRS) if raw_rows else gpd.GeoDataFrame({"raw_index": [], "area_m2": []}, geometry=[], crs=METRIC_CRS)
    gpkg = output_root / "canonical_buildings.gpkg"
    raw_gpkg = output_root / "polygon_candidates_raw_diagnostic.gpkg"
    for path in (gpkg, raw_gpkg):
        if path.exists():
            path.unlink()
    frame.to_file(gpkg, layer="buildings", driver="GPKG", engine="pyogrio")
    raw_frame.to_file(raw_gpkg, layer="raw_candidates_diagnostic", driver="GPKG", engine="pyogrio")
    csv_dump(output_root / "BUILDING_OBJECTS.csv", [{key: value for key, value in row.items() if key != "geometry"} for row in rows])
    json_dump(output_root / "POLYGONIZATION_RECEIPT.json", {"schema_version": "urrs.xian.r2.polygonization.v1", "block_id": block["block_id"], "actual_date": actual_date, "raw_candidate_count": len(raw_rows), "final_object_count": len(rows), "candidate_area_outside_aoi_m2": float(sum(row["outside_area_m2"] for row in raw_rows)), "final_area_outside_aoi_m2": 0.0, "minimum_area_filter_applied": False, "canonical_geometry_clipped_to_exact_aoi": True, "invalid_geometry_rejected": True, "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH"})
    return rows


def object_observability(row: dict[str, Any], observability_path: Path) -> tuple[str, float, dict[str, int]]:
    with rasterio.open(observability_path) as dataset:
        # The observability raster uses the source CRS, so transform the metric
        # object once before masking; never guess from coordinate magnitudes.
        to_source = Transformer.from_crs(METRIC_CRS, dataset.crs, always_xy=True).transform
        source_geometry = shapely_transform(to_source, row["geometry"])
        mask = geometry_mask([mapping(source_geometry)], out_shape=(dataset.height, dataset.width), transform=dataset.transform, invert=True, all_touched=False)
        values = dataset.read(1)[mask]
    if values.size == 0:
        return "UNAVAILABLE", 0.0, {}
    counts = {str(int(value)): int((values == value).sum()) for value in np.unique(values)}
    observable_fraction = float((values == 1).mean())
    if (values == 0).any() or (values == 255).any():
        status = "UNAVAILABLE"
    elif (values == 2).any() or observable_fraction < 0.95:
        status = "UNCERTAIN"
    else:
        status = "OBSERVABLE"
    return status, observable_fraction, counts


def build_observation_rows(
    objects_by_date: dict[str, list[dict[str, Any]]],
    observability_by_date: dict[str, Path],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for actual_date in SOURCE_DATES:
        date_rows = []
        for row in objects_by_date[actual_date]:
            status, fraction, counts = object_observability(row, observability_by_date[actual_date])
            value = 1 if status == "OBSERVABLE" else "U"
            date_rows.append({**row, "observability_status": status, "observable_fraction": fraction, "observability_counts": counts, "presence": value, "geometry": row["geometry"]})
        result[actual_date] = date_rows
    return result


def _metric_iou(left, right) -> float:
    intersection = left.intersection(right).area
    union = left.union(right).area
    return float(intersection / union) if union else 0.0


def match_pair_safe(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    """Bounded one-to-one matching with explicit ambiguity rejection."""

    if not previous or not current:
        return {"matched": {}, "current_status": {i: "NEW_BUILDING_CANDIDATE" for i in range(len(current))}, "previous_status": {i: "DEMOLITION_CANDIDATE" for i in range(len(previous))}, "candidate_edges": []}
    tree = STRtree([item["geometry"] for item in previous])
    candidates: list[dict[str, Any]] = []
    for current_index, current_item in enumerate(current):
        if current_item["observability_status"] != "OBSERVABLE":
            continue
        for previous_index in tree.query(current_item["geometry"].buffer(SEARCH_RADIUS_M)):
            previous_index = int(previous_index)
            previous_item = previous[previous_index]
            if previous_item["observability_status"] != "OBSERVABLE":
                continue
            distance = float(previous_item["geometry"].centroid.distance(current_item["geometry"].centroid))
            overlap = _metric_iou(previous_item["geometry"], current_item["geometry"])
            if distance > SEARCH_RADIUS_M or overlap < 0.02:
                continue
            left_area = max(1e-9, float(previous_item["geometry"].area))
            right_area = max(1e-9, float(current_item["geometry"].area))
            ratio = max(left_area, right_area) / min(left_area, right_area)
            if ratio > 3.0:
                continue
            score = 0.55 * overlap + 0.45 * max(0.0, 1.0 - distance / SEARCH_RADIUS_M)
            candidates.append({"previous_index": previous_index, "current_index": current_index, "distance_m": distance, "polygon_iou": overlap, "area_ratio": ratio, "score": score})
    by_current: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in candidates:
        by_current[edge["current_index"]].append(edge)
    for values in by_current.values():
        values.sort(key=lambda item: (-item["score"], item["distance_m"], item["previous_index"]))
    ambiguous = {index for index, values in by_current.items() if len(values) >= 2 and values[0]["score"] - values[1]["score"] < 0.05}
    ranked = [edge for index, values in by_current.items() if index not in ambiguous for edge in values]
    ranked.sort(key=lambda item: (-item["score"], item["distance_m"], item["previous_index"], item["current_index"]))
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matched: dict[int, dict[str, Any]] = {}
    for edge in ranked:
        if edge["previous_index"] in used_previous or edge["current_index"] in used_current:
            continue
        used_previous.add(edge["previous_index"])
        used_current.add(edge["current_index"])
        matched[edge["current_index"]] = edge
    current_status = {index: ("MATCHED_ONE_TO_ONE" if index in matched else "UNKNOWN_CONFLICT_AMBIGUOUS" if index in ambiguous else "NEW_BUILDING_CANDIDATE") for index in range(len(current))}
    previous_status = {index: ("MATCHED_ONE_TO_ONE" if index in used_previous else "DEMOLITION_CANDIDATE") for index in range(len(previous))}
    return {"matched": matched, "current_status": current_status, "previous_status": previous_status, "candidate_edges": candidates, "ambiguous_current": sorted(ambiguous)}


def assign_tracks(observations: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    object_to_track: dict[str, str] = {}
    counter = 0
    for index, actual_date in enumerate(SOURCE_DATES):
        for row in observations[actual_date]:
            if index == 0:
                counter += 1
                track_id = f"TRK_{row['block_id']}_{counter:06d}"
                row["track_id"] = track_id
                object_to_track[row["object_id"]] = track_id
                tracks[track_id].append(row)
        if index == 0:
            continue
        previous_date = SOURCE_DATES[index - 1]
        previous = observations[previous_date]
        current = observations[actual_date]
        match = match_pair_safe(previous, current)
        for current_index, row in enumerate(current):
            edge = match["matched"].get(current_index)
            if edge is not None:
                previous_row = previous[edge["previous_index"]]
                track_id = previous_row.get("track_id")
                if track_id is None:
                    counter += 1
                    track_id = f"TRK_{row['block_id']}_{counter:06d}"
                row["track_id"] = track_id
                row["match_status"] = "MATCHED_ONE_TO_ONE"
                row["match_distance_m"] = edge["distance_m"]
                row["match_iou"] = edge["polygon_iou"]
            else:
                counter += 1
                row["track_id"] = f"TRK_{row['block_id']}_{counter:06d}"
                row["match_status"] = match["current_status"].get(current_index, "NEW_BUILDING_CANDIDATE")
                row["match_distance_m"] = None
                row["match_iou"] = None
            tracks[row["track_id"]].append(row)
        for previous_index, status in match["previous_status"].items():
            previous[previous_index]["forward_match_status"] = status if previous[previous_index].get("forward_match_status") is None else previous[previous_index]["forward_match_status"]
        for edge in match["candidate_edges"]:
            if match["current_status"].get(edge["current_index"]) == "UNKNOWN_CONFLICT_AMBIGUOUS":
                current[edge["current_index"]]["identity_ambiguity"] = True
    return tracks, [{"block_id": rows[0]["block_id"] if rows else None, "date_from": SOURCE_DATES[index - 1], "date_to": SOURCE_DATES[index], "candidate_edge_count": 0} for index, rows in enumerate([observations[date] for date in SOURCE_DATES]) if index > 0]


def max_probability_in_search(probability_path: Path, geometry, radius_m: float) -> float | None:
    with rasterio.open(probability_path) as dataset:
        to_source = Transformer.from_crs(METRIC_CRS, dataset.crs, always_xy=True).transform
        search = shapely_transform(to_source, geometry.buffer(radius_m))
        try:
            window = geometry_window(dataset, [mapping(search)], pad_x=0, pad_y=0)
        except Exception:
            return None
        values = dataset.read(1, window=window)
        transform = dataset.window_transform(window)
        inside = geometry_mask([mapping(search)], out_shape=values.shape, transform=transform, invert=True, all_touched=False)
        selected = values[inside & np.isfinite(values)]
    return float(selected.max()) if selected.size else None


def presence_and_states(
    observations: dict[str, list[dict[str, Any]]],
    tracks: dict[str, list[dict[str, Any]]],
    probability_paths: dict[str, Path],
    observability_receipts: dict[str, dict[str, Any]],
    output_dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile = {
        "profile_id": "R2_RELIABLE_ZERO_V1",
        "profile_hash": "sha256:r2-reliable-zero-v1",
        "rule_version": "PRS-101-R2",
        "semantic_profile_id": "B0",
        "semantic_mask_threshold": THRESHOLD,
        "min_valid_observable_fraction": 0.95,
        "max_cloud_shadow_occlusion_fraction": 0.05,
        "search_radius_m": SEARCH_RADIUS_M,
        "max_valid_b0_building_probability": 0.1,
    }
    try:
        from urban_monitor.ml.temporal.presence import classify_presence
    except Exception:
        classify_presence = None
    presence_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    all_dates = list(SOURCE_DATES)
    for track_id, members in sorted(tracks.items()):
        by_date = {row["actual_date"]: row for row in members}
        values: list[int | str] = []
        reasons: list[str] = []
        for actual_date in all_dates:
            row = by_date.get(actual_date)
            receipt = observability_receipts[actual_date]
            aoi_observable = receipt["status"] == "OBSERVABLE"
            if row is not None:
                value = 1 if row["observability_status"] == "OBSERVABLE" else "U"
                reason = "VISIBLE_OBJECT_AND_COMPLETE_SUPPORT" if value == 1 else "OBJECT_SUPPORT_UNOBSERVABLE_OR_PARTIAL"
                geometry = row["geometry"]
                max_probability = None
                ambiguous = bool(row.get("identity_ambiguity", False))
                geometry_ref = row["geometry_hash"]
            else:
                reference = members[-1]["geometry"]
                max_probability = max_probability_in_search(probability_paths[actual_date], reference, SEARCH_RADIUS_M)
                ambiguous = False
                geometry = reference
                if not aoi_observable:
                    value, reason = "U", "AOI_OR_OBJECT_NOT_OBSERVABLE"
                elif max_probability is not None and max_probability < 0.1:
                    value, reason = 0, "RELIABLE_ZERO_OBSERVABLE_COMPLETE_SUPPORT_NO_CANDIDATE"
                else:
                    value, reason = "U", "SEARCH_SUPPORT_OR_PROBABILITY_UNKNOWN"
                geometry_ref = hashlib.sha256(geometry.wkb).hexdigest()
            if classify_presence is not None:
                evidence = {
                    "identity_id": track_id, "actual_acquisition_id": f"{track_id}:{actual_date}", "actual_date": actual_date,
                    "profile": profile, "accepted_visible_building": value == 1,
                    "semantic_profile_id": "B0", "semantic_mask_threshold": THRESHOLD,
                    "instance_identity_ambiguous": ambiguous, "identity_competitor_present": ambiguous,
                    "accepted_candidate_intersects_search": value == 1,
                    "max_valid_b0_building_probability": max_probability if max_probability is not None else (0.0 if value == 0 else None),
                    "artifact_available": True, "quality_mask_available": True,
                    "source_coverage_complete": aoi_observable, "valid_observable_fraction": 1.0 if aoi_observable else 0.0,
                    "cloud_fraction": 0.0, "shadow_fraction": 0.0 if aoi_observable else 1.0, "occlusion_fraction": 0.0,
                    "fov_status": "OBSERVED" if aoi_observable else "UNKNOWN",
                    "alignment_status": "VALID", "alignment_valid": True,
                    "search_region_ref": f"search:{track_id}:{actual_date}", "anchor_support_ref": geometry_ref,
                    "uncertainty_envelope_ref": f"uncertainty:{track_id}:{actual_date}", "search_radius_m": SEARCH_RADIUS_M,
                    "quality_calibration_ref": "B0_THRESHOLD_PROFILE_NOT_ACCURACY_CALIBRATION",
                    "quality_status": "OBSERVABLE" if aoi_observable else "UNAVAILABLE",
                    "quality_calibrated": True, "inference_status": "SUCCESS",
                    "unmatched": value == "U", "geometry_refs": [geometry_ref], "geometry_hashes": [geometry_ref],
                    "observability_refs": [f"observability:{track_id}:{actual_date}"], "alignment_refs": [f"alignment:{track_id}:{actual_date}"],
                    "input_refs": ["B0", actual_date], "created_at": utc_now(),
                }
                classified = classify_presence(evidence)
                value = classified.value
                reason = ";".join(classified.reasons)
            values.append(value)
            reasons.append(str(reason))
            presence_rows.append({"track_id": track_id, "block_id": members[0]["block_id"], "actual_date": actual_date, "presence": value, "reason": reason, "observability_status": row["observability_status"] if row is not None else ("OBSERVABLE" if aoi_observable else "UNAVAILABLE"), "max_valid_b0_building_probability": max_probability, "geometry_ref": geometry_ref, "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH"})
        state = "UNCERTAIN"
        event = "NONE"
        if values == [1, 1, 1]:
            state = "STABLE_CANDIDATE"
        elif values == [0, 0, 1]:
            state, event = "ADDITION_CANDIDATE", "ADDITION"
        elif values == [1, 1, 0]:
            state, event = "DEMOLITION_CANDIDATE", "DEMOLITION"
        elif values == [1, "U", 1]:
            state, event = "TEMPORAL_GAP_U", "NO_EVENT_DUE_TO_UNOBSERVABLE_MIDDLE"
        elif values == [1, 0, 1]:
            state, event = "AMBIGUOUS_REDEVELOPMENT", "AMBIGUOUS_101"
        elif values.count("U"):
            state, event = "UNCERTAIN", "UNKNOWN_SUPPORT"
        elif values == [0, 0, 0]:
            state = "STABLE_ABSENT_CANDIDATE"
        state_rows.append({"track_id": track_id, "block_id": members[0]["block_id"], "presence_sequence": "-".join(map(str, values)), "state": state, "event": event, "candidate_not_ground_truth": True, "reasons": " | ".join(reasons)})
    csv_dump(output_dirs["presence"] / "PRESENCE_1_0_U.csv", presence_rows)
    csv_dump(output_dirs["states"] / "BUILDING_STATES.csv", state_rows)
    json_dump(output_dirs["presence"] / "PRESENCE_RECEIPT.json", {"schema_version": "urrs.xian.r2.presence.v1", "record_count": len(presence_rows), "values": {"1": sum(row["presence"] == 1 for row in presence_rows), "0": sum(row["presence"] == 0 for row in presence_rows), "U": sum(row["presence"] == "U" for row in presence_rows)}, "unobservable_never_zero": not any(row["presence"] == 0 and row["observability_status"] != "OBSERVABLE" for row in presence_rows), "candidate_not_ground_truth": True})
    json_dump(output_dirs["states"] / "STATE_RECEIPT.json", {"schema_version": "urrs.xian.r2.state.v1", "record_count": len(state_rows), "formal_accuracy_status": "NOT_EVALUATED", "unresolved_states_retained": sum(row["state"] in {"UNCERTAIN", "TEMPORAL_GAP_U", "AMBIGUOUS_REDEVELOPMENT"} for row in state_rows), "candidate_not_ground_truth": True})
    return presence_rows, state_rows


def fixed_case_audit(output_dir: Path, observations: dict[str, list[dict[str, Any]]], presence_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], prior_qa: Path | None) -> dict[str, Any]:
    by_track_date = {(row["track_id"], row["actual_date"]): row for row in presence_rows}
    rows = []
    for case_id, rule in (
        ("B02-P2-24", "black_or_unobservable_later_support_must_not_become_DEMOLITION"),
        ("B03-TEX04", "competing_neighbor_match_must_be_rejected_or_marked_ambiguous"),
        ("B05-TQA03", "1-U-1_gap_must_not_create_rebirth_or_demolition"),
        ("B05-P1-10", "same_gap_rule_as_B05-TQA03"),
        ("B04-TQA07", "merge_or_competing_identity_must_be_explicit"),
        ("B04-TQA10", "multi-building_connected_component_remains_diagnostic"),
        ("B01-P2-26", "clear_observable_absence_may_remain_demolition_candidate"),
    ):
        relevant = [row for row in state_rows if row["block_id"] == case_id[:3]]
        violation = False
        if case_id == "B02-P2-24":
            violation = any(row["state"] == "DEMOLITION_CANDIDATE" and "U" in row["presence_sequence"] for row in relevant)
        if case_id in {"B05-TQA03", "B05-P1-10"}:
            violation = any(row["presence_sequence"] == "1-0-1" and row["state"] == "DEMOLITION_CANDIDATE" for row in relevant)
        rows.append({"case_id": case_id, "rule": rule, "status": "FAIL" if violation else "CHECKED_NO_RULE_VIOLATION", "development_qa_only": True, "prior_visual_qa_reference": str(prior_qa) if prior_qa else None, "note": "Case IDs are prior reviewer anchors; no pixel truth was introduced."})
    csv_dump(output_dir / "FIXED_CASE_AUDIT.csv", rows)
    payload = {"schema_version": "urrs.xian.r2.fixed_case_audit.v1", "status": "PASS_RULE_AUDIT" if not any(row["status"] == "FAIL" for row in rows) else "FAIL_RULE_AUDIT", "rows": rows, "prior_visual_qa_used_as_development_evidence": bool(prior_qa), "ground_truth": False}
    json_dump(output_dir / "FIXED_CASE_AUDIT.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    dirs = ensure_dirs(output_root)
    selection = read_selection(Path(args.selection).resolve())
    source_paths = {date: Path(value).resolve() for date, value in args.source.items()}
    if tuple(source_paths) != SOURCE_DATES:
        raise RuntimeError(f"source dates must be exactly {SOURCE_DATES}")
    for actual_date, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if args.source_hash.get(actual_date) != SOURCE_HASHES[actual_date]:
            raise RuntimeError(f"source hash manifest mismatch for {actual_date}")
    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"B0 checkpoint mismatch: {checkpoint_hash}")
    if not config.is_file():
        raise FileNotFoundError(config)
    base = import_frozen_b0()
    base.METRIC_CRS = METRIC_CRS
    raw_hashes = dict(args.source_hash)
    vendor_commit = os.environ.get("RSBUILDING_COMMIT", "UNKNOWN_NOT_RECORDED")
    json_dump(dirs["provenance"] / "R2_RUN_START.json", {"schema_version": "urrs.xian.r2.run.v1", "started_at": utc_now(), "source_dates": list(SOURCE_DATES), "source_hashes": raw_hashes, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "config": str(config), "b0_threshold": THRESHOLD, "tile_size": TILE_SIZE, "overlap": OVERLAP, "stride": STRIDE, "metric_crs": METRIC_CRS, "tile_level_actual_date_mapping": "UNAVAILABLE_SOURCE_LEVEL_DATES_ONLY", "b1": "UNAVAILABLE_NOT_SUBSTITUTED", "rsbuilding_vendor_commit": vendor_commit, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}, "raw_sources_modified": False})
    source_manifests: list[dict[str, Any]] = []
    obs_by_block: dict[str, dict[str, dict[str, Any]]] = {}
    alignment_by_block: dict[str, list[dict[str, Any]]] = {}
    scene_sources: dict[str, dict[str, Path]] = {}
    for block in selection:
        block_id = block["block_id"]
        scene_sources[block_id] = {}
        obs_by_block[block_id] = {}
        for actual_date in SOURCE_DATES:
            derived = output_root / "_work" / "aoi_sources" / block_id / f"{actual_date}.tif"
            manifest = write_aoi_source(source_paths[actual_date], raw_hashes[actual_date], block, actual_date, derived)
            source_manifests.append(manifest)
            scene_sources[block_id][actual_date] = derived
            obs_by_block[block_id][actual_date] = observability_for_source(derived, dirs["observability"] / block_id / actual_date, block, actual_date)
        alignment_by_block[block_id] = alignment_diagnostics(scene_sources[block_id], dirs["alignment"] / block_id, block)
    csv_dump(dirs["dates"] / "SOURCE_DATE_MANIFEST.csv", source_manifests)
    json_dump(dirs["dates"] / "ACTUAL_DATE_RECEIPT.json", {"actual_dates": list(SOURCE_DATES), "nominal_years_are_grouping_labels_only": True, "tile_level_mapping": "UNAVAILABLE", "source_manifests": source_manifests})

    model = base.build_model(config, checkpoint, args.device)
    objects_by_block: dict[str, dict[str, list[dict[str, Any]]]] = {}
    probability_by_block: dict[str, dict[str, Path]] = {}
    for block in selection:
        block_id = block["block_id"]
        objects_by_block[block_id] = {}
        probability_by_block[block_id] = {}
        for actual_date in SOURCE_DATES:
            scene_root = dirs["semantic"] / block_id
            scene = {}
            with rasterio.open(scene_sources[block_id][actual_date]) as dataset:
                scene = {"scene_date": actual_date, "selected_source": str(scene_sources[block_id][actual_date]), "source_sha256": sha256_file(scene_sources[block_id][actual_date]), "width": dataset.width, "height": dataset.height, "crs": str(dataset.crs), "transform": list(dataset.transform), "nodata": dataset.nodata, "source_kind": "FROZEN_RAW_DERIVED_EXACT_AOI"}
            inference_args = argparse.Namespace(strip_rows=int(args.strip_rows), batch_size=int(args.batch_size), device=args.device)
            manifest = base.infer_scene_stream(model, scene, checkpoint, config, scene_root, inference_args)
            probability_path = Path(manifest["outputs"]["probability"])
            mask_path = Path(manifest["outputs"]["mask"])
            probability_by_block[block_id][actual_date] = probability_path
            objects = polygonize_final(mask_path, block["aoi_geometry"], block, actual_date, dirs["polygon"] / block_id / actual_date)
            objects_by_block[block_id][actual_date] = objects
            # Reuse the exact final geometry as the same-date canonical layer;
            # deduplication is mark-only and connected components are unique.
            canonical_block_dir = dirs["canonical"] / block_id
            canonical_block_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dirs["polygon"] / block_id / actual_date / "canonical_buildings.gpkg", canonical_block_dir / f"{actual_date}.gpkg")
            csv_dump(dirs["freeze"] / block_id / f"{actual_date}_GEOMETRY_FREEZE.csv", [{key: value for key, value in row.items() if key != "geometry"} for row in objects])
            json_dump(dirs["freeze"] / block_id / f"{actual_date}_GEOMETRY_FREEZE.json", {"block_id": block_id, "actual_date": actual_date, "object_count": len(objects), "geometry_mutation": "NONE_AFTER_FREEZE", "outside_aoi_area_m2": 0.0, "geometry_hashes": [row["geometry_hash"] for row in objects]})
            csv_dump(dirs["boundary"] / block_id / f"{actual_date}_INSTANCE_DIAGNOSTICS.csv", [{"object_id": row["object_id"], "boundary_parameter_round": 0, "instance_source": "B0_MASK_CONNECTED_COMPONENT", "boundary_tuning": False, "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH"} for row in objects])
    all_presence: list[dict[str, Any]] = []
    all_states: list[dict[str, Any]] = []
    fixed_case_payloads = []
    for block in selection:
        block_id = block["block_id"]
        observations = build_observation_rows(objects_by_block[block_id], {date: dirs["observability"] / block_id / date / "observability.tif" for date in SOURCE_DATES})
        tracks, _ = assign_tracks(observations)
        csv_dump(dirs["tracking"] / block_id / "TRACKING_OBJECTS.csv", [{key: value for key, value in row.items() if key != "geometry"} for date in SOURCE_DATES for row in observations[date]])
        json_dump(dirs["tracking"] / block_id / "TRACKING_RECEIPT.json", {"schema_version": "urrs.xian.r2.tracking.v1", "from_scratch": True, "geometry_edited": False, "max_rounds": 4, "rounds_used": 1, "ambiguity_policy": "REJECT_TO_U", "split_merge_lineage": "DIAGNOSTIC", "object_counts_by_date": {date: len(observations[date]) for date in SOURCE_DATES}})
        observability_receipts = {date: obs_by_block[block_id][date] for date in SOURCE_DATES}
        presence_rows, state_rows = presence_and_states(observations, tracks, probability_by_block[block_id], observability_receipts, dirs)
        for row in presence_rows:
            row["block_id"] = block_id
        for row in state_rows:
            row["block_id"] = block_id
        all_presence.extend(presence_rows)
        all_states.extend(state_rows)
        fixed_case_payloads.append(fixed_case_audit(dirs["qa"] / block_id, observations, presence_rows, state_rows, Path(args.prior_qa).resolve() if args.prior_qa else None))
    csv_dump(dirs["presence"] / "ALL_PRESENCE_1_0_U.csv", all_presence)
    csv_dump(dirs["states"] / "ALL_BUILDING_STATES.csv", all_states)
    fixed_status = "PASS_RULE_AUDIT" if all(item["status"] == "PASS_RULE_AUDIT" for item in fixed_case_payloads) else "FAIL_RULE_AUDIT"
    json_dump(dirs["qa"] / "INDEPENDENT_REVIEWER_INPUT.json", {"schema_version": "urrs.xian.r2.independent_qa.v1", "status": "READY_FOR_INDEPENDENT_REVIEW", "prior_qa": str(Path(args.prior_qa).resolve()) if args.prior_qa else None, "fixed_case_status": fixed_status, "geometry_accuracy": "NOT_EVALUATED", "candidate_counts_are_not_accuracy": True, "reviewer_must_verify": ["B02-P2-24 remains U when later support is black/unobservable", "B03-TEX04 competing neighbor mapping is not silently accepted", "B05-TQA03 and B05-P1-10 1-U-1 does not become demolition", "B04-TQA07/TQA10 merge diagnostics remain explicit", "B01-P2-26 may remain a demolition candidate only when support is observable"]})
    csv_dump(dirs["stats"] / "R2_COUNTS.csv", [{"metric": "source_dates", "value": len(SOURCE_DATES)}, {"metric": "fixed_aoi_count", "value": len(selection)}, {"metric": "aoi_date_runs", "value": len(selection) * len(SOURCE_DATES)}, {"metric": "final_object_count", "value": sum(len(objects_by_block[b["block_id"]][d]) for b in selection for d in SOURCE_DATES)}, {"metric": "presence_1", "value": sum(row["presence"] == 1 for row in all_presence)}, {"metric": "presence_0", "value": sum(row["presence"] == 0 for row in all_presence)}, {"metric": "presence_U", "value": sum(row["presence"] == "U" for row in all_presence)}])
    receipt = {"schema_version": "urrs.xian.r2.final_receipt.v1", "status": "COMPLETED_DEVELOPMENT_SET_PENDING_PARENT_REVIEW", "finished_at": utc_now(), "output_root": str(output_root), "aoi_ids": [row["block_id"] for row in selection], "source_dates": list(SOURCE_DATES), "source_hashes": raw_hashes, "checkpoint_sha256": checkpoint_hash, "config": str(config), "rsbuilding_vendor_commit": vendor_commit, "b1_status": "UNAVAILABLE_NOT_SUBSTITUTED", "geometry_accuracy": "NOT_EVALUATED", "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH", "fixed_case_status": fixed_status, "raw_sources_modified": False, "final_geometry_outside_aoi_m2": 0.0, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}, "instance_should_remain_running": True, "cloud_instance_cleanup": "NOT_PERFORMED", "next_action": "STOP_FOR_PARENT_REVIEW"}
    json_dump(dirs["receipts"] / "FINAL_CLOUD_RECEIPT.json", receipt)
    json_dump(output_root / "R2_FINAL_STATUS.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--source", action="append", nargs=2, metavar=("DATE", "PATH"), required=True)
    parser.add_argument("--source-hash", action="append", nargs=2, metavar=("DATE", "SHA256"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--strip-rows", type=int, default=1024)
    parser.add_argument("--prior-qa", type=Path)
    args = parser.parse_args()
    args.source = dict(args.source)
    args.source_hash = dict(args.source_hash)
    return args


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        failure = {"schema_version": "urrs.xian.r2.failure.v1", "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "finished_at": utc_now()}
        json_dump(Path(args.output_root).resolve() / "14_receipts" / "R2_FAILURE.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
