#!/usr/bin/env python3
"""Render a bounded submeter-to-10m tie diagnostic for one fixed case.

This reads the user-provided 10m RGB rasters, actual-date registry and
submeter tiles. It does not alter or resample any source in place and does not
promote automatic shifts to corrections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry, unary_union
try:
    import fiona
except ImportError:  # pragma: no cover - exercised in the Pyogrio-only runtime
    fiona = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xian_r5.alignment.base_tie import (
    estimate_component_gradient_translation as _corr_component,
    estimate_gradient_translation as _corr_score,
    estimate_local_gradient_field as _corr_grid,
)
from xian_r5.alignment.field_io import direct_base_tie_field_from_epoch
from xian_r5.alignment.preview import (
    PREVIEW_QUALITY_CODES,
    render_candidate_alignment_preview,
)


STAGE2 = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2")
BASE_DIR = Path(r"F:\xian\xian底图\xian搴曞浘")
RAW_ROOT = Path(r"F:\xian\xian")
DATE_CSV = RAW_ROOT / "date_audit" / "tables" / "tile_actual_date.csv"
FIXED_MAP = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv"
FIXED_GPKG = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_GEOMETRY.gpkg"
R4_FREEZE = STAGE2 / "13_receipts" / "R4_FREEZE_STAGE2.json"
SCENE = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
GRID_ORIGIN_X, GRID_ORIGIN_Y = 420534, 311895
GRID_ORIGIN_LON, GRID_ORIGIN_LAT = 108.75778198242188, 34.16130065917969
GRID_STEP_DEG = 0.0006866455078125
TO_UTM = Transformer.from_crs(4326, 32649, always_xy=True)
ROOF_MASK_RASTERIZATION_POLICIES = ("all_touched", "pixel_center")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rasterize_exclusion_mask(geometries, out_shape, transform, policy: str) -> np.ndarray:
    """Rasterize a diagnostic roof-exclusion mask under an explicit pixel rule."""
    if policy not in ROOF_MASK_RASTERIZATION_POLICIES:
        raise ValueError(
            f"policy must be one of {ROOF_MASK_RASTERIZATION_POLICIES}, got {policy!r}")
    return rasterize(
        geometries,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=(policy == "all_touched"),
    )


def _pixel_window_for_bounds(bounds, transform, width: int, height: int) -> Window:
    """Return the integer pixel window covering bounds without GDAL transforms."""
    inverse = ~transform
    left, bottom, right, top = bounds
    pixel_corners = [
        (inverse.a * x + inverse.b * y + inverse.c,
         inverse.d * x + inverse.e * y + inverse.f)
        for x, y in ((left, bottom), (left, top), (right, bottom), (right, top))
    ]
    col0 = max(0, math.floor(min(point[0] for point in pixel_corners)))
    row0 = max(0, math.floor(min(point[1] for point in pixel_corners)))
    col1 = min(width, math.ceil(max(point[0] for point in pixel_corners)))
    row1 = min(height, math.ceil(max(point[1] for point in pixel_corners)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("Requested bounds do not overlap the raster grid")
    return Window(col0, row0, col1 - col0, row1 - row0)


def _write_candidate_alignment_tifs(
    output_dir: Path,
    stem: str,
    preview: Any,
    *,
    crs: Any,
    transform: Any,
    actual_date: str,
    reference_year: int,
    field_id: str,
) -> tuple[Path, Path]:
    rgb_path = output_dir / f"{stem}_ALIGNMENT_PREVIEW_CANDIDATE_ONLY.tif"
    quality_path = output_dir / f"{stem}_ALIGNMENT_QUALITY_CANDIDATE_ONLY.tif"
    height, width = preview.quality.shape
    profile = {
        "driver": "GTiff", "width": width, "height": height,
        "crs": crs, "transform": transform, "compress": "deflate",
    }
    with rasterio.open(rgb_path, "w", count=4, dtype="uint8", **profile) as dst:
        dst.write(preview.rgba)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green,
                           ColorInterp.blue, ColorInterp.alpha)
        dst.update_tags(
            PRODUCT_KIND="IMAGE_ALIGNMENT_PREVIEW_CANDIDATE_ONLY",
            MOVING_ACTUAL_DATE=actual_date,
            REFERENCE_LABEL=f"Xian_{reference_year}_TrueColor",
            REFERENCE_ACQUISITION_DATE_STATUS="UNKNOWN_ANNUAL_FILE_LABEL_ONLY",
            ALIGNMENT_FIELD_ID=field_id,
            POSITIONAL_UNCERTAINTY="UNKNOWN_NOT_ESTIMATED",
            POSITIONAL_ACCURACY_CLAIMED="NO",
            ROOF_TO_GROUND_CORRECTION="NO",
            FORMAL_CORRECTION_ACCEPTED="NO",
            QUALITY_RASTER=str(quality_path),
        )
    with rasterio.open(quality_path, "w", count=1, dtype="uint8",
                       nodata=255, **profile) as dst:
        dst.write(preview.quality, 1)
        dst.update_tags(
            PRODUCT_KIND="IMAGE_ALIGNMENT_QUALITY_CANDIDATE_ONLY",
            MOVING_ACTUAL_DATE=actual_date,
            REFERENCE_LABEL=f"Xian_{reference_year}_TrueColor",
            REFERENCE_ACQUISITION_DATE_STATUS="UNKNOWN_ANNUAL_FILE_LABEL_ONLY",
            ALIGNMENT_FIELD_ID=field_id,
            QUALITY_CODES=json.dumps(PREVIEW_QUALITY_CODES, sort_keys=True),
            POSITIONAL_UNCERTAINTY="UNKNOWN_NOT_ESTIMATED",
            POSITIONAL_ACCURACY_CLAIMED="NO",
            INTERPOLATION="NONE",
            FIELD_ID=field_id,
            FORMAL_CORRECTION_ACCEPTED="NO",
            ROOF_TO_GROUND_CORRECTION="NO",
        )
    return rgb_path, quality_path


def _read_vector_features(path: Path, layer: str | None = None, *, bbox=None):
    """Read a bounded vector layer through Fiona or the installed Pyogrio stack."""
    if fiona is not None:
        with fiona.open(path, layer=layer) as source:
            crs = source.crs_wkt or source.crs
            features = source.filter(bbox=bbox) if bbox is not None else iter(source)
            records = [{
                "properties": dict(feature["properties"]),
                "geometry": feature.get("geometry"),
            } for feature in features]
        return crs, records

    try:
        import pyogrio
    except ImportError as exc:  # pragma: no cover - environment-specific guard
        raise RuntimeError("Base-tie diagnostics require Fiona or Pyogrio") from exc

    frame = pyogrio.read_dataframe(path, layer=layer, bbox=bbox, use_arrow=False)
    geometry_column = frame.geometry.name
    fields = [name for name in frame.columns if name != geometry_column]
    records = []
    for _, row in frame.iterrows():
        records.append({
            "properties": {name: row[name] for name in fields},
            "geometry": row[geometry_column],
        })
    return frame.crs, records


def _vector_layer_crs(path: Path, layer: str | None = None):
    if fiona is not None:
        with fiona.open(path, layer=layer) as source:
            value = source.crs_wkt or source.crs
    else:
        import pyogrio
        info = pyogrio.read_info(path, layer=layer)
        value = info.get("crs")
    if not value:
        raise ValueError(f"Roof-exclusion layer has no CRS: {path}")
    return CRS.from_user_input(value)


def _vector_component_hashes(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    if path.suffix.casefold() == ".shp":
        components = [path.with_suffix(suffix) for suffix in
                      (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix")]
    else:
        components = [path, *(Path(f"{path}{suffix}") for suffix in
                              ("-wal", "-shm", "-journal"))]
    existing = [component for component in components if component.is_file()]
    if not path.is_file():
        raise FileNotFoundError(path)
    return {str(component.resolve()): _sha256(component)
            for component in sorted(existing)}


def _projected_bounds_to_source(bounds, source_crs: CRS) -> tuple[float, float, float, float]:
    transformed = transform_bounds(
        CRS.from_epsg(32649), source_crs, *bounds, densify_pts=21,
    )
    return tuple(float(value) for value in transformed)


def _project_polygon_features(features: list[dict], source_crs: CRS,
                              target_bounds: tuple[float, float, float, float]):
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(32649), always_xy=True)
    clip = box(*target_bounds)
    projected = []
    invalid_count = 0
    polygon_count = 0
    for feature in features:
        geometry = _feature_geometry(feature.get("geometry"))
        if (geometry is None or geometry.is_empty or not geometry.is_valid
                or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
            invalid_count += 1
            continue
        candidate = transform_geometry(transformer.transform, geometry)
        if candidate.is_empty or not candidate.is_valid:
            invalid_count += 1
            continue
        polygon_count += 1
        candidate = candidate.intersection(clip)
        if not candidate.is_empty and candidate.area > 0:
            projected.append(candidate)
    return projected, {
        "feature_count_read": len(features),
        "valid_polygon_count": polygon_count,
        "invalid_or_nonpolygon_count": invalid_count,
        "overlap_count_in_target_bounds": len(projected),
    }


def _feature_geometry(value):
    if value is None:
        return None
    return value if hasattr(value, "geom_type") else shape(value)


def _load_case(case_id: str):
    with FIXED_MAP.open("r", encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["case_id"] == case_id and row.get("pixel_available", "").lower() == "true"]
    date_values = {}
    for row in rows:
        if row.get("actual_date", "").startswith("20"):
            date_values.setdefault(row["epoch_label"], set()).add(row["actual_date"])
    mixed = {epoch: sorted(values) for epoch, values in date_values.items() if len(values) != 1}
    if mixed:
        raise ValueError(f"Fixed case crosses actual dates within an epoch; split by date before mosaicking: {mixed}")
    dates = {epoch: next(iter(values)) for epoch, values in date_values.items()}
    centers = {}
    for row in rows:
        if row.get("tile_x") and row.get("tile_y"):
            centers.setdefault(row["epoch_label"], set()).add((int(row["tile_x"]), int(row["tile_y"])))
    geometries = {epoch: [] for epoch in dates}
    _, features = _read_vector_features(FIXED_GPKG, "fixed_case_polygons")
    for feature in features:
        props = feature["properties"]
        geometry = _feature_geometry(feature["geometry"])
        if (geometry is not None and props["case_id"] == case_id
                and props["role"] == "TARGET"):
            geometries.setdefault(str(props["epoch_label"]), []).append(geometry)
    return rows, dates, centers, geometries


def _tile_catalog(epochs: set[str], wanted: set[tuple[int, int]]) -> dict[tuple[str, int, int], str | None]:
    catalog = {}
    scene_to_epoch = {SCENE[e]: e for e in epochs}
    x_values = [x for x, _ in wanted]
    y_values = [y for _, y in wanted]
    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    with DATE_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            epoch = scene_to_epoch.get(row["task_name"])
            if epoch is None:
                continue
            x, y = int(row["tile_x"]), int(row["tile_y"])
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
            key = (epoch, x, y)
            date = row["actual_capture_date"] or None
            pixel = RAW_ROOT / row["task_name"] / "20" / str(x) / f"gesh_{x}_{y}_20.jpg"
            value = date if row["tile_present"] == "1" and date and pixel.is_file() else None
            if key in catalog and catalog[key] != value:
                catalog[key] = None
            else:
                catalog[key] = value
    return catalog


def _tile_image(epoch: str, x: int, y: int) -> np.ndarray | None:
    path = RAW_ROOT / SCENE[epoch] / "20" / str(x) / f"gesh_{x}_{y}_20.jpg"
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return rgb if rgb.shape == (256, 256, 3) else None
    except OSError:
        return None


def _summarize_cli_epoch(epoch: dict) -> dict:
    """Keep CLI receipts compact; complete source-component lineage stays in JSON."""
    registration = epoch.get("registration_diagnostic") or {}
    local = epoch.get("local_spatial_tie_diagnostic") or {}
    preview = epoch.get("candidate_alignment_preview") or {}
    local_cells = local.get("cells") or []
    preview_cells = preview.get("cells") or []
    scope_counts = Counter(
        str(cell.get("source_component_scope"))
        for cell in preview_cells if cell.get("source_component_scope")
    )
    return {
        "actual_date": epoch.get("actual_date"),
        "exact_date_coverage_fraction": epoch.get("submeter_tiles", {}).get(
            "exact_date_coverage_fraction"),
        "registration_backend": registration.get("preprocessing_backend"),
        "global_tie": registration.get("best"),
        "local_cell_count": len(local_cells),
        "local_numeric_tie_count": sum(cell.get("best") is not None for cell in local_cells),
        "local_field_interpolated": local.get("field_interpolated", False),
        "local_field_applied": local.get("field_applied", False),
        "candidate_preview_status": preview.get("status"),
        "candidate_numeric_cell_count": preview.get("candidate_cells_with_numeric_vectors", 0),
        "candidate_rendered_cell_count": sum(
            bool(cell.get("preview_candidate_rendered")) for cell in preview_cells),
        "component_scope_counts": dict(sorted(scope_counts.items())),
        "quality_pixel_counts": preview.get("quality_pixel_counts", {}),
        "correction_accepted": preview.get("correction_accepted", False),
        "field_applied": preview.get("field_applied", False),
        "roof_to_ground_correction": preview.get("roof_to_ground_correction", False),
    }


def _source_component_local_tie_diagnostic(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    reference_valid_mask: np.ndarray,
    moving_valid_mask: np.ndarray,
    base_transform: Any,
    source_tile_entries: list[dict],
    *,
    actual_date: str,
    radius: int,
    min_samples: int,
) -> dict:
    """Estimate candidate ties independently within exact-date source tiles."""
    from affine import Affine

    reference = np.asarray(reference_rgb)
    moving = np.asarray(moving_rgb)
    if (reference.ndim != 3 or reference.shape[2] != 3
            or moving.shape != reference.shape):
        raise ValueError("Source-component tie images must share an RGB grid")
    if (np.asarray(reference_valid_mask).shape != reference.shape[:2]
            or np.asarray(moving_valid_mask).shape != reference.shape[:2]):
        raise ValueError("Source-component tie masks must share the image grid")
    if radius < 1 or min_samples < 3:
        raise ValueError("Source-component tie search settings are invalid")

    transform = (base_transform if isinstance(base_transform, Affine)
                 else Affine(*base_transform[:6]))
    height, width = reference.shape[:2]
    corners = [transform * point for point in
               ((0, 0), (width, 0), (width, height), (0, height))]
    reference_window = box(
        min(point[0] for point in corners), min(point[1] for point in corners),
        max(point[0] for point in corners), max(point[1] for point in corners))
    cells = []
    skipped_without_pixels = 0
    seen_tile_ids = set()
    for entry in source_tile_entries:
        if entry.get("actual_date") != actual_date:
            continue
        tile_id = str(entry.get("tile_id") or "")
        if not tile_id or tile_id in seen_tile_ids:
            raise ValueError(f"Missing or duplicate exact-date source tile ID: {tile_id!r}")
        seen_tile_ids.add(tile_id)
        polygon_value = entry.get("support_polygon_epsg32649")
        if not isinstance(polygon_value, dict):
            raise ValueError(f"Source tile lacks projected support geometry: {tile_id}")
        source_support = shape(polygon_value)
        if (source_support.is_empty or not source_support.is_valid
                or source_support.geom_type not in {"Polygon", "MultiPolygon"}):
            raise ValueError(f"Invalid projected source-tile support: {tile_id}")
        support = source_support.intersection(reference_window)
        if support.is_empty or support.area <= 0:
            continue
        component_mask = rasterize(
            [(mapping(support), 1)], out_shape=(height, width),
            transform=transform, fill=0, dtype="uint8", all_touched=False,
        ).astype(bool)
        if not component_mask.any():
            skipped_without_pixels += 1
            continue

        tie = _corr_component(
            reference, moving, reference_valid_mask, moving_valid_mask,
            component_mask, radius=radius, min_samples=min_samples,
        )
        best, runner_up = tie["best"], tie["runner_up"]
        cells.append({
            "cell_id": "t" + tile_id.replace(":", "_"),
            "tile_id": tile_id,
            "actual_date": actual_date,
            "source_tile_path": entry.get("path"),
            "source_tile_sha256": entry.get("sha256"),
            "base_pixel_window": tie["base_pixel_window"],
            "support_polygon_epsg32649": mapping(support),
            "valid_pixel_count": int((np.asarray(reference_valid_mask, dtype=bool)
                                       & component_mask).sum()),
            "component_support_pixel_count": tie["component_support_pixel_count"],
            "component_interior_pixel_count": tie["component_interior_pixel_count"],
            "maximum_masked_samples_available": tie["maximum_masked_samples_available"],
            "minimum_samples_required": tie["minimum_samples_required"],
            "sample_support_status": (
                "NUMERIC_PEAK_AVAILABLE" if best else
                "INSUFFICIENT_MASKED_SAMPLES" if tie["sample_support_limited"] else
                "NO_VALID_CORRELATION_SCORE"),
            "component_local_tie_status": tie["component_local_tie_status"],
            "best": best,
            "runner_up": runner_up,
            "peak_margin": (best["corr"] - runner_up["corr"]
                            if best and runner_up else None),
            "status": ("COMPONENT_TILE_TIE_CANDIDATE_NOT_APPLIED"
                       if best else "NO_VALID_COMPONENT_TILE_TIE"),
        })

    return {
        "status": "DIAGNOSTIC_ONLY_NOT_ACCEPTED",
        "method": "EXACT_SOURCE_TILE_COMPONENT_GRADIENT_NCC",
        "support_mode": "EXACT_SOURCE_TILE_COMPONENTS",
        "actual_date": actual_date,
        "search_radius_px": radius,
        "search_radius_10m_pixels": radius,
        "minimum_samples_required": min_samples,
        "source_tile_count_exact_date": len(seen_tile_ids),
        "component_count_with_base_pixels": len(cells),
        "component_tiles_skipped_without_base_pixels": skipped_without_pixels,
        "numeric_component_peak_count": sum(cell["best"] is not None for cell in cells),
        "field_interpolated": False,
        "field_applied": False,
        "position_accuracy_claimed": False,
        "positional_uncertainty": "UNKNOWN_NOT_ESTIMATED",
        "roof_to_ground_correction": False,
        "cells": cells,
    }


def run(case_id: str, output: Path, radius_tiles: int, building_buffer_m: float = 5.0,
        local_grid_size: int = 0, base_reference_year: int | None = None,
        render_alignment_preview_candidate_only: bool = False,
        local_search_radius_px: int = 3,
        local_minimum_samples: int = 100,
        source_component_local_ties: bool = False,
        external_roof_exclusion_path: Path | None = None,
        external_roof_exclusion_layer: str | None = None,
        external_roof_exclusion_epoch_label: str | None = None,
        external_roof_exclusion_buffer_m: float | None = None,
        roof_mask_rasterization_policy: str = "all_touched") -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if building_buffer_m < 0:
        raise ValueError("building_buffer_m must be non-negative")
    if roof_mask_rasterization_policy not in ROOF_MASK_RASTERIZATION_POLICIES:
        raise ValueError(
            "roof_mask_rasterization_policy must be one of "
            f"{ROOF_MASK_RASTERIZATION_POLICIES}")
    if external_roof_exclusion_buffer_m is not None and external_roof_exclusion_buffer_m < 0:
        raise ValueError("external_roof_exclusion_buffer_m must be non-negative")
    if local_search_radius_px < 1:
        raise ValueError("local_search_radius_px must be positive")
    if local_minimum_samples < 3:
        raise ValueError("local_minimum_samples must be at least 3")
    if base_reference_year is not None and base_reference_year not in range(2017, 2023):
        raise ValueError("base_reference_year must be one of 2017..2022")
    if source_component_local_ties and local_grid_size:
        raise ValueError("Choose either uniform local grid or exact source-component ties")
    if (external_roof_exclusion_path is None) != (external_roof_exclusion_epoch_label is None):
        raise ValueError(
            "External roof exclusion requires both its path and user-reported source epoch label")
    if external_roof_exclusion_path is None and external_roof_exclusion_buffer_m is not None:
        raise ValueError("External roof-exclusion buffer requires an external roof-exclusion source")
    external_buffer_m = (building_buffer_m if external_roof_exclusion_buffer_m is None
                         else external_roof_exclusion_buffer_m)
    if render_alignment_preview_candidate_only and not (local_grid_size >= 1
                                                        or source_component_local_ties):
        raise ValueError("Candidate previews require local-grid or source-component ties")
    rows, dates, centers, geoms_by_epoch = _load_case(case_id)
    if not dates:
        raise ValueError(f"No date-resolved imagery rows for {case_id}")
    all_centers = {tile for values in centers.values() for tile in values}
    wanted = {(x + dx, y + dy) for x, y in all_centers
              for dx in range(-radius_tiles, radius_tiles + 1)
              for dy in range(-radius_tiles, radius_tiles + 1)}
    catalog = _tile_catalog(set(dates), wanted)
    freeze = json.loads(R4_FREEZE.read_text(encoding="utf-8"))
    geometries = [g for values in geoms_by_epoch.values() for g in values]
    case_geom = unary_union(geometries)
    case_bounds = case_geom.bounds
    xs = [x for x, _ in wanted]
    ys = [y for _, y in wanted]
    x0, y0 = min(xs), min(ys)
    nx, ny = max(xs) - x0 + 1, max(ys) - y0 + 1
    lon_w = GRID_ORIGIN_LON + (x0 - GRID_ORIGIN_X) * GRID_STEP_DEG
    lat_s = GRID_ORIGIN_LAT + (y0 - GRID_ORIGIN_Y) * GRID_STEP_DEG
    lon_e, lat_n = lon_w + nx * GRID_STEP_DEG, lat_s + ny * GRID_STEP_DEG
    from affine import Affine
    tl = TO_UTM.transform(lon_w, lat_n)
    tr = TO_UTM.transform(lon_e, lat_n)
    bl = TO_UTM.transform(lon_w, lat_s)
    br = TO_UTM.transform(lon_e, lat_s)
    source_transform = Affine((tr[0] - tl[0]) / (nx * 256), (bl[0] - tl[0]) / (ny * 256), tl[0],
                              (tr[1] - tl[1]) / (nx * 256), (bl[1] - tl[1]) / (ny * 256), tl[1])
    source_bounds = (min(tl[0], bl[0]), min(bl[1], br[1]), max(tr[0], br[0]), max(tl[1], tr[1]))
    local_buildings = []
    local_building_counts = {}
    mask_geometry_sources = {}
    for mask_epoch in sorted(dates):
        geometry_source = freeze["files"][f"geometry_{mask_epoch}"]
        mask_geometry_sources[mask_epoch] = {
            "path": geometry_source["path"], "sha256": geometry_source["sha256"],
            "freeze_status": "FROZEN_SOURCE_FROM_RECEIPT"}
        count = 0
        source_crs, features = _read_vector_features(
            Path(geometry_source["path"]), "buildings", bbox=source_bounds,
        )
        if CRS.from_user_input(source_crs).to_epsg() != 32649:
            raise ValueError(f"Unexpected R4 geometry CRS: {geometry_source['path']}")
        for feature in features:
            candidate = _feature_geometry(feature.get("geometry"))
            if candidate is not None and not candidate.is_empty and candidate.is_valid:
                local_buildings.append(candidate.buffer(building_buffer_m))
                count += 1
        local_building_counts[mask_epoch] = count
    external_roof_geometries = []
    external_roof_exclusion_receipt = None
    external_roof_exclusion_hashes_before = {}
    if external_roof_exclusion_path is not None:
        external_roof_exclusion_path = Path(external_roof_exclusion_path).resolve()
        external_roof_exclusion_hashes_before = _vector_component_hashes(
            external_roof_exclusion_path)
        external_crs = _vector_layer_crs(
            external_roof_exclusion_path, external_roof_exclusion_layer)
        query_bounds = box(*source_bounds).buffer(external_buffer_m).bounds
        query_bbox = _projected_bounds_to_source(query_bounds, external_crs)
        source_crs, features = _read_vector_features(
            external_roof_exclusion_path, external_roof_exclusion_layer,
            bbox=query_bbox,
        )
        source_crs = CRS.from_user_input(source_crs)
        if source_crs != external_crs:
            raise ValueError("External roof-exclusion layer CRS changed between metadata and read")
        external_roof_geometries, geometry_counts = _project_polygon_features(
            features, source_crs, query_bounds,
        )
        external_roof_exclusion_receipt = {
            "source_path": str(external_roof_exclusion_path),
            "source_layer": external_roof_exclusion_layer,
            "source_crs": source_crs.to_string(),
            "source_epoch_label": external_roof_exclusion_epoch_label,
            "source_epoch_label_semantics": "USER_REPORTED_LABEL_NOT_EXACT_ACQUISITION_DATE",
            "geometry_semantics": "USER_REPORTED_ROOF_BOUNDARY_FROM_IMAGE_UNVERIFIED",
            "use_semantics": "EXCLUSION_MASK_ONLY_NOT_GROUND_CONTROL_OR_FOOTPRINT_TRUTH",
            "buffer_m": external_buffer_m,
            "query_bbox_source_crs": list(query_bbox),
            "query_target_bounds_epsg32649": list(query_bounds),
            "source_component_hashes_sha256": external_roof_exclusion_hashes_before,
            **geometry_counts,
            "buffered_polygon_count": len(external_roof_geometries),
        }
    output.mkdir(parents=True)
    preview_dir = output / "alignment_preview_candidate_only"
    if render_alignment_preview_candidate_only:
        preview_dir.mkdir()
    outputs = {}
    base_hash_cache: dict[Path, str] = {}
    preview_base_grid_signature = None
    for epoch, actual_date in sorted(dates.items()):
        # Keep only tiles on the exact target acquisition date. Mixed-date mosaics
        # are represented as holes, never silently blended.
        mosaic = np.full((ny * 256, nx * 256, 3), np.nan, dtype=np.float32)
        validity = np.zeros((ny * 256, nx * 256), dtype=np.uint8)
        included = []
        counts = Counter()
        for y in range(y0 + ny - 1, y0 - 1, -1):
            for x in range(x0, x0 + nx):
                date = catalog.get((epoch, x, y))
                counts[str(date or "MISSING_OR_DATE_UNKNOWN")] += 1
                if date != actual_date:
                    continue
                image = _tile_image(epoch, x, y)
                if image is None:
                    continue
                col, row = x - x0, (y0 + ny - 1) - y
                ys0, xs0 = row * 256, col * 256
                mosaic[ys0:ys0 + 256, xs0:xs0 + 256] = image.astype(np.float32)
                validity[ys0:ys0 + 256, xs0:xs0 + 256] = 1
                tile_path = RAW_ROOT / SCENE[epoch] / "20" / str(x) / f"gesh_{x}_{y}_20.jpg"
                tile_col0, tile_row0 = col * 256, row * 256
                tile_ring = [
                    [float(value) for value in source_transform * (tile_col0, tile_row0)],
                    [float(value) for value in source_transform * (tile_col0 + 256, tile_row0)],
                    [float(value) for value in source_transform * (tile_col0 + 256, tile_row0 + 256)],
                    [float(value) for value in source_transform * (tile_col0, tile_row0 + 256)],
                ]
                tile_ring.append(tile_ring[0])
                included.append({
                    "tile_id": f"{x}:{y}", "actual_date": actual_date,
                    "path": str(tile_path), "sha256": _sha256(tile_path),
                    "source_pixel_window": [tile_col0, tile_row0, 256, 256],
                    "support_polygon_epsg32649": {
                        "type": "Polygon", "coordinates": [tile_ring]},
                })
        geometry_source = freeze["files"][f"geometry_{epoch}"]
        reference_year = base_reference_year if base_reference_year is not None else int(actual_date[:4])
        base_path = BASE_DIR / f"Xian_{reference_year}_TrueColor.tif"
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        with rasterio.open(base_path) as base:
            if base.crs.to_epsg() != 32649 or abs(base.res[0] - 10) > 1e-6 or abs(base.res[1] - 10) > 1e-6:
                raise ValueError(f"Unexpected base profile: {base_path}")
            if render_alignment_preview_candidate_only:
                current_grid_signature = (
                    str(base.crs), tuple(float(value) for value in base.transform),
                    int(base.width), int(base.height), int(base.count),
                    tuple(float(value) for value in base.res),
                )
                if (preview_base_grid_signature is not None
                        and current_grid_signature != preview_base_grid_signature):
                    raise ValueError(
                        "Candidate preview annual references must share one exact full raster grid")
                preview_base_grid_signature = current_grid_signature
            window = _pixel_window_for_bounds(
                source_bounds, base.transform, base.width, base.height,
            )
            base_rgb = base.read(window=window, boundless=False)
            base_rgb = np.moveaxis(base_rgb, 0, 2)
            base_transform = base.window_transform(window)
            high10_float = np.full(base_rgb.shape, np.nan, dtype=np.float32)
            valid10 = np.zeros(base_rgb.shape[:2], dtype=np.uint8)
            for band in range(3):
                reproject(source=mosaic[:, :, band], destination=high10_float[:, :, band],
                          src_transform=source_transform, src_crs="EPSG:32649",
                          dst_transform=base_transform, dst_crs=base.crs,
                          resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
            reproject(source=validity, destination=valid10, src_transform=source_transform,
                      src_crs="EPSG:32649", dst_transform=base_transform, dst_crs=base.crs,
                      resampling=Resampling.nearest, dst_nodata=0)
            high10 = np.uint8(np.nan_to_num(np.clip(high10_float, 0, 255), nan=0.0))
            roof_mask = rasterize(((g, 1) for g in geoms_by_epoch.get(epoch, [])), out_shape=base_rgb.shape[:2],
                                  transform=base_transform, fill=0, dtype="uint8")
            r4_building_mask = (
                _rasterize_exclusion_mask(
                    ((g, 1) for g in local_buildings), base_rgb.shape[:2],
                    base_transform, roof_mask_rasterization_policy)
                if local_buildings else np.zeros(base_rgb.shape[:2], dtype=np.uint8)
            )
            external_buffered = [geometry.buffer(external_buffer_m)
                                 for geometry in external_roof_geometries]
            external_building_mask = (
                _rasterize_exclusion_mask(
                    ((g, 1) for g in external_buffered), base_rgb.shape[:2],
                    base_transform, roof_mask_rasterization_policy)
                if external_buffered else np.zeros(base_rgb.shape[:2], dtype=np.uint8)
            )
            all_building_mask = ((r4_building_mask > 0) | (external_building_mask > 0))
            valid_reference = (valid10 > 0) & np.any(base_rgb > 0, axis=2)
            valid_without_external_mask = valid_reference & (r4_building_mask == 0)
            valid = valid_without_external_mask & (external_building_mask == 0)
            tie = _corr_score(base_rgb, high10, valid)
            if source_component_local_ties:
                local_tie = _source_component_local_tie_diagnostic(
                    base_rgb, high10, valid, valid, base_transform, included,
                    actual_date=actual_date, radius=local_search_radius_px,
                    min_samples=local_minimum_samples,
                )
            else:
                local_tie = (_corr_grid(base_rgb, high10, valid, grid_size=local_grid_size,
                                        radius=local_search_radius_px,
                                        min_samples=local_minimum_samples)
                             if local_grid_size else None)
            base_sha256 = base_hash_cache.get(base_path)
            if base_sha256 is None:
                base_sha256 = _sha256(base_path)
                base_hash_cache[base_path] = base_sha256
            base_grid_receipt = {
                "crs": str(base.crs), "resolution": list(base.res),
                "window": [int(window.col_off), int(window.row_off),
                           int(window.width), int(window.height)],
                "transform": list(base_transform),
            }
            preview_receipt = None
            if render_alignment_preview_candidate_only:
                field = direct_base_tie_field_from_epoch(
                    epoch,
                    {"actual_date": actual_date, "base_path": str(base_path),
                     "base_sha256": base_sha256, "base_grid": base_grid_receipt,
                     "submeter_tiles": {"tile_ids": included},
                     "local_spatial_tie_diagnostic": local_tie},
                    reference_year=reference_year,
                    diagnostic_path=output / "BASE_TIE_DIAGNOSTIC.json",
                )
                preview = render_candidate_alignment_preview(
                    mosaic, validity, source_transform, base_rgb.shape[:2],
                    base_transform, field,
                )
                stem = f"{epoch}_{actual_date}"
                rgb_path, quality_path = _write_candidate_alignment_tifs(
                    preview_dir, stem, preview, crs=base.crs,
                    transform=base_transform, actual_date=actual_date,
                    reference_year=reference_year, field_id=field.field_id,
                )
                quality_counts = {
                    str(code): int((preview.quality == code).sum())
                    for code in PREVIEW_QUALITY_CODES
                }
                preview_receipt = {
                    "status": "GENERATED_CANDIDATE_ONLY_NOT_ACCEPTED",
                    "rgb_tif": str(rgb_path), "quality_tif": str(quality_path),
                    "field_id": field.field_id,
                    "candidate_cells_with_numeric_vectors": sum(
                        cell.correction_dx_m_candidate is not None for cell in field.cells),
                    "candidate_cells_rendered": sum(
                        bool(cell["preview_candidate_rendered"])
                        for cell in preview.cells),
                    "quality_pixel_counts": quality_counts,
                    "quality_codes": PREVIEW_QUALITY_CODES,
                    "cells": list(preview.cells),
                    "position_accuracy_claimed": False,
                    "positional_uncertainty": "UNKNOWN_NOT_ESTIMATED",
                    "correction_accepted": False,
                    "roof_to_ground_correction": False,
                }
            scale = 4
            def enlarge(img):
                return np.asarray(Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.Resampling.NEAREST))
            left = enlarge(base_rgb).copy()
            middle = enlarge(high10).copy()
            blend = np.uint8(np.clip(0.5 * base_rgb.astype(np.float32) + 0.5 * high10.astype(np.float32), 0, 255))
            blended = enlarge(blend).copy()
            roof_big = enlarge(roof_mask * 255).copy()
            for panel in (left, middle, blended):
                panel[roof_big > 0] = np.array([255, 40, 20], dtype=np.uint8)
            canvas = Image.new("RGB", (left.shape[1] * 3, left.shape[0] + 32), (245, 245, 245))
            canvas.paste(Image.fromarray(left), (0, 32))
            canvas.paste(Image.fromarray(middle), (left.shape[1], 32))
            canvas.paste(Image.fromarray(blended), (left.shape[1] * 2, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"10m base {reference_year} / actual-date imagery {actual_date} / red: R4 roof candidates", fill=(0, 0, 0))
            panel_names = ["10m baseline", "submeter projected to same 10m grid", "50:50 overlay"]
            for index, name in enumerate(panel_names):
                draw.text((index * left.shape[1] + 8, 32 + left.shape[0] - 18), name, fill=(255, 255, 0))
            png_path = output / f"{epoch}_{actual_date}_base_tie.png"
            canvas.save(png_path)
            outputs[epoch] = {
                "actual_date": actual_date, "base_reference_year": reference_year,
                "base_path": str(base_path),
                "base_grid": base_grid_receipt,
                "submeter_tiles": {"wanted": nx * ny, "exact_date_read": len(included),
                                   "exact_date_coverage_fraction": len(included) / (nx * ny),
                                   "date_histogram": dict(counts), "tile_ids": included},
                "base_sha256": base_sha256,
                "case_geometry_bounds_epsg32649": list(case_bounds),
                "mosaic_bounds_epsg32649": list(source_bounds),
                "r4_roof_exclusion": {"source_path": geometry_source["path"],
                                      "frozen_sha256": geometry_source["sha256"],
                                      "masked_epochs": sorted(dates),
                                      "mask_geometry_sources": mask_geometry_sources,
                                      "local_buildings_found_by_epoch": local_building_counts,
                                      "local_buildings_found_total": sum(local_building_counts.values()),
                                      "buffer_m": building_buffer_m,
                                      "rasterization_policy": roof_mask_rasterization_policy,
                                      "excluded_10m_pixels": int((r4_building_mask > 0).sum()),
                                      "ground_pixels_used_without_external_mask": int(valid_without_external_mask.sum())},
                "external_roof_exclusion": (
                    {**external_roof_exclusion_receipt,
                     "excluded_10m_pixels": int((external_building_mask > 0).sum()),
                     "rasterization_policy": roof_mask_rasterization_policy,
                     "ground_pixels_used_after_r4_and_external_mask": int(valid.sum())}
                    if external_roof_exclusion_receipt is not None else None
                ),
                "combined_roof_exclusion": {
                    "excluded_10m_pixels": int(all_building_mask.sum()),
                    "ground_pixels_used": int(valid.sum()),
                    "mask_is_ground_control": False,
                },
                "registration_diagnostic": tie,
                "local_spatial_tie_diagnostic": local_tie,
                "review_png": str(png_path),
                "candidate_alignment_preview": preview_receipt,
            }
    if external_roof_exclusion_path is not None:
        external_roof_exclusion_hashes_after = _vector_component_hashes(
            external_roof_exclusion_path)
        if external_roof_exclusion_hashes_after != external_roof_exclusion_hashes_before:
            raise RuntimeError("External roof-exclusion source changed during the diagnostic")
    else:
        external_roof_exclusion_hashes_after = {}
    report = {"schema_version": "xian_r5.base_tie_diagnostic.v1", "case_id": case_id,
              "base_dir": str(BASE_DIR), "scope": "FIXED_CASE_LOCAL_ONLY",
              "parameters": {"radius_tiles": radius_tiles,
                             "roof_exclusion_buffer_m": building_buffer_m,
                             "roof_mask_uses_all_observed_epochs": True,
                             "roof_mask_rasterization_policy": roof_mask_rasterization_policy,
                             "roof_mask_pixel_rule": (
                                 "ANY_PIXEL_TOUCHED_BY_GEOMETRY" if roof_mask_rasterization_policy == "all_touched"
                                 else "PIXEL_CENTER_INSIDE_GEOMETRY"),
                             "external_roof_exclusion_path": (
                                 str(external_roof_exclusion_path)
                                 if external_roof_exclusion_path is not None else None),
                             "external_roof_exclusion_layer": external_roof_exclusion_layer,
                             "external_roof_exclusion_source_epoch_label": external_roof_exclusion_epoch_label,
                             "external_roof_exclusion_buffer_m": (
                                 external_buffer_m
                                 if external_roof_exclusion_receipt is not None else None),
                             "external_roof_exclusion_is_mask_only": (
                                 external_roof_exclusion_receipt is not None),
                             "local_grid_size": local_grid_size,
                             "local_support_mode": (
                                 "EXACT_SOURCE_TILE_COMPONENTS" if source_component_local_ties
                                 else "UNIFORM_GRID" if local_grid_size else "DISABLED"),
                             "local_search_radius_px": local_search_radius_px,
                             "local_minimum_samples": local_minimum_samples,
                             "base_reference_year": base_reference_year,
                             "local_field_interpolated_or_applied": False,
                             "candidate_alignment_preview_only": render_alignment_preview_candidate_only,
                             "candidate_preview_reference_policy": (
                                 "FIXED_ANNUAL_REFERENCE" if base_reference_year is not None
                                 else "ACTUAL_DATE_YEAR_MATCHED_WITH_EXACT_SHARED_GRID_VALIDATION"),
                             "candidate_preview_full_grid_signature": (
                                 list(preview_base_grid_signature)
                                 if preview_base_grid_signature is not None else None)},
              "mask_semantics": (
                  "BUFFERED_UNION_OF_FROZEN_R4_GEOMETRIES_ACROSS_ACTUAL_DATES_PLUS_USER_REPORTED_ROOF_BOUNDARIES_SOURCE_LABEL_"
                  f"{external_roof_exclusion_epoch_label}_MASK_ONLY;NOT_GROUND_CONTROL_OR_PROOF_OF_STABLE_GROUND"
                  if external_roof_exclusion_receipt is not None else
                  "BUFFERED_UNION_OF_FROZEN_R4_BUILDING_GEOMETRIES_ACROSS_ACTUAL_DATE_EPOCHS;NOT_PROOF_OF_STABLE_GROUND"
              ),
              "inputs_sha256": {"fixed_case_map": _sha256(FIXED_MAP), "fixed_case_geometry": _sha256(FIXED_GPKG),
                                "actual_date_registry": _sha256(DATE_CSV), "r4_freeze_receipt": _sha256(R4_FREEZE),
                                "external_roof_exclusion_before": external_roof_exclusion_hashes_before,
                                "external_roof_exclusion_after": external_roof_exclusion_hashes_after},
              "geometry_semantics": "R4_roof_candidates_not_ground_truth",
              "external_roof_exclusion_hash_recheck": (
                  "PASS" if external_roof_exclusion_receipt is not None else "NOT_APPLICABLE"),
              "automatic_registration_promoted": False, "epochs": outputs}
    (output / "BASE_TIE_DIAGNOSTIC.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="B03-TEX04")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-tiles", type=int, default=5)
    parser.add_argument("--building-buffer-m", type=float, default=5.0)
    parser.add_argument("--local-grid-size", type=int, default=0,
                        help="Optional square block grid for coarse local-tie diagnostics; 0 disables it.")
    parser.add_argument("--source-component-local-ties", action="store_true",
                        help="Run candidate-only local ties independently on exact-date source-tile support polygons.")
    parser.add_argument("--local-search-radius-px", type=int, default=3,
                        help="Local-tie search radius in 10m reference pixels; default 3.")
    parser.add_argument("--local-minimum-samples", type=int, default=100,
                        help="Minimum valid 10m pixels for a local tie; lower values are diagnostic-only sensitivity candidates, never automatic corrections.")
    parser.add_argument("--base-reference-year", type=int, choices=range(2017, 2023), default=None,
                        help="Use one fixed 10m base raster for every actual date instead of year-matched bases.")
    parser.add_argument("--external-roof-exclusion-path", type=Path, default=None,
                        help="Optional user-reported roof-boundary layer used only to exclude suspected roofs from tie samples.")
    parser.add_argument("--external-roof-exclusion-layer", default=None,
                        help="Optional vector layer name; defaults to the dataset's first layer.")
    parser.add_argument("--external-roof-exclusion-epoch-label", default=None,
                        help="Required provenance label for the optional roof-mask source; not an exact acquisition date.")
    parser.add_argument("--external-roof-exclusion-buffer-m", type=float, default=None,
                        help="Optional separate buffer for the external roof mask; defaults to --building-buffer-m.")
    parser.add_argument("--roof-mask-rasterization-policy",
                        choices=ROOF_MASK_RASTERIZATION_POLICIES, default="all_touched",
                        help="Rasterization rule for R4 and external roof-exclusion masks. pixel_center is a candidate-support sensitivity diagnostic only; the default preserves prior behavior.")
    parser.add_argument("--render-alignment-preview-candidate-only", action="store_true",
                        help="Write transparent RGB and quality GeoTIFFs from unaccepted local ties; requires one local tie mode and exactly co-gridded annual references (fixed-year or actual-date-year-matched).")
    args = parser.parse_args()
    if args.local_grid_size < 0:
        parser.error("--local-grid-size must be non-negative")
    if args.source_component_local_ties and args.local_grid_size:
        parser.error("--source-component-local-ties cannot be combined with --local-grid-size")
    result = run(args.case_id, args.output, args.radius_tiles,
                 building_buffer_m=args.building_buffer_m,
                 local_grid_size=args.local_grid_size,
                 base_reference_year=args.base_reference_year,
                 render_alignment_preview_candidate_only=args.render_alignment_preview_candidate_only,
                 local_search_radius_px=args.local_search_radius_px,
                 local_minimum_samples=args.local_minimum_samples,
                 source_component_local_ties=args.source_component_local_ties,
                 external_roof_exclusion_path=args.external_roof_exclusion_path,
                 external_roof_exclusion_layer=args.external_roof_exclusion_layer,
                 external_roof_exclusion_epoch_label=args.external_roof_exclusion_epoch_label,
                 external_roof_exclusion_buffer_m=args.external_roof_exclusion_buffer_m,
                 roof_mask_rasterization_policy=args.roof_mask_rasterization_policy)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "diagnostic_manifest": str((args.output / "BASE_TIE_DIAGNOSTIC.json").resolve()),
        "epochs": {
            key: _summarize_cli_epoch(value)
            for key, value in result["epochs"].items()
        },
    }, ensure_ascii=False, separators=(",", ":")))
