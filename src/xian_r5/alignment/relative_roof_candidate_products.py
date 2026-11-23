"""Export spatially local relative-roof alignment as candidate-only products.

These products are on the selected 10 m reference grid but are not ground
footprints and do not resolve absolute position. Unsupported cells are NoData
in the candidate masks; no displacement is interpolated across them.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.affinity import translate
from shapely.geometry import Polygon, mapping
from shapely.geometry.base import BaseGeometry


RELATIVE_QUALITY_CODES = {
    "relative_pose_cycle_candidate": 1,
    "relative_pose_tree_no_cycle_check": 2,
    "insufficient_relative_support": 3,
    "anchor_epoch_raw_reference_only": 4,
    "outside_reference_window": 255,
}
_SUPPORTED_POSE_STATUSES = {
    "RELATIVE_POSE_CYCLE_DIAGNOSTIC_CANDIDATE",
    "RELATIVE_POSE_TREE_NO_CYCLE_CHECK",
}
_FIELD_NODATA = -9999.0
_MASK_NODATA = 255


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_grid_window(
    reference_raster: Path,
    reference_raster_stack: Sequence[Path],
    roi_bounds: Sequence[float],
) -> tuple[dict[str, Any], Window, Any]:
    if len(roi_bounds) != 4:
        raise ValueError("ROI bounds must be xmin, ymin, xmax, ymax")
    xmin, ymin, xmax, ymax = (float(value) for value in roi_bounds)
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError("ROI bounds must be finite")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("ROI bounds must have positive area")
    reference_raster = Path(reference_raster)
    if not reference_raster.is_file():
        raise FileNotFoundError(reference_raster)

    grid_receipts = []
    with rasterio.open(reference_raster) as reference:
        if reference.crs is None or reference.crs.to_epsg() != 32649:
            raise ValueError("Reference raster must use EPSG:32649")
        transform = reference.transform
        if (not math.isclose(transform.a, 10.0, abs_tol=1e-8)
                or not math.isclose(transform.e, -10.0, abs_tol=1e-8)
                or not math.isclose(transform.b, 0.0, abs_tol=1e-10)
                or not math.isclose(transform.d, 0.0, abs_tol=1e-10)):
            raise ValueError("Reference raster must be an unrotated 10 m grid")
        if (xmin < reference.bounds.left or xmax > reference.bounds.right
                or ymin < reference.bounds.bottom or ymax > reference.bounds.top):
            raise ValueError("ROI is not fully covered by the reference raster")
        # Avoid rasterio's CRS-backed rowcol path here: the base grid has
        # already been validated as north-up, unrotated, and exactly 10 m.
        col_start = (xmin - transform.c) / transform.a
        col_end = (xmax - transform.c) / transform.a
        row_start = (transform.f - ymax) / -transform.e
        row_end = (transform.f - ymin) / -transform.e
        col0 = math.floor(col_start + 1e-9)
        row0 = math.floor(row_start + 1e-9)
        col1 = math.ceil(col_end - 1e-9)
        row1 = math.ceil(row_end - 1e-9)
        window = Window(col0, row0, col1 - col0, row1 - row0)
        if window.width < 1 or window.height < 1:
            raise ValueError("ROI produced an empty reference-grid window")
        output_transform = reference.window_transform(window)
        reference_grid = {
            "path": str(reference_raster),
            "sha256": _sha256(reference_raster),
            "crs": reference.crs.to_string(),
            "width": reference.width,
            "height": reference.height,
            "transform": list(transform),
            "window": {
                "column_offset": int(window.col_off),
                "row_offset": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
                "transform": list(output_transform),
            },
            "source_file_tag": reference.tags().get("SOURCE_FILE"),
            "per_pixel_source_component_date_and_view_angle": "UNKNOWN",
        }
        stack_paths = sorted({Path(path) for path in reference_raster_stack} | {reference_raster})
        for path in stack_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            with rasterio.open(path) as dataset:
                if (dataset.crs != reference.crs
                        or dataset.width != reference.width
                        or dataset.height != reference.height
                        or dataset.transform != reference.transform):
                    raise ValueError(f"Base raster grid differs from selected reference: {path}")
                grid_receipts.append({
                    "path": str(path), "sha256": _sha256(path),
                    "crs": dataset.crs.to_string(), "width": dataset.width,
                    "height": dataset.height, "transform": list(dataset.transform),
                    "source_file_tag": dataset.tags().get("SOURCE_FILE"),
                    "per_pixel_source_component_date_and_view_angle": "UNKNOWN",
                })
    reference_grid["verified_same_grid_base_rasters"] = grid_receipts
    return reference_grid, window, output_transform


def _write_raster(
    path: Path,
    array: np.ndarray,
    *,
    crs: str,
    transform: Any,
    nodata: int | float,
    tags: Mapping[str, Any],
    descriptions: Sequence[str] | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite relative-roof candidate: {path}")
    data = array[np.newaxis, ...] if array.ndim == 2 else array
    dtype = str(data.dtype)
    with rasterio.open(
        path, "w", driver="GTiff", width=int(data.shape[2]),
        height=int(data.shape[1]), count=int(data.shape[0]), dtype=dtype,
        crs=crs, transform=transform, nodata=nodata,
        compress="deflate", predictor=2 if dtype.startswith("float") else 1,
    ) as target:
        target.write(data)
        if descriptions:
            for index, description in enumerate(descriptions, start=1):
                target.set_band_description(index, description)
        target.update_tags(**{str(key): str(value) for key, value in tags.items()})


def _rasterize_or_fill(
    shapes: Sequence[tuple[Mapping[str, Any], int | float]],
    *,
    out_shape: tuple[int, int],
    transform: Any,
    fill: int | float,
    dtype: str,
) -> np.ndarray:
    if not shapes:
        return np.full(out_shape, fill, dtype=dtype)
    return rasterize(
        shapes, out_shape=out_shape, transform=transform,
        fill=fill, dtype=dtype, all_touched=False)


def write_relative_roof_candidate_products(
    output_dir: Path,
    reference_raster: Path,
    reference_raster_stack: Sequence[Path],
    roi_bounds: Sequence[float],
    local_roofs_by_epoch: Mapping[str, Sequence[Mapping[str, Any]]],
    geometry_source_paths: Mapping[str, Path | str],
    geometry_source_sha256: Mapping[str, str],
    actual_dates: Mapping[str, str],
    pose_graph: Mapping[str, Any],
    *,
    outline_source_path: str,
    outline_source_component_sha256: Mapping[str, str],
    outline_source_epoch_label: str,
    outline_source_geometry_semantics: str,
) -> dict[str, Any]:
    """Write relative-aligned roof candidates and quality fields on a 10 m grid.

    Only roofs wholly contained in a single supported pose-graph cell are
    translated. A translated geometry must also remain in that same cell.
    Candidate masks are NoData outside supported cells, so unsupported pixels
    cannot be mistaken for observed absence.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory must already exist: {output_dir}")
    if pose_graph.get("schema_version") != "xian_r5.local_relative_pose_graph_candidates.v1":
        raise ValueError("Unsupported relative pose-graph schema")
    if (pose_graph.get("absolute_ground_position_resolved") is not False
            or pose_graph.get("applied") is not False
            or pose_graph.get("interpolated") is not False
            or pose_graph.get("source_semantics")
            != "RELATIVE_TO_UNVERIFIED_STATIC_REFERENCE_ONLY"):
        raise ValueError("Relative pose graph must remain candidate-only and unapplied")
    if outline_source_geometry_semantics != "ROOF_BOUNDARY_FROM_IMAGE":
        raise ValueError("The external reference semantics must remain an image-derived roof boundary")
    if not outline_source_epoch_label:
        raise ValueError("A user-reported outline source epoch label is required")
    if (not outline_source_component_sha256
            or any(len(str(value)) != 64
                   or any(character not in "0123456789abcdefABCDEF" for character in str(value))
                   for value in outline_source_component_sha256.values())):
        raise ValueError("Outline source component SHA-256 receipts are required")

    epoch_to_date = {str(epoch): str(actual_date)
                     for epoch, actual_date in actual_dates.items()}
    dates = sorted(set(epoch_to_date.values()))
    anchor_date = str(pose_graph.get("anchor_id"))
    if anchor_date not in dates:
        raise ValueError("Pose-graph anchor is not one of the actual observation dates")
    if set(local_roofs_by_epoch) != set(epoch_to_date):
        raise ValueError("Roof geometry epochs do not match the actual-date registry")
    if set(geometry_source_paths) != set(epoch_to_date) or set(geometry_source_sha256) != set(epoch_to_date):
        raise ValueError("Geometry source lineage must cover each epoch exactly once")

    reference_grid, window, output_transform = _reference_grid_window(
        reference_raster, reference_raster_stack, roi_bounds)
    height, width = int(window.height), int(window.width)
    rows, columns = pose_graph.get("grid_shape_rows_columns", (0, 0))
    if not rows or not columns:
        raise ValueError("Pose graph has an invalid spatial grid")
    cells = list(pose_graph.get("cells", []))
    if len(cells) != rows * columns:
        raise ValueError("Pose-graph cell count does not match its grid shape")

    cell_records = []
    for cell in cells:
        polygon = Polygon.from_bounds(*cell["bounds_epsg32649"])
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("Pose graph contains invalid cell bounds")
        cell_records.append((cell, polygon))

    candidate_features_by_date: dict[str, list[dict[str, Any]]] = {date: [] for date in dates}
    decisions_by_epoch: dict[str, list[dict[str, Any]]] = {
        epoch: [] for epoch in epoch_to_date}
    decision_counts: dict[str, dict[str, int]] = {}
    field_products = []
    mask_products = []
    quality_products = []
    raster_qa = []

    for epoch, actual_date in sorted(epoch_to_date.items(), key=lambda item: item[1]):
        quality_shapes = []
        dx_shapes = []
        dy_shapes = []
        supported_shapes = []
        for cell, polygon in cell_records:
            status = str(cell["status"])
            offsets = cell.get("candidate_corrections_to_anchor_m", {})
            correction = offsets.get(actual_date)
            supported = status in _SUPPORTED_POSE_STATUSES and correction is not None
            if actual_date == anchor_date and supported:
                quality_code = RELATIVE_QUALITY_CODES["anchor_epoch_raw_reference_only"]
            elif supported and status == "RELATIVE_POSE_CYCLE_DIAGNOSTIC_CANDIDATE":
                quality_code = RELATIVE_QUALITY_CODES["relative_pose_cycle_candidate"]
            elif supported:
                quality_code = RELATIVE_QUALITY_CODES["relative_pose_tree_no_cycle_check"]
            else:
                quality_code = RELATIVE_QUALITY_CODES["insufficient_relative_support"]
            quality_shapes.append((mapping(polygon), quality_code))
            if not supported:
                continue
            dx, dy = (float(value) for value in correction)
            if not math.isfinite(dx) or not math.isfinite(dy):
                raise ValueError("Pose graph candidate correction must be finite")
            supported_shapes.append((mapping(polygon), 0))
            dx_shapes.append((mapping(polygon), dx))
            dy_shapes.append((mapping(polygon), dy))

        quality_array = _rasterize_or_fill(
            quality_shapes, out_shape=(height, width), transform=output_transform,
            fill=_MASK_NODATA, dtype="uint8")
        dx_array = _rasterize_or_fill(
            dx_shapes, out_shape=(height, width), transform=output_transform,
            fill=_FIELD_NODATA, dtype="float32")
        dy_array = _rasterize_or_fill(
            dy_shapes, out_shape=(height, width), transform=output_transform,
            fill=_FIELD_NODATA, dtype="float32")
        mask_array = _rasterize_or_fill(
            supported_shapes, out_shape=(height, width), transform=output_transform,
            fill=_MASK_NODATA, dtype="uint8")

        roi_polygon = Polygon.from_bounds(*roi_bounds)
        for item in local_roofs_by_epoch[epoch]:
            geometry: BaseGeometry = item["geometry"]
            object_id = str(item["object_id"])
            source_ref = (
                f"{geometry_source_paths[epoch]}#layer=buildings#object_id={object_id}")
            representative = geometry.representative_point()
            owning = [(cell, polygon) for cell, polygon in cell_records
                      if polygon.covers(representative)]
            status = "INSUFFICIENT_RELATIVE_SUPPORT"
            selected_cell = None
            correction_dx = correction_dy = None
            candidate_geometry = None
            pose_status = None
            if len(owning) == 1:
                selected_cell, selected_polygon = owning[0]
                pose_status = str(selected_cell["status"])
                correction = selected_cell.get("candidate_corrections_to_anchor_m", {}).get(actual_date)
                if pose_status not in _SUPPORTED_POSE_STATUSES or correction is None:
                    status = "INSUFFICIENT_RELATIVE_SUPPORT"
                elif not roi_polygon.covers(geometry):
                    status = "ROOF_CROSSES_BOUNDED_ROI"
                elif not selected_polygon.contains(geometry):
                    status = "ROOF_CROSSES_RELATIVE_CELL_BOUNDARY"
                else:
                    correction_dx, correction_dy = (float(value) for value in correction)
                    candidate_geometry = translate(
                        geometry, xoff=correction_dx, yoff=correction_dy)
                    if not selected_polygon.contains(candidate_geometry):
                        status = "CANDIDATE_CROSSES_RELATIVE_CELL_BOUNDARY"
                        candidate_geometry = None
                    else:
                        status = ("ANCHOR_REFERENCE_RAW_UNVERIFIED"
                                  if actual_date == anchor_date
                                  else "RELATIVE_ALIGNMENT_CANDIDATE_EMITTED")
            elif not roi_polygon.covers(geometry):
                status = "ROOF_CROSSES_BOUNDED_ROI"
            else:
                status = "NO_UNIQUE_RELATIVE_CELL"

            decision = {
                "geometry": geometry, "candidate_geometry": candidate_geometry,
                "candidate_uid": f"{actual_date}:{object_id}",
                "source_ref": source_ref, "epoch": epoch,
                "actual_date": actual_date,
                "cell_id": (f"r{selected_cell['row']}c{selected_cell['column']}"
                            if selected_cell is not None else None),
                "pose_status": pose_status, "decision": status,
                "correction_dx_m": correction_dx, "correction_dy_m": correction_dy,
                "closure_max_m": (selected_cell.get("closure_max_m")
                                   if selected_cell is not None else None),
                # This is the number of pairwise roof ties contributing to
                # this date's local relative solution, not an accuracy metric.
                "support_sample_count": (
                    sum(int(edge.get("sample_count", 0))
                        for edge in selected_cell.get("pairwise_candidate_edges", [])
                        if actual_date in (edge.get("reference_id"), edge.get("moving_id")))
                    if selected_cell is not None else 0),
            }
            decisions_by_epoch[epoch].append(decision)
            if candidate_geometry is not None:
                candidate_features_by_date[actual_date].append(decision)

        for decision in candidate_features_by_date[actual_date]:
            mask_array = rasterize(
                [(mapping(decision["candidate_geometry"]), 1)],
                out=mask_array, transform=output_transform,
                all_touched=False, dtype="uint8")

        field_path = output_dir / f"{actual_date}_relative_to_{anchor_date}_roof_field.tif"
        quality_path = output_dir / f"{actual_date}_relative_roof_alignment_quality.tif"
        mask_path = output_dir / f"{actual_date}_relative_registered_roof_candidate.tif"
        common_tags = {
            "PRODUCT_SCHEMA": "xian_r5.relative_roof_candidate_products.v1",
            "PRODUCT_CRS": "EPSG:32649",
            "REFERENCE_BASE_GRID": reference_grid["path"],
            "REFERENCE_BASE_GRID_SHA256": reference_grid["sha256"],
            "REFERENCE_GRID_PIXEL_SIZE_M": "10",
            "ACTUAL_DATE": actual_date,
            "RELATIVE_ANCHOR_DATE": anchor_date,
            "SOURCE_OUTLINE_EPOCH_LABEL": outline_source_epoch_label,
            "SOURCE_OUTLINE_EPOCH_LABEL_PROVENANCE": "USER_REPORTED_NOT_EXACT_ACQUISITION_DATE",
            "SOURCE_OUTLINE_GEOMETRY_SEMANTICS": outline_source_geometry_semantics,
            "SOURCE_OUTLINE_GEOMETRY_SEMANTICS_USER_REPORTED": "IMAGE_DRAWN_ROOF_BOUNDARY",
            "SOURCE_OUTLINE_SEMANTICS_PROVENANCE": "USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED",
            "SOURCE_OUTLINE_CAPTURE_DATE": "UNKNOWN",
            "SOURCE_OUTLINE_ACCURACY_M": "UNKNOWN",
            "SOURCE_OUTLINE_COMPONENT_SHA256_JSON": json.dumps(
                outline_source_component_sha256, sort_keys=True),
            "SOURCE_OUTLINE_GROUND_TRUTH": "NO",
            "ABSOLUTE_GROUND_POSITION_RESOLVED": "NO",
            "POSITIONAL_ACCURACY_CLAIMED": "NO",
            "FORMAL_GROUND_FOOTPRINT": "NO",
            "FORMAL_CHANGE_USE": "NO",
            "INTERPOLATED": "NO",
            "SOURCE_COMPONENT_PROVENANCE": "UNKNOWN_PER_PIXEL_SOURCE_COMPONENT_DATE_AND_VIEW_ANGLE",
        }
        _write_raster(
            field_path, np.stack((dx_array, dy_array)), crs="EPSG:32649",
            transform=output_transform, nodata=_FIELD_NODATA,
            tags={**common_tags,
                  "PRODUCT_KIND": "RELATIVE_ROOF_ALIGNMENT_FIELD_CANDIDATE",
                  "FIELD_SEMANTICS": "DX_DY_TO_RELATIVE_ANCHOR_METRES_NOT_GROUND_CORRECTION"},
            descriptions=("candidate_dx_to_anchor_m", "candidate_dy_to_anchor_m"))
        _write_raster(
            quality_path, quality_array, crs="EPSG:32649",
            transform=output_transform, nodata=_MASK_NODATA,
            tags={**common_tags,
                  "PRODUCT_KIND": "RELATIVE_ROOF_ALIGNMENT_QUALITY",
                  "QUALITY_CODES_JSON": json.dumps(RELATIVE_QUALITY_CODES, sort_keys=True),
                  "CLOSURE_IS_ABSOLUTE_ACCURACY": "NO"})
        _write_raster(
            mask_path, mask_array, crs="EPSG:32649",
            transform=output_transform, nodata=_MASK_NODATA,
            tags={**common_tags,
                  "PRODUCT_KIND": "RELATIVE_REGISTERED_ROOF_CANDIDATE_MASK",
                  "MASK_CODES_JSON": json.dumps({"background": 0,
                                                 "relative_roof_candidate": 1,
                                                 "unsupported_or_outside": 255}, sort_keys=True),
                  "RASTERIZATION": "PIXEL_CENTER;NO_INTERPOLATION"})

        for path in (field_path, quality_path, mask_path):
            with rasterio.open(path) as product:
                if (product.crs.to_epsg() != 32649
                        or product.transform != output_transform
                        or product.width != width or product.height != height):
                    raise RuntimeError(f"Relative candidate raster grid QA failed: {path}")
        if not set(int(value) for value in np.unique(quality_array)).issubset(
                set(RELATIVE_QUALITY_CODES.values())):
            raise RuntimeError("Relative candidate quality raster contains an undeclared code")
        if not set(int(value) for value in np.unique(mask_array)).issubset({0, 1, _MASK_NODATA}):
            raise RuntimeError("Relative roof candidate mask contains an undeclared code")

        field_products.append({
            "actual_date": actual_date, "path": str(field_path),
            "sha256": _sha256(field_path),
            "quality_path": str(quality_path), "quality_sha256": _sha256(quality_path),
            "supported_cell_count": len(supported_shapes),
            "unsupported_cell_count": rows * columns - len(supported_shapes),
            "absolute_ground_position_resolved": False,
            "applied_to_formal_ground_footprints": False,
        })
        quality_products.append({
            "actual_date": actual_date, "path": str(quality_path),
            "sha256": _sha256(quality_path),
            "quality_codes": RELATIVE_QUALITY_CODES,
        })
        mask_products.append({
            "actual_date": actual_date, "path": str(mask_path),
            "sha256": _sha256(mask_path),
            "candidate_feature_count": len(candidate_features_by_date[actual_date]),
            "candidate_pixel_count": int(np.count_nonzero(mask_array == 1)),
            "valid_candidate_cell_count": len(supported_shapes),
            "nodata_pixel_count": int(np.count_nonzero(mask_array == _MASK_NODATA)),
            "formal_ground_footprint_product": False,
            "formal_change_use": False,
        })
        raster_qa.append({
            "actual_date": actual_date, "crs": "EPSG:32649",
            "window_transform": list(output_transform),
            "width": width, "height": height,
            "quality_value_counts": {
                str(int(value)): int(count)
                for value, count in zip(*np.unique(quality_array, return_counts=True))},
            "roof_mask_value_counts": {
                str(int(value)): int(count)
                for value, count in zip(*np.unique(mask_array, return_counts=True))},
            "status": "PASS",
        })

    gpkg_path = output_dir / "RELATIVE_ROOF_CANDIDATES.gpkg"
    if gpkg_path.exists():
        raise FileExistsError(f"Refusing to overwrite relative-roof candidates: {gpkg_path}")
    decision_schema = {"geometry": "Unknown", "properties": {
        "candidate_uid": "str:96", "source_ref": "str:512",
        "epoch": "str:16", "actual_date": "str:16",
        "cell_id": "str:24", "pose_status": "str:48",
        "decision": "str:56", "correction_dx_m": "float",
        "correction_dy_m": "float", "closure_max_m": "float",
        "support_sample_count": "int",
        "source_outline_epoch": "str:32", "source_outline_semantics": "str:48",
        "source_outline_epoch_provenance": "str:64",
        "source_outline_semantics_user_reported": "str:40",
        "source_outline_semantics_provenance": "str:64",
        "source_outline_ref": "str:512",
        "source_outline_ground_truth": "int", "absolute_ground": "int",
        "formal_ground_use": "int", "formal_change_use": "int",
        "position_sigma_m": "float"}}
    candidate_schema = {"geometry": "Unknown", "properties": {
        **decision_schema["properties"], "status": "str:32"}}
    with fiona.open(gpkg_path, "w", driver="GPKG",
                    layer="candidate_decisions", schema=decision_schema,
                    crs="EPSG:32649") as sink:
        for epoch in sorted(decisions_by_epoch):
            for record in decisions_by_epoch[epoch]:
                sink.write({"geometry": mapping(record["geometry"]), "properties": {
                    "candidate_uid": record["candidate_uid"],
                    "source_ref": record["source_ref"], "epoch": record["epoch"],
                    "actual_date": record["actual_date"],
                    "cell_id": record["cell_id"] or "",
                    "pose_status": record["pose_status"] or "",
                    "decision": record["decision"],
                    "correction_dx_m": record["correction_dx_m"],
                    "correction_dy_m": record["correction_dy_m"],
                    "closure_max_m": record["closure_max_m"],
                    "support_sample_count": record["support_sample_count"],
                    "source_outline_epoch": outline_source_epoch_label,
                    "source_outline_semantics": outline_source_geometry_semantics,
                    "source_outline_epoch_provenance": (
                        "USER_REPORTED_NOT_EXACT_ACQUISITION_DATE"),
                    "source_outline_semantics_user_reported": "IMAGE_DRAWN_ROOF_BOUNDARY",
                    "source_outline_semantics_provenance": (
                        "USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED"),
                    "source_outline_ref": outline_source_path,
                    "source_outline_ground_truth": 0, "absolute_ground": 0,
                    "formal_ground_use": 0, "formal_change_use": 0,
                    "position_sigma_m": None}})
    with fiona.open(gpkg_path, "w", driver="GPKG",
                    layer="relative_registered_roof_candidates",
                    schema=candidate_schema, crs="EPSG:32649") as sink:
        for actual_date in dates:
            for record in candidate_features_by_date[actual_date]:
                sink.write({"geometry": mapping(record["candidate_geometry"]), "properties": {
                    "candidate_uid": record["candidate_uid"],
                    "source_ref": record["source_ref"], "epoch": record["epoch"],
                    "actual_date": record["actual_date"],
                    "cell_id": record["cell_id"],
                    "pose_status": record["pose_status"],
                    "decision": record["decision"],
                    "correction_dx_m": record["correction_dx_m"],
                    "correction_dy_m": record["correction_dy_m"],
                    "closure_max_m": record["closure_max_m"],
                    "support_sample_count": record["support_sample_count"],
                    "source_outline_epoch": outline_source_epoch_label,
                    "source_outline_semantics": outline_source_geometry_semantics,
                    "source_outline_epoch_provenance": (
                        "USER_REPORTED_NOT_EXACT_ACQUISITION_DATE"),
                    "source_outline_semantics_user_reported": "IMAGE_DRAWN_ROOF_BOUNDARY",
                    "source_outline_semantics_provenance": (
                        "USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED"),
                    "source_outline_ref": outline_source_path,
                    "source_outline_ground_truth": 0, "absolute_ground": 0,
                    "formal_ground_use": 0, "formal_change_use": 0,
                    "position_sigma_m": None,
                    "status": "UNVERIFIED_RELATIVE_ROOF_CANDIDATE"}})

    for epoch, records in decisions_by_epoch.items():
        decision_counts[epoch] = dict(sorted(
            {status: sum(record["decision"] == status for record in records)
             for status in {record["decision"] for record in records}}.items()))
    output_hashes = {str(path.name): _sha256(path) for path in (
        gpkg_path, *[Path(item["path"]) for item in field_products],
        *[Path(item["path"]) for item in quality_products],
        *[Path(item["path"]) for item in mask_products])}
    manifest = {
        "schema_version": "xian_r5.relative_roof_candidate_products.v1",
        "reference_grid": reference_grid,
        "relative_anchor_actual_date": anchor_date,
        "relative_pose_grid_shape_rows_columns": [rows, columns],
        "relative_pose_minimum_support_per_edge": pose_graph.get(
            "minimum_support_per_pairwise_edge"),
        "relative_pose_graph_semantics": pose_graph.get("source_semantics"),
        "support_sample_count_semantics": (
            "SUM_OF_PAIRWISE_ROOF_TIES_INCIDENT_ON_ACTUAL_DATE_NOT_ACCURACY"),
        "cycle_closure_semantics": "INTERNAL_RELATIVE_CONSISTENCY_NOT_ABSOLUTE_POSITION_ACCURACY",
        "outline_source": {
            "path": outline_source_path,
            "component_sha256_revalidated": dict(outline_source_component_sha256),
            "source_epoch_label": outline_source_epoch_label,
            "source_epoch_label_provenance": "USER_REPORTED_EPOCH_LABEL_NOT_EXACT_ACQUISITION_DATE",
            "source_geometry_semantics": outline_source_geometry_semantics,
            "source_geometry_semantics_user_reported": "IMAGE_DRAWN_ROOF_BOUNDARY",
            "source_geometry_semantics_provenance": "USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED",
            "capture_date": None, "accuracy_m": None, "ground_truth": False,
        },
        "geometry_source_lineage": {
            epoch: {"path": str(geometry_source_paths[epoch]),
                    "sha256": geometry_source_sha256[epoch],
                    "feature_count_in_roi": len(local_roofs_by_epoch[epoch]),
                    "actual_date": epoch_to_date[epoch]}
            for epoch in sorted(epoch_to_date)},
        "candidate_vector_path": str(gpkg_path),
        "candidate_vector_sha256": output_hashes[gpkg_path.name],
        "candidate_vector_layers": {
            "decision_layer": "candidate_decisions",
            "relative_registered_candidates": "relative_registered_roof_candidates",
        },
        "field_products": field_products,
        "quality_products": quality_products,
        "roof_candidate_mask_products": mask_products,
        "decision_counts_by_epoch": decision_counts,
        "raster_grid_qa": raster_qa,
        "output_sha256": output_hashes,
        "ground_footprint_products_written": False,
        "formal_change_products_written": False,
        "position_accuracy_claimed": False,
        "absolute_ground_position_resolved": False,
        "interpolated": False,
        "applied_to_formal_ground_footprints": False,
    }
    manifest_path = output_dir / "RELATIVE_ROOF_CANDIDATE_PRODUCTS.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite relative-roof candidate manifest: {manifest_path}")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
