"""Export exact-source-component, multi-reference tie diagnostics.

The output is a diagnostic candidate on the shared 10 m grid. Agreement across
annual reference rasters is not ground control and is never applied to a roof
geometry or formal change product.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import fiona
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from .base_tie import image_shift_to_projected_correction


COMPONENT_TIE_QUALITY_CODES = {
    "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE": 1,
    "MULTI_REFERENCE_SAME_SHIFT_LOW_CONFIDENCE": 2,
    "MULTI_REFERENCE_SHIFT_DISAGREEMENT": 3,
    "INSUFFICIENT_MULTI_REFERENCE_NUMERIC_SUPPORT": 4,
    "INSUFFICIENT_TIE_SAMPLES": 5,
    "NO_NUMERIC_TIE": 6,
    "OVERLAPPING_SOURCE_COMPONENTS": 7,
    "OUTSIDE_SOURCE_COMPONENT_SUPPORT": 255,
}
_FIELD_NODATA = -9999.0
_QUALITY_NODATA = 255


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_reports(
    paths: Sequence[Path | str],
    *,
    minimum_correlation: float,
    minimum_peak_margin: float,
    agreement_tolerance_m: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if len(paths) < 2:
        raise ValueError("At least two independent reference diagnostics are required")
    if not 0.0 <= minimum_correlation <= 1.0:
        raise ValueError("minimum_correlation must be between zero and one")
    if minimum_peak_margin < 0.0 or agreement_tolerance_m < 0.0:
        raise ValueError("Tie thresholds must be non-negative")

    reports = []
    report_hashes = {}
    seen_years = set()
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != "xian_r5.base_tie_diagnostic.v1":
            raise ValueError(f"Unsupported base-tie diagnostic schema: {path}")
        if report.get("scope") != "FIXED_CASE_LOCAL_ONLY":
            raise ValueError(f"Diagnostic is not fixed-case local: {path}")
        if report.get("automatic_registration_promoted") is not False:
            raise ValueError(f"Diagnostic unexpectedly promotes registration: {path}")
        parameters = report.get("parameters", {})
        year = parameters.get("base_reference_year")
        if year is None or int(year) in seen_years:
            raise ValueError(f"Reference years must be present and unique: {path}")
        seen_years.add(int(year))
        if parameters.get("local_support_mode") != "EXACT_SOURCE_TILE_COMPONENTS":
            raise ValueError(f"Diagnostic does not use exact source-tile support: {path}")
        if parameters.get("local_field_interpolated_or_applied") is not False:
            raise ValueError(f"Diagnostic tie field was interpolated or applied: {path}")
        reports.append({"path": path, "sha256": digest, "year": int(year), "report": report})
        report_hashes[str(path)] = digest

    case_ids = {row["report"].get("case_id") for row in reports}
    if len(case_ids) != 1 or not next(iter(case_ids)):
        raise ValueError("Diagnostics must identify the same fixed case")
    return sorted(reports, key=lambda row: row["year"]), report_hashes


def _epoch_records(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    date_maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    base_grid_signatures = set()
    base_rasters: dict[str, dict[str, Any]] = {}
    for report_row in reports:
        report = report_row["report"]
        for epoch_label, epoch in report.get("epochs", {}).items():
            actual_date = str(epoch.get("actual_date") or "")
            if not actual_date:
                raise ValueError(f"Missing actual observation date in {report_row['path']}")
            local_tie = epoch.get("local_spatial_tie_diagnostic") or {}
            if (local_tie.get("support_mode") != "EXACT_SOURCE_TILE_COMPONENTS"
                    or local_tie.get("field_interpolated") is not False
                    or local_tie.get("field_applied") is not False):
                raise ValueError(f"Unsafe or unsupported local-tie semantics: {actual_date}")
            base_grid = epoch.get("base_grid") or {}
            signature = (
                base_grid.get("crs"), tuple(base_grid.get("resolution", ())),
                tuple(base_grid.get("window", ())), tuple(base_grid.get("transform", ())),
            )
            base_grid_signatures.add(signature)
            raster_path = Path(epoch.get("base_path") or "").resolve()
            expected_base_hash = str(epoch.get("base_sha256") or "")
            if not raster_path.is_file() or len(expected_base_hash) != 64:
                raise ValueError(f"Base raster provenance is incomplete: {actual_date}/{raster_path}")
            prior_raster = base_rasters.get(str(raster_path))
            if prior_raster and prior_raster["sha256"] != expected_base_hash:
                raise ValueError(f"Conflicting base-raster hashes: {raster_path}")
            base_rasters[str(raster_path)] = {
                "path": raster_path, "sha256": expected_base_hash,
                "reference_year": report_row["year"],
            }
            by_year = date_maps.setdefault(actual_date, {})
            year_key = str(report_row["year"])
            if year_key in by_year:
                raise ValueError(
                    f"Reference diagnostic contains duplicate actual date {actual_date}: "
                    f"{report_row['path']}")
            by_year[year_key] = {
                "epoch_label": str(epoch_label), "epoch": epoch,
                "diagnostic_path": str(report_row["path"]),
                "reference_year": report_row["year"],
            }
    if len(base_grid_signatures) != 1:
        raise ValueError("Reference diagnostics do not share one exact local 10 m grid")
    expected_years = {str(row["year"]) for row in reports}
    if any(set(by_year) != expected_years for by_year in date_maps.values()):
        raise ValueError("Actual observation dates differ across reference diagnostics")
    return ({date: [by_year[year] for year in sorted(by_year, key=int)]
             for date, by_year in sorted(date_maps.items())}, base_rasters)


def _validate_base_rasters(
    base_rasters: Mapping[str, Mapping[str, Any]],
    *,
    expected_grid: Mapping[str, Any],
) -> dict[str, str]:
    grid_hashes = {}
    crs = expected_grid.get("crs")
    window_values = tuple(int(value) for value in expected_grid.get("window", ()))
    transform_values = tuple(float(value) for value in expected_grid.get("transform", ()))
    if len(window_values) != 4 or len(transform_values) not in (6, 9):
        raise ValueError("Diagnostic contains an invalid local base-grid window")
    window = rasterio.windows.Window(*window_values)
    local_transform = Affine(*transform_values[:6])
    full_grid_signature = None
    for path_text, entry in base_rasters.items():
        path = Path(path_text)
        actual_hash = _sha256(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Base-raster hash mismatch: {path}")
        with rasterio.open(path) as raster:
            if (raster.crs is None or raster.crs.to_string() != crs
                    or abs(raster.transform.a - 10.0) > 1e-8
                    or abs(raster.transform.e + 10.0) > 1e-8
                    or abs(raster.transform.b) > 1e-10
                    or abs(raster.transform.d) > 1e-10):
                raise ValueError(f"Reference must be a north-up 10 m EPSG:32649 raster: {path}")
            current_full_grid = (
                raster.crs.to_string(), tuple(raster.transform),
                raster.width, raster.height,
            )
            if full_grid_signature is None:
                full_grid_signature = current_full_grid
            elif current_full_grid != full_grid_signature:
                raise ValueError("Annual reference rasters do not share one exact full grid")
            computed = raster.window_transform(window)
            if any(not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-7)
                   for actual, wanted in zip(computed[:6], local_transform[:6])):
                raise ValueError(f"Local diagnostic window differs from base raster: {path}")
            if (window.col_off < 0 or window.row_off < 0
                    or window.col_off + window.width > raster.width
                    or window.row_off + window.height > raster.height):
                raise ValueError(f"Diagnostic window is outside base raster: {path}")
        grid_hashes[path_text] = actual_hash
    return grid_hashes


def _load_case_roi_geometry(
    path_value: Path | str,
    reports: Sequence[Mapping[str, Any]],
) -> tuple[BaseGeometry, dict[str, Any]]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hashes = {
        str(row["report"].get("inputs_sha256", {}).get("fixed_case_geometry") or "")
        for row in reports
    }
    if len(expected_hashes) != 1 or len(next(iter(expected_hashes))) != 64:
        raise ValueError("Fixed-case geometry hash is missing or differs across diagnostics")
    expected_hash = next(iter(expected_hashes))
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Fixed-case geometry hash mismatch: {path}")

    case_id = str(reports[0]["report"]["case_id"])
    geometries = []
    with fiona.open(path, layer="fixed_case_polygons") as source:
        crs = rasterio.crs.CRS.from_user_input(source.crs_wkt or source.crs)
        if crs.to_epsg() != 32649:
            raise ValueError("Fixed-case target geometry must use EPSG:32649")
        for feature in source:
            properties = feature["properties"]
            if (properties.get("case_id") != case_id
                    or properties.get("role") != "TARGET"):
                continue
            geometry = shape(feature["geometry"])
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError("Fixed-case target geometry contains invalid features")
            geometries.append(geometry)
    if not geometries:
        raise ValueError(f"Fixed-case target geometry is empty for {case_id}")
    from shapely.ops import unary_union

    roi = unary_union(geometries)
    if roi.is_empty or not roi.is_valid:
        raise ValueError(f"Fixed-case target geometry union is invalid for {case_id}")
    return roi, {
        "path": str(path), "sha256": actual_hash, "case_id": case_id,
        "feature_count": len(geometries), "crs": crs.to_string(),
        "semantics": "FROZEN_FIXED_CASE_TARGET_GEOMETRIES_UNION_NOT_GROUND_REFERENCE",
    }


def _component_maps(
    date_records: Sequence[Mapping[str, Any]],
    *,
    transform_values: Sequence[float],
    minimum_correlation: float,
    minimum_peak_margin: float,
    agreement_tolerance_m: float,
) -> list[dict[str, Any]]:
    by_reference = []
    for reference in date_records:
        epoch = reference["epoch"]
        local_tie = epoch["local_spatial_tie_diagnostic"]
        cells = local_tie.get("cells", [])
        cell_map = {}
        for cell in cells:
            tile_id = str(cell.get("tile_id") or "")
            if not tile_id or tile_id in cell_map:
                raise ValueError(f"Missing or duplicate source tile ID: {tile_id!r}")
            if cell.get("actual_date") != epoch.get("actual_date"):
                raise ValueError(f"Source component date mismatch: {tile_id}")
            geometry = cell.get("support_polygon_epsg32649")
            if not isinstance(geometry, dict):
                raise ValueError(f"Source component lacks support polygon: {tile_id}")
            polygon = shape(geometry)
            if polygon.is_empty or not polygon.is_valid:
                raise ValueError(f"Invalid source component support polygon: {tile_id}")
            cell_map[tile_id] = cell
        by_reference.append((reference, cell_map))

    all_tile_ids = sorted(set().union(*(set(cell_map) for _, cell_map in by_reference)))
    affine_values = tuple(float(value) for value in transform_values)
    result = []
    for tile_id in all_tile_ids:
        rows = [cell_map.get(tile_id) for _, cell_map in by_reference]
        existing = [cell for cell in rows if cell is not None]
        geometries = [shape(cell["support_polygon_epsg32649"]) for cell in existing]
        geometry = geometries[0]
        if any(geometry.symmetric_difference(other).area > 0.01 for other in geometries[1:]):
            raise ValueError(f"Source component geometry changes across references: {tile_id}")

        evidence = []
        numeric_vectors = []
        low_sample_seen = False
        for (reference, _), cell in zip(by_reference, rows):
            year = int(reference["reference_year"])
            if cell is None:
                evidence.append({"reference_year": year, "status": "COMPONENT_MISSING"})
                continue
            best = cell.get("best")
            if not isinstance(best, dict):
                low_sample_seen |= cell.get("sample_support_status") == "INSUFFICIENT_MASKED_SAMPLES"
                evidence.append({
                    "reference_year": year,
                    "status": cell.get("sample_support_status") or "NO_NUMERIC_PEAK",
                    "minimum_samples_required": cell.get("minimum_samples_required"),
                    "maximum_masked_samples_available": cell.get("maximum_masked_samples_available"),
                    "source_tile_path": cell.get("source_tile_path"),
                    "source_tile_sha256": cell.get("source_tile_sha256"),
                })
                continue
            dx_px, dy_px = best.get("dx_px"), best.get("dy_px")
            corr, margin = best.get("corr"), cell.get("peak_margin")
            samples, minimum_samples = best.get("n"), cell.get("minimum_samples_required")
            if not all(_finite_number(value) for value in (dx_px, dy_px, corr, samples)):
                raise ValueError(f"Non-finite source-component tie candidate: {tile_id}/{year}")
            dx_m, dy_m = image_shift_to_projected_correction(
                float(dx_px), float(dy_px), affine_values)
            enough_samples = (minimum_samples is not None
                              and int(samples) >= int(minimum_samples))
            confidence_pass = (
                float(corr) >= minimum_correlation
                and _finite_number(margin)
                and float(margin) >= minimum_peak_margin
                and enough_samples
            )
            vector = {"dx_m": dx_m, "dy_m": dy_m}
            numeric_vectors.append({
                "reference_year": year, "vector": vector,
                "high_confidence_gate_pass": confidence_pass,
            })
            evidence.append({
                "reference_year": year,
                "status": "NUMERIC_CANDIDATE",
                "observed_image_shift_px": [float(dx_px), float(dy_px)],
                "candidate_moving_to_reference_correction_m": [dx_m, dy_m],
                "correlation": float(corr),
                "peak_margin": float(margin) if _finite_number(margin) else None,
                "sample_count": int(samples),
                "minimum_samples_required": (int(minimum_samples)
                                              if minimum_samples is not None else None),
                "maximum_masked_samples_available": (
                    int(cell["maximum_masked_samples_available"])
                    if cell.get("maximum_masked_samples_available") is not None else None),
                "high_confidence_gate_pass": confidence_pass,
                "source_tile_path": cell.get("source_tile_path"),
                "source_tile_sha256": cell.get("source_tile_sha256"),
            })

        if len(existing) != len(by_reference) or len(numeric_vectors) < 2:
            status = "INSUFFICIENT_MULTI_REFERENCE_NUMERIC_SUPPORT"
        else:
            dxs = [item["vector"]["dx_m"] for item in numeric_vectors]
            dys = [item["vector"]["dy_m"] for item in numeric_vectors]
            agrees = (max(dxs) - min(dxs) <= agreement_tolerance_m
                      and max(dys) - min(dys) <= agreement_tolerance_m)
            all_confident = all(item["high_confidence_gate_pass"]
                                for item in numeric_vectors)
            if not agrees:
                status = "MULTI_REFERENCE_SHIFT_DISAGREEMENT"
            elif all_confident:
                status = "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE"
            else:
                status = "MULTI_REFERENCE_SAME_SHIFT_LOW_CONFIDENCE"
        if not numeric_vectors:
            status = ("INSUFFICIENT_TIE_SAMPLES" if low_sample_seen
                      else "NO_NUMERIC_TIE")
        candidate_dx = candidate_dy = None
        if status == "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE":
            candidate_dx = float(np.median([item["vector"]["dx_m"]
                                            for item in numeric_vectors]))
            candidate_dy = float(np.median([item["vector"]["dy_m"]
                                            for item in numeric_vectors]))
        correlations = [item["correlation"] for item in evidence
                        if item.get("status") == "NUMERIC_CANDIDATE"]
        margins = [item["peak_margin"] for item in evidence
                   if item.get("status") == "NUMERIC_CANDIDATE"
                   and item.get("peak_margin") is not None]
        tile_provenance = next((item for item in evidence
                                if item.get("source_tile_path")), {})
        result.append({
            "tile_id": tile_id, "geometry": geometry, "decision": status,
            "roi_geometry": None,
            "reference_count": len(by_reference),
            "numeric_reference_count": len(numeric_vectors),
            "candidate_dx_m": candidate_dx, "candidate_dy_m": candidate_dy,
            "minimum_correlation_observed": min(correlations) if correlations else None,
            "minimum_peak_margin_observed": min(margins) if margins else None,
            "source_tile_path": tile_provenance.get("source_tile_path"),
            "source_tile_sha256": tile_provenance.get("source_tile_sha256"),
            "reference_evidence": evidence,
        })
    return result


def _write_candidate_raster(
    path: Path,
    array: np.ndarray,
    *,
    transform: Affine,
    descriptions: Sequence[str] | None = None,
    tags: Mapping[str, Any],
    nodata: int | float,
) -> None:
    data = array[np.newaxis, ...] if array.ndim == 2 else array
    dtype = str(data.dtype)
    with rasterio.open(
        path, "w", driver="GTiff", width=int(data.shape[2]),
        height=int(data.shape[1]), count=int(data.shape[0]), dtype=dtype,
        crs="EPSG:32649", transform=transform, nodata=nodata,
        compress="deflate", predictor=2 if dtype.startswith("float") else 1,
    ) as target:
        target.write(data)
        if descriptions:
            for index, description in enumerate(descriptions, start=1):
                target.set_band_description(index, description)
        target.update_tags(**{str(key): str(value) for key, value in tags.items()})


def export_component_tie_consensus_candidates(
    output_dir: Path | str,
    diagnostic_paths: Sequence[Path | str],
    *,
    fixed_case_geometry_path: Path | str,
    minimum_correlation: float = 0.6,
    minimum_peak_margin: float = 0.1,
    agreement_tolerance_m: float = 0.01,
) -> dict[str, Any]:
    """Export conservative component-wise tie agreement and quality products.

    Only exact source-tile components present in every reference diagnostic,
    with matching component hashes/geometries and high-confidence same-shift
    candidates, receive numeric offsets. The exported vector shifts remain
    diagnostic-only; the function never edits input geometries or images.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory must already exist: {output_dir}")
    reports, report_hashes = _load_reports(
        diagnostic_paths, minimum_correlation=minimum_correlation,
        minimum_peak_margin=minimum_peak_margin,
        agreement_tolerance_m=agreement_tolerance_m)
    case_roi_geometry, case_geometry_receipt = _load_case_roi_geometry(
        fixed_case_geometry_path, reports)
    date_records, base_rasters = _epoch_records(reports)
    expected_grid = date_records[sorted(date_records)[0]][0]["epoch"]["base_grid"]
    base_hashes = _validate_base_rasters(base_rasters, expected_grid=expected_grid)

    case_id = str(reports[0]["report"]["case_id"])
    reference_years = [row["year"] for row in reports]
    window_values = tuple(int(value) for value in expected_grid["window"])
    transform_values = tuple(float(value) for value in expected_grid["transform"])
    transform = Affine(*transform_values[:6])
    width, height = window_values[2], window_values[3]
    if width < 1 or height < 1:
        raise ValueError("Diagnostic local base-grid window is empty")
    grid_corners = [transform @ point for point in ((0, 0), (width, 0),
                                                     (width, height), (0, height))]
    local_grid_polygon = box(
        min(point[0] for point in grid_corners), min(point[1] for point in grid_corners),
        max(point[0] for point in grid_corners), max(point[1] for point in grid_corners))
    if case_roi_geometry.difference(local_grid_polygon).area > 1e-4:
        raise ValueError("Fixed-case target geometry is not fully covered by the output grid")
    vector_path = output_dir / f"{case_id}_source_component_tie_consensus_candidates.gpkg"
    manifest_path = output_dir / "COMPONENT_TIE_CONSENSUS_CANDIDATES.json"
    planned_paths = [vector_path, manifest_path]
    for actual_date in date_records:
        planned_paths.extend((
            output_dir / f"{case_id}_{actual_date}_component_shift_candidate.tif",
            output_dir / f"{case_id}_{actual_date}_component_tie_quality.tif",
        ))
    existing_outputs = [path for path in planned_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite component-tie candidates: {existing_outputs}")

    all_decisions = {}
    component_hashes = {}
    for actual_date, records in date_records.items():
        components = _component_maps(
            records, transform_values=expected_grid["transform"],
            minimum_correlation=minimum_correlation,
            minimum_peak_margin=minimum_peak_margin,
            agreement_tolerance_m=agreement_tolerance_m)
        for component in components:
            component["roi_geometry"] = component["geometry"].intersection(case_roi_geometry)
            hashes = {item.get("source_tile_sha256") for item in component["reference_evidence"]
                      if item.get("source_tile_sha256")}
            paths = {item.get("source_tile_path") for item in component["reference_evidence"]
                     if item.get("source_tile_path")}
            if (len(hashes) > 1 or len(paths) > 1
                    or any(len(str(value)) != 64 for value in hashes)
                    or len(paths) != 1):
                raise ValueError(f"Source component identity differs across references: {actual_date}/{component['tile_id']}")
            component["source_tile_path"] = next(iter(paths), None)
            component["source_tile_sha256"] = next(iter(hashes), None)
            component_hashes[str(component["source_tile_path"])] = str(component["source_tile_sha256"])
        all_decisions[actual_date] = components

    # Revalidate exact source tiles represented in this bounded case. These are
    # image receipts, not proof that their pixels represent ground positions.
    source_hashes = {}
    for path_text, expected in component_hashes.items():
        path = Path(path_text)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Source tile hash mismatch: {path}")
        source_hashes[path_text] = expected
    for path_text, expected in report_hashes.items():
        if _sha256(Path(path_text)) != expected:
            raise RuntimeError(f"Diagnostic changed while being compiled: {path_text}")

    products = {}
    output_paths = []
    for actual_date, components in all_decisions.items():
        field_path = output_dir / f"{case_id}_{actual_date}_component_shift_candidate.tif"
        quality_path = output_dir / f"{case_id}_{actual_date}_component_tie_quality.tif"
        quality = np.full((height, width), _QUALITY_NODATA, dtype="uint8")
        coverage = np.zeros((height, width), dtype="uint16")
        dx = np.full((height, width), _FIELD_NODATA, dtype="float32")
        dy = np.full((height, width), _FIELD_NODATA, dtype="float32")
        roi_components = [component for component in components
                          if component["roi_geometry"] is not None
                          and not component["roi_geometry"].is_empty
                          and component["roi_geometry"].area > 0]
        for component in roi_components:
            covered = rasterize(
                [(mapping(component["roi_geometry"]), 1)],
                out_shape=(height, width), transform=transform,
                fill=0, dtype="uint8", all_touched=False).astype(bool)
            coverage += covered.astype("uint16")
            code = COMPONENT_TIE_QUALITY_CODES[component["decision"]]
            quality[covered] = code
            if code == 1:
                dx[covered] = component["candidate_dx_m"]
                dy[covered] = component["candidate_dy_m"]
        overlaps = coverage > 1
        quality[overlaps] = COMPONENT_TIE_QUALITY_CODES["OVERLAPPING_SOURCE_COMPONENTS"]
        dx[overlaps] = _FIELD_NODATA
        dy[overlaps] = _FIELD_NODATA

        common_tags = {
            "PRODUCT_SCHEMA": "xian_r5.component_tie_consensus_candidates.v1",
            "PRODUCT_SCOPE": "FIXED_CASE_LOCAL_ONLY",
            "ACTUAL_DATE": actual_date,
            "REFERENCE_YEARS": ",".join(str(year) for year in reference_years),
            "REFERENCE_GRID": "EPSG:32649_10M",
            "TIE_QUALITY_CODES": json.dumps(COMPONENT_TIE_QUALITY_CODES, sort_keys=True),
            "MINIMUM_CORRELATION": minimum_correlation,
            "MINIMUM_PEAK_MARGIN": minimum_peak_margin,
            "AGREEMENT_TOLERANCE_M": agreement_tolerance_m,
            "SHIFT_SEMANTICS": "NEGATIVE_OBSERVED_INTEGER_PIXEL_PEAK_TO_MAP_METERS",
            "POSITIONAL_UNCERTAINTY": "UNKNOWN_NOT_ESTIMATED",
            "ABSOLUTE_GROUND_POSITION_RESOLVED": "NO",
            "ROOF_TO_GROUND_CORRECTION": "NO",
            "INTERPOLATED": "NO",
            "APPLIED": "NO",
            "FORMAL_CHANGE_USE": "NO",
        }
        _write_candidate_raster(
            field_path, np.stack((dx, dy)), transform=transform,
            descriptions=("candidate_dx_m_not_accepted", "candidate_dy_m_not_accepted"),
            tags={**common_tags, "PRODUCT_KIND": "SOURCE_COMPONENT_SHIFT_CANDIDATE"},
            nodata=_FIELD_NODATA)
        _write_candidate_raster(
            quality_path, quality, transform=transform,
            tags={**common_tags, "PRODUCT_KIND": "SOURCE_COMPONENT_TIE_QUALITY"},
            nodata=_QUALITY_NODATA)
        counts = Counter(component["decision"] for component in roi_components)
        products[actual_date] = {
            "field_path": str(field_path), "quality_path": str(quality_path),
            "diagnostic_window_component_count": len(components),
            "case_roi_component_count": len(roi_components),
            "decision_counts_within_case_roi": dict(sorted(counts.items())),
            "high_confidence_same_shift_candidates": counts[
                "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE"],
            "high_confidence_candidates_outside_case_roi": sum(
                component["decision"] ==
                "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE"
                and (component["roi_geometry"] is None
                     or component["roi_geometry"].is_empty
                     or component["roi_geometry"].area <= 0)
                for component in components),
            "overlap_pixel_count": int(overlaps.sum()),
        }
        output_paths.extend((field_path, quality_path))

    schema = {
        "geometry": "Unknown",
        "properties": {
            "actual_date": "str:20", "tile_id": "str:32", "decision": "str:64",
            "reference_count": "int", "numeric_ref_count": "int",
            "candidate_dx_m": "float", "candidate_dy_m": "float",
            "min_correlation": "float", "min_peak_margin": "float",
            "source_tile_path": "str:254", "source_tile_sha256": "str:64",
            "reference_evidence": "str",
        },
    }
    with fiona.open(vector_path, "w", driver="GPKG",
                    layer="source_component_tie_consensus", schema=schema,
                    crs="EPSG:32649") as sink:
        for actual_date, components in all_decisions.items():
            for component in components:
                roi_geometry = component["roi_geometry"]
                if roi_geometry is None or roi_geometry.is_empty or roi_geometry.area <= 0:
                    continue
                sink.write({
                    "geometry": mapping(roi_geometry),
                    "properties": {
                        "actual_date": actual_date,
                        "tile_id": component["tile_id"],
                        "decision": component["decision"],
                        "reference_count": component["reference_count"],
                        "numeric_ref_count": component["numeric_reference_count"],
                        "candidate_dx_m": component["candidate_dx_m"],
                        "candidate_dy_m": component["candidate_dy_m"],
                        "min_correlation": component["minimum_correlation_observed"],
                        "min_peak_margin": component["minimum_peak_margin_observed"],
                        "source_tile_path": component["source_tile_path"] or "",
                        "source_tile_sha256": component["source_tile_sha256"] or "",
                        "reference_evidence": json.dumps(
                            component["reference_evidence"], ensure_ascii=False,
                            sort_keys=True, separators=(",", ":")),
                    },
                })

    output_hashes = {path.name: _sha256(path) for path in (*output_paths, vector_path)}
    manifest = {
        "schema_version": "xian_r5.component_tie_consensus_candidates.v1",
        "case_id": case_id,
        "scope": "FIXED_CASE_LOCAL_ONLY",
        "reference_years": reference_years,
        "actual_dates": sorted(all_decisions),
        "reference_grid": {
            "crs": expected_grid["crs"], "resolution_m": expected_grid["resolution"],
            "window": expected_grid["window"], "transform": expected_grid["transform"],
            "width": width, "height": height,
        },
        "thresholds": {
            "minimum_correlation": minimum_correlation,
            "minimum_peak_margin": minimum_peak_margin,
            "agreement_tolerance_m": agreement_tolerance_m,
        },
        "source_diagnostics": [
            {"path": str(row["path"]), "sha256": row["sha256"],
             "reference_year": row["year"]} for row in reports],
        "fixed_case_geometry": case_geometry_receipt,
        "reference_base_rasters": [
            {"path": path, "sha256": digest,
             "reference_year": base_rasters[path]["reference_year"]}
            for path, digest in sorted(base_hashes.items())],
        "source_component_images_revalidated": len(source_hashes),
        "products_by_actual_date": products,
        "candidate_vector_path": str(vector_path),
        "candidate_vector_layer": "source_component_tie_consensus",
        "output_sha256": output_hashes,
        "support_semantics": "EXACT_HASHED_SOURCE_TILE_COMPONENT_POLYGONS",
        "consensus_semantics": (
            "SAME_INTEGER_IMAGE_SHIFT_ACROSS_ALL_REFERENCES_AND_CONFIDENCE_GATES;"
            "CANDIDATE_ONLY_NOT_POSITIONAL_ACCURACY"),
        "absolute_ground_position_resolved": False,
        "ground_footprints_written": False,
        "roof_to_ground_correction": False,
        "interpolated": False,
        "applied": False,
        "formal_change_products_written": False,
        "position_accuracy_claimed": False,
        "positional_uncertainty": "UNKNOWN_NOT_ESTIMATED",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
