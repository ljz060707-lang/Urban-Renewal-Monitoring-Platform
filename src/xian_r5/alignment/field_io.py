"""Read traceable image-alignment candidates from paired receipts and vectors."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping

try:
    import fiona
except ImportError:  # pragma: no cover - Fiona is optional for receipt-only adapters
    fiona = None
import rasterio
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .base_tie import image_shift_to_projected_correction
from .field import DisplacementCell, LocalDisplacementField
from .local_field import LOCAL_FIELD_METHOD, LOCAL_TRANSFORM_MODEL
from .mosaic_grid import MosaicTileGrid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_number(actual: object, expected: object, *, tolerance: float = 1e-7) -> bool:
    if expected is None:
        return actual is None or (isinstance(actual, (int, float)) and math.isnan(actual))
    return (actual is not None and math.isfinite(float(actual))
            and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance))


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _base_tie_settings_signature(settings: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Keep annual-anchor ties comparable under identical local support rules."""
    names = (
        "radius_tiles", "roof_exclusion_buffer_m", "local_grid_size",
        "local_support_mode", "local_search_radius_px", "local_minimum_samples",
    )
    return tuple((name, json.dumps(settings.get(name), sort_keys=True)) for name in names)


def _require_fiona(operation: str) -> None:
    if fiona is None:
        raise RuntimeError(f"{operation} requires Fiona to read its GeoPackage layer")


def _prepare_source_component_index(
    tile_entries: object,
    actual_date: str,
) -> dict[str, object] | None:
    """Validate one exact-date tile inventory and index it for repeated cell queries."""
    if not isinstance(tile_entries, list) or not tile_entries:
        return None
    items = []
    incomplete_provenance = False
    for entry in tile_entries:
        if not isinstance(entry, dict) or entry.get("actual_date") != actual_date:
            incomplete_provenance = True
            continue
        tile_id = str(entry.get("tile_id") or "")
        path = entry.get("path")
        digest = entry.get("sha256")
        polygon_value = entry.get("support_polygon_epsg32649")
        if (not re.fullmatch(r"\d+:\d+", tile_id)
                or not isinstance(path, str) or not path.strip()
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
                or not isinstance(polygon_value, dict)):
            incomplete_provenance = True
            continue
        geometry = shape(polygon_value)
        if (geometry.is_empty or not geometry.is_valid
                or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
            incomplete_provenance = True
            continue
        items.append((entry, geometry))
    geometries = [geometry for _, geometry in items]
    return {
        "items": items,
        "tree": STRtree(geometries) if geometries else None,
        "incomplete_provenance": incomplete_provenance,
    }


def _source_component_scope_for_support(
    support: Polygon,
    tile_entries: object,
    actual_date: str,
    *,
    prepared_index: dict[str, object] | None = None,
) -> dict[str, object]:
    """Identify exact-date source tiles covering one projected alignment cell."""
    unknown = {
        "source_component_scope": "SOURCE_COMPONENT_SCOPE_UNKNOWN",
        "source_component_ids": [],
        "source_component_coverage_fraction": None,
    }
    if prepared_index is not None:
        items = prepared_index["items"]
        tree = prepared_index["tree"]
        if tree is None:
            candidate_indices = ()
        else:
            candidate_indices = tree.query(support, predicate="intersects")
        intersecting = [items[int(index)] for index in candidate_indices]
        components = []
        threshold = max(1e-8, support.area * 1e-12)
        for entry, geometry in intersecting:
            if support.intersection(geometry).area > threshold:
                components.append({
                    "actual_date": actual_date,
                    "tile_id": str(entry["tile_id"]),
                    "path": entry["path"],
                    "sha256": str(entry["sha256"]).lower(),
                })
        if prepared_index["incomplete_provenance"] or not items:
            return {
                "source_component_scope": "SOURCE_COMPONENT_SCOPE_UNKNOWN",
                "source_component_ids": sorted(components, key=lambda item: item["tile_id"]),
                "source_component_coverage_fraction": None,
            }
        if not intersecting:
            coverage = 0.0
        else:
            covered = unary_union([geometry for _, geometry in intersecting])
            coverage = min(1.0, max(
                0.0, float(support.intersection(covered).area / support.area)))
        if coverage < 1.0 - 1e-8:
            scope = "INCOMPLETE_SOURCE_COMPONENT_COVERAGE"
        elif len(components) == 1:
            scope = "SINGLE_SOURCE_COMPONENT_PER_CELL"
        else:
            scope = "MULTIPLE_SOURCE_COMPONENTS_PER_CELL"
        return {
            "source_component_scope": scope,
            "source_component_ids": sorted(components, key=lambda item: item["tile_id"]),
            "source_component_coverage_fraction": coverage,
        }
    if not isinstance(tile_entries, list) or not tile_entries:
        return unknown
    components = []
    component_geometries = []
    incomplete_provenance = False
    for entry in tile_entries:
        if not isinstance(entry, dict) or entry.get("actual_date") != actual_date:
            incomplete_provenance = True
            continue
        tile_id = str(entry.get("tile_id") or "")
        path = entry.get("path")
        digest = entry.get("sha256")
        polygon_value = entry.get("support_polygon_epsg32649")
        if (not re.fullmatch(r"\d+:\d+", tile_id)
                or not isinstance(path, str) or not path.strip()
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
                or not isinstance(polygon_value, dict)):
            incomplete_provenance = True
            continue
        geometry = shape(polygon_value)
        if (geometry.is_empty or not geometry.is_valid
                or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
            incomplete_provenance = True
            continue
        component_geometries.append((entry, geometry))
        if support.intersection(geometry).area > max(1e-8, support.area * 1e-12):
            components.append({
                "actual_date": actual_date,
                "tile_id": tile_id,
                "path": path,
                "sha256": digest.lower(),
            })

    if incomplete_provenance or not component_geometries:
        return {
            "source_component_scope": "SOURCE_COMPONENT_SCOPE_UNKNOWN",
            "source_component_ids": components,
            "source_component_coverage_fraction": None,
        }
    covered = unary_union([geometry for _, geometry in component_geometries])
    coverage = min(1.0, max(0.0, float(support.intersection(covered).area / support.area)))
    if coverage < 1.0 - 1e-8:
        scope = "INCOMPLETE_SOURCE_COMPONENT_COVERAGE"
    elif len(components) == 1:
        scope = "SINGLE_SOURCE_COMPONENT_PER_CELL"
    else:
        scope = "MULTIPLE_SOURCE_COMPONENTS_PER_CELL"
    return {
        "source_component_scope": scope,
        "source_component_ids": sorted(components, key=lambda item: item["tile_id"]),
        "source_component_coverage_fraction": coverage,
    }


def paired_source_component_scope_for_support(
    support: Polygon,
    tiles_by_actual_date: Mapping[str, object],
    actual_dates: Iterable[str],
) -> dict[str, object]:
    """Resolve exact-date source components for a moving-to-reference tie cell.

    Each date must have complete geometric tile coverage. A field cell is scoped
    only when exactly one source tile covers it on every date and both dates use
    the same tile ID. Mosaic-wide or partial supports remain explicit candidates.
    """
    dates = tuple(str(value) for value in actual_dates)
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("Paired source-component scope requires unique actual dates")
    for actual_date in dates:
        try:
            date.fromisoformat(actual_date)
        except ValueError as exc:
            raise ValueError("Paired source-component scope requires ISO actual dates") from exc

    per_date = [
        _source_component_scope_for_support(
            support, tiles_by_actual_date.get(actual_date), actual_date)
        for actual_date in dates
    ]
    components = [
        component
        for receipt in per_date
        for component in receipt["source_component_ids"]
    ]
    coverages = [receipt["source_component_coverage_fraction"] for receipt in per_date]
    coverage = (min(float(value) for value in coverages)
                if all(isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in coverages) else None)
    scopes = [receipt["source_component_scope"] for receipt in per_date]
    if "SOURCE_COMPONENT_SCOPE_UNKNOWN" in scopes:
        scope = "SOURCE_COMPONENT_SCOPE_UNKNOWN"
    elif "INCOMPLETE_SOURCE_COMPONENT_COVERAGE" in scopes:
        scope = "INCOMPLETE_SOURCE_COMPONENT_COVERAGE"
    elif "MULTIPLE_SOURCE_COMPONENTS_PER_CELL" in scopes:
        scope = "MULTIPLE_SOURCE_COMPONENTS_PER_CELL"
    elif len(dates) > 1:
        tile_ids_by_date = {
            actual_date: {
                component["tile_id"] for component in receipt["source_component_ids"]
            }
            for actual_date, receipt in zip(dates, per_date)
        }
        if len({tuple(sorted(ids)) for ids in tile_ids_by_date.values()}) != 1:
            scope = "SOURCE_COMPONENT_ID_MISMATCH"
        else:
            scope = "SINGLE_SOURCE_COMPONENT_PER_CELL"
    else:
        scope = "SINGLE_SOURCE_COMPONENT_PER_CELL"

    return {
        "source_component_scope": scope,
        "source_component_ids": sorted(
            components, key=lambda item: (item["actual_date"], item["tile_id"])),
        "source_component_coverage_fraction": coverage,
    }


def _lightglue_source_component_scope(
    provenance: dict,
    from_date: str,
    to_date: str,
) -> dict[str, object]:
    """Require one complete, same-key source tile for each paired date cell."""
    tiles_by_date = provenance.get("tiles_by_actual_date")
    if not isinstance(tiles_by_date, dict):
        return {
            "source_component_scope": "SOURCE_COMPONENT_SCOPE_UNKNOWN",
            "source_component_ids": [],
        }
    components = []
    ids = []
    incomplete = False
    multiple = False
    coverage_fractions = []
    for actual_date in (from_date, to_date):
        group = tiles_by_date.get(actual_date)
        if not isinstance(group, dict):
            incomplete = True
            continue
        expected = group.get("expected_tile_ids")
        present = group.get("present_tiles")
        missing = group.get("missing_tile_ids")
        if not isinstance(expected, list) or not isinstance(present, list) or not isinstance(missing, list):
            incomplete = True
            continue
        if missing or len(present) != len(expected):
            incomplete = True
        if expected:
            coverage_fractions.append(min(1.0, len(present) / len(expected)))
        if len(expected) > 1 or len(present) > 1:
            multiple = True
        for record in present:
            if not isinstance(record, dict):
                incomplete = True
                continue
            component = {
                "actual_date": actual_date,
                "tile_id": str(record.get("tile_id") or ""),
                "path": record.get("path"),
                "sha256": record.get("sha256"),
            }
            if (not re.fullmatch(r"\d+:\d+", component["tile_id"])
                    or not isinstance(component["path"], str)
                    or not component["path"].strip()
                    or not isinstance(component["sha256"], str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", component["sha256"])):
                incomplete = True
            else:
                component["sha256"] = component["sha256"].lower()
                ids.append(component["tile_id"])
            components.append(component)
    if incomplete:
        scope = "INCOMPLETE_SOURCE_COMPONENT_COVERAGE"
    elif multiple:
        scope = "MULTIPLE_SOURCE_COMPONENTS_PER_CELL"
    elif len(ids) != 2:
        scope = "INCOMPLETE_SOURCE_COMPONENT_COVERAGE"
    elif ids[0] != ids[1]:
        scope = "SOURCE_COMPONENT_ID_MISMATCH"
    else:
        scope = "SINGLE_SOURCE_COMPONENT_PER_CELL"
    return {
        "source_component_scope": scope,
        "source_component_ids": components,
        "source_component_coverage_fraction": (
            min(coverage_fractions) if len(coverage_fractions) == 2 else None),
    }


def _verify_composed_source_diagnostics(manifest: dict) -> dict[Path, str]:
    """Verify source receipts when a composed field records their provenance."""
    source_diagnostics = manifest.get("source_diagnostics")
    if source_diagnostics is None:
        return {}
    if not isinstance(source_diagnostics, dict):
        raise ValueError("Composed-alignment source diagnostics must be an object")

    verified: dict[Path, str] = {}
    for prefix in ("base_pair", "base_tie"):
        path_value = source_diagnostics.get(f"{prefix}_path")
        digest_before = source_diagnostics.get(f"{prefix}_sha256_before")
        digest_after = source_diagnostics.get(f"{prefix}_sha256_after")
        if not path_value or not digest_before or not digest_after:
            raise ValueError(f"Incomplete composed-alignment {prefix} source receipt")
        if digest_before != digest_after:
            raise ValueError(f"Composed-alignment {prefix} source changed during composition")
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != digest_after:
            raise ValueError(f"Composed-alignment source diagnostic hash mismatch: {path}")
        verified[path] = actual
    return verified


def read_composed_alignment_fields(gpkg_path: Path, manifest_path: Path) -> list[LocalDisplacementField]:
    """Load unapplied fields and verify every vector against its JSON source receipt.

    The GeoPackage is never trusted alone: dates, candidate vectors, support bounds,
    and review flags must agree with the paired diagnostic JSON. No uncertainty is
    inferred from cycle-closure statistics.
    """
    gpkg_path = Path(gpkg_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not gpkg_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Both composed alignment GeoPackage and diagnostic manifest are required")
    before = (_sha256(gpkg_path), _sha256(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hashes = _verify_composed_source_diagnostics(manifest)
    if manifest.get("schema_version") != "xian_r5.composed_alignment_candidates.v1":
        raise ValueError("Unsupported composed-alignment diagnostic schema")
    products = manifest.get("products", {})
    expected_gpkg = (manifest_path.parent / str(products.get("geopackage", ""))).resolve()
    if gpkg_path != expected_gpkg:
        raise ValueError("Alignment GeoPackage does not match the manifest product reference")
    if (manifest.get("field_applied") is not False
            or manifest.get("field_interpolated") is not False
            or products.get("position_accuracy_claimed") is not False
            or products.get("accepted_cells") != 0):
        raise ValueError("Only unapplied, unaccepted diagnostic candidate fields can be loaded")
    if manifest.get("crs") != "EPSG:32649":
        raise ValueError("Composed alignment manifest must use EPSG:32649")
    if manifest.get("anchor_acquisition_date_status") != "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN":
        raise ValueError("Expected annual-base reference acquisition date to remain explicitly unknown")

    expected_rows = {}
    for row in manifest.get("date_fields", []):
        key = (str(row["actual_date"]), str(row["cell_id"]))
        if key in expected_rows:
            raise ValueError(f"Duplicate cell in composed-alignment manifest: {key}")
        expected_rows[key] = row
    grouped: dict[str, list[DisplacementCell]] = {}
    field_meta: dict[str, tuple[str | None, str, str]] = {}
    seen = set()
    _require_fiona("Composed alignment field loading")
    with fiona.open(gpkg_path, layer="composed_alignment_candidates") as source:
        if source.crs.to_epsg() != 32649:
            raise ValueError("Composed alignment GeoPackage must use EPSG:32649")
        for feature in source:
            props = dict(feature["properties"])
            moving_date = str(props["moving_date"])
            cell_id = str(props["cell_id"])
            key = (moving_date, cell_id)
            if key not in expected_rows or key in seen:
                raise ValueError(f"Unexpected or duplicate composed-alignment cell: {key}")
            seen.add(key)
            expected = expected_rows[key]
            if (str(props["field_id"]) != f"{moving_date}_to_{props['reference_label']}"
                    or props["reference_date_status"] != "REFERENCE_ACQUISITION_DATE_UNKNOWN"
                    or props["reference_date"] not in (None, "")):
                raise ValueError(f"Composed-alignment date/reference semantics mismatch: {key}")
            if props["status"] != expected["status"]:
                raise ValueError(f"Composed-alignment status mismatch: {key}")
            if (not _same_number(props["candidate_dx_m"], _pair_component(expected, 0))
                    or not _same_number(props["candidate_dy_m"], _pair_component(expected, 1))):
                raise ValueError(f"Composed-alignment vector mismatch: {key}")
            if (int(props["reviewed"] or 0) != 0 or int(props["ground_evidence"] or 0) != 0
                    or int(props["applied"] or 0) != 0):
                raise ValueError(f"Composed-alignment review/application gate is not closed: {key}")
            geometry = shape(feature["geometry"])
            bounds = expected["support_bounds_epsg32649"]
            if any(not math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-6)
                   for actual, wanted in zip(geometry.bounds, bounds)):
                raise ValueError(f"Composed-alignment support geometry mismatch: {key}")
            field_id = str(props["field_id"])
            reference_label = str(props["reference_label"])
            meta = (None, reference_label, "REFERENCE_ACQUISITION_DATE_UNKNOWN")
            if field_id in field_meta and field_meta[field_id] != meta:
                raise ValueError(f"Inconsistent field metadata: {field_id}")
            field_meta[field_id] = meta
            grouped.setdefault(field_id, []).append(DisplacementCell(
                cell_id=cell_id,
                support_geometry=geometry,
                correction_dx_m_candidate=(float(props["candidate_dx_m"])
                                           if props["candidate_dx_m"] is not None else None),
                correction_dy_m_candidate=(float(props["candidate_dy_m"])
                                           if props["candidate_dy_m"] is not None else None),
                fit_dispersion_m=None,
                status=str(props["status"]),
                method="SUBMETER_TO_ANNUAL_BASE_PLUS_ANNUAL_BASE_POSE_GRAPH",
                source_ref=f"{gpkg_path}#composed_alignment_candidates/{cell_id}",
                reviewed=False,
                independent_ground_evidence=False,
                positional_uncertainty_m=None,
                applied=False,
                metadata={name: props.get(name) for name in (
                    "base_dx_m", "base_dy_m", "highres_dx_m", "highres_dy_m",
                    "cycle_rmse_m", "cycle_max_m", "cycle_dof", "peak_corr",
                    "peak_margin", "eligible_px")},
            ))
    if seen != set(expected_rows):
        raise ValueError("Composed-alignment GeoPackage is missing manifest cells")

    fields = []
    anchor_source = str(manifest["anchor_path"])
    for field_id in sorted(grouped):
        moving_date = field_id.split("_to_", 1)[0]
        _, reference_label, reference_status = field_meta[field_id]
        fields.append(LocalDisplacementField(
            field_id=field_id,
            reference_date=None,
            moving_date=moving_date,
            cells=tuple(sorted(grouped[field_id], key=lambda cell: cell.cell_id)),
            reference_label=reference_label,
            reference_source=anchor_source,
            reference_date_status=reference_status,
        ))
    if before != (_sha256(gpkg_path), _sha256(manifest_path)):
        raise RuntimeError("Composed alignment inputs changed while being loaded")
    if source_hashes and source_hashes != {
            path: _sha256(path) for path in source_hashes}:
        raise RuntimeError("Composed alignment source diagnostics changed while being loaded")
    return fields


def read_spatial_alignment_composition_fields(
    manifest_path: Path,
    *,
    expected_base_raster: Path | None = None,
    expected_base_sha256: str | None = None,
    expected_case_id: str | None = None,
) -> list[LocalDisplacementField]:
    """Load spatial-overlap-composed candidates from their paired JSON/GPKG.

    Source diagnostics, the annual reference raster, cell vectors, support
    geometries, and review/application flags are revalidated. These fields are
    relative image-grid candidates only; the reader never promotes them to
    ground corrections or roof footprints.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_hash_before = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "xian_r5.spatially_composed_alignment_candidates.v1":
        raise ValueError("Unsupported spatially composed alignment diagnostic schema")
    if expected_case_id is not None and manifest.get("case_id") != expected_case_id:
        raise ValueError("Spatial-composition diagnostic belongs to a different fixed case")
    products = manifest.get("products")
    if not isinstance(products, dict):
        raise ValueError("Spatial-composition manifest is missing its products receipt")
    gpkg_path = (manifest_path.parent / str(products.get("geopackage") or "")).resolve()
    if not gpkg_path.is_file():
        raise FileNotFoundError(gpkg_path)
    gpkg_hash_before = _sha256(gpkg_path)

    if (manifest.get("field_applied") is not False
            or manifest.get("field_interpolated") is not False
            or manifest.get("absolute_ground_position_resolved") is not False
            or manifest.get("roof_to_ground_parallax_resolved") is not False
            or manifest.get("formal_change_use") is not False
            or products.get("field_applied") is not False
            or products.get("field_interpolated") is not False
            or products.get("formal_ground_footprint_product") is not False):
        raise ValueError("Spatial-composition candidates must remain unapplied and non-formal")
    if (manifest.get("crs") != "EPSG:32649"
            or manifest.get("anchor_acquisition_date_status")
            != "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"):
        raise ValueError("Spatial-composition reference CRS/date semantics are invalid")
    anchor_label = str(manifest.get("anchor_label") or "")
    source_inputs = manifest.get("source_inputs")
    if not anchor_label or not isinstance(source_inputs, dict):
        raise ValueError("Spatial-composition manifest lacks its reference/source receipt")

    verified_sources: dict[Path, str] = {}

    def verify_source(record: object, label: str) -> Path:
        if not isinstance(record, dict):
            raise ValueError(f"Spatial-composition {label} source receipt is invalid")
        path_value = record.get("path")
        before_hash, after_hash = record.get("sha256_before"), record.get("sha256_after")
        if (not path_value or not isinstance(before_hash, str)
                or not isinstance(after_hash, str) or before_hash != after_hash
                or not re.fullmatch(r"[0-9a-fA-F]{64}", before_hash)):
            raise ValueError(f"Spatial-composition {label} source hashes are incomplete or unstable")
        path = Path(str(path_value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest.lower() != after_hash.lower():
            raise ValueError(f"Spatial-composition {label} source hash mismatch: {path}")
        verified_sources[path] = digest
        return path

    direct_source = verify_source(
        source_inputs.get("direct_base_tie_diagnostic"), "direct base-tie diagnostic")
    relative_records = source_inputs.get("relative_diagnostics")
    if not isinstance(relative_records, list) or not relative_records:
        raise ValueError("Spatial-composition manifest requires relative diagnostic receipts")
    relative_sources = [verify_source(row, f"relative diagnostic {index}")
                        for index, row in enumerate(relative_records)]
    base_record = source_inputs.get("annual_reference_raster")
    base_source = verify_source(base_record, "annual reference raster")
    if (base_record.get("acquisition_date_status")
            != "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"
            or base_record.get("per_pixel_mosaic_component_provenance") != "NOT_RECORDED"):
        raise ValueError("Annual-reference acquisition/provenance uncertainty was not preserved")
    if expected_base_raster is not None:
        expected_path = Path(expected_base_raster).resolve()
        if str(base_source).casefold() != str(expected_path).casefold():
            raise ValueError("Spatial-composition candidates target a different base raster")
    if expected_base_sha256 is not None:
        if verified_sources[base_source].lower() != expected_base_sha256.lower():
            raise ValueError("Spatial-composition base raster hash differs from current base stack")

    candidate_rows = manifest.get("candidate_cells")
    if not isinstance(candidate_rows, list):
        raise ValueError("Spatial-composition manifest must enumerate candidate cells")
    if int(products.get("geopackage_feature_count", -1)) != len(candidate_rows):
        raise ValueError("Spatial-composition GeoPackage feature count differs from the manifest")
    expected: dict[tuple[str, str], dict] = {}
    allowed_statuses = {
        "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
        "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED",
        "SPATIALLY_COMPOSED_CANDIDATE_TREE_NO_CYCLE_CHECK",
    }
    for row in candidate_rows:
        if not isinstance(row, dict):
            raise ValueError("Spatial-composition candidate cell receipt is invalid")
        actual_date = str(row.get("actual_date") or "")
        try:
            date.fromisoformat(actual_date)
        except ValueError as exc:
            raise ValueError("Spatial-composition fields require real actual dates") from exc
        cell_id = str(row.get("cell_id") or "")
        key = (actual_date, cell_id)
        if not cell_id or key in expected:
            raise ValueError(f"Missing or duplicate spatial-composition cell: {key}")
        if (row.get("field_id") != f"SPATIAL_{actual_date}_TO_{anchor_label}"
                or row.get("reference_label") != anchor_label
                or row.get("reference_date") is not None
                or row.get("reference_date_status") != "REFERENCE_ACQUISITION_DATE_UNKNOWN"):
            raise ValueError(f"Spatial-composition actual-date/reference semantics mismatch: {key}")
        if row.get("status") not in allowed_statuses:
            raise ValueError(f"Unsupported spatial-composition status for {key}")
        if (row.get("reviewed") is not False
                or row.get("independent_ground_evidence") is not False
                or row.get("positional_uncertainty_m") is not None
                or row.get("applied") is not False):
            raise ValueError(f"Spatial-composition review/application gate is not closed: {key}")
        correction = row.get("candidate_correction_m")
        geometry_value = row.get("support_geometry_epsg32649")
        metadata = row.get("metadata")
        if (not isinstance(correction, list) or len(correction) != 2
                or not all(isinstance(value, (int, float))
                           and not isinstance(value, bool) and math.isfinite(float(value))
                           for value in correction)
                or not isinstance(geometry_value, dict)
                or not isinstance(metadata, dict)):
            raise ValueError(f"Spatial-composition vector, support or provenance is incomplete: {key}")
        expected[key] = row

    grouped: dict[str, list[DisplacementCell]] = {}
    _require_fiona("Spatially composed alignment field loading")
    seen: set[tuple[str, str]] = set()
    with fiona.open(gpkg_path, layer="spatial_alignment_candidates") as source:
        if source.crs.to_epsg() != 32649:
            raise ValueError("Spatial-composition GeoPackage must use EPSG:32649")
        for feature in source:
            props = dict(feature["properties"])
            actual_date = str(props.get("actual_date") or "")
            cell_id = str(props.get("cell_id") or "")
            key = (actual_date, cell_id)
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate spatial-composition vector cell: {key}")
            row = expected[key]
            for name in ("field_id", "reference_label", "status", "reference_date_status"):
                if str(props.get(name) or "") != str(row.get(name) or ""):
                    raise ValueError(f"Spatial-composition GeoPackage {name} mismatch: {key}")
            correction = row["candidate_correction_m"]
            if (not _same_number(props.get("candidate_dx_m"), correction[0])
                    or not _same_number(props.get("candidate_dy_m"), correction[1])):
                raise ValueError(f"Spatial-composition vector candidate mismatch: {key}")
            metadata = row["metadata"]
            if (int(props.get("reviewed") or 0) != 0
                    or int(props.get("ground_evidence") or 0) != 0
                    or int(props.get("applied") or 0) != 0):
                raise ValueError(f"Spatial-composition GeoPackage bypasses review gate: {key}")
            source_refs = json.loads(str(props.get("source_refs") or "[]"))
            if source_refs != metadata.get("source_candidate_cell_refs", []):
                raise ValueError(f"Spatial-composition source-cell provenance mismatch: {key}")
            geometry = shape(feature["geometry"])
            expected_geometry = shape(row["support_geometry_epsg32649"])
            if (not geometry.is_valid or geometry.is_empty
                    or geometry.symmetric_difference(expected_geometry).area
                    > max(1e-6, expected_geometry.area * 1e-10)):
                raise ValueError(f"Spatial-composition support geometry mismatch: {key}")

            cell_metadata = dict(metadata)
            cell_metadata.setdefault("source_component_scope", "SOURCE_COMPONENT_SCOPE_UNKNOWN")
            cell_metadata.setdefault("reference_raster_sha256", verified_sources[base_source])
            cell_metadata.setdefault(
                "reference_component_id_status",
                "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT")
            cell_metadata.setdefault("candidate_status", row["status"])
            grouped.setdefault(str(row["field_id"]), []).append(DisplacementCell(
                cell_id=cell_id,
                support_geometry=geometry,
                correction_dx_m_candidate=float(correction[0]),
                correction_dy_m_candidate=float(correction[1]),
                fit_dispersion_m=None,
                status=str(row["status"]),
                method="SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE",
                source_ref=f"{gpkg_path}#spatial_alignment_candidates/{cell_id}",
                reviewed=False,
                independent_ground_evidence=False,
                positional_uncertainty_m=None,
                applied=False,
                metadata=cell_metadata,
            ))
            seen.add(key)
    if seen != set(expected):
        raise ValueError("Spatial-composition GeoPackage is missing manifest candidate cells")

    fields = [LocalDisplacementField(
        field_id=field_id,
        reference_date=None,
        moving_date=field_id[len("SPATIAL_"):].split("_TO_", 1)[0],
        cells=tuple(sorted(cells, key=lambda cell: cell.cell_id)),
        crs="EPSG:32649",
        correction_scope="IMAGE_TO_REFERENCE_GRID",
        reference_label=anchor_label,
        reference_source=str(base_source),
        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
    ) for field_id, cells in sorted(grouped.items())]

    if (_sha256(gpkg_path) != gpkg_hash_before
            or _sha256(manifest_path) != manifest_hash_before
            or {path: _sha256(path) for path in verified_sources} != verified_sources):
        raise RuntimeError("Spatial-composition inputs changed while being loaded")
    return fields


def _pair_component(row: dict, index: int) -> float | None:
    pair = row.get("composed_correction_m_candidate")
    return None if pair is None else float(pair[index])


def _lightglue_mosaic_tile_index(
    mosaic_path: Path, diagnostic_grid: dict, actual_dates: list[str],
    *, mosaic_grids_by_source_epoch: dict[str, MosaicTileGrid] | None = None,
) -> tuple[dict, dict[str, dict[tuple[int, int], dict]]]:
    """Validate the shared tile grid and index exact-date source tiles by XY."""
    mosaic = json.loads(mosaic_path.read_text(encoding="utf-8"))
    if mosaic.get("schema_version") != "xian_r5.mosaic_registration_diagnostic.v1":
        raise ValueError("Unsupported LightGlue source mosaic diagnostic schema")
    mosaic_grid = mosaic.get("grid")
    if not isinstance(mosaic_grid, dict):
        raise ValueError("LightGlue source mosaic is missing its tile grid")
    for key in ("crs", "tile_pixel_size", "tile_window", "pixel_transform", "bounds"):
        actual, expected = diagnostic_grid.get(key), mosaic_grid.get(key)
        if key in {"crs", "tile_window"}:
            same = actual == expected
        else:
            try:
                same = (len(actual) == len(expected)
                        and all(_same_number(a, b, tolerance=1e-8)
                                for a, b in zip(actual, expected)))
            except (TypeError, ValueError):
                same = False
        if not same:
            raise ValueError(f"LightGlue diagnostic and source mosaic grid disagree: {key}")
    if mosaic_grid.get("crs") != "EPSG:32649":
        raise ValueError("LightGlue source mosaic grid must use EPSG:32649")

    tile_size_values = mosaic_grid.get("tile_pixel_size")
    window = mosaic_grid.get("tile_window")
    if not isinstance(tile_size_values, list) or len(tile_size_values) != 2:
        raise ValueError("Invalid LightGlue source mosaic tile dimensions")
    if not isinstance(window, dict) or any(
            name not in window for name in ("x0", "y0", "nx", "ny")):
        raise ValueError("Invalid LightGlue source mosaic tile window")
    tile_size = [int(value) for value in tile_size_values]
    x0, y0 = int(window["x0"]), int(window["y0"])
    nx, ny = int(window["nx"]), int(window["ny"])
    if min(tile_size) < 1 or min(nx, ny) < 1:
        raise ValueError("Invalid LightGlue source mosaic tile dimensions")
    transform_values = [float(value) for value in mosaic_grid["pixel_transform"]]
    if len(transform_values) not in (6, 9):
        raise ValueError("Invalid LightGlue source mosaic affine transform")
    affine = Affine(*transform_values[:6])
    width, height = nx * tile_size[0], ny * tile_size[1]
    full_grid = Polygon((affine @ (0, 0), affine @ (width, 0),
                         affine @ (width, height), affine @ (0, height)))
    reported_bounds = mosaic_grid.get("bounds")
    if (not isinstance(reported_bounds, list) or len(reported_bounds) != 4
            or any(not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-5)
                   for a, b in zip(full_grid.bounds, reported_bounds))):
        raise ValueError("LightGlue source mosaic bounds disagree with its affine tile grid")

    records_by_date: dict[str, dict[tuple[int, int], dict]] = {}
    source_tiles = mosaic.get("submeter_tiles")
    if not isinstance(source_tiles, dict) or set(source_tiles) != set(actual_dates):
        raise ValueError("LightGlue source mosaic actual-date tile lists do not match the field dates")
    for actual_date in actual_dates:
        date_records: dict[tuple[int, int], dict] = {}
        entries = source_tiles.get(actual_date)
        if not isinstance(entries, list):
            raise ValueError(f"LightGlue source mosaic has no tile list for {actual_date}")
        for entry in entries:
            match = re.fullmatch(r"(\d+):(\d+)", str(entry.get("tile_id", "")))
            if match is None or not entry.get("path") or not entry.get("sha256"):
                raise ValueError(f"Invalid exact-date tile provenance for {actual_date}")
            tile_x, tile_y = (int(value) for value in match.groups())
            coord = (tile_x, tile_y)
            if not (x0 <= tile_x < x0 + nx and y0 <= tile_y < y0 + ny):
                raise ValueError(f"Source tile lies outside its mosaic window: {entry['tile_id']}")
            if coord in date_records:
                raise ValueError(f"Duplicate source tile for {actual_date}: {entry['tile_id']}")
            record = {
                "tile_id": f"{tile_x}:{tile_y}",
                "path": str(entry["path"]),
                "sha256": str(entry["sha256"]),
                "selected_source_epoch": entry.get("selected_source_epoch"),
                "duplicate_source_status": entry.get("duplicate_source_status"),
            }
            if mosaic_grids_by_source_epoch is not None:
                reported_epoch = str(entry.get("selected_source_epoch") or "")
                if reported_epoch:
                    source_epoch = reported_epoch
                    source_grid = mosaic_grids_by_source_epoch.get(source_epoch)
                    if source_grid is None:
                        raise ValueError(
                            f"Missing stitched-raster grid for source epoch {source_epoch}: "
                            f"{actual_date}/{entry['tile_id']}"
                        )
                    expected_tile_path = source_grid.source_path_for_tile(tile_x, tile_y)
                    if expected_tile_path.resolve() != Path(str(entry["path"])).resolve():
                        raise ValueError(
                            f"Selected source epoch disagrees with tile path: "
                            f"{actual_date}/{entry['tile_id']}"
                        )
                else:
                    epoch_matches = []
                    for candidate_epoch, candidate_grid in mosaic_grids_by_source_epoch.items():
                        try:
                            expected_tile_path = candidate_grid.source_path_for_tile(
                                tile_x, tile_y)
                        except ValueError:
                            continue
                        if expected_tile_path.resolve() == Path(str(entry["path"])).resolve():
                            epoch_matches.append((candidate_epoch, candidate_grid))
                    if len(epoch_matches) != 1:
                        raise ValueError(
                            f"Cannot uniquely resolve stitched-raster source epoch from tile path: "
                            f"{actual_date}/{entry['tile_id']}"
                        )
                    source_epoch, source_grid = epoch_matches[0]
                    record["selected_source_epoch"] = source_epoch
                if source_grid is None:
                    raise ValueError(
                        f"Missing stitched-raster grid for source epoch {source_epoch}: "
                        f"{actual_date}/{entry['tile_id']}"
                    )
                component = source_grid.component_receipt(
                    tile_x,
                    tile_y,
                    actual_date=actual_date,
                    actual_date_source_ref=(
                        f"{mosaic_path.resolve()}#submeter_tiles/{actual_date}/{entry['tile_id']}"
                    ),
                    expected_sha256=str(entry["sha256"]),
                    expected_path=str(entry["path"]),
                )
                if (entry.get("sha256_after_read")
                        and component["source_tile_sha256"] != entry["sha256_after_read"]):
                    raise ValueError(
                        f"Source tile changed after mosaic diagnostic read: {entry['tile_id']}"
                    )
                record["stitched_mosaic_grid"] = {
                    "source_component_window_verified": True,
                    "geid_path": component["geid_path"],
                    "nominal_task_date": component["nominal_task_date"],
                    "mosaic_path": component["mosaic_path"],
                    "mosaic_crs": component["mosaic_crs"],
                    "pixel_window_col_row_width_height": (
                        component["pixel_window_col_row_width_height"]),
                    "window_bounds_epsg4326": component["window_bounds_epsg4326"],
                }
            date_records[coord] = record
        records_by_date[actual_date] = date_records
    return mosaic_grid, records_by_date


def _lightglue_cell_tile_provenance(
    cell: dict, *, cell_id: str, grid: dict,
    records_by_date: dict[str, dict[tuple[int, int], dict]],
) -> tuple[dict, Polygon]:
    """Resolve a cell's north-origin pixel patch to exact-date z20 tile sources."""
    match = re.fullmatch(r"r(\d+)c(\d+)", cell_id)
    if match is None:
        raise ValueError(f"Invalid LightGlue tile-grid cell ID: {cell_id}")
    cell_row, cell_col = (int(value) for value in match.groups())
    try:
        col_raw = cell["tile_col0"]
        row_raw = cell["tile_row0_north_to_south"]
        patch_raw = cell["patch_tiles"]
        tile_rows_raw = cell.get("tile_rows_in_cell", patch_raw)
        tile_cols_raw = cell.get("tile_cols_in_cell", patch_raw)
        col0, row0, patch_tiles = int(col_raw), int(row_raw), int(patch_raw)
        tile_rows = int(tile_rows_raw)
        tile_cols = int(tile_cols_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing LightGlue tile-grid position: {cell_id}") from exc
    if any(isinstance(raw, bool) or not _same_number(raw, integer)
           for raw, integer in ((col_raw, col0), (row_raw, row0), (patch_raw, patch_tiles),
                                (tile_rows_raw, tile_rows), (tile_cols_raw, tile_cols))):
        raise ValueError(f"Non-integer LightGlue tile-grid position: {cell_id}")
    window = grid["tile_window"]
    x0, y0 = int(window["x0"]), int(window["y0"])
    nx, ny = int(window["nx"]), int(window["ny"])
    tile_width, tile_height = (int(value) for value in grid["tile_pixel_size"])
    if (patch_tiles < 1 or col0 != cell_col * patch_tiles
            or row0 != cell_row * patch_tiles or col0 >= nx or row0 >= ny
            or tile_cols != min(patch_tiles, nx - col0)
            or tile_rows != min(patch_tiles, ny - row0)):
        raise ValueError(f"LightGlue cell ID and tile-grid position disagree: {cell_id}")

    pixel_col0, pixel_row0 = col0 * tile_width, row0 * tile_height
    pixel_col1, pixel_row1 = ((col0 + tile_cols) * tile_width,
                              (row0 + tile_rows) * tile_height)
    affine = Affine(*[float(value) for value in grid["pixel_transform"][:6]])
    geometry = Polygon((affine @ (pixel_col0, pixel_row0),
                        affine @ (pixel_col1, pixel_row0),
                        affine @ (pixel_col1, pixel_row1),
                        affine @ (pixel_col0, pixel_row1)))
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Invalid affine tile support polygon: {cell_id}")

    tiles_by_date = {}
    for actual_date, records in records_by_date.items():
        expected_ids = []
        present = []
        missing = []
        for row_delta in range(tile_rows):
            for col_delta in range(tile_cols):
                tile_x = x0 + col0 + col_delta
                tile_y = y0 + ny - 1 - (row0 + row_delta)
                tile_id = f"{tile_x}:{tile_y}"
                expected_ids.append(tile_id)
                record = records.get((tile_x, tile_y))
                if record is None:
                    missing.append(tile_id)
                else:
                    present.append(record)
        tiles_by_date[actual_date] = {
            "expected_tile_ids": expected_ids,
            "present_tiles": present,
            "missing_tile_ids": missing,
            "coverage_fraction": len(present) / len(expected_ids),
        }
    provenance = {
        "tile_window": {key: int(value) for key, value in window.items()},
        "tile_pixel_size": [tile_width, tile_height],
        "pixel_transform": [float(value) for value in grid["pixel_transform"]],
        "tile_col0": col0,
        "tile_row0_north_to_south": row0,
        "tile_rows_in_cell": tile_rows,
        "tile_cols_in_cell": tile_cols,
        "patch_tiles": patch_tiles,
        "pixel_window": [pixel_col0, pixel_row0, pixel_col1, pixel_row1],
        "tiles_by_actual_date": tiles_by_date,
    }
    return provenance, geometry


def direct_base_tie_field_from_epoch(
    epoch_label: str,
    epoch: dict,
    *,
    reference_year: int,
    diagnostic_path: Path,
) -> LocalDisplacementField:
    """Build the shared field contract from one validated diagnostic epoch."""
    moving_date = str(epoch.get("actual_date") or "")
    try:
        date.fromisoformat(moving_date)
    except ValueError as exc:
        raise ValueError(f"Base-tie epoch lacks a real actual date: {epoch_label}") from exc
    expected_label = f"Xian_{int(reference_year)}_TrueColor"
    base_path = Path(str(epoch.get("base_path") or ""))
    if base_path.name != f"{expected_label}.tif":
        raise ValueError(f"Epoch {moving_date} was not tied directly to {expected_label}")
    base_grid = epoch.get("base_grid", {})
    resolution = [float(value) for value in base_grid.get("resolution", ())]
    if (base_grid.get("crs") != "EPSG:32649" or len(resolution) != 2
            or any(not math.isclose(float(value), 10.0, abs_tol=1e-8)
                   for value in resolution)):
        raise ValueError("Direct base-tie grid must be 10 m EPSG:32649")
    transform = [float(value) for value in base_grid.get("transform", ())]
    if len(transform) not in (6, 9):
        raise ValueError(f"Missing base-grid affine transform for {moving_date}")
    a, b, c, d, e, f = transform[:6]
    if abs(b) > 1e-12 or abs(d) > 1e-12 or a <= 0 or e >= 0:
        raise ValueError("Direct base-tie adapter requires a north-up reference grid")
    base_window = [int(value) for value in base_grid.get("window", ())]
    if len(base_window) != 4 or base_window[2] <= 0 or base_window[3] <= 0:
        raise ValueError(f"Invalid base raster window for {moving_date}")
    if not str(epoch.get("base_sha256") or ""):
        raise ValueError(f"Missing base raster hash in direct tie receipt: {moving_date}")

    local = epoch.get("local_spatial_tie_diagnostic")
    if (not isinstance(local, dict) or local.get("field_interpolated") is not False
            or local.get("field_applied") is not False):
        raise ValueError(f"Invalid or applied local tie field for {moving_date}")
    support_mode = str(local.get("support_mode") or "UNIFORM_GRID")
    rows, cols = int(local.get("grid_rows", 0)), int(local.get("grid_cols", 0))
    radius = int(local.get("search_radius_10m_pixels",
                           local.get("search_radius_px", 0)))
    cells = local.get("cells", [])
    if (radius < 1 or not isinstance(cells, list) or not cells
            or (support_mode == "UNIFORM_GRID"
                and (rows < 1 or cols < 1 or len(cells) != rows * cols))
            or support_mode not in {"UNIFORM_GRID", "EXACT_SOURCE_TILE_COMPONENTS"}):
        raise ValueError(f"Incomplete local tie grid for {moving_date}")
    base_left, base_right = c, c + a * base_window[2]
    base_top, base_bottom = f, f + e * base_window[3]
    reference_window_geometry = box(
        min(base_left, base_right), min(base_top, base_bottom),
        max(base_left, base_right), max(base_top, base_bottom))
    cell_ids = set()
    field_cells = []
    source_tile_entries = (epoch.get("submeter_tiles", {}).get("tile_ids")
                           if isinstance(epoch.get("submeter_tiles"), dict) else None)
    source_component_index = _prepare_source_component_index(
        source_tile_entries, moving_date)
    for tie_cell in cells:
        if support_mode == "UNIFORM_GRID":
            row, col = int(tie_cell["row"]), int(tie_cell["col"])
            cell_id = f"r{row}c{col}"
            if not (0 <= row < rows and 0 <= col < cols):
                raise ValueError(f"Invalid local tie cell: {moving_date}/{cell_id}")
        else:
            cell_id = str(tie_cell.get("cell_id") or "")
            if not cell_id:
                raise ValueError(f"Missing source-component tie cell ID: {moving_date}")
        if cell_id in cell_ids:
            raise ValueError(f"Duplicate local tie cell: {moving_date}/{cell_id}")
        cell_ids.add(cell_id)
        x0, y0, x1, y1 = (int(value) for value in tie_cell["base_pixel_window"])
        if not (0 <= x0 < x1 <= base_window[2] and 0 <= y0 < y1 <= base_window[3]):
            raise ValueError(f"Local tie support is outside the reference window: {moving_date}/{cell_id}")
        left, right = c + a * x0, c + a * x1
        top, bottom = f + e * y0, f + e * y1
        if support_mode == "EXACT_SOURCE_TILE_COMPONENTS":
            support_polygon = tie_cell.get("support_polygon_epsg32649")
            if not isinstance(support_polygon, dict):
                raise ValueError(f"Missing exact source-tile support polygon: {moving_date}/{cell_id}")
            support = shape(support_polygon)
            if (support.is_empty or not support.is_valid
                    or support.geom_type not in {"Polygon", "MultiPolygon"}
                    or support.difference(reference_window_geometry).area
                    > max(1e-7, reference_window_geometry.area * 1e-12)):
                raise ValueError(f"Invalid or out-of-window source-tile support: {moving_date}/{cell_id}")
        else:
            support = box(left, min(top, bottom), right, max(top, bottom))
        best = tie_cell.get("best")
        peak = best.get("subpixel_peak") if best else None
        at_boundary = bool(best and (abs(int(best["dx_px"])) >= radius
                                     or abs(int(best["dy_px"])) >= radius))
        correction = None
        if (peak and not at_boundary
                and peak.get("status") == "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED"):
            observed_dx = float(peak["dx_px"])
            observed_dy = float(peak["dy_px"])
            correction = image_shift_to_projected_correction(
                observed_dx, observed_dy, transform)
            status = "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED"
        elif not best:
            status = "DIRECT_BASE_TIE_NO_VALID_TIE"
        elif at_boundary:
            status = "DIRECT_BASE_TIE_SEARCH_BOUNDARY_UNRESOLVED"
        else:
            status = "DIRECT_BASE_TIE_COARSE_ONLY_UNRESOLVED"
        full_pixel_window = [base_window[0] + x0, base_window[1] + y0,
                             base_window[0] + x1, base_window[1] + y1]
        reference_window_evidence = {
            "reference_year_label": int(reference_year),
            "reference_raster_path": str(base_path.resolve()),
            "reference_raster_sha256": str(epoch["base_sha256"]),
            "reference_acquisition_date_status": (
                "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"),
            "reference_component_id_status": (
                "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
            "reference_grid_window_full_raster": base_window,
            "cell_window_within_reference_grid": [x0, y0, x1, y1],
            "cell_window_full_reference_raster": full_pixel_window,
            "reference_grid_transform_local": transform,
            "cell_support_bounds_epsg32649": list(support.bounds),
            "cell_support_mode": support_mode,
            "status": status,
            "correction_dx_m_candidate": (
                None if correction is None else float(correction[0])),
            "correction_dy_m_candidate": (
                None if correction is None else float(correction[1])),
            "best_correlation": best.get("corr") if best else None,
            "peak_margin": tie_cell.get("peak_margin"),
            "eligible_pixels": best.get("n") if best else 0,
            "sample_support_status": tie_cell.get("sample_support_status"),
        }
        source_component_metadata = _source_component_scope_for_support(
            support, source_tile_entries, moving_date,
            prepared_index=source_component_index)
        field_cells.append(DisplacementCell(
            cell_id=cell_id,
            support_geometry=support,
            correction_dx_m_candidate=None if correction is None else correction[0],
            correction_dy_m_candidate=None if correction is None else correction[1],
            fit_dispersion_m=None,
            status=status,
            method="DIRECT_ACTUAL_DATE_TO_SELECTED_10M_BASE_GRADIENT_NCC",
            source_ref=f"{diagnostic_path}#epochs/{epoch_label}/{cell_id}",
            reviewed=False,
            independent_ground_evidence=False,
            positional_uncertainty_m=None,
            applied=False,
            metadata={
                "source_epoch_label": str(epoch_label),
                "source_status": tie_cell.get("status"),
                "actual_date": moving_date,
                "base_acquisition_date_status": "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN",
                "reference_year_label": int(reference_year),
                "reference_raster_path": str(base_path.resolve()),
                "reference_raster_sha256": str(epoch["base_sha256"]),
                "reference_grid_window_full_raster": base_window,
                "cell_window_within_reference_grid": [x0, y0, x1, y1],
                "cell_window_full_reference_raster": full_pixel_window,
                "reference_grid_transform_local": transform,
                "reference_window_evidence": reference_window_evidence,
                "cell_support_mode": support_mode,
                "source_component_tile_id": tie_cell.get("tile_id"),
                **source_component_metadata,
                "reference_component_id_status": (
                    "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
                "best_correlation": best.get("corr") if best else None,
                "peak_margin": tie_cell.get("peak_margin"),
                "eligible_pixels": best.get("n") if best else 0,
                "maximum_masked_samples_available": tie_cell.get(
                    "maximum_masked_samples_available"),
                "minimum_samples_required": tie_cell.get("minimum_samples_required"),
                "sample_support_status": tie_cell.get("sample_support_status"),
                "observed_shift_px": ([float(peak["dx_px"]), float(peak["dy_px"])]
                                       if correction is not None else None),
                "search_radius_10m_pixels": radius,
                "interpolated": False,
                "applied": False,
            },
        ))
    return LocalDisplacementField(
        field_id=f"{moving_date}_to_{expected_label}_DIRECT",
        reference_date=None,
        moving_date=moving_date,
        cells=tuple(sorted(field_cells, key=lambda cell: cell.cell_id)),
        reference_label=expected_label,
        reference_source=str(base_path),
        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
    )


def read_direct_base_tie_fields(
    diagnostic_path: Path,
    *,
    verified_reference_raster_hashes: Mapping[str, str] | None = None,
) -> list[LocalDisplacementField]:
    """Adapt date-specific base ties on one shared coordinate grid.

    Only a genuine subpixel peak is retained. Coarse-only and search-boundary
    peaks stay spatially covered but have no numeric correction vector. When
    dates reference different annual mosaics, the caller must supply their
    independently verified hashes; their pixel grids must still match exactly.
    """
    diagnostic_path = Path(diagnostic_path).resolve()
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    digest_before = _sha256(diagnostic_path)
    report = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "xian_r5.base_tie_diagnostic.v1":
        raise ValueError("Unsupported base-tie diagnostic schema")
    if report.get("automatic_registration_promoted") is not False:
        raise ValueError("Base-tie candidates must not be promoted")
    configured_reference_year = report.get("parameters", {}).get("base_reference_year")
    configured_local_search_radius = report.get("parameters", {}).get(
        "local_search_radius_px")
    if configured_reference_year is not None:
        try:
            configured_reference_year = int(configured_reference_year)
        except (TypeError, ValueError) as exc:
            raise ValueError("Direct base-tie reference year must be an integer") from exc
    if configured_local_search_radius is not None:
        try:
            configured_local_search_radius = int(configured_local_search_radius)
        except (TypeError, ValueError) as exc:
            raise ValueError("Direct base-tie local search radius must be an integer") from exc
        if configured_local_search_radius < 1:
            raise ValueError("Direct base-tie local search radius must be positive")
    epochs = report.get("epochs")
    if not isinstance(epochs, dict) or not epochs:
        raise ValueError("Base-tie report contains no actual-date epochs")

    fields: list[LocalDisplacementField] = []
    seen_dates: set[str] = set()
    reference_hashes_by_path: dict[str, str] = {}
    reference_grid_signature = None
    for epoch_label, epoch in sorted(epochs.items()):
        local_diagnostic = epoch.get("local_spatial_tie_diagnostic") or {}
        if configured_local_search_radius is not None:
            try:
                epoch_local_radius = int(local_diagnostic["search_radius_px"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing local search radius in direct base-tie epoch {epoch_label}") from exc
            if epoch_local_radius != configured_local_search_radius:
                raise ValueError(
                    f"Direct base-tie local search radius mismatch: {epoch_label}")
        epoch_reference_year = epoch.get("base_reference_year", configured_reference_year)
        if epoch_reference_year is None:
            raise ValueError(
                f"Direct base-tie epoch {epoch_label} does not name its annual reference raster")
        try:
            epoch_reference_year = int(epoch_reference_year)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Direct base-tie reference year must be an integer: {epoch_label}") from exc
        if (configured_reference_year is not None
                and epoch_reference_year != configured_reference_year):
            raise ValueError(
                f"Direct base-tie epoch {epoch_label} differs from the fixed reference year")
        field = direct_base_tie_field_from_epoch(
            str(epoch_label), epoch, reference_year=epoch_reference_year,
            diagnostic_path=diagnostic_path)
        if field.moving_date in seen_dates:
            raise ValueError(f"Duplicate actual-date fields must be collapsed: {field.moving_date}")
        seen_dates.add(field.moving_date)
        base_grid = epoch["base_grid"]
        signature = (str(base_grid.get("crs")),
                     tuple(float(value) for value in base_grid["resolution"]),
                     tuple(float(value) for value in base_grid["transform"]),
                     tuple(int(value) for value in base_grid["window"]))
        if reference_grid_signature is not None and signature != reference_grid_signature:
            raise ValueError(
                "Direct-date fields do not share one exact reference pixel grid and window")
        reference_grid_signature = signature
        reference_path_key = (str(Path(epoch["base_path"]).resolve())
                              .replace("\\", "/").casefold())
        reference_digest = str(epoch.get("base_sha256") or "").lower()
        prior_digest = reference_hashes_by_path.setdefault(reference_path_key, reference_digest)
        if not reference_digest or prior_digest != reference_digest:
            raise ValueError(
                f"Direct base-tie receipt has a missing or conflicting reference hash: {epoch_label}")
        if verified_reference_raster_hashes is not None:
            verified_hashes = {
                str(Path(path).resolve()).replace("\\", "/").casefold(): str(digest).lower()
                for path, digest in verified_reference_raster_hashes.items()
            }
            verified_digest = verified_hashes.get(reference_path_key)
            if verified_digest is None:
                raise ValueError(
                    f"Direct base-tie reference is outside the verified 10 m raster stack: {epoch_label}")
            if verified_digest != reference_digest:
                raise ValueError(
                    f"Direct base-tie reference hash differs from the verified 10 m stack: {epoch_label}")
        fields.append(field)
    if (len(reference_hashes_by_path) > 1
            and verified_reference_raster_hashes is None):
        raise ValueError(
            "Date-specific annual references require a verified shared-grid raster stack")
    if digest_before != _sha256(diagnostic_path):
        raise RuntimeError("Direct base-tie diagnostic changed while being read")
    return fields


def read_multi_anchor_base_tie_fields(
    diagnostic_paths: Iterable[Path],
    *,
    canonical_reference_raster: Path,
) -> tuple[list[LocalDisplacementField], dict]:
    """Aggregate date/cell ties across annual rasters on one verified grid.

    The component-wise median is an image-grid candidate only. ``fit_dispersion_m``
    records the median radial deviation among annual-anchor estimates; it is not
    positional uncertainty, independent ground control, or roof-to-ground
    parallax. A cell needs at least two numeric, non-boundary subpixel candidates
    to receive an aggregate vector. No cells are interpolated, reviewed, or
    applied by this adapter.
    """
    paths = sorted({Path(value).resolve() for value in diagnostic_paths})
    if len(paths) < 2:
        raise ValueError("Multi-anchor base ties require at least two annual diagnostics")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("A multi-anchor base-tie diagnostic is missing")

    canonical = Path(canonical_reference_raster).resolve()
    if not canonical.is_file():
        raise FileNotFoundError(f"Canonical 10 m reference raster is missing: {canonical}")
    if canonical.name != "Xian_2019_TrueColor.tif":
        raise ValueError("Multi-anchor fields must target the fixed Xian_2019 reference grid")
    with rasterio.open(canonical) as anchor:
        if (anchor.crs is None or anchor.crs.to_epsg() != 32649
                or any(not math.isclose(abs(float(value)), 10.0, abs_tol=1e-8)
                       for value in anchor.res)):
            raise ValueError("Canonical reference must be a 10 m EPSG:32649 raster")
        canonical_profile = (anchor.crs.to_string(), tuple(anchor.transform),
                             anchor.width, anchor.height)

    report_hashes_before = {str(path): _sha256(path) for path in paths}
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    years: dict[int, tuple[Path, str]] = {}
    reference_profiles = {}
    common_signature = None
    common_dates = None
    common_case = None
    common_inputs = None
    common_settings = None
    fields_by_year: dict[int, dict[str, LocalDisplacementField]] = {}
    epoch_records_by_year: dict[int, dict[str, dict]] = {}

    for path, report in reports:
        if report.get("schema_version") != "xian_r5.base_tie_diagnostic.v1":
            raise ValueError(f"Unsupported base-tie diagnostic schema: {path}")
        if report.get("automatic_registration_promoted") is not False:
            raise ValueError(f"Base-tie input is not candidate-only: {path}")
        if report.get("scope") != "FIXED_CASE_LOCAL_ONLY":
            raise ValueError(f"Multi-anchor ties must be fixed-case diagnostics: {path}")
        case_id = str(report.get("case_id") or "")
        if not case_id:
            raise ValueError(f"Base-tie report has no fixed case ID: {path}")
        reference_year = report.get("parameters", {}).get("base_reference_year")
        try:
            reference_year = int(reference_year)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Base-tie report has no valid reference year: {path}") from exc
        if reference_year not in range(2017, 2023):
            raise ValueError(f"Unexpected annual base reference year: {reference_year}")
        if reference_year in years:
            raise ValueError(f"Duplicate annual base reference year: {reference_year}")

        settings = report.get("parameters", {})
        settings_signature = _base_tie_settings_signature(settings)
        inputs = report.get("inputs_sha256")
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError(f"Base-tie report has no frozen input hashes: {path}")
        if common_case is None:
            common_case, common_inputs, common_settings = case_id, inputs, settings_signature
        elif (case_id != common_case or inputs != common_inputs
              or settings_signature != common_settings):
            raise ValueError("Multi-anchor reports differ in fixed-case inputs or tie settings")

        epochs = report.get("epochs")
        if not isinstance(epochs, dict) or not epochs:
            raise ValueError(f"Base-tie report contains no date-resolved epochs: {path}")
        dates = {str(item.get("actual_date") or "") for item in epochs.values()}
        if "" in dates:
            raise ValueError(f"Base-tie report contains an epoch without an actual date: {path}")
        if common_dates is None:
            common_dates = dates
        elif dates != common_dates:
            raise ValueError("Annual base-tie reports do not cover the same actual dates")

        current_fields = read_direct_base_tie_fields(path)
        fields_by_date = {field.moving_date: field for field in current_fields}
        if set(fields_by_date) != dates:
            raise ValueError(f"Direct tie adapter did not preserve actual dates: {path}")
        epochs_by_date = {str(item["actual_date"]): item for item in epochs.values()}

        for moving_date, epoch in epochs_by_date.items():
            base_path = Path(str(epoch.get("base_path") or "")).resolve()
            expected_name = f"Xian_{reference_year}_TrueColor.tif"
            if base_path.name != expected_name or base_path.parent != canonical.parent:
                raise ValueError(f"Annual base-tie raster does not match reference year {reference_year}")
            expected_digest = str(epoch.get("base_sha256") or "")
            if not expected_digest or _sha256(base_path) != expected_digest:
                raise ValueError(f"Annual base-tie raster hash mismatch: {base_path}")
            with rasterio.open(base_path) as raster:
                if (raster.crs is None or raster.crs.to_epsg() != 32649
                        or len(raster.res) != 2
                        or any(not math.isclose(abs(float(value)), 10.0, abs_tol=1e-8)
                               for value in raster.res)):
                    raise ValueError(f"Annual reference must be a 10 m EPSG:32649 raster: {base_path}")
                profile = (raster.crs.to_string(), tuple(raster.transform),
                           raster.width, raster.height)
                if reference_year == 2019 and base_path != canonical:
                    raise ValueError("The 2019 annual report must use the canonical reference path")
                reference_profiles[str(base_path)] = {
                    "crs": raster.crs.to_string(),
                    "transform": list(raster.transform),
                    "width": raster.width, "height": raster.height,
                    "resolution": list(raster.res),
                    "sha256": expected_digest,
                    "raster_tags": raster.tags(),
                    "image_structure_tags": raster.tags(ns="IMAGE_STRUCTURE"),
                }
            base_grid = epoch.get("base_grid", {})
            window_values = [int(value) for value in base_grid.get("window", ())]
            local_transform = [float(value) for value in base_grid.get("transform", ())]
            if len(window_values) != 4 or len(local_transform) not in (6, 9):
                raise ValueError(f"Annual base-tie report has an invalid local grid: {path}/{moving_date}")
            if profile is None:
                raise ValueError(f"Could not inspect annual base raster: {base_path}")
            signature = (tuple(local_transform), tuple(window_values),
                         tuple(float(value) for value in base_grid.get("resolution", ())),
                         base_grid.get("crs"))
            if common_signature is None:
                common_signature = signature
            elif signature != common_signature:
                raise ValueError("Annual base-tie reports do not share one local reference grid")
            reference_grid_window = Window(*window_values)
            with rasterio.open(canonical) as anchor:
                expected_window_transform = anchor.window_transform(reference_grid_window)
                if any(not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-7)
                       for actual, wanted in zip(local_transform[:6], expected_window_transform[:6])):
                    raise ValueError("Annual tie window is not on the canonical 2019 reference grid")

            # All annual references must share the exact full-grid transform, not
            # merely the same EPSG code and nominal resolution.
            if base_path != canonical and profile != canonical_profile:
                raise ValueError(f"Annual base raster is not co-gridded with Xian_2019: {base_path}")

            highres_signature = json.dumps({
                "submeter_tiles": epoch.get("submeter_tiles"),
                "mosaic_bounds_epsg32649": epoch.get("mosaic_bounds_epsg32649"),
                "mask_geometry_sources": (epoch.get("r4_roof_exclusion") or {}).get(
                    "mask_geometry_sources"),
            }, sort_keys=True, separators=(",", ":"))
            epoch_records_by_year.setdefault(reference_year, {})[moving_date] = {
                "epoch": epoch, "highres_signature": highres_signature,
            }

        years[reference_year] = (base_path, expected_digest)
        fields_by_year[reference_year] = fields_by_date

    if 2019 not in years:
        raise ValueError("The canonical Xian_2019 raster must be present among anchor diagnostics")
    if canonical not in {path for path, _ in years.values()}:
        raise ValueError("No annual diagnostic points to the canonical Xian_2019 raster")

    reference_years = sorted(years)
    for moving_date in common_dates or ():
        signatures = {epoch_records_by_year[year][moving_date]["highres_signature"]
                      for year in reference_years}
        if len(signatures) != 1:
            raise ValueError(f"High-resolution date sources or masks differ across anchors: {moving_date}")

    date_fields = []
    for moving_date in sorted(common_dates or ()):
        cell_maps = {year: {cell.cell_id: cell for cell in field[moving_date].cells}
                     for year, field in fields_by_year.items()}
        expected_cells = set(cell_maps[2019])
        if any(set(cells) != expected_cells for cells in cell_maps.values()):
            raise ValueError(f"Annual base-tie reports have different support cells: {moving_date}")

        aggregate_cells = []
        for cell_id in sorted(expected_cells):
            anchor_rows = []
            anchor_window_evidence = []
            component_signatures = {
                json.dumps({
                    "scope": cell_maps[year][cell_id].metadata.get(
                        "source_component_scope"),
                    "ids": cell_maps[year][cell_id].metadata.get(
                        "source_component_ids"),
                    "coverage": cell_maps[year][cell_id].metadata.get(
                        "source_component_coverage_fraction"),
                }, sort_keys=True, separators=(",", ":"))
                for year in reference_years
            }
            if len(component_signatures) != 1:
                raise ValueError(
                    f"Annual anchors disagree on source-component support: {moving_date}/{cell_id}")
            source_cell_metadata = cell_maps[2019][cell_id].metadata
            source_component_metadata = {
                name: source_cell_metadata[name]
                for name in ("source_component_scope", "source_component_ids",
                             "source_component_coverage_fraction")
                if name in source_cell_metadata
            }
            for year in reference_years:
                cell = cell_maps[year][cell_id]
                anchor_cell_geometry = cell.support_geometry
                if anchor_cell_geometry.symmetric_difference(
                        cell_maps[2019][cell_id].support_geometry).area > 1e-6:
                    raise ValueError(f"Annual base-tie support geometry differs: {moving_date}/{cell_id}")
                evidence = cell.metadata.get("reference_window_evidence")
                if not isinstance(evidence, dict):
                    raise ValueError(
                        f"Annual base-tie cell has no reference-window provenance: {moving_date}/{cell_id}/{year}")
                anchor_window_evidence.append({
                    **evidence,
                    "reference_raster_grid": reference_profiles[str(years[year][0])],
                })
                if cell.correction_dx_m_candidate is None:
                    continue
                anchor_rows.append({
                    "year": year,
                    "dx_m": float(cell.correction_dx_m_candidate),
                    "dy_m": float(cell.correction_dy_m_candidate),
                    "correlation": cell.metadata.get("best_correlation"),
                    "peak_margin": cell.metadata.get("peak_margin"),
                    "source_ref": cell.source_ref,
                })

            enough = len(anchor_rows) >= 2
            if enough:
                dx = float(median(row["dx_m"] for row in anchor_rows))
                dy = float(median(row["dy_m"] for row in anchor_rows))
                radial = [math.hypot(row["dx_m"] - dx, row["dy_m"] - dy)
                          for row in anchor_rows]
                dispersion = float(median(radial))
                correlations = [float(row["correlation"]) for row in anchor_rows
                                if row["correlation"] is not None
                                and math.isfinite(float(row["correlation"]))]
                margins = [float(row["peak_margin"]) for row in anchor_rows
                           if row["peak_margin"] is not None
                           and math.isfinite(float(row["peak_margin"]))]
                status = "MULTI_ANCHOR_BASE_TIE_CANDIDATE_NOT_REVIEWED"
            else:
                dx = dy = dispersion = None
                correlations = margins = []
                status = "MULTI_ANCHOR_BASE_TIE_INSUFFICIENT_SUPPORT"

            anchor_summary = {
                "anchor_count": len(anchor_rows),
                "reference_years": [row["year"] for row in anchor_rows],
                "anchor_vectors_m": {str(row["year"]): [row["dx_m"], row["dy_m"]]
                                      for row in anchor_rows},
                "median_correlation": float(median(correlations)) if correlations else None,
                "median_peak_margin": float(median(margins)) if margins else None,
                "median_radial_deviation_m": dispersion,
                "dispersion_is_positional_uncertainty": False,
            }
            aggregate_cells.append(DisplacementCell(
                cell_id=cell_id,
                support_geometry=cell_maps[2019][cell_id].support_geometry,
                correction_dx_m_candidate=dx,
                correction_dy_m_candidate=dy,
                fit_dispersion_m=dispersion,
                status=status,
                method="MULTI_ANCHOR_BASE_TIE_CONSENSUS",
                source_ref=(f"multi-anchor:{common_case}:{moving_date}:{cell_id}:"
                            + ",".join(str(row["year"]) for row in anchor_rows)),
                reviewed=False,
                independent_ground_evidence=False,
                positional_uncertainty_m=None,
                applied=False,
                metadata={
                    "actual_date": moving_date,
                    "anchor_summary": anchor_summary,
                    "anchor_reports": [str(path) for path in paths],
                    "anchor_reference_years": reference_years,
                    "anchor_reference_sources": {
                        str(year): {"path": str(years[year][0]), "sha256": years[year][1]}
                        for year in reference_years},
                    "reference_window_evidence": anchor_window_evidence,
                    **source_component_metadata,
                    "reference_component_id_status": (
                        "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
                    "reference_grid_sha256": years[2019][1],
                    "base_acquisition_date_status": (
                        "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"),
                    "sample_support_status": ("MULTI_ANCHOR_CANDIDATE_SUPPORT" if enough
                                              else "INSUFFICIENT_ANCHOR_CANDIDATES"),
                    "maximum_masked_samples_available": None,
                    "minimum_samples_required": None,
                    "interpolated": False,
                    "applied": False,
                },
            ))
        date_fields.append(LocalDisplacementField(
            field_id=f"{moving_date}_to_Xian_2019_MULTI_ANCHOR",
            reference_date=None,
            moving_date=moving_date,
            cells=tuple(aggregate_cells),
            reference_label="Xian_2019_TrueColor_MULTI_ANCHOR",
            reference_source=str(canonical),
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
        ))

    integrity_before = {
        **report_hashes_before,
        **{str(path): digest for path, digest in years.values()},
    }
    integrity_after = {path: _sha256(Path(path)) for path in integrity_before}
    if integrity_after != integrity_before:
        raise RuntimeError("Multi-anchor diagnostic or reference raster changed while being read")

    summary = {
        "schema_version": "xian_r5.multi_anchor_base_tie_candidates.v1",
        "case_id": common_case,
        "scope": "FIXED_CASE_LOCAL_ONLY",
        "reference_years": reference_years,
        "canonical_reference_raster": str(canonical),
        "canonical_reference_sha256": years[2019][1],
        "reference_rasters": [
            {"year": year, "path": str(years[year][0]), "sha256": years[year][1],
             "grid": reference_profiles[str(years[year][0])]} for year in reference_years],
        "source_diagnostics": [
            {"path": str(path), "sha256": report_hashes_before[str(path)]}
            for path in paths],
        "actual_dates": sorted(common_dates or ()),
        "field_cell_summaries": [
            {"field_id": field.field_id, "moving_date": field.moving_date,
             "cell_id": cell.cell_id, "status": cell.status,
             "correction_dx_m_candidate": cell.correction_dx_m_candidate,
             "correction_dy_m_candidate": cell.correction_dy_m_candidate,
             "anchor_summary": cell.metadata["anchor_summary"],
             "reference_window_evidence": cell.metadata[
                 "reference_window_evidence"]}
            for field in date_fields for cell in field.cells],
        "field_interpolated": False,
        "field_applied": False,
        "accepted_cells": 0,
        "dispersion_is_positional_uncertainty": False,
        "semantics": (
            "COMPONENTWISE_MEDIAN_OF_ANNUAL_IMAGE_GRID_TIES;UNREVIEWED_CANDIDATE_ONLY;"
            "CROSS_ANCHOR_DISPERSION_IS_NOT_POSITIONAL_UNCERTAINTY_OR_ROOF_TO_GROUND_PARALLAX"),
        "integrity_files_sha256": integrity_before,
    }
    return date_fields, summary


def read_component_tie_consensus_fields(
    manifest_path: Path | str,
    *,
    canonical_reference_raster: Path | str,
) -> tuple[list[LocalDisplacementField], dict[str, object]]:
    """Adapt exact-source-component consensus into unapplied R5 alignment fields.

    The adapter verifies the candidate products and source receipts, then keeps
    confidence failures as explicit no-vector cells. Even a high-confidence
    same-shift candidate remains unreviewed image-to-grid evidence; this reader
    cannot accept it or treat it as roof-to-ground parallax.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    source_hashes: dict[str, str] = {}

    def remember(path_value: Path | str, expected: str | None = None) -> Path:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Component-tie source hash mismatch: {path}")
        prior = source_hashes.setdefault(str(path), digest)
        if prior != digest:
            raise ValueError(f"Conflicting component-tie source hashes: {path}")
        return path

    manifest_hash = _sha256(manifest_path)
    source_hashes[str(manifest_path)] = manifest_hash
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "xian_r5.component_tie_consensus_candidates.v1":
        raise ValueError("Unsupported source-component tie-consensus manifest")
    if (receipt.get("scope") != "FIXED_CASE_LOCAL_ONLY"
            or receipt.get("absolute_ground_position_resolved") is not False
            or receipt.get("ground_footprints_written") is not False
            or receipt.get("roof_to_ground_correction") is not False
            or receipt.get("interpolated") is not False
            or receipt.get("applied") is not False
            or receipt.get("formal_change_products_written") is not False
            or receipt.get("position_accuracy_claimed") is not False):
        raise ValueError("Component-tie receipt bypasses the candidate-only gate")

    canonical = remember(canonical_reference_raster)
    canonical_key = str(canonical).replace("\\", "/").casefold()
    reference_rows = receipt.get("reference_base_rasters", [])
    canonical_rows = [row for row in reference_rows
                      if str(Path(row.get("path", "")).resolve())
                      .replace("\\", "/").casefold() == canonical_key]
    if len(canonical_rows) != 1:
        raise ValueError("Canonical reference raster is not uniquely listed in the component receipt")
    if _sha256(canonical) != canonical_rows[0].get("sha256"):
        raise ValueError("Canonical reference raster hash differs from the component receipt")
    if not reference_rows or len({int(row["reference_year"]) for row in reference_rows}) < 2:
        raise ValueError("At least two annual reference rasters are required for component consensus")
    for row in reference_rows:
        remember(row["path"], row.get("sha256"))
    for row in receipt.get("source_diagnostics", []):
        remember(row["path"], row.get("sha256"))
    fixed_geometry = receipt.get("fixed_case_geometry") or {}
    geometry_path = remember(fixed_geometry.get("path", ""), fixed_geometry.get("sha256"))
    if fixed_geometry.get("crs") != "EPSG:32649":
        raise ValueError("Component-tie fixed-case geometry is not EPSG:32649")

    product_directory = manifest_path.parent
    vector_path = remember(
        receipt.get("candidate_vector_path", ""),
        receipt.get("output_sha256", {}).get(
            Path(receipt.get("candidate_vector_path", "")).name))
    if vector_path.parent != product_directory:
        raise ValueError("Component-tie vector must remain beside its manifest")
    output_hashes = receipt.get("output_sha256", {})
    for filename, expected in output_hashes.items():
        candidate = vector_path if filename == vector_path.name else product_directory / filename
        remember(candidate, expected)

    with fiona.open(geometry_path, layer="fixed_case_polygons") as case_source:
        if case_source.crs.to_epsg() != 32649:
            raise ValueError("Fixed-case geometry source must use EPSG:32649")
        case_geometries = [shape(feature["geometry"])
                           for feature in case_source
                           if feature["properties"].get("case_id") == receipt.get("case_id")
                           and feature["properties"].get("role") == "TARGET"]
    if not case_geometries:
        raise ValueError("Component-tie manifest case has no fixed target geometry")
    case_union = unary_union(case_geometries)

    actual_dates = [str(value) for value in receipt.get("actual_dates", [])]
    try:
        parsed_dates = [date.fromisoformat(value) for value in actual_dates]
    except ValueError as exc:
        raise ValueError("Component-tie observation dates must be ISO dates") from exc
    if (not parsed_dates or parsed_dates != sorted(set(parsed_dates))
            or len(parsed_dates) != len(set(parsed_dates))):
        raise ValueError("Component-tie actual dates must be unique and chronological")

    grouped: dict[str, list[DisplacementCell]] = {actual_date: []
                                                  for actual_date in actual_dates}
    seen = set()
    expected_statuses = {
        "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE",
        "MULTI_REFERENCE_SAME_SHIFT_LOW_CONFIDENCE",
        "MULTI_REFERENCE_SHIFT_DISAGREEMENT",
        "INSUFFICIENT_MULTI_REFERENCE_NUMERIC_SUPPORT",
        "INSUFFICIENT_TIE_SAMPLES", "NO_NUMERIC_TIE",
    }
    sample_status_by_decision = {
        "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE": "NUMERIC_PEAK_AVAILABLE",
        "MULTI_REFERENCE_SAME_SHIFT_LOW_CONFIDENCE": "INSUFFICIENT_ANCHOR_CANDIDATES",
        "MULTI_REFERENCE_SHIFT_DISAGREEMENT": "INSUFFICIENT_ANCHOR_CANDIDATES",
        "INSUFFICIENT_MULTI_REFERENCE_NUMERIC_SUPPORT": "INSUFFICIENT_ANCHOR_CANDIDATES",
        "INSUFFICIENT_TIE_SAMPLES": "INSUFFICIENT_MASKED_SAMPLES",
        "NO_NUMERIC_TIE": "NO_VALID_CORRELATION_SCORE",
    }
    _require_fiona("component-tie consensus field loading")
    with fiona.open(vector_path, layer=receipt.get("candidate_vector_layer")) as source:
        if source.crs.to_epsg() != 32649:
            raise ValueError("Component-tie candidate vector must use EPSG:32649")
        for feature in source:
            props = dict(feature["properties"])
            actual_date = str(props.get("actual_date"))
            tile_id = str(props.get("tile_id"))
            key = (actual_date, tile_id)
            if actual_date not in grouped or not tile_id or key in seen:
                raise ValueError(f"Unexpected or duplicate component-tie cell: {key}")
            seen.add(key)
            decision = str(props.get("decision"))
            if decision not in expected_statuses:
                raise ValueError(f"Unknown component-tie decision status: {decision}")
            geometry = shape(feature["geometry"])
            if (geometry.is_empty or not geometry.is_valid
                    or geometry.geom_type not in {"Polygon", "MultiPolygon"}
                    or geometry.difference(case_union).area > 1e-4):
                raise ValueError(f"Component-tie support is outside its fixed case ROI: {key}")

            source_tile_path = remember(
                props.get("source_tile_path", ""), props.get("source_tile_sha256"))
            try:
                reference_evidence = json.loads(props.get("reference_evidence") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid reference evidence JSON for {key}") from exc
            if (not isinstance(reference_evidence, list)
                    or len(reference_evidence) != len(reference_rows)):
                raise ValueError(f"Incomplete multi-reference evidence for {key}")
            evidence_years = sorted(int(item.get("reference_year"))
                                    for item in reference_evidence)
            expected_years = sorted(int(item["reference_year"]) for item in reference_rows)
            if evidence_years != expected_years:
                raise ValueError(f"Reference years differ from component manifest: {key}")
            if any(item.get("source_tile_path") != str(source_tile_path)
                   or item.get("source_tile_sha256") != props.get("source_tile_sha256")
                   for item in reference_evidence
                   if item.get("source_tile_path") is not None):
                raise ValueError(f"Source tile lineage differs across references: {key}")

            dx, dy = props.get("candidate_dx_m"), props.get("candidate_dy_m")
            if decision == "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE":
                if not _finite_number(dx) or not _finite_number(dy):
                    raise ValueError(f"High-confidence component candidate has no numeric vector: {key}")
                dx, dy = float(dx), float(dy)
                status = "MULTI_ANCHOR_COMPONENT_TIE_CANDIDATE_NOT_REVIEWED"
            else:
                if dx is not None or dy is not None:
                    raise ValueError(f"Rejected component candidate carries a correction: {key}")
                dx = dy = None
                status = decision

            anchor_evidence = []
            for item in reference_evidence:
                reference_year = int(item["reference_year"])
                base_row = next(row for row in reference_rows
                                if int(row["reference_year"]) == reference_year)
                correction = item.get("candidate_moving_to_reference_correction_m")
                anchor_evidence.append({
                    "reference_year_label": reference_year,
                    "reference_raster_path": base_row["path"],
                    "reference_raster_sha256": base_row["sha256"],
                    "candidate_correction_dx_m": correction[0] if correction else None,
                    "candidate_correction_dy_m": correction[1] if correction else None,
                    "correlation": item.get("correlation"),
                    "peak_margin": item.get("peak_margin"),
                    "sample_count": item.get("sample_count"),
                    "minimum_samples_required": item.get("minimum_samples_required"),
                    "maximum_masked_samples_available": item.get(
                        "maximum_masked_samples_available"),
                    "high_confidence_gate_pass": item.get("high_confidence_gate_pass"),
                })
            available_counts = [
                int(item.get("maximum_masked_samples_available"))
                if item.get("maximum_masked_samples_available") is not None
                else int(item["sample_count"])
                for item in anchor_evidence
                if (item.get("maximum_masked_samples_available") is not None
                    or item.get("sample_count") is not None)]
            required_counts = [int(item["minimum_samples_required"])
                               for item in anchor_evidence
                               if item.get("minimum_samples_required") is not None]
            metadata = {
                "actual_date": actual_date,
                "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
                "source_component_ids": [{
                    "actual_date": actual_date, "tile_id": tile_id,
                    "path": str(source_tile_path),
                    "sha256": str(props.get("source_tile_sha256")),
                }],
                "source_component_coverage_fraction": 1.0,
                "sample_support_status": sample_status_by_decision[decision],
                "maximum_masked_samples_available": (
                    min(available_counts) if available_counts else None),
                "minimum_samples_required": (
                    max(required_counts) if required_counts else None),
                "component_consensus_status": decision,
                "reference_window_evidence": anchor_evidence,
                "reference_component_id_status": (
                    "HASH_REVALIDATED_EXACT_SOURCE_TILE_COMPONENT_ACROSS_ANCHORS"),
                "cross_anchor_dispersion_is_positional_uncertainty": False,
                "interpolated": False,
                "applied": False,
                "position_accuracy_claimed": False,
                "roof_to_ground_correction": False,
            }
            grouped[actual_date].append(DisplacementCell(
                cell_id=tile_id, support_geometry=geometry,
                correction_dx_m_candidate=dx, correction_dy_m_candidate=dy,
                fit_dispersion_m=None, status=status,
                method="MULTI_ANCHOR_SOURCE_COMPONENT_TIE_CONSENSUS",
                source_ref=(f"{vector_path}#layer={receipt['candidate_vector_layer']}"
                            f"&actual_date={actual_date}&tile_id={tile_id}"),
                reviewed=False, independent_ground_evidence=False,
                positional_uncertainty_m=None, applied=False, metadata=metadata,
            ))

    fields = []
    for actual_date in actual_dates:
        cells = tuple(grouped[actual_date])
        expected_count = receipt.get("products_by_actual_date", {}).get(
            actual_date, {}).get("case_roi_component_count")
        if not cells or len(cells) != expected_count:
            raise ValueError(f"Component-tie cell inventory mismatch for {actual_date}")
        fields.append(LocalDisplacementField(
            field_id=f"{actual_date}_to_component_multi_anchor",
            reference_date=None, moving_date=actual_date, cells=cells,
            reference_label="MULTI_REFERENCE_EXACT_SOURCE_COMPONENT_TIES",
            reference_source=str(canonical),
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
        ))

    after = {source_path: _sha256(Path(source_path))
             for source_path in source_hashes}
    if after != source_hashes:
        raise RuntimeError("Component-tie sources changed while loading the field")
    summary = {
        "schema_version": "xian_r5.component_tie_consensus_field_adapter.v1",
        "case_id": receipt.get("case_id"),
        "scope": "FIXED_CASE_LOCAL_ONLY",
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
        "candidate_vector_path": str(vector_path),
        "canonical_reference_raster": str(canonical),
        "reference_years": sorted(int(row["reference_year"]) for row in reference_rows),
        "actual_dates": actual_dates,
        "field_cell_count_by_actual_date": {
            actual_date: len(grouped[actual_date]) for actual_date in actual_dates},
        "field_cell_summaries": [
            {
                "field_id": field.field_id, "moving_date": field.moving_date,
                "cell_id": cell.cell_id, "status": cell.status,
                "component_consensus_status": cell.metadata[
                    "component_consensus_status"],
                "correction_dx_m_candidate": cell.correction_dx_m_candidate,
                "correction_dy_m_candidate": cell.correction_dy_m_candidate,
                "source_component_ids": cell.metadata["source_component_ids"],
                "sample_support_status": cell.metadata["sample_support_status"],
            }
            for field in fields for cell in field.cells],
        "numeric_candidate_cell_count": sum(
            cell.correction_dx_m_candidate is not None
            for field in fields for cell in field.cells),
        "accepted_cell_count": sum(cell.accepted for field in fields for cell in field.cells),
        "field_applied": False,
        "field_interpolated": False,
        "absolute_ground_position_resolved": False,
        "formal_change_use": False,
        "integrity_files_sha256": source_hashes,
    }
    return fields, summary


def read_lightglue_local_field_candidates(
    diagnostic_path: Path, *, expected_base_raster: Path | None = None,
    expected_base_sha256: str | None = None,
    mosaic_grids_by_source_epoch: dict[str, MosaicTileGrid] | None = None,
) -> list[LocalDisplacementField]:
    """Adapt an ALIKED/LightGlue local-field receipt into unapplied tie fields.

    The paired JSON, candidate GeoPackage, source mosaic receipt, model receipt,
    and cached weights are hash-checked. Candidate geometry and attributes must
    agree between JSON and GeoPackage. This adapter cannot mark a tie reviewed,
    promote a correction, interpolate a field, or correct roof-to-ground parallax.
    """
    diagnostic_path = Path(diagnostic_path).resolve()
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    source_hashes: dict[Path, str] = {}

    def remember(path_value: str | Path, expected: str | None = None) -> Path:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"LightGlue field source hash mismatch: {path}")
        source_hashes[path] = digest
        return path

    report_hash = _sha256(diagnostic_path)
    source_hashes[diagnostic_path] = report_hash
    report = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "xian_r5.local_tile_lightglue_field_diagnostic.v1":
        raise ValueError("Unsupported ALIKED/LightGlue local-field diagnostic schema")
    reported_transform_model = report.get("transform_model")
    if (reported_transform_model is not None
            and reported_transform_model != LOCAL_TRANSFORM_MODEL):
        raise ValueError("Unsupported ALIKED/LightGlue local transform model")
    if (report.get("scope") != "FIXED_CASE_LOCAL_ONLY"
            or report.get("registration_promoted") is not False
            or report.get("registration_applied") is not False
            or report.get("field_interpolated") is not False):
        raise ValueError("Only fixed-case, unpromoted, unapplied LightGlue candidates can be loaded")
    if report.get("mask", {}).get("ground_mask_reviewed") is not False:
        raise ValueError("LightGlue candidate ground mask is not independently reviewed")
    grid = report.get("grid", {})
    if grid.get("crs") != "EPSG:32649":
        raise ValueError("LightGlue candidate grid must use EPSG:32649")

    base = report.get("base_raster", {})
    if (base.get("crs") != "EPSG:32649"
            or tuple(float(value) for value in base.get("resolution", ())) != (10.0, 10.0)
            or not base.get("sha256")):
        raise ValueError("LightGlue local-field QA must reference a 10 m EPSG:32649 raster")
    base_path = Path(base.get("path", "")).resolve()
    if expected_base_raster is not None:
        expected_path = Path(expected_base_raster).resolve()
        if str(base_path).casefold() != str(expected_path).casefold():
            raise ValueError("LightGlue local-field diagnostic targets a different base raster")
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        if expected_base_sha256 is not None:
            if expected_base_sha256 != base["sha256"]:
                raise ValueError("LightGlue local-field base-raster hash differs from current base stack")
            source_hashes[base_path] = expected_base_sha256
        else:
            remember(base_path, base["sha256"])
    else:
        remember(base_path, base["sha256"])

    mosaic_path = remember(report.get("input_mosaic_diagnostic", ""),
                           report.get("input_mosaic_diagnostic_sha256"))
    model = report.get("model", {})
    remember(model.get("receipt_path", ""), model.get("receipt_sha256"))
    weight_hashes = model.get("weights_sha256", {})
    if len(weight_hashes) < 2:
        raise ValueError("LightGlue diagnostic does not identify both model weights")
    for path_value, digest in weight_hashes.items():
        remember(path_value, digest)

    vector_path = remember(report.get("outputs", {}).get("candidate_vector", ""))
    expected_vector_path = (diagnostic_path.parent / "LOCAL_TILE_FIELD_CANDIDATES.gpkg").resolve()
    if vector_path != expected_vector_path:
        raise ValueError("LightGlue candidate GeoPackage does not match the diagnostic directory")

    dates = report.get("actual_dates_in_order", [])
    try:
        parsed_dates = [date.fromisoformat(str(value)) for value in dates]
    except (TypeError, ValueError) as exc:
        raise ValueError("LightGlue field dates must be real ISO acquisition dates") from exc
    if (len(parsed_dates) < 2 or parsed_dates != sorted(set(parsed_dates))
            or len(parsed_dates) != len(set(parsed_dates))):
        raise ValueError("LightGlue diagnostic dates must be unique and chronologically ordered")
    dates = [str(value) for value in dates]
    mosaic_grid, mosaic_tiles_by_date = _lightglue_mosaic_tile_index(
        mosaic_path, grid, dates,
        mosaic_grids_by_source_epoch=mosaic_grids_by_source_epoch)
    if mosaic_grids_by_source_epoch is not None:
        for records in mosaic_tiles_by_date.values():
            for record in records.values():
                source_path = Path(record["path"]).resolve()
                prior = source_hashes.setdefault(source_path, record["sha256"])
                if prior != record["sha256"]:
                    raise ValueError(f"Conflicting source tile receipts: {source_path}")

    expected_rows: dict[tuple[str, str, str], dict] = {}
    tile_provenance_by_key: dict[tuple[str, str, str], dict] = {}
    pair_metadata: dict[tuple[str, str], dict] = {}
    pair_rows = report.get("pairwise_fields", [])
    for pair in pair_rows:
        from_date, to_date = str(pair.get("from_date")), str(pair.get("to_date"))
        if (from_date not in dates or to_date not in dates
                or date.fromisoformat(to_date) <= date.fromisoformat(from_date)):
            raise ValueError(f"Invalid LightGlue date-pair ordering: {from_date}/{to_date}")
        pair_key = (from_date, to_date)
        if pair_key in pair_metadata:
            raise ValueError(f"Duplicate LightGlue date pair: {pair_key}")
        if (pair.get("field_interpolated") is not False
                or pair.get("field_applied") is not False):
            raise ValueError(f"LightGlue pair was interpolated or applied: {pair_key}")
        pair_metadata[pair_key] = pair
        cells = pair.get("cells", [])
        if len(cells) != int(pair.get("cell_count", -1)) or not cells:
            raise ValueError(f"Incomplete LightGlue local cell list: {pair_key}")
        for cell in cells:
            cell_id = str(cell.get("cell_id", ""))
            key = (from_date, to_date, cell_id)
            if not cell_id or key in expected_rows:
                raise ValueError(f"Missing or duplicate LightGlue local cell: {key}")
            if (cell.get("ground_mask_reviewed") is not False
                    or cell.get("field_interpolated") is not False
                    or cell.get("applied_to_image_or_geometry") is not False
                    or str(cell.get("status", "")).startswith("ACCEPTED")):
                raise ValueError(f"LightGlue cell bypasses the unreviewed-candidate gate: {key}")
            cell_transform_model = cell.get("transform_model")
            if (cell_transform_model is not None
                    and cell_transform_model != LOCAL_TRANSFORM_MODEL):
                raise ValueError(f"Unsupported LightGlue cell transform model: {key}")
            dx, dy = cell.get("correction_dx_m_candidate"), cell.get("correction_dy_m_candidate")
            if (dx is None) != (dy is None):
                raise ValueError(f"Incomplete LightGlue displacement vector: {key}")
            if "polygon_epsg32649" not in cell:
                raise ValueError(f"Missing LightGlue support geometry: {key}")
            tile_provenance, affine_geometry = _lightglue_cell_tile_provenance(
                cell, cell_id=cell_id, grid=mosaic_grid,
                records_by_date=mosaic_tiles_by_date)
            report_geometry = shape(cell["polygon_epsg32649"])
            if (not report_geometry.is_valid or report_geometry.is_empty
                    or report_geometry.symmetric_difference(affine_geometry).area > 1e-5):
                raise ValueError(
                    f"LightGlue support geometry differs from affine tile grid: {key}")
            expected_rows[key] = cell
            tile_provenance_by_key[key] = tile_provenance

    expected_pairs = {(dates[i], dates[j]) for i in range(len(dates))
                       for j in range(i + 1, len(dates))}
    if set(pair_metadata) != expected_pairs:
        raise ValueError("LightGlue diagnostic does not contain every date-coherent pair")

    grouped: dict[tuple[str, str], list[DisplacementCell]] = {}
    seen: set[tuple[str, str, str]] = set()
    _require_fiona("LightGlue alignment field loading")
    with fiona.open(vector_path, layer="local_tile_field_candidates") as source:
        if source.crs.to_epsg() != 32649:
            raise ValueError("LightGlue candidate GeoPackage must use EPSG:32649")
        for feature in source:
            props = dict(feature["properties"])
            key = (str(props.get("from_date")), str(props.get("to_date")),
                   str(props.get("cell_id")))
            if key not in expected_rows or key in seen:
                raise ValueError(f"Unexpected or duplicate LightGlue GeoPackage cell: {key}")
            row = expected_rows[key]
            comparisons = (
                ("status", props.get("status"), row.get("status")),
                ("raw_n", props.get("raw_n"), row.get("matched_features_raw")),
                ("eligible_n", props.get("eligible_n"), row.get("eligible_matches")),
                ("inlier_n", props.get("inlier_n"), row.get("inliers")),
                ("ratio", props.get("ratio"), row.get("inlier_ratio")),
                ("coverage", props.get("coverage"), row.get("coverage_4x4")),
                ("p90_px", props.get("p90_px"), row.get("p90_residual_px")),
                ("p90_m", props.get("p90_m"), row.get("p90_residual_m")),
                ("corr_dx_m", props.get("corr_dx_m"), row.get("correction_dx_m_candidate")),
                ("corr_dy_m", props.get("corr_dy_m"), row.get("correction_dy_m_candidate")),
            )
            for label, actual, expected in comparisons:
                same = (actual == expected if label == "status"
                        else _same_number(actual, expected, tolerance=1e-5))
                if not same:
                    raise ValueError(f"LightGlue GeoPackage {label} differs from JSON: {key}")
            vector_transform_model = props.get("transform_model")
            expected_transform_model = row.get("transform_model", LOCAL_TRANSFORM_MODEL)
            if (vector_transform_model is not None
                    and vector_transform_model != expected_transform_model):
                raise ValueError(
                    f"LightGlue GeoPackage transform model differs from JSON: {key}")
            if int(props.get("reviewed") or 0) != 0 or int(props.get("applied") or 0) != 0:
                raise ValueError(f"LightGlue GeoPackage has a reviewed/applied cell: {key}")
            geometry = shape(feature["geometry"])
            expected_geometry = shape(row["polygon_epsg32649"])
            if (not geometry.is_valid or geometry.is_empty
                    or geometry.symmetric_difference(expected_geometry).area > 1e-5):
                raise ValueError(f"LightGlue GeoPackage support geometry differs from JSON: {key}")

            from_date, to_date, cell_id = key
            metadata = {
                name: row.get(name) for name in (
                    "candidate_ground_mask_fraction_from", "candidate_ground_mask_fraction_to",
                    "matched_features_raw", "eligible_matches", "inliers", "inlier_ratio",
                    "coverage_4x4", "median_residual_px", "p90_residual_px",
                    "median_residual_m", "p90_residual_m",
                    "observed_dx_px", "observed_dy_px", "ground_mask_reviewed",
                    "field_interpolated", "applied_to_image_or_geometry")
            }
            metadata["model_device"] = model.get("device")
            metadata["transform_model"] = row.get(
                "transform_model", reported_transform_model or LOCAL_TRANSFORM_MODEL)
            metadata["model_weight_sha256"] = weight_hashes
            metadata["source_tile_grid"] = {
                **tile_provenance_by_key[key],
                "mosaic_diagnostic_path": str(mosaic_path),
                "mosaic_diagnostic_sha256": source_hashes[mosaic_path],
            }
            metadata.update(_lightglue_source_component_scope(
                tile_provenance_by_key[key], from_date, to_date))
            grouped.setdefault((from_date, to_date), []).append(DisplacementCell(
                cell_id=cell_id,
                support_geometry=geometry,
                correction_dx_m_candidate=(float(row["correction_dx_m_candidate"])
                                           if row["correction_dx_m_candidate"] is not None else None),
                correction_dy_m_candidate=(float(row["correction_dy_m_candidate"])
                                           if row["correction_dy_m_candidate"] is not None else None),
                fit_dispersion_m=None,
                status=str(row["status"]),
                method=LOCAL_FIELD_METHOD,
                source_ref=f"{diagnostic_path}#pair:{from_date}/{to_date}/cell:{cell_id}",
                reviewed=False,
                independent_ground_evidence=False,
                positional_uncertainty_m=None,
                applied=False,
                metadata=metadata,
            ))
            seen.add(key)
    if seen != set(expected_rows):
        raise ValueError("LightGlue candidate GeoPackage is missing JSON cells")

    for path, expected in source_hashes.items():
        if (path == base_path and expected_base_raster is not None
                and expected_base_sha256 is not None):
            # The production caller independently hashes the reference raster
            # before and after reconstruction; avoid a third city-raster scan.
            continue
        if _sha256(path) != expected:
            raise RuntimeError(f"LightGlue field source changed while being read: {path}")
    fields = []
    for from_date, to_date in sorted(grouped):
        fields.append(LocalDisplacementField(
            field_id=f"ALIKED_{from_date}_to_{to_date}",
            reference_date=from_date,
            moving_date=to_date,
            cells=tuple(sorted(grouped[(from_date, to_date)], key=lambda cell: cell.cell_id)),
            reference_label=from_date,
            reference_source=str(mosaic_path),
        ))
    return fields
