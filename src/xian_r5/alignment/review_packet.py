"""Export a fail-closed, traceable package for human alignment review.

The base-grid review layer contains numeric direct ties to each date's recorded,
co-gridded 10 m reference raster. Relative date-pair candidates are kept in a
separate diagnostic GeoPackage because they cannot independently anchor an
image to the base grid. Neither output asserts absolute ground accuracy or
roof-to-ground parallax correction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from shapely.geometry import MultiPolygon, box, mapping, shape
from shapely.ops import unary_union

from .field_io import read_spatial_alignment_composition_fields
from .local_field import LOCAL_FIELD_METHOD


DIRECT_METHOD = "DIRECT_ACTUAL_DATE_TO_SELECTED_10M_BASE_GRADIENT_NCC"
RELATIVE_METHOD = LOCAL_FIELD_METHOD
LEGACY_RELATIVE_METHOD = "ALIKED_LIGHTGLUE_LOCAL_AFFINE_RANSAC_CANDIDATE"
RELATIVE_METHODS = {RELATIVE_METHOD, LEGACY_RELATIVE_METHOD}
MULTI_ANCHOR_METHOD = "MULTI_ANCHOR_BASE_TIE_CONSENSUS"
SPATIAL_COMPOSITION_METHOD = "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE"
SOURCE_LAYER = "image_alignment_candidates"
REVIEW_LAYER = "reviewed_image_alignment_cells"
RELATIVE_LAYER = "relative_date_pair_candidates"
R5_RELATIVE_REVIEW_SCOPE_ID = "XIAN_R5_GROUNDREG_FREEZE_0929"
R5_RELATIVE_REVIEW_SCOPE = (
    ("c0007_a00007", "2018-04-29"),
    ("c0007_a00007", "2018-11-01"),
    ("c0007_a00007", "2019-12-02"),
    ("c0007_a00012", "2018-04-29"),
    ("c0007_a00012", "2018-11-01"),
    ("c0007_a00012", "2019-12-02"),
    ("c0007_a00013", "2018-04-29"),
    ("c0007_a00013", "2018-11-01"),
    ("c0007_a00013", "2019-12-02"),
)
SUPPORTED_PRODUCT_SCHEMAS = {
    "xian_r5.grounded_products.v4",
    "xian_r5.grounded_products.v5",
    "xian_r5.grounded_products.v6",
    "xian_r5.grounded_products.v7",
    "xian_r5.grounded_products.v8",
    "xian_r5.grounded_products.v13",
    "xian_r5.grounded_products.v14",
    "xian_r5.grounded_products.v15",
    "xian_r5.grounded_products.v16",
    "xian_r5.grounded_products.v17",
}

REVIEW_FIELDS = {
    "field_id": "str:160",
    "cell_id": "str:160",
    "moving_date": "str:16",
    "reference_source": "str:1024",
    "reference_source_sha256": "str:64",
    "candidate_source_ref": "str:2048",
    "correction_dx_m": "float",
    "correction_dy_m": "float",
    "independent_ground_evidence": "int",
    "ground_evidence_type": "str:48",
    "review_basis": "str:48",
    "alignment_evidence_type": "str:64",
    "reviewed": "int",
    "positional_uncertainty_m": "float",
    "positional_uncertainty_semantics": "str:40",
    "uncertainty_semantics": "str:40",
    "relative_registration_uncertainty_m": "float",
    "fit_max_adjusted_bound_m": "float",
    "holdout_max_adjusted_bound_m": "float",
    "nonbuilding_tie_pairs": "int",
    "holdout_tie_pairs": "int",
    "relative_ties_within_support_cell": "int",
    "reviewer_id": "str:128",
    "reviewed_at": "str:48",
    "evidence_path": "str:1024",
    "evidence_sha256": "str:64",
    "review_state": "str:24",
    "candidate_status": "str:80",
    "method": "str:96",
    "fit_dispersion_m": "float",
    "candidate_position_uncertainty_m": "float",
    "source_feature_id": "str:32",
    "source_run_gpkg_sha256": "str:64",
    "reference_date_status": "str:48",
    "correction_scope": "str:48",
    "anchor_count": "int",
    "anchor_years": "str:64",
    "anchor_median_correlation": "float",
    "anchor_median_peak_margin": "float",
    "fit_dispersion_kind": "str:80",
    "source_component_scope": "str:48",
    "reference_component_id_status": "str:96",
    "cycle_closure_max_m": "float",
    "source_edge_count": "int",
    "accepted": "int",
    "applied": "int",
}

RELATIVE_FIELDS = {
    "field_id": "str:160",
    "cell_id": "str:160",
    "reference_date": "str:16",
    "moving_date": "str:16",
    "reference_source": "str:1024",
    "candidate_source_ref": "str:2048",
    "candidate_status": "str:80",
    "method": "str:96",
    "correction_dx_m_candidate": "float",
    "correction_dy_m_candidate": "float",
    "fit_dispersion_m_candidate": "float",
    "candidate_position_uncertainty_m": "float",
    "matched_features_raw": "int",
    "eligible_matches": "int",
    "inliers": "int",
    "inlier_ratio": "float",
    "coverage_4x4": "float",
    "median_residual_px": "float",
    "p90_residual_px": "float",
    "candidate_ground_mask_fraction_from": "float",
    "candidate_ground_mask_fraction_to": "float",
    "ground_mask_reviewed": "int",
    "field_interpolated": "int",
    "applied_to_image_or_geometry": "int",
    "accepted": "int",
    "applied": "int",
    "review_state": "str:64",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _flag(value: object, name: str) -> bool:
    if value in (False, 0, None):
        return False
    if value in (True, 1):
        return True
    raise ValueError(f"{name} must be stored as 0 or 1")


def _absolute_path(value: object, *, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Candidate source path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _read_vector(path: Path, layer: str) -> tuple[list[dict[str, Any]], str]:
    """Read vector features using Fiona, with Pyogrio as an optional fallback."""
    try:
        import fiona
    except ImportError:
        fiona = None

    if fiona is not None:
        with fiona.open(path, layer=layer) as source:
            crs = source.crs_wkt or str(source.crs)
            records = []
            for ordinal, feature in enumerate(source):
                if not feature.get("geometry"):
                    raise ValueError(f"Missing candidate geometry at source feature {feature.id}")
                records.append({
                    "feature_id": str(feature.id if feature.id is not None else ordinal),
                    "properties": dict(feature["properties"]),
                    "geometry": shape(feature["geometry"]),
                })
            return records, crs

    try:
        import pyogrio
    except ImportError as exc:
        raise RuntimeError("Review packet export requires Fiona or Pyogrio") from exc
    frame = pyogrio.read_dataframe(path, layer=layer)
    info = pyogrio.read_info(path, layer=layer)
    records = []
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            raise ValueError(f"Missing candidate geometry at source feature {ordinal}")
        records.append({
            "feature_id": str(ordinal),
            "properties": {name: row[name] for name in info["fields"]},
            "geometry": geometry,
        })
    return records, str(info.get("crs") or "")


def _multipolygon(geometry):
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Candidate support geometry must be nonempty and valid")
    if geometry.geom_type == "Polygon":
        return MultiPolygon([geometry])
    if geometry.geom_type == "MultiPolygon":
        return geometry
    raise ValueError(f"Candidate support must be Polygon or MultiPolygon, got {geometry.geom_type}")


def _write_vector(path: Path, layer: str, records: list[dict[str, Any]], fields: dict[str, str]) -> None:
    """Write one-layer GeoPackage through Fiona, or Pyogrio when Fiona is absent."""
    schema = {"geometry": "MultiPolygon", "properties": fields}
    try:
        import fiona
    except ImportError:
        fiona = None

    if fiona is not None:
        with fiona.open(path, "w", driver="GPKG", layer=layer, schema=schema,
                        crs="EPSG:32649") as sink:
            for record in records:
                sink.write({
                    "geometry": mapping(_multipolygon(record["geometry"])),
                    "properties": {
                        name: record["properties"].get(name) for name in fields
                    },
                })
        return

    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import pyogrio
    except ImportError as exc:
        raise RuntimeError("Review packet export requires Fiona or GeoPandas/Pyogrio") from exc

    rows = []
    for record in records:
        rows.append({**{name: record["properties"].get(name) for name in fields},
                     "geometry": _multipolygon(record["geometry"])})
    if rows:
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32649")
    else:
        empty_columns = {}
        for name, specification in fields.items():
            dtype = ("object" if specification.startswith("str") else
                     "int64" if specification == "int" else "float64")
            empty_columns[name] = pd.Series([], dtype=dtype)
        empty_geometry = gpd.GeoSeries([], crs="EPSG:32649")
        frame = gpd.GeoDataFrame(empty_columns, geometry=empty_geometry,
                                 crs="EPSG:32649")
    for name, specification in fields.items():
        if specification.startswith("str"):
            frame[name] = frame[name].fillna("").astype(object)
        elif specification == "int":
            frame[name] = frame[name].fillna(0).astype("int64")
        elif specification == "float":
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    pyogrio.write_dataframe(frame, path, layer=layer, driver="GPKG",
                            geometry_type="MultiPolygon")


def _validate_run(source_run: Path) -> tuple[dict, dict, dict, dict[str, Path]]:
    qa_path = source_run / "ENGINEERING_QA.json"
    if not qa_path.is_file():
        audit_paths = sorted(source_run.glob("ENGINEERING_QA_INDEPENDENT_AUDIT_*.json"))
        if len(audit_paths) == 1:
            qa_path = audit_paths[0]
        elif len(audit_paths) > 1:
            raise ValueError(
                "Fixed-case product has no canonical engineering QA and multiple independent audits"
            )
    manifest_path = source_run / "PRODUCT_MANIFEST.json"
    receipt_path = source_run / "EVIDENCE_RECEIPT.json"
    gpkg_path = source_run / "building_lineage.gpkg"
    for path in (qa_path, manifest_path, receipt_path, gpkg_path):
        if not path.is_file():
            raise FileNotFoundError(f"Fixed-case product is incomplete: {path}")

    qa, product, receipt = _json(qa_path), _json(manifest_path), _json(receipt_path)
    if qa.get("schema_version") != "xian_r5.reconstruction_engineering_qa.v1":
        raise ValueError("Unsupported engineering QA schema")
    if qa.get("structural_qa") != "PASS" or qa.get("errors") != []:
        raise ValueError("Review packet requires a fixed case with passing structural QA")
    for key in ("r4_status_sha256_unchanged", "tracked_input_hashes_unchanged",
                "pixel_input_hashes_unchanged", "base_raster_hashes_unchanged"):
        if qa.get(key) is not True:
            raise ValueError(f"Source QA does not certify unchanged inputs: {key}")
    if product.get("schema_version") not in SUPPORTED_PRODUCT_SCHEMAS:
        raise ValueError("Unsupported grounded-product manifest schema")
    if product.get("crs") != "EPSG:32649" or product.get("image_alignment_field_applied") is not False:
        raise ValueError("Review packet requires unapplied EPSG:32649 alignment candidates")
    if receipt.get("schema_version") != "xian_r5.grounded_fixed_case_run.v1":
        raise ValueError("Unsupported fixed-case evidence receipt schema")
    if qa.get("case_id") != receipt.get("case_id"):
        raise ValueError("Engineering QA and evidence receipt refer to different fixed cases")
    if receipt.get("image_grid_alignment_applied_count") != 0:
        raise ValueError("Fixed-case receipt must explicitly report zero applied image-alignment corrections")

    sidecars = {
        "engineering_qa": qa_path,
        "product_manifest": manifest_path,
        "evidence_receipt": receipt_path,
        "source_geopackage": gpkg_path,
    }
    return qa, product, receipt, sidecars


def _receipt_sources(receipt: dict, field_name: str) -> dict[Path, str]:
    if field_name not in receipt or receipt[field_name] is None:
        return {}
    value = receipt[field_name]
    rows = value if isinstance(value, list) else [value]
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ValueError(f"Malformed candidate source receipt: {field_name}")
        path = Path(row["path"]).resolve()
        digest = str(row["sha256"]).lower()
        if path in result and result[path] != digest:
            raise ValueError(f"Conflicting source receipts for {path}")
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Candidate source receipt hash mismatch: {path}")
        result[path] = digest
    return result


def _checked_multi_anchor_sources(receipt: dict) -> tuple[dict | None, dict[Path, str]]:
    """Verify the embedded multi-anchor report and every hashed source it names."""
    report = receipt.get("multi_anchor_base_tie_source_report")
    if report is None:
        return None, {}
    if not isinstance(report, dict):
        raise ValueError("Malformed multi-anchor base-tie source report")
    if (report.get("schema_version") != "xian_r5.multi_anchor_base_tie_candidates.v1"
            or report.get("case_id") != receipt.get("case_id")
            or report.get("field_interpolated") is not False
            or report.get("field_applied") is not False
            or report.get("accepted_cells") != 0
            or report.get("dispersion_is_positional_uncertainty") is not False
            or receipt.get("multi_anchor_base_tie_field_applied") is not False):
        raise ValueError("Unsupported, accepted, interpolated or applied multi-anchor report")

    inventory: dict[Path, str] = {}

    def add_source(path_value: object, digest_value: object, label: str) -> Path:
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Malformed multi-anchor {label} path")
        if (not isinstance(digest_value, str) or len(digest_value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in digest_value)):
            raise ValueError(f"Malformed multi-anchor {label} SHA-256")
        path = Path(path_value).resolve()
        digest = digest_value.lower()
        if path in inventory and inventory[path] != digest:
            raise ValueError(f"Conflicting multi-anchor source receipts for {path}")
        inventory[path] = digest
        return path

    diagnostics = report.get("source_diagnostics")
    references = report.get("reference_rasters")
    integrity = report.get("integrity_files_sha256")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("Multi-anchor report has no source diagnostics")
    if not isinstance(references, list) or not references:
        raise ValueError("Multi-anchor report has no reference-raster inventory")
    if not isinstance(integrity, dict) or not integrity:
        raise ValueError("Multi-anchor report has no integrity inventory")

    diagnostic_paths = set()
    for row in diagnostics:
        if not isinstance(row, dict):
            raise ValueError("Malformed multi-anchor diagnostic inventory")
        diagnostic_paths.add(add_source(row.get("path"), row.get("sha256"), "diagnostic"))

    years: dict[int, tuple[Path, str, dict]] = {}
    grid_signature = None
    for row in references:
        if not isinstance(row, dict):
            raise ValueError("Malformed multi-anchor reference-raster inventory")
        try:
            year = int(row["year"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Multi-anchor reference raster has no valid year") from exc
        if year in years:
            raise ValueError(f"Duplicate multi-anchor reference year: {year}")
        path = add_source(row.get("path"), row.get("sha256"), "reference raster")
        grid = row.get("grid")
        if not isinstance(grid, dict):
            raise ValueError(f"Multi-anchor reference raster has no grid profile: {year}")
        try:
            signature = (
                str(grid["crs"]), tuple(float(value) for value in grid["transform"][:6]),
                int(grid["width"]), int(grid["height"]),
                tuple(float(value) for value in grid["resolution"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"Malformed multi-anchor reference grid profile: {year}") from exc
        if signature[0] != "EPSG:32649" or len(signature[1]) != 6:
            raise ValueError(f"Multi-anchor reference is not on the EPSG:32649 grid: {year}")
        if grid_signature is None:
            grid_signature = signature
        elif signature != grid_signature:
            raise ValueError("Multi-anchor reference rasters are not on one identical grid")
        grid_digest = grid.get("sha256")
        if isinstance(grid_digest, str) and grid_digest.casefold() != inventory[path].casefold():
            raise ValueError(f"Reference grid hash differs from raster inventory: {path}")
        years[year] = (path, inventory[path], grid)

    declared_years = report.get("reference_years")
    try:
        declared_years = [int(year) for year in declared_years]
    except (TypeError, ValueError) as exc:
        raise ValueError("Multi-anchor reference years are malformed") from exc
    if sorted(years) != declared_years or len(set(declared_years)) != len(declared_years):
        raise ValueError("Multi-anchor reference years conflict with the raster inventory")
    if 2019 not in years:
        raise ValueError("Multi-anchor report lacks its canonical Xian_2019 reference grid")
    canonical = Path(str(report.get("canonical_reference_raster") or "")).resolve()
    if (canonical != years[2019][0]
            or str(report.get("canonical_reference_sha256") or "").casefold()
            != years[2019][1].casefold()):
        raise ValueError("Multi-anchor canonical reference differs from its 2019 raster receipt")

    integrity_inventory = {}
    for path_value, digest_value in integrity.items():
        path = add_source(path_value, digest_value, "integrity entry")
        integrity_inventory[path] = inventory[path]
    if not diagnostic_paths.issubset(integrity_inventory):
        raise ValueError("Multi-anchor integrity inventory omits a source diagnostic")
    if not set(path for path, _, _ in years.values()).issubset(integrity_inventory):
        raise ValueError("Multi-anchor integrity inventory omits a reference raster")

    # The small JSON receipts are checked here. Reference rasters are already
    # hashed by the fixed-case producer; rechecking them is required for a
    # review packet because they are part of the candidate's source lineage.
    for path, expected in inventory.items():
        if not path.is_file() or _sha256(path).casefold() != expected.casefold():
            raise ValueError(f"Multi-anchor source hash mismatch: {path}")

    if not isinstance(report.get("field_cell_summaries"), list):
        raise ValueError("Multi-anchor report has no field/cell summaries")
    numeric_count = sum(
        _number(row.get("correction_dx_m_candidate")) is not None
        and _number(row.get("correction_dy_m_candidate")) is not None
        for row in report["field_cell_summaries"] if isinstance(row, dict)
    )
    if receipt.get("multi_anchor_base_tie_numeric_cell_count") != numeric_count:
        raise ValueError("Multi-anchor numeric cell count differs from its source report")
    return report, inventory


def _checked_references(
    receipt: dict,
) -> tuple[dict[Path, str], dict[Path, str], dict[Path, str]]:
    direct = _receipt_sources(receipt, "direct_base_tie_field_sources")
    relative = _receipt_sources(receipt, "lightglue_local_field_sources")
    multi_anchor, multi_sources = _checked_multi_anchor_sources(receipt)
    if (not direct and multi_anchor is None and not relative
            and receipt.get("spatial_alignment_composition") is None):
        raise ValueError("Evidence receipt has no alignment candidate diagnostic source")
    return direct, relative, multi_sources


def _checked_spatial_composition_sources(
    receipt: dict,
    product: dict,
    *,
    case_id: str,
) -> tuple[dict[tuple[str, str], tuple[Any, Any]], dict[Path, str], dict]:
    """Load hash-verified local base-grid candidates from the composition seam."""
    record = receipt.get("spatial_alignment_composition")
    if record is None:
        return {}, {}, {}
    if not isinstance(record, dict):
        raise ValueError("Malformed spatial-alignment composition receipt")
    for key in ("field_applied", "reviewed", "absolute_ground_position_resolved",
                "roof_to_ground_parallax_resolved", "formal_change_use"):
        if record.get(key) is not False:
            raise ValueError(f"Spatial-composition review source must remain pending: {key}")
    if (record.get("candidate_fields_in_product_manifest") is not True
            or record.get("per_pixel_base_component_provenance") != "NOT_RECORDED"
            or record.get("annual_reference_acquisition_date") != "UNKNOWN"):
        raise ValueError("Spatial-composition provenance uncertainty was not preserved")

    inventory = record.get("input_paths_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Spatial-composition receipt has no hashed input inventory")
    source_receipt = {"_spatial_composition_sources": list(inventory.values())}
    sources = _receipt_sources(source_receipt, "_spatial_composition_sources")

    manifest_path = Path(str(record.get("manifest_path") or "")).resolve()
    manifest_source = inventory.get("spatial_alignment_manifest")
    gpkg_source = inventory.get("spatial_alignment_geopackage")
    if not all(isinstance(value, dict) for value in (manifest_source, gpkg_source)):
        raise ValueError("Spatial-composition receipt lacks manifest or GeoPackage hashes")
    if manifest_path != Path(str(manifest_source.get("path") or "")).resolve():
        raise ValueError("Spatial-composition receipt manifest path conflicts with its hash inventory")
    manifest = _json(manifest_path)
    source_inputs = manifest.get("source_inputs")
    base_record = source_inputs.get("annual_reference_raster") if isinstance(source_inputs, dict) else None
    if not isinstance(base_record, dict):
        raise ValueError("Spatial-composition manifest lacks its annual reference-raster receipt")
    base_path = Path(str(base_record.get("path") or "")).resolve()
    base_digest = str(base_record.get("sha256_after") or "").lower()
    if (not base_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", base_digest)
            or _sha256(base_path).lower() != base_digest):
        raise ValueError("Spatial-composition annual reference raster hash is invalid")
    reference_grid = product.get("reference_grid")
    if (not isinstance(reference_grid, dict)
            or Path(str(reference_grid.get("path") or "")).resolve() != base_path):
        raise ValueError("Spatial-composition candidates do not use the fixed-case reference raster")

    fields = read_spatial_alignment_composition_fields(
        manifest_path,
        expected_base_raster=base_path,
        expected_base_sha256=base_digest,
        expected_case_id=case_id,
    )
    gpkg_path = Path(str(gpkg_source.get("path") or "")).resolve()
    declared_gpkg = (manifest_path.parent / str(
        manifest.get("products", {}).get("geopackage") or "")).resolve()
    if declared_gpkg != gpkg_path:
        raise ValueError("Spatial-composition GeoPackage path differs from its manifest")

    indexed: dict[tuple[str, str], tuple[Any, Any]] = {}
    date_counts: dict[str, int] = {}
    for field in fields:
        for cell in field.cells:
            key = (field.field_id, cell.cell_id)
            if key in indexed:
                raise ValueError(f"Duplicate spatial-composition field/cell: {key}")
            indexed[key] = (field, cell)
            date_counts[field.moving_date] = date_counts.get(field.moving_date, 0) + 1
    if base_path in sources and sources[base_path].casefold() != base_digest.casefold():
        raise ValueError("Spatial-composition base raster hash conflicts with its receipt inventory")
    sources[base_path] = base_digest
    if (int(record.get("field_count", -1)) != len(fields)
            or int(record.get("cell_count", -1)) != len(indexed)
            or record.get("cells_by_actual_date") != date_counts):
        raise ValueError("Spatial-composition receipt counts differ from its verified fields")

    return indexed, sources, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sources[manifest_path],
        "geopackage_path": str(gpkg_path),
        "geopackage_sha256": sources[gpkg_path],
        "reference_path": str(base_path),
        "reference_sha256": sources[base_path],
        "candidate_count": len(indexed),
        "fields": fields,
        "date_counts": date_counts,
    }


def _verify_multi_anchor_candidate(
    *,
    report: dict,
    reference_rasters: dict[Path, str],
    properties: dict,
    source_ref: str,
    correction_dx_m: float,
    correction_dy_m: float,
    support_geometry,
) -> tuple[Path, str, dict]:
    """Cross-check one aggregate candidate against its six-anchor provenance."""
    field_id = str(properties.get("field_id") or "")
    cell_id = str(properties.get("cell_id") or "")
    moving_date = str(properties.get("moving_date") or "")
    if (properties.get("status") != "MULTI_ANCHOR_BASE_TIE_CANDIDATE_NOT_REVIEWED"
            or properties.get("reference_date_status") != "REFERENCE_ACQUISITION_DATE_UNKNOWN"
            or properties.get("scope") != "IMAGE_TO_REFERENCE_GRID"):
        raise ValueError("Multi-anchor candidate has conflicting status or correction semantics")
    if field_id != f"{moving_date}_to_Xian_2019_MULTI_ANCHOR":
        raise ValueError("Multi-anchor field ID conflicts with its actual date or canonical grid")

    matches = [row for row in report["field_cell_summaries"]
               if isinstance(row, dict) and row.get("field_id") == field_id
               and row.get("cell_id") == cell_id and row.get("moving_date") == moving_date]
    if len(matches) != 1:
        raise ValueError(f"Multi-anchor source cell is missing or duplicated: {field_id}/{cell_id}")
    summary = matches[0]
    if summary.get("status") != properties.get("status"):
        raise ValueError(f"Multi-anchor source status differs from product: {field_id}/{cell_id}")
    source_dx = _number(summary.get("correction_dx_m_candidate"))
    source_dy = _number(summary.get("correction_dy_m_candidate"))
    if (source_dx is None or source_dy is None
            or not math.isclose(correction_dx_m, source_dx, rel_tol=0.0, abs_tol=1e-5)
            or not math.isclose(correction_dy_m, source_dy, rel_tol=0.0, abs_tol=1e-5)):
        raise ValueError(f"Multi-anchor vector differs from its source summary: {field_id}/{cell_id}")

    anchor_summary = summary.get("anchor_summary")
    evidence_rows = summary.get("reference_window_evidence")
    if not isinstance(anchor_summary, dict) or not isinstance(evidence_rows, list):
        raise ValueError(f"Multi-anchor source evidence is incomplete: {field_id}/{cell_id}")
    anchor_years = anchor_summary.get("reference_years")
    anchor_vectors = anchor_summary.get("anchor_vectors_m")
    if (not isinstance(anchor_years, list) or not isinstance(anchor_vectors, dict)
            or len(anchor_years) < 2 or anchor_summary.get("dispersion_is_positional_uncertainty") is not False):
        raise ValueError(f"Multi-anchor candidate lacks minimum anchor provenance: {field_id}/{cell_id}")
    try:
        years = [int(year) for year in anchor_years]
        anchor_count = int(anchor_summary["anchor_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Multi-anchor candidate has malformed anchor counts: {field_id}/{cell_id}") from exc
    if (years != sorted(set(years)) or anchor_count != len(years)
            or set(str(year) for year in years) != set(anchor_vectors)):
        raise ValueError(f"Multi-anchor candidate anchor list is inconsistent: {field_id}/{cell_id}")
    expected_ref = (f"multi-anchor:{report['case_id']}:{moving_date}:{cell_id}:"
                    + ",".join(str(year) for year in years))
    if source_ref != expected_ref:
        raise ValueError(f"Multi-anchor candidate source reference is inconsistent: {field_id}/{cell_id}")

    support_bounds = None
    evidence_by_year = {}
    for evidence in evidence_rows:
        if not isinstance(evidence, dict):
            raise ValueError(f"Malformed multi-anchor reference-window evidence: {field_id}/{cell_id}")
        try:
            year = int(evidence["reference_year_label"])
            path = Path(str(evidence["reference_raster_path"])).resolve()
            digest = str(evidence["reference_raster_sha256"]).lower()
            bounds = tuple(float(value) for value in evidence["cell_support_bounds_epsg32649"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Incomplete multi-anchor reference-window evidence: {field_id}/{cell_id}") from exc
        if len(bounds) != 4 or path not in reference_rasters:
            raise ValueError(f"Invalid multi-anchor reference-window key: {field_id}/{cell_id}/{year}")
        if reference_rasters[path].casefold() != digest.casefold():
            raise ValueError(f"Multi-anchor window reference hash differs: {field_id}/{cell_id}/{year}")
        if year in evidence_by_year:
            raise ValueError(f"Duplicate multi-anchor window evidence: {field_id}/{cell_id}/{year}")
        if support_bounds is None:
            support_bounds = bounds
        elif bounds != support_bounds:
            raise ValueError(f"Multi-anchor support cells differ across anchors: {field_id}/{cell_id}")
        evidence_by_year[year] = evidence
    expected_years = {int(year) for year in report["reference_years"]}
    if set(evidence_by_year) != expected_years:
        raise ValueError(f"Multi-anchor cell lacks a reference-window record: {field_id}/{cell_id}")

    canonical = Path(str(report["canonical_reference_raster"])).resolve()
    canonical_hash = str(report["canonical_reference_sha256"]).lower()
    if canonical not in reference_rasters or reference_rasters[canonical].casefold() != canonical_hash:
        raise ValueError("Multi-anchor canonical reference is not in the verified raster inventory")
    row_reference = _absolute_path(properties.get("reference_source"), base=canonical.parent)
    if row_reference != canonical:
        raise ValueError(f"Multi-anchor candidate names a different reference grid: {field_id}/{cell_id}")
    row_hash = properties.get("reference_sha256")
    if isinstance(row_hash, str) and row_hash.strip() and row_hash.casefold() != canonical_hash:
        raise ValueError(f"Multi-anchor canonical reference hash differs: {field_id}/{cell_id}")
    if support_bounds is None:
        raise ValueError(f"Multi-anchor candidate has no support geometry: {field_id}/{cell_id}")
    expected_geometry = box(*support_bounds)
    if support_geometry.hausdorff_distance(expected_geometry) > 1e-6:
        raise ValueError(f"Multi-anchor product support differs from its source cell: {field_id}/{cell_id}")
    return canonical, canonical_hash, anchor_summary


def _verify_direct_candidate_source(
    *,
    diagnostic_path: Path,
    diagnostic_cache: dict[Path, dict],
    source_ref: str,
    field_id: str,
    cell_id: str,
    moving_date: str,
    reference_path: Path,
    reference_sha256: str,
    correction_dx_m: float,
    correction_dy_m: float,
    support_geometry,
) -> None:
    """Cross-check one product-layer candidate against its hashed base-tie receipt."""
    if diagnostic_path not in diagnostic_cache:
        diagnostic_cache[diagnostic_path] = _json(diagnostic_path)
    diagnostic = diagnostic_cache[diagnostic_path]
    if (diagnostic.get("schema_version") != "xian_r5.base_tie_diagnostic.v1"
            or diagnostic.get("automatic_registration_promoted") is not False):
        raise ValueError(f"Unsupported or promoted direct base-tie diagnostic: {diagnostic_path}")

    try:
        fragment = source_ref.split("#", 1)[1]
        parts = fragment.split("/")
        if len(parts) != 3 or parts[0] != "epochs":
            raise ValueError
        epoch_label, source_cell_id = parts[1], parts[2]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed direct-base candidate source reference: {source_ref}") from exc
    if source_cell_id != cell_id:
        raise ValueError(f"Direct-base source reference cell mismatch: {source_ref}")
    epoch = diagnostic.get("epochs", {}).get(epoch_label)
    if not isinstance(epoch, dict):
        raise ValueError(f"Missing direct-base epoch in candidate source: {source_ref}")
    epoch_base = _absolute_path(epoch.get("base_path"), base=diagnostic_path.parent)
    if epoch_base != reference_path:
        raise ValueError(f"Candidate reference raster differs from base-tie diagnostic: {source_ref}")
    if str(epoch.get("base_sha256") or "").casefold() != reference_sha256.casefold():
        raise ValueError(f"Candidate reference hash differs from base-tie diagnostic: {source_ref}")
    if str(epoch.get("actual_date") or "") != moving_date:
        raise ValueError(f"Candidate actual date differs from base-tie diagnostic: {source_ref}")
    if field_id != f"{moving_date}_to_{reference_path.stem}_DIRECT":
        raise ValueError(f"Candidate field ID conflicts with date/reference: {source_ref}")

    local = epoch.get("local_spatial_tie_diagnostic")
    if (not isinstance(local, dict) or local.get("field_interpolated") is not False
            or local.get("field_applied") is not False):
        raise ValueError(f"Direct-base source field is missing or applied: {source_ref}")
    matches = []
    for candidate in local.get("cells", []):
        try:
            candidate_id = f"r{int(candidate['row'])}c{int(candidate['col'])}"
        except (KeyError, TypeError, ValueError):
            continue
        if candidate_id == cell_id:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"Direct-base source cell is missing or duplicated: {source_ref}")
    tie_cell = matches[0]
    best = tie_cell.get("best")
    peak = best.get("subpixel_peak") if isinstance(best, dict) else None
    if (not isinstance(peak, dict)
            or peak.get("status") != "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED"):
        raise ValueError(f"Numeric product candidate has no matching subpixel source peak: {source_ref}")
    try:
        dx_px, dy_px = float(peak["dx_px"]), float(peak["dy_px"])
        transform = [float(value) for value in epoch["base_grid"]["transform"]]
        x0, y0, x1, y1 = [int(value) for value in tie_cell["base_pixel_window"]]
        base_window = [int(value) for value in epoch["base_grid"]["window"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Incomplete direct-base source evidence: {source_ref}") from exc
    if len(transform) < 6 or len(base_window) != 4:
        raise ValueError(f"Invalid direct-base source affine/window: {source_ref}")
    a, b, _, d, e, _ = transform[:6]
    expected_dx, expected_dy = (-a * dx_px - b * dy_px, -d * dx_px - e * dy_px)
    if (not math.isclose(correction_dx_m, expected_dx, rel_tol=0.0, abs_tol=1e-5)
            or not math.isclose(correction_dy_m, expected_dy, rel_tol=0.0, abs_tol=1e-5)):
        raise ValueError(f"Product candidate vector differs from source subpixel peak: {source_ref}")
    if not (0 <= x0 < x1 <= base_window[2] and 0 <= y0 < y1 <= base_window[3]):
        raise ValueError(f"Direct-base source cell lies outside its reference window: {source_ref}")
    left, right = transform[2] + a * x0, transform[2] + a * x1
    top, bottom = transform[5] + e * y0, transform[5] + e * y1
    source_support = box(left, min(top, bottom), right, max(top, bottom))
    if support_geometry.hausdorff_distance(source_support) > 1e-6:
        raise ValueError(f"Product support geometry differs from source tie cell: {source_ref}")


def _verify_relative_candidate_source(
    *,
    diagnostic_path: Path,
    diagnostic_cache: dict[Path, dict],
    relative_receipts: dict[Path, str],
    source_ref: str,
    case_id: str,
    field_id: str,
    cell_id: str,
    reference_date: str,
    moving_date: str,
    candidate_status: str,
    correction_dx_m: float,
    correction_dy_m: float,
    reference_source: str,
    support_geometry,
) -> dict:
    """Verify one relative LightGlue cell and return its measured QA fields."""
    diagnostic_path = Path(diagnostic_path).resolve()
    if diagnostic_path not in relative_receipts:
        raise ValueError(f"Relative candidate source is absent from the evidence receipt: {diagnostic_path}")
    if diagnostic_path not in diagnostic_cache:
        diagnostic_cache[diagnostic_path] = _json(diagnostic_path)
    diagnostic = diagnostic_cache[diagnostic_path]
    if (diagnostic.get("schema_version") != "xian_r5.local_tile_lightglue_field_diagnostic.v1"
            or diagnostic.get("case_id") != case_id
            or diagnostic.get("scope") != "FIXED_CASE_LOCAL_ONLY"
            or diagnostic.get("registration_promoted") is not False
            or diagnostic.get("registration_applied") is not False
            or diagnostic.get("field_interpolated") is not False
            or diagnostic.get("mask", {}).get("ground_mask_reviewed") is not False):
        raise ValueError(f"Unsupported, promoted, or reviewed relative diagnostic: {diagnostic_path}")

    expected_fragment = f"pair:{reference_date}/{moving_date}/cell:{cell_id}"
    try:
        source_path_text, fragment = source_ref.split("#", 1)
    except ValueError as exc:
        raise ValueError(f"Malformed relative candidate source reference: {source_ref}") from exc
    if (Path(source_path_text).resolve() != diagnostic_path
            or fragment != expected_fragment
            or field_id != f"ALIKED_{reference_date}_to_{moving_date}"):
        raise ValueError(f"Relative candidate source reference conflicts with its field/cell: {source_ref}")

    actual_dates = diagnostic.get("actual_dates_in_order")
    if (not isinstance(actual_dates, list) or reference_date not in actual_dates
            or moving_date not in actual_dates or actual_dates != sorted(set(actual_dates))):
        raise ValueError("Relative diagnostic dates are missing, duplicated, or unordered")
    expected_mosaic = Path(str(diagnostic.get("input_mosaic_diagnostic") or "")).resolve()
    if (not expected_mosaic.is_file()
            or expected_mosaic not in relative_receipts
            or Path(reference_source).resolve() != expected_mosaic):
        raise ValueError("Relative candidate does not identify its hashed mosaic reference")
    candidate_vector = Path(str(diagnostic.get("outputs", {}).get("candidate_vector") or "")).resolve()
    if candidate_vector not in relative_receipts or not candidate_vector.is_file():
        raise ValueError("Relative diagnostic candidate vector is missing from its source receipt")

    pairs = [row for row in diagnostic.get("pairwise_fields", [])
             if isinstance(row, dict)
             and row.get("from_date") == reference_date
             and row.get("to_date") == moving_date]
    if len(pairs) != 1:
        raise ValueError(f"Relative diagnostic date pair is missing or duplicated: {reference_date}/{moving_date}")
    pair = pairs[0]
    if pair.get("field_interpolated") is not False or pair.get("field_applied") is not False:
        raise ValueError("Relative diagnostic date pair was interpolated or applied")
    cells = [row for row in pair.get("cells", [])
             if isinstance(row, dict) and row.get("cell_id") == cell_id]
    if len(cells) != 1:
        raise ValueError(f"Relative diagnostic cell is missing or duplicated: {field_id}/{cell_id}")
    cell = cells[0]
    if (cell.get("status") != candidate_status
            or cell.get("ground_mask_reviewed") is not False
            or cell.get("field_interpolated") is not False
            or cell.get("applied_to_image_or_geometry") is not False
            or str(cell.get("status", "")).startswith("ACCEPTED")):
        raise ValueError(f"Relative diagnostic cell has conflicting review/application state: {field_id}/{cell_id}")
    source_dx = _number(cell.get("correction_dx_m_candidate"))
    source_dy = _number(cell.get("correction_dy_m_candidate"))
    if (source_dx is None or source_dy is None
            or not math.isclose(correction_dx_m, source_dx, rel_tol=0.0, abs_tol=1e-5)
            or not math.isclose(correction_dy_m, source_dy, rel_tol=0.0, abs_tol=1e-5)):
        raise ValueError(f"Relative candidate vector differs from its source cell: {field_id}/{cell_id}")
    source_geometry = cell.get("polygon_epsg32649")
    if not isinstance(source_geometry, dict):
        raise ValueError(f"Relative diagnostic cell has no support geometry: {field_id}/{cell_id}")
    source_support = shape(source_geometry)
    if (not source_support.is_valid or source_support.is_empty
            or source_support.hausdorff_distance(support_geometry) > 1e-6):
        raise ValueError(f"Relative product support differs from its source cell: {field_id}/{cell_id}")
    return cell


def _review_context_sources(
    direct_sources: dict[Path, str],
    relative_sources: dict[Path, str],
) -> list[dict[str, str]]:
    """Collect hashed, date-linked PNG context from validated diagnostics."""
    records: list[dict[str, str]] = []
    seen: set[Path] = set()

    def add(path_value: object, *, diagnostic: Path, kind: str, label: str,
            expected_sha256: object = None, dates: str = "") -> None:
        if not path_value:
            return
        path = _absolute_path(path_value, base=diagnostic.parent)
        if not path.is_file():
            raise FileNotFoundError(f"Review context image is missing: {path}")
        digest = _sha256(path)
        if expected_sha256 is not None:
            if (not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                    or digest.casefold() != expected_sha256.casefold()):
                raise ValueError(f"Review context image hash mismatch: {path}")
        if path in seen:
            return
        with path.open("rb") as source:
            if source.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Review context image is not a PNG: {path}")
        seen.add(path)
        records.append({
            "kind": kind,
            "label": label,
            "dates": dates,
            "source_path": str(path),
            "source_sha256": digest,
            "source_diagnostic": str(diagnostic.resolve()),
            "semantics": (
                "DATE_LEVEL_BASE_GRID_VISUAL_CONTEXT_NOT_REVIEWED_TIE_EVIDENCE"
                if kind == "BASE_GRID_CONTEXT"
                else "RELATIVE_DATE_PAIR_MATCH_CONTEXT_NOT_A_BASE_ANCHOR"
            ),
        })

    for diagnostic in sorted(direct_sources):
        report = _json(diagnostic)
        if report.get("schema_version") != "xian_r5.base_tie_diagnostic.v1":
            continue
        epochs = report.get("epochs")
        if not isinstance(epochs, dict):
            continue
        for epoch_key, epoch in sorted(epochs.items()):
            if not isinstance(epoch, dict):
                continue
            actual_date = str(epoch.get("actual_date") or epoch_key)
            add(epoch.get("review_png"), diagnostic=diagnostic,
                kind="BASE_GRID_CONTEXT", label=actual_date,
                dates=actual_date)

    for diagnostic in sorted(relative_sources):
        if diagnostic.suffix.casefold() != ".json":
            continue
        report = _json(diagnostic)
        if report.get("schema_version") != "xian_r5.local_tile_lightglue_field_diagnostic.v1":
            continue
        outputs = report.get("outputs")
        sheets = outputs.get("match_review_contact_sheets") if isinstance(outputs, dict) else None
        if not isinstance(sheets, list):
            continue
        for sheet in sheets:
            if not isinstance(sheet, dict):
                raise ValueError(f"Malformed LightGlue review contact sheet: {diagnostic}")
            from_date = str(sheet.get("from_date") or "UNKNOWN")
            to_date = str(sheet.get("to_date") or "UNKNOWN")
            add(sheet.get("path"), diagnostic=diagnostic,
                kind="RELATIVE_MATCH_CONTEXT",
                label=f"{from_date}_to_{to_date}",
                expected_sha256=sheet.get("sha256"),
                dates=f"{from_date};{to_date}")
    return records


def _copy_review_context_images(
    output_dir: Path,
    sources: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not sources:
        return []
    context_dir = output_dir / "review_context"
    context_dir.mkdir()
    copied = []
    used_names: dict[str, str] = {}
    for source in sources:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_",
                      f"{source['kind'].lower()}_{source['label']}").strip("._-")
        filename = f"{stem or 'review_context'}.png"
        source_hash = source["source_sha256"]
        if filename in used_names and used_names[filename] != source_hash:
            filename = f"{stem}_{source_hash[:8]}.png"
        used_names[filename] = source_hash
        target = context_dir / filename
        shutil.copyfile(source["source_path"], target)
        copied_hash = _sha256(target)
        if copied_hash.casefold() != source_hash.casefold():
            raise RuntimeError(f"Copied review context hash mismatch: {target}")
        copied.append({
            **source,
            "path": target.relative_to(output_dir).as_posix(),
            "sha256": copied_hash,
        })
    return copied


def _write_candidate_cell_context_images(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    copied_contexts: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, str]]:
    """Render candidate-local crops from hash-checked base-grid review images.

    These crops are navigation aids only. They show the 10 m target and the
    moving image averaged onto that grid; they do not encode reviewed ties.
    """
    from PIL import Image, ImageDraw

    base_contexts = [row for row in copied_contexts
                     if row["kind"] == "BASE_GRID_CONTEXT"]
    if not candidates or not base_contexts:
        return [], [{
            "field_id": str(row["properties"].get("field_id") or ""),
            "cell_id": str(row["properties"].get("cell_id") or ""),
            "status": "NO_DATE_MATCHED_BASE_GRID_CONTEXT",
        } for row in candidates], {}

    report_cache: dict[Path, dict[str, Any]] = {}
    output_dir_cells = output_dir / "review_context" / "cells"
    output_dir_cells.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    source_tile_hashes: dict[str, str] = {}
    verified_tile_hashes: dict[Path, str] = {}

    def normalized(path_value: object) -> str:
        return str(Path(str(path_value)).resolve()).replace("\\", "/").casefold()

    def image_context_for(row: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]] | None:
        props = row["properties"]
        actual_date = str(props.get("moving_date") or "")
        reference_source = normalized(props.get("reference_source") or "")
        reference_sha256 = str(props.get("reference_source_sha256") or "").casefold()
        source_ref = str(props.get("candidate_source_ref") or "")
        candidate_diagnostic = (
            normalized(source_ref.split("#", 1)[0])
            if props.get("method") == DIRECT_METHOD and "#" in source_ref else None)
        matches = []
        for context in base_contexts:
            if context["label"] != actual_date:
                continue
            diagnostic_path = Path(context["source_diagnostic"]).resolve()
            report = report_cache.get(diagnostic_path)
            if report is None:
                report = _json(diagnostic_path)
                report_cache[diagnostic_path] = report
            for epoch in report.get("epochs", {}).values():
                if not isinstance(epoch, dict) or str(epoch.get("actual_date") or "") != actual_date:
                    continue
                base_path = epoch.get("base_path")
                if not base_path or normalized(base_path) != reference_source:
                    continue
                base_hash = str(epoch.get("base_sha256") or "").casefold()
                if reference_sha256 and base_hash != reference_sha256:
                    continue
                if (candidate_diagnostic is not None
                        and normalized(context["source_diagnostic"]) != candidate_diagnostic):
                    continue
                matches.append((context, epoch))
        if len(matches) != 1:
            return None
        return matches[0]

    def source_components_for(
        epoch: dict[str, Any], geometry: Any, actual_date: str,
    ) -> tuple[list[dict[str, Any]], str, float]:
        tile_inventory = epoch.get("submeter_tiles")
        entries = tile_inventory.get("tile_ids") if isinstance(tile_inventory, dict) else None
        if not isinstance(entries, list) or not entries:
            return [], "SOURCE_COMPONENT_PROVENANCE_NOT_AVAILABLE", 0.0
        items = []
        component_geometries = []
        incomplete = False
        seen_ids: set[str] = set()
        threshold = max(1e-8, geometry.area * 1e-12)
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("actual_date") != actual_date:
                incomplete = True
                continue
            tile_id = str(entry.get("tile_id") or "")
            tile_path_value = entry.get("path")
            tile_digest = entry.get("sha256")
            polygon_value = entry.get("support_polygon_epsg32649")
            if (not re.fullmatch(r"\d+:\d+", tile_id)
                    or not isinstance(tile_path_value, str) or not tile_path_value.strip()
                    or not isinstance(tile_digest, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", tile_digest)
                    or not isinstance(polygon_value, dict)):
                incomplete = True
                continue
            tile_geometry = shape(polygon_value)
            if (tile_geometry.is_empty or not tile_geometry.is_valid
                    or tile_geometry.geom_type not in {"Polygon", "MultiPolygon"}):
                incomplete = True
                continue
            component_geometries.append(tile_geometry)
            overlap_area = float(geometry.intersection(tile_geometry).area)
            if overlap_area <= threshold:
                continue
            if tile_id in seen_ids:
                raise ValueError(f"Duplicate source tile in review context: {actual_date}/{tile_id}")
            seen_ids.add(tile_id)
            tile_path = Path(tile_path_value).resolve()
            expected_hash = tile_digest.casefold()
            if not tile_path.is_file():
                raise FileNotFoundError(f"Review context source tile is missing: {tile_path}")
            actual_hash = verified_tile_hashes.get(tile_path)
            if actual_hash is None:
                actual_hash = _sha256(tile_path).casefold()
                verified_tile_hashes[tile_path] = actual_hash
            if actual_hash != expected_hash:
                raise ValueError(f"Review context source tile hash mismatch: {tile_path}")
            path_key = str(tile_path)
            prior_hash = source_tile_hashes.get(path_key)
            if prior_hash is not None and prior_hash.casefold() != actual_hash:
                raise ValueError(f"Conflicting source tile hashes in review context: {tile_path}")
            source_tile_hashes[path_key] = actual_hash
            items.append({
                "actual_date": actual_date,
                "tile_id": tile_id,
                "path": path_key,
                "sha256": actual_hash,
                "overlap_area_m2": overlap_area,
            })
        if not items:
            return [], "NO_SOURCE_TILE_OVERLAPS_CANDIDATE", 0.0
        covered = unary_union(component_geometries) if component_geometries else None
        coverage = min(1.0, max(0.0,
            float(geometry.intersection(covered).area / geometry.area)
            if covered is not None else 0.0))
        if incomplete:
            scope = "SOURCE_COMPONENT_PROVENANCE_INCOMPLETE"
        elif coverage < 1.0 - 1e-8:
            scope = "INCOMPLETE_SOURCE_COMPONENT_COVERAGE"
        elif len(items) == 1:
            scope = "SINGLE_SOURCE_COMPONENT_PER_CANDIDATE_CELL"
        else:
            scope = "MULTIPLE_SOURCE_COMPONENTS_PER_CANDIDATE_CELL"
        return sorted(items, key=lambda item: item["tile_id"]), scope, coverage

    for index, row in enumerate(candidates, start=1):
        props = row["properties"]
        field_id = str(props.get("field_id") or "")
        cell_id = str(props.get("cell_id") or "")
        match = image_context_for(row)
        if match is None:
            skipped.append({
                "field_id": field_id,
                "cell_id": cell_id,
                "status": "NO_UNIQUE_DATE_AND_REFERENCE_MATCHED_CONTEXT",
            })
            continue
        context, epoch = match
        actual_date = str(props.get("moving_date") or "")
        source_component_tiles, source_component_scope, source_component_coverage = (
            source_components_for(epoch, row["geometry"], actual_date))
        grid = epoch.get("base_grid")
        window = grid.get("window") if isinstance(grid, dict) else None
        transform = grid.get("transform") if isinstance(grid, dict) else None
        if (not isinstance(window, list) or len(window) != 4
                or not isinstance(transform, list) or len(transform) not in (6, 9)):
            raise ValueError(f"Base-grid review context has no local affine: {context['source_diagnostic']}")
        grid_width, grid_height = int(window[2]), int(window[3])
        a, b, c, d, e, f = (float(value) for value in transform[:6])
        if (grid_width < 1 or grid_height < 1 or a <= 0 or e >= 0
                or abs(b) > 1e-10 or abs(d) > 1e-10):
            raise ValueError(f"Unsupported base-grid review context affine: {context['source_diagnostic']}")
        with Image.open(output_dir / context["path"]) as source_image:
            source_image = source_image.convert("RGB")
            panel_width = source_image.width // 3
            panel_height = source_image.height - 32
            if (source_image.width % 3 or panel_height <= 0
                    or panel_width % grid_width or panel_height % grid_height
                    or panel_width // grid_width != panel_height // grid_height):
                raise ValueError(f"Unexpected three-panel base-grid review image: {context['path']}")
            scale = panel_width // grid_width
            scale_up = 3
            geometry = row["geometry"]
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError(f"Invalid candidate review support: {field_id}/{cell_id}")

            def pixel_xy(x: float, y: float) -> tuple[float, float]:
                return ( (x - c) / a, (y - f) / e )

            projected_rings = []
            parts = ([geometry] if geometry.geom_type == "Polygon"
                     else list(geometry.geoms))
            for polygon in parts:
                if polygon.geom_type != "Polygon":
                    continue
                projected_rings.append([pixel_xy(x, y) for x, y in polygon.exterior.coords])
            all_points = [point for ring in projected_rings for point in ring]
            if not all_points:
                raise ValueError(f"Candidate review support has no polygon rings: {field_id}/{cell_id}")
            margin = 2
            col0 = max(0, math.floor(min(point[0] for point in all_points)) - margin)
            row0 = max(0, math.floor(min(point[1] for point in all_points)) - margin)
            col1 = min(grid_width, math.ceil(max(point[0] for point in all_points)) + margin)
            row1 = min(grid_height, math.ceil(max(point[1] for point in all_points)) + margin)
            if col1 <= col0 or row1 <= row0:
                skipped.append({"field_id": field_id, "cell_id": cell_id,
                                "status": "CANDIDATE_OUTSIDE_REVIEW_IMAGE_WINDOW"})
                continue

            crop_panels = []
            for panel_index, panel_label in enumerate(("10 m base", "moving → 10 m", "50:50 overlay")):
                panel = source_image.crop((panel_index * panel_width, 32,
                                           (panel_index + 1) * panel_width,
                                           32 + panel_height))
                crop = panel.crop((col0 * scale, row0 * scale,
                                   col1 * scale, row1 * scale))
                crop = crop.resize((crop.width * scale_up, crop.height * scale_up),
                                   Image.Resampling.NEAREST)
                draw = ImageDraw.Draw(crop)
                grid_step = scale * scale_up
                for x in range(0, crop.width + 1, grid_step):
                    draw.line((x, 0, x, crop.height), fill=(115, 115, 115), width=1)
                for y in range(0, crop.height + 1, grid_step):
                    draw.line((0, y, crop.width, y), fill=(115, 115, 115), width=1)
                for ring in projected_rings:
                    points = [((x - col0) * scale * scale_up,
                               (y - row0) * scale * scale_up) for x, y in ring]
                    draw.line(points, fill=(255, 230, 0), width=3, joint="curve")
                crop_panels.append((panel_label, crop))

        gap, panel_header, title_height, footer_height = 8, 20, 46, 32
        panel_pixel_width = crop_panels[0][1].width
        panel_pixel_height = crop_panels[0][1].height
        canvas = Image.new(
            "RGB",
            (panel_pixel_width * len(crop_panels) + gap * (len(crop_panels) - 1),
             title_height + panel_header + panel_pixel_height + footer_height),
            (248, 248, 248),
        )
        draw = ImageDraw.Draw(canvas)
        dx = _number(props.get("correction_dx_m"))
        dy = _number(props.get("correction_dy_m"))
        vector = "unknown" if dx is None or dy is None else f"({dx:.2f}, {dy:.2f}) m"
        title = (f"{props.get('moving_date')} | {cell_id} | candidate shift {vector} | "
                 f"{props.get('candidate_status') or 'UNREVIEWED'}")
        draw.text((4, 4), title[:180], fill=(0, 0, 0))
        for panel_index, (panel_label, panel_image) in enumerate(crop_panels):
            left = panel_index * (panel_pixel_width + gap)
            draw.text((left + 4, title_height + 3), panel_label, fill=(0, 0, 0))
            canvas.paste(panel_image, (left, title_height + panel_header))
        minx, miny, maxx, maxy = geometry.bounds
        footer = (f"EPSG:32649 | support bounds {minx:.1f},{miny:.1f} to "
                  f"{maxx:.1f},{maxy:.1f} | grid cell = 10 m")
        draw.text((4, canvas.height - footer_height + 4), footer, fill=(0, 0, 0))
        filename_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", cell_id).strip("._-") or "cell"
        filename = f"{index:03d}_{str(props.get('moving_date') or 'date')}_{filename_stem}.png"
        target = output_dir_cells / filename
        canvas.save(target, format="PNG")
        rendered.append({
            "field_id": field_id,
            "cell_id": cell_id,
            "moving_date": str(props.get("moving_date") or ""),
            "method": str(props.get("method") or ""),
            "candidate_status": str(props.get("candidate_status") or ""),
            "candidate_dx_m": dx,
            "candidate_dy_m": dy,
            "moving_source_component_scope": source_component_scope,
            "moving_source_component_coverage_fraction": source_component_coverage,
            "moving_source_component_tiles": source_component_tiles,
            "reference_source": str(props.get("reference_source") or ""),
            "reference_source_sha256": str(props.get("reference_source_sha256") or ""),
            "source_diagnostic": context["source_diagnostic"],
            "source_context_path": context["path"],
            "source_context_sha256": context["sha256"],
            "path": target.relative_to(output_dir).as_posix(),
            "sha256": _sha256(target),
            "support_bounds_epsg32649": [float(value) for value in geometry.bounds],
            "crop_pixel_window_within_context": [col0, row0, col1, row1],
            "semantics": "CANDIDATE_CELL_VISUAL_CONTEXT_ONLY_NOT_TIE_EVIDENCE_OR_REVIEW_VERDICT",
        })
    return rendered, skipped, source_tile_hashes


_TIE_PAIR_TEMPLATE_FIELDS = (
    "field_id", "cell_id", "moving_date", "reference_source",
    "reference_source_sha256", "pair_index", "tie_id", "FIT_or_HOLDOUT",
    "feature_type", "precision_eligible",
    "ge_x", "ge_y", "reference_x", "reference_y", "candidate_dx_m",
    "candidate_dy_m", "residual_dx_m", "residual_dy_m", "residual_m",
    "ge_localization_bound_m", "reference_localization_bound_m",
    "adjusted_bound_m", "localization_basis", "evidence_path",
    "evidence_sha256", "within_exact_support", "independence_group",
    "feature_class", "stable_nonbuilding_ground_evidence", "stability_basis",
    "precision_ineligibility_reason",
)


def _write_tie_pair_review_template(path: Path, candidates: list[dict[str, Any]]) -> int:
    """Write six empty, role-separated tie slots for every absolute candidate."""
    count = 0
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=_TIE_PAIR_TEMPLATE_FIELDS)
        writer.writeheader()
        for row in candidates:
            props = row["properties"]
            for role in ("FIT", "HOLDOUT"):
                for pair_index in range(1, 4):
                    writer.writerow({
                        "field_id": props["field_id"],
                        "cell_id": props["cell_id"],
                        "moving_date": props["moving_date"],
                        "reference_source": props["reference_source"],
                        "reference_source_sha256": props["reference_source_sha256"],
                        "candidate_dx_m": props["correction_dx_m"],
                        "candidate_dy_m": props["correction_dy_m"],
                        "pair_index": pair_index,
                        "FIT_or_HOLDOUT": role,
                    })
                    count += 1
    return count


def _apply_review_scope_allowlist(
    direct_rows: list[dict[str, Any]],
    relative_rows: list[dict[str, Any]],
    scope_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Apply the one frozen nine-cell review scope without candidate fallback."""
    if scope_id is None:
        return direct_rows, relative_rows, None
    if scope_id != R5_RELATIVE_REVIEW_SCOPE_ID:
        raise ValueError(f"Unknown alignment review scope allowlist: {scope_id}")
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in direct_rows:
        props = row["properties"]
        key = (str(props.get("cell_id") or ""), str(props.get("moving_date") or ""))
        by_key.setdefault(key, []).append(row)

    selected = []
    for key in R5_RELATIVE_REVIEW_SCOPE:
        matches = by_key.get(key, [])
        if len(matches) != 1:
            raise ValueError(
                "Frozen review-scope allowlist requires exactly one candidate per "
                f"date-cell; {key} matched {len(matches)}")
        selected.append(matches[0])
    if len(selected) != 9:
        raise ValueError(f"Frozen review-scope allowlist resolved {len(selected)} candidates, expected 9")
    selected_keys = {
        (str(row["properties"].get("cell_id") or ""),
         str(row["properties"].get("moving_date") or ""))
        for row in selected
    }
    if len(selected_keys) != 9 or selected_keys != set(R5_RELATIVE_REVIEW_SCOPE):
        raise ValueError("Frozen review-scope allowlist did not resolve the exact nine date-cell keys")
    return selected, [], {
        "allowlist_id": R5_RELATIVE_REVIEW_SCOPE_ID,
        "expected_date_cell_count": 9,
        "requested_date_cells": [
            {"cell_id": cell_id, "moving_date": moving_date}
            for cell_id, moving_date in R5_RELATIVE_REVIEW_SCOPE
        ],
        "matched_date_cell_count": len(selected),
        "exact_set_match": True,
        "matched_date_cells": [
            {"cell_id": row["properties"]["cell_id"],
             "moving_date": row["properties"]["moving_date"]}
            for row in selected
        ],
        "relative_date_pair_diagnostics": "OMITTED_BY_REVIEW_SCOPE",
        "candidate_fallback": False,
    }


def _scope_review_context_sources(
    contexts: list[dict[str, str]],
    scope_id: str | None,
) -> list[dict[str, str]]:
    """Keep only date-level base context in the frozen date-cell scope."""
    if scope_id is None:
        return contexts
    if scope_id != R5_RELATIVE_REVIEW_SCOPE_ID:
        raise ValueError(f"Unknown alignment review scope allowlist: {scope_id}")
    allowed_dates = {moving_date for _, moving_date in R5_RELATIVE_REVIEW_SCOPE}
    return [
        record for record in contexts
        if record.get("kind") == "BASE_GRID_CONTEXT"
        and record.get("dates") in allowed_dates
    ]


def _readme(
    case_id: str,
    direct_count: int,
    relative_count: int | None,
    multi_anchor_count: int = 0,
    spatial_composition_count: int = 0,
    spatial_excluded_status_counts: dict[str, int] | None = None,
    review_context_count: int = 0,
    tie_pair_template_count: int = 0,
    cell_context_count: int = 0,
    source_component_image_count: int = 0,
    review_scope_allowlist_id: str | None = None,
) -> str:
    if review_scope_allowlist_id is not None:
        relative_description = (
            "- `" + RELATIVE_LAYER + "` was omitted by frozen review scope "
            f"`{review_scope_allowlist_id}`; this does not mean the source diagnostic had zero candidates."
        )
    elif relative_count is None:
        relative_description = (
            f"- `{RELATIVE_LAYER}` in `relative_date_pair_candidates.gpkg`: no relative-date "
            "diagnostic was supplied. The empty layer is a schema placeholder; this is not "
            "evidence that the candidate count is zero."
        )
    else:
        relative_description = (
            f"- `{RELATIVE_LAYER}` in `relative_date_pair_candidates.gpkg`: {relative_count} "
            "numeric ALIKED/LightGlue date-pair candidates. These are relative image-to-image "
            "vectors and cannot independently anchor an observation to the 10 m reference grid. "
            "The layer also preserves match counts, inlier support, coverage, residuals, and "
            "the unreviewed-ground-mask flag from the hashed source diagnostic."
        )
    if multi_anchor_count:
        multi_anchor_description = (
            f"- The absolute layer also contains {multi_anchor_count} numeric multi-anchor "
            "consensus cells. Their annual references share one grid, but the annual image "
            "acquisition dates and per-pixel source components are unknown. Cross-anchor "
            "dispersion is not positional uncertainty or ground control."
        )
    else:
        multi_anchor_description = ""
    spatial_excluded_status_counts = spatial_excluded_status_counts or {}
    spatial_excluded_description = (
        "; ".join(f"{status}: {count}" for status, count
                   in sorted(spatial_excluded_status_counts.items()))
        or "none"
    )
    if spatial_composition_count:
        spatial_description = (
            f"- The absolute review layer also contains {spatial_composition_count} "
            "cycle-screen-only spatial-composition cells. They are local image-to-base "
            "grid candidates composed from direct base ties and actual-date relative ties; "
            "cycle closure is an internal consistency screen, not positional accuracy. "
            "The 10 m reference acquisition date and per-pixel mosaic component are unknown. "
            f"Spatial candidates excluded from review because of status: {spatial_excluded_description}."
        )
    elif spatial_excluded_status_counts:
        spatial_description = (
            "- Spatial-composition diagnostics were supplied but no cycle-screen-only "
            "cell was eligible for the review layer. Excluded statuses: "
            f"{spatial_excluded_description}."
        )
    else:
        spatial_description = ""
    return f"""# Alignment candidate review packet — {case_id}

This packet contains **unreviewed image-to-grid candidates**, not corrected ground footprints.
It does not prove roof-to-ground parallax, building footprint accuracy, or change accuracy.

## Layers and counts

- `{REVIEW_LAYER}` in `absolute_base_review_template.gpkg`: {direct_count} numeric image-to-grid candidate cells, including direct, multi-anchor, and eligible spatial-composition cycle-screen candidates where present. Every row is pending, unreviewed, unaccepted, and unapplied.
{multi_anchor_description}
{spatial_description}
{relative_description}

## Human review requirements

The 10 m raster is the user's selected reference grid. A direct image-to-base candidate can be reviewed under either of two explicitly different bases:

1. For `review_basis=RELATIVE_REFERENCE_IMAGE_FEATURES`, inspect stable **non-building** ground features in the moving image and the named base raster. Use `R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1` and the current `xian_r5.relative_alignment_tie_sidecar.v2` schema. For every tie, compute `residual_vector=(GE_point+candidate_vector)-reference_point`, `residual_m=hypot(residual_dx_m,residual_dy_m)`, and `adjusted_bound_m=residual_m+ge_localization_bound_m+reference_localization_bound_m`. The evidence artifacts must support both finite, nonnegative localization bounds, and `localization_basis` must explain the locating feature and distance range rather than provide a confidence label. Use only `fit_max_adjusted_bound_m=max(FIT adjusted bounds)`, `holdout_max_adjusted_bound_m=max(HOLDOUT adjusted bounds)`, and set `relative_registration_uncertainty_m=holdout_max_adjusted_bound_m`; both maxima must be <=2 m and there must be at least 3 independent precision-eligible ties per role, all stable non-building features inside exact support. Do not use RMSE, averages, percentiles, standard deviation, or normal confidence intervals. For this relative review only, set `positional_uncertainty_m` equal to `relative_registration_uncertainty_m` and persist both `positional_uncertainty_semantics` and `uncertainty_semantics` as `RELATIVE_IMAGE_GRID_ONLY`; `absolute_position_accuracy` stays `UNKNOWN`.

2. For `review_basis=INDEPENDENT_GROUND_CONTROL`, set `independent_ground_evidence=1`, `alignment_evidence_type=INDEPENDENT_GROUND_CONTROL`, `ground_evidence_type` to one of `SURVEY_CONTROL`, `INDEPENDENT_ORTHO`, `AUTHORITATIVE_MAP`, `OTHER_INDEPENDENT_GROUND_CONTROL`, and `positional_uncertainty_m` in [0, 2]. Leave relative-only fields empty.

Both paths require `reviewed=1`, `reviewer_id`, timezone-bearing ISO `reviewed_at`, `evidence_path`, and the matching `evidence_sha256`. Do not use building roofs or the 2019 roof-outline layer as stable ground ties.

Do not change `field_id`, `cell_id`, `moving_date`, `reference_source`, its SHA-256, `candidate_source_ref`, the correction vector, or support geometry. The existing review adapter checks these against the source candidate and reference raster. Submit only completed, accepted rows in a separate GeoPackage layer named `{REVIEW_LAYER}`; do not include untouched pending rows in that accepted-only layer.

Date-pair relative candidates remain unaccepted and unapplied in this packet. They still require composition with a reviewed tie to the selected 10 m base grid. For a spatial-composition cell with unknown source-component scope, only `RELATIVE_REFERENCE_IMAGE_FEATURES` review is eligible: inspect the current-date image directly against stable non-building features in the named base raster, and validate at least 3 fit pairs plus 3 held-out pairs wholly inside that exact support cell. Do not treat cycle closure or a relative date-pair tie as independent ground evidence. This narrowly authorizes a cell-local shift to the selected raster grid; source-component provenance, absolute ground accuracy, and roof-to-ground parallax remain unknown.

The external roof-outline candidate data is not a ground control source. The 10 m raster defines the target grid, but by itself does not support an absolute ground-position accuracy claim.

Even a valid accepted tie authorizes only image-to-reference-grid alignment. It does **not** establish that a shifted roof outline is the building's ground footprint. Preserve the raw roof geometry and keep the ground-footprint status unresolved unless separate roof-to-ground evidence supports a footprint correction.

## Provenance

`REVIEW_PACKET_MANIFEST.json` records source-product hashes, candidate-source hashes, reference-raster hashes, counts, and the no-application boundary. The fixed-case source run is read-only; this packet was written to a new output directory.

## Review materials

`review_context/` contains {review_context_count} hash-checked PNG context image(s), when available. Base-grid context is date-level visual context; relative match contact sheets show date-pair matches only and cannot anchor a date to the 10 m raster. None of these images is a human review verdict.

`review_context/cells/` contains {cell_context_count} candidate-local crops with the support outline and EPSG:32649 bounds. These are crops of the date-level three-panel preview, where the moving image has already been averaged onto the 10 m grid; they are navigation aids, not raw high-resolution tie evidence. The manifest lists and hash-checks {source_component_image_count} unique intersecting moving-image source tile(s) where the diagnostic recorded them. The 10 m reference raster's per-pixel source-component provenance remains unknown.

`tie_pairs_review_template.csv` contains {tie_pair_template_count} blank rows: three `FIT` and three `HOLDOUT` slots per absolute candidate cell. These are empty slots, not ties: `tie_id`, coordinates, bounds, basis, evidence paths, and independence groups are intentionally blank. Keep holdouts out of fitting; do not lower a localization bound to reach the 3+3 minimum. A reference feature that cannot be localized from the native 10 m evidence must be marked `TIE_NOT_PRECISION_ELIGIBLE` and excluded from the counts.
"""


def export_alignment_review_packet(
    source_run_dir: Path,
    output_dir: Path,
    *,
    review_scope_allowlist: str | None = None,
) -> dict[str, Any]:
    """Create a new review package from one structurally valid fixed-case run.

    Existing output directories are never overwritten. Only numeric, unapplied
    direct-base candidates enter the review template; numeric date-pair ties
    remain in a separate relative-only diagnostic layer unless a strict review
    scope intentionally omits them.
    """
    source_run = Path(source_run_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Review package output already exists: {output}")
    if not source_run.is_dir():
        raise NotADirectoryError(source_run)
    if source_run == output or source_run in output.parents:
        raise ValueError("Review package output must be outside the immutable source run")

    qa, product, receipt, source_files = _validate_run(source_run)
    gpkg_path = source_files["source_geopackage"]
    source_hashes_before = {str(path): _sha256(path) for path in source_files.values()}
    direct_receipts, relative_receipts, multi_anchor_receipts = _checked_references(receipt)
    spatial_candidate_lookup, spatial_sources, spatial_summary = (
        _checked_spatial_composition_sources(
            receipt, product, case_id=str(qa.get("case_id") or receipt.get("case_id") or "")))
    multi_anchor_report = receipt.get("multi_anchor_base_tie_source_report")
    relative_diagnostic_provided = bool(relative_receipts)
    source_hashes_before.update({str(path): digest for path, digest in direct_receipts.items()})
    source_hashes_before.update({str(path): digest for path, digest in relative_receipts.items()})
    source_hashes_before.update({str(path): digest for path, digest in multi_anchor_receipts.items()})
    source_hashes_before.update({str(path): digest for path, digest in spatial_sources.items()})
    review_context_sources = _review_context_sources(direct_receipts, relative_receipts)
    review_context_source_count_before_scope = len(review_context_sources)
    review_context_sources = _scope_review_context_sources(
        review_context_sources, review_scope_allowlist)
    review_context_sources_omitted_by_scope = (
        review_context_source_count_before_scope - len(review_context_sources))
    for context in review_context_sources:
        context_path = Path(context["source_path"])
        context_hash = context["source_sha256"]
        previous_hash = source_hashes_before.get(str(context_path))
        if previous_hash is not None and previous_hash.casefold() != context_hash.casefold():
            raise ValueError(f"Review context image conflicts with a recorded source hash: {context_path}")
        source_hashes_before[str(context_path)] = context_hash

    base_candidates, crs = _read_vector(gpkg_path, SOURCE_LAYER)
    if "32649" not in crs:
        raise ValueError(f"Alignment candidate layer must use EPSG:32649, got {crs!r}")
    product_spatial_keys = {
        (str(record["properties"].get("field_id") or ""),
         str(record["properties"].get("cell_id") or ""))
        for record in base_candidates
        if record["properties"].get("method") == SPATIAL_COMPOSITION_METHOD
    }
    if product_spatial_keys != set(spatial_candidate_lookup):
        raise ValueError("Fixed-case spatial-composition candidates differ from their verified receipt")

    direct_rows = []
    relative_rows = []
    multi_anchor_count = 0
    spatial_composition_count = 0
    spatial_excluded_status_counts: dict[str, int] = {}
    direct_reference_hashes: dict[Path, str] = {}
    direct_diagnostic_cache: dict[Path, dict] = {}
    for record in base_candidates:
        props = record["properties"]
        method = props.get("method")
        dx, dy = _number(props.get("correction_dx_m")), _number(props.get("correction_dy_m"))
        numeric = dx is not None and dy is not None
        if method not in ({DIRECT_METHOD, MULTI_ANCHOR_METHOD,
                           SPATIAL_COMPOSITION_METHOD} | RELATIVE_METHODS) or not numeric:
            continue
        for name in ("reviewed", "independent_ground", "accepted", "applied"):
            if _flag(props.get(name), name):
                raise ValueError(f"Source candidate is already reviewed or applied: {props.get('source_ref')}")
        geometry = _multipolygon(record["geometry"])
        source_ref = props.get("source_ref")
        if not isinstance(source_ref, str) or not source_ref:
            raise ValueError("Numeric alignment candidate is missing its exact source reference")
        if method == DIRECT_METHOD:
            if "#" not in source_ref:
                raise ValueError("Direct-base candidate lacks its exact source reference")
            source_diagnostic = Path(source_ref.split("#", 1)[0]).resolve()
            if (props.get("reference_date_status") != "REFERENCE_ACQUISITION_DATE_UNKNOWN"
                    or props.get("scope") != "IMAGE_TO_REFERENCE_GRID"
                    or props.get("status") != "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED"):
                raise ValueError("Direct-base candidate has conflicting reference or correction semantics")
            reference_date = props.get("reference_date")
            if (reference_date is not None
                    and not (isinstance(reference_date, float) and math.isnan(reference_date))
                    and not (isinstance(reference_date, str) and not reference_date.strip())):
                raise ValueError("Annual reference acquisition date must remain unknown")
            reference = _absolute_path(props.get("reference_source"), base=source_run)
            if not reference.is_file():
                raise FileNotFoundError(f"Direct-base reference raster is missing: {reference}")
            if source_diagnostic not in direct_receipts:
                raise ValueError(f"Direct-base source diagnostic is absent from the evidence receipt: {source_diagnostic}")
            reference_hash = direct_reference_hashes.get(reference)
            if reference_hash is None:
                reference_hash = _sha256(reference)
            row_reference_hash = props.get("reference_sha256")
            if isinstance(row_reference_hash, str) and row_reference_hash.strip():
                if row_reference_hash.casefold() != reference_hash.casefold():
                    raise ValueError(f"Reference raster hash differs from candidate receipt: {reference}")
            _verify_direct_candidate_source(
                diagnostic_path=source_diagnostic,
                diagnostic_cache=direct_diagnostic_cache,
                source_ref=source_ref,
                field_id=str(props.get("field_id") or ""),
                cell_id=str(props.get("cell_id") or ""),
                moving_date=str(props.get("moving_date") or ""),
                reference_path=reference,
                reference_sha256=reference_hash,
                correction_dx_m=dx,
                correction_dy_m=dy,
                support_geometry=geometry,
            )
            if reference in direct_reference_hashes and direct_reference_hashes[reference] != reference_hash:
                raise ValueError(f"Reference raster changed while exporting: {reference}")
            direct_reference_hashes[reference] = reference_hash
            direct_rows.append({
                "feature_id": record["feature_id"],
                "geometry": geometry,
                "properties": {
                    "field_id": str(props["field_id"]),
                    "cell_id": str(props["cell_id"]),
                    "moving_date": str(props["moving_date"]),
                    "reference_source": str(reference),
                    "reference_source_sha256": reference_hash,
                    "candidate_source_ref": source_ref,
                    "correction_dx_m": dx,
                    "correction_dy_m": dy,
                    "independent_ground_evidence": 0,
                    "ground_evidence_type": "",
                    "review_basis": "",
                    "alignment_evidence_type": "",
                    "reviewed": 0,
                    "positional_uncertainty_m": None,
                    "positional_uncertainty_semantics": "",
                    "uncertainty_semantics": "",
                    "relative_registration_uncertainty_m": None,
                    "fit_max_adjusted_bound_m": None,
                    "holdout_max_adjusted_bound_m": None,
                    "nonbuilding_tie_pairs": 0,
                    "holdout_tie_pairs": 0,
                    "relative_ties_within_support_cell": 0,
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "evidence_path": "",
                    "evidence_sha256": "",
                    "review_state": "PENDING",
                    "candidate_status": str(props.get("status") or ""),
                    "method": str(method),
                    "fit_dispersion_m": _number(props.get("fit_disp_m")),
                    "candidate_position_uncertainty_m": _number(props.get("position_unc_m")),
                    "source_feature_id": str(record["feature_id"]),
                    "source_run_gpkg_sha256": source_hashes_before[str(gpkg_path)],
                    "reference_date_status": str(props.get("reference_date_status") or ""),
                    "correction_scope": str(props.get("correction_scope") or "IMAGE_TO_REFERENCE_GRID"),
                    "anchor_count": 0,
                    "anchor_years": "",
                    "anchor_median_correlation": None,
                    "anchor_median_peak_margin": None,
                    "fit_dispersion_kind": "LOCAL_FIT_DISPERSION_NOT_POSITIONAL_UNCERTAINTY",
                    "accepted": 0,
                    "applied": 0,
                },
            })
        elif method == MULTI_ANCHOR_METHOD:
            if not isinstance(multi_anchor_report, dict):
                raise ValueError("Multi-anchor candidate has no embedded source report")
            if source_ref.startswith("#") or source_ref.startswith("multi-anchor:") is False:
                raise ValueError("Multi-anchor candidate has a malformed source reference")
            reference, reference_hash, anchor_summary = _verify_multi_anchor_candidate(
                report=multi_anchor_report,
                reference_rasters={
                    Path(row["path"]).resolve(): str(row["sha256"]).lower()
                    for row in multi_anchor_report.get("reference_rasters", [])
                    if isinstance(row, dict) and row.get("path") and row.get("sha256")
                },
                properties=props,
                source_ref=source_ref,
                correction_dx_m=dx,
                correction_dy_m=dy,
                support_geometry=geometry,
            )
            if reference not in multi_anchor_receipts:
                raise ValueError(f"Multi-anchor canonical raster is absent from its receipt: {reference}")
            if multi_anchor_receipts[reference].casefold() != reference_hash.casefold():
                raise ValueError(f"Multi-anchor canonical raster hash differs: {reference}")
            if reference in direct_reference_hashes and direct_reference_hashes[reference] != reference_hash:
                raise ValueError(f"Reference raster changed while exporting: {reference}")
            direct_reference_hashes[reference] = reference_hash
            anchor_years = [str(year) for year in anchor_summary["reference_years"]]
            direct_rows.append({
                "feature_id": record["feature_id"],
                "geometry": geometry,
                "properties": {
                    "field_id": str(props["field_id"]),
                    "cell_id": str(props["cell_id"]),
                    "moving_date": str(props["moving_date"]),
                    "reference_source": str(reference),
                    "reference_source_sha256": reference_hash,
                    "candidate_source_ref": source_ref,
                    "correction_dx_m": dx,
                    "correction_dy_m": dy,
                    "independent_ground_evidence": 0,
                    "ground_evidence_type": "",
                    "review_basis": "",
                    "alignment_evidence_type": "",
                    "reviewed": 0,
                    "positional_uncertainty_m": None,
                    "positional_uncertainty_semantics": "",
                    "uncertainty_semantics": "",
                    "relative_registration_uncertainty_m": None,
                    "fit_max_adjusted_bound_m": None,
                    "holdout_max_adjusted_bound_m": None,
                    "nonbuilding_tie_pairs": 0,
                    "holdout_tie_pairs": 0,
                    "relative_ties_within_support_cell": 0,
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "evidence_path": "",
                    "evidence_sha256": "",
                    "review_state": "PENDING",
                    "candidate_status": str(props.get("status") or ""),
                    "method": str(method),
                    "fit_dispersion_m": _number(props.get("fit_disp_m")),
                    "candidate_position_uncertainty_m": None,
                    "source_feature_id": str(record["feature_id"]),
                    "source_run_gpkg_sha256": source_hashes_before[str(gpkg_path)],
                    "reference_date_status": str(props.get("reference_date_status") or ""),
                    "correction_scope": "IMAGE_TO_REFERENCE_GRID",
                    "anchor_count": int(anchor_summary["anchor_count"]),
                    "anchor_years": ",".join(anchor_years),
                    "anchor_median_correlation": _number(anchor_summary.get("median_correlation")),
                    "anchor_median_peak_margin": _number(anchor_summary.get("median_peak_margin")),
                    "fit_dispersion_kind": "CROSS_ANCHOR_MEDIAN_RADIAL_DEVIATION_NOT_POSITIONAL_UNCERTAINTY",
                    "accepted": 0,
                    "applied": 0,
                },
            })
            multi_anchor_count += 1
        elif method == SPATIAL_COMPOSITION_METHOD:
            key = (str(props.get("field_id") or ""), str(props.get("cell_id") or ""))
            field_cell = spatial_candidate_lookup.get(key)
            if field_cell is None:
                raise ValueError(f"Spatial-composition candidate is absent from its verified source: {key}")
            alignment_field, cell = field_cell
            try:
                source_path_text, source_fragment = source_ref.split("#", 1)
            except ValueError as exc:
                raise ValueError("Spatial-composition candidate lacks its exact source reference") from exc
            if (Path(source_path_text).resolve() != Path(spatial_summary["geopackage_path"]).resolve()
                    or source_fragment != f"spatial_alignment_candidates/{cell.cell_id}"
                    or cell.source_ref != source_ref):
                raise ValueError(f"Spatial-composition source reference mismatch: {key}")
            if (alignment_field.field_id != key[0]
                    or alignment_field.moving_date != str(props.get("moving_date") or "")
                    or alignment_field.reference_date is not None
                    or alignment_field.reference_date_status != "REFERENCE_ACQUISITION_DATE_UNKNOWN"
                    or props.get("reference_date_status") != "REFERENCE_ACQUISITION_DATE_UNKNOWN"
                    or props.get("scope") != "IMAGE_TO_REFERENCE_GRID"):
                raise ValueError(f"Spatial-composition date/reference semantics mismatch: {key}")
            if (cell.status != props.get("status")
                    or not math.isclose(float(cell.correction_dx_m_candidate), dx,
                                        rel_tol=0.0, abs_tol=1e-5)
                    or not math.isclose(float(cell.correction_dy_m_candidate), dy,
                                        rel_tol=0.0, abs_tol=1e-5)
                    or geometry.hausdorff_distance(cell.support_geometry) > 1e-6):
                raise ValueError(f"Spatial-composition product differs from its source cell: {key}")
            reference = _absolute_path(props.get("reference_source"), base=source_run)
            reference_hash = spatial_summary["reference_sha256"]
            if (reference != Path(spatial_summary["reference_path"]).resolve()
                    or _sha256(reference).casefold() != reference_hash.casefold()):
                raise ValueError(f"Spatial-composition reference raster changed: {reference}")
            allowed_spatial_statuses = {
                "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
                "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED",
                "SPATIALLY_COMPOSED_CANDIDATE_TREE_NO_CYCLE_CHECK",
            }
            if cell.status not in allowed_spatial_statuses:
                raise ValueError(f"Unsupported spatial-composition candidate status: {cell.status}")
            if cell.status != "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY":
                spatial_excluded_status_counts[cell.status] = (
                    spatial_excluded_status_counts.get(cell.status, 0) + 1)
                continue
            if reference in direct_reference_hashes and direct_reference_hashes[reference] != reference_hash:
                raise ValueError(f"Reference raster changed while exporting: {reference}")
            direct_reference_hashes[reference] = reference_hash
            spatial_composition_count += 1
            metadata = cell.metadata
            direct_rows.append({
                "feature_id": record["feature_id"],
                "geometry": geometry,
                "properties": {
                    "field_id": key[0],
                    "cell_id": key[1],
                    "moving_date": alignment_field.moving_date,
                    "reference_source": str(reference),
                    "reference_source_sha256": reference_hash,
                    "candidate_source_ref": source_ref,
                    "correction_dx_m": dx,
                    "correction_dy_m": dy,
                    "independent_ground_evidence": 0,
                    "ground_evidence_type": "",
                    "review_basis": "",
                    "alignment_evidence_type": "",
                    "reviewed": 0,
                    "positional_uncertainty_m": None,
                    "positional_uncertainty_semantics": "",
                    "uncertainty_semantics": "",
                    "relative_registration_uncertainty_m": None,
                    "fit_max_adjusted_bound_m": None,
                    "holdout_max_adjusted_bound_m": None,
                    "nonbuilding_tie_pairs": 0,
                    "holdout_tie_pairs": 0,
                    "relative_ties_within_support_cell": 0,
                    "reviewer_id": "",
                    "reviewed_at": "",
                    "evidence_path": "",
                    "evidence_sha256": "",
                    "review_state": "PENDING",
                    "candidate_status": str(props.get("status") or ""),
                    "method": str(method),
                    "fit_dispersion_m": None,
                    "candidate_position_uncertainty_m": None,
                    "source_feature_id": str(record["feature_id"]),
                    "source_run_gpkg_sha256": source_hashes_before[str(gpkg_path)],
                    "reference_date_status": alignment_field.reference_date_status,
                    "correction_scope": "IMAGE_TO_REFERENCE_GRID",
                    "anchor_count": 0,
                    "anchor_years": "",
                    "anchor_median_correlation": None,
                    "anchor_median_peak_margin": None,
                    "fit_dispersion_kind": "POSE_GRAPH_CYCLE_CLOSURE_NOT_POSITIONAL_UNCERTAINTY",
                    "source_component_scope": str(metadata.get("source_component_scope") or "SOURCE_COMPONENT_SCOPE_UNKNOWN"),
                    "reference_component_id_status": str(metadata.get("reference_component_id_status") or "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
                    "cycle_closure_max_m": _number(metadata.get("closure_max_m")),
                    "source_edge_count": int(metadata.get("source_edge_count") or 0),
                    "accepted": 0,
                    "applied": 0,
                },
            })
        else:
            source_diagnostic = Path(source_ref.split("#", 1)[0]).resolve()
            if props.get("reference_date_status") != "ACTUAL_DATE_KNOWN":
                raise ValueError("Relative candidate must identify its actual reference acquisition date")
            cell = _verify_relative_candidate_source(
                diagnostic_path=source_diagnostic,
                diagnostic_cache=direct_diagnostic_cache,
                relative_receipts=relative_receipts,
                source_ref=source_ref,
                case_id=str(qa.get("case_id") or receipt.get("case_id") or ""),
                field_id=str(props.get("field_id") or ""),
                cell_id=str(props.get("cell_id") or ""),
                reference_date=str(props.get("reference_date") or ""),
                moving_date=str(props.get("moving_date") or ""),
                candidate_status=str(props.get("status") or ""),
                correction_dx_m=dx,
                correction_dy_m=dy,
                reference_source=str(props.get("reference_source") or ""),
                support_geometry=geometry,
            )

            def _as_count(name: str) -> int:
                value = _number(cell.get(name))
                return int(value) if value is not None else 0

            relative_rows.append({
                "feature_id": record["feature_id"],
                "geometry": geometry,
                "properties": {
                    "field_id": str(props["field_id"]),
                    "cell_id": str(props["cell_id"]),
                    "reference_date": str(props.get("reference_date") or ""),
                    "moving_date": str(props["moving_date"]),
                    "reference_source": str(props.get("reference_source") or ""),
                    "candidate_source_ref": source_ref,
                    "candidate_status": str(props.get("status") or ""),
                    "method": str(method),
                    "correction_dx_m_candidate": dx,
                    "correction_dy_m_candidate": dy,
                    "fit_dispersion_m_candidate": _number(props.get("fit_disp_m")),
                    "candidate_position_uncertainty_m": _number(props.get("position_unc_m")),
                    "matched_features_raw": _as_count("matched_features_raw"),
                    "eligible_matches": _as_count("eligible_matches"),
                    "inliers": _as_count("inliers"),
                    "inlier_ratio": _number(cell.get("inlier_ratio")),
                    "coverage_4x4": _number(cell.get("coverage_4x4")),
                    "median_residual_px": _number(cell.get("median_residual_px")),
                    "p90_residual_px": _number(cell.get("p90_residual_px")),
                    "candidate_ground_mask_fraction_from": _number(
                        cell.get("candidate_ground_mask_fraction_from")),
                    "candidate_ground_mask_fraction_to": _number(
                        cell.get("candidate_ground_mask_fraction_to")),
                    "ground_mask_reviewed": 0,
                    "field_interpolated": 0,
                    "applied_to_image_or_geometry": 0,
                    "accepted": 0,
                    "applied": 0,
                    "review_state": "RELATIVE_ONLY_NOT_BASE_ANCHORED",
                },
            })

    direct_rows, relative_rows, review_scope_manifest = _apply_review_scope_allowlist(
        direct_rows, relative_rows, review_scope_allowlist)
    if review_scope_manifest is not None:
        review_scope_manifest["review_context_source_count"] = len(review_context_sources)
        review_scope_manifest["review_context_sources_omitted"] = (
            review_context_sources_omitted_by_scope)
        multi_anchor_count = sum(
            row["properties"].get("method") == MULTI_ANCHOR_METHOD for row in direct_rows)
        spatial_composition_count = sum(
            row["properties"].get("method") == SPATIAL_COMPOSITION_METHOD for row in direct_rows)
    if not direct_rows and not relative_rows:
        raise ValueError("No numeric absolute or relative alignment candidates are available for review")
    if len({(row["properties"]["field_id"], row["properties"]["cell_id"])
            for row in direct_rows}) != len(direct_rows):
        raise ValueError("Direct-base field/cell candidates are not unique")
    if len({(row["properties"]["field_id"], row["properties"]["cell_id"])
            for row in relative_rows}) != len(relative_rows):
        raise ValueError("Relative date-pair field/cell candidates are not unique")
    if review_scope_manifest is None:
        direct_rows.sort(key=lambda row: (row["properties"]["moving_date"],
                                         row["properties"]["field_id"],
                                         row["properties"]["cell_id"]))
    relative_rows.sort(key=lambda row: (row["properties"]["reference_date"],
                                       row["properties"]["moving_date"],
                                       row["properties"]["field_id"],
                                       row["properties"]["cell_id"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.tmp-", dir=output.parent) as temp_name:
        temp = Path(temp_name)
        _write_vector(temp / "absolute_base_review_template.gpkg", REVIEW_LAYER,
                      direct_rows, REVIEW_FIELDS)
        _write_vector(temp / "relative_date_pair_candidates.gpkg", RELATIVE_LAYER,
                      relative_rows, RELATIVE_FIELDS)
        copied_review_context = _copy_review_context_images(temp, review_context_sources)
        cell_context_images, cell_context_skipped, source_component_image_hashes = (
            _write_candidate_cell_context_images(temp, direct_rows, copied_review_context))
        for tile_path, tile_hash in source_component_image_hashes.items():
            previous_hash = source_hashes_before.get(tile_path)
            if previous_hash is not None and previous_hash.casefold() != tile_hash.casefold():
                raise ValueError(f"Source-tile hash conflicts with another receipt: {tile_path}")
            source_hashes_before[tile_path] = tile_hash
        tie_pair_template_count = _write_tie_pair_review_template(
            temp / "tie_pairs_review_template.csv", direct_rows)
        (temp / "README.md").write_text(
            _readme(str(qa.get("case_id") or receipt.get("case_id") or "fixed-case"),
                    len(direct_rows),
                    len(relative_rows) if relative_diagnostic_provided else None,
                    multi_anchor_count,
                    spatial_composition_count,
                    spatial_excluded_status_counts,
                    len(copied_review_context),
                    tie_pair_template_count,
                    len(cell_context_images),
                    len(source_component_image_hashes),
                    review_scope_allowlist if review_scope_manifest is not None else None),
            encoding="utf-8",
        )

        reference_hashes_after = {str(path): _sha256(path) for path in direct_reference_hashes}
        source_hashes_after = {str(path): _sha256(Path(path)) for path in source_hashes_before}
        if source_hashes_after != source_hashes_before:
            raise RuntimeError("Fixed-case source products changed while exporting review packet")
        if reference_hashes_after != {str(path): digest for path, digest in direct_reference_hashes.items()}:
            raise RuntimeError("Reference raster changed while exporting review packet")

        packet_manifest = {
            "schema_version": "xian_r5.alignment_review_packet.v6",
            "review_scope_allowlist": (review_scope_manifest or {
                "allowlist_id": None, "status": "NOT_APPLIED",
                "candidate_fallback": False,
            }),
            "case_id": qa.get("case_id"),
            "source_run_dir": str(source_run),
            "source_candidate_layer": SOURCE_LAYER,
            "source_crs": "EPSG:32649",
            "source_feature_count": len(base_candidates),
            "absolute_base_review": {
                "path": "absolute_base_review_template.gpkg",
                "layer": REVIEW_LAYER,
                "status": ("NUMERIC_CANDIDATES_AVAILABLE" if direct_rows
                           else "NO_NUMERIC_ABSOLUTE_CANDIDATES"),
                "numeric_candidate_count": len(direct_rows),
                "direct_single_anchor_candidate_count": (
                    len(direct_rows) - multi_anchor_count - spatial_composition_count),
                "multi_anchor_candidate_count": multi_anchor_count,
                "spatial_composition_candidate_count": spatial_composition_count,
                "accepted_count": 0,
                "applied_count": 0,
                "reference_rasters_sha256": {str(path): digest
                                              for path, digest in direct_reference_hashes.items()},
                "semantics": "UNREVIEWED_IMAGE_TO_USER_BASE_GRID_CANDIDATES_NOT_GROUND_FOOTPRINTS_OR_ABSOLUTE_ACCURACY",
                "relative_feature_review_allowed": True,
                "relative_review_contract": {
                    "contract_amendment": "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_AMENDMENT_V2",
                    "uncertainty_rule": "R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1",
                    "sidecar_schema": "xian_r5.relative_alignment_tie_sidecar.v2",
                    "adjusted_bound_definition": (
                        "RESIDUAL_PLUS_GE_AND_REFERENCE_LOCALIZATION_BOUNDS"),
                    "minimum_nonbuilding_fit_pairs": 3,
                    "minimum_heldout_pairs": 3,
                    "fit_statistic": "MAX_ADJUSTED_BOUND",
                    "holdout_statistic": "MAX_ADJUSTED_BOUND",
                    "relative_registration_uncertainty_m": "HOLDOUT_MAX_ADJUSTED_BOUND",
                    "allowed_alternative_statistics": [],
                    "positional_uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                    "uncertainty_semantics": "RELATIVE_IMAGE_GRID_ONLY",
                    "absolute_position_accuracy": "UNKNOWN",
                    "ties_and_holdouts_within_exact_support_cell": True,
                    "unknown_source_scope_exception": "SPATIAL_COMPOSITION_CYCLE_SCREEN_ONLY_RELATIVE_REVIEW",
                    "absolute_ground_position_claimed": False,
                    "roof_to_ground_parallax_established": False,
                },
                "absolute_position_accuracy_claimed": False,
            },
            "multi_anchor_base_tie_source": ({
                "status": "PROVIDED",
                "reference_years": multi_anchor_report.get("reference_years"),
                "source_diagnostic_count": len(multi_anchor_report.get("source_diagnostics", [])),
                "reference_raster_sha256": {
                    str(row["path"]): str(row["sha256"]).lower()
                    for row in multi_anchor_report.get("reference_rasters", [])
                    if isinstance(row, dict) and row.get("path") and row.get("sha256")
                },
                "field_interpolated": False,
                "field_applied": False,
                "dispersion_is_positional_uncertainty": False,
                "semantics": "MULTI_ANCHOR_IMAGE_GRID_CANDIDATES_NOT_GROUND_CONTROL_OR_ROOF_TO_GROUND_PARALLAX",
            } if isinstance(multi_anchor_report, dict) else {
                "status": "NOT_PROVIDED",
                "numeric_candidate_count": None,
                "semantics": "NOT_PROVIDED_NOT_A_ZERO_CANDIDATE_RESULT",
            }),
            "spatial_composition_review": ({
                "status": "PROVIDED",
                "candidate_source_manifest": spatial_summary["manifest_path"],
                "candidate_source_manifest_sha256": spatial_summary["manifest_sha256"],
                "candidate_source_geopackage": spatial_summary["geopackage_path"],
                "candidate_source_geopackage_sha256": spatial_summary["geopackage_sha256"],
                "reference_raster": spatial_summary["reference_path"],
                "reference_raster_sha256": spatial_summary["reference_sha256"],
                "source_candidate_count": spatial_summary["candidate_count"],
                "reviewable_cycle_screen_only_count": spatial_composition_count,
                "excluded_status_counts": spatial_excluded_status_counts,
                "accepted_count": 0,
                "applied_count": 0,
                "reference_acquisition_date": "UNKNOWN",
                "per_pixel_reference_component_provenance": "NOT_RECORDED",
                "cycle_closure_is_positional_uncertainty": False,
                "roof_to_ground_parallax_established": False,
                "semantics": "LOCAL_SPATIAL_POSE_GRAPH_CANDIDATES_FOR_IMAGE_TO_SELECTED_BASE_GRID_REVIEW_ONLY",
            } if spatial_summary else {
                "status": "NOT_PROVIDED",
                "candidate_count": None,
                "semantics": "NOT_PROVIDED_NOT_A_ZERO_CANDIDATE_RESULT",
            }),
            "relative_date_pair_diagnostics": {
                "path": "relative_date_pair_candidates.gpkg",
                "layer": RELATIVE_LAYER,
                "status": (
                    "OMITTED_BY_REVIEW_SCOPE" if review_scope_manifest is not None else
                    ("PROVIDED" if relative_diagnostic_provided else "NOT_PROVIDED")),
                "source_file_count": len(relative_receipts),
                "numeric_candidate_count": (
                    None if review_scope_manifest is not None else
                    (len(relative_rows) if relative_diagnostic_provided else None)),
                "accepted_count": 0,
                "applied_count": 0,
                "semantics": (
                    "SOURCE_DIAGNOSTIC_NOT_EXPORTED_BY_EXPLICIT_REVIEW_SCOPE_NOT_A_ZERO_RESULT"
                    if review_scope_manifest is not None else
                    "RELATIVE_IMAGE_TO_IMAGE_ONLY_REQUIRES_COMPOSITION_WITH_ABSOLUTE_BASE_TIE"
                    if relative_diagnostic_provided else
                    "NOT_PROVIDED_EMPTY_SCHEMA_PLACEHOLDER_NOT_A_ZERO_CANDIDATE_RESULT"),
            },
            "review_materials": {
                "context_images": copied_review_context,
                "context_image_count": len(copied_review_context),
                "context_semantics": (
                    "HASHED_DATE_LEVEL_OR_DATE_PAIR_VISUAL_CONTEXT_ONLY_NOT_TIE_EVIDENCE_OR_REVIEW_VERDICT"
                ),
                "candidate_cell_context_images": cell_context_images,
                "candidate_cell_context_image_count": len(cell_context_images),
                "candidate_cell_context_skipped": cell_context_skipped,
                "moving_source_component_image_count": len(source_component_image_hashes),
                "moving_source_component_hashes_sha256": source_component_image_hashes,
                "moving_source_component_semantics": (
                    "HASHED_EXACT_DATE_HIGHRES_SOURCE_TILES_INTERSECTING_EACH_CANDIDATE_SUPPORT"
                ),
                "reference_source_component_semantics": (
                    "10M_REFERENCE_PER_PIXEL_SOURCE_TILE_PROVENANCE_UNKNOWN"
                ),
                "candidate_cell_context_semantics": (
                    "CROPPED_10M_PREVIEW_WITH_SUPPORT_OUTLINE_ONLY_NOT_RAW_TIE_EVIDENCE"
                ),
                "tie_pairs_template": "tie_pairs_review_template.csv",
                "tie_pairs_template_row_count": tie_pair_template_count,
                "slots_per_absolute_candidate": {"fit": 3, "holdout": 3},
                "coordinates_crs": "EPSG:32649",
                "blank_slots_are_evidence": False,
            },
            "source_files_sha256": source_hashes_after,
            "source_hash_recheck": "PASS",
            "reference_hash_recheck": (
                "PASS" if direct_reference_hashes
                else "NOT_APPLICABLE_NO_ABSOLUTE_BASE_CANDIDATES"),
            "formal_ground_correction": qa.get("formal_ground_correction"),
            "formal_change_accuracy": qa.get("formal_change_accuracy"),
            "roof_to_ground_parallax_correction": "NOT_ESTABLISHED_BY_IMAGE_GRID_ALIGNMENT",
            "created_files_sha256": {
                path.relative_to(temp).as_posix(): _sha256(path)
                for path in sorted(temp.rglob("*")) if path.is_file()
            },
        }
        packet_manifest["base_grid_alignment_review"] = dict(
            packet_manifest["absolute_base_review"])
        packet_manifest["absolute_base_review"]["legacy_manifest_key"] = True
        (temp / "REVIEW_PACKET_MANIFEST.json").write_text(
            json.dumps(packet_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"Review package output appeared during export: {output}")
        os.rename(temp, output)

    return packet_manifest
