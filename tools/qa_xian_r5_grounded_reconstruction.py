#!/usr/bin/env python3
"""Bounded structural QA for one isolated actual-imagery engineering output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import mapping, shape

try:
    import fiona
except ImportError:
    fiona = None

SUPPORTED_PRODUCT_SCHEMAS = frozenset({
    "xian_r5.grounded_products.v2",
    "xian_r5.grounded_products.v3",
    "xian_r5.grounded_products.v4",
    "xian_r5.grounded_products.v5",
    "xian_r5.grounded_products.v6",
    "xian_r5.grounded_products.v7",
    "xian_r5.grounded_products.v8",
    "xian_r5.grounded_products.v9",
    "xian_r5.grounded_products.v10",
    "xian_r5.grounded_products.v11",
    "xian_r5.grounded_products.v12",
    "xian_r5.grounded_products.v13",
    "xian_r5.grounded_products.v14",
    "xian_r5.grounded_products.v15",
    "xian_r5.grounded_products.v16",
    "xian_r5.grounded_products.v17",
})
CANDIDATE_METRIC_PRODUCT_SCHEMAS = frozenset({
    "xian_r5.grounded_products.v3",
    "xian_r5.grounded_products.v4",
    "xian_r5.grounded_products.v5",
    "xian_r5.grounded_products.v6",
    "xian_r5.grounded_products.v7",
    "xian_r5.grounded_products.v8",
    "xian_r5.grounded_products.v9",
    "xian_r5.grounded_products.v10",
    "xian_r5.grounded_products.v11",
    "xian_r5.grounded_products.v12",
    "xian_r5.grounded_products.v13",
    "xian_r5.grounded_products.v14",
    "xian_r5.grounded_products.v15",
    "xian_r5.grounded_products.v16",
    "xian_r5.grounded_products.v17",
})
TIF_METADATA_PRODUCT_SCHEMAS = frozenset({
    "xian_r5.grounded_products.v5",
    "xian_r5.grounded_products.v6",
    "xian_r5.grounded_products.v7",
    "xian_r5.grounded_products.v8",
    "xian_r5.grounded_products.v9",
    "xian_r5.grounded_products.v10",
    "xian_r5.grounded_products.v11",
    "xian_r5.grounded_products.v12",
    "xian_r5.grounded_products.v13",
    "xian_r5.grounded_products.v14",
    "xian_r5.grounded_products.v15",
    "xian_r5.grounded_products.v16",
    "xian_r5.grounded_products.v17",
})
_ROOF_SUPPORT_QUALITY_LABELS = {
    "INSUFFICIENT_MASKED_SAMPLES": "insufficient_masked_tie_samples",
    "NO_VALID_CORRELATION_SCORE": "no_valid_correlation_peak",
    "INSUFFICIENT_ANCHOR_CANDIDATES": "insufficient_multi_anchor_support",
}
_SOURCE_COMPONENT_QUALITY_LABELS = {
    "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED": "mosaic_component_boundary_unresolved",
    "SOURCE_COMPONENT_COVERAGE_INCOMPLETE": "source_component_coverage_incomplete",
    "SOURCE_COMPONENT_ID_MISMATCH": "source_component_id_mismatch",
    "SOURCE_COMPONENT_SCOPE_UNKNOWN": "source_component_scope_unknown",
}


def _valid_unknown_scope_spatial_relative_review(
    cell: dict, product_schema: str | None,
) -> bool:
    """Validate the narrowly allowed relative-only composed-cell exception."""
    closure = cell.get("cycle_closure_max_m")
    relative_uncertainty = cell.get("relative_unc_m")
    return bool(
        product_schema in {"xian_r5.grounded_products.v16",
                           "xian_r5.grounded_products.v17"}
        and cell.get("method") == "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE"
        and cell.get("candidate_status")
            == "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY"
        and cell.get("status") == "ACCEPTED_REVIEWED_RELATIVE_IMAGE_TIE"
        and cell.get("alignment_review_basis") == "RELATIVE_REFERENCE_IMAGE_FEATURES"
        and cell.get("accepted") == 1 and cell.get("reviewed") == 1
        and cell.get("applied") == 1 and cell.get("independent_ground") == 0
        and cell.get("position_unc_m") is None
        and isinstance(relative_uncertainty, (int, float))
        and not isinstance(relative_uncertainty, bool)
        and math.isfinite(float(relative_uncertainty))
        and 0 <= float(relative_uncertainty) <= 2.0
        and cell.get("relative_ties_within_support_cell") == 1
        and cell.get("relative_cell_scope_review_authorized") == 1
        and all(isinstance(cell.get(name), int) and cell.get(name) >= 3
                for name in ("nonbuilding_tie_pairs", "holdout_tie_pairs"))
        and cell.get("source_component_scope") == "SOURCE_COMPONENT_SCOPE_UNKNOWN"
        and cell.get("source_component_ids_json") == "[]"
        and cell.get("source_component_coverage") is None
        and cell.get("reference_component_id_status")
            == "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"
        and isinstance(cell.get("source_edge_count"), int)
        and not isinstance(cell.get("source_edge_count"), bool)
        and cell.get("source_edge_count") >= 2
        and isinstance(closure, (int, float)) and not isinstance(closure, bool)
        and math.isfinite(float(closure))
        and cell.get("fit_dispersion_kind")
            == "POSE_GRAPH_CYCLE_CLOSURE_NOT_POSITIONAL_UNCERTAINTY"
    )


def _roof_support_quality_code(
    support_status: str | None, quality_codes: dict,
) -> int:
    """Return the mapped failure-reason code, preserving legacy code 3."""
    label = _ROOF_SUPPORT_QUALITY_LABELS.get(support_status or "")
    if label is None or label not in quality_codes:
        return 3
    return int(quality_codes[label])


def _source_component_quality_code(
    status: str | None, quality_codes: dict,
) -> int | None:
    label = _SOURCE_COMPONENT_QUALITY_LABELS.get(status or "")
    if label is None or label not in quality_codes:
        return None
    return int(quality_codes[label])


_SPATIAL_QUALITY_CODES = {
    "background": 0,
    "reviewed_ground": 1,
    "roof_only_unresolved": 2,
    "unverified_ground_candidate": 3,
    "ambiguous_ground_overlap": 4,
}


def _spatial_quality_codes_valid(quality_codes: dict) -> bool:
    return quality_codes == _SPATIAL_QUALITY_CODES


def _spatial_quality_matches_footprint(
    quality: np.ndarray, footprint: np.ndarray, quality_codes: dict,
    footprint_codes: dict,
) -> bool:
    """Ensure spatial QA classes describe the paired footprint-state raster."""
    if not _spatial_quality_codes_valid(quality_codes) or quality.shape != footprint.shape:
        return False
    quality_background = int(quality_codes["background"])
    quality_ground = int(quality_codes["reviewed_ground"])
    quality_unresolved = [
        int(quality_codes["roof_only_unresolved"]),
        int(quality_codes["unverified_ground_candidate"]),
        int(quality_codes["ambiguous_ground_overlap"]),
    ]
    footprint_background = int(footprint_codes["background"])
    footprint_ground = int(footprint_codes["confirmed_ground"])
    footprint_unresolved = int(footprint_codes["unresolved"])
    allowed_footprint = [footprint_background, footprint_ground, footprint_unresolved]
    nodata = footprint_codes.get("nodata")
    if nodata is not None:
        allowed_footprint.append(int(nodata))
    if (not np.isin(quality, [quality_background, quality_ground, *quality_unresolved]).all()
            or not np.isin(footprint, allowed_footprint).all()):
        return False
    background_mask = footprint == footprint_background
    if nodata is not None:
        background_mask |= footprint == int(nodata)
    return (
        np.array_equal(quality == quality_background, background_mask)
        and np.array_equal(quality == quality_ground, footprint == footprint_ground)
        and np.array_equal(np.isin(quality, quality_unresolved),
                           footprint == footprint_unresolved)
    )


class _PyogrioFeatureCollection:
    """Small read-only Fiona-compatible view over one Pyogrio layer."""

    def __init__(self, path: Path, layer: str):
        try:
            import pyogrio
        except ImportError as exc:
            raise RuntimeError("Engineering QA requires Fiona or Pyogrio") from exc

        info = pyogrio.read_info(path, layer=layer)
        frame = pyogrio.read_dataframe(path, layer=layer)
        self.schema = {
            "geometry": info.get("geometry_type") or "Unknown",
            "properties": dict(zip(info.get("fields", ()), info.get("dtypes", ()))),
        }
        geometry_accessor = getattr(frame, "geometry", None)
        geometry_column = getattr(geometry_accessor, "name", None)
        self._features = []
        for ordinal, (_, row) in enumerate(frame.iterrows()):
            geometry = row[geometry_column] if geometry_column in frame.columns else None
            properties = {}
            for name in info.get("fields", ()):
                value = row[name]
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, float) and math.isnan(value):
                    value = None
                properties[name] = value
            self._features.append({
                "id": str(ordinal),
                "geometry": mapping(geometry) if geometry is not None else None,
                "properties": properties,
            })

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self._features)

    def __len__(self):
        return len(self._features)


def _open_vector(path: Path, *, layer: str):
    """Open one vector layer read-only with Fiona or the Pyogrio fallback."""
    if fiona is not None:
        return fiona.open(path, layer=layer)
    return _PyogrioFeatureCollection(path, layer)


def _candidate_metrics_required(product_schema: str | None) -> bool:
    if product_schema not in SUPPORTED_PRODUCT_SCHEMAS:
        supported = ", ".join(sorted(SUPPORTED_PRODUCT_SCHEMAS))
        raise ValueError(f"Unsupported product manifest schema; expected one of: {supported}")
    return product_schema in CANDIDATE_METRIC_PRODUCT_SCHEMAS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygon_overlap_metrics(left, right) -> tuple[float, float, float]:
    intersection_area = float(left.intersection(right).area)
    union_area = float(left.union(right).area)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou, intersection_area, float(left.distance(right))


def _coarsen_candidate_mask(array: np.ndarray, factor: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Candidate fine grid does not contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    result = np.zeros(blocks.shape[:2], dtype=np.uint8)
    result[np.any(blocks == 1, axis=2)] = 1
    result[np.any(blocks == 2, axis=2)] = 2
    return result


def _coarsen_candidate_quality(array: np.ndarray, factor: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Candidate quality grid does not contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    return blocks.max(axis=2).astype(np.uint8)


def _coarsen_candidate_identity_ids(
    array: np.ndarray, factor: int, unknown: int,
) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Candidate identity grid does not contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(
        0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    unknown_mask = blocks == unknown
    identity_pixels = (blocks > 0) & ~unknown_mask
    has_identity = np.any(identity_pixels, axis=2)
    low_values = np.where(identity_pixels, blocks, np.iinfo(np.uint32).max)
    high_values = np.where(identity_pixels, blocks, 0)
    low, high = low_values.min(axis=2), high_values.max(axis=2)
    result = np.zeros(low.shape, dtype=np.uint32)
    single = has_identity & (low == high)
    result[single] = low[single]
    result[unknown_mask.any(axis=2) | (has_identity & (low != high))] = unknown
    return result


def check(root: Path, receipt_name: str = "ENGINEERING_QA.json") -> dict:
    receipt = json.loads((root / "EVIDENCE_RECEIPT.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "PRODUCT_MANIFEST.json").read_text(encoding="utf-8"))
    epochs = manifest["epochs"]
    manifest_product_schema = manifest.get("schema_version")
    product_schema = manifest_product_schema
    # v14-v17 extend the v13 product contract. Keep the exact manifest version
    # for TIF tag checks and use v13's shared structural gates.
    if product_schema in {"xian_r5.grounded_products.v14",
                          "xian_r5.grounded_products.v15",
                          "xian_r5.grounded_products.v16",
                          "xian_r5.grounded_products.v17"}:
        product_schema = "xian_r5.grounded_products.v13"
    actual_date_schema = product_schema in {
        "xian_r5.grounded_products.v8", "xian_r5.grounded_products.v9",
        "xian_r5.grounded_products.v10", "xian_r5.grounded_products.v11",
        "xian_r5.grounded_products.v12", "xian_r5.grounded_products.v13"}
    if actual_date_schema:
        actual_date_order = manifest.get("actual_date_order")
        output_groups = manifest.get("output_groups")
        group_metadata = manifest.get("output_group_metadata")
        parsed_dates = []
        try:
            parsed_dates = [date.fromisoformat(value) for value in actual_date_order]
        except (TypeError, ValueError):
            pass
        if (not isinstance(actual_date_order, list)
                or parsed_dates != sorted(set(parsed_dates))
                or len(parsed_dates) != len(actual_date_order)
                or actual_date_order != [value.isoformat() for value in parsed_dates]
                or not isinstance(output_groups, list)
                or output_groups != epochs
                or not isinstance(group_metadata, dict)
                or set(group_metadata) != set(output_groups)
                or [group for group in output_groups if group in actual_date_order]
                != actual_date_order):
            raise ValueError("Actual-date output-group manifest is invalid")
        for group in output_groups:
            item = group_metadata[group]
            if group in actual_date_order:
                if (item.get("actual_date") != group
                        or item.get("actual_date_status") != "VERIFIED_ACTUAL_DATE"):
                    raise ValueError(f"Verified output group metadata is invalid: {group}")
            elif (item.get("actual_date") is not None
                  or item.get("actual_date_status") != "ACTUAL_DATE_UNRESOLVED"):
                raise ValueError(f"Unresolved output group metadata is invalid: {group}")
    candidate_metrics_required = _candidate_metrics_required(manifest_product_schema)
    footprint_codes = manifest.get("footprint_codes", {})
    id_codes = manifest.get("id_codes", {})
    change_codes = manifest.get("change_codes", {})
    candidate_codes = manifest.get("image_grid_roof_candidate_codes", {})
    if footprint_codes.get("unresolved") == footprint_codes.get("nodata"):
        raise ValueError("Footprint unresolved and NoData must have distinct codes")
    if change_codes.get("UNRESOLVED") == change_codes.get("nodata"):
        raise ValueError("Change unresolved and NoData must have distinct codes")
    if id_codes.get("unresolved") == id_codes.get("nodata"):
        raise ValueError("Identity unresolved and NoData must have distinct codes")
    if candidate_codes.get("unresolved") == candidate_codes.get("nodata"):
        raise ValueError("Roof-candidate unresolved and NoData must have distinct codes")
    if product_schema == "xian_r5.grounded_products.v10":
        expected_candidate_policy = {
            "full_cell": "ONE_UNIQUE_FIELD_CELL_COVERS_THE_WHOLE_ROOF",
            "dominant_cell_candidate_only": {
                "method": "MAXIMUM_ROOF_AREA_INTERSECTION_FRACTION",
                "minimum_fraction": 0.8,
                "accepted_partial_cell_application": False,
                "whole_geometry_translation": True,
            },
            "piecewise_warp": False,
            "roof_to_ground_correction": False,
        }
        if manifest.get("image_grid_roof_candidate_policy") != expected_candidate_policy:
            raise ValueError("Roof candidate dominant-cell policy is invalid")
    elif product_schema in {"xian_r5.grounded_products.v11",
                            "xian_r5.grounded_products.v12",
                            "xian_r5.grounded_products.v13"}:
        source_component_gate = (
            "ONE_HASH_IDENTIFIED_SOURCE_TILE_PER_ACTUAL_DATE_REQUIRED_EXCEPT_EXACT_SPATIAL_COMPOSITION_CELL_WITH_REVIEWED_RELATIVE_TIES_AND_HOLDOUTS_WITHIN_CELL;"
            "EXCEPTION_IS_RELATIVE_GRID_ONLY_NOT_ABSOLUTE_GROUND_OR_PARALLAX"
            if manifest_product_schema in {"xian_r5.grounded_products.v16",
                                           "xian_r5.grounded_products.v17"} else
            "ONE_HASH_IDENTIFIED_SOURCE_TILE_PER_ACTUAL_DATE_REQUIRED_FOR_ANY_SHIFT")
        expected_candidate_policy = {
            "full_cell": "ONE_UNIQUE_FIELD_CELL_COVERS_THE_WHOLE_ROOF",
            "source_component_gate": source_component_gate,
            "dominant_cell_candidate_only": {
                "method": "MAXIMUM_ROOF_AREA_INTERSECTION_FRACTION",
                "minimum_fraction": 0.8,
                "accepted_partial_cell_application": False,
                "whole_geometry_translation": True,
            },
            "piecewise_warp": False,
            "roof_to_ground_correction": False,
            "cross_component_candidate_shift": False,
        }
        if manifest.get("image_grid_roof_candidate_policy") != expected_candidate_policy:
            raise ValueError("Roof candidate source-component policy is invalid")
    tifs = sorted(root.glob("*.tif"))
    errors = []
    registration_backend = receipt.get("registration_backend")
    if registration_backend and registration_backend.get("status") == "BLOCKED_NO_OPENCV":
        if registration_backend.get("available") is not False:
            errors.append("REGISTRATION_BACKEND_BLOCKED_RECEIPT_CONTRADICTION")
        if any(row.get("status") != "BLOCKED_NO_OPENCV"
               for row in receipt.get("registrations", [])):
            errors.append("REGISTRATION_ROW_NOT_BLOCKED_WITH_BACKEND")
        if any(row.get("registration_status") != "BLOCKED_NO_OPENCV"
               for row in receipt.get("observations", [])):
            errors.append("OBSERVATION_REGISTRATION_STATUS_NOT_BLOCKED")
    if product_schema in {"xian_r5.grounded_products.v9",
                          "xian_r5.grounded_products.v10",
                          "xian_r5.grounded_products.v11",
                          "xian_r5.grounded_products.v12",
                          "xian_r5.grounded_products.v13"}:
        metadata_contract = manifest.get("tif_metadata_contract", {})
        expected_component_provenance = (
            "UNKNOWN_PER_PIXEL_SOURCE_COMPONENT_DATE_AND_VIEW_ANGLE"
            if manifest.get("reference_grid") else "NOT_APPLICABLE")
        if (metadata_contract.get("version") != "xian_r5.tif_metadata.v2"
                or metadata_contract.get("base_grid_pixel_provenance_status")
                != expected_component_provenance):
            errors.append("TIF_BASE_GRID_PROVENANCE_CONTRACT_INVALID")
    grids = set()
    profiles = {}
    codes = {}
    spatial_quality_by_group = {}
    footprint_state_by_group = {}
    quality_codes = manifest.get("quality_codes", {})
    if (product_schema in {"xian_r5.grounded_products.v12",
                           "xian_r5.grounded_products.v13"}
            and not _spatial_quality_codes_valid(quality_codes)):
        errors.append("SPATIAL_QUALITY_CODE_CONTRACT_INVALID")
    for tif in tifs:
        with rasterio.open(tif) as source:
            profile = {"crs": str(source.crs), "transform": source.transform,
                       "width": source.width, "height": source.height,
                       "resolution": source.res, "bounds": source.bounds}
            tags = source.tags()
            profiles[tif.name] = profile
            grids.add((profile["crs"], tuple(profile["transform"]), profile["width"], profile["height"], profile["resolution"]))
            data = source.read(1)
            if product_schema in TIF_METADATA_PRODUCT_SCHEMAS:
                if (tags.get("PRODUCT_SCHEMA_VERSION") != manifest_product_schema
                        or not tags.get("PRODUCT_KIND")):
                    errors.append(f"TIF_METADATA_CONTRACT_MISSING:{tif.name}")
                if product_schema in {"xian_r5.grounded_products.v9",
                                      "xian_r5.grounded_products.v10",
                                      "xian_r5.grounded_products.v11",
                                      "xian_r5.grounded_products.v12",
                                      "xian_r5.grounded_products.v13"}:
                    expected_component_provenance = manifest.get(
                        "tif_metadata_contract", {}).get(
                            "base_grid_pixel_provenance_status")
                    if (not expected_component_provenance
                            or tags.get("BASE_GRID_PIXEL_PROVENANCE_STATUS")
                            != expected_component_provenance):
                        errors.append(f"TIF_BASE_GRID_PROVENANCE_TAG_MISMATCH:{tif.name}")
                if (tags.get("PRODUCT_CRS") != profile["crs"]
                        or not math.isclose(float(tags.get("PIXEL_SIZE_X_M", "nan")),
                                            float(source.res[0]), abs_tol=1e-8)
                        or not math.isclose(float(tags.get("PIXEL_SIZE_Y_M", "nan")),
                                            float(source.res[1]), abs_tol=1e-8)):
                    errors.append(f"TIF_METADATA_GRID_MISMATCH:{tif.name}")
                if actual_date_schema and "OUTPUT_GROUP_ID" in tags:
                    group_id = tags["OUTPUT_GROUP_ID"]
                    group = manifest.get("output_group_metadata", {}).get(group_id)
                    try:
                        tagged_dates = json.loads(tags.get("ACTUAL_DATES_JSON", "null"))
                        tagged_labels = json.loads(tags.get("EPOCH_LABELS_JSON", "null"))
                        tagged_raw_dates = json.loads(tags.get("ACTUAL_DATE_RAW_VALUES_JSON", "null"))
                    except json.JSONDecodeError:
                        tagged_dates = tagged_labels = tagged_raw_dates = None
                    if group is None:
                        errors.append(f"TIF_OUTPUT_GROUP_UNKNOWN:{tif.name}")
                    else:
                        expected_dates = [group_id] if group.get("actual_date") else []
                        if (not tif.name.startswith(f"{group_id}_")
                                or tags.get("ACTUAL_DATE") != (group.get("actual_date") or "UNKNOWN")
                                or tags.get("ACTUAL_DATE_STATUS") != group.get("actual_date_status")
                                or tagged_dates != expected_dates
                                or tagged_labels != group.get("epoch_labels")
                                or tagged_raw_dates != group.get("unresolved_actual_date_values")
                                or tags.get("ACTUAL_DATE_SOURCE") != "TILE_ACTUAL_DATE_REGISTRY"
                                or (group.get("actual_date") is None
                                    and group_id in manifest.get("actual_date_order", []))):
                            errors.append(f"TIF_ACTUAL_DATE_GROUP_TAG_MISMATCH:{tif.name}")
                elif actual_date_schema and "FROM_OUTPUT_GROUP" in tags and "TO_OUTPUT_GROUP" in tags:
                    left, right = tags["FROM_OUTPUT_GROUP"], tags["TO_OUTPUT_GROUP"]
                    order = manifest.get("actual_date_order", [])
                    adjacent = any(order[index:index + 2] == [left, right]
                                   for index in range(max(0, len(order) - 1)))
                    if (not adjacent or not tif.name.startswith(f"{left}_to_{right}_")
                            or tags.get("FROM_ACTUAL_DATE") != left
                            or tags.get("TO_ACTUAL_DATE") != right
                            or tags.get("ACTUAL_DATE_ORDER_BASIS") != "UNIQUE_ACTUAL_ACQUISITION_DATES"
                            or tags.get("ACTUAL_DATE_SOURCE") != "TILE_ACTUAL_DATE_REGISTRY"):
                        errors.append(f"TIF_INTERVAL_DATE_GROUP_TAG_MISMATCH:{tif.name}")
                elif actual_date_schema and "CROSS_ACTUAL_DATES_JSON" in tags:
                    try:
                        tagged_cross_dates = json.loads(tags.get("CROSS_ACTUAL_DATES_JSON", "null"))
                    except json.JSONDecodeError:
                        tagged_cross_dates = None
                    if (tagged_cross_dates != manifest.get("actual_date_order")
                            or tags.get("ACTUAL_DATE_SOURCE") != "TILE_ACTUAL_DATE_REGISTRY"):
                        errors.append(f"TIF_CROSS_DATE_TAG_MISMATCH:{tif.name}")
                elif "EPOCH_LABEL" in tags:
                    epoch = tags["EPOCH_LABEL"]
                    try:
                        tagged_dates = json.loads(tags.get("ACTUAL_DATES_JSON", "null"))
                    except json.JSONDecodeError:
                        tagged_dates = None
                    if (epoch not in manifest.get("actual_dates_by_epoch", {})
                            or tagged_dates != manifest["actual_dates_by_epoch"].get(epoch)
                            or tags.get("ACTUAL_DATE_SOURCE") != "TILE_ACTUAL_DATE_REGISTRY"):
                        errors.append(f"TIF_ACTUAL_DATE_TAG_MISMATCH:{tif.name}")
                elif "FROM_EPOCH_LABEL" in tags and "TO_EPOCH_LABEL" in tags:
                    left, right = tags["FROM_EPOCH_LABEL"], tags["TO_EPOCH_LABEL"]
                    try:
                        left_dates = json.loads(tags.get("FROM_ACTUAL_DATES_JSON", "null"))
                        right_dates = json.loads(tags.get("TO_ACTUAL_DATES_JSON", "null"))
                    except json.JSONDecodeError:
                        left_dates = right_dates = None
                    if (left not in manifest.get("actual_dates_by_epoch", {})
                            or right not in manifest.get("actual_dates_by_epoch", {})
                            or left_dates != manifest["actual_dates_by_epoch"].get(left)
                            or right_dates != manifest["actual_dates_by_epoch"].get(right)
                            or tags.get("ACTUAL_DATE_SOURCE") != "TILE_ACTUAL_DATE_REGISTRY"):
                        errors.append(f"TIF_INTERVAL_DATE_TAG_MISMATCH:{tif.name}")
                elif "MOVING_ACTUAL_DATE" in tags:
                    available_dates = set(manifest.get("actual_date_order", [])) if actual_date_schema else {
                        value for values in manifest.get("actual_dates_by_epoch", {}).values()
                        for value in values
                    }
                    moving_date = tags["MOVING_ACTUAL_DATE"]
                    if (moving_date != "UNKNOWN" and moving_date not in available_dates):
                        errors.append(f"TIF_MOVING_DATE_TAG_MISMATCH:{tif.name}")
                else:
                    errors.append(f"TIF_TEMPORAL_TAG_MISSING:{tif.name}")
            state_layer = any(token in tif.name for token in ("_stable", "_addition", "_demolition"))
            is_spatial_quality = tif.name.endswith("_spatial_quality.tif")
            if ("ground_footprint" in tif.name or "_change" in tif.name
                    or state_layer or is_spatial_quality):
                codes[tif.name] = {str(int(value)): int(count) for value, count in zip(*np.unique(data, return_counts=True))}
            if product_schema in {"xian_r5.grounded_products.v12",
                                  "xian_r5.grounded_products.v13"} and is_spatial_quality:
                expected_quality_tags = {
                    str(int(code)): label.upper()
                    for label, code in quality_codes.items()
                }
                try:
                    tagged_quality_codes = json.loads(tags.get("VALUE_CODES_JSON", "null"))
                except json.JSONDecodeError:
                    tagged_quality_codes = None
                if (tags.get("PRODUCT_KIND") != "SPATIAL_QUALITY"
                        or tagged_quality_codes != expected_quality_tags
                        or source.nodata is not None):
                    errors.append(f"SPATIAL_QUALITY_TIF_CONTRACT_INVALID:{tif.name}")
                group_id = tif.name.removesuffix("_spatial_quality.tif")
                spatial_quality_by_group[group_id] = data
            elif (product_schema in {"xian_r5.grounded_products.v12",
                                    "xian_r5.grounded_products.v13"}
                  and tif.name.endswith("_ground_footprint.tif")):
                group_id = tif.name.removesuffix("_ground_footprint.tif")
                footprint_state_by_group[group_id] = data
            if "ground_footprint" in tif.name and source.nodata != footprint_codes.get("nodata"):
                errors.append(f"FOOTPRINT_NODATA_METADATA_MISMATCH:{tif.name}")
            if "_change" in tif.name and source.nodata != change_codes.get("nodata"):
                errors.append(f"CHANGE_NODATA_METADATA_MISMATCH:{tif.name}")
            if state_layer and source.nodata != change_codes.get("nodata"):
                errors.append(f"CHANGE_STATE_NODATA_METADATA_MISMATCH:{tif.name}")
            if "physical_id" in tif.name and source.nodata is not None:
                errors.append(f"IDENTITY_NODATA_MUST_NOT_MASK_UNKNOWN:{tif.name}")
    if product_schema in {"xian_r5.grounded_products.v12",
                          "xian_r5.grounded_products.v13"}:
        expected_groups = set(manifest.get("output_groups", []))
        if (set(spatial_quality_by_group) != expected_groups
                or set(footprint_state_by_group) != expected_groups):
            errors.append("SPATIAL_QUALITY_EPOCH_PRODUCTS_INCOMPLETE")
        for group_id in expected_groups & set(spatial_quality_by_group) & set(footprint_state_by_group):
            if not _spatial_quality_matches_footprint(
                    spatial_quality_by_group[group_id], footprint_state_by_group[group_id],
                    quality_codes, footprint_codes):
                errors.append(f"SPATIAL_QUALITY_FOOTPRINT_STATE_MISMATCH:{group_id}")
    if any(profile["crs"] != "EPSG:32649" for profile in profiles.values()):
        errors.append("CRS_NOT_EPSG32649")
    has_base_grid = bool(manifest.get("native_reference_grid_products_written"))
    epoch_product_count = 6 if product_schema in {
        "xian_r5.grounded_products.v6", "xian_r5.grounded_products.v7",
        "xian_r5.grounded_products.v8", "xian_r5.grounded_products.v9",
        "xian_r5.grounded_products.v10", "xian_r5.grounded_products.v11",
        "xian_r5.grounded_products.v12", "xian_r5.grounded_products.v13"} else 5
    interval_count = max(0, len(manifest.get("actual_date_order", [])) - 1) if actual_date_schema else max(0, len(epochs) - 1)
    temporal_group_count = len(manifest.get("actual_date_order", [])) if actual_date_schema else len(epochs)
    expected_tifs = len(epochs) * epoch_product_count + interval_count * 4
    alignment_field_count = len(manifest.get("image_alignment_quality_products", {}))
    roof_candidate_product_count = len(manifest.get("image_grid_roof_candidate_products", []))
    expected_tifs += alignment_field_count
    expected_tifs += 2 * roof_candidate_product_count
    if has_base_grid:
        base_epoch_product_count = 4 if product_schema in {
            "xian_r5.grounded_products.v6", "xian_r5.grounded_products.v7",
            "xian_r5.grounded_products.v8", "xian_r5.grounded_products.v9",
            "xian_r5.grounded_products.v10", "xian_r5.grounded_products.v11",
            "xian_r5.grounded_products.v12", "xian_r5.grounded_products.v13"} else 3
        expected_tifs += len(epochs) * base_epoch_product_count + interval_count * 4
        expected_tifs += alignment_field_count
        expected_tifs += 2 * roof_candidate_product_count
    if manifest_product_schema in {"xian_r5.grounded_products.v15",
                                   "xian_r5.grounded_products.v16",
                                   "xian_r5.grounded_products.v17"}:
        expected_tifs += 2 * len(epochs)
        if has_base_grid:
            expected_tifs += 2 * len(epochs)
    if manifest_product_schema == "xian_r5.grounded_products.v17":
        # Five explicit geometry/presence/registration rasters per actual-date
        # group, three cross-date canonical layers, and five non-ground
        # diagnostic products for each registered and canonical transition.
        expected_tifs += 5 * len(epochs) + 3 + 10 * interval_count
        if has_base_grid:
            expected_tifs += 5 * len(epochs) + 3 + 10 * interval_count
    if manifest.get("registration_status_contract"):
        expected_tifs += len(epochs) * (2 if has_base_grid else 1)
        for group in epochs:
            for suffix in ("", "_BASE10M") if has_base_grid else ("",):
                if f"{group}_registration_status{suffix}.tif" not in profiles:
                    errors.append(f"REGISTRATION_STATUS_MISSING:{group}{suffix}")
    if len(tifs) != expected_tifs:
        errors.append("MISSING_OR_EXTRA_TIF")
    fine = [p for name, p in profiles.items() if "_BASE10M" not in name]
    native = [p for name, p in profiles.items() if "_BASE10M" in name]
    if fine:
        if len({(p["crs"], tuple(p["transform"]), p["width"], p["height"]) for p in fine}) != 1:
            errors.append("FINE_TIF_GRID_MISMATCH")
        if has_base_grid and (len(native) != expected_tifs - len(fine) or
                              len({(p["crs"], tuple(p["transform"]), p["width"], p["height"]) for p in native}) != 1):
            errors.append("NATIVE_BASE_TIF_GRID_MISMATCH")
        if has_base_grid and native:
            fp, bp = fine[0], native[0]
            factor = int(manifest["grid"]["fine_pixels_per_reference_pixel"])
            if (fp["bounds"] != bp["bounds"] or fp["crs"] != bp["crs"] or
                    fp["transform"].c != bp["transform"].c or fp["transform"].f != bp["transform"].f or
                    fp["width"] != bp["width"] * factor or fp["height"] != bp["height"] * factor or
                    abs(fp["resolution"][0] * factor - bp["resolution"][0]) > 1e-8):
                errors.append("FINE_NATIVE_GRID_NESTING_FAILED")
            reference = manifest["reference_grid"]
            with rasterio.open(reference["path"]) as base:
                if list(base.transform) != reference["transform"] or str(base.crs) != reference["crs"]:
                    errors.append("REFERENCE_GRID_RECEIPT_MISMATCH")
                base_window = reference["base_window"]
                expected_c = base.transform.c + base_window[0] * base.res[0]
                expected_f = base.transform.f - base_window[1] * base.res[1]
                if abs(bp["transform"].c - expected_c) > 1e-8 or abs(bp["transform"].f - expected_f) > 1e-8:
                    errors.append("NATIVE_GRID_NOT_ANCHORED_TO_BASE")
    if receipt["input_sha256"]["r4_status_before"] != receipt["input_sha256"]["r4_status_after"]:
        errors.append("R4_STATUS_CHANGED")
    input_hashes = receipt.get("input_sha256", {})
    input_paths = receipt.get("input_paths", {})
    tracked_input_hashes_unchanged = True
    for name in ("fixed_case_map", "fixed_case_geometry", "actual_date_registry"):
        before, after = input_hashes.get(f"{name}_before"), input_hashes.get(f"{name}_after")
        if before is not None and after is not None and before != after:
            errors.append(f"INPUT_CHANGED:{name}")
            tracked_input_hashes_unchanged = False
        path_value = input_paths.get(name)
        expected = input_hashes.get(name)
        if path_value and expected:
            path = Path(path_value)
            if not path.is_file() or _sha256(path) != expected:
                errors.append(f"INPUT_LIVE_HASH_MISMATCH:{name}")
                tracked_input_hashes_unchanged = False
    r4_path_value = input_paths.get("r4_status")
    r4_expected = input_hashes.get("r4_status_after")
    if r4_path_value and r4_expected:
        r4_path = Path(r4_path_value)
        if not r4_path.is_file() or _sha256(r4_path) != r4_expected:
            errors.append("INPUT_LIVE_HASH_MISMATCH:r4_status")
            tracked_input_hashes_unchanged = False
    pixel_hashes_unchanged = True
    for item in receipt.get("pixel_inputs", []):
        before = item.get("sha256_before")
        after = item.get("sha256_after", item.get("sha256"))
        path = Path(item["path"])
        if (before is not None and before != after) or (not path.is_file() or _sha256(path) != after):
            errors.append(f"PIXEL_INPUT_HASH_MISMATCH:{path.name}")
            pixel_hashes_unchanged = False
    base_hashes_unchanged = True
    for item in receipt.get("10m_base_stack", {}).get("rasters", []):
        path = Path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            errors.append(f"BASE_RASTER_HASH_MISMATCH:{path.name}")
            base_hashes_unchanged = False
    for obs in receipt["observations"]:
        if obs["footprint_status"] != "GROUND_VERIFIED" and obs["physical_id"] is not None:
            errors.append(f"UNVERIFIED_PHYSICAL_ID:{obs['uid']}")
    if receipt["ground_footprints_confirmed"] == 0:
        for name, histogram in codes.items():
            is_candidate_footprint = "ground_footprint_candidates" in name
            if "ground_footprint" in name and not is_candidate_footprint and "1" in histogram:
                errors.append(f"ROOF_LAUNDERED_AS_GROUND:{name}")
            if (temporal_group_count > 1 and "_change" in name
                    and any(value in histogram for value in ("1", "2", "3"))):
                errors.append(f"UNCONFIRMED_CHANGE:{name}")
        if not any("ground_footprint" in name and "ground_footprint_candidates" not in name
                   and str(footprint_codes["unresolved"]) in histogram
                   for name, histogram in codes.items()):
            errors.append("UNRESOLVED_FOOTPRINT_CODE_MISSING")
        if temporal_group_count > 1:
            if not any("_change" in name and str(change_codes["UNRESOLVED"]) in histogram
                       for name, histogram in codes.items()):
                errors.append("UNRESOLVED_CHANGE_CODE_MISSING")
            if not any(any(token in name for token in ("_stable", "_addition", "_demolition"))
                       and str(change_codes["UNRESOLVED"]) in histogram
                       for name, histogram in codes.items()):
                errors.append("UNRESOLVED_STATE_LAYER_CODE_MISSING")
        unknown_id = int(id_codes["unresolved"])
        unknown_id_seen = False
        for tif in tifs:
            if "physical_id" not in tif.name:
                continue
            with rasterio.open(tif) as source:
                if np.any(source.read(1) == unknown_id):
                    unknown_id_seen = True
                    break
        if not unknown_id_seen:
            errors.append("UNRESOLVED_ID_CODE_MISSING")
    with _open_vector(root / "building_lineage.gpkg", layer="roof_candidates") as source:
        roof_features = [feature for feature in source]
    if len(roof_features) != len(receipt["observations"]):
        errors.append("ROOF_LINEAGE_COUNT_MISMATCH")
    with _open_vector(root / "building_lineage.gpkg", layer="ground_footprints") as source:
        ground_features = [feature for feature in source if feature.get("geometry")]
        if len(ground_features) != receipt["ground_footprints_confirmed"]:
            errors.append("GROUND_LINEAGE_COUNT_MISMATCH")
    raw_roofs = {feature["properties"]["uid"]: shape(feature["geometry"])
                 for feature in roof_features if feature.get("geometry")}
    raw_roof_sources = {feature["properties"]["uid"]: feature["properties"].get("source_ref")
                        for feature in roof_features if feature.get("geometry")}
    ground_geometries = {feature["properties"]["uid"]: shape(feature["geometry"])
                         for feature in ground_features}
    observations_by_uid = {str(row["uid"]): row for row in receipt["observations"]}
    alignment_cells = {}
    with _open_vector(root / "building_lineage.gpkg", layer="image_alignment_candidates") as source:
        for feature in source:
            props = dict(feature["properties"])
            key = (str(props.get("field_id")), str(props.get("cell_id")))
            if key in alignment_cells:
                errors.append(f"DUPLICATE_IMAGE_ALIGNMENT_CELL:{key[0]}:{key[1]}")
            alignment_cells[key] = (props, shape(feature["geometry"]))
    registered_roofs = {}
    registered_rows = {}
    with _open_vector(root / "building_lineage.gpkg", layer="registered_roof_candidates") as source:
        for feature in source:
            props = dict(feature["properties"])
            uid = str(props.get("uid"))
            if uid in registered_roofs:
                errors.append(f"DUPLICATE_REGISTERED_ROOF:{uid}")
            registered_rows[uid] = props
            registered_roofs[uid] = shape(feature["geometry"])
    alignment_applications = receipt.get("image_grid_alignment_applications", [])
    application_by_uid = {str(row.get("uid")): row for row in alignment_applications}
    if len(application_by_uid) != len(alignment_applications):
        errors.append("DUPLICATE_IMAGE_ALIGNMENT_APPLICATION_UID")
    if set(application_by_uid) != set(observations_by_uid):
        errors.append("IMAGE_ALIGNMENT_APPLICATION_OBSERVATION_SET_MISMATCH")
    applied_count = 0
    for uid, observation in observations_by_uid.items():
        application = application_by_uid.get(uid)
        if application is None:
            continue
        applied = application.get("applied") is True
        applied_count += int(applied)
        if bool(observation.get("image_alignment_applied")) != applied:
            errors.append(f"IMAGE_ALIGNMENT_APPLICATION_RECEIPT_MISMATCH:{uid}")
        if applied != (uid in registered_roofs):
            errors.append(f"REGISTERED_ROOF_COUNT_MISMATCH:{uid}")
        if not applied:
            continue
        props = registered_rows[uid]
        field_id, cell_id = str(props.get("field_id")), str(props.get("cell_id"))
        cell_entry = alignment_cells.get((field_id, cell_id))
        raw = raw_roofs.get(uid)
        if cell_entry is None or raw is None:
            errors.append(f"REGISTERED_ROOF_SOURCE_MISSING:{uid}")
            continue
        cell_props, support_geometry = cell_entry
        dx, dy = props.get("correction_dx_m"), props.get("correction_dy_m")
        basis = props.get("alignment_review_basis") or "UNREVIEWED"
        sigma = props.get("position_unc_m")
        relative_sigma = props.get("relative_unc_m")
        app_dx, app_dy = application.get("correction_dx_m"), application.get("correction_dy_m")
        if any(value is None or not math.isfinite(float(value))
               for value in (dx, dy, app_dx, app_dy)):
            errors.append(f"REGISTERED_ROOF_METRICS_INCOMPLETE:{uid}")
            continue
        relative_review = basis == "RELATIVE_REFERENCE_IMAGE_FEATURES"
        if relative_review:
            count_values = (props.get("nonbuilding_tie_pairs"), props.get("holdout_tie_pairs"))
            if (sigma is not None or not isinstance(relative_sigma, (int, float))
                    or isinstance(relative_sigma, bool)
                    or not math.isfinite(float(relative_sigma))
                    or not 0 <= float(relative_sigma) <= 2.0
                    or props.get("independent_ground") != 0
                    or props.get("ground_evidence_type") not in (None, "")
                    or props.get("alignment_evidence_type")
                    != "STABLE_NONBUILDING_IMAGE_FEATURE_TIES"
                     or any(not isinstance(value, int) or isinstance(value, bool) or value < 3
                            for value in count_values)):
                errors.append(f"REGISTERED_ROOF_RELATIVE_UNCERTAINTY_INVALID:{uid}")
                continue
        elif (basis not in ("INDEPENDENT_GROUND_CONTROL", "UNREVIEWED")
              or not isinstance(sigma, (int, float)) or isinstance(sigma, bool)
              or not math.isfinite(float(sigma)) or not 0 <= float(sigma) <= 2.0
              or relative_sigma is not None):
            errors.append(f"REGISTERED_ROOF_ABSOLUTE_UNCERTAINTY_INVALID:{uid}")
            continue
        cell_dx, cell_dy = cell_props.get("correction_dx_m"), cell_props.get("correction_dy_m")
        if any(value is None or not math.isfinite(float(value)) for value in (cell_dx, cell_dy)):
            errors.append(f"REGISTERED_ROOF_CELL_METRICS_INCOMPLETE:{uid}")
            continue
        spatial_relative_scope_exception = (
            _valid_unknown_scope_spatial_relative_review(
                cell_props, manifest_product_schema)
            and props.get("relative_cell_scope_review_authorized") == 1
            and props.get("relative_ties_within_support_cell") == 1
            and props.get("source_component_scope") == "SOURCE_COMPONENT_SCOPE_UNKNOWN"
            and props.get("reference_component_id_status")
                == "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT")
        component_gate_failed = (
            product_schema in {"xian_r5.grounded_products.v11",
                               "xian_r5.grounded_products.v12",
                               "xian_r5.grounded_products.v13"}
            and (cell_props.get("source_component_scope")
                 != "SINGLE_SOURCE_COMPONENT_PER_CELL"
                 or not cell_props.get("source_component_ids_json"))
            and not spatial_relative_scope_exception)
        if relative_review:
            cell_relative_sigma = cell_props.get("relative_unc_m")
            component_gate_failed = component_gate_failed or (
                cell_props.get("alignment_review_basis") != basis
                or cell_props.get("position_unc_m") is not None
                or not isinstance(cell_relative_sigma, (int, float))
                or isinstance(cell_relative_sigma, bool)
                or not math.isclose(float(cell_relative_sigma), float(relative_sigma), abs_tol=1e-5)
                or cell_props.get("alignment_evidence_type")
                != "STABLE_NONBUILDING_IMAGE_FEATURE_TIES"
                or cell_props.get("independent_ground") != 0
                or (manifest_product_schema in {"xian_r5.grounded_products.v16",
                                                "xian_r5.grounded_products.v17"}
                    and (cell_props.get("relative_ties_within_support_cell") != 1
                         or props.get("relative_ties_within_support_cell") != 1))
                or (cell_props.get("source_component_scope")
                    == "SOURCE_COMPONENT_SCOPE_UNKNOWN"
                    and not spatial_relative_scope_exception))
        else:
            cell_absolute_sigma = cell_props.get("position_unc_m")
            component_gate_failed = component_gate_failed or (
                (cell_props.get("alignment_review_basis") is not None
                 and cell_props.get("alignment_review_basis") not in
                 ("INDEPENDENT_GROUND_CONTROL", "UNREVIEWED"))
                or not isinstance(cell_absolute_sigma, (int, float))
                or isinstance(cell_absolute_sigma, bool)
                or not math.isclose(float(cell_absolute_sigma), float(sigma), abs_tol=1e-5)
                or cell_props.get("relative_unc_m") is not None)
        if (cell_props.get("accepted") != 1 or cell_props.get("reviewed") != 1
                or cell_props.get("independent_ground") != (0 if relative_review else 1)
                or (not relative_review and float(sigma) > 2.0)
                or cell_props.get("applied") != 1
                or component_gate_failed
                or not support_geometry.covers(raw)):
            errors.append(f"REGISTERED_ROOF_REVIEW_GATE_FAILED:{uid}")
        if (props.get("actual_date") != observation.get("actual_date")
                or props.get("status") != "ACCEPTED_IMAGE_GRID_SHIFT_ROOF_NOT_GROUND"
                or (product_schema == "xian_r5.grounded_products.v13"
                    and application.get("review_basis") is not None
                    and basis != application.get("review_basis"))
                or (relative_review and (
                    application.get("positional_uncertainty_m") is not None
                    or application.get("relative_registration_uncertainty_m") is None
                    or not math.isclose(
                        float(application.get("relative_registration_uncertainty_m")),
                        float(relative_sigma), abs_tol=1e-5)))
                or (not relative_review and (
                    application.get("relative_registration_uncertainty_m") is not None
                    or application.get("positional_uncertainty_m") is None
                    or not math.isclose(float(application.get("positional_uncertainty_m")),
                                        float(sigma), abs_tol=1e-5)))
                or props.get("alignment_source_ref") != cell_props.get("source_ref")
                or props.get("field_id") != application.get("field_id")
                or props.get("cell_id") != application.get("cell_id")
                or props.get("field_id") != observation.get("image_alignment_field_id")
                or props.get("cell_id") != observation.get("image_alignment_cell_id")
                or not math.isclose(float(dx), float(cell_dx), abs_tol=1e-5)
                or not math.isclose(float(dy), float(cell_dy), abs_tol=1e-5)
                or not math.isclose(float(dx), float(app_dx), abs_tol=1e-5)
                or not math.isclose(float(dy), float(app_dy), abs_tol=1e-5)):
            errors.append(f"REGISTERED_ROOF_CELL_PROVENANCE_MISMATCH:{uid}")
        lineage_fields = (
                ("reviewer_id", "reviewer_id"), ("reviewed_at", "reviewed_at"),
                ("review_evidence_path", "review_evidence_path"),
                ("review_evidence_sha256", "review_evidence_sha256"),
                ("reference_sha256", "reference_sha256"))
        if not relative_review:
            lineage_fields = (("ground_evidence_type", "ground_evidence_type"),
                              *lineage_fields)
        for registered_name, cell_name in lineage_fields:
            if not props.get(registered_name) or props.get(registered_name) != cell_props.get(cell_name):
                errors.append(f"REGISTERED_ROOF_REVIEW_LINEAGE_MISMATCH:{uid}:{registered_name}")
        expected_geometry = translate(raw, xoff=float(dx), yoff=float(dy))
        if registered_roofs[uid].hausdorff_distance(expected_geometry) > 1e-7:
            errors.append(f"REGISTERED_ROOF_TRANSLATION_MISMATCH:{uid}")
        if (props.get("geometry_basis") != "IMAGE_GRID_ALIGNED_ROOF_PROXY_NOT_GROUND"
                or props.get("roof_source_ref") != observation.get("roof_source")):
            errors.append(f"REGISTERED_ROOF_BASIS_INVALID:{uid}")
    if (applied_count != manifest.get("registered_roof_candidate_count")
            or bool(applied_count) != bool(manifest.get("image_alignment_field_applied"))):
        errors.append("IMAGE_ALIGNMENT_MANIFEST_APPLICATION_MISMATCH")
    for field_record in manifest.get("image_alignment_fields", []):
        field_id = str(field_record.get("field_id"))
        field_rows = [props for (row_field, _), (props, _) in alignment_cells.items()
                      if row_field == field_id]
        if (sum(props.get("accepted") == 1 for props in field_rows)
                != field_record.get("accepted_cell_count")
                or sum(props.get("applied") == 1 for props in field_rows)
                != field_record.get("applied_cell_count")):
            errors.append(f"IMAGE_ALIGNMENT_FIELD_COUNTS_MISMATCH:{field_id}")
    if applied_count:
        reviewed_alignment = receipt.get("reviewed_image_alignment_source")
        if (not reviewed_alignment
                or reviewed_alignment.get("run_end_hash_recheck") != "PASS"
                or reviewed_alignment.get("field_application_performed") is not False):
            errors.append("REVIEWED_IMAGE_ALIGNMENT_PROVENANCE_MISSING")
        else:
            reviewed_rows = reviewed_alignment.get("reviewed_cells", [])
            reviewed_by_key = {(str(row.get("field_id")), str(row.get("cell_id"))): row
                               for row in reviewed_rows}
            if (len(reviewed_by_key) != len(reviewed_rows)
                    or len(reviewed_rows) != reviewed_alignment.get("accepted_cell_count")
                    or reviewed_alignment.get("run_end_integrity_files_sha256")
                    != reviewed_alignment.get("integrity_files_sha256")):
                errors.append("REVIEWED_IMAGE_ALIGNMENT_RECEIPT_INTEGRITY_MISMATCH")
            for uid, props in registered_rows.items():
                key = (str(props.get("field_id")), str(props.get("cell_id")))
                reviewed = reviewed_by_key.get(key)
                if reviewed is None:
                    errors.append(f"REGISTERED_ROOF_REVIEW_RECORD_MISSING:{uid}")
                    continue
                relative_review = (
                    props.get("alignment_review_basis") == "RELATIVE_REFERENCE_IMAGE_FEATURES")
                review_metrics = [reviewed.get("correction_dx_m"),
                                  reviewed.get("correction_dy_m"),
                                  props.get("correction_dx_m"),
                                  props.get("correction_dy_m")]
                if relative_review:
                    review_metrics.extend((reviewed.get("relative_registration_uncertainty_m"),
                                           props.get("relative_unc_m")))
                    uncertainty_invalid = (
                        reviewed.get("positional_uncertainty_m") is not None
                        or props.get("position_unc_m") is not None
                        or reviewed.get("independent_ground_evidence") is not False
                        or reviewed.get("review_basis") != "RELATIVE_REFERENCE_IMAGE_FEATURES"
                        or reviewed.get("alignment_evidence_type")
                        != "STABLE_NONBUILDING_IMAGE_FEATURE_TIES"
                        or reviewed.get("ground_evidence_type") not in (None, "")
                        or not isinstance(reviewed.get("nonbuilding_tie_pairs"), int)
                        or isinstance(reviewed.get("nonbuilding_tie_pairs"), bool)
                        or reviewed.get("nonbuilding_tie_pairs", 0) < 3
                        or not isinstance(reviewed.get("holdout_tie_pairs"), int)
                        or isinstance(reviewed.get("holdout_tie_pairs"), bool)
                        or reviewed.get("holdout_tie_pairs", 0) < 3
                        or reviewed.get("relative_registration_uncertainty_m")
                        != props.get("relative_unc_m"))
                else:
                    review_metrics.extend((reviewed.get("positional_uncertainty_m"),
                                           props.get("position_unc_m")))
                    uncertainty_invalid = (
                        reviewed.get("review_basis") not in
                        (None, "INDEPENDENT_GROUND_CONTROL")
                        or reviewed.get("positional_uncertainty_m")
                        != props.get("position_unc_m"))
                if (any(value is None or not math.isfinite(float(value))
                        for value in review_metrics) or uncertainty_invalid):
                    errors.append(f"REGISTERED_ROOF_REVIEW_METRICS_INVALID:{uid}")
                    continue
                if (reviewed.get("moving_date") != props.get("actual_date")
                        or reviewed.get("candidate_source_ref") != props.get("alignment_source_ref")
                        or (not relative_review
                            and reviewed.get("ground_evidence_type")
                            != props.get("ground_evidence_type"))
                        or (not relative_review and props.get("independent_ground") != 1)
                        or (relative_review and props.get("independent_ground") != 0)
                        or (reviewed.get("review_basis") is not None
                            and reviewed.get("review_basis")
                            != props.get("alignment_review_basis"))
                        or (reviewed.get("alignment_evidence_type") is not None
                            and reviewed.get("alignment_evidence_type")
                            != props.get("alignment_evidence_type"))
                        or reviewed.get("reviewer_id") != props.get("reviewer_id")
                        or reviewed.get("reviewed_at") != props.get("reviewed_at")
                        or reviewed.get("evidence_path") != props.get("review_evidence_path")
                        or reviewed.get("evidence_sha256") != props.get("review_evidence_sha256")
                        or reviewed.get("reference_source") != props.get("reference_source")
                        or not math.isclose(float(reviewed["correction_dx_m"]),
                                            float(props["correction_dx_m"]), abs_tol=1e-5)
                        or not math.isclose(float(reviewed["correction_dy_m"]),
                                            float(props["correction_dy_m"]), abs_tol=1e-5)
                        or (relative_review and not math.isclose(
                            float(reviewed["relative_registration_uncertainty_m"]),
                            float(props["relative_unc_m"]), abs_tol=1e-5))
                        or (not relative_review and not math.isclose(
                            float(reviewed["positional_uncertainty_m"]),
                            float(props["position_unc_m"]), abs_tol=1e-5))):
                    errors.append(f"REGISTERED_ROOF_REVIEW_RECORD_MISMATCH:{uid}")
    image_roof_geometries = {**raw_roofs, **registered_roofs}
    identity_nodes = {}
    with _open_vector(root / "building_lineage.gpkg", layer="identity_graph_nodes") as source:
        node_schema_fields = set(source.schema["properties"])
        node_features = [feature for feature in source if feature.get("geometry")]
    graph_rows = {str(row["uid"]): row for row in receipt.get("identity_graph", [])}
    if len(node_features) != len(graph_rows) or len(node_features) != manifest.get("identity_graph_node_count"):
        errors.append("IDENTITY_GRAPH_NODE_COUNT_MISMATCH")
    if "physical_id" not in node_schema_fields:
        errors.append("IDENTITY_GRAPH_NODE_ID_FIELD_MISSING")
    for feature in node_features:
        props = dict(feature["properties"])
        uid = str(props.get("uid"))
        identity_nodes[uid] = shape(feature["geometry"])
        expected = graph_rows.get(uid)
        if expected is None:
            errors.append(f"IDENTITY_GRAPH_NODE_UNKNOWN_UID:{uid}")
            continue
        if props.get("status") != expected.get("status"):
            errors.append(f"IDENTITY_GRAPH_NODE_STATUS_MISMATCH:{uid}")
        physical_id = props.get("physical_id")
        if physical_id is not None and int(physical_id) != expected.get("physical_id"):
            errors.append(f"IDENTITY_GRAPH_NODE_ID_MISMATCH:{uid}")
        if expected.get("physical_id") is None and physical_id is not None:
            errors.append(f"UNRESOLVED_NODE_HAS_PHYSICAL_ID:{uid}")
        geometry_basis = props.get("geometry_basis")
        expected_geometry = (ground_geometries.get(uid) if geometry_basis == "GROUND_VERIFIED"
                             else registered_roofs.get(uid) if geometry_basis == "IMAGE_GRID_ROOF"
                             else raw_roofs.get(uid) if geometry_basis == "ROOF_PROXY"
                             else None)
        if expected_geometry is None or identity_nodes[uid].hausdorff_distance(expected_geometry) > 1e-7:
            errors.append(f"IDENTITY_GRAPH_NODE_GEOMETRY_MISMATCH:{uid}")
    candidate_edges = receipt.get("candidate_edges", [])
    graph_products = receipt.get("identity_graph_products", {})
    if (graph_products.get("node_layer") != "building_lineage.gpkg#identity_graph_nodes"
            or graph_products.get("candidate_edge_layer") != "building_lineage.gpkg#candidate_identity_edges"
            or graph_products.get("candidate_edges_are_confirmed_physical_ids") is not False):
        errors.append("IDENTITY_GRAPH_PRODUCT_LINEAGE_INVALID")
    if manifest_product_schema == "xian_r5.grounded_products.v17":
        presence_rows = receipt.get("presence")
        presence_codes = manifest.get("presence_table_codes")
        expected_counts = {
            state: sum(row.get("presence") == state for row in presence_rows)
            for state in ("1", "0", "U")
        } if isinstance(presence_rows, list) else {}
        if (not isinstance(presence_rows, list)
                or graph_products.get("presence_state_layer")
                != "building_lineage.gpkg#building_presence"
                or manifest.get("presence_states_layer")
                != "building_lineage.gpkg#building_presence"
                or presence_codes != {"present": "1", "absent": "0", "unresolved": "U"}
                or manifest.get("presence_state_counts") != expected_counts
                or manifest.get("presence_state_semantics") != (
                    "ONE_ROW_PER_OBSERVATION_OR_ACCEPTED_REVIEWED_ABSENCE;"
                    "ABSENCE_REQUIRES_FULL_FOOTPRINT_COVERAGE;UNKNOWN_IS_NOT_ABSENCE")):
            errors.append("PRESENCE_STATE_MANIFEST_INVALID")
        else:
            with _open_vector(root / "building_lineage.gpkg", layer="building_presence") as source:
                presence_fields = set(source.schema["properties"])
                presence_features = [feature for feature in source]
            required_presence_fields = {
                "observation_id", "epoch_label", "actual_date", "physical_id",
                "identity_candidate", "presence_state", "presence_evidence_type",
                "presence_evidence_ref", "coverage_complete", "review_status",
                "confidence", "notes", "confirmed_identity_key", "identity_reviewed",
                "identity_evidence_type", "identity_evidence_ref", "identity_evidence_path",
                "identity_evidence_sha256", "identity_reviewer_id", "identity_reviewed_at",
                "footprint_source_ref", "footprint_method", "footprint_sigma_m",
                "image_ref", "reviewed", "reviewer_id", "reviewed_at",
                "evidence_path", "evidence_sha256",
            }
            if (not required_presence_fields.issubset(presence_fields)
                    or len(presence_features) != len(presence_rows)):
                errors.append("PRESENCE_STATE_LAYER_SCHEMA_OR_COUNT_INVALID")
            else:
                for feature, expected in zip(presence_features, presence_rows):
                    props = dict(feature["properties"])
                    pairs = (("observation_id", "uid"), ("epoch_label", "epoch_label"),
                             ("actual_date", "actual_date"), ("presence_state", "presence"),
                             ("identity_candidate", "candidate_identity"),
                             ("confirmed_identity_key", "confirmed_identity_key"),
                             ("identity_evidence_type", "identity_evidence_type"),
                             ("identity_evidence_ref", "identity_evidence_ref"),
                             ("identity_evidence_path", "identity_evidence_path"),
                             ("identity_evidence_sha256", "identity_evidence_sha256"),
                             ("identity_reviewer_id", "identity_reviewer_id"),
                             ("identity_reviewed_at", "identity_reviewed_at"),
                             ("presence_evidence_type", "presence_evidence_type"),
                             ("presence_evidence_ref", "presence_evidence_ref"),
                             ("review_status", "review_status"), ("notes", "notes"),
                             ("footprint_source_ref", "footprint_source_ref"),
                             ("footprint_method", "footprint_method"),
                             ("image_ref", "image_ref"), ("reviewer_id", "reviewer_id"),
                             ("reviewed_at", "reviewed_at"), ("evidence_path", "evidence_path"),
                             ("evidence_sha256", "evidence_sha256"))
                    if any(props.get(target) != expected.get(source_name)
                           for target, source_name in pairs):
                        errors.append("PRESENCE_STATE_ROW_MISMATCH")
                        break
                    expected_id = expected.get("physical_id")
                    actual_id = props.get("physical_id")
                    if ((actual_id is None) != (expected_id is None)
                            or (actual_id is not None and int(actual_id) != int(expected_id))
                            or int(props.get("identity_reviewed") or 0)
                            != int(bool(expected.get("identity_reviewed")))):
                        errors.append("PRESENCE_STATE_IDENTITY_MISMATCH")
                        break
                    for target, source_name in (("coverage_complete", "coverage_complete"),
                                                ("reviewed", "reviewed")):
                        wanted = expected.get(source_name)
                        actual = props.get(target)
                        if ((actual is None) != (wanted is None)
                                or (actual is not None and int(actual) != int(bool(wanted)))):
                            errors.append("PRESENCE_STATE_REVIEW_LINEAGE_MISMATCH")
                            break
                    if errors and errors[-1] == "PRESENCE_STATE_REVIEW_LINEAGE_MISMATCH":
                        break
                    expected_sigma = expected.get("footprint_sigma_m")
                    actual_sigma = props.get("footprint_sigma_m")
                    if ((actual_sigma is None) != (expected_sigma is None)
                            or (actual_sigma is not None and not math.isclose(
                                float(actual_sigma), float(expected_sigma), abs_tol=1e-6))):
                        errors.append("PRESENCE_STATE_ACCURACY_MISMATCH")
                        break
        geometry_contract = manifest.get("geometry_layer_contract", {})
        if (geometry_contract.get("roof_raw") != "building_lineage.gpkg#roof_raw"
                or geometry_contract.get("roof_registered") != "building_lineage.gpkg#roof_registered"
                or geometry_contract.get("ground_footprints") != "building_lineage.gpkg#ground_footprints"
                or geometry_contract.get("canonical_building_geometry")
                != "building_lineage.gpkg#canonical_building_geometry"
                or geometry_contract.get("raw_roof_promoted_to_ground") is not False
                or geometry_contract.get("registered_roof_promoted_to_ground") is not False):
            errors.append("GEOMETRY_LAYER_CONTRACT_INVALID")
        else:
            with _open_vector(root / "building_lineage.gpkg", layer="roof_raw") as source:
                raw_fields = set(source.schema["properties"])
                raw_features = [feature for feature in source]
            raw_required = {"uid", "actual_date", "date_status", "source_ref", "image_path",
                            "source_tile_ids_json", "source_image_paths_json", "source_hashes_json",
                            "crs", "geometry_semantics"}
            receipt_observations = {row["uid"]: row for row in receipt.get("observations", [])}
            if (not raw_required.issubset(raw_fields)
                    or len(raw_features) != len(receipt_observations)
                    or len(raw_features) != manifest.get("roof_raw_count")):
                errors.append("RAW_ROOF_LAYER_SCHEMA_OR_COUNT_INVALID")
            else:
                for feature in raw_features:
                    props = dict(feature["properties"])
                    expected = receipt_observations.get(props.get("uid"))
                    try:
                        tile_ids = json.loads(props.get("source_tile_ids_json") or "[]")
                        image_paths = json.loads(props.get("source_image_paths_json") or "[]")
                        source_hashes = json.loads(props.get("source_hashes_json") or "[]")
                    except json.JSONDecodeError:
                        tile_ids = image_paths = source_hashes = None
                    if (expected is None or tile_ids != expected.get("source_tile_ids", [])
                            or image_paths != expected.get("source_image_paths", [])
                            or source_hashes != expected.get("source_hashes", [])
                            or len(tile_ids or []) != len(image_paths or [])
                            or len(tile_ids or []) != len(source_hashes or [])):
                        errors.append("RAW_ROOF_SOURCE_PROVENANCE_MISMATCH")
                        break
            with _open_vector(root / "building_lineage.gpkg", layer="roof_registered") as source:
                explicit_registered = {
                    feature["properties"]["uid"]: shape(feature["geometry"])
                    for feature in source if feature.get("geometry")
                }
            if (set(explicit_registered) != set(registered_roofs)
                    or any(explicit_registered[uid].hausdorff_distance(registered_roofs[uid]) > 1e-7
                           for uid in explicit_registered)):
                errors.append("REGISTERED_ROOF_LAYER_MISMATCH")
            canonical_manifest = manifest.get("canonical_geometry", {})
            with _open_vector(root / "building_lineage.gpkg", layer="canonical_building_geometry") as source:
                canonical_fields = set(source.schema["properties"])
                canonical_features = [feature for feature in source]
            required_canonical = {"physical_id", "canonical_uid", "canonical_date",
                                  "support_dates_json", "canonical_status", "geometry_basis",
                                  "ground_truth"}
            if (not required_canonical.issubset(canonical_fields)
                    or len(canonical_features) != canonical_manifest.get("row_count")
                    or sum(bool(feature.get("geometry")) for feature in canonical_features)
                    != canonical_manifest.get("resolved_count")
                    or canonical_manifest.get("resolved_count", 0)
                    + canonical_manifest.get("unresolved_count", 0)
                    != canonical_manifest.get("row_count")):
                errors.append("CANONICAL_GEOMETRY_LAYER_SCHEMA_OR_COUNT_INVALID")
            else:
                registered_uids = set(explicit_registered)
                physical_ids = {int(row["physical_id"]) for row in receipt.get("identity_graph", [])
                                if row.get("physical_id") is not None}
                for feature in canonical_features:
                    props = dict(feature["properties"])
                    if (int(props["physical_id"]) not in physical_ids
                            or int(props.get("ground_truth") or 0) != 0
                            or props.get("geometry_basis") != "REGISTERED_ROOF_MEDOID_NOT_GROUND"
                            or (props.get("canonical_uid") is not None
                                and props.get("canonical_uid") not in registered_uids)):
                        errors.append("CANONICAL_GEOMETRY_PROVENANCE_INVALID")
                        break
            with _open_vector(root / "building_lineage.gpkg", layer="identity_graph_nodes") as source:
                node_fields = set(source.schema["properties"])
            if not {"candidate_identity", "confirmed_identity_key", "identity_reviewed",
                    "physical_id", "geometry_status"}.issubset(node_fields):
                errors.append("IDENTITY_GRAPH_NODE_SCHEMA_INVALID")

        if manifest_product_schema == "xian_r5.grounded_products.v17":
            grid_path = root / manifest.get("grid_alignment_receipt", "GRID_ALIGNMENT_RECEIPT.json")
            if not grid_path.is_file():
                errors.append("GRID_ALIGNMENT_RECEIPT_MISSING")
            else:
                grid_receipt = json.loads(grid_path.read_text(encoding="utf-8"))
                fine_grid = grid_receipt.get("fine_grid", {})
                grid = manifest.get("grid", {})
                has_reference_grid = manifest.get("native_reference_grid_products_written") is True
                if (grid_receipt.get("crs") != "EPSG:32649"
                        or fine_grid.get("transform") != grid.get("transform")
                        or fine_grid.get("width") != grid.get("width")
                        or fine_grid.get("height") != grid.get("height")
                        or (has_reference_grid and (
                            grid_receipt.get("fine_to_10m_nesting_pass") is not True
                            or grid_receipt.get("fine_to_reference_alignment_pass") is not True
                            or manifest.get("grid_alignment_receipt_status", {}).get(
                                "fine_to_10m_nesting_pass") is not True
                            or manifest.get("grid_alignment_receipt_status", {}).get(
                                "fine_to_reference_alignment_pass") is not True))
                        or (not has_reference_grid and (
                            grid_receipt.get("nested_10m_grid") is not None
                            or grid_receipt.get("fine_to_10m_nesting_pass") is not None))):
                    errors.append("GRID_ALIGNMENT_RECEIPT_INVALID")
            for group_id in manifest.get("output_groups", []):
                for suffix in ("roof_raw", "roof_registered", "registration_quality",
                               "registration_uncertainty", "presence"):
                    filename = f"{group_id}_{suffix}.tif"
                    if filename not in profiles:
                        errors.append(f"GEOMETRY_STATE_RASTER_MISSING:{filename}")
                raster_path = root / f"{group_id}_presence.tif"
                if raster_path.is_file():
                    with rasterio.open(raster_path) as source:
                        raster = source.read(1)
                        values = set(np.unique(raster).tolist())
                        if not values.issubset({0, 1, 2, 3, 255}):
                            errors.append(f"PRESENCE_RASTER_CODE_INVALID:{group_id}")
            for filename in ("canonical_building_geometry.tif", "canonical_building_id.tif",
                             "canonical_geometry_quality.tif"):
                if filename not in profiles:
                    errors.append(f"CANONICAL_RASTER_MISSING:{filename}")
    with _open_vector(root / "building_lineage.gpkg", layer="candidate_identity_edges") as source:
        edge_schema_fields = set(source.schema["properties"])
        edge_features = [feature for feature in source if feature.get("geometry")]
    if len(edge_features) != len(candidate_edges) or len(edge_features) != manifest.get("candidate_identity_edge_count"):
        errors.append("CANDIDATE_IDENTITY_EDGE_COUNT_MISMATCH")
    if not {"edge_basis", "candidate_reference_json"}.issubset(edge_schema_fields):
        errors.append("CANDIDATE_EDGE_EVIDENCE_FIELDS_MISSING")
    if "physical_id" in edge_schema_fields or "physical_id" in {
            str(key).lower() for key in edge_schema_fields}:
        errors.append("CANDIDATE_EDGE_EXPOSES_CONFIRMED_PHYSICAL_ID")
    for feature, expected in zip(edge_features, candidate_edges):
        props = dict(feature["properties"])
        try:
            recorded_candidate_references = json.loads(
                props.get("candidate_reference_json") or "[]")
        except (TypeError, ValueError):
            recorded_candidate_references = None
        if (props.get("from_uid") != expected.get("from_uid")
                or props.get("to_uid") != expected.get("to_uid")
                or props.get("status") != expected.get("status")
                or props.get("relation") != expected.get("relation")
                or props.get("edge_basis") != expected.get("edge_basis", "TBIA_GEOMETRIC")
                or recorded_candidate_references != expected.get("candidate_references", [])
                or int(props.get("candidate_count") or 0) != int(expected.get("candidate_count") or 0)
                or abs(float(props.get("affinity") or 0.0) - float(expected.get("affinity") or 0.0)) > 1e-5):
            errors.append("CANDIDATE_IDENTITY_EDGE_ATTRIBUTE_MISMATCH")
            continue
        if expected.get("edge_basis") == "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE":
            references = expected.get("candidate_references", [])
            if (expected.get("status") != "CANDIDATE_ONLY"
                    or not expected.get("temporal_order_valid")
                    or expected.get("affinity") is not None
                    or not references
                    or any(item.get("verification_status") != "UNVERIFIED_CANDIDATE"
                           for item in references)):
                errors.append("UNVERIFIED_OUTLINE_EDGE_PROMOTED_OR_MALFORMED")
        geometry_basis = props.get("geometry_basis")
        geometry_lookup = ground_geometries if geometry_basis == "GROUND" else image_roof_geometries
        from_geometry = geometry_lookup.get(str(expected.get("from_uid")))
        to_uid = expected.get("to_uid")
        edge_geometry = shape(feature["geometry"])
        if from_geometry is None:
            errors.append("CANDIDATE_IDENTITY_EDGE_SOURCE_NODE_MISSING")
            continue
        left = from_geometry.centroid
        if to_uid is None:
            if edge_geometry.geom_type != "Point" or edge_geometry.distance(left) > 1e-7:
                errors.append("UNMATCHED_CANDIDATE_EDGE_GEOMETRY_MISMATCH")
        else:
            to_geometry = geometry_lookup.get(str(to_uid))
            if to_geometry is None:
                errors.append("CANDIDATE_IDENTITY_EDGE_TARGET_NODE_MISSING")
                continue
            right = to_geometry.centroid
            if left.equals(right):
                valid_geometry = edge_geometry.geom_type == "Point" and edge_geometry.distance(left) <= 1e-7
            else:
                valid_geometry = (edge_geometry.geom_type == "LineString"
                                  and edge_geometry.hausdorff_distance(
                                      type(edge_geometry)([(left.x, left.y), (right.x, right.y)])) <= 1e-7)
            if not valid_geometry:
                errors.append("CANDIDATE_IDENTITY_EDGE_GEOMETRY_MISMATCH")
        if expected.get("status") == "CANDIDATE_ONLY" and not expected.get("temporal_order_valid"):
            errors.append("INVALID_TEMPORAL_EDGE_MARKED_AS_CANDIDATE")
    if manifest.get("candidate_edges_promoted_to_physical_identity") is not False:
        errors.append("CANDIDATE_EDGES_PROMOTED_TO_PHYSICAL_ID")
    edge_basis_counts = {
        basis: sum(edge.get("edge_basis", "TBIA_GEOMETRIC") == basis
                   for edge in candidate_edges)
        for basis in sorted({edge.get("edge_basis", "TBIA_GEOMETRIC")
                             for edge in candidate_edges})}
    if manifest.get("candidate_identity_edge_basis_counts") != edge_basis_counts:
        errors.append("CANDIDATE_IDENTITY_EDGE_BASIS_COUNTS_MISMATCH")
    with _open_vector(root / "building_lineage.gpkg", layer="image_alignment_candidates") as source:
        alignment_rows = [dict(feature["properties"]) for feature in source]
        alignment_schema_fields = set(source.schema["properties"])
    support_fields = {"max_masked_n", "min_required_n", "sample_support"}
    if not support_fields.issubset(alignment_schema_fields):
        errors.append("ALIGNMENT_SAMPLE_SUPPORT_FIELDS_MISSING")
    source_component_fields = {
        "source_component_scope", "source_component_ids_json", "source_component_coverage"}
    if (product_schema in {"xian_r5.grounded_products.v11",
                           "xian_r5.grounded_products.v12",
                           "xian_r5.grounded_products.v13"}
            and not source_component_fields.issubset(alignment_schema_fields)):
        errors.append("ALIGNMENT_SOURCE_COMPONENT_FIELDS_MISSING")
    for row in alignment_rows:
        if (product_schema in {"xian_r5.grounded_products.v11",
                               "xian_r5.grounded_products.v12",
                               "xian_r5.grounded_products.v13"}
                and (int(row.get("accepted") or 0) == 1
                     or int(row.get("applied") or 0) == 1)):
            single_source_scoped = (
                row.get("source_component_scope") == "SINGLE_SOURCE_COMPONENT_PER_CELL"
                and bool(row.get("source_component_ids_json"))
                and row.get("source_component_coverage") is not None
                and float(row["source_component_coverage"]) >= 1.0 - 1e-8)
            spatial_relative_review = _valid_unknown_scope_spatial_relative_review(
                row, manifest_product_schema)
            if (not single_source_scoped and not spatial_relative_review):
                errors.append(f"COMPONENT_UNSCOPED_ALIGNMENT_ACCEPTED:{row.get('cell_id')}")
        support_status = row.get("sample_support")
        maximum, minimum = row.get("max_masked_n"), row.get("min_required_n")
        has_vector = row.get("correction_dx_m") is not None and row.get("correction_dy_m") is not None
        if support_status == "INSUFFICIENT_MASKED_SAMPLES":
            if maximum is None or minimum is None or int(maximum) >= int(minimum) or has_vector:
                errors.append(f"ALIGNMENT_SAMPLE_SUPPORT_REJECT_GATE_MISMATCH:{row.get('cell_id')}")
            if int(row.get("accepted") or 0) != 0 or int(row.get("applied") or 0) != 0:
                errors.append(f"SUPPORT_LIMITED_ALIGNMENT_PROMOTED:{row.get('cell_id')}")
        elif support_status == "NUMERIC_PEAK_AVAILABLE":
            if maximum is None or minimum is None or int(maximum) < int(minimum):
                errors.append(f"ALIGNMENT_SAMPLE_SUPPORT_PEAK_MISMATCH:{row.get('cell_id')}")
        elif support_status == "NO_VALID_CORRELATION_SCORE":
            if maximum is None or minimum is None or int(maximum) < int(minimum) or has_vector:
                errors.append(f"ALIGNMENT_SAMPLE_SUPPORT_SCORE_MISMATCH:{row.get('cell_id')}")
    multi_anchor_report = receipt.get("multi_anchor_base_tie_source_report")
    if multi_anchor_report is not None:
        if (multi_anchor_report.get("schema_version")
                != "xian_r5.multi_anchor_base_tie_candidates.v1"
                or multi_anchor_report.get("field_applied") is not False
                or multi_anchor_report.get("field_interpolated") is not False
                or multi_anchor_report.get("accepted_cells") != 0
                or multi_anchor_report.get("dispersion_is_positional_uncertainty") is not False
                or receipt.get("multi_anchor_base_tie_field_applied") is not False
                or receipt.get("multi_anchor_base_tie_accepted_cell_count") != 0):
            errors.append("MULTI_ANCHOR_CANDIDATE_ONLY_GATE_MISMATCH")
        integrity = multi_anchor_report.get("integrity_files_sha256")
        if not isinstance(integrity, dict) or not integrity:
            errors.append("MULTI_ANCHOR_SOURCE_INTEGRITY_MISSING")
        else:
            for source_path, expected_digest in integrity.items():
                candidate_path = Path(source_path)
                if (not candidate_path.is_file()
                        or _sha256(candidate_path) != expected_digest):
                    errors.append(f"MULTI_ANCHOR_SOURCE_HASH_MISMATCH:{source_path}")
        multi_fields = {
            str(field.get("field_id")): field
            for field in manifest.get("image_alignment_fields", [])
            if field.get("method") == "MULTI_ANCHOR_BASE_TIE_CONSENSUS"
        }
        expected_rows = {
            (str(item.get("field_id")), str(item.get("cell_id"))): item
            for item in multi_anchor_report.get("field_cell_summaries", [])
        }
        actual_rows = {
            (str(row.get("field_id")), str(row.get("cell_id"))): row
            for row in alignment_rows
            if row.get("method") == "MULTI_ANCHOR_BASE_TIE_CONSENSUS"
        }
        if (len(multi_fields) != receipt.get("multi_anchor_base_tie_field_count")
                or set(actual_rows) != set(expected_rows)
                or set(multi_fields) != {key[0] for key in expected_rows}):
            errors.append("MULTI_ANCHOR_FIELD_CELL_INVENTORY_MISMATCH")
        for key, expected in expected_rows.items():
            row = actual_rows.get(key)
            if row is None:
                continue
            summary = expected.get("anchor_summary") or {}
            dx, dy = expected.get("correction_dx_m_candidate"), expected.get("correction_dy_m_candidate")
            numeric = dx is not None and dy is not None
            actual_dx, actual_dy = row.get("correction_dx_m"), row.get("correction_dy_m")
            comparisons = [
                row.get("status") == expected.get("status"),
                row.get("scope") == "IMAGE_TO_REFERENCE_GRID",
                row.get("accepted") == 0 and row.get("reviewed") == 0
                and row.get("independent_ground") == 0 and row.get("applied") == 0,
                row.get("position_unc_m") is None,
                row.get("anchor_count") == summary.get("anchor_count"),
                row.get("anchor_years") == (",".join(
                    str(year) for year in summary.get("reference_years", [])) or None),
                row.get("fit_dispersion_kind") == (
                    "CROSS_ANCHOR_MEDIAN_RADIAL_DEVIATION_NOT_POSITIONAL_UNCERTAINTY"),
                numeric == (actual_dx is not None and actual_dy is not None),
            ]
            if numeric:
                comparisons.extend((
                    math.isclose(float(actual_dx), float(dx), rel_tol=0.0, abs_tol=1e-5),
                    math.isclose(float(actual_dy), float(dy), rel_tol=0.0, abs_tol=1e-5),
                    row.get("anchor_count", 0) >= 2,
                    row.get("sample_support") == "MULTI_ANCHOR_CANDIDATE_SUPPORT",
                ))
                dispersion = summary.get("median_radial_deviation_m")
                comparisons.append(
                    dispersion is not None and row.get("fit_disp_m") is not None
                    and math.isclose(float(row["fit_disp_m"]), float(dispersion),
                                     rel_tol=0.0, abs_tol=1e-5))
            else:
                comparisons.append(expected.get("status") ==
                                   "MULTI_ANCHOR_BASE_TIE_INSUFFICIENT_SUPPORT")
            if not all(comparisons):
                errors.append(f"MULTI_ANCHOR_CELL_RECEIPT_MISMATCH:{key[0]}:{key[1]}")
        numeric_count = sum(
            item.get("correction_dx_m_candidate") is not None
            and item.get("correction_dy_m_candidate") is not None
            for item in multi_anchor_report.get("field_cell_summaries", []))
        if numeric_count != receipt.get("multi_anchor_base_tie_numeric_cell_count"):
            errors.append("MULTI_ANCHOR_NUMERIC_CELL_COUNT_MISMATCH")
    component_tie_report = receipt.get("component_tie_consensus_source_report")
    if component_tie_report is not None:
        if (component_tie_report.get("schema_version")
                != "xian_r5.component_tie_consensus_field_adapter.v1"
                or component_tie_report.get("scope") != "FIXED_CASE_LOCAL_ONLY"
                or component_tie_report.get("field_applied") is not False
                or component_tie_report.get("field_interpolated") is not False
                or component_tie_report.get("absolute_ground_position_resolved") is not False
                or component_tie_report.get("formal_change_use") is not False
                or component_tie_report.get("accepted_cell_count") != 0
                or receipt.get("component_tie_consensus_field_applied") is not False
                or receipt.get("component_tie_consensus_accepted_cell_count") != 0):
            errors.append("COMPONENT_TIE_CONSENSUS_CANDIDATE_ONLY_GATE_MISMATCH")
        integrity = component_tie_report.get("integrity_files_sha256")
        if not isinstance(integrity, dict) or not integrity:
            errors.append("COMPONENT_TIE_CONSENSUS_SOURCE_INTEGRITY_MISSING")
        else:
            for source_path, expected_digest in integrity.items():
                candidate_path = Path(source_path)
                if (not candidate_path.is_file()
                        or _sha256(candidate_path) != expected_digest):
                    errors.append(f"COMPONENT_TIE_CONSENSUS_SOURCE_HASH_MISMATCH:{source_path}")
        component_fields = {
            str(field.get("field_id")): field
            for field in manifest.get("image_alignment_fields", [])
            if field.get("method") == "MULTI_ANCHOR_SOURCE_COMPONENT_TIE_CONSENSUS"
        }
        expected_cell_rows = {
            (str(row.get("field_id")), str(row.get("cell_id"))): row
            for row in component_tie_report.get("field_cell_summaries", [])
        }
        actual_cell_rows = {
            (str(row.get("field_id")), str(row.get("cell_id"))): row
            for row in alignment_rows
            if row.get("method") == "MULTI_ANCHOR_SOURCE_COMPONENT_TIE_CONSENSUS"
        }
        if (len(component_fields) != receipt.get("component_tie_consensus_field_count")
                or set(actual_cell_rows) != set(expected_cell_rows)
                or set(component_fields) != {key[0] for key in expected_cell_rows}):
            errors.append("COMPONENT_TIE_CONSENSUS_FIELD_CELL_INVENTORY_MISMATCH")
        for key, expected in expected_cell_rows.items():
            row = actual_cell_rows.get(key)
            if row is None:
                continue
            dx = expected.get("correction_dx_m_candidate")
            dy = expected.get("correction_dy_m_candidate")
            actual_dx, actual_dy = row.get("correction_dx_m"), row.get("correction_dy_m")
            try:
                component_ids = json.loads(row.get("source_component_ids_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                component_ids = None
            expected_ids = expected.get("source_component_ids")
            comparisons = [
                row.get("status") == expected.get("status"),
                row.get("scope") == "IMAGE_TO_REFERENCE_GRID",
                row.get("accepted") == 0 and row.get("reviewed") == 0
                and row.get("independent_ground") == 0 and row.get("applied") == 0,
                row.get("position_unc_m") is None,
                row.get("source_component_scope") == "SINGLE_SOURCE_COMPONENT_PER_CELL",
                row.get("source_component_coverage") is not None
                and math.isclose(float(row["source_component_coverage"]), 1.0,
                                 rel_tol=0.0, abs_tol=1e-8),
                component_ids == expected_ids,
                row.get("sample_support") == expected.get("sample_support_status"),
                (dx is None) == (actual_dx is None),
                (dy is None) == (actual_dy is None),
            ]
            if dx is not None and dy is not None:
                comparisons.extend((
                    math.isclose(float(actual_dx), float(dx), rel_tol=0.0, abs_tol=1e-5),
                    math.isclose(float(actual_dy), float(dy), rel_tol=0.0, abs_tol=1e-5),
                    expected.get("component_consensus_status") ==
                    "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE",
                ))
            if not all(comparisons):
                errors.append(f"COMPONENT_TIE_CONSENSUS_CELL_RECEIPT_MISMATCH:{key[0]}:{key[1]}")
        numeric_count = sum(
            item.get("correction_dx_m_candidate") is not None
            and item.get("correction_dy_m_candidate") is not None
            for item in component_tie_report.get("field_cell_summaries", []))
        if (numeric_count != receipt.get("component_tie_consensus_numeric_cell_count")
                or numeric_count != component_tie_report.get("numeric_candidate_cell_count")
                or sum(component_tie_report.get(
                    "field_cell_count_by_actual_date", {}).values())
                != receipt.get("component_tie_consensus_cell_count")):
            errors.append("COMPONENT_TIE_CONSENSUS_NUMERIC_COUNT_MISMATCH")
    with _open_vector(root / "building_lineage.gpkg", layer="image_grid_roof_candidates") as source:
        roof_alignment_rows = [(dict(feature["properties"]), shape(feature["geometry"]))
                               for feature in source if feature.get("geometry")]
    expected_roof_alignment_rows = sum(
        product.get("observation_count", 0)
        for product in manifest.get("image_grid_roof_candidate_products", []))
    if len(roof_alignment_rows) != expected_roof_alignment_rows:
        errors.append("IMAGE_GRID_ROOF_CANDIDATE_LINEAGE_COUNT_MISMATCH")
    if any(row.get("ground_status") != "NOT_A_GROUND_FOOTPRINT"
           or row.get("geometry_basis") != "ROOF_IMAGE_GRID_CANDIDATE"
           for row, _ in roof_alignment_rows):
        errors.append("IMAGE_GRID_ROOF_CANDIDATE_MISLABELED_AS_GROUND")
    if (product_schema in {"xian_r5.grounded_products.v11",
                           "xian_r5.grounded_products.v12",
                           "xian_r5.grounded_products.v13"}
            and any(row.get("source_component_scope") is None
                    or row.get("source_component_ids_json") is None
                    for row, _ in roof_alignment_rows)):
        errors.append("IMAGE_GRID_ROOF_SOURCE_COMPONENT_FIELDS_MISSING")
    roof_rows_by_field: dict[str, list[tuple[dict, object]]] = {}
    for row, candidate_geometry in roof_alignment_rows:
        shifted = bool(row.get("candidate_shift"))
        dx, dy = row.get("candidate_dx_m"), row.get("candidate_dy_m")
        support_fraction = row.get("cell_support_fraction")
        if (product_schema in {"xian_r5.grounded_products.v11",
                               "xian_r5.grounded_products.v12",
                               "xian_r5.grounded_products.v13"} and shifted
                and row.get("source_component_scope")
                != "SINGLE_SOURCE_COMPONENT_PER_CELL"):
            errors.append(f"IMAGE_GRID_ROOF_CROSS_COMPONENT_SHIFT:{row.get('uid')}")
        dominant_candidate = row.get("status") == (
            "UNACCEPTED_DOMINANT_CELL_IMAGE_GRID_SHIFT_ROOF_CANDIDATE")
        if dominant_candidate and (
                not shifted or support_fraction is None
                or not 0.8 <= float(support_fraction) <= 1.0):
            errors.append("IMAGE_GRID_ROOF_DOMINANT_CELL_SUPPORT_INVALID")
        if shifted and (support_fraction is None or not 0.0 < float(support_fraction) <= 1.0):
            errors.append("IMAGE_GRID_ROOF_CELL_SUPPORT_FRACTION_INVALID")
        raw = raw_roofs.get(row.get("uid"))
        if shifted != (dx is not None and dy is not None) or raw is None:
            errors.append("IMAGE_GRID_ROOF_CANDIDATE_SHIFT_PROVENANCE_MISMATCH")
            continue
        expected_geometry = translate(raw, xoff=float(dx), yoff=float(dy)) if shifted else raw
        if candidate_geometry.hausdorff_distance(expected_geometry) > 1e-7:
            errors.append(f"IMAGE_GRID_ROOF_GEOMETRY_RELATION_MISMATCH:{row.get('uid')}")
        roof_rows_by_field.setdefault(str(row["field_id"]), []).append((row, candidate_geometry))
    roof_candidates_by_uid_field = {
        (str(row.get("uid")), str(row.get("field_id"))): (row, geometry)
        for row, geometry in roof_alignment_rows
    }
    products_by_field = {str(item["field_id"]): item
                         for item in manifest.get("image_grid_roof_candidate_products", [])}
    if set(products_by_field) != set(roof_rows_by_field):
        errors.append("IMAGE_GRID_ROOF_FIELD_PRODUCT_SET_MISMATCH")
    for field_id, product in products_by_field.items():
        candidate_path = root / product["candidate_tif"]
        quality_path = root / product["quality_tif"]
        if not candidate_path.is_file() or not quality_path.is_file():
            errors.append(f"IMAGE_GRID_ROOF_TIF_MISSING:{field_id}")
            continue
        with rasterio.open(candidate_path) as candidate_ds, rasterio.open(quality_path) as quality_ds:
            if candidate_ds.nodata != candidate_codes.get("nodata"):
                errors.append(f"IMAGE_GRID_ROOF_NODATA_METADATA_MISMATCH:{field_id}")
            if product_schema in TIF_METADATA_PRODUCT_SCHEMAS:
                expected_tags = {
                    "MOVING_ACTUAL_DATE": product.get("moving_date") or "UNKNOWN",
                    "REFERENCE_LABEL": product.get("reference_label") or "UNKNOWN",
                    "REFERENCE_GRID_PATH": product.get("reference_source") or "UNKNOWN",
                    "REFERENCE_ACQUISITION_DATE_STATUS": (
                        product.get("reference_date_status") or "UNKNOWN"),
                    "ALIGNMENT_FIELD_ID": field_id,
                    "ROOF_TO_GROUND_CORRECTION": "NO",
                    "POSITIONAL_ACCURACY_CLAIMED": "NO",
                }
                for dataset, expected_kind in (
                        (candidate_ds, "IMAGE_GRID_ROOF_ALIGNMENT_CANDIDATE"),
                        (quality_ds, "IMAGE_GRID_ROOF_ALIGNMENT_QUALITY")):
                    tags = dataset.tags()
                    if product_schema == "xian_r5.grounded_products.v10":
                        expected_tags["ROOF_CELL_SUPPORT_POLICY"] = (
                            "FULL_SINGLE_CELL_OR_UNACCEPTED_UNIQUE_DOMINANT_CELL_AREA_FRACTION_GE_0.80")
                    elif product_schema in {"xian_r5.grounded_products.v11",
                                            "xian_r5.grounded_products.v12",
                                            "xian_r5.grounded_products.v13"}:
                        expected_tags["ROOF_CELL_SUPPORT_POLICY"] = (
                            "SINGLE_SOURCE_COMPONENT_REQUIRED;FULL_CELL_OR_UNACCEPTED_DOMINANT_CELL_AREA_FRACTION_GE_0.80")
                    if (tags.get("PRODUCT_SCHEMA_VERSION") != manifest_product_schema
                            or tags.get("PRODUCT_KIND") != expected_kind
                            or any(tags.get(name) != value
                                   for name, value in expected_tags.items())):
                        errors.append(f"IMAGE_GRID_ROOF_TIF_METADATA_MISMATCH:{field_id}")
            expected_candidate = np.zeros((candidate_ds.height, candidate_ds.width), dtype=np.uint8)
            expected_quality = np.zeros_like(expected_candidate)
            occupied = np.zeros_like(expected_candidate, dtype=bool)
            for row, geometry in roof_rows_by_field.get(field_id, []):
                mask = rasterize([(geometry, 1)], out_shape=expected_candidate.shape,
                                 transform=candidate_ds.transform, fill=0, dtype="uint8").astype(bool)
                value = (candidate_codes["unaccepted_candidate"] if bool(row.get("candidate_shift"))
                         else candidate_codes["unresolved"])
                if value == 1:
                    status = str(row.get("status", ""))
                    code = (1 if status.startswith("ACCEPTED_") else
                            5 if status == "UNACCEPTED_DOMINANT_CELL_IMAGE_GRID_SHIFT_ROOF_CANDIDATE"
                            else 2)
                else:
                    if row.get("status") == "CROSSES_OR_OVERLAPS_ALIGNMENT_CELLS":
                        code = 4
                    else:
                        quality_codes = manifest.get(
                            "image_grid_roof_candidate_quality_codes", {})
                        code = _source_component_quality_code(
                            row.get("status"), quality_codes)
                        if code is None:
                            cell_key = (field_id, str(row.get("cell_id") or ""))
                            cell_record = alignment_cells.get(cell_key)
                            support_status = (cell_record[0].get("sample_support")
                                              if cell_record is not None else None)
                            code = _roof_support_quality_code(support_status, quality_codes)
                overlap = mask & occupied
                fresh = mask & ~occupied
                expected_candidate[fresh] = value
                expected_quality[fresh] = code
                expected_candidate[overlap] = candidate_codes["unresolved"]
                expected_quality[overlap] = 4
                occupied[mask] = True
            if (not np.array_equal(candidate_ds.read(1), expected_candidate)
                    or not np.array_equal(quality_ds.read(1), expected_quality)):
                errors.append(f"IMAGE_GRID_ROOF_TIF_VECTOR_MISMATCH:{field_id}")
        if has_base_grid:
            base_candidate_name = product.get("candidate_base10m_tif")
            base_quality_name = product.get("quality_base10m_tif")
            if not base_candidate_name or not base_quality_name:
                errors.append(f"IMAGE_GRID_ROOF_BASE10M_TIF_MISSING:{field_id}")
                continue
            base_candidate_path = root / base_candidate_name
            base_quality_path = root / base_quality_name
            if not base_candidate_path.is_file() or not base_quality_path.is_file():
                errors.append(f"IMAGE_GRID_ROOF_BASE10M_TIF_MISSING:{field_id}")
                continue
            factor = int(manifest["grid"]["fine_pixels_per_reference_pixel"])
            with rasterio.open(candidate_path) as fine_candidate, rasterio.open(quality_path) as fine_quality, rasterio.open(base_candidate_path) as base_candidate, rasterio.open(base_quality_path) as base_quality:
                if base_candidate.nodata != candidate_codes.get("nodata"):
                    errors.append(f"IMAGE_GRID_ROOF_BASE10M_NODATA_METADATA_MISMATCH:{field_id}")
                if product_schema in TIF_METADATA_PRODUCT_SCHEMAS:
                    candidate_tags = base_candidate.tags()
                    quality_tags = base_quality.tags()
                    if (candidate_tags.get("GRID_LEVEL") != "REFERENCE_10M"
                            or candidate_tags.get("DERIVED_FROM_RASTER") != product["candidate_tif"]
                            or quality_tags.get("GRID_LEVEL") != "REFERENCE_10M"
                            or quality_tags.get("DERIVED_FROM_RASTER") != product["quality_tif"]
                            or candidate_tags.get("ROOF_TO_GROUND_CORRECTION") != "NO"
                            or quality_tags.get("ROOF_TO_GROUND_CORRECTION") != "NO"):
                        errors.append(f"IMAGE_GRID_ROOF_BASE10M_TIF_METADATA_MISMATCH:{field_id}")
                    quality_codes = manifest.get(
                        "image_grid_roof_candidate_quality_codes", {})
                    if any(label in quality_codes for label in
                           _ROOF_SUPPORT_QUALITY_LABELS.values()):
                        if quality_tags.get("AGGREGATION_POLICY") != "MAX_FINE_QUALITY_CODE":
                            errors.append(
                                f"IMAGE_GRID_ROOF_BASE10M_QUALITY_AGGREGATION_MISMATCH:{field_id}")
                if (not np.array_equal(base_candidate.read(1),
                                       _coarsen_candidate_mask(fine_candidate.read(1), factor))
                        or not np.array_equal(base_quality.read(1),
                                              _coarsen_candidate_quality(fine_quality.read(1), factor))):
                    errors.append(f"IMAGE_GRID_ROOF_BASE10M_COARSEN_MISMATCH:{field_id}")
    external_source = receipt.get("external_footprint_candidate_source")
    source_tile_date_report = (external_source or {}).get("source_tile_date_association")
    candidate_observation_count = sum(len(obs.get("footprint_candidates", [])) for obs in receipt["observations"])
    candidate_vector_count = None
    with _open_vector(root / "building_lineage.gpkg", layer="footprint_candidates") as source:
        candidate_schema_fields = set(source.schema.get("properties", {}))
        candidate_features = [
            (dict(feature["properties"]),
             shape(feature["geometry"]) if feature.get("geometry") else None)
            for feature in source
        ]
    candidate_rows = [properties for properties, _ in candidate_features]
    candidate_vector_count = len(candidate_rows)
    if (product_schema in {"xian_r5.grounded_products.v12",
                           "xian_r5.grounded_products.v13"}
            and "source_epoch_label" not in candidate_schema_fields):
        errors.append("FOOTPRINT_CANDIDATE_SOURCE_EPOCH_FIELD_MISSING")
    if (product_schema == "xian_r5.grounded_products.v13"
            and not {"source_geometry_semantics",
                     "source_geometry_semantics_provenance"}.issubset(candidate_schema_fields)):
        errors.append("FOOTPRINT_CANDIDATE_SOURCE_GEOMETRY_SEMANTICS_FIELDS_MISSING")
    if any(row.get("status") != "UNVERIFIED_CANDIDATE" for row in candidate_rows):
        errors.append("FOOTPRINT_CANDIDATE_PROMOTED")
    outline_diagnostic_rows = []
    if product_schema in {"xian_r5.grounded_products.v7", "xian_r5.grounded_products.v8",
                          "xian_r5.grounded_products.v9", "xian_r5.grounded_products.v10",
                          "xian_r5.grounded_products.v11",
                          "xian_r5.grounded_products.v12",
                          "xian_r5.grounded_products.v13"}:
        diagnostic_contract = manifest.get("image_grid_roof_outline_diagnostics")
        if (not isinstance(diagnostic_contract, dict)
                or diagnostic_contract.get("layer")
                != "building_lineage.gpkg#image_grid_roof_outline_diagnostics"
                or diagnostic_contract.get("formal_ground_use") is not False
                or diagnostic_contract.get("formal_change_use") is not False
                or diagnostic_contract.get("ground_status") != "NOT_VERIFIED"):
            errors.append("IMAGE_GRID_ROOF_OUTLINE_DIAGNOSTIC_MANIFEST_INVALID")
            if not isinstance(diagnostic_contract, dict):
                diagnostic_contract = {}
        with _open_vector(root / "building_lineage.gpkg",
                          layer="image_grid_roof_outline_diagnostics") as source:
            outline_diagnostic_rows = [
                (dict(feature["properties"]),
                 shape(feature["geometry"]) if feature.get("geometry") else None)
                for feature in source
            ]
        receipt_candidates_by_uid = {
            str(obs["uid"]): obs.get("footprint_candidates", [])
            for obs in receipt["observations"]
        }
        expected_diagnostic_keys = {
            (uid, field_id, str(candidate.get("candidate_id")))
            for uid, field_id in roof_candidates_by_uid_field
            for candidate in receipt_candidates_by_uid.get(uid, [])
        }
        actual_diagnostic_by_key = {}
        for props, geometry in outline_diagnostic_rows:
            key = (str(props.get("uid")), str(props.get("field_id")),
                   str(props.get("candidate_id")))
            if key in actual_diagnostic_by_key:
                errors.append(f"IMAGE_GRID_ROOF_OUTLINE_DIAGNOSTIC_DUPLICATE:{key}")
            actual_diagnostic_by_key[key] = (props, geometry)
        if (set(actual_diagnostic_by_key) != expected_diagnostic_keys
                or len(outline_diagnostic_rows) != len(expected_diagnostic_keys)
                or diagnostic_contract.get("row_count") != len(expected_diagnostic_keys)):
            errors.append("IMAGE_GRID_ROOF_OUTLINE_DIAGNOSTIC_COUNT_OR_KEYS_MISMATCH")
        external_geometry_by_key = {
            (str(props.get("uid")), str(props.get("candidate_id"))): geometry
            for props, geometry in candidate_features
            if geometry is not None and props.get("candidate_id") not in {
                "FOOTPRINT_EVIDENCE", "UNSPECIFIED_CANDIDATE"}
        }

        def metric_matches(actual, expected):
            if expected is None:
                return actual is None
            if actual is None:
                return False
            return math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-5)

        for key, (props, geometry) in actual_diagnostic_by_key.items():
            uid, field_id, candidate_id = key
            roof_entry = roof_candidates_by_uid_field.get((uid, field_id))
            raw_roof = raw_roofs.get(uid)
            external_geometry = external_geometry_by_key.get((uid, candidate_id))
            candidate_properties = next((row for row in candidate_rows
                                         if str(row.get("uid")) == uid
                                         and str(row.get("candidate_id")) == candidate_id), None)
            if roof_entry is None or raw_roof is None or external_geometry is None:
                errors.append(f"IMAGE_GRID_ROOF_OUTLINE_DIAGNOSTIC_LINEAGE_MISSING:{key}")
                continue
            roof_props, shifted_roof = roof_entry
            if (geometry is None or geometry.hausdorff_distance(external_geometry) > 1e-7
                    or props.get("candidate_source_ref") != (candidate_properties or {}).get("source_ref")
                    or props.get("candidate_source_crs") != (candidate_properties or {}).get("source_crs")
                    or props.get("candidate_source_date") != (candidate_properties or {}).get("source_date")
                    or props.get("candidate_source_accuracy_m") != (candidate_properties or {}).get("source_accuracy_m")
                    or props.get("roof_source_ref") != raw_roof_sources.get(uid)
                    or props.get("alignment_status") != roof_props.get("status")
                    or props.get("candidate_shift") != roof_props.get("candidate_shift")
                    or not metric_matches(props.get("candidate_dx_m"), roof_props.get("candidate_dx_m"))
                    or not metric_matches(props.get("candidate_dy_m"), roof_props.get("candidate_dy_m"))
                    or props.get("ground_status") != "NOT_VERIFIED"
                    or props.get("geometry_basis") != "UNVERIFIED_EXTERNAL_OUTLINE"
                    or props.get("formal_ground_use") != 0
                    or props.get("formal_change_use") != 0):
                errors.append(f"IMAGE_GRID_ROOF_OUTLINE_DIAGNOSTIC_STATUS_OR_PROVENANCE_MISMATCH:{key}")
            raw_iou, raw_intersection, raw_distance = (
                _polygon_overlap_metrics(raw_roof, external_geometry))
            if (not metric_matches(props.get("raw_iou"), raw_iou)
                    or not metric_matches(props.get("raw_intersection_m2"), raw_intersection)
                    or not metric_matches(props.get("raw_distance_m"), raw_distance)):
                errors.append(f"IMAGE_GRID_ROOF_OUTLINE_RAW_METRICS_MISMATCH:{key}")
            if bool(roof_props.get("candidate_shift")):
                image_iou, image_intersection, image_distance = (
                    _polygon_overlap_metrics(shifted_roof, external_geometry))
                residual_dx = external_geometry.centroid.x - shifted_roof.centroid.x
                residual_dy = external_geometry.centroid.y - shifted_roof.centroid.y
                if (not metric_matches(props.get("image_grid_iou"), image_iou)
                        or not metric_matches(props.get("image_grid_iou_gain"), image_iou - raw_iou)
                        or not metric_matches(props.get("image_grid_intersection_m2"), image_intersection)
                        or not metric_matches(props.get("image_grid_distance_m"), image_distance)
                        or not metric_matches(props.get("outline_dx_from_shifted_m"), residual_dx)
                        or not metric_matches(props.get("outline_dy_from_shifted_m"), residual_dy)):
                    errors.append(f"IMAGE_GRID_ROOF_OUTLINE_SHIFTED_METRICS_MISMATCH:{key}")
            elif any(props.get(name) is not None for name in (
                    "image_grid_iou", "image_grid_iou_gain", "image_grid_intersection_m2",
                    "image_grid_distance_m", "outline_dx_from_shifted_m",
                    "outline_dy_from_shifted_m")):
                errors.append(f"IMAGE_GRID_ROOF_OUTLINE_UNRENDERED_SHIFT_HAS_METRICS:{key}")
    if product_schema in {"xian_r5.grounded_products.v6", "xian_r5.grounded_products.v7",
                          "xian_r5.grounded_products.v8", "xian_r5.grounded_products.v9",
                          "xian_r5.grounded_products.v10", "xian_r5.grounded_products.v11",
                          "xian_r5.grounded_products.v12",
                          "xian_r5.grounded_products.v13"}:
        ground_candidate_codes = manifest.get("ground_footprint_candidate_codes", {})
        candidate_value = ground_candidate_codes.get("unverified_candidate")
        if (ground_candidate_codes.get("background") == candidate_value
                or ground_candidate_codes.get("nodata") == candidate_value
                or candidate_value is None):
            errors.append("GROUND_CANDIDATE_CODES_INVALID")
        group_field = "output_group" if actual_date_schema else "epoch"
        products_by_epoch = {
            str(item.get(group_field)): item
            for item in manifest.get("ground_footprint_candidate_products", [])
        }
        if (len(products_by_epoch) != len(epochs)
                or set(products_by_epoch) != set(epochs)):
            errors.append("GROUND_CANDIDATE_EPOCH_PRODUCT_SET_MISMATCH")
        factor = int(manifest["grid"]["fine_pixels_per_reference_pixel"] or 1)
        for epoch in epochs:
            product = products_by_epoch.get(epoch)
            if product is None:
                continue
            group_info = manifest.get("output_group_metadata", {}).get(epoch, {})
            def belongs_to_group(properties):
                if not actual_date_schema:
                    return properties.get("epoch") == epoch
                if group_info.get("actual_date") is not None:
                    return properties.get("actual_date") == group_info["actual_date"]
                raw_date = properties.get("actual_date")
                try:
                    date.fromisoformat(str(raw_date))
                    return False
                except (TypeError, ValueError):
                    return properties.get("epoch") in group_info.get("epoch_labels", [])
            candidate_path = root / str(product.get("candidate_tif", ""))
            if not candidate_path.is_file():
                errors.append(f"GROUND_CANDIDATE_TIF_MISSING:{epoch}")
                continue
            expected_geometries = [
                geometry for properties, geometry in candidate_features
                if belongs_to_group(properties) and geometry is not None
            ]
            with rasterio.open(candidate_path) as candidate_ds:
                expected = (
                    rasterize([(geometry, int(candidate_value)) for geometry in expected_geometries],
                              out_shape=(candidate_ds.height, candidate_ds.width),
                              transform=candidate_ds.transform, fill=0, dtype="uint8")
                    if expected_geometries else
                    np.zeros((candidate_ds.height, candidate_ds.width), dtype="uint8")
                )
                tags = candidate_ds.tags()
                if candidate_ds.nodata != footprint_codes.get("nodata"):
                    errors.append(f"GROUND_CANDIDATE_NODATA_MISMATCH:{epoch}")
                if (tags.get("PRODUCT_SCHEMA_VERSION") != manifest_product_schema
                        or tags.get("PRODUCT_KIND") != "UNVERIFIED_GROUND_FOOTPRINT_CANDIDATE"
                        or (tags.get("OUTPUT_GROUP_ID") != epoch if actual_date_schema
                            else tags.get("EPOCH_LABEL") != epoch)
                        or tags.get("FORMAL_GROUND_FOOTPRINT_PRODUCT") != "NO"
                        or tags.get("GROUND_FOOTPRINT_CONFIRMED") != "NO"
                        or tags.get("POSITIONAL_ACCURACY_CLAIMED") != "NO"
                        or tags.get("ROOF_TO_GROUND_CORRECTION") != "NO"):
                    errors.append(f"GROUND_CANDIDATE_TIF_METADATA_MISMATCH:{epoch}")
                if not np.array_equal(candidate_ds.read(1), expected):
                    errors.append(f"GROUND_CANDIDATE_TIF_VECTOR_MISMATCH:{epoch}")
                if int(np.count_nonzero(candidate_ds.read(1))) != product.get("candidate_pixel_count"):
                    errors.append(f"GROUND_CANDIDATE_PIXEL_COUNT_MISMATCH:{epoch}")
            epoch_features = [properties for properties, geometry in candidate_features
                              if belongs_to_group(properties) and geometry is not None]
            if (len(epoch_features) != product.get("candidate_feature_count")
                    or len({row.get("uid") for row in epoch_features})
                    != product.get("candidate_observation_count")
                    or product.get("formal_ground_footprint_product") is not False
                    or product.get("position_accuracy_claimed") is not False
                    or product.get("roof_to_ground_correction") is not False):
                errors.append(f"GROUND_CANDIDATE_MANIFEST_MISMATCH:{epoch}")
            if has_base_grid:
                base_name = product.get("candidate_base10m_tif")
                base_path = root / str(base_name or "")
                if not base_name or not base_path.is_file():
                    errors.append(f"GROUND_CANDIDATE_BASE10M_TIF_MISSING:{epoch}")
                    continue
                with rasterio.open(candidate_path) as fine_ds, rasterio.open(base_path) as base_ds:
                    base_tags = base_ds.tags()
                    if (base_ds.nodata != footprint_codes.get("nodata")
                            or base_tags.get("GRID_LEVEL") != "REFERENCE_10M"
                            or base_tags.get("DERIVED_FROM_RASTER") != product["candidate_tif"]
                            or base_tags.get("FORMAL_GROUND_FOOTPRINT_PRODUCT") != "NO"
                            or not np.array_equal(
                                base_ds.read(1),
                                _coarsen_candidate_mask(fine_ds.read(1), factor))):
                        errors.append(f"GROUND_CANDIDATE_BASE10M_MISMATCH:{epoch}")
    elif manifest.get("ground_footprint_candidate_products"):
        errors.append("UNVERSIONED_GROUND_CANDIDATE_PRODUCTS")
    if manifest_product_schema in {"xian_r5.grounded_products.v15",
                                   "xian_r5.grounded_products.v16",
                                   "xian_r5.grounded_products.v17"}:
        unknown_track = int(manifest.get("candidate_track_id_codes", {}).get(
            "unresolved_or_ambiguous", -1))
        expected_quality_codes = {
            "background": 0,
            "unique_candidate_track": 1,
            "candidate_track_missing": 2,
            "candidate_track_repeated_within_date": 3,
            "overlapping_candidate_tracks": 4,
        }
        track_semantics = manifest.get("candidate_track_id_semantics", {})
        track_codebook_contract = manifest.get("candidate_track_id_codebook", {})
        if (manifest.get("candidate_track_ids_promoted_to_physical_ids") is not False
                or manifest.get("candidate_track_id_quality_codes") != expected_quality_codes
                or track_semantics.get("physical_identity_confirmed") is not False
                or track_semantics.get("formal_physical_id_product") is not False
                or track_semantics.get("used_for_presence_or_change") is not False
                or track_semantics.get("repeated_label_within_actual_date") != "ALL_PIXELS_UNRESOLVED"
                or track_semantics.get("overlapping_different_labels") != "OVERLAP_PIXELS_UNRESOLVED"
                or unknown_track != 4294967295):
            errors.append("CANDIDATE_TRACK_ID_SEMANTICS_INVALID")
        codebook_path = root / str(track_codebook_contract.get("path") or "")
        codebook = None
        if (not codebook_path.is_file()
                or not track_codebook_contract.get("sha256")
                or _sha256(codebook_path) != track_codebook_contract.get("sha256")):
            errors.append("CANDIDATE_TRACK_ID_CODEBOOK_HASH_MISMATCH")
        else:
            try:
                codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                errors.append("CANDIDATE_TRACK_ID_CODEBOOK_INVALID")
        code_to_label = ((codebook or {}).get("code_to_source_candidate_identity") or {})
        valid_track_codes = set()
        labels_seen = set()
        for raw_code, label in code_to_label.items():
            try:
                code = int(raw_code)
            except (TypeError, ValueError):
                errors.append("CANDIDATE_TRACK_ID_CODEBOOK_CODE_INVALID")
                continue
            expected_code = int.from_bytes(hashlib.sha256(
                str(label).encode("utf-8")).digest()[:4], "big")
            if (code in {0, unknown_track} or code != expected_code
                    or str(label) in labels_seen or code in valid_track_codes):
                errors.append("CANDIDATE_TRACK_ID_CODEBOOK_ENTRY_INVALID")
            labels_seen.add(str(label))
            valid_track_codes.add(code)
        if (not isinstance(codebook, dict)
                or codebook.get("schema_version") != "xian_r5.candidate_track_id_codebook.v1"
                or codebook.get("semantics")
                != "HASHED_SOURCE_CANDIDATE_TRACK_LABEL_NOT_PHYSICAL_BUILDING_ID"
                or track_codebook_contract.get("candidate_identity_label_count") != len(valid_track_codes)
                or len(labels_seen) != len(valid_track_codes)):
            errors.append("CANDIDATE_TRACK_ID_CODEBOOK_CONTRACT_INVALID")
        track_products = manifest.get("candidate_track_id_products", [])
        track_products_by_group = {
            str(item.get("output_group")): item for item in track_products
        }
        expected_groups = set(manifest.get("output_groups", []))
        if (len(track_products_by_group) != len(track_products)
                or set(track_products_by_group) != expected_groups):
            errors.append("CANDIDATE_TRACK_ID_PRODUCT_GROUPS_MISMATCH")
        factor = int(manifest.get("grid", {}).get(
            "fine_pixels_per_reference_pixel") or 1)
        for group_id in expected_groups & set(track_products_by_group):
            product = track_products_by_group[group_id]
            track_path = root / str(product.get("candidate_tif") or "")
            quality_path = root / str(product.get("quality_tif") or "")
            if not track_path.is_file() or not quality_path.is_file():
                errors.append(f"CANDIDATE_TRACK_ID_TIFS_MISSING:{group_id}")
                continue
            with rasterio.open(track_path) as track_ds, rasterio.open(quality_path) as quality_ds:
                track_ids = track_ds.read(1)
                track_quality = quality_ds.read(1)
                tags = track_ds.tags()
                quality_tags = quality_ds.tags()
                allowed_ids = {0, unknown_track} | valid_track_codes
                if (not np.isin(track_ids, list(allowed_ids)).all()
                        or not np.isin(track_quality, list(expected_quality_codes.values())).all()
                        or np.any((track_quality == 0) != (track_ids == 0))
                        or np.any((track_quality == 1)
                                  & ((track_ids == 0) | (track_ids == unknown_track)))
                        or np.any((track_quality >= 2) & (track_ids != unknown_track))):
                    errors.append(f"CANDIDATE_TRACK_ID_PIXELS_INVALID:{group_id}")
                forbidden_tags = {
                    "FORMAL_PHYSICAL_ID_PRODUCT": "NO",
                    "PHYSICAL_IDENTITY_CONFIRMED": "NO",
                    "CANDIDATE_TRACK_ID_PROMOTED_TO_PHYSICAL_ID": "NO",
                    "CHANGE_USE": "NO",
                    "GROUND_FOOTPRINT_CONFIRMED": "NO",
                    "ROOF_TO_GROUND_CORRECTION": "NO",
                    "POSITIONAL_ACCURACY_CLAIMED": "NO",
                }
                if (tags.get("PRODUCT_SCHEMA_VERSION") != manifest_product_schema
                        or tags.get("PRODUCT_KIND")
                        != "CANDIDATE_TRACK_ID_ROOF_PROXY_NOT_PHYSICAL_ID"
                        or tags.get("OUTPUT_GROUP_ID") != group_id
                        or tags.get("CODEBOOK_JSON") != track_codebook_contract.get("path")
                        or tags.get("CODEBOOK_SHA256") != track_codebook_contract.get("sha256")
                        or any(tags.get(key) != value for key, value in forbidden_tags.items())
                        or quality_tags.get("PRODUCT_KIND")
                        != "CANDIDATE_TRACK_ID_QUALITY_NOT_PHYSICAL_ID_QUALITY"
                        or quality_tags.get("OUTPUT_GROUP_ID") != group_id
                        or quality_tags.get("FORMAL_PHYSICAL_ID_PRODUCT") != "NO"
                        or quality_tags.get("CHANGE_USE") != "NO"
                        or track_ds.nodata is not None or quality_ds.nodata is not None):
                    errors.append(f"CANDIDATE_TRACK_ID_TIF_METADATA_INVALID:{group_id}")
                expected_observations = [
                    obs for obs in receipt.get("observations", [])
                    if ((manifest.get("output_group_metadata", {}).get(group_id, {}).get("actual_date")
                         is not None
                         and obs.get("actual_date") == group_id)
                        or (manifest.get("output_group_metadata", {}).get(group_id, {}).get("actual_date")
                            is None
                            and obs.get("actual_date") in (None, "", "DATE_UNKNOWN",
                                                            "NO_TILE_DATE_RECORD")
                            and obs.get("epoch_label") in manifest.get(
                                "output_group_metadata", {}).get(group_id, {}).get("epoch_labels", [])))
                ]
                labels_for_group = [
                    str(obs.get("candidate_identity")).strip()
                    for obs in expected_observations
                    if isinstance(obs.get("candidate_identity"), str)
                    and obs.get("candidate_identity").strip()
                ]
                repeated_labels = sorted(label for label, count in Counter(
                    labels_for_group).items() if count > 1)
                if (product.get("candidate_observation_count") != len(labels_for_group)
                        or product.get("source_track_label_count") != len(set(labels_for_group))
                        or product.get("repeated_source_track_labels") != repeated_labels
                        or product.get("unique_track_pixel_count") != int(np.count_nonzero(
                            (track_ids != 0) & (track_ids != unknown_track)))
                        or product.get("unresolved_track_pixel_count") != int(np.count_nonzero(
                            track_ids == unknown_track))
                        or product.get("formal_physical_id_product") is not False
                        or product.get("candidate_ids_promoted_to_physical_ids") is not False
                        or product.get("change_use") is not False
                        or product.get("ground_footprint_confirmed") is not False):
                    errors.append(f"CANDIDATE_TRACK_ID_MANIFEST_MISMATCH:{group_id}")
            if has_base_grid:
                base_track_path = root / str(product.get("candidate_base10m_tif") or "")
                base_quality_path = root / str(product.get("quality_base10m_tif") or "")
                if not base_track_path.is_file() or not base_quality_path.is_file():
                    errors.append(f"CANDIDATE_TRACK_ID_BASE10M_TIFS_MISSING:{group_id}")
                    continue
                with rasterio.open(track_path) as fine_track, rasterio.open(quality_path) as fine_quality, \
                        rasterio.open(base_track_path) as base_track, rasterio.open(base_quality_path) as base_quality:
                    if (base_track.tags().get("GRID_LEVEL") != "REFERENCE_10M"
                            or base_track.tags().get("DERIVED_FROM_RASTER") != product.get("candidate_tif")
                            or base_quality.tags().get("GRID_LEVEL") != "REFERENCE_10M"
                            or base_quality.tags().get("DERIVED_FROM_RASTER") != product.get("quality_tif")
                            or not np.array_equal(
                                base_track.read(1),
                                _coarsen_candidate_identity_ids(
                                    fine_track.read(1), factor, unknown_track))
                            or not np.array_equal(
                                base_quality.read(1),
                                _coarsen_candidate_quality(fine_quality.read(1), factor))):
                        errors.append(f"CANDIDATE_TRACK_ID_BASE10M_MISMATCH:{group_id}")
    elif manifest.get("candidate_track_id_products"):
        errors.append("UNVERSIONED_CANDIDATE_TRACK_ID_PRODUCTS")
    positive_overlap_rows = [row for row in candidate_rows
                             if row.get("association") == "POSITIVE_AREA_OVERLAP_ONLY"]
    nearby_candidate_rows = [row for row in candidate_rows
                             if row.get("association") == "NEARBY_WITHIN_SEARCH_RADIUS"]
    metric_names = (
        "match_iou", "intersection_area_m2", "roof_overlap_fraction",
        "candidate_overlap_fraction", "candidate_centroid_dx_from_roof_m",
        "candidate_centroid_dy_from_roof_m", "centroid_distance_m",
    )
    external_schema = (external_source or {}).get("schema_version")
    new_candidate_metrics_required = external_schema in {
        "xian_r5.external_footprint_candidates.v3",
        "xian_r5.external_footprint_candidates.v4",
        "xian_r5.external_footprint_candidates.v5",
    }
    externally_associated_rows = positive_overlap_rows + nearby_candidate_rows
    if new_candidate_metrics_required:
        receipt_candidate_rows = [
            (obs["uid"], candidate)
            for obs in receipt["observations"]
            for candidate in obs.get("footprint_candidates", [])
        ]
        receipt_by_key = {
            (uid, str(candidate.get("candidate_id"))): candidate
            for uid, candidate in receipt_candidate_rows
        }
        vector_by_key = {
            (row.get("uid"), str(row.get("candidate_id"))): row
            for row in externally_associated_rows
        }
        if (len(receipt_by_key) != len(receipt_candidate_rows)
                or len(vector_by_key) != len(externally_associated_rows)
                or receipt_by_key.keys() != vector_by_key.keys()):
            errors.append("EXTERNAL_CANDIDATE_RECEIPT_VECTOR_KEYS_MISMATCH")
        receipt_vector_metric_names = (
            "match_iou", "intersection_area_m2", "roof_overlap_fraction",
            "candidate_overlap_fraction", "candidate_centroid_dx_from_roof_m",
            "candidate_centroid_dy_from_roof_m", "centroid_distance_m",
            "candidate_polygon_distance_m", "candidate_search_radius_m",
            "centroid_aligned_roof_iou", "centroid_aligned_iou_gain", "area_m2",
        )
        for key in receipt_by_key.keys() & vector_by_key.keys():
            candidate = receipt_by_key[key]
            row = vector_by_key[key]
            if (candidate.get("source_ref") != row.get("source_ref")
                    or candidate.get("source_crs") != row.get("source_crs")
                    or candidate.get("source_date") != row.get("source_date")
                    or (product_schema in {"xian_r5.grounded_products.v12",
                                           "xian_r5.grounded_products.v13"}
                        and candidate.get("source_epoch_label")
                        != row.get("source_epoch_label"))
                    or (product_schema == "xian_r5.grounded_products.v13"
                        and (candidate.get("source_geometry_semantics")
                             != row.get("source_geometry_semantics")
                             or candidate.get("source_geometry_semantics_provenance")
                             != row.get("source_geometry_semantics_provenance")))
                    or candidate.get("source_accuracy_m") != row.get("source_accuracy_m")
                    or candidate.get("association") != row.get("association")
                    or candidate.get("status") != row.get("status")):
                errors.append(f"EXTERNAL_CANDIDATE_RECEIPT_VECTOR_PROVENANCE_MISMATCH:{key[0]}:{key[1]}")
            if source_tile_date_report is not None:
                try:
                    vector_source_dates = json.loads(
                        row.get("source_date_candidates_json") or "[]")
                    vector_source_tiles = json.loads(
                        row.get("source_tile_ids_json") or "[]")
                    vector_tile_actual_dates = json.loads(
                        row.get("source_tile_actual_dates_json") or "[]")
                except (TypeError, json.JSONDecodeError):
                    vector_source_dates = vector_source_tiles = vector_tile_actual_dates = None
                if (candidate.get("source_date") is not None
                        or candidate.get("source_date_status") not in {
                            "NO_SOURCE_TILE_SUPPORT", "INCOMPLETE_SOURCE_TILE_DATES",
                            "SINGLE_ACTUAL_DATE_CANDIDATE", "MULTIPLE_ACTUAL_DATES"}
                        or candidate.get("source_date_candidates") != vector_source_dates
                        or candidate.get("source_tile_ids") != vector_source_tiles
                        or candidate.get("source_tile_actual_dates") != vector_tile_actual_dates
                        or candidate.get("source_date_status") != row.get("source_date_status")
                        or candidate.get("source_scene_key") != row.get("source_scene_key")
                        or candidate.get("source_scene_linkage_status")
                        != row.get("source_scene_linkage_status")
                        or candidate.get("source_date_provenance")
                        != row.get("source_date_provenance")):
                    errors.append(
                        f"EXTERNAL_CANDIDATE_SOURCE_TILE_DATE_PROVENANCE_MISMATCH:{key[0]}:{key[1]}")
                tile_actual_dates = candidate.get("source_tile_actual_dates")
                if not isinstance(tile_actual_dates, list):
                    errors.append(
                        f"EXTERNAL_CANDIDATE_SOURCE_TILE_DATE_MAP_INVALID:{key[0]}:{key[1]}")
                    tile_actual_dates = []
                mapped_tile_ids = [item.get("tile_id") for item in tile_actual_dates
                                   if isinstance(item, dict)]
                mapped_dates = sorted({item.get("actual_date") for item in tile_actual_dates
                                       if isinstance(item, dict)
                                       and item.get("actual_date") is not None})
                has_unknown_tile_date = any(
                    isinstance(item, dict) and item.get("actual_date") is None
                    for item in tile_actual_dates)
                if (candidate.get("source_tile_ids") != mapped_tile_ids
                        or candidate.get("source_date_candidates") != mapped_dates):
                    errors.append(
                        f"EXTERNAL_CANDIDATE_SOURCE_TILE_DATE_MAP_CONTENT_INVALID:{key[0]}:{key[1]}")
                if candidate.get("source_date_status") == "NO_SOURCE_TILE_SUPPORT":
                    expected_status = "NO_SOURCE_TILE_SUPPORT" if not mapped_tile_ids else None
                elif not mapped_tile_ids:
                    expected_status = "NO_SOURCE_TILE_SUPPORT"
                elif has_unknown_tile_date:
                    expected_status = "INCOMPLETE_SOURCE_TILE_DATES"
                elif len(mapped_dates) == 1:
                    expected_status = "SINGLE_ACTUAL_DATE_CANDIDATE"
                else:
                    expected_status = "MULTIPLE_ACTUAL_DATES"
                if expected_status != candidate.get("source_date_status"):
                    errors.append(
                        f"EXTERNAL_CANDIDATE_SOURCE_TILE_DATE_STATUS_INVALID:{key[0]}:{key[1]}")
                epoch_label = str(candidate.get("source_epoch_label"))
                expected_scene = source_tile_date_report.get("scene_mapping", {}).get(epoch_label)
                expected_link_status = (
                    "EPOCH_LABEL_TO_SCENE_MAPPING_ONLY_SOURCE_VECTOR_NOT_HASH_BOUND"
                    if expected_scene is not None else "NO_SCENE_MAPPING_FOR_USER_EPOCH_LABEL")
                if (candidate.get("source_scene_key") != expected_scene
                        or candidate.get("source_scene_linkage_status") != expected_link_status):
                    errors.append(
                        f"EXTERNAL_CANDIDATE_SCENE_LINKAGE_CLAIM_INVALID:{key[0]}:{key[1]}")
            for name in receipt_vector_metric_names:
                left, right = candidate.get(name), row.get(name)
                if left is None or right is None:
                    matches = left is None and right is None
                else:
                    matches = math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-5)
                if not matches:
                    errors.append(f"EXTERNAL_CANDIDATE_RECEIPT_VECTOR_METRIC_MISMATCH:{key[0]}:{key[1]}:{name}")
            try:
                vector_properties = json.loads(row.get("source_properties_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                vector_properties = None
            if candidate.get("source_properties") != vector_properties:
                errors.append(f"EXTERNAL_CANDIDATE_RECEIPT_VECTOR_PROPERTIES_MISMATCH:{key[0]}:{key[1]}")
        if source_tile_date_report is not None:
            candidate_status_counts = dict(Counter(
                candidate.get("source_date_status")
                for _, candidate in receipt_candidate_rows))
            candidate_date_counts = dict(Counter(
                source_date
                for _, candidate in receipt_candidate_rows
                for source_date in candidate.get("source_date_candidates", [])))
            if (source_tile_date_report.get("schema_version")
                    != "xian_r5.external_candidate_source_tile_dates.v1"
                    or source_tile_date_report.get("capture_date_field_promoted") is not False
                    or source_tile_date_report.get("neighbor_date_inference_used") is not False
                    or source_tile_date_report.get("source_vector_linkage_status")
                    != "USER_REPORTED_EPOCH_LABEL_MAPPED_TO_SOURCE_SCENE;"
                       "VECTOR_SOURCE_RASTER_HASH_NOT_VERIFIED"
                    or source_tile_date_report.get("source_date_registry_sha256")
                    != input_hashes.get("actual_date_registry_before")
                    or source_tile_date_report.get("observation_candidate_association_count")
                    != len(receipt_candidate_rows)
                    or source_tile_date_report.get(
                        "candidate_status_counts_by_observation_association")
                    != candidate_status_counts
                    or source_tile_date_report.get(
                        "candidate_date_counts_by_observation_association")
                    != candidate_date_counts):
                errors.append("EXTERNAL_CANDIDATE_SOURCE_TILE_DATE_REPORT_INVALID")
            if (manifest.get("external_candidate_source_date_status_counts")
                    != candidate_status_counts
                    or manifest.get("external_candidate_source_date_candidates")
                    != sorted(candidate_date_counts)
                    or manifest.get("external_candidate_source_scene_linkage_verified") is not False):
                errors.append("EXTERNAL_CANDIDATE_SOURCE_TILE_DATE_MANIFEST_MISMATCH")
    if candidate_metrics_required:
        for index, row in enumerate(externally_associated_rows):
            required_names = list(metric_names)
            if new_candidate_metrics_required:
                required_names.extend(("candidate_polygon_distance_m",
                                       "centroid_aligned_roof_iou",
                                       "centroid_aligned_iou_gain"))
            if any(row.get(name) is None or not math.isfinite(float(row[name]))
                   for name in required_names):
                errors.append(f"EXTERNAL_CANDIDATE_METRICS_INCOMPLETE:{index}")
                continue
            association = row.get("association")
            bad_intersection = (
                association == "POSITIVE_AREA_OVERLAP_ONLY"
                and row["intersection_area_m2"] <= 0.01
            ) or (
                association == "NEARBY_WITHIN_SEARCH_RADIUS"
                and row["intersection_area_m2"] > 0.01
            )
            if (bad_intersection
                    or not 0 <= row["match_iou"] <= 1
                    or not 0 <= row["roof_overlap_fraction"] <= 1
                    or not 0 <= row["candidate_overlap_fraction"] <= 1
                    or row["centroid_distance_m"] < 0):
                errors.append(f"EXTERNAL_CANDIDATE_METRICS_OUT_OF_RANGE:{index}")
                continue
            expected_distance = math.hypot(row["candidate_centroid_dx_from_roof_m"],
                                           row["candidate_centroid_dy_from_roof_m"])
            if not math.isclose(row["centroid_distance_m"], expected_distance,
                                rel_tol=1e-6, abs_tol=1e-3):
                errors.append(f"EXTERNAL_CANDIDATE_CENTROID_DISTANCE_MISMATCH:{index}")
            if new_candidate_metrics_required:
                if (row["candidate_polygon_distance_m"] < 0
                        or not 0 <= row["centroid_aligned_roof_iou"] <= 1
                        or not -1 <= row["centroid_aligned_iou_gain"] <= 1
                        or not math.isclose(
                            row["centroid_aligned_iou_gain"],
                            row["centroid_aligned_roof_iou"] - row["match_iou"],
                            rel_tol=1e-6, abs_tol=1e-5)):
                    errors.append(f"EXTERNAL_CANDIDATE_TRANSLATION_DIAGNOSTIC_INVALID:{index}")
                if row.get("association") == "POSITIVE_AREA_OVERLAP_ONLY":
                    if (row["candidate_polygon_distance_m"] > 1e-4
                            or row.get("candidate_search_radius_m") is not None):
                        errors.append(f"EXTERNAL_CANDIDATE_OVERLAP_DISTANCE_INVALID:{index}")
                elif row.get("association") == "NEARBY_WITHIN_SEARCH_RADIUS":
                    radius = row.get("candidate_search_radius_m")
                    if (row["intersection_area_m2"] > 0.01 or radius is None
                            or not math.isfinite(float(radius)) or radius < 0
                            or row["candidate_polygon_distance_m"] > radius + 1e-3):
                        errors.append(f"EXTERNAL_NEARBY_CANDIDATE_RADIUS_INVALID:{index}")
        if new_candidate_metrics_required and candidate_vector_count != len(externally_associated_rows):
            errors.append("EXTERNAL_CANDIDATE_ASSOCIATION_TYPE_MISMATCH")
    complete_metric_rows = [
        row for row in positive_overlap_rows
        if all(row.get(name) is not None and math.isfinite(float(row[name])) for name in metric_names)
    ]
    if external_source:
        if external_source.get("ground_truth") is not False:
            errors.append("EXTERNAL_SOURCE_MISLABELED_GROUND_TRUTH")
        if external_source.get("source_capture_date") is not None or external_source.get("source_accuracy_m") is not None:
            errors.append("UNKNOWN_EXTERNAL_PROVENANCE_INVENTED")
        if (external_source.get("source_epoch_label") is not None
                and external_source.get("source_epoch_label_provenance")
                != "USER_REPORTED_EPOCH_LABEL_NOT_EXACT_ACQUISITION_DATE"):
            errors.append("EXTERNAL_EPOCH_LABEL_PROVENANCE_INVALID")
        if external_source.get("source_geometry_semantics") is not None:
            allowed_semantics = {"ROOF_BOUNDARY_FROM_IMAGE",
                                 "GROUND_FOOTPRINT_FROM_ORTHO_OR_SURVEY", "UNKNOWN"}
            if (external_source.get("source_geometry_semantics") not in allowed_semantics
                    or external_source.get("source_geometry_semantics_provenance")
                    != "USER_REPORTED_SEMANTICS_NOT_INDEPENDENTLY_VERIFIED"):
                errors.append("EXTERNAL_SOURCE_GEOMETRY_SEMANTICS_PROVENANCE_INVALID")
            if (external_source.get("source_geometry_semantics") == "ROOF_BOUNDARY_FROM_IMAGE"
                    and manifest.get("external_candidate_source_geometry_semantics")
                    != ["ROOF_BOUNDARY_FROM_IMAGE"]):
                errors.append("EXTERNAL_ROOF_BOUNDARY_SEMANTICS_NOT_PRESERVED")
        elif (external_source.get("schema_version")
              == "xian_r5.external_footprint_candidates.v5"
              and external_source.get("source_geometry_semantics_provenance")
              != "NOT_PROVIDED"):
            errors.append("EXTERNAL_SOURCE_GEOMETRY_SEMANTICS_UNKNOWN_PROVENANCE_INVALID")
        if candidate_vector_count != external_source.get("candidate_associations"):
            errors.append("EXTERNAL_CANDIDATE_VECTOR_COUNT_MISMATCH")
        if any(row.get("source_date") is not None or row.get("source_accuracy_m") is not None
               for row in candidate_rows):
            errors.append("UNKNOWN_EXTERNAL_PROVENANCE_NOT_PRESERVED")
    elif candidate_vector_count < candidate_observation_count:
        errors.append("FOOTPRINT_CANDIDATE_VECTOR_COUNT_MISMATCH")
    outcome = {
        "schema_version": "xian_r5.reconstruction_engineering_qa.v1",
        "case_id": receipt["case_id"], "structural_qa": "PASS" if not errors else "FAIL",
        "formal_ground_correction": "NOT_DEMONSTRATED_NO_REVIEWED_GROUND_EVIDENCE" if receipt["ground_footprints_confirmed"] == 0 else "REQUIRES_SEPARATE_SPATIAL_REVIEW",
        "formal_change_accuracy": "NOT_EVALUATED",
        "errors": errors, "raster_count": len(tifs), "common_grid_count": len(grids),
        "fine_grid_count": len({(p["crs"], tuple(p["transform"]), p["width"], p["height"]) for p in fine}),
        "native_base_grid_count": len({(p["crs"], tuple(p["transform"]), p["width"], p["height"]) for p in native}),
        "observation_count": len(receipt["observations"]),
        "image_grid_roof_candidate_rows": len(roof_alignment_rows),
        "image_grid_roof_outline_diagnostic_rows": len(outline_diagnostic_rows),
        "external_candidate_associations": candidate_vector_count,
        "unverified_ground_candidate_products": len(
            manifest.get("ground_footprint_candidate_products", [])),
        "external_candidate_provenance": "UNKNOWN_PRESERVED_AS_NULL" if external_source else "NO_EXTERNAL_SOURCE",
        "external_candidate_source_epoch_label": (
            external_source.get("source_epoch_label") if external_source else None),
        "external_candidate_source_geometry_semantics": (
            external_source.get("source_geometry_semantics") if external_source else None),
        "external_candidate_source_geometry_semantics_provenance": (
            external_source.get("source_geometry_semantics_provenance")
            if external_source else None),
        "external_candidate_source_date_status_counts": (
            manifest.get("external_candidate_source_date_status_counts", {})
            if external_source else {}),
        "external_candidate_source_date_candidates": (
            manifest.get("external_candidate_source_date_candidates", [])
            if external_source else []),
        "external_candidate_source_scene_linkage_verified": (
            manifest.get("external_candidate_source_scene_linkage_verified", False)
            if external_source else False),
        "external_candidate_overlap_metrics": {
            "schema": ("v5" if external_schema == "xian_r5.external_footprint_candidates.v5"
                       else "v4" if new_candidate_metrics_required
                       else "v3" if candidate_metrics_required
                       else "not_present_in_legacy_v2"),
            "positive_overlap_rows": len(positive_overlap_rows),
            "nearby_nonoverlap_rows": len(nearby_candidate_rows),
            "complete_rows": len(complete_metric_rows) if candidate_metrics_required else None,
            "median_intersection_area_m2": (float(np.median([row["intersection_area_m2"] for row in complete_metric_rows]))
                                             if candidate_metrics_required and complete_metric_rows else None),
            "median_roof_overlap_fraction": (float(np.median([row["roof_overlap_fraction"] for row in complete_metric_rows]))
                                             if candidate_metrics_required and complete_metric_rows else None),
            "median_candidate_overlap_fraction": (float(np.median([row["candidate_overlap_fraction"] for row in complete_metric_rows]))
                                                   if candidate_metrics_required and complete_metric_rows else None),
            "median_centroid_distance_m": (float(np.median([row["centroid_distance_m"] for row in complete_metric_rows]))
                                           if candidate_metrics_required and complete_metric_rows else None),
            "median_centroid_aligned_roof_iou": (
                float(np.median([row["centroid_aligned_roof_iou"]
                                 for row in externally_associated_rows]))
                if new_candidate_metrics_required and externally_associated_rows else None),
        },
        "actual_dates": dict(Counter(o["actual_date"] for o in receipt["observations"])),
        "registration_statuses": dict(Counter(r["status"] for r in receipt["registrations"])),
        "raster_code_histograms": codes,
        "r4_status_sha256_unchanged": receipt["input_sha256"]["r4_status_before"] == receipt["input_sha256"]["r4_status_after"],
        "tracked_input_hashes_unchanged": tracked_input_hashes_unchanged,
        "pixel_input_hashes_unchanged": pixel_hashes_unchanged,
        "base_raster_hashes_unchanged": base_hashes_unchanged,
    }
    if Path(receipt_name).name != receipt_name:
        raise ValueError("QA receipt name must be a filename, not a path")
    path = root / receipt_name
    if path.exists():
        raise FileExistsError(f"QA receipt already exists: {path}")
    path.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt-name", default="ENGINEERING_QA.json")
    args = parser.parse_args()
    result = check(args.output, args.receipt_name)
    print(json.dumps({key: result[key] for key in ("structural_qa", "formal_ground_correction", "formal_change_accuracy", "errors", "raster_count", "registration_statuses")}, ensure_ascii=False))
    sys.exit(0 if result["structural_qa"] == "PASS" else 1)
