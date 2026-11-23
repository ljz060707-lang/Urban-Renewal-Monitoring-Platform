"""One EPSG:32649 grid for ground footprints, IDs, changes and spatial QA."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, mapping
from shapely.ops import unary_union

from xian_r5.alignment.field import LocalDisplacementField
from xian_r5.alignment.roof_candidate import (
    RoofGridCandidate,
    align_roof_candidate_to_reference_grid,
)
from .core import Observation, Reconstruction, reviewed_identity_is_valid


ID_UNKNOWN = np.uint32(4294967295)
U = np.uint8(2)
NODATA = np.uint8(255)
STATE_CODES = {"STABLE": 1, "ADDITION": 2, "DEMOLITION": 3, "UNRESOLVED": 4}
PRODUCT_SCHEMA_VERSION = "xian_r5.grounded_products.v17"
CANDIDATE_TRACK_QUALITY_CODES = {
    "background": 0,
    "unique_candidate_track": 1,
    "candidate_track_missing": 2,
    "candidate_track_repeated_within_date": 3,
    "overlapping_candidate_tracks": 4,
}
_TIE_SUPPORT_QUALITY_CODES = {
    "INSUFFICIENT_MASKED_SAMPLES": 8,
    "NO_VALID_CORRELATION_SCORE": 9,
    "INSUFFICIENT_ANCHOR_CANDIDATES": 10,
}
_TIE_SUPPORT_QUALITY_LABELS = {
    "insufficient_masked_tie_samples": 8,
    "no_valid_correlation_peak": 9,
    "insufficient_multi_anchor_support": 10,
}
_SOURCE_COMPONENT_QUALITY_LABELS = {
    "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED": "mosaic_component_boundary_unresolved",
    "SOURCE_COMPONENT_COVERAGE_INCOMPLETE": "source_component_coverage_incomplete",
    "SOURCE_COMPONENT_ID_MISMATCH": "source_component_id_mismatch",
    "SOURCE_COMPONENT_SCOPE_UNKNOWN": "source_component_scope_unknown",
}
_SOURCE_COMPONENT_QUALITY_CODES = {
    "mosaic_component_boundary_unresolved": 11,
    "source_component_coverage_incomplete": 12,
    "source_component_id_mismatch": 13,
    "source_component_scope_unknown": 14,
}


@dataclass(frozen=True)
class _RoofAlignmentProduct:
    observation: Observation
    field: LocalDisplacementField
    candidate: RoofGridCandidate


def _tie_support_quality_code(cell: Any | None) -> int | None:
    """Map an explicit failed-tie support reason to a non-acceptance QA code."""
    if cell is None or not isinstance(cell.metadata, dict):
        return None
    return _TIE_SUPPORT_QUALITY_CODES.get(cell.metadata.get("sample_support_status"))


def _source_tile_properties(cell: Any | None) -> dict[str, Any]:
    """Expose verified tile-grid provenance without changing candidate status."""
    provenance = (cell.metadata.get("source_tile_grid")
                  if cell is not None else None)
    if not isinstance(provenance, dict):
        provenance = None
    return {
        "tile_mosaic_source": (provenance.get("mosaic_diagnostic_path")
                               if provenance else None),
        "tile_mosaic_sha256": (provenance.get("mosaic_diagnostic_sha256")
                               if provenance else None),
        "tile_col0": provenance.get("tile_col0") if provenance else None,
        "tile_row0_north_to_south": (
            provenance.get("tile_row0_north_to_south") if provenance else None),
        "patch_tiles": provenance.get("patch_tiles") if provenance else None,
        "tile_window_json": (json.dumps(provenance["tile_window"], sort_keys=True,
                                         separators=(",", ":"))
                             if provenance else None),
        "pixel_window_json": (json.dumps(provenance["pixel_window"],
                                          separators=(",", ":"))
                              if provenance else None),
        "tile_prov_json": (json.dumps(provenance, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"))
                           if provenance else None),
    }


def _source_component_properties(cell: Any | None) -> dict[str, Any]:
    metadata = cell.metadata if cell is not None else {}
    components = metadata.get("source_component_ids", [])
    return {
        "source_component_scope": metadata.get(
            "source_component_scope", "SOURCE_COMPONENT_SCOPE_UNKNOWN"),
        "source_component_ids_json": json.dumps(
            components if isinstance(components, list) else [],
            ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "source_component_coverage": metadata.get(
            "source_component_coverage_fraction"),
    }


def _reference_window_evidence(cell: Any | None) -> list[dict[str, Any]]:
    if cell is None:
        return []
    evidence = cell.metadata.get("reference_window_evidence")
    if isinstance(evidence, dict):
        return [evidence]
    if isinstance(evidence, list):
        return [item for item in evidence if isinstance(item, dict)]
    return []


def _reference_window_properties(cell: Any | None) -> dict[str, Any]:
    evidence = _reference_window_evidence(cell)
    years = sorted({int(item["reference_year_label"]) for item in evidence
                    if item.get("reference_year_label") is not None})
    return {
        "reference_anchor_count": len(evidence),
        "reference_anchor_years": ",".join(str(year) for year in years) or None,
        "reference_window_evidence_json": (
            json.dumps(evidence, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) if evidence else None),
    }


def _field_tile_grid_summary(field: LocalDisplacementField) -> dict[str, Any]:
    provenances = [cell.metadata.get("source_tile_grid") for cell in field.cells
                   if isinstance(cell.metadata.get("source_tile_grid"), dict)]
    if not provenances:
        return {"source_tile_cell_count": 0}
    signatures = {
        (item.get("mosaic_diagnostic_path"), item.get("mosaic_diagnostic_sha256"),
         json.dumps(item.get("tile_window"), sort_keys=True))
        for item in provenances}
    if len(signatures) != 1:
        raise ValueError(f"Alignment field mixes source tile mosaics: {field.field_id}")
    first = provenances[0]
    present_by_date: dict[str, set[str]] = {}
    verified_component_windows: set[tuple[str, str, str, str, tuple[int, ...]]] = set()
    for provenance in provenances:
        for actual_date, tile_group in provenance.get("tiles_by_actual_date", {}).items():
            for row in tile_group.get("present_tiles", []):
                present_by_date.setdefault(actual_date, set()).add(row["tile_id"])
                grid = row.get("stitched_mosaic_grid")
                if (isinstance(grid, dict)
                        and grid.get("source_component_window_verified") is True):
                    pixel_window = grid.get("pixel_window_col_row_width_height")
                    if isinstance(pixel_window, list) and len(pixel_window) == 4:
                        verified_component_windows.add((
                            actual_date, row["tile_id"],
                            str(grid.get("nominal_task_date") or ""),
                            str(grid.get("mosaic_path") or ""),
                            tuple(int(value) for value in pixel_window)))
    return {
        "source_tile_cell_count": len(provenances),
        "source_tile_mosaic_diagnostic": first.get("mosaic_diagnostic_path"),
        "source_tile_mosaic_sha256": first.get("mosaic_diagnostic_sha256"),
        "source_tile_window": first.get("tile_window"),
        "source_tile_counts_by_actual_date": {
            actual_date: len(tile_ids)
            for actual_date, tile_ids in sorted(present_by_date.items())},
        "stitched_mosaic_component_window_verified_count": (
            len(verified_component_windows)),
    }


def _field_source_component_summary(field: LocalDisplacementField) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for cell in field.cells:
        scope = str(cell.metadata.get(
            "source_component_scope", "SOURCE_COMPONENT_SCOPE_UNKNOWN"))
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "source_component_scope_counts": dict(sorted(counts.items())),
        "single_source_component_cell_count": sum(
            cell.source_component_scoped for cell in field.cells),
    }


def _field_reference_window_summary(field: LocalDisplacementField) -> dict[str, Any]:
    cells_with_evidence = [cell for cell in field.cells
                           if _reference_window_evidence(cell)]
    if not cells_with_evidence:
        return {"reference_window_evidence_cell_count": 0,
                "reference_anchor_sources": []}
    first_evidence = _reference_window_evidence(cells_with_evidence[0])
    source_rows = {}
    statuses = set()
    years = set()
    for cell in cells_with_evidence:
        for evidence in _reference_window_evidence(cell):
            year = evidence.get("reference_year_label")
            path = evidence.get("reference_raster_path")
            digest = evidence.get("reference_raster_sha256")
            if year is not None and path and digest:
                key = (int(year), str(path), str(digest))
                source_rows[key] = {"year_label": int(year), "path": str(path),
                                    "sha256": str(digest)}
                years.add(int(year))
            if evidence.get("reference_component_id_status"):
                statuses.add(str(evidence["reference_component_id_status"]))
    return {
        "reference_window_evidence_cell_count": len(cells_with_evidence),
        "reference_anchor_year_labels": sorted(years),
        "reference_anchor_sources": [source_rows[key] for key in sorted(source_rows)],
        "reference_component_id_status": (
            next(iter(statuses)) if len(statuses) == 1 else "MIXED_OR_UNKNOWN"),
        "reference_window_evidence_anchor_count_per_first_cell": len(first_evidence),
    }


def _roof_alignment_products(
    result: Reconstruction,
    fields: tuple[LocalDisplacementField, ...],
    anchor_raster: Path | None,
    alignment_reference_rasters: tuple[Path | str, ...],
) -> list[_RoofAlignmentProduct]:
    """Build candidates for fields tied to the verified co-gridded base stack."""
    if anchor_raster is None:
        return []
    reference_sources = {
        str(Path(path).resolve()).replace("\\", "/").casefold()
        for path in (anchor_raster, *alignment_reference_rasters)
    }
    products = []
    for field in fields:
        if not field.reference_source:
            continue
        reference = str(Path(field.reference_source).resolve()).replace("\\", "/").casefold()
        if reference not in reference_sources or field.reference_date is not None:
            continue
        for observation in result.observations:
            if observation.actual_date != field.moving_date:
                continue
            candidate = align_roof_candidate_to_reference_grid(
                observation.roof, observation.actual_date, field)
            products.append(_RoofAlignmentProduct(observation, field, candidate))
    return products


def _polygon_comparison_metrics(left: Any, right: Any) -> tuple[float, float, float]:
    intersection_area = float(left.intersection(right).area)
    union_area = float(left.union(right).area)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou, intersection_area, float(left.distance(right))


def _shapes(items: list, value: int, shape: tuple[int, int], transform: Any, dtype: str) -> np.ndarray:
    if not items:
        return np.zeros(shape, dtype=dtype)
    return rasterize(((g, value) for g in items if g is not None and not g.is_empty),
                     out_shape=shape, transform=transform, fill=0, dtype=dtype)


def _coarsen_presence(array: np.ndarray, factor: int) -> np.ndarray:
    """Preserve a single presence code; mixed or unresolved blocks become U=3."""
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    low = blocks.min(axis=2)
    high = blocks.max(axis=2)
    out = low.astype(np.uint8)
    out[low != high] = 3
    out[np.any(blocks == 3, axis=2)] = 3
    return out


def _coarsen_float_max(array: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate positional uncertainty conservatively; all-unknown stays NaN."""
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    finite = np.isfinite(blocks)
    out = np.full(blocks.shape[:2], np.nan, dtype=np.float32)
    if finite.any():
        safe = np.where(finite, blocks, -np.inf)
        maxima = safe.max(axis=2)
        present = finite.any(axis=2)
        out[present] = maxima[present]
    return out


def _canonical_geometry_records(result: Reconstruction) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a dated medoid from reviewed IDs and accepted registered roofs only."""
    by_id: dict[int, list[Observation]] = {}
    reviewed_ids: dict[int, str] = {}
    for obs in result.observations:
        if obs.physical_id is None or not reviewed_identity_is_valid(obs):
            continue
        reviewed_ids[int(obs.physical_id)] = str(obs.confirmed_identity_key)
        by_id.setdefault(int(obs.physical_id), []).append(obs)

    geometries: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for physical_id in sorted(reviewed_ids):
        observations = by_id.get(physical_id, [])
        dated = []
        for obs in observations:
            if (obs.registered_roof is None
                    or obs.image_alignment_status != "ACCEPTED_IMAGE_GRID_SHIFT_ROOF_NOT_GROUND"):
                continue
            try:
                actual_date = date.fromisoformat(str(obs.actual_date)).isoformat()
            except (TypeError, ValueError):
                continue
            dated.append((actual_date, obs))

        per_date: dict[str, list[Observation]] = {}
        for actual_date, obs in dated:
            per_date.setdefault(actual_date, []).append(obs)
        incompatible = any(
            any(not group[0].registered_roof.equals(item.registered_roof)
                for item in group[1:])
            for group in per_date.values())
        candidates = []
        if not incompatible:
            for actual_date, group in sorted(per_date.items()):
                candidates.append(min(
                    group,
                    key=lambda obs: (
                        float(obs.image_alignment_positional_uncertainty_m)
                        if isinstance(obs.image_alignment_positional_uncertainty_m, (int, float))
                        and math.isfinite(float(obs.image_alignment_positional_uncertainty_m))
                        else float("inf"),
                        obs.uid,
                    )))

        if incompatible:
            status = "UNRESOLVED_MULTIPLE_REGISTERED_ROOFS_ON_SAME_DATE"
            selected = None
        elif len(candidates) < 2:
            status = "UNRESOLVED_FEWER_THAN_TWO_REGISTERED_ACTUAL_DATES"
            selected = None
        else:
            # A medoid preserves one observed geometry; no union, average, or warp is invented.
            scored = []
            for candidate in candidates:
                pairwise = [candidate.registered_roof.boundary.hausdorff_distance(
                    other.registered_roof.boundary)
                    for other in candidates if other.uid != candidate.uid]
                score = sum(pairwise) / len(pairwise) if pairwise else 0.0
                uncertainty = candidate.image_alignment_positional_uncertainty_m
                uncertainty = (float(uncertainty) if isinstance(uncertainty, (int, float))
                               and math.isfinite(float(uncertainty)) else float("inf"))
                scored.append((score, uncertainty, candidate.uid, candidate))
            selected = min(scored, key=lambda item: item[:3])[3]
            status = "CROSS_DATE_REGISTERED_ROOF_MEDOID_NOT_GROUND_TRUTH"

        for actual_date, obs in dated:
            comparison = selected.registered_roof if selected is not None else None
            if comparison is None:
                metrics = {"centroid_distance_m": None, "iou": None,
                           "hausdorff_boundary_distance_m": None,
                           "area_ratio": None, "shape_similarity": None}
            else:
                geom = obs.registered_roof
                intersection = float(geom.intersection(comparison).area)
                union = float(geom.union(comparison).area)
                iou = intersection / union if union > 0 else 0.0
                metrics = {
                    "centroid_distance_m": float(geom.centroid.distance(comparison.centroid)),
                    "iou": iou,
                    "hausdorff_boundary_distance_m": float(
                        geom.boundary.hausdorff_distance(comparison.boundary)),
                    "area_ratio": (min(geom.area, comparison.area) / max(geom.area, comparison.area)
                                   if max(geom.area, comparison.area) > 0 else 0.0),
                    "shape_similarity": iou * (
                        min(geom.area, comparison.area) / max(geom.area, comparison.area)
                        if max(geom.area, comparison.area) > 0 else 0.0),
                }
            evidence.append({
                "physical_id": physical_id, "actual_date": actual_date, "uid": obs.uid,
                "selected_as_canonical": bool(selected is not None and selected.uid == obs.uid),
                "canonical_status": status, **metrics,
            })
        geometries.append({
            "physical_id": physical_id, "confirmed_identity_key": reviewed_ids[physical_id],
            "geometry": selected.registered_roof if selected is not None else None,
            "canonical_uid": selected.uid if selected is not None else None,
            "canonical_date": selected.actual_date if selected is not None else None,
            "support_dates": sorted(per_date), "support_count": len(per_date),
            "status": status, "position_uncertainty_m": (
                selected.image_alignment_positional_uncertainty_m
                if selected is not None else None),
        })
    return geometries, evidence


def _unverified_ground_candidate_geometries(observation: Observation) -> list:
    """Return only geometries written to the unverified footprint-candidate layer."""
    geometries = [candidate.geometry for candidate in observation.footprint_candidates]
    evidence = observation.footprint_evidence
    if evidence is not None and not evidence.accepted:
        geometries.append(evidence.geometry)
    elif observation.footprint_candidate is not None and not observation.footprint_candidates:
        geometries.append(observation.footprint_candidate)
    return geometries


def _coarsen_candidate_footprints(array: np.ndarray, factor: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor)
    return blocks.max(axis=(1, 3)).astype(np.uint8)


def _tag_value(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _write_tif(path: Path, array: np.ndarray, transform: Any, *,
               nodata: int | float | None,
               tags: dict[str, Any] | None = None) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0],
                       count=1, dtype=array.dtype, crs="EPSG:32649", transform=transform,
                       tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=nodata) as sink:
        sink.write(array, 1)
        metadata = {
            "PRODUCT_SCHEMA_VERSION": PRODUCT_SCHEMA_VERSION,
            "PRODUCT_KIND": path.stem,
            "PRODUCT_CRS": "EPSG:32649",
            "PIXEL_SIZE_X_M": abs(float(transform.a)),
            "PIXEL_SIZE_Y_M": abs(float(transform.e)),
        }
        metadata.update(tags or {})
        sink.update_tags(**{key: _tag_value(value) for key, value in metadata.items()})


def _coarsen_binary_state(array: np.ndarray, factor: int, unknown: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    has_unknown = np.any(blocks == unknown, axis=2)
    has_value = np.any(blocks == 1, axis=2)
    result = np.zeros(has_unknown.shape, dtype=array.dtype)
    result[has_value] = 1
    result[has_unknown] = unknown
    return result


def _coarsen_ids(array: np.ndarray, factor: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    unknown = blocks == ID_UNKNOWN
    identity_pixels = (blocks > 0) & ~unknown
    has_identity = np.any(identity_pixels, axis=2)
    low_values = np.where(identity_pixels, blocks, np.iinfo(np.uint32).max)
    high_values = np.where(identity_pixels, blocks, 0)
    low, high = low_values.min(axis=2), high_values.max(axis=2)
    result = np.zeros(low.shape, dtype=np.uint32)
    single = has_identity & (low == high)
    result[single] = low[single]
    result[unknown.any(axis=2) | (has_identity & (low != high))] = ID_UNKNOWN
    return result


def _candidate_track_codebook(observations: Iterable[Observation]) -> dict[str, int]:
    """Create stable codes for source track labels without promoting them to IDs."""
    labels = sorted({
        obs.candidate_identity.strip()
        for obs in observations
        if isinstance(obs.candidate_identity, str) and obs.candidate_identity.strip()
    })
    label_to_code: dict[str, int] = {}
    code_to_label: dict[int, str] = {}
    for label in labels:
        code = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")
        if code in {0, int(ID_UNKNOWN)} or code in code_to_label:
            raise ValueError("Candidate track ID hash collision or reserved value")
        label_to_code[label] = code
        code_to_label[code] = label
    return label_to_code


def _candidate_track_rasters(
    records: list[Observation], shape: tuple[int, int], transform: Any,
    track_codes: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rasterize R4 track labels as roof-proxy candidates with conflict sentinels."""
    track_ids = np.zeros(shape, dtype=np.uint32)
    quality = np.zeros(shape, dtype=np.uint8)
    labels = [obs.candidate_identity.strip()
              if isinstance(obs.candidate_identity, str) and obs.candidate_identity.strip()
              else None for obs in records]
    label_counts: dict[str, int] = {}
    for label in labels:
        if label is not None:
            label_counts[label] = label_counts.get(label, 0) + 1

    missing_mask = np.zeros(shape, dtype=bool)
    repeated_mask = np.zeros(shape, dtype=bool)
    conflict_mask = np.zeros(shape, dtype=bool)
    for obs, label in zip(records, labels):
        geometry = obs.registered_roof or obs.roof
        mask = _shapes([geometry], 1, shape, transform, "uint8").astype(bool)
        if label is None:
            missing_mask |= mask
            continue
        if label_counts[label] > 1:
            repeated_mask |= mask
            continue
        code = np.uint32(track_codes[label])
        occupied = (track_ids != 0) & (track_ids != ID_UNKNOWN)
        conflict_mask |= mask & occupied & (track_ids != code)
        track_ids[mask & (track_ids == 0)] = code

    conflict_mask |= (missing_mask | repeated_mask) & (track_ids != 0) & (track_ids != ID_UNKNOWN)
    conflict_mask |= missing_mask & repeated_mask
    unresolved = missing_mask | repeated_mask | conflict_mask
    track_ids[unresolved] = ID_UNKNOWN
    quality[track_ids != 0] = CANDIDATE_TRACK_QUALITY_CODES["unique_candidate_track"]
    quality[missing_mask] = CANDIDATE_TRACK_QUALITY_CODES["candidate_track_missing"]
    quality[repeated_mask] = CANDIDATE_TRACK_QUALITY_CODES["candidate_track_repeated_within_date"]
    quality[conflict_mask] = CANDIDATE_TRACK_QUALITY_CODES["overlapping_candidate_tracks"]
    repeated_labels = sorted(label for label, count in label_counts.items() if count > 1)
    return track_ids, quality, repeated_labels


def _coarsen_quality(array: np.ndarray, factor: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    return blocks.max(axis=2).astype(np.uint8)


def _coarsen_alignment_quality(array: np.ndarray, factor: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    result = np.zeros(blocks.shape[:2], dtype=np.uint8)
    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            values = set(blocks[row, col].tolist()) - {0}
            if len(values) == 1:
                result[row, col] = values.pop()
            elif len(values) > 1:
                result[row, col] = 255
    return result


def _coarsen_change(array: np.ndarray, factor: int) -> np.ndarray:
    h, w = array.shape
    if h % factor or w % factor:
        raise ValueError("Fine-grid extent must contain complete base pixels")
    blocks = array.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3).reshape(h // factor, w // factor, factor * factor)
    unresolved = np.any(blocks == STATE_CODES["UNRESOLVED"], axis=2)
    state_pixels = (blocks >= 1) & (blocks <= 3)
    has_state = np.any(state_pixels, axis=2)
    low_values = np.where(state_pixels, blocks, 255)
    high_values = np.where(state_pixels, blocks, 0)
    low, high = low_values.min(axis=2), high_values.max(axis=2)
    result = np.zeros(low.shape, dtype=np.uint8)
    single = has_state & (low == high)
    result[single] = low[single]
    result[unresolved | (has_state & (low != high))] = STATE_CODES["UNRESOLVED"]
    return result


def _write_vectors(path: Path, result: Reconstruction,
                   alignment_fields: tuple[LocalDisplacementField, ...],
                   roof_alignment_records: list[_RoofAlignmentProduct],
                   canonical_rows: list[dict[str, Any]],
                   canonical_evidence: list[dict[str, Any]]) -> None:
    applied_alignment_cells = {
        (obs.image_alignment_field_id, obs.image_alignment_cell_id)
        for obs in result.observations if obs.registered_roof is not None
    }
    schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
        "physical_id": "int", "source_ref": "str:254", "status": "str:40",
        "area_m2": "float", "sigma_m": "float"}}
    with fiona.open(path, "w", driver="GPKG", layer="roof_candidates", schema=schema, crs="EPSG:32649") as sink:
        for obs in result.observations:
            sink.write({"geometry": mapping(obs.roof), "properties": {
                "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                "physical_id": obs.physical_id, "source_ref": obs.source_ref,
                "status": obs.footprint_status, "area_m2": obs.roof.area,
                "sigma_m": obs.registration_sigma_m}})
    raw_schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:32", "actual_date": "str:16",
        "date_status": "str:48", "tile_id": "str:64", "source_ref": "str:1024",
        "image_path": "str:1024", "source_tile_ids_json": "str:8192",
        "source_image_paths_json": "str:32767", "source_hashes_json": "str:8192",
        "crs": "str:16", "geometry_semantics": "str:64",
        "area_m2": "float", "status": "str:48"}}
    with fiona.open(path, "w", driver="GPKG", layer="roof_raw",
                    schema=raw_schema, crs="EPSG:32649") as sink:
        for obs in result.observations:
            sink.write({"geometry": mapping(obs.roof), "properties": {
                "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                "date_status": obs.date_status, "tile_id": obs.tile_id,
                "source_ref": obs.source_ref, "image_path": obs.image_path,
                "source_tile_ids_json": json.dumps(obs.source_tile_ids, ensure_ascii=False),
                "source_image_paths_json": json.dumps(obs.source_image_paths, ensure_ascii=False),
                "source_hashes_json": json.dumps(obs.source_hashes, ensure_ascii=False),
                "crs": "EPSG:32649", "geometry_semantics": "IMAGE_TRACED_ROOF_BOUNDARY",
                "area_m2": obs.roof.area, "status": "RAW_UNCORRECTED_ROOF"}})
    ground_schema = {"geometry": "Unknown", "properties": {
        **schema["properties"], "method": "str:40", "reviewer_id": "str:96",
        "reviewed_at": "str:40", "evidence_path": "str:1024",
        "evidence_sha256": "str:64", "independent_ground_evidence": "int",
        "reviewed": "int", "roof_to_ground_dx_m": "float",
        "roof_to_ground_dy_m": "float", "confirmed_identity_key": "str:96",
        "identity_reviewed": "int"}}
    with fiona.open(path, "w", driver="GPKG", layer="ground_footprints", schema=ground_schema, crs="EPSG:32649") as sink:
        for obs in result.observations:
            if obs.ground_footprint is None:
                continue
            evidence = obs.footprint_evidence
            sink.write({"geometry": mapping(obs.ground_footprint), "properties": {
                "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                "physical_id": obs.physical_id, "source_ref": evidence.source_ref if evidence else "",
                "status": obs.footprint_status, "area_m2": obs.ground_footprint.area,
                "sigma_m": evidence.positional_sigma_m if evidence else None,
                "method": evidence.method if evidence else "",
                "reviewer_id": evidence.reviewer_id if evidence else "",
                "reviewed_at": evidence.reviewed_at if evidence else "",
                "evidence_path": evidence.evidence_path if evidence else "",
                "evidence_sha256": evidence.evidence_sha256 if evidence else "",
                "independent_ground_evidence": int(evidence.independent_ground_evidence) if evidence else 0,
                "reviewed": int(evidence.reviewed) if evidence else 0,
                "roof_to_ground_dx_m": evidence.roof_to_ground_dx_m if evidence else None,
                "roof_to_ground_dy_m": evidence.roof_to_ground_dy_m if evidence else None,
                "confirmed_identity_key": obs.confirmed_identity_key,
                "identity_reviewed": int(obs.identity_reviewed)}})
    with fiona.open(path, "w", driver="GPKG", layer="instance_repair_candidates", schema=schema, crs="EPSG:32649") as sink:
        for child in result.repair_candidates:
            geometry = child["geometry"]
            sink.write({"geometry": mapping(geometry), "properties": {
                "uid": f"{child['merged_uid']}#{child['child_index']}", "epoch": "",
                "actual_date": None, "physical_id": None,
                "source_ref": child["prior_uid"], "status": child["status"],
                "area_m2": geometry.area, "sigma_m": None}})
    candidate_schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
        "candidate_id": "str:80", "source_ref": "str:254", "source_crs": "str:32",
        "source_date": "str:16", "source_epoch_label": "str:32",
        "source_date_status": "str:48", "source_date_candidates_json": "str:256",
        "source_tile_ids_json": "str:1024", "source_tile_actual_dates_json": "str:4096",
        "source_scene_key": "str:24",
        "source_scene_linkage_status": "str:80", "source_date_provenance": "str:512",
        "source_geometry_semantics": "str:48",
        "source_geometry_semantics_provenance": "str:64",
        "source_accuracy_m": "float", "association": "str:40",
        "source_properties_json": "str:4096", "match_iou": "float",
        "intersection_area_m2": "float", "roof_overlap_fraction": "float",
        "candidate_overlap_fraction": "float", "candidate_centroid_dx_from_roof_m": "float",
        "candidate_centroid_dy_from_roof_m": "float", "centroid_distance_m": "float",
        "candidate_polygon_distance_m": "float", "candidate_search_radius_m": "float",
        "centroid_aligned_roof_iou": "float", "centroid_aligned_iou_gain": "float",
        "status": "str:40", "area_m2": "float"}}
    with fiona.open(path, "w", driver="GPKG", layer="footprint_candidates", schema=candidate_schema, crs="EPSG:32649") as sink:
        for obs in result.observations:
            for candidate in obs.footprint_candidates:
                geometry = candidate.geometry
                sink.write({"geometry": mapping(geometry), "properties": {
                    "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                    "candidate_id": candidate.candidate_id, "source_ref": candidate.source_ref,
                    "source_crs": candidate.source_crs, "source_date": candidate.source_capture_date,
                    "source_date_status": candidate.source_date_status,
                    "source_date_candidates_json": json.dumps(
                        candidate.source_date_candidates, ensure_ascii=False),
                    "source_tile_ids_json": json.dumps(
                        candidate.source_tile_ids, ensure_ascii=False),
                    "source_tile_actual_dates_json": json.dumps([
                        {"tile_id": tile_id, "actual_date": actual_date}
                        for tile_id, actual_date in candidate.source_tile_actual_dates
                    ], ensure_ascii=False),
                    "source_scene_key": candidate.source_scene_key,
                    "source_scene_linkage_status": candidate.source_scene_linkage_status,
                    "source_date_provenance": candidate.source_date_provenance,
                    "source_epoch_label": candidate.source_epoch_label,
                    "source_geometry_semantics": candidate.source_geometry_semantics,
                    "source_geometry_semantics_provenance": candidate.source_geometry_semantics_provenance,
                    "source_accuracy_m": candidate.source_accuracy_m,
                    "association": candidate.association_method,
                    "source_properties_json": json.dumps(candidate.source_properties, ensure_ascii=False, sort_keys=True),
                    "match_iou": candidate.match_iou,
                    "intersection_area_m2": candidate.intersection_area_m2,
                    "roof_overlap_fraction": candidate.roof_overlap_fraction,
                    "candidate_overlap_fraction": candidate.candidate_overlap_fraction,
                    "candidate_centroid_dx_from_roof_m": candidate.candidate_centroid_dx_from_roof_m,
                    "candidate_centroid_dy_from_roof_m": candidate.candidate_centroid_dy_from_roof_m,
                    "centroid_distance_m": candidate.centroid_distance_m,
                    "candidate_polygon_distance_m": candidate.candidate_polygon_distance_m,
                    "candidate_search_radius_m": candidate.candidate_search_radius_m,
                    "centroid_aligned_roof_iou": candidate.centroid_aligned_roof_iou,
                    "centroid_aligned_iou_gain": candidate.centroid_aligned_iou_gain,
                    "status": "UNVERIFIED_CANDIDATE", "area_m2": geometry.area}})

            evidence = obs.footprint_evidence
            if evidence is not None and not evidence.accepted:
                geometry = evidence.geometry
                sink.write({"geometry": mapping(geometry), "properties": {
                    "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                    "candidate_id": "FOOTPRINT_EVIDENCE", "source_ref": evidence.source_ref,
                    "source_crs": evidence.crs, "source_date": None,
                    "source_date_status": "NOT_APPLICABLE",
                    "source_date_candidates_json": "[]", "source_tile_ids_json": "[]",
                    "source_tile_actual_dates_json": "[]",
                    "source_scene_key": None,
                    "source_scene_linkage_status": "NOT_APPLICABLE",
                    "source_date_provenance": None,
                    "source_epoch_label": None,
                    "source_geometry_semantics": None,
                    "source_geometry_semantics_provenance": None,
                    "source_accuracy_m": evidence.positional_sigma_m,
                    "association": evidence.method, "source_properties_json": "{}", "match_iou": None,
                    "intersection_area_m2": None, "roof_overlap_fraction": None,
                    "candidate_overlap_fraction": None, "candidate_centroid_dx_from_roof_m": None,
                    "candidate_centroid_dy_from_roof_m": None, "centroid_distance_m": None,
                    "candidate_polygon_distance_m": None, "candidate_search_radius_m": None,
                    "centroid_aligned_roof_iou": None, "centroid_aligned_iou_gain": None,
                    "status": "UNVERIFIED_CANDIDATE", "area_m2": geometry.area}})
            elif obs.footprint_candidate is not None and not obs.footprint_candidates:
                geometry = obs.footprint_candidate
                sink.write({"geometry": mapping(geometry), "properties": {
                    "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                    "candidate_id": "UNSPECIFIED_CANDIDATE", "source_ref": obs.source_ref,
                    "source_crs": "EPSG:32649", "source_date": None,
                    "source_date_status": "NOT_APPLICABLE",
                    "source_date_candidates_json": "[]", "source_tile_ids_json": "[]",
                    "source_tile_actual_dates_json": "[]",
                    "source_scene_key": None,
                    "source_scene_linkage_status": "NOT_APPLICABLE",
                    "source_date_provenance": None,
                    "source_epoch_label": None,
                    "source_geometry_semantics": None,
                    "source_geometry_semantics_provenance": None,
                    "source_accuracy_m": None, "association": "UNSPECIFIED",
                    "source_properties_json": "{}",
                    "match_iou": None, "intersection_area_m2": None,
                    "roof_overlap_fraction": None, "candidate_overlap_fraction": None,
                    "candidate_centroid_dx_from_roof_m": None,
                    "candidate_centroid_dy_from_roof_m": None, "centroid_distance_m": None,
                    "candidate_polygon_distance_m": None, "candidate_search_radius_m": None,
                    "centroid_aligned_roof_iou": None, "centroid_aligned_iou_gain": None,
                    "status": "UNVERIFIED_CANDIDATE",
                    "area_m2": geometry.area}})

    roof_source_schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
        "candidate_id": "str:80", "source_ref": "str:254",
        "candidate_method": "str:80", "source_properties_json": "str:4096"}}
    with fiona.open(path, "w", driver="GPKG", layer="footprint_candidate_roof_sources",
                    schema=roof_source_schema, crs="EPSG:32649") as sink:
        for obs in result.observations:
            for candidate in obs.footprint_candidates:
                if candidate.source_roof_geometry is None:
                    continue
                sink.write({"geometry": mapping(candidate.source_roof_geometry), "properties": {
                    "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                    "candidate_id": candidate.candidate_id, "source_ref": candidate.source_ref,
                    "candidate_method": str(candidate.source_properties.get("candidate_method", "")),
                    "source_properties_json": json.dumps(
                        candidate.source_properties, ensure_ascii=False, sort_keys=True)}})

    alignment_schema = {"geometry": "Unknown", "properties": {
        "field_id": "str:48", "cell_id": "str:80", "reference_date": "str:16",
        "reference_label": "str:80", "reference_source": "str:254",
        "reference_date_status": "str:48", "moving_date": "str:16",
        "status": "str:48", "method": "str:64",
        "source_ref": "str:254", "correction_dx_m": "float", "correction_dy_m": "float",
        "fit_disp_m": "float", "position_unc_m": "float", "reviewed": "int",
        "independent_ground": "int", "accepted": "int", "applied": "int",
        "max_masked_n": "int", "min_required_n": "int",
        "sample_support": "str:40", "anchor_count": "int",
        "anchor_years": "str:48", "anchor_median_corr": "float",
        "anchor_median_margin": "float", "fit_dispersion_kind": "str:64",
        "alignment_review_basis": "str:48", "alignment_evidence_type": "str:64",
        "relative_unc_m": "float", "nonbuilding_tie_pairs": "int",
        "holdout_tie_pairs": "int", "ground_evidence_type": "str:48",
        "reviewer_id": "str:96",
        "reviewed_at": "str:40", "review_evidence_path": "str:512",
        "review_evidence_sha256": "str:64", "reference_sha256": "str:64",
        "scope": "str:40", "tile_mosaic_source": "str:512",
        "tile_mosaic_sha256": "str:64", "tile_col0": "int",
        "tile_row0_north_to_south": "int", "patch_tiles": "int",
        "tile_window_json": "str:128", "pixel_window_json": "str:128",
        "tile_prov_json": "str:32767", "reference_anchor_count": "int",
        "reference_anchor_years": "str:64",
        "source_component_scope": "str:48",
        "source_component_ids_json": "str:8192",
        "source_component_coverage": "float",
        "candidate_status": "str:80",
        "reference_component_id_status": "str:96",
        "cycle_closure_max_m": "float",
        "source_edge_count": "int",
        "relative_ties_within_support_cell": "int",
        "relative_cell_scope_review_authorized": "int",
        "reference_window_evidence_json": "str:32767"}}
    with fiona.open(path, "w", driver="GPKG", layer="image_alignment_candidates",
                    schema=alignment_schema, crs="EPSG:32649") as sink:
        for field in alignment_fields:
            for cell in field.cells:
                sink.write({"geometry": mapping(cell.support_geometry), "properties": {
                    "field_id": field.field_id, "cell_id": cell.cell_id,
                    "reference_date": field.reference_date, "moving_date": field.moving_date,
                    "reference_label": field.reference_label,
                    "reference_source": field.reference_source,
                    "reference_date_status": field.reference_date_status,
                    "status": cell.status, "method": cell.method, "source_ref": cell.source_ref,
                    "correction_dx_m": cell.correction_dx_m_candidate,
                    "correction_dy_m": cell.correction_dy_m_candidate,
                    "fit_disp_m": cell.fit_dispersion_m,
                    "position_unc_m": cell.positional_uncertainty_m,
                    "reviewed": int(cell.reviewed),
                    "independent_ground": int(cell.independent_ground_evidence),
                    "accepted": int(cell.accepted),
                    "applied": int((field.field_id, cell.cell_id) in applied_alignment_cells),
                    "max_masked_n": cell.metadata.get("maximum_masked_samples_available"),
                    "min_required_n": cell.metadata.get("minimum_samples_required"),
                    "sample_support": cell.metadata.get("sample_support_status"),
                    "anchor_count": (cell.metadata.get("anchor_summary") or {}).get("anchor_count"),
                    "anchor_years": ",".join(str(year) for year in
                                               (cell.metadata.get("anchor_summary") or {}).get("reference_years", [])) or None,
                    "anchor_median_corr": (cell.metadata.get("anchor_summary") or {}).get("median_correlation"),
                    "anchor_median_margin": (cell.metadata.get("anchor_summary") or {}).get("median_peak_margin"),
                    "fit_dispersion_kind": (
                        "CROSS_ANCHOR_MEDIAN_RADIAL_DEVIATION_NOT_POSITIONAL_UNCERTAINTY"
                        if cell.method == "MULTI_ANCHOR_BASE_TIE_CONSENSUS" else
                        "POSE_GRAPH_CYCLE_CLOSURE_NOT_POSITIONAL_UNCERTAINTY"
                        if cell.method == "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE"
                        else None),
                    "alignment_review_basis": cell.alignment_review_basis,
                    "alignment_evidence_type": cell.metadata.get("alignment_evidence_type"),
                    "relative_unc_m": cell.relative_registration_uncertainty_m,
                    "nonbuilding_tie_pairs": cell.metadata.get("nonbuilding_tie_pairs"),
                    "holdout_tie_pairs": cell.metadata.get("holdout_tie_pairs"),
                    "ground_evidence_type": cell.metadata.get("ground_evidence_type"),
                    "reviewer_id": cell.metadata.get("reviewer_id"),
                    "reviewed_at": cell.metadata.get("reviewed_at"),
                    "review_evidence_path": cell.metadata.get("review_evidence_path"),
                    "review_evidence_sha256": cell.metadata.get("review_evidence_sha256"),
                    "reference_sha256": cell.metadata.get("review_reference_sha256"),
                    "candidate_status": cell.metadata.get("candidate_status", cell.status),
                    "reference_component_id_status": cell.metadata.get(
                        "reference_component_id_status",
                        "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
                    "cycle_closure_max_m": cell.metadata.get("closure_max_m"),
                    "source_edge_count": cell.metadata.get("source_edge_count"),
                    "relative_ties_within_support_cell": int(
                        cell.metadata.get("relative_ties_within_support_cell") is True),
                    "relative_cell_scope_review_authorized": int(
                        cell.relative_cell_source_scope_review_authorized),
                    "scope": field.correction_scope,
                    **_source_component_properties(cell),
                    **_source_tile_properties(cell),
                    **_reference_window_properties(cell)}})
    roof_alignment_schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
        "field_id": "str:80", "cell_id": "str:80", "status": "str:64",
        "geometry_basis": "str:48", "ground_status": "str:32",
        "candidate_shift": "int", "candidate_dx_m": "float", "candidate_dy_m": "float",
        "grid_unc_m": "float", "relative_unc_m": "float",
        "cell_support_fraction": "float",
        "reference_label": "str:80", "reference_source": "str:254",
        "roof_source_ref": "str:254", "alignment_source_ref": "str:254",
        "raw_area_m2": "float", "candidate_area_m2": "float",
        "tile_mosaic_source": "str:512", "tile_mosaic_sha256": "str:64",
        "tile_col0": "int", "tile_row0_north_to_south": "int",
        "patch_tiles": "int", "tile_window_json": "str:128",
        "pixel_window_json": "str:128", "tile_prov_json": "str:32767",
        "reference_anchor_count": "int", "reference_anchor_years": "str:64",
        "source_component_scope": "str:48",
        "source_component_ids_json": "str:8192",
        "source_component_coverage": "float",
        "candidate_status": "str:80",
        "reference_component_id_status": "str:96",
        "cycle_closure_max_m": "float",
        "source_edge_count": "int",
        "relative_ties_within_support_cell": "int",
        "relative_cell_scope_review_authorized": "int",
        "reference_window_evidence_json": "str:32767"}}
    with fiona.open(path, "w", driver="GPKG", layer="image_grid_roof_candidates",
                    schema=roof_alignment_schema, crs="EPSG:32649") as sink:
        for record in roof_alignment_records:
            obs, field, candidate = record.observation, record.field, record.candidate
            cell = next((item for item in field.cells if item.cell_id == candidate.cell_id), None)
            sink.write({"geometry": mapping(candidate.geometry), "properties": {
                "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                "field_id": field.field_id, "cell_id": candidate.cell_id or "",
                "status": candidate.status,
                "geometry_basis": "ROOF_IMAGE_GRID_CANDIDATE",
                "ground_status": "NOT_A_GROUND_FOOTPRINT",
                "candidate_shift": int(candidate.candidate_shift_rendered),
                "candidate_dx_m": candidate.correction_dx_m_candidate,
                "candidate_dy_m": candidate.correction_dy_m_candidate,
                "grid_unc_m": candidate.grid_positional_uncertainty_m,
                "relative_unc_m": candidate.relative_registration_uncertainty_m,
                "cell_support_fraction": candidate.cell_support_fraction,
                "reference_label": field.reference_label or "",
                "reference_source": field.reference_source or "",
                "roof_source_ref": obs.source_ref,
                "alignment_source_ref": cell.source_ref if cell else field.reference_source or "",
                "raw_area_m2": obs.roof.area,
                "candidate_area_m2": candidate.geometry.area,
                "candidate_status": cell.metadata.get("candidate_status", cell.status)
                    if cell else None,
                "reference_component_id_status": cell.metadata.get(
                    "reference_component_id_status") if cell else None,
                "cycle_closure_max_m": cell.metadata.get("closure_max_m") if cell else None,
                "source_edge_count": cell.metadata.get("source_edge_count") if cell else None,
                "relative_ties_within_support_cell": int(
                    cell.metadata.get("relative_ties_within_support_cell") is True)
                    if cell else 0,
                "relative_cell_scope_review_authorized": int(
                    cell.relative_cell_source_scope_review_authorized) if cell else 0,
                **_source_component_properties(cell),
                **_source_tile_properties(cell),
                **_reference_window_properties(cell)}})

    outline_diagnostic_schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
        "candidate_identity": "str:254", "candidate_id": "str:80",
        "candidate_source_ref": "str:1024", "candidate_source_crs": "str:32",
        "candidate_source_date": "str:16", "candidate_source_accuracy_m": "float",
        "roof_source_ref": "str:1024",
        "field_id": "str:128", "cell_id": "str:80", "reference_date": "str:16",
        "reference_source": "str:512", "alignment_status": "str:64",
        "candidate_shift": "int", "candidate_dx_m": "float",
        "candidate_dy_m": "float", "grid_unc_m": "float",
        "relative_unc_m": "float",
        "raw_iou": "float", "image_grid_iou": "float", "image_grid_iou_gain": "float",
        "raw_intersection_m2": "float", "image_grid_intersection_m2": "float",
        "raw_distance_m": "float", "image_grid_distance_m": "float",
        "outline_dx_from_shifted_m": "float", "outline_dy_from_shifted_m": "float",
        "alignment_cell_support_fraction": "float",
        "geometry_basis": "str:40", "ground_status": "str:24",
        "formal_ground_use": "int", "formal_change_use": "int"}}
    with fiona.open(path, "w", driver="GPKG",
                    layer="image_grid_roof_outline_diagnostics",
                    schema=outline_diagnostic_schema, crs="EPSG:32649") as sink:
        for record in roof_alignment_records:
            obs, field, aligned = record.observation, record.field, record.candidate
            for external in obs.footprint_candidates:
                raw_iou, raw_intersection, raw_distance = _polygon_comparison_metrics(
                    obs.roof, external.geometry)
                if aligned.candidate_shift_rendered:
                    image_iou, image_intersection, image_distance = _polygon_comparison_metrics(
                        aligned.geometry, external.geometry)
                    outline_centroid = external.geometry.centroid
                    roof_centroid = aligned.geometry.centroid
                    residual_dx = float(outline_centroid.x - roof_centroid.x)
                    residual_dy = float(outline_centroid.y - roof_centroid.y)
                    iou_gain = image_iou - raw_iou
                else:
                    image_iou = image_intersection = image_distance = None
                    residual_dx = residual_dy = iou_gain = None
                sink.write({"geometry": mapping(external.geometry), "properties": {
                    "uid": obs.uid, "epoch": obs.epoch_label,
                    "actual_date": obs.actual_date,
                    "candidate_identity": obs.candidate_identity,
                    "candidate_id": external.candidate_id,
                    "candidate_source_ref": external.source_ref,
                    "candidate_source_crs": external.source_crs,
                    "candidate_source_date": external.source_capture_date,
                    "candidate_source_accuracy_m": external.source_accuracy_m,
                    "roof_source_ref": obs.source_ref,
                    "field_id": field.field_id, "cell_id": aligned.cell_id or "",
                    "reference_date": field.reference_date,
                    "reference_source": field.reference_source or "",
                    "alignment_status": aligned.status,
                    "candidate_shift": int(aligned.candidate_shift_rendered),
                    "candidate_dx_m": aligned.correction_dx_m_candidate,
                    "candidate_dy_m": aligned.correction_dy_m_candidate,
                    "grid_unc_m": aligned.grid_positional_uncertainty_m,
                    "relative_unc_m": aligned.relative_registration_uncertainty_m,
                    "raw_iou": raw_iou, "image_grid_iou": image_iou,
                    "image_grid_iou_gain": iou_gain,
                    "raw_intersection_m2": raw_intersection,
                    "image_grid_intersection_m2": image_intersection,
                    "raw_distance_m": raw_distance,
                    "image_grid_distance_m": image_distance,
                    "outline_dx_from_shifted_m": residual_dx,
                    "outline_dy_from_shifted_m": residual_dy,
                    "alignment_cell_support_fraction": aligned.cell_support_fraction,
                    "geometry_basis": "UNVERIFIED_EXTERNAL_OUTLINE",
                    "ground_status": "NOT_VERIFIED",
                    "formal_ground_use": 0, "formal_change_use": 0}})

    registered_schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
        "field_id": "str:120", "cell_id": "str:120", "status": "str:64",
        "geometry_basis": "str:48", "correction_dx_m": "float",
        "correction_dy_m": "float", "position_unc_m": "float",
        "relative_unc_m": "float", "alignment_review_basis": "str:48",
        "alignment_evidence_type": "str:64", "nonbuilding_tie_pairs": "int",
        "holdout_tie_pairs": "int", "independent_ground": "int",
        "reference_source": "str:512", "alignment_source_ref": "str:254",
        "roof_source_ref": "str:254", "ground_evidence_type": "str:48",
        "reviewer_id": "str:96", "reviewed_at": "str:40",
        "review_evidence_path": "str:512", "review_evidence_sha256": "str:64",
        "reference_sha256": "str:64", "tile_mosaic_source": "str:512",
        "tile_mosaic_sha256": "str:64", "tile_col0": "int",
        "tile_row0_north_to_south": "int", "patch_tiles": "int",
        "tile_window_json": "str:128", "pixel_window_json": "str:128",
        "tile_prov_json": "str:32767", "reference_anchor_count": "int",
        "reference_anchor_years": "str:64",
        "source_component_scope": "str:48",
        "source_component_ids_json": "str:8192",
        "source_component_coverage": "float",
        "candidate_status": "str:80",
        "reference_component_id_status": "str:96",
        "cycle_closure_max_m": "float",
        "source_edge_count": "int",
        "relative_ties_within_support_cell": "int",
        "relative_cell_scope_review_authorized": "int",
        "reference_window_evidence_json": "str:32767"}}
    cells_by_key = {(field.field_id, cell.cell_id): (field, cell)
                    for field in alignment_fields for cell in field.cells}
    with fiona.open(path, "w", driver="GPKG", layer="registered_roof_candidates",
                    schema=registered_schema, crs="EPSG:32649") as sink:
        for obs in result.observations:
            if obs.registered_roof is None:
                continue
            field, cell = cells_by_key[(obs.image_alignment_field_id,
                                        obs.image_alignment_cell_id)]
            metadata = cell.metadata
            sink.write({"geometry": mapping(obs.registered_roof), "properties": {
                "uid": obs.uid, "epoch": obs.epoch_label,
                "actual_date": obs.actual_date, "field_id": field.field_id,
                "cell_id": cell.cell_id, "status": obs.image_alignment_status,
                "geometry_basis": "IMAGE_GRID_ALIGNED_ROOF_PROXY_NOT_GROUND",
                "correction_dx_m": obs.image_alignment_dx_m,
                "correction_dy_m": obs.image_alignment_dy_m,
                "position_unc_m": obs.image_alignment_positional_uncertainty_m,
                "relative_unc_m": obs.image_alignment_relative_uncertainty_m,
                "alignment_review_basis": (obs.image_alignment_review_basis
                                            or metadata.get("review_basis")),
                "alignment_evidence_type": metadata.get("alignment_evidence_type"),
                "nonbuilding_tie_pairs": metadata.get("nonbuilding_tie_pairs"),
                "holdout_tie_pairs": metadata.get("holdout_tie_pairs"),
                "independent_ground": int(cell.independent_ground_evidence),
                "reference_source": field.reference_source or "",
                "alignment_source_ref": cell.source_ref,
                "roof_source_ref": obs.source_ref,
                "ground_evidence_type": metadata.get("ground_evidence_type"),
                "reviewer_id": metadata.get("reviewer_id"),
                "reviewed_at": metadata.get("reviewed_at"),
                "review_evidence_path": metadata.get("review_evidence_path"),
                "review_evidence_sha256": metadata.get("review_evidence_sha256"),
                "reference_sha256": metadata.get("review_reference_sha256"),
                "candidate_status": metadata.get("candidate_status", cell.status),
                "reference_component_id_status": metadata.get(
                    "reference_component_id_status",
                    "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
                "cycle_closure_max_m": metadata.get("closure_max_m"),
                "source_edge_count": metadata.get("source_edge_count"),
                "relative_ties_within_support_cell": int(
                    metadata.get("relative_ties_within_support_cell") is True),
                "relative_cell_scope_review_authorized": int(
                    cell.relative_cell_source_scope_review_authorized),
                **_source_component_properties(cell),
                **_source_tile_properties(cell),
                **_reference_window_properties(cell)}})

    roof_registered_schema = {"geometry": "Unknown", "properties": {
        "raw_roof_id": "str:96", "raw_geometry_hash": "str:64",
        "transform_type": "str:32", "registration_status": "str:32",
        "uid": "str:96", "epoch": "str:32", "actual_date": "str:16",
        "field_id": "str:120", "cell_id": "str:120", "status": "str:64",
        "geometry_basis": "str:48", "correction_dx_m": "float",
        "correction_dy_m": "float", "position_unc_m": "float",
        "relative_unc_m": "float", "alignment_review_basis": "str:48",
        "reference_source": "str:1024", "alignment_source_ref": "str:1024",
        "roof_source_ref": "str:1024", "reviewer_id": "str:96",
        "reviewed_at": "str:40", "review_evidence_path": "str:1024",
        "review_evidence_sha256": "str:64", "reference_sha256": "str:64",
        "source_tile_ids_json": "str:8192", "source_image_paths_json": "str:32767",
        "source_hashes_json": "str:8192"}}
    with fiona.open(path, "w", driver="GPKG", layer="roof_registered",
                    schema=roof_registered_schema, crs="EPSG:32649") as sink:
        for obs in result.observations:
            if obs.registered_roof is None:
                continue
            field, cell = cells_by_key[(obs.image_alignment_field_id,
                                        obs.image_alignment_cell_id)]
            metadata = cell.metadata
            sink.write({"geometry": mapping(obs.registered_roof), "properties": {
                "uid": obs.uid, "epoch": obs.epoch_label, "actual_date": obs.actual_date,
                "raw_roof_id": obs.uid,
                "raw_geometry_hash": hashlib.sha256(obs.roof.wkb).hexdigest(),
                "transform_type": "TRANSLATION",
                "registration_status": "ACCEPTED_REGISTERED",
                "field_id": field.field_id, "cell_id": cell.cell_id,
                "status": obs.image_alignment_status,
                "geometry_basis": "IMAGE_GRID_ALIGNED_ROOF_PROXY_NOT_GROUND",
                "correction_dx_m": obs.image_alignment_dx_m,
                "correction_dy_m": obs.image_alignment_dy_m,
                "position_unc_m": obs.image_alignment_positional_uncertainty_m,
                "relative_unc_m": obs.image_alignment_relative_uncertainty_m,
                "alignment_review_basis": obs.image_alignment_review_basis or metadata.get("review_basis"),
                "reference_source": field.reference_source or "",
                "alignment_source_ref": cell.source_ref,
                "roof_source_ref": obs.source_ref,
                "reviewer_id": metadata.get("reviewer_id"),
                "reviewed_at": metadata.get("reviewed_at"),
                "review_evidence_path": metadata.get("review_evidence_path"),
                "review_evidence_sha256": metadata.get("review_evidence_sha256"),
                "reference_sha256": metadata.get("review_reference_sha256"),
                "source_tile_ids_json": json.dumps(obs.source_tile_ids),
                "source_image_paths_json": json.dumps(obs.source_image_paths),
                "source_hashes_json": json.dumps(obs.source_hashes)}})

    observations_by_uid = {obs.uid: obs for obs in result.observations}
    node_schema = {"geometry": "Unknown", "properties": {
        "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
        "candidate_identity": "str:254", "confirmed_identity_key": "str:96",
        "identity_reviewed": "int", "identity_evidence_type": "str:64",
        "identity_evidence_ref": "str:1024", "identity_evidence_sha256": "str:64",
        "physical_id": "int", "status": "str:48",
        "geometry_status": "str:48", "geometry_basis": "str:48"}}
    with fiona.open(path, "w", driver="GPKG", layer="identity_graph_nodes",
                    schema=node_schema, crs="EPSG:32649") as sink:
        for row in result.identity_graph:
            obs = observations_by_uid[row["uid"]]
            ground_based = obs.ground_footprint is not None
            geometry = obs.tracking_geometry
            sink.write({"geometry": mapping(geometry), "properties": {
                "uid": obs.uid, "epoch": obs.epoch_label,
                "actual_date": obs.actual_date,
                "candidate_identity": row.get("candidate_identity"),
                "confirmed_identity_key": (obs.confirmed_identity_key
                                           if reviewed_identity_is_valid(obs) else None),
                "identity_reviewed": int(reviewed_identity_is_valid(obs)),
                "identity_evidence_type": obs.identity_evidence_type,
                "identity_evidence_ref": obs.identity_evidence_ref,
                "identity_evidence_sha256": obs.identity_evidence_sha256,
                "physical_id": row.get("physical_id"), "status": row["status"],
                "geometry_status": ("REVIEWED_GROUND_FOOTPRINT" if ground_based else
                                    "ACCEPTED_REGISTERED_ROOF_NOT_GROUND"
                                    if obs.registered_roof is not None else "RAW_ROOF_PROXY_ONLY"),
                "geometry_basis": ("GROUND_VERIFIED" if ground_based else
                                   "IMAGE_GRID_ROOF" if obs.registered_roof is not None else
                                   "ROOF_PROXY")}})

    edge_schema = {"geometry": "Unknown", "properties": {
        "from_uid": "str:254", "to_uid": "str:254",
        "from_date": "str:16", "to_date": "str:16",
        "relation": "str:16", "affinity": "float",
        "search_radius_m": "float", "candidate_count": "int",
        "candidate_rank": "int", "temporal_valid": "int",
        "status": "str:40", "geometry_basis": "str:24",
        "resolver_stage": "str:64", "edge_basis": "str:64",
        "candidate_reference_json": "str",
        "hard_negative_json": "str:2048"}}
    with fiona.open(path, "w", driver="GPKG", layer="candidate_identity_edges",
                    schema=edge_schema, crs="EPSG:32649") as sink:
        for edge in result.candidate_edges:
            previous = observations_by_uid[edge["from_uid"]]
            current = observations_by_uid.get(edge["to_uid"] or "")
            use_ground = (edge["geometry_basis"] == "GROUND" and current is not None
                          and previous.ground_footprint is not None
                          and current.ground_footprint is not None)
            left_geometry = (previous.ground_footprint if use_ground else
                             previous.registered_roof or previous.roof)
            right_geometry = ((current.ground_footprint if use_ground else
                               current.registered_roof or current.roof)
                              if current is not None else None)
            left = left_geometry.centroid
            if right_geometry is None:
                geometry = left
            else:
                right = right_geometry.centroid
                geometry = (Point(left.x, left.y) if left.equals(right)
                            else LineString([(left.x, left.y), (right.x, right.y)]))
            sink.write({"geometry": mapping(geometry), "properties": {
                "from_uid": edge["from_uid"], "to_uid": edge["to_uid"],
                "from_date": edge["from_date"], "to_date": edge["to_date"],
                "relation": edge["relation"], "affinity": edge["affinity"],
                "search_radius_m": edge["search_radius_m"],
                "candidate_count": edge["candidate_count"],
                "candidate_rank": edge["candidate_rank"],
                "temporal_valid": int(edge["temporal_order_valid"]),
                "status": edge["status"],
                "geometry_basis": (
                    "GROUND" if use_ground else
                    "ROOF_IMAGE_GRID" if (previous.registered_roof is not None
                                           or (current is not None and current.registered_roof is not None))
                    else "ROOF_PROXY"),
                "resolver_stage": edge["resolver_stage"],
                "edge_basis": edge.get("edge_basis", "TBIA_GEOMETRIC"),
                "candidate_reference_json": json.dumps(
                    edge.get("candidate_references", []), ensure_ascii=False, sort_keys=True),
                "hard_negative_json": json.dumps(edge["hard_negative_ids"],
                                                  ensure_ascii=False)}})

    presence_schema = {"geometry": "None", "properties": {
        "observation_id": "str:96", "epoch_label": "str:32", "actual_date": "str:16",
        "identity_candidate": "str:254", "physical_id": "int",
        "confirmed_identity_key": "str:96", "identity_reviewed": "int",
        "identity_evidence_type": "str:64", "identity_evidence_ref": "str:1024",
        "identity_evidence_path": "str:1024", "identity_evidence_sha256": "str:64",
        "identity_reviewer_id": "str:96", "identity_reviewed_at": "str:40",
        "presence_state": "str:1", "presence_evidence_type": "str:64",
        "presence_evidence_ref": "str:1024", "coverage_complete": "int",
        "review_status": "str:32", "confidence": "float", "notes": "str:512",
        "footprint_source_ref": "str:1024", "footprint_method": "str:48",
        "footprint_sigma_m": "float", "image_ref": "str:1024", "reviewed": "int",
        "reviewer_id": "str:96", "reviewed_at": "str:40",
        "evidence_path": "str:1024", "evidence_sha256": "str:64"}}
    with fiona.open(path, "w", driver="GPKG", layer="building_presence",
                    schema=presence_schema) as sink:
        for row in result.presence:
            state = row.get("presence")
            if state not in {"1", "0", "U"}:
                raise ValueError(f"Unsupported presence state for {row.get('uid')}: {state!r}")
            sink.write({"geometry": None, "properties": {
                "observation_id": row.get("uid"), "epoch_label": row.get("epoch_label"),
                "actual_date": row.get("actual_date"),
                "identity_candidate": row.get("candidate_identity"),
                "physical_id": row.get("physical_id"), "presence_state": state,
                "confirmed_identity_key": row.get("confirmed_identity_key"),
                "identity_reviewed": int(bool(row.get("identity_reviewed"))),
                "identity_evidence_type": row.get("identity_evidence_type"),
                "identity_evidence_ref": row.get("identity_evidence_ref"),
                "identity_evidence_path": row.get("identity_evidence_path"),
                "identity_evidence_sha256": row.get("identity_evidence_sha256"),
                "identity_reviewer_id": row.get("identity_reviewer_id"),
                "identity_reviewed_at": row.get("identity_reviewed_at"),
                "presence_evidence_type": row.get("presence_evidence_type"),
                "presence_evidence_ref": row.get("presence_evidence_ref"),
                "coverage_complete": (
                    int(bool(row.get("coverage_complete")))
                    if row.get("coverage_complete") is not None else None),
                "review_status": row.get("review_status"),
                "confidence": row.get("confidence"), "notes": row.get("notes"),
                "footprint_source_ref": row.get("footprint_source_ref"),
                "footprint_method": row.get("footprint_method"),
                "footprint_sigma_m": row.get("footprint_sigma_m"),
                "image_ref": row.get("image_ref"),
                "reviewed": (int(bool(row.get("reviewed")))
                             if row.get("reviewed") is not None else None),
                "reviewer_id": row.get("reviewer_id"),
                "reviewed_at": row.get("reviewed_at"),
                "evidence_path": row.get("evidence_path"),
                "evidence_sha256": row.get("evidence_sha256")}})

    canonical_schema = {"geometry": "Unknown", "properties": {
        "physical_id": "int", "confirmed_identity_key": "str:96",
        "canonical_uid": "str:96", "canonical_date": "str:16",
        "support_dates_json": "str:2048", "support_count": "int",
        "canonical_status": "str:64", "geometry_basis": "str:64",
        "position_unc_m": "float", "ground_truth": "int"}}
    with fiona.open(path, "w", driver="GPKG", layer="canonical_building_geometry",
                    schema=canonical_schema, crs="EPSG:32649") as sink:
        for row in canonical_rows:
            geom = row["geometry"]
            sink.write({"geometry": mapping(geom) if geom is not None else None,
                        "properties": {
                            "physical_id": row["physical_id"],
                            "confirmed_identity_key": row["confirmed_identity_key"],
                            "canonical_uid": row["canonical_uid"],
                            "canonical_date": row["canonical_date"],
                            "support_dates_json": json.dumps(row["support_dates"]),
                            "support_count": row["support_count"],
                            "canonical_status": row["status"],
                            "geometry_basis": "REGISTERED_ROOF_MEDOID_NOT_GROUND",
                            "position_unc_m": row["position_uncertainty_m"],
                            "ground_truth": 0}})
    canonical_evidence_schema = {"geometry": "None", "properties": {
        "physical_id": "int", "actual_date": "str:16", "uid": "str:96",
        "selected_as_canonical": "int", "canonical_status": "str:64",
        "centroid_distance_m": "float", "iou": "float",
        "hausdorff_boundary_distance_m": "float", "area_ratio": "float",
        "shape_similarity": "float"}}
    with fiona.open(path, "w", driver="GPKG", layer="canonical_geometry_evidence",
                    schema=canonical_evidence_schema) as sink:
        for row in canonical_evidence:
            sink.write({"geometry": None, "properties": {
                **row, "selected_as_canonical": int(row["selected_as_canonical"])}})


def write_products(output_dir: Path, result: Reconstruction, *, resolution_m: float = 1.0,
                   min_area_m2: float = 100.0, anchor_raster: Path | None = None,
                   alignment_fields: Iterable[LocalDisplacementField] = (),
                   alignment_reference_rasters: Iterable[Path | str] = ()) -> dict[str, Any]:
    """Write candidate QA and formal products without ever rasterizing roofs as ground.

    Must be called only with a fresh output directory.  The 100 m² filter is
    applied here, after identity and state determination.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite reconstruction output: {output_dir}")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    alignment_fields = tuple(alignment_fields)
    alignment_reference_rasters = tuple(alignment_reference_rasters)
    roof_alignment_records = _roof_alignment_products(
        result, alignment_fields, anchor_raster, alignment_reference_rasters)
    outline_diagnostic_count = sum(
        len(record.observation.footprint_candidates)
        for record in roof_alignment_records)
    geoms = [g for obs in result.observations
             for g in (obs.roof, obs.registered_roof, obs.ground_footprint, obs.footprint_candidate,
                       *(candidate.geometry for candidate in obs.footprint_candidates)) if g is not None]
    geoms.extend(cell.support_geometry for field in alignment_fields for cell in field.cells)
    geoms.extend(record.candidate.geometry for record in roof_alignment_records)
    if not geoms:
        raise ValueError("No observations")
    xmin, ymin, xmax, ymax = unary_union(geoms).bounds
    padding = 10.0
    anchor_metadata = None
    anchor_resolution = None
    if anchor_raster is not None:
        with rasterio.open(anchor_raster) as reference:
            if reference.crs is None or reference.crs.to_epsg() != 32649:
                raise ValueError("Reference grid must be EPSG:32649")
            rx, ry = reference.res
            if abs(rx - ry) > 1e-8 or rx <= resolution_m:
                raise ValueError("Reference grid must be square and coarser than the output grid")
            factor = rx / resolution_m
            if abs(factor - round(factor)) > 1e-8:
                raise ValueError("Output resolution must divide the reference pixel size")
            if any(abs(v) > 1e-12 for v in (reference.transform.b, reference.transform.d)) or reference.transform.a <= 0 or reference.transform.e >= 0:
                raise ValueError("Rotated or south-up reference grids are not supported")
            anchor_resolution = float(rx)
            anchor_x, anchor_y = reference.transform.c, reference.transform.f
            west = anchor_x + math.floor((xmin - padding - anchor_x) / rx) * rx
            east = anchor_x + math.ceil((xmax + padding - anchor_x) / rx) * rx
            north = anchor_y - math.floor((anchor_y - (ymax + padding)) / ry) * ry
            south = anchor_y - math.ceil((anchor_y - (ymin - padding)) / ry) * ry
            anchor_metadata = {"path": str(anchor_raster), "crs": str(reference.crs),
                               "transform": list(reference.transform), "resolution_m": [rx, ry],
                               "width": reference.width, "height": reference.height,
                               "acquisition_date_status": "UNKNOWN_ANNUAL_FILE_LABEL_ONLY",
                               "base_window": [int(round((west - anchor_x) / rx)),
                                               int(round((anchor_y - north) / ry)),
                                               int(round((east - west) / rx)),
                                               int(round((north - south) / ry))]}
            col, row, cols, rows = anchor_metadata["base_window"]
            if col < 0 or row < 0 or col + cols > reference.width or row + rows > reference.height:
                raise ValueError("Output extent falls outside the reference 10m raster")
    else:
        west = math.floor((xmin - padding) / resolution_m) * resolution_m
        east = math.ceil((xmax + padding) / resolution_m) * resolution_m
        south = math.floor((ymin - padding) / resolution_m) * resolution_m
        north = math.ceil((ymax + padding) / resolution_m) * resolution_m
    width = int(round((east - west) / resolution_m))
    height = int(round((north - south) / resolution_m))
    if width * height > 25_000_000:
        raise ValueError("Fixed-case output exceeds 25 million pixels; partition before writing")
    transform = from_origin(west, north, resolution_m, resolution_m)
    shape = height, width
    output_dir.mkdir(parents=True)
    canonical_rows, canonical_evidence = _canonical_geometry_records(result)
    _write_vectors(output_dir / "building_lineage.gpkg", result, alignment_fields,
                   roof_alignment_records, canonical_rows, canonical_evidence)
    physical_id_codebook = {
        "schema_version": "xian_r5.physical_id_codebook.v1",
        "semantics": "REVIEWED_PHYSICAL_IDENTITY_KEY_HASH;NOT_CANDIDATE_TRACK_ID",
        "hash": "SHA256_FIRST_32_BITS_BIG_ENDIAN;ZERO_AND_UINT32_MAX_RESERVED",
        "code_to_reviewed_identity": {
            str(row["physical_id"]): {
                "identity_key": row["confirmed_identity_key"],
                "identity_evidence": sorted({
                    obs.identity_evidence_ref or ""
                    for obs in result.observations
                    if obs.physical_id == row["physical_id"] and reviewed_identity_is_valid(obs)
                }),
            }
            for row in canonical_rows
        },
    }
    physical_id_codebook_bytes = (json.dumps(
        physical_id_codebook, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    (output_dir / "PHYSICAL_ID_CODEBOOK.json").write_bytes(physical_id_codebook_bytes)
    physical_id_codebook_sha256 = hashlib.sha256(physical_id_codebook_bytes).hexdigest()
    nominal_epochs = result.epoch_order or sorted({o.epoch_label for o in result.observations})
    actual_dates_by_epoch = {
        epoch: sorted({o.actual_date for o in result.observations
                       if o.epoch_label == epoch and o.actual_date})
        for epoch in nominal_epochs
    }
    for row in result.presence:
        epoch = row.get("epoch_label")
        actual_date = row.get("actual_date")
        if epoch in actual_dates_by_epoch and actual_date:
            actual_dates_by_epoch[epoch] = sorted(
                set(actual_dates_by_epoch[epoch]) | {str(actual_date)})

    def canonical_date(value: Any) -> str | None:
        if not isinstance(value, str) or value in {"", "DATE_UNKNOWN", "NO_TILE_DATE_RECORD"}:
            return None
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            return None
        return parsed.isoformat() if parsed.isoformat() == value else None

    def safe_group_id(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
        return safe or "UNSPECIFIED"

    actual_date_order = list(result.actual_date_order)
    if not actual_date_order:
        actual_date_order = sorted({
            parsed for parsed in (canonical_date(obs.actual_date) for obs in result.observations)
            if parsed is not None
        } | {
            parsed for parsed in (canonical_date(row.get("actual_date")) for row in result.presence)
            if parsed is not None
        })
    records_by_group: dict[str, list[Observation]] = {}
    group_epoch_labels: dict[str, set[str]] = {}
    group_raw_dates: dict[str, set[str]] = {}
    unknown_group_by_epoch: dict[str, str] = {}
    unknown_group_owner: dict[str, str] = {}
    for obs in result.observations:
        observed_date = canonical_date(obs.actual_date)
        if observed_date is not None:
            group_id = observed_date
        else:
            group_id = unknown_group_by_epoch.get(obs.epoch_label)
            if group_id is None:
                group_id = f"DATE_UNKNOWN_{safe_group_id(obs.epoch_label)}"
                other_label = unknown_group_owner.get(group_id)
                if other_label is not None and other_label != obs.epoch_label:
                    raise ValueError(
                        "Unresolved-date output labels collide after filename normalization: "
                        f"{other_label!r} and {obs.epoch_label!r}")
                unknown_group_owner[group_id] = obs.epoch_label
                unknown_group_by_epoch[obs.epoch_label] = group_id
        records_by_group.setdefault(group_id, []).append(obs)
        group_epoch_labels.setdefault(group_id, set()).add(obs.epoch_label)
        if observed_date is None and obs.actual_date:
            group_raw_dates.setdefault(group_id, set()).add(str(obs.actual_date))
    for row in result.presence:
        observed_date = canonical_date(row.get("actual_date"))
        if observed_date is not None:
            group_id = observed_date
        else:
            epoch = str(row.get("epoch_label") or "UNSPECIFIED")
            group_id = unknown_group_by_epoch.get(epoch)
            if group_id is None:
                continue
        if row.get("epoch_label"):
            group_epoch_labels.setdefault(group_id, set()).add(str(row["epoch_label"]))
        if observed_date is None and row.get("actual_date"):
            group_raw_dates.setdefault(group_id, set()).add(str(row["actual_date"]))
    output_groups = actual_date_order + sorted(
        group_id for group_id in records_by_group if group_id not in actual_date_order)
    output_group_metadata = {
        group_id: {
            "actual_date": group_id if group_id in actual_date_order else None,
            "actual_date_status": "VERIFIED_ACTUAL_DATE" if group_id in actual_date_order
                                  else "ACTUAL_DATE_UNRESOLVED",
            "epoch_labels": sorted(group_epoch_labels.get(group_id, set())),
            "unresolved_actual_date_values": sorted(group_raw_dates.get(group_id, set())),
        }
        for group_id in output_groups
    }
    reference_tags = {
        "REFERENCE_GRID_PATH": str(anchor_raster.resolve()) if anchor_raster else "NOT_APPLICABLE",
        "REFERENCE_GRID_LABEL": anchor_raster.stem if anchor_raster else "NOT_APPLICABLE",
        "REFERENCE_ACQUISITION_DATE_STATUS": (
            "UNKNOWN_ANNUAL_FILE_LABEL_ONLY" if anchor_raster else "NOT_APPLICABLE"),
        "BASE_GRID_PIXEL_PROVENANCE_STATUS": (
            "UNKNOWN_PER_PIXEL_SOURCE_COMPONENT_DATE_AND_VIEW_ANGLE"
            if anchor_raster else "NOT_APPLICABLE"),
    }

    def output_tif_tags(product_kind: str, group_id: str, **extra: Any) -> dict[str, Any]:
        metadata = output_group_metadata[group_id]
        observed_date = metadata["actual_date"]
        dates = [observed_date] if observed_date else []
        return {
            "PRODUCT_KIND": product_kind,
            "OUTPUT_GROUP_ID": group_id,
            "ACTUAL_DATE": observed_date or "UNKNOWN",
            "ACTUAL_DATE_STATUS": metadata["actual_date_status"],
            "EPOCH_LABELS_JSON": metadata["epoch_labels"],
            "ACTUAL_DATES_JSON": dates,
            "ACTUAL_DATE_RAW_VALUES_JSON": metadata["unresolved_actual_date_values"],
            "ACTUAL_DATE_SOURCE": "TILE_ACTUAL_DATE_REGISTRY",
            **reference_tags,
            **extra,
        }

    def alignment_tif_tags(product_kind: str, field: LocalDisplacementField,
                           value_codes: dict[str, Any]) -> dict[str, Any]:
        uncertainty_status = (
            "UNKNOWN_NOT_ESTIMATED" if any(
                cell.positional_uncertainty_m is None for cell in field.cells)
            else "CELLWISE_SEE_GPKG_LINEAGE")
        relative_uncertainty_status = (
            "UNKNOWN_NOT_ESTIMATED" if any(
                cell.relative_registration_uncertainty_m is None for cell in field.cells)
            else "CELLWISE_SEE_GPKG_LINEAGE")
        review_basis_counts = {
            basis: sum(cell.alignment_review_basis == basis for cell in field.cells)
            for basis in sorted({cell.alignment_review_basis for cell in field.cells})
        }
        return {
            "PRODUCT_KIND": product_kind,
            "MOVING_ACTUAL_DATE": field.moving_date,
            "ACTUAL_DATE_SOURCE": "TILE_ACTUAL_DATE_REGISTRY",
            "REFERENCE_ACTUAL_DATE": field.reference_date,
            "REFERENCE_LABEL": field.reference_label,
            "REFERENCE_GRID_PATH": field.reference_source or "UNKNOWN",
            "REFERENCE_ACQUISITION_DATE_STATUS": field.reference_date_status,
            "BASE_GRID_PIXEL_PROVENANCE_STATUS": reference_tags[
                "BASE_GRID_PIXEL_PROVENANCE_STATUS"],
            "ALIGNMENT_FIELD_ID": field.field_id,
            "CORRECTION_SCOPE": field.correction_scope,
            "VALUE_CODES_JSON": value_codes,
            "POSITIONAL_UNCERTAINTY_STATUS": uncertainty_status,
            "ABSOLUTE_POSITIONAL_UNCERTAINTY_STATUS": uncertainty_status,
            "RELATIVE_REGISTRATION_UNCERTAINTY_STATUS": relative_uncertainty_status,
            "ALIGNMENT_REVIEW_BASIS_COUNTS_JSON": review_basis_counts,
            "ABSOLUTE_POSITION_ACCURACY_CLAIMED": False,
            "POSITIONAL_ACCURACY_CLAIMED": False,
            "ROOF_TO_GROUND_CORRECTION": False,
        }

    def interval_tif_tags(product_kind: str, left: str, right: str,
                          **extra: Any) -> dict[str, Any]:
        return {
            "PRODUCT_KIND": product_kind,
            "FROM_OUTPUT_GROUP": left,
            "TO_OUTPUT_GROUP": right,
            "FROM_ACTUAL_DATE": left,
            "TO_ACTUAL_DATE": right,
            "ACTUAL_DATE_ORDER_BASIS": "UNIQUE_ACTUAL_ACQUISITION_DATES",
            "ACTUAL_DATE_SOURCE": "TILE_ACTUAL_DATE_REGISTRY",
            **reference_tags,
            **extra,
        }

    epoch_arrays: dict[str, dict[str, np.ndarray]] = {}
    ground_candidate_products: list[dict[str, Any]] = []
    candidate_track_id_products: list[dict[str, Any]] = []
    candidate_track_codes = _candidate_track_codebook(result.observations)
    candidate_track_codebook_name = "CANDIDATE_TRACK_ID_CODEBOOK.json"
    candidate_track_codebook = {
        "schema_version": "xian_r5.candidate_track_id_codebook.v1",
        "semantics": "HASHED_SOURCE_CANDIDATE_TRACK_LABEL_NOT_PHYSICAL_BUILDING_ID",
        "hash": "SHA256_FIRST_32_BITS_BIG_ENDIAN;ZERO_AND_UINT32_MAX_RESERVED",
        "code_to_source_candidate_identity": {
            str(code): label for label, code in sorted(
                candidate_track_codes.items(), key=lambda item: item[1])
        },
    }
    candidate_track_codebook_text = json.dumps(
        candidate_track_codebook, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    candidate_track_codebook_bytes = candidate_track_codebook_text.encode("utf-8")
    (output_dir / candidate_track_codebook_name).write_bytes(
        candidate_track_codebook_bytes)
    candidate_track_codebook_sha256 = hashlib.sha256(
        candidate_track_codebook_bytes).hexdigest()
    observations_by_uid = {obs.uid: obs for obs in result.observations}
    observations_by_physical_id: dict[int, list[Observation]] = {}
    for obs in result.observations:
        if obs.physical_id is not None:
            observations_by_physical_id.setdefault(int(obs.physical_id), []).append(obs)
    canonical_by_id = {int(row["physical_id"]): row for row in canonical_rows}
    roof_alignment_by_uid = {
        row.observation.uid: row for row in roof_alignment_records
    }
    presence_raster_diagnostics: dict[str, dict[str, int]] = {}
    for group_id in output_groups:
        records = records_by_group.get(group_id, [])
        roof = _shapes([o.roof for o in records], 1, shape, transform, "uint8")
        _write_tif(
            output_dir / f"{group_id}_roof_candidate_DIAGNOSTIC.tif", roof, transform,
            nodata=None, tags=output_tif_tags(
                "RAW_ROOF_CANDIDATE_DIAGNOSTIC", group_id,
                GEOMETRY_BASIS="SOURCE_IMAGE_ROOF_PROXY_NOT_GROUND_FOOTPRINT",
                GROUND_FOOTPRINT_CONFIRMED=False))
        roof_registered = np.zeros(shape, dtype=np.uint8)
        registration_quality = np.zeros(shape, dtype=np.uint8)
        registration_uncertainty = np.full(shape, np.nan, dtype=np.float32)
        registration_quality_codes = {
            "0": "BACKGROUND", "1": "REVIEWED_IMAGE_GRID_SHIFT_APPLIED",
            "2": "UNACCEPTED_NUMERIC_IMAGE_GRID_CANDIDATE",
            "3": "RAW_ROOF_RETAINED_ALIGNMENT_UNRESOLVED",
            "255": "NODATA",
        }
        for obs in records:
            raw_mask = _shapes([obs.roof], 1, shape, transform, "uint8").astype(bool)
            if obs.registered_roof is not None:
                registered_mask = _shapes([obs.registered_roof], 1, shape,
                                          transform, "uint8").astype(bool)
                roof_registered[registered_mask] = 1
                registration_quality[registered_mask] = 1
                sigma_value = obs.image_alignment_positional_uncertainty_m
                if (isinstance(sigma_value, (int, float))
                        and not isinstance(sigma_value, bool)
                        and math.isfinite(float(sigma_value))):
                    registration_uncertainty[registered_mask] = float(sigma_value)
            else:
                candidate = roof_alignment_by_uid.get(obs.uid)
                code = (2 if candidate is not None and
                        candidate.candidate.candidate_shift_rendered else 3)
                registration_quality[raw_mask & (registration_quality == 0)] = code
        _write_tif(output_dir / f"{group_id}_roof_raw.tif", roof, transform,
                   nodata=None, tags=output_tif_tags(
                       "ROOF_RAW", group_id,
                       GEOMETRY_BASIS="SOURCE_IMAGE_ROOF_BOUNDARY_NOT_GROUND_FOOTPRINT",
                       SOURCE_LAYER="building_lineage.gpkg#roof_raw",
                       GROUND_FOOTPRINT_CONFIRMED=False))
        _write_tif(output_dir / f"{group_id}_roof_registered.tif", roof_registered,
                   transform, nodata=None, tags=output_tif_tags(
                       "ROOF_REGISTERED", group_id,
                       VALUE_CODES_JSON={"0": "BACKGROUND", "1": "ACCEPTED_IMAGE_GRID_REGISTERED_ROOF"},
                       GEOMETRY_BASIS="ACCEPTED_IMAGE_GRID_ROOF_NOT_GROUND_FOOTPRINT",
                       SOURCE_LAYER="building_lineage.gpkg#roof_registered",
                       GROUND_FOOTPRINT_CONFIRMED=False))
        _write_tif(output_dir / f"{group_id}_registration_quality.tif",
                   registration_quality, transform, nodata=None,
                   tags=output_tif_tags(
                       "REGISTRATION_QUALITY", group_id,
                       VALUE_CODES_JSON=registration_quality_codes,
                       QUALITY_BASIS="PER_OBSERVATION_IMAGE_GRID_REVIEW_STATUS",
                       ABSOLUTE_GROUND_ACCURACY_CLAIMED=False))
        _write_tif(output_dir / f"{group_id}_registration_status.tif",
                   registration_quality, transform, nodata=None,
                   tags=output_tif_tags(
                       "REGISTRATION_STATUS", group_id,
                       VALUE_CODES_JSON={**registration_quality_codes,
                                         "4": "RAW", "5": "REJECTED"},
                       QUALITY_BASIS="PER_OBSERVATION_IMAGE_GRID_REVIEW_STATUS",
                       ABSOLUTE_GROUND_ACCURACY_CLAIMED=False))
        _write_tif(output_dir / f"{group_id}_registration_uncertainty.tif",
                   registration_uncertainty, transform, nodata=np.nan,
                   tags=output_tif_tags(
                       "REGISTRATION_UNCERTAINTY_METERS", group_id,
                       VALUE_UNITS="METERS",
                       UNCERTAINTY_SEMANTICS=(
                           "REVIEWED_ABSOLUTE_POSITIONAL_UNCERTAINTY_ONLY;"
                           "NAN_WHERE_UNKNOWN_OR_RELATIVE_ONLY")))

        presence_raster = np.zeros(shape, dtype=np.uint8)
        presence_claim_count = 0
        presence_unmapped_count = 0
        presence_rows_for_group = []
        for row in result.presence:
            actual_group = canonical_date(row.get("actual_date"))
            if actual_group is None:
                actual_group = unknown_group_by_epoch.get(str(row.get("epoch_label") or ""))
            if actual_group == group_id:
                presence_rows_for_group.append(row)
        for row in presence_rows_for_group:
            state = row.get("presence")
            value = {"1": 1, "0": 2, "U": 3}.get(state)
            if value is None:
                raise ValueError(f"Unsupported presence state: {state!r}")
            obs = observations_by_uid.get(row.get("uid")) if row.get("uid") else None
            geometry = None
            if state == "0":
                physical_id = row.get("physical_id")
                canonical = canonical_by_id.get(int(physical_id)) if physical_id is not None else None
                if canonical is not None:
                    geometry = canonical.get("geometry")
                if geometry is None and physical_id is not None:
                    history = observations_by_physical_id.get(int(physical_id), [])
                    target_date = canonical_date(row.get("actual_date"))
                    prior = [item for item in history
                             if canonical_date(item.actual_date) is not None
                             and canonical_date(item.actual_date) < target_date]
                    pool = prior or history
                    if pool:
                        known_ground = [item for item in pool if item.ground_footprint is not None]
                        known_registered = [item for item in pool if item.registered_roof is not None]
                        selected = (max(known_ground, key=lambda item: item.actual_date or "")
                                    if known_ground else
                                    max(known_registered, key=lambda item: item.actual_date or "")
                                    if known_registered else
                                    max(pool, key=lambda item: item.actual_date or ""))
                        geometry = selected.ground_footprint or selected.registered_roof or selected.roof
            elif obs is not None:
                geometry = obs.ground_footprint or obs.registered_roof or obs.roof
            if geometry is None or geometry.is_empty:
                presence_unmapped_count += 1
                continue
            mask = _shapes([geometry], 1, shape, transform, "uint8").astype(bool)
            if not mask.any():
                presence_unmapped_count += 1
                continue
            overlap = mask & (presence_raster != 0) & (presence_raster != value)
            presence_raster[mask & (presence_raster == 0)] = value
            presence_raster[overlap] = 3
            presence_claim_count += 1
        _write_tif(output_dir / f"{group_id}_presence.tif", presence_raster,
                   transform, nodata=int(NODATA), tags=output_tif_tags(
                       "BUILDING_PRESENCE_STATE", group_id,
                       VALUE_CODES_JSON={"0": "BACKGROUND", "1": "PRESENT",
                                         "2": "REVIEWED_ABSENT", "3": "UNRESOLVED",
                                         str(int(NODATA)): "NODATA"},
                       PRESENCE_TABLE="building_lineage.gpkg#building_presence",
                       SPATIALIZATION_POLICY=(
                           "GROUND_FOOTPRINT_IF_REVIEWED_ELSE_ACCEPTED_REGISTERED_ROOF_ELSE_RAW_OBSERVED_ROOF;"
                           "ABSENCE_USES_LAST_KNOWN_GEOMETRY;QUALITY_AND_GEOMETRY_BASIS_REMAIN_EXPLICIT"),
                       RAW_ROOF_AS_GROUND_FOOTPRINT=False))
        presence_raster_diagnostics[group_id] = {
            "table_row_count": len(presence_rows_for_group),
            "spatial_claim_count": presence_claim_count,
            "unmapped_table_row_count": presence_unmapped_count,
            "pixel_counts": {str(code): int(np.count_nonzero(presence_raster == code))
                             for code in (0, 1, 2, 3)},
        }
        footprint = np.zeros(shape, dtype=np.uint8)
        ground_candidate = np.zeros(shape, dtype=np.uint8)
        candidate_feature_count = 0
        candidate_observation_uids: set[str] = set()
        ids = np.zeros(shape, dtype=np.uint32)
        quality = np.zeros(shape, dtype=np.uint8)
        sigma = np.full(shape, np.nan, dtype=np.float32)
        unresolved_candidate_mask = np.zeros(shape, dtype=bool)
        ground_claim_count = np.zeros(shape, dtype=np.uint16)
        for obs in records:
            candidate_geometries = _unverified_ground_candidate_geometries(obs)
            if not candidate_geometries:
                continue
            candidate_feature_count += len(candidate_geometries)
            candidate_observation_uids.add(obs.uid)
            mask = _shapes(candidate_geometries, 1, shape, transform, "uint8").astype(bool)
            ground_candidate[mask] = 1
        # Unknown rooftop candidate locations cannot be mistaken for confirmed ground.
        for obs in records:
            if obs.ground_footprint is not None:
                continue
            roof_mask = _shapes([obs.registered_roof or obs.roof], 1, shape,
                                transform, "uint8").astype(bool)
            candidate_geometries = _unverified_ground_candidate_geometries(obs)
            if obs.footprint_candidate is not None:
                candidate_geometries.append(obs.footprint_candidate)
            candidate_mask = _shapes(candidate_geometries, 1, shape,
                                     transform, "uint8").astype(bool)
            unresolved_mask = roof_mask | candidate_mask
            unresolved_candidate_mask |= unresolved_mask
            footprint[unresolved_mask] = U
            ids[unresolved_mask] = ID_UNKNOWN
            quality[roof_mask & (quality == 0)] = 2
            quality[candidate_mask] = 3
        rasterized_ground_claims: set[tuple[int | None, bytes]] = set()
        for obs in records:
            if obs.ground_footprint is None:
                continue
            if obs.ground_footprint.area < min_area_m2:
                continue
            claim_key = (obs.physical_id, bytes(obs.ground_footprint.wkb))
            if claim_key in rasterized_ground_claims:
                continue
            rasterized_ground_claims.add(claim_key)
            mask = _shapes([obs.ground_footprint], 1, shape, transform, "uint8").astype(bool)
            # Ground geometry is meaningful independently of whether a physical
            # identity has been confirmed. Keep the ID sentinel when it has not.
            ground_claim_count[mask] += 1
            footprint[mask] = 1
            ids[mask] = obs.physical_id if obs.physical_id is not None else ID_UNKNOWN
            quality[mask] = 1
            sigma[mask] = obs.footprint_evidence.positional_sigma_m if obs.footprint_evidence else np.nan
        ambiguous_ground = ((ground_claim_count > 1)
                            | ((ground_claim_count > 0) & unresolved_candidate_mask))
        footprint[ambiguous_ground] = U
        ids[ambiguous_ground] = ID_UNKNOWN
        quality[ambiguous_ground] = 4
        sigma[ambiguous_ground] = np.nan
        _write_tif(
            output_dir / f"{group_id}_ground_footprint.tif", footprint, transform,
            nodata=int(NODATA), tags=output_tif_tags(
                "GROUND_FOOTPRINT_STATE_OR_UNRESOLVED", group_id,
                VALUE_CODES_JSON={"0": "BACKGROUND", "1": "REVIEWED_GROUND_FOOTPRINT",
                                  str(int(U)): "UNRESOLVED_CANDIDATE_AREA",
                                  str(int(NODATA)): "NODATA"},
                GEOMETRY_BASIS="REVIEWED_GROUND_ONLY;ROOF_AND_UNVERIFIED_POLYGON_CANDIDATES_REMAIN_UNRESOLVED",
                GROUND_FOOTPRINT_CONFIRMED=bool(np.any(footprint == 1))))
        candidate_name = f"{group_id}_ground_footprint_candidates_UNVERIFIED.tif"
        candidate_tags = output_tif_tags(
            "UNVERIFIED_GROUND_FOOTPRINT_CANDIDATE", group_id,
            VALUE_CODES_JSON={"0": "BACKGROUND", "1": "UNVERIFIED_GROUND_FOOTPRINT_CANDIDATE",
                              str(int(NODATA)): "NODATA"},
            GEOMETRY_BASIS="EXTERNAL_OR_UNVERIFIED_CANDIDATE_VECTOR_NOT_FORMAL_GROUND",
            GROUND_FOOTPRINT_CONFIRMED=False,
            FORMAL_GROUND_FOOTPRINT_PRODUCT=False,
            ROOF_TO_GROUND_CORRECTION=False,
            POSITIONAL_ACCURACY_CLAIMED=False)
        _write_tif(output_dir / candidate_name, ground_candidate, transform,
                   nodata=int(NODATA), tags=candidate_tags)
        candidate_product = {
            "epoch": group_id,
            "output_group": group_id,
            "candidate_tif": candidate_name,
            "candidate_base10m_tif": None,
            "candidate_feature_count": candidate_feature_count,
            "candidate_observation_count": len(candidate_observation_uids),
            "candidate_pixel_count": int(np.count_nonzero(ground_candidate)),
            "formal_ground_footprint_product": False,
            "position_accuracy_claimed": False,
            "roof_to_ground_correction": False,
        }
        ground_candidate_products.append(candidate_product)
        _write_tif(
            output_dir / f"{group_id}_physical_id.tif", ids, transform, nodata=None,
            tags=output_tif_tags(
                "PHYSICAL_BUILDING_ID_OR_UNRESOLVED", group_id,
                VALUE_CODES_JSON={"0": "BACKGROUND", str(int(ID_UNKNOWN)): "IDENTITY_UNRESOLVED"},
                CONFIRMED_PHYSICAL_ID_COUNT=int(np.unique(ids[(ids > 0) & (ids != ID_UNKNOWN)]).size)))
        _write_tif(
            output_dir / f"{group_id}_spatial_quality.tif", quality, transform, nodata=None,
            tags=output_tif_tags(
                "SPATIAL_QUALITY", group_id,
                VALUE_CODES_JSON={"0": "BACKGROUND", "1": "REVIEWED_GROUND",
                                  "2": "ROOF_ONLY_UNRESOLVED",
                                  "3": "UNVERIFIED_GROUND_CANDIDATE",
                                  "4": "AMBIGUOUS_GROUND_OVERLAP"}))
        _write_tif(
            output_dir / f"{group_id}_position_sigma_m.tif", sigma, transform, nodata=np.nan,
            tags=output_tif_tags(
                "POSITIONAL_SIGMA_METERS", group_id,
                VALUE_UNITS="METERS",
                SIGMA_SEMANTICS="REVIEWED_GROUND_FOOTPRINT_SIGMA;NAN_WHERE_UNKNOWN"))
        candidate_track_ids, candidate_track_quality, repeated_labels = (
            _candidate_track_rasters(records, shape, transform, candidate_track_codes))
        candidate_track_name = f"{group_id}_candidate_track_id_ROOF_PROXY.tif"
        candidate_track_quality_name = (
            f"{group_id}_candidate_track_id_quality_ROOF_PROXY.tif")
        candidate_track_value_codes = {
            "0": "BACKGROUND",
            str(int(ID_UNKNOWN)): "CANDIDATE_ID_UNRESOLVED_OR_AMBIGUOUS",
        }
        _write_tif(
            output_dir / candidate_track_name, candidate_track_ids, transform,
            nodata=None, tags=output_tif_tags(
                "CANDIDATE_TRACK_ID_ROOF_PROXY_NOT_PHYSICAL_ID", group_id,
                VALUE_CODES_JSON=candidate_track_value_codes,
                CODEBOOK_JSON=candidate_track_codebook_name,
                CODEBOOK_SHA256=candidate_track_codebook_sha256,
                GEOMETRY_BASIS=(
                    "REVIEWED_IMAGE_GRID_ALIGNED_ROOF_PROXY_WHEN_AVAILABLE_ELSE_RAW_ROOF_PROXY"),
                CANDIDATE_ID_SOURCE="SOURCE_CANDIDATE_IDENTITY_LABEL",
                CANDIDATE_TRACK_ID_PROMOTED_TO_PHYSICAL_ID=False,
                FORMAL_PHYSICAL_ID_PRODUCT=False,
                PHYSICAL_IDENTITY_CONFIRMED=False,
                CHANGE_USE=False,
                GROUND_FOOTPRINT_CONFIRMED=False,
                ROOF_TO_GROUND_CORRECTION=False,
                POSITIONAL_ACCURACY_CLAIMED=False))
        _write_tif(
            output_dir / candidate_track_quality_name, candidate_track_quality,
            transform, nodata=None, tags=output_tif_tags(
                "CANDIDATE_TRACK_ID_QUALITY_NOT_PHYSICAL_ID_QUALITY", group_id,
                VALUE_CODES_JSON=CANDIDATE_TRACK_QUALITY_CODES,
                CANDIDATE_TRACK_ID_PROMOTED_TO_PHYSICAL_ID=False,
                FORMAL_PHYSICAL_ID_PRODUCT=False,
                CHANGE_USE=False,
                GEOMETRY_BASIS="SAME_ROOF_PROXY_AS_CANDIDATE_TRACK_ID"))
        candidate_track_id_products.append({
            "output_group": group_id,
            "candidate_tif": candidate_track_name,
            "quality_tif": candidate_track_quality_name,
            "candidate_base10m_tif": None,
            "quality_base10m_tif": None,
            "candidate_observation_count": sum(
                isinstance(obs.candidate_identity, str)
                and bool(obs.candidate_identity.strip()) for obs in records),
            "source_track_label_count": len({
                obs.candidate_identity.strip() for obs in records
                if isinstance(obs.candidate_identity, str)
                and obs.candidate_identity.strip()}),
            "unique_track_pixel_count": int(np.count_nonzero(
                (candidate_track_ids != 0) & (candidate_track_ids != ID_UNKNOWN))),
            "unresolved_track_pixel_count": int(np.count_nonzero(
                candidate_track_ids == ID_UNKNOWN)),
            "repeated_source_track_labels": repeated_labels,
            "formal_physical_id_product": False,
            "candidate_ids_promoted_to_physical_ids": False,
            "change_use": False,
            "ground_footprint_confirmed": False,
        })
        epoch_arrays[group_id] = {"roof_raw": roof,
                               "roof_registered": roof_registered,
                               "registration_quality": registration_quality,
                               "registration_uncertainty": registration_uncertainty,
                               "presence": presence_raster,
                               "footprint": footprint,
                               "ground_candidate": ground_candidate,
                               "ids": ids, "quality": quality,
                               "candidate_track_ids": candidate_track_ids,
                               "candidate_track_quality": candidate_track_quality}
    canonical_geometry_raster = np.zeros(shape, dtype=np.uint8)
    canonical_id_raster = np.zeros(shape, dtype=np.uint32)
    canonical_quality_raster = np.zeros(shape, dtype=np.uint8)
    canonical_occupied = np.zeros(shape, dtype=bool)
    for row in canonical_rows:
        geometry = row["geometry"]
        physical_id = int(row["physical_id"])
        if geometry is None:
            for obs in observations_by_physical_id.get(physical_id, []):
                mask = _shapes([obs.roof], 1, shape, transform, "uint8").astype(bool)
                canonical_quality_raster[mask & (canonical_quality_raster == 0)] = 2
            continue
        mask = _shapes([geometry], 1, shape, transform, "uint8").astype(bool)
        overlap = mask & canonical_occupied & (canonical_id_raster != physical_id)
        canonical_geometry_raster[mask] = 1
        canonical_id_raster[mask & ~overlap] = physical_id
        canonical_id_raster[overlap] = ID_UNKNOWN
        canonical_quality_raster[mask & ~overlap] = 1
        canonical_quality_raster[overlap] = 3
        canonical_occupied |= mask
    cross_date_tags = {
        "ACTUAL_DATE_SOURCE": "TILE_ACTUAL_DATE_REGISTRY",
        "CROSS_ACTUAL_DATES_JSON": actual_date_order,
        "BASE_GRID_PIXEL_PROVENANCE_STATUS": reference_tags[
            "BASE_GRID_PIXEL_PROVENANCE_STATUS"],
        "REFERENCE_GRID_PATH": reference_tags["REFERENCE_GRID_PATH"],
        "REFERENCE_GRID_LABEL": reference_tags["REFERENCE_GRID_LABEL"],
        "REFERENCE_ACQUISITION_DATE_STATUS": reference_tags[
            "REFERENCE_ACQUISITION_DATE_STATUS"],
    }
    _write_tif(output_dir / "canonical_building_geometry.tif",
               canonical_geometry_raster, transform, nodata=None,
               tags={**cross_date_tags,
                     "PRODUCT_KIND": "CANONICAL_CROSS_DATE_GEOMETRY_NOT_GROUND",
                     "GEOMETRY_BASIS": "REVIEWED_PHYSICAL_ID_AND_ACCEPTED_REGISTERED_ROOF_MEDOID",
                     "MINIMUM_DISTINCT_ACTUAL_DATES": 2,
                     "GEOMETRY_SYNTHESIS": "OBSERVED_MEDOID_ONLY_NO_UNION_OR_WARP",
                     "GROUND_TRUTH": False})
    _write_tif(output_dir / "canonical_building_id.tif", canonical_id_raster,
               transform, nodata=None,
               tags={**cross_date_tags,
                     "PRODUCT_KIND": "CANONICAL_BUILDING_ID_NOT_GROUND",
                     "CODEBOOK": "PHYSICAL_ID_CODEBOOK.json",
                     "GEOMETRY_BASIS": "CANONICAL_REGISTERED_ROOF_MEDOID_NOT_GROUND",
                     "VALUE_CODES_JSON": {"0": "BACKGROUND",
                                           str(int(ID_UNKNOWN)): "OVERLAPPING_CANONICAL_IDS"},
                     "GROUND_TRUTH": False})
    _write_tif(output_dir / "canonical_geometry_quality.tif",
               canonical_quality_raster, transform, nodata=None,
               tags={**cross_date_tags,
                     "PRODUCT_KIND": "CANONICAL_GEOMETRY_QUALITY",
                     "VALUE_CODES_JSON": {"0": "BACKGROUND", "1": "CROSS_DATE_MEDOID_RESOLVED",
                                           "2": "IDENTITY_REVIEWED_BUT_CANONICAL_GEOMETRY_UNRESOLVED",
                                           "3": "OVERLAPPING_CANONICAL_IDENTITIES"},
                     "UNRESOLVED_GEOMETRY_BASIS": "RAW_ROOF_LOCATOR_ONLY_NOT_CANONICAL_GEOMETRY",
                     "GROUND_TRUTH": False})
    alignment_quality_codes = {
        "background_outside_field": 0,
        "reviewed_accepted_image_grid_correction": 1,
        "unreviewed_numeric_candidate": 2,
        "covered_without_usable_tie": 3,
        "composed_cycle_screen_only_unreviewed_candidate": 4,
        "composed_cycle_screen_failed_unreviewed_candidate": 5,
        "composed_without_cycle_check_unreviewed_candidate": 6,
        "composed_anchor_acquisition_date_unknown_candidate": 7,
        **_TIE_SUPPORT_QUALITY_LABELS,
        **_SOURCE_COMPONENT_QUALITY_CODES,
        "overlapping_field_cells": 255,
    }
    composed_quality_codes = {
        "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY": 4,
        "COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED": 5,
        "COMPOSED_CANDIDATE_TREE_HAS_NO_CYCLE_CHECK": 6,
        "COMPOSED_CANDIDATE_ANCHOR_BASE_DATE_UNKNOWN": 7,
        "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY": 4,
        "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED": 5,
        "SPATIALLY_COMPOSED_CANDIDATE_TREE_NO_CYCLE_CHECK": 6,
    }
    alignment_quality_products: dict[str, dict[str, Any]] = {}
    alignment_quality_arrays: dict[str, np.ndarray] = {}
    for field in alignment_fields:
        field_id = field.field_id
        quality = np.zeros(shape, dtype=np.uint8)
        ambiguous = np.zeros(shape, dtype=bool)
        for cell in field.cells:
            if cell.accepted:
                code = 1
            elif (cell.correction_dx_m_candidate is not None
                  and cell.correction_dy_m_candidate is not None):
                if cell.source_component_issue is not None:
                    label = _SOURCE_COMPONENT_QUALITY_LABELS[cell.source_component_issue]
                    code = _SOURCE_COMPONENT_QUALITY_CODES[label]
                else:
                    code = composed_quality_codes.get(cell.status, 2)
            else:
                code = _tie_support_quality_code(cell) or (
                    _SOURCE_COMPONENT_QUALITY_CODES[
                        _SOURCE_COMPONENT_QUALITY_LABELS[cell.source_component_issue]]
                    if cell.source_component_issue is not None else 3)
            mask = _shapes([cell.support_geometry], 1, shape, transform, "uint8").astype(bool)
            ambiguous |= mask & (quality != 0)
            quality[mask & (quality == 0)] = code
        quality[ambiguous] = NODATA
        safe_field_id = re.sub(r"[^A-Za-z0-9._-]+", "_", field_id).strip("._-")
        if not safe_field_id:
            raise ValueError(f"Unsafe empty alignment field ID: {field_id!r}")
        filename = f"{safe_field_id}_alignment_quality.tif"
        if filename in {item["filename"] for item in alignment_quality_products.values()}:
            raise ValueError(f"Alignment field IDs collide as output filenames: {field_id}")
        _write_tif(output_dir / filename, quality,
                   transform, nodata=None, tags=alignment_tif_tags(
                       "IMAGE_ALIGNMENT_FIELD_QUALITY", field, alignment_quality_codes))
        alignment_quality_arrays[field_id] = quality
        alignment_quality_products[field_id] = {
            "filename": filename,
            "moving_date": field.moving_date,
            "reference_date": field.reference_date,
            "reference_label": field.reference_label,
        }
    roof_alignment_quality_codes = {
        "background": 0,
        "accepted_image_grid_shift_roof_only_not_ground": 1,
        "unaccepted_numeric_image_grid_candidate_roof_only": 2,
        "raw_roof_retained_unresolved": 3,
        "cross_cell_or_geometry_overlap_unresolved": 4,
        "unaccepted_dominant_cell_image_grid_candidate_roof_only": 5,
        **_TIE_SUPPORT_QUALITY_LABELS,
        **_SOURCE_COMPONENT_QUALITY_CODES,
    }
    roof_alignment_products: list[dict[str, Any]] = []
    roof_fields = {record.field.field_id: record.field
                   for record in roof_alignment_records}
    for field_id, field in roof_fields.items():
        records = [record for record in roof_alignment_records
                   if record.field.field_id == field_id]
        candidate_raster = np.zeros(shape, dtype=np.uint8)
        quality_raster = np.zeros(shape, dtype=np.uint8)
        occupied = np.zeros(shape, dtype=bool)
        statuses: dict[str, int] = {}
        for record in records:
            candidate = record.candidate
            cell = next((item for item in record.field.cells
                         if item.cell_id == candidate.cell_id), None)
            statuses[candidate.status] = statuses.get(candidate.status, 0) + 1
            mask = _shapes([candidate.geometry], 1, shape, transform, "uint8").astype(bool)
            if not mask.any():
                continue
            if candidate.candidate_shift_rendered:
                value = 1
                if candidate.status.startswith("ACCEPTED_"):
                    quality_code = 1
                elif candidate.status == "UNACCEPTED_DOMINANT_CELL_IMAGE_GRID_SHIFT_ROOF_CANDIDATE":
                    quality_code = 5
                else:
                    quality_code = 2
            else:
                value = U
                if candidate.status in _SOURCE_COMPONENT_QUALITY_LABELS:
                    quality_code = _SOURCE_COMPONENT_QUALITY_CODES[
                        _SOURCE_COMPONENT_QUALITY_LABELS[candidate.status]]
                else:
                    quality_code = (
                        4 if candidate.status == "CROSSES_OR_OVERLAPS_ALIGNMENT_CELLS"
                        else _tie_support_quality_code(cell) or 3)
            overlap = mask & occupied
            fresh = mask & ~occupied
            candidate_raster[fresh] = value
            quality_raster[fresh] = quality_code
            candidate_raster[overlap] = U
            quality_raster[overlap] = 4
            occupied[mask] = True

        safe_field_id = re.sub(r"[^A-Za-z0-9._-]+", "_", field_id).strip("._-")
        if not safe_field_id:
            raise ValueError(f"Unsafe empty roof-alignment field ID: {field_id!r}")
        candidate_name = f"roof_alignment_candidate_{safe_field_id}.tif"
        quality_name = f"roof_alignment_quality_{safe_field_id}.tif"
        candidate_tags = alignment_tif_tags(
            "IMAGE_GRID_ROOF_ALIGNMENT_CANDIDATE", field,
            {"0": "BACKGROUND", "1": "UNACCEPTED_SHIFTED_ROOF_CANDIDATE",
             str(int(U)): "UNRESOLVED_ROOF_CANDIDATE", str(int(NODATA)): "NODATA"})
        candidate_tags.update({
            "GEOMETRY_BASIS": "IMAGE_GRID_ALIGNED_ROOF_PROXY_NOT_GROUND_FOOTPRINT",
            "FORMAL_GROUND_FOOTPRINT_PRODUCT": False,
            "ROOF_CELL_SUPPORT_POLICY": (
                "SINGLE_SOURCE_COMPONENT_REQUIRED;FULL_CELL_OR_UNACCEPTED_DOMINANT_CELL_AREA_FRACTION_GE_0.80"),
        })
        quality_tags = alignment_tif_tags(
            "IMAGE_GRID_ROOF_ALIGNMENT_QUALITY", field, roof_alignment_quality_codes)
        quality_tags["ROOF_CELL_SUPPORT_POLICY"] = candidate_tags["ROOF_CELL_SUPPORT_POLICY"]
        _write_tif(output_dir / candidate_name, candidate_raster, transform,
                   nodata=int(NODATA), tags=candidate_tags)
        _write_tif(output_dir / quality_name, quality_raster, transform,
                   nodata=None, tags=quality_tags)
        product = {
            "field_id": field_id,
            "moving_date": field.moving_date,
            "reference_label": field.reference_label,
            "reference_source": field.reference_source,
            "reference_date_status": field.reference_date_status,
            "candidate_tif": candidate_name,
            "quality_tif": quality_name,
            "observation_count": len(records),
            "candidate_shift_rendered_count": sum(
                record.candidate.candidate_shift_rendered for record in records),
            "unresolved_count": sum(
                not record.candidate.candidate_shift_rendered for record in records),
            "status_counts": statuses,
            "quality_pixel_counts": {
                str(code): int((quality_raster == code).sum())
                for code in roof_alignment_quality_codes.values()},
            "position_accuracy_claimed": False,
            "roof_to_ground_correction": False,
            "formal_ground_footprint_product": False,
            "base10m_quality_aggregation": "MAX_FINE_QUALITY_CODE",
        }
        if anchor_resolution is not None:
            factor = int(round(anchor_resolution / resolution_m))
            base_candidate_name = f"roof_alignment_candidate_{safe_field_id}_BASE10M.tif"
            base_quality_name = f"roof_alignment_quality_{safe_field_id}_BASE10M.tif"
            _write_tif(output_dir / base_candidate_name,
                       _coarsen_binary_state(candidate_raster, factor, int(U)),
                       from_origin(west, north, anchor_resolution, anchor_resolution),
                       nodata=int(NODATA), tags={
                           **candidate_tags, "GRID_LEVEL": "REFERENCE_10M",
                           "DERIVED_FROM_RASTER": candidate_name})
            _write_tif(output_dir / base_quality_name,
                       _coarsen_quality(quality_raster, factor),
                       from_origin(west, north, anchor_resolution, anchor_resolution),
                       nodata=None, tags={
                           **quality_tags, "GRID_LEVEL": "REFERENCE_10M",
                           "DERIVED_FROM_RASTER": quality_name,
                           "AGGREGATION_POLICY": "MAX_FINE_QUALITY_CODE"})
            product["candidate_base10m_tif"] = base_candidate_name
            product["quality_base10m_tif"] = base_quality_name
        roof_alignment_products.append(product)
    interval_arrays: dict[str, np.ndarray] = {}
    registered_interval_arrays: dict[str, np.ndarray] = {}
    canonical_interval_arrays: dict[str, np.ndarray] = {}
    for left, right in zip(actual_date_order, actual_date_order[1:]):
        records = records_by_group.get(left, []) + records_by_group.get(right, [])
        extent_geoms = [g for o in records
                        for g in ((o.ground_footprint,) if o.ground_footprint is not None
                                  else (o.registered_roof or o.roof,
                                        o.footprint_candidate)) if g is not None]
        extent = _shapes(extent_geoms,
                         1, shape, transform, "uint8").astype(bool)
        change = np.zeros(shape, dtype=np.uint8)
        change[extent] = STATE_CODES["UNRESOLVED"]
        for row in result.transitions:
            if (row.get("from_output_group") != left
                    or row.get("to_output_group") != right
                    or row["state"] not in {"STABLE", "ADDITION", "DEMOLITION"}):
                continue
            relevant = [o.ground_footprint for o in records if o.physical_id == row["physical_id"]
                        and o.ground_footprint is not None and o.ground_footprint.area >= min_area_m2]
            mask = _shapes(relevant, 1, shape, transform, "uint8").astype(bool)
            change[mask] = STATE_CODES[row["state"]]
        stem = f"{left}_to_{right}"
        change_codes = {"0": "BACKGROUND", "1": "STABLE", "2": "ADDITION",
                        "3": "DEMOLITION", "4": "UNRESOLVED", str(int(NODATA)): "NODATA"}
        _write_tif(
            output_dir / f"{stem}_change.tif", change, transform, nodata=int(NODATA),
            tags=interval_tif_tags(
                "BUILDING_CHANGE_STATE", left, right,
                VALUE_CODES_JSON=change_codes,
                CHANGE_EVIDENCE_POLICY="CONFIRMED_PHYSICAL_IDENTITY_AND_REVIEWED_GROUND_FOOTPRINTS_ONLY",
                MIN_AREA_M2_POST_STATE=min_area_m2))
        interval_arrays[stem] = change
        for state in ("STABLE", "ADDITION", "DEMOLITION"):
            layer = np.zeros(shape, dtype=np.uint8)
            layer[change == STATE_CODES["UNRESOLVED"]] = STATE_CODES["UNRESOLVED"]
            layer[change == STATE_CODES[state]] = 1
            _write_tif(
                output_dir / f"{stem}_{state.lower()}.tif", layer, transform,
                nodata=int(NODATA), tags=interval_tif_tags(
                    f"{state}_AREA_MASK", left, right,
                    VALUE_CODES_JSON={"0": "BACKGROUND", "1": state,
                                      "4": "UNRESOLVED", str(int(NODATA)): "NODATA"},
                    STATE_CLASS=state,
                    CHANGE_EVIDENCE_POLICY="CONFIRMED_PHYSICAL_IDENTITY_AND_REVIEWED_GROUND_FOOTPRINTS_ONLY",
                    MIN_AREA_M2_POST_STATE=min_area_m2))
        registered_change = np.zeros(shape, dtype=np.uint8)
        canonical_change = np.zeros(shape, dtype=np.uint8)

        def paint_change(target: np.ndarray, geometry: Any, value: int) -> None:
            if geometry is None or geometry.is_empty:
                return
            mask = _shapes([geometry], 1, shape, transform, "uint8").astype(bool)
            overlap = mask & (target != 0) & (target != value)
            target[mask & (target == 0)] = value
            target[overlap] = STATE_CODES["UNRESOLVED"]

        for row in result.transitions:
            if (row.get("from_output_group") != left
                    or row.get("to_output_group") != right):
                continue
            physical_id = row.get("physical_id")
            transition_state = row.get("state")
            code = STATE_CODES.get(transition_state, STATE_CODES["UNRESOLVED"])
            history = observations_by_physical_id.get(int(physical_id), []) if physical_id is not None else []
            left_obs = [obs for obs in history if canonical_date(obs.actual_date) == left]
            right_obs = [obs for obs in history if canonical_date(obs.actual_date) == right]
            if transition_state == "STABLE":
                registered_geometries = [obs.registered_roof for obs in (*left_obs, *right_obs)
                                         if obs.registered_roof is not None]
            elif transition_state == "ADDITION":
                registered_geometries = [obs.registered_roof for obs in right_obs
                                         if obs.registered_roof is not None]
            elif transition_state == "DEMOLITION":
                registered_geometries = [obs.registered_roof for obs in left_obs
                                         if obs.registered_roof is not None]
            else:
                registered_geometries = [obs.registered_roof for obs in (*left_obs, *right_obs)
                                         if obs.registered_roof is not None]
            for geometry in registered_geometries:
                paint_change(registered_change, geometry, code)
            canonical = canonical_by_id.get(int(physical_id)) if physical_id is not None else None
            if canonical is not None:
                paint_change(canonical_change, canonical.get("geometry"), code)

        # Raw roof support is uncertainty support only, never confirmed change geometry.
        # Missing accepted registration/identity must not disappear into background.
        for obs in records_by_group.get(left, []) + records_by_group.get(right, []):
            if obs.physical_id is None or obs.registered_roof is None:
                support = obs.registered_roof if obs.registered_roof is not None else obs.roof
                paint_change(registered_change, support,
                             STATE_CODES["UNRESOLVED"])
                paint_change(canonical_change, support, STATE_CODES["UNRESOLVED"])

        registered_interval_arrays[stem] = registered_change
        canonical_interval_arrays[stem] = canonical_change

        def write_non_ground_change(prefix: str, values: np.ndarray,
                                    geometry_basis: str) -> None:
            value_codes = {"0": "BACKGROUND", "1": "STABLE", "2": "ADDITION",
                           "3": "DEMOLITION", "4": "UNRESOLVED",
                           str(int(NODATA)): "NODATA"}
            common = interval_tif_tags(
                f"{prefix.upper()}_CHANGE_DIAGNOSTIC_NOT_GROUND", left, right,
                VALUE_CODES_JSON=value_codes,
                GEOMETRY_BASIS=geometry_basis,
                GROUND_FOOTPRINT_CONFIRMED=False,
                CHANGE_USE="DIAGNOSTIC_ONLY_NOT_GROUND_CHANGE",
                IDENTITY_SOURCE="REVIEWED_PHYSICAL_IDENTITY_GRAPH_WHEN_AVAILABLE")
            _write_tif(output_dir / f"{stem}_{prefix}_change_NON_GROUND.tif",
                       values, transform, nodata=int(NODATA), tags=common)
            for state_name, code in (("stable", 1), ("addition", 2),
                                     ("demolition", 3), ("unresolved", 4)):
                layer = np.zeros(shape, dtype=np.uint8)
                layer[values == code] = 1
                _write_tif(output_dir / f"{stem}_{prefix}_{state_name}_NON_GROUND.tif",
                           layer, transform, nodata=int(NODATA),
                           tags=interval_tif_tags(
                               f"{prefix.upper()}_{state_name.upper()}_DIAGNOSTIC_NOT_GROUND",
                               left, right,
                               VALUE_CODES_JSON={"0": "BACKGROUND", "1": state_name.upper(),
                                                 str(int(NODATA)): "NODATA"},
                               GEOMETRY_BASIS=geometry_basis,
                               GROUND_FOOTPRINT_CONFIRMED=False,
                               CHANGE_USE="DIAGNOSTIC_ONLY_NOT_GROUND_CHANGE"))

        write_non_ground_change(
            "registered", registered_change,
            "ACCEPTED_IMAGE_GRID_REGISTERED_ROOF_ONLY_NOT_GROUND")
        write_non_ground_change(
            "canonical", canonical_change,
            "REVIEWED_IDENTITY_CROSS_DATE_REGISTERED_ROOF_MEDOID_NOT_GROUND")
    base_grid_products = False
    if anchor_metadata is not None:
        factor = int(round(anchor_resolution / resolution_m))
        base_transform = from_origin(west, north, anchor_resolution, anchor_resolution)
        for group_id, arrays in epoch_arrays.items():
            _write_tif(output_dir / f"{group_id}_roof_raw_BASE10M.tif",
                       _coarsen_binary_state(arrays["roof_raw"], factor, 255),
                       base_transform, nodata=None, tags=output_tif_tags(
                           "ROOF_RAW", group_id, GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="ANY_FINE_ROOF_PIXEL",
                           GEOMETRY_BASIS="SOURCE_IMAGE_ROOF_NOT_GROUND"))
            _write_tif(output_dir / f"{group_id}_roof_registered_BASE10M.tif",
                       _coarsen_binary_state(arrays["roof_registered"], factor, 255),
                       base_transform, nodata=None, tags=output_tif_tags(
                           "ROOF_REGISTERED", group_id, GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="ANY_ACCEPTED_REGISTERED_ROOF_PIXEL",
                           GEOMETRY_BASIS="IMAGE_GRID_REGISTERED_ROOF_NOT_GROUND"))
            _write_tif(output_dir / f"{group_id}_registration_quality_BASE10M.tif",
                       _coarsen_quality(arrays["registration_quality"], factor),
                       base_transform, nodata=None, tags=output_tif_tags(
                           "REGISTRATION_QUALITY", group_id, GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="MAX_FINE_QUALITY_CODE"))
            _write_tif(output_dir / f"{group_id}_registration_status_BASE10M.tif",
                       _coarsen_quality(arrays["registration_quality"], factor),
                       base_transform, nodata=None, tags=output_tif_tags(
                           "REGISTRATION_STATUS", group_id, GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="MAX_FINE_QUALITY_CODE"))
            _write_tif(output_dir / f"{group_id}_registration_uncertainty_BASE10M.tif",
                       _coarsen_float_max(arrays["registration_uncertainty"], factor),
                       base_transform, nodata=np.nan, tags=output_tif_tags(
                           "REGISTRATION_UNCERTAINTY_METERS", group_id,
                           GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="MAX_FINITE_ABSOLUTE_SIGMA;NAN_IF_UNKNOWN"))
            _write_tif(output_dir / f"{group_id}_presence_BASE10M.tif",
                       _coarsen_presence(arrays["presence"], factor),
                       base_transform, nodata=int(NODATA), tags=output_tif_tags(
                           "BUILDING_PRESENCE_STATE", group_id, GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="SINGLE_STATE_ONLY;MIXED_OR_UNRESOLVED_TO_U",
                           VALUE_CODES_JSON={"0": "BACKGROUND", "1": "PRESENT",
                                             "2": "REVIEWED_ABSENT", "3": "UNRESOLVED",
                                             str(int(NODATA)): "NODATA"}))
            _write_tif(output_dir / f"{group_id}_ground_footprint_BASE10M.tif",
                       _coarsen_binary_state(arrays["footprint"], factor, int(U)),
                       base_transform, nodata=int(NODATA), tags=output_tif_tags(
                           "GROUND_FOOTPRINT_STATE_OR_UNRESOLVED", group_id,
                           GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="ANY_SINGLE_GROUND_STATE;UNRESOLVED_OR_MIXED_TO_U",
                           VALUE_CODES_JSON={"0": "BACKGROUND", "1": "REVIEWED_GROUND_FOOTPRINT",
                                             str(int(U)): "UNRESOLVED_CANDIDATE_AREA",
                                             str(int(NODATA)): "NODATA"},
                           GEOMETRY_BASIS="REVIEWED_GROUND_ONLY;ROOF_AND_UNVERIFIED_POLYGON_CANDIDATES_REMAIN_UNRESOLVED"))
            candidate_product = next(item for item in ground_candidate_products
                                     if item["output_group"] == group_id)
            candidate_base_name = f"{group_id}_ground_footprint_candidates_UNVERIFIED_BASE10M.tif"
            _write_tif(
                output_dir / candidate_base_name,
                _coarsen_candidate_footprints(arrays["ground_candidate"], factor),
                base_transform, nodata=int(NODATA), tags=output_tif_tags(
                    "UNVERIFIED_GROUND_FOOTPRINT_CANDIDATE", group_id,
                    GRID_LEVEL="REFERENCE_10M",
                    AGGREGATION_POLICY="ANY_FINE_PIXEL_CANDIDATE",
                    DERIVED_FROM_RASTER=candidate_product["candidate_tif"],
                    VALUE_CODES_JSON={"0": "BACKGROUND", "1": "UNVERIFIED_GROUND_FOOTPRINT_CANDIDATE",
                                      str(int(NODATA)): "NODATA"},
                    GEOMETRY_BASIS="EXTERNAL_OR_UNVERIFIED_CANDIDATE_VECTOR_NOT_FORMAL_GROUND",
                    GROUND_FOOTPRINT_CONFIRMED=False,
                    FORMAL_GROUND_FOOTPRINT_PRODUCT=False,
                    ROOF_TO_GROUND_CORRECTION=False,
                    POSITIONAL_ACCURACY_CLAIMED=False))
            candidate_product["candidate_base10m_tif"] = candidate_base_name
            _write_tif(output_dir / f"{group_id}_physical_id_BASE10M.tif",
                       _coarsen_ids(arrays["ids"], factor), base_transform, nodata=None,
                       tags=output_tif_tags(
                           "PHYSICAL_BUILDING_ID_OR_UNRESOLVED", group_id,
                           GRID_LEVEL="REFERENCE_10M",
                           AGGREGATION_POLICY="SINGLE_ID_ONLY;MULTIPLE_OR_UNKNOWN_TO_SENTINEL",
                           VALUE_CODES_JSON={"0": "BACKGROUND", str(int(ID_UNKNOWN)): "IDENTITY_UNRESOLVED"}))
            _write_tif(output_dir / f"{group_id}_spatial_quality_BASE10M.tif",
                       _coarsen_quality(arrays["quality"], factor), base_transform, nodata=None,
                       tags=output_tif_tags(
                           "SPATIAL_QUALITY", group_id, GRID_LEVEL="REFERENCE_10M",
                           VALUE_CODES_JSON={"0": "BACKGROUND", "1": "REVIEWED_GROUND",
                                             "2": "ROOF_ONLY_UNRESOLVED",
                                             "3": "UNVERIFIED_GROUND_CANDIDATE",
                                             "4": "AMBIGUOUS_GROUND_OVERLAP"}))
            track_product = next(item for item in candidate_track_id_products
                                 if item["output_group"] == group_id)
            track_base_name = f"{group_id}_candidate_track_id_ROOF_PROXY_BASE10M.tif"
            track_quality_base_name = (
                f"{group_id}_candidate_track_id_quality_ROOF_PROXY_BASE10M.tif")
            _write_tif(
                output_dir / track_base_name,
                _coarsen_ids(arrays["candidate_track_ids"], factor),
                base_transform, nodata=None,
                tags=output_tif_tags(
                    "CANDIDATE_TRACK_ID_ROOF_PROXY_NOT_PHYSICAL_ID", group_id,
                    GRID_LEVEL="REFERENCE_10M",
                    AGGREGATION_POLICY="SINGLE_CANDIDATE_ID_ONLY;MULTIPLE_OR_UNKNOWN_TO_SENTINEL",
                    DERIVED_FROM_RASTER=track_product["candidate_tif"],
                    CODEBOOK_JSON=candidate_track_codebook_name,
                    CODEBOOK_SHA256=candidate_track_codebook_sha256,
                    VALUE_CODES_JSON={
                        "0": "BACKGROUND",
                        str(int(ID_UNKNOWN)): "CANDIDATE_ID_UNRESOLVED_OR_AMBIGUOUS"},
                    CANDIDATE_TRACK_ID_PROMOTED_TO_PHYSICAL_ID=False,
                    FORMAL_PHYSICAL_ID_PRODUCT=False,
                    PHYSICAL_IDENTITY_CONFIRMED=False,
                    CHANGE_USE=False,
                    GROUND_FOOTPRINT_CONFIRMED=False,
                    ROOF_TO_GROUND_CORRECTION=False,
                    POSITIONAL_ACCURACY_CLAIMED=False))
            _write_tif(
                output_dir / track_quality_base_name,
                _coarsen_quality(arrays["candidate_track_quality"], factor),
                base_transform, nodata=None,
                tags=output_tif_tags(
                    "CANDIDATE_TRACK_ID_QUALITY_NOT_PHYSICAL_ID_QUALITY", group_id,
                    GRID_LEVEL="REFERENCE_10M",
                    AGGREGATION_POLICY="MAX_FINE_QUALITY_CODE",
                    DERIVED_FROM_RASTER=track_product["quality_tif"],
                    VALUE_CODES_JSON=CANDIDATE_TRACK_QUALITY_CODES,
                    FORMAL_PHYSICAL_ID_PRODUCT=False,
                    CHANGE_USE=False))
            track_product["candidate_base10m_tif"] = track_base_name
            track_product["quality_base10m_tif"] = track_quality_base_name
        _write_tif(output_dir / "canonical_building_geometry_BASE10M.tif",
                   _coarsen_binary_state(canonical_geometry_raster, factor, 255),
                   base_transform, nodata=None,
                   tags={**cross_date_tags,
                         "PRODUCT_KIND": "CANONICAL_CROSS_DATE_GEOMETRY_NOT_GROUND",
                         "GRID_LEVEL": "REFERENCE_10M",
                         "AGGREGATION_POLICY": "ANY_FINE_MEDOID_PIXEL",
                         "GEOMETRY_BASIS": "REGISTERED_ROOF_MEDOID_NOT_GROUND",
                         "GROUND_TRUTH": False})
        _write_tif(output_dir / "canonical_building_id_BASE10M.tif",
                   _coarsen_ids(canonical_id_raster, factor), base_transform, nodata=None,
                   tags={**cross_date_tags,
                         "PRODUCT_KIND": "CANONICAL_BUILDING_ID_NOT_GROUND",
                         "GRID_LEVEL": "REFERENCE_10M",
                         "AGGREGATION_POLICY": "SINGLE_ID_ONLY;MULTIPLE_IDS_TO_SENTINEL",
                         "CODEBOOK": "PHYSICAL_ID_CODEBOOK.json",
                         "GROUND_TRUTH": False})
        _write_tif(output_dir / "canonical_geometry_quality_BASE10M.tif",
                   _coarsen_quality(canonical_quality_raster, factor),
                   base_transform, nodata=None,
                   tags={**cross_date_tags,
                         "PRODUCT_KIND": "CANONICAL_GEOMETRY_QUALITY",
                         "GRID_LEVEL": "REFERENCE_10M",
                         "AGGREGATION_POLICY": "MAX_FINE_QUALITY_CODE",
                         "GROUND_TRUTH": False})
        for field_id, quality in alignment_quality_arrays.items():
            safe_field_id = re.sub(r"[^A-Za-z0-9._-]+", "_", field_id).strip("._-")
            field = next(item for item in alignment_fields if item.field_id == field_id)
            _write_tif(output_dir / f"{safe_field_id}_alignment_quality_BASE10M.tif",
                       _coarsen_alignment_quality(quality, factor), base_transform, nodata=None,
                       tags={
                           **alignment_tif_tags(
                               "IMAGE_ALIGNMENT_FIELD_QUALITY", field, alignment_quality_codes),
                           "GRID_LEVEL": "REFERENCE_10M",
                           "DERIVED_FROM_RASTER": alignment_quality_products[field_id]["filename"],
                       })
        for stem, change in interval_arrays.items():
            base_change = _coarsen_change(change, factor)
            left, right = stem.split("_to_", 1)
            change_codes = {"0": "BACKGROUND", "1": "STABLE", "2": "ADDITION",
                            "3": "DEMOLITION", "4": "UNRESOLVED", str(int(NODATA)): "NODATA"}
            _write_tif(
                output_dir / f"{stem}_change_BASE10M.tif", base_change, base_transform,
                nodata=int(NODATA), tags=interval_tif_tags(
                    "BUILDING_CHANGE_STATE", left, right, GRID_LEVEL="REFERENCE_10M",
                    AGGREGATION_POLICY="ANY_SUBPIXEL_SINGLE_STATE;MIXED_OR_UNRESOLVED_TO_U",
                    VALUE_CODES_JSON=change_codes,
                    CHANGE_EVIDENCE_POLICY="CONFIRMED_PHYSICAL_IDENTITY_AND_REVIEWED_GROUND_FOOTPRINTS_ONLY",
                    MIN_AREA_M2_POST_STATE=min_area_m2))
            for code, label in ((1, "stable"), (2, "addition"), (3, "demolition")):
                layer = np.zeros(base_change.shape, dtype=np.uint8)
                layer[base_change == STATE_CODES["UNRESOLVED"]] = STATE_CODES["UNRESOLVED"]
                layer[base_change == code] = 1
                state = label.upper()
                _write_tif(
                    output_dir / f"{stem}_{label}_BASE10M.tif", layer,
                    base_transform, nodata=int(NODATA), tags=interval_tif_tags(
                        f"{state}_AREA_MASK", left, right, GRID_LEVEL="REFERENCE_10M",
                        AGGREGATION_POLICY="ANY_SUBPIXEL_SINGLE_STATE;MIXED_OR_UNRESOLVED_TO_U",
                        VALUE_CODES_JSON={"0": "BACKGROUND", "1": state,
                                          "4": "UNRESOLVED", str(int(NODATA)): "NODATA"},
                        STATE_CLASS=state,
                        CHANGE_EVIDENCE_POLICY="CONFIRMED_PHYSICAL_IDENTITY_AND_REVIEWED_GROUND_FOOTPRINTS_ONLY",
                        MIN_AREA_M2_POST_STATE=min_area_m2))
            for prefix, diagnostic in (("registered", registered_interval_arrays[stem]),
                                       ("canonical", canonical_interval_arrays[stem])):
                base_diagnostic = _coarsen_change(diagnostic, factor)
                non_ground_codes = {"0": "BACKGROUND", "1": "STABLE", "2": "ADDITION",
                                    "3": "DEMOLITION", "4": "UNRESOLVED",
                                    str(int(NODATA)): "NODATA"}
                _write_tif(
                    output_dir / f"{stem}_{prefix}_change_NON_GROUND_BASE10M.tif",
                    base_diagnostic, base_transform, nodata=int(NODATA),
                    tags=interval_tif_tags(
                        f"{prefix.upper()}_CHANGE_DIAGNOSTIC_NOT_GROUND", left, right,
                        GRID_LEVEL="REFERENCE_10M",
                        AGGREGATION_POLICY="SINGLE_SUBPIXEL_STATE;MIXED_OR_UNRESOLVED_TO_U",
                        VALUE_CODES_JSON=non_ground_codes,
                        GROUND_FOOTPRINT_CONFIRMED=False,
                        CHANGE_USE="DIAGNOSTIC_ONLY_NOT_GROUND_CHANGE"))
                for state_name, code in (("stable", 1), ("addition", 2),
                                         ("demolition", 3), ("unresolved", 4)):
                    layer = np.zeros(base_diagnostic.shape, dtype=np.uint8)
                    layer[base_diagnostic == code] = 1
                    _write_tif(
                        output_dir / f"{stem}_{prefix}_{state_name}_NON_GROUND_BASE10M.tif",
                        layer, base_transform, nodata=int(NODATA),
                        tags=interval_tif_tags(
                            f"{prefix.upper()}_{state_name.upper()}_DIAGNOSTIC_NOT_GROUND",
                            left, right, GRID_LEVEL="REFERENCE_10M",
                            VALUE_CODES_JSON={"0": "BACKGROUND", "1": state_name.upper(),
                                              str(int(NODATA)): "NODATA"},
                            GROUND_FOOTPRINT_CONFIRMED=False,
                            CHANGE_USE="DIAGNOSTIC_ONLY_NOT_GROUND_CHANGE"))
        base_grid_products = True
    grid_receipt = {
        "schema_version": "xian_r5.grid_alignment_receipt.v1",
        "crs": "EPSG:32649",
        "fine_grid": {
            "resolution_m": resolution_m, "width": width, "height": height,
            "transform": list(transform),
            "bounds": [west, south, east, north],
        },
        "reference_grid": anchor_metadata,
        "nested_10m_grid": None,
        "fine_to_10m_nesting_pass": None,
        "fine_to_reference_alignment_pass": None,
    }
    if anchor_metadata is not None:
        factor = int(round(anchor_resolution / resolution_m))
        base_transform = from_origin(west, north, anchor_resolution, anchor_resolution)
        grid_receipt["nested_10m_grid"] = {
            "resolution_m": anchor_resolution,
            "width": width // factor, "height": height // factor,
            "transform": list(base_transform),
            "bounds": [west, south, east, north],
            "fine_pixels_per_10m_pixel": factor,
        }
        grid_receipt["fine_to_10m_nesting_pass"] = bool(
            width % factor == 0 and height % factor == 0
            and abs(base_transform.c - transform.c) < 1e-9
            and abs(base_transform.f - transform.f) < 1e-9
            and abs(base_transform.a - transform.a * factor) < 1e-9
            and abs(base_transform.e - transform.e * factor) < 1e-9
            and width == (width // factor) * factor
            and height == (height // factor) * factor)
        col, row, cols, rows = anchor_metadata["base_window"]
        reference_transform = anchor_metadata["transform"]
        grid_receipt["fine_to_reference_alignment_pass"] = bool(
            abs(west - (reference_transform[2] + col * anchor_resolution)) < 1e-8
            and abs(north - (reference_transform[5] - row * anchor_resolution)) < 1e-8
            and cols == width // factor and rows == height // factor)
    grid_receipt_path = output_dir / "GRID_ALIGNMENT_RECEIPT.json"
    grid_receipt_path.write_text(
        json.dumps(grid_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": PRODUCT_SCHEMA_VERSION, "crs": "EPSG:32649",
        "registration_status_contract": "v1_STATUS_ALIAS_OF_REGISTRATION_QUALITY",
        "unresolved_change_support": "RAW_OR_REGISTERED_ROOF_SUPPORT_IS_UNRESOLVED_ONLY_NOT_CONFIRMED_CHANGE",
        "tif_metadata_contract": {
            "version": "xian_r5.tif_metadata.v2",
            "actual_dates_source": "TILE_ACTUAL_DATE_REGISTRY",
            "annual_reference_date_status": "UNKNOWN_ANNUAL_FILE_LABEL_ONLY",
            "base_grid_pixel_provenance_status": (
                "UNKNOWN_PER_PIXEL_SOURCE_COMPONENT_DATE_AND_VIEW_ANGLE"
                if anchor_metadata else "NOT_APPLICABLE"),
            "candidate_roof_semantics": "IMAGE_GRID_ROOF_PROXY_NOT_GROUND_FOOTPRINT",
            "positional_accuracy_claimed": False,
            "relative_alignment_is_absolute_ground_accuracy": False,
        },
        "grid": {"west": west, "north": north, "resolution_m": resolution_m, "width": width, "height": height,
                 "transform": list(transform), "anchored_to_reference_pixels": anchor_metadata is not None,
                 "reference_pixel_size_m": anchor_resolution,
                 "fine_pixels_per_reference_pixel": int(round(anchor_resolution / resolution_m)) if anchor_resolution else None},
        "grid_alignment_receipt": "GRID_ALIGNMENT_RECEIPT.json",
        "grid_alignment_receipt_status": {
            "fine_to_10m_nesting_pass": grid_receipt["fine_to_10m_nesting_pass"],
            "fine_to_reference_alignment_pass": grid_receipt["fine_to_reference_alignment_pass"],
        },
        "geometry_layer_contract": {
            "roof_raw": "building_lineage.gpkg#roof_raw",
            "roof_registered": "building_lineage.gpkg#roof_registered",
            "ground_footprints": "building_lineage.gpkg#ground_footprints",
            "canonical_building_geometry": "building_lineage.gpkg#canonical_building_geometry",
            "canonical_geometry_evidence": "building_lineage.gpkg#canonical_geometry_evidence",
            "registered_roof_acceptance": (
                "ONLY_ACCEPTED_REVIEWED_IMAGE_GRID_SHIFT;NEVER_GROUND_FOOTPRINT"),
            "canonical_geometry_acceptance": (
                "REVIEWED_PHYSICAL_IDENTITY_PLUS_ACCEPTED_REGISTERED_ROOFS_ON_AT_LEAST_TWO_ACTUAL_DATES;"
                "OBSERVED_MEDOID_ONLY_NOT_GROUND_TRUTH"),
            "raw_roof_promoted_to_ground": False,
            "registered_roof_promoted_to_ground": False,
        },
        "canonical_geometry": {
            "row_count": len(canonical_rows),
            "resolved_count": sum(row["geometry"] is not None for row in canonical_rows),
            "unresolved_count": sum(row["geometry"] is None for row in canonical_rows),
            "evidence_count": len(canonical_evidence),
            "metrics": ["centroid_distance_m", "iou", "hausdorff_boundary_distance_m",
                        "area_ratio", "shape_similarity"],
            "selection": "MINIMUM_MEAN_BOUNDARY_HAUSDORFF_MEDOID;THEN_LOWER_POSITION_UNCERTAINTY;THEN_UID",
            "physical_id_codebook": {
                "path": "PHYSICAL_ID_CODEBOOK.json",
                "sha256": physical_id_codebook_sha256,
                "count": len(physical_id_codebook["code_to_reviewed_identity"]),
            },
        },
        "reference_grid": anchor_metadata,
        "reference_acquisition_date_status": (
            "UNKNOWN_ANNUAL_FILE_LABEL_ONLY" if anchor_metadata else "NOT_APPLICABLE"),
        "native_reference_grid_products_written": base_grid_products,
        "epochs": output_groups,
        "nominal_epochs": nominal_epochs,
        "actual_date_order": actual_date_order,
        "output_groups": output_groups,
        "output_group_metadata": output_group_metadata,
        "epoch_order_basis": "UNIQUE_ACTUAL_ACQUISITION_DATE;UNKNOWN_DATES_EXCLUDED_FROM_TRANSITIONS",
        "actual_dates_by_epoch": actual_dates_by_epoch,
        "epochs_with_mixed_actual_dates": [epoch for epoch, values in actual_dates_by_epoch.items()
                                           if len(values) > 1],
        "min_area_m2_post_state": min_area_m2,
        "footprint_codes": {"background": 0, "confirmed_ground": 1, "unresolved": int(U), "nodata": int(NODATA)},
        "id_codes": {"background": 0, "unresolved": int(ID_UNKNOWN), "nodata": None},
        "candidate_track_id_codes": {
            "background": 0, "unresolved_or_ambiguous": int(ID_UNKNOWN), "nodata": None},
        "candidate_track_id_codebook": {
            "path": candidate_track_codebook_name,
            "sha256": candidate_track_codebook_sha256,
            "candidate_identity_label_count": len(candidate_track_codes),
            "code_semantics": "HASHED_SOURCE_CANDIDATE_TRACK_LABEL_NOT_PHYSICAL_BUILDING_ID",
        },
        "candidate_track_id_semantics": {
            "geometry": "REVIEWED_IMAGE_GRID_ALIGNED_ROOF_PROXY_WHEN_AVAILABLE_ELSE_RAW_ROOF_PROXY",
            "identity_source": "SOURCE_CANDIDATE_IDENTITY_LABEL",
            "physical_identity_confirmed": False,
            "formal_physical_id_product": False,
            "used_for_presence_or_change": False,
            "repeated_label_within_actual_date": "ALL_PIXELS_UNRESOLVED",
            "overlapping_different_labels": "OVERLAP_PIXELS_UNRESOLVED",
            "missing_label": "ROOF_PIXELS_UNRESOLVED",
        },
        "candidate_track_id_quality_codes": CANDIDATE_TRACK_QUALITY_CODES,
        "candidate_track_id_products": candidate_track_id_products,
        "candidate_track_ids_promoted_to_physical_ids": False,
        "change_codes": {"background": 0, **STATE_CODES, "nodata": int(NODATA)},
        "base_change_aggregation": "ANY_SUBPIXEL_SINGLE_STATE;MIXED_STATES_OR_UNRESOLVED_TO_U",
        "quality_codes": {"background": 0, "reviewed_ground": 1, "roof_only_unresolved": 2,
                          "unverified_ground_candidate": 3, "ambiguous_ground_overlap": 4},
        "ground_footprint_candidate_codes": {
            "background": 0, "unverified_candidate": 1, "nodata": int(NODATA)},
        "external_candidate_source_epoch_labels": sorted({
            candidate.source_epoch_label
            for obs in result.observations for candidate in obs.footprint_candidates
            if candidate.source_epoch_label is not None
        }),
        "external_candidate_source_epoch_label_semantics": (
            "USER_REPORTED_EPOCH_LABEL_NOT_EXACT_ACQUISITION_DATE;"
            "NOT_GROUND_TRUTH_OR_POSITIONAL_ACCURACY"),
        "external_candidate_source_date_status_counts": {
            status: sum(
                candidate.source_date_status == status
                for obs in result.observations for candidate in obs.footprint_candidates
            )
            for status in sorted({
                candidate.source_date_status
                for obs in result.observations for candidate in obs.footprint_candidates
            })
        },
        "external_candidate_source_date_candidates": sorted({
            source_date
            for obs in result.observations for candidate in obs.footprint_candidates
            for source_date in candidate.source_date_candidates
        }),
        "external_candidate_source_scene_linkage_verified": False,
        "external_candidate_source_geometry_semantics": sorted({
            candidate.source_geometry_semantics
            for obs in result.observations for candidate in obs.footprint_candidates
            if candidate.source_geometry_semantics is not None
        }),
        "external_candidate_source_geometry_semantics_provenance": (
            "USER_REPORTED_SEMANTICS_NOT_INDEPENDENTLY_VERIFIED;"
            "ROOF_BOUNDARY_FROM_IMAGE_IS_NOT_A_GROUND_FOOTPRINT"),
        "ground_footprint_candidate_vector_layer": "building_lineage.gpkg#footprint_candidates",
        "ground_footprint_candidate_products": ground_candidate_products,
        "footprint_candidate_roof_source_layer": "building_lineage.gpkg#footprint_candidate_roof_sources",
        "footprint_candidate_roof_source_count": sum(
            candidate.source_roof_geometry is not None
            for obs in result.observations for candidate in obs.footprint_candidates),
        "rbf_sam_candidate_semantics": (
            "NEGATIVE_FINAL_OFFSET_TRANSLATED_ROOF_MASK_CANDIDATE_ONLY;"
            "NOT_INDEPENDENT_GROUND_EVIDENCE;POSITION_UNCERTAINTY_NOT_EMITTED"),
        "image_alignment_quality_codes": alignment_quality_codes,
        "image_alignment_quality_products": alignment_quality_products,
        "image_alignment_scope": "IMAGE_TO_REFERENCE_GRID_ONLY_NOT_ROOF_TO_GROUND_PARALLAX",
        "image_alignment_evidence_semantics": {
            "relative_reference_feature_review": (
                "MAY_AUTHORIZE_IMAGE_TO_USER_BASE_GRID_CORRECTION_ONLY;"
                "ABSOLUTE_GROUND_POSITION_ACCURACY_REMAINS_UNKNOWN"),
            "unknown_source_scope_spatial_composition_relative_review": (
                "EXACT_CELL_FIT_AND_HOLDOUT_TIES_REQUIRED;SOURCE_COMPONENT_AND_PER_PIXEL_REFERENCE_PROVENANCE_REMAIN_UNKNOWN;"
                "IMAGE_GRID_ONLY_NOT_ABSOLUTE_GROUND_OR_PARALLAX"),
            "independent_ground_control_review": (
                "MAY_REPORT_SEPARATE_ABSOLUTE_POSITIONAL_UNCERTAINTY"),
            "roof_to_ground_parallax": "NOT_ESTABLISHED_BY_EITHER_IMAGE_GRID_REVIEW_BASIS",
        },
        "source_tile_provenance_semantics": (
            "EXACT_DATE_Z20_TILE_IDS_PATHS_AND_SOURCE_HASHES_FROM_VERIFIED_MOSAIC_RECEIPT;"
            "TRACEABILITY_ONLY_NOT_GROUND_CONTROL_OR_POSITIONAL_ACCURACY"),
        "source_component_scope_semantics": (
            "ABSOLUTE_GROUND_REVIEW_AND_UNREVIEWED_ALIGNMENT_REQUIRE_FULL_CELL_COVERAGE_BY_ONE_HASH_IDENTIFIED_SOURCE_TILE;"
            "UNKNOWN_SCOPE_SPATIAL_COMPOSITION_MAY_ONLY_USE_EXACT_CELL_RELATIVE_FEATURE_REVIEW_WITHIN_CELL_FIT_AND_HOLDOUT_TIES;"
            "SOURCE_COMPONENT_AND_PER_PIXEL_REFERENCE_PROVENANCE_STAY_UNKNOWN_AND_NO_GROUND_OR_PARALLAX_CLAIM_IS_MADE"),
        "source_tile_provenance_layers": [
            "image_alignment_candidates", "image_grid_roof_candidates",
            "registered_roof_candidates", "roof_raw", "roof_registered"],
        "reference_window_provenance_semantics": (
            "CELL_PIXEL_WINDOWS_AND_ANNUAL_RASTER_PATH_HASH_ARE_TRACEABILITY_ONLY;"
            "PER_PIXEL_MOSAIC_COMPONENT_ACQUISITION_DATE_AND_VIEW_ANGLE_REMAIN_UNKNOWN"),
        "reference_window_provenance_layers": [
            "image_alignment_candidates", "image_grid_roof_candidates",
            "registered_roof_candidates", "roof_registered"],
        "image_alignment_field_interpolation": False,
        "image_alignment_field_applied": any(
            obs.registered_roof is not None for obs in result.observations),
        "registered_roof_candidate_count": sum(
            obs.registered_roof is not None for obs in result.observations),
        "registered_roof_candidate_layer": "building_lineage.gpkg#registered_roof_candidates",
        "roof_raw_layer": "building_lineage.gpkg#roof_raw",
        "roof_raw_count": len(result.observations),
        "roof_registered_layer": "building_lineage.gpkg#roof_registered",
        "roof_registered_count": sum(obs.registered_roof is not None
                                      for obs in result.observations),
        "ground_footprint_layer": "building_lineage.gpkg#ground_footprints",
        "ground_footprint_count": sum(obs.ground_footprint is not None
                                       for obs in result.observations),
        "image_alignment_candidate_preview_rendered": bool(roof_alignment_products),
        "image_grid_roof_candidate_codes": {
            "background": 0, "unaccepted_candidate": 1,
            "unresolved": int(U), "nodata": int(NODATA),
        },
        "image_grid_roof_candidate_policy": {
            "full_cell": "ONE_UNIQUE_FIELD_CELL_COVERS_THE_WHOLE_ROOF",
            "source_component_gate": (
                "ONE_HASH_IDENTIFIED_SOURCE_TILE_PER_ACTUAL_DATE_REQUIRED_EXCEPT_EXACT_SPATIAL_COMPOSITION_CELL_WITH_REVIEWED_RELATIVE_TIES_AND_HOLDOUTS_WITHIN_CELL;"
                "EXCEPTION_IS_RELATIVE_GRID_ONLY_NOT_ABSOLUTE_GROUND_OR_PARALLAX"),
            "dominant_cell_candidate_only": {
                "method": "MAXIMUM_ROOF_AREA_INTERSECTION_FRACTION",
                "minimum_fraction": 0.8,
                "accepted_partial_cell_application": False,
                "whole_geometry_translation": True,
            },
            "piecewise_warp": False,
            "roof_to_ground_correction": False,
            "cross_component_candidate_shift": False,
        },
        "image_grid_roof_candidate_quality_codes": roof_alignment_quality_codes,
        "image_grid_roof_candidate_products": roof_alignment_products,
        "image_grid_roof_outline_diagnostics": {
            "layer": "building_lineage.gpkg#image_grid_roof_outline_diagnostics",
            "row_count": outline_diagnostic_count,
            "comparison": "RAW_ROOF_VS_IMAGE_GRID_SHIFTED_ROOF_AGAINST_UNVERIFIED_EXTERNAL_OUTLINE",
            "alignment_scope": "IMAGE_TO_REFERENCE_GRID_ONLY",
            "unshifted_metrics": "NULL_WHEN_NO_NUMERIC_IMAGE_GRID_SHIFT_RENDERED",
            "ground_status": "NOT_VERIFIED",
            "formal_ground_use": False,
            "formal_change_use": False,
            "metric_semantics": {
                "raw_iou": "Raw roof versus external outline polygon IoU",
                "image_grid_iou": "Image-grid-shifted roof candidate versus external outline; null when no shift rendered",
                "image_grid_iou_gain": "Image-grid candidate IoU minus raw roof IoU; diagnostic only",
                "outline_dx_from_shifted_m": "External outline centroid easting minus shifted roof centroid easting",
                "outline_dy_from_shifted_m": "External outline centroid northing minus shifted roof centroid northing",
            },
        },
        "formal_ground_products_use_candidate_roof_geometry": False,
        "footprint_candidate_metric_semantics": {
            "intersection_area_m2": "positive intersection area in EPSG:32649",
            "roof_overlap_fraction": "intersection divided by roof candidate area",
            "candidate_overlap_fraction": "intersection divided by candidate outline area",
            "candidate_centroid_dx_from_roof_m": "candidate centroid easting minus roof centroid easting",
            "candidate_centroid_dy_from_roof_m": "candidate centroid northing minus roof centroid northing",
            "centroid_distance_m": "centroid separation; diagnostic only and never applied as a shift",
            "candidate_polygon_distance_m": "minimum separation between roof and external candidate polygons",
            "candidate_search_radius_m": "explicit maximum polygon distance for a disjoint candidate; null for positive-area overlaps",
            "centroid_aligned_roof_iou": "IoU after translating a copy of roof geometry by candidate centroid delta; diagnostic only, never applied",
            "centroid_aligned_iou_gain": "centroid-aligned roof IoU minus original IoU; diagnostic only",
        },
        "image_alignment_fields": [{
            "field_id": field.field_id,
            "method": field.cells[0].method if field.cells else None,
            "reference_date": field.reference_date,
            "reference_label": field.reference_label,
            "reference_source": field.reference_source,
            "reference_date_status": field.reference_date_status,
            "moving_date": field.moving_date,
            "cell_count": len(field.cells),
            "accepted_cell_count": sum(cell.accepted for cell in field.cells),
            "applied_cell_count": sum(
                (field.field_id, cell.cell_id) in {
                    (obs.image_alignment_field_id, obs.image_alignment_cell_id)
                    for obs in result.observations if obs.registered_roof is not None
                } for cell in field.cells),
            "numeric_candidate_cell_count": sum(cell.correction_dx_m_candidate is not None
                                                  for cell in field.cells),
            "review_basis_counts": {
                basis: sum(cell.alignment_review_basis == basis for cell in field.cells)
                for basis in sorted({cell.alignment_review_basis for cell in field.cells})
            },
            "relative_reviewed_cell_count": sum(
                cell.alignment_review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES"
                and cell.accepted for cell in field.cells),
            "relative_cell_source_scope_authorization_count": sum(
                cell.relative_cell_source_scope_review_authorized for cell in field.cells),
            "absolute_ground_control_cell_count": sum(
                cell.alignment_review_basis == "INDEPENDENT_GROUND_CONTROL"
                and cell.accepted for cell in field.cells),
            "absolute_position_accuracy_claimed": False,
            "cross_anchor_dispersion_is_positional_uncertainty": (
                False if field.cells and field.cells[0].method.startswith("MULTI_ANCHOR_")
                else None),
            "correction_scope": field.correction_scope,
            **_field_source_component_summary(field),
            **_field_tile_grid_summary(field),
            **_field_reference_window_summary(field),
        } for field in alignment_fields],
        "roof_products_are_diagnostic": True,
        "formal_ground_observations": sum(o.ground_footprint is not None for o in result.observations),
        "confirmed_ground_geometry_observations": sum(o.ground_footprint is not None for o in result.observations),
        "confirmed_physical_identity_observations": sum(o.physical_id is not None for o in result.observations),
        "identity_and_ground_jointly_confirmed_observations": sum(
            o.ground_footprint is not None and o.physical_id is not None
            for o in result.observations),
        "unresolved_observations": sum(o.ground_footprint is None or o.physical_id is None for o in result.observations),
        "identity_graph_node_count": len(result.identity_graph),
        "presence_states_layer": "building_lineage.gpkg#building_presence",
        "presence_table_codes": {"present": "1", "absent": "0", "unresolved": "U"},
        "presence_raster_codes": {
            "background": 0, "present": 1, "absent": 2,
            "unresolved": 3, "nodata": int(NODATA)},
        "presence_state_counts": {
            state: sum(row.get("presence") == state for row in result.presence)
            for state in ("1", "0", "U")},
        "presence_state_semantics": (
            "ONE_ROW_PER_OBSERVATION_OR_ACCEPTED_REVIEWED_ABSENCE;"
            "ABSENCE_REQUIRES_FULL_FOOTPRINT_COVERAGE;UNKNOWN_IS_NOT_ABSENCE"),
        "presence_raster_diagnostics": presence_raster_diagnostics,
        "presence_raster_geometry_policy": (
            "TABLE_IS_AUTHORITATIVE;RASTER_USES_REVIEWED_GROUND_ELSE_REGISTERED_ROOF_ELSE_RAW_OBSERVATION_LOCATOR;"
            "ABSENCE_RASTER_USES_LAST_KNOWN_GEOMETRY;NOT_A_GROUND_FOOTPRINT_CLAIM"),
        "non_ground_change_products": {
            "registered": sorted(
                f"{stem}_registered_change_NON_GROUND.tif" for stem in registered_interval_arrays),
            "canonical": sorted(
                f"{stem}_canonical_change_NON_GROUND.tif" for stem in canonical_interval_arrays),
            "ground_change_derived_from_non_ground_geometry": False,
        },
        "registration_method_registry": [{
            "field_id": field.field_id,
            "methods": sorted({cell.method for cell in field.cells}),
            "cell_count": len(field.cells),
            "accepted_cell_count": sum(cell.accepted for cell in field.cells),
            "applied_roof_count": sum(
                obs.registered_roof is not None
                and obs.image_alignment_field_id == field.field_id
                for obs in result.observations),
            "run_status": "EXISTING_METHOD_RECEIPT_ONLY_NO_NEW_DIAGNOSTIC_EXPERIMENT",
        } for field in alignment_fields],
        "candidate_identity_edge_count": len(result.candidate_edges),
        "candidate_identity_edge_basis_counts": {
            basis: sum(edge.get("edge_basis", "TBIA_GEOMETRIC") == basis
                       for edge in result.candidate_edges)
            for basis in sorted({edge.get("edge_basis", "TBIA_GEOMETRIC")
                                 for edge in result.candidate_edges})},
        "candidate_identity_edge_semantics": (
            "TBIA_GEOMETRIC_AND_SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCES;"
            "CANDIDATE_ONLY_NEVER_PROMOTED_TO_PHYSICAL_ID"),
        "candidate_identity_edge_search_radius_semantics": {
            "TBIA_GEOMETRIC": (
                "PAIR_SPECIFIC_MAXIMUM_CENTROID_SEPARATION;KNOWN_ENDPOINT_SIGMAS_COMBINED_IN_QUADRATURE;"
                "WHEN_ANY_ENDPOINT_SIGMA_IS_UNKNOWN_USE_CONFIGURED_SEARCH_ONLY_BUFFER_FLOOR"),
            "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE": "NULL_NOT_APPLICABLE",
            "unknown_search_buffer_is_positional_accuracy": False,
            "search_radius_promotes_identity": False,
        },
        "candidate_identity_edge_status_counts": {
            status: sum(edge["status"] == status for edge in result.candidate_edges)
            for status in sorted({edge["status"] for edge in result.candidate_edges})},
        "identity_graph_nodes_use_roof_proxy_when_ground_unverified": True,
        "candidate_edges_promoted_to_physical_identity": False,
    }
    (output_dir / "PRODUCT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
