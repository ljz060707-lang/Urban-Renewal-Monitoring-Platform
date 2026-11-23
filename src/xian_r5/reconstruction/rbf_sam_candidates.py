"""Ingest RBF-SAM roof-mask/offset proposals as unverified ground candidates.

This module deliberately separates inference from geospatial reconstruction.
It accepts a provenance-bearing prediction bundle, checks its exact observation
date/tile/image association, converts model-input pixels to EPSG:32649, and
keeps both the predicted roof geometry and translated footprint candidate.
Model proposals never satisfy the independent reviewed-ground evidence gate.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from affine import Affine
import numpy as np
from pyproj import CRS, Transformer
from rasterio.features import shapes as raster_shapes
from shapely.affinity import affine_transform, translate
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union

from ..alignment.mosaic_grid import MosaicTileGrid
from .core import ExternalFootprintCandidate, Observation


PREDICTION_SCHEMA = "xian_r5.rbf_sam_prediction_bundle.v2"
OFFSET_SEMANTICS = "RBF_SAM_BONAI_FINAL_OFFSET;APPLY_NEGATIVE_TO_ROOF_MASK"
_HEX_256 = re.compile(r"[0-9a-fA-F]{64}")
_PREDICTION_ID = re.compile(r"[A-Za-z0-9_.-]{1,72}")
_ALLOWED_PROMPT_SOURCES = {
    "OBSERVATION_ROOF", "DETECTED_ROOF_INSTANCE", "LOCAL_RESEGMENTED_ROOF",
}
_CANDIDATE_METHOD = "RBF_SAM_MASK_TRANSLATED_BY_NEGATIVE_FINAL_OFFSET"


def polygonize_rbf_sam_roof_mask(mask: Any) -> dict[str, Any]:
    """Convert one already-binarized RBF-SAM roof mask to pixel-space GeoJSON.

    The model's test path emits a binary mask after thresholding. This helper
    intentionally does not choose a probability threshold; non-binary inputs
    are rejected so the mask policy remains an explicit inference decision.
    """
    if hasattr(mask, "detach") and hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2 or not array.size:
        raise ValueError("RBF-SAM roof mask must be a nonempty 2D binary array")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("RBF-SAM roof mask must contain finite binary values")
    if not np.isin(array, (0, 1, False, True)).all():
        raise ValueError("RBF-SAM roof mask must already be binarized to 0/1")
    binary = array.astype(np.uint8, copy=False)
    polygons = [shape(geometry) for geometry, value in raster_shapes(
        binary, mask=binary.astype(bool), transform=Affine.identity(), connectivity=4)
        if int(value) == 1]
    if not polygons:
        raise ValueError("RBF-SAM roof mask contains no positive pixels")
    geometry = unary_union(polygons)
    if (geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or geometry.is_empty or not geometry.is_valid):
        raise ValueError("Polygonized RBF-SAM roof mask is not a valid polygon geometry")
    return mapping(geometry)


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while being hashed: {path}")
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _finite_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label} must contain numeric values, not strings or booleans")
    pair = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in pair):
        raise ValueError(f"{label} must be finite")
    return pair


def _date(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an exact ISO acquisition date")
    try:
        normalized = date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(f"{label} must be an exact ISO acquisition date") from exc
    if normalized != value:
        raise ValueError(f"{label} must use canonical YYYY-MM-DD format")
    return normalized


def _polygon_geometry(value: Any, label: str) -> BaseGeometry:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a GeoJSON Polygon or MultiPolygon")
    try:
        geometry = shape(value)
    except Exception as exc:
        raise ValueError(f"Invalid {label}") from exc
    if (geometry.geom_type not in {"Polygon", "MultiPolygon"}
            or geometry.is_empty or not geometry.is_valid or geometry.area <= 0):
        raise ValueError(f"{label} must be a nonempty valid Polygon or MultiPolygon")
    return geometry


def _pixel_geometry_to_ground(
    pixel_geometry: BaseGeometry,
    *,
    pixel_to_crs: tuple[float, float, float, float, float, float],
    pixel_crs: CRS,
) -> BaseGeometry:
    affine = Affine(*pixel_to_crs)
    # shapely.affinity uses [a, b, d, e, xoff, yoff], matching Affine's
    # pixel-corner mapping (x, y) -> (a*x+b*y+c, d*x+e*y+f).
    geometry_in_crs = affine_transform(
        pixel_geometry,
        [affine.a, affine.b, affine.d, affine.e, affine.c, affine.f],
    )
    if pixel_crs.to_epsg() != 32649:
        transformer = Transformer.from_crs(pixel_crs, "EPSG:32649", always_xy=True)
        geometry_in_crs = transform(transformer.transform, geometry_in_crs)
    if (geometry_in_crs.is_empty or not geometry_in_crs.is_valid
            or geometry_in_crs.geom_type not in {"Polygon", "MultiPolygon"}):
        raise ValueError("Transformed candidate geometry is invalid in EPSG:32649")
    return geometry_in_crs


def _read_bundle(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    digest = _sha256_file(path)
    content = path.read_text(encoding="utf-8")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Prediction bundle changed while being read: {path}")
    try:
        bundle = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid RBF-SAM prediction JSON: {path}") from exc
    if not isinstance(bundle, dict):
        raise ValueError("RBF-SAM prediction bundle root must be an object")
    return bundle, digest


def _read_mosaic_diagnostic(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    digest = _sha256_file(path)
    content = path.read_text(encoding="utf-8")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Mosaic date diagnostic changed while being read: {path}")
    try:
        diagnostic = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid mosaic date diagnostic JSON: {path}") from exc
    if (not isinstance(diagnostic, dict)
            or diagnostic.get("schema_version") != "xian_r5.mosaic_registration_diagnostic.v1"
            or diagnostic.get("scope") != "FIXED_CASE_LOCAL_ONLY"):
        raise ValueError("RBF-SAM import requires a fixed-case mosaic date diagnostic")
    if not isinstance(diagnostic.get("submeter_tiles"), dict):
        raise ValueError("Mosaic date diagnostic has no exact-date source-tile registry")
    return diagnostic, digest


def _resolve_source_tile_grid(
    *,
    actual_date: str,
    tile_id: str,
    source_image_path: Path,
    source_image_sha256: str,
    diagnostic: dict[str, Any],
    diagnostic_path: Path,
    mosaic_grids_by_source_epoch: Mapping[str, MosaicTileGrid],
) -> tuple[str, MosaicTileGrid, dict[str, Any]]:
    match = re.fullmatch(r"(\d+):(\d+)", tile_id)
    if match is None:
        raise ValueError(f"Observation tile ID is not an exact z20 coordinate: {tile_id}")
    tile_x, tile_y = (int(value) for value in match.groups())
    entries = diagnostic["submeter_tiles"].get(actual_date)
    if not isinstance(entries, list):
        raise ValueError(f"Mosaic date diagnostic has no source-tile list for {actual_date}")
    tile_entries = [entry for entry in entries
                    if isinstance(entry, dict) and entry.get("tile_id") == tile_id]
    if len(tile_entries) != 1:
        raise ValueError(
            f"Mosaic date diagnostic must identify exactly one tile {actual_date}/{tile_id}"
        )
    entry = tile_entries[0]
    entry_path = entry.get("path")
    if not isinstance(entry_path, str) or Path(entry_path).resolve() != source_image_path:
        raise ValueError("Mosaic actual-date receipt source path does not match prediction image")
    receipt_hash = entry.get("sha256")
    if (not isinstance(receipt_hash, str)
            or _require_hash(receipt_hash, "mosaic source tile sha256") != source_image_sha256):
        raise ValueError("Mosaic actual-date receipt hash does not match prediction source image")

    grid_matches: list[tuple[str, MosaicTileGrid]] = []
    for epoch, grid in mosaic_grids_by_source_epoch.items():
        if not isinstance(grid, MosaicTileGrid):
            raise ValueError(f"Invalid stitched source grid for epoch {epoch}")
        try:
            expected_path = grid.source_path_for_tile(tile_x, tile_y).resolve()
        except ValueError:
            continue
        if expected_path == source_image_path:
            grid_matches.append((str(epoch), grid))
    if len(grid_matches) != 1:
        raise ValueError(
            f"Prediction image must resolve to exactly one validated source grid; "
            f"found {len(grid_matches)} for {actual_date}/{tile_id}"
        )
    epoch, grid = grid_matches[0]
    recorded_epoch = entry.get("selected_source_epoch")
    if recorded_epoch is not None and str(recorded_epoch) != epoch:
        raise ValueError("Mosaic actual-date source epoch disagrees with exact tile path")
    component = grid.component_receipt(
        tile_x,
        tile_y,
        actual_date=actual_date,
        actual_date_source_ref=(
            f"{diagnostic_path.resolve()}#submeter_tiles/{actual_date}/{tile_id}"
        ),
        expected_sha256=source_image_sha256,
        expected_path=source_image_path,
    )
    if component["source_tile_sha256"] != receipt_hash.lower():
        raise ValueError("Source tile bytes changed since the actual-date mosaic receipt")
    return epoch, grid, component


def attach_rbf_sam_footprint_candidates(
    observations: Iterable[Observation],
    prediction_path: Path,
    *,
    mosaic_diagnostic_path: Path,
    mosaic_grids_by_source_epoch: Mapping[str, MosaicTileGrid],
) -> dict[str, Any]:
    """Attach exact-date RBF-SAM proposals without accepting them as ground.

    Each prediction row contains ``observation_uid``, ``actual_date``,
    ``tile_id``, ``source_image_path``/``source_image_sha256``,
    ``model_input_sha256``, model-input dimensions, pixel-space GeoJSON
    ``roof_geometry_pixel``, ``predicted_offset_xy_px``, and a non-benchmark
    ``prompt_source``. The affine is derived here from the exact source-tile
    window in the validated GEID/mosaic grid; callers may not provide one.
    The caller remains responsible for producing the model bundle using a
    current detected/observed roof prompt, not benchmark GT.
    """
    records = list(observations)
    if not records:
        raise ValueError("At least one observation is required")
    by_uid: dict[str, Observation] = {}
    for observation in records:
        if not observation.uid or observation.uid in by_uid:
            raise ValueError("Observation UIDs must be nonempty and unique")
        by_uid[observation.uid] = observation

    prediction_path = Path(prediction_path).resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    mosaic_diagnostic_path = Path(mosaic_diagnostic_path).resolve()
    if not mosaic_diagnostic_path.is_file():
        raise FileNotFoundError(mosaic_diagnostic_path)
    if not mosaic_grids_by_source_epoch:
        raise ValueError("Validated stitched-mosaic grids are required for RBF-SAM import")
    mosaic_diagnostic, mosaic_diagnostic_sha256 = _read_mosaic_diagnostic(
        mosaic_diagnostic_path)
    bundle, bundle_sha256 = _read_bundle(prediction_path)
    if bundle.get("schema_version") != PREDICTION_SCHEMA:
        raise ValueError(f"Unsupported RBF-SAM prediction schema: {bundle.get('schema_version')}")
    if bundle.get("scope") != "FIXED_CASE_LOCAL_ONLY":
        raise ValueError("RBF-SAM candidate import is restricted to fixed-case local bundles")
    if bundle.get("offset_semantics") != OFFSET_SEMANTICS:
        raise ValueError("RBF-SAM bundle does not declare the audited negative-offset convention")

    model = bundle.get("model")
    if not isinstance(model, dict) or model.get("name") != "RBF-SAM":
        raise ValueError("RBF-SAM model metadata is required")
    model_revision = model.get("revision")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("RBF-SAM source revision is required")
    model_source = model.get("source")
    if not isinstance(model_source, str) or not model_source.strip():
        raise ValueError("RBF-SAM source reference is required")
    checkpoint_sha256 = _require_hash(model.get("checkpoint_sha256"), "checkpoint_sha256")
    config_sha256 = _require_hash(model.get("config_sha256"), "config_sha256")

    predictions = bundle.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("RBF-SAM prediction bundle must contain prediction rows")

    staged: dict[str, list[ExternalFootprintCandidate]] = {uid: [] for uid in by_uid}
    source_hash_cache: dict[Path, str] = {}
    seen_prediction_ids: set[str] = set()
    for row_number, row in enumerate(predictions, start=1):
        label = f"predictions[{row_number - 1}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        prediction_id = row.get("prediction_id")
        if not isinstance(prediction_id, str) or _PREDICTION_ID.fullmatch(prediction_id) is None:
            raise ValueError(f"{label}.prediction_id must be 1-72 safe ASCII characters")
        if prediction_id in seen_prediction_ids:
            raise ValueError(f"Duplicate RBF-SAM prediction_id: {prediction_id}")
        seen_prediction_ids.add(prediction_id)

        uid = row.get("observation_uid")
        observation = by_uid.get(uid) if isinstance(uid, str) else None
        if observation is None:
            raise ValueError(f"{label} does not identify a supplied observation UID")
        actual_date = _date(row.get("actual_date"), f"{label}.actual_date")
        if observation.actual_date is None or actual_date != observation.actual_date:
            raise ValueError(f"{label} actual date does not exactly match observation {uid}")
        if not observation.tile_id or row.get("tile_id") != observation.tile_id:
            raise ValueError(f"{label} tile ID does not exactly match observation {uid}")
        if not observation.image_path:
            raise ValueError(f"Observation {uid} has no image path for RBF-SAM provenance")
        source_image_value = row.get("source_image_path")
        if not isinstance(source_image_value, str) or not source_image_value.strip():
            raise ValueError(f"{label}.source_image_path is required")
        try:
            source_image_path = Path(source_image_value).resolve(strict=True)
            observation_image_path = Path(observation.image_path).resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{label} source image path cannot be resolved") from exc
        if source_image_path != observation_image_path:
            raise ValueError(f"{label} source image path does not match observation {uid}")
        recorded_source_hash = _require_hash(
            row.get("source_image_sha256"), f"{label}.source_image_sha256")
        actual_source_hash = source_hash_cache.get(source_image_path)
        if actual_source_hash is None:
            actual_source_hash = _sha256_file(source_image_path)
            source_hash_cache[source_image_path] = actual_source_hash
        if actual_source_hash != recorded_source_hash:
            raise ValueError(f"{label} source image hash mismatch")
        model_input_sha256 = _require_hash(
            row.get("model_input_sha256"), f"{label}.model_input_sha256")

        if "pixel_to_crs_affine" in row or "pixel_crs" in row:
            raise ValueError(
                f"{label} must not supply a pixel affine/CRS; these are derived "
                "from the audited GEID mosaic tile grid"
            )

        width, height = row.get("model_input_width"), row.get("model_input_height")
        if (isinstance(width, bool) or isinstance(height, bool)
                or not isinstance(width, int) or not isinstance(height, int)
                or width <= 0 or height <= 0):
            raise ValueError(f"{label} model-input dimensions must be positive integers")
        prompt_source = row.get("prompt_source")
        if (not isinstance(prompt_source, str)
                or prompt_source not in _ALLOWED_PROMPT_SOURCES):
            raise ValueError(f"{label} prompt_source must be a current roof-instance source")
        prompt_type = row.get("prompt_type")
        if (prompt_type is not None
                and (not isinstance(prompt_type, str)
                     or prompt_type not in {
                         "building_box", "roof_box", "footprint_box", "noisy_box"
                     })):
            raise ValueError(f"{label} prompt_type is not a supported RBF-SAM box prompt")
        prompt_box = row.get("prompt_box_xyxy_pixel")
        if prompt_box is not None:
            if (not isinstance(prompt_box, (list, tuple)) or len(prompt_box) != 4
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           for value in prompt_box)):
                raise ValueError(f"{label} prompt_box_xyxy_pixel must be four numeric coordinates")
            prompt_box = [float(value) for value in prompt_box]
            if not all(math.isfinite(value) for value in prompt_box):
                raise ValueError(f"{label} prompt_box_xyxy_pixel must be finite")
            prompt_x1, prompt_y1, prompt_x2, prompt_y2 = prompt_box
            if not (0 <= prompt_x1 < prompt_x2 <= width
                    and 0 <= prompt_y1 < prompt_y2 <= height):
                raise ValueError(f"{label} prompt_box_xyxy_pixel must be inside the model input")

        source_epoch, source_grid, component = _resolve_source_tile_grid(
            actual_date=actual_date,
            tile_id=observation.tile_id,
            source_image_path=source_image_path,
            source_image_sha256=actual_source_hash,
            diagnostic=mosaic_diagnostic,
            diagnostic_path=mosaic_diagnostic_path,
            mosaic_grids_by_source_epoch=mosaic_grids_by_source_epoch,
        )
        tile_x, tile_y = (int(value) for value in observation.tile_id.split(":"))
        model_grid = source_grid.model_input_grid_for_tile(
            tile_x,
            tile_y,
            model_input_width_px=width,
            model_input_height_px=height,
        )
        pixel_crs = CRS.from_user_input(model_grid.pixel_crs)
        affine_values = model_grid.pixel_to_crs_affine
        affine = Affine(*affine_values)

        roof_pixel = _polygon_geometry(row.get("roof_geometry_pixel"), f"{label}.roof_geometry_pixel")
        min_x, min_y, max_x, max_y = roof_pixel.bounds
        if min_x < 0 or min_y < 0 or max_x > width or max_y > height:
            raise ValueError(f"{label}.roof_geometry_pixel extends beyond the model-input image")
        offset_x, offset_y = _finite_pair(
            row.get("predicted_offset_xy_px"), f"{label}.predicted_offset_xy_px")

        source_roof = _pixel_geometry_to_ground(
            roof_pixel, pixel_to_crs=affine_values, pixel_crs=pixel_crs)
        translated_pixel = translate(roof_pixel, xoff=-offset_x, yoff=-offset_y)
        candidate_geometry = _pixel_geometry_to_ground(
            translated_pixel, pixel_to_crs=affine_values, pixel_crs=pixel_crs)
        candidate_id = f"RBF-{prediction_id}"
        if any(item.candidate_id == candidate_id for item in observation.footprint_candidates):
            raise ValueError(f"Candidate already attached: {uid}/{candidate_id}")

        properties = {
            "candidate_method": _CANDIDATE_METHOD,
            "candidate_status": "UNVERIFIED_MODEL_CANDIDATE",
            "prediction_id": prediction_id,
            "prediction_bundle_path": str(prediction_path),
            "prediction_bundle_sha256": bundle_sha256,
            "observation_uid": uid,
            "actual_date": actual_date,
            "tile_id": observation.tile_id,
            "source_image_path": str(source_image_path),
            "source_image_sha256": actual_source_hash,
            "model_input_sha256": model_input_sha256,
            "model_input_width": width,
            "model_input_height": height,
            "pixel_crs": pixel_crs.to_string(),
            "pixel_to_crs_affine": list(affine_values),
            "pixel_to_crs_affine_status": "DERIVED_FROM_VALIDATED_GEID_MOSAIC_TILE_GRID",
            "source_grid_epoch": source_epoch,
            "source_grid_scene_id": source_grid.scene_id,
            "source_mosaic_path": str(source_grid.mosaic_path),
            "source_mosaic_transform": list(source_grid.mosaic_transform),
            "source_pixel_window_col_row_width_height": list(
                model_grid.source_pixel_window),
            "actual_date_source_ref": component["actual_date_source_ref"],
            "mosaic_diagnostic_path": str(mosaic_diagnostic_path),
            "mosaic_diagnostic_sha256": mosaic_diagnostic_sha256,
            "predicted_offset_xy_px": [offset_x, offset_y],
            "applied_mask_translation_xy_px": [-offset_x, -offset_y],
            "offset_semantics": OFFSET_SEMANTICS,
            "prompt_source": prompt_source,
            "prompt_type": prompt_type,
            "prompt_box_xyxy_pixel": prompt_box,
            "model_name": "RBF-SAM",
            "model_source": model_source,
            "model_revision": model_revision,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "positional_uncertainty_m": None,
            "offset_confidence": "NOT_EMITTED",
            "independent_ground_evidence": False,
        }
        staged[uid].append(ExternalFootprintCandidate(
            geometry=candidate_geometry,
            candidate_id=candidate_id,
            source_ref=f"RBF-SAM:{checkpoint_sha256[:16]}:{prediction_id}",
            source_crs="EPSG:32649",
            source_capture_date=actual_date,
            association_method="RBF_SAM_BY_OBSERVATION_UID",
            source_properties=properties,
            source_roof_geometry=source_roof,
        ))

    for uid, candidates in staged.items():
        by_uid[uid].footprint_candidates.extend(candidates)
    attached = [candidate for candidates in staged.values() for candidate in candidates]
    return {
        "schema_version": "xian_r5.rbf_sam_candidate_import.v2",
        "prediction_bundle_path": str(prediction_path),
        "prediction_bundle_sha256": bundle_sha256,
        "mosaic_diagnostic_path": str(mosaic_diagnostic_path),
        "mosaic_diagnostic_sha256": mosaic_diagnostic_sha256,
        "model_revision": model_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "prediction_count": len(predictions),
        "attached_candidate_count": len(attached),
        "candidate_ids": [candidate.candidate_id for candidate in attached],
        "ground_footprints_confirmed": 0,
        "position_uncertainty_emitted": False,
        "status": "UNVERIFIED_CANDIDATES_ATTACHED",
    }
