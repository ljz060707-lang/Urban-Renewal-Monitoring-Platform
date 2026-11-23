"""Attach human-drawn ground-boundary candidates without promoting them.

These records preserve a reviewer interpretation of a visible boundary while
keeping absolute positional uncertainty explicit. They never satisfy the
formal reviewed-ground-footprint or physical-identity gates.
"""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import fiona
import rasterio
from pyproj import CRS
from shapely.geometry import box, shape

from .core import ExternalFootprintCandidate, Observation


LAYER_NAME = "ground_footprint_review_candidates"
REQUIRED_FIELDS = {
    "uid", "actual_date", "candidate_id", "source_ref", "review_status",
    "reviewer_id", "reviewed_at", "evidence_path", "evidence_sha256",
    "uncertainty_basis", "positional_sigma_m", "notes",
}
UNCERTAINTY_BASES = {"UNKNOWN_ABSOLUTE_GEOREF", "RELATIVE_SOURCE_GRID_ONLY"}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_component_hashes(path: Path) -> dict[str, str]:
    components = [path]
    components.extend(
        Path(f"{path}{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
        if Path(f"{path}{suffix}").is_file()
    )
    return {str(component.resolve()): _sha256(component)
            for component in sorted(components)}


def _parse_date(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must use canonical YYYY-MM-DD form")
    return value


def _parse_reviewed_at(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reviewed_at is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO timestamp with timezone") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone")
    return value


def attach_reviewed_ground_footprint_candidates(
    observations: Iterable[Observation], source_path: Path, *,
    layer: str = LAYER_NAME,
) -> dict:
    """Attach exact-UID/date visual candidates; no candidate is ground truth.

    Candidate records must acknowledge that the image-grid absolute position
    is unknown or only relative. ``positional_sigma_m`` must therefore remain
    null; observations with independent absolute control belong in the strict
    ``reviewed_ground_footprints`` layer instead.
    """
    records = list(observations)
    if not records:
        raise ValueError("At least one observation is required")
    by_uid: dict[str, Observation] = {}
    for observation in records:
        if observation.uid in by_uid:
            raise ValueError(f"Duplicate observation UID: {observation.uid}")
        by_uid[observation.uid] = observation

    source_path = Path(source_path).resolve()
    if source_path.suffix.lower() != ".gpkg":
        raise ValueError("Reviewed footprint candidates must be supplied as a GeoPackage")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_hashes_before = _source_component_hashes(source_path)
    evidence_hashes_before: dict[str, str] = {}
    pending: dict[str, list[ExternalFootprintCandidate]] = {}
    candidate_keys: set[tuple[str, str]] = set()
    feature_count = 0
    source_schema = None

    with fiona.open(source_path, layer=layer) as source:
        source_crs_input = source.crs_wkt or source.crs
        if not source_crs_input:
            raise ValueError("Reviewed footprint candidate layer has no CRS")
        source_crs = CRS.from_user_input(source_crs_input)
        if source_crs.to_epsg() != 32649:
            raise ValueError("Reviewed footprint candidate layer must use EPSG:32649")
        source_schema = {
            "geometry": source.schema.get("geometry"),
            "properties": dict(source.schema.get("properties", {})),
        }
        missing = REQUIRED_FIELDS - set(source_schema["properties"])
        if missing:
            raise ValueError(f"Reviewed footprint candidate layer is missing fields: {sorted(missing)}")
        if source_schema["geometry"] not in {"Polygon", "MultiPolygon", "Unknown"}:
            raise ValueError("Reviewed footprint candidate geometry must be polygonal")

        feature_count = len(source)
        for row_number, feature in enumerate(source, start=1):
            props = dict(feature["properties"])
            uid = props.get("uid")
            if not isinstance(uid, str) or not uid.strip() or uid not in by_uid:
                raise ValueError(f"Candidate row {row_number} references an unknown observation UID")
            observation = by_uid[uid]
            actual_date = _parse_date(props.get("actual_date"), "actual_date")
            if observation.actual_date is None or actual_date != observation.actual_date:
                raise ValueError(f"Candidate actual date does not match observation {uid}")
            candidate_id = props.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError(f"candidate_id is required for {uid}")
            key = (uid, candidate_id.strip())
            if key in candidate_keys:
                raise ValueError(f"Duplicate reviewed candidate ID for observation: {uid}/{candidate_id}")
            candidate_keys.add(key)

            if props.get("review_status") != "VISIBLE_GROUND_BOUNDARY_CANDIDATE":
                raise ValueError(f"Unsupported candidate review_status for {uid}/{candidate_id}")
            uncertainty_basis = props.get("uncertainty_basis")
            if uncertainty_basis not in UNCERTAINTY_BASES:
                raise ValueError(f"Unsupported uncertainty_basis for {uid}/{candidate_id}")
            sigma = props.get("positional_sigma_m")
            if sigma is not None and not (isinstance(sigma, float) and math.isnan(sigma)):
                raise ValueError(
                    "positional_sigma_m must remain null for candidate-only image-grid geometry")

            source_ref = props.get("source_ref")
            reviewer_id = props.get("reviewer_id")
            if not isinstance(source_ref, str) or not source_ref.strip():
                raise ValueError(f"source_ref is required for {uid}/{candidate_id}")
            if not isinstance(reviewer_id, str) or not reviewer_id.strip():
                raise ValueError(f"reviewer_id is required for {uid}/{candidate_id}")
            reviewed_at = _parse_reviewed_at(props.get("reviewed_at"))

            evidence_path_raw = props.get("evidence_path")
            evidence_digest = props.get("evidence_sha256")
            if not isinstance(evidence_path_raw, str) or not evidence_path_raw.strip():
                raise ValueError(f"evidence_path is required for {uid}/{candidate_id}")
            if not isinstance(evidence_digest, str) or not _SHA256_RE.fullmatch(evidence_digest):
                raise ValueError(f"evidence_sha256 must be a 64-character SHA-256 for {uid}/{candidate_id}")
            evidence_path = Path(evidence_path_raw)
            if not evidence_path.is_absolute():
                evidence_path = source_path.parent / evidence_path
            evidence_path = evidence_path.resolve()
            if evidence_path.name != f"{actual_date}_RGB_UTM.tif":
                raise ValueError(
                    f"Candidate evidence must be the exact-date scene raster for {uid}")
            if str(evidence_path) in source_hashes_before:
                raise ValueError("The review GeoPackage cannot serve as independent image evidence")
            if not evidence_path.is_file():
                raise FileNotFoundError(f"Candidate image evidence does not exist: {evidence_path}")
            evidence_key = str(evidence_path)
            if evidence_key not in evidence_hashes_before:
                evidence_hashes_before[evidence_key] = _sha256(evidence_path)
            if evidence_hashes_before[evidence_key].casefold() != evidence_digest.casefold():
                raise ValueError(f"Evidence SHA-256 mismatch for {uid}/{candidate_id}")

            geometry_value = feature.get("geometry")
            if not geometry_value:
                raise ValueError(f"Candidate geometry is missing for {uid}/{candidate_id}")
            geometry = shape(geometry_value)
            if (geometry.geom_type not in {"Polygon", "MultiPolygon"}
                    or geometry.is_empty or not geometry.is_valid or geometry.area <= 0):
                raise ValueError(f"Candidate geometry is invalid for {uid}/{candidate_id}")
            with rasterio.open(evidence_path) as scene:
                scene_tags = scene.tags()
                if (scene.crs is None or scene.crs.to_epsg() != 32649
                        or scene_tags.get("PRODUCT_KIND") != "RAW_ACTUAL_DATE_SCENE_REVIEW_CONTEXT"
                        or scene_tags.get("ACTUAL_DATE") != actual_date
                        or scene_tags.get("ORTHORECTIFIED") != "false"):
                    raise ValueError(
                        f"Candidate evidence is not the tagged unorthorectified scene for {uid}")
                if not geometry.within(box(*scene.bounds)):
                    raise ValueError(
                        f"Candidate geometry falls outside the exact-date scene bounds for {uid}")

            intersection = float(observation.roof.intersection(geometry).area)
            union = float(observation.roof.union(geometry).area)
            roof_center, candidate_center = observation.roof.centroid, geometry.centroid
            source_properties = {
                "review_status": "VISIBLE_GROUND_BOUNDARY_CANDIDATE",
                "reviewer_id": reviewer_id.strip(),
                "reviewed_at": reviewed_at,
                "uncertainty_basis": uncertainty_basis,
                "positional_sigma_m": None,
                "notes": str(props.get("notes") or ""),
                "candidate_only": True,
                "formal_ground_footprint_confirmed": False,
            }
            pending.setdefault(uid, []).append(ExternalFootprintCandidate(
                geometry=geometry,
                candidate_id=candidate_id.strip(),
                source_ref=source_ref.strip(),
                source_crs="EPSG:32649",
                source_capture_date=actual_date,
                source_date_status="EXACT_ACTUAL_DATE",
                source_tile_ids=tuple(observation.covered_tiles),
                source_scene_linkage_status="HASH_VERIFIED_ACTUAL_DATE_SCENE",
                source_date_provenance=evidence_key,
                source_geometry_semantics="VISIBLE_GROUND_BOUNDARY_CANDIDATE",
                source_geometry_semantics_provenance="MANUAL_REVIEW_NOT_ORTHORECTIFIED",
                source_accuracy_m=None,
                association_method="MANUAL_VISUAL_CANDIDATE_ONLY",
                match_iou=intersection / union if union > 0 else 0.0,
                intersection_area_m2=intersection,
                roof_overlap_fraction=intersection / observation.roof.area
                if observation.roof.area > 0 else 0.0,
                candidate_overlap_fraction=intersection / geometry.area
                if geometry.area > 0 else 0.0,
                candidate_centroid_dx_from_roof_m=float(candidate_center.x - roof_center.x),
                candidate_centroid_dy_from_roof_m=float(candidate_center.y - roof_center.y),
                centroid_distance_m=float(roof_center.distance(candidate_center)),
                source_properties=source_properties,
                candidate_polygon_distance_m=float(observation.roof.distance(geometry)),
                source_roof_geometry=observation.roof,
            ))

    source_hashes_after = _source_component_hashes(source_path)
    evidence_hashes_after = {path: _sha256(Path(path)) for path in evidence_hashes_before}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(f"Reviewed candidate GeoPackage changed while being read: {source_path}")
    if evidence_hashes_after != evidence_hashes_before:
        raise RuntimeError("Candidate image evidence changed while being read")

    for uid, candidates in pending.items():
        by_uid[uid].footprint_candidates.extend(candidates)
    integrity_files = {**source_hashes_after, **evidence_hashes_after}
    return {
        "schema_version": "xian_r5.reviewed_ground_footprint_candidates.v1",
        "source_path": str(source_path),
        "source_layer": layer,
        "source_crs": "EPSG:32649",
        "source_feature_count": feature_count,
        "source_schema": source_schema,
        "attached_candidate_count": len(candidate_keys),
        "attached_observation_count": len(pending),
        "attached_candidates": [
            {"uid": uid, "candidate_id": candidate.candidate_id,
             "actual_date": candidate.source_capture_date,
             "geometry_semantics": "VISIBLE_GROUND_BOUNDARY_CANDIDATE",
             "positional_sigma_m": None,
             "formal_ground_footprint_confirmed": False}
            for uid, candidates in pending.items() for candidate in candidates
        ],
        "ground_footprints_confirmed": 0,
        "physical_identity_or_change_state_created": False,
        "source_component_sha256": source_hashes_before,
        "source_component_sha256_after": source_hashes_after,
        "evidence_file_sha256": evidence_hashes_before,
        "evidence_file_sha256_after": evidence_hashes_after,
        "integrity_files_sha256": integrity_files,
        "source_hash_recheck": "PASS",
        "evidence_hash_recheck": "PASS",
        "geometry_policy": "EPSG:32649_CANDIDATE_ONLY_NO_REPROJECTION_NO_REPAIR_NO_PROMOTION",
    }


def recheck_reviewed_candidate_sources(report: dict) -> dict[str, str]:
    """Verify candidate review and scene evidence stayed unchanged after load."""
    expected = report.get("integrity_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Reviewed candidate report has no integrity file inventory")
    actual = {path: _sha256(Path(path)) for path in expected}
    if actual != expected:
        raise RuntimeError("Reviewed candidate source or scene evidence changed during reconstruction")
    return actual
