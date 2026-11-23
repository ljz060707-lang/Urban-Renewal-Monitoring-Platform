"""Fail-closed input adapter for independently reviewed ground footprints."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import fiona
from pyproj import CRS
from shapely.geometry import shape

from .core import FootprintEvidence, Observation


LAYER_NAME = "reviewed_ground_footprints"
REQUIRED_FIELDS = {
    "uid", "actual_date", "source_ref", "method",
    "independent_ground_evidence", "reviewed", "positional_sigma_m",
    "roof_to_ground_dx_m", "roof_to_ground_dy_m", "reviewer_id",
    "reviewed_at", "evidence_path", "evidence_sha256",
    "physical_identity_key", "identity_reviewed",
}
METHODS = {"SURVEY", "VISIBLE_GROUND_BOUNDARY", "VALIDATED_3D_PROJECTION"}
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


def _as_flag(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field_name} must be stored as 0 or 1")


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


def attach_reviewed_ground_footprints(
    observations: Iterable[Observation], source_path: Path, *,
    layer: str = LAYER_NAME,
) -> dict:
    """Attach only exact-UID/date, hash-verified reviewed ground geometries.

    No reprojection, geometry repair, spatial matching or roof translation is
    performed. Any malformed or ambiguous record aborts the bounded run.
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
        raise ValueError("Reviewed ground footprints must be supplied as a GeoPackage")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_hashes_before = _source_component_hashes(source_path)
    evidence_hashes_before: dict[str, str] = {}
    source_feature_count = 0
    source_schema = None
    attached_uids: list[str] = []
    pending_evidence: dict[str, FootprintEvidence] = {}
    pending_identities: dict[str, tuple[str | None, bool]] = {}

    with fiona.open(source_path, layer=layer) as source:
        source_crs_input = source.crs_wkt or source.crs
        if not source_crs_input:
            raise ValueError("Reviewed ground footprint layer has no CRS")
        source_crs = CRS.from_user_input(source_crs_input)
        if source_crs.to_epsg() != 32649:
            raise ValueError("Reviewed ground footprint layer must use EPSG:32649")
        source_schema = {
            "geometry": source.schema.get("geometry"),
            "properties": dict(source.schema.get("properties", {})),
        }
        missing_fields = REQUIRED_FIELDS - set(source_schema["properties"])
        if missing_fields:
            raise ValueError(f"Reviewed ground footprint layer is missing fields: {sorted(missing_fields)}")
        if source_schema["geometry"] not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Reviewed ground footprint geometry type must be Polygon or MultiPolygon")

        source_feature_count = len(source)
        for feature in source:
            properties = dict(feature["properties"])
            uid = properties.get("uid")
            if not isinstance(uid, str) or not uid.strip():
                raise ValueError("Every reviewed footprint must have a non-empty observation UID")
            if uid not in by_uid:
                raise ValueError(f"Reviewed footprint references unknown observation UID: {uid}")
            if uid in attached_uids:
                raise ValueError(f"More than one reviewed footprint record for observation UID: {uid}")

            actual_date = _parse_date(properties.get("actual_date"), "actual_date")
            observation = by_uid[uid]
            if observation.actual_date is None:
                raise ValueError(f"Observation date is unresolved for reviewed footprint UID: {uid}")
            if actual_date != observation.actual_date:
                raise ValueError(f"Reviewed footprint date does not match observation {uid}: {actual_date} != {observation.actual_date}")

            identity_key_raw = properties.get("physical_identity_key")
            identity_key = (identity_key_raw.strip()
                            if isinstance(identity_key_raw, str) and identity_key_raw.strip()
                            else None)
            if identity_key_raw is not None and not isinstance(identity_key_raw, str):
                raise ValueError(f"physical_identity_key must be text or null for {uid}")
            identity_reviewed = _as_flag(properties.get("identity_reviewed"), "identity_reviewed")
            if identity_reviewed != bool(identity_key):
                raise ValueError(f"A physical identity key and identity_reviewed=1 must be supplied together for {uid}")

            method = properties.get("method")
            if method not in METHODS:
                raise ValueError(f"Unsupported reviewed footprint method for {uid}: {method!r}")
            independent = _as_flag(properties.get("independent_ground_evidence"), "independent_ground_evidence")
            reviewed = _as_flag(properties.get("reviewed"), "reviewed")
            if not independent or not reviewed:
                raise ValueError(f"Footprint {uid} must have independent ground evidence and completed human review")

            source_ref = properties.get("source_ref")
            reviewer_id = properties.get("reviewer_id")
            if not isinstance(source_ref, str) or not source_ref.strip():
                raise ValueError(f"source_ref is required for reviewed footprint {uid}")
            if not isinstance(reviewer_id, str) or not reviewer_id.strip():
                raise ValueError(f"reviewer_id is required for reviewed footprint {uid}")
            reviewed_at = _parse_reviewed_at(properties.get("reviewed_at"))

            sigma_raw = properties.get("positional_sigma_m")
            if isinstance(sigma_raw, bool) or not isinstance(sigma_raw, (int, float)):
                raise ValueError(f"positional_sigma_m must be numeric for reviewed footprint {uid}")
            sigma = float(sigma_raw)
            if not math.isfinite(sigma) or sigma < 0 or sigma > 2.0:
                raise ValueError(f"positional_sigma_m must be finite and between 0 and 2 m for {uid}")

            evidence_path_raw = properties.get("evidence_path")
            evidence_digest = properties.get("evidence_sha256")
            if not isinstance(evidence_path_raw, str) or not evidence_path_raw.strip():
                raise ValueError(f"evidence_path is required for reviewed footprint {uid}")
            if not isinstance(evidence_digest, str) or not _SHA256_RE.fullmatch(evidence_digest):
                raise ValueError(f"evidence_sha256 must be a 64-character SHA-256 for {uid}")
            evidence_path = Path(evidence_path_raw)
            if not evidence_path.is_absolute():
                evidence_path = source_path.parent / evidence_path
            evidence_path = evidence_path.resolve()
            if str(evidence_path) in source_hashes_before:
                raise ValueError(f"Review GeoPackage components cannot be independent evidence for {uid}")
            if not evidence_path.is_file():
                raise FileNotFoundError(f"Reviewed footprint evidence file does not exist: {evidence_path}")
            evidence_key = str(evidence_path)
            if evidence_key not in evidence_hashes_before:
                evidence_hashes_before[evidence_key] = _sha256(evidence_path)
            actual_digest = evidence_hashes_before[evidence_key]
            if actual_digest.casefold() != evidence_digest.casefold():
                raise ValueError(f"Evidence SHA-256 mismatch for reviewed footprint {uid}")

            dx = properties.get("roof_to_ground_dx_m")
            dy = properties.get("roof_to_ground_dy_m")
            offsets = []
            for name, value in (("roof_to_ground_dx_m", dx), ("roof_to_ground_dy_m", dy)):
                if value is None:
                    offsets.append(None)
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite numeric or null for {uid}")
                else:
                    offsets.append(float(value))

            geometry_value = feature.get("geometry")
            if not geometry_value:
                raise ValueError(f"Reviewed footprint geometry is missing for {uid}")
            geometry = shape(geometry_value)
            if (geometry.geom_type not in {"Polygon", "MultiPolygon"}
                    or geometry.is_empty or not geometry.is_valid or geometry.area <= 0):
                raise ValueError(f"Reviewed footprint geometry is invalid for {uid}; no automatic repair is allowed")

            footprint_evidence = FootprintEvidence(
                geometry=geometry,
                source_ref=source_ref.strip(),
                method=method,
                independent_ground_evidence=True,
                reviewed=True,
                positional_sigma_m=sigma,
                roof_to_ground_dx_m=offsets[0],
                roof_to_ground_dy_m=offsets[1],
                crs="EPSG:32649",
                reviewer_id=reviewer_id.strip(),
                reviewed_at=reviewed_at,
                evidence_path=evidence_key,
                evidence_sha256=evidence_digest.lower(),
                evidence_integrity_verified=True,
            )
            if not footprint_evidence.accepted:
                raise ValueError(f"Reviewed footprint failed the formal acceptance gate for {uid}")
            pending_evidence[uid] = footprint_evidence
            pending_identities[uid] = (identity_key, identity_reviewed)
            attached_uids.append(uid)

    source_hashes_after = _source_component_hashes(source_path)
    evidence_hashes_after = {path: _sha256(Path(path)) for path in evidence_hashes_before}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(f"Reviewed footprint source changed while being read: {source_path}")
    if evidence_hashes_after != evidence_hashes_before:
        raise RuntimeError("Reviewed footprint evidence changed while being read")

    for uid, evidence in pending_evidence.items():
        observation = by_uid[uid]
        observation.footprint_evidence = evidence
        observation.confirmed_identity_key, observation.identity_reviewed = pending_identities[uid]
        if observation.identity_reviewed:
            observation.identity_evidence_type = "REVIEWED_GROUND_FOOTPRINT_RECORD"
            observation.identity_evidence_ref = evidence.source_ref
            observation.identity_evidence_path = evidence.evidence_path
            observation.identity_evidence_sha256 = evidence.evidence_sha256
            observation.identity_reviewer_id = evidence.reviewer_id
            observation.identity_reviewed_at = evidence.reviewed_at
            observation.identity_evidence_integrity_verified = True

    integrity_files = {**source_hashes_after, **evidence_hashes_after}
    return {
        "schema_version": "xian_r5.reviewed_ground_footprints.v1",
        "source_path": str(source_path),
        "source_layer": layer,
        "source_crs": "EPSG:32649",
        "source_feature_count": source_feature_count,
        "source_schema": source_schema,
        "attached_observation_count": len(attached_uids),
        "attached_uids": attached_uids,
        "identity_reviewed_observation_count": sum(reviewed for _, reviewed in pending_identities.values()),
        "source_component_sha256": source_hashes_before,
        "source_component_sha256_after": source_hashes_after,
        "evidence_file_sha256": evidence_hashes_before,
        "evidence_file_sha256_after": evidence_hashes_after,
        "integrity_files_sha256": integrity_files,
        "source_hash_recheck": "PASS",
        "evidence_hash_recheck": "PASS",
        "ground_truth_semantics": "HUMAN_ASSERTED_REVIEWED_GROUND_GEOMETRY; HASHES_VERIFY_LINEAGE_NOT_SEMANTIC_TRUTH",
        "geometry_policy": "EPSG:32649_POLYGON_ONLY_NO_REPROJECTION_NO_REPAIR_NO_AUTOMATIC_TRANSLATION",
    }


def recheck_review_sources(report: dict) -> dict[str, str]:
    """Verify review geometry and evidence inputs stayed unchanged after load."""
    expected = report.get("integrity_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Reviewed footprint report has no integrity file inventory")
    actual = {path: _sha256(Path(path)) for path in expected}
    if actual != expected:
        raise RuntimeError("Reviewed footprint or evidence source changed during reconstruction")
    return actual
