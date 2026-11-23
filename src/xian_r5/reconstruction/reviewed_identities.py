"""Fail-closed adapter for identity review independent of ground geometry."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
from pathlib import Path
import re
from typing import Iterable

import fiona

from .core import Observation


LAYER_NAME = "reviewed_physical_identities"
REQUIRED_FIELDS = {
    "uid", "actual_date", "physical_identity_key", "reviewed",
    "evidence_type", "evidence_ref", "evidence_path", "evidence_sha256",
    "reviewer_id", "reviewed_at",
}
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
    return {str(item.resolve()): _sha256(item)
            for item in sorted(components)}


def _reviewed_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError("reviewed must be stored as 0 or 1")


def _canonical_date(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("actual_date must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("actual_date must be an ISO date") from error
    if parsed.isoformat() != value:
        raise ValueError("actual_date must use canonical YYYY-MM-DD form")
    return value


def _reviewed_at(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reviewed_at is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("reviewed_at must be a timezone-aware ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewed_at must be a timezone-aware ISO timestamp")
    return value


def attach_reviewed_physical_identities(
    observations: Iterable[Observation], source_path: Path, *,
    layer: str = LAYER_NAME,
) -> dict:
    """Attach reviewed physical IDs without requiring a ground-footprint row.

    The adapter does not edit geometry. Every record must bind an exact
    observation UID and actual date to independently reviewable evidence.
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
        raise ValueError("Reviewed physical identities must be supplied as a GeoPackage")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_hashes_before = _source_component_hashes(source_path)
    evidence_hashes_before: dict[str, str] = {}
    pending: dict[str, tuple[str, str, str, str, str, str, str]] = {}
    with fiona.open(source_path, layer=layer) as source:
        properties = dict(source.schema.get("properties", {}))
        missing = REQUIRED_FIELDS - set(properties)
        if missing:
            raise ValueError(f"Reviewed identity layer is missing fields: {sorted(missing)}")
        if source.schema.get("geometry") not in {None, "None"}:
            raise ValueError("Reviewed physical identity layer must be non-spatial")
        for feature in source:
            row = dict(feature["properties"])
            uid = row.get("uid")
            if not isinstance(uid, str) or uid not in by_uid:
                raise ValueError(f"Reviewed identity references unknown observation UID: {uid!r}")
            if uid in pending:
                raise ValueError(f"Duplicate reviewed identity record for observation UID: {uid}")
            observation = by_uid[uid]
            actual_date = _canonical_date(row.get("actual_date"))
            if observation.actual_date is None or actual_date != observation.actual_date:
                raise ValueError(
                    f"Reviewed identity date does not match observation {uid}: "
                    f"{actual_date} != {observation.actual_date}")
            if not _reviewed_flag(row.get("reviewed")):
                raise ValueError(f"Identity review is not accepted for {uid}")
            identity_key = row.get("physical_identity_key")
            if not isinstance(identity_key, str) or not identity_key.strip():
                raise ValueError(f"physical_identity_key is required for {uid}")
            evidence_type = row.get("evidence_type")
            evidence_ref = row.get("evidence_ref")
            reviewer_id = row.get("reviewer_id")
            if not all(isinstance(value, str) and value.strip()
                       for value in (evidence_type, evidence_ref, reviewer_id)):
                raise ValueError(f"Identity review provenance is incomplete for {uid}")
            reviewed_at = _reviewed_at(row.get("reviewed_at"))
            evidence_path_text = row.get("evidence_path")
            if not isinstance(evidence_path_text, str) or not evidence_path_text.strip():
                raise ValueError(f"evidence_path is required for {uid}")
            evidence_path = Path(evidence_path_text)
            if not evidence_path.is_absolute():
                evidence_path = source_path.parent / evidence_path
            evidence_path = evidence_path.resolve()
            if not evidence_path.is_file():
                raise FileNotFoundError(evidence_path)
            digest = row.get("evidence_sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"evidence_sha256 must be a 64-character SHA-256 for {uid}")
            evidence_hash = _sha256(evidence_path)
            if evidence_hash != digest.lower():
                raise ValueError(f"Identity evidence SHA-256 mismatch for {uid}")
            evidence_hashes_before[str(evidence_path)] = evidence_hash
            existing_key = observation.confirmed_identity_key
            if observation.identity_reviewed and existing_key != identity_key.strip():
                raise ValueError(f"Conflicting reviewed physical identities for {uid}")
            pending[uid] = (
                identity_key.strip(), evidence_type.strip(), evidence_ref.strip(),
                str(evidence_path), evidence_hash, reviewer_id.strip(), reviewed_at,
            )

    source_hashes_after = _source_component_hashes(source_path)
    evidence_hashes_after = {path: _sha256(Path(path))
                             for path in evidence_hashes_before}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(f"Reviewed identity source changed while being read: {source_path}")
    if evidence_hashes_after != evidence_hashes_before:
        raise RuntimeError("Reviewed identity evidence changed while being read")

    for uid, values in pending.items():
        observation = by_uid[uid]
        (observation.confirmed_identity_key, observation.identity_evidence_type,
         observation.identity_evidence_ref, observation.identity_evidence_path,
         observation.identity_evidence_sha256, observation.identity_reviewer_id,
         observation.identity_reviewed_at) = values
        observation.identity_reviewed = True
        observation.identity_evidence_integrity_verified = True

    integrity_files = {**source_hashes_after, **evidence_hashes_after}
    return {
        "schema_version": "xian_r5.reviewed_physical_identities.v1",
        "source_path": str(source_path), "source_layer": layer,
        "source_feature_count": len(pending), "attached_uids": sorted(pending),
        "reviewed_identity_observation_count": len(pending),
        "physical_identity_key_count": len({values[0] for values in pending.values()}),
        "source_component_sha256": source_hashes_before,
        "source_component_sha256_after": source_hashes_after,
        "evidence_file_sha256": evidence_hashes_before,
        "evidence_file_sha256_after": evidence_hashes_after,
        "integrity_files_sha256": integrity_files,
        "source_hash_recheck": "PASS", "evidence_hash_recheck": "PASS",
        "ground_geometry_required": False,
        "geometry_modified": False,
    }


def recheck_reviewed_identity_sources(report: dict) -> dict[str, str]:
    """Verify identity review and evidence sources remain byte-identical."""
    expected = report.get("integrity_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Reviewed identity report has no integrity file inventory")
    actual = {path: _sha256(Path(path)) for path in expected}
    if actual != expected:
        changed = [path for path in expected if actual.get(path) != expected[path]]
        raise RuntimeError(f"Reviewed identity source changed during reconstruction: {changed}")
    return actual
