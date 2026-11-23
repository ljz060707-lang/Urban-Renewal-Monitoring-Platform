"""Fail-closed adapter for reviewed full-coverage absence evidence."""

from __future__ import annotations

import csv
from datetime import date, datetime
import hashlib
from pathlib import Path
import re
from typing import Iterable

from .core import AbsenceEvidence, Observation, reviewed_identity_is_valid


REQUIRED_FIELDS = {
    "confirmed_identity_key", "epoch_label", "actual_date", "image_ref",
    "full_footprint_coverage", "reviewed", "reviewer_id", "reviewed_at",
    "evidence_path", "evidence_sha256",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_date(value: str, field: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a canonical YYYY-MM-DD date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must be a canonical YYYY-MM-DD date")
    return value


def _flag(value: str, field: str) -> bool:
    normalized = value.strip() if isinstance(value, str) else ""
    if normalized not in {"0", "1"}:
        raise ValueError(f"{field} must be 0 or 1")
    return normalized == "1"


def load_reviewed_absences(
    observations: Iterable[Observation], source_path: Path,
) -> tuple[list[AbsenceEvidence], dict]:
    """Load explicit absence reviews; absence is never inferred from no match."""
    records = list(observations)
    if not records:
        raise ValueError("At least one observation is required")
    source_path = Path(source_path).resolve()
    if source_path.suffix.lower() != ".csv":
        raise ValueError("Reviewed absence evidence must be supplied as a CSV")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    confirmed_keys = {
        obs.confirmed_identity_key for obs in records
        if reviewed_identity_is_valid(obs)
    }
    dates_by_epoch: dict[str, set[str]] = {}
    epochs_by_date: dict[str, set[str]] = {}
    for obs in records:
        if obs.actual_date:
            dates_by_epoch.setdefault(obs.epoch_label, set()).add(obs.actual_date)
            epochs_by_date.setdefault(obs.actual_date, set()).add(obs.epoch_label)
    epochs = {obs.epoch_label for obs in records}

    source_hash_before = _sha256(source_path)
    evidence_hashes_before: dict[str, str] = {}
    loaded: list[AbsenceEvidence] = []
    seen: set[tuple[str, str]] = set()
    with source_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("Reviewed absence CSV must include a header")
        missing = REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Reviewed absence CSV is missing fields: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            identity_key = (row.get("confirmed_identity_key") or "").strip()
            epoch_label = (row.get("epoch_label") or "").strip()
            if not identity_key or identity_key not in confirmed_keys:
                raise ValueError(f"Row {row_number} references an identity without reviewed identity evidence")
            if not epoch_label or epoch_label not in epochs:
                raise ValueError(f"Row {row_number} references an unknown epoch label")
            actual_date = _parse_date(row.get("actual_date"), f"actual_date at row {row_number}")
            pair = (identity_key, actual_date)
            if pair in seen:
                raise ValueError(
                    f"Duplicate reviewed absence for identity/actual-date: {identity_key}/{actual_date}")
            if actual_date not in dates_by_epoch.get(epoch_label, set()):
                if actual_date not in epochs_by_date:
                    raise ValueError(f"Absence date is not an observed actual date for {identity_key}/{epoch_label}")
                raise ValueError(
                    f"Absence actual date is not observed under the supplied epoch label "
                    f"for {identity_key}/{epoch_label}")

            full_coverage = _flag(row.get("full_footprint_coverage"), "full_footprint_coverage")
            reviewed = _flag(row.get("reviewed"), "reviewed")
            if not full_coverage or not reviewed:
                raise ValueError(f"Absence row {row_number} requires full-footprint coverage and human review")
            image_ref = (row.get("image_ref") or "").strip()
            reviewer_id = (row.get("reviewer_id") or "").strip()
            reviewed_at = (row.get("reviewed_at") or "").strip()
            if not image_ref or not reviewer_id or not reviewed_at:
                raise ValueError(f"Absence row {row_number} is missing image/reviewer/timestamp provenance")
            try:
                review_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"reviewed_at at row {row_number} must be an ISO timestamp with timezone") from error
            if review_time.tzinfo is None or review_time.utcoffset() is None:
                raise ValueError(f"reviewed_at at row {row_number} must include a timezone")

            evidence_path_raw = (row.get("evidence_path") or "").strip()
            evidence_digest = (row.get("evidence_sha256") or "").strip()
            if not evidence_path_raw:
                raise ValueError(f"evidence_path is required at row {row_number}")
            if not _SHA256_RE.fullmatch(evidence_digest):
                raise ValueError(f"evidence_sha256 at row {row_number} must be a 64-character SHA-256")
            evidence_path = Path(evidence_path_raw)
            if not evidence_path.is_absolute():
                evidence_path = source_path.parent / evidence_path
            evidence_path = evidence_path.resolve()
            if evidence_path == source_path:
                raise ValueError(f"Reviewed absence CSV cannot be its own supporting evidence at row {row_number}")
            if not evidence_path.is_file():
                raise FileNotFoundError(f"Absence evidence file does not exist: {evidence_path}")
            evidence_key = str(evidence_path)
            if evidence_key not in evidence_hashes_before:
                evidence_hashes_before[evidence_key] = _sha256(evidence_path)
            if evidence_hashes_before[evidence_key].casefold() != evidence_digest.casefold():
                raise ValueError(f"Absence evidence SHA-256 mismatch at row {row_number}")

            evidence = AbsenceEvidence(
                confirmed_identity_key=identity_key,
                epoch_label=epoch_label,
                actual_date=actual_date,
                image_ref=image_ref,
                full_footprint_coverage=True,
                reviewed=True,
                reviewer_id=reviewer_id,
                reviewed_at=reviewed_at,
                evidence_path=evidence_key,
                evidence_sha256=evidence_digest.lower(),
                evidence_integrity_verified=True,
            )
            if not evidence.accepted:
                raise ValueError(f"Absence row {row_number} failed the formal evidence gate")
            loaded.append(evidence)
            seen.add(pair)

    source_hash_after = _sha256(source_path)
    evidence_hashes_after = {path: _sha256(Path(path)) for path in evidence_hashes_before}
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"Reviewed absence CSV changed while being read: {source_path}")
    if evidence_hashes_after != evidence_hashes_before:
        raise RuntimeError("Reviewed absence evidence changed while being read")
    integrity_files = {str(source_path): source_hash_after, **evidence_hashes_after}
    report = {
        "schema_version": "xian_r5.reviewed_absences.v1",
        "source_path": str(source_path),
        "source_sha256": source_hash_before,
        "source_sha256_after": source_hash_after,
        "evidence_file_sha256": evidence_hashes_before,
        "evidence_file_sha256_after": evidence_hashes_after,
        "integrity_files_sha256": integrity_files,
        "reviewed_absence_count": len(loaded),
        "identity_epoch_pairs": [
            {"confirmed_identity_key": item.confirmed_identity_key,
             "epoch_label": item.epoch_label, "actual_date": item.actual_date}
            for item in loaded
        ],
        "identity_actual_date_pairs": [
            {"confirmed_identity_key": item.confirmed_identity_key,
             "actual_date": item.actual_date, "source_epoch_label": item.epoch_label}
            for item in loaded
        ],
        "source_hash_recheck": "PASS",
        "evidence_hash_recheck": "PASS",
        "acceptance_semantics": "EXPLICIT_REVIEWED_ABSENCE_WITH_FULL_PRIOR_FOOTPRINT_COVERAGE_AT_EXACT_ACTUAL_DATE;NO_MATCH_IS_NOT_ABSENCE",
        "hash_semantics": "HASHES_VERIFY_LINEAGE_NOT_SEMANTIC_TRUTH",
    }
    return loaded, report
