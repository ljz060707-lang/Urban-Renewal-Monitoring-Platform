from __future__ import annotations

from datetime import datetime, timezone


ALLOWED_ENCODINGS = {"wgs84", "cgcs2000", "gcj02", "bd09", "projected", "unknown"}
TRAINING_SAFE_ENCODINGS = {"wgs84", "cgcs2000", "projected"}


def validate_coordinate_encoding(value: str | None) -> str:
    """Normalize and validate a declared coordinate encoding."""
    encoding = str(value or "unknown").strip().lower()
    if encoding not in ALLOWED_ENCODINGS:
        raise ValueError(f"Unsupported coordinate_encoding={value!r}; allowed={sorted(ALLOWED_ENCODINGS)}")
    return encoding


def require_training_safe_encoding(value: str | None) -> str:
    """Reject GCJ-02, BD-09 and unknown vectors before training or metric use."""
    encoding = validate_coordinate_encoding(value)
    if encoding not in TRAINING_SAFE_ENCODINGS:
        raise ValueError(
            f"coordinate_encoding={encoding!r} cannot enter training before a verified conversion"
        )
    return encoding


def conversion_provenance(
    source_encoding: str,
    target_encoding: str,
    *,
    method: str,
    version: str,
) -> dict[str, str]:
    """Build an auditable record for an externally verified conversion."""
    source = validate_coordinate_encoding(source_encoding)
    target = validate_coordinate_encoding(target_encoding)
    if source == target:
        raise ValueError("source_encoding and target_encoding are identical")
    if not method.strip() or not version.strip():
        raise ValueError("conversion method and version are required")
    return {
        "source_encoding": source,
        "target_encoding": target,
        "conversion_method": method.strip(),
        "conversion_version": version.strip(),
        "conversion_timestamp": datetime.now(timezone.utc).isoformat(),
    }
