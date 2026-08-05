"""Frozen, auditable development validation split for SECOND Train only."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .adapters.second import semantic_ids
from .schema import ChangeSample
from .supervision_v2 import TestReadGuard


MANIFEST_COLUMNS = ("sample_id", "source_split", "development_split", "stratum", "has_new", "has_demolition")
STRATA = ("both", "new_only", "demolition_only", "background")


@dataclass(frozen=True)
class FrozenSecondDevelopmentSplit:
    records: list[ChangeSample]
    manifest_path: Path
    manifest_sha256: str
    audit_path: Path
    audit: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def second_transition_stratum(
    gt1: np.ndarray, gt2: np.ndarray, *, building_class_id: int, ignore_index: int = 255
) -> tuple[str, bool, bool]:
    """Classify a canonical-ID SECOND pair without ever treating ignore as a class."""
    valid = (gt1 != ignore_index) & (gt2 != ignore_index)
    new = valid & (gt1 != building_class_id) & (gt2 == building_class_id)
    demolition = valid & (gt1 == building_class_id) & (gt2 != building_class_id)
    has_new, has_demolition = bool(new.any()), bool(demolition.any())
    if has_new and has_demolition:
        return "both", has_new, has_demolition
    if has_new:
        return "new_only", has_new, has_demolition
    if has_demolition:
        return "demolition_only", has_new, has_demolition
    return "background", has_new, has_demolition


def _development_config(adapter) -> dict:
    config = dict(adapter.config.get("development_validation", {}))
    if not config.get("enabled", False):
        raise ValueError("SECOND development_validation.enabled must be true for V2")
    return config


def _manifest_path(config: dict) -> Path:
    value = config.get("manifest")
    if not value:
        raise ValueError("SECOND development_validation.manifest is required")
    return Path(value)


def _validate_config(adapter, config: dict) -> tuple[int, int, int]:
    if str(config.get("source_split", "")).casefold() != "train":
        raise ValueError("SECOND development validation must use source_split=train only")
    if not bool(adapter.mapping_verified) or adapter.building_class_id is None:
        raise ValueError("SECOND development validation requires mapping_verified=true and building_class_id")
    count = int(config.get("count", 0))
    if count <= 0:
        raise ValueError("SECOND development_validation.count must be positive")
    return count, int(config.get("seed", 20260726)), int(adapter.config.get("semantic_ignore_index", adapter.config.get("ignore_index", 255)))


def _read_manifest(path: Path, candidates: dict[str, ChangeSample], *, requested_count: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if (tuple(rows[0].keys()) if rows else ()) != MANIFEST_COLUMNS:
        raise ValueError(f"SECOND development manifest has invalid columns: {path}")
    ids = [row.get("sample_id", "") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(candidates):
        raise ValueError("SECOND development manifest does not exactly match current SECOND Train IDs; refusing to redraw")
    if any(row.get("source_split", "").casefold() != "train" for row in rows):
        raise ValueError("SECOND development manifest contains a non-train source_split")
    if any(":test:" in sample_id.casefold() for sample_id in ids):
        raise ValueError("SECOND development manifest contains a test ID")
    if any(row.get("development_split") not in {"train", "validation"} for row in rows):
        raise ValueError("SECOND development manifest has invalid development_split")
    if sum(row["development_split"] == "validation" for row in rows) != requested_count:
        raise ValueError("SECOND development manifest validation count differs from configuration")
    return rows


def _allocation(groups: dict[str, list[dict]], count: int) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())
    if total <= count:
        raise ValueError(f"SECOND Train has {total} samples, insufficient to retain a development train split after {count} validation samples")
    capacity = {key: max(0, len(groups[key]) - 1) for key in STRATA}
    if sum(capacity.values()) < count:
        raise ValueError("SECOND development validation would exhaust a stratum")
    allocation = {key: 0 for key in STRATA}
    # Keep every non-singleton stratum represented, while retaining one train record.
    for key in STRATA:
        if len(groups[key]) > 1 and sum(allocation.values()) < count:
            allocation[key] = 1
    remainder = count - sum(allocation.values())
    ideal = {key: len(groups[key]) * count / total for key in STRATA}
    while remainder:
        candidates = [key for key in STRATA if allocation[key] < capacity[key]]
        if not candidates:
            raise ValueError("SECOND stratified allocation has no remaining capacity")
        key = max(candidates, key=lambda item: (ideal[item] - allocation[item], len(groups[item]), item))
        allocation[key] += 1
        remainder -= 1
    return allocation


def _rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _audit(*, config: dict, manifest_path: Path, rows: list[dict[str, str]], building: int, ignore: int, test_receipt) -> dict:
    total = Counter(row["stratum"] for row in rows)
    validation = Counter(row["stratum"] for row in rows if row["development_split"] == "validation")
    return {
        "dataset": "SECOND", "source_split": "train", "seed": int(config["seed"]),
        "requested_validation_count": int(config["count"]),
        "actual_validation_count": sum(row["development_split"] == "validation" for row in rows),
        "development_train_count": sum(row["development_split"] == "train" for row in rows),
        "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "building_class_id": building, "ignore_index": ignore, "mapping_verified": True,
        "test_images_read": test_receipt.test_images_read,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stratum_counts_total": {key: int(total[key]) for key in STRATA},
        "stratum_counts_validation": {key: int(validation[key]) for key in STRATA},
    }


def build_or_load_second_development_split(records: Iterable[ChangeSample], adapter) -> FrozenSecondDevelopmentSplit:
    """Create once or byte-for-byte reuse a SECOND Train-only development split."""
    config = _development_config(adapter)
    requested_count, seed, ignore_index = _validate_config(adapter, config)
    config["seed"] = seed
    config["count"] = requested_count
    guard = TestReadGuard("second-development-validation", allow_test=False)
    source = list(records)
    if any(record.split.casefold() == "test" for record in source):
        guard.check("test", "SECOND development split input")
    if any(record.split.casefold() != "train" for record in source):
        raise ValueError("SECOND development validation resolver accepts only discovered source split=train records")
    candidates = {record.sample_id: record for record in source}
    if len(candidates) != len(source):
        raise ValueError("SECOND Train contains duplicate sample IDs")
    manifest_path = _manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        rows = _read_manifest(manifest_path, candidates, requested_count=requested_count)
    else:
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in source:
            gt1 = semantic_ids(record.gt_t1_path, adapter.color_mapping)
            gt2 = semantic_ids(record.gt_t2_path, adapter.color_mapping)
            stratum, has_new, has_demolition = second_transition_stratum(
                gt1, gt2, building_class_id=int(adapter.building_class_id), ignore_index=ignore_index
            )
            groups[stratum].append({"sample_id": record.sample_id, "stratum": stratum, "has_new": str(has_new).lower(), "has_demolition": str(has_demolition).lower()})
        allocation = _allocation({key: groups[key] for key in STRATA}, requested_count)
        validation_ids = set()
        for key in STRATA:
            validation_ids.update(row["sample_id"] for row in sorted(groups[key], key=lambda row: _rank(seed, row["sample_id"]))[:allocation[key]])
        rows = []
        for key in STRATA:
            for row in groups[key]:
                rows.append({**row, "source_split": "train", "development_split": "validation" if row["sample_id"] in validation_ids else "train"})
        rows.sort(key=lambda row: row["sample_id"])
        with manifest_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    audit_path = manifest_path.with_name("SECOND_DEV_VALIDATION_AUDIT.json")
    if audit_path.exists():
        previous_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        previous_hash = previous_audit.get("manifest_sha256")
        current_hash = _sha256(manifest_path)
        if previous_hash and previous_hash != current_hash:
            raise ValueError("SECOND development manifest SHA256 does not match its audit; refusing to continue")
    audit = _audit(config=config, manifest_path=manifest_path, rows=rows, building=int(adapter.building_class_id), ignore=ignore_index, test_receipt=guard.receipt())
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    by_id = {row["sample_id"]: row for row in rows}
    resolved = [replace(record, split="val" if by_id[record.sample_id]["development_split"] == "validation" else "train", source_split="train") for record in source]
    return FrozenSecondDevelopmentSplit(resolved, manifest_path, audit["manifest_sha256"], audit_path, audit)
