from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .reviewed_regions import write_reviewed_region_index


REVIEW_FILES = (
    "review_state.json",
    "review_events.jsonl",
    "review_decisions.csv",
    "review_decisions.jsonl",
    "REVIEW_INVENTORY.json",
    "REVIEW_INVENTORY.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path, fallback):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def migrate_existing_review(project_root: Path, review_dir: Path, output_dir: Path | None = None) -> dict:
    """Import reviews into an immutable provenance snapshot without rewriting source review files."""
    project_root = Path(project_root).resolve()
    review_dir = Path(review_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir).resolve() if output_dir else (
        project_root / "outputs" / "urrs" / "review" / f"review_migration_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    before = {name: _sha256(review_dir / name) for name in REVIEW_FILES if (review_dir / name).exists()}
    inventory = _load_json(review_dir / "REVIEW_INVENTORY.json", {"records": []})
    state = _load_json(review_dir / "review_state.json", {"decisions": {}})
    decisions = state.get("decisions", {})
    records = inventory.get("records", [])
    provenance: list[dict] = []
    transition_rows: list[dict] = []
    queue_rows = {key: [] for key in ("hard_negative", "missed_object", "fragmentation", "semantic_confusion")}
    for record in records:
        sample_id = record["sample_id"]
        group = record.get("review_group", "unknown")
        decision = decisions.get(sample_id)
        is_transition = group == "transition"
        if is_transition:
            review_status = "deferred_invalid_preview"
            purpose = "transition_label"
            label_source = "not_labeled"
            ground_truth_status = "not_ground_truth"
        elif decision:
            review_status = "completed_as_recorded"
            purpose = "quality_assurance"
            label_source = "human_reviewed_quality_assessment"
            ground_truth_status = "not_pixel_ground_truth"
        else:
            review_status = "not_recorded"
            purpose = "quality_assurance"
            label_source = "not_labeled"
            ground_truth_status = "not_ground_truth"
        row = {
            "original_review_run": str(review_dir),
            "original_sample_id": sample_id,
            "review_group": group,
            "original_reviewer": decision.get("reviewer", "") if decision else "",
            "original_verdict": decision.get("verdict", "") if decision else "",
            "original_reviewed_at": decision.get("reviewed_at", "") if decision else "",
            "original_updated_at": decision.get("updated_at", "") if decision else "",
            "original_notes": decision.get("notes", "") if decision else "",
            "migration_status": "preserved" if decision else "inventory_preserved_no_decision",
            "review_status": review_status,
            "review_purpose": purpose,
            "label_source": label_source,
            "ground_truth_status": ground_truth_status,
            "requires_regenerated_preview": is_transition,
            "transition_review_failure": False if is_transition else "",
            "source_inventory_hash": record.get("inventory_hash", inventory.get("inventory_hash", "")),
        }
        provenance.append(row)
        if is_transition:
            transition_rows.append(row)
        if decision and not is_transition:
            tags = set(decision.get("issue_tags", []))
            severities = decision.get("severity_fields", {})
            common = {
                "sample_id": sample_id,
                "review_group": group,
                "verdict": decision.get("verdict", ""),
                "reviewer": decision.get("reviewer", ""),
                "reviewed_at": decision.get("reviewed_at", ""),
                "issue_tags": json.dumps(sorted(tags), ensure_ascii=False),
                "notes": decision.get("notes", ""),
                "queue_status": "pending_annotation_or_analysis",
                "auto_training_eligible": False,
            }
            if tags & {"road_confusion", "water_confusion", "shadow_confusion", "vacant_confusion"}:
                queue_rows["hard_negative"].append(common)
            if tags & {"building_miss", "small_building_miss"} or int(severities.get("false_negative_severity", 0)) > 0:
                queue_rows["missed_object"].append(common)
            if tags & {"building_fragmentation", "building_merge"} or int(severities.get("fragmentation_severity", 0)) > 0 or int(severities.get("merging_severity", 0)) > 0:
                queue_rows["fragmentation"].append(common)
            if "surface_state_error" in tags or int(severities.get("semantic_plausibility", 0)) > 0:
                queue_rows["semantic_confusion"].append(common)
    fields = list(provenance[0]) if provenance else ["original_sample_id"]
    _write_csv(output_dir / "REVIEW_DECISION_PROVENANCE.csv", provenance, fields)
    deferred = {
        "transition_review_status": "deferred_due_to_invalid_visualization",
        "transition_model_status": "not_human_verified",
        "transition_review_failure": False,
        "preserve_existing_record": True,
        "requires_regenerated_preview": True,
        "transition_inventory_count": len(transition_rows),
        "transition_decision_count": sum(bool(decisions.get(row["original_sample_id"])) for row in transition_rows),
        "records": transition_rows,
    }
    (output_dir / "TRANSITION_REVIEW_DEFERRED.json").write_text(
        json.dumps(deferred, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    feedback_dir = project_root / "outputs" / "urrs" / "construction" / "human_review_feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    queue_fields = [
        "sample_id", "review_group", "verdict", "reviewer", "reviewed_at", "issue_tags",
        "notes", "queue_status", "auto_training_eligible",
    ]
    names = {
        "hard_negative": "HARD_NEGATIVE_QUEUE.csv",
        "missed_object": "MISSED_OBJECT_QUEUE.csv",
        "fragmentation": "FRAGMENTATION_QUEUE.csv",
        "semantic_confusion": "SEMANTIC_CONFUSION_QUEUE.csv",
    }
    for key, name in names.items():
        _write_csv(feedback_dir / name, queue_rows[key], queue_fields)
    region_index = write_reviewed_region_index(records, decisions, feedback_dir / "REVIEWED_REGION_INDEX.gpkg")
    after = {name: _sha256(review_dir / name) for name in before}
    if before != after:
        raise RuntimeError("Source review files changed during read-only migration")
    counts = Counter(record.get("review_group", "unknown") for record in records)
    decision_counts = Counter(
        next((record.get("review_group", "unknown") for record in records if record["sample_id"] == sample_id), "unknown")
        for sample_id in decisions
    )
    receipt = {
        "status": "success",
        "source_review_dir": str(review_dir),
        "output_dir": str(output_dir),
        "source_file_sha256_before": before,
        "source_file_sha256_after": after,
        "source_files_unchanged": before == after,
        "inventory_count": len(records),
        "inventory_counts_by_group": dict(counts),
        "recorded_decision_count": len(decisions),
        "recorded_decisions_by_group": dict(decision_counts),
        "preserved_decision_count": len(decisions),
        "transition_deferred_count": len(transition_rows),
        "transition_review_failure": False,
        "non_transition_decisions_are_pixel_ground_truth": False,
        "feedback_queues": {key: len(rows) for key, rows in queue_rows.items()},
        "reviewed_region_index": region_index,
        "review_state": {
            "geometry_review_complete": decision_counts.get("geometry", 0) == counts.get("geometry", 0),
            "surface_state_review_complete": decision_counts.get("surface_state", 0) == counts.get("surface_state", 0),
            "transition_preview_valid": False,
            "transition_review_complete": False,
            "human_temporal_labels_available": False,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "EXISTING_REVIEW_IMPORT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt
