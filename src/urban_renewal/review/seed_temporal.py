from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SEED_STATES = (
    "stable_building",
    "possible_demolition",
    "cleared_or_exposed_ground",
    "possible_active_construction",
    "possible_structure_rising",
    "possible_completed_building",
    "possible_repair_or_refurbishment",
    "ordinary_bare_land",
    "waste_or_material_storage",
    "uncertain",
)
SEED_TRANSITIONS = (
    "NO_VISIBLE_CHANGE",
    "DEMOLITION_OR_CLEARING",
    "CONSTRUCTION_START",
    "CONTINUING_CONSTRUCTION",
    "STRUCTURE_APPEARANCE",
    "COMPLETION",
    "REPAIR_OR_REFURBISHMENT_CANDIDATE",
    "LIKELY_FALSE_CHANGE",
    "UNCERTAIN",
)
SEED_DECISIONS = ("ACCEPT", "CORRECT", "UNCERTAIN", "REJECT")
SEED_COVERAGE_TARGETS = (
    "urban_built_area",
    "industrial_area",
    "ordinary_bare_land",
    "dense_building_area",
    "suspected_demolition",
    "suspected_new_build",
    "no_visible_change",
    "uncertain_sample",
)
SEED_CSV_FIELDS = (
    "sample_id",
    "track_seed_id",
    "before_date",
    "after_date",
    "before_surface_state",
    "after_surface_state",
    "transition_type",
    "existing_building_before",
    "existing_building_after",
    "construction_activity_visible",
    "demolition_visible",
    "clearing_visible",
    "new_structure_visible",
    "repair_or_refurbishment_visible",
    "confidence",
    "notes",
    "decision",
    "reviewer",
    "human_confirmed",
    "label_source",
    "formal_ground_truth",
    "model_independent",
    "updated_at",
)


def _latest_transition_inventory(review_dir: Path) -> tuple[Path, dict]:
    candidates = sorted(Path(review_dir).parent.glob("review_migration_*/transition_previews/TRANSITION_REVIEW_INVENTORY.json"))
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("records"):
            return path, payload
    raise FileNotFoundError("no regenerated transition preview inventory is available")


def _seed_root(review_dir: Path) -> Path:
    return Path(review_dir).resolve().parent.parent / "construction"


def _latest_seed_dir(review_dir: Path) -> Path | None:
    candidates = sorted(_seed_root(review_dir).glob("human_seed_labels_*/SEED_LABEL_INVENTORY.json"))
    return candidates[-1].parent if candidates else None


def create_seed_temporal_inventory(review_dir: Path, output_dir: Path) -> dict:
    review_dir = Path(review_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path, source = _latest_transition_inventory(review_dir)
    source_rows = sorted(source.get("records", []), key=lambda row: row.get("sample_id", ""))
    records = []
    for index, row in enumerate(source_rows):
        records.append(
            {
                "sample_id": f"seed_temporal_{index:03d}",
                "track_seed_id": f"seed_{row.get('track_id', f'track_{index:03d}')}",
                "before_date": row.get("before_date"),
                "after_date": row.get("after_date"),
                "before_surface_state": None,
                "after_surface_state": None,
                "transition_type": None,
                "existing_building_before": None,
                "existing_building_after": None,
                "construction_activity_visible": None,
                "demolition_visible": None,
                "clearing_visible": None,
                "new_structure_visible": None,
                "repair_or_refurbishment_visible": None,
                "confidence": None,
                "notes": "",
                "decision": None,
                "reviewer": None,
                "human_confirmed": False,
                "label_source": "human_seed_label",
                "formal_ground_truth": False,
                "model_independent": True,
                "seed_status": "unreviewed",
                "supersedes_sample_ids": [row.get("sample_id")],
                "legacy_sample_id": row.get("sample_id"),
                "transition": row.get("transition"),
                "full20_supported": bool(row.get("full20_supported", False)),
                "preview_status": row.get("preview_quality", {}).get("preview_status"),
                "pair_preview_path": row.get("pair_preview_path"),
                "raw_before_path": row.get("raw_before_path"),
                "raw_after_path": row.get("raw_after_path"),
                "source_metadata": row.get("source_metadata", {}),
                "evidence": row.get("evidence", {}),
                "evidence_missing": row.get("evidence_missing", []),
                "coverage_targets": list(SEED_COVERAGE_TARGETS),
            }
        )
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    inventory_hash = hashlib.sha256(canonical).hexdigest()
    inventory = {
        "status": "success",
        "scope": "human_seed_labeling_only",
        "seed_inventory_version": "seed_temporal_v1",
        "source_transition_inventory": str(source_path),
        "source_transition_inventory_hash": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "records": records,
        "seed_count": len(records),
        "inventory_hash": inventory_hash,
        "coverage_transition_requirements": [
            "2018_to_2020",
            "2020_to_2022",
            "2022_to_2023",
            "2018_to_2023",
            "2020_to_2023",
        ],
        "coverage_scene_requirements": list(SEED_COVERAGE_TARGETS),
        "coverage_status": "PENDING HUMAN SEED LABELS",
        "label_source": "human_seed_label",
        "formal_ground_truth": False,
        "model_independent": True,
        "model_prediction_display": False,
        "current_urrs_release_unchanged": True,
        "test_pixel_reads": {"UCS": 0, "CWLD": 0, "WHU": 0, "SECOND": 0},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "SEED_LABEL_INVENTORY.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "SEED_LABEL_DECISIONS.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        csv.DictWriter(stream, fieldnames=SEED_CSV_FIELDS).writeheader()
    (output_dir / "SEED_LABEL_EVENTS.jsonl").write_text("", encoding="utf-8")
    receipt = {
        "status": "success",
        "scope": "human_seed_labeling_only",
        "seed_count": len(records),
        "inventory_hash": inventory_hash,
        "source_transition_preview_count": len(source_rows),
        "model_predictions_included": False,
        "formal_ground_truth": False,
        "label_source_initial": "human_seed_label",
        "label_source_after_explicit_confirmation": "human_verified_seed",
        "coverage_status": "PENDING HUMAN SEED LABELS",
        "coverage_transition_requirements": inventory["coverage_transition_requirements"],
        "coverage_scene_requirements": inventory["coverage_scene_requirements"],
        "supersedes_sample_ids_recorded": True,
        "current_urrs_release_unchanged": True,
        "test_pixel_reads": inventory["test_pixel_reads"],
        "created_at": inventory["created_at"],
    }
    (output_dir / "SEED_LABEL_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "SEED_LABEL_RESUME_RECEIPT.json").write_text(
        json.dumps({**receipt, "resume_status": "PENDING HUMAN SEED LABELS", "labeled_count": 0, "verified_seed_count": 0}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {**inventory, "output_dir": str(output_dir), "seed_dir": str(output_dir)}


def build_seed_temporal_inventory(review_dir: Path) -> dict:
    seed_dir = _latest_seed_dir(review_dir)
    if seed_dir is None:
        return {"status": "not_ready", "records": [], "seed_count": 0, "label_source": "human_seed_label"}
    return json.loads((seed_dir / "SEED_LABEL_INVENTORY.json").read_text(encoding="utf-8")) | {"seed_dir": str(seed_dir)}


class SeedTemporalLabelStore:
    def __init__(self, review_dir: Path, inventory: dict):
        self.review_dir = Path(review_dir).resolve()
        self.inventory = inventory
        self.seed_dir = Path(inventory["seed_dir"]).resolve()
        self.csv_path = self.seed_dir / "SEED_LABEL_DECISIONS.csv"
        self.events_path = self.seed_dir / "SEED_LABEL_EVENTS.jsonl"
        self.resume_path = self.seed_dir / "SEED_LABEL_RESUME_RECEIPT.json"
        self.state_path = self.seed_dir / "SEED_LABEL_STATE.json"

    def _read_decisions(self) -> dict[str, dict]:
        if not self.csv_path.exists():
            return {}
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            return {row["sample_id"]: row for row in csv.DictReader(stream) if row.get("sample_id")}

    def _write_decisions(self, decisions: dict[str, dict]) -> None:
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SEED_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(decisions[sample_id] for sample_id in sorted(decisions))

    def _write_resume(self, decisions: dict[str, dict]) -> None:
        verified = sum(row.get("label_source") == "human_verified_seed" for row in decisions.values())
        payload = json.loads(self.resume_path.read_text(encoding="utf-8")) if self.resume_path.exists() else {}
        payload.update(
            {
                "resume_status": "HUMAN SEED LABELING IN PROGRESS" if decisions else "PENDING HUMAN SEED LABELS",
                "labeled_count": len(decisions),
                "verified_seed_count": verified,
                "remaining_count": max(0, int(self.inventory.get("seed_count", 0)) - len(decisions)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.resume_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self) -> dict:
        decisions = self._read_decisions()
        return {
            "seed_count": int(self.inventory.get("seed_count", 0)),
            "labeled_count": len(decisions),
            "verified_seed_count": sum(row.get("label_source") == "human_verified_seed" for row in decisions.values()),
            "remaining_count": max(0, int(self.inventory.get("seed_count", 0)) - len(decisions)),
            "formal_ground_truth_count": 0,
            "model_prediction_count": 0,
            "coverage_status": "HUMAN SEED LABELING IN PROGRESS" if decisions else "PENDING HUMAN SEED LABELS",
        }

    def state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"decisions": {}, "inventory_hash": self.inventory.get("inventory_hash"), "summary": self.summary()}

    def save(self, payload: dict) -> dict:
        sample_id = str(payload.get("sample_id", ""))
        record = next((row for row in self.inventory.get("records", []) if row.get("sample_id") == sample_id), None)
        if record is None:
            raise ValueError(f"unknown seed sample_id: {sample_id}")
        if payload.get("inventory_hash") != self.inventory.get("inventory_hash"):
            raise ValueError("seed inventory hash mismatch")
        if payload.get("decision") not in SEED_DECISIONS:
            raise ValueError(f"decision must be one of {SEED_DECISIONS}")
        if payload.get("before_surface_state") not in SEED_STATES or payload.get("after_surface_state") not in SEED_STATES:
            raise ValueError("surface states must use the Seed Temporal Labeling vocabulary")
        if payload.get("transition_type") not in SEED_TRANSITIONS:
            raise ValueError("transition_type must use the Seed Temporal Labeling vocabulary")
        reviewer = str(payload.get("reviewer", "")).strip()
        if not reviewer:
            raise ValueError("reviewer is required")
        confidence = float(payload.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        confirmed = bool(payload.get("human_confirmed", False))
        row = {
            "sample_id": sample_id,
            "track_seed_id": record.get("track_seed_id"),
            "before_date": record.get("before_date"),
            "after_date": record.get("after_date"),
            "before_surface_state": payload["before_surface_state"],
            "after_surface_state": payload["after_surface_state"],
            "transition_type": payload["transition_type"],
            "existing_building_before": bool(payload.get("existing_building_before", False)),
            "existing_building_after": bool(payload.get("existing_building_after", False)),
            "construction_activity_visible": bool(payload.get("construction_activity_visible", False)),
            "demolition_visible": bool(payload.get("demolition_visible", False)),
            "clearing_visible": bool(payload.get("clearing_visible", False)),
            "new_structure_visible": bool(payload.get("new_structure_visible", False)),
            "repair_or_refurbishment_visible": bool(payload.get("repair_or_refurbishment_visible", False)),
            "confidence": confidence,
            "notes": str(payload.get("notes", "")),
            "decision": payload["decision"],
            "reviewer": reviewer,
            "human_confirmed": confirmed,
            "label_source": "human_verified_seed" if confirmed else "human_seed_label",
            "formal_ground_truth": False,
            "model_independent": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        decisions = self._read_decisions()
        decisions[sample_id] = row
        self._write_decisions(decisions)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "seed_label_saved", "record": row}, ensure_ascii=False) + "\n")
        state = {"decisions": decisions, "inventory_hash": self.inventory.get("inventory_hash"), "summary": self.summary()}
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_resume(decisions)
        return row

    def export(self) -> list[dict]:
        return list(self._read_decisions().values())
