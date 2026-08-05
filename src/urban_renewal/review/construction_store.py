from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from .construction_schema import normalize_construction_decision


FIELDS = (
    "track_id", "scene_type", "state_2018", "state_2020", "state_2022", "state_2023",
    "transition_2018_2020", "transition_2020_2022", "transition_2022_2023",
    "renewal_candidate", "expansion_candidate", "repair_candidate", "boundary_review",
    "confidence", "label_source", "notes", "reviewer", "reviewed_at", "inventory_hash",
)


class ConstructionReviewStore:
    """Independent store; never reads or writes geometry review decisions."""

    def __init__(self, review_dir, inventory=None):
        self.root = Path(review_dir).resolve() / "construction_activity"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "construction_review_state.json"
        self.csv_path = self.root / "construction_review_decisions.csv"
        self.jsonl_path = self.root / "construction_review_decisions.jsonl"
        self.inventory = inventory or {}

    def _read(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"version": "construction_activity_v1", "decisions": {}, "reviewer": ""}

    def _atomic(self, path: Path, text: str):
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        os.close(fd)
        Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def save(self, payload):
        record = next((row for row in self.inventory.get("records", []) if row.get("track_id") == payload.get("track_id")), None)
        if record is not None and not record.get("review_allowed", False):
            missing = sorted({item for values in record.get("evidence_missing_by_transition", {}).values() for item in values})
            raise ValueError("construction track review is blocked until formal evidence is available: " + ", ".join(missing or ["review_gate"]))
        state = self._read()
        row = normalize_construction_decision(payload, state.get("reviewer", ""))
        state["reviewer"] = row["reviewer"]
        state["decisions"][row["track_id"]] = row
        self._atomic(self.state_path, json.dumps(state, ensure_ascii=False, indent=2))
        self.export(state)
        return row

    def export(self, state=None):
        state = state or self._read()
        rows = list(state.get("decisions", {}).values())
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in FIELDS})
        self.jsonl_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        return rows

    def summary(self, inventory):
        decisions = self._read().get("decisions", {})
        total = len(inventory.get("records", []))
        return {"reviewed_count": len(decisions), "total_count": total, "pending_count": max(0, total - len(decisions))}
