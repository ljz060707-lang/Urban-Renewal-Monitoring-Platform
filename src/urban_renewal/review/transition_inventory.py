from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")


def build_transition_track_inventory(review_dir: Path, preview_inventory: dict, output_dir: Path) -> dict:
    """Upgrade fixed transition samples to track audit rows while preserving legacy IDs."""
    review_dir, output_dir = Path(review_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_track: dict[str, list[dict]] = {}
    for row in preview_inventory.get("records", []):
        by_track.setdefault(row["track_id"], []).append(row)
    tracks = []
    for track_id, rows in sorted(by_track.items()):
        previews = {}
        for row in rows:
            previews[row["before_date"]] = row["raw_before_path"]
            previews[row["after_date"]] = row["raw_after_path"]
        tracks.append({
            "track_id": track_id,
            "object_type": "construction_track",
            "dates": list(DATES),
            "preview_paths": previews,
            "supersedes_sample_ids": sorted({sid for row in rows for sid in row.get("supersedes_sample_ids", [])}),
            "legacy_transition_ids": [row["sample_id"] for row in rows],
            "proposed_states": {date: "unknown_state" for date in DATES},
            "proposed_transitions": {
                "2018_2020": "UNCERTAIN",
                "2020_2022": "UNCERTAIN",
                "2022_2023": "UNCERTAIN",
            },
            "review_status": "deferred_pending_construction_evidence",
            "review_allowed": all(row.get("review_allowed", False) for row in rows),
            "label_source": "model_prediction",
            "formal_training_eligible": False,
            "scientific_status": "experimental_candidate",
        })
    canonical = json.dumps(tracks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload = {
        "status": "success",
        "records": tracks,
        "track_count": len(tracks),
        "building_track_count": 0,
        "inventory_hash": hashlib.sha256(canonical).hexdigest(),
        "transition_review_complete": False,
        "human_temporal_labels_available": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "TRANSITION_TRACK_INVENTORY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "track_id", "object_type", "supersedes_sample_ids", "legacy_transition_ids",
        "review_status", "review_allowed", "label_source", "formal_training_eligible",
        "scientific_status",
    ]
    with (output_dir / "TRANSITION_TRACK_INVENTORY.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore"); writer.writeheader()
        for row in tracks:
            writer.writerow({
                **row,
                "supersedes_sample_ids": json.dumps(row["supersedes_sample_ids"], ensure_ascii=False),
                "legacy_transition_ids": json.dumps(row["legacy_transition_ids"], ensure_ascii=False),
            })
    return payload
