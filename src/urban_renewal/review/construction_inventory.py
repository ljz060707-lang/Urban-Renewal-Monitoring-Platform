from __future__ import annotations
from .construction_inventory_v2 import build_construction_inventory as _canonical_build_construction_inventory

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")


def build_construction_inventory(review_dir: Path, geometry_inventory: dict) -> dict:
    """Build fixed audit tracks from existing previews without inventing predictions."""
    review_dir = Path(review_dir)
    records = geometry_inventory.get("records", [])
    by_id = {row["sample_id"]: row for row in records}
    tracks = []
    for index in range(16):
        previews = {}
        sources = {}
        bounds = None
        for date in DATES:
            sid = f"geometry_{date}_{index:02d}"
            row = by_id.get(sid)
            if row:
                previews[date] = row.get("preview_paths", {}).get("tile")
                sources[date] = row.get("source_paths", {})
                bounds = bounds or row.get("window_bounds")
        if not previews:
            continue
        tracks.append({
            "track_id": f"construction_review_track_{index:02d}",
            "scene_type": "construction_site",
            "dates": list(DATES),
            "preview_paths": previews,
            "source_paths": sources,
            "window_bounds": bounds or [],
            "pseudo_states": {date: "unknown_state" for date in DATES},
            "pseudo_transitions": {
                "2018_2020": "uncertain",
                "2020_2022": "uncertain",
                "2022_2023": "uncertain",
            },
            "label_source": "pseudo_rule",
            "scientific_status": "candidate_not_ground_truth",
            "human_review_required": True,
            "sampling_reason": "fixed_geometry_window_cross_date_track",
        })
    canonical = json.dumps(tracks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    inventory = {
        "version": "construction_activity_v1",
        "records": tracks,
        "inventory_hash": hashlib.sha256(canonical).hexdigest(),
        "label_source": "pseudo_rule",
        "notice": "PSEUDO LABEL - NOT GROUND TRUTH",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target = review_dir / "CONSTRUCTION_REVIEW_INVENTORY.json"
    target.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    return inventory

# Canonical regenerated Transition track inventory; legacy fixed-window code remains import-compatible.
build_construction_inventory = _canonical_build_construction_inventory
