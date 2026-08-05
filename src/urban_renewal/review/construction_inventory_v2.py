from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")


def _latest_preview_inventory(review_dir: Path) -> tuple[Path | None, dict | None]:
    candidates = sorted(review_dir.parent.glob("review_migration_*/transition_previews/TRANSITION_REVIEW_INVENTORY.json"))
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("records"):
            return path, payload
    return None, None


def _runtime_evidence(row: dict) -> tuple[dict, list[str]]:
    evidence = {key: dict(value) for key, value in (row.get("evidence") or {}).items()}
    full20 = dict(evidence.get("full20") or {})
    paths = [path for path in full20.get("paths", []) if any(token in Path(path).name.lower() for token in ("full20", "change", "new", "demolition"))]
    full20["status"] = ("available" if paths else "unavailable") if row.get("full20_supported") else "not_applicable"
    full20["paths"] = paths
    evidence["full20"] = full20
    missing = [key for key, value in evidence.items() if value.get("status") == "unavailable"]
    return evidence, missing


def _from_transition_preview(review_dir: Path) -> dict | None:
    inventory_path, preview_inventory = _latest_preview_inventory(review_dir)
    if inventory_path is None or preview_inventory is None:
        return None
    grouped: dict[str, list[dict]] = {}
    for row in preview_inventory.get("records", []):
        grouped.setdefault(row["track_id"], []).append(row)
    records = []
    for track_id, rows in sorted(grouped.items()):
        evidence_by_transition = {}
        evidence_missing_by_transition = {}
        preview_paths = {}
        for row in rows:
            preview_paths.setdefault(row["before_date"], row["raw_before_path"])
            preview_paths.setdefault(row["after_date"], row["raw_after_path"])
            evidence, missing = _runtime_evidence(row)
            evidence_by_transition[row["transition"]] = evidence
            evidence_missing_by_transition[row["transition"]] = missing
        records.append({
            "track_id": track_id,
            "object_type": "construction_track",
            "scene_type": "construction_site",
            "dates": list(DATES),
            "preview_paths": preview_paths,
            "pair_preview_paths": {row["transition"]: row["pair_preview_path"] for row in rows},
            "evidence_by_transition": evidence_by_transition,
            "evidence_missing_by_transition": evidence_missing_by_transition,
            "supersedes_sample_ids": sorted({sid for row in rows for sid in row.get("supersedes_sample_ids", [])}),
            "legacy_transition_ids": [row["sample_id"] for row in rows],
            "pseudo_states": {date: "unknown_state" for date in DATES},
            "pseudo_transitions": {"2018_2020": "uncertain", "2020_2022": "uncertain", "2022_2023": "uncertain"},
            "review_status": "pending_human_transition_review" if all(row.get("review_allowed", False) for row in rows) else "deferred_pending_formal_evidence",
            "review_allowed": all(row.get("review_allowed", False) for row in rows),
            "label_source": "model_prediction",
            "formal_training_eligible": False,
            "scientific_status": "experimental_candidate",
        })
    canonical = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "status": "success",
        "source_type": "regenerated_transition_preview_inventory",
        "source_migration_dir": str(inventory_path.parents[1]),
        "records": records,
        "track_count": len(records),
        "building_track_count": sum(row.get("object_type") == "building_track" for row in records),
        "inventory_hash": hashlib.sha256(canonical).hexdigest(),
        "transition_review_complete": False,
        "human_temporal_labels_available": False,
        "label_source": "model_prediction",
        "scientific_status": "candidate_not_ground_truth",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_construction_inventory(review_dir: Path, geometry_inventory: dict) -> dict:
    """Load regenerated track evidence, falling back to fixed geometry windows."""
    review_dir = Path(review_dir)
    regenerated = _from_transition_preview(review_dir)
    if regenerated is not None:
        return regenerated
    records = geometry_inventory.get("records", [])
    by_id = {row["sample_id"]: row for row in records}
    tracks = []
    for index in range(16):
        previews, sources, bounds = {}, {}, None
        for scene_date in DATES:
            row = by_id.get(f"geometry_{scene_date}_{index:02d}")
            if row:
                previews[scene_date] = row.get("preview_paths", {}).get("tile")
                sources[scene_date] = row.get("source_paths", {})
                bounds = bounds or row.get("window_bounds")
        if previews:
            tracks.append({
                "track_id": f"construction_review_track_{index:02d}",
                "object_type": "construction_track",
                "scene_type": "construction_site",
                "dates": list(DATES),
                "preview_paths": previews,
                "source_paths": sources,
                "window_bounds": bounds or [],
                "pseudo_states": {date: "unknown_state" for date in DATES},
                "pseudo_transitions": {"2018_2020": "uncertain", "2020_2022": "uncertain", "2022_2023": "uncertain"},
                "label_source": "pseudo_rule",
                "scientific_status": "candidate_not_ground_truth",
                "human_review_required": True,
                "review_allowed": False,
                "review_status": "deferred_pending_formal_evidence",
            })
    canonical = json.dumps(tracks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "version": "construction_activity_v1",
        "source_type": "fixed_geometry_window_fallback",
        "records": tracks,
        "track_count": len(tracks),
        "building_track_count": 0,
        "inventory_hash": hashlib.sha256(canonical).hexdigest(),
        "label_source": "pseudo_rule",
        "notice": "PSEUDO LABEL - NOT GROUND TRUTH",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
