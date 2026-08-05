from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .review_migration import migrate_existing_review
from .transition_inventory import build_transition_track_inventory
from .transition_preview import diagnose_transition_previews, regenerate_transition_previews


def run_transition_review_migration(project_root: Path, review_dir: Path, *, regenerate: bool = True) -> dict:
    project_root, review_dir = Path(project_root), Path(review_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    migration_dir = project_root / "outputs" / "urrs" / "review" / f"review_migration_{stamp}"
    migration = migrate_existing_review(project_root, review_dir, migration_dir)
    diagnosis = diagnose_transition_previews(review_dir, migration_dir)
    preview_inventory = None
    track_inventory = None
    if regenerate:
        preview_dir = migration_dir / "transition_previews"
        preview_inventory = regenerate_transition_previews(project_root, review_dir, preview_dir)
        track_inventory = build_transition_track_inventory(review_dir, preview_inventory, migration_dir / "track_review")
    state = {
        "status": "success",
        "review_migration_dir": str(migration_dir.resolve()),
        "existing_review_import": migration,
        "transition_preview_diagnosis": {
            key: diagnosis[key] for key in (
                "transition_preview_count", "valid_under_display_gate", "invalid_under_display_gate",
                "identical_same_window_across_transitions", "root_causes",
            )
        },
        "transition_preview_regeneration": None if preview_inventory is None else {
            "valid_transition_preview_count": preview_inventory["valid_transition_preview_count"],
            "review_allowed_count": preview_inventory["review_allowed_count"],
            "review_gate_reason": preview_inventory["review_gate_reason"],
            "transition_review_status": preview_inventory["transition_review_status"],
        },
        "construction_track_count": 0 if track_inventory is None else track_inventory["track_count"],
        "building_track_count": 0 if track_inventory is None else track_inventory["building_track_count"],
        "geometry_review_complete": migration["review_state"]["geometry_review_complete"],
        "surface_state_review_complete": migration["review_state"]["surface_state_review_complete"],
        "transition_preview_valid": False if preview_inventory is None else (
            preview_inventory["valid_transition_preview_count"] == len(preview_inventory["records"])
        ),
        "transition_review_complete": False,
        "human_temporal_labels_available": False,
        "transition_review_status": ("PENDING HUMAN TRANSITION REVIEW" if (
            preview_inventory and preview_inventory["review_allowed_count"] > 0
        ) else "PENDING FORMAL CONSTRUCTION EVIDENCE"),
        "construction_tfr_validated": False,
        "construction_mti_evaluated": False,
        "construction_temporal_path_selected": "no_mti",
        "construction_tracks_ready": False,
        "construction_area_audit_ready": False,
        "existing_urrs_candidate_unchanged": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (migration_dir / "CONSTRUCTION_REVIEW_RESUME_STATE.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status_dir = project_root / "outputs" / "urrs" / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "CONSTRUCTION_REVIEW_RESUME_STATE.json"
    status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state
