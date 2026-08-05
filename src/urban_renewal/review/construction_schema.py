from __future__ import annotations

from datetime import datetime, timezone

from urban_renewal.construction.ontology import CONSTRUCTION_STATES, LABEL_SOURCES


BOUNDARY_REVIEWS = (
    "accept",
    "too_small",
    "too_large",
    "fragmented",
    "wrong",
    "needs_redraw",
)

TRANSITIONS = (
    "no_change",
    "demolition_started",
    "cleared",
    "construction_started",
    "continued_construction",
    "structure_rising",
    "completed",
    "repair_candidate",
    "uncertain",
)

SCENE_TYPES = (
    "construction_site",
    "building_track",
)


def normalize_construction_decision(payload: dict, default_reviewer: str = "") -> dict:
    track_id = str(payload.get("track_id", "")).strip()
    if not track_id:
        raise ValueError("track_id is required")
    scene_type = str(payload.get("scene_type", "construction_site"))
    if scene_type not in SCENE_TYPES:
        raise ValueError(f"invalid scene_type: {scene_type}")
    result = {
        "track_id": track_id,
        "scene_type": scene_type,
        "boundary_review": str(payload.get("boundary_review", "accept")),
        "renewal_candidate": bool(payload.get("renewal_candidate", False)),
        "expansion_candidate": bool(payload.get("expansion_candidate", False)),
        "repair_candidate": bool(payload.get("repair_candidate", False)),
        "confidence": float(payload.get("confidence", 0.0)),
        "notes": str(payload.get("notes", "")).strip(),
        "reviewer": str(payload.get("reviewer") or default_reviewer).strip(),
        "inventory_hash": str(payload.get("inventory_hash", "")),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    if result["boundary_review"] not in BOUNDARY_REVIEWS:
        raise ValueError(f"invalid boundary_review: {result['boundary_review']}")
    if not 0.0 <= result["confidence"] <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    for year in (2018, 2020, 2022, 2023):
        key = f"state_{year}"
        value = str(payload.get(key, "unknown_state"))
        if value not in CONSTRUCTION_STATES:
            raise ValueError(f"invalid {key}: {value}")
        result[key] = value
    for interval in ("2018_2020", "2020_2022", "2022_2023"):
        key = f"transition_{interval}"
        value = str(payload.get(key, "uncertain"))
        if value not in TRANSITIONS:
            raise ValueError(f"invalid {key}: {value}")
        result[key] = value
    changed = bool(payload.get("human_modified", False))
    result["label_source"] = "human_corrected" if changed else "human_verified"
    if result["label_source"] not in LABEL_SOURCES:
        raise AssertionError("invalid label provenance")
    return result
