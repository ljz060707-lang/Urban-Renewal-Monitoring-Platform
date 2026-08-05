from __future__ import annotations


CANONICAL_TRANSITIONS = {
    ("vacant_land", "construction_site"): "construction_start",
    ("construction_site", "building"): "completion",
    ("vacant_land", "building"): "newly_completed_building",
    ("building", "vacant_land"): "demolition",
    ("building", "construction_site"): "redevelopment_start",
    ("building", "building"): "stable_building",
    ("construction_site", "construction_site"): "construction_active",
    ("vacant_land", "vacant_land"): "stable_vacant",
}


def derive_transition(before: str, after: str) -> str:
    return CANONICAL_TRANSITIONS.get((before, after), "uncertain_transition")
