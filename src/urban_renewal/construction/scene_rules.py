from __future__ import annotations


def classify_scene(features: dict) -> tuple[str, list[str]]:
    """Conservative rules: single evidence channels never assert active construction."""
    exposed = features.get("exposed_ratio", 0.0)
    dust = features.get("dustproof_net_ratio", 0.0)
    facility = features.get("facility_ratio", 0.0)
    debris = features.get("debris_ratio", 0.0)
    change = features.get("full20_change_evidence", 0.0)
    building_overlap = features.get("building_overlap", 0.0)
    if debris > 0.35 and change < 0.25:
        return "waste_landfill", ["material_debris_dominant", "low_change_evidence"]
    if facility > 0.35 and exposed < 0.15:
        return "unknown_scene", ["engineering_facility_is_not_direct_construction_label"]
    if (dust > 0.08 or exposed > 0.25) and change > 0.35:
        return "demolition_renewal" if building_overlap > 0.20 else "active_construction", ["multiple_element_evidence", "full20_change_support"]
    return "unknown_scene", ["insufficient_multi_source_evidence"]
