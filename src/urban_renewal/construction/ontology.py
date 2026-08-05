from __future__ import annotations

UCS_NATIVE_CLASSES = (
    "ucs_background",
    "ucs_exposed_area",
    "ucs_dustproof_net",
    "ucs_building",
)
CWLD_NATIVE_CLASSES = (
    "cwld_background",
    "cwld_vacant_landfillable",
    "cwld_engineering_facility",
    "cwld_waste_dumping",
)
EVIDENCE_CLASSES = (
    "background_evidence",
    "exposed_ground_evidence",
    "dustproof_net_evidence",
    "building_evidence",
    "engineering_facility_evidence",
    "material_debris_evidence",
    "uncertain_evidence",
)
SCENE_TYPES = (
    "active_construction",
    "infrastructure_construction",
    "demolition_renewal",
    "waste_landfill",
    "material_yard",
    "quarry_mining",
    "vacant_bare_land",
    "completed_site",
    "unknown_scene",
)
CONSTRUCTION_STATES = (
    "stable_building",
    "demolition_started",
    "cleared_site",
    "active_construction_exposed",
    "active_construction_dustproof",
    "foundation_or_excavation_candidate",
    "structure_rising",
    "near_completion",
    "completed_building",
    "idle_vacant",
    "waste_or_material_storage",
    "infrastructure_work",
    "unknown_state",
    "building_repair_or_refurbishment_candidate",
)
LABEL_SOURCES = (
    "official_ground_truth",
    "human_verified",
    "human_corrected",
    "pseudo_rule",
    "model_prediction",
    "unknown",
)

UCS_TO_EVIDENCE = {
    0: ("background_evidence", "strong"),
    1: ("exposed_ground_evidence", "strong"),
    2: ("dustproof_net_evidence", "strong"),
    3: ("building_evidence", "strong"),
}
CWLD_TO_EVIDENCE = {
    0: ("background_evidence", "auxiliary"),
    1: ("exposed_ground_evidence", "auxiliary"),
    2: ("engineering_facility_evidence", "strong"),
    3: ("material_debris_evidence", "strong"),
}
PROHIBITED_INTERPRETATIONS = (
    "cwld_engineering_facility_is_active_construction",
    "cwld_vacant_land_is_active_construction",
    "absence_of_building_is_vacant_land",
    "bare_ground_plus_debris_is_construction_site",
)

FORMAL_METRIC_LABEL_SOURCES = frozenset(
    {"official_ground_truth", "human_verified", "human_corrected"}
)


def evidence_projection(dataset: str, probabilities):
    """Project native probabilities without treating missing classes as negatives."""
    import torch

    mapping = UCS_TO_EVIDENCE if dataset.casefold() == "ucs" else CWLD_TO_EVIDENCE
    out = torch.zeros(
        probabilities.shape[0], len(EVIDENCE_CLASSES), *probabilities.shape[-2:],
        dtype=probabilities.dtype, device=probabilities.device,
    )
    presence = torch.zeros(len(EVIDENCE_CLASSES), dtype=torch.bool, device=probabilities.device)
    for native_index, (name, _) in mapping.items():
        target_index = EVIDENCE_CLASSES.index(name)
        out[:, target_index] = probabilities[:, native_index]
        presence[target_index] = True
    out[:, EVIDENCE_CLASSES.index("uncertain_evidence")] = 1.0 - probabilities.max(1).values
    presence[EVIDENCE_CLASSES.index("uncertain_evidence")] = True
    return out, presence
