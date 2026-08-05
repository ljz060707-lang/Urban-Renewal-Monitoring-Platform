HARD_NEGATIVE_TYPES = (
    "waste_site", "material_yard", "ordinary_bare_land", "fallow_field", "riverbank",
    "quarry_mining", "industrial_vacant_parcel", "parking_lot",
    "completed_unlandscaped_site", "infrastructure_construction",
)


def eligible_hard_negative(record):
    return bool(record.get("source_audited") and record.get("label_source") in {"official_ground_truth", "human_verified", "human_corrected"})
