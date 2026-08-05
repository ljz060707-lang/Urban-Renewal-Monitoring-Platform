from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


FEATURES = (
    "total_area",
    "building_area_t1",
    "building_area_t2",
    "new_building_area",
    "demolition_area",
    "any_change_area",
    "construction_candidate_area",
    "impervious_change_area",
    "green_increase_area",
    "green_decrease_area",
    "water_change_area",
)


def write_mudi_features(
    path: str | Path,
    *,
    pixel_area_m2: float,
    pixel_count: int,
    outputs: dict[str, np.ndarray],
    teacher_status: dict,
) -> Path:
    mapping = {
        "total_area": "__total__",
        "building_area_t1": outputs.get("building_t1_probability"),
        "building_area_t2": outputs.get("building_t2_probability"),
        "new_building_area": outputs.get("new_building_candidate"),
        "demolition_area": outputs.get("demolition_candidate"),
        "any_change_area": outputs.get("any_change_probability"),
        "construction_candidate_area": outputs.get("construction_candidate"),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as sink:
        writer = csv.DictWriter(sink, fieldnames=("feature", "value", "unit", "source", "confidence", "teacher_status"))
        writer.writeheader()
        for feature in FEATURES:
            value = mapping.get(feature)
            available = value is not None
            if isinstance(value, str) and value == "__total__":
                numeric = float(pixel_count * pixel_area_m2)
            elif available:
                numeric = float(np.asarray(value).sum() * pixel_area_m2)
            else:
                numeric = None
            writer.writerow(
                {
                    "feature": feature,
                    "value": numeric if available else "",
                    "unit": "m2",
                    "source": "v2_fusion" if available else "unavailable",
                    "confidence": "model_evidence" if available else "",
                    "teacher_status": "available" if available else "pending_teacher",
                }
            )
    return target
