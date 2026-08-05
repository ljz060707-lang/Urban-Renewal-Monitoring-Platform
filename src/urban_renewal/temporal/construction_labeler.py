from __future__ import annotations

from urban_renewal.construction.ontology import CONSTRUCTION_STATES, LABEL_SOURCES, SCENE_TYPES


def pseudo_label(track_id, states, scene_type="unknown_scene"):
    if scene_type not in SCENE_TYPES or any(state not in CONSTRUCTION_STATES for state in states):
        raise ValueError("unknown construction ontology value")
    return {"track_id": track_id, "states": list(states), "scene_type": scene_type, "label_source": "pseudo_rule", "formal_metric_eligible": False, "warning": "PSEUDO LABEL — NOT GROUND TRUTH"}


def promote_human_label(record, *, corrected=False, reviewer=None):
    if record.get("label_source") != "pseudo_rule":
        raise ValueError("only pseudo labels can be promoted by this workflow")
    if not reviewer:
        raise ValueError("reviewer is required")
    output = dict(record)
    output.update({"label_source": "human_corrected" if corrected else "human_verified", "reviewer": reviewer, "formal_metric_eligible": True})
    return output
