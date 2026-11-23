"""Uncertainty-aware temporal consistency utilities for R5."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def perturbation_catalog() -> dict[str, list[str]]:
    return {
        "weak": ["brightness", "contrast"],
        "strong": ["blur", "rotation", "scale", "partial_occlusion", "small_affine", "simulated_parallax"],
        "instance_stress": ["neighbor_merge", "mask_split", "boundary_erosion", "boundary_dilation"],
    }


def cycle_consistency(edges_18_19: Iterable[Mapping[str, Any]], edges_19_20: Iterable[Mapping[str, Any]], edges_18_20: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    direct = {(str(row.get("source_id")), str(row.get("target_id"))): row for row in edges_18_20}
    middle: dict[str, list[Mapping[str, Any]]] = {}
    for row in edges_18_19:
        middle.setdefault(str(row.get("target_id")), []).append(row)
    later: dict[str, list[Mapping[str, Any]]] = {}
    for row in edges_19_20:
        later.setdefault(str(row.get("source_id")), []).append(row)
    rows: list[dict[str, Any]] = []
    for mid_id, first_edges in middle.items():
        for first in first_edges:
            source_id = str(first.get("source_id"))
            for second in later.get(mid_id, []):
                target_id = str(second.get("target_id"))
                direct_row = direct.get((source_id, target_id))
                rows.append({
                    "source_id": source_id,
                    "middle_id": mid_id,
                    "target_id": target_id,
                    "direct_support": bool(direct_row),
                    "formal_transition_eligible": False,
                    "status": "CYCLE_SUPPORT" if direct_row else "CYCLE_CONFLICT_OR_MISSING_DIRECT_SUPPORT",
                })
    return rows


def resolve_unknown_cascade(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply an evidence-only cascade without forcing scientific labels."""

    stages = [
        "R1_STRICT_DIRECT_IDENTITY",
        "R2_TBIA",
        "R3_ALIGNMENT_AWARE_CANDIDATE",
        "R4_MERGED_MASK_RBF_SAM",
        "R5_THREE_EPOCH_CYCLE",
        "R6_NEIGHBORHOOD_TOPOLOGY",
        "R7_SUPPLEMENTAL_PIXELS",
        "R8_VISUAL_ENSEMBLE",
    ]
    trace: list[dict[str, Any]] = []
    current = "UNRESOLVED"
    for stage in stages:
        value = evidence.get(stage, {})
        status = str(value.get("status", "NOT_AVAILABLE")) if isinstance(value, Mapping) else "NOT_AVAILABLE"
        trace.append({"stage": stage, "status": status})
        if isinstance(value, Mapping) and value.get("scientific_state") in {"STABLE", "ADDITION", "DEMOLITION"} and float(value.get("confidence", 0.0)) >= 0.90:
            current = str(value["scientific_state"])
            break
    operational = current if current != "UNRESOLVED" else "STABLE" if evidence.get("operational_forced_stable") else None
    return {
        "scientific_state": current,
        "operational_state": operational,
        "forced_resolution_flag": operational is not None and current == "UNRESOLVED",
        "trace": trace,
        "model_resolved_unknown": current != "UNRESOLVED",
        "evidence_limited_unknown": current == "UNRESOLVED",
    }
