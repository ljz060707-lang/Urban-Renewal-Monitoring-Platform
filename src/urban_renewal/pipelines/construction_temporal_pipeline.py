from __future__ import annotations


def temporal_status():
    return {
        "status": "PENDING HUMAN LABELS",
        "tfr_implementation": "ready",
        "construction_mti": {"available": True, "enabled": False, "selection_status": "experimental_unselected"},
        "selected_construction_temporal_integration": "no_mti",
        "formal_metrics_allowed_label_sources": ["official_ground_truth", "human_verified", "human_corrected"],
    }
