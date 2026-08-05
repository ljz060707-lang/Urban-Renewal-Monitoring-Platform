from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from affine import Affine
from shapely.geometry import box

from urban_renewal.construction.area_accounting import audit_area
from urban_renewal.temporal.construction_mti import ConstructionMTI
from urban_renewal.temporal.construction_no_mti import ConstructionNoMTI
from urban_renewal.temporal.construction_state_graph import ConstructionStateGraph
from urban_renewal.temporal.construction_tfr import ConstructionTemporalFeatureService
from urban_renewal.tracking.construction_matching import match_construction_objects
from urban_renewal.tracking.split_merge import detect_split_merge


def _tfr_smoke(timesteps: int) -> dict:
    torch.manual_seed(20260729 + timesteps)
    module = ConstructionTemporalFeatureService(8, num_heads=2, num_layers=1)
    features = torch.randn(1, timesteps, 8, 4, 4, requires_grad=True)
    valid = torch.ones(1, timesteps, dtype=torch.bool)
    valid[:, -1] = False
    output = module(features, timestamps=torch.arange(timesteps)[None] * 31, temporal_valid_mask=valid)
    refined = output["refined_features"][0]
    refined.sum().backward()
    return {
        "timesteps": timesteps,
        "input_shape": list(features.shape),
        "output_shape": list(refined.shape),
        "finite": bool(torch.isfinite(refined).all()),
        "invalid_timestep_zero": bool(torch.count_nonzero(refined[:, -1]) == 0),
        "gradient_present": features.grad is not None and bool(torch.isfinite(features.grad).all()),
    }


def run_component_smoke(project_root: Path, *, output_dir: Path | None = None) -> dict:
    """Run synthetic Construction component checks without reading project pixels."""
    project_root = Path(project_root)
    torch.manual_seed(20260729)
    np.random.seed(20260729)

    tfr = {str(timesteps): _tfr_smoke(timesteps) for timesteps in (2, 4)}

    probabilities = np.array(
        [[0.8, 0.15, 0.05], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
        dtype=np.float64,
    )
    valid_mask = np.array([True, True, False, True])
    no_mti = ConstructionNoMTI()(probabilities, valid_mask=valid_mask)
    state_graph = ConstructionStateGraph(
        {
            "states": ["stable_building", "active_construction", "completed_building"],
            "default_transition_cost": 4,
        }
    )
    mti = ConstructionMTI(state_graph)(probabilities)

    previous = [{"track_id": "previous", "geometry": box(0, 0, 10, 10), "scene_type": "active"}]
    current = [
        {"track_id": "current_a", "geometry": box(0, 0, 5, 10), "scene_type": "active"},
        {"track_id": "current_b", "geometry": box(5, 0, 10, 10), "scene_type": "active"},
    ]
    matching = match_construction_objects(previous, current, max_centroid_distance=30)
    split_merge = detect_split_merge(previous, current)

    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:11, 1:11] = 1
    transform = Affine.translation(0, 12) * Affine.scale(1, -1)
    area = audit_area(mask, transform, [box(1, 2, 11, 12)], "EPSG:32650")

    receipt = {
        "status": "success",
        "scope": "development_component_smoke_not_validation",
        "project_root": str(project_root.resolve()),
        "tfr": tfr,
        "no_mti": {
            "integration": no_mti["integration"],
            "invalid_state": int(no_mti["state_sequence"][2]),
            "selected_default": ConstructionNoMTI.selected,
        },
        "mti": {
            "available": ConstructionMTI.available,
            "enabled": ConstructionMTI.enabled,
            "selection_status": ConstructionMTI.selection_status,
            "integration": mti["integration"],
            "sequence_length": len(mti["state_sequence"]),
        },
        "tracking": {
            "candidate_pairs": matching["candidate_pairs"],
            "match_count": len(matching["matches"]),
            "split_events": len([event for event in split_merge if event["event_type"] == "split"]),
        },
        "area": {
            "audit": area.area_audit,
            "spatial_status": area.spatial_status,
            "raster_area_m2": area.raster_area_m2,
            "vector_area_m2": area.vector_area_m2,
        },
        "formal_metrics_allowed": False,
        "checkpoint_registerable": False,
        "test_pixel_reads": {"UCS": 0, "CWLD": 0, "WHU": 0, "SECOND": 0},
        "current_urrs_release_unchanged": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "urrs" / "smoke" / f"construction_component_smoke_{stamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "COMPONENT_SMOKE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**receipt, "receipt": str(receipt_path.resolve())}
