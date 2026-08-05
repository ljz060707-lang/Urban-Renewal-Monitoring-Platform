from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.mti import MultiTaskIntegration
from ..models.surface_state_net import SurfaceStateNet
from ..training.checkpointing import sha256
from .overlay_runtime import _read, _write_tif, alignment_guard


@torch.no_grad()
def calibrated_overlay_infer(config: dict, *, repo: str | Path) -> dict:
    """Apply the existing SurfaceStateNet and calibrated MTI in one forward pass."""
    repo = Path(repo)
    output = Path(config["output"])
    if not output.is_absolute():
        output = repo / output
    output.mkdir(parents=True, exist_ok=False)

    checkpoint = Path(config["surface_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = repo / checkpoint
    selection_path = Path(config["mti_selection"])
    if not selection_path.is_absolute():
        selection_path = repo / selection_path
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    parameters = selection["selected_parameters"]

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SurfaceStateNet()
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    image_t1, metadata_t1 = _read(config["input_t1"])
    image_t2, metadata_t2 = _read(config["input_t2"])
    guard = alignment_guard(metadata_t1, metadata_t2)
    if guard["alignment_status"] != "aligned":
        manifest = {
            "status": "failed_alignment_guard",
            "mti_applied": False,
            **guard,
        }
        (output / "MASTER_OVERLAY_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    cwld = model.forward_cwld(image_t1[None])["probabilities"][0]
    fusu = model.forward_fusu(image_t1[None], image_t2[None])
    p1, p2 = fusu["probabilities_t1"], fusu["probabilities_t2"]
    semantic_change = 1 - (p1 * p2).sum(1)

    confidence = torch.stack([p1.max(1).values, p2.max(1).values], dim=1)
    unary = (
        torch.logit(confidence.clamp(1e-6, 1 - 1e-6))
        * float(parameters["unary_weight"])
    ).sigmoid()
    change = (
        semantic_change[:, None] * float(parameters["change_weight"])
        - float(parameters["persistence_penalty"]) * (1 - semantic_change[:, None])
    ).clamp(0, 1)
    integrated = MultiTaskIntegration(change_weight=1.0)(unary, change)
    state_sequence = integrated["optimal_building_sequence"].float()[0].numpy()
    changed_state = (
        integrated["optimal_building_sequence"][:, 1]
        != integrated["optimal_building_sequence"][:, 0]
    ).float()[0].numpy()

    cwld_np = cwld.numpy()
    fusu_t1 = p1[0].numpy()
    _write_tif(output / "cwld_probabilities.tif", cwld_np, metadata_t1)
    _write_tif(output / "fusu_native_probabilities.tif", fusu_t1, metadata_t1)
    _write_tif(output / "vacant_land_candidate.tif", cwld_np[1], metadata_t1)
    _write_tif(output / "engineering_facility_candidate.tif", cwld_np[2], metadata_t1)
    _write_tif(output / "construction_waste_candidate.tif", cwld_np[3], metadata_t1)
    _write_tif(
        output / "semantic_change_probability.tif",
        semantic_change[0].numpy(),
        metadata_t1,
    )
    _write_tif(
        output / "temporal_consistency.tif",
        1 - semantic_change[0].numpy(),
        metadata_t1,
    )
    _write_tif(output / "mti_semantic_state_sequence.tif", state_sequence, metadata_t1)
    _write_tif(output / "mti_changed_semantic_state.tif", changed_state, metadata_t1)
    renewal = np.maximum.reduce(
        [semantic_change[0].numpy(), cwld_np[1], cwld_np[2], cwld_np[3]]
    )
    _write_tif(output / "renewal_activity_candidate.tif", renewal, metadata_t1)
    Image.fromarray(
        (np.clip(cwld_np[[3, 2, 1]].transpose(1, 2, 0), 0, 1) * 255).astype(
            np.uint8
        )
    ).save(output / "overlay_overview.png")

    outputs = [
        "cwld_probabilities.tif",
        "fusu_native_probabilities.tif",
        "vacant_land_candidate.tif",
        "engineering_facility_candidate.tif",
        "construction_waste_candidate.tif",
        "semantic_change_probability.tif",
        "temporal_consistency.tif",
        "mti_semantic_state_sequence.tif",
        "mti_changed_semantic_state.tif",
        "renewal_activity_candidate.tif",
        "overlay_overview.png",
    ]
    manifest = {
        "status": "success",
        "integration_status": "overlay_candidate",
        "candidate_scope": "fusu_500_sample_domain",
        "scientific_status": "candidate_not_ground_truth",
        "inference_ready": False,
        "surface_checkpoint": str(checkpoint.resolve()),
        "surface_checkpoint_sha256": sha256(checkpoint),
        "mti_selection": {
            "status": "applied",
            "mode": "generic_semantic_state",
            "selection_path": str(selection_path.resolve()),
            "selected_parameters": parameters,
            "calibration_status": selection.get("mti_status"),
        },
        "full20_evidence": config.get("full20_prediction"),
        "rsbuilding_evidence": config.get("rsbuilding_probability"),
        "new_building_output": "unsupported_without_verified_building_semantics",
        "demolition_output": "unsupported_without_verified_building_semantics",
        "outputs": outputs,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "whu_test_reads": 0,
        "second_test_reads": 0,
        **guard,
    }
    (output / "MASTER_OVERLAY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest

