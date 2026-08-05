from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.mti import MultiTaskIntegration
from ..models.surface_state_net import SurfaceStateNet
from ..models.temporal_change_head import TemporalChangeHead
from ..models.tfr import TemporalFeatureRefinement
from ..training.checkpointing import sha256
from ..training.full_trainer import _fusu_forward
from .overlay_runtime import _read, _write_tif, alignment_guard


def _state_hash(state: dict | None) -> str | None:
    if state is None:
        return None
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def final_overlay_infer(config: dict, *, repo: str | Path) -> dict:
    repo = Path(repo)
    output = Path(config["output"])
    if not output.is_absolute():
        output = repo / output
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(config["surface_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = repo / checkpoint
    mti_path = Path(config["mti_selection"])
    if not mti_path.is_absolute():
        mti_path = repo / mti_path
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SurfaceStateNet()
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    use_tfr = state.get("tfr_state_dict") is not None
    tfr = None
    if use_tfr:
        snapshot = state.get("config_snapshot", {})
        tfr = TemporalFeatureRefinement(
            24,
            heads=int(snapshot.get("temporal", {}).get("num_heads", 4)),
            dropout=float(snapshot.get("temporal", {}).get("dropout", 0.1)),
        )
        tfr.load_state_dict(state["tfr_state_dict"], strict=True)
        tfr.eval()
    change_head = None
    if state.get("change_head_state_dict") is not None:
        change_head = TemporalChangeHead(24)
        change_head.load_state_dict(state["change_head_state_dict"], strict=True)
        change_head.eval()
    selection = json.loads(mti_path.read_text(encoding="utf-8"))
    integration_mode = selection.get("integration_mode", "mti")
    parameters = selection.get("selected_parameters")
    t1, metadata1 = _read(config["input_t1"])
    t2, metadata2 = _read(config["input_t2"])
    guard = alignment_guard(metadata1, metadata2)
    if guard["alignment_status"] != "aligned":
        result = {"status": "failed_alignment_guard", **guard}
        (output / "MASTER_OVERLAY_MANIFEST.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result
    cwld = model.forward_cwld(t1[None])["probabilities"][0]
    logits1, logits2, dense = _fusu_forward(
        model, tfr, change_head, t1[None], t2[None], use_tfr
    )
    p1, p2 = logits1.softmax(1), logits2.softmax(1)
    semantic_change = 1 - (p1 * p2).sum(1)
    if integration_mode == "no_mti":
        semantic_states = torch.stack([p1.argmax(1), p2.argmax(1)], 1)
        mti_state = semantic_states.float()[0].numpy()
        mti_change = (semantic_states[:, 1] != semantic_states[:, 0]).float()[0].numpy()
    else:
        if not isinstance(parameters, dict):
            raise RuntimeError("Selected MTI integration requires selected_parameters")
        confidence = torch.stack([p1.max(1).values, p2.max(1).values], 1)
        unary = (
            torch.logit(confidence.clamp(1e-6, 1 - 1e-6))
            * float(parameters["unary_weight"])
        ).sigmoid()
        adjusted = (
            semantic_change[:, None] * float(parameters["change_weight"])
            - float(parameters["persistence_penalty"]) * (1 - semantic_change[:, None])
        ).clamp(0, 1)
        integrated = MultiTaskIntegration(change_weight=1.0)(unary, adjusted)
        mti_state = integrated["optimal_building_sequence"].float()[0].numpy()
        mti_change = (
            integrated["optimal_building_sequence"][:, 1]
            != integrated["optimal_building_sequence"][:, 0]
        ).float()[0].numpy()
    cwld_np, fusu_np = cwld.numpy(), p1[0].numpy()
    arrays = {
        "cwld_probabilities.tif": cwld_np,
        "fusu_native_probabilities.tif": fusu_np,
        "vacant_land_candidate.tif": cwld_np[1],
        "engineering_facility_candidate.tif": cwld_np[2],
        "construction_waste_candidate.tif": cwld_np[3],
        "semantic_change_probability.tif": semantic_change[0].numpy(),
        "temporal_consistency.tif": 1 - semantic_change[0].numpy(),
        "mti_semantic_state_sequence.tif": mti_state,
        "mti_changed_semantic_state.tif": mti_change,
        "renewal_activity_candidate.tif": np.maximum.reduce(
            [semantic_change[0].numpy(), cwld_np[1], cwld_np[2], cwld_np[3]]
        ),
    }
    if dense is not None:
        dense_up = torch.nn.functional.interpolate(
            dense, size=semantic_change.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0].numpy()
        arrays["dense_change_probability.tif"] = dense_up
    numerical = {}
    for name, array in arrays.items():
        _write_tif(output / name, array, metadata1)
        value = np.asarray(array)
        numerical[name] = {
            "finite": bool(np.isfinite(value).all()),
            "min": float(np.nanmin(value)),
            "max": float(np.nanmax(value)),
            "mean": float(np.nanmean(value)),
            "zero_fraction": float((value == 0).mean()),
            "one_fraction": float((value == 1).mean()),
        }
    overview = (
        np.clip(cwld_np[[3, 2, 1]].transpose(1, 2, 0), 0, 1) * 255
    ).astype(np.uint8)
    Image.fromarray(overview).save(output / "overlay_overview.png")
    review = output / "visual_review"
    review.mkdir()
    Image.fromarray(overview).save(review / "contact_sheet.png")
    (review / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Final overlay review</title>"
        "<style>body{font-family:system-ui;background:#111;color:#eee;margin:32px}"
        "img{max-width:100%;image-rendering:auto}</style><h1>Final overlay fixed sample</h1>"
        "<p>Candidate predictions; not Ground Truth.</p><img src='contact_sheet.png'>",
        encoding="utf-8",
    )
    load_audit = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "model_state_loaded_strict": True,
        "tfr_loaded": use_tfr,
        "tfr_state_sha256": _state_hash(state.get("tfr_state_dict")),
        "change_head_loaded": change_head is not None,
        "change_head_state_sha256": _state_hash(state.get("change_head_state_dict")),
        "mti_selection": str(mti_path.resolve()),
        "mti_upstream_checkpoint_sha256": selection.get("upstream_checkpoint_sha256"),
        "mti_integration_mode": integration_mode,
        "mti_status": selection.get("mti_status"),
    }
    (output / "CHECKPOINT_LOAD_AUDIT.json").write_text(
        json.dumps(load_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "NUMERICAL_AUDIT.json").write_text(
        json.dumps(numerical, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "success",
        "integration_status": "overlay_candidate",
        "candidate_scope": "CWLD official Validation + FUSU 500-pair grouped sample domain",
        "scientific_status": "candidate_not_ground_truth",
        "inference_ready": False,
        "checkpoint_load_audit": load_audit,
        "outputs": list(arrays) + ["overlay_overview.png"],
        "automatic_visual_export": "pass",
        "human_visual_review": "pending",
        "full20_evidence": config.get("full20_prediction"),
        "rsbuilding_evidence": config.get("rsbuilding_probability"),
        "new_building_output": "unsupported_without_verified_building_semantics",
        "demolition_output": "unsupported_without_verified_building_semantics",
        "whu_test_reads": 0,
        "second_test_reads": 0,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        **guard,
    }
    (output / "MASTER_OVERLAY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest

