from __future__ import annotations

import torch

from surface_state_v1.models.surface_state_net import SurfaceStateNet
from surface_state_v1.training.cwld_finalization import _class_gate
from surface_state_v1.training.experiment_control import _merge_joint_model


def test_joint_warmstart_uses_selected_shared_and_dataset_heads():
    cwld_model, fusu_model = SurfaceStateNet(), SurfaceStateNet()
    with torch.no_grad():
        for parameter in cwld_model.encoder.parameters():
            parameter.fill_(1)
        for parameter in fusu_model.encoder.parameters():
            parameter.fill_(2)
        for parameter in cwld_model.cwld_decoder_head.parameters():
            parameter.fill_(3)
        for parameter in fusu_model.fusu_decoder_head.parameters():
            parameter.fill_(4)
    merged = _merge_joint_model(
        {"model_state_dict": cwld_model.state_dict()},
        {"model_state_dict": fusu_model.state_dict()},
        "fusu",
    )
    assert next(merged.encoder.parameters()).mean().item() == 2
    assert next(merged.cwld_decoder_head.parameters()).mean().item() == 3
    assert next(merged.fusu_decoder_head.parameters()).mean().item() == 4


def test_cwld_class_gate_rejects_single_class_collapse():
    metrics = {
        "pixel_accuracy": 0.9,
        "confusion_matrix": [
            [90, 0, 0, 0],
            [10, 0, 0, 0],
            [10, 0, 0, 0],
            [10, 0, 0, 0],
        ],
        "per_class": {
            name: {"iou": 0.1, "f1": 0.1, "precision": 0.1, "recall": 0.1}
            for name in (
                "background",
                "vacant_land_candidate",
                "engineering_facility_candidate",
                "construction_waste",
            )
        },
    }
    gate = _class_gate(metrics)
    assert gate["status"] == "rejected_class_collapse"
    assert len(gate["non_background_classes_without_prediction"]) == 3
