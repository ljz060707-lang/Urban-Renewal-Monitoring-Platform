from __future__ import annotations

import pytest


def _batch(torch):
    zero = torch.zeros(1, 4, 4)
    one = torch.ones(1, 4, 4)
    semantic = torch.zeros(4, 4, dtype=torch.long)
    return {
        "dataset_name": ["dsifn", "second"],
        "has_binary": torch.tensor([True, True]),
        "has_directional": torch.tensor([False, True]),
        "has_semantic": torch.tensor([False, True]),
        "binary": [zero, one], "new": [None, one], "demolition": [None, zero],
        "semantic_t1": [None, semantic], "semantic_t2": [None, semantic],
    }


def _outputs(torch):
    return {
        "binary_change": torch.zeros(2, 1, 4, 4, requires_grad=True),
        "new_building": torch.zeros(2, 1, 4, 4, requires_grad=True),
        "demolition": torch.zeros(2, 1, 4, 4, requires_grad=True),
        "semantic_t1": torch.zeros(2, 3, 4, 4, requires_grad=True),
        "semantic_t2": torch.zeros(2, 3, 4, 4, requires_grad=True),
        "uncertainty": torch.zeros(2, 1, 4, 4, requires_grad=True),
    }


def test_masked_loss_ignores_missing_directional_and_semantic_targets():
    torch = pytest.importorskip("torch")
    from urban_monitor.change.losses import MaskedMultiTaskLoss
    criterion = MaskedMultiTaskLoss({"loss_weights": {"binary": 0.5}})
    outputs = _outputs(torch)
    first, details = criterion(outputs, _batch(torch))
    outputs["new_building"] = outputs["new_building"].clone()
    outputs["new_building"][0] = 1000
    second, _ = criterion(outputs, _batch(torch))
    assert torch.allclose(first, second)
    assert "loss_dataset_dsifn" in details and "loss_dataset_second" in details
    first.backward()


def test_masked_loss_rejects_true_flag_with_missing_target():
    torch = pytest.importorskip("torch")
    from urban_monitor.change.losses import MaskedMultiTaskLoss
    batch = _batch(torch)
    batch["new"][1] = None
    with pytest.raises(ValueError, match="target is missing"):
        MaskedMultiTaskLoss()(_outputs(torch), batch)
