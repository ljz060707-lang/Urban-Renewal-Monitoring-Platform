from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.amp_safe_runtime import amp_safe_probability_bce


def test_probability_bce_is_finite_inside_autocast_and_restored():
    original = functional.binary_cross_entropy
    probability = torch.sigmoid(
        torch.tensor([[-3.0, 2.0]], dtype=torch.float32, requires_grad=True)
    )
    target = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    with amp_safe_probability_bce():
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = functional.binary_cross_entropy(probability, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert probability.grad_fn is not None
    assert functional.binary_cross_entropy is original


def test_probability_bce_expands_single_channel_target():
    probability = torch.full((2, 1, 4, 4), 0.5, requires_grad=True)
    target = torch.zeros((2, 4, 4))
    with amp_safe_probability_bce():
        loss = functional.binary_cross_entropy(probability, target)
    assert torch.isfinite(loss)
    loss.backward()

