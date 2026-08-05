from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.scaler_runtime_guard import (
    allow_empty_disabled_scaler_state,
)


def test_empty_disabled_scaler_state_is_skipped_and_method_restored():
    original = torch.amp.GradScaler.load_state_dict
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with allow_empty_disabled_scaler_state():
        assert scaler.load_state_dict({}) is None
    assert torch.amp.GradScaler.load_state_dict is original

