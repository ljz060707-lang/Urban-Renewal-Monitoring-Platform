from __future__ import annotations

import sys
from pathlib import Path

from urban_monitor.compat import create_grad_scaler, grad_scaler_class

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.scaler_runtime_guard import (
    allow_empty_disabled_scaler_state,
)


def test_empty_disabled_scaler_state_is_skipped_and_method_restored():
    scaler_type = grad_scaler_class()
    original = scaler_type.load_state_dict
    scaler = create_grad_scaler("cpu", enabled=False)
    with allow_empty_disabled_scaler_state():
        assert scaler.load_state_dict({}) is None
    assert scaler_type.load_state_dict is original
