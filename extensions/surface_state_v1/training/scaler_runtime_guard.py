"""Process-local GradScaler resume compatibility for CPU-prepared warm starts."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from urban_monitor.compat import grad_scaler_class


@contextmanager
def allow_empty_disabled_scaler_state():
    """Skip only empty scaler states produced by a disabled GradScaler."""

    scaler_type = grad_scaler_class()
    original = scaler_type.load_state_dict

    def guarded(self, state_dict):
        if not state_dict:
            return None
        return original(self, state_dict)

    scaler_type.load_state_dict = guarded
    try:
        yield
    finally:
        scaler_type.load_state_dict = original
