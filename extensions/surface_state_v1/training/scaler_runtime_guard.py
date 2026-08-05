"""Process-local GradScaler resume compatibility for CPU-prepared warm starts."""
from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def allow_empty_disabled_scaler_state():
    """Skip only empty scaler states produced by a disabled GradScaler."""

    original = torch.amp.GradScaler.load_state_dict

    def guarded(self, state_dict):
        if not state_dict:
            return None
        return original(self, state_dict)

    torch.amp.GradScaler.load_state_dict = guarded
    try:
        yield
    finally:
        torch.amp.GradScaler.load_state_dict = original

