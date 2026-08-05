"""Process-local AMP compatibility for probability-space BCE losses.

The immutable prototype trainer computes two change losses from probabilities.
PyTorch intentionally rejects ``binary_cross_entropy`` inside CUDA autocast.
This wrapper keeps the existing loss definition, but evaluates BCE in float32
with autocast disabled.  The patch is limited to the training process.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.nn import functional


def _amp_safe_bce(input, target, *args, **kwargs):
    if input.ndim == target.ndim + 1 and input.shape[1] == 1:
        target = target[:, None]
    device_type = input.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        return _ORIGINAL_BCE(
            input.float().clamp(1e-6, 1.0 - 1e-6),
            target.float(),
            *args,
            **kwargs,
        )


_ORIGINAL_BCE = functional.binary_cross_entropy


@contextmanager
def amp_safe_probability_bce():
    """Install the narrow compatibility patch for one training process."""

    previous = functional.binary_cross_entropy
    functional.binary_cross_entropy = _amp_safe_bce
    try:
        yield
    finally:
        functional.binary_cross_entropy = previous

