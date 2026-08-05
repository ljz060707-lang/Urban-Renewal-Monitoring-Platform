"""Narrow compatibility wrapper for the immutable prototype trainer."""
from __future__ import annotations

import torch.nn.functional as functional

from . import full_trainer


def train(*args, **kwargs):
    original = functional.binary_cross_entropy

    def channel_safe(input, target, *bce_args, **bce_kwargs):
        if input.ndim == target.ndim + 1 and input.shape[1] == 1:
            target = target[:, None]
        return original(input, target, *bce_args, **bce_kwargs)

    # ponytail: process-local shim because the workspace patch helper cannot
    # update the original trainer; remove when safe in-place patching returns.
    functional.binary_cross_entropy = channel_safe
    try:
        return full_trainer.train(*args, **kwargs)
    finally:
        functional.binary_cross_entropy = original
