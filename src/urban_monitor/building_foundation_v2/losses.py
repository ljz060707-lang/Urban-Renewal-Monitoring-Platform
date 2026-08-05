"""Multitask losses with finite-value protection."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))).mean()


def multitask_loss(outputs: dict[str, torch.Tensor | None], batch: dict[str, torch.Tensor], *, height_enabled: bool = False) -> dict[str, torch.Tensor]:
    mask = batch["mask"]
    boundary = batch["boundary"]
    distance = batch["distance"]
    parts = {
        "mask": F.binary_cross_entropy_with_logits(outputs["mask_logits"], mask) + dice_loss(outputs["mask_logits"], mask),
        "boundary": F.binary_cross_entropy_with_logits(outputs["boundary_logits"], boundary),
        "distance": F.smooth_l1_loss(outputs["distance"], distance),
        "instance": F.binary_cross_entropy_with_logits(outputs["instance_logits"], mask),
    }
    total = parts["mask"] + 0.25 * parts["boundary"] + 0.10 * parts["distance"] + 0.10 * parts["instance"]
    if height_enabled and outputs.get("height") is not None and batch.get("height") is not None:
        valid = batch.get("height_valid", torch.ones_like(batch["height"]))
        height_loss = F.smooth_l1_loss(outputs["height"] * valid, batch["height"] * valid)
        parts["height"] = height_loss; total = total + 0.1 * height_loss
    parts["total"] = total
    if not torch.isfinite(total): raise FloatingPointError("Building Foundation v2 loss became non-finite")
    return parts
