from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    safe = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    valid_f = valid[:, None].to(logits.dtype)
    prob = logits.softmax(1) * valid_f
    one_hot = one_hot * valid_f
    numerator = 2.0 * (prob * one_hot).sum((0, 2, 3)) + 1e-6
    denominator = (prob + one_hot).sum((0, 2, 3)) + 1e-6
    return 1.0 - (numerator / denominator).mean()


def boundary_target(mask: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    valid = mask != ignore_index
    safe = mask.masked_fill(~valid, 0).float()[:, None]
    maximum = F.max_pool2d(safe, 3, 1, 1)
    minimum = -F.max_pool2d(-safe, 3, 1, 1)
    return ((maximum != minimum) & valid[:, None]).float()


class PartialLabelLoss(nn.Module):
    """Dataset-specific supervision; absent native classes receive no loss."""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, outputs: dict, target: torch.Tensor, dataset: str) -> dict[str, torch.Tensor]:
        key = f"{dataset.casefold()}_logits"
        if key not in outputs:
            raise KeyError(f"missing dataset-specific logits: {key}")
        logits = outputs[key]
        ce = F.cross_entropy(logits, target.long(), ignore_index=self.ignore_index)
        dice = multiclass_dice_loss(logits, target, self.ignore_index)
        b_target = boundary_target(target, self.ignore_index)
        valid = (target != self.ignore_index)[:, None]
        bce = F.binary_cross_entropy_with_logits(
            outputs["boundary_logits"][valid], b_target[valid]
        ) if bool(valid.any()) else logits.sum() * 0.0
        uncertainty_target = (~valid).float()
        uncertainty = F.binary_cross_entropy_with_logits(
            outputs["uncertainty_logits"], uncertainty_target
        )
        b_weight = 0.25 if dataset.casefold() == "ucs" else 0.20
        u_weight = 0.10 if dataset.casefold() == "ucs" else 0.0
        total = ce + 0.75 * dice + b_weight * bce + u_weight * uncertainty
        return {"total": total, "ce": ce, "dice": dice, "boundary": bce, "uncertainty": uncertainty}
