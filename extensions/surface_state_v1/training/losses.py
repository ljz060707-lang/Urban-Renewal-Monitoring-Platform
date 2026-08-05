from __future__ import annotations

import torch
from torch import nn


def multiclass_dice(logits, target, *, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0
    probabilities = logits.softmax(1)
    safe_target = target.masked_fill(~valid, 0)
    one_hot = nn.functional.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    valid_mask = valid[:, None].to(probabilities.dtype)
    intersection = (probabilities * one_hot * valid_mask).sum((0, 2, 3))
    denominator = ((probabilities + one_hot) * valid_mask).sum((0, 2, 3))
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def cwld_loss(logits, target, class_weights=None) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, target, weight=class_weights, ignore_index=255) + 0.75 * multiclass_dice(logits, target, num_classes=4)


def fusu_loss(logits_t1, logits_t2, target_t1, target_t2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    semantic = (
        nn.functional.cross_entropy(logits_t1, target_t1, ignore_index=255)
        + 0.75 * multiclass_dice(logits_t1, target_t1, num_classes=logits_t1.shape[1])
        + nn.functional.cross_entropy(logits_t2, target_t2, ignore_index=255)
        + 0.75 * multiclass_dice(logits_t2, target_t2, num_classes=logits_t2.shape[1])
    )
    valid = (target_t1 != 255) & (target_t2 != 255)
    ground_truth_change = (target_t1 != target_t2).to(logits_t1.dtype)
    predicted_change = 1 - (logits_t1.softmax(1) * logits_t2.softmax(1)).sum(1)
    change = nn.functional.binary_cross_entropy(predicted_change[valid].clamp(1e-6, 1 - 1e-6), ground_truth_change[valid]) if valid.any() else semantic * 0
    return semantic + 0.25 * change, semantic, change
