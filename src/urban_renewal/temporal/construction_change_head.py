from __future__ import annotations

import torch
from torch import nn


class ConstructionChangeHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(channels * 3, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, 2, 1))

    def forward(self, refined_features, temporal_valid_mask=None):
        batch, timesteps, channels, height, width = refined_features.shape
        outputs = []
        for index in range(timesteps - 1):
            left, right = refined_features[:, index], refined_features[:, index + 1]
            logits = self.head(torch.cat([left, right, (right - left).abs()], 1))
            if temporal_valid_mask is not None:
                pair_valid = temporal_valid_mask[:, index] & temporal_valid_mask[:, index + 1]
                logits = logits.masked_fill(~pair_valid[:, None, None, None], 0)
            outputs.append(logits)
        logits = torch.stack(outputs, 1)
        return {"adjacent_change_logits": logits, "adjacent_change_probabilities": logits.softmax(2)[:, :, 1]}
