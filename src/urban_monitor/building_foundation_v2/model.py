"""Small, shape-compatible static building model used by the v2 extension.

The existing RSBuilding checkpoint remains the frozen B0 external model.  This
module provides an auditable project-native refinement model with the same
building probability contract and an explicit future feature bundle.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BuildingFoundationNet(nn.Module):
    """Multi-head static building network.

    ``height_enabled`` is false by default because the current GloBFP audit
    does not prove height units or image pairing.  All outputs are dense and
    retain the input spatial shape.
    """

    def __init__(self, *, base_channels: int = 24, height_enabled: bool = False) -> None:
        super().__init__()
        self.height_enabled = bool(height_enabled)
        c = int(base_channels)
        self.encoder = nn.ModuleDict({
            "enc1": ConvBlock(3, c),
            "enc2": ConvBlock(c, c * 2),
            "enc3": ConvBlock(c * 2, c * 4),
            "bottleneck": ConvBlock(c * 4, c * 8),
        })
        self.decoder = nn.ModuleDict({
            "up3": nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2),
            "dec3": ConvBlock(c * 8, c * 4),
            "up2": nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2),
            "dec2": ConvBlock(c * 4, c * 2),
            "up1": nn.ConvTranspose2d(c * 2, c, 2, stride=2),
            "dec1": ConvBlock(c * 2, c),
        })
        self.mask_head = nn.Conv2d(c, 1, 1)
        self.boundary_head = nn.Conv2d(c, 1, 1)
        self.distance_head = nn.Conv2d(c, 1, 1)
        self.instance_head = nn.Conv2d(c, 1, 1)
        self.quality_head = nn.Conv2d(c, 1, 1)
        self.uncertainty_head = nn.Conv2d(c, 1, 1)
        self.height_head = nn.Conv2d(c, 1, 1) if self.height_enabled else None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.encoder["enc1"](x)
        e2 = self.encoder["enc2"](F.max_pool2d(e1, 2))
        e3 = self.encoder["enc3"](F.max_pool2d(e2, 2))
        b = self.encoder["bottleneck"](F.max_pool2d(e3, 2))
        return e1, e2, e3, b

    def decode(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        e1, e2, e3, b = features
        d3 = self.decoder["up3"](b)
        d3 = self.decoder["dec3"](torch.cat([d3, e3], dim=1))
        d2 = self.decoder["up2"](d3)
        d2 = self.decoder["dec2"](torch.cat([d2, e2], dim=1))
        d1 = self.decoder["up1"](d2)
        return self.decoder["dec1"](torch.cat([d1, e1], dim=1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.encode(x)
        decoded = self.decode(features)
        logits = {
            "mask_logits": self.mask_head(decoded),
            "boundary_logits": self.boundary_head(decoded),
            "distance": self.distance_head(decoded),
            "instance_logits": self.instance_head(decoded),
            "quality_logits": self.quality_head(decoded),
            "uncertainty_logits": self.uncertainty_head(decoded),
            "height": self.height_head(decoded) if self.height_head is not None else None,
            "feature_tensor": decoded,
        }
        outputs: dict[str, torch.Tensor | None] = dict(logits)
        outputs.update({
            "building_probability": torch.sigmoid(logits["mask_logits"]),
            "boundary_probability": torch.sigmoid(logits["boundary_logits"]),
            "instance_probability": torch.sigmoid(logits["instance_logits"]),
            "quality_score": torch.sigmoid(logits["quality_logits"]),
            "uncertainty_score": torch.sigmoid(logits["uncertainty_logits"]),
            "height_probability_or_value": logits["height"],
        })
        return outputs

    def export_feature_bundle(self, outputs: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
        return {
            "feature_tensor": outputs["feature_tensor"],
            "presence_probability": outputs["building_probability"],
            "boundary_probability": outputs["boundary_probability"],
            "height_probability_or_value": outputs.get("height_probability_or_value"),
            "quality_score": outputs["quality_score"],
            "uncertainty_score": outputs["uncertainty_score"],
        }


def output_contract(outputs: dict[str, torch.Tensor | None]) -> dict[str, Any]:
    result = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value).all().item())}
        elif value is None:
            result[key] = None
    return result
