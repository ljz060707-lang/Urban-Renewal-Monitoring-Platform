from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class DualBranchEncoder(nn.Module):
    """RGB spectral branch plus deterministic luminance texture branch."""

    def __init__(self, width: int = 24):
        super().__init__()
        self.rgb_branch = ConvBlock(3, width)
        self.texture_branch = ConvBlock(1, width // 2)
        self.fusion = ConvBlock(width + width // 2, width * 2)
        self.pool = nn.MaxPool2d(2)
        self.deep = ConvBlock(width * 2, width * 4)
        self.out_channels = (width * 2, width * 4)

    @staticmethod
    def deterministic_gray(rgb: torch.Tensor) -> torch.Tensor:
        weights = rgb.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        return (rgb * weights).sum(1, keepdim=True)

    def forward(self, rgb: torch.Tensor, texture: torch.Tensor | None = None) -> list[torch.Tensor]:
        if texture is None:
            texture = self.deterministic_gray(rgb)
        if texture.shape[1] != 1:
            raise ValueError("texture input must contain one grayscale channel")
        shallow = self.fusion(torch.cat([self.rgb_branch(rgb), self.texture_branch(texture)], dim=1))
        deep = self.deep(self.pool(shallow))
        return [shallow, deep]
