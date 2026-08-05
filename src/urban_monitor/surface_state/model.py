"""Surface-state model interface: background, vacant, construction, building, unknown."""

from __future__ import annotations


class SurfaceStateHead:
    class_names = ("background_or_other", "vacant_land", "construction_site", "building", "unknown")

    def __new__(cls, in_channels: int, num_classes: int = 5):
        import torch.nn as nn

        if num_classes != len(cls.class_names):
            raise ValueError(f"Surface-state canonical schema requires {len(cls.class_names)} classes")
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )
