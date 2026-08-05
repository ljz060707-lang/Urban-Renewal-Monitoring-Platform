from __future__ import annotations

from torch import nn


class NativeSegmentationHead(nn.Sequential):
    def __init__(self, channels: int, classes: int):
        super().__init__(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, classes, 1))


class UnifiedEvidenceProjector(nn.Module):
    def forward(self, native_probabilities, dataset: str):
        from urban_renewal.construction.ontology import evidence_projection

        return evidence_projection(dataset, native_probabilities)
