from __future__ import annotations

from torch import nn
from torch.nn import functional as F

from .boundary_head import BoundaryHead
from .encoder import DualBranchEncoder
from .native_heads import NativeSegmentationHead, UnifiedEvidenceProjector
from .uncertainty_head import UncertaintyHead


class ConstructionElementNet(nn.Module):
    """Shared dual-branch encoder with isolated UCS/CWLD native heads."""

    def __init__(self, width: int = 24):
        super().__init__()
        self.shared_encoder = DualBranchEncoder(width)
        channels = width * 2
        self.decoder = nn.Sequential(nn.Conv2d(width * 4, channels, 1), nn.ReLU(inplace=True))
        self.ucs_native_head = NativeSegmentationHead(channels, 4)
        self.cwld_native_head = NativeSegmentationHead(channels, 4)
        self.unified_evidence_projector = UnifiedEvidenceProjector()
        self.boundary_head = BoundaryHead(channels)
        self.uncertainty_head = UncertaintyHead(channels)

    def features(self, rgb, texture=None):
        shallow, deep = self.shared_encoder(rgb, texture)
        decoded = F.interpolate(self.decoder(deep), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return decoded + shallow

    def forward(self, rgb, dataset: str, texture=None):
        dataset = dataset.casefold()
        if dataset not in {"ucs", "cwld"}:
            raise ValueError("dataset must be 'ucs' or 'cwld'")
        features = self.features(rgb, texture)
        head = self.ucs_native_head if dataset == "ucs" else self.cwld_native_head
        logits = head(features)
        probabilities = logits.softmax(1)
        evidence, presence = self.unified_evidence_projector(probabilities, dataset)
        return {
            f"{dataset}_logits": logits,
            f"{dataset}_probabilities": probabilities,
            "evidence_probabilities": evidence,
            "evidence_presence_mask": presence,
            "boundary_logits": self.boundary_head(features),
            "uncertainty_logits": self.uncertainty_head(features),
            "features": features,
            "input_mode": "rgb_plus_deterministic_gray" if texture is None else "rgb_plus_external_gray",
        }
