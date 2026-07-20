from __future__ import annotations

from pathlib import Path


def _torch_modules():
    import torch
    from torch import nn
    return torch, nn


class ConvBlock:
    """Factory wrapper kept import-safe when Torch is not installed."""

    @staticmethod
    def create(in_channels: int, out_channels: int):
        _, nn = _torch_modules()
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class MultiTaskSiameseUNetBase:
    pass


def _build_class():
    torch, nn = _torch_modules()

    class DecoderBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = ConvBlock.create(out_channels + skip_channels, out_channels)

        def forward(self, value, skip):
            value = self.up(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.block(torch.cat([value, skip], dim=1))

    class MultiTaskSiameseUNet(nn.Module):
        """Shared ResNet34 encoder with multi-scale difference/concatenation fusion."""

        def __init__(self, semantic_classes: int = 7, *, pretrained: bool = False, allow_download: bool = False,
                     weights_path: str | Path | None = None, dropout: float = 0.1):
            super().__init__()
            from torchvision.models import ResNet34_Weights, resnet34

            if pretrained and not allow_download and not weights_path:
                raise ValueError(
                    "pretrained=true requires weights_path or allow_download=true; automatic network download is disabled"
                )
            weights = ResNet34_Weights.DEFAULT if pretrained and allow_download and not weights_path else None
            backbone = resnet34(weights=weights)
            if weights_path:
                payload = torch.load(weights_path, map_location="cpu", weights_only=False)
                state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
                cleaned = {key.removeprefix("module.").removeprefix("encoder."): value for key, value in state.items()}
                missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
                if unexpected:
                    raise ValueError(f"Unexpected ResNet34 weight keys: {unexpected[:8]}")
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
            fused = [64 * 3, 64 * 3, 128 * 3, 256 * 3, 512 * 3]
            self.bottleneck = ConvBlock.create(fused[4], 512)
            self.dec4 = DecoderBlock(512, fused[3], 256)
            self.dec3 = DecoderBlock(256, fused[2], 128)
            self.dec2 = DecoderBlock(128, fused[1], 64)
            self.dec1 = DecoderBlock(64, fused[0], 48)
            self.full = nn.Sequential(nn.ConvTranspose2d(48, 32, 2, stride=2), nn.ReLU(inplace=True), nn.Dropout2d(dropout))
            self.binary_change_head = nn.Conv2d(32, 1, 1)
            self.new_building_head = nn.Conv2d(32, 1, 1)
            self.demolition_head = nn.Conv2d(32, 1, 1)
            self.semantic_t1_head = nn.Conv2d(32, semantic_classes, 1)
            self.semantic_t2_head = nn.Conv2d(32, semantic_classes, 1)
            self.uncertainty_head = nn.Conv2d(32, 1, 1)
            self.semantic_classes = semantic_classes

        def _encode(self, image):
            f0 = self.stem(image)
            f1 = self.layer1(self.pool(f0))
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return [f0, f1, f2, f3, f4]

        @staticmethod
        def _fuse(first, second):
            return [torch.cat([a, b, torch.abs(a - b)], dim=1) for a, b in zip(first, second)]

        def forward(self, t1, t2):
            if t1.shape != t2.shape:
                raise ValueError(f"T1/T2 tensor shape mismatch: {tuple(t1.shape)} != {tuple(t2.shape)}")
            fused = self._fuse(self._encode(t1), self._encode(t2))
            value = self.bottleneck(fused[4])
            value = self.dec4(value, fused[3])
            value = self.dec3(value, fused[2])
            value = self.dec2(value, fused[1])
            value = self.dec1(value, fused[0])
            value = self.full(value)
            if value.shape[-2:] != t1.shape[-2:]:
                value = nn.functional.interpolate(value, size=t1.shape[-2:], mode="bilinear", align_corners=False)
            return {
                "binary_change": self.binary_change_head(value),
                "new_building": self.new_building_head(value),
                "demolition": self.demolition_head(value),
                "semantic_t1": self.semantic_t1_head(value),
                "semantic_t2": self.semantic_t2_head(value),
                "uncertainty": self.uncertainty_head(value),
            }

    return MultiTaskSiameseUNet


MultiTaskSiameseUNet = _build_class()


def build_change_model(config: dict):
    values = config.get("change_model", config.get("model", config))
    backend = values.get("backend", values.get("architecture", "multitask_siamese_unet"))
    if backend == "changemamba":
        from .changemamba_adapter import build_changemamba
        return build_changemamba(values)
    if backend not in {"multitask_siamese_unet", "siamese_unet"}:
        raise ValueError(f"Unknown change model backend: {backend}")
    return MultiTaskSiameseUNet(
        semantic_classes=int(values.get("semantic_classes", 7)),
        pretrained=bool(values.get("pretrained", False)),
        allow_download=bool(values.get("allow_download", False)),
        weights_path=values.get("weights_path"),
        dropout=float(values.get("dropout", 0.1)),
    )
