from __future__ import annotations

from pathlib import Path
import hashlib

import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def _load_compatible(model: torch.nn.Module, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {str(key).replace("module.", ""): value for key, value in state.items()}
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    if not compatible:
        raise ValueError(f"No compatible model weights found in {path}")
    model.load_state_dict(compatible, strict=False)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(
    num_classes: int = 2,
    checkpoint: str | Path | None = None,
    *,
    pretrained: bool = False,
    weights_path: str | Path | None = None,
    allow_pretrained_download: bool = False,
    min_size: int = 512,
    max_size: int = 1024,
    freeze_backbone: bool = False,
) -> torch.nn.Module:
    """Build a transferable Mask R-CNN and optionally load a local checkpoint.

    `num_classes` includes background. The COCO heads are explicitly replaced,
    because torchvision's pretrained detector otherwise keeps its original class
    count.
    """
    weights = None
    if pretrained and not weights_path and not checkpoint:
        filename = Path(MaskRCNN_ResNet50_FPN_Weights.DEFAULT.url).name
        cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not cached.exists() and not allow_pretrained_download:
            raise FileNotFoundError(
                "Pretrained torchvision weights are not cached. Set model.pretrained=false for random initialization, "
                "provide model.weights_path, or explicitly set model.allow_pretrained_download=true."
            )
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, hidden, num_classes)

    source = checkpoint or weights_path
    if source:
        if not Path(source).is_file():
            raise FileNotFoundError(f"Model weights do not exist: {source}")
        _load_compatible(model, source)
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    return model
