from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes: int = 2, checkpoint: str | Path | None = None) -> torch.nn.Module:
    """Build a transferable Mask R-CNN and optionally load a local checkpoint.

    `num_classes` includes background. The COCO heads are explicitly replaced,
    because torchvision's pretrained detector otherwise keeps its original class
    count.
    """
    weights = None if checkpoint else MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, hidden, num_classes)

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {str(k).replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return model
