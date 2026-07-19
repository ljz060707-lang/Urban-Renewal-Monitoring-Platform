from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.transforms.functional import pil_to_tensor


class CocoBuildingDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        with (self.root / "annotations.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.images = payload["images"]
        self.by_image: dict[int, list[dict]] = {}
        for ann in payload["annotations"]:
            self.by_image.setdefault(int(ann["image_id"]), []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        info = self.images[index]
        image = Image.open(self.root / info["file_name"]).convert("RGB")
        tensor = pil_to_tensor(image).float() / 255.0
        anns = self.by_image.get(int(info["id"]), [])
        boxes, labels, masks, areas, crowds = [], [], [], [], []
        height, width = int(info["height"]), int(info["width"])
        for ann in anns:
            seg = ann["segmentation"]
            rles = mask_utils.frPyObjects(seg, height, width)
            merged = mask_utils.merge(rles)
            mask = mask_utils.decode(merged).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask.max(axis=2)
            boxes.append(ann["bbox"])
            x, y, w, h = ann["bbox"]
            boxes[-1] = [x, y, x + w, y + h]
            labels.append(int(ann["category_id"]))
            masks.append(mask)
            areas.append(float(ann["area"]))
            crowds.append(int(ann.get("iscrowd", 0)))
        if boxes:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
                "image_id": torch.tensor([int(info["id"])]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, height, width), dtype=torch.uint8),
                "image_id": torch.tensor([int(info["id"])]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
