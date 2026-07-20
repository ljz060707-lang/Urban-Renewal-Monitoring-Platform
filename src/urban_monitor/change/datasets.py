from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _rgb_tensor(path: str | Path):
    import torch

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _mask_tensor(path: str | Path | None, *, semantic: bool = False):
    if not path or not str(path):
        return None
    import torch

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if semantic:
        return torch.from_numpy(array.astype(np.int64, copy=True))
    return torch.from_numpy((array != 0).astype(np.float32, copy=True)).unsqueeze(0)


class PublicChangeDataset:
    def __init__(self, index: str | Path | pd.DataFrame, *, split: str | None = None, transform=None):
        self.frame = pd.read_csv(index, dtype=str).fillna("") if not isinstance(index, pd.DataFrame) else index.copy()
        if split:
            self.frame = self.frame[self.frame["split"] == split].copy()
        self.frame.reset_index(drop=True, inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        sample = {
            "sample_id": row["sample_id"], "dataset_name": row["dataset_name"], "split": row["split"],
            "t1": _rgb_tensor(row["t1_path"]), "t2": _rgb_tensor(row["t2_path"]),
            "binary": _mask_tensor(row.get("gt_cd_path")), "new": _mask_tensor(row.get("gt_new_path")),
            "demolition": _mask_tensor(row.get("gt_dem_path")),
            "semantic_t1": _mask_tensor(row.get("gt_t1_path"), semantic=True),
            "semantic_t2": _mask_tensor(row.get("gt_t2_path"), semantic=True),
            "has_binary": str(row.get("has_binary", "false")).casefold() == "true",
            "has_directional": str(row.get("has_directional", "false")).casefold() == "true",
            "has_semantic": str(row.get("has_semantic", "false")).casefold() == "true",
        }
        return self.transform(sample) if self.transform else sample


def change_collate(batch: list[dict]) -> dict:
    import torch

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "dataset_name": [item["dataset_name"] for item in batch],
        "t1": torch.stack([item["t1"] for item in batch]),
        "t2": torch.stack([item["t2"] for item in batch]),
        **{name: [item[name] for item in batch] for name in ("binary", "new", "demolition", "semantic_t1", "semantic_t2")},
        **{name: torch.tensor([item[name] for item in batch], dtype=torch.bool) for name in ("has_binary", "has_directional", "has_semantic")},
    }
