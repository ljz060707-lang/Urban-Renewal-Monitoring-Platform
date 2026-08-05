from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import time

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
    def __init__(self, index: str | Path | pd.DataFrame, *, split: str | None = None, transform=None,
                 cache_mode: str = "none", cache_max_gb: float = 0.0):
        self.frame = pd.read_csv(index, dtype=str).fillna("") if not isinstance(index, pd.DataFrame) else index.copy()
        if split:
            self.frame = self.frame[self.frame["split"] == split].copy()
        self.frame.reset_index(drop=True, inplace=True)
        self.transform = transform
        self.cache_mode = str(cache_mode)
        self.cache_max_bytes = int(float(cache_max_gb) * 1024**3)
        self._cache: OrderedDict[tuple[str, bool], object] = OrderedDict()
        self._cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_mode not in {"none", "ram"}:
            raise ValueError("training.cache_mode currently supports only none or ram")

    def __len__(self):
        return len(self.frame)

    def _load(self, path, *, semantic=False, rgb=False):
        key = (str(path), bool(semantic))
        if self.cache_mode == "ram" and key in self._cache:
            self.cache_hits += 1
            value = self._cache.pop(key); self._cache[key] = value
            return value.clone() if value is not None else None
        self.cache_misses += 1
        value = _rgb_tensor(path) if rgb else _mask_tensor(path, semantic=semantic)
        if self.cache_mode == "ram" and self.cache_max_bytes > 0:
            size = int(value.nelement() * value.element_size()) if value is not None else 0
            while self._cache and self._cache_bytes + size > self.cache_max_bytes:
                _, removed = self._cache.popitem(last=False)
                if removed is not None:
                    self._cache_bytes -= int(removed.nelement() * removed.element_size())
            if size <= self.cache_max_bytes:
                self._cache[key] = value
                self._cache_bytes += size
        return value.clone() if value is not None else None

    def __getitem__(self, index: int):
        started = time.perf_counter()
        sampling_category = ""
        if isinstance(index, tuple):
            index, sampling_category = index
        row = self.frame.iloc[index]
        original_target_head = str(row.get("target_head", ""))
        if sampling_category:
            target_head = ("demolition" if sampling_category in {"demolition_positive", "demolition_hard"}
                           else ("new" if sampling_category == "new_positive" else
                                 ("difficult_negative" if sampling_category == "difficult_negative" else "")))
            hard_sample = sampling_category in {"demolition_hard", "new_positive", "demolition_positive"}
        else:
            target_head = original_target_head
            hard_sample = bool(str(row.get("hard_reason", "")).strip() or float(row.get("sample_weight") or 1.0) > 1.0)
        decode_started = time.perf_counter()
        sample = {
            "sample_id": row["sample_id"], "dataset_name": row["dataset_name"], "split": row["split"],
            "t1": self._load(row["t1_path"], rgb=True), "t2": self._load(row["t2_path"], rgb=True),
            "binary": self._load(row.get("gt_cd_path")), "new": self._load(row.get("gt_new_path")),
            "demolition": self._load(row.get("gt_dem_path")),
            "semantic_t1": self._load(row.get("gt_t1_path"), semantic=True),
            "semantic_t2": self._load(row.get("gt_t2_path"), semantic=True),
            "fn_binary": self._load(row.get("false_negative_binary_mask_path")),
            "fn_new": self._load(row.get("false_negative_new_mask_path")),
            "fn_demolition": self._load(row.get("false_negative_demolition_mask_path")),
            "fp_binary": self._load(row.get("false_positive_binary_mask_path")),
            "fp_new": self._load(row.get("false_positive_new_mask_path")),
            "fp_demolition": self._load(row.get("false_positive_demolition_mask_path")),
            "has_binary": str(row.get("has_binary", "false")).casefold() == "true",
            "has_directional": str(row.get("has_directional", "false")).casefold() == "true",
            "has_semantic": str(row.get("has_semantic", "false")).casefold() == "true",
            "target_head": target_head,
            "original_target_head": original_target_head,
            "sampling_category": sampling_category or "ordinary",
            "hard_sample": hard_sample,
        }
        decode_seconds = time.perf_counter() - decode_started
        augment_started = time.perf_counter()
        sample = self.transform(sample) if self.transform else sample
        sample["_timing"] = {
            "image_decode_time": decode_seconds,
            "augmentation_time": time.perf_counter() - augment_started,
            "data_load_time": time.perf_counter() - started,
        }
        return sample


def change_collate(batch: list[dict]) -> dict:
    import torch

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "dataset_name": [item["dataset_name"] for item in batch],
        "t1": torch.stack([item["t1"] for item in batch]),
        "t2": torch.stack([item["t2"] for item in batch]),
        **{name: [item[name] for item in batch] for name in ("binary", "new", "demolition", "semantic_t1", "semantic_t2")},
        **{name: torch.tensor([item[name] for item in batch], dtype=torch.bool) for name in ("has_binary", "has_directional", "has_semantic")},
        "hard_sample": torch.tensor([bool(item.get("hard_sample", False)) for item in batch], dtype=torch.bool),
        "hard_patch_hit": torch.tensor([bool(item.get("hard_patch_hit", False)) for item in batch], dtype=torch.bool),
        "crop_target_head": [item.get("crop_target_head", "") for item in batch],
        "hard_target_head": [item.get("original_target_head", "") for item in batch],
        "sampling_category": [item.get("sampling_category", "ordinary") for item in batch],
        "_timing": [item.get("_timing", {}) for item in batch],
    }
