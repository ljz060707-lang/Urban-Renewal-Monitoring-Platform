from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


class UCSCodec:
    class_names = ("ucs_background", "ucs_exposed_area", "ucs_dustproof_net", "ucs_building")
    ignore_index = 255

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array)
        if values.ndim == 3:
            raise ValueError("UCS labels must be indexed/palette PNG, not RGB")
        output = values.astype(np.int64, copy=True)
        output[(output < 0) | (output > 3)] = self.ignore_index
        return output.astype(np.uint8)

    def encode_file(self, path) -> np.ndarray:
        with Image.open(path) as image:
            return self.encode_array(np.asarray(image))


def _resize_tensor(tensor: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    kwargs = {"size": (size, size), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor[None], **kwargs)[0]


class UCSManifestDataset(Dataset):
    """Read-only UCS adapter. Test pixels are blocked unless explicitly unlocked."""

    def __init__(self, manifest, image_size=256, rgb_bands=(1, 2, 3), allow_test=False):
        self.manifest = Path(manifest)
        self.rows = list(csv.DictReader(self.manifest.open(encoding="utf-8-sig")))
        if not self.rows:
            raise ValueError(f"empty UCS manifest: {self.manifest}")
        if not allow_test and any(row.get("split", "").casefold() == "test" for row in self.rows):
            raise RuntimeError("UCS Test pixel reads are locked")
        self.image_size = int(image_size)
        self.rgb_bands = tuple(int(value) for value in rgb_bands)
        self.allow_test = bool(allow_test)
        self.codec = UCSCodec()
        self.pixel_reads = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import rasterio

        row = self.rows[index]
        if row.get("split", "").casefold() == "test" and not self.allow_test:
            raise RuntimeError("UCS Test pixel reads are locked")
        try:
            with rasterio.open(row["image_path"]) as dataset:
                if max(self.rgb_bands) > dataset.count:
                    raise ValueError(f"rgb_bands={self.rgb_bands} exceed {dataset.count} bands")
                raw = dataset.read(self.rgb_bands).astype(np.float32)
            target = self.codec.encode_file(row["label_path"])
        except Exception as exc:
            raise RuntimeError(f"failed UCS sample_id={row.get('sample_id')}: {exc}") from exc
        self.pixel_reads += 1
        finite = np.isfinite(raw)
        if not finite.any():
            raise ValueError(f"UCS image contains no finite pixels: {row['image_path']}")
        lo = np.nanpercentile(raw[finite], 2.0)
        hi = np.nanpercentile(raw[finite], 98.0)
        raw = np.clip((raw - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        image = _resize_tensor(torch.from_numpy(raw), self.image_size, "bilinear")
        target_tensor = _resize_tensor(torch.from_numpy(target.astype(np.float32))[None], self.image_size, "nearest")[0].long()
        gray = (image * image.new_tensor([0.2989, 0.5870, 0.1140])[:, None, None]).sum(0, keepdim=True)
        return {
            "image": image,
            "texture": gray,
            "target": target_tensor,
            "valid_mask": target_tensor != self.codec.ignore_index,
            "dataset_name": "ucs",
            "sample_id": row["sample_id"],
            "scene_id": row.get("scene_id", row["sample_id"]),
            "split": row["split"],
            "metadata": {
                "source_path": row["image_path"],
                "rgb_bands": list(self.rgb_bands),
                "rgb_band_semantics_verified": False,
                "normalization": "per_image_percentile_2_98",
            },
        }


class CWLDConstructionDataset(Dataset):
    def __init__(self, manifest, image_size=256):
        from extensions.surface_state_v1.datasets.cwld import CWLDManifestDataset
        from extensions.surface_state_v1.datasets.cwld_codec import CWLDCodec

        self.delegate = CWLDManifestDataset(manifest, "train", CWLDCodec(), image_size=image_size)

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, index):
        item = self.delegate[index]
        image = item["image"]
        item.update({
            "texture": (image * image.new_tensor([0.2989, 0.5870, 0.1140])[:, None, None]).sum(0, keepdim=True),
            "valid_mask": item["target"] != 255,
            "dataset_name": "cwld",
            "scene_id": item["sample_id"],
            "metadata": {"codec": "audited_rgba_distance_codec"},
        })
        return item
