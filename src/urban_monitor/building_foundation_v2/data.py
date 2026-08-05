"""WHU Building data adapter with a hard guard against Test discovery."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F
from torch.utils.data import Dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _paired_files(root: Path, split: str) -> list[tuple[Path, Path]]:
    if split not in {"train", "validation"}:
        raise ValueError(f"WHU adapter refuses split={split!r}; Test pixels are locked")
    source = root / ("val" if split == "validation" else "train")
    image_dir, label_dir = source / "image", source / "label"
    images = {p.stem: p for p in image_dir.glob("*.tif")}
    labels = {p.stem: p for p in label_dir.glob("*.tif")}
    if not images or not labels:
        raise FileNotFoundError(f"WHU {split} image/label directories are empty")
    missing_images = sorted(set(labels) - set(images)); missing_labels = sorted(set(images) - set(labels))
    if missing_images or missing_labels:
        raise ValueError(f"WHU pairing mismatch split={split}: missing_images={missing_images[:5]} missing_labels={missing_labels[:5]}")
    return [(images[k], labels[k]) for k in sorted(images)]


def build_whu_manifest(root: Path, output: Path, *, train_limit: int | None = None, validation_limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, limit in (("train", train_limit), ("validation", validation_limit)):
        pairs = _paired_files(root, split)
        if limit is not None: pairs = pairs[:max(0, int(limit))]
        for image, label in pairs:
            rows.append({
                "sample_id": f"whu_{split}_{image.stem}",
                "scene_id": f"whu_{image.stem}",
                "base_id": image.stem,
                "group_id": f"whu_{split}",
                "dataset": "WHU_Building",
                "split": split,
                "image_path": str(image.resolve()),
                "label_path": str(label.resolve()),
                "image_sha256": sha256_file(image),
                "label_sha256": sha256_file(label),
                "crs": None,
                "spatial_isolation_strength": "official_dataset_split_no_geospatial_crs",
                "test_pixel_read": False,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sample_id"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows


class WHUBuildingDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, image_size: int = 256, train: bool = False) -> None:
        self.rows = rows
        self.image_size = int(image_size)
        self.train = bool(train)
        if any(str(r.get("split")) == "test" for r in rows):
            raise ValueError("WHU dataset refuses test rows")

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import rasterio  # type: ignore
        from rasterio.enums import Resampling  # type: ignore
        row = self.rows[index]
        with rasterio.open(row["image_path"]) as ds: image = ds.read(out_shape=(3, self.image_size, self.image_size), resampling=Resampling.bilinear)
        with rasterio.open(row["label_path"]) as ds: label = ds.read(1, out_shape=(self.image_size, self.image_size), resampling=Resampling.nearest)
        image = np.asarray(image, dtype="float32") / 255.0
        label = (np.asarray(label) > 0).astype("float32")
        if self.train and (index * 17) % 2 == 0:
            image = image[:, :, ::-1].copy(); label = label[:, ::-1].copy()
        target = torch.from_numpy(label[None])
        boundary = target.numpy()[0] - ndimage.binary_erosion(label, structure=np.ones((3, 3))).astype("float32")
        distance = ndimage.distance_transform_edt(label) - ndimage.distance_transform_edt(1 - label)
        return {
            "image": torch.from_numpy(image),
            "mask": target,
            "boundary": torch.from_numpy(boundary[None].astype("float32")),
            "distance": torch.from_numpy(np.tanh(distance / 10.0)[None].astype("float32")),
            "sample_id": row["sample_id"], "split": row["split"],
            "metadata": {key: ("" if value is None else str(value)) for key, value in row.items()},
        }
