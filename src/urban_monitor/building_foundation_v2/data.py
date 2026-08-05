"""WHU Building data adapter with a hard guard against Test discovery."""

from __future__ import annotations

import csv
import hashlib
import json
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


def load_label_contract(contract: dict[str, Any] | Path | str | None) -> dict[str, Any]:
    """Return an explicit mask polarity contract.

    The historical adapter used ``label > 0`` unconditionally.  That is kept
    as the backwards-compatible default for old scratch fixtures, but formal
    WHU recovery runs must pass ``SELECTED_WHU_LABEL_CONTRACT.json`` so a local
    inverted 0/1 mask cannot silently become all-positive supervision.
    """
    if contract is None:
        return {"label_polarity": "nonzero_is_building", "building_values": [1], "background_values": [0], "source": "legacy_default"}
    if isinstance(contract, (str, Path)):
        path = Path(contract)
        contract = json.loads(path.read_text(encoding="utf-8"))
        contract = dict(contract)
        contract.setdefault("source", str(path.resolve()))
    if not isinstance(contract, dict):
        raise TypeError("label_contract must be a mapping or JSON path")
    values = dict(contract)
    building = values.get("building_values")
    background = values.get("background_values")
    if not building or not background:
        polarity = values.get("label_polarity")
        if polarity == "zero_is_building":
            building, background = [0], [1]
        elif polarity == "nonzero_is_building":
            building, background = [1], [0]
        else:
            raise ValueError("label contract must specify building_values/background_values or a known label_polarity")
    values["building_values"] = [int(v) for v in building]
    values["background_values"] = [int(v) for v in background]
    values.setdefault("label_polarity", "explicit_values")
    return values


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
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        image_size: int = 256,
        train: bool = False,
        label_contract: dict[str, Any] | Path | str | None = None,
        native_normalization: bool = False,
        duplicate_temporal_input: bool = False,
    ) -> None:
        self.rows = rows
        self.image_size = int(image_size)
        self.train = bool(train)
        self.label_contract = load_label_contract(label_contract)
        self.native_normalization = bool(native_normalization)
        self.duplicate_temporal_input = bool(duplicate_temporal_input)
        if any(str(r.get("split")) == "test" for r in rows):
            raise ValueError("WHU dataset refuses test rows")

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import rasterio  # type: ignore
        from rasterio.enums import Resampling  # type: ignore
        row = self.rows[index]
        with rasterio.open(row["image_path"]) as ds: image = ds.read(out_shape=(3, self.image_size, self.image_size), resampling=Resampling.bilinear)
        with rasterio.open(row["label_path"]) as ds: label = ds.read(1, out_shape=(self.image_size, self.image_size), resampling=Resampling.nearest)
        image = np.asarray(image, dtype="float32")
        if self.native_normalization:
            # Native RSBuilding consumes two identical RGB observations as a
            # six-channel tensor and normalizes after BGR->RGB conversion.
            # Keep this opt-in: the project scratch U-Net still expects [0,1].
            if image.shape[0] != 3:
                raise ValueError(f"WHU adapter expects 3 RGB bands, got {image.shape[0]}")
            image = image[::-1].copy() if self.label_contract.get("native_bgr_to_rgb", True) else image
            if self.duplicate_temporal_input:
                image = np.concatenate([image, image], axis=0)
            default_mean = [123.675, 116.28, 103.53] * (2 if self.duplicate_temporal_input else 1)
            default_std = [58.395, 57.12, 57.375] * (2 if self.duplicate_temporal_input else 1)
            mean = np.asarray(self.label_contract.get("native_mean", default_mean), dtype="float32")[:, None, None]
            std = np.asarray(self.label_contract.get("native_std", default_std), dtype="float32")[:, None, None]
            if mean.shape[0] != image.shape[0] or std.shape[0] != image.shape[0]:
                raise ValueError(
                    "native normalization channel mismatch: "
                    f"image={image.shape[0]} mean={mean.shape[0]} std={std.shape[0]}"
                )
            image = (image - mean) / np.maximum(std, 1e-6)
        else:
            image = image / 255.0
        values = np.asarray(label)
        building_values = np.asarray(self.label_contract["building_values"])
        label = np.isin(values, building_values).astype("float32")
        allowed = np.concatenate([building_values, np.asarray(self.label_contract["background_values"])])
        unknown = np.setdiff1d(np.unique(values), np.unique(allowed))
        if unknown.size:
            raise ValueError(f"label contains values outside contract: {unknown.tolist()}")
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
            "metadata": {
                **{key: ("" if value is None else str(value)) for key, value in row.items()},
                "label_polarity": str(self.label_contract.get("label_polarity")),
                "label_contract_source": str(self.label_contract.get("source", "inline")),
                "native_normalization": str(self.native_normalization).lower(),
            },
        }
