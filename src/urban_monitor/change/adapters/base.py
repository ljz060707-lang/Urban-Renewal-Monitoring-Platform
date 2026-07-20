from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..schema import ChangeSample, validate_sample


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def split_directories(root: Path) -> list[tuple[str, Path]]:
    aliases = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    found = [(aliases[path.name.casefold()], path) for path in root.iterdir() if path.is_dir() and path.name.casefold() in aliases]
    return sorted(found) or [("train", root)]


def named_directory(parent: Path, names: Iterable[str], *, required: bool = True) -> Path | None:
    lookup = {path.name.casefold(): path for path in parent.iterdir() if path.is_dir()}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    if required:
        raise FileNotFoundError(f"Missing one of directories {list(names)} under {parent}")
    return None


def files_by_stem(folder: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f"Duplicate sample stem in {folder}: {result[key]}, {path}")
            result[key] = path
    return result


def require_same_stems(named: dict[str, dict[str, Path]]) -> list[str]:
    if not named:
        return []
    union = set().union(*(set(values) for values in named.values()))
    problems = {name: sorted(union - set(values)) for name, values in named.items() if union - set(values)}
    if problems:
        detail = "; ".join(f"{name} missing {values[:10]}" for name, values in problems.items())
        raise ValueError(f"T1/T2/label filename mismatch: {detail}")
    return sorted(union)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def read_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        elif np.all(array == array[..., :1]):
            array = array[..., 0]
        else:
            raise ValueError(f"Binary label must be single-channel: {path}")
    if array.ndim != 2:
        raise ValueError(f"Binary label must be 2D: {path}")
    return (array != 0).astype(np.uint8)


def write_mask(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


class ChangeAdapter(ABC):
    dataset_name: str

    def __init__(self, root: str | Path, config: dict | None = None):
        self.root = Path(root)
        self.config = config or {}
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.dataset_name} root does not exist: {self.root}")

    @abstractmethod
    def discover(self) -> list[ChangeSample]:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, sample: ChangeSample, output_dir: Path) -> ChangeSample:
        raise NotImplementedError

    def validate(self, samples: list[ChangeSample]) -> None:
        if not samples:
            raise ValueError(f"{self.dataset_name}: no samples found under {self.root}")
        for sample in samples:
            validate_sample(sample)
            sizes = [image_size(sample.t1_path), image_size(sample.t2_path)]
            for path in (sample.gt_cd_path, sample.gt_new_path, sample.gt_dem_path, sample.gt_t1_path, sample.gt_t2_path):
                if path:
                    sizes.append(image_size(path))
            if len(set(sizes)) != 1:
                raise ValueError(f"{sample.sample_id}: image/label size mismatch: {sizes}")
