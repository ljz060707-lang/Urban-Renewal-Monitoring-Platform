"""Native label codecs built only from observed Train label values."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def inspect_label(path: str | Path) -> dict:
    """Read a label without assigning scientific semantics to its values."""
    with Image.open(path) as image:
        palette = image.getpalette()
        array = np.asarray(image)
        mode = image.mode
    if array.ndim == 2:
        values = np.unique(array)
        return {"mode": mode, "shape": list(array.shape), "encoding": "scalar", "values": [int(v) for v in values], "palette": bool(palette)}
    if array.ndim == 3 and array.shape[-1] >= 3:
        rgb = array[..., :3].reshape(-1, 3)
        values = np.unique(rgb, axis=0)
        return {"mode": mode, "shape": list(array.shape), "encoding": "rgb", "values": [list(map(int, row)) for row in values], "palette": bool(palette)}
    raise ValueError(f"Unsupported label shape {array.shape} for {path}")


class NativeLabelCodec:
    """Deterministic scalar/RGB label codec; class names deliberately remain native."""

    def __init__(self, encoding: str, values: list, *, ignore_index: int | None = None, native_prefix: str = "native_class"):
        self.encoding = encoding
        self.values = [tuple(value) if isinstance(value, list) else int(value) for value in values]
        self.ignore_index = ignore_index
        self.native_prefix = native_prefix
        self.index = {value: idx for idx, value in enumerate(self.values)}

    @property
    def class_names(self) -> list[str]:
        return [f"{self.native_prefix}::{value}" for value in self.values]

    @property
    def num_classes(self) -> int:
        return len(self.values)

    def encode(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            value = np.asarray(image)
        if self.encoding == "scalar":
            if value.ndim != 2:
                raise ValueError(f"Expected scalar label for {path}, got {value.shape}")
            out = np.full(value.shape, 255, dtype=np.int64)
            for raw, index in self.index.items():
                out[value == raw] = index
            if self.ignore_index is not None:
                out[value == self.ignore_index] = 255
            unknown = (out == 255) & (value != self.ignore_index)
        else:
            if value.ndim != 3 or value.shape[-1] < 3:
                raise ValueError(f"Expected RGB label for {path}, got {value.shape}")
            rgb = value[..., :3]
            out = np.full(rgb.shape[:2], 255, dtype=np.int64)
            for raw, index in self.index.items():
                out[np.all(rgb == raw, axis=-1)] = index
            unknown = out == 255
        if np.any(unknown):
            raise ValueError(f"Unmapped observed label values in {path}")
        return out

    def as_dict(self) -> dict:
        return {"encoding": self.encoding, "values": [list(v) if isinstance(v, tuple) else v for v in self.values], "ignore_index": self.ignore_index, "native_class_names": self.class_names}


def codec_from_labels(paths: Iterable[str | Path], *, native_prefix: str, ignore_index: int | None = None) -> NativeLabelCodec:
    values: set = set()
    encoding: str | None = None
    for path in paths:
        info = inspect_label(path)
        if encoding is None:
            encoding = info["encoding"]
        elif encoding != info["encoding"]:
            raise ValueError(f"Mixed label encodings: {encoding} and {info['encoding']}")
        values.update(tuple(value) if isinstance(value, list) else value for value in info["values"])
    if not values or encoding is None:
        raise ValueError("No label values found")
    if encoding == "scalar" and ignore_index is not None:
        values.discard(ignore_index)
    ordered = sorted(values)
    return NativeLabelCodec(encoding, [list(v) if isinstance(v, tuple) else v for v in ordered], ignore_index=ignore_index, native_prefix=native_prefix)
