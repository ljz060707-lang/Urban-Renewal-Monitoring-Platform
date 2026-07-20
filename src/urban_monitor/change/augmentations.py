from __future__ import annotations

import random


class ChangePairTransform:
    """Synchronized geometry for T1/T2/labels; independent mild photometry for images."""

    LABEL_KEYS = ("binary", "new", "demolition", "semantic_t1", "semantic_t2")

    def __init__(self, size: int = 256, *, training: bool = True, max_translation_px: int = 0,
                 brightness: float = 0.12, contrast: float = 0.12, noise_std: float = 0.01):
        self.size, self.training = int(size), bool(training)
        self.max_translation_px = max(0, int(max_translation_px))
        self.brightness, self.contrast, self.noise_std = float(brightness), float(contrast), float(noise_std)

    @staticmethod
    def _pad(value, size: int, fill: float = 0):
        import torch.nn.functional as functional
        height, width = value.shape[-2:]
        return functional.pad(value, (0, max(0, size - width), 0, max(0, size - height)), value=fill)

    @staticmethod
    def _photometric(image, brightness: float, contrast: float, noise_std: float):
        import torch
        gain = 1.0 + random.uniform(-contrast, contrast)
        bias = random.uniform(-brightness, brightness)
        value = image * gain + bias
        if noise_std > 0:
            value = value + torch.randn_like(value) * random.uniform(0, noise_std)
        return value.clamp(0, 1)

    def __call__(self, sample: dict):
        import torch
        values = {"t1": sample["t1"], "t2": sample["t2"], **{key: sample.get(key) for key in self.LABEL_KEYS}}
        for key, value in values.items():
            if value is not None:
                values[key] = self._pad(value, self.size)
        height, width = values["t1"].shape[-2:]
        top = random.randint(0, height - self.size) if self.training and height > self.size else max(0, (height - self.size) // 2)
        left = random.randint(0, width - self.size) if self.training and width > self.size else max(0, (width - self.size) // 2)
        for key, value in values.items():
            if value is not None:
                values[key] = value[..., top:top + self.size, left:left + self.size]
        if self.training:
            if random.random() < 0.5:
                values = {key: (torch.flip(value, (-1,)) if value is not None else None) for key, value in values.items()}
            if random.random() < 0.5:
                values = {key: (torch.flip(value, (-2,)) if value is not None else None) for key, value in values.items()}
            rotations = random.randrange(4)
            if rotations:
                values = {key: (torch.rot90(value, rotations, (-2, -1)) if value is not None else None) for key, value in values.items()}
            # A small synchronized translation is explicit and bounded; it never shifts T1 separately from T2.
            if self.max_translation_px:
                dy = random.randint(-self.max_translation_px, self.max_translation_px)
                dx = random.randint(-self.max_translation_px, self.max_translation_px)
                values = {key: (torch.roll(value, (dy, dx), (-2, -1)) if value is not None else None) for key, value in values.items()}
            values["t1"] = self._photometric(values["t1"], self.brightness, self.contrast, self.noise_std)
            values["t2"] = self._photometric(values["t2"], self.brightness, self.contrast, self.noise_std)
        sample.update(values)
        return sample
