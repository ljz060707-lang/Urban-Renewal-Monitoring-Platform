from __future__ import annotations

import random


class ChangePairTransform:
    """Synchronized geometry for T1/T2/labels; independent mild photometry for images."""

    LABEL_KEYS = ("binary", "new", "demolition", "semantic_t1", "semantic_t2")

    def __init__(self, size: int = 256, *, training: bool = True, max_translation_px: int = 0,
                 brightness: float = 0.12, contrast: float = 0.12, noise_std: float = 0.01,
                 gamma: tuple[float, float] = (1.0, 1.0), gaussian_blur_probability: float = 0.0,
                 temporal_alignment_jitter_px: int = 0, change_aware: dict | None = None):
        self.size, self.training = int(size), bool(training)
        self.max_translation_px = max(0, int(max_translation_px))
        self.brightness, self.contrast, self.noise_std = float(brightness), float(contrast), float(noise_std)
        self.gamma = (float(gamma[0]), float(gamma[1]))
        self.gaussian_blur_probability = float(gaussian_blur_probability)
        self.temporal_alignment_jitter_px = max(0, int(temporal_alignment_jitter_px))
        self.change_aware = change_aware or {}

    @staticmethod
    def _pad(value, size: int, fill: float = 0):
        import torch.nn.functional as functional
        height, width = value.shape[-2:]
        return functional.pad(value, (0, max(0, size - width), 0, max(0, size - height)), value=fill)

    @staticmethod
    def _photometric(image, brightness: float, contrast: float, noise_std: float, gamma: tuple[float, float],
                     blur_probability: float):
        import torch
        gain = 1.0 + random.uniform(-contrast, contrast)
        bias = random.uniform(-brightness, brightness)
        value = image * gain + bias
        if noise_std > 0:
            value = value + torch.randn_like(value) * random.uniform(0, noise_std)
        value = value.clamp(0, 1).pow(random.uniform(*gamma))
        if blur_probability > 0 and random.random() < blur_probability:
            value = torch.nn.functional.avg_pool2d(value.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        return value.clamp(0, 1)

    def __call__(self, sample: dict):
        import torch
        values = {"t1": sample["t1"], "t2": sample["t2"], **{key: sample.get(key) for key in self.LABEL_KEYS}}
        for key, value in values.items():
            if value is not None:
                values[key] = self._pad(value, self.size)
        fn_values = {name: sample.get(f"fn_{name}") for name in ("binary", "new", "demolition")}
        fn_values = {name: (self._pad(value, self.size) if value is not None else None) for name, value in fn_values.items()}
        height, width = values["t1"].shape[-2:]
        target_head = str(sample.get("target_head") or "").casefold()
        difficult_negative = target_head == "difficult_negative"
        if difficult_negative:
            target_head = "demolition"
        if target_head == "both":
            candidates = [name for name in ("new", "demolition") if values.get(name) is not None and bool((values[name] > 0).any())]
            target_head = random.choice(candidates) if candidates else "binary"
        if target_head not in {"binary", "new", "demolition"}:
            candidates = [name for name in ("binary", "new", "demolition") if values.get(name) is not None and bool((values[name] > 0).any())]
            target_head = random.choice(candidates) if candidates else "binary"
        target = sample.get("fp_demolition") if difficult_negative else fn_values.get(target_head)
        if target is None:
            target = values.get(target_head)
        aware = self.training and bool(self.change_aware.get("enabled", False)) and target is not None and bool((target > 0).any())
        mode = "random"
        if aware:
            centered = float(self.change_aware.get("target_coverage_probability", 0.70))
            boundary = float(self.change_aware.get("boundary_probability", 0.20))
            draw = random.random()
            mode = "target" if draw < centered else ("boundary" if draw < centered + boundary else "random")
        if aware and mode != "random":
            coordinates = torch.nonzero(target[0] > 0 if target.ndim == 3 else target > 0, as_tuple=False)
            y, x = coordinates[random.randrange(len(coordinates))].tolist()
            jitter = int(self.change_aware.get("jitter_px", self.size // 4))
            if mode == "boundary":
                jitter = max(jitter, self.size // 2)
            y += random.randint(-jitter, jitter)
            x += random.randint(-jitter, jitter)
            top = min(max(0, y - self.size // 2), max(0, height - self.size))
            left = min(max(0, x - self.size // 2), max(0, width - self.size))
        else:
            top = random.randint(0, height - self.size) if self.training and height > self.size else max(0, (height - self.size) // 2)
            left = random.randint(0, width - self.size) if self.training and width > self.size else max(0, (width - self.size) // 2)
        for key, value in values.items():
            if value is not None:
                values[key] = value[..., top:top + self.size, left:left + self.size]
        for key, value in fn_values.items():
            if value is not None:
                fn_values[key] = value[..., top:top + self.size, left:left + self.size]
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
            if self.temporal_alignment_jitter_px:
                dy = random.randint(-self.temporal_alignment_jitter_px, self.temporal_alignment_jitter_px)
                dx = random.randint(-self.temporal_alignment_jitter_px, self.temporal_alignment_jitter_px)
                values["t2"] = torch.roll(values["t2"], (dy, dx), (-2, -1))
            values["t1"] = self._photometric(
                values["t1"], self.brightness, self.contrast, self.noise_std,
                self.gamma, self.gaussian_blur_probability,
            )
            values["t2"] = self._photometric(
                values["t2"], self.brightness, self.contrast, self.noise_std,
                self.gamma, self.gaussian_blur_probability,
            )
        sample.update(values)
        sample.update({f"fn_{key}": value for key, value in fn_values.items()})
        target_after = sample.get(f"fn_{target_head}") if sample.get(f"fn_{target_head}") is not None else sample.get(target_head)
        sample["crop_target_head"] = target_head
        sample["crop_mode"] = mode
        sample["hard_patch_hit"] = bool(target_after is not None and (target_after > 0).any())
        return sample
