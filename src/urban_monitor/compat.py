"""Explicit compatibility helpers for the supported Python/Torch/NumPy runtimes."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch


def grad_scaler_class() -> type:
    modern = getattr(getattr(torch, "amp", None), "GradScaler", None)
    return modern if modern is not None else torch.cuda.amp.GradScaler


def create_grad_scaler(device: str | torch.device, enabled: bool, **kwargs: Any):
    device_type = torch.device(device).type
    enabled = bool(enabled and device_type == "cuda")
    modern = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if modern is not None:
        return modern(device_type, enabled=enabled, **kwargs)
    return torch.cuda.amp.GradScaler(enabled=enabled, **kwargs)


def autocast_context(device: str | torch.device, enabled: bool, **kwargs: Any) -> ContextManager:
    device_type = torch.device(device).type
    enabled = bool(enabled and device_type == "cuda")
    if not enabled:
        return nullcontext()
    modern = getattr(torch, "autocast", None)
    if modern is not None:
        return modern(device_type=device_type, enabled=True, **kwargs)
    return torch.cuda.amp.autocast(enabled=True, **kwargs)


def trapezoid_integral(y: Any, x: Any = None, *, axis: int = -1) -> Any:
    function = getattr(np, "trapezoid", None) or np.trapz
    return function(y, axis=axis) if x is None else function(y, x=x, axis=axis)


def write_text(path: str | Path, text: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding=encoding, newline=newline) as handle:
        handle.write(text)
