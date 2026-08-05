"""Shared runtime normalization for non-training change-model consumers.

Training resolves an ``auto`` batch size by probing and persists the selected
value in ``logs/environment_<stage>.json``.  Evaluation consumers must reuse
that recorded value rather than trying a second probe or calling ``int('auto')``.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import platform


def _integer(value, field: str, *, default: int, minimum: int = 0, allow_auto: bool = False) -> int:
    if value is None:
        value = default
    if value == "auto" and allow_auto:
        return default
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer configuration: {field}={value!r}") from exc
    if resolved < minimum:
        raise ValueError(f"Invalid integer configuration: {field}={value!r} (must be >= {minimum})")
    return resolved


def _float(value, field: str, *, default: float, minimum: float = 0.0, allow_auto: bool = False) -> float:
    if value is None:
        value = default
    if value == "auto" and allow_auto:
        return default
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric configuration: {field}={value!r}") from exc
    if resolved < minimum:
        raise ValueError(f"Invalid numeric configuration: {field}={value!r} (must be >= {minimum})")
    return resolved


def _recorded_batch_size(run: Path) -> int:
    """Use the same persisted fallback as optimization/profile evaluation."""
    environment = run / "logs" / "environment_directional.json"
    if not environment.is_file():
        return 8
    try:
        value = json.loads(environment.read_text(encoding="utf-8")).get("batch_size", 8)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid integer configuration: logs.environment_directional.batch_size={str(environment)!r}") from exc
    return _integer(value, "logs.environment_directional.batch_size", default=8, minimum=1)


def _worker_count(value) -> int:
    if value == "auto":
        # Keep the training policy: Windows defaults to zero workers; other
        # platforms use a conservative fraction of available CPU cores.
        return 0 if platform.system() == "Windows" else min(4, max(0, (os.cpu_count() or 2) // 2))
    return _integer(value, "training.num_workers", default=0, minimum=0)


def resolve_evaluation_runtime_config(config: dict, run: str | Path) -> dict:
    """Normalize fields used by evaluation before any image dataset is built.

    This preserves the established training/evaluation defaults: batch auto
    uses the recorded directional environment (or 8), worker auto follows the
    training platform rule, patch auto falls back to the long-standing 256px
    evaluation default, and cache-size auto resolves to its existing 0GiB.
    """
    copied = deepcopy(config)
    values = copied.setdefault("training", {})
    root = Path(run)
    batch_value = values.get("resolved_batch_size", values.get("batch_size", 8))
    if batch_value == "auto":
        batch_value = _recorded_batch_size(root)
    values["resolved_batch_size"] = _integer(batch_value, "training.batch_size", default=8, minimum=1)
    values["batch_size"] = values["resolved_batch_size"]
    values["resolved_num_workers"] = _worker_count(values.get("num_workers", "auto"))
    values["patch_size"] = _integer(values.get("patch_size", 256), "training.patch_size", default=256, minimum=1, allow_auto=True)
    values["cache_max_gb"] = _float(values.get("cache_max_gb", 0.0), "training.cache_max_gb", default=0.0, minimum=0.0, allow_auto=True)
    cache_mode = values.get("cache_mode", "none")
    if cache_mode == "auto":
        cache_mode = "none"
    if cache_mode not in {"none", "ram"}:
        raise ValueError(f"Invalid configuration: training.cache_mode={cache_mode!r}")
    values["cache_mode"] = cache_mode
    return copied
