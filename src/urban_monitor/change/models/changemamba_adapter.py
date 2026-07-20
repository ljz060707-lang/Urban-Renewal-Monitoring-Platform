from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def build_changemamba(config: dict):
    if platform.system() != "Linux":
        raise RuntimeError("ChangeMamba backend is supported only on Linux/WSL; use multitask_siamese_unet on Windows")
    external = config.get("external_path")
    if not external:
        raise RuntimeError("ChangeMamba requires change_model.external_path pointing to an external checkout")
    root = Path(external).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ChangeMamba external_path does not exist: {root}")
    if not config.get("weights_path"):
        raise RuntimeError("ChangeMamba requires an explicit local weights_path")
    try:
        importlib.import_module("selective_scan")
    except ImportError as exc:
        raise RuntimeError("ChangeMamba requires selective_scan; follow the upstream Linux installation instructions") from exc
    sys.path.insert(0, str(root))
    module_name = config.get("factory_module")
    factory_name = config.get("factory_name", "build_model")
    if not module_name:
        raise RuntimeError("Set ChangeMamba factory_module and factory_name for the external checkout")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(config)
