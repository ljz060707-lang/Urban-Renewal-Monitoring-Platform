from __future__ import annotations

from pathlib import Path

import yaml

from .adapters import DSIFNAdapter, S2LookingAdapter, SECONDAdapter


ADAPTERS = {"s2looking": S2LookingAdapter, "second": SECONDAdapter, "dsifn": DSIFNAdapter}


def load_change_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = str(config_path)
    return config


def enabled_adapters(config: dict):
    result = []
    for name, adapter_type in ADAPTERS.items():
        values = config.get("datasets", {}).get(name, {})
        if not values.get("enabled", False):
            continue
        root = Path(values.get("root", ""))
        if not root.is_absolute():
            root = Path(config["_config_path"]).parent.parent / root
        result.append(adapter_type(root, values))
    if not result:
        raise ValueError("No public change dataset is enabled")
    return result
