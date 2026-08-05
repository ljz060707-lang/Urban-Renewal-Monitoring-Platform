from __future__ import annotations

from pathlib import Path
from copy import deepcopy

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
        values = deepcopy(config.get("datasets", {}).get(name, {}))
        if not values.get("enabled", False):
            continue
        root = Path(values.get("root", ""))
        if not root.is_absolute():
            root = Path(config["_config_path"]).parent.parent / root
        values["root"] = str(root)
        mapping = values.get("class_mapping_file")
        if mapping:
            mapping_path = Path(mapping)
            values["class_mapping_file"] = str(mapping_path if mapping_path.is_absolute() else Path(config["_config_path"]).parent / mapping_path)
        result.append(adapter_type(root, values))
    if not result:
        raise ValueError("No public change dataset is enabled")
    return result
