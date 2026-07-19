from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    defaults: dict[str, Any] = {
        "project": {"name": "urban-renewal-phase1", "version": "0.3.0", "schema_version": "0.3.0"},
        "crs": {
            "source_policy": "preserve",
            "analysis_strategy": "local_utm",
            "exchange_crs": "EPSG:4326",
            "display_crs": "EPSG:3857",
            "reject_missing_crs": True,
            "reject_geographic_metric_operations": True,
            "reject_web_mercator_metric_operations": True,
            "warn_cross_zone_width_km": 100,
        },
        "provenance": {"write_run_manifest": True, "hash_inputs": True, "record_git_commit": True},
    }
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
            warnings.warn(f"Legacy config: using v0.3 default for {key}", stacklevel=2)
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base is None:
        return path.resolve()
    return (Path(base) / path).resolve()
