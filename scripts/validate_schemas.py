#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.schema import validate_annotation_schema, validate_manifest_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v0.3 manifest and available annotation layers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg["manifest"], dtype=str, encoding="utf-8-sig").fillna("")
    warnings = validate_manifest_schema(manifest, require_v030=True)
    validated = 0
    for record in load_manifest(cfg["manifest"], cfg["data_root"]):
        if record.label_path and record.label_path.exists():
            validate_annotation_schema(gpd.read_file(record.label_path))
            validated += 1
    print(f"manifest=ok annotations={validated} warnings={len(warnings)}")


if __name__ == "__main__":
    main()
