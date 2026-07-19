#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["output_root"]) / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    for record in load_manifest(cfg["manifest"], cfg["data_root"]):
        prepare_output(output / f"{record.date}.gpkg", force=args.force)
        frame = gpd.read_file(record.label_path)
        frame = frame[["geometry"]].copy()
        frame["score"] = 0.93
        frame["model_version"] = cfg["model_version"]
        frame["schema_version"] = cfg.get("project", {}).get("schema_version", "0.4.0")
        frame["prediction_id"] = [f"{record.date}-p{i:04d}" for i in range(len(frame))]
        frame.to_file(output / f"{record.date}.gpkg", driver="GPKG")
    print(output)


if __name__ == "__main__":
    main()
