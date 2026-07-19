#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from urban_monitor.config import load_config
from urban_monitor.inference import regularize_prediction_frame
from urban_monitor.manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Regularize existing building prediction polygons in-place or to a new folder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_dir = Path(args.input_dir or Path(cfg["output_root"]) / "predictions")
    output_dir = Path(args.output_dir or Path(cfg["output_root"]) / "predictions_regularized")
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in load_manifest(cfg["manifest"], cfg["data_root"]):
        src_path = input_dir / f"{record.date}.gpkg"
        frame = gpd.read_file(src_path)
        result = regularize_prediction_frame(frame, cfg.get("regularization", {}))
        dst_path = output_dir / f"{record.date}.gpkg"
        result.to_file(dst_path, driver="GPKG")
        print(dst_path)


if __name__ == "__main__":
    main()
