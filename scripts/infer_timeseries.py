#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.inference import infer_raster
from urban_monitor.manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    t = cfg["training"]
    checkpoint = t.get("checkpoint") or str(Path(cfg["output_root"]) / "checkpoints" / "latest.pth")
    for record in load_manifest(cfg["manifest"], cfg["data_root"]):
        output = Path(cfg["output_root"]) / "predictions" / f"{record.date}.gpkg"
        infer_raster(
            record.image_path,
            output,
            checkpoint=checkpoint,
            chip_size=int(cfg["chips"]["size"]),
            overlap=int(cfg["chips"]["overlap"]),
            score_threshold=float(t["score_threshold"]),
            mask_threshold=float(t["mask_threshold"]),
            model_version=str(cfg["model_version"]),
            regularization=cfg.get("regularization", {}),
        )
        print(output)


if __name__ == "__main__":
    main()
