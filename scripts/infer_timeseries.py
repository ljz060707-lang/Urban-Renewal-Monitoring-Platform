#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.inference import infer_raster
from urban_monitor.manifest import load_manifest
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    t = cfg["training"]
    checkpoint = t.get("checkpoint") or str(Path(cfg["output_root"]) / "training" / "checkpoints" / "best.pth")
    for record in load_manifest(cfg["manifest"], cfg["data_root"]):
        output = prepare_output(Path(cfg["output_root"]) / "predictions" / f"{record.date}.gpkg", force=args.force)
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
            rgb_bands=cfg.get("data", {}).get("rgb_bands", [1, 2, 3]),
            nms_iou_threshold=float(cfg.get("inference", {}).get("nms_iou_threshold", 0.4)),
            source_date=record.date,
            min_size=int(cfg.get("model", {}).get("min_size", 512)),
            max_size=int(cfg.get("model", {}).get("max_size", 1024)),
        )
        print(output)


if __name__ == "__main__":
    main()
