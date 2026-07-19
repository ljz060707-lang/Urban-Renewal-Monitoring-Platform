#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.coco import build_coco
from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = [r for r in load_manifest(cfg["manifest"], cfg["data_root"]) if r.split in {"train", "val"}]
    output = Path(cfg["output_root"]) / "coco"
    prepare_output(output / "annotations.json", force=args.force)
    stats = build_coco(
        records,
        output,
        chip_size=int(cfg["chips"]["size"]),
        overlap=int(cfg["chips"]["overlap"]),
        min_polygon_pixels=int(cfg["chips"].get("min_polygon_pixels", 12)),
    )
    print(stats)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
