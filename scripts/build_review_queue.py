#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.review import build_review_queue
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepare_output(Path(cfg["output_root"]) / "review" / "index.html", force=args.force)
    path = build_review_queue(
        Path(cfg["output_root"]) / "tracking" / "tracks_summary.gpkg",
        load_manifest(cfg["manifest"], cfg["data_root"]),
        Path(cfg["output_root"]) / "review",
        padding_m=float(cfg["review"].get("chip_padding_m", 25.0)),
        long_path=Path(cfg["output_root"]) / "tracking" / "tracks_long.gpkg",
        lineage_path=Path(cfg["output_root"]) / "tracking" / "lineage_events.gpkg",
    )
    print(path)


if __name__ == "__main__":
    main()
