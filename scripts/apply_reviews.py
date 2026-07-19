#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.review import apply_reviews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = apply_reviews(
        Path(cfg["output_root"]) / "tracking" / "tracks_summary.gpkg",
        cfg["review"]["review_csv"],
        Path(cfg["output_root"]) / "tracking" / "tracks_reviewed.gpkg",
        lineage_path=Path(cfg["output_root"]) / "tracking" / "lineage_events.gpkg",
        review_version=str(cfg.get("review", {}).get("review_version", "0.3.0")),
    )
    print(result)


if __name__ == "__main__":
    main()
