#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.review import apply_reviews
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    destination = prepare_output(Path(cfg["output_root"]) / "tracking" / "tracks_reviewed.gpkg", force=args.force)
    result = apply_reviews(
        Path(cfg["output_root"]) / "tracking" / "tracks_summary.gpkg",
        cfg["review"]["review_csv"],
        destination,
        lineage_path=Path(cfg["output_root"]) / "tracking" / "lineage_events.gpkg",
        review_version=str(cfg.get("review", {}).get("review_version", "0.4.0")),
        confidence_weights=cfg.get("confidence", {}).get("weights"),
    )
    print(result)


if __name__ == "__main__":
    main()
