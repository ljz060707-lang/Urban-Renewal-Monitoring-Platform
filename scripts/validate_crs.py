#!/usr/bin/env python3
from __future__ import annotations

import argparse

import rasterio
from shapely.geometry import box

from urban_monitor.config import load_config
from urban_monitor.crs import ensure_metric_crs, select_local_utm_crs, validate_source_crs
from urban_monitor.manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate source CRS and select the AOI local UTM CRS.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_manifest(cfg["manifest"], cfg["data_root"])
    if not records:
        raise ValueError("Manifest contains no imagery")
    selected = None
    for record in records:
        with rasterio.open(record.image_path) as source:
            source_crs = validate_source_crs(source.crs)
            ensure_metric_crs(source_crs)
            candidate = select_local_utm_crs(
                box(*source.bounds), source_crs, max_width_km=float(cfg["crs"].get("warn_cross_zone_width_km", 100))
            )
            selected = selected or candidate
            print(f"{record.date} source={source_crs.to_string()} analysis={candidate.to_string()}")
    print(f"selected_analysis_crs={selected.to_string()}")


if __name__ == "__main__":
    main()
