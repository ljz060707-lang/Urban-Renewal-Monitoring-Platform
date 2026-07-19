#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely import affinity
from shapely.geometry import box


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    for folder in ("images", "labels", "quality"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    dates = ["2022-04-01", "2023-08-01", "2024-01-15", "2025-04-01"]
    off_nadir = [2.0, 18.0, 6.0, 12.0]
    seasons = ["spring", "summer", "winter", "spring"]
    shifts = [(0, 0), (10, 2), (-3, -2), (5, 1)]
    transform = from_origin(500000, 3200000, 1, 1)
    crs = "EPSG:32650"
    stable_base = box(500080, 3199820, 500140, 3199880)
    demolished = box(500180, 3199700, 500230, 3199760)
    new_small = box(500300, 3199800, 500325, 3199825)
    new_full = box(500295, 3199795, 500345, 3199845)
    old_a = box(500360, 3199680, 500385, 3199710)
    old_b = box(500390, 3199680, 500415, 3199710)
    redevelopment = box(500356, 3199675, 500420, 3199720)
    manifest = []
    for idx, date_value in enumerate(dates):
        arr = np.zeros((3, 512, 512), dtype=np.uint8)
        arr[:] = 80
        arr[1] = 95
        stable = affinity.translate(stable_base, xoff=shifts[idx][0], yoff=shifts[idx][1])
        geoms = [("stable01", stable)]
        if idx <= 1:
            geoms.append(("demo01", demolished))
            geoms.extend([("old_a", old_a), ("old_b", old_b)])
        if idx == 2:
            geoms.append(("new01", new_small))
            geoms.append(("redevelopment", redevelopment))
        if idx >= 3:
            geoms.append(("new01", new_full))
            geoms.append(("redevelopment", redevelopment))
        with rasterio.open(
            root / "images" / f"{date_value}.tif",
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=3,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(arr)
        with rasterio.open(
            root / "quality" / f"{date_value}.tif",
            "w",
            driver="GTiff",
            height=512,
            width=512,
            count=2,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(np.ones((512, 512), dtype=np.uint8), 1)
            dst.write(np.zeros((512, 512), dtype=np.uint8), 2)
            dst.set_band_description(1, "usable")
            dst.set_band_description(2, "quality_reason")
        labels = gpd.GeoDataFrame(
            [
                {
                    "object_id": oid,
                    "class_name": "building",
                    "geometry_role": "visible_roof",
                    "review_status": "reviewed",
                    "source_date": date_value,
                    "notes": "synthetic stress case",
                    "tracking_geometry_wkt": "",
                    "schema_version": "0.3.0",
                    "geometry": geom,
                }
                for oid, geom in geoms
            ],
            geometry="geometry",
            crs=crs,
        )
        labels.to_file(root / "labels" / f"{date_value}.gpkg", driver="GPKG")
        manifest.append(
            {
                "date": date_value,
                "image_path": f"images/{date_value}.tif",
                "label_path": f"labels/{date_value}.gpkg",
                "quality_mask_path": f"quality/{date_value}.tif",
                "split": "train" if idx < 3 else "val",
                "source": "synthetic_off_nadir_demo",
                "sensor": "LocalOrthoSynthetic",
                "gsd_m": 1.0,
                "off_nadir_deg": off_nadir[idx],
                "sun_elevation_deg": [52, 68, 24, 50][idx],
                "season": seasons[idx],
                "quality_notes": "synthetic smoke test",
                "provider_mask_path": "",
                "manual_mask_path": "",
                "crs": crs,
                "coordinate_encoding": "projected",
                "acquisition_time": f"{date_value}T10:00:00+08:00",
                "registration_status": "passed",
                "notes": "合成压力测试：屋顶位移、拆除、新建与 merge lineage",
            }
        )
    pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False, encoding="utf-8-sig")
    print(root / "manifest.csv")


if __name__ == "__main__":
    main()
