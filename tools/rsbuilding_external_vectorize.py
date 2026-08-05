from __future__ import annotations

"""Vectorize audited external RSBuilding masks with the project GIS runtime.

The frozen RSBuilding inference environment intentionally contains the model
runtime but not Fiona.  This small inference-only adapter consumes the already
audited masks and writes the same ``building_objects.gpkg`` contract required
by the downstream external Construction Activity adapter.  It never opens
training or test data.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import rasterio
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform


SOURCE_MODEL = "rsbuilding_swin_t_whu_iter2000"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def local_utm_epsg(bounds) -> int:
    left, bottom, right, top = bounds
    lon = (left + right) / 2
    lat = (bottom + top) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def main() -> int:
    parser = argparse.ArgumentParser(description="Vectorize audited external RSBuilding masks")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-area-m2", type=float, default=10.0)
    args = parser.parse_args()

    geometry_root = Path(args.output_root).resolve()
    numerical_path = geometry_root / "TONGZHOU_GEOMETRY_NUMERICAL_AUDIT.json"
    if not numerical_path.exists():
        raise FileNotFoundError(f"Numerical audit is required before vectorization: {numerical_path}")
    numerical = json.loads(numerical_path.read_text(encoding="utf-8"))
    if numerical.get("geometry_numerical_status") != "pass":
        raise RuntimeError("Numerical audit is not pass; vectorization refused.")

    scenes_path = geometry_root / "TONGZHOU_SCENES.json"
    scenes_payload = json.loads(scenes_path.read_text(encoding="utf-8"))
    if scenes_payload.get("scene_discovery_status") != "pass":
        raise RuntimeError("Scene discovery is not pass; vectorization refused.")

    results = []
    for scene in scenes_payload["scenes"]:
        date = str(scene["scene_date"])
        output = geometry_root / date
        mask_path = output / "building_mask.tif"
        if not scene.get("crs"):
            skipped = {"status": "skipped_no_crs", "reason": "No CRS; geographic area would be fabricated.", "scene_date": date}
            write_json(output / "VECTORIZATION_SKIPPED.json", skipped)
            results.append(skipped)
            continue
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        gpkg = output / "building_objects.gpkg"
        temporary = output / "building_objects.tmp.gpkg"
        for path in (gpkg, temporary):
            if path.exists():
                path.unlink()

        with rasterio.open(mask_path) as dataset:
            mask = dataset.read(1)
            valid = mask == 1
            metric_crs = f"EPSG:{local_utm_epsg(dataset.bounds)}"
            transformer = Transformer.from_crs(dataset.crs, metric_crs, always_xy=True)
            rows = []
            count = 0
            for geometry_mapping, value in shapes(mask, mask=valid, transform=dataset.transform, connectivity=8):
                if int(value) != 1:
                    continue
                geometry = shape(geometry_mapping)
                if not geometry.is_valid:
                    geometry = geometry.buffer(0)
                if geometry.is_empty:
                    continue
                parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry] if geometry.geom_type == "Polygon" else []
                for part in parts:
                    projected = shapely_transform(transformer.transform, part)
                    if projected.area < float(args.min_area_m2):
                        continue
                    count += 1
                    rows.append(
                        {
                            "object_id": f"{date}_{count:08d}",
                            "scene_date": date,
                            "state": "building_candidate",
                            "area_m2": float(projected.area),
                            "perimeter_m": float(projected.length),
                            "source_model": SOURCE_MODEL,
                            "scientific_status": "candidate_not_ground_truth",
                            "geometry": part,
                        }
                    )
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=dataset.crs)
        frame.to_file(temporary, layer="buildings", driver="GPKG", engine="pyogrio")
        os.replace(temporary, gpkg)
        result = {
            "status": "success",
            "scene_date": date,
            "gpkg": str(gpkg.resolve()),
            "object_count": count,
            "layer": "buildings",
            "minimum_area_m2": float(args.min_area_m2),
            "scientific_status": "candidate_not_ground_truth",
        }
        write_json(output / "VECTORIZATION_RECEIPT.json", result)
        results.append(result)

    payload = {"status": "success", "scenes": results, "runtime": "project_gis_pyogrio", "created_at": utc_now()}
    write_json(geometry_root / "TONGZHOU_VECTORIZATION_SUMMARY.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
