from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

from .manifest import DateRecord


def _polygon_to_coco_segmentation(geom, inv_transform) -> list[list[float]]:
    def world_to_pixel(x, y, z=None):
        col, row = inv_transform * (x, y)
        return (col, row)

    pixel_geom = shapely_transform(world_to_pixel, geom)
    polygons = [pixel_geom] if pixel_geom.geom_type == "Polygon" else list(pixel_geom.geoms)
    segments: list[list[float]] = []
    for poly in polygons:
        coords = np.asarray(poly.exterior.coords[:-1], dtype=float)
        if len(coords) >= 3:
            segments.append(coords.reshape(-1).tolist())
    return segments


def build_coco(
    records: list[DateRecord],
    output_dir: str | Path,
    *,
    chip_size: int,
    overlap: int,
    min_polygon_pixels: int = 12,
) -> dict[str, int]:
    output = Path(output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    stride = chip_size - overlap

    for record in records:
        if record.label_path is None or not record.label_path.exists():
            continue
        labels = gpd.read_file(record.label_path)
        if labels.empty:
            continue
        with rasterio.open(record.image_path) as src:
            if labels.crs != src.crs:
                labels = labels.to_crs(src.crs)
            for row_off in range(0, src.height, stride):
                for col_off in range(0, src.width, stride):
                    width = min(chip_size, src.width - col_off)
                    height = min(chip_size, src.height - row_off)
                    if width < chip_size // 2 or height < chip_size // 2:
                        continue
                    window = Window(col_off, row_off, width, height)
                    chip_transform = src.window_transform(window)
                    chip_bounds = rasterio.windows.bounds(window, src.transform)
                    chip_box = box(*chip_bounds)
                    subset = labels[labels.geometry.intersects(chip_box)].copy()
                    if subset.empty:
                        continue
                    subset["geometry"] = subset.geometry.intersection(chip_box)
                    subset = subset[~subset.geometry.is_empty]
                    if subset.empty:
                        continue
                    data = src.read(window=window)
                    if data.shape[0] >= 3:
                        rgb = np.moveaxis(data[:3], 0, -1)
                    else:
                        rgb = np.repeat(np.moveaxis(data[:1], 0, -1), 3, axis=2)
                    if rgb.dtype != np.uint8:
                        lo, hi = np.nanpercentile(rgb, [2, 98])
                        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                    filename = f"{record.date}_{row_off:06d}_{col_off:06d}.png"
                    Image.fromarray(rgb).save(image_dir / filename)
                    images.append(
                        {
                            "id": image_id,
                            "file_name": f"images/{filename}",
                            "width": int(width),
                            "height": int(height),
                            "date": record.date,
                            "source_image": str(record.image_path),
                            "row_off": int(row_off),
                            "col_off": int(col_off),
                        }
                    )
                    inv = ~chip_transform
                    for _, feature in subset.iterrows():
                        geom = feature.geometry
                        if geom.area <= 0:
                            continue
                        segments = _polygon_to_coco_segmentation(geom, inv)
                        if not segments:
                            continue
                        xs = segments[0][0::2]
                        ys = segments[0][1::2]
                        x0, x1 = max(0.0, min(xs)), min(float(width), max(xs))
                        y0, y1 = max(0.0, min(ys)), min(float(height), max(ys))
                        pixel_area = geom.area / max(abs(chip_transform.a * chip_transform.e), 1e-12)
                        if pixel_area < min_polygon_pixels:
                            continue
                        annotations.append(
                            {
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "segmentation": segments,
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "area": float(pixel_area),
                                "iscrowd": 0,
                                "object_id": str(feature.get("object_id", "")),
                                "review_status": str(feature.get("review_status", "")),
                            }
                        )
                        ann_id += 1
                    image_id += 1

    payload = {
        "info": {"description": "Urban renewal phase-1 building instances"},
        "licenses": [],
        "categories": [{"id": 1, "name": "building", "supercategory": "structure"}],
        "images": images,
        "annotations": annotations,
    }
    with (output / "annotations.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return {"images": len(images), "annotations": len(annotations)}
