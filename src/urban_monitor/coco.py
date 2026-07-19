from __future__ import annotations

import json
import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

from .manifest import DateRecord
from .splitting import split_for_tile


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


def _stable_int(*values: object) -> int:
    digest = hashlib.sha1("|".join(str(value) for value in values).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _rgb_window(source, window: Window, rgb_bands: list[int]) -> np.ndarray:
    if any(band < 1 or band > source.count for band in rgb_bands):
        raise ValueError(f"RGB bands {rgb_bands} exceed image band count {source.count}: {source.name}")
    data = source.read(rgb_bands, window=window)
    rgb = np.moveaxis(data, 0, -1)
    if rgb.dtype != np.uint8:
        valid = rgb[np.isfinite(rgb)]
        if valid.size:
            lo, hi = np.percentile(valid, [2, 98])
            rgb = np.clip((rgb - lo) / max(float(hi - lo), 1e-6) * 255, 0, 255)
        rgb = np.nan_to_num(rgb).astype(np.uint8)
    return rgb


def _rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _tile_annotations(
    labels: gpd.GeoDataFrame,
    tile_box,
    transform,
    width: int,
    height: int,
    *,
    image_id: int,
    min_visibility: float,
) -> list[dict]:
    indices = list(labels.sindex.query(tile_box, predicate="intersects")) if not labels.empty else []
    rows = []
    for index in indices:
        feature = labels.iloc[index]
        original = feature.geometry
        if original is None or original.is_empty or original.area <= 0:
            continue
        clipped = original.intersection(tile_box)
        if clipped.is_empty or clipped.area <= 0:
            continue
        visibility = float(clipped.area / original.area)
        if visibility < min_visibility:
            continue
        mask = rasterize(
            [(mapping(clipped), 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=False,
            dtype="uint8",
        )
        ys, xs = np.where(mask > 0)
        if not len(xs):
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        object_id = str(feature.get("object_id", "")) or hashlib.sha1(original.normalize().wkb).hexdigest()[:16]
        rows.append(
            {
                "id": _stable_int(image_id, object_id, hashlib.sha1(clipped.normalize().wkb).hexdigest()),
                "image_id": image_id,
                "category_id": 1,
                "segmentation": _rle(mask),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": float(mask.sum()),
                "iscrowd": 0,
                "object_id": object_id,
                "visibility": visibility,
                "source_date": str(feature.get("source_date", "")),
            }
        )
    return sorted(rows, key=lambda item: item["id"])


def _write_previews(coco_root: Path, payloads: dict[str, dict], count: int = 20) -> None:
    from pycocotools import mask as mask_utils

    candidates = []
    for split, payload in payloads.items():
        by_image: dict[int, list[dict]] = {}
        for annotation in payload["annotations"]:
            by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        for info in payload["images"]:
            candidates.append((split, info, by_image.get(int(info["id"]), [])))
    if not candidates:
        return
    candidates.sort(key=lambda item: _stable_int(item[0], item[1]["id"]))
    preview_dir = coco_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        split, info, annotations = candidates[index % len(candidates)]
        image = Image.open(coco_root / info["file_name"]).convert("RGBA")
        overlay = np.zeros((int(info["height"]), int(info["width"]), 4), dtype=np.uint8)
        for annotation in annotations:
            encoded = dict(annotation["segmentation"])
            encoded["counts"] = encoded["counts"].encode("ascii")
            mask = mask_utils.decode(encoded).astype(bool)
            overlay[mask] = (255, 0, 0, 90)
        rendered = Image.alpha_composite(image, Image.fromarray(overlay, "RGBA"))
        rendered.convert("RGB").save(preview_dir / f"preview_{index + 1:03d}_{split}.jpg", quality=90)


def build_spatial_coco(
    prepared_dir: str | Path,
    blocks: gpd.GeoDataFrame,
    dates: list[str],
    *,
    tile_size: int = 512,
    overlap: int = 64,
    min_object_visibility: float = 0.30,
    include_empty_tiles: bool = True,
    max_empty_to_positive_ratio: float = 1.0,
    rgb_bands: list[int] | tuple[int, ...] = (1, 2, 3),
) -> dict[str, object]:
    """Window-read aligned GeoTIFFs and write deterministic train/val COCO."""
    prepared = Path(prepared_dir)
    output = prepared / "coco"
    output.mkdir(parents=True, exist_ok=True)
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tiling.overlap must be smaller than tile_size")
    candidates: dict[str, list[dict]] = {"train": [], "val": []}
    for date in sorted(dates):
        image_path = prepared / "images" / f"{date}.tif"
        label_path = prepared / "labels" / f"{date}.gpkg"
        labels = gpd.read_file(label_path)
        with rasterio.open(image_path) as source:
            if labels.crs != source.crs:
                labels = labels.to_crs(source.crs)
            for row_off in range(0, source.height, stride):
                for col_off in range(0, source.width, stride):
                    width = min(tile_size, source.width - col_off)
                    height = min(tile_size, source.height - row_off)
                    if width < tile_size // 2 or height < tile_size // 2:
                        continue
                    window = Window(col_off, row_off, width, height)
                    transform = source.window_transform(window)
                    tile_geometry = box(*rasterio.windows.bounds(window, source.transform))
                    assigned = split_for_tile(tile_geometry, blocks)
                    if assigned is None:
                        continue
                    split, block_id = assigned
                    image_id = _stable_int(date, row_off, col_off, split)
                    annotations = _tile_annotations(
                        labels,
                        tile_geometry,
                        transform,
                        width,
                        height,
                        image_id=image_id,
                        min_visibility=min_object_visibility,
                    )
                    candidates[split].append(
                        {
                            "date": date,
                            "source_path": image_path,
                            "window": window,
                            "row_off": row_off,
                            "col_off": col_off,
                            "width": width,
                            "height": height,
                            "image_id": image_id,
                            "block_id": block_id,
                            "annotations": annotations,
                        }
                    )

    payloads: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for split in ("train", "val"):
        positive = [item for item in candidates[split] if item["annotations"]]
        empty = [item for item in candidates[split] if not item["annotations"]]
        empty.sort(key=lambda item: _stable_int(item["date"], item["row_off"], item["col_off"]))
        empty_limit = int(len(positive) * max_empty_to_positive_ratio)
        selected_empty = empty[:empty_limit] if include_empty_tiles else []
        selected = sorted(positive + selected_empty, key=lambda item: item["image_id"])
        images = []
        annotations = []
        for item in selected:
            filename = f"{item['date']}_{item['row_off']:06d}_{item['col_off']:06d}.png"
            relative = Path(split) / filename
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(item["source_path"]) as source:
                Image.fromarray(_rgb_window(source, item["window"], list(rgb_bands))).save(destination)
            images.append(
                {
                    "id": item["image_id"],
                    "file_name": relative.as_posix(),
                    "width": item["width"],
                    "height": item["height"],
                    "date": item["date"],
                    "row_off": item["row_off"],
                    "col_off": item["col_off"],
                    "block_id": item["block_id"],
                    "split": split,
                }
            )
            annotations.extend(item["annotations"])
        payload = {
            "info": {"description": "Urban renewal v0.4 building instances", "schema_version": "0.4.0"},
            "licenses": [],
            "categories": [{"id": 1, "name": "building", "supercategory": "structure"}],
            "images": images,
            "annotations": sorted(annotations, key=lambda item: item["id"]),
        }
        annotation_path = output / f"annotations_{split}.json"
        annotation_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        payloads[split] = payload
        summary[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "positive_tiles": len(positive),
            "empty_tiles": len(selected_empty),
            "excluded_boundary_tiles": sum(1 for item in candidates[split] if item not in selected),
        }
    result = {"schema_version": "0.4.0", "tile_size": tile_size, "overlap": overlap, "splits": summary}
    (output / "dataset_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_previews(output, payloads, count=20)
    return result
