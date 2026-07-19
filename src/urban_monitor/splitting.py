from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Iterable

import geopandas as gpd
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box


def build_spatial_blocks(
    bounds: tuple[float, float, float, float],
    crs,
    *,
    block_size_m: float = 500.0,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> gpd.GeoDataFrame:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be within (0,1)")
    minx, miny, maxx, maxy = bounds
    extent = box(*bounds)
    rows = []
    col = 0
    x = minx
    while x < maxx:
        row = 0
        y = miny
        while y < maxy:
            geometry = box(x, y, min(x + block_size_m, maxx), min(y + block_size_m, maxy)).intersection(extent)
            rows.append({"block_id": f"block_{col:05d}_{row:05d}", "column": col, "row": row, "geometry": geometry})
            row += 1
            y += block_size_m
        col += 1
        x += block_size_m
    if len(rows) < 2:
        raise ValueError("AOI creates fewer than two spatial blocks; reduce split.block_size_m")
    ids = [item["block_id"] for item in rows]
    rng = random.Random(seed)
    selected = set(rng.sample(ids, min(max(1, round(len(ids) * validation_fraction)), len(ids) - 1)))
    for item in rows:
        item["split"] = "val" if item["block_id"] in selected else "train"
        item["area_m2"] = float(item["geometry"].area)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def assign_geometry_splits(frame: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.crs != blocks.crs:
        frame = frame.to_crs(blocks.crs)
    result = frame.copy()
    centroids = gpd.GeoDataFrame(
        {"_source_index": result.index}, geometry=result.geometry.centroid, crs=result.crs
    )
    joined = gpd.sjoin(centroids, blocks[["block_id", "split", "geometry"]], predicate="within", how="left")
    if joined["split"].isna().any():
        raise ValueError("Some label centroids are outside the spatial split grid")
    by_index = joined.drop_duplicates("_source_index").set_index("_source_index")
    result["block_id"] = [by_index.loc[index, "block_id"] for index in result.index]
    result["split"] = [by_index.loc[index, "split"] for index in result.index]
    return result


def split_for_tile(tile_geometry, blocks: gpd.GeoDataFrame) -> tuple[str, str] | None:
    """Return one split only when a tile is fully inside a single block.

    Boundary-crossing tiles are excluded, preventing overlapping train/val chips
    from sharing pixels across a spatial split boundary.
    """
    candidates = blocks.iloc[list(blocks.sindex.query(tile_geometry, predicate="intersects"))]
    containers = candidates[candidates.geometry.covers(tile_geometry)]
    if len(containers) != 1:
        return None
    row = containers.iloc[0]
    return str(row["split"]), str(row["block_id"])


def _draw_split_map(blocks: gpd.GeoDataFrame, labels: gpd.GeoDataFrame, path: Path) -> None:
    width = height = 1000
    minx, miny, maxx, maxy = blocks.total_bounds
    sx = (width - 40) / max(maxx - minx, 1e-9)
    sy = (height - 40) / max(maxy - miny, 1e-9)

    def point(x, y):
        return 20 + (x - minx) * sx, height - 20 - (y - miny) * sy

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for row in blocks.itertuples():
        left, bottom, right, top = row.geometry.bounds
        x0, y0 = point(left, top)
        x1, y1 = point(right, bottom)
        fill = "#dbeafe" if row.split == "train" else "#fee2e2"
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline="#475569", width=1)
    for geometry in labels.geometry:
        if geometry is None or geometry.is_empty:
            continue
        x, y = point(geometry.centroid.x, geometry.centroid.y)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_split_report(
    blocks: gpd.GeoDataFrame,
    accepted_labels_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = gpd.read_file(accepted_labels_path)
    labels = assign_geometry_splits(labels, blocks) if not labels.empty else labels.assign(block_id="", split="")
    blocks_path = output / "spatial_split.gpkg"
    blocks.to_file(blocks_path, driver="GPKG")
    label_counts = labels.groupby("split").size().to_dict() if not labels.empty else {}
    rows = []
    for split in ("train", "val"):
        rows.append(
            {
                "split": split,
                "area_m2": float(blocks.loc[blocks["split"] == split, "area_m2"].sum()),
                "building_count": int(label_counts.get(split, 0)),
                "positive_tiles": 0,
                "empty_tiles": 0,
            }
        )
    summary_path = output / "split_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    if not labels.empty:
        per_date = labels.groupby(["source_date", "split"]).size().rename("building_count").reset_index()
        per_date.to_csv(output / "split_by_date.csv", index=False, encoding="utf-8-sig")
    _draw_split_map(blocks, labels, output / "split_map.png")
    warnings = []
    if int(label_counts.get("val", 0)) < 20:
        warnings.append(f"Validation has only {int(label_counts.get('val', 0))} buildings (<20)")
    return {"blocks_path": blocks_path, "summary_path": summary_path, "labels": labels, "warnings": warnings}
