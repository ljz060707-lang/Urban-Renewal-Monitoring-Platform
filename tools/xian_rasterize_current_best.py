"""Rasterize the latest Xi'an candidate Geometry and IdentityGraph locally.

The script is intentionally conservative:
* source GPKG polygons are read-only;
* each date is rasterized independently;
* tracking only supplies the numeric label lookup and state selection;
* polygons are never warped by alignment diagnostics;
* the >=100 m2 filter is applied only to product copies.

The output is a candidate product layer, not a scientific accuracy certificate.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work" / "python-packages"))

import numpy as np
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.ops import transform as shapely_transform


warnings.filterwarnings(
    "ignore",
    message=r"Setting the shape on a NumPy array has been deprecated.*",
    category=DeprecationWarning,
)


DELIVERY = Path(os.environ.get("XIAN_DELIVERY", r"F:\xian\西安_当前最佳候选交付_20260917"))
GEOM_DIR = Path(os.environ.get("XIAN_GEOM_DIR", r"F:\xian\西安_三期建筑追踪与三态最终正式候选版_20260916\02_三期建筑几何\remote_full_geometry"))
GEOM_ASCII = Path(os.environ.get("XIAN_GEOM_ASCII", r"F:\xian\_xian_geom_transfer"))
LATEST = Path(os.environ.get("XIAN_LATEST", r"F:\xian\西安_三期建筑追踪与三态最终正式候选版_20260916\14_最终报告与回执"))
FORMAL = Path(os.environ.get("XIAN_FORMAL", r"F:\xian\西安_三期建筑追踪与三态最终正式候选版_20260916"))
RAW_GRID = Path(os.environ.get("XIAN_RAW_GRID", r"F:\xian\xian\xa180429_combined\xa180429_zoom_20.tif"))

SCENES = {
    "2018": ("xa180429", "2018-04-29", "2018-04-29_building_objects.gpkg"),
    "2019": ("xa190101", "2019-01-01", "2019-01-01_building_objects.gpkg"),
    "2020": ("xa201231", "2020-12-31", "2020-12-31_building_objects.gpkg"),
}
CHUNK_SIZE = 6000
BLOCK_SIZE = 512
STRIPE_ROWS = 512
AREA_FILTER = 100.0


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)
    progress = DELIVERY / "01_raster_products" / "RASTER_PROGRESS.md"
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a", encoding="utf-8") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def geom_path(filename: str) -> Path:
    """Resolve the verified local Geometry transfer, with an ASCII fallback."""
    fallback = GEOM_ASCII / filename
    if fallback.exists() and fallback.stat().st_size > 0:
        return fallback
    primary = GEOM_DIR / filename
    if primary.exists() and primary.stat().st_size > 0:
        return primary
    return primary


def grid_profile() -> tuple[dict, dict]:
    with rasterio.open(RAW_GRID) as src:
        profile = {
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": BLOCK_SIZE,
            "blockysize": BLOCK_SIZE,
            "compress": "DEFLATE",
            "predictor": 2,
            "BIGTIFF": "IF_SAFER",
        }
        grid = {
            "source_grid": str(RAW_GRID),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": list(src.transform),
            "resolution_degrees": list(src.res),
            "bounds": list(src.bounds),
        }
    # Approximate metric pixel dimensions at the grid centre, for the report only.
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    x0, y0 = tf.transform((grid["bounds"][0] + grid["bounds"][2]) / 2, (grid["bounds"][1] + grid["bounds"][3]) / 2)
    x1, y1 = tf.transform((grid["bounds"][0] + grid["bounds"][2]) / 2 + grid["resolution_degrees"][0], (grid["bounds"][1] + grid["bounds"][3]) / 2)
    x2, y2 = tf.transform((grid["bounds"][0] + grid["bounds"][2]) / 2, (grid["bounds"][1] + grid["bounds"][3]) / 2 + abs(grid["resolution_degrees"][1]))
    grid["approx_pixel_width_m"] = abs(x1 - x0)
    grid["approx_pixel_height_m"] = abs(y2 - y0)
    grid["approx_pixel_area_m2"] = abs((x1 - x0) * (y2 - y0))
    return profile, grid


def load_identity_maps() -> tuple[dict, dict, dict]:
    """Return (object->code, object->area, code->physical_track_id)."""
    path = LATEST / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv"
    if not path.exists():
        path = FORMAL / "05_建筑追踪" / "tracking_buildings.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing tracking objects CSV: {path}")
    object_code: dict[tuple[str, int], int] = {}
    object_area: dict[tuple[str, int], float] = {}
    physical_to_code: dict[str, int] = {}
    code_to_physical: dict[int, str] = {}
    next_code = 1
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scene = row.get("scene_key", "")
            uid = row.get("object_uid", "")
            try:
                object_id = int(uid.split("::")[-1])
            except Exception:
                continue
            physical = row.get("physical_track_id") or row.get("building_id") or row.get("track_id") or ""
            if not physical:
                continue
            if physical not in physical_to_code:
                physical_to_code[physical] = next_code
                code_to_physical[next_code] = physical
                next_code += 1
            code = physical_to_code[physical]
            key = (scene, object_id)
            object_code[key] = code
            try:
                object_area[key] = float(row.get("area_m2") or 0.0)
            except Exception:
                object_area[key] = 0.0
    lookup = DELIVERY / "01_raster_products" / "BUILDING_ID_LOOKUP.csv"
    lookup.parent.mkdir(parents=True, exist_ok=True)
    with lookup.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["numeric_code", "physical_track_id"])
        w.writerows(sorted(code_to_physical.items()))
    log(f"loaded {len(object_code):,} date-object labels and {len(code_to_physical):,} stable numeric labels from {path.name}")
    return object_code, object_area, code_to_physical


def make_dataset(path: Path, profile: dict, dtype: str, nodata: int = 0):
    if path.exists():
        path.unlink()
    p = dict(profile)
    p.update({"dtype": dtype, "nodata": nodata})
    # Block-wise polygon burns need to read/merge an already written block.
    return rasterio.open(path, "w+", **p)


def block_window(row: int, col: int, height: int, width: int) -> Window:
    r = row * BLOCK_SIZE
    c = col * BLOCK_SIZE
    return Window(c, r, min(BLOCK_SIZE, width - c), min(BLOCK_SIZE, height - r))


def metric_bbox_for_window(win: Window, transform) -> tuple[float, float, float, float]:
    """Return a small EPSG:32649 bbox for a WGS84 output window."""
    left, bottom, right, top = rasterio.windows.bounds(win, transform)
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    corners = [
        to_metric.transform(left, bottom),
        to_metric.transform(left, top),
        to_metric.transform(right, bottom),
        to_metric.transform(right, top),
    ]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return min(xs) - 1.0, min(ys) - 1.0, max(xs) + 1.0, max(ys) + 1.0


def block_ids(geom, transform, height: int, width: int) -> tuple[int, int, int, int] | None:
    try:
        win = from_bounds(*geom.bounds, transform=transform).round_offsets().round_lengths()
    except Exception:
        return None
    r0 = max(0, int(win.row_off) // BLOCK_SIZE)
    c0 = max(0, int(win.col_off) // BLOCK_SIZE)
    r1 = min((height - 1) // BLOCK_SIZE, int(math.ceil((win.row_off + max(1, win.height)) / BLOCK_SIZE)) - 1)
    c1 = min((width - 1) // BLOCK_SIZE, int(math.ceil((win.col_off + max(1, win.width)) / BLOCK_SIZE)) - 1)
    if r1 < r0 or c1 < c0:
        return None
    return r0, r1, c0, c1


def update_mask_id_block(dst_mask, dst_id, shapes, win: Window, transform, current_code: np.ndarray | None) -> None:
    if not shapes:
        return
    h, w = int(win.height), int(win.width)
    local_transform = window_transform(win, transform)
    mask = rasterize([(geom, 1) for geom, _ in shapes], out_shape=(h, w), transform=local_transform, fill=0, dtype="uint8")
    current_mask = dst_mask.read(1, window=win)
    current_mask[mask > 0] = 1
    dst_mask.write(current_mask, 1, window=win)
    if dst_id is not None:
        ids = rasterize(shapes, out_shape=(h, w), transform=local_transform, fill=0, dtype="uint32")
        current = dst_id.read(1, window=win)
        current[ids > 0] = ids[ids > 0]
        dst_id.write(current, 1, window=win)


def rasterize_scene(scene_year: str, profile: dict, object_code: dict, object_area: dict) -> None:
    scene, _, filename = SCENES[scene_year]
    gpkg = geom_path(filename)
    if not gpkg.exists():
        raise FileNotFoundError(f"geometry not ready: {gpkg}")
    out_dir = DELIVERY / "01_raster_products"
    mask_path = out_dir / f"building_boundary_{scene_year}_unfiltered.tif"
    id_path = out_dir / f"building_id_{scene_year}_unfiltered.tif"
    mask100_path = out_dir / f"building_boundary_{scene_year}_ge100m2.tif"
    id100_path = out_dir / f"building_id_{scene_year}_ge100m2.tif"
    width, height = profile["width"], profile["height"]
    tf = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    mask = make_dataset(mask_path, profile, "uint8")
    ids = make_dataset(id_path, profile, "uint32")
    mask100 = make_dataset(mask100_path, profile, "uint8")
    ids100 = make_dataset(id100_path, profile, "uint32")
    try:
        total = int(pyogrio.read_info(gpkg)["features"])
        log(f"{scene_year}: rasterizing {total:,} Geometry objects by {STRIPE_ROWS}-row spatial stripes")
        n_stripes = (height + STRIPE_ROWS - 1) // STRIPE_ROWS
        for stripe_index, row_off in enumerate(range(0, height, STRIPE_ROWS), start=1):
            stripe_height = min(STRIPE_ROWS, height - row_off)
            win = Window(0, row_off, width, stripe_height)
            bbox = metric_bbox_for_window(win, profile["transform"])
            df = pyogrio.read_dataframe(
                gpkg,
                columns=["object_id", "area_m2", "geometry"],
                bbox=bbox,
                use_arrow=False,
            )
            shapes = []
            shapes100 = []
            for obj, area, geom in zip(df["object_id"], df["area_m2"], df.geometry):
                if geom is None or geom.is_empty:
                    continue
                try:
                    oid = int(obj)
                    code = int(object_code.get((scene, oid), 0))
                    if code == 0:
                        continue
                    geom4326 = shapely_transform(tf.transform, geom)
                    shapes.append((geom4326, code))
                    area_value = float(area or 0.0)
                    if area_value >= AREA_FILTER and object_area.get((scene, oid), area_value) >= AREA_FILTER:
                        shapes100.append((geom4326, code))
                except Exception:
                    continue
            if shapes:
                ids_arr = rasterize(
                    shapes,
                    out_shape=(stripe_height, width),
                    transform=window_transform(win, profile["transform"]),
                    fill=0,
                    dtype="uint32",
                )
                ids.write(ids_arr, 1, window=win)
                mask.write((ids_arr > 0).astype("uint8"), 1, window=win)
            if shapes100:
                ids100_arr = rasterize(
                    shapes100,
                    out_shape=(stripe_height, width),
                    transform=window_transform(win, profile["transform"]),
                    fill=0,
                    dtype="uint32",
                )
                ids100.write(ids100_arr, 1, window=win)
                mask100.write((ids100_arr > 0).astype("uint8"), 1, window=win)
            if stripe_index == 1 or stripe_index == n_stripes or stripe_index % 10 == 0:
                log(f"{scene_year}: stripe {stripe_index}/{n_stripes}, candidates in stripe {len(df):,}")
    finally:
        mask.close()
        ids.close()
        mask100.close()
        ids100.close()
    log(f"{scene_year}: finished boundary and ID rasters")


def load_state_object_maps(object_code: dict, object_area: dict) -> dict:
    state_csv = LATEST / "CANDIDATE_STATE_EVENTS_FROM_SCRATCH.csv"
    tracking_csv = LATEST / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv"
    if not state_csv.exists() or not tracking_csv.exists():
        raise FileNotFoundError("latest candidate state/tracking CSVs are not ready")
    track_objects: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with tracking_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scene = row.get("scene_key", "")
            physical = row.get("physical_track_id", "")
            try:
                oid = int(row.get("object_uid", "").split("::")[-1])
            except Exception:
                continue
            if not scene or not physical:
                continue
            area = float(row.get("area_m2") or 0.0)
            track_objects[(physical, scene)].append((oid, area))
    selected: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    scene_pair = {"2018_2019": ("xa180429", "xa190101"), "2019_2020": ("xa190101", "xa201231")}
    with state_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            interval = row.get("interval", "")
            state = row.get("state", "")
            physical = row.get("physical_track_id", "")
            if interval not in scene_pair or state not in {"STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT"}:
                continue
            source_scene, target_scene = scene_pair[interval]
            scenes = [target_scene] if state in {"STABLE", "ADDITION"} else [source_scene] if state == "DEMOLITION" else [source_scene, target_scene]
            for sc in scenes:
                for oid, area in track_objects.get((physical, sc), []):
                    if area >= AREA_FILTER:
                        selected[(sc, oid)].add((interval, state))
    log(f"selected {len(selected):,} date-object/state entries for interval product masks")
    return selected


def make_state_datasets(profile: dict, selected: dict) -> dict:
    out: dict = {}
    for interval in ("2018_2019", "2019_2020"):
        d = DELIVERY / "01_raster_products" / interval
        d.mkdir(parents=True, exist_ok=True)
        out[interval] = {}
        for state in ("STABLE", "ADDITION", "DEMOLITION", "UNKNOWN_CONFLICT"):
            out[interval][state] = make_dataset(d / f"{state}_ge100m2.tif", profile, "uint8")
        out[interval]["COMBINED"] = make_dataset(d / "THREE_STATE_COMBINED_ge100m2.tif", profile, "uint8", nodata=255)
    return out


def rasterize_state_masks(profile: dict, selected: dict) -> None:
    datasets = make_state_datasets(profile, selected)
    try:
        for scene_year, (scene, _, filename) in SCENES.items():
            gpkg = geom_path(filename)
            total = int(pyogrio.read_info(gpkg)["features"])
            tf = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
            n_stripes = (profile["height"] + STRIPE_ROWS - 1) // STRIPE_ROWS
            for stripe_index, row_off in enumerate(range(0, profile["height"], STRIPE_ROWS), start=1):
                stripe_height = min(STRIPE_ROWS, profile["height"] - row_off)
                win = Window(0, row_off, profile["width"], stripe_height)
                bbox = metric_bbox_for_window(win, profile["transform"])
                df = pyogrio.read_dataframe(
                    gpkg,
                    columns=["object_id", "geometry"],
                    bbox=bbox,
                    use_arrow=False,
                )
                groups = defaultdict(list)
                for obj, geom in zip(df["object_id"], df.geometry):
                    if geom is None or geom.is_empty:
                        continue
                    labels = selected.get((scene, int(obj)))
                    if not labels:
                        continue
                    geom4326 = shapely_transform(tf.transform, geom)
                    for interval, state in labels:
                        groups[(interval, state)].append((geom4326, 1))
                local = window_transform(win, profile["transform"])
                for (interval, state), shapes in groups.items():
                    if not shapes:
                        continue
                    ds = datasets[interval][state]
                    burn = rasterize(
                        shapes,
                        out_shape=(stripe_height, profile["width"]),
                        transform=local,
                        fill=0,
                        dtype="uint8",
                    )
                    ds.write(burn, 1, window=win)
                if stripe_index == 1 or stripe_index == n_stripes or stripe_index % 10 == 0:
                    log(f"state masks {scene_year}: stripe {stripe_index}/{n_stripes}, candidates in stripe {len(df):,}")
        for interval, layers in datasets.items():
            combined = layers["COMBINED"]
            n_stripes = (profile["height"] + STRIPE_ROWS - 1) // STRIPE_ROWS
            for stripe_index, row_off in enumerate(range(0, profile["height"], STRIPE_ROWS), start=1):
                stripe_height = min(STRIPE_ROWS, profile["height"] - row_off)
                win = Window(0, row_off, profile["width"], stripe_height)
                stable = layers["STABLE"].read(1, window=win)
                addition = layers["ADDITION"].read(1, window=win)
                demolition = layers["DEMOLITION"].read(1, window=win)
                unknown = layers["UNKNOWN_CONFLICT"].read(1, window=win)
                arr = np.zeros(stable.shape, dtype="uint8")
                arr[stable > 0] = 1
                arr[addition > 0] = 2
                arr[demolition > 0] = 3
                arr[unknown > 0] = 255
                combined.write(arr, 1, window=win)
            log(f"{interval}: combined state product finished")
    finally:
        for layers in datasets.values():
            for ds in layers.values():
                ds.close()


def main() -> None:
    out_dir = DELIVERY / "01_raster_products"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile, grid = grid_profile()
    (out_dir / "GRID_METADATA.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")
    object_code, object_area, _ = load_identity_maps()
    for scene_year in SCENES:
        rasterize_scene(scene_year, profile, object_code, object_area)
    selected = load_state_object_maps(object_code, object_area)
    rasterize_state_masks(profile, selected)
    resolved_geometry = {
        year: str(geom_path(filename)) for year, (_, _, filename) in SCENES.items()
    }
    (out_dir / "RASTER_RECEIPT.json").write_text(json.dumps({
        "status": "COMPLETE_CANDIDATE_RASTER_EXPORT",
        "source_geometry": str(GEOM_DIR),
        "source_geometry_resolved": resolved_geometry,
        "grid": grid,
        "area_filter_m2": AREA_FILTER,
        "alignment_geometry_warped": False,
        "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("all raster products complete")


if __name__ == "__main__":
    main()
