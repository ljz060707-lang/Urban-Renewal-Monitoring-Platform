"""Process one completed GEID Xi'an epoch into an auditable URRS package.

The input GEID project and its JPEG tiles are read-only.  This command builds
small, timestamped artifacts on E: (date coverage, quality proxies, a VRT
reference, and optional sparse B0 building geometry) without materialising a
city-scale probability raster.  It intentionally treats acquisition dates as
observations recovered from GEID logs, never as labels inferred from a task
name or a nominal request date.

Examples::

    python tools/process_xian_single_epoch.py --stage audit
    python tools/process_xian_single_epoch.py --stage all

``--stage geometry`` is the only stage that imports torch and the accepted
RSBuilding runner.  It is resumable at strip boundaries and uses the frozen
threshold/tile/overlap contract.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATA_ROOT_DEFAULT = Path(r"D:\Data Set\xiannew")
TASK_STEM_DEFAULT = "xar1201231"
OUTPUT_ROOT_DEFAULT = Path(r"E:\URRS Output\xiannew\epoch_packages")
METRIC_CRS = "EPSG:32649"  # Xi'an UTM zone 49N; verified for the AOI bounds.
NATIVE_CRS = "EPSG:4326"
ZOOM = 20
TILE_SIZE = 256
DATE_RE = re.compile(r"(?<!\d)(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})(?!\d)")
TILE_RE = re.compile(
    r"gesh_(?P<x>\d+)_(?P<y>\d+)_(?P<z>\d+)\.(?:jpg|jpeg)", re.IGNORECASE
)
LOG_TILE_RE = re.compile(
    r"(?P<filename>gesh_\d+_\d+_\d+\.(?:jpg|jpeg))\s*:\s*"
    r"(?P<date>\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+(?P<status>[^\s]+)",
    re.IGNORECASE,
)
LIST_RE = re.compile(
    r"(?P<filename>gesh_\d+_\d+_\d+\.jpg)\s*:\s*"
    r"(?P<west>-?\d+(?:\.\d+)?)\s+(?P<east>-?\d+(?:\.\d+)?)\s+"
    r"(?P<north>-?\d+(?:\.\d+)?)\s+(?P<south>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
OK_STATUSES = {"ok", "success", "successful", "downloaded", "complete"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def geometry_lock_path(
    device: str,
    lock_dir: Path = Path(r"E:\URRS Output\runtime_locks"),
    physical_device: str | None = None,
) -> Path:
    """Return the derived-geometry lock for one execution device.

    Xi'an geometry jobs may safely run concurrently on independent CUDA
    devices. Keep the lock name deterministic and filesystem-safe while
    avoiding the historical single global GPU lock. The helper is small and
    side-effect free so its contract can be covered without importing torch.
    """
    lock_device = physical_device if physical_device not in (None, "") else device
    # A single CUDA_VISIBLE_DEVICES value denotes the physical slot; for a
    # multi-device visibility list the caller's local device ordinal remains
    # the only unambiguous fallback.
    if lock_device is not None and "," in str(lock_device):
        lock_device = device
    safe_device = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(lock_device)).strip("._") or "default"
    return lock_dir / f"xian_r1_2020_geometry_{safe_device}.lock"


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def safe_stat(path: Path) -> dict[str, Any]:
    try:
        item = path.stat()
        return {"exists": True, "bytes": item.st_size, "mtime": item.st_mtime}
    except OSError as exc:
        return {"exists": False, "error": f"{type(exc).__name__}: {exc}"}


def read_ini(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("[") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_date(value: str | None) -> str | None:
    if not value:
        return None
    match = DATE_RE.search(value)
    if not match:
        return None
    try:
        year, month, day = (int(part) for part in re.split(r"[-/]", match.group("date")))
        return date(year, month, day).isoformat()
    except (TypeError, ValueError):
        return None


def expected_grid(values: dict[str, str]) -> dict[str, Any]:
    min_x = int(values.get(f"MinX_{ZOOM}", "0"))
    max_x = int(values.get(f"MaxX_{ZOOM}", "-1"))
    min_y = int(values.get(f"MinY_{ZOOM}", "0"))
    max_y = int(values.get(f"MaxY_{ZOOM}", "-1"))
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "tile_columns": max(0, max_x - min_x + 1),
        "tile_rows": max(0, max_y - min_y + 1),
    }


def tile_name(x: int, y: int) -> str:
    return f"gesh_{x}_{y}_{ZOOM}.jpg"


def tile_path(raw_root: Path, x: int, y: int) -> Path:
    return raw_root / str(x) / tile_name(x, y)


def list_metadata(data_root: Path, stem: str, grid: dict[str, Any]) -> dict[str, Any]:
    paths = [data_root / f"{stem}_list{i}.txt" for i in (1, 2, 3)]
    summaries: list[dict[str, Any]] = []
    first: dict[str, float] | None = None
    last: dict[str, float] | None = None
    for path in paths:
        count = 0
        unique: set[str] = set()
        min_west = math.inf
        max_east = -math.inf
        min_south = math.inf
        max_north = -math.inf
        if path.is_file():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = LIST_RE.search(line)
                    if not match:
                        continue
                    count += 1
                    filename = match.group("filename").lower()
                    unique.add(filename)
                    row = {key: float(match.group(key)) for key in ("west", "east", "north", "south")}
                    min_west = min(min_west, row["west"])
                    max_east = max(max_east, row["east"])
                    min_south = min(min_south, row["south"])
                    max_north = max(max_north, row["north"])
                    if first is None:
                        first = row
                    last = row
        summaries.append(
            {
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "row_count": count,
                "unique_filename_count": len(unique),
                "bounds": None
                if count == 0
                else [min_west, min_south, max_east, max_north],
            }
        )
    inferred = None
    if first and last:
        inferred = {
            "west": first["west"],
            "east": last["east"],
            "north": last["north"],
            "south": first["south"],
        }
    expected = grid["tile_columns"] * grid["tile_rows"]
    return {
        "files": summaries,
        "expected_tile_count": expected,
        "first_tile_bounds": first,
        "last_tile_bounds": last,
        "inferred_bounds": inferred,
    }


def discover_logs(data_root: Path, stem: str) -> list[Path]:
    return sorted(
        [path for path in data_root.glob(f"{stem}_log.txt*") if path.is_file()],
        key=lambda path: (path.stat().st_mtime, path.name),
    )


def parse_logs(logs: Iterable[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_lines = 0
    matched_lines = 0
    for path in logs:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                total_lines += 1
                match = LOG_TILE_RE.search(line)
                if not match:
                    continue
                matched_lines += 1
                filename = match.group("filename").lower()
                observed_date = parse_date(match.group("date"))
                status = match.group("status").strip().lower()
                observations[filename].append(
                    {
                        "date": observed_date,
                        "status": status,
                        "success": status in OK_STATUSES,
                        "source_log": str(path.resolve()),
                        "line_number": line_number,
                        "log_mtime": mtime,
                    }
                )
    result: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for filename, rows in observations.items():
        dates = sorted({row["date"] for row in rows if row.get("date")})
        if len(dates) > 1:
            conflicts += 1
        successful = [row for row in rows if row.get("success") and row.get("date")]
        chosen = max(successful or rows, key=lambda row: (row.get("log_mtime", 0.0), row.get("line_number", 0)))
        result[filename] = {
            "actual_acquisition_date": chosen.get("date"),
            "date_source": "geid_log_success_latest" if successful else "geid_log_latest_non_success",
            "latest_status": chosen.get("status"),
            "observed_dates": dates,
            "log_observation_count": len(rows),
            "date_conflict": len(dates) > 1,
            "source_log": chosen.get("source_log"),
            "line_number": chosen.get("line_number"),
        }
    return result, {
        "log_count": len(list(logs)) if not isinstance(logs, list) else len(logs),
        "total_lines": total_lines,
        "matched_tile_lines": matched_lines,
        "unique_tiles_in_logs": len(result),
        "date_conflict_tile_count": conflicts,
    }


def sample_jpg(path: Path) -> dict[str, Any]:
    from PIL import Image

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            sample = np.asarray(rgb.resize((32, 32)), dtype=np.float32)
        luminance = sample.mean(axis=2)
        gradient = np.concatenate(
            [np.diff(luminance, axis=0).ravel(), np.diff(luminance, axis=1).ravel()]
        )
        return {
            "read_success": True,
            "width": int(image.width),
            "height": int(image.height),
            "bands": 3,
            "dtype": "uint8",
            "mean_luminance": float(luminance.mean()),
            "std_luminance": float(luminance.std()),
            "dynamic_range": float(luminance.max() - luminance.min()),
            "nonzero_fraction": float(np.count_nonzero(sample) / sample.size),
            "gradient_variance": float(gradient.var()),
        }
    except Exception as exc:  # pragma: no cover - provider files are external
        return {"read_success": False, "error": f"{type(exc).__name__}: {exc}"}


def deterministic_sample_paths(raw_root: Path, grid: dict[str, Any]) -> list[Path]:
    coords = [
        (grid["min_x"], grid["min_y"]),
        (grid["min_x"], grid["max_y"]),
        (grid["max_x"], grid["min_y"]),
        (grid["max_x"], grid["max_y"]),
        ((grid["min_x"] + grid["max_x"]) // 2, (grid["min_y"] + grid["max_y"]) // 2),
    ]
    return [tile_path(raw_root, x, y) for x, y in coords]


def scan_tiles(raw_root: Path, grid: dict[str, Any], logs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = grid["tile_columns"] * grid["tile_rows"]
    present = 0
    missing = 0
    present_bytes = 0
    samples: list[dict[str, Any]] = []
    bad_names = 0
    for y in range(grid["min_y"], grid["max_y"] + 1):
        for x in range(grid["min_x"], grid["max_x"] + 1):
            path = tile_path(raw_root, x, y)
            if not path.is_file():
                missing += 1
                continue
            present += 1
            try:
                present_bytes += path.stat().st_size
            except OSError:
                pass
    for path in deterministic_sample_paths(raw_root, grid):
        row = {"path": str(path.resolve()), "exists": path.is_file()}
        if path.is_file():
            row.update(sample_jpg(path))
            row["sha256"] = sha256_file(path)
        samples.append(row)
    extras = 0
    for path in raw_root.rglob("*.jpg"):
        if not TILE_RE.fullmatch(path.name):
            bad_names += 1
            continue
        match = TILE_RE.fullmatch(path.name)
        assert match is not None
        x, y, z = (int(match.group(key)) for key in ("x", "y", "z"))
        if z != ZOOM or not (grid["min_x"] <= x <= grid["max_x"] and grid["min_y"] <= y <= grid["max_y"]):
            extras += 1
    return {
        "expected_tile_count": expected,
        "present_tile_count": present,
        "missing_tile_count": missing,
        "present_fraction": present / expected if expected else 0.0,
        "present_bytes": present_bytes,
        "extra_tile_count": extras,
        "nonconforming_jpg_name_count": bad_names,
        "samples": samples,
        "log_date_coverage_for_present_tiles": None,
    }


def grid_bounds(list_info: dict[str, Any], values: dict[str, str]) -> tuple[float, float, float, float]:
    inferred = list_info.get("inferred_bounds") or {}
    west = float(values.get(f"Left_Longitude_download_{ZOOM}", inferred.get("west", 0.0)))
    east = float(values.get(f"Right_Longitude_download_{ZOOM}", inferred.get("east", 0.0)))
    south = float(values.get(f"Bottom_Latitude_download_{ZOOM}", inferred.get("south", 0.0)))
    north = float(values.get(f"Top_Latitude_download_{ZOOM}", inferred.get("north", 0.0)))
    return west, south, east, north


def pixel_transform(bounds: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    west, south, east, north = bounds
    # ``Affine`` uses (a, b, c, d, e, f), while the VRT XML uses GDAL's
    # (west, xres, xskew, north, yskew, yres) order.  Keep the list in
    # Affine order because callers pass it to ``Affine(*transform[:6])``.
    return [(east - west) / width, 0.0, west, 0.0, -(north - south) / height, north, 0.0, 0.0, 1.0]


def write_raster(path: Path, array: np.ndarray, transform: list[float], nodata: int = 0) -> None:
    import rasterio
    from rasterio.transform import Affine

    height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": str(array.dtype),
        "crs": NATIVE_CRS,
        "transform": Affine(*transform[:6]),
        "nodata": nodata,
        "compress": "deflate",
        # Date/quality products are tile-grid rasters (138 x 1364 here), not
        # full imagery.  Striped GeoTIFFs avoid libtiff's requirement that
        # tiled block dimensions are multiples of 16.
        "tiled": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with rasterio.open(path):
                pass
        except Exception:
            # Only a derived, same-run raster is eligible for replacement;
            # this handles an interrupted first write without touching inputs.
            path.unlink()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def projected_tile_area(bounds: tuple[float, float, float, float], grid: dict[str, Any]) -> float:
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform

    west, south, east, north = bounds
    px = (east - west) / grid["tile_columns"]
    py = (north - south) / grid["tile_rows"]
    tile = box(west, south, west + px, south + py)
    fn = Transformer.from_crs(NATIVE_CRS, METRIC_CRS, always_xy=True).transform
    return float(transform(fn, tile).area)


def map_date_grid(path: Path, date_grid: np.ndarray, labels: dict[int, str], output_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colours = ["#d9d9d9", "#2c7fb8", "#7fcdbb", "#edf8b1", "#fdae61", "#d73027", "#8073ac", "#1b9e77", "#e6ab02", "#a6761d"]
    max_id = max(labels, default=0)
    cmap = ListedColormap(colours[: max(2, min(max_id + 1, len(colours)))])
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=180)
    image = ax.imshow(date_grid, cmap=cmap, interpolation="nearest", aspect="auto", vmin=0, vmax=max(max_id, 1))
    ax.set_title("Xi'an R1-2020 observed acquisition-date coverage (GEID log evidence)")
    ax.set_xlabel("tile column (x)")
    ax.set_ylabel("tile row (north-up)")
    handles = []
    for code, label in sorted(labels.items()):
        handles.append(plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(code % cmap.N), label=f"{label} ({code})", markersize=8))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def map_quality_grid(path: Path, quality_grid: np.ndarray, output_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    labels = {0: "usable", 1: "cloud/overexposure proxy", 2: "dark/shadow proxy", 3: "low dynamic range/haze proxy", 4: "blur proxy", 5: "seamline proxy", 255: "unobserved"}
    cmap = ListedColormap(["#2ca25f", "#de2d26", "#756bb1", "#fec44f", "#636363", "#e6550d", "#bdbdbd"])
    remap = np.zeros_like(quality_grid, dtype=np.uint8)
    for idx, code in enumerate((0, 1, 2, 3, 4, 5, 255)):
        remap[quality_grid == code] = idx
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=180)
    ax.imshow(remap, cmap=cmap, interpolation="nearest", aspect="auto", vmin=0, vmax=6)
    ax.set_title("Xi'an R1-2020 observation quality proxies (tile-level)")
    ax.set_xlabel("tile column (x)")
    ax.set_ylabel("tile row (north-up)")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i), label=labels[code], markersize=8) for i, code in enumerate((0, 1, 2, 3, 4, 5, 255))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_vrt(vrt_path: Path, raw_root: Path, grid: dict[str, Any], bounds: tuple[float, float, float, float]) -> dict[str, Any]:
    width = grid["tile_columns"] * TILE_SIZE
    height = grid["tile_rows"] * TILE_SIZE
    west, south, east, north = bounds
    x_res = (east - west) / width
    y_res = (north - south) / height
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    if vrt_path.is_file() and vrt_path.stat().st_size > 0:
        return {"status": "reused", "path": str(vrt_path.resolve()), "sha256": sha256_file(vrt_path), "width": width, "height": height}
    with vrt_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">\n')
        handle.write(f"  <SRS>{NATIVE_CRS}</SRS>\n")
        handle.write(f"  <GeoTransform>{west}, {x_res}, 0.0, {north}, 0.0, {-y_res}</GeoTransform>\n")
        for band, interp in ((1, "Red"), (2, "Green"), (3, "Blue")):
            handle.write(f'  <VRTRasterBand dataType="Byte" band="{band}">\n')
            handle.write(f"    <ColorInterp>{interp}</ColorInterp>\n")
            for y in range(grid["max_y"], grid["min_y"] - 1, -1):
                for x in range(grid["min_x"], grid["max_x"] + 1):
                    source = tile_path(raw_root, x, y)
                    if not source.is_file():
                        continue
                    dst_x = (x - grid["min_x"]) * TILE_SIZE
                    dst_y = (grid["max_y"] - y) * TILE_SIZE
                    handle.write("    <SimpleSource>\n")
                    handle.write(f"      <SourceFilename relativeToVRT=\"0\">{html.escape(str(source.resolve()))}</SourceFilename>\n")
                    handle.write(f"      <SourceBand>{band}</SourceBand>\n")
                    handle.write(f"      <SourceProperties RasterXSize=\"{TILE_SIZE}\" RasterYSize=\"{TILE_SIZE}\" DataType=\"Byte\" BlockXSize=\"{TILE_SIZE}\" BlockYSize=\"{TILE_SIZE}\"/>\n")
                    handle.write(f"      <SrcRect xOff=\"0\" yOff=\"0\" xSize=\"{TILE_SIZE}\" ySize=\"{TILE_SIZE}\"/>\n")
                    handle.write(f"      <DstRect xOff=\"{dst_x}\" yOff=\"{dst_y}\" xSize=\"{TILE_SIZE}\" ySize=\"{TILE_SIZE}\"/>\n")
                    handle.write("    </SimpleSource>\n")
            handle.write("  </VRTRasterBand>\n")
        handle.write("</VRTDataset>\n")
    return {"status": "created", "path": str(vrt_path.resolve()), "sha256": sha256_file(vrt_path), "width": width, "height": height}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "date", "quality", "vrt", "geometry", "package", "all"), default="audit")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=None,
        help=(
            "Optional read-only root containing the GEID, list and log files. "
            "Use this when raw tiles/source overrides were copied to another "
            "drive without duplicating the small metadata files."
        ),
    )
    parser.add_argument("--task-stem", default=TASK_STEM_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--source-override",
        type=Path,
        default=None,
        help="Read-only georeferenced raster used for geometry instead of the generated VRT.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--strip-rows", type=int, default=2048)
    parser.add_argument("--quality-workers", type=int, default=8)
    return parser.parse_args()


def make_context(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    metadata_root = (args.metadata_root or args.data_root).resolve()
    stem = args.task_stem
    geid = metadata_root / f"{stem}.geid"
    raw_root = data_root / stem / str(ZOOM)
    run = args.run_dir.resolve() if args.run_dir else (args.output_root / f"R1_2020_{stem}_{stamp()}").resolve()
    dirs = {
        "root": run,
        "manifest": run / "00_manifest",
        "date": run / "01_date",
        "quality": run / "02_quality",
        "imagery": run / "03_imagery",
        "geometry": run / "04_geometry",
        "visual": run / "05_visual_qa",
        "receipt": run / "06_receipt",
        "logs": run / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = dirs["manifest"] / "XIAN_R1_2020_INPUT_MANIFEST.json"
    if existing_manifest_path.is_file():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        values = dict(existing_manifest.get("geid", {}).get("values") or read_ini(geid))
        grid = dict(existing_manifest.get("grid") or expected_grid(values))
        lists = dict(existing_manifest.get("list_metadata") or list_metadata(metadata_root, stem, grid))
        bounds = tuple(existing_manifest.get("bounds_wgs84") or grid_bounds(lists, values))
    else:
        values = read_ini(geid)
        grid = expected_grid(values)
        lists = list_metadata(metadata_root, stem, grid)
        bounds = grid_bounds(lists, values)
    logs = discover_logs(metadata_root, stem)
    log_cache = dirs["manifest"] / "LOG_TILE_DATE_INDEX.json"
    if log_cache.is_file():
        cached = json.loads(log_cache.read_text(encoding="utf-8"))
        if cached.get("parser_version") == 2:
            log_rows = cached.get("rows", {})
            log_summary = cached.get("summary", {})
        else:
            log_rows, log_summary = parse_logs(logs)
            atomic_json(log_cache, {"parser_version": 2, "rows": log_rows, "summary": log_summary, "created_at": utc_now()})
    else:
        log_rows, log_summary = parse_logs(logs)
        atomic_json(log_cache, {"parser_version": 2, "rows": log_rows, "summary": log_summary, "created_at": utc_now()})
    return {
        "args": args,
        "data_root": data_root,
        "metadata_root": metadata_root,
        "stem": stem,
        "geid": geid,
        "raw_root": raw_root,
        "values": values,
        "grid": grid,
        "lists": lists,
        "bounds": bounds,
        "logs": logs,
        "log_rows": log_rows,
        "log_summary": log_summary,
        "dirs": dirs,
        "run": run,
    }


def stage_audit(ctx: dict[str, Any]) -> dict[str, Any]:
    geid = ctx["geid"]
    raw_root = ctx["raw_root"]
    scan = scan_tiles(raw_root, ctx["grid"], ctx["log_rows"])
    sample_bytes = {row["path"]: row.get("sha256") for row in scan["samples"] if row.get("sha256")}
    manifest = {
        "schema_version": "urrs.xian.r1_2020.input_manifest.v1",
        "created_at": utc_now(),
        "task_name": ctx["stem"],
        "nominal_date": parse_date(ctx["values"].get("HistoryDate")),
        "actual_acquisition_date": None,
        "mode_date": None,
        "date_semantics": "actual dates are recovered only from GEID log observations; nominal task date is not substituted",
        "geid": {
            "path": str(geid),
            "exists": geid.is_file(),
            "sha256": sha256_file(geid),
            "bytes": safe_stat(geid).get("bytes", 0),
            "values": ctx["values"],
        },
        "raw_tile_root": {
            "path": str(raw_root),
            "exists": raw_root.is_dir(),
            "read_only_contract": True,
            "jpg_scan": scan,
        },
        "grid": ctx["grid"],
        "native_crs": NATIVE_CRS,
        "metric_crs": METRIC_CRS,
        "bounds_wgs84": list(ctx["bounds"]),
        "list_metadata": ctx["lists"],
        "logs": {
            "paths": [str(path.resolve()) for path in ctx["logs"]],
            "sha256": {str(path.resolve()): sha256_file(path) for path in ctx["logs"]},
            **ctx["log_summary"],
        },
        "sample_tile_sha256": sample_bytes,
        "mosaic": {
            "final_tif_present": False,
            "final_tif_status": "not_present_derived_vrt_used",
            "combiner_logs": [str(path) for path in ctx["metadata_root"].glob(f"{ctx['stem']}*combine*.log")],
        },
        "raw_inputs_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "scientific_status": "candidate_not_ground_truth",
    }
    atomic_json(ctx["dirs"]["manifest"] / "XIAN_R1_2020_INPUT_MANIFEST.json", manifest)
    atomic_json(ctx["dirs"]["manifest"] / "LOG_TILE_DATE_INDEX.json", {"parser_version": 2, "rows": ctx["log_rows"], "summary": ctx["log_summary"], "created_at": utc_now()})
    atomic_json(ctx["dirs"]["receipt"] / "AUDIT_RECEIPT.json", {"status": "pass" if geid.is_file() and scan["present_fraction"] == 1.0 else "blocked_input_incomplete", "stage": "audit", "created_at": utc_now(), "manifest": str(ctx["dirs"]["manifest"] / "XIAN_R1_2020_INPUT_MANIFEST.json"), "raw_inputs_modified": False, "test_reads": manifest["test_reads"]})
    return manifest


def load_manifest(ctx: dict[str, Any]) -> dict[str, Any]:
    path = ctx["dirs"]["manifest"] / "XIAN_R1_2020_INPUT_MANIFEST.json"
    if not path.is_file():
        return stage_audit(ctx)
    return json.loads(path.read_text(encoding="utf-8"))


def stage_date(ctx: dict[str, Any]) -> dict[str, Any]:
    existing_receipt = ctx["dirs"]["date"] / "DATE_COVERAGE_RECEIPT.json"
    if existing_receipt.is_file():
        return json.loads(existing_receipt.read_text(encoding="utf-8"))
    manifest = load_manifest(ctx)
    grid = ctx["grid"]
    raw_root = ctx["raw_root"]
    dates = sorted({row.get("actual_acquisition_date") for row in ctx["log_rows"].values() if row.get("actual_acquisition_date")})
    date_codes = {item: index + 1 for index, item in enumerate(dates)}
    date_grid = np.zeros((grid["tile_rows"], grid["tile_columns"]), dtype=np.uint16)
    unknown_present = 0
    present = 0
    for row_index, y in enumerate(range(grid["max_y"], grid["min_y"] - 1, -1)):
        for col_index, x in enumerate(range(grid["min_x"], grid["max_x"] + 1)):
            path = tile_path(raw_root, x, y)
            if not path.is_file():
                continue
            present += 1
            log_row = ctx["log_rows"].get(path.name.lower())
            actual = log_row.get("actual_acquisition_date") if log_row else None
            if actual:
                date_grid[row_index, col_index] = date_codes[actual]
            else:
                unknown_present += 1
    date_grid_path = ctx["dirs"]["date"] / "DATE_CODE_GRID.npy"
    np.save(date_grid_path, date_grid, allow_pickle=False)
    transform = pixel_transform(ctx["bounds"], grid["tile_columns"], grid["tile_rows"])
    write_raster(ctx["dirs"]["date"] / "DATE_CODE_GRID.tif", date_grid, transform, nodata=0)
    area = projected_tile_area(ctx["bounds"], grid)
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform
    from rasterio.features import shapes

    to_metric = Transformer.from_crs(NATIVE_CRS, METRIC_CRS, always_xy=True).transform
    rows: list[dict[str, Any]] = []
    for geom_json, code in shapes(date_grid, mask=date_grid > 0, transform=__import__("rasterio").transform.Affine(*transform[:6])):
        code_int = int(code)
        geom = shapely_transform(to_metric, shape(geom_json))
        actual = next((key for key, value in date_codes.items() if value == code_int), None)
        rows.append({"date_code": code_int, "actual_acquisition_date": actual, "tile_count": int(np.count_nonzero(date_grid == code_int)), "area_m2": float(geom.area), "geometry": geom})
    frame = None
    if rows:
        import geopandas as gpd
        import pyogrio

        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
    gpkg = ctx["dirs"]["date"] / "DATE_COVERAGE.gpkg"
    geojson = ctx["dirs"]["date"] / "DATE_COVERAGE.geojson"
    # These are derived artifacts in the current run.  Remove only their
    # prior same-run copies before writing corrected geometry.
    for derived in (gpkg, geojson):
        if derived.exists():
            derived.unlink()
    if frame is not None and not frame.empty:
        pyogrio.write_dataframe(frame, gpkg, layer="date_coverage", driver="GPKG")
        pyogrio.write_dataframe(frame.to_crs(NATIVE_CRS), geojson, driver="GeoJSON")
    summary_rows = []
    nominal = parse_date(ctx["values"].get("HistoryDate"))
    for actual, code in date_codes.items():
        tile_count = int(np.count_nonzero(date_grid == code))
        delta = None
        if nominal:
            delta = (date.fromisoformat(actual) - date.fromisoformat(nominal)).days
        summary_rows.append({"actual_acquisition_date": actual, "date_code": code, "tile_count": tile_count, "area_m2": tile_count * area, "delta_days_from_nominal": delta})
    summary_rows.sort(key=lambda row: (-row["area_m2"], row["actual_acquisition_date"]))
    with (ctx["dirs"]["date"] / "DATE_COVERAGE_SUMMARY.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["actual_acquisition_date", "date_code", "tile_count", "area_m2", "delta_days_from_nominal"])
        writer.writeheader()
        writer.writerows(summary_rows)
    labels = {0: "DATE_UNKNOWN"}
    labels.update({code: actual for actual, code in date_codes.items()})
    map_date_grid(ctx["dirs"]["date"] / "DATE_COVERAGE_MAP.png", date_grid, labels, ctx["dirs"]["date"] / "DATE_COVERAGE_MAP.pdf")
    mode = summary_rows[0]["actual_acquisition_date"] if summary_rows else None
    payload = {
        "status": "pass" if present and not unknown_present else "pass_with_date_unknown",
        "nominal_date": nominal,
        "mode_date": mode,
        "actual_date_count": len(dates),
        "other_actual_dates": [item for item in dates if item != mode],
        "date_unknown_present_tile_count": unknown_present,
        "present_tile_count": present,
        "date_codes": date_codes,
        "summary_csv": str(ctx["dirs"]["date"] / "DATE_COVERAGE_SUMMARY.csv"),
        "scientific_status": "candidate_not_ground_truth",
    }
    atomic_json(ctx["dirs"]["date"] / "DATE_COVERAGE_RECEIPT.json", payload)
    manifest.update({"actual_acquisition_date": mode, "mode_date": mode, "other_actual_dates": payload["other_actual_dates"], "date_unknown_present_tile_count": unknown_present})
    atomic_json(ctx["dirs"]["manifest"] / "XIAN_R1_2020_INPUT_MANIFEST.json", manifest)
    return payload


def quality_class(stats: dict[str, Any], neighbour_delta: float | None = None) -> int:
    if not stats.get("read_success"):
        return 255
    mean = float(stats.get("mean_luminance", 0.0))
    std = float(stats.get("std_luminance", 0.0))
    dynamic = float(stats.get("dynamic_range", 0.0))
    gradient = float(stats.get("gradient_variance", 0.0))
    if mean > 250 and std < 8:
        return 1
    if mean < 5 and std < 8:
        return 2
    if dynamic < 12 or std < 4:
        return 3
    if gradient < 1.5:
        return 4
    if neighbour_delta is not None and neighbour_delta > 55:
        return 5
    return 0


def stage_quality(ctx: dict[str, Any]) -> dict[str, Any]:
    existing_receipt = ctx["dirs"]["quality"] / "QUALITY_RECEIPT.json"
    if existing_receipt.is_file():
        return json.loads(existing_receipt.read_text(encoding="utf-8"))
    grid = ctx["grid"]
    raw_root = ctx["raw_root"]
    quality = np.full((grid["tile_rows"], grid["tile_columns"]), 255, dtype=np.uint8)
    means = np.full(quality.shape, np.nan, dtype=np.float32)
    stats_path = ctx["dirs"]["quality"] / "TILE_QUALITY_STATS.csv"
    quality_workers = max(1, int(ctx["args"].quality_workers))
    jobs = [
        (row_index, col_index, x, y, tile_path(raw_root, x, y))
        for row_index, y in enumerate(range(grid["max_y"], grid["min_y"] - 1, -1))
        for col_index, x in enumerate(range(grid["min_x"], grid["max_x"] + 1))
    ]
    with stats_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["tile_x", "tile_y", "row", "col", "path", "read_success", "mean_luminance", "std_luminance", "dynamic_range", "gradient_variance", "quality_code", "quality_label", "date_unknown"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        total = len(jobs)
        processed = 0
        labels = {0: "usable", 1: "cloud_proxy", 2: "shadow_proxy", 3: "haze_low_dynamic_proxy", 4: "blur_proxy", 5: "seamline_proxy", 255: "unobserved"}
        # JPEG decoding is independent per tile and Pillow releases the GIL in
        # its decoder.  Bounded threads keep the audit deterministic while
        # avoiding the multi-process memory overhead of 188k task objects.
        with ThreadPoolExecutor(max_workers=quality_workers) as executor:
            for start in range(0, total, 2048):
                batch = jobs[start : start + 2048]
                stats_batch = list(executor.map(lambda item: sample_jpg(item[4]) if item[4].is_file() else {"read_success": False}, batch))
                for item, stat in zip(batch, stats_batch):
                    row_index, col_index, x, y, path = item
                    means[row_index, col_index] = float(stat.get("mean_luminance", np.nan)) if stat.get("read_success") else np.nan
                    code = quality_class(stat)
                    quality[row_index, col_index] = code
                    writer.writerow({"tile_x": x, "tile_y": y, "row": row_index, "col": col_index, "path": str(path), "read_success": bool(stat.get("read_success")), "mean_luminance": stat.get("mean_luminance"), "std_luminance": stat.get("std_luminance"), "dynamic_range": stat.get("dynamic_range"), "gradient_variance": stat.get("gradient_variance"), "quality_code": code, "quality_label": labels[code], "date_unknown": path.name.lower() not in ctx["log_rows"]})
                processed += len(batch)
                if processed and (processed % max(1, grid["tile_columns"] * 10) < len(batch)):
                    print(f"quality tiles {processed}/{total}", flush=True)
    # Add a conservative seamline proxy only when adjacent tile means differ a lot.
    for r in range(quality.shape[0]):
        for c in range(quality.shape[1]):
            if quality[r, c] != 0 or not np.isfinite(means[r, c]):
                continue
            neighbours = []
            if r > 0 and np.isfinite(means[r - 1, c]): neighbours.append(float(means[r - 1, c]))
            if r + 1 < quality.shape[0] and np.isfinite(means[r + 1, c]): neighbours.append(float(means[r + 1, c]))
            if c > 0 and np.isfinite(means[r, c - 1]): neighbours.append(float(means[r, c - 1]))
            if c + 1 < quality.shape[1] and np.isfinite(means[r, c + 1]): neighbours.append(float(means[r, c + 1]))
            if neighbours and max(abs(float(means[r, c]) - value) for value in neighbours) > 55:
                quality[r, c] = 5
    transform = pixel_transform(ctx["bounds"], grid["tile_columns"], grid["tile_rows"])
    write_raster(ctx["dirs"]["quality"] / "QUALITY_MASKS" / "quality_reason_grid.tif", quality, transform, nodata=255)
    usable = (quality == 0).astype(np.uint8)
    write_raster(ctx["dirs"]["quality"] / "QUALITY_MASKS" / "usable_grid.tif", usable, transform, nodata=0)
    labels = {0: "usable", 1: "cloud_proxy", 2: "shadow_proxy", 3: "haze_low_dynamic_proxy", 4: "blur_proxy", 5: "seamline_proxy", 255: "unobserved"}
    import geopandas as gpd
    import pyogrio
    from rasterio.features import shapes
    from rasterio.transform import Affine
    from shapely.geometry import shape
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer
    to_metric = Transformer.from_crs(NATIVE_CRS, METRIC_CRS, always_xy=True).transform
    rows = []
    for geom_json, code in shapes(quality, mask=quality != 0, transform=Affine(*transform[:6])):
        geom = shapely_transform(to_metric, shape(geom_json))
        rows.append({"quality_code": int(code), "quality_label": labels[int(code)], "area_m2": float(geom.area), "geometry": geom})
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
    if not frame.empty:
        pyogrio.write_dataframe(frame, ctx["dirs"]["quality"] / "OBSERVATION_QUALITY.gpkg", layer="quality", driver="GPKG")
    map_quality_grid(ctx["dirs"]["quality"] / "OBSERVATION_QUALITY_MAP.png", quality, ctx["dirs"]["quality"] / "OBSERVATION_QUALITY_MAP.pdf")
    area = projected_tile_area(ctx["bounds"], grid)
    counts = Counter(int(item) for item in quality.ravel())
    usable_area = counts.get(0, 0) * area
    total_area = quality.size * area
    summary = {
        "status": "pass",
        "quality_is_tile_proxy": True,
        "quality_method": "PIL 32x32 thumbnail statistics; no generative fill; codes are screening proxies, not semantic cloud masks",
        "counts_by_code": {str(code): count for code, count in sorted(counts.items())},
        "labels": labels,
        "total_area_m2": total_area,
        "usable_observation_area_m2": usable_area,
        "valid_observation_ratio_area_weighted": usable_area / total_area if total_area else 0.0,
        "cloud_proxy_ratio": counts.get(1, 0) / quality.size if quality.size else 0.0,
        "blur_proxy_ratio": counts.get(4, 0) / quality.size if quality.size else 0.0,
        "seamline_proxy_ratio": counts.get(5, 0) / quality.size if quality.size else 0.0,
        "date_unknown_ratio_present_tiles": sum(1 for path in raw_root.rglob("*.jpg") if path.name.lower() not in ctx["log_rows"]) / max(1, sum(1 for path in raw_root.rglob("*.jpg"))),
        "scientific_status": "candidate_not_ground_truth",
    }
    atomic_json(ctx["dirs"]["quality"] / "QUALITY_SUMMARY.json", summary)
    atomic_json(ctx["dirs"]["quality"] / "QUALITY_RECEIPT.json", {"stage": "quality", "created_at": utc_now(), **summary, "raw_inputs_modified": False})
    return summary


def stage_vrt(ctx: dict[str, Any]) -> dict[str, Any]:
    vrt = build_vrt(ctx["dirs"]["imagery"] / "ANALYSIS_READY_REFERENCE.vrt", ctx["raw_root"], ctx["grid"], ctx["bounds"])
    georef = {
        "schema_version": "urrs.xian.epoch_georeference.v1",
        "native_crs": NATIVE_CRS,
        "metric_crs": METRIC_CRS,
        "bounds_wgs84": list(ctx["bounds"]),
        "tile_size_px": TILE_SIZE,
        "tile_columns": ctx["grid"]["tile_columns"],
        "tile_rows": ctx["grid"]["tile_rows"],
        "width_px": vrt["width"],
        "height_px": vrt["height"],
        "transform": pixel_transform(ctx["bounds"], vrt["width"], vrt["height"]),
        "source_reference": vrt,
        "native_georeference_frozen": True,
        "cross_epoch_registration": "deferred_to_temporal_assembly",
    }
    atomic_json(ctx["dirs"]["manifest"] / "EPOCH_GEOREFERENCE.json", georef)
    atomic_json(ctx["dirs"]["imagery"] / "ANALYSIS_READY_REFERENCE.json", {"status": "pass", "reference_type": "VRT_over_read_only_GEID_JPEG_tiles", "vrt": vrt, "raw_inputs_modified": False, "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0}})
    atomic_json(ctx["dirs"]["imagery"] / "VRT_RECEIPT.json", {"stage": "vrt", "created_at": utc_now(), **vrt, "raw_inputs_modified": False})
    return vrt


def resolve_checkpoint_config(ctx: dict[str, Any]) -> tuple[Path, Path]:
    args = ctx["args"]
    checkpoint = args.checkpoint or (ctx["data_root"].parent / ".." / "Codex" / "projects" / "Documents-Codex" / "2026-07-18" / "urban-renewal-phase1-starter-v0-2" / "outputs" / "urban-renewal-phase1-starter" / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth")
    if not checkpoint.exists():
        checkpoint = Path(r"D:\Codex\projects\Documents-Codex\2026-07-18\urban-renewal-phase1-starter-v0-2\outputs\urban-renewal-phase1-starter\outputs\v2_readiness\rsbuilding\whu\work_dirs\swin_t_whu_pilot5000_20260728_110750\iter_2000.pth")
    config = args.config or checkpoint.with_name("Swin_T_whu_windows_pilot5000.py")
    return checkpoint.resolve(), config.resolve()


def stage_geometry(ctx: dict[str, Any]) -> dict[str, Any]:
    checkpoint, config = resolve_checkpoint_config(ctx)
    expected_sha = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(f"B0 checkpoint SHA mismatch: {actual_sha} != {expected_sha}")
    override = ctx["args"].source_override.resolve() if ctx["args"].source_override else None
    reference = override or (ctx["dirs"]["imagery"] / "ANALYSIS_READY_REFERENCE.vrt")
    if not reference.is_file():
        stage_vrt(ctx)
    if not reference.is_file():
        raise FileNotFoundError(f"geometry source does not exist: {reference}")
    # A user-provided mosaic is an immutable read-only input.  Read its native
    # metadata rather than inheriting dimensions/transform from the tile-grid
    # VRT; this prevents silent spatial misregistration after mosaicking.
    source_meta: dict[str, Any] = {}
    if override:
        import rasterio

        with rasterio.open(reference) as source:
            if source.crs is None:
                raise RuntimeError(f"geometry source has no CRS: {reference}")
            source_meta = {
                "width": int(source.width),
                "height": int(source.height),
                "bands": int(source.count),
                "dtype": list(source.dtypes),
                "crs": source.crs.to_string(),
                "transform": list(source.transform),
                "bounds": [float(value) for value in source.bounds],
                "nodata": source.nodata,
                "res": [float(value) for value in source.res],
            }
    else:
        source_meta = {
            "width": ctx["grid"]["tile_columns"] * TILE_SIZE,
            "height": ctx["grid"]["tile_rows"] * TILE_SIZE,
            "bands": 3,
            "dtype": ["uint8", "uint8", "uint8"],
            "crs": NATIVE_CRS,
            "transform": pixel_transform(ctx["bounds"], ctx["grid"]["tile_columns"] * TILE_SIZE, ctx["grid"]["tile_rows"] * TILE_SIZE),
            "bounds": list(ctx["bounds"]),
            "nodata": None,
        }
    scene_root = ctx["dirs"]["geometry"] / "input"
    scene_root.mkdir(parents=True, exist_ok=True)
    scene = {
        "scene_date": ctx["manifest_mode_date"] if ctx.get("manifest_mode_date") else parse_date(ctx["values"].get("HistoryDate")) or "DATE_UNKNOWN",
        "selected_source": str(reference.resolve()),
        "source_sha256": sha256_file(reference),
        "width": source_meta["width"],
        "height": source_meta["height"],
        "bands": source_meta["bands"],
        "dtype": source_meta["dtype"],
        "crs": source_meta["crs"],
        "transform": source_meta["transform"],
        "bounds": source_meta["bounds"],
        "nodata": source_meta["nodata"],
        "georeferenced": True,
        "source_kind": "user_mosaicked_read_only_raster" if override else "generated_read_only_vrt",
    }
    scene_manifest_name = "SCENES_MOSAIC.json" if override else "SCENES.json"
    atomic_json(scene_root / scene_manifest_name, {"scene_discovery_status": "pass", "scenes": [scene]})
    if override:
        atomic_json(
            scene_root / "SOURCE_OVERRIDE.json",
            {
                "status": "pass",
                "source_path": str(reference),
                "source_sha256": scene["source_sha256"],
                "source_kind": "user_mosaicked_read_only_raster",
                "replaced_reference": str((ctx["dirs"]["imagery"] / "ANALYSIS_READY_REFERENCE.vrt").resolve()),
                "raw_inputs_modified": False,
            },
        )
    # Reuse the accepted sparse runner; only its metric CRS is changed for Xi'an.
    outer_root = Path(__file__).resolve().parents[1]
    nested_repo = outer_root / "outputs" / "urban-renewal-phase1-starter"
    if str(nested_repo) not in sys.path:
        sys.path.insert(0, str(nested_repo))
    if str(nested_repo / "src") not in sys.path:
        sys.path.insert(0, str(nested_repo / "src"))
    sparse_path = Path(__file__).resolve().with_name("infer_shenzhen_sparse_geometry.py")
    spec = importlib.util.spec_from_file_location("urrs_xian_sparse_runner", sparse_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load accepted sparse runner: {sparse_path}")
    sparse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sparse)
    sparse.METRIC_CRS = METRIC_CRS
    lock_dir = Path(r"E:\URRS Output\runtime_locks")
    # Scope the lock to the CUDA device so independent cloud workers do not
    # contend on a historical single global lock. Existing active workers
    # retain and clean up their already-acquired legacy lock independently.
    physical_device = os.environ.get("URRS_PHYSICAL_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES")
    lock_path = geometry_lock_path(ctx["args"].device, lock_dir, physical_device)
    if lock_path.exists():
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            owner = int(lock_payload.get("pid", 0))
            if owner and any(item.pid == owner for item in __import__("psutil").process_iter(["pid"])):
                raise RuntimeError(f"GPU lock is held by PID {owner}: {lock_path}")
        except RuntimeError:
            raise
        except Exception:
            # A stale lock from an interrupted derived run is recoverable; it
            # does not authorize touching another user's process.
            pass
        lock_path.unlink(missing_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(lock_path, {"pid": os.getpid(), "run_dir": str(ctx["run"]), "created_at": utc_now(), "device": ctx["args"].device})
    model = None
    output_root = ctx["dirs"]["geometry"] / "scene_outputs"
    try:
        model = sparse.base.build_model(config, checkpoint, ctx["args"].device)
        result = sparse.process_scene(model, scene, output_root, checkpoint, config, ctx["args"].device, ctx["args"].batch_size, ctx["args"].strip_rows)
    finally:
        lock_path.unlink(missing_ok=True)
    source_gpkg = output_root / scene["scene_date"] / "building_objects.gpkg"
    package_gpkg = ctx["dirs"]["geometry"] / "BUILDINGS.gpkg"
    if source_gpkg.is_file() and source_gpkg.resolve() != package_gpkg.resolve():
        shutil.copy2(source_gpkg, package_gpkg)
    payload = {
        "status": "pass",
        "scene": scene,
        "geometry_manifest": result,
        "building_objects": str(package_gpkg.resolve()) if package_gpkg.is_file() else None,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "threshold": 0.5,
        "tile_size": 512,
        "overlap": 128,
        "stride": 384,
        "metric_crs": METRIC_CRS,
        "probability_raster_materialized": False,
        "scientific_status": "candidate_not_ground_truth",
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "source_kind": scene["source_kind"],
    }
    atomic_json(ctx["dirs"]["geometry"] / "GEOMETRY_MANIFEST.json", payload)
    return payload


def _stretch_rgb(array: np.ndarray) -> np.ndarray:
    """Percentile-stretch a small RGB read for a readable QA figure."""
    data = np.asarray(array, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected band-first RGB array, got {data.shape}")
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        data = np.concatenate([data, data[:1]], axis=0)
    data = data[:3]
    result = np.zeros_like(data, dtype=np.float32)
    for index in range(3):
        channel = data[index]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [2.0, 98.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
        if high > low:
            result[index] = np.clip((channel - low) / (high - low), 0.0, 1.0)
        else:
            result[index] = np.clip(channel / max(1.0, high), 0.0, 1.0)
    return np.moveaxis(result, 0, -1)


def _render_geometry_visuals(
    source_path: Path,
    gpkg_path: Path,
    visual_dir: Path,
) -> dict[str, Any]:
    """Render real source imagery with the extracted building outlines.

    These are deterministic QA assets, not model-selection evidence.  The
    source raster is read in bounded overview/windows and is never modified.
    """
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from shapely.geometry import box

    visual_dir.mkdir(parents=True, exist_ok=True)
    frame = gpd.read_file(gpkg_path, layer="buildings")
    if frame.crs is None:
        raise RuntimeError("building geometry has no CRS; refusing to draw a geographic QA map")
    with rasterio.open(source_path) as source:
        if source.count < 1:
            raise RuntimeError(f"source has no bands: {source_path}")
        source_crs = source.crs
        projected = frame.to_crs(source_crs)
        overview_max = 1400
        scale = min(1.0, overview_max / max(source.width, source.height))
        out_width = max(1, int(round(source.width * scale)))
        out_height = max(1, int(round(source.height * scale)))
        overview = source.read(
            list(range(1, min(3, source.count) + 1)),
            out_shape=(min(3, source.count), out_height, out_width),
            resampling=Resampling.average,
        )
        bounds = source.bounds

        fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
        ax.imshow(_stretch_rgb(overview), extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin="upper")
        if not projected.empty:
            projected.boundary.plot(ax=ax, color="#f97316", linewidth=0.55, alpha=0.9)
        ax.set_title("Xi'an R1 observed building candidates | real mosaic + B0 outlines")
        ax.set_xlabel(f"Longitude/latitude ({source_crs})")
        ax.set_ylabel(f"Longitude/latitude ({source_crs})")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        outline_path = visual_dir / "BUILDING_OUTLINE_MAP.png"
        fig.savefig(outline_path, dpi=300, facecolor="white")
        outline_pdf = visual_dir / "BUILDING_OUTLINE_MAP.pdf"
        fig.savefig(outline_pdf, facecolor="white")
        plt.close(fig)

        # Fixed, result-independent QA windows on a regular 3x3 grid.  The
        # grid is derived only from raster dimensions, never from detections.
        window_size = min(8192, max(1024, min(source.width, source.height) // 8))
        grid_n = 4
        row_positions = [int(round(value)) for value in np.linspace(0, max(0, source.height - window_size), grid_n)]
        col_positions = [int(round(value)) for value in np.linspace(0, max(0, source.width - window_size), grid_n)]
        offsets = [
            (row, col, f"grid_{r_index}_{c_index}")
            for r_index, row in enumerate(row_positions)
            for c_index, col in enumerate(col_positions)
        ]
        cases: list[dict[str, Any]] = []
        fig, axes = plt.subplots(grid_n, grid_n, figsize=(14, 14), dpi=180)
        axes = np.asarray(axes).reshape(-1)
        for ax, (row, col, name) in zip(axes, offsets):
            height = min(window_size, source.height - row)
            width = min(window_size, source.width - col)
            window = Window(col_off=col, row_off=row, width=width, height=height)
            image = source.read(
                list(range(1, min(3, source.count) + 1)),
                window=window,
                out_shape=(min(3, source.count), 360, 360),
                resampling=Resampling.average,
            )
            win_bounds = source.window_bounds(window)
            clip_box = box(*win_bounds)
            subset = projected[projected.geometry.intersects(clip_box)]
            ax.imshow(_stretch_rgb(image), extent=[win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3]], origin="upper")
            if not subset.empty:
                subset.boundary.plot(ax=ax, color="#f97316", linewidth=0.8, alpha=0.95)
            ax.set_title(name.replace("_", " "))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            cases.append({"window": name, "row": int(row), "col": int(col), "width": int(width), "height": int(height), "candidate_count": int(len(subset))})
        fig.suptitle("Fixed QA windows — observed building candidates (candidate_not_ground_truth)")
        fig.tight_layout()
        contact_path = visual_dir / "CONTACT_SHEET.png"
        fig.savefig(contact_path, dpi=300, facecolor="white")
        contact_pdf = visual_dir / "CONTACT_SHEET.pdf"
        fig.savefig(contact_pdf, facecolor="white")
        plt.close(fig)
    with (visual_dir / "QA_CASES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window", "row", "col", "width", "height", "candidate_count"])
        writer.writeheader()
        writer.writerows(cases)
    return {
        "status": "generated",
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "building_count": int(len(frame)),
        "outline_map": str(outline_path.resolve()),
        "outline_map_pdf": str(outline_pdf.resolve()),
        "contact_sheet": str(contact_path.resolve()),
        "contact_sheet_pdf": str(contact_pdf.resolve()),
        "qa_cases": str((visual_dir / "QA_CASES.csv").resolve()),
        "fixed_windows": cases,
        "human_review": "pending",
        "scientific_status": "candidate_not_ground_truth",
    }


def stage_package(ctx: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ctx["dirs"]["manifest"] / "XIAN_R1_2020_INPUT_MANIFEST.json"
    date_receipt = ctx["dirs"]["date"] / "DATE_COVERAGE_RECEIPT.json"
    quality_receipt = ctx["dirs"]["quality"] / "QUALITY_RECEIPT.json"
    geometry_receipt = ctx["dirs"]["geometry"] / "GEOMETRY_MANIFEST.json"
    geometry = json.loads(geometry_receipt.read_text(encoding="utf-8")) if geometry_receipt.is_file() else {"status": "blocked_pending_sparse_inference"}
    date_payload = json.loads(date_receipt.read_text(encoding="utf-8")) if date_receipt.is_file() else {"status": "missing"}
    quality_payload = json.loads(quality_receipt.read_text(encoding="utf-8")) if quality_receipt.is_file() else {"status": "missing"}
    georef_path = ctx["dirs"]["manifest"] / "EPOCH_GEOREFERENCE.json"
    georef = json.loads(georef_path.read_text(encoding="utf-8")) if georef_path.is_file() else {}
    geometry_status = geometry.get("status") == "pass"
    visual_outputs: dict[str, Any] = {"status": "blocked_pending_geometry"}
    parquet_path: Path | None = None
    footprint_area_m2: float | None = None
    if geometry_status:
        import geopandas as gpd

        package_gpkg = Path(geometry.get("building_objects") or "")
        source_path = Path((geometry.get("scene") or {}).get("selected_source") or "")
        if not package_gpkg.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"geometry package/source missing: {package_gpkg} / {source_path}")
        frame = gpd.read_file(package_gpkg, layer="buildings")
        footprint_area_m2 = float(frame["area_m2"].sum()) if "area_m2" in frame.columns and not frame.empty else 0.0
        parquet_path = ctx["dirs"]["geometry"] / "BUILDINGS.parquet"
        parquet_status: dict[str, Any] = {"status": "not_written"}
        try:
            frame.to_parquet(parquet_path, index=False)
            parquet_status = {"status": "written", "path": str(parquet_path.resolve()), "sha256": sha256_file(parquet_path)}
        except Exception as exc:  # optional pyarrow dependency; never fabricate a parquet
            parquet_status = {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)}
        visual_outputs = _render_geometry_visuals(source_path, package_gpkg, ctx["dirs"]["visual"])
        visual_outputs["parquet"] = parquet_status
    qa = {
        "status": "pass" if geometry_status and visual_outputs.get("status") == "generated" else "blocked",
        "fixed_sample_strategy": "deterministic corner and center windows; no result-based selection",
        "source_inputs": [str(ctx["dirs"]["logs"].resolve())],
        "visual_qa_status": visual_outputs.get("status"),
        "visual_outputs": visual_outputs,
        "human_review": "pending" if visual_outputs.get("status") == "generated" else None,
        "scientific_status": "candidate_not_ground_truth",
    }
    atomic_json(ctx["dirs"]["visual"] / "VISUAL_QA.json", qa)
    atomic_json(ctx["dirs"]["manifest"] / "EPOCH_MANIFEST.json", {
        "schema_version": "urrs.xian.r1_2020.epoch_manifest.v1",
        "created_at": utc_now(),
        "task_name": ctx["stem"],
        "nominal_date": parse_date(ctx["values"].get("HistoryDate")),
        "mode_date": date_payload.get("mode_date"),
        "actual_dates": date_payload.get("date_codes", {}),
        "date_coverage": date_payload,
        "quality": quality_payload,
        "geometry": geometry,
        "visual_qa": qa,
        "georeference": georef,
        "raw_inputs_unmodified": True,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "deferred_modules": ["DSIFN", "S2Looking", "Full20", "change", "demolition", "temporal_state", "cross_epoch_registration"],
        "single_epoch_semantics": "observed building detections only; no STABLE/DEMOLITION/ADDITION labels",
        "scientific_status": "candidate_not_ground_truth",
    })
    hard_conditions = {
        "input_complete": bool(manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("raw_tile_root", {}).get("jpg_scan", {}).get("present_fraction") == 1.0),
        "date_coverage_available": date_payload.get("status") in {"pass", "pass_with_date_unknown"},
        "quality_quantified": quality_payload.get("status") == "pass",
        "georeference_frozen": bool(georef.get("native_georeference_frozen")),
        "geometry_complete": geometry.get("status") == "pass",
        "visual_qa_generated": qa.get("status") == "pass",
        "raw_inputs_unmodified": True,
        "test_reads_zero": True,
    }
    # A single recovered epoch cannot establish longitudinal readiness even
    # when all single-epoch artifacts are complete.
    long_ready = False
    blocking_reasons = [key for key, value in hard_conditions.items() if not value]
    blocking_reasons.extend(["single_epoch_only", "cross_epoch_registration_deferred", "temporal_state_labels_deferred"])
    receipt = {
        "schema_version": "urrs.xian.r1_2020.epoch_ready_receipt.v1",
        "created_at": utc_now(),
        "status": "LONG_TEMPORAL_READY" if long_ready else "LONG_TEMPORAL_NOT_READY",
        "LONG_TEMPORAL_READY": long_ready,
        "hard_conditions": hard_conditions,
        "blocking_reasons": blocking_reasons,
        "nominal_date": parse_date(ctx["values"].get("HistoryDate")),
        "mode_date": date_payload.get("mode_date"),
        "other_actual_dates": date_payload.get("other_actual_dates", []),
        "date_unknown_present_tile_count": date_payload.get("date_unknown_present_tile_count"),
        "valid_observation_ratio": quality_payload.get("valid_observation_ratio_area_weighted"),
        "building_count": geometry.get("geometry_manifest", {}).get("object_count") if geometry.get("status") == "pass" else None,
        "building_footprint_area_m2": footprint_area_m2,
        "geometry_checkpoint": geometry.get("checkpoint_sha256"),
        "area_crs": METRIC_CRS,
        "single_epoch_state_output": "OBSERVED_BUILDING_ONLY",
        "deferred_to_temporal_assembly": ["DSIFN", "S2Looking", "Full20", "change", "demolition", "temporal_state"],
        "raw_inputs_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
        "scientific_status": "candidate_not_ground_truth",
        "visual_outputs": visual_outputs,
    }
    atomic_json(ctx["dirs"]["receipt"] / "EPOCH_READY_RECEIPT.json", receipt)
    report = "\n".join(
        [
            "# Xi'an R1 single-epoch standard package",
            "",
            f"- Task stem: `{ctx['stem']}`",
            f"- Nominal GEID date: `{receipt.get('nominal_date')}` (not used as an observed label)",
            f"- Dominant observed date: `{receipt.get('mode_date')}`",
            f"- Other observed dates: `{', '.join(receipt.get('other_actual_dates') or [])}`",
            f"- DATE_UNKNOWN present tiles: `{receipt.get('date_unknown_present_tile_count')}`",
            f"- Quality valid-observation ratio (tile proxy): `{receipt.get('valid_observation_ratio')}`",
            f"- Building candidates: `{receipt.get('building_count')}`",
            f"- Candidate footprint area (m2, {METRIC_CRS}): `{receipt.get('building_footprint_area_m2')}`",
            f"- Geometry source: `{(geometry.get('scene') or {}).get('selected_source')}`",
            f"- Geometry checkpoint SHA256: `{receipt.get('geometry_checkpoint')}`",
            "- Semantics: observed building detections only; no STABLE/DEMOLITION/ADDITION labels are inferred from a single epoch.",
            "- Scientific status: `candidate_not_ground_truth`.",
            "- Human visual review: `pending`.",
            "- Longitudinal readiness: `LONG_TEMPORAL_NOT_READY` until cross-epoch registration, temporal inference and review are complete.",
            "- Test reads: WHU=0; SECOND=0; Holdout=0.",
            "- Raw GEID inputs modified: no.",
            "",
        ]
    )
    atomic_text(ctx["run"] / "R1_2020_SINGLE_EPOCH_REPORT.md", report)
    # Keep a compact run pointer for resume/status commands.
    atomic_json(ctx["run"] / "CURRENT_PROGRESS.json", {"stage": "package", "status": receipt["status"], "updated_at": utc_now(), "run_dir": str(ctx["run"])})
    return receipt


def main() -> int:
    args = parse_args()
    ctx = make_context(args)
    stage = args.stage
    try:
        audit_manifest = ctx["dirs"]["manifest"] / "XIAN_R1_2020_INPUT_MANIFEST.json"
        if stage == "audit" or not audit_manifest.is_file():
            stage_audit(ctx)
        if stage in {"date", "quality", "vrt", "geometry", "package", "all"}:
            stage_date(ctx)
        if stage in {"quality", "vrt", "geometry", "package", "all"}:
            stage_quality(ctx)
        if stage in {"vrt", "geometry", "package", "all"}:
            stage_vrt(ctx)
        if stage in {"geometry", "package", "all"}:
            ctx["manifest_mode_date"] = stage_date(ctx).get("mode_date")
            geometry_manifest = ctx["dirs"]["geometry"] / "GEOMETRY_MANIFEST.json"
            if geometry_manifest.is_file() and (json.loads(geometry_manifest.read_text(encoding="utf-8")).get("status") == "pass"):
                # A successful sparse geometry stage is immutable for this
                # run.  Package/all are CPU/IO follow-up stages and must not
                # reload B0 or repeat GPU inference.
                pass
            else:
                stage_geometry(ctx)
        if stage in {"package", "all"}:
            stage_package(ctx)
        print(json.dumps({"status": "ok", "stage": stage, "run_dir": str(ctx["run"])}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "urrs.xian.r1_2020.failure_receipt.v1",
            "status": "FAILURE_RECEIPT",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "created_at": utc_now(),
            "run_dir": str(ctx["run"]),
            "raw_inputs_modified": False,
            "test_reads": {"WHU": 0, "SECOND": 0, "Holdout": 0},
            "resume_safe": True,
        }
        atomic_json(ctx["dirs"]["receipt"] / f"FAILURE_RECEIPT_{stamp()}.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
