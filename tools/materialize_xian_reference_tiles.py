"""Materialise an Xi'an VRT by streaming its source JPEG tiles.

The large VRTs are valid, but some GDAL builds can stall when asked to read a
full-width 1024-row window.  This fallback parses only the first VRT raster
band (the RGB source filenames are repeated for the other bands) and writes
each 256x256 source tile to an independent derived GeoTIFF.  Source files and
the VRT are read-only; missing/corrupt tiles remain nodata and are counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from affine import Affine
from rasterio.windows import Window


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_vrt(vrt: Path) -> tuple[dict[str, object], list[tuple[int, int, int, int, str]]]:
    size_re = re.compile(r'<VRTDataset\s+rasterXSize="(\d+)"\s+rasterYSize="(\d+)"')
    gt_re = re.compile(r"<GeoTransform>(.*?)</GeoTransform>")
    srs_re = re.compile(r"<SRS>(.*?)</SRS>")
    source_re = re.compile(r"<SourceFilename[^>]*>(.*?)</SourceFilename>")
    dst_re = re.compile(r'<DstRect\s+xOff="(\d+)"\s+yOff="(\d+)"\s+xSize="(\d+)"\s+ySize="(\d+)"')
    metadata: dict[str, object] = {}
    tiles: list[tuple[int, int, int, int, str]] = []
    in_first_band = False
    current_source: str | None = None
    # Line streaming avoids loading the 300+ MB VRT XML into memory.  The
    # other two bands repeat the same source filenames, so stop at band 2.
    with vrt.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not metadata:
                match = size_re.search(line)
                if match:
                    metadata.update(width=int(match.group(1)), height=int(match.group(2)))
            if "<GeoTransform>" in line:
                match = gt_re.search(line)
                if match:
                    metadata["transform"] = tuple(float(v.strip()) for v in match.group(1).split(","))
            if "<SRS>" in line:
                match = srs_re.search(line)
                if match:
                    metadata["crs"] = match.group(1).strip()
            if '<VRTRasterBand' in line and 'band="1"' in line:
                in_first_band = True
                continue
            if in_first_band and '<VRTRasterBand' in line and 'band="2"' in line:
                break
            if not in_first_band:
                continue
            match = source_re.search(line)
            if match:
                current_source = match.group(1).strip()
                continue
            match = dst_re.search(line)
            if match and current_source:
                tiles.append((int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), current_source))
                current_source = None
    if not metadata.get("width") or not metadata.get("height") or not tiles:
        raise RuntimeError(f"could not parse first RGB band from {vrt}")
    metadata.setdefault("crs", "EPSG:4326")
    if "transform" not in metadata:
        raise RuntimeError(f"VRT has no GeoTransform: {vrt}")
    return metadata, tiles


def affine_from_gdal(values: tuple[float, ...]) -> Affine:
    """Convert a parsed six-value GDAL geotransform to Rasterio Affine."""
    if len(values) != 6:
        raise ValueError(f"GDAL geotransform must have six values, got {len(values)}")
    return Affine.from_gdal(*values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vrt", required=True, type=Path)
    parser.add_argument("--output-tif", required=True, type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()
    source = args.source_vrt.resolve()
    output = args.output_tif.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part.tif")
    state_path = output.with_suffix(output.suffix + ".tiles.state.json")
    source_digest = sha256(source)
    metadata, tiles = parse_vrt(source)
    width = int(metadata["width"])
    height = int(metadata["height"])
    transform = metadata["transform"]
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint8",
        "crs": metadata["crs"],
        # ``parse_vrt`` returns the six coefficients in GDAL order
        # (origin_x, pixel_width, rotation_x, origin_y, rotation_y,
        # pixel_height).  Passing that tuple directly to ``Affine`` swaps
        # the origin/scale terms and produces a zero y scale.  Rasterio then
        # fails with ``ZeroDivisionError`` when a boundless window is read.
        # Use the explicit GDAL conversion so the derived GeoTIFF remains
        # georeferenced and safe for cloud geometry inference.
        "transform": affine_from_gdal(transform),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "YES",
        "compress": "JPEG",
        "jpeg_quality": 90,
        "interleave": "pixel",
        "nodata": 0,
    }
    state = {}
    if state_path.is_file() and part.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    start = 0
    if state.get("source_sha256") == source_digest and state.get("tile_count") == len(tiles):
        start = int(state.get("completed_tiles", 0))
    counters = {"written_tiles": int(state.get("written_tiles", 0)), "missing_tiles": int(state.get("missing_tiles", 0)), "bad_tiles": int(state.get("bad_tiles", 0))}
    mode = "r+" if part.is_file() and start else "w"
    with rasterio.open(part, mode, **(profile if mode == "w" else {})) as dst:
        if mode == "w":
            dst.update_tags(actual_date=args.date, source_vrt=str(source), source_vrt_sha256=source_digest, derived_from_vrt="true", tile_stream_fallback="true")
        for index in range(start, len(tiles)):
            x, y, tile_w, tile_h, filename = tiles[index]
            path = Path(filename)
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if image.size != (tile_w, tile_h):
                        raise ValueError(f"unexpected JPEG size {image.size} != {(tile_w, tile_h)}")
                    array = np.asarray(image, dtype=np.uint8)
                data = np.moveaxis(array, -1, 0)
                valid = bool(np.any(data != 0))
                dst.write(data, window=Window(x, y, tile_w, tile_h))
                dst.write_mask(np.full((tile_h, tile_w), 255 if valid else 0, dtype=np.uint8), window=Window(x, y, tile_w, tile_h))
                counters["written_tiles"] += 1
            except FileNotFoundError:
                counters["missing_tiles"] += 1
            except Exception:
                counters["bad_tiles"] += 1
            completed = index + 1
            if completed % args.checkpoint_every == 0 or completed == len(tiles):
                atomic_json(state_path, {"status": "running" if completed < len(tiles) else "complete", "date": args.date, "source_vrt": str(source), "source_sha256": source_digest, "output_part": str(part), "output_tif": str(output), "width": width, "height": height, "tile_count": len(tiles), "completed_tiles": completed, **counters, "progress_pct": round(completed * 100.0 / len(tiles), 4), "updated_at": now()})
                print(f"{args.date} materialize tiles {completed}/{len(tiles)} written={counters['written_tiles']} missing={counters['missing_tiles']} bad={counters['bad_tiles']}", flush=True)
    os.replace(part, output)
    final_sha = sha256(output)
    atomic_json(state_path, {"status": "complete", "date": args.date, "source_vrt": str(source), "source_sha256": source_digest, "output_tif": str(output), "output_sha256": final_sha, "output_bytes": output.stat().st_size, "width": width, "height": height, "tile_count": len(tiles), "completed_tiles": len(tiles), **counters, "updated_at": now()})
    print(json.dumps({"status": "complete", "output": str(output), "sha256": final_sha, **counters}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
