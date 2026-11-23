"""Materialise one audited Xi'an VRT into an independent GeoTIFF.

This is a derived-data helper only.  It never changes the source VRT/JPEG
tiles, and writes a resumable ``.part.tif`` plus a small state receipt.  Zero
RGB blocks are marked invalid in the derived mask so known missing VRT tiles
cannot be treated as observed pixels by the sparse geometry runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
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
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vrt", required=True, type=Path)
    parser.add_argument("--output-tif", required=True, type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--compress", default="JPEG", choices=("JPEG", "DEFLATE", "ZSTD"))
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    source = args.source_vrt.resolve()
    output = args.output_tif.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.rows < 256:
        raise ValueError("--rows must be >= 256")
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part.tif")
    state_path = output.with_suffix(output.suffix + ".state.json")
    source_digest = sha256(source)

    with rasterio.open(source) as src:
        if src.count < 3 or not src.crs:
            raise RuntimeError(f"source must be georeferenced RGB VRT: {source}")
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=src.width,
            height=src.height,
            count=3,
            dtype="uint8",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="YES",
            compress=args.compress,
            predictor=1,
            nodata=0,
            interleave="pixel",
        )
        if args.compress == "JPEG":
            profile.update(photometric="YCBCR", jpeg_quality=max(1, min(100, args.jpeg_quality)))
        total_strips = (src.height + args.rows - 1) // args.rows
        completed_rows = 0
        if state_path.is_file() and part.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            if state.get("source_sha256") == source_digest and state.get("width") == src.width and state.get("height") == src.height and state.get("rows") == args.rows:
                completed_rows = int(state.get("completed_rows", 0))
        mode = "r+" if part.is_file() and completed_rows else "w"
        with rasterio.open(part, mode, **(profile if mode == "w" else {})) as dst:
            if mode == "w":
                dst.update_tags(
                    actual_date=args.date,
                    source_vrt=str(source),
                    source_vrt_sha256=source_digest,
                    derived_from_vrt="true",
                    zero_rgb_marked_nodata="true",
                )
            for row0 in range(completed_rows, src.height, args.rows):
                height = min(args.rows, src.height - row0)
                window = Window(0, row0, src.width, height)
                data = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                valid = np.any(data != 0, axis=0)
                dst.write(data, window=window)
                dst.write_mask((valid.astype(np.uint8) * 255), window=window)
                completed_rows = row0 + height
                atomic_json(
                    state_path,
                    {
                        "status": "running" if completed_rows < src.height else "complete",
                        "date": args.date,
                        "source_vrt": str(source),
                        "source_sha256": source_digest,
                        "output_part": str(part),
                        "output_tif": str(output),
                        "width": src.width,
                        "height": src.height,
                        "rows": args.rows,
                        "completed_rows": completed_rows,
                        "total_strips": total_strips,
                        "progress_pct": round(100.0 * completed_rows / src.height, 4),
                        "updated_at": now(),
                    },
                )
                print(f"{args.date} materialize rows {completed_rows}/{src.height}", flush=True)
    if completed_rows != src.height:
        raise RuntimeError("materialisation ended before all rows were written")
    os.replace(part, output)
    final_sha = sha256(output)
    atomic_json(
        state_path,
        {
            "status": "complete",
            "date": args.date,
            "source_vrt": str(source),
            "source_sha256": source_digest,
            "output_tif": str(output),
            "output_sha256": final_sha,
            "output_bytes": output.stat().st_size,
            "width": profile["width"],
            "height": profile["height"],
            "rows": args.rows,
            "completed_rows": profile["height"],
            "updated_at": now(),
        },
    )
    print(json.dumps({"status": "complete", "output": str(output), "sha256": final_sha}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
