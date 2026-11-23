"""Completeness / QA scan for candidate raster products.

An epoch raster is only complete when the labelled data actually reaches the
last stripe of the grid.  A killed writer leaves a file that opens fine and has
the right dimensions but has never been written below the abort row, so
existence + dtype + shape is NOT a sufficient gate.

This tool reports, per raster:
  * grid identity vs GRID_METADATA.json
  * per 512-row stripe: number of non-zero pixels
  * the last stripe that contains any non-zero pixel
  * row/col coverage of the non-zero extent
  * nodata / dtype / compression

Usage:
  python xian_raster_completeness_audit.py --glob "*2020*" --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work" / "python-packages"))

import numpy as np
import rasterio
from rasterio.windows import Window

OUT_DIR = Path(r"F:\xian\西安_当前最佳候选交付_20260917\01_raster_products")


def stripe_scan(path: Path, stripe_rows: int = 512) -> dict:
    with rasterio.open(path) as src:
        height, width = src.height, src.width
        n_stripes = (height + stripe_rows - 1) // stripe_rows
        counts = []
        rows_with_data = []
        for idx, row_off in enumerate(range(0, height, stripe_rows), start=1):
            h = min(stripe_rows, height - row_off)
            arr = src.read(1, window=Window(0, row_off, width, h))
            nonzero = int(np.count_nonzero(arr))
            counts.append(nonzero)
            if nonzero:
                rows_with_data.append(idx)
            # also record the deepest true data row inside this stripe
        flat = np.array(counts)
        last_stripe = rows_with_data[-1] if rows_with_data else 0
        maxv = None
        minv = None
        if rows_with_data:
            sample = src.read(1, window=Window(0, (last_stripe - 1) * stripe_rows, width,
                                                min(stripe_rows, height - (last_stripe - 1) * stripe_rows)))
            nz = sample[sample > 0]
            if nz.size:
                maxv, minv = int(nz.max()), int(nz.min())
        return {
            "driver": src.driver,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "compression": str(src.compression) if src.compression else "",
            "width": width,
            "height": height,
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "n_stripes": n_stripes,
            "stripes_with_data": len(rows_with_data),
            "first_stripe_with_data": rows_with_data[0] if rows_with_data else 0,
            "last_stripe_with_data": last_stripe,
            "reaches_last_stripe": bool(rows_with_data and last_stripe == n_stripes),
            "empty_stripes_before_last": int(
                sum(1 for i in range(1, last_stripe) if i not in set(rows_with_data))
            ),
            "nonzero_total": int(flat.sum()),
            "value_min_in_last_stripe": minv,
            "value_max_in_last_stripe": maxv,
            "stripe_nonzero": flat.tolist(),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.tif")
    ap.add_argument("--csv", default="")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--stripe-rows", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    grid = json.loads((out_dir / "GRID_METADATA.json").read_text(encoding="utf-8"))
    gshape = (grid["width"], grid["height"])
    files = sorted(out_dir.glob(args.glob))
    rows = []
    for f in files:
        if not f.is_file():
            continue
        rec = stripe_scan(f, args.stripe_rows)
        rec["file"] = f.name
        rec["path"] = str(f)
        rec["size_bytes"] = f.stat().st_size
        rec["grid_match"] = (rec["width"], rec["height"]) == gshape
        rec["GATE"] = (
            "PASS"
            if rec["grid_match"] and rec["reaches_last_stripe"] and rec["nonzero_total"] > 0
            else "INCOMPLETE_OR_FAIL"
        )
        rows.append(rec)
        print(
            f"{rec['GATE']:<20} {f.name:<52} stripes {rec['first_stripe_with_data']:>3}-"
            f"{rec['last_stripe_with_data']:>3}/{rec['n_stripes']:<3} "
            f"nz={rec['nonzero_total']:>12,} dtype={rec['dtype']:<6} nodata={rec['nodata']}"
        )
    if args.csv:
        with Path(args.csv).open("w", encoding="utf-8-sig", newline="") as f:
            keys = [k for k in rows[0].keys() if k != "stripe_nonzero"] if rows else []
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        Path(args.csv).with_suffix(".stripe_detail.json").write_text(
            json.dumps({r["file"]: r["stripe_nonzero"] for r in rows}, indent=1), encoding="utf-8"
        )
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
