"""Cloud-side crop of the frozen Xi'an 1 km blocks.

The crop is a derived artifact.  It never edits the source raster and records
the source hash, source window, CRS, and actual source date in a manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from pyproj import Transformer


METRIC_CRS = "EPSG:32649"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_selection(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual-date", required=True)
    parser.add_argument("--nominal-year", default="2020")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = read_selection(args.selection)
    source_hash = sha256_file(args.source)
    manifest_rows: list[dict[str, object]] = []
    with rasterio.open(args.source) as source:
        transformer = Transformer.from_crs(METRIC_CRS, source.crs, always_xy=True)
        for block in selection:
            x_min = float(block["x_min"])
            y_min = float(block["y_min"])
            x_max = float(block["x_max"])
            y_max = float(block["y_max"])
            corners = [
                transformer.transform(x_min, y_min),
                transformer.transform(x_min, y_max),
                transformer.transform(x_max, y_min),
                transformer.transform(x_max, y_max),
            ]
            src_minx = min(point[0] for point in corners)
            src_maxx = max(point[0] for point in corners)
            src_miny = min(point[1] for point in corners)
            src_maxy = max(point[1] for point in corners)
            raw_window = from_bounds(src_minx, src_miny, src_maxx, src_maxy, source.transform)
            col0 = max(0, int(raw_window.col_off))
            row0 = max(0, int(raw_window.row_off))
            col1 = min(source.width, int(raw_window.col_off + raw_window.width + 0.999999))
            row1 = min(source.height, int(raw_window.row_off + raw_window.height + 0.999999))
            if col1 <= col0 or row1 <= row0:
                raise RuntimeError(f"empty source window for {block['block_id']}")
            window = Window(col0, row0, col1 - col0, row1 - row0)
            block_id = block["block_id"]
            output_name = f"{block_id}_{safe_name(block['block_name_zh'])}_{args.actual_date}.tif"
            output_path = args.output / output_name
            profile = source.profile.copy()
            # The source is JPEG/YCbCr; derived crops use DEFLATE and must not
            # inherit the source photometric constraint.
            profile.pop("photometric", None)
            profile.pop("jpeg_quality", None)
            profile.pop("jpeg_tables", None)
            profile.update(
                {
                    "driver": "GTiff",
                    "width": int(window.width),
                    "height": int(window.height),
                    "count": min(3, source.count),
                    "dtype": source.dtypes[0],
                    "crs": source.crs,
                    "transform": source.window_transform(window),
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "compress": "deflate",
                    "predictor": 2,
                    "BIGTIFF": "IF_SAFER",
                }
            )
            data = source.read(indexes=list(range(1, profile["count"] + 1)), window=window, resampling=Resampling.nearest)
            with rasterio.open(output_path, "w", **profile) as target:
                target.write(data)
            manifest_rows.append(
                {
                    "block_id": block_id,
                    "block_name_zh": block["block_name_zh"],
                    "actual_date": args.actual_date,
                    "nominal_year": args.nominal_year,
                    "source": str(args.source),
                    "source_sha256": source_hash,
                    "source_crs": str(source.crs),
                    "metric_crs": METRIC_CRS,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "source_window_col_off": col0,
                    "source_window_row_off": row0,
                    "source_window_width": int(window.width),
                    "source_window_height": int(window.height),
                    "output": str(output_path),
                    "output_sha256": sha256_file(output_path),
                    "status": "CROPPED_CLOUD_SIDE",
                }
            )
    manifest_path = args.output / f"裁剪清单_{args.actual_date}.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]) if manifest_rows else ["status"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    receipt = {
        "status": "CROP_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": source_hash,
        "selection": str(args.selection),
        "actual_date": args.actual_date,
        "nominal_year": args.nominal_year,
        "metric_crs": METRIC_CRS,
        "block_count": len(manifest_rows),
        "output_manifest": str(manifest_path),
        "outputs": manifest_rows,
    }
    (args.output / f"裁剪回执_{args.actual_date}.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CROP_COMPLETE", "actual_date": args.actual_date, "block_count": len(manifest_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
