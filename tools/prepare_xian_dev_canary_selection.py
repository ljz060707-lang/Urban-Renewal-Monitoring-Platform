"""Cloud-side, low-resolution selection of fixed 1 km Xi'an DEV-CANARY blocks.

This tool deliberately does not run inference or promote any image proxy to
ground truth.  It reads the source raster on the cloud host, samples each
candidate at low resolution, freezes deterministic spatial blocks, and emits
auditable CSV/JSON receipts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from pyproj import Transformer


BLOCK_M = 1000.0
METRIC_CRS = "EPSG:32649"
SAMPLE_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def transformed_bounds(bounds: tuple[float, float, float, float], transformer: Transformer) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    points = [
        transformer.transform(minx, miny),
        transformer.transform(minx, maxy),
        transformer.transform(maxx, miny),
        transformer.transform(maxx, maxy),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def finite_score(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def sample_proxy(dataset: rasterio.DatasetReader, window: Window) -> dict[str, float]:
    bands = min(3, dataset.count)
    sample = dataset.read(
        indexes=list(range(1, bands + 1)),
        window=window,
        out_shape=(bands, SAMPLE_SIZE, SAMPLE_SIZE),
        resampling=Resampling.bilinear,
        masked=True,
    )
    values = np.ma.filled(sample, np.nan).astype(np.float32)
    if values.size == 0:
        return {"valid_fraction": 0.0, "luma_mean": 0.0, "luma_std": 0.0, "edge_proxy": 0.0}
    valid = np.isfinite(values).all(axis=0)
    if bands >= 3:
        luma = 0.299 * values[0] + 0.587 * values[1] + 0.114 * values[2]
    else:
        luma = values.mean(axis=0)
    finite = np.isfinite(luma)
    if not finite.any():
        return {"valid_fraction": 0.0, "luma_mean": 0.0, "luma_std": 0.0, "edge_proxy": 0.0}
    safe = np.where(finite, luma, 0.0)
    gx = np.abs(np.diff(safe, axis=1)).mean() if safe.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(safe, axis=0)).mean() if safe.shape[0] > 1 else 0.0
    return {
        "valid_fraction": float(valid.mean()),
        "luma_mean": finite_score(float(np.nanmean(luma))),
        "luma_std": finite_score(float(np.nanstd(luma))),
        "edge_proxy": finite_score(float((gx + gy) / 2.0)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def choose_blocks(candidates: list[dict[str, object]], metric_bounds: tuple[float, float, float, float]) -> list[dict[str, object]]:
    minx, miny, maxx, maxy = metric_bounds
    center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    x_span = maxx - minx
    y_span = maxy - miny
    targets = [
        ("西南纹理候选块", "southwest_texture", lambda x, y: x <= center[0] and y <= center[1]),
        ("东南纹理候选块", "southeast_texture", lambda x, y: x >= center[0] and y <= center[1]),
        ("中心纹理候选块", "center_texture", lambda x, y: abs(x - center[0]) <= 0.22 * x_span and abs(y - center[1]) <= 0.22 * y_span),
        ("西北纹理候选块", "northwest_texture", lambda x, y: x <= center[0] and y >= center[1]),
        ("北缘接缝压力块", "north_edge_seam_pressure", lambda x, y: y >= maxy - 2.0 * BLOCK_M),
    ]
    eligible = [row for row in candidates if float(row.get("source_coverage_fraction", 0.0)) >= 0.8]
    if len(eligible) < 5:
        eligible = candidates
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for name, role, predicate in targets:
        sector = [
            row
            for row in eligible
            if predicate(
                (float(row["x_min"]) + float(row["x_max"])) / 2.0,
                (float(row["y_min"]) + float(row["y_max"])) / 2.0,
            )
        ] or eligible
        ranked = sorted(
            sector,
            key=lambda row: (
                row["grid_id"] in used,
                -float(row.get("texture_proxy_score", 0.0)),
            ),
        )
        if not ranked:
            continue
        row = next((item for item in ranked if item["grid_id"] not in used), ranked[0])
        selected_row = dict(row)
        selected_row.update({"block_name_zh": name, "selection_role": role})
        used.add(str(row["grid_id"]))
        selected.append(selected_row)
    for index, row in enumerate(selected, start=1):
        row["block_id"] = f"B{index:02d}"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-year", default="2020")
    parser.add_argument(
        "--known-source-dates",
        default="2018-04-29,2019-01-01,2020-12-31",
        help="Known source-level acquisition dates; not a tile-level date claim.",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(args.source)
    with rasterio.open(args.source) as dataset:
        source_bounds = tuple(float(value) for value in dataset.bounds)
        source_crs = str(dataset.crs)
        to_metric = Transformer.from_crs(dataset.crs, METRIC_CRS, always_xy=True)
        to_source = Transformer.from_crs(METRIC_CRS, dataset.crs, always_xy=True)
        metric_bounds = transformed_bounds(source_bounds, to_metric)
        minx, miny, maxx, maxy = metric_bounds
        grid_minx = math.floor(minx / BLOCK_M) * BLOCK_M
        grid_miny = math.floor(miny / BLOCK_M) * BLOCK_M
        grid_maxx = math.ceil(maxx / BLOCK_M) * BLOCK_M
        grid_maxy = math.ceil(maxy / BLOCK_M) * BLOCK_M
        candidates: list[dict[str, object]] = []
        row_index = 0
        y = grid_miny
        while y < grid_maxy:
            x = grid_minx
            while x < grid_maxx:
                corner_points = [
                    to_source.transform(x, y),
                    to_source.transform(x + BLOCK_M, y + BLOCK_M),
                ]
                src_minx = min(point[0] for point in corner_points)
                src_maxx = max(point[0] for point in corner_points)
                src_miny = min(point[1] for point in corner_points)
                src_maxy = max(point[1] for point in corner_points)
                full_window = from_bounds(src_minx, src_miny, src_maxx, src_maxy, dataset.transform)
                window = full_window
                col0 = max(0.0, float(window.col_off))
                row0 = max(0.0, float(window.row_off))
                col1 = min(float(dataset.width), float(window.col_off + window.width))
                row1 = min(float(dataset.height), float(window.row_off + window.height))
                if col1 <= col0 or row1 <= row0:
                    x += BLOCK_M
                    continue
                window = Window(col0, row0, col1 - col0, row1 - row0)
                full_area = max(1.0, float(full_window.width * full_window.height))
                coverage_fraction = min(1.0, max(0.0, float(window.width * window.height) / full_area))
                if window.width > 1 and window.height > 1:
                    proxy = sample_proxy(dataset, window)
                    edge_distance = min(x - minx, y - miny, maxx - (x + BLOCK_M), maxy - (y + BLOCK_M))
                    row = {
                        "grid_id": f"G{row_index:05d}",
                        "x_min": int(x),
                        "y_min": int(y),
                        "x_max": int(x + BLOCK_M),
                        "y_max": int(y + BLOCK_M),
                        "width_m": int(BLOCK_M),
                        "height_m": int(BLOCK_M),
                        "source_window_col_off": round(float(window.col_off), 3),
                        "source_window_row_off": round(float(window.row_off), 3),
                        "source_window_width": round(float(window.width), 3),
                        "source_window_height": round(float(window.height), 3),
                        "source_coverage_fraction": round(coverage_fraction, 6),
                        "valid_fraction_proxy": round(proxy["valid_fraction"], 6),
                        "luma_mean_proxy": round(proxy["luma_mean"], 6),
                        "luma_std_proxy": round(proxy["luma_std"], 6),
                        "edge_proxy": round(proxy["edge_proxy"], 6),
                        "texture_proxy_score": round(proxy["edge_proxy"] + 0.5 * proxy["luma_std"], 6),
                        "edge_distance_m": round(max(0.0, edge_distance), 3),
                        "proxy_status": "DIAGNOSTIC_ONLY_NOT_GROUND_TRUTH",
                    }
                    candidates.append(row)
                    row_index += 1
                x += BLOCK_M
            y += BLOCK_M
        selected = choose_blocks(candidates, metric_bounds)
        known_dates = [date.strip() for date in args.known_source_dates.split(",") if date.strip()]
        for row in selected:
            row.update(
                {
                    "nominal_year": args.nominal_year,
                    "actual_source_dates": ";".join(known_dates),
                    "source_level_date_diversity_count": len(known_dates),
                    "tile_level_actual_date_mapping": "UNAVAILABLE",
                    "tile_date_claim": "NONE",
                    "crs": METRIC_CRS,
                    "source_crs": source_crs,
                    "selection_method": "deterministic spatial stratification over 1000m grid ranked by low-resolution texture/edge proxy; diagnostic only",
                }
            )
        write_csv(output / "地块选择候选网格.csv", candidates)
        write_csv(output / "地块选择冻结表.csv", selected)
        receipt = {
            "status": "SELECTION_FROZEN",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(args.source),
            "source_sha256": source_hash,
            "source_crs": source_crs,
            "metric_crs": METRIC_CRS,
            "source_bounds": source_bounds,
            "metric_bounds": metric_bounds,
            "nominal_year": args.nominal_year,
            "source_level_actual_dates": known_dates,
            "source_level_date_diversity_count": len(known_dates),
            "tile_level_actual_date_mapping": "UNAVAILABLE",
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "block_size_m": BLOCK_M,
            "proxy_policy": "diagnostic-only; no ground-truth promotion",
            "selected_blocks": selected,
        }
        (output / "地块选择冻结回执.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SELECTION_FROZEN", "candidate_count": len(candidates), "selected_count": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
