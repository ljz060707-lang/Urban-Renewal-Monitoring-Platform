"""Metric-area threshold exports for cloud B0 polygon candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd


METRIC_CRS = "EPSG:32649"
THRESHOLDS = [(0.0, "unfiltered"), (1.0, "ge_1m2"), (10.0, "ge_10m2"), (50.0, "ge_50m2"), (100.0, "ge_100m2")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual-date", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    block_receipts: list[dict[str, object]] = []
    for block_dir in sorted(args.b0_root.iterdir()):
        if not block_dir.is_dir() or not block_dir.name.startswith("B"):
            continue
        source = block_dir / args.actual_date / "building_objects.gpkg"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = gpd.read_file(source, layer="buildings")
        source_crs = str(frame.crs)
        metric = frame.to_crs(METRIC_CRS)
        metric["area_m2"] = metric.geometry.area.astype(float)
        metric["source_geometry_crs"] = source_crs
        metric["area_method"] = "projected_polygon_area_epsg32649"
        metric["source_geometry_preserved"] = True
        block_output = args.output / args.actual_date / block_dir.name
        block_output.mkdir(parents=True, exist_ok=True)
        block_receipt = {
            "block_id": block_dir.name,
            "source_gpkg": str(source),
            "source_sha256": sha256_file(source),
            "source_crs": source_crs,
            "metric_crs": METRIC_CRS,
            "source_object_count": int(len(metric)),
            "thresholds": [],
        }
        for threshold, label in THRESHOLDS:
            output = block_output / f"建筑对象_{label}.gpkg"
            derived = metric.copy()
            derived["threshold_m2"] = threshold
            derived["below_threshold"] = derived["area_m2"] < threshold if threshold > 0 else False
            derived["threshold_filter_method"] = "metric_polygon_area_only"
            if threshold > 0:
                derived = derived.loc[~derived["below_threshold"]].copy()
            derived.to_file(output, layer="buildings", driver="GPKG")
            count = int(len(derived))
            area_sum = float(derived["area_m2"].sum()) if count else 0.0
            row = {
                "block_id": block_dir.name,
                "actual_date": args.actual_date,
                "threshold_m2": threshold,
                "threshold_label": label,
                "object_count": count,
                "total_area_m2": area_sum,
                "mean_area_m2": float(derived["area_m2"].mean()) if count else 0.0,
                "metric_crs": METRIC_CRS,
                "filter_method": "metric_polygon_area_only",
                "geometry_modified": False,
                "output": str(output),
            }
            summary_rows.append(row)
            block_receipt["thresholds"].append(row)
        (block_output / "面积筛选回执.json").write_text(json.dumps(block_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_path = args.output / args.actual_date / "面积筛选统计.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    receipt = {
        "status": "AREA_THRESHOLDS_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actual_date": args.actual_date,
        "metric_crs": METRIC_CRS,
        "thresholds_m2": [threshold for threshold, _ in THRESHOLDS],
        "filter_method": "metric_polygon_area_only",
        "geometry_modified": False,
        "block_count": len({row["block_id"] for row in summary_rows}),
        "output_summary": str(summary_path),
    }
    (args.output / args.actual_date / "面积筛选总回执.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
