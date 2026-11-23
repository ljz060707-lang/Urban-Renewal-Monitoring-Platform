"""Cloud-side B0 smoke screening for candidate 1 km blocks.

This is a diagnostic selection gate only.  A positive smoke result is not a
building label or accuracy claim; it only prevents an obviously empty/invalid
center tile from entering the DEV-CANARY full-run set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


ROLE_SPECS = [
    ("西南烟测候选块", "southwest_smoke_screen", "southwest"),
    ("东南烟测候选块", "southeast_smoke_screen", "southeast"),
    ("中心烟测候选块", "center_smoke_screen", "center"),
    ("西北烟测候选块", "northwest_smoke_screen", "northwest"),
    ("北缘接缝压力块", "north_edge_seam_pressure", "north_edge"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def role_filter(role: str, x: float, y: float, bounds: tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    if role == "southwest":
        return x <= cx and y <= cy
    if role == "southeast":
        return x >= cx and y <= cy
    if role == "center":
        return abs(x - cx) <= 0.22 * (maxx - minx) and abs(y - cy) <= 0.22 * (maxy - miny)
    if role == "northwest":
        return x <= cx and y >= cy
    if role == "north_edge":
        return y >= maxy - 2_000.0
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-role", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(args.selection_receipt.read_text(encoding="utf-8"))
    metric_bounds = tuple(float(value) for value in receipt["metric_bounds"])
    candidates = [
        row
        for row in read_csv(args.candidates)
        if float(row.get("source_coverage_fraction", 0.0)) >= 0.8
    ]
    shortlisted: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, _, role in ROLE_SPECS:
        rows = []
        for row in candidates:
            x = (float(row["x_min"]) + float(row["x_max"])) / 2.0
            y = (float(row["y_min"]) + float(row["y_max"])) / 2.0
            if role_filter(role, x, y, metric_bounds):
                rows.append(row)
        rows.sort(key=lambda row: float(row.get("texture_proxy_score", 0.0)), reverse=True)
        for row in rows[: max(1, args.per_role)]:
            if row["grid_id"] not in seen:
                shortlisted.append(row)
                seen.add(row["grid_id"])

    outer_tools = str(Path(__file__).resolve().parent)
    if outer_tools not in sys.path:
        sys.path.insert(0, outer_tools)
    import infer_rsbuilding_static_geometry as native  # type: ignore

    model = native.build_model(args.config, args.checkpoint, args.device)
    results: list[dict[str, object]] = []
    with rasterio.open(args.source) as source:
        tiles: list[np.ndarray] = []
        for row in shortlisted:
            center_col = float(row["source_window_col_off"]) + float(row["source_window_width"]) / 2.0
            center_row = float(row["source_window_row_off"]) + float(row["source_window_height"]) / 2.0
            col = max(0, min(source.width - 512, int(round(center_col - 256))))
            row_off = max(0, min(source.height - 512, int(round(center_row - 256))))
            tile = source.read([1, 2, 3], window=Window(col, row_off, 512, 512), boundless=True, fill_value=0)
            tiles.append(tile)
            row["screen_col"] = str(col)
            row["screen_row"] = str(row_off)
        for start in range(0, len(tiles), 8):
            probabilities = native.predict_batch(model, tiles[start : start + 8], args.device)
            for candidate, probability in zip(shortlisted[start : start + 8], probabilities):
                finite = bool(np.isfinite(probability).all())
                positive_fraction = float((probability >= native.THRESHOLD).mean()) if finite else 0.0
                results.append(
                    {
                        **candidate,
                        "smoke_status": "PASS" if finite and positive_fraction > 0.0 else "FAIL_EMPTY_OR_INVALID",
                        "probability_min": float(np.nanmin(probability)) if finite else None,
                        "probability_max": float(np.nanmax(probability)) if finite else None,
                        "positive_fraction": positive_fraction,
                        "screen_threshold": native.THRESHOLD,
                        "screen_scientific_status": "DIAGNOSTIC_ONLY_NOT_GROUND_TRUTH",
                    }
                )

    suggested: list[dict[str, object]] = []
    used: set[str] = set()
    for index, (name, role, role_key) in enumerate(ROLE_SPECS, start=1):
        eligible = [row for row in results if row["grid_id"] not in used and role_filter(
            role_key,
            (float(row["x_min"]) + float(row["x_max"])) / 2.0,
            (float(row["y_min"]) + float(row["y_max"])) / 2.0,
            metric_bounds,
        )]
        eligible.sort(key=lambda row: (row["smoke_status"] != "PASS", -float(row.get("texture_proxy_score", 0.0))))
        if not eligible:
            continue
        selected = dict(eligible[0])
        used.add(str(selected["grid_id"]))
        selected.update(
            {
                "block_id": f"B{index:02d}",
                "block_name_zh": name,
                "selection_role": role,
                "selection_status": "SMOKE_SCREEN_PASS" if selected["smoke_status"] == "PASS" else "SMOKE_SCREEN_BLOCKED",
                "nominal_year": receipt["nominal_year"],
                "actual_source_dates": ";".join(receipt["source_level_actual_dates"]),
                "source_level_date_diversity_count": receipt["source_level_date_diversity_count"],
                "tile_level_actual_date_mapping": "UNAVAILABLE",
                "tile_date_claim": "NONE",
                "crs": receipt["metric_crs"],
                "source_crs": receipt["source_crs"],
                "selection_method": "spatially stratified low-resolution proxy shortlist plus B0 center-tile smoke; diagnostic only",
            }
        )
        suggested.append(selected)

    write_csv(args.output / "候选中心烟测.csv", results)
    write_csv(args.output / "建议地块选择冻结表.csv", suggested)
    payload = {
        "status": "SMOKE_SCREEN_COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "candidates_tested": len(results),
        "smoke_pass_count": sum(row["smoke_status"] == "PASS" for row in results),
        "suggested_block_count": len(suggested),
        "suggested_pass_count": sum(row["selection_status"] == "SMOKE_SCREEN_PASS" for row in suggested),
        "threshold": native.THRESHOLD,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": native.sha256_file(args.checkpoint),
        "scientific_status": "DIAGNOSTIC_ONLY_NOT_GROUND_TRUTH",
    }
    (args.output / "候选中心烟测回执.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
