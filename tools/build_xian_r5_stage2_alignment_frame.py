#!/usr/bin/env python3
"""Build bounded, actual-date-coherent alignment candidate pools for R5 Stage-2.

R4 footprint density is used only to propose visual-review candidates. It is
not a semantic/stability label and no candidate is counted as validated until
the paired raw imagery has been reviewed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fiona
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import shape


ROOT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2")
MAP_PATH = ROOT / "00_date_harmonization" / "ACTUAL_DATE_TILE_MAP.parquet"
FREEZE_PATH = ROOT / "13_receipts" / "R4_FREEZE_STAGE2.json"
SCENE_BY_EPOCH = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
GRID_X, GRID_Y = 420534, 311895
GRID_LON, GRID_LAT, GRID_STEP = 108.75778198242188, 34.16130065917969, 0.0006866455078125
PAIR_DEFINITIONS = {
    "PAIR_A": ("2018", "2018-04-29", "2019", "2018-11-01"),
    "PAIR_B": ("2019", "2018-11-01", "2020", "2019-12-10"),
}
SAMPLE_COUNT_PER_PROXY_STRATUM = 150


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_main_pair_tiles() -> dict[tuple[str, str], dict[tuple[int, int], str]]:
    keep = {
        ("2018", "2018-04-29"), ("2019", "2018-11-01"),
        ("2019", "2018-11-08"), ("2020", "2019-12-10"),
    }
    result: dict[tuple[str, str], dict[tuple[int, int], str]] = defaultdict(dict)
    parquet = pq.ParquetFile(MAP_PATH)
    columns = ["epoch_label", "actual_date", "source_scene", "tile_x", "tile_y", "pixel_available", "pixel_path", "source_kind"]
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        for row in batch.to_pylist():
            epoch = str(row["epoch_label"] or "")
            date = str(row["actual_date"] or "")
            if (epoch, date) not in keep or row["source_scene"] != SCENE_BY_EPOCH.get(epoch):
                continue
            key = (int(row["tile_x"]), int(row["tile_y"]))
            path = str(row["pixel_path"] or "")
            if row["pixel_available"] and path:
                result[(epoch, date)][key] = path
    return result


def read_geometry_tile_stats() -> tuple[dict[str, Counter[tuple[int, int]]], dict[str, dict[tuple[int, int], float]], dict[str, dict[tuple[int, int], float]], dict[str, Counter[str]]]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    geometry_paths = {
        epoch: Path(freeze["files"][f"geometry_{epoch}"]["path"])
        for epoch in SCENE_BY_EPOCH
    }
    counts: dict[str, Counter[tuple[int, int]]] = {epoch: Counter() for epoch in SCENE_BY_EPOCH}
    area_sums: dict[str, dict[tuple[int, int], float]] = {epoch: defaultdict(float) for epoch in SCENE_BY_EPOCH}
    max_areas: dict[str, dict[tuple[int, int], float]] = {epoch: defaultdict(float) for epoch in SCENE_BY_EPOCH}
    totals: dict[str, Counter[str]] = {epoch: Counter() for epoch in SCENE_BY_EPOCH}
    transformers = {epoch: Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True) for epoch in SCENE_BY_EPOCH}
    for epoch, path in geometry_paths.items():
        transformer = transformers[epoch]
        with fiona.open(path, layer="buildings") as source:
            for feature in source:
                if not feature.get("geometry"):
                    continue
                geom = shape(feature["geometry"])
                if geom.is_empty:
                    continue
                centroid = geom.centroid
                lon, lat = transformer.transform(centroid.x, centroid.y)
                x = math.floor((lon - GRID_LON) / GRID_STEP + GRID_X)
                y = math.floor((lat - GRID_LAT) / GRID_STEP + GRID_Y)
                key = (x, y)
                area = float((feature.get("properties") or {}).get("area_m2") or geom.area)
                counts[epoch][key] += 1
                area_sums[epoch][key] += area
                max_areas[epoch][key] = max(max_areas[epoch][key], area)
                totals[epoch]["features_read"] += 1
    return counts, area_sums, max_areas, totals


def make_montage(rows: list[dict[str, Any]], output_dir: Path, group_key: str) -> list[str]:
    sheet_size = 30
    tile_size = 104
    cols = 6
    cell_w = tile_size * 2 + 8
    cell_h = tile_size + 30
    paths: list[str] = []
    for start in range(0, len(rows), sheet_size):
        subset = rows[start : start + sheet_size]
        nrows = math.ceil(len(subset) / cols)
        canvas = Image.new("RGB", (cols * cell_w, nrows * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for local_idx, row in enumerate(subset):
            idx = start + local_idx
            col, rnum = local_idx % cols, local_idx // cols
            x0, y0 = col * cell_w, rnum * cell_h
            for side, image_path in enumerate((row["source_pixel_path"], row["target_pixel_path"])):
                with Image.open(image_path) as image:
                    image = image.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    canvas.paste(image, (x0 + 4 + side * (tile_size + 2), y0 + 20))
            label = (
                f"{row['sample_id']} tile={row['tile_x']},{row['tile_y']} "
                f"obj={row['source_object_count']}/{row['target_object_count']} "
                f"maxA={row['source_max_area_m2']:.0f}/{row['target_max_area_m2']:.0f}"
            )
            draw.text((x0 + 2, y0 + 2), label[:56], fill="black")
        sheet_path = output_dir / f"{group_key}_{start + 1:04d}-{start + len(subset):04d}.jpg"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(sheet_path, quality=92)
        paths.append(str(sheet_path))
    return paths


def main() -> int:
    for relative in (
        "04_stable_ground/STABLE_GROUND_CANDIDATE_POOL.csv",
        "05_parallax/PARALLAX_RISK_CANDIDATE_POOL.csv",
        "11_visual_qa/VISUAL_STRATUM_REVIEW_TEMPLATE.csv",
        "13_receipts/ALIGNMENT_SAMPLE_FRAME_RECEIPT.json",
    ):
        if (ROOT / relative).exists():
            raise FileExistsError(f"Refusing to overwrite existing Stage-2 sample artifact: {ROOT / relative}")

    tiles = load_main_pair_tiles()
    counts, area_sums, max_areas, totals = read_geometry_tile_stats()
    stable_rows: list[dict[str, Any]] = []
    parallax_rows: list[dict[str, Any]] = []
    sheet_paths: list[str] = []
    candidate_counts: dict[str, dict[str, int]] = {}

    for pair_id, (source_epoch, source_date, target_epoch, target_date) in PAIR_DEFINITIONS.items():
        source_tiles = tiles.get((source_epoch, source_date), {})
        target_tiles = tiles.get((target_epoch, target_date), {})
        common = sorted(set(source_tiles) & set(target_tiles))
        stable_candidates: list[dict[str, Any]] = []
        parallax_candidates: list[dict[str, Any]] = []
        for key in common:
            n0, n1 = counts[source_epoch][key], counts[target_epoch][key]
            a0, a1 = area_sums[source_epoch].get(key, 0.0), area_sums[target_epoch].get(key, 0.0)
            m0, m1 = max_areas[source_epoch].get(key, 0.0), max_areas[target_epoch].get(key, 0.0)
            base = {
                "pair_id": pair_id,
                "source_epoch": source_epoch,
                "source_date": source_date,
                "target_epoch": target_epoch,
                "target_date": target_date,
                "actual_interval_days": (datetime.fromisoformat(target_date) - datetime.fromisoformat(source_date)).days,
                "tile_id": f"MAIN_Z20:{key[0]}:{key[1]}",
                "tile_x": key[0],
                "tile_y": key[1],
                "source_pixel_path": source_tiles[key],
                "target_pixel_path": target_tiles[key],
                "source_object_count": n0,
                "target_object_count": n1,
                "source_total_area_m2": round(a0, 2),
                "target_total_area_m2": round(a1, 2),
                "source_max_area_m2": round(m0, 2),
                "target_max_area_m2": round(m1, 2),
                "r4_geometry_proxy_not_label": True,
                "visual_review_status": "UNREVIEWED_CANDIDATE_ONLY",
            }
            if n0 <= 1 and n1 <= 1 and a0 <= 250.0 and a1 <= 250.0:
                stable_candidates.append({**base, "proxy_rule": "LOW_R4_FOOTPRINT_DENSITY;VISUAL_GROUND_SCREEN_REQUIRED"})
            if max(n0, n1) >= 3 or max(m0, m1) >= 500.0:
                parallax_candidates.append({**base, "proxy_rule": "DENSE_OR_LARGE_R4_FOOTPRINT;HIGH_RISE_VISUAL_SCREEN_REQUIRED"})

        def choose(candidates: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
            ordered = sorted(candidates, key=lambda row: sha(f"{pair_id}:{tag}:{row['tile_id']}"))
            selected = ordered[:SAMPLE_COUNT_PER_PROXY_STRATUM]
            for i, row in enumerate(selected, 1):
                row["sample_id"] = f"{pair_id}_{tag}_{i:03d}"
            return selected

        stable_selected = choose(stable_candidates, "STABLE_PROXY")
        parallax_selected = choose(parallax_candidates, "PARALLAX_PROXY")
        stable_rows.extend(stable_selected)
        parallax_rows.extend(parallax_selected)
        candidate_counts[pair_id] = {
            "common_pixel_tiles": len(common),
            "stable_ground_proxy_candidates": len(stable_candidates),
            "stable_ground_proxy_selected": len(stable_selected),
            "parallax_proxy_candidates": len(parallax_candidates),
            "parallax_proxy_selected": len(parallax_selected),
        }
        for label, rows, output_dir in (
            ("STABLE_GROUND", stable_selected, ROOT / "11_visual_qa" / "alignment_candidates" / "stable_ground"),
            ("PARALLAX_RISK", parallax_selected, ROOT / "11_visual_qa" / "alignment_candidates" / "parallax_risk"),
        ):
            sheet_paths.extend(make_montage(rows, output_dir, f"{pair_id}_{label}"))

    review_rows = []
    for row in stable_rows + parallax_rows:
        review_rows.append({
            **row,
            "visual_class": "",
            "reviewer_notes": "",
            "reviewed_by": "",
            "reviewed_at_utc": "",
        })
    write_csv(ROOT / "04_stable_ground" / "STABLE_GROUND_CANDIDATE_POOL.csv", stable_rows)
    write_csv(ROOT / "05_parallax" / "PARALLAX_RISK_CANDIDATE_POOL.csv", parallax_rows)
    write_csv(ROOT / "11_visual_qa" / "VISUAL_STRATUM_REVIEW_TEMPLATE.csv", review_rows)
    receipt = {
        "schema_version": "xian_r5.alignment_sample_frame.v1",
        "generated_at_utc": now_utc(),
        "pairs": candidate_counts,
        "r4_geometry_objects_read": {epoch: totals[epoch]["features_read"] for epoch in SCENE_BY_EPOCH},
        "stratum_status": "PROXY_CANDIDATES_ONLY_VISUAL_REVIEW_REQUIRED",
        "stable_ground_n_validated": 0,
        "parallax_risk_n_validated": 0,
        "lightglue_ab_status": "NOT_RUN_NO_VISUALLY_VALIDATED_STRATA",
        "false_change_status": "NOT_RUN_NO_100_VISUALLY_CONFIRMED_STABLE_BUILDINGS",
        "random_seed_rule": "SHA256_RANKED_BY_PAIR_STRATUM_AND_TILE_ID",
        "sample_counts_per_pair_and_proxy_stratum": SAMPLE_COUNT_PER_PROXY_STRATUM,
        "montage_paths": sheet_paths,
        "limitations": [
            "R4 centroid counts and footprint area nominate candidates only; they do not establish land cover, building height, stability, or parallax.",
            "Montages show exact-date paired raw tiles at reduced resolution; each accepted sample still requires visual review and a documented ROI.",
            "No production geometry is warped or modified.",
        ],
    }
    out = ROOT / "13_receipts" / "ALIGNMENT_SAMPLE_FRAME_RECEIPT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "CANDIDATE_FRAME_READY_FOR_VISUAL_REVIEW", "pairs": candidate_counts, "montages": len(sheet_paths)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
