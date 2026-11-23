#!/usr/bin/env python3
"""Enumerate exact-date spatial overlaps and fixed-case support for Stage-2."""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pyproj import Geod


ROOT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2")
MAP = ROOT / "00_date_harmonization" / "ACTUAL_DATE_TILE_MAP.parquet"
SCENE = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
ORIGIN_X, ORIGIN_Y = 420534, 311895
ORIGIN_LON, ORIGIN_LAT, STEP = 108.75778198242188, 34.16130065917969, 0.0006866455078125


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    out_csv = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_CANDIDATES_STAGE2_V2.csv"
    out_json = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION_STAGE2_V2.json"
    for path in (out_csv, out_json):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite date-pair alternative artifact: {path}")

    target_dates = {
        "2018": ["2018-04-29"],
        "2019": ["2018-11-01", "2018-11-08"],
        "2020": ["2019-12-02", "2019-12-10", "2020-02-09", "2020-02-16", "2020-02-21", "2020-04-24", "2020-06-14", "2020-11-14"],
    }
    tile_sets: dict[tuple[str, str], set[tuple[int, int]]] = {(epoch, date): set() for epoch, dates in target_dates.items() for date in dates}
    parquet = pq.ParquetFile(MAP)
    for batch in parquet.iter_batches(batch_size=100_000, columns=["epoch_label", "actual_date", "source_scene", "tile_x", "tile_y", "pixel_available"]):
        for row in batch.to_pylist():
            epoch, date = str(row["epoch_label"] or ""), str(row["actual_date"] or "")
            if (epoch, date) not in tile_sets or row["source_scene"] != SCENE[epoch] or not row["pixel_available"]:
                continue
            tile_sets[(epoch, date)].add((int(row["tile_x"]), int(row["tile_y"])))

    strata: dict[tuple[str, str], dict[str, float]] = {}
    for row in read_csv(ROOT / "00_date_harmonization" / "DATE_STRATA_COVERAGE.csv"):
        if row.get("epoch_label") in target_dates and row.get("actual_date") in target_dates[row["epoch_label"]]:
            strata[(row["epoch_label"], row["actual_date"])] = row

    fixed_rows = read_csv(ROOT / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv")
    fixed: dict[tuple[str, str], dict[str, set[str] | int]] = {}
    for epoch, dates in target_dates.items():
        for date in dates:
            rows = [row for row in fixed_rows if row.get("epoch_label") == epoch and row.get("actual_date") == date and row.get("pixel_available", "").lower() == "true"]
            fixed[(epoch, date)] = {
                "rows": len(rows),
                "cases": {row["case_id"] for row in rows},
                "object_uids": {row.get("object_uid", "") for row in rows if row.get("object_uid")},
            }

    geod = Geod(ellps="WGS84")
    area_cache: dict[int, float] = {}

    def tile_area(y: int) -> float:
        if y not in area_cache:
            ymin = ORIGIN_LAT + (y - ORIGIN_Y) * STEP
            xmax = ORIGIN_LON + STEP
            ymax = ymin + STEP
            area, _ = geod.polygon_area_perimeter([ORIGIN_LON, xmax, xmax, ORIGIN_LON], [ymin, ymin, ymax, ymax])
            area_cache[y] = abs(float(area)) / 1_000_000.0
        return area_cache[y]

    defs: list[tuple[str, str, str, str, str]] = []
    for target_date in target_dates["2019"]:
        defs.append(("PAIR_A", "2018", "2018-04-29", "2019", target_date))
    for source_date in target_dates["2019"]:
        for target_date in target_dates["2020"]:
            defs.append(("PAIR_B", "2019", source_date, "2020", target_date))

    rows: list[dict[str, Any]] = []
    for base_pair, source_epoch, source_date, target_epoch, target_date in defs:
        left = tile_sets[(source_epoch, source_date)]
        right = tile_sets[(target_epoch, target_date)]
        common = left & right
        common_area = sum(tile_area(y) for _, y in common)
        source_fixed = fixed[(source_epoch, source_date)]
        target_fixed = fixed[(target_epoch, target_date)]
        paired_cases = set(source_fixed["cases"]) & set(target_fixed["cases"])
        primary = (base_pair == "PAIR_A" and target_date == "2018-11-01") or (base_pair == "PAIR_B" and source_date == "2018-11-01" and target_date == "2019-12-10")
        if base_pair == "PAIR_A":
            pair_id = "PAIR_A_2018-04-29_TO_" + target_date
        else:
            pair_id = "PAIR_B_" + source_date + "_TO_" + target_date
        source_stat = strata.get((source_epoch, source_date), {})
        target_stat = strata.get((target_epoch, target_date), {})
        rows.append({
                "pair_id": pair_id,
                "base_pair": base_pair,
                "source_epoch": source_epoch,
                "source_date": source_date,
                "target_epoch": target_epoch,
                "target_date": target_date,
                "actual_interval_days": (datetime.fromisoformat(target_date) - datetime.fromisoformat(source_date)).days,
                "date_coherent": True,
                "mosaic_dates_combined": False,
                "source_pixel_tile_count": len(left),
                "target_pixel_tile_count": len(right),
                "same_spatial_tile_count": len(common),
                "same_spatial_coverage_area_km2": round(common_area, 6),
                "source_r4_centroid_objects": int(float(source_stat.get("r4_geometry_centroid_object_count") or 0)),
                "target_r4_centroid_objects": int(float(target_stat.get("r4_geometry_centroid_object_count") or 0)),
                "fixed_case_pixel_object_rows_source": source_fixed["rows"],
                "fixed_case_pixel_object_rows_target": target_fixed["rows"],
                "fixed_case_pixel_cases_source": len(source_fixed["cases"]),
                "fixed_case_pixel_cases_target": len(target_fixed["cases"]),
                "fixed_case_pairable_case_n": len(paired_cases),
                "fixed_case_pairable_case_ids": ";".join(sorted(paired_cases)),
                "primary_date_preference": primary,
                "selected_for_bounded_stage2": bool(common) and (primary or bool(paired_cases)),
                "selection_role": "DOMINANT_DATE_PRIMARY" if primary else ("FIXED_CASE_DATE_ALTERNATIVE" if paired_cases else "ALTERNATIVE_NOT_SELECTED"),
                "status": "ELIGIBLE_FOR_BOUNDED_CANDIDATE_QA" if common else "BLOCKED_NO_COMMON_PIXEL_TILES",
        })

    rows.sort(key=lambda row: (row["base_pair"], not row["selected_for_bounded_stage2"], -row["same_spatial_coverage_area_km2"], -row["fixed_case_pairable_case_n"], row["source_date"], row["target_date"]))
    write_csv(out_csv, rows)
    # Preserve the first pair-expansion attempt; V1 duplicated rows because its
    # candidate loop repeated each source date once per target date.
    v1_csv = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_CANDIDATES_STAGE2.csv"
    v1_json = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION_STAGE2.json"
    if v1_csv.exists():
        archive = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_CANDIDATES_STAGE2_V1_DUPLICATED.csv"
        if not archive.exists():
            shutil.copy2(v1_csv, archive)
    if v1_json.exists():
        archive = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION_STAGE2_V1_DUPLICATED.json"
        if not archive.exists():
            shutil.copy2(v1_json, archive)
    selected = [row for row in rows if row["selected_for_bounded_stage2"]]
    payload = {
        "schema_version": "xian_r5.date_coherent_pair_selection_stage2.v1",
        "generated_at_utc": now_utc(),
        "pair_candidates_total": len(rows),
        "selected_bounded_pair_strata": selected,
        "pair_candidates_csv": str(out_csv),
        "selection_rule": "retain dominant actual-date pairs for spatial coverage; additionally retain exact-date alternatives only where visually registered fixed cases have pixels in both epochs; never combine acquisition dates into a mosaic",
        "date_heterogeneity_in_selected_tile_pairs": "NONE_BY_CONSTRUCTION;each pair is keyed to one exact source date and one exact target date",
        "fixed_case_candidate_not_ground_truth": True,
        "promotion_or_inference_claim": False,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    initial = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION.json"
    backup = ROOT / "00_date_harmonization" / "DATE_COHERENT_PAIR_SELECTION_PRIMARY_INITIAL.json"
    if initial.exists() and not backup.exists():
        shutil.copy2(initial, backup)
    if backup.exists():
        prior = json.loads(backup.read_text(encoding="utf-8"))
        prior["initial_primary_selection_receipt"] = str(backup)
        prior["expanded_stage2_candidate_receipt"] = str(out_json)
        prior["selected_for_bounded_stage2"] = [row["pair_id"] for row in selected]
        prior["selected_pair_details"] = selected
        prior["date_heterogeneity_in_selected_tile_pairs"] = "NONE_BY_CONSTRUCTION;exact-date pairs are kept separate"
        initial.write_text(json.dumps(prior, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DATE_PAIR_ALTERNATIVES_READY", "candidates": len(rows), "selected": len(selected), "selected_ids": [row["pair_id"] for row in selected]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
