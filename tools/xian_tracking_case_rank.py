"""Rank tracking-case candidates by GEOMETRY QUALITY, not by raw size.

The raw pool ranked by area alone surfaced 140,000 m2 airport/industrial tracts,
which are useless as showcase cases.  This script re-ranks with real per-epoch
geometry:

  compactness = 4*pi*A / P^2   (1.0 = circle; regular rectangles ~0.7-0.9)
  isolation   = other-date-object count inside 150 m of the case centroid
  legibility  = area inside a mid-size band, legible at the delivered pixel size
  contrast    = how unambiguous the presence/state signal is for its category

Output: TRACKING_CASE_CANDIDATE_POOL_RANKED.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "work" / "python-packages"))
spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402

CASE_ROOT = core.DELIVERY / "03_tracking_cases"
CATEGORIES = (
    "stable", "addition", "demolition", "unknown_conflict",
    "split", "merge", "split_merge", "gap_return", "parallax",
)
SCENE_OF = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
SIZE_MIN, SIZE_MAX = 400.0, 20_000.0
NEIGHBOUR_RADIUS_M = 150.0
PIXEL_M = 0.247


def scene_profiles() -> tuple[dict, dict, dict]:
    """year -> object_id -> (area, compactness, cx, cy, minx, miny, maxx, maxy)."""
    profs, nbrs, cell_lists = {}, {}, {}
    for year, scene in SCENE_OF.items():
        gpkg = core.geom_path(core.SCENES[year][2])
        df = pyogrio.read_dataframe(gpkg, columns=["object_id", "area_m2", "geometry"], use_arrow=False)
        geoms = list(df.geometry)
        arr = np.empty(len(geoms), dtype=object)
        arr[:] = geoms
        bnd = shapely.bounds(arr)
        cent = shapely.centroid(arr)
        perim = shapely.length(arr)
        areas = df["area_m2"].to_numpy(dtype=float)
        comp = np.where(perim > 0, 4.0 * math.pi * areas / (perim**2), 0.0)
        oids = df["object_id"].to_numpy()
        prof = {
            int(oids[i]): (
                float(areas[i]), float(comp[i]),
                float(shapely.get_x(cent[i])), float(shapely.get_y(cent[i])),
                float(bnd[i, 0]), float(bnd[i, 1]), float(bnd[i, 2]), float(bnd[i, 3]),
            )
            for i in range(len(geoms))
        }
        profs[year] = prof
        del df, geoms, arr
        # coarse grid index for neighbour counting
        cell = NEIGHBOUR_RADIUS_M
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        ids = list(prof.keys())
        for oid in ids:
            _, _, cx, cy, *_ = prof[oid]
            buckets[(int(cx // cell), int(cy // cell))].append(oid)
        nbr = {}
        r2 = NEIGHBOUR_RADIUS_M**2
        for oid in ids:
            _, _, cx, cy, *_ = prof[oid]
            kx, ky = int(cx // cell), int(cy // cell)
            n = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other in buckets.get((kx + dx, ky + dy), ()):
                        if other == oid:
                            continue
                        _, _, ox, oy, *_ = prof[other]
                        if (ox - cx) ** 2 + (oy - cy) ** 2 <= r2:
                            n += 1
            nbr[oid] = n
        nbrs[year] = nbr
        cell_lists[year] = buckets
        print(f"{year}: {len(prof):,} geometry profiles indexed")
    return profs, nbrs, cell_lists


def load_track_objects() -> dict:
    """physical_track_id -> {scene -> [object_id, ...]} from the tracking table."""
    path = core.LATEST / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv"
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            physical = row.get("physical_track_id", "")
            scene = row.get("scene_key", "")
            if not physical or not scene:
                continue
            try:
                oid = int(row.get("object_uid", "").split("::")[-1])
            except Exception:
                continue
            out[physical][scene].append(oid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(CASE_ROOT / "TRACKING_CASE_CANDIDATE_POOL.csv"))
    ap.add_argument("--per-category", type=int, default=40)
    args = ap.parse_args()

    pool_rows = list(csv.DictReader(Path(args.pool).open(encoding="utf-8-sig", newline="")))
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in pool_rows:
        by_cat[r["case_category"]].append(r)

    profs, nbrs, _ = scene_profiles()
    track_objects = load_track_objects()

    def contrast(cat: str, seq: str) -> float:
        if cat == "stable":
            return 1.0 if seq == "1-1-1" else 0.0
        if cat == "addition":
            return 1.0 if seq == "0-1-1" else (0.75 if seq == "0-0-1" else 0.35)
        if cat == "demolition":
            return 1.0 if seq == "1-1-0" else (0.75 if seq == "1-0-0" else 0.35)
        if cat == "gap_return":
            return 1.0 if seq == "1-0-1" else 0.0
        if cat == "unknown_conflict":
            return 1.0 if seq == "1-1-1" else 0.6
        return 0.7

    out_rows = []
    for cat in CATEGORIES:
        scored = []
        for row in by_cat.get(cat, []):
            physical = row["physical_track_id"]
            per_scene = track_objects.get(physical, {})
            if not per_scene:
                continue
            # choose the epoch object that best represents the case
            best = None
            for year, scene in SCENE_OF.items():
                for oid in per_scene.get(scene, []):
                    p = profs[year].get(oid)
                    if not p:
                        continue
                    area, comp, cx, cy, minx, miny, maxx, maxy = p
                    if not (SIZE_MIN <= area <= SIZE_MAX):
                        continue
                    nbr = nbrs[year].get(oid, 999)
                    score = comp * (1.0 / (1.0 + nbr)) * math.log10(area)
                    if best is None or score > best["shape_score"]:
                        best = {
                            "year": year, "scene": scene, "object_id": oid, "area_m2": area,
                            "compactness": comp, "n_objects_within_150m": nbr,
                            "shape_score": score, "cx": cx, "cy": cy,
                            "bbox_32649": [minx, miny, maxx, maxy],
                        }
            if best is None:
                continue
            seq = row.get("full_presence", "")
            con = contrast(cat, seq)
            if con <= 0.0:
                continue
            best.update({k: v for k, v in row.items()})
            best["contrast"] = con
            best["final_score"] = con * best["shape_score"]
            scored.append(best)
        scored.sort(key=lambda r: -r["final_score"])
        for rank, r in enumerate(scored[: args.per_category], start=1):
            r["rank_quality"] = rank
            out_rows.append(r)

    fields = [
        "case_category", "rank_quality", "physical_track_id", "final_score", "contrast",
        "shape_score", "compactness", "n_objects_within_150m", "area_m2", "year", "scene",
        "object_id", "cx", "cy", "full_presence", "state_2018_2019", "state_2019_2020",
        "trigger_interval", "unknown_intervals", "gap_return", "n_date_objects", "area_by_scene",
        "reason_2018_2019", "reason_2019_2020",
    ]
    out = CASE_ROOT / "TRACKING_CASE_CANDIDATE_POOL_RANKED.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    counts = {c: sum(1 for r in out_rows if r["case_category"] == c) for c in CATEGORIES}
    (CASE_ROOT / "TRACKING_CASE_POOL_RANKED_SUMMARY.json").write_text(
        json.dumps(
            {
                "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
                "size_band_m2": [SIZE_MIN, SIZE_MAX],
                "neighbour_radius_m": NEIGHBOUR_RADIUS_M,
                "per_category": counts,
                "note": "quality-ranked candidate pool; visual grading still required before recommendation",
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out, len(out_rows), "rows")
    print(counts)


if __name__ == "__main__":
    main()
