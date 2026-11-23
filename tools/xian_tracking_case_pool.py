"""Build the tracking-case candidate pool for the delivery.

Reads the candidate tracking objects and state table and ranks physical tracks
into the case categories requested for the delivery:

    stable, addition, demolition, unknown_conflict,
    split, merge, split_merge, gap_return, parallax

For every candidate the pool records the geometry, the presence sequence, the
state, the epoch areas and a ranking score.  Visual grading is a separate step
(see xian_tracking_case_sheets.py) because no candidate may be recommended
without actually looking at the imagery.
"""

from __future__ import annotations

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
import shapely  # noqa: E402

CASE_ROOT = core.DELIVERY / "03_tracking_cases"
CATEGORIES = (
    "stable",
    "addition",
    "demolition",
    "unknown_conflict",
    "split",
    "merge",
    "split_merge",
    "gap_return",
    "parallax",
)


def load_tracking() -> tuple[dict, dict]:
    """physical_track_id -> per-scene records; and -> per-interval state rows."""
    path = core.LATEST / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv"
    per_scene: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scene = row.get("scene_key", "")
            physical = row.get("physical_track_id", "")
            if not scene or not physical:
                continue
            try:
                oid = int(row.get("object_uid", "").split("::")[-1])
            except Exception:
                continue
            try:
                area = float(row.get("area_m2") or 0.0)
            except Exception:
                area = 0.0
            rec = per_scene[physical].get(scene)
            # a physical track can hold several date objects in one scene
            if rec is None:
                per_scene[physical][scene] = {
                    "epoch_object_ids": [oid],
                    "area_m2": area,
                    "n_objects": 1,
                    "identity_sources": row.get("identity_sources", ""),
                    "local_actual_date": row.get("local_actual_date", ""),
                    "formal_date": row.get("formal_date", ""),
                }
            else:
                rec["epoch_object_ids"].append(oid)
                rec["area_m2"] += area
                rec["n_objects"] += 1
    return per_scene, {}


def load_states() -> dict:
    path = core.LATEST / "CANDIDATE_STATE_EVENTS_FROM_SCRATCH.csv"
    states: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            physical = row.get("physical_track_id", "")
            interval = row.get("interval", "")
            if not physical or interval not in {"2018_2019", "2019_2020"}:
                continue
            def num(k):
                try:
                    return float(row.get(k) or 0.0)
                except Exception:
                    return 0.0
            states[physical][interval] = {
                "state": row.get("state", ""),
                "presence_sequence": row.get("presence_sequence", ""),
                "source_presence": row.get("source_presence", ""),
                "target_presence": row.get("target_presence", ""),
                "state_reason": row.get("state_reason", ""),
                "identity_sources": row.get("identity_sources", ""),
                "gap_return": row.get("gap_return", ""),
                "strict_change_eligible": row.get("strict_change_eligible", ""),
                "source_area_m2": num("source_area_m2"),
                "target_area_m2": num("target_area_m2"),
                "union_area_m2": num("union_area_m2"),
                "area_m2": num("area_m2"),
                "event": row.get("event", ""),
            }
    return states


def main() -> None:
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        (CASE_ROOT / cat).mkdir(parents=True, exist_ok=True)

    per_scene, _ = load_tracking()
    states = load_states()
    print(f"loaded {len(per_scene):,} physical tracks, {len(states):,} with state rows")

    # candidate geometries for overlap diagnostics, one epoch per scene
    pools: dict[str, list[dict]] = defaultdict(list)
    for physical, scenes in per_scene.items():
        st = states.get(physical)
        if not st:
            continue
        seqs = {iv: st[iv]["presence_sequence"] for iv in ("2018_2019", "2019_2020") if iv in st}
        areas = {sc: rec["area_m2"] for sc, rec in scenes.items()}
        max_area = max(areas.values()) if areas else 0.0
        n_objects = {sc: rec["n_objects"] for sc, rec in scenes.items()}
        total_objects = sum(n_objects.values())
        row = {
            "physical_track_id": physical,
            "scenes": sorted(scenes.keys()),
            "n_scenes": len(scenes),
            "n_date_objects": total_objects,
            "area_max_m2": max_area,
            "area_by_scene": {k: round(v, 3) for k, v in areas.items()},
            "presence_2018_2019": seqs.get("2018_2019", ""),
            "presence_2019_2020": seqs.get("2019_2020", ""),
            "state_2018_2019": st.get("2018_2019", {}).get("state", ""),
            "state_2019_2020": st.get("2019_2020", {}).get("state", ""),
            "reason_2018_2019": st.get("2018_2019", {}).get("state_reason", ""),
            "reason_2019_2020": st.get("2019_2020", {}).get("state_reason", ""),
            "identity_sources": (st.get("2018_2019", {}).get("identity_sources", "") or "")[:400],
            "gap_return": st.get("2018_2019", {}).get("gap_return", ""),
        }
        row["full_presence"] = "-".join(
            "1" if sc in scenes else "0" for sc in ("xa180429", "xa190101", "xa201231")
        )

        if max_area < core.AREA_FILTER:
            continue

        # ---- category rules -------------------------------------------------
        s1 = row["state_2018_2019"]
        s2 = row["state_2019_2020"]
        if s1 == "STABLE" and s2 == "STABLE" and row["full_presence"] == "1-1-1":
            row["category"] = "stable"
            row["score"] = min(areas.values()) if len(areas) == 3 else 0.0
            pools["stable"].append(row)
        if s1 == "ADDITION" or s2 == "ADDITION":
            iv = "2018_2019" if s1 == "ADDITION" else "2019_2020"
            row["category"] = "addition"
            row["trigger_interval"] = iv
            row["score"] = st[iv]["target_area_m2"] or max_area
            pools["addition"].append(row)
        if s1 == "DEMOLITION" or s2 == "DEMOLITION":
            iv = "2018_2019" if s1 == "DEMOLITION" else "2019_2020"
            row["category"] = "demolition"
            row["trigger_interval"] = iv
            row["score"] = st[iv]["source_area_m2"] or max_area
            pools["demolition"].append(row)
        if s1 == "UNKNOWN_CONFLICT" or s2 == "UNKNOWN_CONFLICT":
            row["category"] = "unknown_conflict"
            row["score"] = max_area
            row["unknown_intervals"] = "+".join(
                iv for iv in ("2018_2019", "2019_2020") if st.get(iv, {}).get("state") == "UNKNOWN_CONFLICT"
            )
            pools["unknown_conflict"].append(row)
        if total_objects >= 2 and len(scenes) >= 2:
            split_like = any(v >= 2 for v in n_objects.values())
            if split_like and row["full_presence"] == "1-1-1":
                row["category"] = "split"
                row["score"] = max_area
                row["object_counts"] = n_objects
                pools["split"].append(row)
        if row["full_presence"] == "1-0-1" or row["gap_return"].strip().upper() == "YES":
            row["category"] = "gap_return"
            row["score"] = max_area
            pools["gap_return"].append(row)
    print({k: len(v) for k, v in pools.items()})

    rows_out = []
    for cat in CATEGORIES:
        items = pools.get(cat, [])
        items.sort(key=lambda r: -r.get("score", 0.0))
        for rank, item in enumerate(items[:4000], start=1):
            item["rank_in_category"] = rank
            rows_out.append({"case_category": cat, **{k: v for k, v in item.items() if k != "category"}})

    detail = CASE_ROOT / "TRACKING_CASE_CANDIDATE_POOL.csv"
    fields = [
        "case_category", "rank_in_category", "physical_track_id", "score", "area_max_m2",
        "full_presence", "n_scenes", "n_date_objects", "presence_2018_2019", "presence_2019_2020",
        "state_2018_2019", "state_2019_2020", "reason_2018_2019", "reason_2019_2020",
        "trigger_interval", "unknown_intervals", "gap_return", "area_by_scene", "identity_sources",
    ]
    with detail.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    (CASE_ROOT / "TRACKING_CASE_POOL_SUMMARY.json").write_text(
        json.dumps(
            {
                "status": "CURRENT_BEST_CANDIDATE_DELIVERY",
                "scientific_status": "CANDIDATE_NOT_GROUND_TRUTH",
                "pool_counts": {k: len(v) for k, v in pools.items()},
                "pool_rows_written": len(rows_out),
                "note": "candidate pool only; no case is recommended before visual grading",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", detail)


if __name__ == "__main__":
    main()
