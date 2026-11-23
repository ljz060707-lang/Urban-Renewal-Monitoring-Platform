"""Read-only absence evidence diagnostic for unresolved compound residuals."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from shapely.ops import unary_union


SCENES = ("xa180429", "xa190101", "xa201231")
PAIRS = {"2018_2019": ("xa180429", "xa190101"), "2019_2020": ("xa190101", "xa201231")}


def quality(src, x: float, y: float) -> dict[str, float]:
    # The raw rasters are geographic, while the frozen footprints are UTM.
    to_geo = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True).transform
    lon, lat = to_geo(x, y)
    col = int((lon - src.transform.c) / src.transform.a)
    row = int((src.transform.f - lat) / abs(src.transform.e))
    size = 96
    win = Window(col - size // 2, row - size // 2, size, size)
    arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
    valid = np.all(arr > 0, axis=0)
    black = np.all(arr == 0, axis=0)
    gray = arr.astype(np.float32).mean(axis=0)
    return {
        "valid_ratio": float(valid.mean()),
        "black_ratio": float(black.mean()),
        "gray_std": float(gray[valid].std()) if valid.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("compound", "u05"), default="compound")
    ap.add_argument("--per-interval", type=int, default=100)
    ap.add_argument("--output-name", default="21_rfinal_phase3_absence_diagnostic_20260917")
    args = ap.parse_args()
    root = Path("/data/xian_r4_100km2_20260914")
    out = root / "output" / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    phase1 = root / "output/18_rfinal_baseline_and_residual_diagnosis_20260916"
    v2 = root / "output/16_candidate_identitygraph_v2_20260916"
    formal = root / "output/西安_R4_辅助影像接入前_三期三态候选版_20260914"
    gpkg = formal / "04_identity_graph/tracking_buildings.gpkg"
    if args.mode == "compound":
        input_path = phase1 / "phase1_unresolved_compound_high_area.csv"
    else:
        input_path = v2 / "RESIDUAL_UNKNOWN_PHASE_B_V2.csv"
    residual = pd.read_csv(input_path, low_memory=False)
    if args.mode == "u05":
        residual = residual[residual["unknown_cause"].astype(str).eq("U05_UNOBSERVABLE")].copy()
        residual = residual[~residual["presence_sequence"].astype(str).isin({"1-0-1", "0-1-0"})].copy()
    residual["area_m2"] = pd.to_numeric(residual["area_m2"], errors="coerce").fillna(0.0)
    pieces = []
    for interval in sorted(residual["interval"].astype(str).unique()):
        pieces.append(residual[residual["interval"].astype(str).eq(interval)].sort_values("area_m2", ascending=False).head(args.per_interval))
    samples = pd.concat(pieces, ignore_index=True) if pieces else residual.head(0).copy()
    tr = pd.read_csv(v2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv", usecols=["physical_track_id", "object_uid"], low_memory=False)
    tr["physical_track_id"] = tr["physical_track_id"].astype(str); tr["object_uid"] = tr["object_uid"].astype(str)
    by_pid = tr.groupby("physical_track_id")["object_uid"].apply(list).to_dict()
    needed = sorted({u for pid in samples["physical_track_id"].astype(str) for u in by_pid.get(pid, [])})
    geom = {}
    for start in range(0, len(needed), 400):
        vals = ",".join("'" + u.replace("'", "''") + "'" for u in needed[start:start + 400])
        g = pyogrio.read_dataframe(str(gpkg), layer="tracking_buildings", where=f"object_uid IN ({vals})", columns=["object_uid", "scene_key"])
        for r in g.itertuples():
            geom[str(r.object_uid)] = r
    raw_paths = {s: root / "input/base" / f"{s}_zoom_20.tif" for s in SCENES}
    raw = {s: rasterio.open(raw_paths[s]) for s in SCENES}
    rows = []
    to_geo = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True).transform
    for rec in samples.to_dict("records"):
        interval = str(rec["interval"]); left, right = PAIRS[interval]; seq = str(rec["presence_sequence"])
        pid = str(rec["physical_track_id"]); active = [geom[u] for u in by_pid.get(pid, []) if u in geom]
        side = right if str(rec["target_presence"]) == "U" and str(rec["source_presence"]) == "1" else (left if str(rec["source_presence"]) == "U" and str(rec["target_presence"]) == "1" else "")
        target_scene = side
        if not active or not target_scene:
            rows.append({**rec, "target_scene_checked": target_scene, "candidate_count_in_buffer": -1, "valid_ratio": -1.0, "black_ratio": -1.0, "gray_std": -1.0, "absence_diagnostic": "NOT_APPLICABLE_SEQUENCE_OR_NO_ACTIVE_OBJECT"})
            continue
        union = unary_union([r.geometry for r in active if getattr(r, "geometry", None) is not None])
        if union.is_empty:
            rows.append({**rec, "target_scene_checked": target_scene, "candidate_count_in_buffer": -1, "valid_ratio": -1.0, "black_ratio": -1.0, "gray_std": -1.0, "absence_diagnostic": "NO_GEOMETRY"})
            continue
        cx, cy = union.centroid.x, union.centroid.y
        g = pyogrio.read_dataframe(str(gpkg), layer="tracking_buildings", bbox=(union.bounds[0] - 100, union.bounds[1] - 100, union.bounds[2] + 100, union.bounds[3] + 100), columns=["object_uid", "scene_key"])
        target_rows = [r for r in g.itertuples() if str(r.scene_key) == target_scene]
        buffered = union.buffer(90.0)
        nearby = [r for r in target_rows if getattr(r, "geometry", None) is not None and r.geometry.intersects(buffered)]
        q = quality(raw[target_scene], cx, cy)
        clear = q["valid_ratio"] >= 0.98 and q["black_ratio"] <= 0.05 and q["gray_std"] >= 3.0
        no_candidate = len(nearby) == 0
        if clear and no_candidate:
            diag = "POTENTIAL_RELIABLE_CLEAR_ABSENCE_REVIEW"
        elif not clear:
            diag = "RETAIN_UNKNOWN_PIXEL_QUALITY_NOT_CLEAR"
        else:
            diag = "RETAIN_UNKNOWN_OTHER_CANONICAL_OBJECT_IN_90M_BUFFER"
        rows.append({**rec, "target_scene_checked": target_scene, "candidate_count_in_buffer": len(nearby), **q, "clear_pixel_gate": clear, "no_candidate_in_buffer": no_candidate, "absence_diagnostic": diag})
    for src in raw.values():
        src.close()
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out / "phase3_compound_absence_diagnostic.csv", index=False, encoding="utf-8-sig")
    summary = out_df.groupby(["interval", "absence_diagnostic"], dropna=False).agg(rows=("physical_track_id", "size"), area_m2=("area_m2", "sum")).reset_index()
    summary.to_csv(out / "phase3_compound_absence_summary.csv", index=False, encoding="utf-8-sig")
    receipt = {"phase": f"PHASE_3_{args.mode.upper()}_CLEAR_ABSENCE_DIAGNOSTIC", "input_path": str(input_path), "input_rows": len(samples), "output_rows": len(out_df), "potential_clear_absence_rows": int((out_df["absence_diagnostic"] == "POTENTIAL_RELIABLE_CLEAR_ABSENCE_REVIEW").sum()), "potential_clear_absence_area_m2": float(out_df.loc[out_df["absence_diagnostic"] == "POTENTIAL_RELIABLE_CLEAR_ABSENCE_REVIEW", "area_m2"].sum()), "formal_outputs_modified": False, "promotion_count": 0, "rule": "clear 96x96 raw pixels plus no canonical B0 object in a conservative 90m buffer; diagnostic only, not a Presence/State promotion", "next_action": "targeted visual and interval-level absence review for potential rows"}
    (out / "PHASE3_ABSENCE_DIAGNOSTIC_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
