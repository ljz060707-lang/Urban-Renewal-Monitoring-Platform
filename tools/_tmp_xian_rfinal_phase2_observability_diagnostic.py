from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


BASE = Path("/data/xian_r4_100km2_20260914")
V2 = BASE / "output/16_candidate_identitygraph_v2_20260916"
FORMAL = BASE / "output/西安_R4_辅助影像接入前_三期三态候选版_20260914"
RESIDUAL = V2 / "RESIDUAL_UNKNOWN_PHASE_B_V2.csv"
TRACKING = V2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv"
GPKG = FORMAL / "04_identity_graph/tracking_buildings.gpkg"
RASTERS = {
    "xa180429": BASE / "input/base/xa180429_zoom_20.tif",
    "xa190101": BASE / "input/base/xa190101_zoom_20.tif",
    "xa201231": BASE / "input/base/xa201231_zoom_20.tif",
}
SCENE_DATE = {"xa180429": "2018-04-29", "xa190101": "2019-01-01", "xa201231": "2020-12-31"}
SCENES = ["xa180429", "xa190101", "xa201231"]


def pct(value: float, total: float) -> float:
    return float(value / total * 100.0) if total else 0.0


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "output/19_rfinal_phase2_observability_diagnostic_20260916"
    out.mkdir(parents=True, exist_ok=True)

    residual = pd.read_csv(RESIDUAL, low_memory=False)
    residual["area_m2"] = pd.to_numeric(residual["area_m2"], errors="coerce").fillna(0.0)
    # Only the largest area U05 group is probed first. Identity/compound/parallax blockers stay closed.
    eligible = residual[
        residual["unknown_cause"].eq("U05_UNOBSERVABLE")
        & residual["area_m2"].gt(0)
        & ~residual["identity_sources"].fillna("").str.contains("COMPOUND|PARALLAX", case=False, regex=True)
        & ~residual["presence_sequence"].isin(["1-0-1", "0-1-0"])
    ].copy()
    eligible = eligible.sort_values("area_m2", ascending=False)
    # Area-priority budget: at most 6,000 residual rows per interval in this diagnostic round.
    selected = (
        eligible.groupby("interval", group_keys=False)
        .head(6000)
        .sort_values("area_m2", ascending=False)
        .copy()
    )
    selected["missing_scene"] = selected.apply(
        lambda r: SCENES[SCENES.index("xa180429") + 0] if False else "", axis=1
    )

    # Derive the missing scenes from the three-state sequence. A row can contribute multiple missing scenes;
    # retaining the same track/scene once prevents duplicate pixel reads.
    missing_rows = []
    for row in selected.itertuples(index=False):
        vals = str(row.presence_sequence).split("-")
        for scene, val in zip(SCENES, vals):
            if val == "U":
                missing_rows.append(
                    {
                        "physical_track_id": str(row.physical_track_id),
                        "interval": str(row.interval),
                        "scene_key": scene,
                        "presence_sequence": str(row.presence_sequence),
                        "area_m2": float(row.area_m2),
                        "unknown_cause": str(row.unknown_cause),
                        "identity_sources": "" if pd.isna(row.identity_sources) else str(row.identity_sources),
                        "source_presence": str(row.source_presence),
                        "target_presence": str(row.target_presence),
                    }
                )
    missing = pd.DataFrame(missing_rows).drop_duplicates(["physical_track_id", "scene_key"])
    if missing.empty:
        raise RuntimeError("no eligible U05 missing-scene rows")

    target_tracks = set(missing["physical_track_id"].astype(str))
    track_cols = ["physical_track_id", "object_uid", "scene_key", "area_m2", "building_id", "canonical_object_id"]
    tracks = pd.read_csv(TRACKING, usecols=track_cols, low_memory=False)
    tracks["physical_track_id"] = tracks["physical_track_id"].astype(str)
    tracks["area_m2"] = pd.to_numeric(tracks["area_m2"], errors="coerce").fillna(0.0)
    tracks = tracks[tracks["physical_track_id"].isin(target_tracks)].copy()
    # Object UIDs are unique in the canonical object table. Querying the formal GPKG in bounded batches
    # avoids copying the 3 GB geometry database and retains only one centroid per physical track.
    obj_to_track = tracks[["object_uid", "physical_track_id"]].drop_duplicates("object_uid").set_index("object_uid")["physical_track_id"].to_dict()
    import pyogrio

    feature_count = int(pyogrio.read_info(str(GPKG))["features"])
    chunks: list[pd.DataFrame] = []
    for offset in range(0, feature_count, 50000):
        gdf = pyogrio.read_dataframe(str(GPKG), columns=["object_uid"], skip_features=offset, max_features=50000)
        if gdf.empty:
            continue
        gdf["physical_track_id"] = gdf["object_uid"].map(obj_to_track)
        gdf = gdf[gdf["physical_track_id"].notna()].copy()
        if gdf.empty:
            continue
        cent = gdf.geometry.centroid
        gdf["x"] = cent.x.to_numpy()
        gdf["y"] = cent.y.to_numpy()
        chunks.append(gdf.groupby("physical_track_id", as_index=False).agg(x=("x", "mean"), y=("y", "mean")))
    if not chunks:
        raise RuntimeError("no centroids matched selected physical tracks")
    centroids = pd.concat(chunks, ignore_index=True).groupby("physical_track_id", as_index=False).agg(x=("x", "mean"), y=("y", "mean"))
    missing = missing.merge(centroids, on="physical_track_id", how="left")
    missing = missing[missing["x"].notna() & missing["y"].notna()].copy()

    # The raster quality probe uses the same conservative core thresholds as the current system, plus
    # a bounded multi-scale check. It is evidence for a possible closure only; it is not a promotion.
    transformer_cache = {}
    source_cache: dict[str, rasterio.DatasetReader] = {}
    block_cache: dict[tuple[str, int, int], dict] = {}

    def quality(scene: str, x: float, y: float) -> dict:
        src = source_cache[scene]
        if scene not in transformer_cache:
            transformer_cache[scene] = Transformer.from_crs("EPSG:32649", src.crs, always_xy=True)
        lon, lat = transformer_cache[scene].transform(float(x), float(y))
        row, col = src.index(lon, lat)
        key = (scene, max(0, row // 128), max(0, col // 128))
        if key in block_cache:
            result = dict(block_cache[key])
            result.update({"lon": float(lon), "lat": float(lat), "row": int(row), "col": int(col)})
            return result
        size = 96
        r0, c0 = max(0, row - size // 2), max(0, col - size // 2)
        h, w = min(size, src.height - r0), min(size, src.width - c0)
        if h <= 0 or w <= 0:
            result = {"valid_fraction": 0.0, "black_fraction": 1.0, "gray_std": 0.0, "gray_p10": 0.0, "gray_p90": 0.0, "clear": False, "quality_reason": "OUT_OF_BOUNDS"}
        else:
            values = src.read([1, 2, 3], window=rasterio.windows.Window(c0, r0, w, h), out_shape=(3, min(64, h), min(64, w)), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            gray = values.mean(axis=0)
            mask = src.dataset_mask(window=rasterio.windows.Window(c0, r0, w, h))
            valid_fraction = float((mask > 0).mean()) if mask.size else 0.0
            black_fraction = float((gray <= 1.0).mean()) if gray.size else 1.0
            std = float(gray.std()) if gray.size else 0.0
            p10, p90 = np.percentile(gray, [10, 90]) if gray.size else (0.0, 0.0)
            clear = bool(valid_fraction >= 0.98 and black_fraction <= 0.05 and std >= 3.0)
            result = {
                "valid_fraction": valid_fraction,
                "black_fraction": black_fraction,
                "gray_std": std,
                "gray_p10": float(p10),
                "gray_p90": float(p90),
                "clear": clear,
                "quality_reason": "CLEAR_CORE_RULE" if clear else "CORE_RULE_NOT_MET",
            }
        block_cache[key] = dict(result)
        result.update({"lon": float(lon), "lat": float(lat), "row": int(row), "col": int(col)})
        return result

    try:
        for scene, path in RASTERS.items():
            source_cache[scene] = rasterio.open(path)
        qrows = []
        for row in missing.itertuples(index=False):
            q = quality(str(row.scene_key), float(row.x), float(row.y))
            qrows.append({**row._asdict(), **q, "formal_zero_candidate": bool(q["clear"]), "candidate_reason": "FRESH_MAIN_PIXEL_CLEAR_AND_NO_COMPOUND_OR_PARALLAX" if q["clear"] else "FRESH_MAIN_PIXEL_NOT_CLEAR"})
    finally:
        for src in source_cache.values():
            src.close()

    result = pd.DataFrame(qrows)
    result = result.sort_values("area_m2", ascending=False)
    result.to_csv(out / "phase2_fresh_observability_candidates.csv", index=False, encoding="utf-8-sig")
    summary = (
        result.groupby(["interval", "scene_key", "formal_zero_candidate"], dropna=False)
        .agg(rows=("physical_track_id", "size"), area_m2=("area_m2", "sum"), mean_black_fraction=("black_fraction", "mean"), mean_gray_std=("gray_std", "mean"))
        .reset_index()
    )
    summary.to_csv(out / "phase2_fresh_observability_summary.csv", index=False, encoding="utf-8-sig")

    potential = result[result["formal_zero_candidate"]].copy()
    potential_by_interval = potential.groupby("interval")["area_m2"].sum().to_dict()
    receipt = {
        "phase": "PHASE_2_ISOLATED_FRESH_MAIN_PIXEL_OBSERVABILITY_DIAGNOSTIC",
        "input_residual": str(RESIDUAL),
        "selection_rule": "U05_UNOBSERVABLE, positive area, no COMPOUND/PARALLAX identity source, exclude 1-0-1 and 0-1-0 for this first round, top 6000 rows per interval by area",
        "selected_residual_rows": int(len(selected)),
        "missing_track_scene_rows": int(len(missing)),
        "tracks_with_centroid": int(missing["physical_track_id"].nunique()),
        "raster_quality_rule": "valid_fraction>=0.98, black_fraction<=0.05, gray_std>=3.0 on 96x96 local window; cached by 128-pixel block",
        "fresh_clear_candidate_rows": int(len(potential)),
        "fresh_clear_candidate_area_m2_by_interval": {str(k): float(v) for k, v in potential_by_interval.items()},
        "fresh_clear_candidate_area_m2_total": float(potential["area_m2"].sum()),
        "block_cache_entries": int(len(block_cache)),
        "formal_outputs_modified": False,
        "candidate_not_ground_truth": True,
        "promotion_status": "NOT_PROMOTED_PENDING_DATE_SUPPORT_IDENTITY_GATE_VISUAL_QA",
        "next_action": "audit candidate date support and identity competitors, then generate targeted visual QA before any Presence/State candidate rerun",
    }
    (out / "PHASE2_OBSERVABILITY_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
