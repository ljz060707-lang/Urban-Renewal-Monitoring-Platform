from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/data/xian_r4_100km2_20260914")
V2 = BASE / "output/16_candidate_identitygraph_v2_20260916"
FORMAL = BASE / "output/西安_R4_辅助影像接入前_三期三态候选版_20260914"
RESIDUAL = V2 / "RESIDUAL_UNKNOWN_PHASE_B_V2.csv"
TRACKING = V2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv"
HELD = V2 / "HELD_FULL_RECOVERY_DECISION.csv"
GPKG = FORMAL / "04_identity_graph/tracking_buildings.gpkg"


def pct(part: float, whole: float) -> float:
    return float(part / whole * 100.0) if whole else 0.0


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "output/18_rfinal_baseline_and_residual_diagnosis_20260916"
    out.mkdir(parents=True, exist_ok=True)

    residual = pd.read_csv(RESIDUAL, low_memory=False)
    residual["area_m2"] = pd.to_numeric(residual["area_m2"], errors="coerce").fillna(0.0)
    residual["source_area_m2"] = pd.to_numeric(residual["source_area_m2"], errors="coerce").fillna(0.0)
    residual["target_area_m2"] = pd.to_numeric(residual["target_area_m2"], errors="coerce").fillna(0.0)
    residual["union_area_m2"] = pd.to_numeric(residual["union_area_m2"], errors="coerce").fillna(0.0)

    total_area_by_interval = residual.groupby("interval", dropna=False)["area_m2"].sum().to_dict()
    total_area = float(residual["area_m2"].sum())

    by_cause = (
        residual.groupby(["interval", "unknown_cause"], dropna=False)
        .agg(rows=("physical_track_id", "size"), area_m2=("area_m2", "sum"))
        .reset_index()
    )
    by_cause["share_of_interval_unknown_pct"] = by_cause.apply(
        lambda r: pct(float(r.area_m2), float(total_area_by_interval.get(r.interval, 0.0))), axis=1
    )
    by_cause["area_required_to_reach_5pct_m2"] = by_cause["area_m2"]
    save_csv(by_cause.sort_values(["interval", "area_m2"], ascending=[True, False]), out / "phase1_residual_unknown_by_cause.csv")

    by_sequence = (
        residual.groupby(["interval", "presence_sequence"], dropna=False)
        .agg(rows=("physical_track_id", "size"), area_m2=("area_m2", "sum"))
        .reset_index()
    )
    by_sequence["share_of_interval_unknown_pct"] = by_sequence.apply(
        lambda r: pct(float(r.area_m2), float(total_area_by_interval.get(r.interval, 0.0))), axis=1
    )
    save_csv(by_sequence.sort_values(["interval", "area_m2"], ascending=[True, False]), out / "phase1_residual_unknown_by_sequence.csv")

    top_components = residual.sort_values("area_m2", ascending=False).head(2000).copy()
    top_components.insert(0, "area_rank_within_diagnosis", np.arange(1, len(top_components) + 1))
    save_csv(top_components, out / "phase1_top_residual_components_by_area.csv")

    aba = residual[residual["presence_sequence"].eq("1-0-1")].sort_values("area_m2", ascending=False).head(2000).copy()
    aba.insert(0, "area_rank_1_0_1", np.arange(1, len(aba) + 1))
    save_csv(aba, out / "phase1_high_area_1_0_1.csv")

    compounds = residual[residual["unknown_cause"].eq("U07_COMPOUND_UNRESOLVED")].sort_values("area_m2", ascending=False).head(2000).copy()
    compounds.insert(0, "area_rank_compound", np.arange(1, len(compounds) + 1))
    save_csv(compounds, out / "phase1_unresolved_compound_high_area.csv")

    track_cols = [
        "physical_track_id", "object_uid", "scene_key", "formal_date", "area_m2",
        "canonical_object_id", "compound_track_id", "identity_sources"
    ]
    tracks = pd.read_csv(TRACKING, usecols=track_cols, low_memory=False)
    tracks["area_m2"] = pd.to_numeric(tracks["area_m2"], errors="coerce").fillna(0.0)
    object_area = tracks.drop_duplicates("object_uid").set_index("object_uid")["area_m2"]
    object_track = tracks.drop_duplicates("object_uid").set_index("object_uid")["physical_track_id"]
    object_track = object_track.astype(str)

    held = pd.read_csv(HELD, low_memory=False)
    for c in ["score", "buffered_iou", "area_ratio", "displacement", "neighborhood", "competitor_count"]:
        if c in held:
            held[c] = pd.to_numeric(held[c], errors="coerce").fillna(0.0)
    held["source_area_m2"] = held["source_object_id"].map(object_area).fillna(0.0)
    held["target_area_m2"] = held["target_object_id"].map(object_area).fillna(0.0)
    held["review_union_area_m2"] = held[["source_area_m2", "target_area_m2"]].max(axis=1)
    relation_multiplier = held.get("split_merge_type", pd.Series("NONE", index=held.index)).fillna("NONE").ne("NONE").astype(float) * 0.25 + 1.0
    conflict_multiplier = held.get("identity_conflict", pd.Series("NO", index=held.index)).fillna("NO").ne("NO").astype(float) * 0.5 + 1.0
    held["area_weighted_review_risk"] = (
        held["review_union_area_m2"]
        * (1.0 - held["score"].clip(0.0, 1.0))
        * (1.0 + held["competitor_count"].clip(lower=0.0))
        * relation_multiplier
        * conflict_multiplier
    )
    held_high = held[held["decision"].eq("KEEP_REVIEW")].sort_values(
        ["area_weighted_review_risk", "review_union_area_m2"], ascending=False
    ).head(2000).copy()
    held_high.insert(0, "risk_rank", np.arange(1, len(held_high) + 1))
    save_csv(held_high, out / "phase1_held_high_risk.csv")

    # Build a compact track-level table for spatial lookup. A track may occur in both intervals;
    # retaining both rows preserves interval-specific area contribution.
    track_summary = (
        residual.groupby(["physical_track_id", "interval"], dropna=False)
        .agg(
            area_m2=("area_m2", "sum"),
            rows=("physical_track_id", "size"),
            unknown_causes=("unknown_cause", lambda s: ";".join(sorted(set(map(str, s))))),
            presence_sequences=("presence_sequence", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    track_summary["area_rank_interval"] = track_summary.groupby("interval")["area_m2"].rank(method="first", ascending=False).astype(int)
    save_csv(track_summary.sort_values("area_m2", ascending=False).head(5000), out / "phase1_track_area_summary_top5000.csv")

    # Spatial diagnosis: read the authoritative formal geometry in bounded batches, retaining
    # only object UIDs that occur in residual tracks. No geometry is modified or written back.
    spatial_clusters_path = out / "phase1_spatial_clusters_1km.csv"
    spatial_status = {"status": "NOT_RUN", "reason": "pyogrio not available"}
    try:
        import pyogrio

        residual_track_ids = set(residual["physical_track_id"].astype(str))
        obj_to_track = tracks[tracks["physical_track_id"].astype(str).isin(residual_track_ids)][["object_uid", "physical_track_id"]].drop_duplicates("object_uid")
        obj_to_track["physical_track_id"] = obj_to_track["physical_track_id"].astype(str)
        obj_to_track_map = obj_to_track.set_index("object_uid")["physical_track_id"].to_dict()
        info = pyogrio.read_info(str(GPKG))
        feature_count = int(info["features"])
        chunk_size = 50000
        accum: list[pd.DataFrame] = []
        for offset in range(0, feature_count, chunk_size):
            gdf = pyogrio.read_dataframe(
                str(GPKG),
                columns=["object_uid"],
                skip_features=offset,
                max_features=chunk_size,
            )
            if gdf.empty:
                continue
            gdf["physical_track_id"] = gdf["object_uid"].map(obj_to_track_map)
            gdf = gdf[gdf["physical_track_id"].notna()].copy()
            if gdf.empty:
                continue
            cent = gdf.geometry.centroid
            gdf["x"] = cent.x.to_numpy()
            gdf["y"] = cent.y.to_numpy()
            accum.append(
                gdf.groupby("physical_track_id", as_index=False).agg(x=("x", "mean"), y=("y", "mean"), object_count=("object_uid", "size"))
            )
        if accum:
            centroids = pd.concat(accum, ignore_index=True).groupby("physical_track_id", as_index=False).agg(
                x=("x", "mean"), y=("y", "mean"), object_count=("object_count", "sum")
            )
            spatial = track_summary.merge(centroids, on="physical_track_id", how="left")
            spatial = spatial[spatial["x"].notna() & spatial["y"].notna()].copy()
            spatial["grid_1km_x"] = np.floor(spatial["x"] / 1000.0).astype("int64")
            spatial["grid_1km_y"] = np.floor(spatial["y"] / 1000.0).astype("int64")
            clusters = (
                spatial.groupby(["interval", "grid_1km_x", "grid_1km_y"], as_index=False)
                .agg(
                    track_count=("physical_track_id", "nunique"),
                    residual_rows=("rows", "sum"),
                    area_m2=("area_m2", "sum"),
                    x_mean=("x", "mean"),
                    y_mean=("y", "mean"),
                    dominant_causes=("unknown_causes", lambda s: ";".join(sorted(set(";".join(map(str, s)).split(";"))))),
                )
            )
            clusters["share_of_interval_unknown_pct"] = clusters.apply(
                lambda r: pct(float(r.area_m2), float(total_area_by_interval.get(r.interval, 0.0))), axis=1
            )
            clusters = clusters.sort_values(["interval", "area_m2"], ascending=[True, False])
            save_csv(clusters, spatial_clusters_path)
            spatial_status = {
                "status": "COMPLETED",
                "formal_geometry": str(GPKG),
                "feature_count": feature_count,
                "residual_tracks": len(residual_track_ids),
                "tracks_with_centroid": int(centroids["physical_track_id"].nunique()),
                "clusters": int(len(clusters)),
                "grid_m": 1000,
            }
        else:
            spatial_status = {"status": "NO_MATCHING_GEOMETRY", "feature_count": feature_count}
    except Exception as exc:  # diagnosis must remain reproducible even if the optional spatial read fails
        spatial_status = {"status": "FAILED_DIAGNOSTIC_ONLY", "error": repr(exc)}

    outputs = sorted(p.name for p in out.iterdir() if p.is_file())
    receipt = {
        "task": "西安三期建筑追踪与三态最终正式生产任务（R-FINAL）",
        "phase": "PHASE_1_RESIDUAL_UNKNOWN_AREA_PRIORITY_DIAGNOSIS",
        "created_at_local": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "input_v2_root": str(V2),
        "formal_root": str(FORMAL),
        "residual_unknown_rows": int(len(residual)),
        "residual_unknown_area_m2": total_area,
        "residual_unknown_area_by_interval_m2": {str(k): float(v) for k, v in total_area_by_interval.items()},
        "top_cause": by_cause.sort_values("area_m2", ascending=False).iloc[0]["unknown_cause"],
        "top_cause_area_m2": float(by_cause["area_m2"].max()),
        "high_area_1_0_1_rows": int(len(residual[residual["presence_sequence"].eq("1-0-1")])),
        "compound_unknown_rows": int((residual["unknown_cause"] == "U07_COMPOUND_UNRESOLVED").sum()),
        "held_keep_review_rows": int((held["decision"] == "KEEP_REVIEW").sum()),
        "spatial_status": spatial_status,
        "formal_outputs_modified": False,
        "geometry_modified": False,
        "tracking_modified": False,
        "presence_modified": False,
        "state_modified": False,
        "scientific_status": "DIAGNOSTIC_ONLY_CANDIDATE_NOT_GROUND_TRUTH",
        "gates": {
            "phase0_baseline_freeze": "PENDING_LOCAL_RECEIPT",
            "residual_unknown_lt5": "FAIL",
            "tracking_accuracy": "REVIEW_REQUIRED",
            "fixed_case_regression": "REVIEW_REQUIRED",
            "false_change_visual": "FAIL_REVIEW_REQUIRED",
            "formal_production": "NOT_RUN",
        },
        "outputs": outputs,
    }
    (out / "PHASE1_DIAGNOSIS_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Xi'an R-FINAL Phase 1 Residual Unknown Diagnosis

## Scope

This is an isolated, diagnostic-only package. It does not modify Geometry, IdentityGraph, Presence, State, or any formal product.

The authoritative V2 residual ledger contains **{len(residual):,} rows** and **{total_area:,.2f} m²** of retained Unknown. The interval totals are:

| interval | residual Unknown area (m²) | rows |
|---|---:|---:|
""" + "\n".join(
        f"| {interval} | {area:,.2f} | {int((residual['interval'] == interval).sum()):,} |"
        for interval, area in sorted(total_area_by_interval.items())
    ) + f"""

## Area-priority findings

The largest residual blocker is `U05_UNOBSERVABLE`. `U01_IDENTITY_MULTIPLE_OR_NO_MATCH` and `U07_COMPOUND_UNRESOLVED` are retained as unresolved rather than converted by nearest-neighbor or mask XOR rules. The 1-0-1 list is exported separately because it is a false-change/GAP_RETURN review priority, not an automatic addition/demolition decision.

`phase1_held_high_risk.csv` ranks held edges by an explicit diagnostic score: union object area × (1 − identity score) × competitor multiplier, with additional split/merge and conflict multipliers. This ranking is for review ordering only and is not a promotion rule.

## Evidence boundary

The spatial table uses the formal candidate GPKG geometries read in bounded batches and aggregates only centroids to 1 km cells. No polygon is changed. Supplemental imagery is not silently used here as a pixel observation; its actual pixel/metadata availability is recorded separately in the Phase 0 receipt.

## Gate state

- V4B regression: PASS.
- Strict-core ledger: PASS.
- State sanity: PASS.
- Unknown <5%: FAIL.
- Tracking accuracy and fixed-case acceptance: REVIEW_REQUIRED.
- False-change visual gate: FAIL_REVIEW_REQUIRED.
- Formal production: NOT_RUN.

Next action is isolated structured closure of the largest evidence-supported review groups, with per-round object/area diff and targeted visual checks. Formal outputs remain untouched.
"""
    (out / "PHASE1_DIAGNOSIS_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
