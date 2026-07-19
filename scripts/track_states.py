#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.quality import geometry_quality_fraction
from urban_monitor.tracking import assign_tracks, detect_lineage_events, summarize_tracks


def _manual_quality_overrides(review_csv: str | Path | None) -> dict[tuple[str, str], bool]:
    if not review_csv or not Path(review_csv).exists():
        return {}
    frame = pd.read_csv(review_csv, dtype=str).fillna("")
    if "date_quality_json" not in frame.columns:
        return {}
    if "row_type" in frame.columns:
        frame = frame[frame["row_type"] == "track"]
    overrides: dict[tuple[str, str], bool] = {}
    for _, row in frame.iterrows():
        track_id = str(row.get("track_id") or row.get("record_id") or "")
        try:
            values = json.loads(row.get("date_quality_json", "") or "{}")
        except json.JSONDecodeError:
            values = {}
        for date_value, quality in values.items():
            overrides[(track_id, str(date_value))] = str(quality) == "usable"
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_manifest(cfg["manifest"], cfg["data_root"])
    by_date = {}
    crs = None
    for record in records:
        regularized = Path(cfg["output_root"]) / "predictions_regularized" / f"{record.date}.gpkg"
        path = regularized if regularized.exists() else Path(cfg["output_root"]) / "predictions" / f"{record.date}.gpkg"
        frame = gpd.read_file(path)
        crs = crs or frame.crs
        by_date[record.date] = frame
    p = cfg["tracking"]
    date_metadata = {
        r.date: {
            "off_nadir_deg": r.off_nadir_deg,
            "sun_elevation_deg": r.sun_elevation_deg,
            "season": r.season,
            "source": r.source,
        }
        for r in records
    }
    observations = assign_tracks(
        by_date,
        min_iou=float(p["min_iou"]),
        max_centroid_distance_m=float(p["max_centroid_distance_m"]),
        max_track_gap=int(p.get("max_track_gap", 1)),
        max_track_gap_days=int(p["max_track_gap_days"]) if p.get("max_track_gap_days") is not None else None,
        buffer_m=float(p.get("buffer_m", 4.0)),
        min_buffer_iou=float(p.get("min_buffer_iou", 0.12)),
        min_bbox_iou=float(p.get("min_bbox_iou", 0.05)),
        match_weights=p.get("match_weights"),
        date_metadata=date_metadata,
        off_nadir_scale_m_per_degree=float(p.get("off_nadir_scale_m_per_degree", 0.35)),
        max_dynamic_buffer_m=float(p.get("max_dynamic_buffer_m", 20.0)),
        force_new_track_on_merge=bool(p.get("force_new_track_on_merge", True)),
        merge_component_overlap=float(p.get("lineage", {}).get("min_component_overlap", 0.25)),
        merge_parent_coverage=float(p.get("lineage", {}).get("min_event_coverage", 0.70)),
        merge_child_coverage=float(p.get("lineage", {}).get("min_counterpart_coverage", 0.25)),
        lineage_buffer_m=float(p.get("lineage", {}).get("buffer_m", 2.0)),
        ambiguity_margin_threshold=float(p.get("ambiguity_margin_threshold", 0.08)),
    )
    representative = {}
    for obs in observations:
        if obs.track_id not in representative or obs.score > representative[obs.track_id].score:
            representative[obs.track_id] = obs
    usability = {}
    min_quality = float(p.get("min_quality_fraction", 0.8))
    for track_id, obs in representative.items():
        for record in records:
            observed = next((o for o in observations if o.track_id == track_id and o.date == record.date), None)
            geometry = observed.geometry if observed is not None else obs.geometry
            fraction = geometry_quality_fraction(record.quality_mask_path, geometry)
            usability[(track_id, record.date)] = fraction >= min_quality
    usability.update(_manual_quality_overrides(cfg.get("review", {}).get("review_csv")))

    long_gdf, intervals, summary = summarize_tracks(
        observations,
        [r.date for r in records],
        min_persistence=int(p["min_persistence"]),
        isolated_gap_repair=bool(p["isolated_gap_repair"]),
        construction_area_ratio=float(p["construction_area_ratio"]),
        low_confidence_threshold=float(p["low_confidence_threshold"]),
        usability=usability,
        max_soft_gap_epochs=int(p.get("max_soft_gap_epochs", p.get("max_track_gap", 1))),
        max_soft_gap_days=int(p.get("max_soft_gap_days", 800)),
        construction_gap_buffer_m=float(p.get("construction_gap_buffer_m", p.get("buffer_m", 4.0))),
        construction_gap_max_buffer_iou=float(p.get("construction_gap_max_buffer_iou", 0.35)),
    )
    long_gdf.set_crs(crs, allow_override=True, inplace=True)
    summary.set_crs(crs, allow_override=True, inplace=True)

    lineage_cfg = p.get("lineage", {})
    lineage = detect_lineage_events(
        by_date,
        observations,
        min_component_overlap=float(lineage_cfg.get("min_component_overlap", 0.25)),
        min_event_coverage=float(lineage_cfg.get("min_event_coverage", 0.70)),
        min_counterpart_coverage=float(lineage_cfg.get("min_counterpart_coverage", 0.25)),
        lineage_buffer_m=float(lineage_cfg.get("buffer_m", 2.0)),
    )
    if not lineage.empty:
        parent_counts: dict[str, int] = {}
        child_counts: dict[str, int] = {}
        for _, row in lineage.iterrows():
            for tid in str(row.get("parent_track_ids", "")).split(";"):
                if tid:
                    parent_counts[tid] = parent_counts.get(tid, 0) + 1
            for tid in str(row.get("child_track_ids", "")).split(";"):
                if tid:
                    child_counts[tid] = child_counts.get(tid, 0) + 1
        summary["lineage_parent_events"] = summary["track_id"].map(parent_counts).fillna(0).astype(int)
        summary["lineage_child_events"] = summary["track_id"].map(child_counts).fillna(0).astype(int)
        summary.loc[(summary["lineage_parent_events"] + summary["lineage_child_events"]) > 0, "review_status"] = "priority"
    else:
        summary["lineage_parent_events"] = 0
        summary["lineage_child_events"] = 0

    output = Path(cfg["output_root"]) / "tracking"
    output.mkdir(parents=True, exist_ok=True)
    long_gdf.to_file(output / "tracks_long.gpkg", driver="GPKG")
    summary.to_file(output / "tracks_summary.gpkg", driver="GPKG")
    intervals.to_csv(output / "state_intervals.csv", index=False, encoding="utf-8-sig")
    long_gdf.drop(columns="geometry").to_csv(output / "observation_matrix.csv", index=False, encoding="utf-8-sig")
    if not lineage.empty:
        lineage.to_file(output / "lineage_events.gpkg", driver="GPKG")
        lineage.drop(columns="geometry").to_csv(output / "lineage_events.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(
            columns=[
                "event_id", "event_type", "date_from", "date_to", "parent_track_ids", "child_track_ids",
                "n_parent", "n_child", "coverage", "review_status"
            ]
        ).to_csv(output / "lineage_events.csv", index=False, encoding="utf-8-sig")
    print(summary[["track_id", "state", "confidence", "temporal_flag", "first_seen", "last_seen"]].to_string(index=False))


if __name__ == "__main__":
    main()
