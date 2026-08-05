"""Match the extracted 2020 GloBFP slice to sealed RSBuilding 2019/2022."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FULL20_CHANGE_THRESHOLD = 0.75
EDGE_DISTANCE_M = 20.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_tracking(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["previous_object_id"]] = row
    return rows


def choose_best_matches(glo, rs, metric_crs: str) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    from shapely.strtree import STRtree  # type: ignore

    glo_metric = glo.to_crs(metric_crs)
    rs_metric = rs.to_crs(metric_crs)
    tree = STRtree(list(rs_metric.geometry))
    records: list[dict[str, Any]] = []
    used_rs: set[int] = set()
    for glo_index, glo_row in glo_metric.iterrows():
        glo_geom = glo_row.geometry
        candidate_indices = tree.query(glo_geom, predicate="intersects")
        candidates = []
        for candidate_index in candidate_indices:
            rs_row = rs_metric.iloc[int(candidate_index)]
            rs_geom = rs_row.geometry
            intersection = glo_geom.intersection(rs_geom)
            intersection_area = float(intersection.area)
            if intersection_area <= 0:
                continue
            union_area = float(glo_geom.union(rs_geom).area)
            iou = intersection_area / union_area if union_area else 0.0
            candidates.append(
                {
                    "rs_position": int(candidate_index),
                    "rs_object_id": str(rs_row["object_id"]),
                    "intersection_area_m2": intersection_area,
                    "union_area_m2": union_area,
                    "footprint_iou": iou,
                    "globfp_coverage_rate": intersection_area / float(glo_geom.area) if glo_geom.area else 0.0,
                    "rsbuilding_coverage_rate": intersection_area / float(rs_geom.area) if rs_geom.area else 0.0,
                    "centroid_distance_m": float(glo_geom.centroid.distance(rs_geom.centroid)),
                    "area_ratio": float(glo_geom.area / rs_geom.area) if rs_geom.area else None,
                    "boundary_distance_m": float(glo_geom.boundary.distance(rs_geom.boundary)),
                }
            )
        candidates.sort(key=lambda item: (item["footprint_iou"], item["intersection_area_m2"]), reverse=True)
        best = candidates[0] if candidates else None
        if best:
            used_rs.add(best["rs_position"])
        row = {
            "globfp_position": int(glo_index),
            "globfp_object_id": str(glo_row["globfp_object_id"]),
            "rsbuilding_object_id": best["rs_object_id"] if best else None,
            "match_found": bool(best),
            "candidate_match_count": len(candidates),
            "one_to_many": len(candidates) > 1,
            "unmatched_globfp": not bool(best),
            **({key: value for key, value in best.items() if key != "rs_position"} if best else {
                "intersection_area_m2": 0.0,
                "union_area_m2": None,
                "footprint_iou": 0.0,
                "globfp_coverage_rate": 0.0,
                "rsbuilding_coverage_rate": 0.0,
                "centroid_distance_m": None,
                "area_ratio": None,
                "boundary_distance_m": None,
            }),
        }
        records.append(row)
    unmatched_rs = set(range(len(rs_metric))) - used_rs
    return records, used_rs, unmatched_rs


def full20_metrics(glo, raster_path: Path) -> dict[str, dict[str, Any]]:
    import numpy as np  # type: ignore
    import rasterio  # type: ignore
    from rasterio.features import geometry_mask  # type: ignore

    output: dict[str, dict[str, Any]] = {}
    with rasterio.open(raster_path) as dataset:
        values = dataset.read(1).astype("float32")
        for _, row in glo.iterrows():
            geom = row.geometry
            try:
                mask = geometry_mask([geom], out_shape=values.shape, transform=dataset.transform, invert=True, all_touched=True)
                selected = values[mask]
                selected = selected[np.isfinite(selected)]
                if selected.size:
                    output[str(row["globfp_object_id"])] = {
                        "full20_any_change_mean": float(selected.mean()),
                        "full20_any_change_max": float(selected.max()),
                        "full20_any_change_p95": float(np.quantile(selected, 0.95)),
                        "full20_pixels_observed": int(selected.size),
                        "full20_low_change": bool(float(selected.max()) < FULL20_CHANGE_THRESHOLD),
                        "full20_threshold": FULL20_CHANGE_THRESHOLD,
                    }
                else:
                    output[str(row["globfp_object_id"])] = {
                        "full20_any_change_mean": None,
                        "full20_any_change_max": None,
                        "full20_any_change_p95": None,
                        "full20_pixels_observed": 0,
                        "full20_low_change": False,
                        "full20_threshold": FULL20_CHANGE_THRESHOLD,
                    }
            except Exception as exc:
                output[str(row["globfp_object_id"])] = {
                    "full20_any_change_mean": None,
                    "full20_any_change_max": None,
                    "full20_any_change_p95": None,
                    "full20_pixels_observed": 0,
                    "full20_low_change": False,
                    "full20_threshold": FULL20_CHANGE_THRESHOLD,
                    "full20_error": f"{type(exc).__name__}: {exc}",
                }
    return output


def run_match(project_root: Path, data_root: Path, old_modular_root: Path, output_root: Path) -> dict[str, Any]:
    import geopandas as gpd  # type: ignore
    import pyogrio  # type: ignore
    from shapely.geometry import box  # type: ignore

    match_root = output_root / "02_globfp_matching"
    match_root.mkdir(parents=True, exist_ok=True)
    extract_path = data_root / "01_nanjing_extract" / "globfp_nanjing_valid.gpkg"
    glo = pyogrio.read_dataframe(extract_path, layer="globfp_nanjing_valid")
    old_rs_root = old_modular_root / "03_baseline_rsbuilding"
    rs_frames = {}
    for date in ("2019-03-16", "2022-12-02"):
        path = old_rs_root / date / "building_objects.gpkg"
        rs_frames[date] = pyogrio.read_dataframe(path, layer="buildings")
    tracking_path = old_modular_root / "05_baseline_tracking" / "cumulative_primary" / "2019-03-16_to_2022-12-02" / "BUILDING_MATCHES.csv"
    tracking = read_tracking(tracking_path)
    full20_path = old_modular_root / "02_baseline_full20" / "2019-03-16_to_2022-12-02" / "rasters" / "any_change_probability.tif"
    full20 = full20_metrics(glo.to_crs("EPSG:4326"), full20_path)
    metric_crs = "EPSG:32650"
    glo_metric = glo.to_crs(metric_crs)
    aoi_bounds = json.loads((data_root / "01_nanjing_extract" / "EXTRACTION_MANIFEST.json").read_text(encoding="utf-8"))["audit"]["bounds"]["sealed_common_grid_bounds_epsg4326"]
    aoi_metric = gpd.GeoSeries([box(*aoi_bounds)], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    records_by_date: dict[str, list[dict[str, Any]]] = {}
    used_by_date: dict[str, set[int]] = {}
    unmatched_by_date: dict[str, set[int]] = {}
    for date, rs in rs_frames.items():
        records, used, unmatched = choose_best_matches(glo, rs, metric_crs)
        records_by_date[date] = records
        used_by_date[date] = used
        unmatched_by_date[date] = unmatched

    rows: list[dict[str, Any]] = []
    for position, (_, glo_row) in enumerate(glo_metric.iterrows()):
        gid = str(glo_row["globfp_object_id"])
        r19 = records_by_date["2019-03-16"][position]
        r22 = records_by_date["2022-12-02"][position]
        track = tracking.get(r19.get("rsbuilding_object_id"), {}) if r19.get("rsbuilding_object_id") else {}
        edge_distance = float(glo_row.geometry.distance(glo_row.geometry.intersection(aoi_metric).boundary)) if glo_row.geometry.intersects(aoi_metric) else 0.0
        inside_aoi = bool(glo_row.geometry.within(aoi_metric))
        temporal_state = track.get("geometry_state") or "unmatched_or_not_in_sealed_tracking"
        stable = bool(
            r19["match_found"]
            and r22["match_found"]
            and r19["footprint_iou"] >= 0.30
            and r22["footprint_iou"] >= 0.30
            and track.get("current_object_id") == r22.get("rsbuilding_object_id")
            and temporal_state == "stable_building"
            and float(track.get("match_score") or 0.0) >= 0.50
            and 0.67 <= float(track.get("area_ratio") or 0.0) <= 1.50
            and full20[gid]["full20_low_change"]
            and inside_aoi
            and edge_distance >= EDGE_DISTANCE_M
            and bool(glo_row.get("height_valid", True))
        )
        conflict = bool(
            r19["match_found"] != r22["match_found"]
            or (temporal_state not in {"stable_building", "unmatched_or_not_in_sealed_tracking"} and r19["match_found"] and r22["match_found"])
            or not full20[gid]["full20_low_change"]
        )
        rows.append(
            {
                "globfp_object_id": gid,
                "source_grid_id": str(glo_row.get("source_grid_id")),
                "globfp_height_m": float(glo_row["globfp_height_m"]),
                "height_valid": bool(glo_row["height_valid"]),
                "inside_common_grid_aoi": inside_aoi,
                "edge_distance_m": edge_distance,
                "rsbuilding_2019_object_id": r19.get("rsbuilding_object_id"),
                "rsbuilding_2019_match_found": r19["match_found"],
                "rsbuilding_2019_candidate_match_count": r19["candidate_match_count"],
                "rsbuilding_2019_one_to_many": r19["one_to_many"],
                "rsbuilding_2019_intersection_area_m2": r19["intersection_area_m2"],
                "rsbuilding_2019_union_area_m2": r19["union_area_m2"],
                "rsbuilding_2019_footprint_iou": r19["footprint_iou"],
                "rsbuilding_2019_globfp_coverage_rate": r19["globfp_coverage_rate"],
                "rsbuilding_2019_rsbuilding_coverage_rate": r19["rsbuilding_coverage_rate"],
                "rsbuilding_2019_centroid_distance_m": r19["centroid_distance_m"],
                "rsbuilding_2019_area_ratio": r19["area_ratio"],
                "rsbuilding_2019_boundary_distance_m": r19["boundary_distance_m"],
                "rsbuilding_2022_object_id": r22.get("rsbuilding_object_id"),
                "rsbuilding_2022_match_found": r22["match_found"],
                "rsbuilding_2022_candidate_match_count": r22["candidate_match_count"],
                "rsbuilding_2022_one_to_many": r22["one_to_many"],
                "rsbuilding_2022_intersection_area_m2": r22["intersection_area_m2"],
                "rsbuilding_2022_union_area_m2": r22["union_area_m2"],
                "rsbuilding_2022_footprint_iou": r22["footprint_iou"],
                "rsbuilding_2022_globfp_coverage_rate": r22["globfp_coverage_rate"],
                "rsbuilding_2022_rsbuilding_coverage_rate": r22["rsbuilding_coverage_rate"],
                "rsbuilding_2022_centroid_distance_m": r22["centroid_distance_m"],
                "rsbuilding_2022_area_ratio": r22["area_ratio"],
                "rsbuilding_2022_boundary_distance_m": r22["boundary_distance_m"],
                "sealed_tracking_geometry_state": temporal_state,
                "sealed_tracking_match_score": track.get("match_score"),
                "sealed_tracking_area_ratio": track.get("area_ratio"),
                **full20[gid],
                "globfp_supported_stable_building": stable,
                "globfp_temporal_conflict": conflict,
                "stability_rejection_reasons": ";".join(
                    reason
                    for reason, failed in (
                        ("missing_2019_rsbuilding", not r19["match_found"]),
                        ("missing_2022_rsbuilding", not r22["match_found"]),
                        ("sealed_tracking_not_stable", temporal_state != "stable_building"),
                        ("full20_change_not_low", not full20[gid]["full20_low_change"]),
                        ("aoi_edge_or_outside", not inside_aoi or edge_distance < EDGE_DISTANCE_M),
                        ("height_invalid", not bool(glo_row["height_valid"])),
                    )
                    if failed
                ),
            }
        )

    row_frame = gpd.GeoDataFrame(rows, geometry=glo_metric.geometry.values, crs=metric_crs)
    match19 = row_frame[[column for column in row_frame.columns if column in {"globfp_object_id", "globfp_height_m", "height_valid", "rsbuilding_2019_object_id", "rsbuilding_2019_match_found", "rsbuilding_2019_intersection_area_m2", "rsbuilding_2019_union_area_m2", "rsbuilding_2019_footprint_iou", "rsbuilding_2019_globfp_coverage_rate", "rsbuilding_2019_rsbuilding_coverage_rate", "rsbuilding_2019_centroid_distance_m", "rsbuilding_2019_area_ratio", "rsbuilding_2019_boundary_distance_m", "globfp_supported_stable_building", "globfp_temporal_conflict", "geometry"}]].copy()
    match22 = row_frame[[column for column in row_frame.columns if column in {"globfp_object_id", "globfp_height_m", "height_valid", "rsbuilding_2022_object_id", "rsbuilding_2022_match_found", "rsbuilding_2022_intersection_area_m2", "rsbuilding_2022_union_area_m2", "rsbuilding_2022_footprint_iou", "rsbuilding_2022_globfp_coverage_rate", "rsbuilding_2022_rsbuilding_coverage_rate", "rsbuilding_2022_centroid_distance_m", "rsbuilding_2022_area_ratio", "rsbuilding_2022_boundary_distance_m", "globfp_supported_stable_building", "globfp_temporal_conflict", "geometry"}]].copy()
    conflicts = row_frame[row_frame["globfp_temporal_conflict"]].copy()
    pyogrio.write_dataframe(match19, match_root / "GLOBFP_RSBUILDING_MATCH_2019.gpkg", layer="match_2019", driver="GPKG")
    pyogrio.write_dataframe(match22, match_root / "GLOBFP_RSBUILDING_MATCH_2022.gpkg", layer="match_2022", driver="GPKG")
    pyogrio.write_dataframe(conflicts, match_root / "GLOBFP_TEMPORAL_CONFLICTS.gpkg", layer="temporal_conflicts", driver="GPKG")
    write_csv(match_root / "GLOBFP_RSBUILDING_MATCH_TABLE.csv", rows)

    stable = [row for row in rows if row["globfp_supported_stable_building"]]
    matched_both = [row for row in rows if row["rsbuilding_2019_match_found"] and row["rsbuilding_2022_match_found"]]
    ious = [float(row["rsbuilding_2019_footprint_iou"]) for row in matched_both]
    ious.extend(float(row["rsbuilding_2022_footprint_iou"]) for row in matched_both)
    centroids = [float(row["rsbuilding_2019_centroid_distance_m"]) for row in matched_both if row["rsbuilding_2019_centroid_distance_m"] is not None]
    centroids.extend(float(row["rsbuilding_2022_centroid_distance_m"]) for row in matched_both if row["rsbuilding_2022_centroid_distance_m"] is not None)
    height_valid = sum(bool(row["height_valid"]) for row in rows)
    stable_count = len(stable)
    if stable_count >= 200 and height_valid / max(len(rows), 1) >= 0.90 and (sorted(ious)[len(ious) // 2] if ious else 0) >= 0.50 and (sorted(centroids)[len(centroids) // 2] if centroids else math.inf) <= 5:
        tier = "A"
    elif stable_count >= 100 and height_valid / max(len(rows), 1) >= 0.80 and (sorted(ious)[len(ious) // 2] if ious else 0) >= 0.30 and (sorted(centroids)[len(centroids) // 2] if centroids else math.inf) <= 10:
        tier = "B"
    else:
        tier = "C"
    audit = {
        "status": "PASS_MATCH_AUDIT_TIER_A_OR_B" if tier in {"A", "B"} else "TIER_C_NOT_ALLOWED_FOR_GLOBFP_PSEUDO_OR_NANJING_HEIGHT_ACCEPTANCE",
        "created_at_utc": now_utc(),
        "metric_crs": metric_crs,
        "source_extract": str(extract_path),
        "sealed_rsbuilding_2019": str(old_rs_root / "2019-03-16" / "building_objects.gpkg"),
        "sealed_rsbuilding_2022": str(old_rs_root / "2022-12-02" / "building_objects.gpkg"),
        "sealed_tracking_source": str(tracking_path),
        "sealed_full20_source": str(full20_path),
        "full20_low_change_threshold": FULL20_CHANGE_THRESHOLD,
        "edge_distance_threshold_m": EDGE_DISTANCE_M,
        "globfp_count": len(rows),
        "rsbuilding_2019_count": len(rs_frames["2019-03-16"]),
        "rsbuilding_2022_count": len(rs_frames["2022-12-02"]),
        "matched_both_count": len(matched_both),
        "matched_2019_count": sum(row["rsbuilding_2019_match_found"] for row in rows),
        "matched_2022_count": sum(row["rsbuilding_2022_match_found"] for row in rows),
        "unmatched_globfp_count": sum(not row["rsbuilding_2019_match_found"] or not row["rsbuilding_2022_match_found"] for row in rows),
        "unmatched_rsbuilding_2019_count": len(unmatched_by_date["2019-03-16"]),
        "unmatched_rsbuilding_2022_count": len(unmatched_by_date["2022-12-02"]),
        "one_to_many_2019_count": sum(row["rsbuilding_2019_one_to_many"] for row in rows),
        "one_to_many_2022_count": sum(row["rsbuilding_2022_one_to_many"] for row in rows),
        "stable_building_count": stable_count,
        "temporal_conflict_count": sum(row["globfp_temporal_conflict"] for row in rows),
        "height_valid_rate": height_valid / max(len(rows), 1),
        "median_iou_across_two_matches": sorted(ious)[len(ious) // 2] if ious else None,
        "median_centroid_distance_m_across_two_matches": sorted(centroids)[len(centroids) // 2] if centroids else None,
        "tier": tier,
        "pseudo_label_gate": "OPEN_TIER_A_OR_B" if tier in {"A", "B"} else "CLOSED_TIER_C",
        "nanjing_height_external_acceptance_gate": "OPEN_TIER_A_OR_B" if tier in {"A", "B"} else "CLOSED_TIER_C",
        "full20_rsbuilding_baseline_recomputed": False,
        "source_data_modified": False,
        "locked_test_pixel_reads": 0,
    }
    write_json(match_root / "GLOBFP_MATCH_AUDIT.json", audit)
    write_json(match_root / "GLOBFP_TEMPORAL_STABILITY_AUDIT.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--old-modular-root", type=Path, default=project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit = run_match(args.project_root, args.data_root, args.old_modular_root, args.output_root)
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
