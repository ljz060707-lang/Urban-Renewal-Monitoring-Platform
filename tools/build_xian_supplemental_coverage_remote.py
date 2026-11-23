"""Build residual-Unknown priority and supplemental-coverage evidence.

This script is intentionally diagnostic.  It does not write to the formal
candidate output and it never promotes supplemental observations to
Geometry, Presence, State, or Tracking.  It reads the frozen residual table
and the existing identity-component geometry, then stops when the available
supplemental pixels cannot support the actual main-date windows.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MAIN_DATES = {
    "2018": "2018-04-29",
    "2019": "2018-11-01",
    "2020": "2019-12-10",
}
SUPP_BBOX_WGS84 = (108.757095336914, 34.1613006591797, 109.173202514648, 34.4325256347656)


def csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def ffloat(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def csv_write(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def top_unknown_rows(residual: Path, fraction: float) -> tuple[int, float, list[dict[str, str]], Counter[str], float]:
    count = 0
    total_area = 0.0
    transition_counts: Counter[str] = Counter()
    for row in csv_rows(residual):
        count += 1
        total_area += ffloat(row.get("union_area_m2"))
        transition_counts[row.get("transition", "")] += 1
    top_n = max(1, math.ceil(count * fraction))
    heap: list[tuple[float, int, dict[str, str]]] = []
    for index, row in enumerate(csv_rows(residual)):
        area = ffloat(row.get("union_area_m2"))
        item = (area, index, row)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif area > heap[0][0]:
            heapq.heapreplace(heap, item)
    ordered = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
    threshold = ffloat(ordered[-1].get("union_area_m2")) if ordered else 0.0
    return count, total_area, ordered, transition_counts, threshold


def component_geometry_index(gpkg: Path, wanted_ids: set[str], batch_size: int = 5000) -> tuple[dict[str, Any], list[Any]]:
    import pyogrio

    info = pyogrio.read_info(gpkg, layer="identity_components")
    total = int(info["features"])
    coverage: dict[str, Any] = {}
    top_geometries: list[Any] = []
    from shapely.geometry import box
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    x0, y0 = transformer.transform(SUPP_BBOX_WGS84[0], SUPP_BBOX_WGS84[1])
    x1, y1 = transformer.transform(SUPP_BBOX_WGS84[2], SUPP_BBOX_WGS84[3])
    supplement_box = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    for offset in range(0, total, batch_size):
        frame = pyogrio.read_dataframe(
            gpkg,
            layer="identity_components",
            columns=["component_id"],
            read_geometry=True,
            skip_features=offset,
            max_features=batch_size,
        )
        for component_id, geometry in zip(frame["component_id"].tolist(), frame.geometry.tolist()):
            if not component_id or geometry is None or geometry.is_empty:
                continue
            intersects = bool(geometry.intersects(supplement_box))
            bounds = geometry.bounds
            coverage[str(component_id)] = {
                "supplement_bbox_relation": "INTERSECTS_DIRECT_TIF_BBOX" if intersects else "OUTSIDE_DIRECT_TIF_BBOX",
                "geometry_bbox_minx_m": bounds[0],
                "geometry_bbox_miny_m": bounds[1],
                "geometry_bbox_maxx_m": bounds[2],
                "geometry_bbox_maxy_m": bounds[3],
            }
            if str(component_id) in wanted_ids:
                top_geometries.append((str(component_id), geometry))
    return coverage, top_geometries


def reconstruct_missing_component_geometry(
    residual: Path,
    known_component_ids: set[str],
    top_ids: set[str],
    geometry_root: Path,
    batch_size: int = 5000,
) -> tuple[dict[str, Any], list[Any], dict[str, int]]:
    """Build diagnostic unions for component IDs absent from the fast GPKG.

    The residual table can contain final components that were not materialized
    in ``IDENTITY_COMPONENTS.gpkg``.  That is a lineage mismatch, not a
    spatial no-coverage result.  Use persisted ``object_uids`` and frozen
    main-date polygons only to recover an audit envelope.  No source geometry
    is modified and no result is promoted.
    """
    import pyogrio
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import unary_union

    missing_components: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in csv_rows(residual):
        cid = row.get("component_id", "")
        if not cid or cid in known_component_ids:
            continue
        for token in str(row.get("object_uids", "")).split(";"):
            token = token.strip()
            if "::" not in token:
                continue
            scene, raw_id = token.rsplit("::", 1)
            try:
                object_id = int(raw_id)
            except ValueError:
                continue
            missing_components[cid].add((scene, object_id))
    uid_to_components: dict[tuple[str, int], set[str]] = defaultdict(set)
    for cid, uids in missing_components.items():
        for uid in uids:
            uid_to_components[uid].add(cid)
    scene_paths = {
        "xa180429": geometry_root / "2018-04-29" / "building_objects.gpkg",
        "xa190101": geometry_root / "2019-01-01" / "building_objects.gpkg",
        "xa201231": geometry_root / "2020-12-31" / "building_objects.gpkg",
    }
    parts: dict[str, list[Any]] = defaultdict(list)
    matched_uids: set[tuple[str, int]] = set()
    for scene, gpkg in scene_paths.items():
        needed = {object_id for scene_name, object_id in uid_to_components if scene_name == scene}
        if not needed or not gpkg.is_file():
            continue
        total = int(pyogrio.read_info(gpkg, layer="buildings")["features"])
        for offset in range(0, total, batch_size):
            frame = pyogrio.read_dataframe(
                gpkg,
                layer="buildings",
                columns=["object_id"],
                read_geometry=True,
                skip_features=offset,
                max_features=batch_size,
            )
            for object_id, geometry in zip(frame["object_id"].tolist(), frame.geometry.tolist()):
                uid = (scene, int(object_id))
                if uid not in uid_to_components or geometry is None or geometry.is_empty:
                    continue
                matched_uids.add(uid)
                for cid in uid_to_components[uid]:
                    parts[cid].append(geometry)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    x0, y0 = transformer.transform(SUPP_BBOX_WGS84[0], SUPP_BBOX_WGS84[1])
    x1, y1 = transformer.transform(SUPP_BBOX_WGS84[2], SUPP_BBOX_WGS84[3])
    supplement_box = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    fallback_coverage: dict[str, Any] = {}
    fallback_top_geometries: list[Any] = []
    for cid, geometries in parts.items():
        if not geometries:
            continue
        geometry = unary_union(geometries)
        if geometry is None or geometry.is_empty:
            continue
        bounds = geometry.bounds
        fallback_coverage[cid] = {
            "supplement_bbox_relation": "INTERSECTS_DIRECT_TIF_BBOX" if geometry.intersects(supplement_box) else "OUTSIDE_DIRECT_TIF_BBOX",
            "geometry_bbox_minx_m": bounds[0],
            "geometry_bbox_miny_m": bounds[1],
            "geometry_bbox_maxx_m": bounds[2],
            "geometry_bbox_maxy_m": bounds[3],
            "geometry_source": "FROZEN_MAIN_BUILDING_OBJECTS_DIAGNOSTIC_UNION",
        }
        if cid in top_ids:
            fallback_top_geometries.append((cid, geometry))
    return fallback_coverage, fallback_top_geometries, {
        "missing_component_ids": len(missing_components),
        "matched_component_ids": len(fallback_coverage),
        "matched_object_uids": len(matched_uids),
        "unmatched_object_uids": len(set(uid_to_components) - matched_uids),
    }


def support_for_transition(transition: str) -> dict[str, str]:
    # The supplement audit found pixel collections near the 2018 main date
    # (2018-02-05 and 2018-08-18), but not within the preferred actual-date
    # windows around 2018-11-01 / 2019-12-10.  The 2019-11-21 and 2020-03-04
    # entries have only .bpw/cache metadata.
    if transition == "2018_2019":
        return {
            "supp_2018_support": "PIXEL_CANDIDATE_BBOX_UNVERIFIED",
            "supp_2019_support": "PIXEL_NEARBY_OUTSIDE_PREFERRED_WINDOW",
            "nearest_capture_date": "2018-02-05;2018-08-18",
            "days_from_target": "83;111;75",
            "coverage_quality": "PARTIAL_TILE_CENSUS_BBOX_UNVERIFIED;NO_TARGET_WINDOW_PIXEL",
            "closure_status": "UNRESOLVED_DUE_TO_NO_PIXEL_EVIDENCE",
        }
    return {
        "supp_2018_support": "NOT_APPLICABLE_TO_INTERVAL",
        "supp_2019_support": "NO_PIXEL_NEAR_TARGET_METADATA_ONLY_2019-11-21",
        "nearest_capture_date": "2019-11-21;2020-03-04;2020-07-30",
        "days_from_target": "20;85;233",
        "coverage_quality": "PARTIAL_TILE_CENSUS_BBOX_UNVERIFIED;NO_TARGET_WINDOW_PIXEL",
        "closure_status": "UNRESOLVED_DUE_TO_NO_PIXEL_EVIDENCE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--bbox", nargs=4, type=float, default=list(SUPP_BBOX_WGS84))
    parser.add_argument("--geometry-root", type=Path, default=Path("/data/xian_r4_100km2_20260914/output/geometry_run/geometry"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    count, total_area, top_rows, transition_counts, threshold = top_unknown_rows(args.residual, args.top_fraction)
    top_ids = {row.get("component_id", "") for row in top_rows if row.get("component_id")}
    coverage_index, top_geometries = component_geometry_index(args.components, top_ids)
    known_component_ids = set(coverage_index)
    fallback_coverage, fallback_top_geometries, fallback_stats = reconstruct_missing_component_geometry(
        args.residual, known_component_ids, top_ids, args.geometry_root
    )
    coverage_index.update(fallback_coverage)
    existing_top_ids = {component_id for component_id, _ in top_geometries}
    top_geometries.extend(
        (component_id, geometry)
        for component_id, geometry in fallback_top_geometries
        if component_id not in existing_top_ids
    )
    top_rank = {id(row): rank for rank, row in enumerate(top_rows, start=1)}
    top_component_rank: dict[str, int] = {}
    component_info: dict[str, dict[str, Any]] = {}
    for rank, row in enumerate(top_rows, start=1):
        cid = row.get("component_id", "")
        top_component_rank[cid] = min(rank, top_component_rank.get(cid, rank))
        item = component_info.setdefault(cid, {"unknown_area_m2": 0.0, "priority_rank": rank, "transitions": set()})
        item["unknown_area_m2"] = max(item["unknown_area_m2"], ffloat(row.get("union_area_m2")))
        item["priority_rank"] = min(item["priority_rank"], rank)
        item["transitions"].add(row.get("transition", ""))

    # Compact top-20% table.
    top_out: list[dict[str, Any]] = []
    for rank, row in enumerate(top_rows, start=1):
        cid = row.get("component_id", "")
        support = support_for_transition(row.get("transition", ""))
        spatial = coverage_index.get(cid, {})
        top_out.append(
            {
                "priority_rank": rank,
                "priority_band": "TOP_20PCT_UNKNOWN_AREA",
                "component_id": cid,
                "transition": row.get("transition", ""),
                "building_id": row.get("building_id", ""),
                "track_id": row.get("track_id", ""),
                "presence_sequence": row.get("presence_sequence", ""),
                "union_area_m2": row.get("union_area_m2", ""),
                "primary_reason_before": row.get("primary_reason_before", ""),
                "secondary_reasons_before": row.get("secondary_reasons_before", ""),
                "residual_reason": row.get("residual_reason", ""),
                "structural_types": row.get("structural_types", ""),
                "candidate_relations": row.get("candidate_relations", ""),
                "identity_confidence": row.get("identity_confidence", ""),
                "max_alignment_aware_displacement_m": row.get("max_alignment_aware_displacement_m", ""),
                "min_neighborhood_consistency": row.get("min_neighborhood_consistency", ""),
                "component_geometry_found": bool(spatial),
                "geometry_source": spatial.get("geometry_source", "IDENTITY_COMPONENTS_GPKG" if spatial else "NOT_FOUND"),
                **spatial,
                **support,
            }
        )
    csv_write(out / "TOP_UNKNOWN_COMPONENTS.csv", top_out)

    # Join every residual row to component geometry coverage and supplemental
    # support status.  No Presence/State value is changed by this join.
    coverage_fields = [
        "transition", "component_id", "building_id", "track_id", "presence_sequence", "union_area_m2",
        "priority_band", "supp_2018_support", "supp_2019_support", "supp_2020_support",
        "nearest_capture_date", "days_from_target", "coverage_quality", "supplement_bbox_relation",
        "geometry_bbox_minx_m", "geometry_bbox_miny_m", "geometry_bbox_maxx_m", "geometry_bbox_maxy_m",
        "geometry_source", "closure_status", "geometry_owner", "formal_result_modified",
    ]
    coverage_out_path = out / "UNKNOWN_SUPPLEMENTAL_COVERAGE.csv"
    with args.residual.open("r", encoding="utf-8-sig", newline="") as source, coverage_out_path.open("w", encoding="utf-8-sig", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=coverage_fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            cid = row.get("component_id", "")
            support = support_for_transition(row.get("transition", ""))
            spatial = coverage_index.get(cid, {})
            writer.writerow(
                {
                    "transition": row.get("transition", ""),
                    "component_id": cid,
                    "building_id": row.get("building_id", ""),
                    "track_id": row.get("track_id", ""),
                    "presence_sequence": row.get("presence_sequence", ""),
                    "union_area_m2": row.get("union_area_m2", ""),
                    "priority_band": "TOP_20PCT_UNKNOWN_AREA" if ffloat(row.get("union_area_m2")) >= threshold else "LOWER_80PCT_UNKNOWN_AREA",
                    "supp_2018_support": support["supp_2018_support"],
                    "supp_2019_support": support["supp_2019_support"],
                    "supp_2020_support": "PIXEL_CANDIDATE_BBOX_UNVERIFIED" if row.get("transition") == "2019_2020" else "NOT_APPLICABLE_TO_INTERVAL",
                    "nearest_capture_date": support["nearest_capture_date"],
                    "days_from_target": support["days_from_target"],
                    "coverage_quality": support["coverage_quality"],
                    "supplement_bbox_relation": spatial.get("supplement_bbox_relation", "COMPONENT_GEOMETRY_NOT_FOUND"),
                    "geometry_bbox_minx_m": spatial.get("geometry_bbox_minx_m", ""),
                    "geometry_bbox_miny_m": spatial.get("geometry_bbox_miny_m", ""),
                    "geometry_bbox_maxx_m": spatial.get("geometry_bbox_maxx_m", ""),
                    "geometry_bbox_maxy_m": spatial.get("geometry_bbox_maxy_m", ""),
                    "geometry_source": spatial.get("geometry_source", "IDENTITY_COMPONENTS_GPKG" if spatial else "NOT_FOUND"),
                    "closure_status": support["closure_status"],
                    "geometry_owner": "MAIN_B0_CANONICAL_ONLY",
                    "formal_result_modified": False,
                }
            )

    # Build a 1 km spatial cluster layer for the highest-priority components.
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union

    top_records: list[dict[str, Any]] = []
    for cid, geometry in top_geometries:
        info = component_info.get(cid, {})
        if not info:
            continue
        centroid = geometry.centroid
        cell_x = math.floor(float(centroid.x) / 1000.0)
        cell_y = math.floor(float(centroid.y) / 1000.0)
        top_records.append(
            {
                "component_id": cid,
                "priority_rank": info.get("priority_rank", ""),
                "unknown_area_m2": info.get("unknown_area_m2", 0.0),
                "transition_set": ";".join(sorted(info.get("transitions", set()))),
                "cluster_id": f"UCL_{cell_x}_{cell_y}",
                "geometry": geometry,
            }
        )
    component_gdf = gpd.GeoDataFrame(top_records, geometry="geometry", crs="EPSG:32649")
    cluster_records: list[dict[str, Any]] = []
    for cluster_id, group in component_gdf.groupby("cluster_id"):
        geometry = unary_union(group.geometry.tolist())
        transitions = sorted({value for cell in group["transition_set"].tolist() for value in str(cell).split(";") if value})
        bounds = geometry.bounds
        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "component_count": int(len(group)),
                "unknown_area_m2": float(group["unknown_area_m2"].sum()),
                "priority_rank_min": int(group["priority_rank"].min()),
                "transitions": ";".join(transitions),
                "bbox_minx_m": bounds[0],
                "bbox_miny_m": bounds[1],
                "bbox_maxx_m": bounds[2],
                "bbox_maxy_m": bounds[3],
                "geometry": geometry,
            }
        )
    clusters_gdf = gpd.GeoDataFrame(cluster_records, geometry="geometry", crs="EPSG:32649")
    cluster_path = out / "TOP_UNKNOWN_CLUSTERS.gpkg"
    if cluster_path.exists():
        cluster_path.unlink()
    clusters_gdf.to_file(cluster_path, layer="top_unknown_clusters", driver="GPKG", index=False)

    # Missing-pixel request: the actual 2019 main evidence window is around
    # 2018-11-01, and the dominant 2020 main evidence is around 2019-12-10.
    request_rows: list[dict[str, Any]] = []
    for row in cluster_records:
        request_rows.append(
            {
                "cluster_id": row["cluster_id"],
                "bbox_minx_m": row["bbox_minx_m"],
                "bbox_miny_m": row["bbox_miny_m"],
                "bbox_maxx_m": row["bbox_maxx_m"],
                "bbox_maxy_m": row["bbox_maxy_m"],
                "target_date": "2018-11-01;2019-12-10",
                "preferred_date_window": "2018-09-01..2019-01-31;2019-09-01..2020-03-31",
                "unknown_area_m2": row["unknown_area_m2"],
                "priority": f"P{row['priority_rank_min']}",
                "reason": "NO_TARGET_WINDOW_PIXEL_PAYLOAD_OR_BBOX_VERIFIED_SUPPLEMENT_COVERAGE",
                "request_status": "UNRESOLVED_DUE_TO_NO_PIXEL_EVIDENCE",
            }
        )
    csv_write(out / "SUPPLEMENTAL_DATA_REQUEST.csv", request_rows)
    request_gdf = gpd.GeoDataFrame(request_rows, geometry=[record["geometry"] for record in cluster_records], crs="EPSG:32649")
    request_path = out / "SUPPLEMENTAL_DATA_REQUEST.gpkg"
    if request_path.exists():
        request_path.unlink()
    request_gdf.to_file(request_path, layer="supplemental_data_request", driver="GPKG", index=False)

    alignment_rows = [
        {
            "supplement_observation": "2017-07-14;2017-09-12;2021-01-29;2021-06-30",
            "pixel_payload": "DIRECT_GEOTIFF",
            "alignment_status": "DEFERRED_SUPPORT_ONLY",
            "translation_or_local_displacement": "NOT_RUN_IN_THIS_EVIDENCE_GATE",
            "alignment_confidence": "NOT_RUN",
            "stable_ground_residual": "NOT_RUN",
            "parallax_status": "NOT_RUN",
            "reason": "No target-date residual-window comparison was promoted before pixel coverage gate.",
            "formal_geometry_modified": False,
        },
        {
            "supplement_observation": "2018-02-05;2018-08-18;2020-07-30;2021-01-29;2021-06-30",
            "pixel_payload": "NESTED_CACHE_PRIOR_CENSUS",
            "alignment_status": "DEFERRED_BBOX_UNVERIFIED",
            "translation_or_local_displacement": "NOT_RUN",
            "alignment_confidence": "NOT_RUN",
            "stable_ground_residual": "NOT_RUN",
            "parallax_status": "NOT_RUN",
            "reason": "Per-tile CRS/transform/bbox was not verified in bounded audit.",
            "formal_geometry_modified": False,
        },
    ]
    csv_write(out / "SUPPLEMENTAL_ALIGNMENT_PARALLAX_DIAGNOSTIC.csv", alignment_rows)
    (out / "SUPPLEMENTAL_VISUAL_QA_STATUS.json").write_text(
        json.dumps({
            "status": "NOT_STARTED_BLOCKED_BY_TARGET_PIXEL_COVERAGE_GATE",
            "required_queues": {
                "largest_unknown": 100,
                "identity_ambiguous": 100,
                "suspected_absence": 100,
                "suspected_false_positive": 100,
            },
            "completed_this_round": 0,
            "reason": "No target-window supplemental pixel payload with verified coverage; no candidate Presence/Identity/State promotion permitted.",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    status = {
        "schema_version": "urrs.xian.supplemental.integration.run_status.v1",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(microsecond=0).isoformat(),
        "task": "XIAN-SUPPLEMENTAL-PIXEL-EVIDENCE-INTEGRATION-001",
        "supplement_pixels_found": True,
        "supplement_pixel_file_count": 1171066,
        "supplement_coverage_km2": "DIRECT_TIF_BBOX_AVAILABLE;NESTED_CACHE_BBOX_UNVERIFIED",
        "unknown_supported_area": 0.0,
        "unknown_area_pct_18_19": 26.578809129101586,
        "unknown_area_pct_19_20": 28.617362723636123,
        "resolved_by_presence": 0.0,
        "resolved_by_identity": 0.0,
        "nonbuilding_candidates": 0,
        "visual_wrong_rate": "NOT_RUN_THIS_ROUND",
        "current_iteration": 0,
        "current_blocker": "UNRESOLVED_DUE_TO_NO_PIXEL_EVIDENCE;2019_AND_2020_TARGET_WINDOWS_LACK_VERIFIED_SUPPLEMENTAL_PIXEL_COVERAGE",
        "tracking_tuning_started": False,
        "formal_geometry_modified": False,
        "formal_tracking_modified": False,
        "formal_presence_modified": False,
        "formal_state_modified": False,
        "safe_to_start_presence_state": False,
        "stop_rule": "B_INSUFFICIENT_SUPPLEMENTAL_TARGET_DATE_PIXEL_EVIDENCE",
        "residual_row_count": count,
        "residual_union_area_m2_sum": total_area,
        "top_fraction": args.top_fraction,
        "top_row_count": len(top_rows),
        "top_area_threshold_m2": threshold,
        "transition_counts": dict(transition_counts),
        "source_residual": str(args.residual),
        "source_identity_components": str(args.components),
        "fallback_geometry_source": str(args.geometry_root),
        "fallback_geometry_stats": fallback_stats,
        "indexed_component_geometry_count": len(coverage_index),
        "top_geometry_count": len(top_geometries),
        "outputs": [
            "SUPPLEMENTAL_PIXEL_INVENTORY.csv (local audit)",
            "SUPPLEMENTAL_ACTUAL_DATE_REGISTRY.csv (local audit)",
            "TOP_UNKNOWN_COMPONENTS.csv",
            "TOP_UNKNOWN_CLUSTERS.gpkg",
            "UNKNOWN_SUPPLEMENTAL_COVERAGE.csv",
            "SUPPLEMENTAL_DATA_REQUEST.csv",
            "SUPPLEMENTAL_DATA_REQUEST.gpkg",
            "SUPPLEMENTAL_ALIGNMENT_PARALLAX_DIAGNOSTIC.csv",
            "SUPPLEMENTAL_VISUAL_QA_STATUS.json",
        ],
    }
    (out / "RUN_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "RUN_PROGRESS.md").write_text(
        "# Xi'an supplemental pixel evidence integration\n\n"
        "## Current gate\n\n"
        "`SUPPLEMENTAL_PIXEL_AUDIT = PASS_BOUNDED`\n\n"
        "`TRACKING_TUNING = STOP`\n\n"
        "`PRESENCE_STATE_PROMOTION = STOP`\n\n"
        "原因：辅助目录中存在 2018/2020 附近的像元集合，但 2019-11-21、2020-03-04、2020-09-08 没有可用像元；嵌套 cache 的本轮逐 tile bbox 也未完成验证。另有部分 residual component 不在 fast-closure GPKG 中，已用冻结主影像对象做诊断性 envelope 回补，但这不改变 IdentityGraph。当前不能把“未检测到”转成可靠 0，不能据此封存 IdentityGraph 或正式减少 Unknown。\n\n"
        "## Required next evidence\n\n"
        "请补充 `2018-11-01` 和 `2019-12-10` 实际日期窗口附近、覆盖请求清单中的 bbox 的可读取像元影像。`.dat/.raw/.datever/.ver/.bpw` 不足以作为像元证据。\n\n"
        "## Formal result protection\n\n"
        "本轮没有修改 Geometry、Tracking、Presence、State、正式 TIF、面积统计或固定案例。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "STOP_INSUFFICIENT_TARGET_DATE_PIXEL_EVIDENCE",
        "out": str(out),
        "residual_rows": count,
        "top_rows": len(top_rows),
        "coverage_components_indexed": len(coverage_index),
        "top_clusters": len(cluster_records),
        "request_rows": len(request_rows),
        "formal_results_modified": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
