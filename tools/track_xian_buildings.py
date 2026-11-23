#!/usr/bin/env python3
"""Auditable same-date canonicalization and cross-date identity tracking.

This tool consumes B0 candidate building polygons after area post-processing.
It deliberately keeps the source geometry and source object id, marks rather
than deletes same-date duplicates, and treats temporal labels as candidates.
It never fuses a missing B1 verifier into B0 and never edits polygon geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.strtree import STRtree


METRIC_CRS = "EPSG:32649"
DEFAULT_DEDUP_OVERLAP = 0.85
DEFAULT_DEDUP_DISTANCE = 2.0
BUFFER_M = 8.0
MAX_DISTANCE_M = 24.0
MAX_AREA_RATIO = 3.0
MAX_CANDIDATES = 16
MIN_POLYGON_IOU = 0.02
AMBIGUITY_MARGIN = 0.05
SCORE_WEIGHTS = {
    "polygon_iou": 0.30,
    "buffered_iou": 0.30,
    "bbox_iou": 0.10,
    "distance_score": 0.20,
    "shape_similarity": 0.10,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def csv_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def calc_geometry(geometry):
    """Return a calculation-only valid geometry; output geometry is untouched."""

    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.is_valid:
        return geometry
    try:
        from shapely import make_valid

        repaired = make_valid(geometry)
    except Exception:
        repaired = geometry.buffer(0)
    return repaired if repaired is not None and not repaired.is_empty else geometry


def safe_area(geometry) -> float:
    try:
        return max(0.0, finite(geometry.area))
    except Exception:
        return 0.0


def safe_iou(left, right) -> float:
    try:
        intersection = left.intersection(right).area
        union = left.union(right).area
        return float(intersection / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def bbox_iou(left, right) -> float:
    try:
        lx0, ly0, lx1, ly1 = left.bounds
        rx0, ry0, rx1, ry1 = right.bounds
        ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
        ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        la = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
        ra = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
        union = la + ra - intersection
        return float(intersection / union) if union > 0 else 0.0
    except Exception:
        return 0.0


def area_ratio(left_area: float, right_area: float) -> float:
    low, high = sorted((max(1e-9, left_area), max(1e-9, right_area)))
    return float(high / low)


def shape_similarity(left, right, left_area: float, right_area: float) -> float:
    try:
        left_perimeter = max(1e-9, float(left.length))
        right_perimeter = max(1e-9, float(right.length))
        perimeter_ratio = min(left_perimeter, right_perimeter) / max(left_perimeter, right_perimeter)
    except Exception:
        perimeter_ratio = 0.0
    return float(min(left_area, right_area) / max(1e-9, max(left_area, right_area)) * perimeter_ratio)


def union_find_groups(links: list[tuple[int, int]], count: int) -> list[list[int]]:
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in links:
        union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        groups[find(index)].append(index)
    return list(groups.values())


def canonicalize(gdf: gpd.GeoDataFrame, block: str, date: str) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Mark same-date duplicates and return metric observations for kept rows."""

    if gdf.crs is None:
        raise ValueError(f"{block} {date}: source geometry has no CRS")
    source = gdf.copy().reset_index(drop=True)
    source_crs = str(source.crs)
    metric = source.to_crs(METRIC_CRS)
    calc = [calc_geometry(value) for value in metric.geometry]
    areas = [safe_area(value) for value in calc]
    tree = STRtree(calc) if calc else None
    links: list[tuple[int, int]] = []
    duplicate_edges: list[dict[str, Any]] = []
    for index, geometry in enumerate(calc):
        if tree is None or geometry is None or geometry.is_empty:
            continue
        for other in tree.query(geometry.buffer(DEFAULT_DEDUP_DISTANCE)):
            other_index = int(other)
            if other_index <= index:
                continue
            other_geometry = calc[other_index]
            if other_geometry is None or other_geometry.is_empty:
                continue
            overlap = safe_iou(geometry, other_geometry)
            minimum_area = max(1e-9, min(areas[index], areas[other_index]))
            containment = float(geometry.intersection(other_geometry).area / minimum_area)
            distance = float(geometry.centroid.distance(other_geometry.centroid))
            if containment >= DEFAULT_DEDUP_OVERLAP and distance <= DEFAULT_DEDUP_DISTANCE:
                links.append((index, other_index))
                duplicate_edges.append({
                    "left_index": index,
                    "right_index": other_index,
                    "polygon_iou": overlap,
                    "containment_ratio": containment,
                    "centroid_distance_m": distance,
                })

    groups = union_find_groups(links, len(source))
    group_by_index: dict[int, list[int]] = {}
    for group in groups:
        for index in group:
            group_by_index[index] = group

    canonical_ids: list[str] = []
    duplicate_status: list[str] = []
    duplicate_group_ids: list[str] = []
    canonical_keep: list[int] = []
    source_object_ids = [str(value) for value in source.get("object_id", pd.Series(range(len(source))))]
    for index, source_id in enumerate(source_object_ids):
        group = group_by_index[index]
        keeper = sorted(group, key=lambda value: (-areas[value], source_object_ids[value]))[0]
        group_id = f"{block}_{date}_D{min(group) + 1:06d}" if len(group) > 1 else ""
        canonical_ids.append(f"{block}_{date}_C{keeper + 1:06d}")
        duplicate_group_ids.append(group_id)
        canonical_keep.append(int(index == keeper))
        duplicate_status.append("DUPLICATE_NONCANONICAL" if index != keeper else "CANONICAL_DUPLICATE_GROUP" if len(group) > 1 else "UNIQUE")

    source["source_object_id"] = source_object_ids
    source["canonical_id"] = canonical_ids
    source["duplicate_group_id"] = duplicate_group_ids
    source["duplicate_status"] = duplicate_status
    source["canonical_keep"] = canonical_keep
    source["geometry_source_crs"] = source_crs
    source["geometry_modified"] = 0
    source["canonicalization_rule"] = "mark_only_overlap_ge_0.85_and_centroid_distance_le_2m"

    observations: list[dict[str, Any]] = []
    for index, row in source.iterrows():
        if not canonical_keep[index]:
            continue
        observations.append({
            "block": block,
            "date": date,
            "row_index": int(index),
            "source_object_id": source_object_ids[index],
            "canonical_id": canonical_ids[index],
            "geometry": metric.geometry.iloc[index],
            "calc_geometry": calc[index],
            "area_m2": areas[index],
            "source_area_m2": finite(row.get("area_m2"), areas[index]),
            "source_geometry_wkb_sha256": hashlib.sha256(row.geometry.wkb).hexdigest() if row.geometry is not None else "",
            "duplicate_status": duplicate_status[index],
            "duplicate_group_id": duplicate_group_ids[index],
        })
    audit = {
        "date": date,
        "source_crs": source_crs,
        "metric_crs": METRIC_CRS,
        "source_count": len(source),
        "canonical_count": len(observations),
        "duplicate_group_count": sum(len(group) > 1 for group in groups),
        "duplicate_noncanonical_count": sum(status == "DUPLICATE_NONCANONICAL" for status in duplicate_status),
        "geometry_modified": False,
        "rule": "mark_only_overlap_ge_0.85_and_centroid_distance_le_2m",
        "duplicate_edges": duplicate_edges,
    }
    return source, observations, audit


def candidate_edges(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    previous_by_index = {index: item for index, item in enumerate(previous)}
    current_candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    previous_candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not previous or not current:
        return current_candidates, previous_candidates
    tree = STRtree([item["calc_geometry"] for item in previous])
    for current_index, current_item in enumerate(current):
        current_geometry = current_item["calc_geometry"]
        if current_geometry is None or current_geometry.is_empty:
            continue
        possible = []
        for previous_index in tree.query(current_geometry.buffer(MAX_DISTANCE_M)):
            previous_index = int(previous_index)
            previous_item = previous_by_index[previous_index]
            previous_geometry = previous_item["calc_geometry"]
            distance = float(previous_geometry.centroid.distance(current_geometry.centroid))
            possible.append((distance, previous_item["canonical_id"], previous_index))
        possible.sort(key=lambda value: (value[0], value[1]))
        for distance, _, previous_index in possible[:MAX_CANDIDATES]:
            previous_item = previous_by_index[previous_index]
            previous_geometry = previous_item["calc_geometry"]
            left_area, right_area = previous_item["area_m2"], current_item["area_m2"]
            ratio = area_ratio(left_area, right_area)
            polygon_overlap = safe_iou(previous_geometry, current_geometry)
            buffered_overlap = safe_iou(previous_geometry.buffer(BUFFER_M), current_geometry.buffer(BUFFER_M))
            box_overlap = bbox_iou(previous_geometry, current_geometry)
            distance_score = max(0.0, 1.0 - distance / MAX_DISTANCE_M)
            shape = shape_similarity(previous_geometry, current_geometry, left_area, right_area)
            score = (
                SCORE_WEIGHTS["polygon_iou"] * polygon_overlap
                + SCORE_WEIGHTS["buffered_iou"] * buffered_overlap
                + SCORE_WEIGHTS["bbox_iou"] * box_overlap
                + SCORE_WEIGHTS["distance_score"] * distance_score
                + SCORE_WEIGHTS["shape_similarity"] * shape
            )
            edge = {
                "previous_index": previous_index,
                "current_index": current_index,
                "previous_id": previous_item["canonical_id"],
                "current_id": current_item["canonical_id"],
                "distance_m": distance,
                "area_ratio": ratio,
                "polygon_iou": polygon_overlap,
                "buffered_iou": buffered_overlap,
                "bbox_iou": box_overlap,
                "distance_score": distance_score,
                "shape_similarity": shape,
                "score": score,
                "area_gate": ratio <= MAX_AREA_RATIO,
                "distance_gate": distance <= MAX_DISTANCE_M,
                "overlap_gate": polygon_overlap >= MIN_POLYGON_IOU,
            }
            current_candidates[current_index].append(edge)
            previous_candidates[previous_index].append(edge)
    for values in current_candidates.values():
        values.sort(key=lambda edge: (-edge["score"], edge["distance_m"], edge["previous_id"]))
    return current_candidates, previous_candidates


def match_pair(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    current_candidates, previous_candidates = candidate_edges(previous, current)
    eligible_by_current: dict[int, list[dict[str, Any]]] = {}
    ambiguous_current: set[int] = set()
    for current_index, edges in current_candidates.items():
        eligible = [edge for edge in edges if edge["area_gate"] and edge["distance_gate"] and edge["overlap_gate"]]
        eligible_by_current[current_index] = eligible
        if len(eligible) >= 2 and (eligible[0]["score"] - eligible[1]["score"]) < AMBIGUITY_MARGIN:
            ambiguous_current.add(current_index)

    ranked = []
    for current_index, edges in eligible_by_current.items():
        if current_index in ambiguous_current:
            continue
        ranked.extend(edges)
    ranked.sort(key=lambda edge: (-edge["score"], edge["distance_m"], edge["previous_id"], edge["current_id"]))
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matched: dict[int, dict[str, Any]] = {}
    for edge in ranked:
        if edge["previous_index"] in used_previous or edge["current_index"] in used_current:
            continue
        used_previous.add(edge["previous_index"])
        used_current.add(edge["current_index"])
        matched[edge["current_index"]] = edge

    current_status: dict[int, str] = {}
    for index in range(len(current)):
        if index in matched:
            current_status[index] = "MATCHED_ONE_TO_ONE"
        elif index in ambiguous_current:
            current_status[index] = "UNKNOWN_CONFLICT_AMBIGUOUS"
        elif current_candidates.get(index):
            current_status[index] = "UNKNOWN_CONFLICT_COMPETING_OR_GATE"
        else:
            current_status[index] = "NEW_BUILDING_CANDIDATE"

    previous_status: dict[int, str] = {}
    for index in range(len(previous)):
        if index in used_previous:
            previous_status[index] = "MATCHED_ONE_TO_ONE"
        elif previous_candidates.get(index):
            previous_status[index] = "UNKNOWN_CONFLICT_UNMATCHED_NEIGHBOR"
        else:
            previous_status[index] = "DEMOLITION_CANDIDATE"
    return {
        "matched": matched,
        "current_status": current_status,
        "previous_status": previous_status,
        "current_candidates": current_candidates,
        "previous_candidates": previous_candidates,
    }


def write_gpkg(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio")


def observations_gdf(observations: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    fields = []
    for item in observations:
        row = {key: value for key, value in item.items() if key not in {"calc_geometry"}}
        fields.append(row)
    if not fields:
        return gpd.GeoDataFrame({"track_id": []}, geometry=[], crs=METRIC_CRS)
    return gpd.GeoDataFrame(fields, geometry="geometry", crs=METRIC_CRS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area-root", required=True, type=Path)
    parser.add_argument("--tracking-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--dates", nargs=3, required=True)
    parser.add_argument("--blocks", nargs="+", default=["B01", "B02", "B03", "B04", "B05"])
    args = parser.parse_args()

    args.tracking_root.mkdir(parents=True, exist_ok=True)
    args.state_root.mkdir(parents=True, exist_ok=True)
    b1_status = {
        "status": "UNAVAILABLE_NOT_SUBSTITUTED",
        "reason": "No independent B1 verifier checkpoint or implementation was supplied in the cloud canary artifacts.",
        "fusion_performed": False,
        "b0_primary": True,
        "scientific_consequence": "B0 outputs remain candidate_not_ground_truth; no verifier accuracy or fused mask is claimed.",
    }
    json_dump(args.tracking_root / "B1_VERIFIER_STATUS.json", b1_status)
    json_dump(args.state_root / "B1_VERIFIER_STATUS.json", b1_status)

    global_rows: list[dict[str, Any]] = []
    all_receipts: list[dict[str, Any]] = []
    for block in args.blocks:
        block_tracking = args.tracking_root / block
        block_state = args.state_root / block
        block_tracking.mkdir(parents=True, exist_ok=True)
        block_state.mkdir(parents=True, exist_ok=True)
        by_date: dict[str, list[dict[str, Any]]] = {}
        dedup_audits: list[dict[str, Any]] = []
        input_receipts: list[dict[str, Any]] = []
        for date in args.dates:
            input_path = args.area_root / date / block / "建筑对象_unfiltered.gpkg"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            gdf = gpd.read_file(input_path)
            source_gdf, observations, audit = canonicalize(gdf, block, date)
            by_date[date] = observations
            dedup_audits.append(audit)
            input_receipts.append({"date": date, "path": str(input_path), "sha256": sha256(input_path), "bytes": input_path.stat().st_size, "rows": len(gdf)})
            canonical_path = block_tracking / f"同日去重_{date}.gpkg"
            write_gpkg(source_gdf, canonical_path, "canonical_objects")
            csv_dump(block_tracking / f"同日去重_{date}.csv", [
                {key: value for key, value in row.items() if key != "geometry"}
                for row in source_gdf.drop(columns="geometry").to_dict(orient="records")
            ])

        states: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        previous_date = args.dates[0]
        next_track_number = 1
        for item in by_date[previous_date]:
            item["track_id"] = f"{block}-T{next_track_number:06d}"
            next_track_number += 1
            item["match_status"] = "BASELINE_NO_PRIOR_DATE"
            item["match_score"] = None
            item["match_distance_m"] = None
            item["match_previous_id"] = ""
            item["candidate_count"] = 0
            item["state"] = "BASELINE_OBSERVATION"
            states.append({
                "block": block,
                "track_id": item["track_id"],
                "state": "BASELINE_OBSERVATION",
                "observation_date": previous_date,
                "event_date": "",
                "object_id": item["canonical_id"],
                "geometry": item["geometry"],
                "reason": "first_actual_date_in_dev_canary",
            })

        for date in args.dates[1:]:
            previous = by_date[previous_date]
            current = by_date[date]
            result = match_pair(previous, current)
            for current_index, item in enumerate(current):
                edge = result["matched"].get(current_index)
                status = result["current_status"][current_index]
                if edge is not None:
                    previous_item = previous[edge["previous_index"]]
                    item["track_id"] = previous_item["track_id"]
                    item["match_status"] = status
                    item["match_score"] = edge["score"]
                    item["match_distance_m"] = edge["distance_m"]
                    item["match_previous_id"] = edge["previous_id"]
                    item["candidate_count"] = len(result["current_candidates"].get(current_index, []))
                    state = "STABLE_BUILDING_CANDIDATE"
                    transition_reason = "one_to_one_identity_match"
                else:
                    item["track_id"] = f"{block}-T{next_track_number:06d}"
                    next_track_number += 1
                    item["match_status"] = status
                    item["match_score"] = None
                    item["match_distance_m"] = None
                    item["match_previous_id"] = ""
                    item["candidate_count"] = len(result["current_candidates"].get(current_index, []))
                    state = "NEW_BUILDING_CANDIDATE" if status == "NEW_BUILDING_CANDIDATE" else "UNKNOWN_CONFLICT"
                    transition_reason = status
                item["state"] = state
                states.append({
                    "block": block,
                    "track_id": item["track_id"],
                    "state": state,
                    "observation_date": date,
                    "event_date": date,
                    "object_id": item["canonical_id"],
                    "geometry": item["geometry"],
                    "reason": transition_reason,
                })
                if edge is not None:
                    transitions.append({
                        "block": block,
                        "from_date": previous_date,
                        "to_date": date,
                        "from_object_id": edge["previous_id"],
                        "to_object_id": edge["current_id"],
                        "track_id": item["track_id"],
                        "transition_state": "STABLE_BUILDING_CANDIDATE",
                        "reason": "one_to_one_identity_match",
                        "score": edge["score"],
                        "distance_m": edge["distance_m"],
                        "area_ratio": edge["area_ratio"],
                        "polygon_iou": edge["polygon_iou"],
                        "buffered_iou": edge["buffered_iou"],
                        "bbox_iou": edge["bbox_iou"],
                        "geometry": item["geometry"],
                    })
                else:
                    transitions.append({
                        "block": block,
                        "from_date": previous_date,
                        "to_date": date,
                        "from_object_id": "",
                        "to_object_id": item["canonical_id"],
                        "track_id": item["track_id"],
                        "transition_state": state,
                        "reason": transition_reason,
                        "score": None,
                        "distance_m": None,
                        "area_ratio": None,
                        "polygon_iou": None,
                        "buffered_iou": None,
                        "bbox_iou": None,
                        "geometry": item["geometry"],
                    })
            for previous_index, previous_item in enumerate(previous):
                if result["previous_status"][previous_index] == "MATCHED_ONE_TO_ONE":
                    continue
                previous_state = "DEMOLITION_CANDIDATE" if result["previous_status"][previous_index] == "DEMOLITION_CANDIDATE" else "UNKNOWN_CONFLICT"
                states.append({
                    "block": block,
                    "track_id": previous_item["track_id"],
                    "state": previous_state,
                    "observation_date": previous_date,
                    "event_date": date,
                    "object_id": previous_item["canonical_id"],
                    "geometry": previous_item["geometry"],
                    "reason": result["previous_status"][previous_index],
                })
                transitions.append({
                    "block": block,
                    "from_date": previous_date,
                    "to_date": date,
                    "from_object_id": previous_item["canonical_id"],
                    "to_object_id": "",
                    "track_id": previous_item["track_id"],
                    "transition_state": previous_state,
                    "reason": result["previous_status"][previous_index],
                    "score": None,
                    "distance_m": None,
                    "area_ratio": None,
                    "polygon_iou": None,
                    "buffered_iou": None,
                    "bbox_iou": None,
                    "geometry": previous_item["geometry"],
                })
            previous_date = date

        track_gdf = observations_gdf([item for date in args.dates for item in by_date[date]])
        transition_gdf = gpd.GeoDataFrame(transitions, geometry="geometry", crs=METRIC_CRS)
        state_gdf = gpd.GeoDataFrame(states, geometry="geometry", crs=METRIC_CRS)
        write_gpkg(track_gdf, block_tracking / "建筑编号追踪.gpkg", "tracking_objects")
        write_gpkg(transition_gdf, block_tracking / "跨期转移事件.gpkg", "transition_events")
        write_gpkg(state_gdf, block_state / "建筑三态.gpkg", "building_states")
        csv_dump(block_tracking / "跨期转移事件.csv", [{key: value for key, value in row.items() if key != "geometry"} for row in transitions])
        csv_dump(block_state / "建筑三态.csv", [{key: value for key, value in row.items() if key != "geometry"} for row in states])

        state_counts = Counter(row["state"] for row in states)
        transition_counts = Counter(row["transition_state"] for row in transitions)
        receipt = {
            "status": "TRACKING_COMPLETE_CANDIDATE_ONLY",
            "created_at": utc_now(),
            "block": block,
            "dates": list(args.dates),
            "input_threshold": "unfiltered",
            "metric_crs": METRIC_CRS,
            "geometry_modified": False,
            "same_date_dedup": {
                "mode": "mark_only_preserve_all_source_objects",
                "overlap_threshold": DEFAULT_DEDUP_OVERLAP,
                "centroid_distance_m": DEFAULT_DEDUP_DISTANCE,
                "audits": dedup_audits,
            },
            "matching_rule": {
                "buffer_m": BUFFER_M,
                "max_area_ratio": MAX_AREA_RATIO,
                "max_candidates": MAX_CANDIDATES,
                "max_distance_m": MAX_DISTANCE_M,
                "min_polygon_iou": MIN_POLYGON_IOU,
                "ambiguity_margin": AMBIGUITY_MARGIN,
                "score_weights": SCORE_WEIGHTS,
                "shape_similarity": "min(area_ratio_reciprocal) * min(perimeter_ratio_reciprocal)",
                "assignment": "deterministic_global_greedy_one_to_one_after_ambiguity_exclusion",
            },
            "inputs": input_receipts,
            "counts": {
                "tracking_objects": len(track_gdf),
                "transition_events": len(transition_gdf),
                "state_events": len(state_gdf),
                "states": dict(state_counts),
                "transitions": dict(transition_counts),
            },
            "model_governance": {
                "b0_role": "primary_semantic_geometry_candidate",
                "b1": b1_status,
                "scientific_status": "candidate_not_ground_truth",
                "accuracy_claim": False,
                "causal_claim": False,
            },
        }
        json_dump(block_tracking / "追踪回执.json", receipt)
        json_dump(block_state / "三态回执.json", receipt)
        global_rows.append({
            "block": block,
            "tracking_objects": len(track_gdf),
            "transition_events": len(transition_gdf),
            "state_events": len(state_gdf),
            **{f"state_{key}": value for key, value in state_counts.items()},
        })
        all_receipts.append(receipt)

    csv_dump(args.tracking_root / "追踪总统计.csv", global_rows)
    json_dump(args.tracking_root / "追踪总回执.json", {
        "status": "TRACKING_COMPLETE_CANDIDATE_ONLY",
        "created_at": utc_now(),
        "blocks": list(args.blocks),
        "dates": list(args.dates),
        "metric_crs": METRIC_CRS,
        "geometry_modified": False,
        "scientific_status": "candidate_not_ground_truth",
        "b1_verifier": b1_status,
        "block_receipts": [str(args.tracking_root / block / "追踪回执.json") for block in args.blocks],
    })
    print(json.dumps({"status": "TRACKING_COMPLETE_CANDIDATE_ONLY", "blocks": len(args.blocks), "dates": list(args.dates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
