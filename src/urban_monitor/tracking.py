from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry.base import BaseGeometry
from shapely import wkt
from shapely.strtree import STRtree
from shapely.ops import unary_union

from .crs import ensure_metric_crs, ensure_same_crs
from .config import DEFAULT_CONFIG

from .geometry import (
    area_similarity,
    bbox_iou,
    buffered_iou,
    centroid_distance,
    intersection_over_area,
    polygon_iou,
)


@dataclass
class Observation:
    date: str
    geometry: BaseGeometry
    score: float
    prediction_id: str
    quality_usable: bool = True
    track_id: str = ""
    tracking_geometry: BaseGeometry | None = None
    matched_from_track: str = ""
    match_iou: float = 0.0
    match_buffer_iou: float = 0.0
    match_bbox_iou: float = 0.0
    match_centroid_distance: float = np.nan
    match_area_similarity: float = 0.0
    match_shape_similarity: float = 0.0
    match_margin: float = 1.0
    current_margin: float = 1.0
    previous_margin: float = 1.0
    match_score: float = 0.0
    match_status: str = "new_track"
    match_ambiguous: bool = False
    lineage_parent_candidates: str = ""


def _parse_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _midpoint(a: str, b: str) -> str:
    da, db = _parse_date(a), _parse_date(b)
    return (da + timedelta(days=(db - da).days // 2)).isoformat()


def _confidence_from_gap(start: str, end: str) -> str:
    gap = (_parse_date(end) - _parse_date(start)).days
    return "high" if gap <= 183 else "medium" if gap <= 730 else "low"


def _dynamic_buffer(
    date_a: str,
    date_b: str,
    *,
    base_buffer_m: float,
    date_metadata: dict[str, dict[str, Any]] | None,
    off_nadir_scale_m_per_degree: float,
    max_dynamic_buffer_m: float,
) -> float:
    buffer_m = base_buffer_m
    if date_metadata:
        a = date_metadata.get(date_a, {})
        b = date_metadata.get(date_b, {})
        try:
            off_a = float(a.get("off_nadir_deg", 0) or 0)
            off_b = float(b.get("off_nadir_deg", 0) or 0)
            buffer_m += abs(off_a - off_b) * off_nadir_scale_m_per_degree
        except (TypeError, ValueError):
            pass
    return min(buffer_m, max_dynamic_buffer_m)


def _pair_metrics(a: Observation, b: Observation, *, buffer_m: float, max_distance: float) -> dict[str, float]:
    geom_a = a.tracking_geometry or a.geometry
    geom_b = b.tracking_geometry or b.geometry
    iou = polygon_iou(geom_a, geom_b)
    b_iou = buffered_iou(geom_a, geom_b, buffer_m)
    bb_iou = bbox_iou(geom_a, geom_b)
    dist = centroid_distance(geom_a, geom_b)
    area_sim = area_similarity(geom_a, geom_b)
    scale = max(np.sqrt(max(float(geom_a.area), float(geom_b.area), 1e-9)), 1e-6)
    shape_sim = max(0.0, 1.0 - float(geom_a.hausdorff_distance(geom_b)) / scale)
    return {
        "iou": iou,
        "buffer_iou": b_iou,
        "bbox_iou": bb_iou,
        "distance": dist,
        "distance_similarity": max(0.0, 1.0 - dist / max(max_distance, 1e-6)),
        "area_similarity": area_sim,
        "shape_similarity": shape_sim,
    }


def _match(
    prev: list[Observation],
    curr: list[Observation],
    *,
    min_iou: float,
    min_buffer_iou: float,
    min_bbox_iou: float,
    max_distance: float,
    base_buffer_m: float,
    date_metadata: dict[str, dict[str, Any]] | None,
    off_nadir_scale_m_per_degree: float,
    max_dynamic_buffer_m: float,
    weights: dict[str, float],
    ambiguity_margin_threshold: float,
    max_identity_area_ratio: float = 2.0,
):
    if not prev or not curr:
        return []
    metrics_cache: dict[tuple[int, int], dict[str, float]] = {}
    weight_sum = max(sum(max(float(v), 0.0) for v in weights.values()), 1e-9)
    prev_geometries = [item.tracking_geometry or item.geometry for item in prev]
    tree = STRtree(prev_geometries)
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    similarities: dict[tuple[int, int], float] = {}
    for j, b in enumerate(curr):
        search_geometry = (b.tracking_geometry or b.geometry).buffer(max_distance)
        for i_value in tree.query(search_geometry, predicate="intersects"):
            i = int(i_value)
            a = prev[i]
            pair_buffer_m = _dynamic_buffer(
                a.date,
                b.date,
                base_buffer_m=base_buffer_m,
                date_metadata=date_metadata,
                off_nadir_scale_m_per_degree=off_nadir_scale_m_per_degree,
                max_dynamic_buffer_m=max_dynamic_buffer_m,
            )
            metrics = _pair_metrics(a, b, buffer_m=pair_buffer_m, max_distance=max_distance)
            metrics["buffer_m"] = pair_buffer_m
            area_ratio = 1.0 / max(metrics["area_similarity"], 1e-12)
            metrics["area_ratio"] = area_ratio
            spatial_gate = (
                (metrics["iou"] > 0 and metrics["iou"] >= min_iou)
                or (metrics["buffer_iou"] > 0 and metrics["buffer_iou"] >= min_buffer_iou)
                or (metrics["bbox_iou"] > 0 and metrics["bbox_iou"] >= min_bbox_iou)
            )
            if metrics["distance"] > max_distance or not spatial_gate or area_ratio > max_identity_area_ratio:
                continue
            similarity = (
                float(weights.get("iou", 0.15)) * metrics["iou"]
                + float(weights.get("buffer_iou", 0.35)) * metrics["buffer_iou"]
                + float(weights.get("bbox_iou", 0.10)) * metrics["bbox_iou"]
                + float(weights.get("centroid", 0.30)) * metrics["distance_similarity"]
                + float(weights.get("area", 0.10)) * metrics["area_similarity"]
                + float(weights.get("shape", 0.0)) * metrics["shape_similarity"]
            ) / weight_sum
            metrics_cache[(i, j)] = metrics
            similarities[(i, j)] = similarity
            adjacency.setdefault(("p", i), set()).add(("c", j))
            adjacency.setdefault(("c", j), set()).add(("p", i))
    by_current: dict[int, list[float]] = {}
    by_previous: dict[int, list[float]] = {}
    for (i, j), similarity in similarities.items():
        by_current.setdefault(j, []).append(similarity)
        by_previous.setdefault(i, []).append(similarity)
    for values in by_current.values():
        values.sort(reverse=True)
    for values in by_previous.values():
        values.sort(reverse=True)
    for (i, j), similarity in similarities.items():
        current_scores = by_current[j]
        previous_scores = by_previous[i]
        current_margin = current_scores[0] - current_scores[1] if len(current_scores) > 1 else 1.0
        previous_margin = previous_scores[0] - previous_scores[1] if len(previous_scores) > 1 else 1.0
        margin = min(current_margin, previous_margin)
        metrics_cache[(i, j)].update(
            similarity=similarity,
            current_margin=float(current_margin),
            previous_margin=float(previous_margin),
            ambiguity_margin=float(margin),
            ambiguous=float(margin < ambiguity_margin_threshold),
        )
    matches = []
    seen: set[tuple[str, int]] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, set()) - component)
        seen.update(component)
        previous_indices = sorted(index for side, index in component if side == "p")
        current_indices = sorted(index for side, index in component if side == "c")
        cost = np.full((len(previous_indices), len(current_indices)), 1e6, dtype=float)
        for row_index, previous_index in enumerate(previous_indices):
            for column_index, current_index in enumerate(current_indices):
                if (previous_index, current_index) in similarities:
                    cost[row_index, column_index] = 1.0 - similarities[(previous_index, current_index)]
        rows, columns = linear_sum_assignment(cost)
        for row_index, column_index in zip(rows, columns):
            i, j = previous_indices[int(row_index)], current_indices[int(column_index)]
            if (i, j) in metrics_cache:
                matches.append((i, j, metrics_cache[(i, j)]))
    return matches


def _forced_merge_children(
    previous: list[Observation],
    current: list[Observation],
    *,
    component_overlap: float,
    parent_coverage: float,
    child_coverage: float,
    buffer_m: float,
) -> dict[int, list[str]]:
    forced: dict[int, list[str]] = {}
    if not previous:
        return forced
    previous_geometries = [(parent.tracking_geometry or parent.geometry).buffer(buffer_m) for parent in previous]
    tree = STRtree(previous_geometries)
    for j, child in enumerate(current):
        child_geom = (child.tracking_geometry or child.geometry).buffer(buffer_m)
        parents: list[Observation] = []
        for index_value in tree.query(child_geom, predicate="intersects"):
            parent = previous[int(index_value)]
            parent_geom = previous_geometries[int(index_value)]
            if intersection_over_area(parent_geom, child_geom, denominator="first") >= component_overlap:
                parents.append(parent)
        parent_ids = sorted({p.track_id for p in parents if p.track_id})
        if len(parent_ids) < 2:
            continue
        parent_union = unary_union([(p.tracking_geometry or p.geometry).buffer(buffer_m) for p in parents])
        absorbed = intersection_over_area(parent_union, child_geom, denominator="first")
        counterpart = intersection_over_area(parent_union, child_geom, denominator="second")
        if absorbed >= parent_coverage and counterpart >= child_coverage:
            forced[j] = parent_ids
    return forced


def assign_tracks(
    by_date: dict[str, gpd.GeoDataFrame],
    *,
    min_iou: float,
    max_centroid_distance_m: float,
    max_track_gap: int = 1,
    max_track_gap_days: int | None = None,
    buffer_m: float = 4.0,
    min_buffer_iou: float = 0.12,
    min_bbox_iou: float = 0.05,
    match_weights: dict[str, float] | None = None,
    date_metadata: dict[str, dict[str, Any]] | None = None,
    off_nadir_scale_m_per_degree: float = 0.35,
    max_dynamic_buffer_m: float = 20.0,
    force_new_track_on_merge: bool = True,
    merge_component_overlap: float = 0.25,
    merge_parent_coverage: float = 0.70,
    merge_child_coverage: float = 0.25,
    lineage_buffer_m: float = 2.0,
    ambiguity_margin_threshold: float = 0.08,
    max_identity_area_ratio: float = 2.0,
) -> list[Observation]:
    """Assign persistent IDs with off-nadir-tolerant geometry matching.

    The primary match score combines raw IoU, buffered IoU, bounding-box IoU,
    centroid proximity and area similarity. A short-memory track can reconnect
    after one or more missing dates, which is important for seasonal vegetation,
    shadow and temporary model failures.
    """
    observations: list[Observation] = []
    crs_values = [frame.crs for frame in by_date.values() if frame.crs is not None]
    if crs_values:
        ensure_metric_crs(ensure_same_crs(*crs_values))
    recent: dict[str, tuple[int, Observation]] = {}
    next_id = 1
    weights = match_weights or {"iou": 0.15, "buffer_iou": 0.35, "bbox_iou": 0.10, "centroid": 0.30, "area": 0.10}
    sorted_dates = sorted(by_date)
    for date_index, date_value in enumerate(sorted_dates):
        frame = by_date[date_value]
        current = []
        for _, row in frame.iterrows():
            tracking_geometry = None
            for field in ("tracking_geometry_wkt", "canonical_geometry_wkt"):
                value = row.get(field, "")
                if value is not None and str(value).strip():
                    try:
                        tracking_geometry = wkt.loads(str(value))
                    except Exception:
                        tracking_geometry = None
                    break
            prediction_id = str(row.get("prediction_id", "")).strip()
            if not prediction_id:
                prediction_id = "p_" + hashlib.sha1(row.geometry.wkb).hexdigest()[:12]
            current.append(
                Observation(
                    date=date_value,
                    geometry=row.geometry,
                    tracking_geometry=tracking_geometry,
                    score=float(row.get("score", 1.0)),
                    prediction_id=prediction_id,
                )
            )
        current.sort(key=lambda item: (*[round(value, 6) for value in item.geometry.bounds], item.prediction_id))
        candidate_ids: list[str] = []
        for tid, (idx, obs) in recent.items():
            if date_index - idx > max_track_gap + 1:
                continue
            if max_track_gap_days is not None and (_parse_date(date_value) - _parse_date(obs.date)).days > max_track_gap_days:
                continue
            candidate_ids.append(tid)
        candidate_obs = [recent[tid][1] for tid in candidate_ids]
        previous_date = sorted_dates[date_index - 1] if date_index > 0 else None
        immediate_previous = [obs for obs in candidate_obs if previous_date is not None and obs.date == previous_date]
        forced_merge = (
            _forced_merge_children(
                immediate_previous,
                current,
                component_overlap=merge_component_overlap,
                parent_coverage=merge_parent_coverage,
                child_coverage=merge_child_coverage,
                buffer_m=lineage_buffer_m,
            )
            if force_new_track_on_merge and immediate_previous
            else {}
        )
        matched_curr: set[int] = set()
        for i, j, metrics in _match(
            candidate_obs,
            current,
            min_iou=min_iou,
            min_buffer_iou=min_buffer_iou,
            min_bbox_iou=min_bbox_iou,
            max_distance=max_centroid_distance_m,
            base_buffer_m=buffer_m,
            date_metadata=date_metadata,
            off_nadir_scale_m_per_degree=off_nadir_scale_m_per_degree,
            max_dynamic_buffer_m=max_dynamic_buffer_m,
            weights=weights,
            ambiguity_margin_threshold=ambiguity_margin_threshold,
            max_identity_area_ratio=max_identity_area_ratio,
        ):
            if j in forced_merge:
                continue
            current[j].track_id = candidate_ids[i]
            current[j].matched_from_track = candidate_ids[i]
            current[j].match_iou = metrics["iou"]
            current[j].match_buffer_iou = metrics["buffer_iou"]
            current[j].match_bbox_iou = metrics["bbox_iou"]
            current[j].match_centroid_distance = metrics["distance"]
            current[j].match_area_similarity = metrics["area_similarity"]
            current[j].match_shape_similarity = metrics["shape_similarity"]
            current[j].match_margin = metrics.get("ambiguity_margin", 1.0)
            current[j].current_margin = metrics.get("current_margin", 1.0)
            current[j].previous_margin = metrics.get("previous_margin", 1.0)
            current[j].match_score = metrics.get("similarity", 0.0)
            current[j].match_ambiguous = bool(metrics.get("ambiguous", 0.0))
            current[j].match_status = "ambiguous_match" if current[j].match_ambiguous else "matched"
            matched_curr.add(j)
        for j, obs in enumerate(current):
            if j not in matched_curr:
                obs.track_id = f"trk{next_id:07d}"
                next_id += 1
            if j in forced_merge:
                obs.lineage_parent_candidates = ";".join(forced_merge[j])
            recent[obs.track_id] = (date_index, obs)
        observations.extend(current)
    return observations


def repair_short_gaps(
    presence: list[int | None],
    dates: list[str],
    *,
    max_gap_epochs: int = 1,
    max_gap_days: int = 800,
) -> tuple[list[int | None], list[int]]:
    """Repair bounded short absent runs while preserving unusable dates as None."""
    repaired = list(presence)
    indices: list[int] = []
    i = 1
    while i < len(presence) - 1:
        if presence[i] != 0:
            i += 1
            continue
        start = i
        while i < len(presence) and presence[i] == 0:
            i += 1
        end = i - 1
        if start > 0 and i < len(presence) and presence[start - 1] == 1 and presence[i] == 1:
            run_len = end - start + 1
            elapsed = (_parse_date(dates[i]) - _parse_date(dates[start - 1])).days
            if run_len <= max_gap_epochs and elapsed <= max_gap_days:
                for idx in range(start, end + 1):
                    repaired[idx] = 1
                    indices.append(idx)
    return repaired, indices


def repair_isolated_gaps(presence: list[int | None]) -> tuple[list[int | None], list[int]]:
    # Backward-compatible helper retained for callers/tests.
    dummy_dates = [f"2000-01-{i + 1:02d}" for i in range(len(presence))]
    return repair_short_gaps(presence, dummy_dates, max_gap_epochs=1, max_gap_days=99999)


def _gap_geometry_change(
    repaired_idx: list[int],
    all_dates: list[str],
    item_by_date: dict[str, Observation],
    *,
    buffer_m: float,
    max_buffer_iou_for_construction: float,
    min_area_ratio_for_construction: float,
) -> tuple[bool, float, float]:
    if not repaired_idx:
        return False, 1.0, 1.0
    first_gap, last_gap = min(repaired_idx), max(repaired_idx)
    before = next((item_by_date[all_dates[i]] for i in range(first_gap - 1, -1, -1) if all_dates[i] in item_by_date), None)
    after = next((item_by_date[all_dates[i]] for i in range(last_gap + 1, len(all_dates)) if all_dates[i] in item_by_date), None)
    if before is None or after is None:
        return False, 1.0, 1.0
    b_iou = buffered_iou(before.geometry, after.geometry, buffer_m)
    ratio = max(before.geometry.area, after.geometry.area) / max(min(before.geometry.area, after.geometry.area), 1e-9)
    changed = b_iou < max_buffer_iou_for_construction or ratio >= min_area_ratio_for_construction
    return changed, b_iou, ratio


OBSERVATION_STATUSES = {
    "present", "usable_absent", "not_detected", "unusable", "occluded",
    "registration_issue", "not_covered", "ambiguous",
}


def _observation_status(obs: Observation | None, context: dict[str, Any], min_quality_fraction: float) -> str:
    quality = str(context.get("date_quality", "usable") or "usable").lower()
    registration = str(context.get("registration_status", "passed") or "passed").lower()
    covered = float(context.get("covered_fraction", 1.0) or 0.0)
    usable = float(context.get("usable_fraction", 1.0) or 0.0)
    if quality in {"unusable", "occluded", "registration_issue"}:
        return quality
    if registration == "failed":
        return "registration_issue"
    if covered < float(context.get("min_covered_fraction", 0.5)):
        return "not_covered"
    if bool(context.get("ambiguous", False)) or (obs is not None and obs.match_ambiguous):
        return "ambiguous"
    if obs is not None:
        return "present"
    if usable >= min_quality_fraction:
        return "usable_absent"
    return "not_detected"


def compute_confidence_score(components: dict[str, float], weights: dict[str, float] | None = None) -> float:
    selected = weights or DEFAULT_CONFIG["confidence"]["weights"]
    denominator = sum(max(float(value), 0.0) for value in selected.values())
    if denominator <= 0:
        raise ValueError("Confidence weights must contain a positive value")
    score = sum(
        max(float(weight), 0.0) * min(1.0, max(0.0, float(components.get(name, 0.0))))
        for name, weight in selected.items()
    ) / denominator
    return min(1.0, max(0.0, float(score)))


def summarize_tracks(
    observations: list[Observation],
    all_dates: list[str],
    *,
    min_persistence: int,
    isolated_gap_repair: bool,
    construction_area_ratio: float,
    low_confidence_threshold: float,
    usability: dict[tuple[str, str], bool] | None = None,
    max_soft_gap_epochs: int = 1,
    max_soft_gap_days: int = 800,
    construction_gap_buffer_m: float = 4.0,
    construction_gap_max_buffer_iou: float = 0.35,
    observation_context: dict[tuple[str, str], dict[str, Any]] | None = None,
    min_quality_fraction: float = 0.8,
    date_metadata: dict[str, dict[str, Any]] | None = None,
    confidence_weights: dict[str, float] | None = None,
    confidence_high_threshold: float = 0.75,
    confidence_medium_threshold: float = 0.50,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    if not observations:
        empty = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=None)
        return empty, pd.DataFrame(), empty
    usability = usability or {}
    observation_context = observation_context or {}
    date_metadata = date_metadata or {}
    grouped: dict[str, list[Observation]] = {}
    for obs in observations:
        grouped.setdefault(obs.track_id, []).append(obs)
    long_rows: list[dict] = []
    summary_rows: list[dict] = []
    interval_rows: list[dict] = []
    for track_id, items in grouped.items():
        items.sort(key=lambda x: x.date)
        item_by_date = {x.date: x for x in items}
        raw_presence: list[int | None] = []
        status_by_date: dict[str, str] = {}
        context_by_date: dict[str, dict[str, Any]] = {}
        for d in all_dates:
            context = dict(observation_context.get((track_id, d), {}))
            if not usability.get((track_id, d), True) and "date_quality" not in context:
                context["date_quality"] = "unusable"
                context["usable_fraction"] = 0.0
            context.setdefault("date_quality", "usable")
            context.setdefault("usable_fraction", 1.0)
            context.setdefault("covered_fraction", 1.0)
            context.setdefault("registration_status", "passed")
            status = _observation_status(item_by_date.get(d), context, min_quality_fraction)
            status_by_date[d] = status
            context_by_date[d] = context
            raw_presence.append(1 if status == "present" else 0 if status == "usable_absent" else None)
        presence = list(raw_presence)
        repaired_idx: list[int] = []
        if isolated_gap_repair:
            presence, repaired_idx = repair_short_gaps(
                presence,
                all_dates,
                max_gap_epochs=max_soft_gap_epochs,
                max_gap_days=max_soft_gap_days,
            )
        present_indices = [i for i, v in enumerate(presence) if v == 1]
        forced_uncertain = not present_indices
        evidence_indices = present_indices or [all_dates.index(item.date) for item in items]
        first_i, last_i = min(evidence_indices), max(evidence_indices)
        first_date, last_date = all_dates[first_i], all_dates[last_i]
        scores = [x.score for x in items]
        mean_score = float(np.mean(scores))
        areas = [x.geometry.area for x in items]
        area_ratio = max(areas) / max(min(areas), 1e-9)
        raw_interior = raw_presence[first_i : last_i + 1]
        nonmonotonic_raw = any(v == 0 for v in raw_interior)
        nonmonotonic_unrepaired = any(
            raw_presence[i] == 0 and i not in repaired_idx for i in range(first_i, last_i + 1)
        )

        construction_transition, gap_buffer_iou, gap_area_ratio = _gap_geometry_change(
            repaired_idx,
            all_dates,
            item_by_date,
            buffer_m=construction_gap_buffer_m,
            max_buffer_iou_for_construction=construction_gap_max_buffer_iou,
            min_area_ratio_for_construction=construction_area_ratio,
        )
        occlusion_candidate = bool(repaired_idx) and not construction_transition

        usable_before = [i for i in range(0, first_i) if presence[i] == 0]
        usable_after = [i for i in range(last_i + 1, len(all_dates)) if presence[i] == 0]
        initial_run = 0
        for v in presence[first_i:]:
            if v == 1:
                initial_run += 1
            elif v is None:
                continue
            else:
                break
        final_absent_run = 0
        for v in presence[last_i + 1 :]:
            if v == 0:
                final_absent_run += 1
            elif v is None:
                continue
            else:
                break

        if forced_uncertain or any(status == "ambiguous" for status in status_by_date.values()):
            state = "uncertain"
        elif construction_transition:
            state = "construction_candidate"
        elif nonmonotonic_unrepaired:
            state = "uncertain"
        elif first_i > 0 and usable_before and initial_run >= min_persistence:
            state = "new_building"
        elif last_i < len(all_dates) - 1 and usable_after and final_absent_run >= min_persistence:
            state = "demolished"
        elif first_i == 0 and last_i == len(all_dates) - 1:
            state = "stable"
        else:
            state = "uncertain"
        if state in {"new_building", "stable"} and len(areas) >= 2 and area_ratio >= construction_area_ratio:
            state = "construction_candidate"
        if mean_score < low_confidence_threshold:
            state = "uncertain"

        interval_start = interval_end = interval_mid = ""
        latest_absent = earliest_present = latest_present = earliest_absent = ""
        confidence = "low"
        if state in {"new_building", "construction_candidate"} and usable_before:
            latest_absent = all_dates[max(usable_before)]
            earliest_present = first_date
            interval_start, interval_end = latest_absent, earliest_present
            interval_mid = _midpoint(interval_start, interval_end)
            confidence = _confidence_from_gap(interval_start, interval_end)
        elif state == "demolished" and usable_after:
            latest_present = last_date
            earliest_absent = all_dates[min(usable_after)]
            interval_start, interval_end = latest_present, earliest_absent
            interval_mid = _midpoint(interval_start, interval_end)
            confidence = _confidence_from_gap(interval_start, interval_end)
        elif state == "stable":
            confidence = "high" if mean_score >= 0.75 else "medium"
        if mean_score < low_confidence_threshold or nonmonotonic_unrepaired:
            confidence = "low"

        temporal_flag = ""
        if occlusion_candidate:
            temporal_flag = "seasonal_occlusion_or_shadow"
        elif construction_transition:
            temporal_flag = "nonmonotonic_redevelopment_transition"
        elif nonmonotonic_raw:
            temporal_flag = "nonmonotonic_needs_review"

        interval_component = 1.0
        if interval_start and interval_end:
            interval_days = max((_parse_date(interval_end) - _parse_date(interval_start)).days, 0)
            interval_component = max(0.0, 1.0 - interval_days / 730.0)
        usable_values = [float(context_by_date[date].get("usable_fraction", 0.0) or 0.0) for date in all_dates]
        registration_values = [
            1.0 if str(context_by_date[date].get("registration_status", "passed")).lower() != "failed" else 0.0
            for date in all_dates
        ]
        off_nadir_values = []
        for item in items:
            try:
                value = date_metadata.get(item.date, {}).get("off_nadir_deg")
                if value is not None and str(value) != "":
                    off_nadir_values.append(float(value))
            except (TypeError, ValueError):
                pass
        off_nadir_component = 1.0
        if len(off_nadir_values) >= 2:
            off_nadir_component = max(0.0, 1.0 - (max(off_nadir_values) - min(off_nadir_values)) / 30.0)
        matched_scores = [item.match_score for item in items if item.matched_from_track]
        matched_margins = [item.match_margin for item in items if item.matched_from_track]
        components = {
            "detection_score": mean_score,
            "valid_observations": min(1.0, sum(status in {"present", "usable_absent"} for status in status_by_date.values()) / 3.0),
            "interval_width": interval_component,
            "usable_fraction": float(np.mean(usable_values)) if usable_values else 0.0,
            "match_score": float(np.mean(matched_scores)) if matched_scores else 0.5,
            "ambiguity_margin": float(np.mean([min(1.0, margin / 0.20) for margin in matched_margins])) if matched_margins else 1.0,
            "registration": float(np.mean(registration_values)) if registration_values else 0.0,
            "off_nadir_consistency": off_nadir_component,
            "canonical_geometry": float(np.mean([item.tracking_geometry is not None for item in items])),
            "lineage_clarity": 0.5 if any(item.lineage_parent_candidates for item in items) else 1.0,
            "manual_review": 1.0 if any(bool(value.get("manual_review", False)) for value in context_by_date.values()) else 0.5,
        }
        confidence_score = compute_confidence_score(components, confidence_weights)
        if state == "uncertain" or any(item.match_ambiguous for item in items):
            confidence_score = min(confidence_score, confidence_medium_threshold - 0.01)
        confidence = (
            "high" if confidence_score >= confidence_high_threshold
            else "medium" if confidence_score >= confidence_medium_threshold
            else "low"
        )

        representative = max(items, key=lambda x: x.score).geometry
        review_status = "pending" if state == "uncertain" or confidence == "low" else "auto"
        if temporal_flag:
            review_status = "priority"
        summary_rows.append(
            {
                "track_id": track_id,
                "state": state,
                "first_seen": first_date,
                "last_seen": last_date,
                "n_seen": len(items),
                "n_usable_dates": sum(v is not None for v in raw_presence),
                "mean_score": mean_score,
                "area_ratio": area_ratio,
                "nonmonotonic": nonmonotonic_raw,
                "repaired_gap_count": len(repaired_idx),
                "occlusion_candidate": occlusion_candidate,
                "construction_transition": construction_transition,
                "gap_buffer_iou": gap_buffer_iou,
                "gap_area_ratio": gap_area_ratio,
                "temporal_flag": temporal_flag,
                "review_status": review_status,
                "confidence": confidence,
                "confidence_level": confidence,
                "confidence_score": confidence_score,
                "confidence_components": json.dumps(components, ensure_ascii=False, sort_keys=True),
                "schema_version": "0.4.0",
                "geometry": representative,
            }
        )
        interval_rows.append(
            {
                "track_id": track_id,
                "state": state,
                "latest_absent_date": latest_absent,
                "earliest_present_date": earliest_present,
                "latest_present_date": latest_present,
                "earliest_absent_date": earliest_absent,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "interval_mid_estimate": interval_mid,
                "confidence": confidence,
                "confidence_level": confidence,
                "confidence_score": confidence_score,
                "mean_score": mean_score,
                "temporal_flag": temporal_flag,
                "schema_version": "0.4.0",
            }
        )
        for d, value in zip(all_dates, raw_presence):
            obs = item_by_date.get(d)
            context = context_by_date[d]
            status = status_by_date[d]
            long_rows.append(
                {
                    "track_id": track_id,
                    "date": d,
                    "observation_status": status,
                    "usable": status in {"present", "usable_absent"},
                    "present": int(obs is not None),
                    "usable_fraction": float(context.get("usable_fraction", 0.0) or 0.0),
                    "covered_fraction": float(context.get("covered_fraction", 0.0) or 0.0),
                    "score": obs.score if obs else np.nan,
                    "prediction_id": obs.prediction_id if obs else "",
                    "match_status": obs.match_status if obs else "absent",
                    "match_score": obs.match_score if obs else np.nan,
                    "ambiguity_margin": obs.match_margin if obs else np.nan,
                    "date_quality": str(context.get("date_quality", "usable")),
                    "geometry_source": "canonical" if obs and obs.tracking_geometry is not None else "observed" if obs else "representative_location",
                    "notes": str(context.get("notes", "")),
                    "matched_from_track": obs.matched_from_track if obs else "",
                    "match_iou": obs.match_iou if obs else np.nan,
                    "match_buffer_iou": obs.match_buffer_iou if obs else np.nan,
                    "match_bbox_iou": obs.match_bbox_iou if obs else np.nan,
                    "match_centroid_distance": obs.match_centroid_distance if obs else np.nan,
                    "match_area_similarity": obs.match_area_similarity if obs else np.nan,
                    "match_shape_similarity": obs.match_shape_similarity if obs else np.nan,
                    "match_margin": obs.match_margin if obs else np.nan,
                    "current_margin": obs.current_margin if obs else np.nan,
                    "previous_margin": obs.previous_margin if obs else np.nan,
                    "match_ambiguous": obs.match_ambiguous if obs else False,
                    "lineage_parent_candidates": obs.lineage_parent_candidates if obs else "",
                    "registration_status": str(context.get("registration_status", "passed")),
                    "schema_version": "0.4.0",
                    "geometry": obs.geometry if obs else None,
                }
            )
    long_gdf = gpd.GeoDataFrame(long_rows, geometry="geometry", crs=None)
    summary_gdf = gpd.GeoDataFrame(summary_rows, geometry="geometry", crs=None)
    return long_gdf, pd.DataFrame(interval_rows), summary_gdf


def detect_lineage_events(
    by_date: dict[str, gpd.GeoDataFrame],
    observations: list[Observation],
    *,
    min_component_overlap: float = 0.25,
    min_event_coverage: float = 0.70,
    min_counterpart_coverage: float = 0.25,
    lineage_buffer_m: float = 2.0,
) -> gpd.GeoDataFrame:
    """Detect many-to-one and one-to-many footprint lineage between dates.

    These events are not forced into the one-to-one track identity. They are
    stored as explicit parent-child relations for demolition/redevelopment and
    partial split/expansion review.
    """
    dates = sorted(by_date)
    obs_lookup = {(o.date, o.prediction_id): o.track_id for o in observations}
    rows: list[dict[str, Any]] = []
    crs = next((gdf.crs for gdf in by_date.values() if gdf.crs is not None), None)

    for d0, d1 in zip(dates[:-1], dates[1:]):
        prev = by_date[d0].copy()
        curr = by_date[d1].copy()
        if prev.empty or curr.empty:
            continue
        prev = prev.assign(_key=prev.geometry.apply(lambda g: tuple(round(v, 6) for v in g.bounds))).sort_values("_key").reset_index(drop=True)
        curr = curr.assign(_key=curr.geometry.apply(lambda g: tuple(round(v, 6) for v in g.bounds))).sort_values("_key").reset_index(drop=True)
        prev_ids = [obs_lookup.get((d0, str(row.get("prediction_id", ""))), "") for _, row in prev.iterrows()]
        curr_ids = [obs_lookup.get((d1, str(row.get("prediction_id", ""))), "") for _, row in curr.iterrows()]
        adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
        for i, prow in prev.iterrows():
            parent = prow.geometry.buffer(lineage_buffer_m)
            candidate_indices = curr.sindex.query(parent, predicate="intersects")
            for j in candidate_indices:
                j = int(j)
                crow = curr.iloc[j]
                child = crow.geometry.buffer(lineage_buffer_m)
                overlap = max(
                    intersection_over_area(parent, child, denominator="first"),
                    intersection_over_area(parent, child, denominator="second"),
                )
                if overlap >= min_component_overlap:
                    adjacency.setdefault(("p", i), set()).add(("c", j))
                    adjacency.setdefault(("c", j), set()).add(("p", i))
        seen: set[tuple[str, int]] = set()
        for start in sorted(adjacency):
            if start in seen:
                continue
            stack = [start]
            component: set[tuple[str, int]] = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(adjacency.get(node, set()) - component)
            seen.update(component)
            parent_idx = sorted(index for side, index in component if side == "p")
            child_idx = sorted(index for side, index in component if side == "c")
            if len(parent_idx) == 1 and len(child_idx) == 1:
                continue
            parent_union = unary_union([prev.iloc[index].geometry.buffer(lineage_buffer_m) for index in parent_idx])
            child_union = unary_union([curr.iloc[index].geometry.buffer(lineage_buffer_m) for index in child_idx])
            coverage = intersection_over_area(parent_union, child_union, denominator="first")
            counterpart = intersection_over_area(parent_union, child_union, denominator="second")
            if coverage < min_event_coverage or counterpart < min_counterpart_coverage:
                continue
            parent_tracks = sorted({prev_ids[index] for index in parent_idx if prev_ids[index]})
            child_tracks = sorted({curr_ids[index] for index in child_idx if curr_ids[index]})
            if len(parent_idx) > 1 and len(child_idx) > 1:
                event_type = "complex_redevelopment"
            elif len(parent_idx) > 1:
                event_type = "merge_redevelopment"
            else:
                event_type = "split_or_expansion"
            signature = f"{d0}|{d1}|{';'.join(parent_tracks)}|{';'.join(child_tracks)}|{event_type}"
            event_id = "lin_" + hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
            rows.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "date_from": d0,
                    "date_to": d1,
                    "parent_track_ids": ";".join(parent_tracks),
                    "child_track_ids": ";".join(child_tracks),
                    "n_parent": len(parent_tracks),
                    "n_child": len(child_tracks),
                    "coverage": coverage,
                    "counterpart_coverage": counterpart,
                    "review_status": "pending",
                    "schema_version": "0.4.0",
                    "geometry": unary_union([curr.iloc[index].geometry for index in child_idx]),
                }
            )

    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                "event_id", "event_type", "date_from", "date_to", "parent_track_ids",
                "child_track_ids", "n_parent", "n_child", "coverage",
                "counterpart_coverage", "review_status", "geometry"
            ],
            geometry="geometry",
            crs=crs,
        )
    return gpd.GeoDataFrame(sorted(rows, key=lambda row: row["event_id"]), geometry="geometry", crs=crs)
