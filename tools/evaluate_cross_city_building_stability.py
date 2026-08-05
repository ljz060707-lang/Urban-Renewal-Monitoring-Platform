"""Cross-city building tracking stability evaluator.

The evaluator is deliberately independent of Construction and human GT.  It
reads sealed building object vectors, creates deterministic one-to-one tracks,
keeps split/merge candidate lineage, and reports normalized metrics that can be
compared between Nanjing and Tongzhou despite different AOIs and date counts.

It does not train a model, read any locked test pixels, or alter source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import base


@dataclass(frozen=True)
class TrackingConfig:
    config_id: str = "T0_CURRENT_BASELINE"
    min_raw_iou: float = 0.08
    min_buffered_iou: float = 0.12
    buffer_m: float = 0.0
    max_centroid_distance_m: float = 35.0
    max_area_ratio: float = 6.0
    max_gap_dates: int = 0
    split_merge_lineage: bool = False
    large_roof_specialist: bool = False


CONFIGS: dict[str, TrackingConfig] = {
    "T0_CURRENT_BASELINE": TrackingConfig(),
    "T1_REGISTRATION_ONLY": TrackingConfig(
        config_id="T1_REGISTRATION_ONLY", max_centroid_distance_m=30.0
    ),
    "T2_BUFFERED_IOU": TrackingConfig(
        config_id="T2_BUFFERED_IOU", buffer_m=1.0, min_buffered_iou=0.18
    ),
    "T3_CENTROID_AREA_SHAPE": TrackingConfig(
        config_id="T3_CENTROID_AREA_SHAPE",
        min_raw_iou=0.03,
        max_centroid_distance_m=22.0,
        max_area_ratio=3.5,
    ),
    "T4_SPLIT_MERGE_AWARE": TrackingConfig(
        config_id="T4_SPLIT_MERGE_AWARE",
        min_raw_iou=0.04,
        buffer_m=1.0,
        max_centroid_distance_m=28.0,
        max_area_ratio=4.0,
        split_merge_lineage=True,
    ),
    "T5_ONE_GAP_BRIDGE": TrackingConfig(
        config_id="T5_ONE_GAP_BRIDGE",
        min_raw_iou=0.04,
        buffer_m=1.0,
        max_centroid_distance_m=28.0,
        max_area_ratio=4.0,
        max_gap_dates=1,
        split_merge_lineage=True,
    ),
    "T6_LARGE_ROOF_SPECIALIST": TrackingConfig(
        config_id="T6_LARGE_ROOF_SPECIALIST",
        min_raw_iou=0.04,
        buffer_m=1.5,
        max_centroid_distance_m=35.0,
        max_area_ratio=4.5,
        split_merge_lineage=True,
        large_roof_specialist=True,
    ),
    "T7_FULL_LEGAL_TRACKER": TrackingConfig(
        config_id="T7_FULL_LEGAL_TRACKER",
        min_raw_iou=0.03,
        min_buffered_iou=0.16,
        buffer_m=1.5,
        max_centroid_distance_m=30.0,
        max_area_ratio=4.0,
        max_gap_dates=1,
        split_merge_lineage=True,
        large_roof_specialist=True,
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _quantile(values: Iterable[float], q: float) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.quantile(clean, q)) if clean else None


def _iou(left: base.BaseGeometry, right: base.BaseGeometry) -> float:
    union = left.union(right).area
    return float(left.intersection(right).area / union) if union else 0.0


def _intersection_over_smaller(left: base.BaseGeometry, right: base.BaseGeometry) -> float:
    denominator = min(left.area, right.area)
    return float(left.intersection(right).area / denominator) if denominator else 0.0


def _compactness(geometry: base.BaseGeometry) -> float:
    perimeter = geometry.length
    return float(4.0 * math.pi * geometry.area / (perimeter * perimeter)) if perimeter else 0.0


def _metric_crs(frame: gpd.GeoDataFrame) -> str:
    if frame.crs and getattr(frame.crs, "is_projected", False):
        return str(frame.crs)
    centroid = frame.to_crs("EPSG:4326").geometry.unary_union.centroid
    zone = int((centroid.x + 180.0) // 6.0) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def read_building_objects(path: Path, date: str, metric_crs: str | None = None) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Building vector has no CRS: {path}")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if not len(frame):
        return gpd.GeoDataFrame(
            columns=["object_id", "scene_date", "source_path", "geometry"],
            geometry="geometry",
            crs=metric_crs or frame.crs,
        )
    frame = frame.to_crs(metric_crs or _metric_crs(frame))
    frame.geometry = frame.geometry.make_valid()
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame = frame.reset_index(drop=True)
    frame["object_id"] = [f"{date}_{index:06d}" for index in range(len(frame))]
    frame["scene_date"] = date
    frame["source_path"] = str(path.resolve())
    frame["area_m2"] = frame.geometry.area.astype(float)
    frame["compactness"] = frame.geometry.map(_compactness)
    return frame


def discover_date_vectors(root: Path, dates: Sequence[str] | None = None) -> dict[str, Path]:
    wanted = set(dates or [])
    candidates: dict[str, Path] = {}
    for path in root.rglob("building_objects.gpkg"):
        parts = path.parts
        date = next(
            (
                part
                for part in reversed(parts)
                if len(part) == 10 and part[4] == "-" and part[7] == "-"
            ),
            None,
        )
        if date and (not wanted or date in wanted):
            current = candidates.get(date)
            if current is None or len(path.parts) < len(current.parts):
                candidates[date] = path
    if wanted:
        missing = sorted(wanted - set(candidates))
        if missing:
            raise FileNotFoundError(f"Missing building vectors for dates: {missing}")
    return dict(sorted(candidates.items()))


def load_series(root: Path, dates: Sequence[str] | None = None) -> dict[str, gpd.GeoDataFrame]:
    paths = discover_date_vectors(root, dates)
    if not paths:
        raise FileNotFoundError(f"No building_objects.gpkg found under {root}")
    first_date, first_path = next(iter(paths.items()))
    first = read_building_objects(first_path, first_date)
    metric = str(first.crs)
    result = {first_date: first}
    for date, path in list(paths.items())[1:]:
        result[date] = read_building_objects(path, date, metric)
    return result


def _candidate_rows(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    config: TrackingConfig,
) -> list[dict[str, Any]]:
    if left.empty or right.empty:
        return []
    right_sindex = right.sindex
    rows: list[dict[str, Any]] = []
    search_distance = max(config.max_centroid_distance_m, config.buffer_m * 2.0)
    for li, lrow in left.iterrows():
        lgeom = lrow.geometry
        query = lgeom.buffer(search_distance)
        for ri in right_sindex.query(query, predicate="intersects"):
            rrow = right.iloc[int(ri)]
            rgeom = rrow.geometry
            distance = float(lgeom.centroid.distance(rgeom.centroid))
            if distance > config.max_centroid_distance_m:
                continue
            area_ratio = max(lgeom.area, rgeom.area) / max(min(lgeom.area, rgeom.area), 1e-9)
            max_ratio = config.max_area_ratio
            if config.large_roof_specialist and max(lgeom.area, rgeom.area) >= 5_000:
                max_ratio = min(max_ratio + 1.0, 5.5)
            if area_ratio > max_ratio:
                continue
            raw_iou = _iou(lgeom, rgeom)
            buffered_iou = (
                _iou(lgeom.buffer(config.buffer_m), rgeom.buffer(config.buffer_m))
                if config.buffer_m > 0
                else raw_iou
            )
            ios = _intersection_over_smaller(lgeom, rgeom)
            if raw_iou < config.min_raw_iou and buffered_iou < config.min_buffered_iou and ios < 0.25:
                continue
            area_similarity = 1.0 / area_ratio
            shape_similarity = 1.0 - min(1.0, abs(_compactness(lgeom) - _compactness(rgeom)))
            distance_score = 1.0 - min(1.0, distance / max(config.max_centroid_distance_m, 1e-9))
            score = 0.42 * raw_iou + 0.18 * buffered_iou + 0.15 * ios + 0.10 * area_similarity + 0.08 * shape_similarity + 0.07 * distance_score
            rows.append(
                {
                    "left_index": int(li),
                    "right_index": int(right.index[int(ri)]),
                    "left_object_id": str(lrow.object_id),
                    "right_object_id": str(rrow.object_id),
                    "raw_iou": raw_iou,
                    "buffered_iou": buffered_iou,
                    "intersection_over_smaller": ios,
                    "centroid_displacement_m": distance,
                    "area_ratio": area_ratio,
                    "score": float(score),
                }
            )
    return rows


def _assign_one_to_one(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    left_ids = list(left.index)
    right_ids = list(right.index)
    li_pos = {value: index for index, value in enumerate(left_ids)}
    ri_pos = {value: index for index, value in enumerate(right_ids)}
    cost = np.full((len(left_ids), len(right_ids)), 10.0, dtype=np.float64)
    by_pair: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in candidates:
        pair = (li_pos[int(row["left_index"])], ri_pos[int(row["right_index"])])
        candidate_cost = 1.0 - float(row["score"])
        if candidate_cost < cost[pair]:
            cost[pair] = candidate_cost
            by_pair[pair] = row
    left_selected, right_selected = linear_sum_assignment(cost)
    result: list[dict[str, Any]] = []
    for li, ri in zip(left_selected, right_selected):
        if cost[li, ri] >= 1.0 or (li, ri) not in by_pair:
            continue
        result.append(dict(by_pair[(li, ri)]))
    return result


def build_tracks(
    series: Mapping[str, gpd.GeoDataFrame], config: TrackingConfig
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = list(series)
    object_to_track: dict[str, str] = {}
    track_last_index: dict[str, int] = {}
    match_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    track_counter = 0

    def new_track(object_id: str, date_index: int) -> str:
        nonlocal track_counter
        track_counter += 1
        track_id = f"TRK_{track_counter:07d}"
        object_to_track[object_id] = track_id
        track_last_index[track_id] = date_index
        return track_id

    for row in series[dates[0]].itertuples():
        new_track(str(row.object_id), 0)

    for date_index in range(1, len(dates)):
        left_date, right_date = dates[date_index - 1], dates[date_index]
        left, right = series[left_date], series[right_date]
        candidates = _candidate_rows(left, right, config)
        assignments = _assign_one_to_one(left, right, candidates)
        matched_right: set[str] = set()
        for row in assignments:
            track_id = object_to_track.get(str(row["left_object_id"]))
            if track_id is None:
                track_id = new_track(str(row["left_object_id"]), date_index - 1)
            object_to_track[str(row["right_object_id"])] = track_id
            track_last_index[track_id] = date_index
            matched_right.add(str(row["right_object_id"]))
            match_rows.append(
                {
                    **row,
                    "left_date": left_date,
                    "right_date": right_date,
                    "track_id": track_id,
                    "gap_dates": 0,
                }
            )

        if config.split_merge_lineage:
            left_degree = pd.Series([r["left_object_id"] for r in candidates]).value_counts()
            right_degree = pd.Series([r["right_object_id"] for r in candidates]).value_counts()
            for row in candidates:
                event = None
                if int(left_degree.get(row["left_object_id"], 0)) > 1:
                    event = "split_candidate"
                if int(right_degree.get(row["right_object_id"], 0)) > 1:
                    event = "merge_candidate" if event is None else "split_merge_complex"
                if event:
                    lineage_rows.append(
                        {
                            "left_date": left_date,
                            "right_date": right_date,
                            "left_object_id": row["left_object_id"],
                            "right_object_id": row["right_object_id"],
                            "event_type": event,
                            "raw_iou": row["raw_iou"],
                        }
                    )

        unmatched = right[~right.object_id.astype(str).isin(matched_right)]
        if config.max_gap_dates and date_index >= 2 and not unmatched.empty:
            previous = series[dates[date_index - 2]]
            eligible = previous[
                previous.object_id.astype(str).map(
                    lambda value: track_last_index.get(object_to_track.get(value, ""), -99)
                    == date_index - 2
                )
            ]
            gap_candidates = _candidate_rows(eligible, unmatched, config)
            gap_assignments = _assign_one_to_one(eligible, unmatched, gap_candidates)
            for row in gap_assignments:
                track_id = object_to_track[str(row["left_object_id"])]
                object_to_track[str(row["right_object_id"])] = track_id
                track_last_index[track_id] = date_index
                matched_right.add(str(row["right_object_id"]))
                match_rows.append(
                    {
                        **row,
                        "left_date": dates[date_index - 2],
                        "right_date": right_date,
                        "track_id": track_id,
                        "gap_dates": 1,
                    }
                )
        for row in right[~right.object_id.astype(str).isin(matched_right)].itertuples():
            new_track(str(row.object_id), date_index)

    observations: list[dict[str, Any]] = []
    for date_index, date in enumerate(dates):
        for row in series[date].itertuples():
            observations.append(
                {
                    "track_id": object_to_track[str(row.object_id)],
                    "object_id": str(row.object_id),
                    "scene_date": date,
                    "date_index": date_index,
                    "area_m2": float(row.geometry.area),
                    "geometry": row.geometry,
                }
            )
    tracks = gpd.GeoDataFrame(observations, geometry="geometry", crs=next(iter(series.values())).crs)
    matches = pd.DataFrame(match_rows)
    lineage = pd.DataFrame(lineage_rows)
    summary_rows: list[dict[str, Any]] = []
    for track_id, group in tracks.groupby("track_id"):
        indices = sorted(group.date_index.astype(int).tolist())
        summary_rows.append(
            {
                "track_id": track_id,
                "observation_count": len(group),
                "first_date": group.sort_values("date_index").iloc[0].scene_date,
                "last_date": group.sort_values("date_index").iloc[-1].scene_date,
                "span_dates": indices[-1] - indices[0] + 1,
                "gap_count": indices[-1] - indices[0] + 1 - len(indices),
                "mean_area_m2": float(group.area_m2.mean()),
                "area_cv": float(group.area_m2.std(ddof=0) / group.area_m2.mean()) if group.area_m2.mean() else 0.0,
            }
        )
    return tracks, pd.DataFrame(summary_rows), matches, lineage


def calculate_metrics(
    series: Mapping[str, gpd.GeoDataFrame],
    tracks: gpd.GeoDataFrame,
    track_summary: pd.DataFrame,
    matches: pd.DataFrame,
    lineage: pd.DataFrame,
    config: TrackingConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    dates = list(series)
    total_observations = len(tracks)
    multi_ids = set(track_summary.loc[track_summary.observation_count >= 2, "track_id"])
    tracked_observations = int(tracks.track_id.isin(multi_ids).sum())
    lengths = track_summary.observation_count.astype(int) if len(track_summary) else pd.Series(dtype=int)
    pair_rows: list[dict[str, Any]] = []
    for left_date, right_date in zip(dates, dates[1:]):
        pair = matches[(matches.left_date == left_date) & (matches.right_date == right_date)] if len(matches) else pd.DataFrame()
        matched_left = pair.left_object_id.nunique() if len(pair) else 0
        matched_right = pair.right_object_id.nunique() if len(pair) else 0
        forward = _safe_ratio(matched_left, len(series[left_date]))
        backward = _safe_ratio(matched_right, len(series[right_date]))
        pair_rows.append(
            {
                "left_date": left_date,
                "right_date": right_date,
                "left_count": len(series[left_date]),
                "right_count": len(series[right_date]),
                "matched_count": len(pair),
                "forward_retention": forward,
                "backward_retention": backward,
                "symmetric_retention": (forward + backward) / 2.0,
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    change_candidates = track_summary[(track_summary.first_date != dates[0]) | (track_summary.last_date != dates[-1])]
    unsupported_appear = int((change_candidates.first_date != dates[0]).sum())
    unsupported_disappear = int((change_candidates.last_date != dates[-1]).sum())
    split_count = int((lineage.event_type.isin(["split_candidate", "split_merge_complex"])).sum()) if len(lineage) else 0
    merge_count = int((lineage.event_type.isin(["merge_candidate", "split_merge_complex"])).sum()) if len(lineage) else 0
    full_span = int(((track_summary.first_date == dates[0]) & (track_summary.last_date == dates[-1]) & (track_summary.gap_count == 0)).sum())
    stable_tracks = track_summary[(track_summary.observation_count >= max(2, math.ceil(len(dates) / 2))) & (track_summary.area_cv <= 0.35)]
    overmerge_matches = matches[matches.area_ratio > 2.0] if len(matches) else pd.DataFrame()
    counts = np.array([len(series[date]) for date in dates], dtype=float)
    metrics = {
        "created_at": _now(),
        "config": asdict(config),
        "date_count": len(dates),
        "dates": dates,
        "total_object_observations": total_observations,
        "tracked_object_observations": tracked_observations,
        "tracked_observation_fraction": _safe_ratio(tracked_observations, total_observations),
        "singleton_observation_fraction": _safe_ratio(total_observations - tracked_observations, total_observations),
        "track_count": len(track_summary),
        "singleton_track_fraction": _safe_ratio(int((lengths == 1).sum()), len(track_summary)),
        "track_length_ge_2_fraction": _safe_ratio(int((lengths >= 2).sum()), len(track_summary)),
        "track_length_ge_half_dates_fraction": _safe_ratio(int((lengths >= math.ceil(len(dates) / 2)).sum()), len(track_summary)),
        "full_span_track_fraction": _safe_ratio(full_span, len(track_summary)),
        "median_track_length": _quantile(lengths.tolist(), 0.5),
        "mean_track_length": float(lengths.mean()) if len(lengths) else 0.0,
        "median_adjacent_retention": _quantile(pairwise.symmetric_retention.tolist(), 0.5) if len(pairwise) else None,
        "minimum_adjacent_retention": float(pairwise.symmetric_retention.min()) if len(pairwise) else None,
        "stable_zone_definition": "tracks spanning at least half of dates with area_cv<=0.35; engineering proxy, not GT",
        "stable_zone_retention_rate": _safe_ratio(int(stable_tracks.observation_count.sum()), len(stable_tracks) * len(dates)) if len(stable_tracks) else 0.0,
        "stable_zone_false_appearance_rate": _safe_ratio(int((stable_tracks.first_date != dates[0]).sum()), len(stable_tracks)),
        "stable_zone_false_disappearance_rate": _safe_ratio(int((stable_tracks.last_date != dates[-1]).sum()), len(stable_tracks)),
        "median_raw_IoU": _quantile(matches.raw_iou.tolist(), 0.5) if len(matches) else None,
        "median_buffered_IoU": _quantile(matches.buffered_iou.tolist(), 0.5) if len(matches) else None,
        "median_centroid_displacement_m": _quantile(matches.centroid_displacement_m.tolist(), 0.5) if len(matches) else None,
        "p95_centroid_displacement_m": _quantile(matches.centroid_displacement_m.tolist(), 0.95) if len(matches) else None,
        "median_area_ratio": _quantile(matches.area_ratio.tolist(), 0.5) if len(matches) else None,
        "p05_area_ratio": _quantile(matches.area_ratio.tolist(), 0.05) if len(matches) else None,
        "p95_area_ratio": _quantile(matches.area_ratio.tolist(), 0.95) if len(matches) else None,
        "one_to_one_rate": _safe_ratio(len(matches), max(total_observations - len(series[dates[0]]), 1)),
        "one_to_many_rate": _safe_ratio(split_count, max(len(matches), 1)),
        "many_to_one_rate": _safe_ratio(merge_count, max(len(matches), 1)),
        "split_event_rate": _safe_ratio(split_count, max(len(matches), 1)),
        "merge_event_rate": _safe_ratio(merge_count, max(len(matches), 1)),
        "overmerge_guardrail_rate": _safe_ratio(len(overmerge_matches), max(len(matches), 1)),
        "appeared_with_full20_support_fraction": None,
        "disappeared_with_full20_support_fraction": None,
        "unsupported_appearance_fraction": _safe_ratio(unsupported_appear, max(len(change_candidates), 1)),
        "unsupported_disappearance_fraction": _safe_ratio(unsupported_disappear, max(len(change_candidates), 1)),
        "building_count_coefficient_of_variation": float(counts.std() / counts.mean()) if counts.mean() else 0.0,
        "stable_zone_count_coefficient_of_variation": float(stable_tracks.observation_count.std(ddof=0) / stable_tracks.observation_count.mean()) if len(stable_tracks) and stable_tracks.observation_count.mean() else 0.0,
        "uncertain_track_fraction": _safe_ratio(int(((lengths > 1) & ~track_summary.track_id.isin(stable_tracks.track_id)).sum()), len(track_summary)),
        "stable_track_count": len(stable_tracks),
        "single_date_anomaly_count": int((lengths == 1).sum()),
        "split_event_count": split_count,
        "merge_event_count": merge_count,
        "gap_bridge_count": int((matches.gap_dates == 1).sum()) if len(matches) else 0,
        "locked_test_pixel_reads": 0,
        "human_gt_used": False,
        "construction_read": False,
        "scientific_status": "normalized_engineering_stability_metrics_not_accuracy",
    }
    return metrics, pairwise


def evaluate(
    input_root: Path,
    output_root: Path,
    city: str,
    dates: Sequence[str] | None,
    config: TrackingConfig,
) -> dict[str, Any]:
    series = load_series(input_root, dates)
    tracks, track_summary, matches, lineage = build_tracks(series, config)
    metrics, pairwise = calculate_metrics(series, tracks, track_summary, matches, lineage, config)
    output_root.mkdir(parents=True, exist_ok=True)
    tracks.to_file(output_root / f"{city.upper()}_BUILDING_TRACKS.gpkg", layer="building_tracks", driver="GPKG")
    track_summary.to_csv(output_root / f"{city.upper()}_TRACK_DISTRIBUTION.csv", index=False)
    pairwise.to_csv(output_root / f"{city.upper()}_PAIRWISE_RETENTION.csv", index=False)
    matches.to_csv(output_root / f"{city.upper()}_MATCHES.csv", index=False)
    lineage.to_csv(output_root / f"{city.upper()}_SPLIT_MERGE_LINEAGE.csv", index=False)
    metrics.update(
        {
            "city": city,
            "input_root": str(input_root.resolve()),
            "output_root": str(output_root.resolve()),
            "input_vectors": [
                {
                    "date": date,
                    "path": str(discover_date_vectors(input_root, [date])[date].resolve()),
                    "sha256": _sha256(discover_date_vectors(input_root, [date])[date]),
                    "object_count": len(series[date]),
                }
                for date in series
            ],
        }
    )
    metrics_path = output_root / f"{city.upper()}_STABILITY_METRICS.json"
    _write_json(metrics_path, metrics)
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="T0_CURRENT_BASELINE")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = evaluate(args.input_root, args.output_root, args.city, args.dates, CONFIGS[args.config])
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
