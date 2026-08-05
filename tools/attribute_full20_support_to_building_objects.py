"""Attribute frozen Full20 change support to sealed Building objects.

This module is intentionally read-only with respect to all source assets.  It
creates a complete, object-level audit in a caller-owned run root.  The
attribution is a support diagnostic, not ground truth and not a model
selection signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask, geometry_window, rasterize
from rasterio.windows import Window
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as shapely_transform


DATES = [
    "2015-01-04",
    "2016-08-18",
    "2017-12-11",
    "2018-02-08",
    "2019-03-16",
    "2022-12-02",
]
METRIC_CRS = "EPSG:32650"
SUPPORTED_STATUS = {
    "supported_new",
    "supported_demolition",
    "supported_change_unknown_direction",
    "weak_change_support",
    "no_change_support",
    "full20_unavailable",
    "registration_conflict",
    "invalid_geometry",
    "outside_valid_coverage",
}
UNSUPPORTED_REASONS = {
    "probability_below_frozen_threshold",
    "insufficient_area_overlap",
    "direction_conflict",
    "alignment_conflict",
    "invalid_full20_grid",
    "outside_valid_mask",
    "missing_date_pair",
    "not_applicable",
}
ATTRIBUTION_VERSION = "full20-object-attribution-v1.0"
MIN_SUPPORT_FRACTION = 0.25


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    temp.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[list(columns)]
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _date_from_name(path: Path) -> str | None:
    for part in reversed(path.parts):
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return None


def discover_building_vectors(building_root: Path, dates: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in building_root.rglob("building_objects.gpkg"):
        date = _date_from_name(path)
        if date in dates:
            previous = result.get(date)
            if previous is None or len(path.parts) < len(previous.parts):
                result[date] = path
    missing = sorted(set(dates) - set(result))
    if missing:
        raise FileNotFoundError(f"Missing Building vectors for dates: {missing}")
    return dict(sorted(result.items()))


def load_building_objects(building_root: Path, dates: Sequence[str], metric_crs: str = METRIC_CRS) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for date, path in discover_building_vectors(building_root, dates).items():
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"Building vector has no CRS: {path}")
        frame = frame.to_crs(metric_crs)
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
        frame.geometry = frame.geometry.make_valid()
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
        if "object_id" not in frame.columns:
            frame["object_id"] = [f"{date}_{i:06d}" for i in range(len(frame))]
        frame["scene_date"] = date
        frame["source_vector"] = str(path.resolve())
        frame["object_area_m2"] = frame.geometry.area.astype(float)
        frame["source_sha256"] = sha256(path)
        frames.append(frame[["object_id", "scene_date", "source_vector", "source_sha256", "object_area_m2", "geometry"]])
    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=metric_crs)
    return result


def discover_track_vector(building_root: Path) -> Path | None:
    candidates = sorted(building_root.rglob("NANJING_BUILDING_TRACKS.gpkg"))
    if not candidates:
        candidates = sorted(building_root.rglob("*BUILDING_TRACKS.gpkg"))
    return candidates[0] if candidates else None


def load_tracks(building_root: Path, objects: gpd.GeoDataFrame, dates: Sequence[str], tracking_path: Path | None = None) -> gpd.GeoDataFrame:
    path = tracking_path if tracking_path is not None and tracking_path.exists() else discover_track_vector(building_root)
    if path is not None:
        tracks = gpd.read_file(path)
        if tracks.crs is None:
            raise ValueError(f"Track vector has no CRS: {path}")
        tracks = tracks.to_crs(objects.crs)
        required = {"track_id", "object_id", "scene_date"}
        if not required.issubset(tracks.columns):
            raise ValueError(f"Track vector missing fields {sorted(required - set(tracks.columns))}: {path}")
        tracks = tracks[["track_id", "object_id", "scene_date", "geometry"]].copy()
        tracks["scene_date"] = tracks["scene_date"].astype(str)
    else:
        tracks = objects[["object_id", "scene_date", "geometry"]].copy()
        tracks["track_id"] = tracks["object_id"]
        tracks = tracks[["track_id", "object_id", "scene_date", "geometry"]]
    tracks["date_index"] = tracks["scene_date"].map({date: i for i, date in enumerate(dates)})
    tracks["date_index"] = tracks["date_index"].astype("Int64")
    tracks = tracks.merge(objects[["object_id", "object_area_m2"]], on="object_id", how="left")
    return gpd.GeoDataFrame(tracks, geometry="geometry", crs=objects.crs)


def discover_lineage_ids(building_root: Path) -> set[str]:
    ids: set[str] = set()
    for path in building_root.rglob("*SPLIT_MERGE_LINEAGE.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for column in ("left_object_id", "right_object_id", "object_id"):
            if column in frame.columns:
                ids.update(frame[column].dropna().astype(str))
    return ids


def candidate_records(tracks: gpd.GeoDataFrame, dates: Sequence[str], lineage_ids: set[str]) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    uncertain_ids: set[str] = set()
    represented: set[tuple[str, str]] = set()
    for track_id, group in tracks.groupby("track_id", sort=True):
        group = group.sort_values("date_index")
        indices = [int(value) for value in group["date_index"].dropna()]
        has_gap = any(right - left > 1 for left, right in zip(indices, indices[1:]))
        is_uncertain = len(group) == 1 or has_gap or bool(set(group.object_id.astype(str)) & lineage_ids)
        if is_uncertain:
            uncertain_ids.update(group.object_id.astype(str))
        first = group.iloc[0]
        last = group.iloc[-1]
        if str(first.scene_date) != dates[0]:
            records.append({
                "object_id": str(first.object_id),
                "track_id": str(track_id),
                "scene_date": str(first.scene_date),
                "direction": "new",
                "candidate_type": "appeared",
                "object_area_m2": float(first.object_area_m2),
                "geometry": first.geometry,
            })
            represented.add((str(first.object_id), str(first.scene_date)))
        if str(last.scene_date) != dates[-1]:
            records.append({
                "object_id": str(last.object_id),
                "track_id": str(track_id),
                "scene_date": str(last.scene_date),
                "direction": "demolition",
                "candidate_type": "disappeared",
                "object_area_m2": float(last.object_area_m2),
                "geometry": last.geometry,
            })
            represented.add((str(last.object_id), str(last.scene_date)))
        if is_uncertain:
            for _, row in group.iterrows():
                records.append({
                    "object_id": str(row.object_id),
                    "track_id": str(track_id),
                    "scene_date": str(row.scene_date),
                    "direction": "unknown",
                    "candidate_type": "uncertain",
                    "object_area_m2": float(row.object_area_m2),
                    "geometry": row.geometry,
                })
                represented.add((str(row.object_id), str(row.scene_date)))
        for _, row in group.iterrows():
            key = (str(row.object_id), str(row.scene_date))
            if key not in represented:
                records.append({
                    "object_id": str(row.object_id),
                    "track_id": str(track_id),
                    "scene_date": str(row.scene_date),
                    "direction": "unknown",
                    "candidate_type": "stable_observation",
                    "object_area_m2": float(row.object_area_m2),
                    "geometry": row.geometry,
                })
                represented.add(key)
    return records, uncertain_ids


@dataclass
class PairRaster:
    date_from: str
    date_to: str
    root: Path
    any_change: Path
    new_evidence: Path
    demolition_evidence: Path
    new_mask: Path
    demolition_mask: Path
    any_threshold: float
    new_threshold: float
    demolition_threshold: float


def discover_pairs(full20_root: Path, dates: Sequence[str]) -> tuple[list[PairRaster], dict[str, Any]]:
    manifest_path = full20_root / "FULL20_BASELINE_MANIFEST.json"
    manifest = read_json(manifest_path)
    if not manifest or manifest.get("status") not in {"completed", "success"}:
        raise ValueError(f"Full20 baseline manifest is not completed: {manifest_path}")
    pairs: list[PairRaster] = []
    for entry in manifest.get("pairs", []):
        pair = entry.get("pair", {})
        date_from, date_to = pair.get("date_from"), pair.get("date_to")
        if date_from not in dates or date_to != dates[-1]:
            continue
        pair_root = full20_root / f"{date_from}_to_{date_to}"
        raster_root = pair_root / "rasters"
        required = {
            "any_change": raster_root / "any_change_probability.tif",
            "new_evidence": raster_root / "new_building_evidence.tif",
            "demolition_evidence": raster_root / "demolition_evidence.tif",
            "new_mask": raster_root / "new_building_mask.tif",
            "demolition_mask": raster_root / "demolition_mask.tif",
        }
        if not all(path.exists() for path in required.values()):
            continue
        thresholds = entry.get("thresholds", {})
        pairs.append(PairRaster(
            date_from=date_from,
            date_to=date_to,
            root=pair_root,
            any_change=required["any_change"],
            new_evidence=required["new_evidence"],
            demolition_evidence=required["demolition_evidence"],
            new_mask=required["new_mask"],
            demolition_mask=required["demolition_mask"],
            any_threshold=float(thresholds.get("any_change", 0.75)),
            new_threshold=float(thresholds.get("new_building", 0.6)),
            demolition_threshold=float(thresholds.get("demolition", 0.35)),
        ))
    if not pairs:
        raise FileNotFoundError(f"No formal Full20 date pairs found under {full20_root}")
    return pairs, manifest


def _clamp_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, min(int(window.col_off), width - 1))
    row_off = max(0, min(int(window.row_off), height - 1))
    col_end = max(col_off + 1, min(int(math.ceil(window.col_off + window.width)), width))
    row_end = max(row_off + 1, min(int(math.ceil(window.row_off + window.height)), height))
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _metric_pixel_area(ds: rasterio.DatasetReader, metric_crs: str) -> float:
    transformer = Transformer.from_crs(ds.crs, metric_crs, always_xy=True)
    x0, y0 = ds.transform * (0.0, 0.0)
    x1, y1 = ds.transform * (1.0, 1.0)
    points = [transformer.transform(x0, y0), transformer.transform(x1, y0), transformer.transform(x1, y1), transformer.transform(x0, y1)]
    return abs(Polygon(points).area)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _read_masked(ds: rasterio.DatasetReader, window: Window) -> tuple[np.ndarray, np.ndarray]:
    data = ds.read(1, window=window, masked=True)
    return np.asarray(data.filled(np.nan), dtype=float), ~np.ma.getmaskarray(data)


class RasterAttributor:
    def __init__(self, pairs: Sequence[PairRaster], metric_crs: str = METRIC_CRS) -> None:
        self.pairs = {(pair.date_from, pair.date_to): pair for pair in pairs}
        self.metric_crs = metric_crs
        self._pixel_area: dict[Path, float] = {}

    def _pair_for(self, date: str) -> PairRaster | None:
        return next((pair for (date_from, _), pair in self.pairs.items() if date_from == date), None)

    def _attribute_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        geometry = record.get("geometry")
        date = str(record["scene_date"])
        direction = str(record["direction"])
        pair = self._pair_for(date)
        base = {
            "object_id": str(record["object_id"]),
            "track_id": str(record["track_id"]),
            "scene_date": date,
            "date_pair": f"{date}_to_{DATES[-1]}" if pair else None,
            "candidate_type": str(record["candidate_type"]),
            "direction_requested": direction,
            "full20_available": bool(pair),
            "full20_change_probability_mean": None,
            "full20_change_probability_max": None,
            "full20_change_covered_fraction": 0.0,
            "full20_new_probability_mean": None,
            "full20_demolition_probability_mean": None,
            "full20_direction": None,
            "full20_supported_area_m2": 0.0,
            "object_area_m2": _safe_float(record.get("object_area_m2")),
            "support_fraction": 0.0,
            "registration_overlap_quality": 0.0,
            "support_status": "full20_unavailable",
            "unsupported_reason": "missing_date_pair" if pair is None else "not_applicable",
            "attribution_version": ATTRIBUTION_VERSION,
        }
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            base["support_status"] = "invalid_geometry"
            base["unsupported_reason"] = "invalid_full20_grid"
            return base
        if pair is None:
            return base
        try:
            geometry_metric = geometry
            raster_path = pair.any_change
            with rasterio.open(raster_path) as any_ds:
                if any_ds.crs is None or not any_ds.transform:
                    base["support_status"] = "invalid_geometry"
                    base["unsupported_reason"] = "invalid_full20_grid"
                    return base
                to_raster = Transformer.from_crs(self.metric_crs, any_ds.crs, always_xy=True).transform
                raster_geometry = shapely_transform(to_raster, geometry_metric)
                raster_bounds = box(*any_ds.bounds)
                if raster_geometry.intersection(raster_bounds).is_empty:
                    base["support_status"] = "outside_valid_coverage"
                    base["unsupported_reason"] = "outside_valid_mask"
                    return base
                window = _clamp_window(geometry_window(any_ds, [mapping(raster_geometry)], pad_x=1, pad_y=1), any_ds.width, any_ds.height)
                change, valid = _read_masked(any_ds, window)
                mask = rasterize([(mapping(raster_geometry), 1)], out_shape=change.shape, transform=any_ds.window_transform(window), fill=0, all_touched=True).astype(bool)
                valid_geom = mask & valid & np.isfinite(change)
                if not valid_geom.any():
                    base["support_status"] = "outside_valid_coverage"
                    base["unsupported_reason"] = "outside_valid_mask"
                    return base
                object_area = float(record.get("object_area_m2") or geometry_metric.area or 0.0)
                pixel_area = self._pixel_area.setdefault(raster_path, _metric_pixel_area(any_ds, self.metric_crs))
                valid_area = float(valid_geom.sum() * pixel_area)
                coverage = min(1.0, valid_area / object_area) if object_area else 0.0
                base["registration_overlap_quality"] = coverage
                base["full20_change_probability_mean"] = _safe_float(np.nanmean(change[valid_geom]))
                base["full20_change_probability_max"] = _safe_float(np.nanmax(change[valid_geom]))
                if coverage < 0.25:
                    base["support_status"] = "registration_conflict"
                    base["unsupported_reason"] = "alignment_conflict"
                    return base
                def read_direction(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
                    with rasterio.open(path) as ds:
                        if ds.crs != any_ds.crs or ds.width != any_ds.width or ds.height != any_ds.height or ds.transform != any_ds.transform:
                            raise ValueError("Full20 direction grid does not align with any-change grid")
                        data, direction_valid = _read_masked(ds, window)
                        return data, direction_valid & valid_geom, self._pixel_area.setdefault(path, _metric_pixel_area(ds, self.metric_crs)), float(ds.nodata if ds.nodata is not None else -9999.0)
                new_prob, new_valid, new_pixel_area, _ = read_direction(pair.new_evidence)
                demolition_prob, demolition_valid, demolition_pixel_area, _ = read_direction(pair.demolition_evidence)
                new_vals = new_prob[new_valid]
                demolition_vals = demolition_prob[demolition_valid]
                new_mean = _safe_float(np.nanmean(new_vals)) if new_vals.size else None
                demolition_mean = _safe_float(np.nanmean(demolition_vals)) if demolition_vals.size else None
                new_support = (new_prob >= pair.new_threshold) & new_valid
                demolition_support = (demolition_prob >= pair.demolition_threshold) & demolition_valid
                new_area = float(new_support.sum() * new_pixel_area)
                demolition_area = float(demolition_support.sum() * demolition_pixel_area)
                new_fraction = min(1.0, new_area / object_area) if object_area else 0.0
                demolition_fraction = min(1.0, demolition_area / object_area) if object_area else 0.0
                any_support = (change >= pair.any_threshold) & valid_geom
                base.update({
                    "full20_new_probability_mean": new_mean,
                    "full20_demolition_probability_mean": demolition_mean,
                    "full20_change_covered_fraction": min(1.0, float(any_support.sum() * pixel_area / object_area)) if object_area else 0.0,
                })
                if direction == "new":
                    requested_fraction = new_fraction
                    opposing_fraction = demolition_fraction
                    requested_status = "supported_new"
                    requested_direction = "new"
                elif direction == "demolition":
                    requested_fraction = demolition_fraction
                    opposing_fraction = new_fraction
                    requested_status = "supported_demolition"
                    requested_direction = "demolition"
                else:
                    requested_fraction = max(new_fraction, demolition_fraction)
                    opposing_fraction = min(new_fraction, demolition_fraction)
                    requested_status = "supported_change_unknown_direction"
                    requested_direction = "unknown"
                base["full20_direction"] = requested_direction
                base["full20_supported_area_m2"] = max(new_area, demolition_area) if direction == "unknown" else (new_area if direction == "new" else demolition_area)
                base["support_fraction"] = requested_fraction
                if direction in {"new", "demolition"} and requested_fraction >= MIN_SUPPORT_FRACTION and opposing_fraction >= MIN_SUPPORT_FRACTION:
                    base["support_status"] = "supported_change_unknown_direction"
                    base["unsupported_reason"] = "direction_conflict"
                elif requested_fraction >= MIN_SUPPORT_FRACTION:
                    base["support_status"] = requested_status
                    base["unsupported_reason"] = "not_applicable"
                elif requested_fraction > 0.0:
                    base["support_status"] = "weak_change_support"
                    base["unsupported_reason"] = "insufficient_area_overlap"
                else:
                    base["support_status"] = "no_change_support"
                    base["unsupported_reason"] = "probability_below_frozen_threshold"
                return base
        except Exception as error:
            base["support_status"] = "registration_conflict"
            base["unsupported_reason"] = "invalid_full20_grid" if "grid" in str(error).lower() else "alignment_conflict"
            base["attribution_error"] = f"{type(error).__name__}: {error}"
            return base

    def attribute(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        rows = [self._attribute_one(record) for record in records]
        frame = pd.DataFrame(rows)
        for value in frame.get("support_status", pd.Series(dtype=str)).dropna().astype(str):
            if value not in SUPPORTED_STATUS:
                raise ValueError(f"Unexpected support_status: {value}")
        for value in frame.get("unsupported_reason", pd.Series(dtype=str)).dropna().astype(str):
            if value not in UNSUPPORTED_REASONS:
                raise ValueError(f"Unexpected unsupported_reason: {value}")
        return frame


def aggregate_tracks(object_attributes: pd.DataFrame, tracks: gpd.GeoDataFrame) -> pd.DataFrame:
    columns = [
        "track_id", "object_id", "scene_date", "date_pair", "full20_available",
        "full20_change_probability_mean", "full20_change_probability_max",
        "full20_change_covered_fraction", "full20_new_probability_mean",
        "full20_demolition_probability_mean", "full20_direction",
        "full20_supported_area_m2", "object_area_m2", "support_fraction",
        "registration_overlap_quality", "support_status", "unsupported_reason",
        "attribution_version",
    ]
    base = tracks[["track_id", "object_id", "scene_date"]].copy()
    base["date_pair"] = None
    base["full20_available"] = False
    base["full20_direction"] = "not_applicable"
    base["support_status"] = "no_change_support"
    base["unsupported_reason"] = "not_applicable"
    base["attribution_version"] = ATTRIBUTION_VERSION
    for column in columns:
        if column not in base.columns:
            base[column] = None
    if len(object_attributes):
        grouped = object_attributes.groupby("track_id", sort=True)
        rows: list[dict[str, Any]] = []
        for track_id, group in grouped:
            statuses = set(group.support_status.astype(str))
            if "supported_new" in statuses and "supported_demolition" in statuses:
                status = "supported_change_unknown_direction"
            elif "supported_new" in statuses:
                status = "supported_new"
            elif "supported_demolition" in statuses:
                status = "supported_demolition"
            elif "weak_change_support" in statuses:
                status = "weak_change_support"
            elif "registration_conflict" in statuses:
                status = "registration_conflict"
            elif "outside_valid_coverage" in statuses:
                status = "outside_valid_coverage"
            else:
                status = "no_change_support"
            rows.append({
                "track_id": str(track_id),
                "object_id": ";".join(sorted(group.object_id.astype(str))),
                "scene_date": ";".join(sorted(group.scene_date.astype(str))),
                "date_pair": ";".join(sorted(set(group.date_pair.dropna().astype(str)))),
                "full20_available": bool(group.full20_available.any()),
                "full20_change_probability_mean": _safe_float(group.full20_change_probability_mean.dropna().mean()),
                "full20_change_probability_max": _safe_float(group.full20_change_probability_max.dropna().max()),
                "full20_change_covered_fraction": _safe_float(group.full20_change_covered_fraction.max()),
                "full20_new_probability_mean": _safe_float(group.full20_new_probability_mean.dropna().mean()),
                "full20_demolition_probability_mean": _safe_float(group.full20_demolition_probability_mean.dropna().mean()),
                "full20_direction": ";".join(sorted(set(group.full20_direction.dropna().astype(str)))),
                "full20_supported_area_m2": _safe_float(group.full20_supported_area_m2.max()),
                "object_area_m2": _safe_float(group.object_area_m2.max()),
                "support_fraction": _safe_float(group.support_fraction.max()),
                "registration_overlap_quality": _safe_float(group.registration_overlap_quality.min()),
                "support_status": status,
                "unsupported_reason": ";".join(sorted(set(group.unsupported_reason.dropna().astype(str)))),
                "attribution_version": ATTRIBUTION_VERSION,
            })
        return pd.DataFrame(rows, columns=columns)
    return base[columns]


def run_attribution(
    run_root: Path,
    building_root: Path,
    full20_root: Path,
    dates: Sequence[str] = DATES,
    metric_crs: str = METRIC_CRS,
    old_metrics: Mapping[str, Any] | None = None,
    tracking_path: Path | None = None,
) -> dict[str, Any]:
    output_root = run_root / "01_full20_attribution"
    output_root.mkdir(parents=True, exist_ok=True)
    objects = load_building_objects(building_root, dates, metric_crs)
    tracks = load_tracks(building_root, objects, dates, tracking_path=tracking_path)
    lineage_ids = discover_lineage_ids(building_root)
    records, uncertain_ids = candidate_records(tracks, dates, lineage_ids)
    pairs, manifest = discover_pairs(full20_root, dates)
    attributor = RasterAttributor(pairs, metric_crs)
    attributes = attributor.attribute(records)
    track_attributes = aggregate_tracks(attributes, tracks)
    required_columns = [
        "object_id", "track_id", "scene_date", "date_pair", "full20_available",
        "full20_change_probability_mean", "full20_change_probability_max",
        "full20_change_covered_fraction", "full20_new_probability_mean",
        "full20_demolition_probability_mean", "full20_direction",
        "full20_supported_area_m2", "object_area_m2", "support_fraction",
        "registration_overlap_quality", "support_status", "unsupported_reason",
        "attribution_version",
    ]
    write_csv(output_root / "BUILDING_OBJECT_FULL20_ATTRIBUTION.csv", attributes, required_columns + ["candidate_type", "direction_requested"])
    write_csv(output_root / "BUILDING_TRACK_FULL20_ATTRIBUTION.csv", track_attributes, required_columns)
    appearance = attributes[attributes.candidate_type == "appeared"].copy() if len(attributes) else attributes.copy()
    disappearance = attributes[attributes.candidate_type == "disappeared"].copy() if len(attributes) else attributes.copy()
    write_csv(output_root / "APPEARANCE_SUPPORT_AUDIT.csv", appearance, required_columns + ["candidate_type", "direction_requested"])
    write_csv(output_root / "DISAPPEARANCE_SUPPORT_AUDIT.csv", disappearance, required_columns + ["candidate_type", "direction_requested"])
    def strict_unsupported(frame: pd.DataFrame, supported_status: str) -> int:
        if not len(frame):
            return 0
        return int((frame.support_status.astype(str) != supported_status).sum())
    appearance_total = len(appearance)
    disappearance_total = len(disappearance)
    old_appearance = _safe_float((old_metrics or {}).get("unsupported_appearance_fraction"))
    old_disappearance = _safe_float((old_metrics or {}).get("unsupported_disappearance_fraction"))
    new_appearance = strict_unsupported(appearance, "supported_new") / appearance_total if appearance_total else 0.0
    new_disappearance = strict_unsupported(disappearance, "supported_demolition") / disappearance_total if disappearance_total else 0.0
    object_count = len(objects)
    expected_keys = {f"{row.object_id}|{row.scene_date}" for _, row in objects.iterrows()}
    attributed_keys = {f"{row['object_id']}|{row['scene_date']}" for row in records}
    # Stable objects are deliberately represented in the track-level table; change
    # candidates are represented in the object-level table.  Completeness is the
    # union of both records, not a silent zero for a missing candidate.
    completeness = {
        "created_at": now(),
        "status": "complete",
        "attribution_version": ATTRIBUTION_VERSION,
        "object_count": object_count,
        "track_count": int(tracks.track_id.nunique()),
        "candidate_record_count": len(attributes),
        "appearance_record_count": appearance_total,
        "disappearance_record_count": disappearance_total,
        "uncertain_object_count": len(uncertain_ids),
        "formal_full20_pair_count": len(pairs),
        "expected_formal_dates": list(dates),
        "missing_date_pairs": sorted(set(dates[:-1]) - {pair.date_from for pair in pairs}),
        "old_incomplete_metric": {
            "unsupported_appearance_fraction": old_appearance,
            "unsupported_disappearance_fraction": old_disappearance,
            "source": "previous sealed gate; retained for comparison only",
        },
        "new_complete_metric": {
            "unsupported_appearance_fraction": new_appearance,
            "unsupported_disappearance_fraction": new_disappearance,
            "appearance_supported_new_count": int((appearance.support_status == "supported_new").sum()) if len(appearance) else 0,
            "disappearance_supported_demolition_count": int((disappearance.support_status == "supported_demolition").sum()) if len(disappearance) else 0,
            "appearance_total": appearance_total,
            "disappearance_total": disappearance_total,
        },
        "all_candidate_rows_have_explicit_status": bool(len(attributes) == 0 or attributes.support_status.notna().all()),
        "all_candidate_rows_have_explicit_reason": bool(len(attributes) == 0 or attributes.unsupported_reason.notna().all()),
        "source_full20_manifest": str((full20_root / "FULL20_BASELINE_MANIFEST.json").resolve()),
        "source_full20_manifest_sha256": sha256(full20_root / "FULL20_BASELINE_MANIFEST.json"),
        "source_building_root": str(building_root.resolve()),
        "source_objects_read_only": True,
        "full20_grid_modified": False,
        "missing_candidate_records": sorted(expected_keys - attributed_keys),
    }
    completeness["complete"] = bool(
        completeness["status"] == "complete"
        and not completeness["missing_candidate_records"]
        and not completeness["missing_date_pairs"]
        and completeness["all_candidate_rows_have_explicit_status"]
        and completeness["all_candidate_rows_have_explicit_reason"]
    )
    write_json(output_root / "FULL20_ATTRIBUTION_COMPLETENESS.json", completeness)
    write_json(output_root / "FULL20_ATTRIBUTION_METHOD.json", {
        "created_at": now(),
        "attribution_version": ATTRIBUTION_VERSION,
        "method": "object geometry rasterized against frozen Full20 probability grids; metric area uses EPSG:32650 object geometry and transformed raster pixel footprints",
        "metric_crs": metric_crs,
        "full20_thresholds_source": "formal pair manifest; thresholds are not changed by this run",
        "direction_rules": {"appearance": "new_building", "disappearance": "demolition", "uncertain": "unknown_direction"},
        "minimum_support_fraction": MIN_SUPPORT_FRACTION,
        "weak_support_definition": "0 < requested-direction support_fraction < minimum_support_fraction",
        "valid_coverage_definition": "non-nodata any_change_probability pixels inside object geometry; class masks are not treated as global valid masks",
        "boundary_rule": "rasterize all_touched=True within a clamped geometry window; out-of-grid objects are explicitly recorded",
        "source_full20_sha256": sha256(full20_root / "FULL20_BASELINE_MANIFEST.json"),
        "source_building_vector_sha256": sorted(set(objects.source_sha256.astype(str))),
        "source_read_only": True,
        "ground_truth": False,
        "model_selection_used": False,
    })
    return {
        "status": "PASS" if completeness["complete"] else "FAIL",
        "run_root": str(run_root.resolve()),
        "object_count": object_count,
        "track_count": int(tracks.track_id.nunique()),
        "candidate_record_count": len(attributes),
        "supported_new_count": int((attributes.support_status == "supported_new").sum()) if len(attributes) else 0,
        "supported_demolition_count": int((attributes.support_status == "supported_demolition").sum()) if len(attributes) else 0,
        "unknown_direction_count": int((attributes.support_status == "supported_change_unknown_direction").sum()) if len(attributes) else 0,
        "no_support_count": int((attributes.support_status == "no_change_support").sum()) if len(attributes) else 0,
        "registration_conflict_count": int((attributes.support_status == "registration_conflict").sum()) if len(attributes) else 0,
        "unsupported_appearance_fraction": new_appearance,
        "unsupported_disappearance_fraction": new_disappearance,
        "old_unsupported_appearance_fraction": old_appearance,
        "old_unsupported_disappearance_fraction": old_disappearance,
        "completeness": bool(completeness["complete"]),
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--building-root", type=Path, required=True)
    parser.add_argument("--full20-root", type=Path, required=True)
    parser.add_argument("--metric-crs", default=METRIC_CRS)
    return parser


def main() -> int:
    args = parser().parse_args()
    result = run_attribution(args.run_root.resolve(), args.building_root.resolve(), args.full20_root.resolve(), metric_crs=args.metric_crs)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
