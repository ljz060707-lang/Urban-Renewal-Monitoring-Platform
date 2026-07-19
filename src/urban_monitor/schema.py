from __future__ import annotations

from datetime import datetime
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely import wkt

from .coordinate_encoding import validate_coordinate_encoding
from .crs import validate_source_crs


SCHEMA_VERSION = "0.4.0"
MANIFEST_V030_COLUMNS = {
    "date", "image_path", "quality_mask_path", "provider_mask_path", "manual_mask_path",
    "sensor", "source", "gsd_m", "off_nadir_deg", "sun_elevation_deg", "season", "crs",
    "coordinate_encoding", "acquisition_time", "registration_status", "notes",
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _validate_dates(series: pd.Series, name: str) -> None:
    for index, value in series.items():
        text = str(value).strip()
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"{name} row {index} has invalid YYYY-MM-DD date: {value!r}") from exc


def _validate_geodata(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, gpd.GeoDataFrame):
        raise TypeError(f"{name} must be a GeoDataFrame")
    validate_source_crs(frame.crs)
    invalid = frame.geometry.isna() | frame.geometry.is_empty
    if invalid.any():
        raise ValueError(f"{name} contains {int(invalid.sum())} empty geometries")


def validate_manifest_schema(frame: pd.DataFrame, *, require_v030: bool = False) -> list[str]:
    """Validate a manifest and return compatibility warnings."""
    _require_columns(frame, {"date", "image_path"}, "manifest")
    _validate_dates(frame["date"].astype(str).str[:10], "manifest")
    warnings: list[str] = []
    missing_v030 = MANIFEST_V030_COLUMNS - set(frame.columns)
    if missing_v030 and require_v030:
        raise ValueError(f"manifest missing v0.3 columns: {sorted(missing_v030)}")
    if missing_v030:
        warnings.append(f"legacy manifest defaults applied for: {sorted(missing_v030)}")
    if "coordinate_encoding" in frame:
        for value in frame["coordinate_encoding"].fillna("unknown"):
            validate_coordinate_encoding(str(value))
    return warnings


def validate_annotation_schema(frame: pd.DataFrame) -> list[str]:
    _require_columns(
        frame,
        {"object_id", "class_name", "geometry_role", "review_status", "source_date", "notes", "tracking_geometry_wkt"},
        "annotation",
    )
    _validate_geodata(frame, "annotation")
    _validate_dates(frame["source_date"].astype(str).str[:10], "annotation")
    allowed_roles = {"visible_roof", "ground_footprint", "canonical_tracking_anchor"}
    bad_roles = set(frame["geometry_role"].astype(str)) - allowed_roles
    if bad_roles:
        raise ValueError(f"annotation has invalid geometry_role: {sorted(bad_roles)}")
    bad_review = set(frame["review_status"].astype(str)) - {"draft", "reviewed", "rejected"}
    if bad_review:
        raise ValueError(f"annotation has invalid review_status: {sorted(bad_review)}")
    for index, value in frame["tracking_geometry_wkt"].items():
        if str(value).strip() and str(value).lower() != "nan":
            try:
                wkt.loads(str(value))
            except Exception as exc:
                raise ValueError(f"annotation row {index} has invalid tracking_geometry_wkt") from exc
    return []


def validate_prediction_schema(frame: pd.DataFrame) -> list[str]:
    _require_columns(frame, {"prediction_id", "score", "model_version", "schema_version"}, "prediction")
    _validate_geodata(frame, "prediction")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    if scores.isna().any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("prediction score must be numeric and within [0,1]")
    return []


def validate_tracking_schema(frame: pd.DataFrame) -> list[str]:
    _require_columns(frame, {"track_id", "date", "present", "schema_version"}, "tracking")
    _validate_dates(frame["date"].astype(str).str[:10], "tracking")
    if isinstance(frame, gpd.GeoDataFrame):
        validate_source_crs(frame.crs)
    return []


def validate_observation_schema(frame: pd.DataFrame) -> list[str]:
    _require_columns(
        frame,
        {"track_id", "date", "observation_status", "present", "usable_fraction", "covered_fraction", "registration_status"},
        "observation",
    )
    _validate_dates(frame["date"].astype(str).str[:10], "observation")
    allowed = {"present", "usable_absent", "not_detected", "unusable", "occluded", "registration_issue", "not_covered", "ambiguous"}
    invalid = set(frame["observation_status"].astype(str)) - allowed
    if invalid:
        raise ValueError(f"observation has invalid observation_status: {sorted(invalid)}")
    return []


def validate_lineage_schema(frame: pd.DataFrame) -> list[str]:
    _require_columns(
        frame,
        {"event_id", "event_type", "date_from", "date_to", "parent_track_ids", "child_track_ids", "review_status"},
        "lineage",
    )
    _validate_dates(frame["date_from"].astype(str).str[:10], "lineage")
    _validate_dates(frame["date_to"].astype(str).str[:10], "lineage")
    return []


def validate_review_schema(frame: pd.DataFrame) -> list[str]:
    _require_columns(frame, {"row_type", "record_id"}, "review")
    bad = set(frame["row_type"].astype(str)) - {"track", "lineage"}
    if bad:
        raise ValueError(f"review has invalid row_type: {sorted(bad)}")
    return []
