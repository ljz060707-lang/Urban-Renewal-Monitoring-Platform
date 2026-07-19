import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from urban_monitor.coordinate_encoding import require_training_safe_encoding, validate_coordinate_encoding
from urban_monitor.schema import validate_annotation_schema, validate_manifest_schema


def test_legacy_manifest_returns_warning():
    frame = pd.DataFrame([{"date": "2024-01-01", "image_path": "a.tif"}])
    assert validate_manifest_schema(frame)


def test_strict_manifest_requires_v030_columns():
    with pytest.raises(ValueError, match="v0.3"):
        validate_manifest_schema(pd.DataFrame([{"date": "2024-01-01", "image_path": "a.tif"}]), require_v030=True)


def test_manifest_rejects_bad_date():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_manifest_schema(pd.DataFrame([{"date": "2024/01/01", "image_path": "a.tif"}]))


def _annotation(wkt_value=""):
    return gpd.GeoDataFrame([{
        "object_id": "o1", "class_name": "building", "geometry_role": "visible_roof",
        "review_status": "reviewed", "source_date": "2024-01-01", "notes": "中文",
        "tracking_geometry_wkt": wkt_value, "geometry": box(0, 0, 1, 1),
    }], crs="EPSG:32650")


def test_annotation_schema_accepts_valid_frame():
    assert validate_annotation_schema(_annotation()) == []


def test_annotation_schema_rejects_bad_wkt():
    with pytest.raises(ValueError, match="tracking_geometry_wkt"):
        validate_annotation_schema(_annotation("NOT WKT"))


@pytest.mark.parametrize("encoding", ["wgs84", "cgcs2000", "projected", "gcj02", "bd09", "unknown"])
def test_coordinate_encoding_enum(encoding):
    assert validate_coordinate_encoding(encoding) == encoding


@pytest.mark.parametrize("encoding", ["gcj02", "bd09", "unknown"])
def test_unsafe_encoding_cannot_enter_training(encoding):
    with pytest.raises(ValueError, match="cannot enter training"):
        require_training_safe_encoding(encoding)
