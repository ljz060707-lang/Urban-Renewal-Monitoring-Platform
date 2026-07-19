import pytest
from shapely.geometry import box

from urban_monitor.crs import ensure_metric_crs, ensure_same_crs, select_local_utm_crs, validate_source_crs


@pytest.mark.parametrize("geometry,epsg", [
    (box(116.39, 39.89, 116.41, 39.91), 32650),
    (box(121.49, 31.21, 121.51, 31.23), 32651),
    (box(103.99, 30.64, 104.01, 30.66), 32648),
])
def test_china_city_utm_selection(geometry, epsg):
    assert select_local_utm_crs(geometry, "EPSG:4326").to_epsg() == epsg


def test_missing_crs_is_explicit_error():
    with pytest.raises(ValueError, match="missing"):
        validate_source_crs(None)


def test_geographic_metric_operation_rejected():
    with pytest.raises(ValueError, match="geographic"):
        ensure_metric_crs("EPSG:4326")


def test_web_mercator_metric_operation_rejected():
    with pytest.raises(ValueError, match="display-only"):
        ensure_metric_crs("EPSG:3857")


def test_mismatched_crs_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        ensure_same_crs("EPSG:32650", "EPSG:32651")


def test_southern_aoi_rejected():
    with pytest.raises(ValueError, match="northern"):
        select_local_utm_crs(box(18, -34, 18.1, -33.9), "EPSG:4326")


def test_oversized_aoi_rejected():
    with pytest.raises(ValueError, match="split"):
        select_local_utm_crs(box(100, 30, 102, 31), "EPSG:4326", max_width_km=100)
