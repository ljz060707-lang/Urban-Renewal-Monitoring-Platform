import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from urban_monitor.ground_truth import normalize_label_frame


def _normalize(frame, **kwargs):
    return normalize_label_frame(
        frame,
        source_date="2024-01-01",
        target_crs="EPSG:32650",
        image_bounds=(0, 0, 100, 100),
        min_building_area_m2=1,
        **kwargs,
    )


def test_geometry_only_ground_truth_gets_defaults():
    frame = gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20)], crs="EPSG:32650")
    result = _normalize(frame)
    row = result.accepted.iloc[0]
    assert row.class_name == "building"
    assert row.geometry_role == "visible_roof"
    assert row.review_status == "reviewed"
    assert row.source_date == "2024-01-01"
    assert row.object_id.startswith("gt_")
    assert row.schema_version == "0.4.0"


def test_chinese_field_aliases_are_reported():
    frame = gpd.GeoDataFrame(
        [{"类别": "building", "日期": "2024-01-01", "编号": "建筑一", "geometry": box(1, 1, 5, 5)}],
        crs="EPSG:32650",
    )
    result = _normalize(frame)
    assert result.field_mapping == {"class_name": "类别", "source_date": "日期", "object_id": "编号"}
    assert result.accepted.iloc[0].object_id == "建筑一"


def test_invalid_polygon_is_repaired():
    bowtie = Polygon([(10, 10), (20, 20), (20, 10), (10, 20), (10, 10)])
    result = _normalize(gpd.GeoDataFrame(geometry=[bowtie], crs="EPSG:32650"))
    assert len(result.accepted) == 1 and result.accepted.geometry.iloc[0].is_valid
    assert result.stats["repaired"] == 1


def test_points_and_lines_are_rejected():
    frame = gpd.GeoDataFrame(geometry=[Point(1, 1), LineString([(1, 1), (2, 2)])], crs="EPSG:32650")
    result = _normalize(frame)
    assert result.accepted.empty and len(result.rejected) == 2


def test_multipolygon_and_courtyard_are_preserved():
    courtyard = Polygon([(10, 10), (40, 10), (40, 40), (10, 40)], holes=[[(20, 20), (30, 20), (30, 30), (20, 30)]])
    multi = MultiPolygon([box(50, 10, 60, 20), box(70, 10, 80, 20)])
    result = _normalize(gpd.GeoDataFrame(geometry=[courtyard, multi], crs="EPSG:32650"))
    assert len(result.accepted.geometry.iloc[0].interiors) == 1
    assert result.accepted.geometry.iloc[1].geom_type == "MultiPolygon"
    assert len(result.accepted.geometry.iloc[1].geoms) == 2


def test_labels_are_clipped_or_rejected_against_image():
    frame = gpd.GeoDataFrame(geometry=[box(90, 90, 110, 110), box(200, 200, 210, 210)], crs="EPSG:32650")
    result = _normalize(frame)
    assert len(result.accepted) == 1 and result.accepted.geometry.iloc[0].bounds == (90, 90, 100, 100)
    assert result.stats["clipped"] == 1 and len(result.rejected) == 1


def test_small_object_is_filtered():
    frame = gpd.GeoDataFrame(geometry=[box(1, 1, 2, 2)], crs="EPSG:32650")
    result = normalize_label_frame(
        frame,
        source_date="2024-01-01",
        target_crs="EPSG:32650",
        image_bounds=(0, 0, 100, 100),
        min_building_area_m2=10,
    )
    assert result.accepted.empty and len(result.filtered_small) == 1


def test_missing_crs_stops_preparation():
    frame = gpd.GeoDataFrame(geometry=[box(1, 1, 5, 5)])
    try:
        _normalize(frame)
    except ValueError as exc:
        assert "CRS is missing" in str(exc)
    else:
        raise AssertionError("missing CRS should fail")
