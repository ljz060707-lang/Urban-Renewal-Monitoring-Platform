from shapely.geometry import MultiPolygon, Polygon, box

from urban_monitor.geometry import regularize_building_polygon


def test_multipolygon_is_preserved():
    source = MultiPolygon([box(0, 0, 10, 10), box(20, 0, 30, 10)])
    result = regularize_building_polygon(source)
    assert result.geom_type == "MultiPolygon" and len(result.geoms) == 2


def test_courtyard_hole_is_preserved():
    source = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], holes=[[(5, 5), (15, 5), (15, 15), (5, 15)]])
    result = regularize_building_polygon(source)
    assert len(result.interiors) == 1


def test_l_shape_is_not_forced_to_rectangle():
    source = Polygon([(0, 0), (12, 0), (12, 4), (4, 4), (4, 12), (0, 12)])
    result = regularize_building_polygon(source, rectangularity_threshold=.85)
    assert result.area < result.minimum_rotated_rectangle.area


def test_regularization_area_guard():
    source = Polygon([(0, 0), (10, 0), (10, 10), (6, 10), (6, 6), (0, 6)])
    result = regularize_building_polygon(source, rectangularity_threshold=.99, max_area_change_ratio=1.05)
    assert .95 <= result.area / source.area <= 1.05
