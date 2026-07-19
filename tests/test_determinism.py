import geopandas as gpd
from shapely.geometry import box

from urban_monitor.tracking import assign_tracks


def _mapping(frame):
    by_date = {"2024-01-01": frame, "2025-01-01": frame.copy()}
    observations = assign_tracks(by_date, min_iou=.1, max_centroid_distance_m=5)
    return {(obs.date, tuple(round(v, 3) for v in obs.geometry.bounds)): obs.track_id for obs in observations}


def test_track_ids_stable_when_input_order_changes():
    frame = gpd.GeoDataFrame([{"prediction_id": "b", "score": .9, "geometry": box(20, 0, 30, 10)}, {"prediction_id": "a", "score": .9, "geometry": box(0, 0, 10, 10)}], crs="EPSG:32650")
    assert _mapping(frame) == _mapping(frame.iloc[::-1].reset_index(drop=True))


def test_equal_candidates_are_marked_ambiguous():
    by_date = {
        "2024-01-01": gpd.GeoDataFrame([{"prediction_id": "a", "score": .9, "geometry": box(0, 0, 10, 10)}, {"prediction_id": "b", "score": .9, "geometry": box(12, 0, 22, 10)}], crs="EPSG:32650"),
        "2025-01-01": gpd.GeoDataFrame([{"prediction_id": "c", "score": .9, "geometry": box(6, 0, 16, 10)}], crs="EPSG:32650"),
    }
    observations = assign_tracks(by_date, min_iou=0, min_buffer_iou=0, min_bbox_iou=0, max_centroid_distance_m=20, ambiguity_margin_threshold=.1)
    assert observations[-1].match_ambiguous
