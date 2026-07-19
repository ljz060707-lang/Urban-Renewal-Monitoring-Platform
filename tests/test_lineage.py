import geopandas as gpd
from shapely.geometry import box

from urban_monitor.tracking import assign_tracks, detect_lineage_events


def _frame(rows):
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32650")


def test_split_lineage_component():
    by_date = {"2024-01-01": _frame([{"prediction_id": "p", "score": .9, "geometry": box(0, 0, 20, 10)}]), "2025-01-01": _frame([{"prediction_id": "a", "score": .9, "geometry": box(0, 0, 10, 10)}, {"prediction_id": "b", "score": .9, "geometry": box(10, 0, 20, 10)}])}
    events = detect_lineage_events(by_date, assign_tracks(by_date, min_iou=.05, max_centroid_distance_m=30), min_component_overlap=.2, min_event_coverage=.8, lineage_buffer_m=0)
    assert list(events.event_type) == ["split_or_expansion"]


def test_complex_three_to_two_lineage_component():
    by_date = {"2024-01-01": _frame([{"prediction_id": f"p{i}", "score": .9, "geometry": box(i*10, 0, (i+1)*10, 10)} for i in range(3)]), "2025-01-01": _frame([{"prediction_id": "c1", "score": .9, "geometry": box(0, 0, 15, 10)}, {"prediction_id": "c2", "score": .9, "geometry": box(15, 0, 30, 10)}])}
    events = detect_lineage_events(by_date, assign_tracks(by_date, min_iou=.05, max_centroid_distance_m=30), min_component_overlap=.2, min_event_coverage=.8, lineage_buffer_m=0)
    assert list(events.event_type) == ["complex_redevelopment"]
    assert events.iloc[0].n_parent == 3 and events.iloc[0].n_child == 2


def test_lineage_event_id_is_stable():
    first = _frame([{"prediction_id": "a", "score": .9, "geometry": box(0, 0, 10, 10)}, {"prediction_id": "b", "score": .9, "geometry": box(10, 0, 20, 10)}])
    second = _frame([{"prediction_id": "c", "score": .9, "geometry": box(0, 0, 20, 10)}])
    ids = []
    for frame in [first, first.iloc[::-1].reset_index(drop=True)]:
        by_date = {"2024-01-01": frame, "2025-01-01": second}
        events = detect_lineage_events(by_date, assign_tracks(by_date, min_iou=.05, max_centroid_distance_m=30), min_component_overlap=.2, min_event_coverage=.8, lineage_buffer_m=0)
        ids.append(events.iloc[0].event_id)
    assert ids[0] == ids[1]
