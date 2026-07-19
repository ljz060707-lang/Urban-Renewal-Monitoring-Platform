import geopandas as gpd
from shapely.geometry import Polygon, box

from urban_monitor.geometry import regularize_building_polygon
from urban_monitor.tracking import Observation, assign_tracks, detect_lineage_events, summarize_tracks


def test_new_building_interval():
    dates = ["2022-01-01", "2023-01-01", "2024-01-01"]
    observations = [
        Observation("2023-01-01", box(0, 0, 10, 10), 0.9, "p1", track_id="trk1"),
        Observation("2024-01-01", box(0, 0, 10, 10), 0.9, "p2", track_id="trk1"),
    ]
    _, intervals, summary = summarize_tracks(
        observations,
        dates,
        min_persistence=2,
        isolated_gap_repair=True,
        construction_area_ratio=2.0,
        low_confidence_threshold=0.5,
    )
    assert summary.iloc[0]["state"] == "new_building"
    assert intervals.iloc[0]["interval_start"] == "2022-01-01"
    assert intervals.iloc[0]["interval_end"] == "2023-01-01"


def test_off_nadir_shift_keeps_track():
    by_date = {
        "2022-01-01": gpd.GeoDataFrame([{"prediction_id": "a", "score": 0.9, "geometry": box(0, 0, 10, 10)}], crs="EPSG:32650"),
        "2023-01-01": gpd.GeoDataFrame([{"prediction_id": "b", "score": 0.9, "geometry": box(12, 0, 22, 10)}], crs="EPSG:32650"),
    }
    observations = assign_tracks(
        by_date,
        min_iou=0.1,
        min_buffer_iou=0.05,
        min_bbox_iou=0.01,
        max_centroid_distance_m=20,
        buffer_m=3,
        date_metadata={"2022-01-01": {"off_nadir_deg": 2}, "2023-01-01": {"off_nadir_deg": 20}},
        off_nadir_scale_m_per_degree=0.4,
    )
    assert len({o.track_id for o in observations}) == 1
    assert observations[1].match_buffer_iou > observations[1].match_iou


def test_gap_can_be_occlusion_or_construction():
    dates = ["2022-01-01", "2023-01-01", "2024-01-01"]
    occlusion = [
        Observation("2022-01-01", box(0, 0, 10, 10), 0.9, "a", track_id="occ"),
        Observation("2024-01-01", box(0.5, 0, 10.5, 10), 0.9, "b", track_id="occ"),
    ]
    _, _, summary = summarize_tracks(
        occlusion,
        dates,
        min_persistence=2,
        isolated_gap_repair=True,
        construction_area_ratio=1.5,
        low_confidence_threshold=0.5,
        max_soft_gap_epochs=1,
        max_soft_gap_days=800,
    )
    assert bool(summary.iloc[0]["occlusion_candidate"])
    assert summary.iloc[0]["state"] == "stable"

    construction = [
        Observation("2022-01-01", box(0, 0, 10, 10), 0.9, "a", track_id="con"),
        Observation("2024-01-01", box(0, 0, 20, 20), 0.9, "b", track_id="con"),
    ]
    _, _, summary2 = summarize_tracks(
        construction,
        dates,
        min_persistence=2,
        isolated_gap_repair=True,
        construction_area_ratio=1.5,
        low_confidence_threshold=0.5,
        max_soft_gap_epochs=1,
        max_soft_gap_days=800,
    )
    assert bool(summary2.iloc[0]["construction_transition"])
    assert summary2.iloc[0]["state"] == "construction_candidate"


def test_merge_lineage_event():
    by_date = {
        "2022-01-01": gpd.GeoDataFrame(
            [
                {"prediction_id": "a", "score": 0.9, "geometry": box(0, 0, 10, 10)},
                {"prediction_id": "b", "score": 0.9, "geometry": box(10, 0, 20, 10)},
            ],
            crs="EPSG:32650",
        ),
        "2023-01-01": gpd.GeoDataFrame(
            [{"prediction_id": "c", "score": 0.9, "geometry": box(0, 0, 20, 10)}], crs="EPSG:32650"
        ),
    }
    observations = assign_tracks(by_date, min_iou=0.05, max_centroid_distance_m=20)
    events = detect_lineage_events(
        by_date,
        observations,
        min_component_overlap=0.3,
        min_event_coverage=0.8,
        lineage_buffer_m=0,
    )
    assert not events.empty
    assert "merge_redevelopment" in set(events["event_type"])
    event = events[events["event_type"] == "merge_redevelopment"].iloc[0]
    assert event["n_parent"] == 2
    assert event["n_child"] == 1
    assert event["child_track_ids"] not in event["parent_track_ids"].split(";")


def test_regularization_reduces_noisy_vertices():
    noisy = Polygon([(0, 0), (4, 0.1), (8, 0), (8.1, 4), (8, 8), (4, 7.9), (0, 8), (-0.1, 4)])
    clean = regularize_building_polygon(noisy, simplify_tolerance=0.2, rectangularity_threshold=0.75)
    assert clean.is_valid
    assert len(clean.exterior.coords) <= len(noisy.exterior.coords)
    assert clean.area > 50
