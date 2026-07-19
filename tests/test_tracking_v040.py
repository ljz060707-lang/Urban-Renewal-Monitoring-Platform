import geopandas as gpd
from shapely.geometry import box

import urban_monitor.tracking as tracking
from urban_monitor.tracking import Observation, assign_tracks, summarize_tracks


def test_spatial_index_bounds_pair_metric_calls(monkeypatch):
    previous = [Observation("2023-01-01", box(i * 1000, 0, i * 1000 + 10, 10), 1, f"p{i}", track_id=f"t{i}") for i in range(100)]
    current = [Observation("2024-01-01", box(i * 1000 + 1, 0, i * 1000 + 11, 10), 1, f"c{i}") for i in range(100)]
    calls = 0
    original = tracking._pair_metrics

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tracking, "_pair_metrics", counted)
    matches = tracking._match(
        previous, current, min_iou=.05, min_buffer_iou=.1, min_bbox_iou=.05,
        max_distance=30, base_buffer_m=4, date_metadata=None,
        off_nadir_scale_m_per_degree=.35, max_dynamic_buffer_m=20,
        weights={"iou": .2, "buffer_iou": .3, "bbox_iou": .1, "centroid": .2, "area": .2},
        ambiguity_margin_threshold=.08,
    )
    assert len(matches) == 100
    assert calls < 300


def test_ambiguity_uses_previous_and_current_margins():
    by_date = {
        "2023-01-01": gpd.GeoDataFrame(
            {"prediction_id": ["old"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:32650"
        ),
        "2024-01-01": gpd.GeoDataFrame(
            {"prediction_id": ["a", "b"]}, geometry=[box(0, 0, 10, 10), box(.2, 0, 10.2, 10)], crs="EPSG:32650"
        ),
    }
    observations = assign_tracks(
        by_date, min_iou=.05, max_centroid_distance_m=20, ambiguity_margin_threshold=.1,
        force_new_track_on_merge=False,
    )
    matched = [item for item in observations if item.date == "2024-01-01" and item.matched_from_track]
    assert len(matched) == 1
    assert matched[0].previous_margin < .1 and matched[0].match_ambiguous
    assert matched[0].match_status == "ambiguous_match"


def test_complete_observation_status_matrix_and_confidence():
    dates = [f"2024-0{month}-01" for month in range(1, 9)]
    observation = Observation(dates[0], box(0, 0, 10, 10), .9, "p1", track_id="trk1")
    context = {
        ("trk1", dates[1]): {"usable_fraction": 1, "covered_fraction": 1, "registration_status": "passed"},
        ("trk1", dates[2]): {"usable_fraction": .2, "covered_fraction": 1, "registration_status": "passed"},
        ("trk1", dates[3]): {"date_quality": "unusable"},
        ("trk1", dates[4]): {"date_quality": "occluded"},
        ("trk1", dates[5]): {"registration_status": "failed"},
        ("trk1", dates[6]): {"covered_fraction": 0},
        ("trk1", dates[7]): {"ambiguous": True},
    }
    long_frame, _, summary = summarize_tracks(
        [observation], dates, min_persistence=2, isolated_gap_repair=False,
        construction_area_ratio=1.5, low_confidence_threshold=.5,
        observation_context=context, min_quality_fraction=.8,
    )
    statuses = dict(zip(long_frame.date, long_frame.observation_status))
    assert statuses == {
        dates[0]: "present", dates[1]: "usable_absent", dates[2]: "not_detected",
        dates[3]: "unusable", dates[4]: "occluded", dates[5]: "registration_issue",
        dates[6]: "not_covered", dates[7]: "ambiguous",
    }
    assert 0 <= summary.iloc[0].confidence_score <= 1
    assert summary.iloc[0].confidence_level == "low"
