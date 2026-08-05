from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "evaluate_cross_city_building_stability.py"
SPEC = importlib.util.spec_from_file_location("cross_city_stability", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

sys.path.insert(0, str(ROOT / "tools"))
CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "nanjing_stable_controller", ROOT / "tools" / "run_nanjing_stable_building_construction.py"
)
assert CONTROLLER_SPEC and CONTROLLER_SPEC.loader
CONTROLLER = importlib.util.module_from_spec(CONTROLLER_SPEC)
sys.modules[CONTROLLER_SPEC.name] = CONTROLLER
CONTROLLER_SPEC.loader.exec_module(CONTROLLER)


def _frame(date: str, geometries: list) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "object_id": [f"{date}_{index}" for index in range(len(geometries))],
            "scene_date": date,
            "area_m2": [geometry.area for geometry in geometries],
            "geometry": geometries,
        },
        geometry="geometry",
        crs="EPSG:32650",
    )


def test_stable_objects_form_multidate_tracks() -> None:
    dates = ["2019-01-01", "2020-01-01", "2021-01-01"]
    series = {
        dates[0]: _frame(dates[0], [box(0, 0, 10, 10), box(100, 0, 110, 10)]),
        dates[1]: _frame(dates[1], [box(0.5, 0, 10.5, 10), box(100, 0, 110, 10)]),
        dates[2]: _frame(dates[2], [box(1, 0, 11, 10), box(100, 0, 110, 10)]),
    }
    tracks, summary, matches, lineage = MODULE.build_tracks(
        series, MODULE.CONFIGS["T0_CURRENT_BASELINE"]
    )
    metrics, pairwise = MODULE.calculate_metrics(
        series,
        tracks,
        summary,
        matches,
        lineage,
        MODULE.CONFIGS["T0_CURRENT_BASELINE"],
    )
    assert len(summary) == 2
    assert set(summary.observation_count) == {3}
    assert metrics["singleton_observation_fraction"] == 0.0
    assert metrics["full_span_track_fraction"] == 1.0
    assert pairwise.matched_count.tolist() == [2, 2]


def test_one_gap_bridge_does_not_create_middle_object() -> None:
    dates = ["2019-01-01", "2020-01-01", "2021-01-01"]
    series = {
        dates[0]: _frame(dates[0], [box(0, 0, 10, 10)]),
        dates[1]: _frame(dates[1], []),
        dates[2]: _frame(dates[2], [box(0.5, 0, 10.5, 10)]),
    }
    tracks, summary, matches, _ = MODULE.build_tracks(
        series, MODULE.CONFIGS["T5_ONE_GAP_BRIDGE"]
    )
    assert len(tracks) == 2
    assert len(summary) == 1
    assert summary.iloc[0].observation_count == 2
    assert matches.iloc[0].gap_dates == 1
    assert set(tracks.scene_date) == {dates[0], dates[2]}


def test_split_candidates_are_lineage_not_forced_matches() -> None:
    dates = ["2019-01-01", "2020-01-01"]
    series = {
        dates[0]: _frame(dates[0], [box(0, 0, 20, 10)]),
        dates[1]: _frame(dates[1], [box(0, 0, 10, 10), box(10, 0, 20, 10)]),
    }
    tracks, summary, matches, lineage = MODULE.build_tracks(
        series, MODULE.CONFIGS["T4_SPLIT_MERGE_AWARE"]
    )
    assert len(matches) == 1
    assert len(tracks) == 3
    assert len(summary) == 2
    assert "split_candidate" in set(lineage.event_type)


def test_metrics_keep_scientific_boundaries() -> None:
    dates = ["2019-01-01", "2020-01-01"]
    series = {
        dates[0]: _frame(dates[0], [box(0, 0, 10, 10)]),
        dates[1]: _frame(dates[1], [box(0, 0, 10, 10)]),
    }
    tracks, summary, matches, lineage = MODULE.build_tracks(
        series, MODULE.CONFIGS["T0_CURRENT_BASELINE"]
    )
    metrics, _ = MODULE.calculate_metrics(
        series,
        tracks,
        summary,
        matches,
        lineage,
        MODULE.CONFIGS["T0_CURRENT_BASELINE"],
    )
    assert metrics["human_gt_used"] is False
    assert metrics["construction_read"] is False
    assert metrics["locked_test_pixel_reads"] == 0
    assert metrics["scientific_status"] == "normalized_engineering_stability_metrics_not_accuracy"


def test_cross_city_gate_fails_when_singletons_and_overmerge_are_high() -> None:
    tongzhou = {
        "singleton_observation_fraction": 0.36,
        "median_adjacent_retention": 0.43,
        "track_length_ge_2_fraction": 0.41,
        "track_length_ge_half_dates_fraction": 0.41,
        "stable_zone_retention_rate": 0.62,
        "unsupported_appearance_fraction": 0.71,
        "unsupported_disappearance_fraction": 0.60,
        "stable_zone_count_coefficient_of_variation": 0.33,
        "overmerge_guardrail_rate": 0.13,
    }
    nanjing = {
        "singleton_observation_fraction": 0.54,
        "median_adjacent_retention": 0.32,
        "track_length_ge_2_fraction": 0.26,
        "track_length_ge_half_dates_fraction": 0.08,
        "stable_zone_retention_rate": 0.57,
        "unsupported_appearance_fraction": 0.87,
        "unsupported_disappearance_fraction": 0.59,
        "stable_zone_count_coefficient_of_variation": 0.19,
        "overmerge_guardrail_rate": 0.26,
    }
    result = CONTROLLER.gate_result(tongzhou, nanjing)
    assert result["status"] == "FAIL"
    assert result["checks"]["singleton_observation_fraction"] is False
    assert result["checks"]["overmerge_guardrail_rate"] is False


def test_runtime_lock_is_atomic_and_owner_released(tmp_path: Path) -> None:
    lock_path = tmp_path / "urrs_gpu.lock"
    first = CONTROLLER.RuntimeLock(lock_path, "test", tmp_path)
    first.acquire()
    try:
        assert lock_path.exists()
        second = CONTROLLER.RuntimeLock(lock_path, "second", tmp_path)
        try:
            second.acquire()
        except RuntimeError as error:
            assert "Lock unavailable" in str(error)
        else:
            raise AssertionError("second owner unexpectedly acquired the lock")
    finally:
        first.release()
    assert not lock_path.exists()


def test_run_tree_does_not_create_html_reports_directory(tmp_path: Path) -> None:
    CONTROLLER.ensure_tree(tmp_path)
    assert not (tmp_path / "reports").exists()
    assert not list(tmp_path.rglob("*.html"))
