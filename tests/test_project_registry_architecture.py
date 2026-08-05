from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from urban_monitor.catalog import load_datasets, load_models
from urban_monitor.contracts import CanonicalModelOutput, OutputStatus
from urban_monitor.datasets.onboarding import audit_dataset
from urban_monitor.safety.test_guard import TestAccessGuard, default_policies
from urban_monitor.surface_state.metrics import surface_state_metrics
from urban_monitor.temporal import QualityState, TemporalObservation, TemporalState, TemporalStateEngine
from urban_monitor.temporal.tracking import TrackedObject, match_objects
from urban_monitor.unified_inference import initialize_inference_run


def test_registry_has_authoritative_models_and_test_policies() -> None:
    models = {row["model_id"]: row for row in load_models()}
    datasets = {row["dataset_id"]: row for row in load_datasets()}
    assert models["full20_multisource_v2"]["status"] == "validation_candidate"
    assert models["rsbuilding_swin_t_whu_candidate"]["status"] == "blocked"
    assert datasets["dsifn"]["test_policy"] == "spent_test"
    assert datasets["s2looking"]["test_policy"] == "spent_test"
    assert datasets["second"]["test_policy"] == "locked_holdout"
    assert datasets["whu_building"]["test_policy"] == "locked_holdout"


def test_canonical_output_marks_missing_heads_unsupported() -> None:
    output = CanonicalModelOutput.unsupported({"any_change_probability": np.ones((2, 2))}, model_id="binary")
    assert output.any_change_probability.status == OutputStatus.AVAILABLE
    assert output.construction_probability.status == OutputStatus.UNSUPPORTED
    assert output.construction_probability.value is None


def test_spent_and_locked_tests_are_blocked_during_development(tmp_path: Path) -> None:
    guard = TestAccessGuard(default_policies(), phase="threshold_tuning", audit_path=tmp_path / "audit.json")
    with pytest.raises(PermissionError, match="SPENT"):
        guard.check("dsifn", "test", "test/a.png")
    with pytest.raises(PermissionError, match="forbidden"):
        guard.check("second", "test", "test/a.png")
    assert guard.read_count == 0
    assert (tmp_path / "audit.json").is_file()


def test_temporal_engine_requires_usable_stable_observations() -> None:
    engine = TemporalStateEngine({
        "probability_threshold": 0.55, "min_consecutive_observations": 2, "min_elapsed_days": 180,
        "min_stable_building_observations": 2, "min_stable_vacant_observations": 2, "max_gap_days": 550,
        "quality_states_allowed_for_presence": ["usable", "unknown_quality"],
        "quality_states_allowed_for_absence": ["usable"],
    })
    observations = [
        TemporalObservation("a", date(2019, 1, 1), .1, .8, .1, QualityState.USABLE),
        TemporalObservation("a", date(2020, 1, 1), .1, .8, .1, QualityState.USABLE),
        TemporalObservation("a", date(2021, 1, 1), .1, .1, .8, QualityState.USABLE),
        TemporalObservation("a", date(2022, 1, 1), .1, .1, .8, QualityState.USABLE),
    ]
    result = engine.infer("track-a", observations)
    assert result.state == TemporalState.NEWLY_COMPLETED_BUILDING
    assert result.events[-1]["event"] == "completion"
    missing = engine.infer("track-b", [
        TemporalObservation("b", date(2020, 1, 1), .9, .05, .05, QualityState.OCCLUDED),
        TemporalObservation("b", date(2021, 1, 1), .9, .05, .05, QualityState.REGISTRATION_ISSUE),
    ])
    assert missing.state == TemporalState.UNCERTAIN


def test_tracking_preserves_split_lineage() -> None:
    parent = TrackedObject("p", "2020", box(0, 0, 10, 10), "building", .9, 100, "", "usable", .1)
    children = [
        TrackedObject("c1", "2021", box(0, 0, 5, 10), "building", .9, 50, "", "usable", .1),
        TrackedObject("c2", "2021", box(5, 0, 10, 10), "building", .9, 50, "", "usable", .1),
    ]
    edges = match_objects([parent], children, search_distance=20, min_iou=.01)
    assert {edge.relation for edge in edges} == {"one_to_many"}


def test_surface_metrics_and_unified_inference_manifest(tmp_path: Path) -> None:
    metrics = surface_state_metrics(np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]), class_names=("bg", "vacant", "construction", "building"))
    assert metrics["macro_f1"] == 1.0
    root = initialize_inference_run("scene", "run", output_root=tmp_path, capabilities={"any_change_probability"})
    manifest = (root / "manifest.json").read_text(encoding="utf-8")
    assert '"any_change_probability"' in manifest and '"pending"' in manifest
    assert '"construction_probability"' in manifest and '"unsupported"' in manifest
    assert not (root / "rasters" / "any_change_probability.tif").exists()


def test_unconfigured_future_dataset_is_blocked_without_pixel_reads(tmp_path: Path) -> None:
    report = audit_dataset("configs/datasets/beijing_waste_template.yaml", tmp_path)
    assert report["status"] == "blocked"
    assert report["image_pixel_reads"] == 0
    assert report["test_image_pixel_reads"] == 0
