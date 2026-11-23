"""Public seams for the additive Xi'an post-inference layer."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "xian_additive_post_inference.py"
_SPEC = spec_from_file_location("xian_additive_post_inference", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_duplicate_actual_observations_are_collapsed_without_inventing_interval():
    rows = [
        {"stem": "nominal_2018", "actual_date": "2015-06-12", "source": "a"},
        {"stem": "nominal_2019", "actual_date": "2015-06-12", "source": "b"},
        {"stem": "nominal_2020", "actual_date": "2015-06-12", "source": "c"},
    ]
    result = _MODULE.fold_actual_observations(rows)
    assert result["status"] == "DUPLICATE_EPOCH"
    assert result["genuine_interval"] is False
    assert len(result["effective_observations"]) == 1


def test_boundary_jitter_is_zero_for_identical_polygon_and_positive_for_shift():
    polygon = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
    }
    shifted = {
        "type": "Polygon",
        "coordinates": [[[2, 0], [12, 0], [12, 10], [2, 10], [2, 0]]],
    }
    same = _MODULE.compute_boundary_jitter(polygon, polygon)
    changed = _MODULE.compute_boundary_jitter(polygon, shifted)
    assert same["polygon_iou"] == pytest.approx(1.0)
    assert same["boundary_jitter_score"] == pytest.approx(0.0)
    assert changed["polygon_iou"] < 1.0
    assert changed["boundary_jitter_score"] > 0.0


def test_confidence_tiers_are_deterministic_and_do_not_create_new_state():
    assert _MODULE.confidence_tier_from_evidence(
        {"temporal_persistence_score": 0.98, "identity_quality": 0.95, "boundary_jitter_score": 0.02, "valid_observation_fraction": 0.99}
    ) == "CONSERVATIVE"
    assert _MODULE.confidence_tier_from_evidence(
        {"temporal_persistence_score": 0.5, "identity_quality": 0.5, "boundary_jitter_score": 0.3, "valid_observation_fraction": 0.5}
    ) == "EXTENDED"
    assert _MODULE.FORMAL_STATES == ("STABLE", "DEMOLITION", "ADDITION")


def test_independent_validation_sample_excludes_calibration_keys():
    rows = [{"sample_key": f"b{i}", "state": "STABLE"} for i in range(8)]
    calibration = {"b1", "b3", "b5"}
    sample = _MODULE.deterministic_validation_sample(rows, calibration_keys=calibration, limit=8)
    assert not calibration.intersection({row["sample_key"] for row in sample})
    assert [row["sample_key"] for row in sample] == ["b0", "b2", "b4", "b6", "b7"]


def test_area_error_matrix_fails_closed_without_reference_labels():
    result = _MODULE.build_area_error_matrix(
        [{"predicted_state": "DEMOLITION", "area_m2": 10.0}],
    )
    assert result["status"] == "NOT_ESTIMABLE_NO_REFERENCE_LABELS"
    assert result["error_adjusted_area_m2"] is None


def test_actual_date_fold_preserves_per_observation_date_provenance():
    result = _MODULE.fold_actual_observations(
        [
            {"stem": "nominal_2018", "nominal_year": 2018, "actual_date": "2015-06-12", "quality_score": 0.8},
            {"stem": "nominal_2019", "nominal_year": 2019, "actual_date": "2015-06-12", "quality_score": 0.9},
            {"stem": "nominal_2020", "nominal_year": 2020, "actual_date": "2020-04-01", "quality_score": 0.7},
        ]
    )
    assert result["genuine_interval"] is True
    assert result["effective_observations"][0]["date_status"] == "DUPLICATE_EPOCH"
    assert result["effective_observations"][1]["date_status"] == "CORE_DATE"
    assert {row["date_status"] for row in result["duplicate_observations"]} == {"DUPLICATE_EPOCH"}


def test_stage_receipt_contains_input_output_and_hashes(tmp_path):
    canonical = tmp_path / "canonical"
    (canonical / "receipts").mkdir(parents=True)
    (canonical / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json").write_text('{"status":"pass"}\n', encoding="utf-8")
    orchestrator = _MODULE.AdditivePostInferenceOrchestrator(canonical, tmp_path / "additive")
    output = orchestrator._stage_dir("S01_PROVENANCE") / "artifact.txt"
    output.parent.mkdir(parents=True)
    output.write_text("ok\n", encoding="utf-8")
    orchestrator._complete("S01_PROVENANCE", {"example": True})
    receipt = _MODULE.read_json(orchestrator._receipt_path("S01_PROVENANCE"))
    assert receipt["status"] == "complete"
    assert receipt["output_files"] == ["s01-provenance/artifact.txt"]
    assert receipt["output_sha256"]["s01-provenance/artifact.txt"]
    assert receipt["validation"]["status"] == "recorded"


def test_orchestrator_emits_additive_package_without_touching_canonical(tmp_path):
    canonical = tmp_path / "canonical"
    (canonical / "receipts").mkdir(parents=True)
    final_receipt = canonical / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json"
    final_receipt.write_text('{"status":"pass"}\n', encoding="utf-8")
    (canonical / "00_provenance").mkdir()
    (canonical / "00_provenance" / "GEOMETRY_SOURCE_SELECTION.csv").write_text(
        "stem,actual_date,nominal_year\nfoo,2020-04-01,2020\n", encoding="utf-8"
    )
    result = _MODULE.AdditivePostInferenceOrchestrator(canonical).run()
    assert result["status"] == "complete"
    package = canonical / "16_additive_post_inference" / "FINAL_XIAN_PACKAGE"
    assert (package / "FINAL_XIAN_RECEIPT.json").is_file()
    assert (package / "FINAL_XIAN_REPORT.md").is_file()
    archive = canonical / "16_additive_post_inference" / "FINAL_XIAN_PACKAGE.zip"
    assert archive.is_file()
    assert _MODULE.zipfile.is_zipfile(archive)
    archive_descriptor = _MODULE.read_json(canonical / "16_additive_post_inference" / "FINAL_XIAN_PACKAGE_ARCHIVE.json")
    assert archive_descriptor["archive"]["sha256"]
    final_output_receipt = _MODULE.read_json(canonical / "16_additive_post_inference" / "ADDITIVE_POST_INFERENCE_RECEIPT.json")
    assert final_output_receipt["package_archive"]["path"].endswith("FINAL_XIAN_PACKAGE.zip")
    assert _MODULE.read_json(final_receipt)["status"] == "pass"


def test_presence_decoder_never_turns_invalid_observation_into_zero():
    assert _MODULE.decode_presence_1_0_u("absent", valid_observation=False) == "U"
    assert _MODULE.decode_presence_1_0_u("absent", valid_observation=True) == "0"
    assert _MODULE.decode_presence_1_0_u(None) == "U"


def test_quality_metrics_preserve_unknown_fields_and_only_derive_from_nodata():
    quality = _MODULE.observation_quality_metrics({"nodata_ratio": 0.2, "cloud_ratio": 0.1})
    assert quality["valid_observation_ratio"] == pytest.approx(0.8)
    assert quality["valid_ratio_source"] == "derived_from_nodata_only"
    assert quality["quality_status"] == "PARTIAL"


def test_inference_gate_rejects_partial_canonical_coverage(tmp_path):
    canonical = tmp_path / "canonical"
    receipt = canonical / "receipts" / "FINAL_TEMPORAL_PACKAGE_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"status":"pass","coverage_status":"partial","missing_geometry_stems":["x"]}\n', encoding="utf-8")
    orchestrator = _MODULE.AdditivePostInferenceOrchestrator(canonical, tmp_path / "additive")
    result = orchestrator.run()
    assert result["status"] == "blocked"
    assert result["failed_stage"] == "S00_INFERENCE_GATE"
    assert list((tmp_path / "additive" / "receipts").glob("FAILURE_RECEIPT*.json"))


def test_temporal_pattern_uses_actual_dates_and_explicit_presence_only():
    rows = [
        {"trajectory_id": "b", "actual_date": "2020-01-01", "presence": 1},
        {"trajectory_id": "b", "actual_date": "2020-02-01", "presence": 0},
        {"trajectory_id": "b", "actual_date": "2020-03-01", "presence": 1, "valid_observation": False},
    ]
    assert _MODULE.temporal_pattern_for_rows(rows)["b"] == "10U"
