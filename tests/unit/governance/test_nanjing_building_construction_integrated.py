"""Acceptance tests for the read-only Nanjing integrated run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = PROJECT_ROOT / "outputs/inference/nanjing_building_construction_integrated_20260804_233025"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration_artifact
def test_integrated_run_has_sealed_independent_branches() -> None:
    assert (RUN_ROOT / "02_building/BUILDING_SYSTEM_SEAL.json").exists()
    assert (RUN_ROOT / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json").exists()
    assert _read(RUN_ROOT / "02_building/BUILDING_SYSTEM_SEAL.json")["status"] == "sealed"
    assert _read(RUN_ROOT / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json")["status"] == "sealed"
    assert (RUN_ROOT / "05_cross_reference/BUILDING_CONSTRUCTION_CROSS_REFERENCE.gpkg").exists()


@pytest.mark.integration_artifact
def test_building_contract_fields_and_metric_crs() -> None:
    import geopandas as gpd

    path = RUN_ROOT / "02_building/single_date/2019-03-16/building_objects.gpkg"
    dataframe = gpd.read_file(path, layer="buildings")
    required = {
        "object_id",
        "scene_date",
        "state",
        "area_m2",
        "perimeter_m",
        "centroid_x",
        "centroid_y",
        "mean_probability",
        "uncertainty",
        "source_model",
        "checkpoint_sha256",
        "input_mode",
        "threshold",
        "mmu_m2",
        "scientific_status",
        "geometry",
    }
    assert required.issubset(set(dataframe.columns))
    assert str(dataframe.crs).upper() == "EPSG:32650"
    assert set(dataframe["state"].dropna()) == {"building_candidate"}
    assert set(dataframe["scientific_status"].dropna()) == {"candidate_not_ground_truth"}


@pytest.mark.integration_artifact
def test_construction_keeps_raw_probability_and_gt_boundary() -> None:
    seal = _read(RUN_ROOT / "04_construction/CONSTRUCTION_SYSTEM_SEAL.json")
    method = _read(RUN_ROOT / "04_construction/ranking/CONSTRUCTION_RANKING_METHOD.json")
    audit = _read(RUN_ROOT / "03_evaluation/construction/CONSTRUCTION_GT_AUDIT.json")
    assert seal["raw_probability_preserved"] is True
    assert seal["ranking_changes_raw_probability"] is False
    assert seal["ranking_uses_ground_truth"] is False
    assert method["uses_ground_truth"] is False
    assert method["changes_raw_probabilities"] is False
    assert audit["formal_site_precision_recall_f1_reported"] is False
    assert audit["full_scene_pixel_accuracy_reported"] is False


@pytest.mark.integration_artifact
def test_current_best_and_protected_assets_are_unchanged() -> None:
    invariance = _read(RUN_ROOT / "09_manifests/PROTECTED_ASSET_INVARIANCE.json")
    assert invariance["status"] == "PASS"
    assert invariance["registry_unchanged"] is True
    assert invariance["baseline_hash_match_count"] == 252
    assert invariance["baseline_hash_total_count"] == 252
    assert invariance["baseline_hash_invariance_passed"] is True
    assert invariance["locked_test_pixel_reads"] == 0
    assert invariance["original_source_modified"] is False
    assert all(item["unchanged"] for item in invariance["checkpoint_checks"])


@pytest.mark.integration_artifact
def test_highres_figures_and_single_report_are_complete() -> None:
    chart_map = _read(RUN_ROOT / "09_manifests/CHART_MAP.json")
    assert chart_map["figure_count"] >= 20
    assert chart_map["all_core_figures_meet_minimum"] is True
    for record in chart_map["figures"]:
        path = RUN_ROOT / record["path"]
        assert path.exists(), path
        assert record["width"] >= 3600
        assert record["height"] >= 2100
    reports = list((RUN_ROOT / "10_reports").glob("*.html"))
    assert [path.name for path in reports] == ["NANJING_BUILDING_AND_CONSTRUCTION_SYSTEM_ZH.html"]
    report_text = reports[0].read_text(encoding="utf-8")
    assert "true construction accuracy" not in report_text.lower()
    assert "full scene accuracy" not in report_text.lower()


@pytest.mark.integration_artifact
def test_summary_tables_cover_all_six_dates() -> None:
    expected = {"2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02"}
    for relative in [
        "02_building/BUILDING_SINGLE_DATE_SUMMARY.csv",
        "04_construction/CONSTRUCTION_SINGLE_DATE_RANKED_SUMMARY.csv",
        "05_cross_reference/INTEGRATED_BUILDING_CONSTRUCTION_SUMMARY.csv",
    ]:
        dataframe = pd.read_csv(RUN_ROOT / relative, encoding="utf-8-sig")
        assert set(dataframe["scene_date"].astype(str)) == expected
        assert "2024-10-20" not in set(dataframe["scene_date"].astype(str))


@pytest.mark.integration_artifact
def test_final_receipt_declares_no_training_and_human_review() -> None:
    receipt = _read(RUN_ROOT / "09_manifests/FINAL_NANJING_INTEGRATED_ACCEPTANCE_RECEIPT.json")
    assert receipt["status"] == "completed_with_scientific_boundaries"
    assert receipt["training_started"] is False
    assert receipt["models_reselected"] is False
    assert receipt["thresholds_changed"] is False
    assert receipt["human_review_required"] is True
    assert receipt["scientific_status"]["Nanjing_construction"] == "candidate_level_not_formal_classifier_accuracy"
