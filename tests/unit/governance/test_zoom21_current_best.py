from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOLS = PROJECT_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_nanjing_zoom21_current_best as zoom21  # noqa: E402


def latest_zoom21_root() -> Path:
    roots = sorted((PROJECT_ROOT / "outputs" / "inference").glob("nanjing_jiangbei_zoom21_current_best_*"))
    assert roots, "Zoom 21 audit output has not been created"
    return roots[-1]


def test_date_and_claimed_zoom_parsing() -> None:
    assert zoom21.parse_date(Path("NJ_jianbei150104_zoom_19.tif")) == "2015-01-04"
    assert zoom21.parse_date(Path("scene_2022-12-02_zoom21.tif")) == "2022-12-02"
    assert zoom21.claimed_zoom(Path("scene_zoom_21.tif")) == "zoom_21"
    assert zoom21.claimed_zoom(Path("scene_z21.tif")) == "zoom_21"


def test_actual_nanjing_source_has_no_zoom21_candidate() -> None:
    rows, candidates, comparison = zoom21.audit_inputs(Path(r"D:\Data Set\Satellite\nanjing_jiangbei"))
    assert len(rows) == 7
    assert candidates == []
    assert len(comparison) == 7
    assert all(row["status"] in {"missing_zoom21", "common_date"} for row in comparison)
    assert all(float(row["estimated_GSD_m"]) > 0.1 for row in rows)
    assert all(row["CRS"] == "EPSG:4326" for row in rows)


def test_zoom21_gate_receipt_is_honest_and_registry_only() -> None:
    root = latest_zoom21_root()
    receipt = json.loads((root / "FINAL_ZOOM21_ACCEPTANCE_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "NOT_READY_MISSING_ZOOM21"
    assert receipt["training_started"] is False
    assert receipt["models_reselected"] is False
    assert receipt["thresholds_changed"] is False
    assert receipt["locked_test_pixel_reads"] == 0
    assert receipt["original_source_modified"] is False
    assert receipt["true_accuracy_reported"] is False
    assert receipt["baseline_hash_match_count"] == 252
    assert receipt["baseline_hash_total_count"] == 252


def test_zoom21_blocked_run_does_not_make_fake_common_grid_or_figures() -> None:
    root = latest_zoom21_root()
    assert not (root / "01_common_grid" / "COMMON_AOI.gpkg").exists()
    assert not (root / "01_common_grid" / "COMMON_VALID_MASK.tif").exists()
    figures = json.loads((root / "09_figures_highres" / "FIGURES_STATUS.json").read_text(encoding="utf-8"))
    assert figures["created_count"] == 0
    assert figures["status"] == "not_created_missing_zoom21"


def test_zoom21_reports_do_not_contain_metric_claim_terms_without_gt() -> None:
    root = latest_zoom21_root()
    banned = ("Accuracy", "F1", "IoU", "准确率", "召回率")
    for path in (root / "10_reports").glob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert not any(term in text for term in banned)


def test_scale_audit_keeps_model_specific_plans_separate() -> None:
    root = latest_zoom21_root()
    payload = json.loads((root / "02_scale_audit" / "MODEL_EXPECTED_SCALE.json").read_text(encoding="utf-8"))
    assert {row["module_id"] for row in payload["models"]} == {"M2_Full20", "M3_RSBuilding", "M6_Construction"}
    assert payload["no_common_preprocessing_tensor"] is True
    plan = json.loads((root / "02_scale_audit" / "MODEL_SPECIFIC_RESAMPLING_PLAN.json").read_text(encoding="utf-8"))
    assert plan["shared_tensor"] is False
