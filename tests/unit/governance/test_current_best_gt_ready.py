from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOLS = PROJECT_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from nanjing_gt_common import EXCLUDED_DATES, GT_DEFAULT, VALID_DATES, inspect_gt  # noqa: E402
import evaluate_nanjing_manual_gt  # noqa: E402
import nanjing_gt_pipeline  # noqa: E402
import validate_nanjing_manual_gt  # noqa: E402


def latest_governance_root() -> Path:
    roots = sorted((PROJECT_ROOT / "outputs" / "governance").glob("current_best_gt_ready_*"))
    assert roots, "static governance output has not been created"
    return roots[-1]


def test_current_best_registry_has_three_independent_entries() -> None:
    root = latest_governance_root()
    registry_path = root / "01_model_registry" / "CURRENT_BEST_MODEL_REGISTRY.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = {entry["module_id"]: entry for entry in payload["entries"]}
    assert set(entries) == {"M2_Full20", "M3_RSBuilding", "M6_Construction"}
    for entry in entries.values():
        assert entry["best_legal_model"] is True
        assert entry["production_ready"] is False
        assert entry["checkpoint"]["exists"] is True
        assert entry["checkpoint"]["sha256_match"] is True
        assert entry["config"]["exists"] is True
        assert entry["config"]["sha256_match"] is True


def test_site_classifier_is_disabled_without_gt() -> None:
    root = latest_governance_root()
    payload = json.loads((root / "01_model_registry" / "SITE_CLASSIFIER_LEGALITY.json").read_text(encoding="utf-8"))
    assert payload["status"] == "disabled_pending_human_gt"
    assert payload["legal_model_present"] is False


def test_historical_rejected_experiments_are_registered() -> None:
    root = latest_governance_root()
    rows = (root / "00_inventory" / "ALL_REJECTED_EXPERIMENTS.csv").read_text(encoding="utf-8")
    assert "C-B1" in rows
    assert "rejected" in rows
    assert "C-B2" in rows and "C-B3" in rows


def test_missing_gt_is_explicitly_awaiting_and_never_created(tmp_path: Path) -> None:
    missing = tmp_path / "GT" / "nanjing_manual_gt_v1.gpkg"
    result = inspect_gt(missing)
    assert result["status"] == "awaiting_manual_annotation"
    assert result["metrics_allowed"] is False
    assert result["gt_modified"] is False
    assert not missing.exists()


def test_gt_contract_dates_exclude_2024() -> None:
    assert "2024-10-20" in EXCLUDED_DATES
    assert "2024-10-20" not in VALID_DATES
    assert GT_DEFAULT.name == "nanjing_manual_gt_v1.gpkg"


def test_validation_entry_is_safe_when_gt_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.gpkg"
    output = tmp_path / "validation.json"
    code = validate_nanjing_manual_gt.main(["--gt", str(missing), "--output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "awaiting_manual_annotation"
    assert payload["locked_test_pixel_reads"] == 0
    assert not missing.exists()


def test_evaluation_entry_never_claims_accuracy_without_gt(tmp_path: Path) -> None:
    missing = tmp_path / "missing.gpkg"
    output = tmp_path / "evaluation.json"
    code = evaluate_nanjing_manual_gt.main(["--gt", str(missing), "--output", str(output), "--module", "all"])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "awaiting_manual_annotation"
    assert payload["metrics"] == {}
    assert payload["true_accuracy_reported"] is False
    assert payload["locked_test_pixel_reads"] == 0


def test_unified_pipeline_is_explicit_and_non_inferential(tmp_path: Path) -> None:
    output_dir = tmp_path / "pipeline"
    code = nanjing_gt_pipeline.main(["--mode", "status", "--gt", str(tmp_path / "missing.gpkg"), "--output-dir", str(output_dir)])
    assert code == 0
    payload = json.loads((output_dir / "NANJING_GT_PIPELINE_STATUS.json").read_text(encoding="utf-8"))
    assert payload["status"] == "awaiting_manual_annotation"
    assert payload["training_started"] is False
    assert payload["inference_started"] is False
    assert payload["true_accuracy_reported"] is False
