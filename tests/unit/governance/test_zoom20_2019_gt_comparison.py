"""Regression tests for the frozen-model Zoom20/2019-GT acceptance run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOLS = PROJECT_ROOT / "tools"
RUN_ROOT = PROJECT_ROOT / "outputs/inference/nanjing_jiangbei_zoom20_2019_gt_20260804_211042"
ZOOM20_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20")


def _load_tool():
    path = TOOLS / "run_nanjing_zoom20_2019_gt_comparison.py"
    spec = importlib.util.spec_from_file_location("zoom20_2019_gt_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_date_and_claimed_zoom_parsing() -> None:
    tool = _load_tool()
    assert tool.parse_date("nj_jiangbei190316_zoom_20.tif") == "2019-03-16"
    assert tool.parse_date("scene_2022-12-02_zoom20.tif") == "2022-12-02"
    assert tool.claimed_zoom(Path("scene_zoom_20.tif")) == "zoom20"
    assert tool.claimed_zoom(Path("scene_z20.tif")) == "zoom20"


def test_positive_object_matching_is_deterministic() -> None:
    from shapely.geometry import box

    tool = _load_tool()
    gt = [{"id": "gt1", "geometry": box(0, 0, 10, 10), "area_m2": 100.0}]
    pred = [{"id": "pred1", "geometry": box(1, 1, 9, 9), "area_m2": 64.0}]
    first = tool._object_score(gt, pred, 0.5)
    second = tool._object_score(gt, pred, 0.5)
    assert first["tp"] == 1
    assert first["f1"] == second["f1"]
    assert first["matches"][0]["iou"] > 0.5


def test_real_gt_inventory_and_positive_only_gate() -> None:
    pyogrio = pytest.importorskip("pyogrio")
    building = ZOOM20_ROOT / "gt_building19.gpkg"
    construction = ZOOM20_ROOT / "gt_construction19.gpkg"
    assert building.exists() and construction.exists()
    b_info = pyogrio.read_info(building, layer="gt_building")
    c_info = pyogrio.read_info(construction, layer="gt_construction")
    assert b_info["features"] == 288
    assert c_info["features"] == 21
    assert list(b_info["fields"]) == ["id"]
    assert list(c_info["fields"]) == ["id"]
    assert str(b_info["crs"]) == "EPSG:4326"
    assert str(c_info["crs"]) == "EPSG:4326"
    readiness = json.loads((RUN_ROOT / "05_gt_evaluation/GT_READINESS.json").read_text(encoding="utf-8"))
    assert readiness["status"] == "ready_for_positive_object_match_only"
    assert readiness["background_class_available"] is False
    assert readiness["pixel_precision_recall_f1_iou_allowed"] is False


def test_gt_deferred_until_inference_seal_and_flags() -> None:
    preseal = json.loads((RUN_ROOT / "03_inference/PRE_GT_INFERENCE_SEAL.json").read_text(encoding="utf-8"))
    assert preseal["status"] == "sealed"
    assert preseal["gt_content_read"] is False
    assert preseal["gt_access_before_inference_completed"] is False
    evaluation = json.loads((RUN_ROOT / "05_gt_evaluation/2019_GT_EVALUATION.json").read_text(encoding="utf-8"))
    assert evaluation["training_started"] is False
    assert evaluation["models_reselected"] is False
    assert evaluation["thresholds_changed"] is False
    assert evaluation["gt_used_for_training"] is False
    assert evaluation["gt_used_for_model_selection"] is False
    assert evaluation["locked_test_pixel_reads"] == 0
    assert evaluation["original_source_modified"] is False


def test_r2_construction_uses_model_scale_grid() -> None:
    rasterio = pytest.importorskip("rasterio")
    native = RUN_ROOT / "03_inference/zoom20_native/06_construction_single_date/2019-03-16/construction_candidate_mask.tif"
    normalized = RUN_ROOT / "03_inference/zoom20_scale_normalized/construction/06_construction_single_date/2019-03-16/construction_candidate_mask.tif"
    with rasterio.open(native) as native_ds, rasterio.open(normalized) as normalized_ds:
        assert (native_ds.width, native_ds.height) == (7424, 6656)
        assert (normalized_ds.width, normalized_ds.height) == (3840, 3328)


def test_baseline_seal_and_highres_outputs() -> None:
    baseline = json.loads((RUN_ROOT / "11_manifests/BASELINE_HASH_AUDIT.json").read_text(encoding="utf-8"))
    assert baseline["baseline_hash_match_count"] == 252
    assert baseline["baseline_hash_total_count"] == 252
    assert baseline["baseline_hash_invariance_passed"] is True
    from PIL import Image

    figures = sorted((RUN_ROOT / "08_figures_highres").glob("*.png"))
    assert figures
    for path in figures:
        width, height = Image.open(path).size
        assert width >= 3600 and height >= 2100, path


def test_registry_hash_is_frozen() -> None:
    registry = PROJECT_ROOT / "outputs/governance/current_best_gt_ready_20260804_173745/01_model_registry/CURRENT_BEST_MODEL_REGISTRY.json"
    digest = hashlib.sha256(registry.read_bytes()).hexdigest()
    assert digest == "48b651e2e0288a8b2bcb73b8034070294064b935ae22ec19b44b2cf65614e8c9"
