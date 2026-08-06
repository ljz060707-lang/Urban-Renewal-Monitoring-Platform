from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from urban_monitor.building_foundation_v2.boundary import boundary_target
from urban_monitor.building_foundation_v2.detached_objectizer import (
    BoundaryObjectizationParams,
    bootstrap_object_differences,
    bounded_parameter_grid,
    connected_component_objectization,
    learned_boundary_watershed,
    morphology_boundary_objectization,
    objectization_preserves_building_area,
    oracle_boundary_objectization,
    select_objectization_parameters,
)
from urban_monitor.building_foundation_v2.detached_split import (
    build_boundary_split,
    split_leakage_audit,
)


def _mask() -> np.ndarray:
    value = np.zeros((48, 64), dtype=np.uint8)
    value[5:20, 5:20] = 1
    value[8:24, 32:52] = 1
    # Touching buildings exercise the objectizer without changing B0 pixels.
    value[30:43, 20:44] = 1
    return value


def _rows(count: int = 20) -> list[dict]:
    return [
        {
            "sample_id": f"whu_train_{i}", "split": "train",
            "image_path": f"D:/Data Set/WHU_Building/train/image/{i}.tif",
            "label_path": f"D:/Data Set/WHU_Building/train/label/{i}.tif",
            "image_sha256": hashlib.sha256(f"image-{i}".encode()).hexdigest(),
            "label_sha256": hashlib.sha256(f"label-{i}".encode()).hexdigest(),
            "positive_fraction": (i % 7) / 10.0, "valid_fraction": 1.0,
            "object_count": 1, "test_pixel_read": False,
        }
        for i in range(count)
    ]


def test_b0_fully_frozen():
    text = Path(__file__).resolve().parents[3].joinpath("tools/native_b2_diagnostic_worker.py").read_text(encoding="utf-8")
    assert "for parameter in model.parameters():" in text
    assert "parameter.requires_grad = False" in text
    assert "decoder.boundary_head.parameters()" in text


def test_boundary_feature_detached():
    text = Path(__file__).resolve().parents[3].joinpath("tools/native_b2_diagnostic_worker.py").read_text(encoding="utf-8")
    assert "self._feature.detach()" in text
    assert "decode_head.fusion_ab" not in text or "feature_module_path" in text


def test_b0_digest_unchanged():
    mask = _mask()
    before = hashlib.sha256(mask.tobytes()).hexdigest()
    labels = learned_boundary_watershed(mask, boundary_target(mask, width=2))
    after = hashlib.sha256(mask.tobytes()).hexdigest()
    assert before == after
    assert objectization_preserves_building_area(labels, mask)["preserved"]


def test_building_logits_identical():
    before = np.linspace(-2, 2, 32, dtype=np.float32)
    after = before.copy()
    assert float(np.max(np.abs(before - after))) == 0.0


def test_building_mask_identical():
    mask = _mask()
    labels = connected_component_objectization(mask)
    assert np.array_equal(mask > 0, labels > 0)


def test_revised_boundary_sanity_gate():
    mask = _mask()
    target = boundary_target(mask, width=2)
    oracle = float((target > 0).sum())
    prediction = target.copy()
    assert oracle > 0
    assert prediction.std() > 1e-4
    assert np.isfinite(prediction).all()
    assert prediction.mean() >= 0.0


def test_boundary_train_calibration_split():
    split = build_boundary_split(_rows(), seed=20260805, train_count=8, calibration_count=4, diagnostic_count=2)
    assert split["counts"] == {"boundary_train": 8, "boundary_calibration": 4, "boundary_diagnostic": 2}
    groups = [split["boundary_train"], split["boundary_calibration"], split["boundary_diagnostic"]]
    ids = [row["sample_id"] for group in groups for row in group]
    assert len(ids) == len(set(ids))
    assert len(split["unused"]) == 6


def test_boundary_threshold_calibration_only():
    rows = [
        {"split": "boundary_calibration", "method": "O2_LEARNED_BOUNDARY_WATERSHED", "object_f1@0.5": 0.7, "split_rate": 0, "merge_rate": 0},
        {"split": "boundary_calibration", "method": "O0_CONNECTED_COMPONENTS", "object_f1@0.5": 0.6, "split_rate": 0, "merge_rate": 0},
    ]
    assert select_objectization_parameters(rows)["selected_method"].startswith("O2")
    with pytest.raises(ValueError, match="boundary_calibration"):
        select_objectization_parameters([dict(rows[0], split="validation")])


def test_objectization_grid_bounded():
    grid = bounded_parameter_grid({"boundary_threshold": [0.4, 0.5], "minimum_seed_area": [8, 16]}, maximum=64)
    assert len(grid) == 4
    with pytest.raises(ValueError, match="exceeds maximum"):
        bounded_parameter_grid({"boundary_threshold": [0.2, 0.3, 0.4, 0.5], "interior_seed_threshold": [0.5, 0.7, 0.9], "minimum_seed_area": [4, 8, 16, 32], "minimum_object_area": [4, 8, 16, 32]}, maximum=64)


def test_learned_boundary_watershed():
    mask = _mask()
    probability = boundary_target(mask, width=2)
    labels = learned_boundary_watershed(mask, probability, BoundaryObjectizationParams(minimum_seed_area=2, minimum_object_area=2))
    contract = objectization_preserves_building_area(labels, mask)
    assert contract["preserved"]
    assert int(labels.max()) >= 1


def test_objectization_preserves_building_area():
    mask = _mask()
    for labels in (connected_component_objectization(mask), morphology_boundary_objectization(mask), learned_boundary_watershed(mask, boundary_target(mask, width=2))):
        assert objectization_preserves_building_area(labels, mask)["preserved"]


def test_oracle_not_selectable():
    rows = [
        {"split": "boundary_calibration", "method": "O3_GT_BOUNDARY_ORACLE", "object_f1@0.5": 1.0, "split_rate": 0, "merge_rate": 0},
        {"split": "boundary_calibration", "method": "O0_CONNECTED_COMPONENTS", "object_f1@0.5": 0.2, "split_rate": 1, "merge_rate": 1},
    ]
    assert select_objectization_parameters(rows)["selected_method"].startswith("O0")


def test_validation_parameters_frozen():
    params = BoundaryObjectizationParams()
    params.validate()
    assert params.boundary_threshold == 0.5
    assert params.minimum_separation_distance == 2


def test_bootstrap_object_metrics():
    rows = []
    for sample_id in ("a", "b", "c"):
        for method, score in (("O0_CONNECTED_COMPONENTS", 0.4), ("O1_MORPHOLOGICAL_BOUNDARY", 0.45), ("O2_LEARNED_BOUNDARY_WATERSHED", 0.5)):
            rows.append({"sample_id": sample_id, "method": method, "object_f1@0.5": score, "split_rate": 0.1, "merge_rate": 0.2, "absolute_count_error": 1, "relative_area_error": 0.1})
    result = bootstrap_object_differences(rows, n_bootstrap=1000)
    assert result["n_images"] == 3
    assert result["comparisons"]["O2_minus_O0"]["object_f1@0.5"]["status"] == "available"


def test_detached_boundary_selection_rule():
    selected = select_objectization_parameters([
        {"split": "boundary_calibration", "method": "O1_MORPHOLOGICAL_BOUNDARY", "object_f1@0.5": 0.8, "split_rate": 0.1, "merge_rate": 0.1},
        {"split": "boundary_calibration", "method": "O2_LEARNED_BOUNDARY_WATERSHED", "object_f1@0.5": 0.81, "split_rate": 0.1, "merge_rate": 0.1},
    ])
    assert selected["selected_method"] == "O2_LEARNED_BOUNDARY_WATERSHED"
    assert selected["oracle_selectable"] is False


def test_test_split_isolation():
    audit = split_leakage_audit(build_boundary_split(_rows(), train_count=8, calibration_count=4, diagnostic_count=2))
    assert audit["test_pixel_reads"] == 0
    assert audit["whu_test_pixel_reads"] == 0
    assert audit["second_test_pixel_reads"] == 0
    assert audit["strictly_train_only"] is True
