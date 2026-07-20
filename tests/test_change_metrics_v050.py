from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from urban_monitor.change.metrics import ChangeMetricAccumulator
from urban_monitor.change.visualization import build_training_visualizations


def test_metrics_report_binary_directional_and_semantic():
    target = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    probability = np.array([[0.1, 0.9], [0.2, 0.8]])
    accumulator = ChangeMetricAccumulator(semantic_classes=2, building_class_id=1)
    accumulator.update(binary_probability=probability, binary_target=target,
                       new_probability=probability, new_target=target,
                       demolition_probability=1-probability, demolition_target=1-target,
                       semantic_t1=target, semantic_t2=target, semantic_target_t1=target, semantic_target_t2=target)
    result = accumulator.result()
    assert result["binary"]["f1"] == 1
    assert result["directional"]["new"]["iou"] == 1
    assert result["semantic"]["mean_iou"] == 1


def test_missing_supervision_is_unavailable_not_fake_zero():
    accumulator = ChangeMetricAccumulator(semantic_classes=2)
    accumulator.update(binary_probability=np.zeros((2,2)), binary_target=np.zeros((2,2)))
    result = accumulator.result()
    assert result["directional"]["new"]["available"] is False
    assert result["directional"]["new"]["f1"] is None
    assert result["semantic"]["available"] is False
    assert result["semantic"]["mean_iou"] is None


def test_static_training_dashboard_and_required_charts(tmp_path: Path):
    metrics = tmp_path / "metrics"; metrics.mkdir()
    pd.DataFrame([{"epoch": 0, "train_loss": 1.0, "val_loss": 0.9, "val_binary_f1": 0.4, "learning_rate": 0.001}]).to_csv(metrics / "train_log_binary.csv", index=False)
    (metrics / "combined.json").write_text(json.dumps({"binary": {"f1": 0.4, "iou": 0.25}}), encoding="utf-8")
    path = build_training_visualizations(tmp_path)
    assert path.is_file() and "静态离线页面" in path.read_text(encoding="utf-8")
    for name in ("loss_curves.png", "learning_rate_curve.png", "binary_metrics_curve.png", "gpu_memory_curve.png", "threshold_sweep.png"):
        assert (tmp_path / "visualizations" / name).is_file()
