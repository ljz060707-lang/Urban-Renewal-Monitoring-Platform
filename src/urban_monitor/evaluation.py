from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from .geometry import polygon_iou


def _match_objects(predicted: gpd.GeoDataFrame, truth: gpd.GeoDataFrame, threshold: float) -> dict[str, float]:
    if predicted.crs != truth.crs:
        predicted = predicted.to_crs(truth.crs)
    candidates = sorted(
        (
            (polygon_iou(predicted.iloc[i].geometry, truth.iloc[j].geometry), i, j)
            for i in range(len(predicted))
            for j in range(len(truth))
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matched_ious: list[float] = []
    for iou, i, j in candidates:
        if iou < threshold or i in used_pred or j in used_truth:
            continue
        used_pred.add(i)
        used_truth.add(j)
        matched_ious.append(iou)
    tp = len(matched_ious)
    fp = len(predicted) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_area = float(predicted.geometry.area.sum()) if not predicted.empty else 0.0
    truth_area = float(truth.geometry.area.sum()) if not truth.empty else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "dice": float(np.mean([2 * value / (1 + value) for value in matched_ious])) if matched_ious else 0.0,
        "area_error_m2": pred_area - truth_area,
        "object_count_error": len(predicted) - len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_phase1(
    predictions_by_date: dict[str, str | Path],
    truth_by_date: dict[str, str | Path],
    output_dir: str | Path,
    *,
    iou_threshold: float = 0.5,
    holdout_name: str = "synthetic_spatial_holdout",
) -> dict[str, object]:
    """Evaluate available object layers and mark unavailable tasks not computable."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for date_value in sorted(set(predictions_by_date) & set(truth_by_date)):
        prediction_path = Path(predictions_by_date[date_value])
        truth_path = Path(truth_by_date[date_value])
        if not prediction_path.exists() or not truth_path.exists():
            continue
        metrics = _match_objects(gpd.read_file(prediction_path), gpd.read_file(truth_path), iou_threshold)
        rows.append({"date": date_value, "holdout": holdout_name, **metrics})
    object_metrics = pd.DataFrame(rows)
    if object_metrics.empty:
        segmentation: object = "not computable"
    else:
        segmentation = {
            key: float(object_metrics[key].mean())
            for key in ["precision", "recall", "f1", "iou", "dice", "area_error_m2", "object_count_error"]
        }
    summary = {
        "segmentation": segmentation,
        "tracking": "not computable: dedicated tracking Ground Truth not supplied",
        "events": "not computable: reviewed event Ground Truth not supplied",
        "grid": "not computable: independent student-grid Ground Truth not supplied",
        "holdout": holdout_name,
    }
    object_metrics.to_csv(output / "object_metrics.csv", index=False, encoding="utf-8-sig")
    for name in ["tracking_metrics.csv", "event_metrics.csv", "grid_metrics.csv"]:
        pd.DataFrame([{"status": "not computable"}]).to_csv(output / name, index=False, encoding="utf-8-sig")
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Phase 1 evaluation", "", f"Spatial holdout: `{holdout_name}`", ""]
    for key, value in summary.items():
        if key == "holdout":
            continue
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    (output / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
