"""Finite segmentation/object metrics for static Building Foundation v2."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    p = np.asarray(probability, dtype="float64") >= float(threshold)
    y = np.asarray(target, dtype="float64") > 0.5
    tp = int(np.logical_and(p, y).sum()); fp = int(np.logical_and(p, ~y).sum()); fn = int(np.logical_and(~p, y).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn); f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": tp / max(1, tp + fp + fn), "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn, "predicted_positive_fraction": float(p.mean()), "actual_positive_fraction": float(y.mean()), "predicted_object_count": int(ndimage.label(p)[0].max()), "target_object_count": int(ndimage.label(y)[0].max()), "object_count_bias": int(ndimage.label(p)[0].max()) - int(ndimage.label(y)[0].max())}


def object_metrics(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    p = ndimage.label(np.asarray(probability) >= threshold)[0]
    y = ndimage.label(np.asarray(target) > 0.5)[0]
    p_count = int(p.max()); y_count = int(y.max())
    return {"predicted_object_count": p_count, "target_object_count": y_count, "object_count_bias": p_count - y_count}


def evaluate_batch_outputs(outputs: dict[str, np.ndarray], target: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    result = binary_metrics(outputs["building_probability"], target, threshold)
    result.update(object_metrics(outputs["building_probability"], target, threshold))
    return result


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows: return {"status": "skipped_no_supervision", "sample_count": 0}
    keys = ["precision", "recall", "f1", "iou", "predicted_positive_fraction", "actual_positive_fraction"]
    summary: dict[str, float | int] = {"status": "available", "sample_count": len(rows)}
    for key in keys: summary[key] = float(np.mean([r[key] for r in rows]))
    for key in ("tp_pixels", "fp_pixels", "fn_pixels", "predicted_object_count", "target_object_count", "object_count_bias"):
        summary[key] = int(sum(int(r[key]) for r in rows))
    return summary
