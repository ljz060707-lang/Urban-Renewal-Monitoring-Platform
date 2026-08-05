"""Finite segmentation/object metrics for static Building Foundation v2."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .object_metrics import area_bias_metrics, object_detection_metrics


def binary_metrics(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    raw_probability = np.asarray(probability, dtype="float64")
    if raw_probability.size == 0:
        raise ValueError("probability array is empty")
    if not np.isfinite(raw_probability).all():
        raise ValueError("probability contains NaN or Inf")
    raw_probability = np.clip(raw_probability, 0.0, 1.0)
    p = raw_probability >= float(threshold)
    y = np.asarray(target, dtype="float64") > 0.5
    tp = int(np.logical_and(p, y).sum()); fp = int(np.logical_and(p, ~y).sum()); fn = int(np.logical_and(~p, y).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn); f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / max(1, tp + fp + fn),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "predicted_positive_fraction": float(p.mean()),
        "actual_positive_fraction": float(y.mean()),
        "probability_mean": float(raw_probability.mean()),
        "probability_std": float(raw_probability.std()),
        "probability_p01": float(np.quantile(raw_probability, 0.01)),
        "probability_p50": float(np.quantile(raw_probability, 0.50)),
        "probability_p99": float(np.quantile(raw_probability, 0.99)),
        "all_positive_sample": float(bool(p.all())),
        "all_negative_sample": float(bool((~p).all())),
        "predicted_object_count": int(ndimage.label(p)[0].max()),
        "target_object_count": int(ndimage.label(y)[0].max()),
        "object_count_bias": int(ndimage.label(p)[0].max()) - int(ndimage.label(y)[0].max()),
    }


def object_metrics(probability: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    p = np.asarray(probability) >= threshold
    y = np.asarray(target) > 0.5
    p_count = int(ndimage.label(p)[0].max()); y_count = int(ndimage.label(y)[0].max())
    matched = object_detection_metrics(p, y, thresholds=(threshold,))[f"iou_{float(threshold):.1f}"]
    area = area_bias_metrics(p, y)
    return {
        "predicted_object_count": p_count,
        "target_object_count": y_count,
        "object_count_bias": p_count - y_count,
        "object_precision": float(matched["precision"]),
        "object_recall": float(matched["recall"]),
        "object_f1": float(matched["f1"]),
        "matched_object_count": int(matched["matched_count"]),
        "unmatched_prediction_count": int(matched["unmatched_prediction_count"]),
        "unmatched_target_count": int(matched["unmatched_target_count"]),
        "split_count": int(matched["split_count"]),
        "merge_count": int(matched["merge_count"]),
        "area_bias": float(area["area_bias"]),
        "area_bias_fraction": float(area["area_bias_fraction"]),
    }


def evaluate_batch_outputs(outputs: dict[str, np.ndarray], target: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    result = binary_metrics(outputs["building_probability"], target, threshold)
    result.update(object_metrics(outputs["building_probability"], target, threshold))
    return result


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows: return {"status": "skipped_no_supervision", "sample_count": 0}
    keys = ["precision", "recall", "f1", "iou", "predicted_positive_fraction", "actual_positive_fraction", "probability_mean", "probability_std", "probability_p01", "probability_p50", "probability_p99"]
    summary: dict[str, float | int] = {"status": "available", "sample_count": len(rows)}
    for key in keys: summary[key] = float(np.mean([r[key] for r in rows]))
    for key in ("tp_pixels", "fp_pixels", "fn_pixels", "predicted_object_count", "target_object_count", "object_count_bias", "matched_object_count", "unmatched_prediction_count", "unmatched_target_count", "split_count", "merge_count"):
        values = [r[key] for r in rows if key in r]
        if values:
            summary[key] = int(sum(int(value) for value in values))
    for key in ("object_precision", "object_recall", "object_f1", "area_bias", "area_bias_fraction"):
        values = [r[key] for r in rows if key in r]
        if values:
            summary[key] = float(np.mean(values))
    summary["all_positive_sample_fraction"] = float(np.mean([r.get("all_positive_sample", 0.0) for r in rows]))
    summary["all_negative_sample_fraction"] = float(np.mean([r.get("all_negative_sample", 0.0) for r in rows]))
    summary.update(prediction_distribution_gate(summary))
    return summary


def prediction_distribution_gate(
    summary: dict[str, float | int],
    *,
    all_positive_threshold: float = 0.95,
    all_negative_threshold: float = 0.005,
    constant_std_threshold: float = 1e-6,
) -> dict[str, str | bool]:
    """Class-collapse gate shared by native recovery and project metrics.

    The thresholds are intentionally diagnostics, not performance claims.  A
    collapsed prediction makes a candidate ineligible even if a mislabeled
    mask can produce a deceptively high overlap score.
    """
    pred = float(summary.get("predicted_positive_fraction", float("nan")))
    prob_std = float(summary.get("probability_std", float("nan")))
    actual = float(summary.get("actual_positive_fraction", float("nan")))
    if not np.isfinite([pred, prob_std, actual]).all():
        return {"distribution_status": "invalid_nonfinite", "candidate_eligible": False}
    if actual >= 0.80:
        return {"distribution_status": "label_contract_suspect", "candidate_eligible": False}
    if pred >= all_positive_threshold:
        return {"distribution_status": "collapsed_all_positive", "candidate_eligible": False}
    if pred <= all_negative_threshold:
        return {"distribution_status": "collapsed_all_negative", "candidate_eligible": False}
    if prob_std <= constant_std_threshold:
        return {"distribution_status": "collapsed_constant_probability", "candidate_eligible": False}
    return {"distribution_status": "healthy_prediction_distribution", "candidate_eligible": True}
