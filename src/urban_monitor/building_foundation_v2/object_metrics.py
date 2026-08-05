"""Auditable connected-component matching for Building Foundation v2.

The native RSBuilding output is a binary building mask.  Pixel overlap alone
cannot reveal split/merge errors, so this module performs deterministic
one-to-one component matching and reports the unmatched topology separately.
It intentionally accepts arrays rather than a model or a dataset, which keeps
the metric code usable for validation fixtures without ever discovering Test
files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Component:
    label: int
    area: int
    pixels: np.ndarray
    # (row_start, row_stop, col_start, col_stop) in the source raster.  The
    # mask stored in ``pixels`` is cropped to this box, which lets matching
    # reject non-overlapping components before touching raster arrays.
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[Component]:
    value = np.asarray(mask).astype(bool)
    if value.ndim != 2:
        raise ValueError(f"object masks must be 2-D, got {value.shape}")
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, count = ndimage.label(value, structure=structure)
    result: list[Component] = []
    objects = ndimage.find_objects(labels)
    for label in range(1, int(count) + 1):
        location = objects[label - 1]
        if location is None:
            continue
        r_slice, c_slice = location
        pixels = labels[location] == label
        bbox = (int(r_slice.start), int(r_slice.stop), int(c_slice.start), int(c_slice.stop))
        result.append(Component(label, int(pixels.sum()), pixels, bbox))
    return result


def _iou_matrix(pred: list[Component], target: list[Component]) -> np.ndarray:
    matrix = np.zeros((len(pred), len(target)), dtype=np.float64)
    for i, p in enumerate(pred):
        pr0, pr1, pc0, pc1 = p.bbox
        for j, t in enumerate(target):
            tr0, tr1, tc0, tc1 = t.bbox
            r0, r1 = max(pr0, tr0), min(pr1, tr1)
            c0, c1 = max(pc0, tc0), min(pc1, tc1)
            if r0 >= r1 or c0 >= c1:
                continue
            p_view = p.pixels[r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0]
            t_view = t.pixels[r0 - tr0:r1 - tr0, c0 - tc0:c1 - tc0]
            intersection = int(np.logical_and(p_view, t_view).sum())
            if intersection:
                union = p.area + t.area - intersection
                matrix[i, j] = intersection / max(1, union)
    return matrix


def match_objects(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.5,
    connectivity: int = 8,
) -> dict[str, Any]:
    """Match connected components with maximum-IoU one-to-one assignment.

    Hungarian assignment is performed on ``1 - IoU`` and pairs below the
    requested threshold are discarded.  Ties are deterministic because the
    component labels are generated in scan order.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("object IoU threshold must be in [0, 1]")
    pred = _components(prediction, connectivity)
    truth = _components(target, connectivity)
    matrix = _iou_matrix(pred, truth)
    matches: list[dict[str, Any]] = []
    if matrix.size:
        rows, cols = linear_sum_assignment(1.0 - matrix)
        for row, col in zip(rows.tolist(), cols.tolist()):
            score = float(matrix[row, col])
            if score >= float(threshold):
                matches.append({
                    "prediction_label": pred[row].label,
                    "target_label": truth[col].label,
                    "iou": score,
                    "prediction_area": pred[row].area,
                    "target_area": truth[col].area,
                })
    matched_pred = {int(item["prediction_label"]) for item in matches}
    matched_target = {int(item["target_label"]) for item in matches}
    # A merge is one predicted component touching multiple GT components at
    # the operating IoU threshold; a split is the symmetric event.
    overlap = matrix >= float(threshold)
    merge_events = sum(int(row.sum() > 1) for row in overlap)
    split_events = sum(int(overlap[:, col].sum() > 1) for col in range(overlap.shape[1])) if overlap.ndim == 2 else 0
    pred_area = int(sum(item.area for item in pred))
    target_area = int(sum(item.area for item in truth))
    matched = len(matches)
    return {
        "threshold": float(threshold),
        "prediction_count": len(pred),
        "target_count": len(truth),
        "matched_count": matched,
        "unmatched_prediction_count": len(pred) - len(matched_pred),
        "unmatched_target_count": len(truth) - len(matched_target),
        "precision": matched / max(1, len(pred)),
        "recall": matched / max(1, len(truth)),
        "f1": (2.0 * matched) / max(1, len(pred) + len(truth)),
        "merge_count": int(merge_events),
        "split_count": int(split_events),
        "area_prediction": pred_area,
        "area_target": target_area,
        "area_bias": pred_area - target_area,
        "area_bias_fraction": (pred_area - target_area) / max(1, target_area),
        "matches": matches,
        "iou_matrix": matrix.tolist(),
    }


def object_detection_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
    connectivity: int = 8,
) -> dict[str, Any]:
    """Return object metrics at all requested IoU thresholds."""
    return {
        f"iou_{float(threshold):.1f}": match_objects(
            prediction, target, threshold=float(threshold), connectivity=connectivity
        )
        for threshold in thresholds
    }


def size_stratified_recall(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    area_thresholds: tuple[float, float] = (256.0, 4096.0),
    iou_threshold: float = 0.5,
    connectivity: int = 8,
) -> dict[str, Any]:
    """Report recall for fixed small/medium/large GT object area bands."""
    truth = _components(target, connectivity)
    pred = _components(prediction, connectivity)
    matrix = _iou_matrix(pred, truth)
    matched_truth: set[int] = set()
    if matrix.size:
        rows, cols = linear_sum_assignment(1.0 - matrix)
        matched_truth = {
            int(col) for row, col in zip(rows.tolist(), cols.tolist())
            if matrix[row, col] >= float(iou_threshold)
        }
    low, high = map(float, area_thresholds)
    bands: dict[str, list[int]] = {"small": [], "medium": [], "large": []}
    for index, component in enumerate(truth):
        name = "small" if component.area < low else "medium" if component.area < high else "large"
        bands[name].append(index)
    output: dict[str, Any] = {"area_thresholds": [low, high], "iou_threshold": float(iou_threshold)}
    for name, indexes in bands.items():
        output[name] = {
            "target_count": len(indexes),
            "matched_count": sum(index in matched_truth for index in indexes),
            "recall": sum(index in matched_truth for index in indexes) / max(1, len(indexes)),
        }
    return output


def area_bias_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Pixel area/count bias independent of a particular object IoU cutoff."""
    pred_count = len(_components(prediction))
    target_count = len(_components(target))
    pred_area = int(np.asarray(prediction).astype(bool).sum())
    target_area = int(np.asarray(target).astype(bool).sum())
    return {
        "predicted_area": float(pred_area),
        "target_area": float(target_area),
        "area_bias": float(pred_area - target_area),
        "area_bias_fraction": float((pred_area - target_area) / max(1, target_area)),
        "predicted_count": float(pred_count),
        "target_count": float(target_count),
        "count_bias": float(pred_count - target_count),
    }
