"""Boundary targets, losses and metrics for the native B2 ablation.

The implementation is deliberately small and array-first.  It never discovers
data on its own, which makes it safe to use with the explicit WHU Train and
Validation registries used by the B2 runner.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage


def _binary(mask: np.ndarray, ignore_value: int = 255) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"building mask must be 2-D, got {arr.shape}")
    valid = arr != ignore_value
    return (arr > 0) & valid, valid


def _structure(connectivity: int = 8) -> np.ndarray:
    return ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)


def boundary_target(
    mask: np.ndarray,
    width: int = 2,
    *,
    method: str = "symmetric",
    ignore_value: int = 255,
    connectivity: int = 8,
    return_valid: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Create a deterministic binary boundary target.

    ``inner`` is the in-object boundary.  ``symmetric`` is the union of an
    in-object and an out-of-object band, which is useful for separating
    touching buildings.  Ignore pixels are never turned into background or
    boundary supervision.
    """
    if int(width) < 1:
        raise ValueError("boundary width must be >= 1")
    if method not in {"inner", "symmetric"}:
        raise ValueError("method must be 'inner' or 'symmetric'")
    building, valid = _binary(mask, ignore_value)
    st = _structure(connectivity)
    eroded = ndimage.binary_erosion(building, structure=st, iterations=int(width), border_value=0)
    inner = building & ~eroded
    if method == "inner":
        result = inner
    else:
        dilated = ndimage.binary_dilation(building, structure=st, iterations=int(width), border_value=0)
        result = dilated & ~eroded
    result &= valid
    result = result.astype(np.float32, copy=False)
    return (result, valid.astype(bool, copy=False)) if return_valid else result


def boundary_object_stats(mask: np.ndarray, width: int = 2, *, ignore_value: int = 255) -> dict[str, float | int]:
    """Return audit statistics for a boundary target without selecting data."""
    building, valid = _binary(mask, ignore_value)
    labels, count = ndimage.label(building, structure=_structure())
    boundary = boundary_target(mask, width, ignore_value=ignore_value)
    object_areas: list[int] = []
    boundary_areas: list[int] = []
    for label in range(1, int(count) + 1):
        obj = labels == label
        object_areas.append(int(obj.sum()))
        boundary_areas.append(int(np.logical_and(obj, boundary > 0).sum()))
    small = [i for i, area in enumerate(object_areas) if area < 256]
    large = [i for i, area in enumerate(object_areas) if area >= 4096]
    return {
        "object_count": int(count),
        "boundary_positive_fraction": float(boundary[valid].mean()) if valid.any() else 0.0,
        "boundary_pixels_per_object": float(boundary.sum() / max(1, count)),
        "small_object_boundary_fraction": float(sum(boundary_areas[i] for i in small) / max(1, sum(object_areas[i] for i in small))),
        "large_object_boundary_fraction": float(sum(boundary_areas[i] for i in large) / max(1, sum(object_areas[i] for i in large))),
        "objects_fully_consumed_by_boundary": int(sum(boundary_areas[i] >= object_areas[i] for i in range(len(object_areas)))),
        "valid_pixel_fraction": float(valid.mean()),
    }


def boundary_loss(
    logits,
    target,
    *,
    valid_mask=None,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
):
    """Boundary BCE + Dice with explicit ignore support."""
    import torch
    import torch.nn.functional as F

    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.to(dtype=logits.dtype, device=logits.device)
    if valid_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=logits.device).bool()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits[valid], target[valid], reduction="mean")
    probability = torch.sigmoid(logits)
    p = probability[valid]
    y = target[valid]
    dice = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)
    value = float(bce_weight) * bce + float(dice_weight) * dice
    if not torch.isfinite(value):
        raise FloatingPointError("boundary loss is NaN/Inf")
    return value


def _match_with_tolerance(pred: np.ndarray, truth: np.ndarray, tolerance: int) -> tuple[int, int, int]:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError(f"boundary shapes differ: {p.shape} != {t.shape}")
    st = _structure()
    p_near = ndimage.binary_dilation(p, structure=st, iterations=int(tolerance))
    t_near = ndimage.binary_dilation(t, structure=st, iterations=int(tolerance))
    tp_p = int(np.logical_and(p, t_near).sum())
    tp_t = int(np.logical_and(t, p_near).sum())
    # A pixel may be used at most once in each side's count.  Precision and
    # recall use the matched pixel counts, not an unbounded pair count.
    tp = min(tp_p, tp_t)
    return tp, int(p.sum()), int(t.sum())


def boundary_metrics(prediction: np.ndarray, target: np.ndarray, *, tolerances: Sequence[int] = (1, 2, 3)) -> dict[str, float | int]:
    p = np.asarray(prediction, dtype=bool)
    t = np.asarray(target, dtype=bool)
    if p.shape != t.shape:
        raise ValueError(f"boundary shapes differ: {p.shape} != {t.shape}")
    result: dict[str, float | int] = {}
    for tolerance in tolerances:
        if int(tolerance) < 0:
            raise ValueError("boundary tolerance must be >= 0")
        matched, p_count, t_count = _match_with_tolerance(p, t, int(tolerance))
        precision = matched / max(1, p_count)
        recall = matched / max(1, t_count)
        result[f"boundary_precision@{int(tolerance)}"] = precision
        result[f"boundary_recall@{int(tolerance)}"] = recall
        result[f"boundary_f1@{int(tolerance)}"] = 2.0 * precision * recall / max(1e-12, precision + recall)
        result[f"boundary_tp@{int(tolerance)}"] = matched
    if p.any() and t.any():
        result["distance_defined"] = True
        dist_to_t = ndimage.distance_transform_edt(~t)
        dist_to_p = ndimage.distance_transform_edt(~p)
        distances = np.concatenate([dist_to_t[p], dist_to_p[t]])
        result["mean_boundary_distance"] = float(np.mean(distances))
        result["p90_boundary_distance"] = float(np.quantile(distances, 0.90))
        result["hausdorff95"] = float(max(np.quantile(dist_to_t[p], 0.95), np.quantile(dist_to_p[t], 0.95)))
    else:
        # Empty-side distances are undefined, but receipts and JSON metrics
        # must remain finite.  Preserve the distinction explicitly instead of
        # leaking ``inf``/``NaN`` into candidate selection or reports.  When
        # only one side is present, the image diagonal is a conservative finite
        # upper bound; when both are empty the distance is exactly zero.
        result["distance_defined"] = False
        if not p.any() and not t.any():
            empty_distance = 0.0
        else:
            empty_distance = float(np.hypot(*p.shape))
        result["mean_boundary_distance"] = empty_distance
        result["p90_boundary_distance"] = empty_distance
        result["hausdorff95"] = empty_distance
    p_components = int(ndimage.label(p, structure=_structure())[1])
    t_components = int(ndimage.label(t, structure=_structure())[1])
    result["boundary_fragmentation"] = float(p_components / max(1, t_components))
    result["predicted_boundary_components"] = p_components
    result["target_boundary_components"] = t_components
    return result


def bootstrap_ci(values: Iterable[float], *, seed: int = 20260805, n_bootstrap: int = 1000, alpha: float = 0.05) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"status": "unavailable", "n": 0}
    rng = np.random.default_rng(int(seed))
    if arr.size == 1:
        return {"status": "available", "n": 1, "mean": float(arr[0]), "lower": float(arr[0]), "upper": float(arr[0]), "n_bootstrap": 0}
    sample = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True).mean(axis=1)
    return {"status": "available", "n": int(arr.size), "mean": float(arr.mean()), "lower": float(np.quantile(sample, alpha / 2)), "upper": float(np.quantile(sample, 1 - alpha / 2)), "n_bootstrap": int(n_bootstrap)}
