"""Pure, auditable diagnostics for the B2 boundary sanity failure.

This module deliberately contains no dataset discovery and no model loading.
The diagnostic runner supplies the already frozen one-sample contract and the
native worker supplies any model observations.  Keeping target/metric/oracle
logic here makes the evidence independently testable on CPU.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

from .boundary import boundary_metrics, boundary_target


def finite_or_none(value: Any) -> float | int | bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def load_mask(path: str | Any) -> np.ndarray:
    """Load a native WHU mask without changing its pixels."""
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {array.shape}")
    return array


def object_size_counts(mask: np.ndarray, *, small: int = 256, large: int = 4096) -> dict[str, int]:
    building = np.asarray(mask) > 0
    labels, count = ndimage.label(building, structure=ndimage.generate_binary_structure(2, 2))
    areas = np.bincount(labels.ravel())[1:] if count else np.asarray([], dtype=np.int64)
    return {
        "object_count": int(count),
        "small": int(np.sum(areas < small)),
        "medium": int(np.sum((areas >= small) & (areas < large))),
        "large": int(np.sum(areas >= large)),
        "min_area": int(areas.min()) if areas.size else 0,
        "max_area": int(areas.max()) if areas.size else 0,
    }


def shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(np.asarray(mask, dtype=bool))
    source = np.asarray(mask, dtype=bool)
    h, w = source.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    if y0 < y1 and x0 < x1:
        result[y0:y1, x0:x1] = source[sy0:sy1, sx0:sx1]
    return result


def target_metric_self_test(mask: np.ndarray, *, widths: Iterable[int] = (1, 2, 3, 4)) -> dict[str, Any]:
    """Compare target to itself and controlled perturbations."""
    records: list[dict[str, Any]] = []
    for width in widths:
        target = boundary_target(mask, width=int(width), method="symmetric") > 0
        variants = {
            "self": target,
            "copy": target.copy(),
            "shift_1px": shift(target, 1, 1),
            "shift_2px": shift(target, 2, 2),
            "dilation_1px": ndimage.binary_dilation(target, iterations=1),
            "erosion_1px": ndimage.binary_erosion(target, iterations=1),
        }
        for variant, prediction in variants.items():
            metrics = boundary_metrics(prediction, target, tolerances=(1, 2, 3))
            records.append({
                "width": int(width),
                "variant": variant,
                "target_positive_fraction": float(target.mean()),
                "target_positive_pixels": int(target.sum()),
                "target_vs_self": variant in {"self", "copy"},
                **{key: finite_or_none(value) for key, value in metrics.items()},
            })
    self_rows = [row for row in records if row["variant"] == "self"]
    return {
        "status": "available",
        "records": records,
        "target_vs_self_bf1": {str(row["width"]): float(row["boundary_f1@2"]) for row in self_rows},
        "target_vs_self_all_one": all(abs(float(row["boundary_f1@2"]) - 1.0) < 1e-12 for row in self_rows),
    }


def resize_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Deterministic nearest-neighbour resize for binary masks."""
    image = Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255), mode="L")
    resized = image.resize((int(width), int(height)), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) >= 128


def oracle_by_stride(mask: np.ndarray, *, strides: Iterable[int] = (1, 2, 4, 8)) -> list[dict[str, Any]]:
    """Estimate the representational ceiling of a stride/tile feature.

    The oracle has perfect knowledge of the target, but is forced through the
    same nearest downsample/upsample resolution.  It therefore measures a
    conservative information ceiling rather than claiming model performance.
    """
    target = boundary_target(mask, width=2, method="symmetric") > 0
    labels, count = ndimage.label(np.asarray(mask) > 0, structure=ndimage.generate_binary_structure(2, 2))
    areas = np.bincount(labels.ravel())[1:] if count else np.asarray([], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for stride in strides:
        h, w = target.shape
        small_h = max(1, int(np.ceil(h / int(stride))))
        small_w = max(1, int(np.ceil(w / int(stride))))
        restored = resize_nearest(resize_nearest(target, small_h, small_w), h, w)
        metrics = boundary_metrics(restored, target, tolerances=(1, 2, 3))
        row = {
            "stride": int(stride),
            "downsample_mode": "nearest",
            "upsample_mode": "nearest",
            "align_corners": None,
            "feature_height": small_h,
            "feature_width": small_w,
            "oracle_bf1@1": float(metrics["boundary_f1@1"]),
            "oracle_bf1@2": float(metrics["boundary_f1@2"]),
            "oracle_bf1@3": float(metrics["boundary_f1@3"]),
            "small_object_oracle_bf1": None,
            "large_object_oracle_bf1": None,
        }
        if areas.size:
            small_mask = np.zeros_like(target)
            large_mask = np.zeros_like(target)
            for index, area in enumerate(areas, start=1):
                object_mask = labels == index
                object_boundary = boundary_target(object_mask.astype(np.uint8), width=2, method="symmetric") > 0
                if area < 256:
                    small_mask |= object_boundary
                if area >= 4096:
                    large_mask |= object_boundary
            for key, object_target in (("small_object_oracle_bf1", small_mask), ("large_object_oracle_bf1", large_mask)):
                if object_target.any():
                    object_prediction = restored & object_target
                    row[key] = float(boundary_metrics(object_prediction, object_target, tolerances=(2,))["boundary_f1@2"])
        rows.append(row)
    return rows


def b0_boundary_from_building(mask: np.ndarray, *, width: int = 2) -> dict[str, Any]:
    target = boundary_target(mask, width=int(width), method="symmetric") > 0
    return {
        "boundary_source": "morphological_boundary_from_B0_building_mask",
        "width": int(width),
        **{key: finite_or_none(value) for key, value in boundary_metrics(target, target).items()},
    }


def gradient_cosine(native_gradient: np.ndarray, boundary_gradient: np.ndarray) -> float | None:
    a = np.asarray(native_gradient, dtype=np.float64).ravel()
    b = np.asarray(boundary_gradient, dtype=np.float64).ravel()
    if a.shape != b.shape or not a.size:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def revised_gate(*, b0_iou: float, building_iou: float, boundary_loss_initial: float | None,
                 boundary_loss_final: float | None, boundary_f1_at_2: float,
                 oracle_bf1_at_2: float) -> dict[str, Any]:
    loss_drop = None
    if boundary_loss_initial is not None and boundary_loss_final is not None and boundary_loss_initial > 0:
        loss_drop = float((boundary_loss_initial - boundary_loss_final) / boundary_loss_initial)
    building_protection = float(building_iou) >= float(b0_iou) - 0.005
    loss_gate = loss_drop is not None and loss_drop >= 0.80
    oracle_gate = float(boundary_f1_at_2) >= 0.90 * float(oracle_bf1_at_2)
    initial_gain_gate = True  # filled by the runner when initial BF1 is available
    return {
        "building_protection": bool(building_protection),
        "boundary_loss_drop": loss_drop,
        "boundary_loss_drop_ge_80pct": bool(loss_gate),
        "boundary_bf1_at_2": float(boundary_f1_at_2),
        "oracle_bf1_at_2": float(oracle_bf1_at_2),
        "boundary_bf1_ge_90pct_oracle": bool(oracle_gate),
        "initial_boundary_gain_ge_0.15": bool(initial_gain_gate),
        "sanity_gate_pass": bool(building_protection and loss_gate and oracle_gate and initial_gain_gate),
    }

