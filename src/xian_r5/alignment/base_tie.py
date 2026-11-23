"""Local image-to-image tie candidates on an already shared raster grid.

The module measures translation candidates from gradient correlation. It does
not resample images, warp geometries, interpolate a displacement field, or
accept a candidate as ground truth. ``dx_px``/``dy_px`` describe where features
in the moving image appear relative to the reference image; aligning the moving
image to the reference requires applying the opposite vector.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Compute scalar Pearson correlation without NumPy's covariance/BLAS path."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or left.size == 0:
        raise ValueError("Correlation inputs must be equal, non-empty vectors")
    left = left - np.mean(left)
    right = right - np.mean(right)
    left_energy = float(np.sum(left * left))
    right_energy = float(np.sum(right * right))
    denominator = math.sqrt(left_energy * right_energy)
    if denominator == 0.0:
        return None
    score = float(np.sum(left * right) / denominator)
    return score if math.isfinite(score) else None


def _numpy_gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    """Compute a Gaussian-smoothed Sobel magnitude without native CV bindings."""
    image = np.asarray(rgb, dtype=np.float32)
    gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]).astype(np.float32)

    # OpenCV's sigma=1, automatically sized Gaussian kernel is 7x7. Use the
    # same support and reflect-101 edge convention in this NumPy fallback.
    offsets = np.arange(-3, 4, dtype=np.float32)
    kernel = np.exp(-0.5 * offsets * offsets)
    kernel /= kernel.sum()
    blurred = gray
    for axis in (1, 0):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (3, 3)
        mode = "reflect" if blurred.shape[axis] > 3 else "edge"
        padded = np.pad(blurred, pad, mode=mode)
        filtered = np.zeros_like(blurred, dtype=np.float32)
        for index, weight in enumerate(kernel):
            if axis == 1:
                filtered += padded[:, index:index + blurred.shape[1]] * weight
            else:
                filtered += padded[index:index + blurred.shape[0], :] * weight
        blurred = filtered

    padded = np.pad(
        blurred, ((1, 1), (1, 1)),
        mode="reflect" if min(blurred.shape) > 1 else "edge",
    )
    top_left, top, top_right = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    bottom_left, bottom, bottom_right = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    grad_x = (top_right - top_left) + 2.0 * (right - left) + (bottom_right - bottom_left)
    grad_y = (bottom_left + 2.0 * bottom + bottom_right) - (top_left + 2.0 * top + top_right)
    return np.hypot(grad_x, grad_y).astype(np.float32, copy=False)


def image_shift_to_projected_correction(
    observed_dx_px: float,
    observed_dy_px: float,
    pixel_to_projected: Any,
) -> tuple[float, float]:
    """Convert moving-feature pixel offset into a moving-to-reference map shift.

    The pixel affine may be a six-value GDAL transform or nine-value matrix.
    The observed feature offset is negated before applying its linear part.
    """
    transform = tuple(float(value) for value in pixel_to_projected)
    if len(transform) not in (6, 9):
        raise ValueError("Pixel-to-projected affine transform must have 6 or 9 values")
    values = (observed_dx_px, observed_dy_px, *transform[:6])
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Image shifts and affine transform must be finite")
    a, b, _, d, e, _ = transform[:6]
    return (float(-a * observed_dx_px - b * observed_dy_px),
            float(-d * observed_dx_px - e * observed_dy_px))


def estimate_gradient_translation(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    radius: int = 5,
    min_samples: int = 100,
    moving_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return the best translation tie within ``radius`` pixels, for diagnosis.

    Inputs must already share one pixel grid. The boolean mask identifies
    candidate comparison pixels in the reference frame. For shifted/no-data
    rasters, callers should erode that mask by the search radius first so every
    tested displacement has complete support. Numerical peaks remain
    unreviewed candidates and are never applied here.
    """
    reference = np.asarray(reference_rgb)
    moving = np.asarray(moving_rgb)
    valid = np.asarray(valid_mask, dtype=bool)
    moving_valid = (None if moving_valid_mask is None else
                    np.asarray(moving_valid_mask, dtype=bool))
    if (reference.ndim != 3 or reference.shape[2] != 3 or
            moving.shape != reference.shape):
        raise ValueError("Reference and moving images must share an H×W×3 RGB grid")
    if valid.shape != reference.shape[:2]:
        raise ValueError("Validity mask must match the shared image grid")
    if moving_valid is not None and moving_valid.shape != valid.shape:
        raise ValueError("Moving-image validity mask must match the shared image grid")
    if radius < 1 or min_samples < 3:
        raise ValueError("Search radius and minimum sample count must be positive")
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        raise ValueError("Tie images must contain only finite values")

    try:
        import cv2
    except (ImportError, OSError):
        ref_edge = _numpy_gradient_magnitude(reference)
        mov_edge = _numpy_gradient_magnitude(moving)
        preprocessing_backend = "NUMPY_GAUSSIAN_SOBEL_FALLBACK"
    else:
        ref_gray = cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_RGB2GRAY)
        mov_gray = cv2.cvtColor(moving.astype(np.float32), cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.GaussianBlur(ref_gray, (0, 0), 1.0)
        mov_gray = cv2.GaussianBlur(mov_gray, (0, 0), 1.0)
        ref_x = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
        ref_y = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
        mov_x = cv2.Sobel(mov_gray, cv2.CV_32F, 1, 0, ksize=3)
        mov_y = cv2.Sobel(mov_gray, cv2.CV_32F, 0, 1, ksize=3)
        ref_edge = cv2.magnitude(ref_x, ref_y)
        mov_edge = cv2.magnitude(mov_x, mov_y)
        preprocessing_backend = "OPENCV_GAUSSIAN_SOBEL"

    candidates: list[tuple[float, int, int, int]] = []
    score_surface: dict[tuple[int, int], float] = {}
    height, width = valid.shape
    max_masked_samples = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            ref = ref_edge[y0:y1, x0:x1]
            mov = mov_edge[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            mask = valid[y0:y1, x0:x1] & (ref > np.percentile(ref, 20))
            if moving_valid is not None:
                mask &= moving_valid[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            masked_samples = int(mask.sum())
            max_masked_samples = max(max_masked_samples, masked_samples)
            if masked_samples < min_samples:
                continue
            a, b = ref[mask], mov[mask]
            score = _pearson_correlation(a, b)
            if score is None:
                continue
            candidates.append((score, dx, dy, int(mask.sum())))
            score_surface[(dx, dy)] = score

    candidates.sort(reverse=True)
    best = None
    if candidates:
        score, dx, dy, count = candidates[0]

        def parabolic_offset(
            f_minus: float | None, f_center: float, f_plus: float | None,
        ) -> tuple[float | None, float | None]:
            if f_minus is None or f_plus is None:
                return None, None
            curvature = f_minus - 2.0 * f_center + f_plus
            if curvature >= -1e-6:
                return None, float(curvature)
            offset = 0.5 * (f_minus - f_plus) / curvature
            if not math.isfinite(offset) or abs(offset) > 0.5:
                return None, float(curvature)
            return float(offset), float(curvature)

        offset_x, curvature_x = parabolic_offset(
            score_surface.get((dx - 1, dy)), score,
            score_surface.get((dx + 1, dy)),
        )
        offset_y, curvature_y = parabolic_offset(
            score_surface.get((dx, dy - 1)), score,
            score_surface.get((dx, dy + 1)),
        )
        subpixel = None
        if offset_x is not None and offset_y is not None:
            subpixel = {
                "dx_px": float(dx + offset_x),
                "dy_px": float(dy + offset_y),
                "axis_peak_curvature": [curvature_x, curvature_y],
                "status": "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED",
            }
        best = {"corr": score, "dx_px": dx, "dy_px": dy,
                "n": count, "subpixel_peak": subpixel}

    return {
        "status": "DIAGNOSTIC_ONLY",
        "method": "GRADIENT_NCC_LOCAL_TRANSLATION",
        "shift_convention": "MOVING_FEATURE_OFFSET_FROM_REFERENCE;ALIGN_MOVING_APPLY_NEGATIVE",
        "search_radius_px": radius,
        "search_radius_10m_pixels": radius,
        "minimum_samples_required": min_samples,
        "preprocessing_backend": preprocessing_backend,
        "maximum_masked_samples_available": max_masked_samples,
        "sample_support_limited": max_masked_samples < min_samples,
        "best": best,
        "runner_up": ({"corr": candidates[1][0], "dx_px": candidates[1][1],
                       "dy_px": candidates[1][2], "n": candidates[1][3]}
                      if len(candidates) > 1 else None),
        "candidate_count": len(candidates),
    }


def _erode_mask(mask: np.ndarray, width: int) -> np.ndarray:
    """Erode a support mask with a square kernel, treating the exterior as false."""
    if width < 0:
        raise ValueError("Mask erosion width cannot be negative")
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("Mask erosion requires a two-dimensional boolean mask")
    if width == 0:
        return source.copy()
    height, cols = source.shape
    padded = np.pad(source, width, mode="constant", constant_values=False)
    eroded = source.copy()
    for dy in range(2 * width + 1):
        for dx in range(2 * width + 1):
            eroded &= padded[dy:dy + height, dx:dx + cols]
    return eroded


def estimate_component_gradient_translation(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    reference_valid_mask: np.ndarray,
    moving_valid_mask: np.ndarray,
    component_support_mask: np.ndarray,
    *,
    radius: int = 1,
    min_samples: int = 20,
) -> dict[str, Any]:
    """Estimate one candidate tie using only one exact source-component support.

    The component mask is applied to both image epochs and eroded one output
    pixel to keep Sobel features away from seams between stitched source tiles.
    The returned vector remains a diagnostic candidate; no field is promoted
    or interpolated. ``base_pixel_window`` identifies the cropped tie support
    in the caller's reference grid.
    """
    reference = np.asarray(reference_rgb)
    moving = np.asarray(moving_rgb)
    reference_valid = np.asarray(reference_valid_mask, dtype=bool)
    moving_valid = np.asarray(moving_valid_mask, dtype=bool)
    component = np.asarray(component_support_mask, dtype=bool)
    if (reference.ndim != 3 or reference.shape[2] != 3
            or moving.shape != reference.shape):
        raise ValueError("Component tie images must share an H×W×3 RGB grid")
    if any(mask.shape != reference.shape[:2]
           for mask in (reference_valid, moving_valid, component)):
        raise ValueError("Component tie masks must share the image grid")
    if radius < 1 or min_samples < 3:
        raise ValueError("Component tie radius and sample minimum are invalid")
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        raise ValueError("Component tie images must contain only finite values")
    if not component.any():
        raise ValueError("Component support mask is empty")

    component_interior = _erode_mask(component, 1)
    ys, xs = np.where(component)
    padding = radius + 2
    height, width = component.shape
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    tie = estimate_gradient_translation(
        reference[y0:y1, x0:x1], moving[y0:y1, x0:x1],
        (reference_valid & component_interior)[y0:y1, x0:x1],
        radius=radius,
        min_samples=min_samples,
        moving_valid_mask=(moving_valid & component_interior)[y0:y1, x0:x1],
    )
    return {
        **tie,
        "base_pixel_window": [x0, y0, x1, y1],
        "component_support_pixel_count": int(component.sum()),
        "component_interior_pixel_count": int(component_interior.sum()),
        "component_local_tie_status": (
            "NUMERIC_COMPONENT_PEAK_AVAILABLE" if tie["best"] else
            "INSUFFICIENT_COMPONENT_GROUND_SAMPLES"
            if tie["sample_support_limited"] else "NO_COMPONENT_CORRELATION_SCORE"),
    }


def estimate_local_gradient_field(
    reference_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    grid_size: int,
    radius: int = 3,
    min_samples: int = 100,
    moving_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate independent cell-local ties; do not fill gaps or apply them."""
    reference = np.asarray(reference_rgb)
    moving = np.asarray(moving_rgb)
    valid = np.asarray(valid_mask, dtype=bool)
    moving_valid = (None if moving_valid_mask is None else
                    np.asarray(moving_valid_mask, dtype=bool))
    if (reference.ndim != 3 or reference.shape[2] != 3 or
            moving.shape != reference.shape or valid.shape != reference.shape[:2]):
        raise ValueError("Images and mask must share one RGB raster grid")
    if moving_valid is not None and moving_valid.shape != valid.shape:
        raise ValueError("Moving-image validity mask must match the shared image grid")
    if grid_size < 1 or radius < 1 or min_samples < 3:
        raise ValueError("grid_size, radius, and min_samples must be positive")

    height, width = valid.shape
    cells = []
    coarse_vectors = []
    subpixel_vectors = []
    for row in range(grid_size):
        y0, y1 = row * height // grid_size, (row + 1) * height // grid_size
        for col in range(grid_size):
            x0, x1 = col * width // grid_size, (col + 1) * width // grid_size
            cell_height, cell_width = y1 - y0, x1 - x0
            maximum_shift_overlap = max(0, cell_height - radius) * max(
                0, cell_width - radius)
            if maximum_shift_overlap < min_samples:
                raise ValueError(
                    "local tie cells cannot provide the minimum sample count at the "
                    f"configured search radius: {cell_width}x{cell_height} pixels, "
                    f"radius={radius}, maximum_overlap={maximum_shift_overlap}, "
                    f"minimum_samples={min_samples}"
                )
            tie = estimate_gradient_translation(
                reference[y0:y1, x0:x1], moving[y0:y1, x0:x1],
                valid[y0:y1, x0:x1], radius=radius, min_samples=min_samples,
                moving_valid_mask=(moving_valid[y0:y1, x0:x1]
                                   if moving_valid is not None else None),
            )
            best = tie["best"]
            cell = {
                "row": row, "col": col,
                "base_pixel_window": [x0, y0, x1, y1],
                "valid_pixel_count": int(valid[y0:y1, x0:x1].sum()),
                "maximum_masked_samples_available": tie["maximum_masked_samples_available"],
                "minimum_samples_required": tie["minimum_samples_required"],
                "sample_support_status": (
                    "NUMERIC_PEAK_AVAILABLE" if best else
                    "INSUFFICIENT_MASKED_SAMPLES" if tie["sample_support_limited"] else
                    "NO_VALID_CORRELATION_SCORE"),
                "best": best,
                "runner_up": tie["runner_up"],
                "peak_margin": (best["corr"] - tie["runner_up"]["corr"]
                                if best and tie["runner_up"] else None),
                "status": "COARSE_TIE_CANDIDATE_NOT_APPLIED" if best else "NO_VALID_TIE",
            }
            cells.append(cell)
            if best:
                coarse_vectors.append([best["dx_px"], best["dy_px"]])
                refined = best.get("subpixel_peak")
                if refined:
                    subpixel_vectors.append([refined["dx_px"], refined["dy_px"]])

    def summarize(vectors: list[list[float]]) -> dict[str, Any]:
        if not vectors:
            return {"supported_cell_count": 0, "median_candidate_shift_px": None,
                    "mad_dx_px": None, "mad_dy_px": None,
                    "median_cell_deviation_px": None}
        values = np.asarray(vectors, dtype=np.float64)
        center = np.median(values, axis=0)
        residual = np.linalg.norm(values - center, axis=1)
        return {
            "supported_cell_count": len(vectors),
            "median_candidate_shift_px": [float(center[0]), float(center[1])],
            "mad_dx_px": float(np.median(np.abs(values[:, 0] - center[0]))),
            "mad_dy_px": float(np.median(np.abs(values[:, 1] - center[1]))),
            "median_cell_deviation_px": float(np.median(residual)),
        }

    if coarse_vectors:
        coarse = np.asarray(coarse_vectors, dtype=np.float64)
        coarse_center = np.median(coarse, axis=0)
        coarse_residual = np.linalg.norm(coarse - coarse_center, axis=1)
        coarse_summary = {
            "median_coarse_shift_px": [float(coarse_center[0]), float(coarse_center[1])],
            "mad_dx_px": float(np.median(np.abs(coarse[:, 0] - coarse_center[0]))),
            "mad_dy_px": float(np.median(np.abs(coarse[:, 1] - coarse_center[1]))),
            "median_cell_deviation_px": float(np.median(coarse_residual)),
            "fraction_within_one_pixel_diagonal": float(np.mean(coarse_residual <= np.sqrt(2.0))),
        }
    else:
        coarse_summary = {"median_coarse_shift_px": None, "mad_dx_px": None,
                          "mad_dy_px": None, "median_cell_deviation_px": None,
                          "fraction_within_one_pixel_diagonal": None}

    subpixel_summary = summarize(subpixel_vectors)
    subpixel_summary["supported_subpixel_cell_count"] = subpixel_summary["supported_cell_count"]
    return {
        "status": "DIAGNOSTIC_ONLY_NOT_ACCEPTED",
        "method": "BLOCKED_GRADIENT_NCC",
        "shift_convention": "MOVING_FEATURE_OFFSET_FROM_REFERENCE;ALIGN_MOVING_APPLY_NEGATIVE",
        "grid_rows": grid_size, "grid_cols": grid_size,
        "search_radius_px": radius,
        "search_radius_10m_pixels": radius,
        "minimum_samples_required": min_samples,
        "supported_cell_count": len(coarse_vectors),
        "summary": coarse_summary,
        "subpixel_candidate_summary": subpixel_summary,
        "cells": cells,
        "field_interpolated": False,
        "field_applied": False,
    }
