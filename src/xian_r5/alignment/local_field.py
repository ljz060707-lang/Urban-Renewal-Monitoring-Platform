"""Estimate read-only, cell-local displacement candidates from masked ties.

The module reports a candidate correction from the later image into the
earlier image. It never edits an image or geometry, and even strong numerical
support remains unreviewed until the eligible ground mask is independently
accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Iterable, Mapping

import numpy as np


LOCAL_TRANSFORM_MODEL = "SIMILARITY_ESTIMATE_AFFINE_PARTIAL_2D"
LOCAL_FIELD_METHOD = "ALIKED_LIGHTGLUE_LOCAL_SIMILARITY_RANSAC_CANDIDATE"


def candidate_cycle_diagnostics(
    ordered_dates: Iterable[str],
    pairwise_fields: Mapping[tuple[str, str], Mapping[str, Mapping[str, object]]],
) -> list[dict]:
    """Report arithmetic closure for numeric local candidates across three dates.

    Weak and unreviewed vectors are retained for diagnosis, but closure is never
    treated as validation, a ground-control observation, or permission to apply
    a displacement field. Only cell IDs present in all three pairwise fields
    are compared.
    """
    dates = tuple(ordered_dates)
    if len(dates) != 3:
        return []

    pair_keys = ((dates[0], dates[1]), (dates[1], dates[2]), (dates[0], dates[2]))
    pair_maps = [pairwise_fields.get(key, {}) for key in pair_keys]
    common_cells = set(pair_maps[0]) & set(pair_maps[1]) & set(pair_maps[2])
    output = []
    for cell_id in sorted(common_cells):
        records = [pair_map[cell_id] for pair_map in pair_maps]
        vectors: list[tuple[float, float] | None] = []
        pairwise_evidence = []
        for pair_key, row in zip(pair_keys, records):
            try:
                dx = float(row.get("correction_dx_m_candidate"))
                dy = float(row.get("correction_dy_m_candidate"))
            except (TypeError, ValueError):
                vector = None
            else:
                vector = (dx, dy) if math.isfinite(dx) and math.isfinite(dy) else None
            vectors.append(vector)
            pairwise_evidence.append({
                "from_date": pair_key[0],
                "to_date": pair_key[1],
                "status": row.get("status"),
                "correction_m_candidate": list(vector) if vector is not None else None,
                "ground_mask_reviewed": row.get("ground_mask_reviewed") is True,
            })

        closure = None
        closure_norm = None
        if all(vector is not None for vector in vectors):
            first, second, direct = vectors
            closure = [first[0] + second[0] - direct[0],
                       first[1] + second[1] - direct[1]]
            closure_norm = math.hypot(*closure)
        output.append({
            "cell_id": cell_id,
            "pairwise_evidence": pairwise_evidence,
            "cycle_closure_m_candidate": closure,
            "cycle_closure_magnitude_candidate": closure_norm,
            "status": ("ARITHMETIC_ONLY_NOT_VALIDATION" if closure is not None
                       else "PAIRWISE_NUMERIC_CANDIDATE_INCOMPLETE"),
            "ground_position_validated": False,
            "field_applied": False,
        })
    return output


@dataclass(frozen=True)
class LocalTileShift:
    status: str
    eligible_matches: int
    inliers: int
    inlier_ratio: float
    coverage_4x4: float
    median_residual_px: float | None
    p90_residual_px: float | None
    observed_dx_px: float | None
    observed_dy_px: float | None
    correction_dx_m_candidate: float | None
    correction_dy_m_candidate: float | None
    ground_mask_reviewed: bool = False
    field_interpolated: bool = False
    applied_to_image_or_geometry: bool = False
    transform_model: str = LOCAL_TRANSFORM_MODEL
    ransac_seed: int | None = None
    inlier_points_from_px: tuple[tuple[float, float], ...] = field(default=(), repr=False, compare=False)
    inlier_points_to_px: tuple[tuple[float, float], ...] = field(default=(), repr=False, compare=False)
    median_residual_m: float | None = None
    p90_residual_m: float | None = None

    def as_dict(self) -> dict:
        values = asdict(self)
        values.pop("inlier_points_from_px")
        values.pop("inlier_points_to_px")
        return values


def _eligible_points(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Correspondence points must have shape (n, 2)")
    height, width = mask.shape
    xy = np.rint(points).astype(np.int64)
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] < width) &
              (xy[:, 1] >= 0) & (xy[:, 1] < height))
    eligible = np.zeros(len(points), dtype=bool)
    indices = np.flatnonzero(inside)
    eligible[indices] = mask[xy[indices, 1], xy[indices, 0]] != 0
    return eligible


def _residuals_in_projected_units(
    residual_delta_px: np.ndarray, pixel_to_projected: tuple[float, ...],
) -> np.ndarray:
    """Convert pixel residual vectors through the affine linear component."""
    linear = np.asarray(
        [[pixel_to_projected[0], pixel_to_projected[1]],
         [pixel_to_projected[3], pixel_to_projected[4]]],
        dtype=np.float64,
    )
    return np.linalg.norm(np.asarray(residual_delta_px, dtype=np.float64) @ linear.T, axis=1)


def estimate_local_tile_shift(
    points_from: Iterable[Iterable[float]],
    points_to: Iterable[Iterable[float]],
    mask_from: np.ndarray,
    mask_to: np.ndarray,
    pixel_to_projected: Iterable[float],
    *,
    min_matches: int = 6,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.25,
    min_coverage_4x4: float = 0.25,
    max_p90_residual_m: float = 2.0,
    ransac_threshold_px: float = 3.0,
) -> LocalTileShift:
    """Fit a robust local similarity tie and report its inverse at cell centre.

    Masks use nonzero for candidate ground pixels; both ends of a tie must be
    eligible. ``pixel_to_projected`` is the six- or nine-value affine transform
    mapping the shared mosaic pixel grid to projected metres. Numerical gates
    label candidate fit strength only; projected residuals are not positional
    uncertainty and this function cannot approve ground truth. The fitted model
    permits rotation, uniform scale, and translation, but not shear or
    anisotropic scale.
    """
    from_points = np.asarray(points_from, dtype=np.float32).reshape(-1, 2)
    to_points = np.asarray(points_to, dtype=np.float32).reshape(-1, 2)
    first_mask = np.asarray(mask_from)
    second_mask = np.asarray(mask_to)
    if len(from_points) != len(to_points):
        raise ValueError("Source and target correspondence counts differ")
    if first_mask.ndim != 2 or second_mask.ndim != 2 or first_mask.shape != second_mask.shape:
        raise ValueError("Candidate masks must be 2D and share one local pixel grid")
    if (min_matches < 3 or min_inliers < 3 or
            not math.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0 or
            not math.isfinite(max_p90_residual_m) or max_p90_residual_m <= 0):
        raise ValueError("Match gates and residual thresholds must be positive and finite")
    transform = tuple(float(value) for value in pixel_to_projected)
    if len(transform) not in (6, 9):
        raise ValueError("Pixel-to-projected affine transform must have 6 or 9 values")
    if not np.isfinite(transform).all():
        raise ValueError("Pixel-to-projected affine transform must be finite")
    linear = np.asarray([[transform[0], transform[1]],
                         [transform[3], transform[4]]], dtype=np.float64)
    if abs(float(np.linalg.det(linear))) <= 1e-12:
        raise ValueError("Pixel-to-projected affine transform must be nonsingular")
    if not np.isfinite(from_points).all() or not np.isfinite(to_points).all():
        raise ValueError("Correspondences must be finite")

    eligible = _eligible_points(from_points, first_mask) & _eligible_points(to_points, second_mask)
    source = from_points[eligible]
    target = to_points[eligible]
    if len(source) < min_matches:
        return LocalTileShift("INSUFFICIENT_GROUND_CANDIDATE_MATCHES", len(source), 0,
                              0.0, 0.0, None, None, None, None, None, None)

    import cv2

    seed_material = source.astype("<f4", copy=False).tobytes() + target.astype("<f4", copy=False).tobytes()
    ransac_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "little", signed=True)
    cv2.setRNGSeed(ransac_seed)
    matrix, raw_inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
        maxIters=2000, confidence=0.99,
    )
    if matrix is None or raw_inliers is None:
        return LocalTileShift("NO_LOCAL_SIMILARITY_INLIERS", len(source), 0,
                              0.0, 0.0, None, None, None, None, None, None,
                              ransac_seed=ransac_seed)

    projected = np.empty_like(source, dtype=np.float64)
    projected[:, 0] = (
        source[:, 0] * float(matrix[0, 0]) +
        source[:, 1] * float(matrix[0, 1]) + float(matrix[0, 2])
    )
    projected[:, 1] = (
        source[:, 0] * float(matrix[1, 0]) +
        source[:, 1] * float(matrix[1, 1]) + float(matrix[1, 2])
    )
    residual_delta = projected - target
    residuals = np.sqrt(np.sum(residual_delta * residual_delta, axis=1))
    inliers = raw_inliers.reshape(-1).astype(bool) & (residuals <= ransac_threshold_px)
    n_inliers = int(inliers.sum())
    ratio = float(n_inliers / len(source)) if len(source) else 0.0
    residual = residuals[inliers]
    residual_m = _residuals_in_projected_units(residual_delta[inliers], transform)
    observed = target[inliers] - source[inliers]
    median_dx = float(np.median(observed[:, 0])) if n_inliers else None
    median_dy = float(np.median(observed[:, 1])) if n_inliers else None

    height, width = first_mask.shape
    points = source[inliers]
    covered = {(min(3, max(0, int(x * 4 / width))),
                min(3, max(0, int(y * 4 / height)))) for x, y in points}
    coverage = len(covered) / 16.0
    median_residual = float(np.median(residual)) if n_inliers else None
    p90_residual = float(np.percentile(residual, 90)) if n_inliers else None
    median_residual_m = float(np.median(residual_m)) if n_inliers else None
    p90_residual_m = float(np.percentile(residual_m, 90)) if n_inliers else None

    correction_dx = correction_dy = None
    if n_inliers >= 3:
        a, b, tx = (float(value) for value in matrix[0])
        c, d, ty = (float(value) for value in matrix[1])
        determinant = a * d - b * c
        if abs(determinant) <= 1e-12:
            raise ValueError("Estimated local similarity transform is singular")
        center_x, center_y = width / 2.0, height / 2.0
        offset_x, offset_y = center_x - tx, center_y - ty
        corrected_x = (d * offset_x - b * offset_y) / determinant
        corrected_y = (-c * offset_x + a * offset_y) / determinant
        shift_px = np.array([corrected_x - center_x, corrected_y - center_y])
        a, b, _, d, e, _ = transform[:6]
        correction_dx = float(a * shift_px[0] + b * shift_px[1])
        correction_dy = float(d * shift_px[0] + e * shift_px[1])

    if n_inliers < 4:
        status = "INSUFFICIENT_GROUND_CANDIDATE_INLIERS"
    elif (n_inliers < min_inliers or ratio < min_inlier_ratio or
          coverage < min_coverage_4x4 or p90_residual_m is None or
          p90_residual_m > max_p90_residual_m):
        status = "WEAK_LOCAL_CANDIDATE"
    else:
        status = "LOCAL_CANDIDATE_NOT_REVIEWED"
    return LocalTileShift(status, len(source), n_inliers, ratio, coverage,
                          median_residual, p90_residual, median_dx, median_dy,
                          correction_dx, correction_dy,
                          ransac_seed=ransac_seed,
                          inlier_points_from_px=tuple(map(tuple, source[inliers].astype(float).tolist())),
                          inlier_points_to_px=tuple(map(tuple, target[inliers].astype(float).tolist())),
                          median_residual_m=median_residual_m,
                          p90_residual_m=p90_residual_m)
