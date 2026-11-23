"""Ground-masked tile registration with explicit candidate/accepted separation.

The automatic mask excludes known roof polygons and strongly green vegetation.
It is *not* proof that all surviving SIFT features lie on the ground: a manual
stable-ground anchor review is still required to accept the displacement.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

try:
    import cv2
except Exception as exc:  # keep non-registration reconstruction usable
    cv2 = None
    _CV2_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _CV2_IMPORT_ERROR = None


ORIGIN_X = 420534
ORIGIN_Y = 311895
ORIGIN_LON = 108.75778198242188
ORIGIN_LAT = 34.16130065917969
STEP_DEG = 0.0006866455078125
TO_WGS = Transformer.from_crs(32649, 4326, always_xy=True)
TO_UTM = Transformer.from_crs(4326, 32649, always_xy=True)


def registration_backend_status() -> dict[str, str | bool | None]:
    """Report whether the optional OpenCV registration backend is usable."""
    available = cv2 is not None
    return {
        "name": "OPENCV_SIFT_GROUND_MASKED_TRANSLATION",
        "available": available,
        "status": "AVAILABLE" if available else "BLOCKED_NO_OPENCV",
        "import_error": None if available else (_CV2_IMPORT_ERROR or "OpenCV module unavailable"),
    }


@dataclass(frozen=True)
class Registration:
    tile_x: int
    tile_y: int
    from_date: str
    to_date: str
    dx_m: float | None
    dy_m: float | None
    sigma_m: float | None
    n_matches: int
    n_inliers: int
    coverage_4x4: float
    median_residual_px: float | None
    status: str
    anchor_reviewed: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MosaicRegistration:
    """Read-only local-feature registration evidence for a date-coherent image pair.

    Every fitted model maps target-image pixel coordinates into source-image
    pixel coordinates. Fits remain candidates even when their support metrics
    are strong; a masked image is not, by itself, proof of stable ground.
    """

    from_date: str
    to_date: str
    status: str
    source_keypoints: int
    target_keypoints: int
    mutual_matches: int
    translation_dx_m: float | None
    translation_dy_m: float | None
    sigma_m: float | None
    models: dict
    valid_mask_fraction: float
    ground_mask_reviewed: bool = False
    applied_to_image_or_geometry: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MosaicPhaseRegistration:
    """Dense phase-correlation shift candidate; never an accepted correction."""

    from_date: str
    to_date: str
    status: str
    translation_dx_px: float | None
    translation_dy_px: float | None
    translation_dx_m: float | None
    translation_dy_m: float | None
    phase_response: float | None
    valid_mask_fraction: float
    applied_to_image_or_geometry: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _tile_ll(tile_x: int, tile_y: int) -> tuple[float, float]:
    return (ORIGIN_LON + (tile_x - ORIGIN_X) * STEP_DEG,
            ORIGIN_LAT + (tile_y - ORIGIN_Y) * STEP_DEG)


def _pixel_box(geom: BaseGeometry, tile_x: int, tile_y: int, width: int, height: int, span: int) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = geom.bounds
    lon0, lat0 = TO_WGS.transform(minx, miny)
    lon1, lat1 = TO_WGS.transform(maxx, maxy)
    west, south = _tile_ll(tile_x, tile_y)
    full_step = STEP_DEG * span
    return (int((min(lon0, lon1) - west) / full_step * width),
            int((south + full_step - max(lat0, lat1)) / full_step * height),
            int((max(lon0, lon1) - west) / full_step * width),
            int((south + full_step - min(lat0, lat1)) / full_step * height))


def _ground_candidate_mask(rgb: np.ndarray, roofs: Iterable[BaseGeometry], tile_x: int, tile_y: int, span: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Avoid vegetation and dark shadow, but do not call remaining pixels verified ground.
    green = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 40)
    dark = hsv[:, :, 2] < 28
    mask = np.where(green | dark, 0, 255).astype(np.uint8)
    for roof in roofs:
        x0, y0, x1, y1 = _pixel_box(roof, tile_x, tile_y, width, height, span)
        if x1 >= 0 and y1 >= 0 and x0 < width and y0 < height:
            cv2.rectangle(mask, (max(0, x0 - 5), max(0, y0 - 5)),
                          (min(width - 1, x1 + 5), min(height - 1, y1 + 5)), 0, -1)
    return mask


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    cells = {(min(3, int(x * 4 / width)), min(3, int(y * 4 / height))) for x, y in points}
    return len(cells) / 16.0


def estimate_mosaic_registration(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    valid_ground_mask: np.ndarray,
    *,
    from_date: str,
    to_date: str,
    pixel_to_projected: tuple[float, float, float, float],
    ground_mask_reviewed: bool = False,
    max_features: int = 15000,
    ratio_test: float = 0.72,
    ransac_threshold_px: float = 2.0,
    include_match_points: bool = False,
    feature_method: str = "SIFT",
) -> MosaicRegistration:
    """Compare translation/similarity/affine/homography on a masked mosaic.

    Inputs must be date-coherent RGB arrays on the same nominal pixel grid.
    ``valid_ground_mask`` is a candidate mask (nonzero means eligible); callers
    must not describe it as ground truth unless its land-cover controls were
    independently reviewed. The returned translation is a target-to-source
    correction candidate. No image or geometry is changed.
    """
    height, width = source_rgb.shape[:2]
    mask = np.asarray(valid_ground_mask) != 0
    valid_fraction = float(mask.mean()) if mask.size else 0.0
    empty_models = {
        name: {"status": "NOT_ESTIMATED", "transform_target_to_source_px": None,
               "n_inliers": 0, "inlier_ratio": None, "coverage_4x4": 0.0,
               "median_reprojection_residual_px": None,
               "p90_reprojection_residual_px": None,
               "median_reprojection_residual_m": None,
               "p90_reprojection_residual_m": None}
        for name in ("translation", "similarity", "affine", "homography")
    }

    def _result(status: str, source_n: int = 0, target_n: int = 0,
                matches: int = 0, dx_m: float | None = None,
                dy_m: float | None = None, sigma_m: float | None = None,
                models: dict | None = None) -> MosaicRegistration:
        return MosaicRegistration(
            from_date=from_date, to_date=to_date, status=status,
            source_keypoints=source_n, target_keypoints=target_n,
            mutual_matches=matches, translation_dx_m=dx_m,
            translation_dy_m=dy_m, sigma_m=sigma_m,
            models=models if models is not None else empty_models,
            valid_mask_fraction=valid_fraction,
            ground_mask_reviewed=ground_mask_reviewed,
            applied_to_image_or_geometry=False,
        )

    try:
        from_day, to_day = date.fromisoformat(from_date), date.fromisoformat(to_date)
    except (TypeError, ValueError):
        return _result("ACTUAL_DATE_UNRESOLVED")
    if from_day == to_day:
        return _result("DUPLICATE_EPOCH")
    if to_day < from_day:
        return _result("INVALID_DATE_ORDER")
    if (source_rgb.ndim != 3 or target_rgb.ndim != 3 or source_rgb.shape != target_rgb.shape
            or source_rgb.shape[2] != 3 or mask.shape != (height, width)
            or height < 32 or width < 32):
        return _result("IMAGE_OR_MASK_SHAPE_MISMATCH")
    if cv2 is None:
        return _result("BLOCKED_NO_OPENCV")

    feature_method = feature_method.upper()
    if feature_method == "SIFT":
        detector = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
        norm_type = cv2.NORM_L2
    elif feature_method == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features, edgeThreshold=15,
                                  patchSize=31, fastThreshold=10)
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported registration feature method: {feature_method}")

    source_gray = cv2.cvtColor(np.asarray(source_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(np.asarray(target_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    cv_mask = np.where(mask, 255, 0).astype(np.uint8)
    source_kp, source_desc = detector.detectAndCompute(source_gray, cv_mask)
    target_kp, target_desc = detector.detectAndCompute(target_gray, cv_mask)
    source_n, target_n = len(source_kp), len(target_kp)
    if source_desc is None or target_desc is None or min(source_n, target_n) < 8:
        return _result("INSUFFICIENT_GROUND_CANDIDATE_FEATURES", source_n, target_n)

    matcher = cv2.BFMatcher(norm_type)
    forward = matcher.knnMatch(source_desc, target_desc, k=2)
    reverse = matcher.knnMatch(target_desc, source_desc, k=2)
    good_forward = [m for pair in forward if len(pair) == 2
                    for m, runner_up in [pair]
                    if m.distance < ratio_test * runner_up.distance]
    good_reverse = {(m.trainIdx, m.queryIdx) for pair in reverse if len(pair) == 2
                    for m, runner_up in [pair]
                    if m.distance < ratio_test * runner_up.distance}
    matches = [m for m in good_forward if (m.queryIdx, m.trainIdx) in good_reverse]
    if len(matches) < 8:
        return _result("INSUFFICIENT_MUTUAL_MATCHES", source_n, target_n, len(matches))

    p_source = np.float32([source_kp[m.queryIdx].pt for m in matches])
    p_target = np.float32([target_kp[m.trainIdx].pt for m in matches])
    affine = np.asarray([[pixel_to_projected[0], pixel_to_projected[1]],
                         [pixel_to_projected[2], pixel_to_projected[3]]], dtype=np.float64)

    delta = p_source - p_target
    best_translation = None
    best_mask = np.zeros(len(delta), dtype=bool)
    for offset in delta:
        translation_residual = delta - offset
        inliers = (np.sqrt(np.sum(translation_residual * translation_residual, axis=1))
                   <= ransac_threshold_px)
        if inliers.sum() > best_mask.sum():
            best_mask = inliers
            best_translation = np.median(delta[inliers], axis=0)
    model_parameters: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        "translation": (np.asarray([[1.0, 0.0, best_translation[0]],
                                    [0.0, 1.0, best_translation[1]]], dtype=np.float64), best_mask)
        if best_translation is not None else None,
        "similarity": None,
        "affine": None,
        "homography": None,
    }
    if len(matches) >= 8:
        partial, partial_mask = cv2.estimateAffinePartial2D(
            p_target, p_source, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px, maxIters=2000,
            confidence=0.99, refineIters=10)
        if partial is not None and partial_mask is not None:
            model_parameters["similarity"] = (partial, partial_mask.ravel().astype(bool))
        full_affine, full_mask = cv2.estimateAffine2D(
            p_target, p_source, method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px, maxIters=2000,
            confidence=0.99, refineIters=10)
        if full_affine is not None and full_mask is not None:
            model_parameters["affine"] = (full_affine, full_mask.ravel().astype(bool))
        homography, homography_mask = cv2.findHomography(
            p_target, p_source, cv2.RANSAC, ransac_threshold_px,
            maxIters=2000, confidence=0.99)
        if homography is not None and homography_mask is not None:
            model_parameters["homography"] = (homography, homography_mask.ravel().astype(bool))

    models = {}
    for name, fit in model_parameters.items():
        if fit is None:
            models[name] = empty_models[name]
            continue
        matrix, _ = fit
        if matrix.shape == (2, 3):
            projected = cv2.transform(p_target.reshape(1, -1, 2), matrix.astype(np.float64))[0]
        else:
            projected_h = np.empty((len(p_target), 3), dtype=np.float64)
            projected_h[:, 0] = (p_target[:, 0] * matrix[0, 0] +
                                 p_target[:, 1] * matrix[0, 1] + matrix[0, 2])
            projected_h[:, 1] = (p_target[:, 0] * matrix[1, 0] +
                                 p_target[:, 1] * matrix[1, 1] + matrix[1, 2])
            projected_h[:, 2] = (p_target[:, 0] * matrix[2, 0] +
                                 p_target[:, 1] * matrix[2, 1] + matrix[2, 2])
            denominator = projected_h[:, 2]
            denominator = np.where(np.abs(denominator) < 1e-12,
                                   np.where(denominator < 0, -1e-12, 1e-12),
                                   denominator)
            projected = projected_h[:, :2] / denominator[:, None]
        residual_vectors = p_source - projected
        residual_px = np.sqrt(np.sum(residual_vectors * residual_vectors, axis=1))
        inliers = residual_px <= ransac_threshold_px
        inlier_vectors = residual_vectors[inliers]
        if inliers.any():
            residual_x_m = (inlier_vectors[:, 0] * affine[0, 0] +
                            inlier_vectors[:, 1] * affine[0, 1])
            residual_y_m = (inlier_vectors[:, 0] * affine[1, 0] +
                            inlier_vectors[:, 1] * affine[1, 1])
            residual_m = np.sqrt(residual_x_m * residual_x_m +
                                 residual_y_m * residual_y_m)
        else:
            residual_m = np.asarray([], dtype=np.float64)
        n_inliers = int(inliers.sum())
        model_record = {
            "status": "MODEL_DIAGNOSTIC_ONLY",
            "transform_target_to_source_px": matrix.tolist(),
            "n_inliers": n_inliers,
            "inlier_ratio": n_inliers / len(matches),
            "coverage_4x4": _coverage(p_source[inliers], width, height) if inliers.any() else 0.0,
            "median_reprojection_residual_px": float(np.median(residual_px[inliers])) if inliers.any() else None,
            "p90_reprojection_residual_px": float(np.percentile(residual_px[inliers], 90)) if inliers.any() else None,
            "median_reprojection_residual_m": float(np.median(residual_m)) if len(residual_m) else None,
            "p90_reprojection_residual_m": float(np.percentile(residual_m, 90)) if len(residual_m) else None,
            "support_diagnostic_pass": bool(n_inliers >= 12 and n_inliers / len(matches) >= 0.45
                                            and _coverage(p_source[inliers], width, height) >= 0.25),
        }
        if include_match_points:
            model_record["inlier_correspondences_px"] = np.column_stack(
                (p_source[inliers], p_target[inliers])
            ).round(4).tolist()
        models[name] = model_record

    translation_fit = model_parameters["translation"]
    translation_metrics = models["translation"]
    if translation_fit is None or translation_metrics["n_inliers"] < 4:
        return _result("WEAK_GROUND_MASKED_MATCH_CANDIDATE", source_n, target_n,
                       len(matches), models=models)
    translation_px = translation_fit[0][:, 2]
    translation_m = np.asarray([
        affine[0, 0] * translation_px[0] + affine[0, 1] * translation_px[1],
        affine[1, 0] * translation_px[0] + affine[1, 1] * translation_px[1],
    ])
    return _result("GROUND_MASKED_CANDIDATE_NOT_REVIEWED", source_n, target_n,
                   len(matches), round(float(translation_m[0]), 4),
                   round(float(translation_m[1]), 4),
                   translation_metrics["median_reprojection_residual_m"], models=models)


def estimate_mosaic_phase_registration(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    valid_ground_mask: np.ndarray,
    *,
    from_date: str,
    to_date: str,
    pixel_to_projected: tuple[float, float, float, float],
    min_valid_pixels: int = 1024,
) -> MosaicPhaseRegistration:
    """Return a masked gradient phase-correlation translation candidate.

    The candidate mask is assumed to be common to both inputs. Remaining
    pixels are not thereby proven to be stable ground. The returned vector is
    a target-to-source correction candidate, and no transform is applied.
    """
    mask = np.asarray(valid_ground_mask) != 0
    valid_fraction = float(mask.mean()) if mask.size else 0.0

    def result(status: str, dx_px: float | None = None, dy_px: float | None = None,
               dx_m: float | None = None, dy_m: float | None = None,
               response: float | None = None) -> MosaicPhaseRegistration:
        return MosaicPhaseRegistration(from_date, to_date, status, dx_px, dy_px,
                                       dx_m, dy_m, response, valid_fraction, False)

    try:
        from_day, to_day = date.fromisoformat(from_date), date.fromisoformat(to_date)
    except (TypeError, ValueError):
        return result("ACTUAL_DATE_UNRESOLVED")
    if from_day == to_day:
        return result("DUPLICATE_EPOCH")
    if to_day < from_day:
        return result("INVALID_DATE_ORDER")
    if (source_rgb.ndim != 3 or target_rgb.ndim != 3 or source_rgb.shape != target_rgb.shape
            or source_rgb.shape[2] != 3 or mask.shape != source_rgb.shape[:2]
            or min(source_rgb.shape[:2]) < 32):
        return result("IMAGE_OR_MASK_SHAPE_MISMATCH")
    if int(mask.sum()) < min_valid_pixels:
        return result("INSUFFICIENT_VALID_MASK_PIXELS")
    if cv2 is None:
        return result("BLOCKED_NO_OPENCV")

    height, width = mask.shape
    source_gray = cv2.cvtColor(np.asarray(source_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(np.asarray(target_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    def normalized_gradient(gray: np.ndarray) -> np.ndarray | None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        values = magnitude[mask]
        spread = float(values.std())
        if not np.isfinite(spread) or spread < 1e-6:
            return None
        normalized = np.zeros_like(magnitude, dtype=np.float32)
        normalized[mask] = (values - float(values.mean())) / spread
        return normalized

    source_features = normalized_gradient(source_gray)
    target_features = normalized_gradient(target_gray)
    if source_features is None or target_features is None:
        return result("INSUFFICIENT_VALID_TEXTURE")
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    window *= mask.astype(np.float32)
    source_features *= window
    target_features *= window
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(source_features, target_features)
    except cv2.error:
        return result("PHASE_CORRELATION_FAILED")
    if not np.isfinite([shift_x, shift_y, response]).all():
        return result("PHASE_CORRELATION_NONFINITE")

    # OpenCV reports source-to-target movement; expose target-to-source
    # correction to match estimate_mosaic_registration's convention.
    correction_px = np.asarray([-shift_x, -shift_y], dtype=np.float64)
    affine = np.asarray([[pixel_to_projected[0], pixel_to_projected[1]],
                         [pixel_to_projected[2], pixel_to_projected[3]]], dtype=np.float64)
    correction_m = affine @ correction_px
    return result("PHASE_CORRELATION_CANDIDATE_NOT_REVIEWED",
                  round(float(correction_px[0]), 4), round(float(correction_px[1]), 4),
                  round(float(correction_m[0]), 4), round(float(correction_m[1]), 4),
                  round(float(response), 6))


def estimate_tile_registration(
    left_path: Path | None, right_path: Path | None, *, tile_x: int, tile_y: int,
    from_date: str, to_date: str,
    left_roofs: Iterable[BaseGeometry] = (), right_roofs: Iterable[BaseGeometry] = (),
    anchor_reviewed: bool = False,
    left_image: np.ndarray | None = None, right_image: np.ndarray | None = None,
    tile_span: int = 1,
) -> Registration:
    """Estimate right-to-left *image* displacement, not roof-ground parallax."""
    if cv2 is None:
        return Registration(tile_x, tile_y, from_date, to_date,
                            None, None, None, 0, 0, 0.0, None,
                            "BLOCKED_NO_OPENCV")
    left = left_image if left_image is not None else np.asarray(Image.open(left_path).convert("RGB"))
    right = right_image if right_image is not None else np.asarray(Image.open(right_path).convert("RGB"))
    if left.shape != right.shape or left.shape[0] < 32 or left.shape[1] < 32:
        return Registration(tile_x, tile_y, from_date, to_date, None, None, None, 0, 0, 0.0, None, "IMAGE_SHAPE_MISMATCH")
    height, width = left.shape[:2]
    mask0 = _ground_candidate_mask(left, left_roofs, tile_x, tile_y, tile_span)
    mask1 = _ground_candidate_mask(right, right_roofs, tile_x, tile_y, tile_span)
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.02)
    kp0, d0 = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), mask0)
    kp1, d1 = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), mask1)
    if d0 is None or d1 is None or len(kp0) < 8 or len(kp1) < 8:
        return Registration(tile_x, tile_y, from_date, to_date, None, None, None, 0, 0, 0.0, None, "INSUFFICIENT_GROUND_CANDIDATE_FEATURES")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, d1, k=2)
    good = [a for pair in matches if len(pair) == 2 for a, b in [pair] if a.distance < 0.72 * b.distance]
    if len(good) < 12:
        return Registration(tile_x, tile_y, from_date, to_date, None, None, None, len(good), 0, 0.0, None, "INSUFFICIENT_MATCHES")
    p0 = np.float32([kp0[m.queryIdx].pt for m in good])
    p1 = np.float32([kp1[m.trainIdx].pt for m in good])
    delta = p0 - p1
    # Translation is safer than an unconstrained homography for small tiles.
    # RANSAC is used to reject moving cars, parallax roof edges, and bad matches.
    best = None
    best_mask = np.zeros(len(delta), dtype=bool)
    for offset in delta:
        distances = np.linalg.norm(delta - offset, axis=1)
        mask = distances <= 2.0
        if mask.sum() > best_mask.sum():
            best_mask = mask
            best = np.median(delta[mask], axis=0)
    assert best is not None
    residuals = np.linalg.norm(delta - best, axis=1)
    best_mask = residuals <= 2.0
    coverage = _coverage(p0[best_mask], width, height)
    median_px = float(np.median(residuals[best_mask])) if best_mask.any() else None
    if best_mask.sum() < 12 or coverage < 0.25 or best_mask.mean() < 0.45:
        status = "REJECTED_WEAK_GROUND_CANDIDATE"
    else:
        status = "ACCEPTED_REVIEWED_GROUND" if anchor_reviewed else "GROUND_MASKED_CANDIDATE_NOT_REVIEWED"
    west, south = _tile_ll(tile_x, tile_y)
    cx = west + STEP_DEG / 2
    cy = south + STEP_DEG / 2
    x0, y0 = TO_UTM.transform(cx, cy)
    x1, y1 = TO_UTM.transform(cx + float(best[0]) / width * STEP_DEG * tile_span,
                              cy - float(best[1]) / height * STEP_DEG * tile_span)
    metre_per_px = max(abs(TO_UTM.transform(cx + STEP_DEG * tile_span / width, cy)[0] - x0),
                       abs(TO_UTM.transform(cx, cy - STEP_DEG * tile_span / height)[1] - y0))
    return Registration(tile_x, tile_y, from_date, to_date,
                        round(x1 - x0, 4) if status != "REJECTED_WEAK_GROUND_CANDIDATE" else None,
                        round(y1 - y0, 4) if status != "REJECTED_WEAK_GROUND_CANDIDATE" else None,
                        round((median_px or 0.0) * metre_per_px, 4) if status != "REJECTED_WEAK_GROUND_CANDIDATE" else None,
                        len(good), int(best_mask.sum()),
                        coverage, median_px, status, anchor_reviewed)
