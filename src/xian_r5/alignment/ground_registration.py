"""SIFT ground-registration baseline for R5.

This module is deliberately an evidence producer.  It estimates transforms
for small, shared z20 JPG tiles and records diagnostics; it never rewrites a
polygon or a production raster.  LightGlue/ALIKED are optional backends.  If
their weights or package are unavailable, the result says so explicitly and
the OpenCV SIFT baseline remains separately identifiable.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .common import utc_now, write_csv, write_json


TILE_RE = re.compile(r"gesh_(?P<x>\d+)_(?P<y>\d+)_20\.(?:jpg|jpeg)$", re.IGNORECASE)


@dataclass(frozen=True)
class TileRef:
    key: str
    path: Path


def _tile_key(path: Path) -> str | None:
    match = TILE_RE.fullmatch(path.name)
    return f"{match.group('x')}_{match.group('y')}" if match else None


def index_tiles(tile_root: Path) -> dict[str, Path]:
    """Index the existing 20/*/*.jpg tile layout without reading pixels."""

    result: dict[str, Path] = {}
    root = tile_root.resolve()
    if not root.is_dir():
        return result
    for path in root.glob("*/*.jpg"):
        key = _tile_key(path)
        if key is not None:
            result[key] = path
    for path in root.glob("*/*.jpeg"):
        key = _tile_key(path)
        if key is not None:
            result[key] = path
    return result


def deterministic_sample(keys: Iterable[str], count: int, seed: str) -> list[str]:
    ranked = sorted(keys, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    return ranked[: max(0, count)]


def _gray(path: Path, max_side: int = 512) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        image = image.convert("L")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _cv2():
    try:
        import cv2

        return cv2
    except Exception:
        return None


def _apply_transform(transform_type: str, params: np.ndarray | None, points: np.ndarray) -> np.ndarray | None:
    if params is None:
        return None
    if transform_type == "translation":
        return points + params.reshape(1, 2)
    if transform_type in {"similarity", "affine"}:
        return points @ params[:, :2].T + params[:, 2]
    if transform_type == "homography":
        homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
        projected = homogeneous @ params.T
        denominator = projected[:, 2:3]
        return projected[:, :2] / np.where(np.abs(denominator) < 1e-9, 1.0, denominator)
    return None


def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    if len(points) == 0:
        return 0.0
    height, width = shape
    xs = np.clip(points[:, 0] / max(1.0, width), 0.0, 0.999)
    ys = np.clip(points[:, 1] / max(1.0, height), 0.0, 0.999)
    cells = {(int(x * 4), int(y * 4)) for x, y in zip(xs, ys)}
    return float(len(cells) / 16.0)


def _fit_candidates(src: np.ndarray, dst: np.ndarray, image_shape: tuple[int, int], cv2: Any) -> dict[str, dict[str, Any]]:
    candidates: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    translation = np.median(dst - src, axis=0).astype(np.float32) if len(src) else None
    candidates["translation"] = (translation, None)
    if len(src) >= 3:
        candidates["similarity"] = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
    else:
        candidates["similarity"] = (None, None)
    if len(src) >= 3:
        candidates["affine"] = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
    else:
        candidates["affine"] = (None, None)
    if len(src) >= 4:
        candidates["homography"] = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.99)
    else:
        candidates["homography"] = (None, None)

    results: dict[str, dict[str, Any]] = {}
    for transform_type, (params, mask) in candidates.items():
        projected = _apply_transform(transform_type, params, src)
        if projected is None:
            results[transform_type] = {"status": "NOT_ESTIMABLE", "n_inliers": 0, "inlier_ratio": 0.0}
            continue
        residual = np.linalg.norm(projected - dst, axis=1)
        if transform_type == "translation":
            inlier_mask = residual <= 3.0
        else:
            inlier_mask = np.asarray(mask).reshape(-1).astype(bool) if mask is not None else residual <= 3.0
        inlier_residual = residual[inlier_mask]
        results[transform_type] = {
            "status": "PASS_DIAGNOSTIC",
            "n_inliers": int(inlier_mask.sum()),
            "inlier_ratio": float(inlier_mask.mean()) if len(inlier_mask) else 0.0,
            "median_reprojection_error": float(np.median(inlier_residual)) if len(inlier_residual) else None,
            "p90_reprojection_error": float(np.percentile(inlier_residual, 90)) if len(inlier_residual) else None,
            "spatial_match_coverage": _coverage(src[inlier_mask], image_shape),
            "parameters": params.tolist() if hasattr(params, "tolist") else params,
        }
    return results


def match_tile_pair(path0: Path, path1: Path, *, extractor: str = "sift") -> dict[str, Any]:
    cv2 = _cv2()
    result: dict[str, Any] = {"path0": str(path0), "path1": str(path1), "extractor": extractor}
    if cv2 is None:
        result.update({"status": "BLOCKED_NO_OPENCV", "error": "opencv is not available in the selected runtime"})
        return result
    if extractor.lower() != "sift":
        result.update({"status": "NOT_RUN_OPTIONAL_BACKEND", "error": f"{extractor} backend requires an isolated LightGlue/ALIKED install and weights"})
        return result
    try:
        image0 = _gray(path0)
        image1 = _gray(path1)
        detector = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
        keypoints0, descriptors0 = detector.detectAndCompute(image0, None)
        keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
        result["n_keypoints0"] = len(keypoints0 or [])
        result["n_keypoints1"] = len(keypoints1 or [])
        if descriptors0 is None or descriptors1 is None:
            result.update({"status": "NO_DESCRIPTORS", "n_matches": 0})
            return result
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw = matcher.knnMatch(descriptors0, descriptors1, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        result["n_matches"] = len(good)
        if len(good) < 4:
            result.update({"status": "INSUFFICIENT_MATCHES", "n_inliers": 0})
            return result
        src = np.float32([keypoints0[m.queryIdx].pt for m in good])
        dst = np.float32([keypoints1[m.trainIdx].pt for m in good])
        candidates = _fit_candidates(src, dst, image0.shape[:2], cv2)
        def score(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float]:
            value = item[1]
            return (float(value.get("inlier_ratio") or 0.0), -float(value.get("median_reprojection_error") or 1e9), float(value.get("spatial_match_coverage") or 0.0))
        best_type, best = max(candidates.items(), key=score)
        result.update({
            "status": "PASS_DIAGNOSTIC",
            "n_inliers": best.get("n_inliers", 0),
            "inlier_ratio": best.get("inlier_ratio", 0.0),
            "median_reprojection_error": best.get("median_reprojection_error"),
            "p90_reprojection_error": best.get("p90_reprojection_error"),
            "spatial_match_coverage": best.get("spatial_match_coverage", 0.0),
            "transform_type": best_type,
            "candidate_models": candidates,
            "match_points0": src[:80].tolist(),
            "match_points1": dst[:80].tolist(),
        })
        return result
    except Exception as exc:  # pragma: no cover - codec/platform dependent
        result.update({"status": "FAILED_TILE_DIAGNOSTIC", "error": repr(exc)})
        return result


def _figure(path0: Path, path1: Path, result: dict[str, Any], output: Path) -> None:
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path0) as left, Image.open(path1) as right:
            left = left.convert("RGB").resize((384, 384))
            right = right.convert("RGB").resize((384, 384))
            canvas = Image.new("RGB", (768, 384), "white")
            canvas.paste(left, (0, 0))
            canvas.paste(right, (384, 0))
            draw = ImageDraw.Draw(canvas)
            for p0, p1 in zip(result.get("match_points0", [])[:30], result.get("match_points1", [])[:30]):
                x0, y0 = float(p0[0]) * 384 / 256, float(p0[1]) * 384 / 256
                x1, y1 = 384 + float(p1[0]) * 384 / 256, float(p1[1]) * 384 / 256
                draw.line((x0, y0, x1, y1), fill=(255, 80, 20), width=1)
                draw.ellipse((x0 - 2, y0 - 2, x0 + 2, y0 + 2), fill=(20, 150, 255))
                draw.ellipse((x1 - 2, y1 - 2, x1 + 2, y1 + 2), fill=(20, 200, 80))
            output.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(output)
    except Exception:
        return


def run_alignment_baseline(
    scene_tile_roots: dict[str, Path],
    output_root: Path,
    *,
    sample_count: int = 100,
    pairs: tuple[tuple[str, str], ...] = (("2018", "2019"), ("2019", "2020"), ("2018", "2020")),
) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    indexes = {epoch: index_tiles(root) for epoch, root in scene_tile_roots.items()}
    tile_rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    pair_receipts: list[dict[str, Any]] = []
    figures = output_root / "correspondence_figures"
    for first, second in pairs:
        left = indexes.get(first, {})
        right = indexes.get(second, {})
        common = sorted(set(left) & set(right))
        sampled = deterministic_sample(common, sample_count, f"{first}:{second}:R5_ALIGN_A01")
        pair_results: list[dict[str, Any]] = []
        for key in sampled:
            result = match_tile_pair(left[key], right[key], extractor="sift")
            result.update({"from_epoch": first, "to_epoch": second, "tile_key": key, "sample_stratum": "UNLABELLED_TILE_PROXY"})
            tile_rows.append(result)
            pair_results.append(result)
        valid = [row for row in pair_results if row.get("status") == "PASS_DIAGNOSTIC"]
        if valid:
            first_valid = valid[0]
            _figure(Path(first_valid["path0"]), Path(first_valid["path1"]), first_valid, figures / f"{first}_{second}_sift_correspondence.png")
        def med(field: str) -> float | None:
            values = [float(row[field]) for row in valid if row.get(field) is not None]
            return float(median(values)) if values else None
        transforms = [row.get("transform_type") for row in valid if row.get("transform_type")]
        best_transform = max(set(transforms), key=transforms.count) if transforms else None
        qa_rows.append({
            "from_epoch": first,
            "to_epoch": second,
            "extractor": "SIFT",
            "sample_count_requested": sample_count,
            "sample_count_available": len(common),
            "sample_count_run": len(sampled),
            "n_keypoints": med("n_keypoints0"),
            "n_matches": med("n_matches"),
            "n_inliers": med("n_inliers"),
            "inlier_ratio": med("inlier_ratio"),
            "median_reprojection_error": med("median_reprojection_error"),
            "p90_reprojection_error": med("p90_reprojection_error"),
            "spatial_match_coverage": med("spatial_match_coverage"),
            "transform_type": best_transform,
            "stable_ground_mask": "NOT_AVAILABLE_SEMANTIC_MASK;SIFT_ALL_TEXTURE_PROXY",
            "temporal_mixing_warning": "CHECK_IMAGE_OBSERVATION_REGISTRY",
            "status": "PASS_DIAGNOSTIC" if valid else "NO_VALID_TILE_DIAGNOSTICS",
        })
        pair_receipts.append({"from_epoch": first, "to_epoch": second, "common_tiles": len(common), "sampled": len(sampled), "valid": len(valid)})
    write_csv(output_root / "ALIGNMENT_TILE_DIAGNOSTICS.csv", tile_rows)
    write_csv(output_root / "GLOBAL_ALIGNMENT_QA.csv", qa_rows)
    receipt = {
        "run_id": "R5_ALIGN_A01",
        "generated_at_utc": utc_now(),
        "backend": "OpenCV SIFT + BFMatcher ratio test + RANSAC/USAC model comparison",
        "pairs": pair_receipts,
        "models": ["translation", "similarity", "affine", "homography"],
        "optional_backends": {"ALIKED_LIGHTGLUE": "NOT_RUN_OPTIONAL_BACKEND_NO_ISOLATED_LIGHTGLUE_WEIGHTS"},
        "stable_ground_sample": {"requested": 100, "completed": 0, "status": "NOT_RUN_NO_STABLE_GROUND_LABEL_OR_MASK"},
        "parallax_risk_sample": {"requested": 100, "completed": 0, "status": "NOT_RUN_NO_CASE_TO_TILE_MAPPING"},
        "polygon_policy": "NO_POLYGON_WARP;EVIDENCE_ONLY",
    }
    write_json(output_root / "GLOBAL_ALIGNMENT_RECEIPT.json", receipt)
    return {"tile_rows": tile_rows, "qa_rows": qa_rows, "receipt": receipt}
