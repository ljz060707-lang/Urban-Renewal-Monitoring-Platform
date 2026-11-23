#!/usr/bin/env python3
"""Run SIFT and official ALIKED+LightGlue on identical Stage-2 tile proposals.

The current pools are intentionally unreviewed proxies. Metrics from this run
are candidate diagnostics only and do not pass stable-ground/parallax gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue


STAGE2 = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2")
METHODS_DIR = STAGE2 / "06_lightglue_ab"
VISUAL_DIR = STAGE2 / "11_visual_qa" / "alignment_method_overlays_v2"
TORCH_HOME = METHODS_DIR / "torch_cache"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def update_status(**values: Any) -> None:
    path = STAGE2 / "13_receipts" / "RUN_STATUS.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(values)
    data["updated_at_utc"] = now_utc()
    data["active_process"] = os.getpid()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_progress(message: str) -> None:
    path = STAGE2 / "13_receipts" / "RUN_PROGRESS.md"
    old = path.read_text(encoding="utf-8") if path.exists() else "# R5 Stage-2 progress\n"
    path.write_text(old.rstrip() + f"\n- {now_utc()} — {message}\n", encoding="utf-8")


def _image_tensor(path: str, device: torch.device) -> torch.Tensor:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"unreadable image tile: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(arr).float().div_(255.0).unsqueeze(0).to(device)


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if not len(points):
        return 0.0
    cells = set()
    for x, y in points:
        cells.add((min(3, max(0, int(float(x) / width * 4))), min(3, max(0, int(float(y) / height * 4)))))
    return len(cells) / 16.0


def _score_matches(points0: np.ndarray, points1: np.ndarray, width: int, height: int) -> dict[str, Any]:
    n = len(points0)
    if n < 3:
        return {"status": "INSUFFICIENT_MATCHES", "match_count": n, "inlier_count": 0, "inlier_ratio": 0.0,
                "spatial_coverage": 0.0, "median_residual_px": None, "p90_residual_px": None,
                "median_dx_px": None, "median_dy_px": None, "inlier_mask": np.zeros(n, dtype=bool), "residuals": np.zeros(n)}
    matrix, raw_mask = cv2.estimateAffinePartial2D(
        points0.astype(np.float32), points1.astype(np.float32), method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99,
    )
    if matrix is None or raw_mask is None:
        return {"status": "NO_AFFINE_INLIERS", "match_count": n, "inlier_count": 0, "inlier_ratio": 0.0,
                "spatial_coverage": 0.0, "median_residual_px": None, "p90_residual_px": None,
                "median_dx_px": None, "median_dy_px": None, "inlier_mask": np.zeros(n, dtype=bool), "residuals": np.zeros(n)}
    pred = points0 @ matrix[:, :2].T + matrix[:, 2]
    residuals = np.linalg.norm(pred - points1, axis=1)
    inliers = raw_mask.reshape(-1).astype(bool) & (residuals <= 3.0)
    res = residuals[inliers]
    diffs = (points1 - points0)[inliers]
    return {
        "status": "PASS_DIAGNOSTIC" if inliers.any() else "NO_AFFINE_INLIERS",
        "match_count": n,
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()) if n else 0.0,
        "spatial_coverage": _coverage(points0[inliers], width, height),
        "median_residual_px": float(np.median(res)) if len(res) else None,
        "p90_residual_px": float(np.percentile(res, 90)) if len(res) else None,
        "median_dx_px": float(np.median(diffs[:, 0])) if len(diffs) else None,
        "median_dy_px": float(np.median(diffs[:, 1])) if len(diffs) else None,
        "inlier_mask": inliers,
        "residuals": residuals,
    }


def _sift(path0: str, path1: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    if image0 is None or image1 is None:
        raise OSError("SIFT input JPEG could not be decoded")
    detector = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.02)
    keypoints0, descriptors0 = detector.detectAndCompute(image0, None)
    keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
    if descriptors0 is None or descriptors1 is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return ({"status": "NO_DESCRIPTORS", "match_count": 0, "n_keypoints0": len(keypoints0 or []),
                 "n_keypoints1": len(keypoints1 or [])}, empty, empty, image0, image1)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors0, descriptors1, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    p0 = np.float32([keypoints0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    p1 = np.float32([keypoints1[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return ({"n_keypoints0": len(keypoints0 or []), "n_keypoints1": len(keypoints1 or []), "ratio_test": 0.75}, p0, p1, image0, image1)


def _lightglue(extractor: Any, matcher: Any, path0: str, path1: str, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image0 = _image_tensor(path0, device)
    image1 = _image_tensor(path1, device)
    with torch.inference_mode():
        feat0 = extractor.extract(image0)
        feat1 = extractor.extract(image1)
        out = matcher({"image0": feat0, "image1": feat1})
    keypoints0 = feat0["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    keypoints1 = feat1["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    matches = out["matches"][0].detach().cpu().numpy().astype(np.int64)
    if matches.ndim != 2 or matches.shape[-1] != 2:
        matches = np.empty((0, 2), dtype=np.int64)
    p0 = keypoints0[matches[:, 0]] if len(matches) else np.empty((0, 2), dtype=np.float32)
    p1 = keypoints1[matches[:, 1]] if len(matches) else np.empty((0, 2), dtype=np.float32)
    rgb0 = cv2.cvtColor(cv2.imread(path0), cv2.COLOR_BGR2RGB)
    rgb1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    score_batch = out.get("scores")
    if isinstance(score_batch, (list, tuple)):
        score_values = score_batch[0] if score_batch else None
    else:
        score_values = score_batch
    mean_score = float(score_values.mean().item()) if isinstance(score_values, torch.Tensor) and score_values.numel() else None
    return ({"n_keypoints0": len(keypoints0), "n_keypoints1": len(keypoints1), "mean_match_confidence": mean_score}, p0, p1, rgb0, rgb1)


def _draw_overlay(image0: np.ndarray, image1: np.ndarray, p0: np.ndarray, p1: np.ndarray, inliers: np.ndarray, path: Path) -> None:
    if image0.ndim == 2:
        image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
    if image1.ndim == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    left = cv2.resize(image0, (384, 384))
    right = cv2.resize(image1, (384, 384))
    canvas = np.concatenate([left, right], axis=1)
    scale0 = np.array([384 / image0.shape[1], 384 / image0.shape[0]])
    scale1 = np.array([384 / image1.shape[1], 384 / image1.shape[0]])
    for i, (a, b) in enumerate(zip(p0, p1)):
        if i >= len(inliers) or not inliers[i]:
            continue
        q0 = tuple(np.round(a * scale0).astype(int))
        q1v = np.round(b * scale1).astype(int)
        q1 = (int(q1v[0] + 384), int(q1v[1]))
        cv2.line(canvas, q0, q1, (255, 100, 10), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> int:
    metrics_path = METHODS_DIR / "ALIGNMENT_METHOD_CANDIDATE_METRICS_V2.csv"
    points_path = METHODS_DIR / "ALIGNMENT_MATCH_POINTS_V2.csv"
    summary_path = METHODS_DIR / "ALIGNMENT_METHOD_CANDIDATE_SUMMARY_V2.csv"
    for path in (metrics_path, points_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing alignment result: {path}")
    METHODS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("LightGlue Stage-2 run requires the isolated CUDA-capable environment")
    extractor = ALIKED(max_num_keypoints=1200).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    model_paths = [
        TORCH_HOME / "hub" / "checkpoints" / "aliked-n16.pth",
        TORCH_HOME / "hub" / "checkpoints" / "aliked_lightglue_v0-1_arxiv.pth",
    ]
    if not all(path.is_file() for path in model_paths):
        raise FileNotFoundError("Official ALIKED/LightGlue weight files were not found in the Stage-2 cache")

    all_metrics: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []
    group_values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_status_counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    processed = 0
    update_status(phase="ALIGNMENT_METHOD_PROXY_DIAGNOSTICS", current_gate="SIFT_ALIKED_LIGHTGLUE_PROXY_RUN", blocker="VISUAL_STRATUM_REVIEW_PENDING;RBF_SAM_CHECKPOINT_BLOCKED", next_action="RUN_SAME_TILE_PROXY_POOL_THEN_VISUAL_REVIEW")
    append_progress("SIFT and official ALIKED+LightGlue candidate diagnostics started on identical actual-date-coherent proxy tile pools; no proxy is counted as a validated stratum.")

    for pool_file in (
        STAGE2 / "04_stable_ground" / "STABLE_GROUND_CANDIDATE_POOL.csv",
        STAGE2 / "05_parallax" / "PARALLAX_RISK_CANDIDATE_POOL.csv",
    ):
        with pool_file.open("r", encoding="utf-8-sig", newline="") as stream:
            pool_rows = list(csv.DictReader(stream))
        proxy_stratum = "STABLE_GROUND_LOW_FOOTPRINT_PROXY" if "stable_ground" in pool_file.name.lower() else "PARALLAX_DENSE_LARGE_FOOTPRINT_PROXY"
        for row in pool_rows:
            image0 = image1 = None
            for method in ("SIFT", "ALIKED_LIGHTGLUE"):
                t0 = time.perf_counter()
                try:
                    if method == "SIFT":
                        meta, points0, points1, image0, image1 = _sift(row["source_pixel_path"], row["target_pixel_path"])
                    else:
                        meta, points0, points1, image0, image1 = _lightglue(extractor, matcher, row["source_pixel_path"], row["target_pixel_path"], device)
                    width, height = image0.shape[1], image0.shape[0]
                    scored = _score_matches(points0, points1, width, height)
                    status = scored["status"]
                    elapsed = time.perf_counter() - t0
                    item = {
                        "sample_id": row["sample_id"],
                        "pair_id": row["pair_id"],
                        "proxy_stratum": proxy_stratum,
                        "visual_review_status": "UNREVIEWED_CANDIDATE_ONLY",
                        "method": method,
                        "source_epoch": row["source_epoch"],
                        "source_date": row["source_date"],
                        "target_epoch": row["target_epoch"],
                        "target_date": row["target_date"],
                        "actual_interval_days": row["actual_interval_days"],
                        "tile_id": row["tile_id"],
                        "tile_x": row["tile_x"],
                        "tile_y": row["tile_y"],
                        "source_object_count": row["source_object_count"],
                        "target_object_count": row["target_object_count"],
                        "source_max_area_m2": row["source_max_area_m2"],
                        "target_max_area_m2": row["target_max_area_m2"],
                        "n_keypoints_source": meta.get("n_keypoints0"),
                        "n_keypoints_target": meta.get("n_keypoints1"),
                        "match_count": scored["match_count"],
                        "inlier_count": scored["inlier_count"],
                        "inlier_ratio": scored["inlier_ratio"],
                        "spatial_coverage_4x4": scored["spatial_coverage"],
                        "median_reprojection_residual_px": scored["median_residual_px"],
                        "p90_reprojection_residual_px": scored["p90_residual_px"],
                        "median_match_dx_px": scored["median_dx_px"],
                        "median_match_dy_px": scored["median_dy_px"],
                        "mean_match_confidence": meta.get("mean_match_confidence"),
                        "runtime_seconds": round(elapsed, 4),
                        "status": status,
                        "transform_estimator": "AFFINE_PARTIAL_RANSAC_3PX_SAME_FOR_BOTH_METHODS",
                    }
                    all_metrics.append(item)
                    group_key = (row["pair_id"], proxy_stratum, method)
                    group_status_counts[group_key][status] += 1
                    g = group_values[group_key]
                    for field in ("match_count", "inlier_ratio", "spatial_coverage", "median_residual_px", "p90_residual_px", "runtime_seconds"):
                        value = elapsed if field == "runtime_seconds" else scored.get(field)
                        if value is not None:
                            g[field].append(float(value))
                    for index, (p0, p1) in enumerate(zip(points0, points1)):
                        residual = float(scored["residuals"][index]) if index < len(scored["residuals"]) else None
                        all_points.append({
                            "sample_id": row["sample_id"], "pair_id": row["pair_id"],
                            "proxy_stratum": proxy_stratum, "method": method,
                            "match_index": index, "x_source_px": float(p0[0]), "y_source_px": float(p0[1]),
                            "x_target_px": float(p1[0]), "y_target_px": float(p1[1]),
                            "inlier": bool(scored["inlier_mask"][index]), "residual_px": residual,
                        })
                    if row["sample_id"].endswith("001"):
                        _draw_overlay(image0, image1, points0, points1, scored["inlier_mask"], VISUAL_DIR / f"{row['sample_id']}_{method}.jpg")
                except Exception as exc:
                    all_metrics.append({
                        "sample_id": row.get("sample_id"), "pair_id": row.get("pair_id"),
                        "proxy_stratum": proxy_stratum, "visual_review_status": "UNREVIEWED_CANDIDATE_ONLY",
                        "method": method, "source_date": row.get("source_date"), "target_date": row.get("target_date"),
                        "tile_id": row.get("tile_id"), "status": "FAILED_TILE_DIAGNOSTIC",
                        "error": repr(exc), "transform_estimator": "AFFINE_PARTIAL_RANSAC_3PX_SAME_FOR_BOTH_METHODS",
                    })
                    group_status_counts[(row["pair_id"], proxy_stratum, method)]["FAILED_TILE_DIAGNOSTIC"] += 1
            processed += 1
            if processed % 20 == 0:
                update_status(phase="ALIGNMENT_METHOD_PROXY_DIAGNOSTICS", current_gate="SIFT_ALIKED_LIGHTGLUE_PROXY_RUN", next_action=f"FINISH_IDENTICAL_TILE_POOL_{processed}")

    write_csv(metrics_path, all_metrics)
    write_csv(points_path, all_points)
    summary_rows = []
    for pair_id, proxy_stratum, method in sorted(set(group_values) | set(group_status_counts)):
        values = group_values[(pair_id, proxy_stratum, method)]
        statuses = group_status_counts[(pair_id, proxy_stratum, method)]
        pass_n = statuses.get("PASS_DIAGNOSTIC", 0)
        field_map = {
            "median_match_count": "match_count", "median_inlier_ratio": "inlier_ratio",
            "median_spatial_coverage_4x4": "spatial_coverage",
            "median_reprojection_residual_px": "median_residual_px",
            "median_p90_reprojection_residual_px": "p90_residual_px",
            "median_runtime_seconds": "runtime_seconds",
        }
        summary_rows.append({
            "pair_id": pair_id, "proxy_stratum": proxy_stratum, "method": method,
            "validated_stratum": False,
            "sample_n_requested": SAMPLE_COUNT_PER_PROXY_STRATUM,
            "attempted_n": sum(statuses.values()),
            "pass_diagnostic_n": pass_n,
            "diagnostic_failure_n": sum(statuses.values()) - pass_n,
            "diagnostic_failure_rate": (sum(statuses.values()) - pass_n) / sum(statuses.values()) if sum(statuses.values()) else None,
            "metric_n": len(values["match_count"]),
            **{out: statistics.median(values[key]) if values[key] else None for out, key in field_map.items()},
        })
    write_csv(summary_path, summary_rows)
    weights = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in model_paths]
    receipt = {
        "schema_version": "xian_r5.stage2.alignment_proxy_diagnostics.v1",
        "generated_at_utc": now_utc(),
        "official_lightglue_repository": "https://github.com/cvg/LightGlue",
        "lightglue_commit": "eb42fee2d71449efb0aa5c10549752b5d75384d8",
        "official_aliked_source": "https://github.com/Shiaoming/ALIKED",
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "weights": weights,
        "input_candidate_n": processed,
        "metric_rows": len(all_metrics),
        "match_point_rows": len(all_points),
        "same_samples_for_both_methods": True,
        "same_geometry_estimator": "AFFINE_PARTIAL_RANSAC_3PX",
        "stable_ground_n_validated": 0,
        "parallax_risk_n_validated": 0,
        "gate_status": "PROXY_DIAGNOSTIC_ONLY_VISUAL_LABELS_PENDING",
        "outputs": [str(metrics_path), str(points_path), str(summary_path)],
        "production_geometry_warped": False,
    }
    receipt["run_version"] = "V2_FIXED_LIGHTGLUE_LIST_OUTPUT_HANDLING"
    receipt["supersedes"] = str(METHODS_DIR / "ALIGNMENT_PROXY_DIAGNOSTIC_RECEIPT.json")
    receipt_path = METHODS_DIR / "ALIGNMENT_PROXY_DIAGNOSTIC_RECEIPT_V2.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_progress(f"SIFT/ALIKED+LightGlue proxy diagnostics completed on {processed} identical tile candidates ({len(all_metrics)} metric rows); visual strata still unvalidated.")
    update_status(phase="ALIGNMENT_METHOD_PROXY_DIAGNOSTICS_COMPLETE", sift_metrics="CANDIDATE_PROXY_ONLY", lightglue_metrics="CANDIDATE_PROXY_ONLY", stable_ground_n=0, parallax_n=0, current_gate="STABLE_AND_PARALLAX_VISUAL_STRATUM_NOT_PASSED", blocker="VISUAL_STRATUM_REVIEW_PENDING;RBF_SAM_CHECKPOINT_BLOCKED;MMDET_MODELS_IMPORT_BLOCKED_NO_MSVC_PYCOCOTOOLS", next_action="CURATE_VISUAL_STRATA_AND_RUN_IDENTITY_AB_WHERE_LABELS_EXIST")
    print(json.dumps({"status": "PROXY_DIAGNOSTICS_COMPLETE", "candidate_n": processed, "metric_rows": len(all_metrics), "point_rows": len(all_points), "summary_rows": len(summary_rows), "gate": receipt["gate_status"]}, ensure_ascii=False, indent=2))
    return 0


SAMPLE_COUNT_PER_PROXY_STRATUM = 150


if __name__ == "__main__":
    raise SystemExit(main())
