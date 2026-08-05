"""Evaluate a native RSBuilding checkpoint on an explicit Validation list."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


def has_test(value: str) -> bool:
    lowered = str(value).lower().replace("/", "\\")
    return "\\test\\" in lowered or lowered.endswith("\\test.txt")


def aggregate_object_metrics(prediction: np.ndarray, target: np.ndarray, totals: dict[str, dict[str, float]]) -> None:
    """Accumulate auditable component counts without serializing matrices.

    The full per-image IoU matrix is intentionally discarded here: the
    validation receipt needs deterministic aggregate object metrics, not a
    multi-gigabyte diagnostic dump.  Matching itself is the shared project
    Hungarian implementation, so the unit-tested semantics stay identical.
    """
    import sys
    project_src = Path(__file__).resolve().parents[1] / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from urban_monitor.building_foundation_v2.object_metrics import match_objects, size_stratified_recall
    for threshold in (0.3, 0.5, 0.7):
        key = f"iou_{threshold:.1f}"
        summary = match_objects(prediction, target, threshold=threshold)
        row = totals.setdefault(key, {"prediction_count": 0.0, "target_count": 0.0, "matched_count": 0.0, "merge_count": 0.0, "split_count": 0.0, "area_prediction": 0.0, "area_target": 0.0})
        for field in row:
            row[field] += float(summary[field])
        row["unmatched_prediction_count"] = row["prediction_count"] - row["matched_count"]
        row["unmatched_target_count"] = row["target_count"] - row["matched_count"]
    # Fixed, auditable size bands (pixels) are aggregated at IoU 0.5.
    bands = size_stratified_recall(prediction, target, iou_threshold=0.5)
    size = totals.setdefault("size_bands", {})
    for band in ("small", "medium", "large"):
        row = size.setdefault(band, {"target_count": 0.0, "matched_count": 0.0})
        row["target_count"] += float(bands[band]["target_count"])
        row["matched_count"] += float(bands[band]["matched_count"])


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit("usage: native_b1_eval_worker.py repo config checkpoint val_list work_dir")
    repo, config_path, checkpoint, val_list, work_dir = map(Path, sys.argv[1:6])
    result = {"status": "failed", "split": "validation", "test_used": False, "test_pixel_reads": 0}
    try:
        if any(has_test(path) for path in (repo, config_path, checkpoint, val_list, work_dir)):
            raise RuntimeError("locked Test path rejected")
        for line in val_list.read_text(encoding="utf-8").splitlines():
            if has_test(line):
                raise RuntimeError(f"locked Test path in validation list: {line}")
        sys.path.insert(0, str(repo))
        import torch
        from mmengine.config import Config
        from mmengine.registry import DefaultScope
        import opencd, opencd.datasets, opencd.evaluation, opencd.models, opencd.visualization  # noqa: F401
        DefaultScope.get_instance("opencd", scope_name="opencd")
        cfg = Config.fromfile(str(config_path))
        cfg.load_from = str(checkpoint)
        cfg.work_dir = str(work_dir)
        cfg.val_dataloader.dataset.data_list = str(val_list)
        cfg.test_dataloader.dataset.data_list = str(val_list)
        cfg.val_dataloader.num_workers = 0; cfg.val_dataloader.persistent_workers = False; cfg.val_dataloader.batch_size = 1
        cfg.test_dataloader.num_workers = 0; cfg.test_dataloader.persistent_workers = False; cfg.test_dataloader.batch_size = 1
        from mmengine.runner import Runner
        runner = Runner.from_cfg(cfg)
        # The native backbone initializer can invoke CUDA NVRTC on this
        # Windows environment.  Evaluation must use the supplied checkpoint,
        # never random initialization; disable only that pre-load initializer
        # and then explicitly load the candidate before iterating Validation.
        runner.model.init_weights = lambda: None
        runner.load_or_resume()
        runner.model.eval()
        tp = fp = fn = tn = 0
        sample_count = 0
        probability_values = []
        object_totals: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            for batch in runner.test_loop.dataloader:
                outputs = runner.model.test_step(batch)
                for sample in outputs:
                    probability = sample.seg_logits.data[0].detach().float().cpu().numpy()
                    target = sample.gt_sem_seg_from.data[0].detach().cpu().numpy() > 0
                    if probability.shape != target.shape:
                        raise RuntimeError(f"shape mismatch {probability.shape} != {target.shape}")
                    if not np.isfinite(probability).all():
                        raise RuntimeError("probability contains NaN/Inf")
                    pred = probability >= 0.5
                    tp += int(np.logical_and(pred, target).sum()); fp += int(np.logical_and(pred, ~target).sum()); fn += int(np.logical_and(~pred, target).sum()); tn += int(np.logical_and(~pred, ~target).sum())
                    sample_count += 1
                    probability_values.append(probability.reshape(-1))
                    aggregate_object_metrics(pred, target, object_totals)
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        for key, row in object_totals.items():
            if key == "size_bands":
                continue
            row["precision"] = row["matched_count"] / max(1.0, row["prediction_count"])
            row["recall"] = row["matched_count"] / max(1.0, row["target_count"])
            row["f1"] = 2.0 * row["matched_count"] / max(1.0, row["prediction_count"] + row["target_count"])
            row["area_bias"] = row["area_prediction"] - row["area_target"]
            row["area_bias_fraction"] = row["area_bias"] / max(1.0, row["area_target"])
        for band, row in object_totals.get("size_bands", {}).items():
            row["recall"] = row["matched_count"] / max(1.0, row["target_count"])
        result.update({"status": "available", "sample_count": sample_count, "object_metrics_available": True, "object_metrics": object_totals, "metrics": {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall), "iou": tp / max(1, tp + fp + fn), "specificity": tn / max(1, tn + fp), "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn, "tn_pixels": tn, "predicted_positive_fraction": (tp + fp) / max(1, tp + fp + fn + tn), "actual_positive_fraction": (tp + fn) / max(1, tp + fp + fn + tn), "probability_mean": float(np.mean([values.mean() for values in probability_values])) if probability_values else float("nan"), "probability_std": float(np.mean([values.std() for values in probability_values])) if probability_values else float("nan")}, "checkpoint": str(checkpoint), "test_used": False, "test_pixel_reads": 0})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
