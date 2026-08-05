"""Full WHU Validation evaluator for B0/B1/B2 native checkpoints."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


def has_test(value: object) -> bool:
    lowered = str(value).lower().replace("/", "\\")
    return "\\test\\" in lowered or lowered.endswith("\\test.txt") or lowered.endswith("\\test_samples.txt")


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit("usage: native_b2_eval_worker.py repo config checkpoint val_list work_dir")
    repo, config_path, checkpoint, val_list, work_dir = map(Path, sys.argv[1:6])
    result = {"status": "failed", "split": "validation", "test_used": False, "test_pixel_reads": 0}
    try:
        if any(has_test(path) for path in (repo, config_path, checkpoint, val_list, work_dir)):
            raise RuntimeError("locked Test path rejected")
        for line in val_list.read_text(encoding="utf-8-sig").splitlines():
            if has_test(line):
                raise RuntimeError(f"locked Test path in validation list: {line}")
        sys.path.insert(0, str(repo))
        project_src = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(project_src))
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from mmengine.config import Config
        from mmengine.registry import DefaultScope
        import opencd, opencd.datasets, opencd.evaluation, opencd.models, opencd.visualization  # noqa: F401
        from opencd.registry import MODELS
        from opencd.models.decode_heads.foundation_decoder import Foundation_Decoder_swin_v1
        from native_b2_worker import register_boundary_decoder, load_b0_state
        from urban_monitor.building_foundation_v2.boundary import boundary_metrics, boundary_target, bootstrap_ci
        from urban_monitor.building_foundation_v2.object_metrics import match_objects, size_stratified_recall
        DefaultScope.get_instance("opencd", scope_name="opencd")
        register_boundary_decoder(torch, nn, F, MODELS, Foundation_Decoder_swin_v1)
        cfg = Config.fromfile(str(config_path))
        cfg.model.decode_head.type = "B2BoundaryDecoder"
        cfg.model.decode_head.boundary_width = 2
        cfg.model.decode_head.boundary_weight = 0.1
        cfg.model.decode_head.boundary_channels = 64
        cfg.work_dir = str(work_dir)
        cfg.val_dataloader.dataset.data_list = str(val_list)
        cfg.test_dataloader.dataset.data_list = str(val_list)
        cfg.val_dataloader.num_workers = 0; cfg.val_dataloader.persistent_workers = False; cfg.val_dataloader.batch_size = 1
        cfg.test_dataloader.num_workers = 0; cfg.test_dataloader.persistent_workers = False; cfg.test_dataloader.batch_size = 1
        cfg.load_from = None
        from mmengine.runner import Runner
        runner = Runner.from_cfg(cfg); runner.model.init_weights = lambda: None
        inheritance = load_b0_state(runner.model, checkpoint)
        runner.model.eval()
        is_b2 = any(key.startswith("decode_head.boundary_head.") for key in torch.load(str(checkpoint), map_location="cpu").get("state_dict", {}).keys())
        totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        rows = []
        probabilities = []
        with torch.no_grad():
            for batch in runner.test_loop.dataloader:
                outputs = runner.model.test_step(batch)
                boundary_probability = None
                if is_b2 and getattr(runner.model.decode_head, "last_boundary_logits", None) is not None:
                    boundary_probability = torch.sigmoid(runner.model.decode_head.last_boundary_logits).detach().float().cpu().numpy()
                for index, sample in enumerate(outputs):
                    probability = sample.seg_logits.data[0].detach().float().cpu().numpy()
                    target = sample.gt_sem_seg_from.data[0].detach().cpu().numpy() > 0
                    pred = probability >= 0.5
                    tp = int(np.logical_and(pred, target).sum()); fp = int(np.logical_and(pred, ~target).sum()); fn = int(np.logical_and(~pred, target).sum()); tn = int(np.logical_and(~pred, ~target).sum())
                    totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn; totals["tn"] += tn
                    if boundary_probability is not None:
                        boundary_prob = boundary_probability[index, 0]
                        boundary_pred = boundary_prob >= 0.5
                    else:
                        boundary_prob = boundary_target(pred.astype(np.uint8), width=2)
                        boundary_pred = boundary_prob > 0
                    boundary_truth = boundary_target(target.astype(np.uint8), width=2) > 0
                    bmetrics = boundary_metrics(boundary_pred, boundary_truth)
                    objects = {f"object_{threshold:.1f}": match_objects(pred, target, threshold=threshold) for threshold in (0.3, 0.5, 0.7)}
                    bands = size_stratified_recall(pred, target, iou_threshold=0.5)
                    row = {"sample_index": len(rows), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn), "predicted_positive_fraction": float(pred.mean()), "actual_positive_fraction": float(target.mean()), "probability_mean": float(probability.mean()), "probability_std": float(probability.std()), "boundary_positive_fraction": float(boundary_pred.mean()), "boundary_probability_mean": float(np.asarray(boundary_prob).mean()), "boundary_probability_std": float(np.asarray(boundary_prob).std()), "object_metrics": objects, "size_bands": bands, "boundary_metrics": bmetrics, "test_pixel_read": False}
                    rows.append(row); probabilities.append(probability.reshape(-1))
        tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        # Aggregate object metrics by image to avoid giant IoU matrices.
        object_summary = {}
        for threshold in (0.3, 0.5, 0.7):
            key = f"object_{threshold:.1f}"
            fields = {name: sum(float(row["object_metrics"][key][name]) for row in rows) for name in ("prediction_count", "target_count", "matched_count", "split_count", "merge_count", "area_bias", "area_bias_fraction")}
            fields["precision"] = fields["matched_count"] / max(1.0, fields["prediction_count"]); fields["recall"] = fields["matched_count"] / max(1.0, fields["target_count"]); fields["f1"] = 2 * fields["matched_count"] / max(1.0, fields["prediction_count"] + fields["target_count"])
            object_summary[key] = fields
        size_summary = {}
        for name in ("small", "medium", "large"):
            target_count = sum(float(row["size_bands"][name]["target_count"]) for row in rows); matched_count = sum(float(row["size_bands"][name]["matched_count"]) for row in rows)
            size_summary[name] = {"target_count": target_count, "matched_count": matched_count, "recall": matched_count / max(1.0, target_count)}
        boundary_summary = {}
        for key in ("boundary_precision@1", "boundary_recall@1", "boundary_f1@1", "boundary_precision@2", "boundary_recall@2", "boundary_f1@2", "boundary_precision@3", "boundary_recall@3", "boundary_f1@3", "mean_boundary_distance", "p90_boundary_distance", "hausdorff95", "boundary_fragmentation"):
            values = [float(row["boundary_metrics"][key]) for row in rows if np.isfinite(float(row["boundary_metrics"][key]))]
            boundary_summary[key] = float(np.mean(values)) if values else float("nan")
        metrics = {"precision": precision, "recall": recall, "f1": f1, "iou": tp / max(1, tp + fp + fn), "specificity": tn / max(1, tn + fp), "balanced_accuracy": 0.5 * (recall + tn / max(1, tn + fp)), "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn, "tn_pixels": tn, "predicted_positive_fraction": (tp + fp) / max(1, tp + fp + fn + tn), "actual_positive_fraction": (tp + fn) / max(1, tp + fp + fn + tn), "probability_mean": float(np.mean([row["probability_mean"] for row in rows])), "probability_std": float(np.mean([row["probability_std"] for row in rows])), "sample_count": len(rows)}
        result.update({"status": "available", "sample_count": len(rows), "metrics": metrics, "object_metrics": object_summary, "size_stratified_recall": size_summary, "boundary_metrics": boundary_summary, "per_sample": rows, "checkpoint": str(checkpoint), "checkpoint_inheritance": inheritance, "boundary_head_evaluated": bool(is_b2), "test_used": False, "test_pixel_reads": 0})
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "B2_VALIDATION_PER_SAMPLE.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())

