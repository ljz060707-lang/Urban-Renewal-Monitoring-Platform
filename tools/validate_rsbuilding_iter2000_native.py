"""Re-evaluate native RSBuilding iter_2000 on WHU Validation only.

The validator uses the native MMEngine/Open-CD model and preprocessing.  It
does not call the native Test split and fails closed if a validation manifest
contains a Test path.  It records both numeric overlap and probability
distribution diagnostics because a high score under an inverted mask is not a
valid semantic reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
DEFAULT_NATIVE = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
DEFAULT_REPO = Path(r"D:\Data Set\RSBuilding\repo")
DEFAULT_CONFIG = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "configs" / "Swin_T_whu_windows_fullval_eval.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


NATIVE_VALIDATE = r'''
import json, sys, traceback, hashlib
from pathlib import Path
repo, config_path, checkpoint, work_dir, max_samples = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve(), int(sys.argv[5])
sys.path.insert(0, str(repo))
result = {"status": "failed", "split": "validation", "test_used": False, "test_pixel_reads": 0, "samples": []}
def contains_test(value):
    value = str(value).lower()
    return "\\test\\" in value or "/test/" in value or value.endswith("\\test.txt") or value.endswith("/test.txt")
try:
    import torch
    from mmengine.config import Config
    from mmengine.registry import DefaultScope
    import opencd, opencd.models, opencd.datasets, opencd.evaluation, opencd.visualization
    DefaultScope.get_instance('opencd', scope_name='opencd')
    cfg = Config.fromfile(str(config_path))
    val_list = cfg.val_dataloader.dataset.data_list
    test_list = cfg.test_dataloader.dataset.data_list
    if contains_test(val_list) or contains_test(test_list):
        raise RuntimeError(f"locked Test list in validation config: val={val_list!s} test={test_list!s}")
    lines = [line.strip() for line in Path(val_list).read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
    for line in lines:
        fields = line.split('\t')
        for field in fields[:5]:
            if contains_test(field):
                raise RuntimeError(f"locked Test path in validation manifest: {field}")
    result["validation_manifest"] = str(Path(val_list).resolve())
    result["validation_manifest_count"] = len(lines)
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(work_dir)
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    runner = __import__('mmengine.runner', fromlist=['Runner']).Runner.from_cfg(cfg)
    runner.load_or_resume()
    runner.model.eval()
    all_tp = all_fp = all_fn = all_tn = 0
    sample_rows = []
    with torch.no_grad():
        for index, batch in enumerate(runner.test_loop.dataloader):
            if max_samples > 0 and index >= max_samples:
                break
            outputs = runner.model.test_step(batch)
            for sample, data_sample in zip(outputs, batch['data_samples']):
                # Foundation decoder returns sigmoid probabilities in seg_logits;
                # channel 0 is static building for an only-building sample.
                probability = sample.seg_logits.data[0].detach().float().cpu().numpy()
                target = sample.gt_sem_seg_from.data[0].detach().cpu().numpy() > 0
                import numpy as np
                if probability.shape != target.shape:
                    raise RuntimeError(f"prediction/target shape mismatch {probability.shape} vs {target.shape}")
                if not np.isfinite(probability).all():
                    raise RuntimeError("native probability contains NaN/Inf")
                pred = probability >= 0.5
                tp = int((pred & target).sum()); fp = int((pred & ~target).sum()); fn = int((~pred & target).sum()); tn = int((~pred & ~target).sum())
                all_tp += tp; all_fp += fp; all_fn += fn; all_tn += tn
                meta = getattr(sample, 'metainfo', {})
                path = str(meta.get('img_a_path', meta.get('img_path', meta.get('ori_filename', ''))))
                sample_rows.append({"sample_index": index, "sample_id": path, "predicted_positive_fraction": float(pred.mean()), "actual_positive_fraction": float(target.mean()), "probability_mean": float(probability.mean()), "probability_std": float(probability.std()), "probability_p01": float(np.quantile(probability, .01)), "probability_p50": float(np.quantile(probability, .50)), "probability_p99": float(np.quantile(probability, .99)), "all_positive": bool(pred.all()), "all_negative": bool((~pred).all()), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    count = len(sample_rows)
    total = all_tp + all_fp + all_fn + all_tn
    precision = all_tp / max(1, all_tp + all_fp); recall = all_tp / max(1, all_tp + all_fn); f1 = 2 * precision * recall / max(1e-12, precision + recall); iou = all_tp / max(1, all_tp + all_fp + all_fn)
    import numpy as np
    result.update({"status": "available", "sample_count": count, "metrics": {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "tp_pixels": all_tp, "fp_pixels": all_fp, "fn_pixels": all_fn, "tn_pixels": all_tn, "predicted_positive_fraction": (all_tp + all_fp) / max(1, total), "actual_positive_fraction": (all_tp + all_fn) / max(1, total), "probability_mean": float(np.mean([r['probability_mean'] for r in sample_rows])), "probability_std": float(np.mean([r['probability_std'] for r in sample_rows])), "probability_p01": float(np.mean([r['probability_p01'] for r in sample_rows])), "probability_p50": float(np.mean([r['probability_p50'] for r in sample_rows])), "probability_p99": float(np.mean([r['probability_p99'] for r in sample_rows])), "all_positive_sample_fraction": float(np.mean([r['all_positive'] for r in sample_rows])), "all_negative_sample_fraction": float(np.mean([r['all_negative'] for r in sample_rows]))}, "samples": sample_rows, "test_used": False, "test_pixel_reads": 0})
except Exception as exc:
    result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
print(json.dumps(result, ensure_ascii=False, default=str))
'''


def acquire_lock(path: Path, run: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "run_dir": str(run), "task": "native_rsbuilding_validation", "created_at_utc": datetime.now(timezone.utc).isoformat()}
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--native-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all 1036 validation samples")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "07_validation"
    out.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint.exists() or sha256(args.checkpoint) != EXPECTED_SHA:
        raise SystemExit("iter_2000 checkpoint missing or SHA256 mismatch")
    if not args.config.exists():
        raise SystemExit(f"native validation config missing: {args.config}")
    lock_path = PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock"
    lock_payload = None
    try:
        lock_payload = acquire_lock(lock_path, run)
        completed = subprocess.run([str(args.native_python), "-c", NATIVE_VALIDATE, str(args.native_repo), str(args.config), str(args.checkpoint), str(out / "native_runner"), str(max(0, args.max_samples))], capture_output=True, text=True, timeout=1800)
    except FileExistsError as exc:
        raise SystemExit(f"GPU lock is held: {lock_path}") from exc
    finally:
        if lock_payload is not None:
            try:
                current = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            if current.get("pid") == os.getpid():
                lock_path.unlink(missing_ok=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except Exception:
        payload = {"status": "failed", "error": {"type": "native_output_parse_error", "stdout_tail": completed.stdout[-10000:], "stderr_tail": completed.stderr[-10000:], "returncode": completed.returncode}}
    metrics = payload.get("metrics", {})
    predicted = float(metrics.get("predicted_positive_fraction", float("nan")))
    probability_std = float(metrics.get("probability_std", float("nan")))
    actual = float(metrics.get("actual_positive_fraction", float("nan")))
    if actual >= 0.80:
        distribution_status = "label_contract_suspect"
    elif predicted >= 0.95:
        distribution_status = "collapsed_all_positive"
    elif predicted <= 0.005:
        distribution_status = "collapsed_all_negative"
    elif probability_std <= 1e-6:
        distribution_status = "collapsed_constant_probability"
    elif not all(map(lambda value: value == value and abs(value) != float("inf"), (predicted, probability_std, actual))):
        distribution_status = "invalid_nonfinite"
    else:
        distribution_status = "healthy_prediction_distribution"
    historical = {"iou": 0.9279, "f1": 0.9626, "precision": 0.9680, "recall": 0.9573}
    payload.update({"run_id": run.name, "checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint), "config_path": str(args.config.resolve()), "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_used": False, "test_pixel_reads": 0, "distribution_status": distribution_status, "candidate_eligible_distribution": distribution_status == "healthy_prediction_distribution", "historical_native_metrics": historical, "historical_metric_absolute_difference_percentage_points": {key: abs(float(metrics.get(key, float("nan"))) - value) * 100.0 for key, value in historical.items()}, "native_reproduction_gate": {"threshold_percentage_points": 0.20, "all_metric_differences_within_threshold": all(abs(float(metrics.get(key, float("nan"))) - value) * 100.0 <= 0.20 for key, value in historical.items()), "status": "pass" if all(abs(float(metrics.get(key, float("nan"))) - value) * 100.0 <= 0.20 for key, value in historical.items()) else "failed"}})
    write_json(out / "NATIVE_ITER2000_VALIDATION.json", payload)
    (out / "NATIVE_ITER2000_VALIDATION.md").write_text("# Native iter_2000 Validation\n\n" + json.dumps({k: v for k, v in payload.items() if k != "samples"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if completed.returncode == 0 and payload.get("status") == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
