"""Run B2 boundary smoke checks and, only after all smoke gates pass, pilot candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_PYTHON = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
NATIVE_CONFIG = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "configs" / "Swin_T_whu_windows_fullval_eval.py"
B0 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def acquire(lock_path: Path, run: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "run_dir": str(run), "task": "building_v2_b2_boundary_ablation", "created_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
    return True


def release(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(dir_path: Path, name: str, rows: list[dict[str, str]]) -> Path:
    samples = dir_path / f"{name}_samples.txt"; master = dir_path / f"{name}.txt"
    samples.write_text("".join(f"{row['image_path']}\t**\t**\t{row['label_path']}\t**\n" for row in rows), encoding="utf-8")
    master.write_text(samples.resolve().as_posix() + "\n", encoding="utf-8")
    return master


def pick_native_positive_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """Pick a positive sample using the native Pillow/TIFF decode semantics.

    Some WHU labels are 1-valued TIFFs that Pillow decodes as a zero-valued
    bilevel image.  Using rasterio here would create a false overfit gate, so
    the gate deliberately follows the native loader's decoding path.
    """
    from PIL import Image
    import numpy as np
    for row in rows:
        with Image.open(row["label_path"]) as image:
            if float((np.asarray(image) > 0).mean()) > 0:
                return row
    raise RuntimeError("B2 smoke requires at least one native-positive WHU Train sample")


def eval_checkpoint(run: Path, candidate: str, checkpoint: Path, val_list: Path) -> dict[str, Any]:
    out = checkpoint.parent / "positive_overfit_eval"
    out.mkdir(parents=True, exist_ok=True)
    lock = PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock"
    if not acquire(lock, run):
        return {"status": "blocked_gpu_lock", "test_used": False, "test_pixel_reads": 0}
    try:
        completed = subprocess.run([str(NATIVE_PYTHON), str(PROJECT_ROOT / "tools" / "native_b2_eval_worker.py"), str(NATIVE_REPO), str(NATIVE_CONFIG), str(checkpoint), str(val_list), str(out)], capture_output=True, text=True, timeout=1800)
    finally:
        release(lock)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except Exception:
        payload = {"status": "failed", "error": {"type": "positive_overfit_eval_parse_error", "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}}
    payload.update({"candidate": candidate, "test_used": False, "test_pixel_reads": 0})
    write_json(out / "POSITIVE_OVERFIT_EVAL.json", payload)
    return payload


def invoke(run: Path, weight_name: str, weight: float, *, mode: str, max_iters: int, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> dict[str, Any]:
    work = run / ("04_smoke" if mode in {"smoke", "one_sample"} or mode.startswith("one_sample") else "05_b2_training") / weight_name / mode
    work.mkdir(parents=True, exist_ok=True)
    train_list = write_manifest(work, "train", train_rows); val_list = write_manifest(work, "validation", val_rows)
    lock = PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock"
    if not acquire(lock, run):
        return {"status": "blocked_gpu_lock", "mode": mode, "boundary_weight": weight, "test_used": False, "test_pixel_reads": 0}
    try:
        command = [str(NATIVE_PYTHON), str(PROJECT_ROOT / "tools" / "native_b2_worker.py"), str(NATIVE_REPO), str(NATIVE_CONFIG), str(B0), str(work), str(train_list), str(val_list), str(int(max_iters)), mode, str(weight)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    finally:
        release(lock)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except Exception:
        payload = {"status": "failed", "error": {"type": "native_output_parse_error", "stdout_tail": completed.stdout[-5000:], "stderr_tail": completed.stderr[-5000:], "returncode": completed.returncode}}
    payload.update({"candidate": weight_name, "boundary_weight": weight, "mode": mode, "returncode": completed.returncode, "stderr_tail": completed.stderr[-2000:], "test_used": False, "test_pixel_reads": 0})
    write_json(work / "NATIVE_RUN_RECEIPT.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "positive-overfit", "pilot", "all"), default="smoke")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--one-iters", type=int, default=100)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    registry = run / "05_b2_training" / "B1_TRAIN_SAMPLE_REGISTRY.csv"
    if not registry.is_file():
        raise SystemExit("B2 blocked: fixed B1 train registry missing")
    rows = read_rows(registry)
    if len(rows) != 256 or any(row.get("split") != "train" for row in rows):
        raise SystemExit("B2 blocked: train registry is not the fixed 256-row B1 registry")
    candidates = [("B2_L005", 0.05), ("B2_L010", 0.10), ("B2_L020", 0.20)]
    outputs = []
    positive_row = pick_native_positive_row(rows)
    existing_summary = run / "04_smoke" / "B2_SMOKE_SUMMARY.json"
    if args.stage != "positive-overfit" or not existing_summary.is_file():
        # The fixed registry is reused.  Smoke uses a deterministic prefix and
        # a native-positive one-sample gate; no easier sample is discovered.
        for name, weight in candidates:
            smoke = invoke(run, name, weight, mode="smoke", max_iters=20, train_rows=rows[:16], val_rows=rows[:8])
            outputs.append(smoke)
            one = invoke(run, name, weight, mode="one_sample", max_iters=20, train_rows=[positive_row], val_rows=[positive_row])
            outputs.append(one)
    smoke_pass = all(item.get("status") == "pass" for item in outputs) if outputs else True
    if args.stage == "positive-overfit":
        prior_candidates = []
        if existing_summary.is_file():
            prior = json.loads(existing_summary.read_text(encoding="utf-8"))
            # The top-level status also includes the corrected overfit gate and
            # therefore may be failed even when the independent 20-step smoke
            # was valid.  Reconstruct the smoke gate from its own entries.
            prior_candidates = [
                item for item in prior.get("candidates", [])
                if item.get("mode") in {"smoke", "one_sample"}
            ]
            smoke_pass = bool(prior_candidates) and all(item.get("status") == "pass" for item in prior_candidates)
            archive = run / "04_smoke" / "B2_POSITIVE_OVERFIT_SUMMARY_100.json"
            if not archive.is_file() and (run / "04_smoke" / "B2_POSITIVE_OVERFIT_SUMMARY.json").is_file():
                shutil.copy2(run / "04_smoke" / "B2_POSITIVE_OVERFIT_SUMMARY.json", archive)
        corrected = []
        for name, weight in candidates:
            mode = f"one_sample_positive_constantlr_{int(args.one_iters)}"
            train_dir = run / "04_smoke" / name / mode
            train_dir.mkdir(parents=True, exist_ok=True)
            train_list = write_manifest(train_dir, "train", [positive_row]); val_list = write_manifest(train_dir, "validation", [positive_row])
            trained = invoke(run, name, weight, mode=mode, max_iters=int(args.one_iters), train_rows=[positive_row], val_rows=[positive_row])
            checkpoint = Path(trained.get("work_dir", train_dir)) / f"iter_{int(args.one_iters)}.pth"
            if not checkpoint.is_file():
                files = sorted(Path(trained.get("work_dir", train_dir)).glob("iter_*.pth")); checkpoint = files[-1] if files else checkpoint
            evaluated = eval_checkpoint(run, name, checkpoint, val_list) if checkpoint.is_file() else {"status": "failed", "error": {"type": "checkpoint_missing"}}
            metrics = evaluated.get("metrics", {}); boundary = evaluated.get("boundary_metrics", {})
            corrected.append({"candidate": name, "training": trained, "evaluation": evaluated, "gates": {"building_iou": float(metrics.get("iou", 0.0)) >= 0.98, "boundary_f1@2": float(boundary.get("boundary_f1@2", 0.0)) >= 0.95, "finite": all(np.isfinite(float(metrics.get(k, 0.0))) for k in ("iou", "f1"))}})
        corrected_pass = all(item["training"].get("status") == "pass" and item["evaluation"].get("status") == "available" and all(item["gates"].values()) for item in corrected)
        write_json(run / "04_smoke" / "B2_POSITIVE_OVERFIT_SUMMARY.json", {"status": "pass" if corrected_pass else "failed", "candidates": corrected, "iterations": int(args.one_iters), "smoke_status": "pass" if smoke_pass else "failed", "test_used": False, "test_pixel_reads": 0})
        smoke_pass = smoke_pass and corrected_pass
        write_json(run / "04_smoke" / "B2_SMOKE_SUMMARY.json", {"status": "pass" if smoke_pass else "failed", "candidates": prior_candidates, "positive_overfit": corrected, "train_count": 16, "validation_count": 8, "test_used": False, "test_pixel_reads": 0})
        manifest_path = run / "RUN_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"run_id": run.name}
        manifest.update({
            "status": "success" if smoke_pass else "failed",
            "scientific_status": "smoke_pass" if smoke_pass else "blocked_b2_training_pipeline_failed",
            "smoke_status": "pass" if bool(prior_candidates) and all(item.get("status") == "pass" for item in prior_candidates) else "failed",
            "positive_overfit_status": "pass" if corrected_pass else "failed",
            "positive_overfit_iterations": int(args.one_iters),
            "test_used": False,
            "test_pixel_reads": 0,
        })
        write_json(manifest_path, manifest)
        print(json.dumps({"status": "positive_overfit_pass" if corrected_pass else "blocked_b2_training_pipeline_failed", "positive_overfit": corrected, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2, default=str)); return 0 if corrected_pass else 3
    write_json(run / "04_smoke" / "B2_SMOKE_SUMMARY.json", {"status": "pass" if smoke_pass else "failed", "candidates": outputs, "train_count": 16, "validation_count": 8, "one_sample_iters": 20, "test_used": False, "test_pixel_reads": 0})
    if not smoke_pass or args.stage == "smoke":
        status = "smoke_pass" if smoke_pass else "blocked_b2_training_pipeline_failed"
        write_json(run / "RUN_MANIFEST.json", {**json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8")), "status": "success" if smoke_pass else "failed", "scientific_status": status, "smoke_status": "pass" if smoke_pass else "failed", "test_used": False, "test_pixel_reads": 0})
        print(json.dumps({"status": status, "run_dir": str(run), "smoke": outputs, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2, default=str)); return 0 if smoke_pass else 3
    # Pilot is deliberately reachable only after all three smoke candidates and
    # their one-sample gates pass.
    train_rows = rows
    val_master = run / "05_b2_training" / "validation.txt"
    val_sample = run / "05_b2_training" / "validation_samples.txt"
    if not val_sample.is_file():
        raise SystemExit("B2 blocked: full Validation companion manifest missing")
    val_rows = []
    for line in val_sample.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            val_rows.append({"sample_id": f"whu_validation_{len(val_rows)}", "split": "validation", "image_path": parts[0], "label_path": parts[3]})
    pilot_outputs = []
    for name, weight in candidates:
        pilot_outputs.append(invoke(run, name, weight, mode="b2", max_iters=int(args.max_iters), train_rows=train_rows, val_rows=val_rows))
    write_json(run / "05_b2_training" / "B2_PILOT_SUMMARY.json", {"status": "available" if all(item.get("status") == "pass" for item in pilot_outputs) else "failed", "candidates": pilot_outputs, "max_iters": int(args.max_iters), "validation_count": len(val_rows), "test_used": False, "test_pixel_reads": 0})
    print(json.dumps({"status": "pilot_complete" if all(item.get("status") == "pass" for item in pilot_outputs) else "blocked_b2_training_pipeline_failed", "pilot": pilot_outputs, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2, default=str))
    return 0 if all(item.get("status") == "pass" for item in pilot_outputs) else 3


if __name__ == "__main__":
    raise SystemExit(main())
