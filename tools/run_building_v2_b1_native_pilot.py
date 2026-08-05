"""Run guarded native sanity checks and the bounded B1 pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITER2000 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
NATIVE_CONFIG = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "configs" / "Swin_T_whu_windows_fullval_eval.py"
NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
WHU_ROOT = Path(r"D:\Data Set\WHU_Building")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def rows(split: str) -> list[dict[str, str]]:
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation are admitted")
    folder = WHU_ROOT / ("train" if split == "train" else "val")
    image_dir, label_dir = folder / "image", folder / "label"
    images = {path.stem: path for path in image_dir.glob("*.tif")}
    labels = {path.stem: path for path in label_dir.glob("*.tif")}
    return [{"sample_id": f"whu_{split}_{stem}", "split": split, "image_path": str(images[stem].resolve()), "label_path": str(labels[stem].resolve())} for stem in sorted(set(images) & set(labels))]


def mask_stats(path: str) -> tuple[float, int, int]:
    import rasterio
    from scipy import ndimage
    with rasterio.open(path) as ds:
        mask = ds.read(1) > 0
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return float(mask.mean()), int(count), int(mask.sum())


def select_samples() -> tuple[dict[str, str], list[dict[str, str]]]:
    candidates = []
    for row in rows("train"):
        fraction, count, area = mask_stats(row["label_path"])
        if 0.05 <= fraction <= 0.60 and count >= 3:
            candidates.append((row, fraction, count, area))
    if not candidates:
        raise RuntimeError("no WHU Train sample satisfies the one-sample gate")
    candidates.sort(key=lambda item: (abs(item[1] - 0.20), item[0]["sample_id"]))
    one = candidates[0][0]
    selected = []
    for low, high in ((0.05, 0.15), (0.15, 0.30), (0.30, 0.60)):
        selected.extend([item[0] for item in candidates if low <= item[1] < high][:2])
    selected.extend(item[0] for item in sorted(candidates, key=lambda item: (-item[3], item[0]["sample_id"]))[:2])
    unique = []
    seen: set[str] = set()
    for row in selected:
        if row["sample_id"] not in seen:
            unique.append(row); seen.add(row["sample_id"])
    return one, unique[:8]


def write_manifest(path: Path, sample_rows: list[dict[str, str]]) -> None:
    # FoundationDataset expects a master list whose lines point to a
    # sample-list file (the sample-list then contains tab-separated pairs).
    specific = path.with_name(path.stem + "_samples.txt")
    with specific.open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(f"{row['image_path']}\t**\t**\t{row['label_path']}\t**\n")
    with path.open("w", encoding="utf-8") as handle:
        # FoundationDataset derives a dataset name by splitting on '/'.
        # Use a POSIX spelling for the master-list entry even on Windows.
        handle.write(specific.resolve().as_posix() + "\n")


def write_registry(path: Path, sample_rows: list[dict[str, str]], role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_path", "label_path", "role", "test_pixel_read"])
        writer.writeheader()
        for row in sample_rows:
            writer.writerow({**row, "role": role, "test_pixel_read": False})


def acquire_lock(path: Path, run: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "run_dir": str(run), "task": "building_v2_b1_native_pilot", "created_at_utc": datetime.now(timezone.utc).isoformat()}
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
    return payload


def run_native(native_python: Path, run: Path, train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], mode: str, max_iters: int) -> dict[str, Any]:
    work = run / ("03_training_sanity" if mode != "b1" else "04_b1_training") / mode
    work.mkdir(parents=True, exist_ok=True)
    train_list, val_list = work / "train.txt", work / "validation.txt"
    write_manifest(train_list, train_rows); write_manifest(val_list, val_rows)
    lock_path = PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock"
    lock = None
    try:
        lock = acquire_lock(lock_path, run)
        command = [str(native_python), str(PROJECT_ROOT / "tools" / "native_b1_worker.py"), str(NATIVE_REPO), str(NATIVE_CONFIG), str(ITER2000), str(work), str(train_list), str(val_list), str(int(max_iters)), mode]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    except FileExistsError:
        return {"status": "blocked_gpu_lock", "mode": mode, "test_used": False, "test_pixel_reads": 0}
    except Exception as exc:
        return {"status": "failed", "mode": mode, "error": {"type": type(exc).__name__, "message": str(exc)}, "test_used": False, "test_pixel_reads": 0}
    finally:
        if lock is not None and lock_path.exists():
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
        payload = {"status": "failed", "mode": mode, "error": {"type": "native_output_parse_error", "stdout_tail": completed.stdout[-6000:], "stderr_tail": completed.stderr[-6000:], "returncode": completed.returncode}}
    payload.update({"mode": mode, "returncode": completed.returncode, "stderr_tail": completed.stderr[-2000:], "test_used": False, "test_pixel_reads": 0})
    write_json(work / "NATIVE_RUN_RECEIPT.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("sanity", "b1", "all"), default="all")
    parser.add_argument("--native-python", type=Path, default=Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"))
    parser.add_argument("--max-iters", type=int, default=500)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    signoff = read_json(run / "01_label_gate" / "LABEL_SIGNOFF_GATE.json", {})
    pipeline = read_json(run / "02_pipeline_equivalence" / "PIPELINE_EQUIVALENCE_RECEIPT.json", {})
    if signoff.get("status") != "pass":
        raise SystemExit("B1 blocked: WHU label signoff is not approved")
    if pipeline.get("status") != "pass":
        raise SystemExit(f"B1 blocked: pipeline equivalence status={pipeline.get('status')}")
    one, eight = select_samples()
    write_registry(run / "03_training_sanity" / "ONE_SAMPLE_SELECTION.csv", [one], "one_sample")
    write_registry(run / "03_training_sanity" / "EIGHT_SAMPLE_SELECTION.csv", eight, "eight_sample")
    fraction, count, area = mask_stats(one["label_path"])
    write_json(run / "03_training_sanity" / "ONE_SAMPLE_SELECTION.json", {"sample_id": one["sample_id"], "positive_fraction": fraction, "object_count": count, "positive_pixels": area, "test_pixel_read": False})
    sanity: dict[str, Any] = {"status": "not_started", "test_used": False, "test_pixel_reads": 0}
    if args.stage in {"sanity", "all"}:
        one_result = run_native(args.native_python, run, [one], [one], "one_sample", 20)
        eight_result = run_native(args.native_python, run, eight, eight, "eight_sample", 40)
        sanity = {"status": "pass" if one_result.get("status") == "pass" and eight_result.get("status") == "pass" else "failed_training_pipeline_sanity", "one_sample": one_result, "eight_sample": eight_result, "test_used": False, "test_pixel_reads": 0}
        write_json(run / "03_training_sanity" / "ONE_SAMPLE_OVERFIT.json", {**one_result, "selection": {"sample_id": one["sample_id"], "positive_fraction": fraction, "object_count": count}, "gates": {"native_initialization": True, "test_pixel_reads": 0}})
        write_json(run / "03_training_sanity" / "EIGHT_SAMPLE_OVERFIT.json", {**eight_result, "sample_count": len(eight), "gates": {"native_initialization": True, "test_pixel_reads": 0}})
        write_json(run / "03_training_sanity" / "TRAINING_SANITY_RECEIPT.json", sanity)
        if sanity["status"] != "pass":
            print(json.dumps({"status": "blocked_training_pipeline_sanity_failed", **sanity}, ensure_ascii=False, indent=2)); return 3
    if args.stage in {"b1", "all"}:
        fixed = rows("train")[:256]
        write_registry(run / "04_b1_training" / "B1_TRAIN_SAMPLE_REGISTRY.csv", fixed, "b1_fixed_train_256")
        write_json(run / "04_b1_training" / "B1_OPTIMIZER_CONTRACT.json", {"source_config": str(NATIVE_CONFIG), "source_base_lr": 0.0001, "pilot_lr": 1e-5, "decoder_lr_multiplier": 1.0, "top_encoder_lr_multiplier": 0.1, "low_mid_backbone": "frozen", "weight_decay": 0.05, "gradient_clip_norm": 0.01, "scheduler_source": "native config", "test_pixel_reads": 0})
        write_json(run / "04_b1_training" / "B1_NATIVE_LOSS_CONTRACT.json", {"source_config": str(NATIVE_CONFIG), "loss_type": "BCELoss", "loss_weights": [1, 1, 1], "ignore_index": 255, "reduction": "mean", "auxiliary_losses": False, "project_losses_not_used": True, "test_pixel_reads": 0})
        b1 = run_native(args.native_python, run, fixed, rows("validation"), "b1", max(1, min(500, args.max_iters)))
        b1["initialization_checkpoint"] = str(ITER2000.resolve()); b1["initialization_checkpoint_sha256"] = sha256(ITER2000)
        write_json(run / "04_b1_training" / "B1_TRAINING_RECEIPT.json", b1)
    print(json.dumps({"run_dir": str(run), "stage": args.stage, "sanity": sanity, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
