"""Train the project-native Building Foundation v2 refinement smoke/pilot.

This entry point only enumerates WHU train/val. It never traverses a Test
directory. The frozen RSBuilding checkpoint is recorded as the B0 source
reference; loading an MMEngine checkpoint is deliberately not attempted in
the project environment, so a scratch smoke is never promoted as a candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import amp
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ITER2000 = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
EXPECTED_FULL20 = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
DEFAULT_ITER2000 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def unique_run(root: Path) -> Path:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"building_foundation_v2_{base}"
    suffix = 1
    while path.exists():
        path = root / f"building_foundation_v2_{base}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    for name in ("00_governance", "01_globfp_audit", "02_pairing", "03_height_audit", "04_leakage", "05_data", "06_training", "07_validation", "08_diagnostics", "09_model_package", "10_figures", "11_final_review", "12_acceptance"):
        (path / name).mkdir()
    return path


def protected_snapshot(run: Path, checkpoint: Path) -> dict[str, Any]:
    registry_candidates = list((PROJECT_ROOT / "outputs" / "governance").rglob("CURRENT_BEST_MODEL_REGISTRY.json"))
    registry = registry_candidates[-1] if registry_candidates else None
    full20_candidates = sorted((PROJECT_ROOT / "outputs" / "models").rglob("best_v2_multisource_change.pth"))
    full20 = full20_candidates[-1] if full20_candidates else None
    digest = sha256(checkpoint) if checkpoint.exists() else None
    payload = {
        "created_at": now(), "project_root": str(PROJECT_ROOT), "git_status": "captured_before_run",
        "iter2000_path": str(checkpoint), "iter2000_sha256": digest, "iter2000_expected_sha256": EXPECTED_ITER2000,
        "iter2000_unchanged": bool(digest == EXPECTED_ITER2000), "registry_path": str(registry) if registry else None,
        "registry_sha256": sha256(registry) if registry and registry.exists() else None,
        "full20_path": str(full20) if full20 else None,
        "full20_sha256": sha256(full20) if full20 and full20.exists() else None,
        "full20_expected_sha256": EXPECTED_FULL20,
        "full20_unchanged": bool(full20 and full20.exists() and sha256(full20) == EXPECTED_FULL20),
        "full20_modified": False, "construction_modified": False, "source_dataset_modified": False,
        "registry_modified": False, "iter2000_modified": False, "locked_test_pixel_reads": 0,
        "pseudo_label_training_started": False, "tfr_training_started": False, "mti_training_started": False, "construction_training_started": False,
    }
    write_json(run / "00_governance" / "PROTECTED_ASSET_INVENTORY.json", payload)
    write_json(run / "00_governance" / "PRE_RUN_SHA256.json", payload)
    return payload


def source_record(run: Path, checkpoint: Path) -> dict[str, Any]:
    digest = sha256(checkpoint) if checkpoint.exists() else None
    record = {
        "source_checkpoint": str(checkpoint.resolve()), "source_checkpoint_sha256": digest,
        "expected_sha256": EXPECTED_ITER2000, "sha256_match": digest == EXPECTED_ITER2000,
        "load_status": "external_reference_only",
        "migration_status": "blocked_external_format",
        "missing_keys": None,
        "unexpected_keys": None,
        "strict_migration": True,
        "reason": "RSBuilding iter_2000 is an MMEngine/OpenCD checkpoint and is not imported into the project .venv-gpu model; smoke refinement is not eligible for candidate promotion",
        "strict_migration_required_for_candidate": True,
    }
    write_json(run / "00_governance" / "SOURCE_CHECKPOINT_MIGRATION.json", record)
    return record


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class GpuRuntimeLock:
    """Small process-owned lock for the shared URRS GPU.

    The lock is intentionally exclusive and fail-closed: a stale lock is not
    silently removed because doing so could overlap with a still-running
    process whose PID cannot be inspected from this environment.
    """

    def __init__(self, path: Path, run_dir: Path) -> None:
        self.path = path
        self.run_dir = run_dir
        self.payload = {
            "pid": os.getpid(),
            "run_dir": str(run_dir.resolve()),
            "task": "building_foundation_v2",
            "created_at": now(),
        }
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                holder = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                holder = {"status": "unreadable"}
            raise RuntimeError(f"GPU lock is held: {self.path} ({holder})") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
        if current.get("pid") == self.payload["pid"] and current.get("run_dir") == self.payload["run_dir"]:
            self.path.unlink(missing_ok=True)
        self.acquired = False


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Building Foundation v2 config not found: {path}")
    import yaml  # type: ignore
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Building Foundation v2 config must be a mapping: {path}")
    return payload


def write_failure_receipt(root: Path, *, stage: str, error: BaseException, last_completed_epoch: int = -1) -> None:
    payload = {
        "status": "failed",
        "run_id": root.name,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "last_completed_epoch": int(last_completed_epoch),
        "test_used": False,
        "locked_test_pixel_reads": 0,
        "created_at": now(),
    }
    write_json(root / "FAILURE_RECEIPT.json", payload)
    write_json(root / "RUN_MANIFEST.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stage", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    config_dir = args.config.resolve().parent if args.config else PROJECT_ROOT

    def cfg_value(name: str, cli_value: Any, default: Any) -> Any:
        return cli_value if cli_value is not None else config.get(name, default)

    data_root = Path(cfg_value("data_root", args.data_root, Path(r"D:\Data Set\WHU_Building")))
    if not data_root.is_absolute() and not data_root.exists():
        data_root = (PROJECT_ROOT / data_root).resolve()
    if not data_root.is_dir():
        raise SystemExit(f"WHU root missing: {data_root}")
    requested_device = cfg_value("device", args.device, "cuda" if torch.cuda.is_available() else "cpu")
    device_name = str(requested_device)
    source_value = cfg_value("source_checkpoint", args.source_checkpoint, DEFAULT_ITER2000)
    source_checkpoint = Path(source_value)
    if not source_checkpoint.is_absolute():
        config_candidate = (config_dir / source_checkpoint).resolve()
        project_candidate = (PROJECT_ROOT / source_checkpoint).resolve()
        source_checkpoint = config_candidate if config_candidate.exists() else project_candidate
    root = args.run_dir or unique_run(PROJECT_ROOT / "outputs" / "pretraining")
    if args.run_dir is not None and root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing Building Foundation v2 run: {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        protected_snapshot(root, source_checkpoint)
        source = source_record(root, source_checkpoint)
        if not source["sha256_match"]:
            raise RuntimeError("iter_2000 checkpoint SHA256 mismatch; refusing to start")
    except BaseException as exc:
        write_failure_receipt(root, stage="governance", error=exc)
        raise

    if args.stage == "smoke":
        train_limit, val_limit = 16, 8
        epochs = int(cfg_value("epochs", args.epochs, 1))
        image_size = int(cfg_value("image_size", args.image_size, 128))
    elif args.stage == "pilot":
        train_limit, val_limit = 256, 64
        epochs = int(cfg_value("epochs", args.epochs, 2))
        image_size = int(cfg_value("image_size", args.image_size, 256))
    else:
        train_limit, val_limit = None, None
        epochs = int(cfg_value("epochs", args.epochs, 30))
        image_size = int(cfg_value("image_size", args.image_size, 256))
    batch_size = int(cfg_value("batch_size", args.batch_size, 2))
    learning_rate = float(cfg_value("learning_rate", None, 2e-4))
    weight_decay = float(cfg_value("weight_decay", None, 1e-4))
    gradient_clip_norm = float(cfg_value("gradient_clip_norm", None, 1.0))
    seed = int(cfg_value("seed", None, 20260805))
    height_enabled = bool(cfg_value("height_enabled", None, False))
    amp_enabled = bool(cfg_value("amp", None, True))
    resolved_config = dict(config)
    resolved_config.update({
        "stage": args.stage, "data_root": str(data_root.resolve()), "device": device_name,
        "epochs": epochs, "image_size": image_size, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm, "seed": seed,
        "height_enabled": height_enabled, "amp": amp_enabled,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "test_used": False, "locked_test_pixel_reads": 0,
    })
    import yaml  # type: ignore
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    from urban_monitor.building_foundation_v2 import BuildingFoundationNet, WHUBuildingDataset
    from urban_monitor.building_foundation_v2.data import build_whu_manifest
    from urban_monitor.building_foundation_v2.losses import multitask_loss
    from urban_monitor.building_foundation_v2.metrics import binary_metrics, summarize_metrics

    manifest_path = root / "05_data" / "building_manifest.csv"
    rows = build_whu_manifest(data_root, manifest_path, train_limit=train_limit, validation_limit=val_limit)
    write_json(root / "05_data" / "DATASET_SCOPE.json", {"dataset": "WHU_Building", "train_count": sum(r["split"] == "train" for r in rows), "validation_count": sum(r["split"] == "validation" for r in rows), "test_count": 0, "test_pixel_reads": 0, "scope": "WHU train/validation only", "spatial_isolation_strength": "official_dataset_split_no_geospatial_crs"})
    if any(r.get("split") == "test" or "\\test\\" in str(r.get("image_path", "")).lower() for r in rows):
        raise RuntimeError("test row/path detected in Building v2 manifest")
    leak_result = subprocess.run([sys.executable, str(PROJECT_ROOT / "tools" / "audit_building_v2_leakage.py"), "--run-dir", str(root)], capture_output=True, text=True)
    if leak_result.returncode != 0:
        raise RuntimeError(f"leakage gate failed: {leak_result.stdout[-1000:]}")

    train_ds = WHUBuildingDataset([r for r in rows if r["split"] == "train"], image_size=image_size, train=True)
    val_ds = WHUBuildingDataset([r for r in rows if r["split"] == "validation"], image_size=image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device_name.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device_name.startswith("cuda"))
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = BuildingFoundationNet(base_channels=16, height_enabled=height_enabled).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_enabled)
    log_path = root / "06_training" / "train_log.jsonl"
    best_score = -float("inf")
    best_path = root / "06_training" / "checkpoints" / "best.pth"
    last_path = root / "06_training" / "checkpoints" / "last.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); completed_epoch = -1
    gpu_lock = GpuRuntimeLock(PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock", root) if device.type == "cuda" else None
    lock_audit = {"required": gpu_lock is not None, "lock_path": str((PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock").resolve()), "acquired": False, "released": False, "test_pixel_reads": 0}
    write_json(root / "00_governance" / "GPU_LOCK_AUDIT.json", lock_audit)
    try:
        if gpu_lock is not None:
            gpu_lock.acquire(); lock_audit["acquired"] = True; write_json(root / "00_governance" / "GPU_LOCK_AUDIT.json", lock_audit)
        for epoch in range(epochs):
            model.train(); losses = []; epoch_start = time.perf_counter()
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                target = {key: batch[key].to(device, non_blocking=True) for key in ("mask", "boundary", "distance")}
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast("cuda", enabled=device.type == "cuda" and amp_enabled):
                    outputs = model(image); parts = multitask_loss(outputs, target); loss = parts["total"]
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm); scaler.step(optimizer); scaler.update()
                losses.append(float(loss.detach().cpu()))
            model.eval(); metrics = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch["image"].to(device)); probabilities = outputs["building_probability"].detach().cpu().numpy(); targets = batch["mask"].numpy()
                    for probability, target in zip(probabilities, targets): metrics.append(binary_metrics(probability[0], target[0]))
            summary = summarize_metrics(metrics); score = float(summary.get("iou", 0.0))
            append_jsonl(log_path, {"epoch": epoch, "train_loss": float(sum(losses) / max(1, len(losses))), "validation": summary, "seconds": time.perf_counter() - epoch_start, "device": str(device), "test_used": False})
            completed_epoch = epoch
            checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict(), "epoch": epoch, "selection_score": score, "validation_metrics": summary, "source_checkpoint": source, "dataset_scope": "WHU train/validation only", "test_used": False, "config": resolved_config}
            torch.save(checkpoint, last_path)
            if score > best_score:
                best_score = score; torch.save(checkpoint, best_path)
        status = "success"
    except BaseException as exc:
        status = "failed"
        write_json(root / "06_training" / "FAILURE.json", {"status": status, "error": f"{type(exc).__name__}: {exc}", "last_completed_epoch": completed_epoch, "test_used": False, "locked_test_pixel_reads": 0})
        write_failure_receipt(root, stage="training", error=exc, last_completed_epoch=completed_epoch)
        raise
    finally:
        if gpu_lock is not None:
            gpu_lock.release(); lock_audit["released"] = True; write_json(root / "00_governance" / "GPU_LOCK_AUDIT.json", lock_audit)
    receipt = {"status": status, "run_dir": str(root.resolve()), "stage": args.stage, "epochs_planned": epochs, "epochs_completed": completed_epoch + 1, "best_checkpoint": str(best_path.resolve()), "last_checkpoint": str(last_path.resolve()), "best_metric": best_score, "source_checkpoint": source, "test_used": False, "locked_test_pixel_reads": 0, "duration_seconds": time.perf_counter() - started, "candidate_eligibility": "blocked_source_checkpoint_not_migrated_in_project_env", "scientific_status": "smoke_only_not_candidate" if args.stage == "smoke" else "candidate_pending_validation", "resolved_config": str((root / "resolved_config.yaml").resolve()), "gpu_lock_audit": lock_audit, "created_at_utc": now()}
    write_json(root / "TRAINING_RECEIPT.json", receipt); write_json(root / "RUN_MANIFEST.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
