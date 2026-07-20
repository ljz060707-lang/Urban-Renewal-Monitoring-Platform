from __future__ import annotations

from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from .augmentations import ChangePairTransform
from .datasets import PublicChangeDataset, change_collate
from .losses import MaskedMultiTaskLoss
from .models import build_change_model
from .samplers import DatasetBalancedSampler, DistributedSamplerView


STAGE_FILES = {
    "binary": "best_binary_change.pth",
    "semantic": "best_semantic_change.pth",
    "directional": "best_directional_change.pth",
    "local_finetune": "best_china_urban_change.pth",
}


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str, *, allow_cpu: bool = False):
    import torch
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is required but unavailable: torch={torch.__version__}, "
                f"torch.cuda.is_available()={torch.cuda.is_available()}, CUDA runtime={torch.version.cuda}. "
                "Verify nvidia-smi, then install a CUDA-enabled Torch build from https://pytorch.org/get-started/locally/"
            )
        return torch.device(requested)
    if requested == "cpu" and allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("Formal change training requires --device cuda:N; CPU is allowed only by change-demo")


def stage_loss_config(config: dict, stage: str) -> dict:
    weights = {"binary": 0.5, "new_building": 1.0, "demolition": 1.0, "semantic": 0.5}
    if stage == "binary":
        weights.update(new_building=0.0, demolition=0.0, semantic=0.0, binary=1.0)
    elif stage == "semantic":
        weights.update(new_building=0.0, demolition=0.0, binary=0.5, semantic=1.0)
    elif stage not in {"directional", "local_finetune"}:
        raise ValueError(f"Unknown training stage: {stage}")
    weights.update(config.get("loss", {}).get("stage_weights", {}).get(stage, {}))
    return {**config.get("loss", {}), "loss_weights": weights}


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().isin({"true", "1", "yes"})


def stage_frame(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage == "binary":
        selected = frame[_flag(frame["has_binary"])]
    elif stage == "semantic":
        selected = frame[_flag(frame["has_semantic"])]
    elif stage in {"directional", "local_finetune"}:
        selected = frame[_flag(frame["has_directional"])]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    if selected.empty:
        raise ValueError(f"No samples provide supervision for stage={stage}")
    return selected.copy()


def _worker_count(value, device) -> int:
    if value != "auto":
        return int(value)
    if platform.system() == "Windows":
        return 0
    return min(4, max(0, (os.cpu_count() or 2) // 2))


def _loader(frame, config, stage, training, device, distributed):
    import torch
    values = config.get("training", {})
    transform = ChangePairTransform(
        int(values.get("patch_size", 256)), training=training,
        max_translation_px=int(values.get("max_translation_px", 0)),
    )
    dataset = PublicChangeDataset(frame, transform=transform)
    sampler = None
    if training:
        names = frame["dataset_name"].tolist()
        sampler = DatasetBalancedSampler(
            names, values.get("dataset_sampling", values.get("dataset_weights", {})),
            num_samples=int(values.get("samples_per_epoch", len(frame))), seed=int(values.get("seed", 42)),
            mode=values.get("sampling_mode", "weighted"), minimum_per_dataset=int(values.get("minimum_per_dataset", 1)),
        )
        if distributed:
            sampler = DistributedSamplerView(sampler, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())
    workers = _worker_count(values.get("num_workers", "auto"), device)
    batch_size = int(values.get("resolved_batch_size", values.get("batch_size", 1)))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=workers,
        collate_fn=change_collate, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0, drop_last=training and len(dataset) >= batch_size,
    ), sampler


def _binary_counts(outputs, batch, threshold=0.5):
    import torch
    indices = torch.where(batch["has_binary"].to(outputs["binary_change"].device))[0]
    if not len(indices):
        return torch.zeros(3, dtype=torch.float64, device=outputs["binary_change"].device)
    target = torch.stack([batch["binary"][int(index)] for index in indices.tolist()]).to(outputs["binary_change"].device) > 0.5
    predicted = torch.sigmoid(outputs["binary_change"][indices]) >= threshold
    return torch.stack([
        (predicted & target).sum(), (predicted & ~target).sum(), (~predicted & target).sum()
    ]).to(torch.float64)


def _checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric, config, stage):
    import torch
    state_model = model.module if hasattr(model, "module") else model
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        git_commit = ""
    payload = {
        "model": state_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "best_metric": best_metric, "config": config, "stage": stage,
        "model_version": "change-teacher-v0.5.0", "git_commit": git_commit,
        "random_seed": int(config.get("training", {}).get("seed", 42)),
        "class_mapping": config.get("datasets", {}).get("second", {}).get("class_mapping", {}),
        "dataset_registry": config.get("datasets", {}),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _hardlink_or_copy(source: Path, target: Path) -> None:
    if target.exists(): target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _environment(device, amp, batch_size, workers):
    import torch
    payload = {
        "device": str(device), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "amp": amp, "batch_size": batch_size, "num_workers": workers,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload.update(gpu_name=properties.name, gpu_memory_bytes=properties.total_memory)
    return payload


def apply_local_finetune_schedule(model, optimizer, epoch: int, config: dict) -> str:
    """Freeze/unfreeze the shared encoder without rebuilding optimizer state."""
    values = config.get("local_finetune", {})
    frozen_epochs = int(values.get("freeze_encoder_epochs", 5))
    partial_epochs = int(values.get("partial_unfreeze_epochs", 10))
    base = model.module if hasattr(model, "module") else model
    encoder_modules = [base.stem, base.layer1, base.layer2, base.layer3, base.layer4]
    if epoch < frozen_epochs:
        for module in encoder_modules:
            for parameter in module.parameters(): parameter.requires_grad = False
        state, learning_rate = "head_only", float(values.get("learning_rate_head", 2e-4))
    elif epoch < frozen_epochs + partial_epochs:
        for module in encoder_modules:
            for parameter in module.parameters(): parameter.requires_grad = False
        for module in (base.layer3, base.layer4):
            for parameter in module.parameters(): parameter.requires_grad = True
        state, learning_rate = "partial_encoder", float(values.get("learning_rate_partial", 5e-5))
    else:
        for module in encoder_modules:
            for parameter in module.parameters(): parameter.requires_grad = True
        state, learning_rate = "full_model", float(values.get("learning_rate_full", 1e-5))
    for group in optimizer.param_groups: group["lr"] = learning_rate
    return state


def train_change_stage(config: dict, index_path: str | Path, run: str | Path, stage: str, *,
                       device_name: str = "cuda:0", resume: bool = False, allow_cpu: bool = False,
                       initial_checkpoint: str | Path | None = None, distributed: bool = False) -> Path:
    import torch
    run = Path(run)
    checkpoints = run / "checkpoints"
    metrics_root = run / "metrics"
    reports = run / "reports"
    for path in (checkpoints, metrics_root, reports, run / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name, allow_cpu=allow_cpu)
    values = config.setdefault("training", {})
    batch_value = values.get("batch_size", 1)
    if batch_value == "auto":
        # Resolution is bounded and recorded; probing is intentionally isolated from the training loop.
        values["resolved_batch_size"], attempts = probe_batch_size(config, index_path, stage, device)
        (run / "logs" / f"auto_batch_{stage}.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
    else:
        values["resolved_batch_size"] = int(batch_value)
    seed = int(values.get("seed", 42))
    seed_everything(seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config.get("gpu", {}).get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(config.get("gpu", {}).get("cudnn_benchmark", True))
    rank = 0
    if distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    frame = stage_frame(pd.read_csv(index_path, dtype=str).fillna(""), stage)
    train_frame = frame[frame["split"].str.casefold() == "train"]
    val_frame = frame[frame["split"].str.casefold().isin(["val", "valid", "validation", "test"])]
    if train_frame.empty or val_frame.empty:
        raise ValueError(f"stage={stage} requires non-empty train and validation splits")
    train_loader, train_sampler = _loader(train_frame, config, stage, True, device, distributed)
    val_loader, _ = _loader(val_frame, config, stage, False, device, False)
    model = build_change_model(config).to(device)
    if initial_checkpoint:
        payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=False)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    criterion = MaskedMultiTaskLoss(stage_loss_config(config, stage)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(values.get("learning_rate", 2e-4)), weight_decay=float(values.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(values.get("epochs", 30))))
    amp = bool(values.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_path, last_path = checkpoints / STAGE_FILES[stage], checkpoints / f"last_{stage}.pth"
    start_epoch, best_metric = 0, -1.0
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        (model.module if hasattr(model, "module") else model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"]); scaler.load_state_dict(payload["scaler"])
        start_epoch, best_metric = int(payload["epoch"]) + 1, float(payload["best_metric"])
    workers = train_loader.num_workers
    environment = _environment(device, amp, values["resolved_batch_size"], workers)
    if rank == 0:
        (run / "logs" / f"environment_{stage}.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    patience, stale = int(values.get("early_stopping_patience", 7)), 0
    history = []
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, int(values.get("epochs", 30))):
            finetune_state = apply_local_finetune_schedule(model, optimizer, epoch, config) if stage == "local_finetune" else "public_pretraining"
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            model.train(); train_loss = 0.0; train_batches = 0
            for batch in train_loader:
                t1 = batch["t1"].to(device, non_blocking=True); t2 = batch["t2"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    outputs = model(t1, t2); loss, _ = criterion(outputs, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(values.get("gradient_clip_norm", 1.0)))
                scaler.step(optimizer); scaler.update()
                train_loss += float(loss.detach()); train_batches += 1
            model.eval(); val_loss = 0.0; val_batches = 0; counts = torch.zeros(3, dtype=torch.float64, device=device)
            with torch.no_grad():
                for batch in val_loader:
                    t1 = batch["t1"].to(device, non_blocking=True); t2 = batch["t2"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=amp):
                        outputs = model(t1, t2); loss, _ = criterion(outputs, batch)
                    val_loss += float(loss); val_batches += 1; counts += _binary_counts(outputs, batch)
            if distributed:
                tensor = torch.tensor([train_loss, train_batches, val_loss, val_batches], dtype=torch.float64, device=device)
                torch.distributed.all_reduce(tensor); torch.distributed.all_reduce(counts)
                train_loss, train_batches, val_loss, val_batches = tensor.tolist()
            tp, fp, fn = counts.tolist()
            f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
            row = {"epoch": epoch, "train_loss": train_loss / max(1, train_batches), "val_loss": val_loss / max(1, val_batches),
                   "val_binary_f1": f1, "learning_rate": optimizer.param_groups[0]["lr"], "finetune_state": finetune_state}
            if device.type == "cuda":
                row["max_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            history.append(row); scheduler.step()
            if rank == 0:
                _checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, max(best_metric, f1), config, stage)
                if (epoch + 1) % int(values.get("save_every_epochs", 1)) == 0:
                    _hardlink_or_copy(last_path, checkpoints / f"{stage}_epoch_{epoch:03d}.pth")
                    snapshots = sorted(checkpoints.glob(f"{stage}_epoch_*.pth"))
                    keep = max(1, int(values.get("keep_epoch_checkpoints", 3)))
                    for old in snapshots[:-keep]: old.unlink()
                if f1 > best_metric:
                    best_metric, stale = f1, 0
                    _hardlink_or_copy(last_path, best_path)
                else:
                    stale += 1
                pd.DataFrame(history).to_csv(metrics_root / f"train_log_{stage}.csv", index=False)
                (metrics_root / f"train_log_{stage}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if stale >= patience:
                break
    except torch.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError("CUDA out of memory. Reduce patch_size or batch_size; auto mode tries only 16,8,4,2,1.") from exc
    if rank == 0:
        report = [f"# {stage} training", "", f"- device: {device}", f"- epochs completed: {len(history)}", f"- best binary F1: {best_metric:.6f}",
                  f"- best checkpoint: {best_path}", f"- last checkpoint: {last_path}", f"- elapsed seconds: {time.perf_counter() - started:.2f}",
                  f"- AMP: {amp}", f"- resumed: {resume and start_epoch > 0}"]
        (reports / f"training_{stage}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return best_path


def train_change_pipeline(config: dict, index_path: str | Path, run: str | Path, *, stages: list[str] | None = None,
                          device_name: str = "cuda:0", resume: bool = False, allow_cpu: bool = False,
                          distributed: bool = False) -> dict[str, Path]:
    """Train public-data stages in order and explicitly inherit the prior best checkpoint."""
    requested = stages or ["binary", "semantic", "directional"]
    order = ["binary", "semantic", "directional", "local_finetune"]
    if any(stage not in order for stage in requested):
        raise ValueError(f"Unknown stage list: {requested}")
    if requested != sorted(requested, key=order.index):
        raise ValueError(f"Stages must preserve order: {order}")
    results: dict[str, Path] = {}
    previous: Path | None = None
    run = Path(run)
    for stage in requested:
        if previous is None:
            position = order.index(stage)
            if position > 0:
                expected = run / "checkpoints" / STAGE_FILES[order[position - 1]]
                if not expected.is_file():
                    raise FileNotFoundError(f"stage={stage} requires prior checkpoint: {expected}")
                previous = expected
        best = train_change_stage(
            config, index_path, run, stage, device_name=device_name, resume=resume,
            allow_cpu=allow_cpu, initial_checkpoint=previous, distributed=distributed,
        )
        results[stage] = best
        previous = best
    return results


def probe_batch_size(config: dict, index_path: str | Path, stage: str, device) -> tuple[int, list[dict]]:
    import torch
    frame = stage_frame(pd.read_csv(index_path, dtype=str).fillna(""), stage)
    frame = frame[frame["split"].str.casefold() == "train"]
    attempts = []
    for size in (16, 8, 4, 2, 1):
        try:
            local = json.loads(json.dumps(config))
            local.setdefault("training", {})["batch_size"] = size
            local["training"]["resolved_batch_size"] = size
            loader, _ = _loader(frame, local, stage, True, device, False)
            batch = next(iter(loader))
            model = build_change_model(local).to(device)
            criterion = MaskedMultiTaskLoss(stage_loss_config(local, stage)).to(device)
            outputs = model(batch["t1"].to(device), batch["t2"].to(device)); loss, _ = criterion(outputs, batch)
            loss.backward()
            attempts.append({"batch_size": size, "status": "success"})
            del model, outputs, loss
            if device.type == "cuda": torch.cuda.empty_cache()
            return size, attempts
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).casefold() and not isinstance(exc, torch.OutOfMemoryError):
                raise
            attempts.append({"batch_size": size, "status": "oom", "error": str(exc)})
            if device.type == "cuda": torch.cuda.empty_cache()
    raise RuntimeError("No auto batch size in [16, 8, 4, 2, 1] fit available GPU memory")
