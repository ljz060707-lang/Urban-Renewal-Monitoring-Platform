from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .dataset import CocoBuildingDataset, collate_fn
from .model import build_model
from .compat import autocast_context, create_grad_scaler


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_runtime(value: int | str, *, device: torch.device, workers: bool = False) -> int:
    if value != "auto":
        return int(value)
    if workers:
        return 0 if os.name == "nt" else min(4, max((os.cpu_count() or 2) // 2, 1))
    return 1 if device.type == "cpu" else 2


def _mask_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    intersection = torch.logical_and(first, second).sum().item()
    union = torch.logical_or(first, second).sum().item()
    return float(intersection / union) if union else 0.0


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    match_iou_threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    matched_ious: list[float] = []
    dice_values: list[float] = []
    predicted_area = truth_area = 0.0
    predicted_count = truth_count = 0
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model([image.to(device) for image in images])
            for prediction, target in zip(predictions, targets):
                keep = prediction["scores"].detach().cpu() >= score_threshold
                pred_masks = prediction["masks"].detach().cpu()[keep, 0] >= mask_threshold
                truth_masks = target["masks"].detach().cpu().bool()
                predicted_count += len(pred_masks)
                truth_count += len(truth_masks)
                predicted_area += float(pred_masks.sum())
                truth_area += float(truth_masks.sum())
                candidates = []
                for pred_index, pred_mask in enumerate(pred_masks):
                    for truth_index, truth_mask in enumerate(truth_masks):
                        candidates.append((_mask_iou(pred_mask, truth_mask), pred_index, truth_index))
                used_pred: set[int] = set()
                used_truth: set[int] = set()
                for iou, pred_index, truth_index in sorted(candidates, reverse=True):
                    if iou < match_iou_threshold or pred_index in used_pred or truth_index in used_truth:
                        continue
                    used_pred.add(pred_index)
                    used_truth.add(truth_index)
                    matched_ious.append(iou)
                    dice_values.append(2 * iou / (1 + iou))
                tp += len(used_pred)
                fp += len(pred_masks) - len(used_pred)
                fn += len(truth_masks) - len(used_truth)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "object_count_error": float(predicted_count - truth_count),
        "building_area_error": float(predicted_area - truth_area),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _loader(dataset, batch_size: int, num_workers: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
    )


def train_teacher(
    coco_root: str | Path,
    output_dir: str | Path,
    *,
    epochs: int,
    batch_size: int | str,
    learning_rate: float,
    weight_decay: float,
    num_workers: int | str,
    num_classes: int = 2,
    checkpoint: str | Path | None = None,
    train_annotations: str = "annotations_train.json",
    val_annotations: str = "annotations_val.json",
    pretrained: bool = False,
    weights_path: str | Path | None = None,
    allow_pretrained_download: bool = False,
    amp: bool = True,
    gradient_clip_norm: float = 5.0,
    early_stopping_patience: int = 7,
    save_every_epochs: int = 1,
    resume: bool = False,
    seed: int = 42,
    min_size: int = 512,
    max_size: int = 1024,
    freeze_backbone: bool = False,
    device_name: str | None = None,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> Path:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    _seed_everything(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    actual_batch_size = resolve_runtime(batch_size, device=device)
    actual_workers = resolve_runtime(num_workers, device=device, workers=True)
    root = Path(coco_root)
    if not (root / train_annotations).exists() and (root / "annotations.json").exists():
        train_annotations = val_annotations = "annotations.json"
    train_dataset = CocoBuildingDataset(root, train_annotations)
    val_dataset = CocoBuildingDataset(root, val_annotations)
    if not train_dataset or not val_dataset:
        raise ValueError("Train and validation COCO datasets must both be non-empty")
    train_loader = _loader(train_dataset, actual_batch_size, actual_workers, shuffle=True, seed=seed)
    val_loader = _loader(val_dataset, 1, actual_workers, shuffle=False, seed=seed)
    model = build_model(
        num_classes=num_classes,
        checkpoint=checkpoint if checkpoint and not resume else None,
        pretrained=pretrained,
        weights_path=weights_path,
        allow_pretrained_download=allow_pretrained_download,
        min_size=min_size,
        max_size=max_size,
        freeze_backbone=freeze_backbone,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    output = Path(output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0
    resumed = False
    resume_path = Path(checkpoint) if checkpoint else checkpoints / "last.pth"
    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state.get("best_metric", -1.0))
        best_epoch = int(state.get("best_epoch", 0))
        resumed = True
    use_amp = bool(amp and device.type == "cuda")
    scaler = create_grad_scaler(device, use_amp)
    logs: list[dict[str, Any]] = []
    patience = 0
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output / "tensorboard")
    except Exception:
        writer = None
    resolved = {
        "device": str(device), "epochs": epochs, "batch_size": actual_batch_size,
        "num_workers": actual_workers, "learning_rate": learning_rate, "weight_decay": weight_decay,
        "amp": use_amp, "pretrained": pretrained, "weights_path": str(weights_path or ""),
        "resume": resume, "seed": seed, "min_size": min_size, "max_size": max_size,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            running = 0.0
            for images, targets in train_loader:
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, use_amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss: {loss.item()}")
                scaler.scale(loss).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.item())
            scheduler.step()
            metrics = evaluate_model(
                model, val_loader, device, score_threshold=score_threshold, mask_threshold=mask_threshold
            )
            mean_loss = running / max(len(train_loader), 1)
            row = {"epoch": epoch, "train_loss": mean_loss, "learning_rate": optimizer.param_groups[0]["lr"], **metrics}
            logs.append(row)
            improved = metrics["f1"] > best_metric or best_epoch == 0
            if improved:
                best_metric = metrics["f1"]
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "metrics": metrics,
                "config": resolved,
            }
            if epoch % max(save_every_epochs, 1) == 0:
                torch.save(state, checkpoints / f"epoch_{epoch:03d}.pth")
            torch.save(state, checkpoints / "last.pth")
            if improved:
                torch.save(state, checkpoints / "best.pth")
            if writer:
                writer.add_scalar("train/loss", mean_loss, epoch)
                writer.add_scalar("validation/f1", metrics["f1"], epoch)
                writer.add_scalar("validation/iou", metrics["mean_matched_iou"], epoch)
            if patience >= early_stopping_patience:
                break
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory. Reduce training.batch_size or tiling.tile_size, close other GPU jobs, then resume from last.pth."
        ) from exc
    finally:
        if writer:
            writer.close()
    if not (checkpoints / "best.pth").exists():
        if (checkpoints / "last.pth").exists():
            (checkpoints / "best.pth").write_bytes((checkpoints / "last.pth").read_bytes())
        else:
            raise RuntimeError("Training produced no checkpoint")
    fieldnames = list(logs[0]) if logs else ["epoch", "train_loss"]
    with (output / "train_log.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(logs)
    (output / "train_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    best_state = torch.load(checkpoints / "best.pth", map_location="cpu", weights_only=False)
    best_metrics = best_state.get("metrics", {})
    duration = time.perf_counter() - started
    summary = [
        "# Training summary", "", f"- Device: `{device}`",
        f"- CUDA available: `{torch.cuda.is_available()}`", f"- CUDA version: `{torch.version.cuda or 'none'}`",
        f"- Started: `{started_at}`", f"- Train images: `{len(train_dataset)}`", f"- Validation images: `{len(val_dataset)}`",
        f"- Epochs requested: `{epochs}`", f"- Best epoch: `{best_state.get('best_epoch', best_epoch)}`",
        f"- Best F1: `{float(best_metrics.get('f1', 0.0)):.6f}`",
        f"- Best IoU: `{float(best_metrics.get('mean_matched_iou', 0.0)):.6f}`",
        f"- Best checkpoint: `{checkpoints / 'best.pth'}`", f"- Last checkpoint: `{checkpoints / 'last.pth'}`",
        f"- Duration seconds: `{duration:.2f}`", f"- Pretrained weights: `{pretrained}`",
        f"- Resumed: `{resumed}`",
    ]
    (output / "training_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return checkpoints / "best.pth"
