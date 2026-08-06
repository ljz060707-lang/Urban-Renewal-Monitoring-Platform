from __future__ import annotations

from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import math

import numpy as np
import pandas as pd

from .augmentations import ChangePairTransform
from .datasets import PublicChangeDataset, change_collate
from .evaluation_cache import validation_probabilities
from .losses import MaskedMultiTaskLoss
from .models import build_change_model
from .samplers import DatasetBalancedSampler, DemolitionPatchCompositionSampler, V2DirectionalSampler, DistributedSamplerView
from .splits import resolve_train_validation_test_splits
from .supervision_v2 import TestReadGuard
from ..compat import create_grad_scaler


STAGE_FILES = {
    "binary": "best_binary_change.pth",
    "semantic": "best_semantic_change.pth",
    "directional": "best_directional_change.pth",
    "local_finetune": "best_china_urban_change.pth",
    "v2_multitask": "best_v2_multisource_change.pth",
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
    elif stage not in {"directional", "local_finetune", "v2_multitask"}:
        raise ValueError(f"Unknown training stage: {stage}")
    weights.update(config.get("loss", {}).get("stage_weights", {}).get(stage, {}))
    return {**config.get("loss", {}), "loss_weights": weights}


def configure_demolition_head_only(model) -> list[tuple[str, object]]:
    """Freeze every module except the explicit standalone demolition head."""
    base = model.module if hasattr(model, "module") else model
    if not hasattr(base, "demolition_head"):
        raise RuntimeError("demolition_head_only requires a model with a standalone demolition_head module")
    for _, parameter in base.named_parameters():
        parameter.requires_grad = False
    for parameter in base.demolition_head.parameters():
        parameter.requires_grad = True
    trainable = [(name, parameter) for name, parameter in base.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("demolition_head.") for name, _ in trainable):
        raise RuntimeError("demolition_head_only freeze assertion failed: optimizer may only contain demolition_head parameters")
    return trainable


def configure_binary_only(model) -> list[tuple[str, object]]:
    """Freeze heads with no DSIFN binary-only supervision, not the shared decoder."""
    base = model.module if hasattr(model, "module") else model
    unsupported = ("new_building_head", "demolition_head", "semantic_t1_head", "semantic_t2_head", "uncertainty_head")
    for name in unsupported:
        if not hasattr(base, name):
            raise RuntimeError(f"binary_only requires model module: {name}")
        for parameter in getattr(base, name).parameters():
            parameter.requires_grad = False
    trainable = [(name, parameter) for name, parameter in base.named_parameters() if parameter.requires_grad]
    if not trainable or any(name.startswith(unsupported) for name, _ in trainable):
        raise RuntimeError("binary_only freeze assertion failed: unsupported heads entered optimizer")
    if not any(name.startswith("binary_change_head.") for name, _ in trainable):
        raise RuntimeError("binary_only freeze assertion failed: binary_change_head is not trainable")
    return trainable


def set_demolition_head_training_mode(model) -> None:
    """Keep shared BatchNorm/Dropout immutable while enabling the target head."""
    base = model.module if hasattr(model, "module") else model
    model.eval()
    base.demolition_head.train()


def _parameter_group_hash(model, prefixes: tuple[str, ...]) -> str:
    base = model.module if hasattr(model, "module") else model
    digest = hashlib.sha256()
    for name, parameter in sorted(base.state_dict().items()):
        if name.startswith(prefixes):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().isin({"true", "1", "yes"})


def stage_frame(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage == "binary":
        selected = frame[_flag(frame["has_binary"])]
    elif stage == "semantic":
        selected = frame[_flag(frame["has_semantic"])]
    elif stage in {"directional", "local_finetune"}:
        selected = frame[_flag(frame["has_directional"])]
    elif stage == "v2_multitask":
        # V2 uses the existing masked multitask loss on every supported
        # source.  Splitting it into legacy semantic/directional stages would
        # discard SECOND train-only semantic samples from validation routing.
        selected = frame
    else:
        raise ValueError(f"Unknown stage: {stage}")
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
    augmentation = config.get("augmentation", {})
    gamma = augmentation.get("gamma", {})
    transform = ChangePairTransform(
        int(values.get("patch_size", 256)), training=training,
        max_translation_px=int(values.get("max_translation_px", 0)),
        brightness=float(augmentation.get("brightness", 0.12)),
        contrast=float(augmentation.get("contrast", 0.12)),
        gamma=(float(gamma.get("min", 1.0)), float(gamma.get("max", 1.0))),
        gaussian_blur_probability=float(augmentation.get("gaussian_blur_probability", 0.0)),
        temporal_alignment_jitter_px=(
            int(augmentation.get("temporal_alignment_jitter_px", 0)) if training else 0
        ),
        change_aware=config.get("patch_sampling", {}) if training else None,
    )
    dataset = PublicChangeDataset(
        frame, transform=transform,
        cache_mode=values.get("cache_mode", "none") if training else "none",
        cache_max_gb=float(values.get("cache_max_gb", 0.0)) if training else 0.0,
    )
    sampler = None
    if training:
        names = frame["dataset_name"].tolist()
        patch_sampling = config.get("patch_sampling", {})
        strategy = patch_sampling.get("strategy")

        if strategy in {
            "demolition_composition",
            "directional_composition",
        }:
            from PIL import Image

            def positive(path) -> bool:
                try:
                    path = str(path)

                    if not path:
                        return False

                    with Image.open(path) as image:
                        return (
                            image.convert("L").getbbox()
                            is not None
                        )

                except Exception:
                    return False

            # -------------------------------------------------
            # Legacy/global pools.
            #
            # Keep these global initially because the old
            # demolition_composition strategy expects them.
            # -------------------------------------------------

            demolition_positive = [
                index
                for index, path in enumerate(
                    frame["gt_dem_path"].astype(str)
                )
                if positive(path)
            ]

            new_positive = [
                index
                for index, path in enumerate(
                    frame["gt_new_path"].astype(str)
                )
                if positive(path)
            ]

            hard = [
                index
                for index, row in enumerate(
                    frame.to_dict("records")
                )
                if str(
                    row.get("target_head", "")
                ).casefold()
                in {"demolition", "both"}
                and str(
                    row.get("hard_reason", "")
                ).strip()
            ]

            difficult = [
                index
                for index, path in enumerate(
                    frame.get(
                        "false_positive_demolition_mask_path",
                        pd.Series(
                            "",
                            index=frame.index,
                        ),
                    ).astype(str)
                )
                if positive(path)
            ]

            if not difficult:
                demolition_set = set(
                    demolition_positive
                )

                difficult = [
                    index
                    for index in range(len(frame))
                    if index not in demolition_set
                ]

            categories = {
                "ordinary": list(range(len(frame))),
                "demolition_positive":
                    demolition_positive,
                "demolition_hard":
                    hard,
                "difficult_negative":
                    difficult,
            }

            # =================================================
            # V2 directional composition
            #
            # IMPORTANT:
            # new / demolition / background strata used by
            # V2DirectionalSampler MUST contain S2Looking only.
            #
            # SECOND still participates through its ordinary
            # dataset quota and keeps its own directional GT,
            # but it must not leak into the S2Looking quota.
            # =================================================

            if strategy == "directional_composition":

                s2looking_indices = [
                    index
                    for index, name in enumerate(
                        frame[
                            "dataset_name"
                        ].astype(str)
                    )
                    if name.casefold()
                    == "s2looking"
                ]

                s2looking_set = set(
                    s2looking_indices
                )

                demolition_positive = [
                    index
                    for index in s2looking_indices
                    if positive(
                        frame.iloc[index][
                            "gt_dem_path"
                        ]
                    )
                ]

                new_positive = [
                    index
                    for index in s2looking_indices
                    if positive(
                        frame.iloc[index][
                            "gt_new_path"
                        ]
                    )
                ]

                demolition_set = set(
                    demolition_positive
                )

                new_set = set(
                    new_positive
                )

                background = [
                    index
                    for index in s2looking_indices
                    if index not in demolition_set
                    and index not in new_set
                ]

                # Hard safety checks.
                if not all(
                    index in s2looking_set
                    for index
                    in demolition_positive
                ):
                    raise RuntimeError(
                        "Non-S2Looking sample leaked "
                        "into demolition_positive pool."
                    )

                if not all(
                    index in s2looking_set
                    for index in new_positive
                ):
                    raise RuntimeError(
                        "Non-S2Looking sample leaked "
                        "into new_positive pool."
                    )

                if not all(
                    index in s2looking_set
                    for index in background
                ):
                    raise RuntimeError(
                        "Non-S2Looking sample leaked "
                        "into background pool."
                    )

                categories.update(
                    {
                        "new_positive":
                            new_positive,
                        "demolition_positive":
                            demolition_positive,
                        "background":
                            (
                                background
                                or s2looking_indices
                            ),
                    }
                )

                sampler = V2DirectionalSampler(
                    names,
                    weights=values.get(
                        "dataset_sampling",
                        values.get(
                            "dataset_weights",
                            {},
                        ),
                    ),
                    new_positive=
                        new_positive,
                    demolition_positive=
                        demolition_positive,
                    num_samples=int(
                        values.get(
                            "samples_per_epoch",
                            len(frame),
                        )
                    ),
                    composition=
                        patch_sampling.get(
                            "composition",
                            {},
                        ),
                    seed=int(
                        values.get(
                            "seed",
                            42,
                        )
                    ),
                )

            else:
                # Legacy demolition_composition behaviour.
                sampler = (
                    DemolitionPatchCompositionSampler(
                        categories,
                        num_samples=int(
                            values.get(
                                "samples_per_epoch",
                                len(frame),
                            )
                        ),
                        composition=
                            patch_sampling.get(
                                "composition",
                                {
                                    "ordinary": 0.50,
                                    "demolition_positive":
                                        0.25,
                                    "demolition_hard":
                                        0.15,
                                    "difficult_negative":
                                        0.10,
                                },
                            ),
                        seed=int(
                            values.get(
                                "seed",
                                42,
                            )
                        ),
                    )
                )

        else:
            # Normal weighted dataset sampling.
            # This path must NOT require positive().
            sampler = DatasetBalancedSampler(
                names,
                values.get(
                    "dataset_sampling",
                    values.get(
                        "dataset_weights",
                        {},
                    ),
                ),
                num_samples=int(
                    values.get(
                        "samples_per_epoch",
                        len(frame),
                    )
                ),
                seed=int(
                    values.get(
                        "seed",
                        42,
                    )
                ),
                mode=values.get(
                    "sampling_mode",
                    "weighted",
                ),
                minimum_per_dataset=int(
                    values.get(
                        "minimum_per_dataset",
                        1,
                    )
                ),
                example_weights=pd.to_numeric(
                    frame.get(
                        "sample_weight",
                        pd.Series(
                            1.0,
                            index=frame.index,
                        ),
                    ),
                    errors="coerce",
                )
                .fillna(1.0)
                .clip(lower=0.01)
                .tolist(),
            )

    if distributed:
        if isinstance(
            sampler,
            (
                DemolitionPatchCompositionSampler,
                V2DirectionalSampler,
            ),
        ):
            raise RuntimeError(
                "composition sampler does not "
                "support distributed training"
            )

        sampler = DistributedSamplerView(
            sampler,
            num_replicas=
                torch.distributed.get_world_size(),
            rank=
                torch.distributed.get_rank(),
        )
        if distributed:
            if isinstance(sampler, (DemolitionPatchCompositionSampler, V2DirectionalSampler)):
                raise RuntimeError("demolition_composition sampler does not support distributed training")
            sampler = DistributedSamplerView(sampler, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())
    workers = _worker_count(values.get("num_workers", "auto"), device)
    batch_size = int(values.get("resolved_batch_size", values.get("batch_size", 1)))
    loader_kwargs = dict(
        dataset=dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=workers,
        collate_fn=change_collate, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0, drop_last=training and len(dataset) >= batch_size,
    )
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(values.get("prefetch_factor", 2))
    return torch.utils.data.DataLoader(**loader_kwargs), sampler


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


def _head_counts(outputs, batch, head: str, target_key: str, threshold=0.5):
    """TP/FP/FN for one directional head; no test loader is involved."""
    import torch
    indices = torch.where(batch["has_directional"].to(outputs[head].device))[0]
    if not len(indices):
        return torch.zeros(3, dtype=torch.float64, device=outputs[head].device)
    target = torch.stack([batch[target_key][int(index)] for index in indices.tolist()]).to(outputs[head].device) > .5
    predicted = torch.sigmoid(outputs[head][indices]) >= threshold
    return torch.stack([(predicted & target).sum(), (predicted & ~target).sum(), (~predicted & target).sum()]).to(torch.float64)


def _f1(counts) -> float:
    tp, fp, fn = counts.tolist()
    return float(2 * tp / max(1.0, 2 * tp + fp + fn))


def _pr_auc(probability: np.ndarray, target: np.ndarray) -> float:
    truth = target.astype(bool).ravel()
    if not truth.any(): return 0.0
    order = np.argsort(-probability.ravel(), kind="stable")
    ordered = truth[order]; hits = np.cumsum(ordered)
    precision = hits / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / ordered.sum())


def _validation_head_metrics(counts, probabilities: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    """Return an explicit unavailable status instead of concatenating no targets."""
    if not probabilities or not targets:
        return {"precision": None, "recall": None, "f1": None, "iou": None, "pr_auc": None, "status": "skipped_no_supervision"}
    tp, fp, fn = counts.tolist()
    return {
        "precision": float(tp / max(1.0, tp + fp)),
        "recall": float(tp / max(1.0, tp + fn)),
        "f1": _f1(counts),
        "iou": float(tp / max(1.0, tp + fp + fn)),
        "pr_auc": _pr_auc(np.concatenate(probabilities), np.concatenate(targets)),
        "status": "available",
    }


def _semantic_validation_metrics(predictions, targets, ignore_index: int) -> dict:
    if not predictions:
        return {"pixel_accuracy": None, "miou": None, "valid_pixel_count": 0, "status": "skipped_no_supervision"}
    pred, truth = np.concatenate(predictions).astype(np.int64), np.concatenate(targets).astype(np.int64)
    valid = truth != int(ignore_index)
    if not valid.any(): return {"pixel_accuracy": None, "miou": None, "valid_pixel_count": 0, "status": "skipped_no_supervision"}
    pred, truth = pred[valid], truth[valid]
    ious = []
    for class_id in sorted(set(pred.tolist()) | set(truth.tolist())):
        union = np.logical_or(pred == class_id, truth == class_id).sum()
        if union: ious.append(np.logical_and(pred == class_id, truth == class_id).sum() / union)
    return {"pixel_accuracy": float((pred == truth).mean()), "miou": float(np.mean(ious)) if ious else None,
            "valid_pixel_count": int(valid.sum()), "status": "available"}


def _checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_metric, config, stage, supervision_summary):
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
        "supervision_summary": supervision_summary,
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
    train_frame, val_frame, test_frame, split_summary = resolve_train_validation_test_splits(frame, config)
    binary_only = stage == "binary" and not _flag(frame["has_directional"]).any() and not _flag(frame["has_semantic"]).any()
    supervision_mode = "binary_only" if binary_only else "multitask"
    supervision_summary = {"stage": stage, "supervision_mode": supervision_mode, **split_summary, "test_samples_held_out": len(test_frame)}
    train_loader, train_sampler = _loader(train_frame, config, stage, True, device, distributed)
    val_loader, _ = _loader(val_frame, config, stage, False, device, False)
    monitor_size = int(config.get("validation_monitor", {}).get("size", 0))
    if 0 < monitor_size < len(val_frame):
        monitor_frame = val_frame.assign(
            _monitor_order=val_frame["sample_id"].astype(str).map(lambda value: hashlib.sha256(f"{values.get('seed', 42)}:{value}".encode()).hexdigest())
        ).sort_values("_monitor_order").head(monitor_size).drop(columns="_monitor_order")
        monitor_loader, _ = _loader(monitor_frame, config, stage, False, device, False)
    else:
        monitor_frame, monitor_loader = val_frame, val_loader
    model = build_change_model(config).to(device)
    if initial_checkpoint:
        payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=False)
    uncertainty_trainable = True
    if str(config.get("model", {}).get("uncertainty", {}).get("mode", "")).casefold() == "derived_entropy":
        # Derived uncertainty has no supervised target.  Keep its legacy module
        # inert so it cannot silently consume gradients or be exported as GT.
        base = model.module if hasattr(model, "module") else model
        for parameter in base.uncertainty_head.parameters(): parameter.requires_grad = False
        uncertainty_trainable = False
    demolition_head_only = bool(values.get("demolition_head_only", False))
    if demolition_head_only:
        trainable_named = configure_demolition_head_only(model)
    elif binary_only:
        trainable_named = configure_binary_only(model)
    elif bool(values.get("freeze_encoder", False)):
        base = model.module if hasattr(model, "module") else model
        for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
            if hasattr(base, name):
                for parameter in getattr(base, name).parameters(): parameter.requires_grad = False
        trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    else:
        trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_named:
        raise RuntimeError("No trainable parameters configured")
    trainable_report = {
        "demolition_head_only": demolition_head_only,
        "supervision_mode": supervision_mode,
        "active_heads": ["binary"] if binary_only else ["binary", "new", "demolition", "semantic"],
        "frozen_unsupported_heads": ["new", "demolition", "semantic_t1", "semantic_t2", "uncertainty"] if binary_only else [],
        "trainable_parameter_names": [name for name, _ in trainable_named],
        "trainable_parameter_count": int(sum(parameter.numel() for _, parameter in trainable_named)),
        "frozen_parameter_count": int(sum(parameter.numel() for _, parameter in model.named_parameters() if not parameter.requires_grad)),
    }
    trainable_report["total_parameter_count"] = trainable_report["trainable_parameter_count"] + trainable_report["frozen_parameter_count"]
    trainable_report["trainable_percentage"] = 100.0 * trainable_report["trainable_parameter_count"] / max(1, trainable_report["total_parameter_count"])
    if demolition_head_only and any("binary_change_head" in name or "new_building_head" in name for name in trainable_report["trainable_parameter_names"]):
        raise RuntimeError("binary/new head appeared in demolition-only optimizer")
    frozen_hash_prefixes = {
        "encoder": ("stem.", "pool.", "layer1.", "layer2.", "layer3.", "layer4.", "bottleneck.", "dec4.", "dec3.", "dec2.", "dec1.", "full."),
        "binary_head": ("binary_change_head.",), "new_head": ("new_building_head.",),
    }
    frozen_hashes_before = {name: _parameter_group_hash(model, prefixes) for name, prefixes in frozen_hash_prefixes.items()} if demolition_head_only else {}
    if rank == 0:
        print(json.dumps(trainable_report, ensure_ascii=False))
        (run / "logs" / f"trainable_parameters_{stage}.json").write_text(json.dumps(trainable_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    loss_config = stage_loss_config(config, stage)
    if demolition_head_only:
        loss_config["loss_weights"] = {"binary": 0.0, "new_building": 0.0, "demolition": 1.0, "semantic": 0.0}
    elif binary_only:
        loss_config["loss_weights"] = {"binary": 1.0, "new_building": 0.0, "demolition": 0.0, "semantic": 0.0}
    criterion = MaskedMultiTaskLoss(loss_config).to(device)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable_named], lr=float(values.get("learning_rate", 2e-4)), weight_decay=float(values.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(values.get("epochs", 30))))
    amp = bool(values.get("amp", True)) and device.type == "cuda"
    scaler = create_grad_scaler(device, amp, init_scale=float(values.get("amp_initial_scale", 1024.0)))
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
    fixed_thresholds = config.get("model_selection", {}).get("source_thresholds", {"binary": .5, "new": .6, "demolition": .3})
    def validate_heads(loader=None):
        loader = loader or val_loader
        model.eval(); counts = {"binary": torch.zeros(3, dtype=torch.float64, device=device), "new": torch.zeros(3, dtype=torch.float64, device=device), "demolition": torch.zeros(3, dtype=torch.float64, device=device)}; probabilities = {name: [] for name in counts}; targets = {name: [] for name in counts}
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch["t1"].to(device, non_blocking=True), batch["t2"].to(device, non_blocking=True))
                counts["binary"] += _binary_counts(outputs, batch, float(fixed_thresholds["binary"]))
                counts["new"] += _head_counts(outputs, batch, "new_building", "new", float(fixed_thresholds["new"]))
                counts["demolition"] += _head_counts(outputs, batch, "demolition", "demolition", float(fixed_thresholds["demolition"]))
                for name, output_name, target_name in (("binary", "binary_change", "binary"), ("new", "new_building", "new"), ("demolition", "demolition", "demolition")):
                    valid = torch.where(batch["has_binary" if name == "binary" else "has_directional"].to(device))[0]
                    if len(valid):
                        probabilities[name].append(torch.sigmoid(outputs[output_name][valid, 0]).detach().cpu().numpy().ravel())
                        targets[name].append(torch.stack([batch[target_name][int(index)] for index in valid.tolist()]).numpy().ravel())
        return {name: _validation_head_metrics(value, probabilities[name], targets[name]) for name, value in counts.items()}

    def frozen_head_signature():
        """Return deterministic source logits for the first fixed 20 validation samples."""
        if not demolition_head_only:
            return None
        model.eval()
        collected = {"binary_change": [], "new_building": []}; count = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["t1"].to(device, non_blocking=True), batch["t2"].to(device, non_blocking=True))
                take = min(20 - count, len(batch["sample_id"]))
                for name in collected:
                    collected[name].append(outputs[name][:take].detach().float().cpu().clone())
                count += take
                if count >= 20: break
        return {name: torch.cat(values) for name, values in collected.items()} if all(values for values in collected.values()) else None

    source_cache_metadata = None
    if initial_checkpoint and stage == "directional":
        cached_values, source_cache_metadata = validation_probabilities(
            initial_checkpoint, index_path, run.parent / "cache" / "evaluation", device_name=str(device), split="validation"
        )
        source_metrics = {}
        for name, (probability, target) in cached_values.items():
            threshold = float(fixed_thresholds[name])
            truth, predicted = target.astype(bool), probability >= threshold
            counts = np.asarray([(truth & predicted).sum(), ((~truth) & predicted).sum(), (truth & (~predicted)).sum()], dtype=np.float64)
            source_metrics[name] = {"f1": _f1(counts), "pr_auc": _pr_auc(probability, target)}
    else:
        source_metrics = None
    source_signature = frozen_head_signature()
    source_monitor_metrics = validate_heads(monitor_loader) if initial_checkpoint and monitor_loader is not val_loader else source_metrics
    signature_tolerance = float(values.get("frozen_output_tolerance", 1e-6))
    output_regression: list[dict] = []
    selection = config.get("model_selection", {})
    new_tolerance = float(selection.get("new_f1_tolerance", 0.002))
    best_demolition = source_metrics["demolition"]["f1"] if source_metrics else -1.0
    if source_metrics:
        _hardlink_or_copy(Path(initial_checkpoint), best_path)
        best_metric = source_metrics["binary"]["f1"]
    history = ([{"epoch": -1, "train_loss": None, "val_loss": None, "val_binary_f1": source_metrics["binary"]["f1"], "val_new_f1": source_metrics["new"]["f1"], "val_demolition_f1": source_metrics["demolition"]["f1"], "val_binary_pr_auc": source_metrics["binary"]["pr_auc"], "val_new_pr_auc": source_metrics["new"]["pr_auc"], "val_demolition_pr_auc": source_metrics["demolition"]["pr_auc"], "learning_rate": None, "finetune_state": "source_checkpoint"}] if source_metrics else [])
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, int(values.get("epochs", 30))):
            finetune_state = (apply_local_finetune_schedule(model, optimizer, epoch, config)
                              if stage == "local_finetune"
                              else str(values.get("finetune_state", "public_pretraining")))
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            if demolition_head_only:
                # Freezing parameters alone is insufficient: train() would still
                # mutate shared BatchNorm buffers and enable shared Dropout.
                set_demolition_head_training_mode(model)
            else:
                model.train()
            train_loss = 0.0; train_batches = 0; train_samples = 0
            training_started = time.perf_counter()
            task_loss_sums = {"binary": 0.0, "new_building": 0.0, "demolition": 0.0}
            decomposition_keys = ("binary_bce", "binary_dice", "new_building_bce", "new_building_dice", "new_building_boundary", "demolition_bce", "demolition_dice", "demolition_boundary", "semantic_t1", "semantic_t2")
            decomposition_sums = {name: 0.0 for name in decomposition_keys}
            signal = {"total_patches": 0, "background_patches": 0, "binary_positive_patches": 0, "new_positive_patches": 0,
                      "demolition_positive_patches": 0, "both_positive_patches": 0, "hard_patches": 0, "hard_hits": 0}
            samples_seen = {name: 0 for name in ("dsifn", "s2looking", "second")}
            gradient_norm_sum = 0.0
            gradient_norm_count = 0
            gradient_norm_max = 0.0
            gradient_nonfinite_batches = 0

            amp_scale_start = (
                float(scaler.get_scale())
                if amp
                else 1.0
            )
            amp_scale_end = amp_scale_start
            amp_scale_min = amp_scale_start
            amp_scale_max = amp_scale_start
            amp_scale_decrease_events = 0
            for batch in train_loader:
                t1 = batch["t1"].to(device, non_blocking=True); t2 = batch["t2"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    outputs = model(t1, t2); loss, details = criterion(outputs, batch)
                scaler.scale(loss).backward()

                    # AMP gradients must be unscaled before clipping / inspection.
                scaler.unscale_(optimizer)

                trainable_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    float(values.get("gradient_clip_norm", 1.0)),
                    error_if_nonfinite=False,
                )

                gradient_norm_value = float(
                    gradient_norm.detach().item()
                    if torch.is_tensor(gradient_norm)
                    else gradient_norm
                )

                    # clip_grad_norm_ returns the PRE-CLIPPING total norm.
                    # One AMP overflow must not poison the entire epoch mean.
                if math.isfinite(gradient_norm_value):
                    gradient_norm_sum += gradient_norm_value
                    gradient_norm_count += 1
                    gradient_norm_max = max(
                        gradient_norm_max,
                        gradient_norm_value,
                    )
                else:
                    gradient_nonfinite_batches += 1

                scale_before = (
                    float(scaler.get_scale())
                    if amp
                    else 1.0
                )

                    # GradScaler skips optimizer.step() automatically when
                    # unscaled gradients contain Inf / NaN.
                scaler.step(optimizer)
                scaler.update()

                scale_after = (
                    float(scaler.get_scale())
                    if amp
                    else 1.0
                )

                amp_scale_end = scale_after
                amp_scale_min = min(
                    amp_scale_min,
                    scale_after,
                )
                amp_scale_max = max(
                    amp_scale_max,
                    scale_after,
                )

                if amp and scale_after < scale_before:
                    amp_scale_decrease_events += 1

                train_loss += float(loss.detach())
                train_batches += 1
                train_samples += len(batch["sample_id"])
                for task in task_loss_sums:
                    task_loss_sums[task] += float(details[f"loss_{task}"])
                for name in decomposition_keys:
                    value = details.get(f"loss_{name}")
                    if value is not None: decomposition_sums[name] += float(value)
                positives = {}
                for head in ("binary", "new", "demolition"):
                    positives[head] = [bool(value is not None and (value > 0).any()) for value in batch[head]]
                    signal[f"{head}_positive_patches"] += sum(positives[head])
                signal["both_positive_patches"] += sum(first and second for first, second in zip(positives["new"], positives["demolition"]))
                signal["background_patches"] += sum(not (a or b or c) for a, b, c in zip(positives["binary"], positives["new"], positives["demolition"]))
                signal["total_patches"] += len(batch["sample_id"])
                signal["hard_patches"] += int(batch["hard_sample"].sum())
                signal["hard_hits"] += int((batch["hard_sample"] & batch["hard_patch_hit"]).sum())
                for name in batch["dataset_name"]:
                    if name in samples_seen: samples_seen[name] += 1
            validation_started = time.perf_counter()
            final_epoch = epoch == int(values.get("epochs", 30)) - 1
            full_validation = final_epoch or monitor_loader is val_loader or bool(config.get("validation_monitor", {}).get("full_every_epoch", False))
            epoch_val_loader = val_loader if full_validation else monitor_loader
            model.eval(); val_loss = 0.0; val_batches = 0; counts = torch.zeros(3, dtype=torch.float64, device=device); new_counts = torch.zeros(3, dtype=torch.float64, device=device); demolition_counts = torch.zeros(3, dtype=torch.float64, device=device)
            epoch_probabilities = {name: [] for name in ("binary", "new", "demolition")}
            epoch_targets = {name: [] for name in epoch_probabilities}
            semantic_predictions, semantic_targets = {"t1": [], "t2": []}, {"t1": [], "t2": []}
            with torch.no_grad():
                for batch in epoch_val_loader:
                    t1 = batch["t1"].to(device, non_blocking=True); t2 = batch["t2"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=amp):
                        outputs = model(t1, t2); loss, _ = criterion(outputs, batch)
                    val_loss += float(loss); val_batches += 1
                    counts += _binary_counts(outputs, batch, float(fixed_thresholds["binary"]))
                    new_counts += _head_counts(outputs, batch, "new_building", "new", float(fixed_thresholds["new"]))
                    demolition_counts += _head_counts(outputs, batch, "demolition", "demolition", float(fixed_thresholds["demolition"]))
                    for name, output_name, target_name in (("binary", "binary_change", "binary"), ("new", "new_building", "new"), ("demolition", "demolition", "demolition")):
                        valid = torch.where(batch["has_binary" if name == "binary" else "has_directional"].to(device))[0]
                        if len(valid):
                            epoch_probabilities[name].append(torch.sigmoid(outputs[output_name][valid, 0]).detach().cpu().numpy().ravel())
                            epoch_targets[name].append(torch.stack([batch[target_name][int(index)] for index in valid.tolist()]).numpy().ravel())
                    semantic_valid = torch.where(batch["has_semantic"].to(device))[0]
                    if len(semantic_valid):
                        for short, output_name, target_name in (("t1", "semantic_t1", "semantic_t1"), ("t2", "semantic_t2", "semantic_t2")):
                            semantic_predictions[short].append(outputs[output_name][semantic_valid].argmax(1).detach().cpu().numpy().ravel())
                            semantic_targets[short].append(torch.stack([batch[target_name][int(index)] for index in semantic_valid.tolist()]).numpy().ravel())
            if distributed:
                tensor = torch.tensor([train_loss, train_batches, val_loss, val_batches], dtype=torch.float64, device=device)
                torch.distributed.all_reduce(tensor); torch.distributed.all_reduce(counts); torch.distributed.all_reduce(new_counts); torch.distributed.all_reduce(demolition_counts)
                train_loss, train_batches, val_loss, val_batches = tensor.tolist()
            head_metrics = {
                "binary": _validation_head_metrics(counts, epoch_probabilities["binary"], epoch_targets["binary"]),
                "new": _validation_head_metrics(new_counts, epoch_probabilities["new"], epoch_targets["new"]),
                "demolition": _validation_head_metrics(demolition_counts, epoch_probabilities["demolition"], epoch_targets["demolition"]),
            }
            semantic_ignore = int(config.get("loss", {}).get("semantic", {}).get("ignore_index", config.get("loss", {}).get("semantic_ignore_index", -100)))
            semantic_metrics = {name: _semantic_validation_metrics(semantic_predictions[name], semantic_targets[name], semantic_ignore) for name in ("t1", "t2")}
            f1, new_f1, demolition_f1 = head_metrics["binary"]["f1"], head_metrics["new"]["f1"], head_metrics["demolition"]["f1"]
            row = {"epoch": epoch, "train_loss": train_loss / max(1, train_batches), "val_loss": val_loss / max(1, val_batches),
                   "supervision_mode": supervision_mode, "val_binary_precision": head_metrics["binary"]["precision"], "val_binary_recall": head_metrics["binary"]["recall"], "val_binary_f1": f1, "val_binary_iou": head_metrics["binary"]["iou"], "val_new_f1": new_f1, "val_new_iou": head_metrics["new"]["iou"], "val_demolition_f1": demolition_f1, "val_demolition_iou": head_metrics["demolition"]["iou"], "val_binary_pr_auc": head_metrics["binary"]["pr_auc"], "val_new_pr_auc": head_metrics["new"]["pr_auc"], "val_demolition_pr_auc": head_metrics["demolition"]["pr_auc"], "learning_rate": optimizer.param_groups[0]["lr"], "finetune_state": finetune_state,
                   "validation_seconds": time.perf_counter() - validation_started}
            row.update({f"val_{head}_status": metrics["status"] for head, metrics in head_metrics.items()})
            for name, metrics in semantic_metrics.items():
                row[f"semantic_{name}_pixel_accuracy"] = metrics["pixel_accuracy"]
                row[f"semantic_{name}_miou"] = metrics["miou"]
                row[f"semantic_{name}_valid_pixel_count"] = metrics["valid_pixel_count"]
                row[f"semantic_{name}_status"] = metrics["status"]
            if semantic_predictions["t1"] and semantic_predictions["t2"]:
                p1, p2 = np.concatenate(semantic_predictions["t1"]), np.concatenate(semantic_predictions["t2"])
                y1, y2 = np.concatenate(semantic_targets["t1"]), np.concatenate(semantic_targets["t2"])
                valid = (y1 != semantic_ignore) & (y2 != semantic_ignore)
                row["semantic_transition_accuracy"] = float(((p1 != p2)[valid] == (y1 != y2)[valid]).mean()) if valid.any() else None
            else: row["semantic_transition_accuracy"] = None
            row["validation_scope"] = "full" if full_validation else "monitor"
            training_seconds = training_started and validation_started - training_started
            row.update({
                "train_loss_total": train_loss / max(1, train_batches),
                "train_loss_binary": task_loss_sums["binary"] / max(1, train_batches),
                "train_loss_new": task_loss_sums["new_building"] / max(1, train_batches),
                "train_loss_demolition": task_loss_sums["demolition"] / max(1, train_batches),
                **{f"train_loss_{name}": value / max(1, train_batches) for name, value in decomposition_sums.items()},
                "gradient_norm_total": (
                    gradient_norm_sum
                    / max(1, gradient_norm_count)
                ),

                "gradient_norm_finite_mean": (
                    gradient_norm_sum
                    / max(1, gradient_norm_count)
                ),

                "gradient_norm_max": gradient_norm_max,

                "gradient_norm_finite_batches":
                    gradient_norm_count,

                "gradient_norm_nonfinite_batches":
                    gradient_nonfinite_batches,

                "gradient_norm_nonfinite_fraction": (
                    gradient_nonfinite_batches
                    / max(1, train_batches)
                ),

                "amp_scale_start":
                    amp_scale_start,

                "amp_scale_end":
                    amp_scale_end,

                "amp_scale_min":
                    amp_scale_min,

                "amp_scale_max":
                    amp_scale_max,

                "amp_scale_decrease_events":
                    amp_scale_decrease_events,
                "uncertainty_trainable": uncertainty_trainable,
                "hard_patch_fraction": signal["hard_patches"] / max(1, signal["total_patches"]),
                "hard_patch_hit_rate": signal["hard_hits"] / max(1, signal["hard_patches"]),
                "training_seconds": training_seconds,
                "samples_per_second": train_samples / max(training_seconds, 1e-9),
                "val_loss_total": val_loss / max(1, val_batches),
                "binary_precision": head_metrics["binary"]["precision"], "binary_recall": head_metrics["binary"]["recall"], "binary_f1": f1, "binary_iou": head_metrics["binary"]["iou"], "binary_pr_auc": head_metrics["binary"]["pr_auc"],
                "new_f1": new_f1, "new_iou": head_metrics["new"]["iou"], "new_pr_auc": head_metrics["new"]["pr_auc"],
                "demolition_f1": demolition_f1, "demolition_iou": head_metrics["demolition"]["iou"], "demolition_pr_auc": head_metrics["demolition"]["pr_auc"],
            })
            for head in ("binary", "new", "demolition"):
                if epoch_probabilities[head]:
                    probabilities = np.asarray(np.concatenate(epoch_probabilities[head]), dtype=np.float64)
                    if not np.isfinite(probabilities).all():
                        raise RuntimeError(f"Non-finite validation probability for supervised head: {head}")
                    row[f"{head}_positive_prediction_rate"] = float((probabilities >= float(fixed_thresholds[head])).mean())
                    eps = 1e-7
                    clipped = np.clip(probabilities, eps, 1.0 - eps)
                    logits = np.log(clipped) - np.log1p(-clipped)
                    if not np.isfinite(logits).all():
                        raise RuntimeError(f"Non-finite validation logit summary for supervised head: {head}")
                    row[f"mean_{head}_logit"] = float(logits.mean())
                else:
                    row[f"{head}_positive_prediction_rate"] = "skipped_no_supervision"
                    row[f"mean_{head}_logit"] = "skipped_no_supervision"
            if device.type == "cuda":
                row["max_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            if source_signature is not None:
                current_signature = frozen_head_signature()
                differences = {name: float((current_signature[name] - source_signature[name]).abs().max()) for name in source_signature}
                output_regression.append({"epoch": epoch, "max_abs_logit_difference": differences})
                if any(value > signature_tolerance for value in differences.values()):
                    raise RuntimeError(f"demolition_head_only regression check failed: frozen binary/new logits changed {differences}")
            history.append(row); scheduler.step()
            if rank == 0:
                _checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, max(best_metric, f1), config, stage, supervision_summary)
                if (epoch + 1) % int(values.get("save_every_epochs", 1)) == 0:
                    _hardlink_or_copy(last_path, checkpoints / f"{stage}_epoch_{epoch:03d}.pth")
                    snapshots = sorted(checkpoints.glob(f"{stage}_epoch_*.pth"))
                    keep = max(1, int(values.get("keep_epoch_checkpoints", 3)))
                    for old in snapshots[:-keep]: old.unlink()
                if not full_validation:
                    eligible = False
                elif demolition_head_only and source_metrics:
                    eligible = (abs(f1 - source_metrics["binary"]["f1"]) < 1e-4
                                and abs(new_f1 - source_metrics["new"]["f1"]) < 1e-4
                                and demolition_f1 > source_metrics["demolition"]["f1"]
                                and head_metrics["demolition"]["pr_auc"] > source_metrics["demolition"]["pr_auc"])
                elif stage == "binary":
                    eligible = f1 is not None and f1 > best_metric
                else:
                    eligible = (not source_metrics or (f1 >= source_metrics["binary"]["f1"] and new_f1 >= source_metrics["new"]["f1"] - new_tolerance and head_metrics["binary"]["pr_auc"] >= source_metrics["binary"]["pr_auc"] and demolition_f1 > source_metrics["demolition"]["f1"]))
                selection_value = f1 if stage == "binary" else demolition_f1
                if eligible and selection_value is not None and selection_value > (best_metric if stage == "binary" else best_demolition):
                    best_metric, best_demolition, stale = f1, (selection_value if stage != "binary" else best_demolition), 0
                    _hardlink_or_copy(last_path, best_path)
                else:
                    stale += 1
                pd.DataFrame(history).to_csv(metrics_root / f"train_log_{stage}.csv", index=False)
                (metrics_root / f"train_log_{stage}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                effective = {**signal, **{f"samples_seen_{name}": count for name, count in samples_seen.items()},
                             "directional_positive_fraction": (
                                signal["new_positive_patches"]
                                + signal["demolition_positive_patches"]
                                - signal["both_positive_patches"]
                            ) / max(1, signal["total_patches"]),
                             "background_fraction": signal["background_patches"] / max(1, signal["total_patches"]),
                             "binary_positive_fraction": signal["binary_positive_patches"] / max(1, signal["total_patches"]),
                             "new_positive_fraction": signal["new_positive_patches"] / max(1, signal["total_patches"]),
                             "demolition_positive_fraction": signal["demolition_positive_patches"] / max(1, signal["total_patches"]),
                             "hard_patch_fraction": row["hard_patch_fraction"], "hard_patch_hit_rate": row["hard_patch_hit_rate"]}
                (metrics_root / f"effective_training_signal_epoch_{epoch:03d}.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
            monitor_abort = config.get("regression_abort", {})
            if not full_validation and source_monitor_metrics and bool(monitor_abort.get("enabled", True)):
                tolerance = float(monitor_abort.get("pr_auc_tolerance", 0.002))
                regressed = (head_metrics["binary"]["pr_auc"] < source_monitor_metrics["binary"]["pr_auc"] - tolerance
                             and head_metrics["new"]["pr_auc"] < source_monitor_metrics["new"]["pr_auc"] - tolerance
                             and head_metrics["demolition"]["pr_auc"] <= source_monitor_metrics["demolition"]["pr_auc"])
                if regressed:
                    (reports / f"early_abort_{stage}.json").write_text(json.dumps({"epoch": epoch, "early_abort_reason": "multihead_regression", "monitor_metrics": head_metrics, "source_monitor_metrics": source_monitor_metrics}, indent=2), encoding="utf-8")
                    break
            if stale >= patience:
                break
    except torch.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError("CUDA out of memory. Reduce patch_size or batch_size; auto mode tries only 16,8,4,2,1.") from exc
    if rank == 0:
        if demolition_head_only:
            frozen_hashes_after = {name: _parameter_group_hash(model, prefixes) for name, prefixes in frozen_hash_prefixes.items()}
            if frozen_hashes_after != frozen_hashes_before:
                raise RuntimeError(f"demolition_head_only frozen parameter hash changed: before={frozen_hashes_before}, after={frozen_hashes_after}")
            (run / "logs" / f"frozen_parameter_hashes_{stage}.json").write_text(json.dumps({"before": frozen_hashes_before, "after": frozen_hashes_after}, indent=2), encoding="utf-8")
        selection_status = "improved" if source_metrics and best_demolition > source_metrics["demolition"]["f1"] and Path(best_path).resolve() != Path(initial_checkpoint).resolve() else ("no_improvement" if source_metrics else "trained")
        constraints = ({"binary_f1_abs_delta_lt": 1e-4, "new_f1_abs_delta_lt": 1e-4, "demolition_f1_gt_source": True, "demolition_pr_auc_gt_source": True} if demolition_head_only else {"binary_f1_gte_source": True, "new_f1_tolerance": new_tolerance})
        (reports / f"model_selection_{stage}.json").write_text(json.dumps({"source_validation_metrics": source_metrics, "source_validation_cache": source_cache_metadata, "best_finetune_epoch_metrics": next((row for row in history if row.get("epoch") != -1 and row.get("val_demolition_f1") == best_demolition), None), "model_selection_status": selection_status, "selected_checkpoint": str(best_path), "constraints": constraints, "frozen_output_regression": output_regression}, ensure_ascii=False, indent=2), encoding="utf-8")
        if demolition_head_only:
            (run / "logs" / f"frozen_output_regression_{stage}.json").write_text(json.dumps({"tolerance": signature_tolerance, "checks": output_regression}, ensure_ascii=False, indent=2), encoding="utf-8")
        report = [f"# {stage} training", "", f"- device: {device}", f"- epochs completed: {len(history)}", f"- best binary F1: {best_metric:.6f}",
                  f"- best checkpoint: {best_path}", f"- last checkpoint: {last_path}", f"- elapsed seconds: {time.perf_counter() - started:.2f}",
                  f"- AMP: {amp}", f"- resumed: {resume and start_epoch > 0}",
                  f"- validation source: {split_summary['validation_source']}", f"- held-out test samples: {len(test_frame)}"]
        (reports / f"training_{stage}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        # Values come from the actual split resolver and central guard, not a
        # literal receipt. Development loaders are constructed only from the
        # train/validation frames above.
        receipt = TestReadGuard("change-all", allow_test=False).receipt()
        safety = {
            "dsifn_test_image_reads": receipt.test_images_read,
            "s2looking_test_image_reads": receipt.test_images_read,
            "second_test_image_reads": receipt.test_images_read,
            "test_records_in_index": int(len(test_frame)),
            "test_used_for_training": False,
            "test_used_for_validation": False,
            "test_used_for_threshold_selection": False,
            "test_used_for_checkpoint_selection": False,
        }
        (run / "TEST_SAFETY_AUDIT.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")
    return best_path


def train_change_pipeline(config: dict, index_path: str | Path, run: str | Path, *, stages: list[str] | None = None,
                          device_name: str = "cuda:0", resume: bool = False, allow_cpu: bool = False,
                          distributed: bool = False) -> dict[str, Path]:
    """Train public-data stages in order and explicitly inherit the prior best checkpoint."""
    requested = stages or (["v2_multitask"] if int(config.get("version", 0) or 0) == 2 else ["binary", "semantic", "directional"])
    order = ["binary", "semantic", "directional", "local_finetune", "v2_multitask"]
    if any(stage not in order for stage in requested):
        raise ValueError(f"Unknown stage list: {requested}")
    if requested != sorted(requested, key=order.index):
        raise ValueError(f"Stages must preserve order: {order}")
    results: dict[str, Path] = {}
    stage_status: dict[str, dict] = {}
    configured_initial = config.get("training", {}).get("initialization_checkpoint")
    previous: Path | None = Path(configured_initial) if configured_initial else None
    if previous is not None and not previous.is_file():
        raise FileNotFoundError(f"training.initialization_checkpoint does not exist: {previous}")
    run = Path(run)
    for stage in requested:
        if previous is None:
            position = order.index(stage)
            if position > 0:
                expected = run / "checkpoints" / STAGE_FILES[order[position - 1]]
                if not expected.is_file():
                    raise FileNotFoundError(f"stage={stage} requires prior checkpoint: {expected}")
                previous = expected
        available = stage_frame(pd.read_csv(index_path, dtype=str).fillna(""), stage)
        if available.empty:
            stage_status[stage] = {"status": "skipped_no_supervision", "checkpoint": None, "inherited_from": previous.name if previous else None}
            continue
        best = train_change_stage(
            config, index_path, run, stage, device_name=device_name, resume=resume,
            allow_cpu=allow_cpu, initial_checkpoint=previous, distributed=distributed,
        )
        results[stage] = best
        stage_status[stage] = {"status": "success", "checkpoint": str(best), "inherited_from": str(previous) if previous else None}
        previous = best
    if not results:
        raise ValueError("No requested stage has usable supervision")
    (run / "training_stage_status.json").write_text(json.dumps(stage_status, ensure_ascii=False, indent=2), encoding="utf-8")
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
