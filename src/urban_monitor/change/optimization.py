from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image

from .losses import MaskedMultiTaskLoss
from .models import build_change_model
from .splits import resolve_train_validation_test_splits
from .training import _loader, resolve_device, stage_frame, stage_loss_config
from .runtime_config import resolve_evaluation_runtime_config


def _checkpoint(config: dict, run: Path) -> Path:
    value = config.get("training", {}).get("initialization_checkpoint")
    path = Path(value) if value else run / "checkpoints" / "best_directional_change.pth"
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_model(config: dict, run: Path, device):
    import torch
    checkpoint = _checkpoint(config, run)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_change_model(payload.get("config", config)).to(device)
    model.load_state_dict(payload.get("model", payload))
    return model, checkpoint


def _runtime_config(config: dict, run: Path) -> dict:
    return resolve_evaluation_runtime_config(config, run)


def _gpu_utilization() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        return float(result.stdout.strip().splitlines()[0]) if result.returncode == 0 else None
    except Exception:
        return None


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.quantile(array, .9)),
            "min": float(array.min()), "max": float(array.max())} if len(array) else {}


def profile_training(config: dict, index_path: str | Path, run: str | Path, *, device_name: str = "cuda:0",
                     batches: int = 100) -> dict:
    """Profile real train batches without reading validation/test images."""
    import torch
    run, index_path = Path(run), Path(index_path)
    config = _runtime_config(config, run)
    device = resolve_device(device_name)
    stage = "v2_multitask" if int(config.get("version", 0) or 0) == 2 else "directional"
    frame = stage_frame(pd.read_csv(index_path, dtype=str).fillna(""), stage)
    train, _, test, _ = resolve_train_validation_test_splits(frame, config)
    loader, sampler = _loader(train, config, "directional", True, device, False)
    model, checkpoint = _load_model(config, run, device); model.train()
    criterion = MaskedMultiTaskLoss(stage_loss_config(config, "directional")).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    records, iterator, previous = [], iter(loader), time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index in range(min(int(batches), len(loader))):
        batch = next(iterator)
        batch_ready = time.perf_counter()
        timing = batch.get("_timing", [])
        host_started = time.perf_counter()
        t1 = batch["t1"].to(device, non_blocking=True); t2 = batch["t2"].to(device, non_blocking=True)
        if device.type == "cuda": torch.cuda.synchronize(device)
        host_done = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        forward_started = time.perf_counter()
        outputs = model(t1, t2)
        if device.type == "cuda": torch.cuda.synchronize(device)
        forward_done = time.perf_counter()
        loss, _ = criterion(outputs, batch)
        if device.type == "cuda": torch.cuda.synchronize(device)
        loss_done = time.perf_counter()
        loss.backward()
        if device.type == "cuda": torch.cuda.synchronize(device)
        backward_done = time.perf_counter()
        optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        optimizer_done = time.perf_counter()
        records.append({
            "batch": batch_index, "batch_size": len(batch["sample_id"]),
            "data_load_time": batch_ready - previous,
            "image_decode_time": sum(float(item.get("image_decode_time", 0)) for item in timing),
            "augmentation_time": sum(float(item.get("augmentation_time", 0)) for item in timing),
            "host_to_gpu_time": host_done - host_started, "forward_time": forward_done - forward_started,
            "loss_time": loss_done - forward_done, "backward_time": backward_done - loss_done,
            "optimizer_time": optimizer_done - backward_done, "total_batch_time": optimizer_done - previous,
            "gpu_utilization": _gpu_utilization() if batch_index % 10 == 0 else None,
            "gpu_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
            "gpu_memory_reserved": int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0,
        })
        previous = optimizer_done
    elapsed = sum(row["total_batch_time"] for row in records)
    samples = sum(row["batch_size"] for row in records)
    keys = ("data_load_time", "image_decode_time", "augmentation_time", "host_to_gpu_time", "forward_time",
            "loss_time", "backward_time", "optimizer_time", "total_batch_time")
    summary = {
        "checkpoint": str(checkpoint.resolve()), "batches": len(records), "samples": samples,
        "test_samples_held_out": int(len(test)), "test_images_read": 0,
        "timing_seconds": {key: _stats([row[key] for row in records]) for key in keys},
        "samples_per_second": samples / max(elapsed, 1e-9), "patches_per_second": samples / max(elapsed, 1e-9),
        "gpu_utilization_mean": float(np.mean([row["gpu_utilization"] for row in records if row["gpu_utilization"] is not None])) if records else None,
        "gpu_peak_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_memory_reserved": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "cache_hit_rate": loader.dataset.cache_hits / max(1, loader.dataset.cache_hits + loader.dataset.cache_misses),
    }
    mean_times = {key: value.get("mean", 0) for key, value in summary["timing_seconds"].items()}
    components = {key: mean_times[key] for key in ("data_load_time", "forward_time", "backward_time", "optimizer_time")}
    summary["primary_bottleneck"] = max(components, key=components.get)
    metrics = run / "metrics"; reports = run / "reports"; metrics.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    (metrics / "training_profile.json").write_text(json.dumps({"summary": summary, "batches": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Training Efficiency Audit", "", f"- profiled real train batches: {len(records)}",
             f"- source checkpoint: `{checkpoint}`", f"- held-out test images read: 0 / {len(test)}",
             f"- primary measured bottleneck: **{summary['primary_bottleneck']}**",
             f"- samples/sec: {summary['samples_per_second']:.3f}",
             f"- GPU utilization mean: {summary['gpu_utilization_mean']}",
             f"- GPU peak allocated: {summary['gpu_peak_memory_allocated'] / 1024**3:.3f} GiB", "",
             "## Mean seconds per batch", ""]
    lines.extend(f"- {key}: {value.get('mean', 0):.6f}" for key, value in summary["timing_seconds"].items())
    (reports / "TRAINING_EFFICIENCY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _mask_stats(path: str) -> tuple[int, int, list[int]]:
    if not path or not Path(path).is_file():
        return 0, 0, []
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    try:
        import cv2
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
    except Exception:
        areas = [int(mask.sum())] if mask.any() else []
    return int(mask.sum()), int(mask.size), areas


def audit_training_signal(config: dict, index_path: str | Path, run: str | Path, *, sample_patches: int = 500) -> dict:
    """Audit all train masks plus a deterministic stream of actual cropped patches."""
    import torch
    run = Path(run)
    config = _runtime_config(config, run)
    stage = "v2_multitask" if int(config.get("version", 0) or 0) == 2 else "directional"
    frame = stage_frame(pd.read_csv(index_path, dtype=str).fillna(""), stage)
    train, _, test, _ = resolve_train_validation_test_splits(frame, config)
    heads = {"binary": "gt_cd_path", "new": "gt_new_path", "demolition": "gt_dem_path"}
    result = {"train_images": len(train), "test_images_read": 0, "test_samples_held_out": len(test), "heads": {}}
    existing_path = run / "metrics" / "training_signal_audit.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.is_file() else {}
    if existing.get("train_images") == len(train) and existing.get("heads") and all("size_strata" in value for value in existing["heads"].values()):
        result["heads"] = existing["heads"]
    else:
        for head, column in heads.items():
            pixels, totals, components, positives = [], [], [], 0
            for path in train[column].astype(str):
                positive, total, areas = _mask_stats(path)
                pixels.append(positive); totals.append(total); components.extend(areas); positives += int(positive > 0)
            ratios = np.divide(pixels, np.maximum(totals, 1))
            quantiles = lambda values: {name: float(np.quantile(values, q)) if len(values) else 0.0 for name, q in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9))}
            component_array = np.asarray(components)
            lower, upper = (float(np.quantile(component_array, .25)), float(np.quantile(component_array, .75))) if len(component_array) else (0.0, 0.0)
            result["heads"][head] = {"positive_images": positives, "positive_image_fraction": positives / max(1, len(train)),
                                     "positive_pixel_ratio": quantiles(ratios), "component_count": len(components),
                                     "component_area_pixels": quantiles(component_array),
                                     "size_strata": {"method": "component area P25/P75", "small": int((component_array < lower).sum()),
                                                     "medium": int(((component_array >= lower) & (component_array <= upper)).sum()),
                                                     "large": int((component_array > upper).sum())}}
    device = torch.device("cpu")
    audit_config = json.loads(json.dumps(config))
    audit_config.setdefault("training", {})["batch_size"] = 1
    audit_config["training"]["resolved_batch_size"] = 1
    loader, sampler = _loader(train, audit_config, "directional", True, device, False)
    if hasattr(sampler, "set_epoch"): sampler.set_epoch(0)
    counts = {"total_patches": 0, "background_patches": 0, "binary_positive_patches": 0, "new_positive_patches": 0,
              "demolition_positive_patches": 0, "both_positive_patches": 0, "hard_patches": 0, "hard_hits": 0}
    head_hard = {name: {"selected": 0, "hits": 0} for name in ("binary", "new", "demolition", "both")}
    for batch in loader:
        counts["total_patches"] += 1
        positive = {}
        for head in ("binary", "new", "demolition"):
            value = batch[head][0]
            positive[head] = bool(value is not None and (value > 0).any())
            counts[f"{head}_positive_patches"] += int(positive[head])
        counts["both_positive_patches"] += int(positive["new"] and positive["demolition"])
        counts["background_patches"] += int(not any(positive.values()))
        hard = bool(batch["hard_sample"][0]); hit = bool(batch["hard_patch_hit"][0])
        counts["hard_patches"] += int(hard); counts["hard_hits"] += int(hard and hit)
        if hard:
            target = batch["hard_target_head"][0] or batch["crop_target_head"][0] or "binary"
            head_hard.setdefault(target, {"selected": 0, "hits": 0})
            head_hard[target]["selected"] += 1; head_hard[target]["hits"] += int(hit)
        if counts["total_patches"] >= sample_patches: break
    total = max(1, counts["total_patches"])
    result["epoch_patch_sample"] = {**counts, "background_fraction": counts["background_patches"] / total,
                                    "hard_patch_fraction": counts["hard_patches"] / total,
                                    "hard_patch_hit_rate": counts["hard_hits"] / max(1, counts["hard_patches"]),
                                    "hard_by_head": {head: {**value, "hit_rate": value["hits"] / max(1, value["selected"])} for head, value in head_hard.items()}}
    legacy_config = json.loads(json.dumps(audit_config))
    legacy_config.setdefault("patch_sampling", {})["enabled"] = False
    legacy_loader, legacy_sampler = _loader(train, legacy_config, "directional", True, device, False)
    if hasattr(legacy_sampler, "set_epoch"): legacy_sampler.set_epoch(0)
    legacy = {"hard_patches": 0, "hard_hits": 0}
    for index, batch in enumerate(legacy_loader):
        legacy["hard_patches"] += int(batch["hard_sample"].sum())
        legacy["hard_hits"] += int((batch["hard_sample"] & batch["hard_patch_hit"]).sum())
        if index + 1 >= sample_patches: break
    legacy["hard_patch_hit_rate"] = legacy["hard_hits"] / max(1, legacy["hard_patches"])
    result["legacy_random_crop_sample"] = legacy
    metrics = run / "metrics"; reports = run / "reports"; metrics.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    (metrics / "training_signal_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Training Signal Audit", "", f"- train images audited: {len(train)}", f"- test images read: 0 / {len(test)}", ""]
    for head, value in result["heads"].items():
        lines.extend([f"## {head}", "", f"- positive images: {value['positive_images']} ({value['positive_image_fraction']:.4%})",
                      f"- components: {value['component_count']}", f"- positive pixel ratio quantiles: `{value['positive_pixel_ratio']}`",
                      f"- component area quantiles: `{value['component_area_pixels']}`", ""])
    lines.extend(["## Cropped patch sample", "", f"- patches: {counts['total_patches']}",
                  f"- background fraction: {result['epoch_patch_sample']['background_fraction']:.4%}",
                  f"- hard patch fraction: {result['epoch_patch_sample']['hard_patch_fraction']:.4%}",
                  f"- hard patch hit rate: {result['epoch_patch_sample']['hard_patch_hit_rate']:.4%}"])
    (reports / "TRAINING_SIGNAL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def audit_gradients(config: dict, index_path: str | Path, run: str | Path, *, device_name: str = "cuda:0",
                    batches: int = 50) -> pd.DataFrame:
    """Measure task-gradient cosine similarity on shared parameters."""
    import torch
    run = Path(run); config = _runtime_config(config, run); device = resolve_device(device_name)
    stage = "v2_multitask" if int(config.get("version", 0) or 0) == 2 else "directional"
    frame = stage_frame(pd.read_csv(index_path, dtype=str).fillna(""), stage)
    train, _, test, _ = resolve_train_validation_test_splits(frame, config)
    loader, _ = _loader(train, config, stage, True, device, False)
    model, _ = _load_model(config, run, device); model.train()
    criterion = MaskedMultiTaskLoss({**stage_loss_config(config, stage), "loss_weights": {"binary": 1, "new_building": 1, "demolition": 1, "semantic": 1}}).to(device)
    shared = [(name, parameter) for name, parameter in model.named_parameters()
              if parameter.requires_grad and not any(token in name for token in ("binary_change_head", "new_building_head", "demolition_head", "semantic_", "uncertainty_head"))]
    metrics = run / "metrics"; reports = run / "reports"; metrics.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    rows = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches: break
        outputs = model(batch["t1"].to(device), batch["t2"].to(device))
        binary_indices = torch.where(batch["has_binary"].to(device))[0]
        directional_indices = torch.where(batch["has_directional"].to(device))[0]
        semantic_indices = torch.where(batch["has_semantic"].to(device))[0]
        gradients = {}
        losses = {}
        if len(binary_indices):
            target = torch.stack([batch["binary"][int(i)] for i in binary_indices.tolist()]).to(device)
            losses["binary"] = criterion._binary_per_sample(outputs["binary_change"][binary_indices], target).mean()
        if len(directional_indices):
            for task, output_name, target_name in (("new", "new_building", "new"), ("demolition", "demolition", "demolition")):
                target = torch.stack([batch[target_name][int(i)] for i in directional_indices.tolist()]).to(device)
                losses[task] = criterion._binary_per_sample(outputs[output_name][directional_indices], target).mean()
        if len(semantic_indices):
            semantic = []
            for output_name, target_name in (("semantic_t1", "semantic_t1"), ("semantic_t2", "semantic_t2")):
                target = torch.stack([batch[target_name][int(i)] for i in semantic_indices.tolist()]).to(device)
                semantic.append(criterion._semantic_per_sample(outputs[output_name][semantic_indices], target).mean())
            losses["semantic"] = torch.stack(semantic).mean()
        for task, loss in losses.items():
            grads = torch.autograd.grad(loss, [parameter for _, parameter in shared], retain_graph=True, allow_unused=True)
            gradients[task] = tuple(gradient.detach() for gradient in grads if gradient is not None)
        item = {"batch": batch_index}
        norms = {task: torch.sqrt(sum((gradient.double() ** 2).sum() for gradient in values)) for task, values in gradients.items()}
        for task, norm in norms.items(): item[f"{task}_gradient_norm"] = float(norm)
        for first, second in __import__("itertools").combinations(gradients, 2):
            dot = sum((left.double() * right.double()).sum() for left, right in zip(gradients[first], gradients[second]))
            item[f"cos_{first}_{second}"] = float(dot / (norms[first] * norms[second]).clamp_min(1e-12))
        rows.append(item)
        pd.DataFrame(rows).to_csv(metrics / "gradient_conflict.csv", index=False, encoding="utf-8-sig")
        del outputs, gradients
    output = pd.DataFrame(rows)
    output.to_csv(metrics / "gradient_conflict.csv", index=False, encoding="utf-8-sig")
    long_rows = []
    for _, item in output.iterrows():
        for column, value in item.items():
            if column.startswith("cos_"):
                first, second = column.removeprefix("cos_").split("_", 1)
                long_rows.append({"epoch": 0, "step": int(item["batch"]), "head_a": first, "head_b": second, "cosine": value, "negative": bool(value < 0)})
    pd.DataFrame(long_rows).to_csv(metrics / "gradient_cosine.csv", index=False, encoding="utf-8-sig")
    lines = ["# Gradient Conflict Audit", "", f"- representative train batches: {len(output)}", f"- held-out test images read: 0 / {len(test)}", ""]
    for column in [name for name in output.columns if name.startswith("cos_")]:
        lines.append(f"- {column}: mean={output[column].mean():.6f}, median={output[column].median():.6f}, negative_fraction={(output[column] < 0).mean():.4%}")
    (reports / "GRADIENT_CONFLICT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def benchmark_loader(config: dict, index_path: str | Path, run: str | Path, *, device_name: str = "cuda:0",
                     batch_sizes=(4, 8, 12, 16), workers=(0, 2, 4), batches: int = 20) -> pd.DataFrame:
    """Benchmark real train steps; no validation/test image is opened."""
    import torch
    run = Path(run); device = resolve_device(device_name)
    frame = stage_frame(pd.read_csv(index_path, dtype=str).fillna(""), "directional")
    train, _, test, _ = resolve_train_validation_test_splits(frame, config)
    model, _ = _load_model(config, run, device); model.train()
    criterion = MaskedMultiTaskLoss(stage_loss_config(config, "directional")).to(device)
    rows = []
    for batch_size in batch_sizes:
        for worker_count in workers:
            current = json.loads(json.dumps(config)); values = current.setdefault("training", {})
            values["batch_size"] = values["resolved_batch_size"] = int(batch_size); values["num_workers"] = int(worker_count)
            try:
                loader, _ = _loader(train, current, "directional", True, device, False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
                iterator = iter(loader)
                # Exclude Windows worker startup and CUDA warm-up from steady-state throughput.
                for _ in range(2):
                    batch = next(iterator); optimizer.zero_grad(set_to_none=True)
                    warm = model(batch["t1"].to(device, non_blocking=True), batch["t2"].to(device, non_blocking=True))
                    warm_loss, _ = criterion(warm, batch); warm_loss.backward(); optimizer.step()
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); samples = completed = 0
                for completed in range(1, batches + 1):
                    batch = next(iterator)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(batch["t1"].to(device, non_blocking=True), batch["t2"].to(device, non_blocking=True))
                    loss, _ = criterion(outputs, batch); loss.backward(); optimizer.step()
                    samples += len(batch["sample_id"])
                torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
                rows.append({"batch_size": batch_size, "num_workers": worker_count, "status": "ok",
                             "samples_per_second": samples / max(elapsed, 1e-9), "estimated_epoch_seconds": elapsed * len(train) / max(1, samples),
                             "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "test_images_read": 0})
            except (RuntimeError, OSError) as exc:
                if "out of memory" in str(exc).casefold() and device.type == "cuda": torch.cuda.empty_cache()
                rows.append({"batch_size": batch_size, "num_workers": worker_count, "status": "failed", "error": str(exc), "test_images_read": 0})
    output = pd.DataFrame(rows)
    valid = output[output["status"] == "ok"].copy()
    if not valid.empty:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        safe = valid[valid["gpu_peak_memory_bytes"] <= .85 * total_memory]
        chosen = (safe if not safe.empty else valid).sort_values("samples_per_second", ascending=False).iloc[0]
        output.attrs["recommended"] = chosen.to_dict()
    metrics = run / "metrics"; reports = run / "reports"; metrics.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    output.to_csv(metrics / "dataloader_batch_benchmark.csv", index=False, encoding="utf-8-sig")
    table = output.fillna("").to_csv(index=False)
    (reports / "DATALOADER_BATCH_BENCHMARK.md").write_text(
        "# DataLoader and Batch Benchmark\n\n```csv\n" + table + f"```\n\n- held-out test images read: 0 / {len(test)}\n",
        encoding="utf-8",
    )
    return output


def experiment_summary(output_root: str | Path, baseline_run: str = "20260721_232528") -> Path:
    root = Path(output_root); baseline = root / baseline_run
    baseline_metrics = {"binary_pr_auc": .3670, "new_pr_auc": .3936, "demolition_pr_auc": .2175}
    rows = []
    for run in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = run / "RUN_MANIFEST.json"
        if not manifest_path.is_file(): continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection_path = run / "reports" / "model_selection_directional.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
        metric = selection.get("best_finetune_epoch_metrics") or selection.get("source_validation_metrics") or {}
        log_path = run / "metrics" / "train_log_directional.csv"
        log = pd.read_csv(log_path) if log_path.is_file() else pd.DataFrame()
        completed_log = log[pd.to_numeric(log.get("epoch", pd.Series(dtype=float)), errors="coerce") >= 0] if not log.empty else log
        trainable_path = run / "logs" / "trainable_parameters_directional.json"
        trainable = json.loads(trainable_path.read_text(encoding="utf-8")) if trainable_path.is_file() else {}
        training_minutes = float((pd.to_numeric(completed_log.get("training_seconds", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                                  + pd.to_numeric(completed_log.get("validation_seconds", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) / 60) if not completed_log.empty else None
        row = {"run_id": run.name, "experiment_name": manifest.get("finetune_state", manifest.get("command", "")),
               "checkpoint_sha": manifest.get("best_checkpoint_sha256", ""), "trainable_parameters": trainable.get("trainable_parameter_count"),
               "epochs": int(len(completed_log)),
               "training_minutes": training_minutes, "learning_rate": metric.get("learning_rate") if isinstance(metric, dict) else None,
               "hard_fraction": manifest.get("demolition_sampling", {}).get("hard_fraction"), "sampling_strategy": manifest.get("finetune_state", ""),
               "status": manifest.get("status"), "abort_reason": manifest.get("early_abort_reason", ""),
               "accepted": manifest.get("model_selection_status") == "improved"}
        for head in ("binary", "new", "demolition"):
            value = metric.get(head, {}) if isinstance(metric.get(head), dict) else {}
            row[f"{head}_pr_auc"] = value.get("pr_auc", metric.get(f"val_{head}_pr_auc"))
            row[f"{head}_f1"] = value.get("f1", metric.get(f"val_{head}_f1"))
            row[f"delta_{head}"] = (row[f"{head}_pr_auc"] - baseline_metrics[f"{head}_pr_auc"]) if row[f"{head}_pr_auc"] is not None else None
        hours = (row["training_minutes"] or 0) / 60
        row["improvement_per_gpu_hour"] = row["delta_demolition"] / hours if hours else None
        rows.append(row)
    reports = root / "reports"; reports.mkdir(exist_ok=True)
    path = reports / "experiment_leaderboard.csv"; pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    project_reports = Path(__file__).resolve().parents[3] / "reports"; project_reports.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(project_reports / "experiment_leaderboard.csv", index=False, encoding="utf-8-sig")
    return path
