from __future__ import annotations

"""Detached UCS-386 task-candidate training worker.

This worker is intentionally separate from the legacy 20-step smoke command.
It consumes the formal UCS-386 gate, uses the canonical ConstructionElementNet,
never opens the locked Test pixels, and leaves a heartbeat/receipt trail that can
be inspected without attaching to the training process.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from urban_monitor.compat import autocast_context, create_grad_scaler


TOOL_PATH = Path(__file__).resolve()
PROJECT_ROOT = TOOL_PATH.parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from urban_renewal.construction.json_io import atomic_write_text, dump_json, load_json


MODEL_NAME = "URRS-Construction-UCS386-v1.0-task-candidate"
CLASS_NAMES = ("ucs_background", "ucs_exposed_area", "ucs_dustproof_net", "ucs_building")
IGNORE_INDEX = 255


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    dump_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    return load_json(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is preferred for this candidate screen; benchmark speed
    # is not allowed to silently change the seed/split contract.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"training config must be a mapping: {config_path}")
    return config


def _latest_allowed_gate(project_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((project_root / "outputs" / "urrs" / "audits").glob("ucs386_subset_readiness_*/UCS386_SUBSET_GATE.json"))
    for path in reversed(candidates):
        payload = _read_json(path)
        if payload.get("status") == "FORMAL_SUBSET_TRAINING_ALLOWED":
            return path.parent, payload
    raise RuntimeError("no FORMAL_SUBSET_TRAINING_ALLOWED UCS386 gate found")


def _require_cuda(device_name: str):
    import torch

    requested = str(device_name).casefold()
    if requested not in {"cuda", "cuda:0"}:
        raise RuntimeError(f"formal detached UCS training requires explicit CUDA; received device={device_name!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for formal training but torch.cuda.is_available() is false; CPU fallback is forbidden")
    return torch.device("cuda")


def _environment_payload(device) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(device)) if torch.cuda.is_available() else None,
        "torch_num_threads": int(torch.get_num_threads()),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        payload["gpu_memory_free_bytes"] = int(free)
        payload["gpu_memory_total_bytes"] = int(total)
    else:
        payload["gpu_memory_free_bytes"] = 0
        payload["gpu_memory_total_bytes"] = 0
    return payload


def _model_and_loss(width: int):
    from urban_renewal.construction.partial_labels import PartialLabelLoss
    from urban_renewal.models.construction import ConstructionElementNet

    return ConstructionElementNet(width=width), PartialLabelLoss(ignore_index=IGNORE_INDEX)


def _loader(manifest: Path, image_size: int, batch_size: int, rgb_mapping: Sequence[int], *, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader

    from urban_renewal.construction.datasets import UCSManifestDataset

    dataset = UCSManifestDataset(manifest, image_size=image_size, rgb_bands=tuple(rgb_mapping), allow_test=False)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=True,
        generator=generator,
        drop_last=False,
    )
    return dataset, loader


def _amp_context(device, enabled: bool):
    import torch
    return autocast_context(device, bool(enabled and device.type == "cuda"))


def _new_scaler(device, enabled: bool):
    import torch
    return create_grad_scaler(device, bool(enabled and device.type == "cuda"))


def _selection_score(metrics: Mapping[str, Any]) -> float:
    values = (
        float(metrics.get("non_background_miou", float("nan"))),
        float(metrics.get("non_background_macro_f1", float("nan"))),
        float(metrics.get("boundary_f1", float("nan"))),
        float(metrics.get("uncertainty_ece", float("nan"))),
    )
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    return 0.50 * values[0] + 0.25 * values[1] + 0.15 * values[2] + 0.10 * (1.0 - values[3])


def _boundary_distance(pred: np.ndarray, target: np.ndarray) -> float | None:
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return None
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")
    distance_to_target = distance_transform_edt(~target)
    distance_to_pred = distance_transform_edt(~pred)
    return float((distance_to_target[pred].mean() + distance_to_pred[target].mean()) / 2.0)


def _ece_from_bins(bin_count: np.ndarray, bin_confidence: np.ndarray, bin_accuracy: np.ndarray) -> float:
    total = float(bin_count.sum())
    if total <= 0:
        return float("nan")
    return float(np.sum((bin_count / total) * np.abs(bin_accuracy - bin_confidence)))


def evaluate_model(model, loader, criterion, device, *, amp_enabled: bool, collect_visuals: bool = False) -> dict[str, Any]:
    import torch

    model.eval()
    confusion = np.zeros((4, 4), dtype=np.int64)
    boundary_tp = boundary_fp = boundary_fn = 0
    boundary_displacements: list[float] = []
    ece_count = np.zeros(10, dtype=np.float64)
    ece_confidence = np.zeros(10, dtype=np.float64)
    ece_accuracy = np.zeros(10, dtype=np.float64)
    uncertainty_count = np.zeros(10, dtype=np.float64)
    uncertainty_confidence = np.zeros(10, dtype=np.float64)
    uncertainty_accuracy = np.zeros(10, dtype=np.float64)
    losses: list[float] = []
    sample_records: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    nan_count = 0
    total_valid = 0
    non_background_pred_pixels = 0
    all_background_predictions = 0
    all_class_predictions = 0
    probability_nonfinite_count = 0
    boundary_nonfinite_count = 0
    uncertainty_nonfinite_count = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            texture = batch["texture"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with _amp_context(device, amp_enabled):
                output = model(image, "ucs", texture)
                loss_values = criterion(output, target, "ucs")
            loss_value = float(loss_values["total"].detach().float().cpu())
            if not math.isfinite(loss_value):
                nan_count += 1
            else:
                losses.append(loss_value)
            probabilities = output["ucs_probabilities"].float()
            probability_nonfinite_count += int((~torch.isfinite(probabilities)).sum().detach().cpu())
            boundary_nonfinite_count += int((~torch.isfinite(output["boundary_logits"])).sum().detach().cpu())
            uncertainty_nonfinite_count += int((~torch.isfinite(output["uncertainty_logits"])).sum().detach().cpu())
            prediction = probabilities.argmax(1)
            valid = target != IGNORE_INDEX
            target_np = target.detach().cpu().numpy()
            prediction_np = prediction.detach().cpu().numpy()
            probability_np = probabilities.detach().cpu().numpy()
            uncertainty_np = output["uncertainty_logits"].sigmoid().detach().cpu().numpy()[:, 0]
            for index in range(target_np.shape[0]):
                valid_np = valid[index].detach().cpu().numpy().astype(bool)
                if valid_np.any():
                    values = target_np[index][valid_np] * 4 + prediction_np[index][valid_np]
                    confusion += np.bincount(values.astype(np.int64), minlength=16).reshape(4, 4)
                    total_valid += int(valid_np.sum())
                    non_background_pred_pixels += int(np.count_nonzero((prediction_np[index] > 0) & valid_np))
                if not np.any(prediction_np[index] > 0):
                    all_background_predictions += 1
                if valid_np.any() and np.all(prediction_np[index][valid_np] > 0):
                    all_class_predictions += 1
                correct_np = (prediction_np[index] == target_np[index]) & valid_np
                confidence_np = probability_np[index].max(0)
                bins = np.minimum((confidence_np * 10).astype(np.int64), 9)
                for bin_index in range(10):
                    selected = valid_np & (bins == bin_index)
                    count = int(selected.sum())
                    if count:
                        ece_count[bin_index] += count
                        ece_confidence[bin_index] += float(confidence_np[selected].sum())
                        ece_accuracy[bin_index] += float(correct_np[selected].sum())
                uncertainty_confidence_np = 1.0 - uncertainty_np[index]
                uncertainty_bins = np.minimum((uncertainty_confidence_np * 10).astype(np.int64), 9)
                for bin_index in range(10):
                    selected = valid_np & (uncertainty_bins == bin_index)
                    count = int(selected.sum())
                    if count:
                        uncertainty_count[bin_index] += count
                        uncertainty_confidence[bin_index] += float(uncertainty_confidence_np[selected].sum())
                        uncertainty_accuracy[bin_index] += float(correct_np[selected].sum())
                from urban_renewal.construction.partial_labels import boundary_target

                target_tensor = target[index : index + 1]
                prediction_tensor = prediction[index : index + 1]
                target_boundary = boundary_target(target_tensor).squeeze(1).detach().cpu().numpy()[0].astype(bool) & valid_np
                prediction_boundary = boundary_target(prediction_tensor).squeeze(1).detach().cpu().numpy()[0].astype(bool) & valid_np
                boundary_tp += int(np.count_nonzero(target_boundary & prediction_boundary))
                boundary_fp += int(np.count_nonzero(~target_boundary & prediction_boundary))
                boundary_fn += int(np.count_nonzero(target_boundary & ~prediction_boundary))
                displacement = _boundary_distance(prediction_boundary, target_boundary)
                if displacement is not None and math.isfinite(displacement):
                    boundary_displacements.append(displacement)
                sample_records.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "scene_id": str(batch["scene_id"][index]),
                        "valid_pixel_count": int(valid_np.sum()),
                        "predicted_non_background_fraction": float(np.count_nonzero((prediction_np[index] > 0) & valid_np) / max(int(valid_np.sum()), 1)),
                        "mean_confidence": float(confidence_np[valid_np].mean()) if valid_np.any() else None,
                        "mean_uncertainty": float(uncertainty_np[index][valid_np].mean()) if valid_np.any() else None,
                        "all_background_prediction": bool(not np.any(prediction_np[index] > 0)),
                        "all_class_prediction": bool(valid_np.any() and np.all(prediction_np[index][valid_np] > 0)),
                    }
                )
                if collect_visuals and len(visuals) < 24:
                    visuals.append(
                        {
                            "sample_id": str(batch["sample_id"][index]),
                            "image": image[index].detach().float().cpu().numpy(),
                            "target": target_np[index].copy(),
                            "prediction": prediction_np[index].copy(),
                            "uncertainty": uncertainty_np[index].copy(),
                        }
                    )

    class_metrics: list[dict[str, Any]] = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        union = tp + fp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / union if union else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics.append({"class_id": index, "class_name": name, "tp": tp, "fp": fp, "fn": fn, "iou": iou, "f1": f1, "precision": precision, "recall": recall})
    boundary_precision = boundary_tp / (boundary_tp + boundary_fp) if boundary_tp + boundary_fp else 0.0
    boundary_recall = boundary_tp / (boundary_tp + boundary_fn) if boundary_tp + boundary_fn else 0.0
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if boundary_precision + boundary_recall else 0.0
    confidence = np.divide(ece_confidence, ece_count, out=np.zeros_like(ece_confidence), where=ece_count > 0)
    accuracy = np.divide(ece_accuracy, ece_count, out=np.zeros_like(ece_accuracy), where=ece_count > 0)
    uncertainty_confidence_mean = np.divide(uncertainty_confidence, uncertainty_count, out=np.zeros_like(uncertainty_confidence), where=uncertainty_count > 0)
    uncertainty_accuracy_mean = np.divide(uncertainty_accuracy, uncertainty_count, out=np.zeros_like(uncertainty_accuracy), where=uncertainty_count > 0)
    ece = _ece_from_bins(ece_count, confidence, accuracy)
    uncertainty_ece = _ece_from_bins(uncertainty_count, uncertainty_confidence_mean, uncertainty_accuracy_mean)
    non_background_ious = [class_metrics[index]["iou"] for index in range(1, 4)]
    non_background_f1 = [class_metrics[index]["f1"] for index in range(1, 4)]
    metrics = {
        "sample_count": len(sample_records),
        "effective_pixel_count": int(total_valid),
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(np.trace(confusion) / max(total_valid, 1)),
        "macro_miou": float(np.mean([item["iou"] for item in class_metrics])),
        "non_background_miou": float(np.mean(non_background_ious)),
        "macro_f1": float(np.mean([item["f1"] for item in class_metrics])),
        "non_background_macro_f1": float(np.mean(non_background_f1)),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "average_boundary_displacement": float(np.mean(boundary_displacements)) if boundary_displacements else None,
        "ece": ece,
        "uncertainty_ece": uncertainty_ece,
        "uncertainty_confidence_histogram": uncertainty_count.astype(int).tolist(),
        "confidence_histogram": ece_count.astype(int).tolist(),
        "all_background_prediction_fraction": all_background_predictions / max(len(sample_records), 1),
        "all_class_prediction_fraction": all_class_predictions / max(len(sample_records), 1),
        "non_background_predicted_pixel_count": int(non_background_pred_pixels),
        "nan_count": int(nan_count),
        "probability_nonfinite_count": int(probability_nonfinite_count),
        "boundary_nonfinite_count": int(boundary_nonfinite_count),
        "uncertainty_nonfinite_count": int(uncertainty_nonfinite_count),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
    }
    reject_reasons: list[str] = []
    if not all(math.isfinite(float(metrics[key])) for key in ("loss", "accuracy", "non_background_miou", "non_background_macro_f1", "boundary_f1", "uncertainty_ece")):
        reject_reasons.append("nan_or_nonfinite_metric")
    if metrics["non_background_predicted_pixel_count"] == 0 or metrics["all_background_prediction_fraction"] >= 1.0:
        reject_reasons.append("all_background_prediction")
    if metrics["boundary_f1"] < 0.01:
        reject_reasons.append("bad_boundary_f1")
    if metrics["uncertainty_ece"] > 1.0:
        reject_reasons.append("uncertainty_calibration_failure")
    metrics["selection_score"] = _selection_score(metrics)
    metrics["selection_reject_reasons"] = reject_reasons
    metrics["selection_valid"] = not reject_reasons and math.isfinite(float(metrics["selection_score"]))
    metrics["ece_bins"] = {
        "count": ece_count.astype(int).tolist(),
        "confidence": confidence.tolist(),
        "accuracy": accuracy.tolist(),
        "uncertainty_confidence": uncertainty_confidence_mean.tolist(),
        "uncertainty_accuracy": uncertainty_accuracy_mean.tolist(),
    }
    metrics["sample_stability"] = {
        "all_background_prediction_fraction": metrics["all_background_prediction_fraction"],
        "all_class_prediction_fraction": metrics["all_class_prediction_fraction"],
        "nan_count": int(nan_count),
        "scene_count": len({item["scene_id"] for item in sample_records}),
        "mean_predicted_non_background_fraction": float(np.mean([item["predicted_non_background_fraction"] for item in sample_records])) if sample_records else None,
    }
    return {"metrics": metrics, "sample_records": sample_records, "visuals": visuals}


def _train_one_epoch(model, loader, criterion, optimizer, scaler, device, *, amp_enabled: bool, accumulation_steps: int, clip_norm: float) -> dict[str, Any]:
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: list[float] = []
    optimizer_steps = 0
    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        texture = batch["texture"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with _amp_context(device, amp_enabled):
            outputs = model(image, "ucs", texture)
            losses = criterion(outputs, target, "ucs")
            loss = losses["total"] / max(int(accumulation_steps), 1)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"non-finite training loss at loader step {step}")
        scaler.scale(loss).backward()
        if step % max(int(accumulation_steps), 1) == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        totals.append(float(losses["total"].detach().float().cpu()))
    return {"loss": float(np.mean(totals)) if totals else float("nan"), "optimizer_steps": optimizer_steps, "loader_steps": len(totals)}


def _checkpoint_payload(model, optimizer, scaler, *, epoch: int, global_step: int, stage: str, config_sha256: str, metrics: Mapping[str, Any], settings: Mapping[str, Any]):
    import torch

    return {
        "module_id": "construction_activity_v1",
        "model_name": MODEL_NAME,
        "model_type": "research_task_model_candidate",
        "architecture": "ConstructionElementNet",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "stage": stage,
        "config_sha256": config_sha256,
        "selection_metrics": dict(metrics),
        "settings": dict(settings),
        "scientific_status": "experimental_unselected" if stage != "final_frozen" else "candidate_not_ground_truth",
        "production_ready": False,
        "national_generalization_verified": False,
        "selection_used_test": False,
        "test_pixel_reads": 0,
        "created_at": _now(),
    }


def _save_torch(path: Path, payload: Mapping[str, Any]) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)
    return _sha256(path)


def _next_recovery_dir(run_dir: Path) -> Path:
    for index in range(1, 100):
        candidate = run_dir / f"recovery_attempt_{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"too many recovery attempts already exist under {run_dir}")


def _checkpoint_completeness(path: Path, config_sha256: str) -> dict[str, Any]:
    import torch

    if not path.exists():
        return {"path": str(path.resolve()), "exists": False, "complete": False, "missing_fields": ["checkpoint_file"]}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"path": str(path.resolve()), "exists": True, "complete": False, "missing_fields": [f"load_error:{type(exc).__name__}: {exc}"]}
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "epoch", "config_sha256", "global_step", "random_state"}
    missing = sorted(key for key in required if key not in payload)
    if payload.get("config_sha256") != config_sha256:
        missing.append("config_sha256_mismatch")
    return {
        "path": str(path.resolve()),
        "exists": True,
        "complete": not missing,
        "missing_fields": missing,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "has_scheduler_state": "scheduler_state_dict" in payload,
        "has_random_state": "random_state" in payload,
        "model_state": "model_state_dict" in payload,
        "optimizer_state": "optimizer_state_dict" in payload,
        "scaler_state": "scaler_state_dict" in payload,
    }


def _inspect_initialization_candidates(project_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    from urban_renewal.models.construction import ConstructionElementNet

    width = int(config["model"]["width"])
    canonical = ConstructionElementNet(width=width).state_dict()
    records: list[dict[str, Any]] = []
    candidates = config["training"]["initialization_screen"].get("candidates", [])
    allowed_registry_status = {"active", "registered", "external_pretrained", "validation_candidate"}
    for candidate in candidates:
        path = _resolve(project_root, str(candidate.get("path", "")))
        record: dict[str, Any] = {
            "name": candidate.get("name"),
            "path": str(path.resolve()),
            "exists": path.exists(),
            "expected_architecture": candidate.get("expected_architecture"),
            "registry_status": candidate.get("registry_status"),
            "registry_status_allowed": candidate.get("registry_status") in allowed_registry_status,
            "state_dict_load_ok": False,
            "strict_canonical_compatibility": False,
            "excluded_reason": None,
            "excluded_reasons": [],
            "pixel_reads": 0,
            "test_pixel_reads": 0,
        }
        if not path.exists():
            record["excluded_reasons"].append("checkpoint_missing")
            record["excluded_reason"] = "checkpoint_missing"
            records.append(record)
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            if not isinstance(state, dict):
                raise TypeError("checkpoint has no state_dict mapping")
            record["state_dict_load_ok"] = True
            keys_match = set(state) == set(canonical)
            shapes_match = keys_match and all(tuple(state[key].shape) == tuple(canonical[key].shape) for key in canonical)
            record["strict_canonical_compatibility"] = bool(keys_match and shapes_match)
            record["checkpoint_sha256"] = _sha256(path)
            record["state_key_count"] = len(state)
            if not record["registry_status_allowed"]:
                record["excluded_reasons"].append("registry_status_not_allowed")
            if not record["strict_canonical_compatibility"]:
                record["excluded_reasons"].append("incompatible_architecture_or_width")
            if record["excluded_reasons"]:
                record["excluded_reason"] = ";".join(record["excluded_reasons"])
            else:
                record["eligible_for_three_epoch_screen"] = True
        except Exception as exc:
            record["excluded_reasons"].append(f"checkpoint_invalid: {type(exc).__name__}: {exc}")
            record["excluded_reason"] = record["excluded_reasons"][0]
        records.append(record)
    eligible = [record for record in records if record.get("eligible_for_three_epoch_screen")]
    return {
        "status": "INITIALIZATION_SCREEN_ONLY",
        "candidate_records": records,
        "valid_initialization_count": len(eligible),
        "valid_initializations": [record["name"] for record in eligible],
        "screen_epochs": int(config["training"]["initialization_screen"].get("epochs", 3)),
        "screen_seed": int(config["training"]["seed"]),
        "screen_train_manifest_only": True,
        "screen_validation_manifest_only": True,
        "test_used": False,
        "test_pixel_reads": 0,
        "registerable": False,
        "selected_initialization": "scratch_random_seed_20260803" if not eligible else "pending_three_epoch_screen",
        "notes": [
            "Incompatible, unregistered, hash-invalid, and incomplete weights are excluded without conversion.",
            "No current checkpoint passed both strict canonical compatibility and registry checks; Stage A will use the fixed-seed scratch initialization.",
        ],
    }


def prepare_run(project_root: Path, config_path: Path, ucs_root: Path, *, run_dir: Path | None = None) -> dict[str, Any]:
    import torch

    config = _load_config(config_path)
    gate_dir, gate = _latest_allowed_gate(project_root)
    if gate.get("test_pixel_reads") != 0 or gate.get("scope_limitations", {}).get("full_ucs_training") is not False:
        raise RuntimeError("UCS386 gate does not satisfy the locked-test and subset-only contract")
    device = _require_cuda(config.get("training", {}).get("device", "cuda"))
    _set_seed(int(config["training"]["seed"]))
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = project_root / "outputs" / "training" / "construction_activity_v1" / f"{MODEL_NAME}_{stamp}"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = dict(config)
    run_config.setdefault("dataset", {})["ucs_root"] = str(ucs_root.resolve())
    run_config["dataset"]["subset_gate_dir"] = str(gate_dir.resolve())
    run_config["dataset"]["train_manifest"] = str((gate_dir / "UCS_TRAIN_MANIFEST.csv").resolve())
    run_config["dataset"]["validation_manifest"] = str((gate_dir / "UCS_VALIDATION_MANIFEST.csv").resolve())
    run_config["dataset"]["locked_test_manifest"] = str((gate_dir / "UCS_TEST_LOCKED_MANIFEST.csv").resolve())
    run_config_path = run_dir / "run_config.yaml"
    atomic_write_text(run_config_path, yaml.safe_dump(run_config, sort_keys=False, allow_unicode=True))
    config_sha = _sha256(run_config_path)
    screen = _inspect_initialization_candidates(project_root, run_config)
    _write_json(run_dir / "INITIALIZATION_SCREEN.json", screen)
    env = _environment_payload(device)
    train_count = int(run_config["dataset"].get("train_count", 296))
    batch = int(run_config["training"].get("batch_size", 2))
    max_epochs = int(run_config["training"].get("max_epochs", 80))
    approx_optimizer_steps = math.ceil(train_count / max(batch, 1)) * max_epochs
    estimated_minutes = round(approx_optimizer_steps * 0.55 / 60.0, 1)
    command = [
        str((project_root / ".venv-gpu" / "Scripts" / "python.exe").resolve()),
        str((project_root / "tools" / "construction_training_worker.py").resolve()),
        "--mode",
        "train",
        "--project-root",
        str(project_root.resolve()),
        "--run-dir",
        str(run_dir.resolve()),
        "--config",
        str(run_config_path.resolve()),
    ]
    receipt = {
        "status": "PREPARED",
        "model_name": MODEL_NAME,
        "model_type": "research_task_model_candidate",
        "dataset_id": "UCS-386-available-image-subset",
        "scientific_status": "experimental_unselected",
        "production_ready": False,
        "national_generalization_verified": False,
        "gate_status": gate.get("status"),
        "gate_directory": str(gate_dir.resolve()),
        "run_directory": str(run_dir.resolve()),
        "run_config": str(run_config_path.resolve()),
        "run_config_sha256": config_sha,
        "environment": env,
        "gpu_memory": {key: env[key] for key in ("gpu_memory_free_bytes", "gpu_memory_total_bytes")},
        "estimated_duration_minutes": estimated_minutes,
        "estimated_optimizer_steps": approx_optimizer_steps,
        "command": command,
        "command_text": " ".join(f'"{value}"' if " " in value else value for value in command),
        "initialization_screen": str((run_dir / "INITIALIZATION_SCREEN.json").resolve()),
        "initialization_selected": screen["selected_initialization"],
        "training_process_started": False,
        "detached": False,
        "test_pixel_reads": 0,
        "locked_test_used_for_selection": False,
        "full_ucs_training": False,
        "subset_training": True,
        "full_record_coverage": "386/600",
        "missingness_bias_not_ruled_out": True,
        "license_status": "confirmed_training_allowed",
        "license_source": "explicit_user_confirmation",
        "created_at": _now(),
    }
    _write_json(run_dir / "PREPARE_RECEIPT.json", receipt)
    _write_json(run_dir / "TRAINING_STATE.json", {"status": "PREPARED", "run_directory": str(run_dir.resolve()), "updated_at": _now(), "test_pixel_reads": 0})
    return {**receipt, "run_dir": str(run_dir.resolve())}


def _write_heartbeat(run_dir: Path, payload: Mapping[str, Any]) -> None:
    state = {**dict(payload), "run_directory": str(run_dir.resolve()), "updated_at": _now(), "test_pixel_reads": 0}
    _write_json(run_dir / "TRAINING_STATE.json", state)
    _write_json(run_dir / "HEARTBEAT.json", state)
    _write_json(run_dir / "TRAINING_RUN_STATE.json", state)
    _write_json(run_dir / "heartbeat.json", state)
    progress = {
        "status": state.get("status"),
        "stage": state.get("stage"),
        "epoch": state.get("epoch", 0),
        "max_epochs": state.get("max_epochs"),
        "best_epoch": state.get("best_epoch"),
        "best_score": state.get("best_score"),
        "updated_at": state["updated_at"],
        "test_pixel_reads": 0,
    }
    _write_json(run_dir / "progress.json", progress)


def train_run(project_root: Path, run_dir: Path, config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    import torch

    config = _load_config(config_path)
    device = _require_cuda(config["training"].get("device", "cuda"))
    _set_seed(int(config["training"]["seed"]))
    gate_dir, gate = _latest_allowed_gate(project_root)
    if str(gate_dir.resolve()) != str(Path(config["dataset"]["subset_gate_dir"]).resolve()):
        raise RuntimeError("run config gate directory is not the latest allowed UCS386 gate")
    if gate.get("test_pixel_reads") != 0:
        raise RuntimeError("locked Test pixel-read invariant is not zero")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_sha = _sha256(config_path)
    artifact_dir = run_dir
    recovery_info: dict[str, Any] = {
        "resume": False,
        "resume_origin": "new_training_run",
        "resume_epoch": 0,
        "resume_global_step": 0,
        "checkpoint_available": False,
        "checkpoint_complete": False,
        "partial_uncheckpointed_steps_discarded": False,
        "same_run_directory": True,
        "new_training_run_created": False,
    }
    if resume:
        previous_checkpoint = run_dir / "checkpoints" / "last.pth"
        checkpoint_info = _checkpoint_completeness(previous_checkpoint, config_sha)
        artifact_dir = _next_recovery_dir(run_dir)
        recovery_info = {
            "resume": True,
            "resume_origin": "partial_checkpoint_incomplete_restart_epoch0" if not checkpoint_info["complete"] else "last_checkpoint",
            "resume_epoch": int(checkpoint_info.get("epoch") or 0) if checkpoint_info["complete"] else 0,
            "resume_global_step": int(checkpoint_info.get("global_step") or 0) if checkpoint_info["complete"] else 0,
            "checkpoint_available": bool(checkpoint_info.get("exists")),
            "checkpoint_complete": bool(checkpoint_info.get("complete")),
            "checkpoint_info": checkpoint_info,
            "partial_uncheckpointed_steps_discarded": not bool(checkpoint_info.get("complete")),
            "same_run_directory": True,
            "new_training_run_created": False,
            "recovery_directory": str(artifact_dir.resolve()),
        }
        _write_json(run_dir / f"{artifact_dir.name}.json", {**recovery_info, "created_at": _now()})
        _write_json(artifact_dir / "RESUME_ORIGIN.json", {**recovery_info, "created_at": _now()})
    logs_dir = artifact_dir / "logs"
    checkpoints_dir = artifact_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    width = int(config["model"].get("width", 24))
    image_size = int(config["dataset"].get("image_size", 256))
    batch_size = int(config["training"].get("batch_size", 2))
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    max_epochs = int(config["training"].get("max_epochs", 80))
    patience = int(config["training"].get("patience", 12))
    rgb_mapping = tuple(int(value) for value in config["dataset"].get("rgb_mapping", [3, 2, 1]))
    train_manifest = Path(config["dataset"]["train_manifest"])
    validation_manifest = Path(config["dataset"]["validation_manifest"])
    model, criterion = _model_and_loss(width)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"].get("learning_rate", 3e-4)), weight_decay=float(config["training"].get("weight_decay", 0.01)))
    scaler = _new_scaler(device, bool(config["training"].get("amp", True)))
    start_epoch = 1
    global_step = 0
    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        if recovery_info["checkpoint_complete"]:
            resume_checkpoint = Path(recovery_info["checkpoint_info"]["path"])
            payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            if payload.get("scaler_state_dict"):
                scaler.load_state_dict(payload["scaler_state_dict"])
            start_epoch = int(payload.get("epoch", 0)) + 1
            global_step = int(payload.get("global_step", 0))
            best_score = float(payload.get("best_score", -float("inf")))
            best_epoch = int(payload.get("best_epoch", 0))
            bad_epochs = int(payload.get("bad_epochs", 0))
            history = list(payload.get("history", []))
            random_state = payload.get("random_state") or {}
            if random_state.get("python"):
                random.setstate(tuple(random_state["python"]))
            if random_state.get("numpy"):
                numpy_state = list(random_state["numpy"])
                if len(numpy_state) > 1:
                    numpy_state[1] = np.asarray(numpy_state[1], dtype=np.uint32)
                np.random.set_state(tuple(numpy_state))
            if random_state.get("torch") is not None:
                torch.set_rng_state(random_state["torch"])
            if random_state.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(random_state["cuda"])
    train_dataset, train_loader = _loader(train_manifest, image_size, batch_size, rgb_mapping, shuffle=True, seed=int(config["training"]["seed"]))
    val_dataset, val_loader = _loader(validation_manifest, image_size, batch_size, rgb_mapping, shuffle=False, seed=int(config["training"]["seed"]) + 1)
    settings = {"image_size": image_size, "batch_size": batch_size, "gradient_accumulation_steps": accumulation_steps, "width": width, "rgb_mapping": list(rgb_mapping)}
    _write_heartbeat(run_dir, {"status": "RUNNING", "stage": "A_UCS", "epoch": start_epoch - 1, "max_epochs": max_epochs, "best_epoch": best_epoch, "best_score": best_score, "global_step": global_step, "settings": settings, "artifact_directory": str(artifact_dir.resolve()), **recovery_info})
    with (logs_dir / "training.jsonl").open("a", encoding="utf-8") as log_stream:
        epoch = start_epoch
        while epoch <= max_epochs:
            try:
                train_metrics = _train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    scaler,
                    device,
                    amp_enabled=bool(config["training"].get("amp", True)),
                    accumulation_steps=accumulation_steps,
                    clip_norm=float(config["training"].get("gradient_clip_norm", 1.0)),
                )
            except RuntimeError as exc:
                message = str(exc).casefold()
                if "out of memory" not in message or not bool(config["training"].get("auto_oom_fallback", {}).get("enabled", True)):
                    raise
                fallback = config["training"].get("auto_oom_fallback", {})
                old_settings = dict(settings)
                if batch_size > int(fallback.get("batch_size_floor", 1)):
                    batch_size = max(int(fallback.get("batch_size_floor", 1)), batch_size // 2)
                elif accumulation_steps < int(fallback.get("accumulation_ceiling", 16)):
                    accumulation_steps = min(int(fallback.get("accumulation_ceiling", 16)), accumulation_steps * 2)
                elif image_size > int(fallback.get("image_size_floor", 128)):
                    image_size = max(int(fallback.get("image_size_floor", 128)), image_size - 32)
                else:
                    raise RuntimeError("CUDA OOM persisted after batch/accumulation/crop fallback") from exc
                torch.cuda.empty_cache()
                settings = {"image_size": image_size, "batch_size": batch_size, "gradient_accumulation_steps": accumulation_steps, "width": width, "rgb_mapping": list(rgb_mapping)}
                train_dataset, train_loader = _loader(train_manifest, image_size, batch_size, rgb_mapping, shuffle=True, seed=int(config["training"]["seed"]) + epoch)
                val_dataset, val_loader = _loader(validation_manifest, image_size, batch_size, rgb_mapping, shuffle=False, seed=int(config["training"]["seed"]) + 1)
                _write_json(run_dir / "OOM_FALLBACK.json", {"epoch": epoch, "old_settings": old_settings, "new_settings": settings, "reason": str(exc), "created_at": _now()})
                continue
            evaluation = evaluate_model(model, val_loader, criterion, device, amp_enabled=bool(config["training"].get("amp", True)))
            metrics = evaluation["metrics"]
            score = float(metrics["selection_score"])
            improved = bool(metrics["selection_valid"] and math.isfinite(score) and score > best_score)
            if improved:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            global_step += int(train_metrics.get("optimizer_steps", 0))
            row = {"epoch": epoch, "stage": "A_UCS", "train": train_metrics, "validation": metrics, "best_score": best_score, "best_epoch": best_epoch, "global_step": global_step, "improved": improved, "settings": settings, "created_at": _now(), "resume_origin": recovery_info["resume_origin"]}
            history.append(row)
            log_stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            log_stream.flush()
            payload = _checkpoint_payload(model, optimizer, scaler, epoch=epoch, global_step=global_step, stage="A_UCS", config_sha256=config_sha, metrics=metrics, settings=settings)
            payload.update({"best_score": best_score, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history})
            _save_torch(checkpoints_dir / "last.pth", payload)
            if improved:
                _save_torch(checkpoints_dir / "best_validation.pth", payload)
            if epoch % int(config["training"].get("checkpoint_period_epochs", 10)) == 0:
                _save_torch(checkpoints_dir / f"epoch_{epoch:03d}.pth", payload)
            _write_heartbeat(run_dir, {"status": "RUNNING", "stage": "A_UCS", "epoch": epoch, "max_epochs": max_epochs, "best_epoch": best_epoch, "best_score": best_score, "global_step": global_step, "bad_epochs": bad_epochs, "settings": settings, "artifact_directory": str(artifact_dir.resolve()), "validation_selection_score": score, "selection_valid": metrics["selection_valid"], **recovery_info})
            if bad_epochs >= patience:
                break
            epoch += 1
    final_status = "COMPLETED_STAGE_A" if (checkpoints_dir / "best_validation.pth").exists() else "FAILED_NO_VALIDATION_CHECKPOINT"
    result = {
        "status": final_status,
        "stage": "A_UCS",
        "run_directory": str(run_dir.resolve()),
        "artifact_directory": str(artifact_dir.resolve()),
        "best_checkpoint": str((checkpoints_dir / "best_validation.pth").resolve()) if (checkpoints_dir / "best_validation.pth").exists() else None,
        "best_epoch": best_epoch,
        "global_step": global_step,
        "best_selection_score": best_score,
        "completed_epochs": len(history),
        "stage_b": {"enabled": bool(config["training"].get("stage_b_joint", {}).get("enabled", False)), "status": "NOT_RUN_OPTIONAL"},
        "selection_used_test": False,
        "test_pixel_reads": 0,
        "full_ucs_training": False,
        "subset_training": True,
        "scientific_status": "experimental_unselected",
        **recovery_info,
        "created_at": _now(),
    }
    _write_json(artifact_dir / "TRAINING_COMPLETION.json", result)
    if resume:
        _write_json(run_dir / "TRAINING_RESUME_COMPLETION.json", result)
    else:
        _write_json(run_dir / "TRAINING_COMPLETION.json", result)
    _write_heartbeat(run_dir, {**result, "status": final_status})
    return result


def status_run(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "TRAINING_STATE.json"
    receipt_path = run_dir / "TRAINING_LAUNCH_RECEIPT.json"
    result = {"status": "NOT_PREPARED", "run_directory": str(run_dir.resolve()), "test_pixel_reads": 0}
    if state_path.exists():
        result.update(_read_json(state_path))
    if receipt_path.exists():
        result["launch_receipt"] = _read_json(receipt_path)
    result["observed_at"] = _now()
    return result


def finalize_run(project_root: Path, run_dir: Path, config_path: Path) -> dict[str, Any]:
    from construction_finalize import finalize_existing_run

    return finalize_existing_run(project_root.resolve(), run_dir.resolve(), config_path.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detached UCS-386 construction task-candidate worker")
    parser.add_argument("--mode", choices=["prepare", "train", "status", "finalize"], required=True)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    parser.add_argument("--ucs-root", default=r"D:\Data Set\URS\600 manually delineated UCS labels part2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    try:
        if args.mode == "prepare":
            config_path = _resolve(project_root, args.config or "configs/urrs/construction/ucs386_task_candidate.yaml")
            result = prepare_run(project_root, config_path, Path(args.ucs_root), run_dir=Path(args.run_dir) if args.run_dir else None)
        elif args.mode == "status":
            if not args.run_dir:
                raise ValueError("--run-dir is required for status")
            result = status_run(Path(args.run_dir))
        elif args.mode == "train":
            if not args.run_dir:
                raise ValueError("--run-dir is required for train")
            if not args.config:
                raise ValueError("--config is required for train")
            result = train_run(project_root, Path(args.run_dir), Path(args.config), resume=args.resume)
        else:
            if not args.run_dir or not args.config:
                raise ValueError("--run-dir and --config are required for finalize")
            result = finalize_run(project_root, Path(args.run_dir), Path(args.config))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except Exception as exc:
        # A read-only Status/Finalize call must never overwrite a training
        # failure receipt or heartbeat merely because one historical JSON is
        # malformed.  Only Prepare/Train own failure-state writes.
        if args.run_dir and args.mode in {"prepare", "train"}:
            try:
                _write_json(Path(args.run_dir) / "TRAINING_FAILURE.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "test_pixel_reads": 0, "created_at": _now()})
                _write_heartbeat(Path(args.run_dir), {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
