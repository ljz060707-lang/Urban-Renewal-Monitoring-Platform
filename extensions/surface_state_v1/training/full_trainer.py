"""Full epoch trainer for the existing SurfaceStateNet extension."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..datasets.cwld import CWLDManifestDataset
from ..datasets.cwld_codec import CLASS_NAMES as CWLD_CLASS_NAMES, CWLDCodec
from ..datasets.fusu import FUSUSampleDataset
from ..models.surface_state_net import SurfaceStateNet
from ..models.temporal_change_head import TemporalChangeHead
from ..models.tfr import TemporalFeatureRefinement
from urban_monitor.compat import create_grad_scaler
from .checkpointing import save_checkpoint, select_better, sha256, write_hashes
from .losses import cwld_loss, fusu_loss
from .metrics import BinaryCounts, Confusion

FUSU_CLASS_NAMES = [f"native_fusu::{index}" for index in range(17)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_state(repo: Path) -> dict:
    def command(*args):
        result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
        return result.stdout.strip()
    return {"commit": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current"), "status_short": command("status", "--short"), "diff_stat": command("diff", "--stat")}


def _manifest_hash(path: str | Path) -> str:
    return sha256(path)


def _run_directory(repo: Path, kind: str, requested: str | None = None) -> Path:
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = repo / path
        path.mkdir(parents=True, exist_ok=True)
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = repo / "outputs" / "training" / "surface_state_v1" / f"{kind}_{stamp}"
    suffix = 1
    while path.exists():
        path = path.with_name(f"{kind}_{stamp}_{suffix:02d}")
        suffix += 1
    for name in ("checkpoints", "logs", "metrics", "visualizations"):
        (path / name).mkdir(parents=True, exist_ok=True)
    return path


def _resolve_path(repo: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else repo / path)


def _loader(dataset, config, *, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", config.get("batch_size_per_dataset", 2))),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=bool(config.get("pin_memory", True) and torch.cuda.is_available()),
        persistent_workers=False,
    )


def compute_cwld_class_weights(manifest: str, codec: CWLDCodec, output: Path, minimum: float, maximum: float) -> torch.Tensor:
    cache = output / "CWLD_CLASS_WEIGHTS.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return torch.tensor(payload["weights"], dtype=torch.float32)
    rows = [row for row in csv.DictReader(Path(manifest).open(encoding="utf-8-sig")) if row["split"] == "train"]
    counts = np.zeros(4, dtype=np.int64)
    ignored = 0
    for row in rows:
        target = codec.encode_file(row["mask_t1"])
        counts += np.bincount(target[target != 255], minlength=4)
        ignored += int((target == 255).sum())
    frequencies = counts / max(1, counts.sum())
    median = float(np.median(frequencies[frequencies > 0]))
    weights = np.clip(median / np.maximum(frequencies, 1e-12), minimum, maximum)
    payload = {"class_names": list(CWLD_CLASS_NAMES), "pixel_counts": counts.tolist(), "pixel_frequencies": frequencies.tolist(), "ignored_pixels": ignored, "weights": weights.tolist(), "source_split": "train"}
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return torch.tensor(weights, dtype=torch.float32)


def _fusu_forward(model, tfr, change_head, t1, t2, use_tfr: bool):
    if not use_tfr:
        output = model.forward_fusu(t1, t2)
        return output["logits_t1"], output["logits_t2"], None
    batch, _, height, width = t1.shape
    images = torch.stack([t1, t2], 1)
    encoded = model.decoder(model.encoder(images.reshape(batch * 2, 3, height, width)))
    _, channels, feature_height, feature_width = encoded.shape
    sequence = encoded.reshape(batch, 2, channels, feature_height, feature_width)
    refined = tfr(sequence, temporal_valid_mask=torch.ones((batch, 2), dtype=torch.bool, device=t1.device))["refined_features"][0]
    flattened = refined.reshape(batch * 2, channels, feature_height, feature_width)
    upsampled = nn.functional.interpolate(flattened, size=(height, width), mode="bilinear", align_corners=False)
    logits = model.fusu_decoder_head(upsampled).reshape(batch, 2, 17, height, width)
    change = change_head(refined)["dense_change_probability"] if change_head is not None else None
    return logits[:, 0], logits[:, 1], change


@torch.no_grad()
def evaluate_cwld(model, loader, device, class_weights) -> dict:
    model.eval()
    confusion = Confusion(4)
    losses = []
    ignored = total = 0
    for batch in loader:
        image, target = batch["image"].to(device), batch["target"].to(device)
        logits = model.forward_cwld(image)["logits"]
        losses.append(float(cwld_loss(logits, target, class_weights).detach().cpu()))
        prediction = logits.argmax(1).cpu().numpy()
        target_numpy = target.cpu().numpy()
        confusion.update(prediction, target_numpy)
        ignored += int((target_numpy == 255).sum())
        total += target_numpy.size
    report = confusion.report(list(CWLD_CLASS_NAMES))
    report.update(validation_loss=float(np.mean(losses)), ignored_pixel_fraction=ignored / max(1, total), sample_count=len(loader.dataset), split="validation", test_read_count=0)
    return report


@torch.no_grad()
def evaluate_fusu(model, tfr, change_head, loader, device, use_tfr: bool) -> dict:
    model.eval()
    if tfr is not None:
        tfr.eval()
    if change_head is not None:
        change_head.eval()
    first, second, change_counts = Confusion(17), Confusion(17), BinaryCounts()
    losses, violations, valid_pixels = [], 0, 0
    for batch in loader:
        t1, t2 = batch["image_t1"].to(device), batch["image_t2"].to(device)
        y1, y2 = batch["mask_t1"].to(device), batch["mask_t2"].to(device)
        logits1, logits2, change_probability = _fusu_forward(model, tfr, change_head, t1, t2, use_tfr)
        loss, _, _ = fusu_loss(logits1, logits2, y1, y2)
        losses.append(float(loss.detach().cpu()))
        p1, p2 = logits1.argmax(1), logits2.argmax(1)
        first.update(p1.cpu().numpy(), y1.cpu().numpy())
        second.update(p2.cpu().numpy(), y2.cpu().numpy())
        ground_truth_change = (y1 != y2) & (y1 != 255) & (y2 != 255)
        predicted_change = p1 != p2
        change_counts.update(predicted_change.cpu().numpy(), ground_truth_change.cpu().numpy())
        stable = ~ground_truth_change
        violations += int((predicted_change & stable).sum().cpu())
        valid_pixels += int(stable.sum().cpu())
    t1_report, t2_report, change_report = first.report(FUSU_CLASS_NAMES), second.report(FUSU_CLASS_NAMES), change_counts.report()
    average_miou = (t1_report["macro_mIoU"] + t2_report["macro_mIoU"]) / 2
    violation_rate = violations / max(1, valid_pixels)
    return {
        "title": "FUSU 500-pair sample spatial holdout — NOT full FUSU validation",
        "validation_loss": float(np.mean(losses)),
        "t1_macro_mIoU": t1_report["macro_mIoU"],
        "t2_macro_mIoU": t2_report["macro_mIoU"],
        "average_macro_mIoU": average_miou,
        "per_class_t1": t1_report["per_class"],
        "per_class_t2": t2_report["per_class"],
        "semantic_change_precision": change_report["precision"],
        "semantic_change_recall": change_report["recall"],
        "semantic_change_F1": change_report["f1"],
        "temporal_consistency_violation_rate": violation_rate,
        "confusion_matrix_t1": t1_report["confusion_matrix"],
        "confusion_matrix_t2": t2_report["confusion_matrix"],
        "sample_count": len(loader.dataset),
        "full_fusu_validated": False,
        "national_generalization_verified": False,
        "test_read_count": 0,
    }


def _checkpoint_payload(model, tfr, change_head, optimizer, scheduler, scaler, epoch, global_step, score, metrics, config, repo, manifests, kind):
    return {
        "model_state_dict": model.state_dict(),
        "tfr_state_dict": tfr.state_dict() if tfr is not None else None,
        "change_head_state_dict": change_head.state_dict() if change_head is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "selection_score": score,
        "validation_metrics": metrics,
        "cwld_class_names": list(CWLD_CLASS_NAMES),
        "fusu_class_names": FUSU_CLASS_NAMES,
        "cwld_codec": config.get("codec"),
        "fusu_class_mapping": "native_0_to_16_no_semantic_guess",
        "ontology_mapping": {"engineering_facility_is_not_construction_site": True},
        "config_snapshot": config,
        "manifest_hashes": {name: _manifest_hash(path) for name, path in manifests.items()},
        "dataset_scope": config.get("dataset_scope", "cwld_official_train_validation"),
        "split_scope": config.get("validation_scope", "official_validation" if kind == "cwld" else "sample_spatial_holdout"),
        "git_status": _git_state(repo),
        "software_versions": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
        "created_at": _now(),
        "scientific_status": "validation_candidate" if kind == "cwld" else "sample_validation_candidate",
    }


def _restore_selection_state(run: Path) -> tuple[dict | None, int]:
    """Restore best-model and early-stopping state from durable epoch records."""

    best = None
    selection_path = run / "CHECKPOINT_SELECTION.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        candidate = selection.get("best")
        if isinstance(candidate, dict):
            best = candidate

    patience = 0
    log_path = run / "logs" / "epochs.jsonl"
    if log_path.exists():
        rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in reversed(rows):
            if row.get("improved"):
                break
            patience += 1
    return best, patience

def train(config_path: str | Path, repo: str | Path, *, run_dir: str | None = None) -> dict:
    repo = Path(repo)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kind = config["kind"]
    _seed(int(config.get("seed", 20260729)))
    run = _run_directory(repo, kind, run_dir)
    (run / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    receipt_path = run / "TRAINING_RECEIPT.json"
    previous_receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.exists()
        else {}
    )
    receipt = {
        "status": "running",
        "kind": kind,
        "run_dir": str(run.resolve()),
        "started_at": previous_receipt.get("started_at", _now()),
        "resumed_at": _now() if previous_receipt else None,
        "resume_count": int(previous_receipt.get("resume_count", 0))
        + (1 if previous_receipt else 0),
        "git": _git_state(repo),
        "test_read_count": 0,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda:0" if torch.cuda.is_available() and str(config.get("device", "cuda")).startswith("cuda") else "cpu")
    model = SurfaceStateNet().to(device)
    use_tfr = bool(config.get("use_tfr", False) and kind in {"fusu_sample", "tfr_sample", "joint_sample"})
    tfr = TemporalFeatureRefinement(24, heads=int(config.get("temporal", {}).get("num_heads", 4)), dropout=float(config.get("temporal", {}).get("dropout", 0.1))).to(device) if use_tfr else None
    change_head = TemporalChangeHead(24).to(device) if bool(config.get("use_change_head", use_tfr)) and kind != "cwld" else None
    parameters = list(model.parameters()) + (list(tfr.parameters()) if tfr else []) + (list(change_head.parameters()) if change_head else [])
    optimizer_config = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(parameters, lr=float(optimizer_config.get("decoder_lr", config.get("learning_rate", 3e-4))), weight_decay=float(optimizer_config.get("weight_decay", 0.01)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["epochs"])))
    amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = create_grad_scaler(device, amp)

    codec = CWLDCodec(**config.get("codec", {"max_anchor_distance": 128, "min_distance_margin": 16, "ignore_index": 255}))
    manifests = {}
    cwld_train_loader = cwld_val_loader = fusu_train_loader = fusu_val_loader = None
    class_weights = None
    if kind in {"cwld", "joint_sample"}:
        manifest = _resolve_path(repo, config["cwld_manifest"])
        manifests["cwld"] = manifest
        cwld_train_loader = _loader(CWLDManifestDataset(manifest, "train", codec, config["image_size"]), config, shuffle=True)
        cwld_val_loader = _loader(CWLDManifestDataset(manifest, "validation", codec, config["image_size"]), config, shuffle=False)
        class_weights = compute_cwld_class_weights(manifest, codec, run / "metrics", float(config.get("class_weight_min", 0.25)), float(config.get("class_weight_max", 4.0))).to(device)
    if kind in {"fusu_sample", "tfr_sample", "joint_sample"}:
        train_manifest = _resolve_path(repo, config["fusu_train_manifest"])
        val_manifest = _resolve_path(repo, config["fusu_validation_manifest"])
        manifests.update(fusu_train=train_manifest, fusu_validation=val_manifest)
        fusu_train_loader = _loader(FUSUSampleDataset(train_manifest, config["image_size"]), config, shuffle=True)
        fusu_val_loader = _loader(FUSUSampleDataset(val_manifest, config["image_size"]), config, shuffle=False)

    start_epoch = 0
    global_step = 0
    best = None
    patience = 0
    last_path = run / "checkpoints" / "last.pth"
    if config.get("resume") == "auto" and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if tfr and state.get("tfr_state_dict"):
            tfr.load_state_dict(state["tfr_state_dict"])
        if change_head and state.get("change_head_state_dict"):
            change_head.load_state_dict(state["change_head_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best, patience = _restore_selection_state(run)
        if start_epoch > 0 and best is None:
            raise RuntimeError(
                "Cannot safely resume: CHECKPOINT_SELECTION.json has no historical best"
            )

    log_path = run / "logs" / "epochs.jsonl"
    interrupted = False
    try:
        for epoch in range(start_epoch, int(config["epochs"])):
            model.train()
            if tfr:
                tfr.train()
            if change_head:
                change_head.train()
            epoch_started = time.perf_counter()
            cwld_iterator = iter(cwld_train_loader) if cwld_train_loader else None
            fusu_iterator = iter(fusu_train_loader) if fusu_train_loader else None
            cycles = max(len(cwld_train_loader) if cwld_train_loader else 0, len(fusu_train_loader) if fusu_train_loader else 0)
            optimizer.zero_grad(set_to_none=True)
            epoch_losses = []
            for cycle in range(cycles):
                joint = torch.zeros((), device=device)
                detail = {}
                if cwld_iterator is not None:
                    try:
                        batch = next(cwld_iterator)
                    except StopIteration:
                        cwld_iterator = iter(cwld_train_loader)
                        batch = next(cwld_iterator)
                    with torch.amp.autocast(device_type=device.type, enabled=amp):
                        cwld_value = cwld_loss(model.forward_cwld(batch["image"].to(device))["logits"], batch["target"].to(device), class_weights)
                    joint = joint + cwld_value
                    detail["cwld_loss"] = float(cwld_value.detach().cpu())
                if fusu_iterator is not None:
                    try:
                        batch = next(fusu_iterator)
                    except StopIteration:
                        fusu_iterator = iter(fusu_train_loader)
                        batch = next(fusu_iterator)
                    t1, t2, y1, y2 = (batch[key].to(device) for key in ("image_t1", "image_t2", "mask_t1", "mask_t2"))
                    with torch.amp.autocast(device_type=device.type, enabled=amp):
                        logits1, logits2, change_probability = _fusu_forward(model, tfr, change_head, t1, t2, use_tfr)
                        fusu_value, semantic_value, change_value = fusu_loss(logits1, logits2, y1, y2)
                        if change_probability is not None:
                            target_change = (y1 != y2).to(change_probability.dtype)[:, None]
                            target_change = nn.functional.interpolate(target_change, size=change_probability.shape[-2:], mode="nearest")[:, 0]
                            fusu_value = fusu_value + 0.25 * nn.functional.binary_cross_entropy(change_probability.clamp(1e-6, 1 - 1e-6), target_change)
                    joint = joint + fusu_value
                    detail.update(fusu_semantic_loss=float(semantic_value.detach().cpu()), fusu_change_loss=float(change_value.detach().cpu()))
                if not torch.isfinite(joint):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} cycle={cycle}")
                scaled = joint / int(config.get("gradient_accumulation_steps", 1))
                scaler.scale(scaled).backward()
                if (cycle + 1) % int(config.get("gradient_accumulation_steps", 1)) == 0 or cycle + 1 == cycles:
                    scaler.unscale_(optimizer)
                    grad_norm = float(nn.utils.clip_grad_norm_(parameters, float(config.get("gradient_clip_norm", 1.0))).detach().cpu())
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                else:
                    grad_norm = None
                epoch_losses.append(float(joint.detach().cpu()))
            scheduler.step()

            metrics = {}
            if cwld_val_loader:
                metrics["cwld"] = evaluate_cwld(model, cwld_val_loader, device, class_weights)
            if fusu_val_loader:
                metrics["fusu"] = evaluate_fusu(model, tfr, change_head, fusu_val_loader, device, use_tfr)
            if kind == "cwld":
                score = metrics["cwld"]["macro_mIoU"]
                macro_f1 = metrics["cwld"]["macro_F1"]
            elif kind in {"fusu_sample", "tfr_sample"}:
                fusu = metrics["fusu"]
                score = 0.60 * fusu["average_macro_mIoU"] + 0.30 * fusu["semantic_change_F1"] + 0.10 * (1 - fusu["temporal_consistency_violation_rate"])
                macro_f1 = fusu["semantic_change_F1"]
            else:
                score = 0.45 * metrics["cwld"]["macro_mIoU"] + 0.40 * metrics["fusu"]["average_macro_mIoU"] + 0.15 * metrics["fusu"]["semantic_change_F1"]
                macro_f1 = (metrics["cwld"]["macro_F1"] + metrics["fusu"]["semantic_change_F1"]) / 2
            candidate = {"selection_score": float(score), "macro_F1": float(macro_f1), "epoch": epoch}
            payload = _checkpoint_payload(model, tfr, change_head, optimizer, scheduler, scaler, epoch, global_step, score, metrics, config, repo, manifests, kind)
            save_checkpoint(last_path, payload)
            improved = select_better(candidate, best, kind)
            if improved:
                best = candidate
                best_name = {"cwld": "cwld_best_validation.pth", "fusu_sample": "fusu_sample_best_validation.pth", "tfr_sample": "temporal_tfr_sample_best_validation.pth", "joint_sample": "surface_joint_sample_best_validation.pth"}[kind]
                best_path = run / "checkpoints" / best_name
                save_checkpoint(best_path, payload)
                patience = 0
            else:
                patience += 1
            log = {"epoch": epoch, "global_step": global_step, "train_loss": float(np.mean(epoch_losses)), "selection_score": score, "improved": improved, "learning_rate": optimizer.param_groups[0]["lr"], "training_seconds": time.perf_counter() - epoch_started, "gpu_memory_peak": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, "metrics": metrics}
            with log_path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(log, ensure_ascii=False) + "\n")
            (run / "VALIDATION_REPORT.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            (run / "CHECKPOINT_SELECTION.json").write_text(json.dumps({"best": best, "candidate_status": config.get("candidate_status"), "full_fusu_validated": False if kind != "cwld" else None}, indent=2), encoding="utf-8")
            if patience >= int(config.get("early_stopping_patience", 8)):
                break
    except KeyboardInterrupt:
        interrupted = True
        receipt.update(status="interrupted", interrupted_at=_now(), last_epoch=max(start_epoch - 1, epoch - 1), global_step=global_step)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    except Exception as exc:
        receipt.update(status="failed", error=repr(exc), failed_at=_now(), global_step=global_step)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        raise

    checkpoints = list((run / "checkpoints").glob("*.pth"))
    write_hashes(run, checkpoints)
    receipt.update(status="success", completed_at=_now(), global_step=global_step, best=best, checkpoint_hashes={path.name: sha256(path) for path in checkpoints}, test_read_count=0, interrupted=interrupted)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "MODEL_CARD.md").write_text(
        f"# {config['experiment_name']}\n\n- Candidate scope: `{config.get('candidate_scope', 'cwld_validation')}`\n- Full FUSU validated: `false`\n- National generalization verified: `false`\n- Inference ready: `false`\n- Test reads: `0`\n",
        encoding="utf-8",
    )
    return receipt
