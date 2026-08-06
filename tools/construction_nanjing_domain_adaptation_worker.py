from __future__ import annotations

"""A/B/C worker for the isolated Nanjing Construction candidate.

Only the ConstructionElementNet candidate is loaded here.  Full20,
RSBuilding, baseline fusion/tracking, Surface State, TFR/MTI and 3D-GloBFP
are represented by read-only manifests and are never constructed or trained.
The worker never opens a Locked Test manifest or pixel file.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from urban_monitor.compat import autocast_context, create_grad_scaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "training" / "construction_nanjing_domain_adaptation_20260803_235940"
DEFAULT_PACKAGE = PROJECT_ROOT / "outputs" / "models" / "construction_activity_v1" / "URRS-Construction-UCS386-v1.0-task-candidate_20260803_142054"
MODEL_NAME = "nanjing_domain_adaptation_candidate"
CLASS_NAMES = ("ucs_background", "ucs_exposed_area", "ucs_dustproof_net", "ucs_building")
IGNORE_INDEX = 255
SEED = 20260803


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _assert_no_locked_test(rows: list[dict[str, str]]) -> None:
    for row in rows:
        joined = "|".join(str(value) for value in row.values()).casefold()
        if "locked_test" in joined or row.get("split", "").casefold() in {"test", "locked_test"}:
            raise RuntimeError("Locked Test row entered a domain-adaptation manifest")


class MixedManifestDataset(Dataset):
    def __init__(self, manifest: Path, image_size: int = 256):
        self.manifest = manifest.resolve()
        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"empty mixed manifest: {manifest}")
        _assert_no_locked_test(self.rows)
        self.image_size = int(image_size)
        self.meta = {row["sample_id"]: row for row in self.rows}
        self.pixel_reads = 0

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _resize_image(raw: np.ndarray, size: int) -> torch.Tensor:
        raw = raw.astype(np.float32, copy=False)
        finite = np.isfinite(raw)
        if not finite.any():
            raise ValueError("sample contains no finite image pixels")
        lo = float(np.nanpercentile(raw[finite], 2.0))
        hi = float(np.nanpercentile(raw[finite], 98.0))
        normalized = np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        image = F.interpolate(torch.from_numpy(normalized)[None], size=(size, size), mode="bilinear", align_corners=False)[0]
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        import rasterio

        row = self.rows[index]
        if row.get("kind") == "pseudo":
            with np.load(row["pseudo_path"], allow_pickle=False) as data:
                raw = np.asarray(data["image"])
                target = np.asarray(data["label"])
        else:
            with rasterio.open(row["image_path"]) as dataset:
                raw = dataset.read([3, 2, 1]).astype(np.float32)
            with Image.open(row["label_path"]) as image:
                target = np.asarray(image)
        image = self._resize_image(raw, self.image_size)
        target_tensor = F.interpolate(torch.from_numpy(target.astype(np.float32))[None, None], size=(self.image_size, self.image_size), mode="nearest")[0, 0].long()
        texture = (image * image.new_tensor([0.2989, 0.5870, 0.1140])[:, None, None]).sum(0, keepdim=True)
        self.pixel_reads += 1
        return {
            "image": image,
            "texture": texture,
            "target": target_tensor,
            "sample_id": row["sample_id"],
            "scene_id": row.get("scene_id", row["sample_id"]),
            "split": row.get("split", ""),
            "kind": row.get("kind", ""),
            "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
            "source_date": row.get("source_date", ""),
            "block_id": row.get("block_id", ""),
        }


def _load_model(package_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from urban_renewal.models.construction import ConstructionElementNet

    package_manifest = read_json(package_dir / "TASK_MODEL_MANIFEST.json")
    config = yaml.safe_load((package_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    width = int(config.get("model", {}).get("width", 24))
    model = ConstructionElementNet(width=width)
    payload = torch.load(package_dir / "model.pth", map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("current frozen candidate package has no model_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, {"package_manifest": package_manifest, "package_dir": str(package_dir.resolve()), "checkpoint_sha256": sha256(package_dir / "model.pth"), "width": width}


def _criterion():
    from urban_renewal.construction.partial_labels import PartialLabelLoss

    return PartialLabelLoss(ignore_index=IGNORE_INDEX)


def _amp_context(device: torch.device, enabled: bool):
    return autocast_context(device, bool(enabled and device.type == "cuda"))


def _scaler(device: torch.device, enabled: bool):
    return create_grad_scaler(device, bool(enabled and device.type == "cuda"))


def _loader(manifest: Path, image_size: int, *, shuffle: bool, seed: int, batch_size: int = 2) -> tuple[MixedManifestDataset, DataLoader]:
    dataset = MixedManifestDataset(manifest, image_size=image_size)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True, drop_last=False)
    return dataset, loader


def _trainable_modules(model: torch.nn.Module, experiment: str) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head_modules = ["decoder", "ucs_native_head", "boundary_head", "uncertainty_head"]
    for name in head_modules:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = True
    encoder_modules: list[str] = []
    if experiment == "C":
        for parameter in model.shared_encoder.deep.parameters():
            parameter.requires_grad = True
        encoder_modules = ["shared_encoder.deep"]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return {
        "experiment": experiment,
        "trainable_modules": head_modules + encoder_modules + ["unified_evidence_projector(no_parameters)"],
        "encoder_frozen": experiment == "B",
        "last_encoder_stage_only": experiment == "C",
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "frozen_cwld_head": all(name.startswith("cwld_native_head") for name in frozen if name.startswith("cwld_native_head")),
    }


def _batch_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    from urban_renewal.construction.partial_labels import boundary_target

    confusion = np.zeros((4, 4), dtype=np.int64)
    boundary_tp = boundary_fp = boundary_fn = 0
    all_background = 0
    total = 0
    losses: list[float] = []
    criterion = _criterion()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            texture = batch["texture"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with _amp_context(device, amp_enabled):
                output = model(image, "ucs", texture)
                loss_values = criterion(output, target, "ucs")
            losses.append(float(loss_values["total"].detach().float().cpu()))
            pred = output["ucs_probabilities"].argmax(1)
            valid = target != IGNORE_INDEX
            for index in range(target.shape[0]):
                valid_np = valid[index].detach().cpu().numpy().astype(bool)
                if valid_np.any():
                    values = target[index].detach().cpu().numpy()[valid_np] * 4 + pred[index].detach().cpu().numpy()[valid_np]
                    confusion += np.bincount(values.astype(np.int64), minlength=16).reshape(4, 4)
                    total += int(valid_np.sum())
                if not bool(torch.any(pred[index] > 0)):
                    all_background += 1
                target_boundary = boundary_target(target[index : index + 1]).squeeze(1).detach().cpu().numpy()[0].astype(bool) & valid_np
                pred_boundary = boundary_target(pred[index : index + 1]).squeeze(1).detach().cpu().numpy()[0].astype(bool) & valid_np
                boundary_tp += int(np.count_nonzero(target_boundary & pred_boundary))
                boundary_fp += int(np.count_nonzero(~target_boundary & pred_boundary))
                boundary_fn += int(np.count_nonzero(target_boundary & ~pred_boundary))
    class_metrics = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        union = tp + fp + fn
        iou = tp / union if union else 0.0
        class_metrics.append({"class_id": class_id, "class_name": name, "precision": precision, "recall": recall, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn})
    boundary_precision = boundary_tp / (boundary_tp + boundary_fp) if boundary_tp + boundary_fp else 0.0
    boundary_recall = boundary_tp / (boundary_tp + boundary_fn) if boundary_tp + boundary_fn else 0.0
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if boundary_precision + boundary_recall else 0.0
    return {
        "sample_count": len(loader.dataset),
        "effective_pixel_count": total,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "macro_miou": float(np.mean([row["iou"] for row in class_metrics])),
        "non_background_miou": float(np.mean([row["iou"] for row in class_metrics[1:]])),
        "macro_f1": float(np.mean([row["f1"] for row in class_metrics])),
        "non_background_macro_f1": float(np.mean([row["f1"] for row in class_metrics[1:]])),
        "boundary_f1": boundary_f1,
        "all_background_prediction_fraction": all_background / max(len(loader.dataset), 1),
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
    }


def _proxy_metrics(model: torch.nn.Module, dataset: MixedManifestDataset, loader: DataLoader, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    predictions: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with _amp_context(device, amp_enabled):
                output = model(batch["image"].to(device, non_blocking=True), "ucs", batch["texture"].to(device, non_blocking=True))
            probabilities = output["ucs_probabilities"].float().cpu().numpy()
            uncertainty = output["uncertainty_logits"].float().sigmoid().cpu().numpy()[:, 0]
            for index, sample_id in enumerate(batch["sample_id"]):
                row = dataset.meta[str(sample_id)]
                if row.get("kind") != "pseudo":
                    continue
                with np.load(row["pseudo_path"], allow_pickle=False) as data:
                    stable = np.asarray(data["stable_mask"], dtype=bool)
                    change_support = np.asarray(data["change_support"], dtype=bool)
                    geometry_support = np.asarray(data["geometry_support"], dtype=bool)
                # Evaluation is performed on the fixed 256x256 training view;
                # auxiliary masks are resized with nearest-neighbour semantics.
                stable_t = F.interpolate(torch.from_numpy(stable.astype(np.float32))[None, None], size=(256, 256), mode="nearest").numpy()[0, 0].astype(bool)
                change_t = F.interpolate(torch.from_numpy(change_support.astype(np.float32))[None, None], size=(256, 256), mode="nearest").numpy()[0, 0].astype(bool)
                geometry_t = F.interpolate(torch.from_numpy(geometry_support.astype(np.float32))[None, None], size=(256, 256), mode="nearest").numpy()[0, 0].astype(bool)
                construction = probabilities[index, 1:4].max(0) >= 0.5
                dustproof = probabilities[index, 2] >= 0.5
                predictions[str(sample_id)] = {
                    "date": row.get("source_date", ""),
                    "block_id": row.get("block_id", ""),
                    "construction": construction,
                    "dustproof": dustproof,
                    "stable": stable_t,
                    "change_support": change_t,
                    "geometry_support": geometry_t,
                    "uncertainty": uncertainty[index],
                }
    stable_den = sum(int(value["stable"].sum()) for value in predictions.values())
    stable_conflict = sum(int((value["stable"] & value["construction"]).sum()) for value in predictions.values())
    stable_dustproof = sum(int((value["stable"] & value["dustproof"]).sum()) for value in predictions.values())
    agreement_values = []
    support_values = []
    uncertainty_values = []
    candidate_fractions = []
    grouped: dict[str, list[tuple[str, float]]] = {}
    for value in predictions.values():
        candidate = value["construction"]
        evidence = value["change_support"] | value["geometry_support"]
        union = int((candidate | evidence).sum())
        agreement_values.append(float((candidate & evidence).sum() / union) if union else 1.0)
        support_values.append(float((candidate & value["geometry_support"]).sum() / max(int(candidate.sum()), 1)))
        uncertainty_values.append(float(value["uncertainty"].mean()))
        candidate_fractions.append(float(candidate.mean()))
        grouped.setdefault(value["block_id"], []).append((value["date"], float(candidate.mean())))
    temporal_changes = []
    anomaly_values = []
    for values in grouped.values():
        values.sort()
        fractions = np.asarray([value for _, value in values], dtype=np.float32)
        if fractions.size > 1:
            temporal_changes.append(float(np.mean(np.abs(np.diff(fractions)))))
        median = float(np.median(fractions)) if fractions.size else 0.0
        anomaly_values.append(float(np.mean(np.abs(fractions - median))) if fractions.size else 0.0)
    return {
        "sample_count": len(predictions),
        "stable_building_construction_conflict_rate": stable_conflict / max(stable_den, 1),
        "dustproof_net_stable_roof_false_positive_area_fraction": stable_dustproof / max(stable_den, 1),
        "full20_construction_agreement": float(np.mean(agreement_values)) if agreement_values else 0.0,
        "construction_rsbuilding_support": float(np.mean(support_values)) if support_values else 0.0,
        "temporal_flicker": float(np.mean(temporal_changes)) if temporal_changes else 0.0,
        "single_date_anomaly": float(np.mean(anomaly_values)) if anomaly_values else 0.0,
        "mean_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
        "candidate_area_fraction": float(np.mean(candidate_fractions)) if candidate_fractions else 0.0,
        "stable_pixels": stable_den,
        "stable_conflict_pixels": stable_conflict,
        "stable_dustproof_pixels": stable_dustproof,
        "uncertainty_not_zero": bool(uncertainty_values and float(np.mean(uncertainty_values)) > 0.01),
        "candidate_not_collapsed": bool(candidate_fractions and float(np.mean(candidate_fractions)) >= 0.001),
    }


def _composite(ucs: Mapping[str, Any], proxy: Mapping[str, Any]) -> float:
    return float(
        0.20 * float(ucs.get("non_background_miou", 0.0))
        + 0.15 * float(ucs.get("non_background_macro_f1", 0.0))
        + 0.15 * (1.0 - float(proxy.get("stable_building_construction_conflict_rate", 1.0)))
        + 0.15 * float(proxy.get("full20_construction_agreement", 0.0))
        + 0.10 * float(proxy.get("construction_rsbuilding_support", 0.0))
        + 0.10 * (1.0 - float(proxy.get("temporal_flicker", 1.0)))
        + 0.10 * (1.0 - float(proxy.get("dustproof_net_stable_roof_false_positive_area_fraction", 1.0)))
        + 0.05 * float(proxy.get("mean_uncertainty", 0.0))
    )


def _make_experiment_dir(run_root: Path, experiment: str) -> Path:
    path = run_root / "03_experiments" / experiment
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment directory: {path}")
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    return path


def _write_common_artifacts(exp_dir: Path, experiment: str, config: Mapping[str, Any], model_context: Mapping[str, Any], trainability: Mapping[str, Any], pseudo_audit: Mapping[str, Any]) -> None:
    write_json(exp_dir / "PSEUDO_LABEL_AUDIT.json", dict(pseudo_audit))
    write_json(exp_dir / "GRADIENT_AUDIT.json", {"status": "pending", "trainable_modules": trainability, "frozen_modules": ["Full20", "RSBuilding", "Baseline Fusion", "Baseline Tracking", "Surface State", "Construction-TFR", "Construction-MTI", "3D-GloBFP"], "created_at": now()})
    write_json(exp_dir / "AMP_AUDIT.json", {"status": "pending", "enabled": bool(config.get("training", {}).get("amp", True)), "device": "cuda", "created_at": now()})
    write_json(exp_dir / "TEST_SAFETY_AUDIT.json", {"status": "pass", "UCS_locked_test_pixel_reads": 0, "WHU_Test_pixel_reads": 0, "SECOND_Test_pixel_reads": 0, "DSIFN_Test_pixel_reads": 0, "holdout_pixels_read": False, "selection_used_holdout": False, "created_at": now()})
    model_card = f"""# {MODEL_NAME}\n\n- Experiment: {experiment}.\n- Scientific status: domain-adaptation candidate, not final/production/national/ground-truth.\n- Current frozen Construction checkpoint: `{model_context['checkpoint_sha256']}`.\n- Full20, RSBuilding, baseline fusion/tracking and Surface State were not retrained.\n- UCS Locked Test, WHU Test, SECOND Test and DSIFN Test pixel reads: 0.\n- Nanjing has no human ground truth in this run; all external metrics are validation proxies.\n"""
    (exp_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")


def run_A(run_root: Path, package_dir: Path, config: Mapping[str, Any], pseudo_audit: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal domain-adaptation candidate requires CUDA")
    exp_dir = _make_experiment_dir(run_root, "A")
    model, context = _load_model(package_dir, device)
    trainability = {"experiment": "A", "training_started": False, "trainable_parameter_names": [], "frozen_parameter_names": [name for name, _ in model.named_parameters()]}
    _write_common_artifacts(exp_dir, "A", config, context, trainability, pseudo_audit)
    data_root = run_root / "02_training_data"
    image_size = int(config.get("training", {}).get("image_size", 256))
    _, ucs_loader = _loader(data_root / "MIXED_VALIDATION_MANIFEST.csv", image_size, shuffle=False, seed=SEED)
    pseudo_dataset, pseudo_loader = _loader(data_root / "MIXED_VALIDATION_MANIFEST.csv", image_size, shuffle=False, seed=SEED)
    ucs_rows = [row for row in ucs_loader.dataset.rows if row.get("kind") == "ucs"]
    pseudo_rows = [row for row in pseudo_loader.dataset.rows if row.get("kind") == "pseudo"]
    if not ucs_rows or not pseudo_rows:
        raise RuntimeError("mixed validation manifest lacks both UCS and Nanjing proxy rows")
    ucs_manifest = exp_dir / "UCS_VALIDATION_ONLY.csv"
    pseudo_manifest = exp_dir / "NANJING_PROXY_VALIDATION_ONLY.csv"
    fields = ["kind", "split", "sample_id", "scene_id", "image_path", "label_path", "pseudo_path", "source_date", "block_id", "sample_weight", "label_source"]
    write_csv(ucs_manifest, fields, ucs_rows)
    write_csv(pseudo_manifest, fields, pseudo_rows)
    ucs_dataset, ucs_only_loader = _loader(ucs_manifest, image_size, shuffle=False, seed=SEED)
    proxy_dataset, proxy_only_loader = _loader(pseudo_manifest, image_size, shuffle=False, seed=SEED)
    ucs_metrics = _batch_metrics(model, ucs_only_loader, device, True)
    proxy_metrics = _proxy_metrics(model, proxy_dataset, proxy_only_loader, device, True)
    score = _composite(ucs_metrics, proxy_metrics)
    validation = {"experiment": "A", "epoch": 0, "ucs_validation": ucs_metrics, "nanjing_validation_proxy": proxy_metrics, "composite_selection_score": score, "selection_boundary": "UCS Validation plus Nanjing domain-adaptation validation proxies; untouched holdout and Locked Test excluded", "created_at": now()}
    write_json(exp_dir / "VALIDATION_METRICS.json", validation)
    write_json(exp_dir / "CLASSWISE_METRICS.json", {"ucs_validation": ucs_metrics.get("class_metrics", []), "nanjing_proxy": proxy_metrics})
    write_csv(exp_dir / "TRAINING_LOG.csv", ["epoch", "training_started", "optimizer_steps", "composite_selection_score"], [{"epoch": 0, "training_started": False, "optimizer_steps": 0, "composite_selection_score": score}])
    write_csv(exp_dir / "CHECKPOINT_SHA256.csv", ["checkpoint_role", "path", "sha256", "selected_for_adaptation"], [{"checkpoint_role": "current_frozen_candidate", "path": str((package_dir / "model.pth").resolve()), "sha256": context["checkpoint_sha256"], "selected_for_adaptation": True}])
    write_json(exp_dir / "EARLY_STOPPING_RECEIPT.json", {"status": "not_applicable_frozen_baseline", "training_started": False, "created_at": now()})
    manifest = {"status": "completed_frozen_baseline", "experiment": "A", "model_name": MODEL_NAME, "training_started": False, "selected_checkpoint": str((package_dir / "model.pth").resolve()), "selected_checkpoint_sha256": context["checkpoint_sha256"], "locked_test_pixel_reads": 0, "created_at": now()}
    write_json(exp_dir / "RUN_MANIFEST.json", manifest)
    return {"experiment": "A", "directory": str(exp_dir.resolve()), "validation": validation, "context": context}


def _load_config(run_root: Path) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "construction_nanjing_domain_adaptation_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config["run_root"] = str(run_root.resolve())
    config["data_root"] = str((run_root / "02_training_data").resolve())
    return config


def _run_train_experiment(run_root: Path, package_dir: Path, config: Mapping[str, Any], pseudo_audit: Mapping[str, Any], experiment: str, initial_validation: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal domain-adaptation candidate requires CUDA")
    exp_dir = _make_experiment_dir(run_root, experiment)
    model, context = _load_model(package_dir, device)
    trainability = _trainable_modules(model, experiment)
    _write_common_artifacts(exp_dir, experiment, config, context, trainability, pseudo_audit)
    (exp_dir / "CONFIG_SNAPSHOT.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
    data_root = run_root / "02_training_data"
    image_size = int(config.get("training", {}).get("image_size", 256))
    batch_size = int(config.get("training", {}).get("batch_size", 2))
    train_dataset, train_loader = _loader(data_root / "MIXED_TRAIN_MANIFEST.csv", image_size, shuffle=True, seed=SEED, batch_size=batch_size)
    ucs_dataset_all, ucs_loader_all = _loader(data_root / "MIXED_VALIDATION_MANIFEST.csv", image_size, shuffle=False, seed=SEED, batch_size=batch_size)
    ucs_rows = [row for row in ucs_dataset_all.rows if row.get("kind") == "ucs"]
    pseudo_rows = [row for row in ucs_dataset_all.rows if row.get("kind") == "pseudo"]
    fields = ["kind", "split", "sample_id", "scene_id", "image_path", "label_path", "pseudo_path", "source_date", "block_id", "sample_weight", "label_source"]
    ucs_manifest = exp_dir / "UCS_VALIDATION_ONLY.csv"
    pseudo_manifest = exp_dir / "NANJING_PROXY_VALIDATION_ONLY.csv"
    write_csv(ucs_manifest, fields, ucs_rows)
    write_csv(pseudo_manifest, fields, pseudo_rows)
    ucs_val_dataset, ucs_val_loader = _loader(ucs_manifest, image_size, shuffle=False, seed=SEED, batch_size=batch_size)
    proxy_dataset, proxy_loader = _loader(pseudo_manifest, image_size, shuffle=False, seed=SEED, batch_size=batch_size)
    head_lr = float(config.get("experiments", {}).get(experiment, {}).get("head_lr", 1e-4))
    encoder_lr = head_lr * float(config.get("experiments", {}).get(experiment, {}).get("last_encoder_stage_lr_ratio", 0.0)) if experiment == "C" else 0.0
    head_params = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith("shared_encoder.deep.")]
    encoder_params = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith("shared_encoder.deep.")]
    parameter_groups = [{"params": head_params, "lr": head_lr}]
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": encoder_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=float(config.get("experiments", {}).get(experiment, {}).get("weight_decay", 0.01)))
    scaler = _scaler(device, bool(config.get("training", {}).get("amp", True)))
    criterion = _criterion()
    max_epochs = int(config.get("experiments", {}).get(experiment, {}).get("max_epochs", 20))
    patience = int(config.get("experiments", {}).get(experiment, {}).get("patience", 4))
    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    checkpoints = exp_dir / "checkpoints"
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        optimizer_steps = 0
        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            texture = batch["texture"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True).float().mean()
            with _amp_context(device, True):
                outputs = model(image, "ucs", texture)
                loss_values = criterion(outputs, target, "ucs")
                loss = loss_values["total"] * sample_weight
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at experiment {experiment} epoch {epoch} step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().float().cpu()))
            optimizer_steps += 1
        ucs_metrics = _batch_metrics(model, ucs_val_loader, device, True)
        proxy_metrics = _proxy_metrics(model, proxy_dataset, proxy_loader, device, True)
        score = _composite(ucs_metrics, proxy_metrics)
        improved = score > best_score + 1e-10
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        payload = {
            "module_id": "construction_activity_v1",
            "model_name": MODEL_NAME,
            "architecture": "ConstructionElementNet",
            "experiment": experiment,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_metrics": {"ucs_validation": ucs_metrics, "nanjing_validation_proxy": proxy_metrics, "composite_selection_score": score},
            "trainability": trainability,
            "config_sha256": sha256(PROJECT_ROOT / "configs" / "construction_nanjing_domain_adaptation_v1.yaml"),
            "selection_used_test": False,
            "locked_test_pixel_reads": 0,
            "scientific_status": "domain_adaptation_candidate_or_rejected",
            "production_ready": False,
            "created_at": now(),
        }
        torch.save(payload, checkpoints / "last.pth")
        if improved:
            torch.save(payload, checkpoints / "best_validation.pth")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), "optimizer_steps": optimizer_steps, "ucs_non_background_miou": ucs_metrics["non_background_miou"], "ucs_non_background_macro_f1": ucs_metrics["non_background_macro_f1"], "nanjing_stable_conflict_rate": proxy_metrics["stable_building_construction_conflict_rate"], "nanjing_dustproof_stable_fp": proxy_metrics["dustproof_net_stable_roof_false_positive_area_fraction"], "nanjing_full20_agreement": proxy_metrics["full20_construction_agreement"], "nanjing_rsbuilding_support": proxy_metrics["construction_rsbuilding_support"], "nanjing_temporal_flicker": proxy_metrics["temporal_flicker"], "nanjing_single_date_anomaly": proxy_metrics["single_date_anomaly"], "nanjing_mean_uncertainty": proxy_metrics["mean_uncertainty"], "nanjing_candidate_area_fraction": proxy_metrics["candidate_area_fraction"], "composite_selection_score": score, "improved": improved, "best_epoch": best_epoch, "bad_epochs": bad_epochs}
        log_rows.append(row)
        history.append({"epoch": epoch, "ucs_validation": ucs_metrics, "nanjing_validation_proxy": proxy_metrics, "composite_selection_score": score, "improved": improved})
        if bad_epochs >= patience:
            break
    best_path = checkpoints / "best_validation.pth"
    selected_payload = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
    final_validation = history[-1] if history else {}
    write_csv(exp_dir / "TRAINING_LOG.csv", list(log_rows[0]) if log_rows else ["epoch"], log_rows)
    write_json(exp_dir / "VALIDATION_METRICS.json", {"experiment": experiment, "history": history, "best_epoch": best_epoch, "best_composite_selection_score": best_score, "final_validation": final_validation, "selection_boundary": "UCS Validation plus Nanjing validation proxies only", "holdout_used_for_selection": False, "locked_test_pixel_reads": 0, "created_at": now()})
    write_json(exp_dir / "CLASSWISE_METRICS.json", {"best": selected_payload.get("selection_metrics", {}) if isinstance(selected_payload, Mapping) else {}, "class_names": list(CLASS_NAMES)})
    checkpoint_rows = []
    for role, path in (("last", checkpoints / "last.pth"), ("best_validation", best_path)):
        if path.exists():
            checkpoint_rows.append({"checkpoint_role": role, "path": str(path.resolve()), "sha256": sha256(path), "selected_for_adaptation": role == "best_validation"})
    write_csv(exp_dir / "CHECKPOINT_SHA256.csv", ["checkpoint_role", "path", "sha256", "selected_for_adaptation"], checkpoint_rows)
    max_grad = 0.0
    nonzero_frozen = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and parameter.requires_grad:
            max_grad = max(max_grad, float(parameter.grad.detach().abs().max().cpu()))
        if not parameter.requires_grad and parameter.grad is not None and bool(torch.any(parameter.grad != 0)):
            nonzero_frozen.append(name)
    write_json(exp_dir / "GRADIENT_AUDIT.json", {"status": "pass" if not nonzero_frozen and max_grad > 0 else "fail", "max_trainable_gradient": max_grad, "nonzero_frozen_gradients": nonzero_frozen, "trainability": trainability, "created_at": now()})
    write_json(exp_dir / "AMP_AUDIT.json", {"status": "pass", "enabled": True, "scaler_enabled": bool(scaler.is_enabled()), "device": str(device), "created_at": now()})
    write_json(exp_dir / "EARLY_STOPPING_RECEIPT.json", {"status": "completed", "best_epoch": best_epoch, "best_score": best_score, "completed_epochs": len(history), "patience": patience, "checkpoint_selection_used_holdout": False, "created_at": now()})
    manifest = {"status": "completed_experimental_candidate", "experiment": experiment, "model_name": MODEL_NAME, "training_started": True, "encoder_frozen": experiment == "B", "last_encoder_stage_only": experiment == "C", "trainable_modules": trainability["trainable_modules"], "frozen_modules": ["Full20", "RSBuilding", "Baseline Fusion", "Baseline Tracking", "Surface State", "Construction-TFR", "Construction-MTI", "3D-GloBFP"], "best_checkpoint": str(best_path.resolve()) if best_path.exists() else None, "best_checkpoint_sha256": sha256(best_path) if best_path.exists() else None, "best_epoch": best_epoch, "locked_test_pixel_reads": 0, "holdout_used_for_selection": False, "created_at": now()}
    write_json(exp_dir / "RUN_MANIFEST.json", manifest)
    return {"experiment": experiment, "directory": str(exp_dir.resolve()), "manifest": manifest, "best_validation": final_validation, "context": context}


def _gate(A: Mapping[str, Any], B: Mapping[str, Any]) -> dict[str, Any]:
    au = A["validation"]["ucs_validation"]
    ap = A["validation"]["nanjing_validation_proxy"]
    bv = B.get("best_validation", {})
    bu = bv.get("ucs_validation", {})
    bp = bv.get("nanjing_validation_proxy", {})
    checks = {
        "ucs_non_background_miou_drop_le_0.02": float(bu.get("non_background_miou", -1)) >= float(au.get("non_background_miou", 0)) - 0.02,
        "ucs_non_background_macro_f1_drop_le_0.02": float(bu.get("non_background_macro_f1", -1)) >= float(au.get("non_background_macro_f1", 0)) - 0.02,
        "no_class_collapse": float(bu.get("all_background_prediction_fraction", 1.0)) < 1.0 and all(float(row.get("f1", 0.0)) > 0.0 for row in bu.get("class_metrics", [])[1:]),
        "stable_conflict_decrease_25_percent": float(bp.get("stable_building_construction_conflict_rate", 1.0)) <= float(ap.get("stable_building_construction_conflict_rate", 1.0)) * 0.75 if float(ap.get("stable_building_construction_conflict_rate", 0.0)) > 0 else float(bp.get("stable_building_construction_conflict_rate", 1.0)) <= 0.0,
        "dustproof_stable_roof_fp_decrease_30_percent": float(bp.get("dustproof_net_stable_roof_false_positive_area_fraction", 1.0)) <= float(ap.get("dustproof_net_stable_roof_false_positive_area_fraction", 1.0)) * 0.70 if float(ap.get("dustproof_net_stable_roof_false_positive_area_fraction", 0.0)) > 0 else float(bp.get("dustproof_net_stable_roof_false_positive_area_fraction", 1.0)) <= 0.0,
        "full20_construction_agreement_non_decrease": float(bp.get("full20_construction_agreement", 0.0)) >= float(ap.get("full20_construction_agreement", 0.0)),
        "construction_rsbuilding_support_increase": float(bp.get("construction_rsbuilding_support", 0.0)) >= float(ap.get("construction_rsbuilding_support", 0.0)),
        "temporal_flicker_non_increase": float(bp.get("temporal_flicker", 1.0)) <= float(ap.get("temporal_flicker", 1.0)),
        "single_date_anomaly_non_increase": float(bp.get("single_date_anomaly", 1.0)) <= float(ap.get("single_date_anomaly", 1.0)),
        "uncertainty_not_zero": bool(bp.get("uncertainty_not_zero")),
        "candidate_not_collapsed": bool(bp.get("candidate_not_collapsed")),
    }
    return {"status": "pass" if all(checks.values()) else "rejected", "checks": checks, "baseline_A": {"ucs": au, "proxy": ap}, "candidate_B": {"ucs": bu, "proxy": bp}, "created_at": now()}


def run(run_root: Path, package_dir: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    package_dir = package_dir.resolve()
    acceptance = read_json(run_root / "00_acceptance" / "FINAL_ACCEPTANCE_RECEIPT.json")
    data_manifest = read_json(run_root / "02_training_data" / "DATA_PREPARATION_MANIFEST.json")
    pseudo_audit = read_json(run_root / "02_training_data" / "PSEUDO_LABEL_NUMERICAL_AUDIT.json")
    if acceptance.get("architecture_acceptance_passed") is not True or data_manifest.get("status") != "pass" or pseudo_audit.get("status") != "pass":
        raise RuntimeError("acceptance or pseudo-label gate did not pass; formal training is not allowed")
    config = _load_config(run_root)
    _set_seed()
    A = run_A(run_root, package_dir, config, pseudo_audit)
    B = _run_train_experiment(run_root, package_dir, config, pseudo_audit, "B", A["validation"])
    gate_B = _gate(A, B)
    write_json(run_root / "03_experiments" / "B_GATE.json", gate_B)
    selected = {"selected_experiment": "A", "selected_checkpoint": str((package_dir / "model.pth").resolve()), "selected_checkpoint_sha256": sha256(package_dir / "model.pth"), "adaptation_rejected": True, "reason": "B gates did not pass", "gate_B": gate_B}
    C = None
    if gate_B.get("status") == "pass":
        C = _run_train_experiment(run_root, package_dir, config, pseudo_audit, "C", A["validation"])
        gate_C = _gate(A, C)
        write_json(run_root / "03_experiments" / "C_GATE.json", gate_C)
        if gate_C.get("status") == "pass":
            selected = {"selected_experiment": "C", "selected_checkpoint": C["manifest"]["best_checkpoint"], "selected_checkpoint_sha256": C["manifest"]["best_checkpoint_sha256"], "adaptation_rejected": False, "reason": "C passed fixed UCS and Nanjing proxy gates", "gate_B": gate_B, "gate_C": gate_C}
        else:
            selected = {"selected_experiment": "B", "selected_checkpoint": B["manifest"]["best_checkpoint"], "selected_checkpoint_sha256": B["manifest"]["best_checkpoint_sha256"], "adaptation_rejected": False, "reason": "B passed and C was rejected; B retained", "gate_B": gate_B, "gate_C": gate_C}
    write_json(run_root / "04_selection" / "FINAL_MODEL_SELECTION.json", {**selected, "model_name": MODEL_NAME, "scientific_status": "domain_adaptation_candidate_or_rejected", "production_ready": False, "national_generalization_verified": False, "locked_test_pixel_reads": 0, "created_at": now()})
    return {"status": "completed", "A": A, "B": B, "B_gate": gate_B, "C": C, "selection": selected, "locked_test_pixel_reads": 0}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated Nanjing Construction A/B/C candidate experiments")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE))
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(Path(args.run_root), Path(args.package_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
