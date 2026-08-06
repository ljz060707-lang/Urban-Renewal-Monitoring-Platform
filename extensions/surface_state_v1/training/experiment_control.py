from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from urban_monitor.compat import create_grad_scaler

from ..models.surface_state_net import SurfaceStateNet
from ..models.temporal_change_head import TemporalChangeHead
from ..models.tfr import TemporalFeatureRefinement
from .checkpointing import save_checkpoint, sha256
from .runtime_compat import train


def _load_best(run: Path) -> tuple[Path, dict]:
    paths = list((run / "checkpoints").glob("*best_validation.pth"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one best checkpoint in {run}, found {len(paths)}")
    path = paths[0]
    return path, torch.load(path, map_location="cpu", weights_only=False)


def select_temporal_branch(
    baseline_run: str | Path, tfr_run: str | Path, output: str | Path
) -> dict:
    baseline_path, baseline = _load_best(Path(baseline_run))
    tfr_path, tfr = _load_best(Path(tfr_run))
    left = baseline["validation_metrics"]["fusu"]
    right = tfr["validation_metrics"]["fusu"]
    miou_delta = right["average_macro_mIoU"] - left["average_macro_mIoU"]
    change_delta = right["semantic_change_F1"] - left["semantic_change_F1"]
    violation_delta = (
        left["temporal_consistency_violation_rate"]
        - right["temporal_consistency_violation_rate"]
    )
    criteria = {
        "higher_macro_mIoU": miou_delta > 0,
        "higher_change_f1_with_miou_guard": change_delta > 0 and miou_delta >= -0.005,
        "lower_violation_without_material_regression": (
            violation_delta >= 0.01 and miou_delta >= -0.005 and change_delta >= -0.005
        ),
    }
    selected_tfr = any(criteria.values())
    selection = {
        "selected_temporal_branch": "tfr" if selected_tfr else "baseline",
        "tfr_status": (
            "tfr_sample_validation_candidate"
            if selected_tfr
            else "implemented_but_not_selected"
        ),
        "baseline_checkpoint": str(baseline_path.resolve()),
        "baseline_sha256": sha256(baseline_path),
        "tfr_checkpoint": str(tfr_path.resolve()),
        "tfr_sha256": sha256(tfr_path),
        "selected_checkpoint": str((tfr_path if selected_tfr else baseline_path).resolve()),
        "criteria": criteria,
        "delta_average_macro_mIoU": miou_delta,
        "delta_semantic_change_F1": change_delta,
        "temporal_consistency_violation_improvement": violation_delta,
        "validation_scope": "FUSU 500-pair grouped sample Validation; weak spatial isolation",
        "full_fusu_validated": False,
        "test_reads": 0,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "FUSU_BASELINE_VS_TFR.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "branch",
                "average_macro_mIoU",
                "semantic_change_F1",
                "temporal_consistency_violation_rate",
                "checkpoint",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "branch": "baseline",
                "average_macro_mIoU": left["average_macro_mIoU"],
                "semantic_change_F1": left["semantic_change_F1"],
                "temporal_consistency_violation_rate": left[
                    "temporal_consistency_violation_rate"
                ],
                "checkpoint": baseline_path,
                "sha256": sha256(baseline_path),
            }
        )
        writer.writerow(
            {
                "branch": "tfr",
                "average_macro_mIoU": right["average_macro_mIoU"],
                "semantic_change_F1": right["semantic_change_F1"],
                "temporal_consistency_violation_rate": right[
                    "temporal_consistency_violation_rate"
                ],
                "checkpoint": tfr_path,
                "sha256": sha256(tfr_path),
            }
        )
    (output / "FUSU_TEMPORAL_BRANCH_SELECTION.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def _component(state: dict, prefix: str) -> dict:
    values = {key: value for key, value in state.items() if key.startswith(prefix + ".")}
    if not values:
        raise RuntimeError(f"Checkpoint has no required component: {prefix}")
    return values


def _merge_joint_model(cwld: dict, fusu: dict, encoder_source: str) -> SurfaceStateNet:
    model = SurfaceStateNet()
    target = model.state_dict()
    shared = cwld["model_state_dict"] if encoder_source == "cwld" else fusu["model_state_dict"]
    sources = [
        _component(shared, "encoder"),
        _component(shared, "decoder"),
        _component(cwld["model_state_dict"], "cwld_decoder_head"),
        _component(fusu["model_state_dict"], "fusu_decoder_head"),
    ]
    merged = {}
    for source in sources:
        merged.update(source)
    missing = sorted(set(target) - set(merged))
    unexpected = sorted(set(merged) - set(target))
    if missing or unexpected:
        raise RuntimeError(
            f"Strict joint component load failed: missing={missing}, unexpected={unexpected}"
        )
    for key, value in merged.items():
        if target[key].shape != value.shape:
            raise RuntimeError(
                f"Joint component shape mismatch for {key}: {value.shape} != {target[key].shape}"
            )
    model.load_state_dict(merged, strict=True)
    return model


def _prepare_joint_run(
    config: dict,
    run: Path,
    cwld_checkpoint: Path,
    fusu_checkpoint: Path,
    encoder_source: str,
) -> dict:
    run.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "logs", "metrics", "visualizations"):
        (run / name).mkdir()
    cwld = torch.load(cwld_checkpoint, map_location="cpu", weights_only=False)
    fusu = torch.load(fusu_checkpoint, map_location="cpu", weights_only=False)
    model = _merge_joint_model(cwld, fusu, encoder_source)
    use_tfr = bool(config.get("use_tfr", False))
    tfr = (
        TemporalFeatureRefinement(
            24,
            heads=int(config.get("temporal", {}).get("num_heads", 4)),
            dropout=float(config.get("temporal", {}).get("dropout", 0.1)),
        )
        if use_tfr
        else None
    )
    if tfr is not None:
        source = fusu.get("tfr_state_dict")
        if source is None:
            raise RuntimeError("Selected TFR branch has no tfr_state_dict")
        tfr.load_state_dict(source, strict=True)
    change_head = TemporalChangeHead(24)
    if fusu.get("change_head_state_dict") is not None:
        change_head.load_state_dict(fusu["change_head_state_dict"], strict=True)
    parameters = list(model.parameters())
    if tfr is not None:
        parameters += list(tfr.parameters())
    parameters += list(change_head.parameters())
    optimizer_config = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config.get("decoder_lr", 2e-4)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["epochs"]))
    )
    scaler = create_grad_scaler("cpu", False)
    payload = {
        "model_state_dict": model.state_dict(),
        "tfr_state_dict": tfr.state_dict() if tfr is not None else None,
        "change_head_state_dict": change_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": -1,
        "global_step": 0,
        "selection_score": float("-inf"),
        "validation_metrics": {},
        "config_snapshot": config,
        "warmstart": {
            "encoder_source": encoder_source,
            "cwld_checkpoint": str(cwld_checkpoint.resolve()),
            "cwld_sha256": sha256(cwld_checkpoint),
            "fusu_checkpoint": str(fusu_checkpoint.resolve()),
            "fusu_sha256": sha256(fusu_checkpoint),
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
        },
    }
    save_checkpoint(run / "checkpoints" / "last.pth", payload)
    audit = payload["warmstart"]
    (run / "JOINT_CHECKPOINT_LOAD_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def run_joint_probes(
    base_config: str | Path,
    cwld_checkpoint: str | Path,
    fusu_checkpoint: str | Path,
    use_tfr: bool,
    output_root: str | Path,
    *,
    repo: str | Path,
) -> dict:
    repo, output_root = Path(repo), Path(output_root)
    config = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    config.update(epochs=2, early_stopping_patience=2, use_tfr=bool(use_tfr))
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for source in ("cwld", "fusu"):
        run = output_root / f"probe_{source}"
        _prepare_joint_run(
            config, run, Path(cwld_checkpoint), Path(fusu_checkpoint), source
        )
        probe_config = run / "probe_config.yaml"
        probe_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        receipt = train(probe_config, repo, run_dir=str(run))
        checkpoint, state = _load_best(run)
        metrics = state["validation_metrics"]
        score = (
            0.45 * metrics["cwld"]["macro_mIoU"]
            + 0.40 * metrics["fusu"]["average_macro_mIoU"]
            + 0.15 * metrics["fusu"]["semantic_change_F1"]
        )
        results.append(
            {
                "encoder_source": source,
                "selection_score": score,
                "cwld_macro_mIoU": metrics["cwld"]["macro_mIoU"],
                "fusu_average_macro_mIoU": metrics["fusu"]["average_macro_mIoU"],
                "fusu_semantic_change_F1": metrics["fusu"]["semantic_change_F1"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "status": receipt["status"],
                "test_reads": 0,
            }
        )
    selected = max(results, key=lambda row: row["selection_score"])
    with (output_root / "JOINT_WARMSTART_PROBES.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    selection = {
        "selected_encoder_source": selected["encoder_source"],
        "selected_probe_checkpoint": selected["checkpoint"],
        "probes": results,
        "selection_formula": "0.45*cwld_mIoU + 0.40*fusu_mIoU + 0.15*fusu_change_F1",
        "test_reads": 0,
    }
    (output_root / "JOINT_WARMSTART_SELECTION.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection


def run_joint_from_selection(
    base_config: str | Path,
    selection_path: str | Path,
    output_root: str | Path,
    *,
    repo: str | Path,
) -> dict:
    repo, output_root = Path(repo), Path(output_root)
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    source_checkpoint = Path(selection["selected_probe_checkpoint"])
    state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    config = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    config["use_tfr"] = state.get("tfr_state_dict") is not None
    output_root.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "logs", "metrics", "visualizations"):
        (output_root / name).mkdir()
    model = SurfaceStateNet()
    model.load_state_dict(state["model_state_dict"], strict=True)
    tfr = (
        TemporalFeatureRefinement(
            24,
            heads=int(config.get("temporal", {}).get("num_heads", 4)),
            dropout=float(config.get("temporal", {}).get("dropout", 0.1)),
        )
        if config["use_tfr"]
        else None
    )
    if tfr is not None:
        tfr.load_state_dict(state["tfr_state_dict"], strict=True)
    change_head = TemporalChangeHead(24)
    change_head.load_state_dict(state["change_head_state_dict"], strict=True)
    parameters = list(model.parameters()) + list(change_head.parameters())
    if tfr is not None:
        parameters += list(tfr.parameters())
    optimizer_config = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config.get("decoder_lr", 2e-4)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["epochs"]))
    )
    scaler = create_grad_scaler("cpu", False)
    payload = dict(state)
    payload.update(
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        epoch=-1,
        global_step=0,
        selection_score=float("-inf"),
        validation_metrics={},
        config_snapshot=config,
        warmstart_selection=selection,
    )
    save_checkpoint(output_root / "checkpoints" / "last.pth", payload)
    config_path = output_root / "joint_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return train(config_path, repo, run_dir=str(output_root))
