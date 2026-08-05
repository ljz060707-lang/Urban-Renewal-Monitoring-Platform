from __future__ import annotations

"""Formal, load-only closeout for the existing UCS-386 Stage A run.

The closeout is intentionally separated from the training worker.  It audits
the recovery best checkpoint, re-evaluates only the fixed Train/Validation
selection boundary, freezes the candidate, evaluates the 46 locked Test
samples exactly once, and then packages the same checkpoint.  There is no
training, resume, Stage B, split mutation, or source-data write in this file.
"""

import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from urban_renewal.construction.json_io import atomic_write_text, dump_json, load_json


MODEL_NAME = "URRS-Construction-UCS386-v1.0-task-candidate"
CLASS_NAMES = ("ucs_background", "ucs_exposed_area", "ucs_dustproof_net", "ucs_building")
RGB_MAPPING = (3, 2, 1)
FIXED_WEIGHTS = {
    "non_background_mIoU": 0.45,
    "non_background_macro_F1": 0.25,
    "boundary_F1": 0.20,
    "one_minus_uncertainty_ECE": 0.10,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return load_json(path)


def _csv_text(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(headers))
    for row in rows:
        writer.writerow(list(row))
    return stream.getvalue()


def _write_csv(path: Path, headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    atomic_write_text(path, _csv_text(headers, rows))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping YAML: {path}")
    return payload


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fixed_selection_score(metrics: Mapping[str, Any]) -> float:
    values = (
        float(metrics["non_background_miou"]),
        float(metrics["non_background_macro_f1"]),
        float(metrics["boundary_f1"]),
        float(metrics["uncertainty_ece"]),
    )
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    return (
        FIXED_WEIGHTS["non_background_mIoU"] * values[0]
        + FIXED_WEIGHTS["non_background_macro_F1"] * values[1]
        + FIXED_WEIGHTS["boundary_F1"] * values[2]
        + FIXED_WEIGHTS["one_minus_uncertainty_ECE"] * (1.0 - values[3])
    )


def _split_hashes(gate_dir: Path, config: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "gate": gate_dir / "UCS386_SUBSET_GATE.json",
        "train_manifest": Path(config["dataset"]["train_manifest"]),
        "validation_manifest": Path(config["dataset"]["validation_manifest"]),
        "locked_test_manifest": Path(config["dataset"]["locked_test_manifest"]),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _load_checkpoint_audit(run_dir: Path, checkpoint: Path, config: Mapping[str, Any], state: Mapping[str, Any], gate: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    import torch

    from construction_training_worker import _model_and_loss
    from urban_renewal.models.construction import ConstructionElementNet

    checkpoint_hash = _sha256(checkpoint)
    model, criterion = _model_and_loss(int(config["model"].get("width", 24)))
    payload: Any = None
    load_error: str | None = None
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint payload is not a mapping")
        model_state = payload.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint has no model_state_dict")
        incompatible = model.load_state_dict(model_state, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(f"model state mismatch: missing={missing_keys}, unexpected={unexpected_keys}")
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    old_info = state.get("checkpoint_info") if isinstance(state.get("checkpoint_info"), Mapping) else {}
    input_weight_shapes = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if name.endswith("weight") and len(parameter.shape) == 4
    }
    audit = {
        "status": "success" if load_error is None else "CHECKPOINT_LOAD_FAILED",
        "checkpoint": str(checkpoint.resolve()),
        "sha256": checkpoint_hash,
        "architecture": "ConstructionElementNet",
        "width": int(config["model"].get("width", 24)),
        "state_dict_missing_keys": missing_keys,
        "state_dict_unexpected_keys": unexpected_keys,
        "strict_model_load": load_error is None,
        "load_error": load_error,
        "class_order": list(CLASS_NAMES),
        "input_channels": 3,
        "texture_channels": 1,
        "input_mode": "rgb_plus_external_gray",
        "rgb_mapping": list(RGB_MAPPING),
        "rgb_semantic_order": ["R", "G", "B"],
        "preprocessing": "per_image_percentile_2_98",
        "checkpoint_payload_keys": sorted(str(key) for key in payload.keys()) if isinstance(payload, Mapping) else [],
        "checkpoint_config_sha256": payload.get("config_sha256") if isinstance(payload, Mapping) else None,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "four_dimensional_weight_shapes": input_weight_shapes,
        "training_resume_checkpoint_complete": bool(old_info.get("complete", False)),
        "training_resume_checkpoint_missing_fields": list(old_info.get("missing_fields", [])),
        "inference_checkpoint_loadable": load_error is None,
        "old_last_checkpoint_used_for_selection": False,
        "old_last_checkpoint_used_for_test": False,
        "gate_rgb_mapping_verified": gate.get("rgb_mapping", {}).get("verified") is True,
        "audited_at": _now(),
    }
    return audit, model, criterion, payload if isinstance(payload, Mapping) else {}


def _training_trace_audit(run_dir: Path, artifact_dir: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    log_path = artifact_dir / "logs" / "training.jsonl"
    rows: list[dict[str, Any]] = []
    if log_path.exists():
        for line_number, line in enumerate(log_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid training log line {line_number}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    stderr_candidates = [run_dir / "logs" / "training_resume_stderr.log", run_dir / "logs" / "detached_stderr.log"]
    stderr_text = "\n".join(path.read_text(encoding="utf-8-sig", errors="replace") for path in stderr_candidates if path.exists())
    progress_path = run_dir / "progress.json"
    progress_payload = _json(progress_path) if progress_path.exists() else None
    nonfinite_count = 0
    nonfinite_selection_score_count = 0
    for row in rows:
        train = row.get("train", {})
        validation = row.get("validation", {})
        values = [train.get("mean_loss"), train.get("total_loss"), train.get("loss"), validation.get("loss")]
        nonfinite_count += sum(1 for value in values if value is not None and not _finite(value))
        nonfinite_count += int(validation.get("nan_count", 0) or 0)
        if validation.get("selection_score") is not None and not _finite(validation.get("selection_score")):
            nonfinite_selection_score_count += 1
    return {
        "artifact_directory": str(artifact_dir.resolve()),
        "training_jsonl": str(log_path.resolve()) if log_path.exists() else None,
        "training_jsonl_exists": log_path.exists(),
        "training_jsonl_epoch_rows": len(rows),
        "metrics_history_csv": str((artifact_dir / "metrics_history.csv").resolve()),
        "metrics_history_csv_exists_at_closeout_start": (artifact_dir / "metrics_history.csv").exists(),
        "metrics_source_for_closeout": "recovery_attempt_01/logs/training.jsonl",
        "training_traceback_after_recovery": "Traceback (most recent call last)" in stderr_text,
        "training_stderr_tail": stderr_text[-2000:] if stderr_text else "",
        "nonfinite_loss_count": int(nonfinite_count),
        "nonfinite_selection_score_count": int(nonfinite_selection_score_count),
        "state_status": state.get("status"),
        "state_completed_epochs": state.get("completed_epochs"),
        "state_best_epoch": state.get("best_epoch"),
        "state_global_step": state.get("global_step"),
        "state_best_selection_score": state.get("best_selection_score"),
        "progress_json": str(progress_path.resolve()) if progress_path.exists() else None,
        "progress_json_exists": progress_path.exists(),
        "progress_payload": progress_payload,
        "created_at": _now(),
    }


def _validation_gate(metrics: Mapping[str, Any], checkpoint_audit: Mapping[str, Any], consistent: bool) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not checkpoint_audit.get("inference_checkpoint_loadable"):
        reasons.append("inference_checkpoint_not_loadable")
    required = ("non_background_miou", "non_background_macro_f1", "boundary_f1", "uncertainty_ece")
    if not all(_finite(metrics.get(key)) for key in required):
        reasons.append("nan_or_nonfinite_validation_metric")
    if int(metrics.get("nan_count", 0)) != 0 or int(metrics.get("probability_nonfinite_count", 0)) != 0 or int(metrics.get("boundary_nonfinite_count", 0)) != 0 or int(metrics.get("uncertainty_nonfinite_count", 0)) != 0:
        reasons.append("nonfinite_loss_or_output_count_nonzero")
    if float(metrics.get("all_background_prediction_fraction", 1.0)) >= 1.0:
        reasons.append("all_background_prediction")
    if float(metrics.get("all_class_prediction_fraction", 0.0)) >= 1.0:
        reasons.append("all_non_background_class_prediction")
    if int(metrics.get("non_background_predicted_pixel_count", 0)) <= 0:
        reasons.append("zero_non_background_prediction_pixels")
    predicted_class_ids = {int(item.get("class_id")) for item in metrics.get("class_metrics", []) if int(item.get("tp", 0)) + int(item.get("fp", 0)) > 0}
    if not all(class_id in predicted_class_ids for class_id in (1, 2, 3)):
        reasons.append("non_background_class_collapse")
    if not consistent:
        reasons.append("validation_reinference_mismatch")
    return ("VALIDATION_GATE_PASSED_LIMITED" if not reasons else "VALIDATION_GATE_FAILED"), reasons


def _class_rows(metrics: Mapping[str, Any]) -> list[list[Any]]:
    return [[item.get(key) for key in ("class_id", "class_name", "tp", "fp", "fn", "iou", "f1", "precision", "recall")] for item in metrics.get("class_metrics", [])]


def _write_metric_artifacts(output_dir: Path, prefix: str, evaluation: Mapping[str, Any], *, fixed_score: float | None = None) -> None:
    metrics = dict(evaluation["metrics"])
    if fixed_score is not None:
        metrics["fixed_closeout_selection_score"] = fixed_score
    metrics_path = output_dir / f"{prefix}_METRICS.json"
    class_path = output_dir / f"{prefix}_CLASS_METRICS.csv"
    confusion_path = output_dir / f"{prefix}_CONFUSION_MATRIX.csv"
    dump_json(metrics_path, {"status": "success", "metrics": metrics, "sample_records": evaluation.get("sample_records", []), "generated_at": _now()})
    _write_csv(class_path, ("class_id", "class_name", "tp", "fp", "fn", "iou", "f1", "precision", "recall"), _class_rows(metrics))
    matrix = metrics.get("confusion_matrix", [])
    _write_csv(confusion_path, ["target\\prediction", *CLASS_NAMES], [[CLASS_NAMES[index], *row] for index, row in enumerate(matrix)])


def _make_contact_sheet(path: Path, visuals: list[Mapping[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    if not visuals:
        Image.new("RGB", (512, 128), "#eeeeee").save(path)
        return
    cell_w, cell_h = 512, 128
    cols = 1
    rows = len(visuals)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")
    palette = np.asarray([[245, 245, 245], [232, 170, 80], [150, 215, 120], [105, 145, 230]], dtype=np.uint8)
    for row_index, visual in enumerate(visuals):
        image = np.asarray(visual["image"], dtype=np.float32)
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        target = np.asarray(visual["target"], dtype=np.int64)
        prediction = np.asarray(visual["prediction"], dtype=np.int64)
        uncertainty = np.clip(np.asarray(visual["uncertainty"], dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        rgb = Image.fromarray(image, mode="RGB").resize((128, 128))
        target_image = Image.fromarray(palette[np.clip(target, 0, 3)], mode="RGB").resize((128, 128))
        prediction_image = Image.fromarray(palette[np.clip(prediction, 0, 3)], mode="RGB").resize((128, 128))
        uncertainty_image = Image.fromarray(np.repeat(uncertainty[..., None], 3, axis=2), mode="RGB").resize((128, 128))
        for column, tile in enumerate((rgb, target_image, prediction_image, uncertainty_image)):
            sheet.paste(tile, (column * 128, row_index * cell_h))
        draw = ImageDraw.Draw(sheet)
        draw.text((3, row_index * cell_h + 3), str(visual.get("sample_id", "")), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _write_review(output_dir: Path, prefix: str, evaluation: Mapping[str, Any], gate_status: str, reasons: list[str]) -> None:
    review_dir = output_dir / f"{prefix}_REVIEW"
    review_dir.mkdir(parents=True, exist_ok=True)
    _make_contact_sheet(review_dir / "contact_sheet.png", list(evaluation.get("visuals", [])))
    rows = []
    for item in evaluation.get("sample_records", []):
        rows.append(
            "<tr>" + "".join(f"<td>{item.get(key, '')}</td>" for key in ("sample_id", "scene_id", "valid_pixel_count", "predicted_non_background_fraction", "mean_confidence", "mean_uncertainty", "all_background_prediction")) + "</tr>"
        )
    html = f"""<!doctype html>
<meta charset="utf-8"><title>UCS386 {prefix} review</title>
<h1>UCS386 {prefix} review</h1>
<p>Status: <code>{gate_status}</code></p>
<p>Gate reasons: <code>{', '.join(reasons) or 'none'}</code></p>
<p>Contact sheet shows RGB, target, prediction, and uncertainty for the first reviewed samples.</p>
<img src="contact_sheet.png" style="max-width:100%;image-rendering:auto">
<table border="1" cellspacing="0" cellpadding="4"><thead><tr><th>sample_id</th><th>scene_id</th><th>valid_pixels</th><th>pred_non_bg_fraction</th><th>mean_confidence</th><th>mean_uncertainty</th><th>all_background</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
"""
    atomic_write_text(review_dir / "index.html", html)


def _validation_numerical_audit(run_dir: Path, evaluation: Mapping[str, Any], checkpoint_audit: Mapping[str, Any], gate_status: str, reasons: list[str], *, fixed_score: float, validation_reads: int, consistent: bool) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    return {
        "status": gate_status,
        "gate_reasons": reasons,
        "checkpoint_loadable": checkpoint_audit.get("inference_checkpoint_loadable", False),
        "class_order": list(CLASS_NAMES),
        "rgb_mapping": list(RGB_MAPPING),
        "validation_sample_count": metrics.get("sample_count"),
        "validation_pixel_reads": validation_reads,
        "expected_validation_pixel_reads": 44,
        "fixed_closeout_formula": "0.45*non_background_mIoU + 0.25*non_background_macro_F1 + 0.20*boundary_F1 + 0.10*(1-uncertainty_ECE)",
        "fixed_closeout_weights": FIXED_WEIGHTS,
        "fixed_closeout_selection_score": fixed_score,
        "output_probabilities_finite": all(_finite(metrics.get(key)) for key in ("non_background_miou", "non_background_macro_f1", "boundary_f1", "uncertainty_ece")) and int(metrics.get("nan_count", 0)) == 0 and int(metrics.get("probability_nonfinite_count", 0)) == 0 and int(metrics.get("boundary_nonfinite_count", 0)) == 0 and int(metrics.get("uncertainty_nonfinite_count", 0)) == 0,
        "empty_output_count": int(round(float(metrics.get("all_background_prediction_fraction", 0.0)) * int(metrics.get("sample_count", 0)))),
        "full_non_background_output_count": int(round(float(metrics.get("all_class_prediction_fraction", 0.0)) * int(metrics.get("sample_count", 0)))),
        "class_collapse": bool(float(metrics.get("all_background_prediction_fraction", 1.0)) >= 1.0 or int(metrics.get("non_background_predicted_pixel_count", 0)) == 0 or not all(int(item.get("class_id")) in {int(other.get("class_id")) for other in metrics.get("class_metrics", []) if int(other.get("tp", 0)) + int(other.get("fp", 0)) > 0} for item in ({"class_id": 1}, {"class_id": 2}, {"class_id": 3}))),
        "nan_inf_count": int(metrics.get("nan_count", 0)) + int(metrics.get("probability_nonfinite_count", 0)) + int(metrics.get("boundary_nonfinite_count", 0)) + int(metrics.get("uncertainty_nonfinite_count", 0)),
        "validation_reinference_consistent_with_training_log": bool(consistent),
        "locked_test_pixel_reads": 0,
        "other_dataset_test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0},
        "generated_at": _now(),
    }


def _write_selection_report(path: Path, config: Mapping[str, Any], training: Mapping[str, Any], validation_metrics: Mapping[str, Any], checkpoint_audit: Mapping[str, Any], gate_status: str, reasons: list[str], checkpoint_hash: str) -> None:
    fixed_score = validation_metrics.get("fixed_closeout_selection_score")
    text = f"""# {MODEL_NAME} — Stage A selection report

## Decision

- Validation gate: **{gate_status}**
- Candidate checkpoint: `{checkpoint_hash}`
- Selection boundary: fixed UCS386 Validation (44 samples); Locked Test was not used for selection.
- Fixed closeout score: `{fixed_score}`
- Training-recorded score (historical worker formula): `{training.get('state_best_selection_score')}`
- Gate reasons: `{', '.join(reasons) or 'none'}`

## Scope limits

This is a Stage A UCS task candidate trained on the available UCS386 subset: Train 296, Validation 44, and Locked Test 46. The remaining **214 UCS images are missing**. Coverage is **386/600**; **missingness bias not ruled out**. This is a **candidate not ground truth** artifact. **National generalization not verified**. It is not a production or national release.

## Metrics

| Metric | Value |
|---|---:|
| non-background mIoU | {validation_metrics.get('non_background_miou')} |
| non-background macro F1 | {validation_metrics.get('non_background_macro_f1')} |
| boundary F1 | {validation_metrics.get('boundary_f1')} |
| uncertainty ECE | {validation_metrics.get('uncertainty_ece')} |
| empty/full output counts | {validation_metrics.get('all_background_prediction_fraction')} / {validation_metrics.get('all_class_prediction_fraction')} |
| NaN count | {validation_metrics.get('nan_count')} |

## Checkpoint distinction

`training_resume_checkpoint_complete` for the historical interrupted `last.pth` is `{checkpoint_audit.get('training_resume_checkpoint_complete')}`. The recovery best checkpoint audited here is inference-loadable with a strict model-state load: `{checkpoint_audit.get('inference_checkpoint_loadable')}`. The old failed checkpoint was not used for selection, test, or packaging.
"""
    atomic_write_text(path, text)


def _frozen_files(run_dir: Path, config: Mapping[str, Any], checkpoint: Path, checkpoint_hash: str, validation_metrics: Mapping[str, Any], split_hashes: Mapping[str, str]) -> dict[str, Path]:
    ontology = {
        "class_names": list(CLASS_NAMES),
        "valid_label_values": [0, 1, 2, 3],
        "ignore_index": 255,
        "label_encoding": "indexed_palette_png",
    }
    preprocessing = {
        "image_size": int(config["dataset"].get("image_size", 256)),
        "rgb_mapping": list(RGB_MAPPING),
        "rgb_semantic_order": ["R", "G", "B"],
        "normalization": "per_image_percentile_2_98",
        "input_channels": 3,
        "texture_channels": 1,
        "input_mode": "rgb_plus_external_gray",
    }
    thresholds = {
        "frozen": True,
        "class_probability_thresholds": {name: 0.5 for name in CLASS_NAMES},
        "boundary_probability_threshold": 0.5,
        "uncertainty_review_threshold": 0.5,
        "selection_used_for_threshold_tuning": False,
    }
    frozen_config = dict(config)
    frozen_config["selected_checkpoint"] = str(checkpoint.resolve())
    frozen_config["selected_checkpoint_sha256"] = checkpoint_hash
    frozen_config["selected_stage"] = "A_UCS"
    frozen_config["stage_b_used"] = False
    frozen_config["selection_used_test"] = False
    frozen_config["fixed_closeout_selection_score"] = validation_metrics.get("fixed_closeout_selection_score")
    paths = {
        "config": run_dir / "FROZEN_CONFIG.yaml",
        "ontology": run_dir / "FROZEN_ONTOLOGY.json",
        "preprocessing": run_dir / "FROZEN_PREPROCESSING.json",
        "rgb_mapping": run_dir / "FROZEN_RGB_MAPPING.json",
        "thresholds": run_dir / "FROZEN_THRESHOLDS.json",
    }
    atomic_write_text(paths["config"], yaml.safe_dump(frozen_config, allow_unicode=True, sort_keys=False))
    dump_json(paths["ontology"], ontology)
    dump_json(paths["preprocessing"], preprocessing)
    dump_json(paths["rgb_mapping"], {"one_based_bands": list(RGB_MAPPING), "semantic_order": "R,G,B", "verified": True, "frozen": True})
    dump_json(paths["thresholds"], thresholds)
    return paths


def _package_files(project_root: Path, package_dir: Path, frozen_paths: Mapping[str, Path], validation_path: Path, test_path: Path, checkpoint: Path, config: Mapping[str, Any], checkpoint_hash: str, freeze_receipt: Mapping[str, Any]) -> dict[str, Path]:
    package_dir.mkdir(parents=True, exist_ok=False)
    files = {
        "model": package_dir / "model.pth",
        "config": package_dir / "config.yaml",
        "ontology": package_dir / "ontology.json",
        "preprocessing": package_dir / "preprocessing.json",
        "rgb_mapping": package_dir / "rgb_mapping.json",
        "thresholds": package_dir / "thresholds.json",
        "validation": package_dir / "validation_metrics.json",
        "test": package_dir / "test_metrics.json",
    }
    shutil.copyfile(checkpoint, files["model"])
    shutil.copyfile(frozen_paths["config"], files["config"])
    shutil.copyfile(frozen_paths["ontology"], files["ontology"])
    shutil.copyfile(frozen_paths["preprocessing"], files["preprocessing"])
    shutil.copyfile(frozen_paths["rgb_mapping"], files["rgb_mapping"])
    shutil.copyfile(frozen_paths["thresholds"], files["thresholds"])
    shutil.copyfile(validation_path, files["validation"])
    shutil.copyfile(test_path, files["test"])
    card = f"""# {MODEL_NAME}

Research task-model candidate for UCS construction-element segmentation.

- Stage: A_UCS; Stage B used: false.
- Scope: UCS386 available subset, Train296 / Validation44 / LockedTest46.
- Coverage: 386/600; **214 UCS images missing**.
- **Missingness bias not ruled out.**
- **Candidate not ground truth.**
- **National generalization not verified.**
- Production ready: false.
- Checkpoint SHA256: `{checkpoint_hash}`.
- Frozen receipt: `{freeze_receipt.get('receipt_path')}`.

The package is intended for auditable research inference with human review required. It does not establish full-record, national, or production validity.
"""
    atomic_write_text(package_dir / "model_card.md", card)
    requirements = [
        f"python={sys.version.split()[0]}",
        "torch=runtime-installed",
        "numpy=runtime-installed",
        "rasterio=runtime-installed",
        "pyyaml=runtime-installed",
        "Pillow=runtime-installed",
    ]
    atomic_write_text(package_dir / "requirements_snapshot.txt", "\n".join(requirements) + "\n")
    example = f"""$ErrorActionPreference = 'Stop'
$ProjectRoot = '{project_root}'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot 'scripts\\run_construction_task_model.ps1') -PackageDir '{package_dir}' -Manifest '<NON_TEST_MANIFEST.csv>' -OutputDir '<OUTPUT_DIR>' -Device cuda
"""
    atomic_write_text(package_dir / "inference_example.ps1", example)
    return files


def _write_package_manifest(package_dir: Path, files: Mapping[str, Path], freeze_receipt: Mapping[str, Any], test_receipt: Mapping[str, Any], validation_metrics: Mapping[str, Any], test_metrics: Mapping[str, Any]) -> Path:
    manifest = {
        "status": "research_task_model_candidate_packaged",
        "model_name": MODEL_NAME,
        "release_type": "research_task_model_candidate",
        "dataset_scope": "UCS_386_available_subset",
        "training_stage": "A_UCS",
        "stage_b_used": False,
        "production_ready": False,
        "candidate_not_ground_truth": True,
        "full_ucs_training": False,
        "full_record_coverage": "386/600",
        "missing_ucs_images": 214,
        "missingness_bias_not_ruled_out": True,
        "national_generalization_verified": False,
        "class_order": list(CLASS_NAMES),
        "rgb_mapping": list(RGB_MAPPING),
        "human_review_required": True,
        "candidate_frozen": True,
        "selection_used_test": False,
        "test_evaluation_runs": 1,
        "ucs_locked_test_sample_reads": 46,
        "test_receipt": test_receipt.get("receipt_path"),
        "validation_gate": validation_metrics.get("status"),
        "test_status": test_receipt.get("status"),
        "files": {name: path.name for name, path in files.items()},
        "sha256sums_excludes": ["sha256sums.txt", "TASK_MODEL_MANIFEST.json"],
        "freeze_receipt": freeze_receipt.get("receipt_path"),
        "reload_validation": {"status": "pending"},
        "created_at": _now(),
    }
    path = package_dir / "TASK_MODEL_MANIFEST.json"
    dump_json(path, manifest)
    return path


def _write_sha256sums(package_dir: Path) -> Path:
    excluded = {"sha256sums.txt", "TASK_MODEL_MANIFEST.json"}
    lines = []
    for path in sorted(package_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in excluded:
            lines.append(f"{_sha256(path)}  {path.name}")
    output = package_dir / "sha256sums.txt"
    atomic_write_text(output, "\n".join(lines) + "\n")
    return output


def _run_independent_reload(project_root: Path, package_dir: Path, validation_manifest: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(validation_manifest.open(encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError("validation manifest is empty for independent reload")
    sample_id = rows[0].get("sample_id")
    output_dir = package_dir / "VALIDATION_RELOAD"
    command = [
        sys.executable,
        str(project_root / "tools" / "urrs.py"),
        "construction",
        "task-model",
        "infer",
        "--package-dir",
        str(package_dir),
        "--manifest",
        str(validation_manifest),
        "--output-dir",
        str(output_dir),
        "--sample-id",
        str(sample_id),
        "--device",
        "cuda",
    ]
    completed = subprocess.run(command, cwd=str(project_root), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"independent task-model reload failed: stdout={completed.stdout[-2000:]} stderr={completed.stderr[-2000:]}")
    output_manifest = output_dir / "CONSTRUCTION_ELEMENT_MANIFEST.json"
    receipt = _json(output_manifest)
    if receipt.get("status") != "success" or not receipt.get("all_outputs_finite"):
        raise RuntimeError(f"independent task-model reload receipt failed: {receipt}")
    return {
        "status": "success",
        "python_executable": sys.executable,
        "command": command,
        "sample_id": sample_id,
        "output_manifest": str(output_manifest.resolve()),
        "model_sha256": receipt.get("package_model_sha256"),
        "output_manifest_payload": receipt,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "test_pixel_reads": 0,
        "created_at": _now(),
    }


def _test_report(path: Path, test_receipt: Mapping[str, Any], metrics: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    text = f"""# UCS386 Locked Test report

- Status: **{test_receipt.get('status')}**
- Evaluation runs: `{test_receipt.get('test_evaluation_runs')}`
- Locked Test samples read: `{test_receipt.get('ucs_locked_test_sample_reads')}` (expected 46)
- Used for selection/tuning: **false**
- Validation gate: `{validation.get('status')}`

| Metric | Value |
|---|---:|
| non-background mIoU | {metrics.get('non_background_miou')} |
| non-background macro F1 | {metrics.get('non_background_macro_f1')} |
| boundary F1 | {metrics.get('boundary_f1')} |
| uncertainty ECE | {metrics.get('uncertainty_ece')} |
| NaN count | {metrics.get('nan_count')} |

Interpretation is limited to the UCS386 available-image subset. **214 UCS images missing**, **missingness bias not ruled out**, **candidate not ground truth**, and **national generalization not verified**. No post-Test tuning or retraining was performed.
"""
    atomic_write_text(path, text)


def finalize_existing_run(project_root: Path, run_dir: Path, config_path: Path) -> dict[str, Any]:
    """Complete the closeout without starting a training process."""

    run_dir = run_dir.resolve()
    existing_receipt = run_dir / "FINALIZE_RECEIPT.json"
    if existing_receipt.exists():
        return _json(existing_receipt)
    state_path = run_dir / "TRAINING_RUN_STATE.json"
    state = _json(state_path if state_path.exists() else run_dir / "TRAINING_STATE.json")
    config = _read_yaml(config_path)
    if state.get("status") != "COMPLETED_STAGE_A":
        return {"status": "FINALIZE_BLOCKED", "reason": "existing Stage A run is not COMPLETED_STAGE_A", "state": state, "test_pixel_reads": 0, "selection_used_test": False}
    if state.get("stage_b", {}).get("enabled") or state.get("stage_b", {}).get("status") not in {"NOT_RUN_OPTIONAL", "NOT_RUN"}:
        return {"status": "FINALIZE_BLOCKED", "reason": "Stage B state is not disabled/not-run", "test_pixel_reads": 0, "selection_used_test": False}
    artifact_dir = Path(state.get("artifact_directory", "")).resolve()
    checkpoint = Path(state.get("best_checkpoint", "")).resolve()
    if not checkpoint.exists() or checkpoint.parent.parent != artifact_dir:
        return {"status": "FINALIZE_BLOCKED", "reason": "recovery best checkpoint is missing or is not under the recorded recovery artifact directory", "test_pixel_reads": 0, "selection_used_test": False}
    if checkpoint.name != "best_validation.pth":
        return {"status": "FINALIZE_BLOCKED", "reason": "only best_validation.pth may be selected", "test_pixel_reads": 0, "selection_used_test": False}

    gate_dir = Path(config["dataset"]["subset_gate_dir"]).resolve()
    gate = _json(gate_dir / "UCS386_SUBSET_GATE.json")
    if gate.get("status") != "FORMAL_SUBSET_TRAINING_ALLOWED" or gate.get("test_pixel_reads") != 0:
        return {"status": "FINALIZE_BLOCKED", "reason": "UCS386 gate is not the previously approved zero-Test-read gate", "test_pixel_reads": 0, "selection_used_test": False}

    import torch
    from torch.utils.data import DataLoader
    from urban_renewal.construction.datasets import UCSManifestDataset
    from construction_training_worker import _loader, _model_and_loss, _require_cuda, evaluate_model

    checkpoint_hash = _sha256(checkpoint)
    checkpoint_audit, model, criterion, payload = _load_checkpoint_audit(run_dir, checkpoint, config, state, gate)
    dump_json(run_dir / "CHECKPOINT_LOAD_AUDIT.json", checkpoint_audit)
    atomic_write_text(run_dir / "CHECKPOINT_SHA256.txt", checkpoint_hash + "  best_validation.pth\n")
    training_trace = _training_trace_audit(run_dir, artifact_dir, state)
    dump_json(run_dir / "TRAINING_CLOSEOUT_TRACE_AUDIT.json", training_trace)
    if not checkpoint_audit["inference_checkpoint_loadable"]:
        failure = {"status": "VALIDATION_GATE_FAILED", "gate_reasons": ["inference_checkpoint_not_loadable"], "checkpoint_audit": checkpoint_audit, "test_pixel_reads": 0, "selection_used_test": False, "created_at": _now()}
        dump_json(run_dir / "VALIDATION_NUMERICAL_AUDIT.json", failure)
        return failure

    device = _require_cuda(str(config.get("training", {}).get("device", "cuda")))
    model.to(device)
    validation_manifest = Path(config["dataset"]["validation_manifest"]).resolve()
    validation_dataset, validation_loader = _loader(validation_manifest, int(config["dataset"].get("image_size", 256)), int(config["training"].get("batch_size", 2)), RGB_MAPPING, shuffle=False, seed=int(config["training"].get("seed", 20260803)) + 1)
    validation_evaluation = evaluate_model(model, validation_loader, criterion, device, amp_enabled=bool(config["training"].get("amp", True)), collect_visuals=True)
    validation_metrics = dict(validation_evaluation["metrics"])
    fixed_score = _fixed_selection_score(validation_metrics)
    validation_metrics["fixed_closeout_selection_score"] = fixed_score
    validation_metrics["training_recorded_best_selection_score"] = state.get("best_selection_score")
    validation_metrics["training_recorded_selection_formula"] = "0.50/0.25/0.15/0.10"
    validation_metrics["fixed_closeout_selection_formula"] = "0.45/0.25/0.20/0.10"
    validation_metrics["validation_pixel_reads"] = validation_dataset.pixel_reads
    validation_metrics["expected_validation_pixel_reads"] = 44
    # Compare the selected epoch's saved Validation values, if available.  This
    # is a consistency diagnostic only; it never changes checkpoint selection.
    training_rows = []
    training_log = artifact_dir / "logs" / "training.jsonl"
    if training_log.exists():
        training_rows = [json.loads(line) for line in training_log.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    selected_row = next((row for row in training_rows if int(row.get("epoch", -1)) == int(state.get("best_epoch", -2))), None)
    consistent = True
    if selected_row:
        old_metrics = selected_row.get("validation", {})
        for key in ("non_background_miou", "non_background_macro_f1", "boundary_f1", "uncertainty_ece"):
            if _finite(old_metrics.get(key)) and abs(float(old_metrics[key]) - float(validation_metrics[key])) > 1e-4:
                consistent = False
    gate_status, gate_reasons = _validation_gate(validation_metrics, checkpoint_audit, consistent)
    validation_metrics["validation_reinference_consistent_with_training_log"] = consistent
    validation_metrics["status"] = gate_status
    validation_evaluation["metrics"] = validation_metrics
    _write_metric_artifacts(run_dir, "VALIDATION", validation_evaluation, fixed_score=fixed_score)
    numerical_audit = _validation_numerical_audit(run_dir, validation_evaluation, checkpoint_audit, gate_status, gate_reasons, fixed_score=fixed_score, validation_reads=validation_dataset.pixel_reads, consistent=consistent)
    dump_json(run_dir / "VALIDATION_NUMERICAL_AUDIT.json", numerical_audit)
    _write_review(run_dir, "VALIDATION", validation_evaluation, gate_status, gate_reasons)
    _write_selection_report(run_dir / "MODEL_SELECTION_REPORT.md", config, training_trace, validation_metrics, checkpoint_audit, gate_status, gate_reasons, checkpoint_hash)
    if gate_status != "VALIDATION_GATE_PASSED_LIMITED":
        return {"status": gate_status, "gate_reasons": gate_reasons, "validation_metrics": validation_metrics, "test_pixel_reads": 0, "selection_used_test": False, "checkpoint_audit": checkpoint_audit, "created_at": _now()}

    split_hashes = _split_hashes(gate_dir, config)
    frozen_paths = _frozen_files(run_dir, config, checkpoint, checkpoint_hash, validation_metrics, split_hashes)
    freeze_receipt = {
        "status": "MODEL_FROZEN",
        "candidate_frozen": True,
        "selected_stage": "A_UCS",
        "stage_b_used": False,
        "candidate_checkpoint": str(checkpoint.resolve()),
        "candidate_checkpoint_sha256": checkpoint_hash,
        "architecture": "ConstructionElementNet",
        "class_order": list(CLASS_NAMES),
        "preprocessing": "per_image_percentile_2_98",
        "rgb_mapping": list(RGB_MAPPING),
        "thresholds_frozen": True,
        "split_hashes": split_hashes,
        "best_epoch": state.get("best_epoch"),
        "training_recorded_selection_score": state.get("best_selection_score"),
        "fixed_closeout_selection_score": fixed_score,
        "selection_used_test": False,
        "test_evaluation_runs": 0,
        "ucs_locked_test_sample_reads": 0,
        "test_pixel_reads": 0,
        "full_ucs_training": False,
        "full_record_coverage": "386/600",
        "missing_ucs_images": 214,
        "missingness_bias_not_ruled_out": True,
        "national_generalization_verified": False,
        "frozen_files": {key: str(path.resolve()) for key, path in frozen_paths.items()},
        "freeze_time": _now(),
    }
    freeze_receipt["receipt_path"] = str((run_dir / "MODEL_FREEZE_RECEIPT.json").resolve())
    dump_json(run_dir / "MODEL_FREEZE_RECEIPT.json", freeze_receipt)

    # The following block is the only code path allowed to instantiate a
    # locked-Test dataset.  It executes once and records the exact read count.
    test_manifest = Path(config["dataset"]["locked_test_manifest"]).resolve()
    test_dataset = UCSManifestDataset(test_manifest, image_size=int(config["dataset"].get("image_size", 256)), rgb_bands=RGB_MAPPING, allow_test=True)
    if len(test_dataset) != 46:
        raise RuntimeError(f"locked Test manifest count changed: {len(test_dataset)} != 46")
    test_loader = DataLoader(test_dataset, batch_size=int(config["training"].get("batch_size", 2)), shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_evaluation = evaluate_model(model, test_loader, criterion, device, amp_enabled=bool(config["training"].get("amp", True)), collect_visuals=False)
    if test_dataset.pixel_reads != 46:
        raise RuntimeError(f"locked Test pixel-read invariant failed: {test_dataset.pixel_reads} != 46")
    test_metrics = dict(test_evaluation["metrics"])
    test_metrics["fixed_closeout_selection_score"] = _fixed_selection_score(test_metrics)
    test_metrics["used_for_selection"] = False
    test_metrics["used_for_tuning"] = False
    test_metrics["validation_gate"] = gate_status
    test_evaluation["metrics"] = test_metrics
    test_result_status = "limited_candidate" if test_metrics.get("selection_valid") else "rejected_candidate"
    test_receipt = {
        "status": test_result_status,
        "test_evaluation_runs": 1,
        "ucs_locked_test_sample_reads": int(test_dataset.pixel_reads),
        "selection_used_test": False,
        "used_for_tuning": False,
        "candidate_frozen_before_test": True,
        "candidate_checkpoint_sha256": checkpoint_hash,
        "frozen_config_sha256": _sha256(frozen_paths["config"]),
        "frozen_ontology_sha256": _sha256(frozen_paths["ontology"]),
        "frozen_preprocessing_sha256": _sha256(frozen_paths["preprocessing"]),
        "frozen_thresholds_sha256": _sha256(frozen_paths["thresholds"]),
        "test_pixel_reads": int(test_dataset.pixel_reads),
        "other_dataset_test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0},
        "test_metrics_path": str((run_dir / "UCS386_FINAL_TEST_METRICS.json").resolve()),
        "receipt_path": str((run_dir / "UCS386_FINAL_TEST_RECEIPT.json").resolve()),
        "created_at": _now(),
    }
    _write_metric_artifacts(run_dir, "UCS386_FINAL_TEST", test_evaluation, fixed_score=test_metrics["fixed_closeout_selection_score"])
    _test_report(run_dir / "UCS386_FINAL_TEST_REPORT.md", test_receipt, test_metrics, validation_metrics)
    dump_json(run_dir / "UCS386_FINAL_TEST_RECEIPT.json", test_receipt)
    freeze_receipt["test_evaluation_runs"] = 1
    freeze_receipt["ucs_locked_test_sample_reads"] = int(test_dataset.pixel_reads)
    freeze_receipt["test_receipt"] = str((run_dir / "UCS386_FINAL_TEST_RECEIPT.json").resolve())
    dump_json(run_dir / "MODEL_FREEZE_RECEIPT.json", freeze_receipt)

    package_root = project_root / "outputs" / "models" / "construction_activity_v1"
    package_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = package_root / f"{MODEL_NAME}_{package_stamp}"
    package_files = _package_files(project_root, package_dir, frozen_paths, run_dir / "VALIDATION_METRICS.json", run_dir / "UCS386_FINAL_TEST_METRICS.json", checkpoint, config, checkpoint_hash, freeze_receipt)
    package_manifest_path = _write_package_manifest(package_dir, package_files, freeze_receipt, test_receipt, validation_metrics, test_metrics)
    _write_sha256sums(package_dir)
    reload_audit = _run_independent_reload(project_root, package_dir, validation_manifest)
    package_manifest = _json(package_manifest_path)
    package_manifest["reload_validation"] = reload_audit
    dump_json(package_manifest_path, package_manifest)
    final_receipt = {
        "status": "COMPLETED_STAGE_A_CLOSEOUT",
        "run_directory": str(run_dir.resolve()),
        "artifact_directory": str(artifact_dir.resolve()),
        "candidate_checkpoint": str(checkpoint.resolve()),
        "candidate_checkpoint_sha256": checkpoint_hash,
        "checkpoint_load_audit": str((run_dir / "CHECKPOINT_LOAD_AUDIT.json").resolve()),
        "validation_gate": gate_status,
        "validation_metrics": str((run_dir / "VALIDATION_METRICS.json").resolve()),
        "model_freeze_receipt": str((run_dir / "MODEL_FREEZE_RECEIPT.json").resolve()),
        "test_receipt": str((run_dir / "UCS386_FINAL_TEST_RECEIPT.json").resolve()),
        "test_metrics": str((run_dir / "UCS386_FINAL_TEST_METRICS.json").resolve()),
        "test_evaluation_runs": 1,
        "ucs_locked_test_sample_reads": 46,
        "selection_used_test": False,
        "stage_b_used": False,
        "package_dir": str(package_dir.resolve()),
        "package_manifest": str(package_manifest_path.resolve()),
        "independent_reload": reload_audit,
        "test_pixel_reads": 46,
        "other_dataset_test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0},
        "full_ucs_training": False,
        "subset_training": True,
        "full_record_coverage": "386/600",
        "missing_ucs_images": 214,
        "missingness_bias_not_ruled_out": True,
        "candidate_not_ground_truth": True,
        "national_generalization_verified": False,
        "training_started_by_finalize": False,
        "resume_started_by_finalize": False,
        "new_training_run_created": False,
        "created_at": _now(),
    }
    dump_json(existing_receipt, final_receipt)
    return final_receipt
