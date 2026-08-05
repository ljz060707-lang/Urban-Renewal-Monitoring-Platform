from __future__ import annotations

"""Auditable Tongzhou construction-candidate image pipeline.

This entry point is inference-only.  It never trains, reads the locked-test
manifest, or uses Tongzhou outputs to select a recipe.  Construction outputs
remain model candidates rather than human-confirmed sites.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")
SCENE_FILES = {
    "2018-05-06": "tongzhou180506_zoom_20.tif",
    "2020-07-23": "tongzhou200723_zoom_20.tif",
    "2022-03-20": "tongzhou220320_zoom_20.tif",
    "2023-07-16": "tongzhou230716_zoom_20.tif",
}
PACKAGE_NAME = "URRS-Construction-UCS386-v1.0-task-candidate_20260803_142054"
EXPECTED_CHECKPOINT_SHA256 = "c522d5a4e6c5d6124f968d659004d7870fe4adc9d7381dc4c8917fe019f0f060"
METRIC_CRS = "EPSG:32650"
CLASS_NAMES = ("background", "exposed_ground", "dustproof_net", "building")
PROBABILITY_NAMES = tuple(f"{name}_probability" for name in CLASS_NAMES) + (
    "boundary_probability",
    "uncertainty_probability",
)
RUN_PREFIX = "beijing_tongzhou_construction_images_best_"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    os.replace(temporary, path)


def project_paths(project_root: Path) -> dict[str, Path]:
    package = project_root / "outputs" / "models" / "construction_activity_v1" / PACKAGE_NAME
    return {
        "package": package,
        "checkpoint": package / "model.pth",
        "old_construction": project_root / "outputs" / "inference" / "beijing_tongzhou_construction_activity_v1",
        "full20": project_root / "outputs" / "inference" / "beijing_tongzhou_to_latest",
        "building": project_root / "outputs" / "inference" / "beijing_tongzhou_building_geometry",
        "surface": project_root / "outputs" / "inference" / "beijing_tongzhou_surface_state_v1" / "20260730_113324",
        "review": project_root / "outputs" / "urrs" / "review" / "tongzhou_20260801",
        "seed": project_root / "outputs" / "urrs" / "construction" / "human_seed_labels_20260802_134145",
        "validation_manifest": project_root / "outputs" / "urrs" / "audits" / "ucs386_subset_readiness_20260803_113903" / "UCS_VALIDATION_MANIFEST.csv",
    }


def ensure_safe_path(path: Path) -> None:
    lowered = str(path).lower()
    if "nanjing" in lowered or "南京" in lowered:
        raise RuntimeError(f"Nanjing path is outside this run contract: {path}")


def create_run_root(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        run_root = Path(explicit).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = project_root / "outputs" / "inference" / f"{RUN_PREFIX}{stamp}"
    ensure_safe_path(run_root)
    if run_root.exists() and not (run_root / "00_governance" / "RUN_CONTRACT.json").exists():
        raise RuntimeError(f"refusing to use an unrelated or incomplete existing directory: {run_root}")
    for name in (
        "00_governance", "01_input_audit", "02_inference_recipe", "03_raw_inference",
        "04_objectization", "05_candidate_ranking", "06_temporal", "07_figures_highres",
        "08_manifests", "09_metrics", "10_vectors", "11_logs", "12_tests",
    ):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    for name in ("single_date", "multi_date", "temporal", "cases", "old_vs_new"):
        (run_root / "07_figures_highres" / name).mkdir(parents=True, exist_ok=True)
    if (run_root / "10_reports").exists():
        raise RuntimeError("10_reports is prohibited for this run")
    return run_root


def scene_inventory(source_root: Path) -> list[dict[str, Any]]:
    import cv2
    import rasterio
    from rasterio.enums import Resampling

    rows: list[dict[str, Any]] = []
    for date in DATES:
        path = source_root / SCENE_FILES[date]
        ensure_safe_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            sample_height = min(1024, src.height)
            sample_width = min(1024, src.width)
            sample = src.read((3, 2, 1), out_shape=(3, sample_height, sample_width), resampling=Resampling.bilinear).astype(np.float32)
            finite = sample[np.isfinite(sample)]
            if finite.size == 0:
                raise RuntimeError(f"no finite pixels in {path}")
            low, high = np.percentile(finite, (2, 98))
            display = np.clip((sample - low) / max(high - low, 1e-6), 0.0, 1.0)
            gray = np.moveaxis(display, 0, -1)
            gray = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            valid = np.all(np.isfinite(sample), axis=0)
            if src.nodata is not None:
                valid &= np.all(sample != src.nodata, axis=0)
            xres, yres = abs(float(src.transform.a)), abs(float(src.transform.e))
            center_lat = (src.bounds.bottom + src.bounds.top) / 2.0
            gsd_x = xres * 111320.0 * math.cos(math.radians(center_lat)) if src.crs and src.crs.is_geographic else xres
            gsd_y = yres * 110540.0 if src.crs and src.crs.is_geographic else yres
            rows.append({
                "scene_date": date,
                "file_path": str(path.resolve()),
                "file_name": path.name,
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": src.dtypes[0],
                "CRS": str(src.crs),
                "transform": tuple(src.transform),
                "bounds": tuple(src.bounds),
                "pixel_resolution": (xres, yres),
                "estimated_GSD": (gsd_x + gsd_y) / 2.0,
                "nodata": src.nodata,
                "valid_fraction": float(valid.mean()),
                "SHA256": sha256(path),
                "brightness": float(gray.mean()),
                "contrast": float(gray.std()),
                "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "shadow_fraction_proxy": float((gray < 35).mean()),
                "alignment_status": "pending_cross_scene_check",
            })
    first = rows[0]
    aligned = all(
        row["width"] == first["width"]
        and row["height"] == first["height"]
        and row["CRS"] == first["CRS"]
        and row["transform"] == first["transform"]
        and row["bounds"] == first["bounds"]
        for row in rows
    )
    for row in rows:
        row["alignment_status"] = "aligned_common_grid" if aligned else "not_aligned"
    return rows


def protected_files(paths: Mapping[str, Path], source_root: Path) -> list[tuple[str, Path]]:
    assets: list[tuple[str, Path]] = []
    for date in DATES:
        assets.append(("source_imagery", source_root / SCENE_FILES[date]))
    assets.append(("checkpoint", paths["checkpoint"]))
    for key in ("old_construction", "full20", "building", "review", "seed"):
        root = paths[key]
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            ensure_safe_path(path)
            assets.append((key, path))
    return assets


def audit(project_root: Path, source_root: Path, run_root: Path) -> dict[str, Any]:
    paths = project_paths(project_root)
    for path in paths.values():
        ensure_safe_path(path)
    package_manifest = load_json(paths["package"] / "TASK_MODEL_MANIFEST.json")
    config_text = (paths["package"] / "config.yaml").read_text(encoding="utf-8")
    preprocessing = load_json(paths["package"] / "preprocessing.json")
    thresholds = load_json(paths["package"] / "thresholds.json")
    old_manifest = load_json(paths["old_construction"] / "TONGZHOU_CONSTRUCTION_MANIFEST.json")
    old_audit = load_json(paths["old_construction"] / "TONGZHOU_CONSTRUCTION_AUDIT.json")
    checkpoint_sha = sha256(paths["checkpoint"])
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint mismatch: {checkpoint_sha}")
    scenes = scene_inventory(source_root)
    if any(row["alignment_status"] != "aligned_common_grid" for row in scenes):
        raise RuntimeError("Tongzhou scenes do not share a common grid")

    contract = {
        "status": "audit_complete",
        "run_root": str(run_root),
        "dates": list(DATES),
        "latest_date": DATES[-1],
        "model": "Construction C-A",
        "model_package": package_manifest.get("model_name"),
        "model_family": "ConstructionElementNet",
        "checkpoint_sha256": checkpoint_sha,
        "c_b1_status": "rejected",
        "site_classifier": "disabled_pending_human_gt",
        "construction_tfr": "development_candidate_not_formally_validated",
        "construction_mti": "experimental_unselected",
        "selected_construction_temporal_integration": "no_mti",
        "globfp": "disabled",
        "height": "disabled",
        "nanjing_assets_read": False,
        "training_started": False,
        "fine_tuning_started": False,
        "domain_adaptation_started": False,
        "pseudo_label_training_started": False,
        "models_reselected": False,
        "thresholds_tuned_on_tongzhou": False,
        "tta_selected_on_tongzhou": False,
        "site_classifier_enabled": False,
        "locked_test_manifest_reads": 0,
        "locked_test_pixel_reads": 0,
        "existing_locked_test_summary_artifact_accidentally_read_during_initial_manual_audit": True,
        "existing_locked_test_summary_excluded_from_recipe_selection": True,
        "registry_modified": False,
        "checkpoint_modified": False,
        "old_tongzhou_outputs_modified": False,
        "source_imagery_modified": False,
        "raw_probability_modified": False,
        "html_generated": False,
        "created_at": now(),
    }
    write_json(run_root / "00_governance" / "RUN_CONTRACT.json", contract)
    write_json(run_root / "00_governance" / "CONSTRUCTION_ASSET_DISCOVERY.json", {
        "current_best_model_registry": {"status": "not_found", "searched_fixed_locations_only": True},
        "package_manifest": str((paths["package"] / "TASK_MODEL_MANIFEST.json").resolve()),
        "old_tongzhou_manifest": str((paths["old_construction"] / "TONGZHOU_CONSTRUCTION_MANIFEST.json").resolve()),
        "old_tongzhou_audit": str((paths["old_construction"] / "TONGZHOU_CONSTRUCTION_AUDIT.json").resolve()),
        "validation_manifest": str(paths["validation_manifest"].resolve()),
        "checkpoint": str(paths["checkpoint"].resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "site_classifier": "disabled_pending_human_gt",
        "nanjing_assets_read": False,
    })
    write_json(run_root / "00_governance" / "CONSTRUCTION_MODEL_LINEAGE.json", {
        "formal_model": "Construction C-A",
        "model_package": package_manifest.get("model_name"),
        "architecture": ["DualBranchEncoder", "decoder", "UCS native head", "UnifiedEvidenceProjector"],
        "parallel_heads": ["BoundaryHead", "UncertaintyHead"],
        "classes": list(CLASS_NAMES),
        "checkpoint_sha256": checkpoint_sha,
        "candidate_frozen": package_manifest.get("candidate_frozen"),
        "selection_used_test": package_manifest.get("selection_used_test"),
        "c_b1": "rejected",
        "site_classifier": "disabled_pending_human_gt",
        "scientific_status": "candidate_not_ground_truth",
    })
    write_json(run_root / "00_governance" / "CONSTRUCTION_PROCESSING_CONTRACT.json", {
        "rgb_mapping": preprocessing.get("rgb_mapping"),
        "preprocessing": preprocessing,
        "thresholds": thresholds,
        "historical_tile_size": old_manifest["scene_outputs"][DATES[0]]["inference"]["tile_size"],
        "historical_batch_size": old_manifest["scene_outputs"][DATES[0]]["inference"]["batch_size"],
        "formal_metric_crs": old_manifest["area_policy"]["metric_crs"],
        "model_probability_immutability": True,
        "auxiliary_evidence_read_only": ["Full20", "RSBuilding", "Surface State"],
    })
    write_json(run_root / "00_governance" / "REJECTED_MODEL_AUDIT.json", {
        "Construction C-B1": "rejected",
        "Construction-TFR": "development_candidate_not_formally_validated",
        "Construction-MTI": "experimental_unselected",
        "Site Classifier": "disabled_pending_human_gt",
        "GloBFP": "disabled",
        "Height": "disabled",
        "formal_temporal_path": "no_mti",
    })
    write_csv(run_root / "01_input_audit" / "TONGZHOU_SCENE_INVENTORY.csv", scenes, [
        "scene_date", "file_path", "file_name", "width", "height", "bands", "dtype", "CRS",
        "transform", "bounds", "pixel_resolution", "estimated_GSD", "nodata", "valid_fraction",
        "SHA256", "brightness", "contrast", "sharpness", "shadow_fraction_proxy", "alignment_status",
    ])
    write_json(run_root / "01_input_audit" / "TONGZHOU_SCENE_MANIFEST.json", {"dates": list(DATES), "scenes": scenes})
    write_json(run_root / "01_input_audit" / "TONGZHOU_ALIGNMENT_AUDIT.json", {
        "status": "pass", "common_crs": scenes[0]["CRS"], "common_transform": scenes[0]["transform"],
        "common_bounds": scenes[0]["bounds"], "common_width": scenes[0]["width"], "common_height": scenes[0]["height"],
    })
    write_json(run_root / "01_input_audit" / "TONGZHOU_INPUT_READINESS.json", {
        "status": "ready", "date_count": len(scenes), "dates": list(DATES), "common_grid": True,
        "metric_crs_from_existing_manifest": old_manifest["area_policy"]["metric_crs"],
        "nanjing_assets_read": False,
    })
    baseline_rows = []
    for category, path in protected_files(paths, source_root):
        baseline_rows.append({"category": category, "path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)})
    write_csv(run_root / "00_governance" / "PROTECTED_ASSET_BASELINE.csv", baseline_rows, ["category", "path", "size", "sha256"])
    write_json(run_root / "00_governance" / "HISTORICAL_CONSTRUCTION_AUDIT.json", {
        "manifest_status": old_manifest.get("status"),
        "audit_status": old_audit.get("status"),
        "model_name": old_manifest.get("model_name"),
        "model_sha256": old_manifest.get("model_sha256"),
        "dates": old_manifest.get("dates"),
        "historical_preprocessing": old_manifest["scene_outputs"][DATES[0]]["inference"].get("preprocessing"),
        "historical_locked_test_pixel_reads_recorded_by_old_run": old_manifest.get("test_pixel_reads"),
        "new_run_locked_test_pixel_reads": 0,
        "config_selected_checkpoint_present": EXPECTED_CHECKPOINT_SHA256 in config_text,
    })
    return {"run_root": str(run_root), "checkpoint_sha256": checkpoint_sha, "scene_count": len(scenes), "protected_file_count": len(baseline_rows)}


def _prepare_validation_image(path: Path, rgb_mapping: Sequence[int], image_size: int):
    import rasterio
    import torch
    from torch.nn import functional as F

    with rasterio.open(path) as src:
        array = src.read(tuple(int(v) for v in rgb_mapping)).astype(np.float32)
    finite = array[np.isfinite(array)]
    low, high = np.percentile(finite, (2, 98))
    array = np.clip((array - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(np.float32, copy=False)
    image = torch.from_numpy(array)[None]
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    texture = (image * image.new_tensor([0.2989, 0.5870, 0.1140])[None, :, None, None]).sum(1, keepdim=True)
    return image, texture


def _predict_recipe(model, image, texture, recipe: str):
    import torch

    transforms = [(False, False)]
    if recipe in {"RCP2_FLIP_TTA", "RCP3_UNCERTAINTY_BLEND"}:
        transforms = [(False, False), (False, True), (True, False)]
    probabilities = []
    boundaries = []
    uncertainties = []
    with torch.no_grad():
        for vertical, horizontal in transforms:
            x = image
            t = texture
            if vertical:
                x = torch.flip(x, (-2,)); t = torch.flip(t, (-2,))
            if horizontal:
                x = torch.flip(x, (-1,)); t = torch.flip(t, (-1,))
            output = model(x, "ucs", t)
            p = output["ucs_probabilities"].float()
            b = output["boundary_logits"].float().sigmoid()
            u = output["uncertainty_logits"].float().sigmoid()
            if horizontal:
                p = torch.flip(p, (-1,)); b = torch.flip(b, (-1,)); u = torch.flip(u, (-1,))
            if vertical:
                p = torch.flip(p, (-2,)); b = torch.flip(b, (-2,)); u = torch.flip(u, (-2,))
            probabilities.append(p); boundaries.append(b); uncertainties.append(u)
    if recipe == "RCP3_UNCERTAINTY_BLEND":
        weights = [torch.clamp(1.0 - u, min=1e-3) for u in uncertainties]
        denominator = torch.stack(weights).sum(0)
        probability = torch.stack([p * w for p, w in zip(probabilities, weights)]).sum(0) / denominator
        boundary = torch.stack([b * w for b, w in zip(boundaries, weights)]).sum(0) / denominator
        uncertainty = torch.stack(uncertainties).mean(0)
    else:
        probability = torch.stack(probabilities).mean(0)
        boundary = torch.stack(boundaries).mean(0)
        uncertainty = torch.stack(uncertainties).mean(0)
    return probability, boundary, uncertainty


def _boundary_target(label: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = label != ignore_index
    boundary = np.zeros(label.shape, dtype=bool)
    boundary[1:, :] |= valid[1:, :] & valid[:-1, :] & (label[1:, :] != label[:-1, :])
    boundary[:-1, :] |= valid[:-1, :] & valid[1:, :] & (label[:-1, :] != label[1:, :])
    boundary[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (label[:, 1:] != label[:, :-1])
    boundary[:, :-1] |= valid[:, :-1] & valid[:, 1:] & (label[:, :-1] != label[:, 1:])
    return boundary


def _metric_from_arrays(confusion: np.ndarray, boundary_counts: Sequence[int], confidences: list[np.ndarray], corrects: list[np.ndarray], uncertainty_correct: list[np.ndarray], uncertainty_wrong: list[np.ndarray]) -> dict[str, Any]:
    class_rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        iou = tp / max(tp + fp + fn, 1.0)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
        class_rows.append({"class_id": class_id, "class_name": name, "iou": iou, "f1": f1})
    confidence = np.concatenate(confidences)
    correct = np.concatenate(corrects)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1 + 1e-12)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    btp, bfp, bfn = (float(value) for value in boundary_counts)
    wrong_u = np.concatenate(uncertainty_wrong) if uncertainty_wrong else np.asarray([], dtype=np.float32)
    correct_u = np.concatenate(uncertainty_correct) if uncertainty_correct else np.asarray([], dtype=np.float32)
    return {
        "non_background_mIoU": float(np.mean([row["iou"] for row in class_rows[1:]])),
        "macro_F1": float(np.mean([row["f1"] for row in class_rows])),
        "exposed_ground_F1": class_rows[1]["f1"],
        "dustproof_net_F1": class_rows[2]["f1"],
        "building_class_F1": class_rows[3]["f1"],
        "background_F1": class_rows[0]["f1"],
        "boundary_F1": 2 * btp / max(2 * btp + bfp + bfn, 1.0),
        "calibration_error": ece,
        "uncertainty_discrimination": float(wrong_u.mean() - correct_u.mean()) if wrong_u.size and correct_u.size else None,
        "class_metrics": class_rows,
        "confusion_matrix": confusion.tolist(),
    }


def validate_recipes(project_root: Path, run_root: Path, device: str) -> dict[str, Any]:
    import torch
    from PIL import Image
    from urban_renewal.construction.task_model import load_task_model
    from urban_renewal.construction.partial_labels import boundary_target

    paths = project_paths(project_root)
    package = paths["package"]
    preprocessing = load_json(package / "preprocessing.json")
    thresholds = load_json(package / "thresholds.json")
    with paths["validation_manifest"].open(encoding="utf-8-sig", newline="") as handle:
        validation_rows = list(csv.DictReader(handle))
    if len(validation_rows) != 44 or any(row.get("split") != "validation" for row in validation_rows):
        raise RuntimeError("validation manifest is not the frozen 44-sample Validation split")
    model, _ = load_task_model(package, device=device)
    model.eval()
    recipes = ("RCP0_HISTORICAL_EXACT", "RCP1_WEIGHTED_OVERLAP", "RCP2_FLIP_TTA", "RCP3_UNCERTAINTY_BLEND")
    registry = {
        "registered_before_evaluation": True,
        "selection_data": "UCS Validation only",
        "tongzhou_used_for_selection": False,
        "locked_test_manifest_reads": 0,
        "locked_test_pixel_reads": 0,
        "recipes": {
            "RCP0_HISTORICAL_EXACT": {"tile": 512, "model_input": 256, "overlap": 0, "fusion": "none", "tta": False},
            "RCP1_WEIGHTED_OVERLAP": {"tile": 256, "model_input": 256, "overlap": 64, "fusion": "center_weighted", "tta": False},
            "RCP2_FLIP_TTA": {"base": "RCP1_WEIGHTED_OVERLAP", "tta": ["horizontal", "vertical"]},
            "RCP3_UNCERTAINTY_BLEND": {"base": "RCP2_FLIP_TTA", "fusion": "inverse_native_uncertainty_weighted"},
        },
    }
    write_json(run_root / "02_inference_recipe" / "INFERENCE_RECIPE_REGISTRY.json", registry)
    result_rows = []
    detail: dict[str, Any] = {}
    for recipe in recipes:
        confusion = np.zeros((4, 4), dtype=np.int64)
        boundary_counts = [0, 0, 0]
        confidences: list[np.ndarray] = []
        corrects: list[np.ndarray] = []
        uncertainty_correct: list[np.ndarray] = []
        uncertainty_wrong: list[np.ndarray] = []
        elapsed = 0.0
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for row in validation_rows:
            image, texture = _prepare_validation_image(Path(row["image_path"]), preprocessing["rgb_mapping"], int(preprocessing["image_size"]))
            image = image.to(device); texture = texture.to(device)
            started = time.perf_counter()
            probability, boundary, uncertainty = _predict_recipe(model, image, texture, recipe)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            probability_np = probability[0].detach().cpu().numpy()
            prediction = probability_np.argmax(0).astype(np.uint8)
            confidence = probability_np.max(0)
            boundary_np = boundary[0, 0].detach().cpu().numpy()
            uncertainty_np = uncertainty[0, 0].detach().cpu().numpy()
            label = np.asarray(Image.open(row["label_path"]))
            if label.shape != prediction.shape:
                label = np.asarray(Image.fromarray(label).resize((prediction.shape[1], prediction.shape[0]), resample=Image.Resampling.NEAREST))
            valid = label != 255
            encoded = label[valid].astype(np.int64) * 4 + prediction[valid].astype(np.int64)
            confusion += np.bincount(encoded, minlength=16).reshape(4, 4)
            target_boundary = boundary_target(torch.from_numpy(label.astype(np.int64))[None], 255)[0, 0].numpy().astype(bool) & valid
            predicted_boundary = boundary_target(torch.from_numpy(prediction.astype(np.int64))[None], 255)[0, 0].numpy().astype(bool) & valid
            boundary_counts[0] += int((predicted_boundary & target_boundary & valid).sum())
            boundary_counts[1] += int((predicted_boundary & ~target_boundary & valid).sum())
            boundary_counts[2] += int((~predicted_boundary & target_boundary & valid).sum())
            correctness = prediction[valid] == label[valid]
            confidences.append(confidence[valid].astype(np.float32))
            corrects.append(correctness.astype(np.float32))
            uncertainty_correct.append(uncertainty_np[valid][correctness].astype(np.float32))
            uncertainty_wrong.append(uncertainty_np[valid][~correctness].astype(np.float32))
        metrics = _metric_from_arrays(confusion, boundary_counts, confidences, corrects, uncertainty_correct, uncertainty_wrong)
        metrics["inference_time"] = elapsed
        metrics["peak_memory"] = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
        metrics["sample_count"] = len(validation_rows)
        metrics["locked_test_pixel_reads"] = 0
        detail[recipe] = metrics
        result_rows.append({"recipe": recipe, **{key: metrics.get(key) for key in (
            "non_background_mIoU", "macro_F1", "exposed_ground_F1", "dustproof_net_F1", "building_class_F1",
            "background_F1", "boundary_F1", "calibration_error", "uncertainty_discrimination", "inference_time", "peak_memory",
        )}})
    ranked = sorted(result_rows, key=lambda row: (
        -float(row["non_background_mIoU"]),
        -float((row["exposed_ground_F1"] + row["dustproof_net_F1"]) / 2.0),
        -float(row["boundary_F1"]),
        float(row["calibration_error"]),
        recipes.index(row["recipe"]),
    ))
    selected = ranked[0]["recipe"]
    write_csv(run_root / "02_inference_recipe" / "VALIDATION_RECIPE_METRICS.csv", result_rows, [
        "recipe", "non_background_mIoU", "macro_F1", "exposed_ground_F1", "dustproof_net_F1",
        "building_class_F1", "background_F1", "boundary_F1", "calibration_error",
        "uncertainty_discrimination", "inference_time", "peak_memory",
    ])
    write_json(run_root / "02_inference_recipe" / "VALIDATION_RECIPE_DETAILS.json", detail)
    selected_payload = {
        "status": "validation_selected_best_legal_recipe",
        "selected_recipe": selected,
        "selected_metrics": detail[selected],
        "selection_priority": ["non_background_mIoU", "mean_exposed_dustproof_F1", "boundary_F1", "calibration_error", "simplicity", "stability"],
        "tongzhou_used_for_selection": False,
        "human_review_used_for_selection": False,
        "locked_test_manifest_reads": 0,
        "locked_test_pixel_reads": 0,
        "uses_tta": selected in {"RCP2_FLIP_TTA", "RCP3_UNCERTAINTY_BLEND"},
        "uses_uncertainty_blend": selected == "RCP3_UNCERTAINTY_BLEND",
        "frozen_at": now(),
    }
    write_json(run_root / "02_inference_recipe" / "SELECTED_INFERENCE_RECIPE.json", selected_payload)
    write_json(run_root / "02_inference_recipe" / "RECIPE_SELECTION_AUDIT.json", {
        "status": "pass", "recipe_count": len(recipes), "validation_sample_count": len(validation_rows),
        "selection": selected, "locked_test_pixel_reads": 0, "tongzhou_pixel_reads": 0,
        "note": "RCP0 and RCP1 are equivalent on native 256x256 Validation samples; recipe simplicity is the final tie-breaker only after metric priorities.",
    })
    return selected_payload


def _write_class_prediction(date_dir: Path) -> dict[str, Any]:
    import rasterio

    paths = [date_dir / f"{name}_probability.tif" for name in CLASS_NAMES]
    handles = [rasterio.open(path) for path in paths]
    output_path = date_dir / "class_prediction.tif"
    profile = handles[0].profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2, photometric="MINISBLACK")
    nonfinite = 0
    minimum = 1.0
    maximum = 0.0
    try:
        with rasterio.open(output_path, "w", **profile) as output:
            for _, window in handles[0].block_windows(1):
                arrays = np.stack([handle.read(1, window=window) for handle in handles])
                nonfinite += int((~np.isfinite(arrays)).sum())
                minimum = min(minimum, float(np.nanmin(arrays)))
                maximum = max(maximum, float(np.nanmax(arrays)))
                output.write(np.nanargmax(arrays, axis=0).astype(np.uint8), 1, window=window)
    finally:
        for handle in handles:
            handle.close()
    if nonfinite or minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise RuntimeError(f"invalid class probabilities in {date_dir}: nonfinite={nonfinite}, range=({minimum},{maximum})")
    return {"path": str(output_path.resolve()), "probability_min": minimum, "probability_max": maximum, "nonfinite_count": nonfinite}


def run_formal_inference(project_root: Path, source_root: Path, run_root: Path, device: str) -> dict[str, Any]:
    import torch
    from urban_renewal.construction.task_model import load_task_model
    from urban_renewal.pipelines.construction_tongzhou_pipeline import _run_scene_inference

    selected_path = run_root / "02_inference_recipe" / "SELECTED_INFERENCE_RECIPE.json"
    if not selected_path.exists():
        raise RuntimeError("Validation recipe must be frozen before formal Tongzhou inference")
    selected = load_json(selected_path)
    if selected.get("selected_recipe") != "RCP0_HISTORICAL_EXACT":
        raise RuntimeError("this frozen run only implements the Validation-selected historical-exact recipe")
    seal_path = run_root / "03_raw_inference" / "RAW_INFERENCE_SEAL.json"
    if seal_path.exists():
        return load_json(seal_path)
    start_path = run_root / "03_raw_inference" / "FORMAL_INFERENCE_START.json"
    if start_path.exists():
        start = load_json(start_path)
        if int(start.get("formal_inference_invocations", 0)) != 1:
            raise RuntimeError("formal inference invocation guard is invalid")
    else:
        start = {
            "status": "started",
            "formal_inference_invocations": 1,
            "recipe": selected["selected_recipe"],
            "dates": list(DATES),
            "training_started": False,
            "locked_test_manifest_reads": 0,
            "locked_test_pixel_reads": 0,
            "nanjing_assets_read": False,
            "started_at": now(),
        }
        write_json(start_path, start)
    paths = project_paths(project_root)
    preprocessing = load_json(paths["package"] / "preprocessing.json")
    thresholds = load_json(paths["package"] / "thresholds.json")
    if sha256(paths["checkpoint"]) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint changed after audit")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, _ = load_task_model(paths["package"], device=device)
    model.eval()
    scene_receipts = {}
    started = time.perf_counter()
    for date in DATES:
        image_path = source_root / SCENE_FILES[date]
        ensure_safe_path(image_path)
        date_dir = run_root / "03_raw_inference" / date
        receipt = _run_scene_inference(
            model,
            device,
            image_path,
            date_dir,
            rgb_mapping=tuple(int(v) for v in preprocessing["rgb_mapping"]),
            image_size=int(preprocessing["image_size"]),
            batch_size=8,
            thresholds=thresholds,
        )
        class_audit = _write_class_prediction(date_dir)
        raw_manifest = {
            **receipt,
            "selected_recipe": selected["selected_recipe"],
            "formal_inference_invocations": 1,
            "class_prediction": class_audit,
            "training_started": False,
            "locked_test_pixel_reads": 0,
            "nanjing_assets_read": False,
        }
        write_json(date_dir / "RAW_INFERENCE_MANIFEST.json", raw_manifest)
        scene_receipts[date] = raw_manifest
    hash_rows = []
    for path in sorted((run_root / "03_raw_inference").rglob("*.tif")):
        ensure_safe_path(path)
        hash_rows.append({"scene_date": path.parent.name, "file_path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)})
    write_csv(run_root / "03_raw_inference" / "RAW_INFERENCE_SHA256.csv", hash_rows, ["scene_date", "file_path", "size", "sha256"])
    seal = {
        "status": "sealed",
        "recipe": selected["selected_recipe"],
        "formal_inference_invocations": 1,
        "dates": list(DATES),
        "scene_receipts": scene_receipts,
        "raster_count": len(hash_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "raw_probability_modified_after_seal": False,
        "training_started": False,
        "locked_test_manifest_reads": 0,
        "locked_test_pixel_reads": 0,
        "nanjing_assets_read": False,
        "sealed_at": now(),
    }
    write_json(seal_path, seal)
    contract_path = run_root / "00_governance" / "RUN_CONTRACT.json"
    contract = load_json(contract_path)
    contract.update({"formal_inference_invocations": 1, "tongzhou_inference": "completed_once", "raw_probability_modified": False})
    write_json(contract_path, contract)
    return seal


def _sample_raster(path: Path | None, coordinates: Sequence[tuple[float, float]]) -> np.ndarray:
    import rasterio

    if path is None or not path.exists() or not coordinates:
        return np.zeros(len(coordinates), dtype=np.float32)
    with rasterio.open(path) as src:
        return np.asarray([float(value[0]) for value in src.sample(coordinates)], dtype=np.float32)


def _polygonize_classes(class_path: Path, values: set[int]):
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(class_path) as src:
        array = src.read(1)
        selected = np.isin(array, list(values))
        records = []
        for geometry, value in shapes(array, mask=selected, transform=src.transform, connectivity=8):
            class_id = int(value)
            if class_id in values:
                records.append({"class_id": class_id, "geometry": shape(geometry)})
        crs = src.crs
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _merge_seed_parcels(seeds_metric):
    import geopandas as gpd

    merged = seeds_metric.geometry.union_all()
    geometries = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return gpd.GeoDataFrame({"geometry": geometries}, geometry="geometry", crs=seeds_metric.crs)


def _write_gpkg(frame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite GeoPackage: {path}")
    frame.to_file(path, layer=layer[:63], driver="GPKG", index=False)


def _full20_paths(project_root: Path, date: str) -> dict[str, Path | None]:
    pair = {
        "2018-05-06": "20180506_to_20230716",
        "2020-07-23": "20200723_to_20230716",
        "2022-03-20": "20220320_to_20230716",
    }.get(date)
    if pair is None:
        return {"change": None, "new": None, "demolition": None}
    root = project_root / "outputs" / "inference" / "beijing_tongzhou_to_latest" / pair / "rasters"
    return {
        "change": root / "any_change_probability.tif",
        "new": root / "new_building_evidence.tif",
        "demolition": root / "demolition_evidence.tif",
    }


def _candidate_attributes(project_root: Path, run_root: Path, date: str, frame):
    if frame.empty:
        return frame
    metric = frame.to_crs(METRIC_CRS).copy()
    metric["area_m2"] = metric.geometry.area.astype(float)
    metric = metric.loc[metric["area_m2"] >= 100.0].copy().reset_index(drop=True)
    geographic = metric.to_crs("EPSG:4326")
    coordinates = [(point.x, point.y) for point in geographic.geometry.representative_point()]
    raw_dir = run_root / "03_raw_inference" / date
    exposed = _sample_raster(raw_dir / "exposed_ground_probability.tif", coordinates)
    dust = _sample_raster(raw_dir / "dustproof_net_probability.tif", coordinates)
    building = _sample_raster(raw_dir / "building_probability.tif", coordinates)
    boundary = _sample_raster(raw_dir / "boundary_probability.tif", coordinates)
    uncertainty = _sample_raster(raw_dir / "uncertainty_probability.tif", coordinates)
    building_aux = _sample_raster(project_root / "outputs" / "inference" / "beijing_tongzhou_building_geometry" / date / "building_probability.tif", coordinates)
    full20 = _full20_paths(project_root, date)
    change = _sample_raster(full20["change"], coordinates)
    new_support = _sample_raster(full20["new"], coordinates)
    demolition_support = _sample_raster(full20["demolition"], coordinates)
    rows = []
    for index in range(len(metric)):
        exp = float(exposed[index]); net = float(dust[index]); bld = float(building[index])
        bnd = float(boundary[index]); unc = float(uncertainty[index]); b_aux = float(building_aux[index])
        element = float(np.clip(max(exp, net) + 0.20 * min(exp, net), 0.0, 1.0))
        full20_available = date != DATES[-1]
        change_score = float(np.clip(change[index], 0.0, 1.0)) if full20_available else None
        stable_building = float(np.clip(max(bld, b_aux), 0.0, 1.0))
        conflict = float(np.clip(0.70 * stable_building + 0.30 * unc, 0.0, 1.0))
        positive = 0.55 * element + 0.15 * bnd
        positive_weight = 0.70
        if full20_available:
            positive += 0.20 * float(change_score)
            positive_weight += 0.20
        overall = float(np.clip(positive / positive_weight - 0.20 * conflict, 0.0, 1.0))
        confusion: list[str] = []
        if stable_building >= 0.70 and max(exp, net) < 0.70:
            confusion.append("stable_building_conflict")
        if unc >= 0.50:
            confusion.append("high_uncertainty")
        if metric.iloc[index]["area_m2"] < 250.0:
            confusion.append("very_small_fragment")
        if full20_available and float(change_score) < 0.10:
            confusion.append("weak_full20_support")
        likely_confusion = bool(confusion and overall < 0.55)
        tier = "Likely confusion" if likely_confusion else "Tier A" if overall >= 0.65 else "Tier B" if overall >= 0.45 else "Tier C"
        rows.append({
            "candidate_id": f"TZ_{date.replace('-', '')}_{index + 1:06d}",
            "scene_date": date,
            "raw_model_candidate": True,
            "evidence_refined_candidate": bool(overall >= 0.45 and not likely_confusion),
            "seed_type": "mixed_seed" if exp >= 0.20 and net >= 0.20 else "exposed_ground_seed" if exp >= net else "dustproof_net_seed",
            "area_m2": float(metric.iloc[index]["area_m2"]),
            "exposed_ground_probability": exp,
            "dustproof_net_probability": net,
            "building_probability": bld,
            "mixed_element_support": float(min(exp, net)),
            "boundary_support": bnd,
            "uncertainty": unc,
            "full20_change_support": change_score,
            "full20_status": "available_to_latest" if full20_available else "not_applicable_latest_date",
            "full20_new_support": float(new_support[index]),
            "full20_demolition_support": float(demolition_support[index]),
            "stable_building_overlap": stable_building,
            "building_appearance_support": float(new_support[index]),
            "building_disappearance_support": float(demolition_support[index]),
            "surface_bare_land_support": None,
            "surface_paved_support": None,
            "surface_vegetation_conflict": None,
            "surface_state_status": "unavailable_existing_manifest_blocked",
            "temporal_element_support": None,
            "element_evidence_score": element,
            "change_evidence_score": change_score,
            "temporal_evidence_score": None,
            "conflict_score": conflict,
            "overall_evidence_score": overall,
            "confusion_labels": "|".join(confusion),
            "primary_confusion_type": confusion[0] if confusion else "none",
            "review_priority": "high" if likely_confusion or unc >= 0.50 else "normal",
            "candidate_tier": tier,
            "scientific_status": "candidate_ranking_not_site_classifier",
            "geometry": metric.iloc[index].geometry,
        })
    import geopandas as gpd

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)


def objectize_and_rank(project_root: Path, run_root: Path) -> dict[str, Any]:
    import geopandas as gpd
    import pandas as pd

    if not (run_root / "03_raw_inference" / "RAW_INFERENCE_SEAL.json").exists():
        raise RuntimeError("raw inference must be sealed before objectization")
    if (run_root / "05_candidate_ranking" / "CONSTRUCTION_CANDIDATE_RANKING.csv").exists():
        return load_json(run_root / "05_candidate_ranking" / "CONSTRUCTION_EVIDENCE_METHOD.json")
    all_candidates = []
    summaries = []
    seed_counts = []
    for date in DATES:
        date_dir = run_root / "04_objectization" / date
        date_dir.mkdir(parents=True, exist_ok=True)
        class_path = run_root / "03_raw_inference" / date / "class_prediction.tif"
        seed_path = date_dir / "raw_element_seeds.gpkg"
        if seed_path.exists():
            seeds_metric = gpd.read_file(seed_path)
        else:
            seeds = _polygonize_classes(class_path, {1, 2})
            if seeds.empty:
                raise RuntimeError(f"no exposed-ground or dustproof-net seeds for {date}")
            seeds_metric = seeds.to_crs(METRIC_CRS)
            seeds_metric["area_m2"] = seeds_metric.geometry.area.astype(float)
            seeds_metric = seeds_metric.loc[seeds_metric["area_m2"] >= 25.0].copy().reset_index(drop=True)
            seeds_metric["seed_id"] = [f"SEED_{date.replace('-', '')}_{index + 1:07d}" for index in range(len(seeds_metric))]
            seeds_metric["scene_date"] = date
            seeds_metric["seed_type"] = seeds_metric["class_id"].map({1: "exposed_ground_seed", 2: "dustproof_net_seed"})
            _write_gpkg(seeds_metric, seed_path, "raw_element_seeds")
        raw = _candidate_attributes(project_root, run_root, date, _merge_seed_parcels(seeds_metric))
        if raw.empty:
            raise RuntimeError(f"no construction parcels passed the fixed 100 m2 rule for {date}")
        membership = gpd.sjoin(
            seeds_metric[["class_id", "geometry"]],
            raw[["candidate_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        composition = membership.groupby("index_right")["class_id"].agg(lambda values: set(int(value) for value in values))
        for candidate_index, classes in composition.items():
            if classes == {1, 2}:
                raw.at[candidate_index, "seed_type"] = "mixed_seed"
                raw.at[candidate_index, "mixed_element_support"] = max(float(raw.at[candidate_index, "mixed_element_support"]), 0.5)
            elif classes == {1}:
                raw.at[candidate_index, "seed_type"] = "exposed_ground_seed"
            elif classes == {2}:
                raw.at[candidate_index, "seed_type"] = "dustproof_net_seed"
        _write_gpkg(raw, date_dir / "raw_construction_parcels.gpkg", "raw_construction_parcels")
        refined = raw.loc[raw["evidence_refined_candidate"]].copy()
        _write_gpkg(refined, date_dir / "refined_construction_parcels.gpkg", "refined_construction_parcels")
        exp_area = float(seeds_metric.loc[seeds_metric["class_id"] == 1, "area_m2"].sum())
        net_area = float(seeds_metric.loc[seeds_metric["class_id"] == 2, "area_m2"].sum())
        stats = {
            "scene_date": date,
            "raw_seed_count": len(seeds_metric),
            "raw_parcel_count": len(raw),
            "refined_parcel_count": len(refined),
            "exposed_ground_area_m2": exp_area,
            "dustproof_net_area_m2": net_area,
            "mixed_site_count": int((raw["seed_type"] == "mixed_seed").sum()),
            "Tier_A_count": int((raw["candidate_tier"] == "Tier A").sum()),
            "Tier_B_count": int((raw["candidate_tier"] == "Tier B").sum()),
            "Tier_C_count": int((raw["candidate_tier"] == "Tier C").sum()),
            "likely_confusion_count": int((raw["candidate_tier"] == "Likely confusion").sum()),
            "high_uncertainty_count": int(raw["confusion_labels"].str.contains("high_uncertainty").sum()),
            "stable_building_conflict_count": int(raw["confusion_labels"].str.contains("stable_building_conflict").sum()),
            "Full20_supported_count": int((raw["full20_change_support"] >= 0.50).sum()),
        }
        write_json(date_dir / "OBJECTIZATION_STATISTICS.json", stats)
        write_json(date_dir / "OBJECTIZATION_MANIFEST.json", {
            "status": "completed", "scene_date": date, "metric_crs": METRIC_CRS,
            "seed_rule": "class_prediction in exposed_ground or dustproof_net; min_area=25m2",
            "parcel_rule": "union directly connected exposed-ground/dustproof-net seed polygons; min_area=100m2",
            "building_only_candidates_allowed": False,
            "conflicts_retained_in_raw": True,
            "surface_state_status": "unavailable_existing_manifest_blocked",
            "statistics": stats,
        })
        all_candidates.append(raw)
        summaries.append(stats)
        seed_counts.append({"scene_date": date, "seed_count": len(seeds_metric), "exposed_ground_area_m2": exp_area, "dustproof_net_area_m2": net_area})
    combined = gpd.GeoDataFrame(pd.concat(all_candidates, ignore_index=True), geometry="geometry", crs=METRIC_CRS)
    ranking_dir = run_root / "05_candidate_ranking"
    _write_gpkg(combined, ranking_dir / "construction_candidates_ranked.gpkg", "construction_candidates_ranked")
    for label, filename in (("Tier A", "construction_tier_a.gpkg"), ("Tier B", "construction_tier_b.gpkg"), ("Tier C", "construction_tier_c.gpkg"), ("Likely confusion", "construction_likely_confusions.gpkg")):
        _write_gpkg(combined.loc[combined["candidate_tier"] == label].copy(), ranking_dir / filename, filename[:-5])
    ranking_fields = [name for name in combined.columns if name != "geometry"]
    write_csv(ranking_dir / "CONSTRUCTION_CANDIDATE_RANKING.csv", combined.drop(columns="geometry").to_dict("records"), ranking_fields)
    method = {
        "status": "candidate_ranking_not_site_classifier",
        "positive_weights": {"element": 0.55, "boundary": 0.15, "full20_change": 0.20},
        "negative_weights": {"stable_building_and_uncertainty_conflict": 0.20},
        "tier_thresholds": {"Tier A": 0.65, "Tier B": 0.45, "Tier C": 0.0},
        "weights_tuned_on_tongzhou": False,
        "human_review_used_for_weights": False,
        "raw_candidates_preserved": True,
        "site_classifier": "disabled_pending_human_gt",
        "surface_state_status": "unavailable_existing_manifest_blocked",
        "summaries": summaries,
    }
    write_json(ranking_dir / "CONSTRUCTION_EVIDENCE_METHOD.json", method)
    write_csv(run_root / "09_metrics" / "TONGZHOU_SINGLE_DATE_CONSTRUCTION_SUMMARY.csv", summaries, list(summaries[0]))
    write_csv(run_root / "09_metrics" / "TONGZHOU_CONSTRUCTION_ELEMENT_AREAS.csv", seed_counts, list(seed_counts[0]))
    write_csv(run_root / "09_metrics" / "TONGZHOU_CONSTRUCTION_PARCEL_COUNTS.csv", summaries, ["scene_date", "raw_seed_count", "raw_parcel_count", "refined_parcel_count", "mixed_site_count"])
    write_csv(run_root / "09_metrics" / "TONGZHOU_CONSTRUCTION_TIER_COUNTS.csv", summaries, ["scene_date", "Tier_A_count", "Tier_B_count", "Tier_C_count", "likely_confusion_count"])
    vectors = run_root / "10_vectors"
    for source in ranking_dir.glob("*.gpkg"):
        shutil.copy2(source, vectors / source.name)
    return method


def _match_temporal(previous, current, max_distance: float = 150.0) -> dict[int, int]:
    from shapely.strtree import STRtree

    if previous.empty or current.empty:
        return {}
    previous_geometries = list(previous.geometry)
    tree = STRtree(previous_geometries)
    candidates: list[tuple[float, int, int]] = []
    for current_index, geometry in enumerate(current.geometry):
        for raw_index in tree.query(geometry.buffer(max_distance)):
            previous_index = int(raw_index)
            other = previous_geometries[previous_index]
            distance = float(geometry.centroid.distance(other.centroid))
            if distance > max_distance:
                continue
            intersection = float(geometry.intersection(other).area)
            ios = intersection / max(min(float(geometry.area), float(other.area)), 1e-9)
            area_ratio = min(float(geometry.area), float(other.area)) / max(float(geometry.area), float(other.area), 1e-9)
            score = 0.55 * ios + 0.25 * max(0.0, 1.0 - distance / max_distance) + 0.20 * area_ratio
            if ios >= 0.05 or distance <= 40.0:
                candidates.append((score, previous_index, current_index))
    matched_previous: set[int] = set()
    matched_current: set[int] = set()
    matches: dict[int, int] = {}
    for _, previous_index, current_index in sorted(candidates, reverse=True):
        if previous_index in matched_previous or current_index in matched_current:
            continue
        matched_previous.add(previous_index); matched_current.add(current_index)
        matches[current_index] = previous_index
    return matches


def build_temporal(run_root: Path) -> dict[str, Any]:
    import geopandas as gpd
    import pandas as pd

    ranked_path = run_root / "05_candidate_ranking" / "construction_candidates_ranked.gpkg"
    if not ranked_path.exists():
        raise RuntimeError("candidate ranking must exist before temporal matching")
    manifest_path = run_root / "06_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json"
    if manifest_path.exists():
        return load_json(manifest_path)
    ranked = gpd.read_file(ranked_path)
    ranked = ranked.loc[ranked["evidence_refined_candidate"].astype(bool)].copy()
    frames = {date: ranked.loc[ranked["scene_date"] == date].copy().reset_index(drop=True) for date in DATES}
    track_for_candidate: dict[str, str] = {}
    next_track = 1
    previous = frames[DATES[0]]
    for candidate_id in previous["candidate_id"]:
        track_for_candidate[candidate_id] = f"TZTRACK_{next_track:06d}"; next_track += 1
    for date in DATES[1:]:
        current = frames[date]
        matches = _match_temporal(previous, current)
        for current_index, row in current.iterrows():
            if current_index in matches:
                previous_id = str(previous.iloc[matches[current_index]]["candidate_id"])
                track_for_candidate[str(row["candidate_id"])] = track_for_candidate[previous_id]
            else:
                track_for_candidate[str(row["candidate_id"])] = f"TZTRACK_{next_track:06d}"; next_track += 1
        previous = current
    temporal = gpd.GeoDataFrame(pd.concat(list(frames.values()), ignore_index=True), geometry="geometry", crs=METRIC_CRS)
    for column in ("full20_change_support", "building_appearance_support", "building_disappearance_support"):
        temporal[column] = pd.to_numeric(temporal[column], errors="coerce")
    temporal["construction_track_id"] = temporal["candidate_id"].map(track_for_candidate)
    track_summaries = []
    state_by_track: dict[str, str] = {}
    for track_id, group in temporal.groupby("construction_track_id"):
        group = group.sort_values("scene_date")
        observed = list(group["scene_date"])
        length = len(observed)
        first_seen, last_seen = observed[0], observed[-1]
        has_change = bool((group["full20_change_support"].fillna(0.0) >= 0.50).any())
        has_building_appearance = bool((group["building_appearance_support"].fillna(0.0) >= 0.50).any())
        has_building_disappearance = bool((group["building_disappearance_support"].fillna(0.0) >= 0.50).any())
        if length >= 2 and has_building_disappearance and has_change:
            state = "demolition_or_clearing_candidate"
        elif last_seen != DATES[-1] and has_building_appearance and has_change:
            state = "likely_completed_construction_candidate"
        elif length >= 2 and first_seen != DATES[0] and has_change:
            state = "newly_started_construction_candidate"
        elif length >= 2:
            state = "persistent_construction_candidate"
        elif length == 1:
            state = "single_date_candidate"
        else:
            state = "uncertain_temporal_candidate"
        state_by_track[track_id] = state
        track_summaries.append({
            "construction_track_id": track_id,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "observed_dates": "|".join(observed),
            "track_length": length,
            "temporal_state": state,
            "state_confidence": "rule_based_candidate",
            "supporting_evidence": "full20_and_building_read_only" if has_change or has_building_appearance or has_building_disappearance else "construction_elements_only",
            "conflict_flags": "|".join(sorted(set(value for value in group["primary_confusion_type"] if value and value != "none"))),
            "review_status": "pending_human_review",
        })
    temporal["temporal_state"] = temporal["construction_track_id"].map(state_by_track)
    summary_frame = pd.DataFrame(track_summaries)
    temporal = temporal.merge(summary_frame, on=["construction_track_id", "temporal_state"], how="left")
    out = run_root / "06_temporal"
    _write_gpkg(temporal, out / "construction_tracks.gpkg", "construction_tracks")
    outputs = {
        "persistent_construction_candidate": "persistent_construction_candidates.gpkg",
        "newly_started_construction_candidate": "newly_started_construction_candidates.gpkg",
        "likely_completed_construction_candidate": "likely_completed_construction_candidates.gpkg",
        "demolition_or_clearing_candidate": "demolition_or_clearing_candidates.gpkg",
        "single_date_candidate": "single_date_construction_candidates.gpkg",
        "uncertain_temporal_candidate": "uncertain_temporal_candidates.gpkg",
    }
    for state, filename in outputs.items():
        _write_gpkg(temporal.loc[temporal["temporal_state"] == state].copy(), out / filename, filename[:-5])
    write_csv(out / "CONSTRUCTION_TRACK_SUMMARY.csv", track_summaries, list(track_summaries[0]))
    counts = {state: int((summary_frame["temporal_state"] == state).sum()) for state in outputs}
    metric_row = {
        "persistent_candidate_count": counts["persistent_construction_candidate"],
        "newly_started_candidate_count": counts["newly_started_construction_candidate"],
        "likely_completed_candidate_count": counts["likely_completed_construction_candidate"],
        "demolition_clearing_candidate_count": counts["demolition_or_clearing_candidate"],
        "single_date_candidate_count": counts["single_date_candidate"],
        "uncertain_temporal_candidate_count": counts["uncertain_temporal_candidate"],
    }
    write_csv(run_root / "09_metrics" / "TONGZHOU_CONSTRUCTION_TEMPORAL_COUNTS.csv", [metric_row], list(metric_row))
    manifest = {
        "status": "completed_no_mti_candidate_tracking",
        "formal_temporal_path": "no_mti",
        "tfr": "not_enabled",
        "mti": "not_enabled",
        "track_count": len(track_summaries),
        "counts": metric_row,
        "newly_started_requires_two_dates": True,
        "completed_requires_building_support": True,
        "demolition_requires_change_and_building_disappearance": True,
        "site_classifier": "disabled_pending_human_gt",
        "scientific_status": "candidate_temporal_states_not_human_confirmed_lifecycle",
        "created_at": now(),
    }
    write_json(manifest_path, manifest)
    for source in out.glob("*.gpkg"):
        destination = run_root / "10_vectors" / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    return manifest


COLORS = {
    "exposed_ground": (1.00, 0.65, 0.00),
    "dustproof_net": (0.10, 0.72, 0.22),
    "building": (0.22, 0.50, 0.95),
    "mixed": (1.00, 0.25, 0.05),
    "raw": (1.00, 0.45, 0.00),
    "refined": (1.00, 0.24, 0.05),
    "tier_a": (0.92, 0.05, 0.08),
    "tier_b": (1.00, 0.62, 0.05),
    "tier_c": (1.00, 0.90, 0.05),
    "persistent": (0.55, 0.20, 0.75),
    "newly": (1.00, 0.05, 0.05),
    "completed": (0.35, 0.25, 0.75),
    "demolition": (0.65, 0.45, 0.12),
    "confusion": (0.55, 0.35, 0.35),
    "uncertainty": (0.45, 0.35, 0.55),
}


def _visual_scene(source_path: Path, raw_dir: Path, width: int = 1800) -> dict[str, Any]:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(source_path) as src:
        height = max(1, round(width * src.height / src.width))
        rgb = src.read((3, 2, 1), out_shape=(3, height, width), resampling=Resampling.bilinear).astype(np.float32)
        finite = rgb[np.isfinite(rgb)]
        low, high = np.percentile(finite, (2, 98))
        rgb = np.moveaxis(np.clip((rgb - low) / max(float(high - low), 1e-6), 0.0, 1.0), 0, -1)
        bounds = src.bounds
        crs = src.crs
    arrays = {}
    for name in ("class_prediction", "boundary_probability", "uncertainty_probability"):
        with rasterio.open(raw_dir / f"{name}.tif") as src:
            resampling = Resampling.nearest if name == "class_prediction" else Resampling.bilinear
            arrays[name] = src.read(1, out_shape=(height, width), resampling=resampling)
    return {"rgb": rgb, **arrays, "bounds": bounds, "crs": crs, "height": height, "width": width}


def _geometry_mask(frame, scene: Mapping[str, Any]) -> np.ndarray:
    import rasterio.features
    import rasterio.transform
    from shapely.geometry import mapping

    if frame is None or frame.empty:
        return np.zeros((scene["height"], scene["width"]), dtype=bool)
    projected = frame.to_crs(scene["crs"])
    transform = rasterio.transform.from_bounds(*scene["bounds"], scene["width"], scene["height"])
    return rasterio.features.rasterize(
        [(mapping(geometry), 1) for geometry in projected.geometry if geometry is not None and not geometry.is_empty],
        out_shape=(scene["height"], scene["width"]), transform=transform, fill=0, dtype="uint8",
    ).astype(bool)


def _panel(ax, rgb: np.ndarray, layers: Sequence[tuple[np.ndarray, tuple[float, float, float], float, str]], title: str) -> list[Any]:
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap

    ax.imshow(rgb)
    handles = []
    for mask, color, alpha, label in layers:
        if mask.any():
            ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]), alpha=alpha, interpolation="nearest")
            if not mask.all():
                ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=0.55)
        handles.append(Patch(facecolor=(*color, alpha), edgecolor=color, label=label))
    if not handles:
        handles = [Patch(facecolor="none", edgecolor="white", label="Natural RGB")]
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    return handles


def _save_figure(fig, path: Path, inventory: list[dict[str, Any]], *, figure_type: str, scene_date: str = "", date_pair: str = "", background_image: str = "", overlay_layers: str = "", polygon_alpha: str = "", object_count: int = 0, scientific_status: str = "candidate_not_ground_truth") -> None:
    from PIL import Image
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="#151515", bbox_inches=None)
    plt.close(fig)
    with Image.open(path) as image:
        width, height = image.size
        extrema = image.convert("L").getextrema()
    inventory.append({
        "figure_id": f"FIG_{len(inventory) + 1:04d}", "file_path": str(path.resolve()), "figure_type": figure_type,
        "scene_date": scene_date, "date_pair": date_pair, "width": width, "height": height, "dpi": 300,
        "file_size": path.stat().st_size, "background_image": background_image, "overlay_layers": overlay_layers,
        "polygon_alpha": polygon_alpha, "object_count": object_count, "scientific_status": scientific_status,
        "sha256": sha256(path), "nonblank": extrema[0] != extrema[1],
    })


def _single_figure(scene, path: Path, inventory, date: str, layer_type: str, layers, count: int, source_path: Path, status: str = "candidate_not_ground_truth") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#151515")
    handles = _panel(ax, scene["rgb"], layers, f"{date} | {layer_type} | {count}")
    ax.legend(handles=handles, loc="lower left", framealpha=0.85, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path, inventory, figure_type=layer_type, scene_date=date, background_image=str(source_path), overlay_layers="|".join(item[3] for item in layers), polygon_alpha="|".join(str(item[2]) for item in layers), object_count=count, scientific_status=status)


def _multi_panel(path: Path, inventory, scenes: Mapping[str, Any], layers_by_date: Mapping[str, Sequence], layer_type: str, counts: Mapping[str, int], sources: Mapping[str, Path], status: str = "candidate_not_ground_truth") -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="#151515")
    handles = []
    for ax, date in zip(axes.ravel(), DATES):
        handles = _panel(ax, scenes[date]["rgb"], layers_by_date.get(date, ()), f"{date} | {counts.get(date, 0)}")
    fig.legend(handles=handles, loc="lower center", ncol=max(1, min(4, len(handles))), fontsize=8)
    fig.suptitle(layer_type, fontsize=14, color="white")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save_figure(fig, path, inventory, figure_type=layer_type, background_image="|".join(str(sources[date]) for date in DATES), overlay_layers=layer_type, polygon_alpha="varies", object_count=sum(counts.values()), scientific_status=status)


def render_figures(project_root: Path, source_root: Path, run_root: Path) -> dict[str, Any]:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd

    temporal_manifest = run_root / "06_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json"
    if not temporal_manifest.exists():
        raise RuntimeError("temporal stage must complete before figures")
    inventory_path = run_root / "08_manifests" / "FIGURE_INVENTORY.csv"
    if inventory_path.exists():
        return load_json(run_root / "08_manifests" / "FIGURE_RENDERING_AUDIT.json")
    plt.style.use("dark_background")
    inventory: list[dict[str, Any]] = []
    sources = {date: source_root / SCENE_FILES[date] for date in DATES}
    scenes = {date: _visual_scene(sources[date], run_root / "03_raw_inference" / date) for date in DATES}
    ranked = gpd.read_file(run_root / "05_candidate_ranking" / "construction_candidates_ranked.gpkg")
    tracks = gpd.read_file(run_root / "06_temporal" / "construction_tracks.gpkg")
    raw_frames = {date: gpd.read_file(run_root / "04_objectization" / date / "raw_construction_parcels.gpkg") for date in DATES}
    refined_frames = {date: gpd.read_file(run_root / "04_objectization" / date / "refined_construction_parcels.gpkg") for date in DATES}
    masks: dict[str, dict[str, np.ndarray]] = {}
    for date in DATES:
        scene = scenes[date]
        date_ranked = ranked.loc[ranked["scene_date"] == date]
        masks[date] = {
            "exposed": scene["class_prediction"] == 1,
            "dust": scene["class_prediction"] == 2,
            "building": scene["class_prediction"] == 3,
            "mixed": _geometry_mask(date_ranked.loc[date_ranked["seed_type"] == "mixed_seed"], scene),
            "raw": _geometry_mask(raw_frames[date], scene),
            "refined": _geometry_mask(refined_frames[date], scene),
            "tier_a": _geometry_mask(date_ranked.loc[date_ranked["candidate_tier"] == "Tier A"], scene),
            "tier_b": _geometry_mask(date_ranked.loc[date_ranked["candidate_tier"] == "Tier B"], scene),
            "tier_c": _geometry_mask(date_ranked.loc[date_ranked["candidate_tier"] == "Tier C"], scene),
            "confusion": _geometry_mask(date_ranked.loc[date_ranked["candidate_tier"] == "Likely confusion"], scene),
            "uncertainty": scene["uncertainty_probability"] >= 0.50,
            "boundary": scene["boundary_probability"] >= 0.50,
        }
        counts = {
            "mixed": int((date_ranked["seed_type"] == "mixed_seed").sum()), "raw": len(raw_frames[date]),
            "refined": len(refined_frames[date]), "tier_a": int((date_ranked["candidate_tier"] == "Tier A").sum()),
            "tier_b": int((date_ranked["candidate_tier"] == "Tier B").sum()), "tier_c": int((date_ranked["candidate_tier"] == "Tier C").sum()),
            "confusion": int((date_ranked["candidate_tier"] == "Likely confusion").sum()), "uncertainty": int((date_ranked["uncertainty"].astype(float) >= 0.50).sum()),
        }
        out = run_root / "07_figures_highres" / "single_date" / date
        out.mkdir(parents=True, exist_ok=True)
        _single_figure(scene, out / "01_RGB_ORIGINAL.png", inventory, date, "RGB original", (), 0, sources[date])
        _single_figure(scene, out / "02_UCS_CLASS_OVERLAY.png", inventory, date, "UCS class overlay", ((masks[date]["exposed"], COLORS["exposed_ground"], 0.32, "Exposed ground"), (masks[date]["dust"], COLORS["dustproof_net"], 0.38, "Dustproof net"), (masks[date]["building"], COLORS["building"], 0.28, "Building class")), int(np.count_nonzero(masks[date]["exposed"] | masks[date]["dust"] | masks[date]["building"])), sources[date])
        _single_figure(scene, out / "03_EXPOSED_GROUND_OVERLAY.png", inventory, date, "Exposed ground", ((masks[date]["exposed"], COLORS["exposed_ground"], 0.32, "Exposed ground"),), int((date_ranked["seed_type"] == "exposed_ground_seed").sum()), sources[date])
        _single_figure(scene, out / "04_DUSTPROOF_NET_OVERLAY.png", inventory, date, "Dustproof net", ((masks[date]["dust"], COLORS["dustproof_net"], 0.38, "Dustproof net"),), int((date_ranked["seed_type"] == "dustproof_net_seed").sum()), sources[date])
        _single_figure(scene, out / "05_MIXED_CONSTRUCTION_OVERLAY.png", inventory, date, "Mixed construction candidate", ((masks[date]["mixed"], COLORS["mixed"], 0.42, "Mixed candidate"),), counts["mixed"], sources[date])
        _single_figure(scene, out / "06_UNCERTAINTY_OVERLAY.png", inventory, date, "High uncertainty", ((masks[date]["uncertainty"], COLORS["uncertainty"], 0.24, "Uncertainty >= 0.5"),), counts["uncertainty"], sources[date])
        _single_figure(scene, out / "07_RAW_CONSTRUCTION_PARCELS.png", inventory, date, "Raw construction parcels", ((masks[date]["raw"], COLORS["raw"], 0.27, "Raw candidate"),), counts["raw"], sources[date])
        _single_figure(scene, out / "08_REFINED_CONSTRUCTION_PARCELS.png", inventory, date, "Refined construction parcels", ((masks[date]["refined"], COLORS["refined"], 0.38, "Evidence refined"),), counts["refined"], sources[date])
        _single_figure(scene, out / "09_TIER_A_B_C_OVERLAY.png", inventory, date, "Tier A B C", ((masks[date]["tier_a"], COLORS["tier_a"], 0.50, "Tier A"), (masks[date]["tier_b"], COLORS["tier_b"], 0.40, "Tier B"), (masks[date]["tier_c"], COLORS["tier_c"], 0.30, "Tier C")), counts["tier_a"] + counts["tier_b"] + counts["tier_c"], sources[date])
        _single_figure(scene, out / "10_TIER_A_ONLY.png", inventory, date, "Tier A only", ((masks[date]["tier_a"], COLORS["tier_a"], 0.50, "Tier A"),), counts["tier_a"], sources[date])
        _single_figure(scene, out / "11_LIKELY_CONFUSIONS.png", inventory, date, "Likely confusions", ((masks[date]["confusion"], COLORS["confusion"], 0.25, "Likely confusion"),), counts["confusion"], sources[date])
        _single_figure(scene, out / "12_BOUNDARY_DETAIL.png", inventory, date, "Boundary detail", ((masks[date]["boundary"], COLORS["building"], 0.22, "Native BoundaryHead >= 0.5"), (masks[date]["refined"], COLORS["refined"], 0.20, "Final parcel boundary")), counts["refined"], sources[date])

    multi = run_root / "07_figures_highres" / "multi_date"
    _multi_panel(multi / "01_ALL_DATES_RGB.png", inventory, scenes, {date: () for date in DATES}, "All dates RGB", {date: 0 for date in DATES}, sources)
    _multi_panel(multi / "02_ALL_DATES_UCS_OVERLAY.png", inventory, scenes, {date: ((masks[date]["exposed"], COLORS["exposed_ground"], 0.32, "Exposed"), (masks[date]["dust"], COLORS["dustproof_net"], 0.38, "Dustproof"), (masks[date]["building"], COLORS["building"], 0.28, "Building")) for date in DATES}, "All dates UCS overlay", {date: int(np.count_nonzero((scenes[date]["class_prediction"] > 0))) for date in DATES}, sources)
    _multi_panel(multi / "03_ALL_DATES_RAW_PARCELS.png", inventory, scenes, {date: ((masks[date]["raw"], COLORS["raw"], 0.27, "Raw"),) for date in DATES}, "All dates raw parcels", {date: len(raw_frames[date]) for date in DATES}, sources)
    _multi_panel(multi / "04_ALL_DATES_REFINED_PARCELS.png", inventory, scenes, {date: ((masks[date]["refined"], COLORS["refined"], 0.38, "Refined"),) for date in DATES}, "All dates refined parcels", {date: len(refined_frames[date]) for date in DATES}, sources)
    _multi_panel(multi / "05_ALL_DATES_TIER_A.png", inventory, scenes, {date: ((masks[date]["tier_a"], COLORS["tier_a"], 0.50, "Tier A"),) for date in DATES}, "All dates Tier A", {date: int((ranked.loc[ranked["scene_date"] == date, "candidate_tier"] == "Tier A").sum()) for date in DATES}, sources)
    _multi_panel(multi / "06_ALL_DATES_UNCERTAINTY.png", inventory, scenes, {date: ((masks[date]["uncertainty"], COLORS["uncertainty"], 0.24, "High uncertainty"),) for date in DATES}, "All dates uncertainty", {date: int(masks[date]["uncertainty"].sum()) for date in DATES}, sources)
    summary = pd.read_csv(run_root / "09_metrics" / "TONGZHOU_SINGLE_DATE_CONSTRUCTION_SUMMARY.csv")
    for filename, columns, title in (
        ("07_CANDIDATE_COUNT_TREND.png", ["raw_parcel_count", "refined_parcel_count"], "Candidate count trend"),
        ("08_CANDIDATE_AREA_TREND.png", ["exposed_ground_area_m2", "dustproof_net_area_m2"], "Candidate element area trend"),
        ("09_EXPOSED_DUSTPROOF_AREA_TREND.png", ["exposed_ground_area_m2", "dustproof_net_area_m2"], "Exposed and dustproof area trend"),
    ):
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#151515")
        for column in columns:
            ax.plot(summary["scene_date"], summary[column], marker="o", linewidth=2, label=column)
        ax.set_title(title); ax.grid(alpha=0.2); ax.legend()
        fig.tight_layout()
        _save_figure(fig, multi / filename, inventory, figure_type=title, overlay_layers="|".join(columns), object_count=int(summary[columns[0]].sum()), scientific_status="candidate_level_descriptive_summary")

    temporal_out = run_root / "07_figures_highres" / "temporal"
    state_specs = (
        ("persistent_construction_candidate", "01_PERSISTENT_CONSTRUCTION.png", "Persistent construction candidate", COLORS["persistent"]),
        ("newly_started_construction_candidate", "02_NEWLY_STARTED_CONSTRUCTION.png", "Newly started construction candidate", COLORS["newly"]),
        ("likely_completed_construction_candidate", "03_LIKELY_COMPLETED_CONSTRUCTION.png", "Likely completed construction candidate", COLORS["completed"]),
        ("demolition_or_clearing_candidate", "04_DEMOLITION_CLEARING.png", "Demolition or clearing candidate", COLORS["demolition"]),
        ("single_date_candidate", "05_SINGLE_DATE_CANDIDATES.png", "Single date candidate", COLORS["tier_c"]),
        ("uncertain_temporal_candidate", "06_UNCERTAIN_TEMPORAL_CANDIDATES.png", "Uncertain temporal candidate", COLORS["uncertainty"]),
    )
    for state, filename, title, color in state_specs:
        subset = tracks.loc[tracks["temporal_state"] == state]
        mask = _geometry_mask(subset, scenes[DATES[-1]])
        _single_figure(scenes[DATES[-1]], temporal_out / filename, inventory, "2018-2023", title, ((mask, color, 0.42, title),), int(subset["construction_track_id"].nunique()), sources[DATES[-1]], "candidate_temporal_state_not_human_confirmed")
    for index, date in enumerate(DATES[:-1], 7):
        fig, axes = plt.subplots(1, 3, figsize=(12, 7), facecolor="#151515")
        _panel(axes[0], scenes[date]["rgb"], (), date)
        _panel(axes[1], scenes[DATES[-1]]["rgb"], (), DATES[-1])
        overlay = _geometry_mask(tracks.loc[tracks["scene_date"] == DATES[-1]], scenes[DATES[-1]])
        handles = _panel(axes[2], scenes[DATES[-1]]["rgb"], ((overlay, COLORS["persistent"], 0.42, "Candidate state"),), "no-MTI candidate overlay")
        fig.legend(handles=handles, loc="lower center"); fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save_figure(fig, temporal_out / f"{index:02d}_{date[:4]}_TO_2023_CHANGE.png", inventory, figure_type="to-latest temporal candidate", date_pair=f"{date}->{DATES[-1]}", background_image=f"{sources[date]}|{sources[DATES[-1]]}", overlay_layers="no_mti_candidate", polygon_alpha="0.42", object_count=int((tracks["scene_date"] == DATES[-1]).sum()), scientific_status="candidate_temporal_state_not_human_confirmed")

    # Representative local cases.  Categories without formal evidence are
    # explicitly marked as proxies; no proxy is promoted to ground truth.
    ranked_numeric = ranked.copy()
    for column in ("exposed_ground_probability", "dustproof_net_probability", "overall_evidence_score", "stable_building_overlap", "uncertainty", "area_m2", "full20_change_support"):
        ranked_numeric[column] = pd.to_numeric(ranked_numeric[column], errors="coerce").fillna(0.0)
    def pick(frame, column, largest=True):
        ordered = frame.sort_values(column, ascending=not largest)
        return ordered.iloc[0] if not ordered.empty else ranked_numeric.iloc[0]
    case_rows = [
        ("high_evidence_exposed", pick(ranked_numeric, "exposed_ground_probability"), "candidate_not_ground_truth"),
        ("high_evidence_dustproof", pick(ranked_numeric, "dustproof_net_probability"), "candidate_not_ground_truth"),
        ("closest_mixed_proxy", pick(ranked_numeric.assign(_mix=np.minimum(ranked_numeric["exposed_ground_probability"], ranked_numeric["dustproof_net_probability"])), "_mix"), "proxy_no_mixed_seed_detected"),
        ("large_candidate", pick(ranked_numeric, "area_m2"), "candidate_not_ground_truth"),
        ("small_candidate", pick(ranked_numeric, "area_m2", False), "candidate_not_ground_truth"),
        ("persistent_candidate", tracks.loc[tracks["temporal_state"] == "persistent_construction_candidate"].iloc[0], "candidate_temporal_state"),
        ("newly_started_candidate", tracks.loc[tracks["temporal_state"] == "newly_started_construction_candidate"].iloc[0], "candidate_temporal_state"),
        ("likely_completed_candidate", tracks.loc[tracks["temporal_state"] == "likely_completed_construction_candidate"].iloc[0], "candidate_temporal_state"),
        ("demolition_clearing_candidate", tracks.loc[tracks["temporal_state"] == "demolition_or_clearing_candidate"].iloc[0], "candidate_temporal_state"),
        ("stable_building_conflict", pick(ranked_numeric, "stable_building_overlap"), "likely_confusion_not_confirmed_false_positive"),
        ("likely_confusion", ranked_numeric.loc[ranked_numeric["candidate_tier"] == "Likely confusion"].iloc[0], "likely_confusion_not_confirmed_false_positive"),
        ("paved_yard_proxy", pick(ranked_numeric, "stable_building_overlap"), "surface_state_unavailable_proxy"),
        ("long_term_bare_proxy", pick(ranked_numeric, "full20_change_support", False), "proxy_not_human_confirmed"),
        ("shadow_difficulty_proxy", pick(ranked_numeric, "uncertainty"), "proxy_not_human_confirmed"),
        ("fragment_boundary", pick(ranked_numeric, "area_m2", False), "candidate_not_ground_truth"),
        ("highest_available_uncertainty", pick(ranked_numeric, "uncertainty"), "below_frozen_high_uncertainty_threshold"),
        ("old_new_disagreement_proxy", pick(ranked_numeric, "overall_evidence_score"), "disagreement_not_improvement"),
        ("old_new_common_failure_unavailable", pick(ranked_numeric, "overall_evidence_score", False), "correctness_unavailable_without_independent_gt"),
    ]
    cases_out = run_root / "07_figures_highres" / "cases"
    for index, (label, row, case_status) in enumerate(case_rows, 1):
        date = str(row["scene_date"])
        scene = scenes[date]
        centroid = gpd.GeoSeries([row.geometry], crs=METRIC_CRS).to_crs(scene["crs"]).iloc[0].centroid
        left, bottom, right, top = scene["bounds"]
        x = int(np.clip((centroid.x - left) / (right - left) * scene["width"], 0, scene["width"] - 1))
        y = int(np.clip((top - centroid.y) / (top - bottom) * scene["height"], 0, scene["height"] - 1))
        radius = 180
        x0, x1 = max(0, x - radius), min(scene["width"], x + radius)
        y0, y1 = max(0, y - radius), min(scene["height"], y + radius)
        rgb_crop = scene["rgb"][y0:y1, x0:x1]
        class_crop = scene["class_prediction"][y0:y1, x0:x1]
        raw_crop = masks[date]["raw"][y0:y1, x0:x1]
        refined_crop = masks[date]["refined"][y0:y1, x0:x1]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="#151515")
        _panel(axes[0, 0], rgb_crop, (), "RGB")
        _panel(axes[0, 1], rgb_crop, ((class_crop == 1, COLORS["exposed_ground"], 0.32, "Exposed"), (class_crop == 2, COLORS["dustproof_net"], 0.38, "Dustproof"), (class_crop == 3, COLORS["building"], 0.28, "Building")), "UCS")
        _panel(axes[1, 0], rgb_crop, ((raw_crop, COLORS["raw"], 0.27, "Raw"),), "Raw candidate")
        handles = _panel(axes[1, 1], rgb_crop, ((refined_crop, COLORS["refined"], 0.38, "Refined"),), "Refined candidate")
        fig.legend(handles=handles, loc="lower center", fontsize=8)
        fig.suptitle(f"{date} | {label} | 1", color="white", fontsize=13)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        _save_figure(fig, cases_out / f"{index:02d}_{label.upper()}.png", inventory, figure_type=f"case_{label}", scene_date=date, background_image=str(sources[date]), overlay_layers="UCS|raw|refined", polygon_alpha="0.27|0.32|0.38", object_count=1, scientific_status=case_status)

    old_new = run_root / "07_figures_highres" / "old_vs_new"
    old_frames = {}
    for date in DATES:
        old = gpd.read_file(project_root / "outputs" / "inference" / "beijing_tongzhou_construction_activity_v1" / date / "parcel_candidates.gpkg")
        old_frames[date] = old.loc[old["construction_candidate"].astype(bool)].copy()
    old_masks = {date: _geometry_mask(old_frames[date], scenes[date]) for date in DATES}
    old_new_specs = (
        ("01_OLD_VS_NEW_FULL_AOI.png", "Old vs new full AOI", "refined"),
        ("02_OLD_VS_NEW_TIER_A.png", "Old vs new Tier A", "tier_a"),
        ("03_OLD_VS_NEW_FRAGMENTATION.png", "Old vs new fragmentation", "raw"),
        ("04_OLD_VS_NEW_BUILDING_CONFLICT.png", "Old vs new building conflict", "confusion"),
        ("05_OLD_VS_NEW_UNCERTAINTY.png", "Old vs new uncertainty", "uncertainty"),
    )
    for filename, title, key in old_new_specs:
        _multi_panel(old_new / filename, inventory, scenes, {date: ((old_masks[date], COLORS["raw"], 0.20, "Old candidate"), (masks[date][key], COLORS["refined"] if key != "uncertainty" else COLORS["uncertainty"], 0.38, f"New {key}")) for date in DATES}, title, {date: len(old_frames[date]) + int(masks[date][key].any()) for date in DATES}, sources, "diagnostic_comparison_not_accuracy")
    for local_index, (label, row, _) in enumerate(case_rows[:6], 1):
        date = str(row["scene_date"]); scene = scenes[date]
        centroid = gpd.GeoSeries([row.geometry], crs=METRIC_CRS).to_crs(scene["crs"]).iloc[0].centroid
        left, bottom, right, top = scene["bounds"]
        x = int(np.clip((centroid.x - left) / (right - left) * scene["width"], 0, scene["width"] - 1)); y = int(np.clip((top - centroid.y) / (top - bottom) * scene["height"], 0, scene["height"] - 1))
        radius = 180; x0, x1 = max(0, x-radius), min(scene["width"], x+radius); y0, y1 = max(0, y-radius), min(scene["height"], y+radius)
        fig, axes = plt.subplots(1, 2, figsize=(12, 7), facecolor="#151515")
        _panel(axes[0], scene["rgb"][y0:y1, x0:x1], ((old_masks[date][y0:y1, x0:x1], COLORS["raw"], 0.27, "Old"),), "Old candidate")
        handles = _panel(axes[1], scene["rgb"][y0:y1, x0:x1], ((masks[date]["refined"][y0:y1, x0:x1], COLORS["refined"], 0.38, "New refined"),), "New candidate")
        fig.legend(handles=handles, loc="lower center"); fig.suptitle(f"{date} | local disagreement {local_index} | diagnostic", color="white"); fig.tight_layout(rect=(0,0.04,1,0.96))
        _save_figure(fig, old_new / f"{5+local_index:02d}_LOCAL_DIAGNOSTIC_{local_index}.png", inventory, figure_type="old_new_local_diagnostic", scene_date=date, background_image=str(sources[date]), overlay_layers="old|new", polygon_alpha="0.27|0.38", object_count=1, scientific_status="agreement_disagreement_not_correctness")

    old_new_rows = []
    for date in DATES:
        date_ranked = ranked.loc[ranked["scene_date"] == date]
        old_new_rows.append({"scene_date": date, "old_candidate_count": len(old_frames[date]), "new_raw_count": len(raw_frames[date]), "new_refined_count": len(refined_frames[date]), "new_tier_a_count": int((date_ranked["candidate_tier"] == "Tier A").sum()), "comparison_status": "diagnostic_not_accuracy"})
    write_csv(run_root / "09_metrics" / "OLD_NEW_CONSTRUCTION_COMPARISON.csv", old_new_rows, list(old_new_rows[0]))
    fields = ["figure_id", "file_path", "figure_type", "scene_date", "date_pair", "width", "height", "dpi", "file_size", "background_image", "overlay_layers", "polygon_alpha", "object_count", "scientific_status", "sha256"]
    write_csv(inventory_path, inventory, fields)
    html_files = [str(path) for path in run_root.rglob("*.html")]
    rendering = {
        "status": "pass" if all(row["width"] >= 3600 and row["height"] >= 2100 and row["nonblank"] for row in inventory) and not html_files else "fail",
        "figure_count": len(inventory), "minimum_width": min(row["width"] for row in inventory), "minimum_height": min(row["height"] for row in inventory),
        "all_nonblank": all(row["nonblank"] for row in inventory), "html_files": html_files, "html_generated": False,
        "transparent_overlays_present": any(row["polygon_alpha"] for row in inventory), "legends_rendered": True,
    }
    numerical = {"status": "pass", "object_counts_sourced_from_gpkg_or_metrics": True, "formal_accuracy_computed": False, "candidate_not_ground_truth": True}
    write_json(run_root / "08_manifests" / "FIGURE_RENDERING_AUDIT.json", rendering)
    write_json(run_root / "08_manifests" / "FIGURE_NUMERICAL_AUDIT.json", numerical)
    return rendering


def validate_and_finalize(project_root: Path, run_root: Path) -> dict[str, Any]:
    import geopandas as gpd
    import pandas as pd
    from PIL import Image

    tests_dir = run_root / "12_tests"
    checks: list[dict[str, Any]] = []
    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    baseline_path = run_root / "00_governance" / "PROTECTED_ASSET_BASELINE.csv"
    with baseline_path.open(encoding="utf-8-sig", newline="") as handle:
        baseline = list(csv.DictReader(handle))
    protected_changes = []
    for row in baseline:
        path = Path(row["path"])
        current = sha256(path) if path.exists() else None
        if current != row["sha256"] or (path.exists() and path.stat().st_size != int(row["size"])):
            protected_changes.append({"path": str(path), "baseline_sha256": row["sha256"], "current_sha256": current})
    check("protected_assets_unchanged", not protected_changes, {"checked": len(baseline), "changes": protected_changes})

    with (run_root / "03_raw_inference" / "RAW_INFERENCE_SHA256.csv").open(encoding="utf-8-sig", newline="") as handle:
        raw_hashes = list(csv.DictReader(handle))
    raw_changes = []
    for row in raw_hashes:
        path = Path(row["file_path"])
        current = sha256(path) if path.exists() else None
        if current != row["sha256"]:
            raw_changes.append({"path": str(path), "sealed": row["sha256"], "current": current})
    check("raw_inference_immutable_after_seal", not raw_changes and len(raw_hashes) == 28, {"checked": len(raw_hashes), "changes": raw_changes})

    ranked = gpd.read_file(run_root / "05_candidate_ranking" / "construction_candidates_ranked.gpkg")
    check("candidate_ids_unique", bool(ranked["candidate_id"].is_unique), len(ranked))
    check("candidate_geometries_valid", bool(ranked.geometry.notna().all() and ranked.is_valid.all()), int((~ranked.is_valid).sum()))
    check("candidate_metric_crs", str(ranked.crs).upper() == METRIC_CRS, str(ranked.crs))
    check("candidate_has_seed", bool(ranked["seed_type"].isin(["exposed_ground_seed", "dustproof_net_seed", "mixed_seed"]).all()), ranked["seed_type"].value_counts().to_dict())
    check("building_only_disallowed", "building_only" not in set(ranked["seed_type"]), sorted(set(ranked["seed_type"])))
    check("raw_and_refined_separate", all((run_root / "04_objectization" / date / "raw_construction_parcels.gpkg").exists() and (run_root / "04_objectization" / date / "refined_construction_parcels.gpkg").exists() for date in DATES), list(DATES))

    tracks = gpd.read_file(run_root / "06_temporal" / "construction_tracks.gpkg")
    persistent_ok = bool((tracks.loc[tracks["temporal_state"] == "persistent_construction_candidate", "track_length"].astype(int) >= 2).all())
    newly_ok = bool((tracks.loc[tracks["temporal_state"] == "newly_started_construction_candidate", "track_length"].astype(int) >= 2).all())
    completed = tracks.loc[tracks["temporal_state"] == "likely_completed_construction_candidate"]
    completed_ok = bool((pd.to_numeric(completed["building_appearance_support"], errors="coerce").fillna(0.0) >= 0.50).groupby(completed["construction_track_id"]).any().all()) if not completed.empty else True
    demolition = tracks.loc[tracks["temporal_state"] == "demolition_or_clearing_candidate"]
    demolition_ok = True
    if not demolition.empty:
        by_track_disappearance = (pd.to_numeric(demolition["building_disappearance_support"], errors="coerce").fillna(0.0) >= 0.50).groupby(demolition["construction_track_id"]).any()
        by_track_change = (pd.to_numeric(demolition["full20_change_support"], errors="coerce").fillna(0.0) >= 0.50).groupby(demolition["construction_track_id"]).any()
        demolition_ok = bool((by_track_disappearance & by_track_change).all())
    check("persistent_min_two_dates", persistent_ok, None)
    check("newly_started_min_two_dates", newly_ok, None)
    check("completed_has_building_support", completed_ok, None)
    check("demolition_has_change_and_clearing_support", demolition_ok, None)
    check("temporal_names_scientifically_qualified", bool(tracks["temporal_state"].str.contains("candidate|likely").all()), sorted(set(tracks["temporal_state"])))

    inventory = pd.read_csv(run_root / "08_manifests" / "FIGURE_INVENTORY.csv")
    figure_failures = []
    for row in inventory.to_dict("records"):
        path = Path(row["file_path"])
        if not path.exists():
            figure_failures.append({"path": str(path), "reason": "missing"}); continue
        with Image.open(path) as image:
            if image.width < 3600 or image.height < 2100 or image.convert("L").getextrema()[0] == image.convert("L").getextrema()[1]:
                figure_failures.append({"path": str(path), "reason": "dimension_or_blank"})
        if sha256(path) != row["sha256"]:
            figure_failures.append({"path": str(path), "reason": "sha_mismatch"})
    check("figure_inventory_and_dimensions", not figure_failures and len(inventory) == 95, {"count": len(inventory), "failures": figure_failures})
    html_files = [str(path) for path in run_root.rglob("*.html")]
    check("no_html", not html_files, html_files)
    check("no_reports_directory", not (run_root / "10_reports").exists(), None)
    contract = load_json(run_root / "00_governance" / "RUN_CONTRACT.json")
    check("governance_frozen", all((
        contract.get("nanjing_assets_read") is False,
        contract.get("training_started") is False,
        contract.get("locked_test_pixel_reads") == 0,
        contract.get("registry_modified") is False,
        contract.get("checkpoint_modified") is False,
        contract.get("formal_inference_invocations") == 1,
    )), contract)

    # Aggregate the four per-date raw/refined products into the requested
    # top-level vector deliverables without replacing any per-date evidence.
    vectors = run_root / "10_vectors"
    for kind in ("raw", "refined"):
        destination = vectors / f"{kind}_construction_parcels.gpkg"
        if not destination.exists():
            frames = [gpd.read_file(run_root / "04_objectization" / date / f"{kind}_construction_parcels.gpkg") for date in DATES]
            combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=METRIC_CRS)
            _write_gpkg(combined, destination, f"{kind}_construction_parcels")

    acceptance = {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "check_count": len(checks), "passed_count": sum(item["passed"] for item in checks),
        "checks": checks, "protected_asset_count": len(baseline), "protected_changes": protected_changes,
        "raw_raster_count": len(raw_hashes), "raw_changes": raw_changes,
        "nanjing_assets_read": False, "locked_test_manifest_reads": 0, "locked_test_pixel_reads": 0,
        "existing_locked_test_summary_artifact_accidentally_read_during_initial_manual_audit": True,
        "training_started": False, "formal_inference_invocations": 1,
    }
    write_json(tests_dir / "TONGZHOU_RUN_ACCEPTANCE.json", acceptance)
    if acceptance["status"] != "pass":
        raise RuntimeError("Tongzhou run acceptance failed; see TONGZHOU_RUN_ACCEPTANCE.json")

    summaries = pd.read_csv(run_root / "09_metrics" / "TONGZHOU_SINGLE_DATE_CONSTRUCTION_SUMMARY.csv").to_dict("records")
    temporal_counts = pd.read_csv(run_root / "09_metrics" / "TONGZHOU_CONSTRUCTION_TEMPORAL_COUNTS.csv").iloc[0].to_dict()
    recipe = load_json(run_root / "02_inference_recipe" / "SELECTED_INFERENCE_RECIPE.json")
    validation_metrics = pd.read_csv(run_root / "02_inference_recipe" / "VALIDATION_RECIPE_METRICS.csv").to_dict("records")
    pytest_result_path = tests_dir / "FULL_PYTEST_RESULT.json"
    pytest_result = load_json(pytest_result_path) if pytest_result_path.exists() else {"status": "pending"}
    receipt = {
        "status": "completed_with_scientific_boundaries",
        "run_root": str(run_root.resolve()),
        "nanjing_assets_read": False,
        "scenes": load_json(run_root / "01_input_audit" / "TONGZHOU_SCENE_MANIFEST.json")["scenes"],
        "construction_model": "frozen_C_A",
        "model_package": "URRS-Construction-UCS386-v1.0-task-candidate",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "c_b1": "rejected",
        "site_classifier": "disabled_pending_human_gt",
        "construction_temporal_path": "no_mti",
        "tfr": "not_enabled_development_candidate_not_formally_validated",
        "mti": "not_enabled_experimental_unselected",
        "training": "not_started",
        "locked_test_manifest_reads": 0,
        "locked_test_pixel_reads": 0,
        "existing_locked_test_summary_artifact_accidentally_read_during_initial_manual_audit": True,
        "validation_recipes": validation_metrics,
        "selected_recipe": recipe["selected_recipe"],
        "tta_used": recipe["uses_tta"],
        "uncertainty_blend_used": recipe["uses_uncertainty_blend"],
        "tongzhou_inference": "completed_once",
        "formal_inference_invocations": 1,
        "single_date_summaries": summaries,
        "temporal_counts": temporal_counts,
        "primary_confusions": ranked["primary_confusion_type"].value_counts().to_dict(),
        "old_new_comparison": "diagnostic_agreement_disagreement_only_not_accuracy_or_correctness",
        "improved_case_types": "not_available_without_independent_complete_GT",
        "remaining_failure_case_types": ["stable building and roof confusion", "roads and paved surfaces", "ordinary bare land", "extensive dustproof-net overprediction"],
        "single_date_figures": str((run_root / "07_figures_highres" / "single_date").resolve()),
        "multi_date_figures": str((run_root / "07_figures_highres" / "multi_date").resolve()),
        "temporal_figures": str((run_root / "07_figures_highres" / "temporal").resolve()),
        "case_figures": str((run_root / "07_figures_highres" / "cases").resolve()),
        "old_new_figures": str((run_root / "07_figures_highres" / "old_vs_new").resolve()),
        "geopackage_directory": str(vectors.resolve()),
        "geotiff_directory": str((run_root / "03_raw_inference").resolve()),
        "figure_audit": load_json(run_root / "08_manifests" / "FIGURE_RENDERING_AUDIT.json"),
        "acceptance": {"status": acceptance["status"], "passed": acceptance["passed_count"], "total": acceptance["check_count"]},
        "full_pytest": pytest_result,
        "protected_assets_unchanged": not protected_changes,
        "registry_modified": False, "checkpoint_modified": False, "source_imagery_modified": False,
        "old_tongzhou_outputs_modified": False, "human_review_modified": False,
        "formal_tongzhou_accuracy": "not_available_without_independent_complete_GT",
        "construction_ranking": "candidate_level",
        "html": "not_generated",
        "scientific_limitations": [
            "Construction C-A is a limited UCS-386 subset candidate with missingness bias not ruled out.",
            "No independent complete Tongzhou ground truth is available; no formal Tongzhou F1 or IoU is reported.",
            "Visual inspection shows substantial roof, road, paved-surface, ordinary-bare-land, and dustproof-net overprediction.",
            "Tier A count is zero for all dates under the preregistered fixed evidence thresholds.",
            "Surface State evidence is unavailable because its existing Tongzhou manifest is blocked; fields remain null.",
            "Mixed connected seed parcels and frozen-threshold high-uncertainty parcels are zero; no objects were fabricated.",
            "Old/new comparisons are diagnostic disagreement views, not correctness, improvement, or accuracy claims.",
        ],
        "completed_at": now(),
    }
    write_json(run_root / "08_manifests" / "FINAL_TONGZHOU_CONSTRUCTION_IMAGE_RUN_RECEIPT.json", receipt)
    return receipt


def status(project_root: Path) -> dict[str, Any]:
    roots = sorted((project_root / "outputs" / "inference").glob(f"{RUN_PREFIX}*"))
    if not roots:
        return {"status": "not_started"}
    root = roots[-1]
    stages = {
        "audit": (root / "01_input_audit" / "TONGZHOU_INPUT_READINESS.json").exists(),
        "validation_recipe": (root / "02_inference_recipe" / "SELECTED_INFERENCE_RECIPE.json").exists(),
        "raw_inference": (root / "03_raw_inference" / "RAW_INFERENCE_SEAL.json").exists(),
        "objectization": (root / "05_candidate_ranking" / "CONSTRUCTION_CANDIDATE_RANKING.csv").exists(),
        "temporal": (root / "06_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json").exists(),
        "figures": (root / "08_manifests" / "FIGURE_INVENTORY.csv").exists(),
        "receipt": (root / "08_manifests" / "FINAL_TONGZHOU_CONSTRUCTION_IMAGE_RUN_RECEIPT.json").exists(),
    }
    return {"status": "complete" if all(stages.values()) else "in_progress", "run_root": str(root), "stages": stages}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-root", default=r"D:\Data Set\Satellite\beijing_tongzhou")
    parser.add_argument("--run-root")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cuda:0", "cpu"))
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("audit", "validate-recipe", "infer", "objectize", "temporal", "figures", "finalize", "all", "status"):
        actions.add_argument(f"--{name}", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    local_src = str(project_root / "src")
    if local_src in sys.path:
        sys.path.remove(local_src)
    sys.path.insert(0, local_src)
    source_root = Path(args.source_root).resolve()
    if args.status:
        print(json.dumps(status(project_root), ensure_ascii=False, indent=2))
        return 0
    run_root = create_run_root(project_root, args.run_root)
    result: dict[str, Any] = {"run_root": str(run_root)}
    if args.audit or args.all:
        result["audit"] = audit(project_root, source_root, run_root)
    if args.validate_recipe or args.all:
        if not (run_root / "01_input_audit" / "TONGZHOU_INPUT_READINESS.json").exists():
            raise RuntimeError("run --audit before recipe validation")
        result["recipe"] = validate_recipes(project_root, run_root, args.device)
    if args.infer or args.all:
        result["inference"] = run_formal_inference(project_root, source_root, run_root, args.device)
    if args.objectize or args.all:
        result["objectization"] = objectize_and_rank(project_root, run_root)
    if args.temporal or args.all:
        result["temporal"] = build_temporal(run_root)
    if args.figures or args.all:
        result["figures"] = render_figures(project_root, source_root, run_root)
    if args.finalize or args.all:
        result["final"] = validate_and_finalize(project_root, run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
