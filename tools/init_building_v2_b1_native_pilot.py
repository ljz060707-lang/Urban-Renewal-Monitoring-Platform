"""Create and gate an isolated B1 native-resume pilot run.

This command is intentionally conservative.  It imports the prior recovery
evidence, verifies the human label sign-off, and compares the project adapter
with the *validation* portion of the native preprocessing contract.  It never
walks a WHU/SECOND Test directory and never edits the source recovery run.
Training is a separate command so a failed input gate cannot accidentally
start a GPU job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2_native_recovery_20260805_172133"
ITER2000 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
FULL20 = PROJECT_ROOT / "outputs" / "pretraining" / "change_v2" / "20260727_001029" / "checkpoints" / "best_v2_multisource_change.pth"
WHU_ROOT = Path(r"D:\Data Set\WHU_Building")
ITER2000_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
REQUIRED_SIGNOFF = {
    "review_status": "approved",
    "preview_reviewed": True,
    "label_polarity": "nonzero_is_building",
    "building_values": [1],
    "background_values": [0],
    "systematic_offset_found": False,
    "polarity_inversion_found": False,
    "label_alignment_accepted": True,
    "formal_training_label_gate": True,
}


def sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return {"branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD"), "status_short": run("status", "--short").splitlines(), "diff_stat": run("diff", "--stat").splitlines()}


def new_run(run_dir: Path | None) -> Path:
    parent = PROJECT_ROOT / "outputs" / "pretraining"
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = parent / f"building_foundation_v2_b1_native_pilot_{stamp}"
        suffix = 1
        while run_dir.exists():
            run_dir = parent / f"building_foundation_v2_b1_native_pilot_{stamp}_{suffix:02d}"
            suffix += 1
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("00_governance", "01_label_gate", "02_pipeline_equivalence", "03_training_sanity", "04_b1_training", "05_validation", "06_object_metrics", "07_diagnostics", "08_figures", "09_candidate_package", "10_tests", "11_acceptance"):
        (run_dir / name).mkdir()
    return run_dir


def validate_signoff(path: Path) -> tuple[bool, dict[str, Any], list[str]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return False, {"status": "missing_or_invalid", "source": str(path)}, ["signoff_file_missing_or_not_object"]
    errors: list[str] = []
    for key, expected in REQUIRED_SIGNOFF.items():
        if payload.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {payload.get(key)!r}")
    # Do not infer approval from reviewer notes or from the old audit status.
    return not errors, payload, errors


def fixed_sample_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, folder in (("train", "train"), ("validation", "val")):
        image_dir = WHU_ROOT / folder / "image"
        label_dir = WHU_ROOT / folder / "label"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        images = {path.stem: path for path in image_dir.glob("*.tif")}
        labels = {path.stem: path for path in label_dir.glob("*.tif")}
        # Explicitly never inspect a Test path; only these two known folders.
        for stem in sorted(set(images) & set(labels)):
            rows.append({"sample_id": f"whu_{split}_{stem}", "split": split, "image_path": str(images[stem].resolve()), "label_path": str(labels[stem].resolve())})
    selected = [row for row in rows if row["split"] == "train"][:32] + [row for row in rows if row["split"] == "validation"][:32]
    return selected


def write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_path", "label_path", "test_pixel_read"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "test_pixel_read": False})


def native_contract(old: Path) -> dict[str, Any]:
    audit = read_json(old / "02_native_recovery" / "NATIVE_CONFIG_AUDIT.json", {})
    pre = audit.get("data_preprocessor", {})
    return {
        "source_config": audit.get("config_path"),
        "dataset_class": "opencd.datasets.FoundationDataset",
        "data_prefix": {"train": audit.get("train_data_list"), "validation": audit.get("val_data_list"), "test": "registered_only_not_read"},
        "loader": {"channels": 6, "bgr_to_rgb": bool(pre.get("bgr_to_rgb", True)), "dtype": "float32", "size_divisor": pre.get("size_divisor", 32)},
        "channel_order": "BGR source converted to RGB, repeated for two identical observations",
        "mean": pre.get("mean"),
        "std": pre.get("std"),
        "resize_crop_padding": {"validation": "native full 512x512 image; size_divisor padding only when required", "train": "native RandomMosaic/RandomCrop 512, recorded but not used by equivalence comparison", "pad_val": pre.get("pad_val", 0), "seg_pad_val": pre.get("seg_pad_val", 255)},
        "interpolation": {"image": "native loader/OpenCV", "label": "nearest/native annotation loader"},
        "label_mapping": {"local_whu_values": {"0": 0, "1": 1}, "building_values": [1], "background_values": [0], "ignore_index": 255, "reduce_zero_label": False},
        "preprocessor": pre,
        "formal_test_access": "forbidden",
    }


def compare_pipeline(run: Path, rows: list[dict[str, str]], contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from urban_monitor.building_foundation_v2.data import WHUBuildingDataset
    import rasterio

    native_dir, project_dir, preview_dir = (run / "02_pipeline_equivalence" / name for name in ("native_samples", "project_samples", "INPUT_PIPELINE_PREVIEWS"))
    native_dir.mkdir(exist_ok=True); project_dir.mkdir(exist_ok=True); preview_dir.mkdir(exist_ok=True)
    dataset_rows = {row["sample_id"]: row for row in rows}
    dataset = WHUBuildingDataset(rows, image_size=512, train=False, label_contract=contract_path, native_normalization=True, duplicate_temporal_input=True)
    result_rows: list[dict[str, Any]] = []
    from PIL import Image, ImageDraw
    for index, row in enumerate(rows):
        project_item = dataset[index]
        with rasterio.open(row["image_path"]) as image_ds:
            raw = image_ds.read(out_shape=(3, 512, 512), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
        with rasterio.open(row["label_path"]) as label_ds:
            raw_label = label_ds.read(1, out_shape=(512, 512), resampling=rasterio.enums.Resampling.nearest)
        native_image = raw[::-1].copy()
        native_image = np.concatenate([native_image, native_image], axis=0)
        mean = np.asarray(contract["native_mean"], dtype="float32")[:, None, None] if contract.get("native_mean") else np.asarray(contract["mean"], dtype="float32")[:, None, None]
        std = np.asarray(contract["native_std"], dtype="float32")[:, None, None] if contract.get("native_std") else np.asarray(contract["std"], dtype="float32")[:, None, None]
        native_image = (native_image - mean) / np.maximum(std, 1e-6)
        native_mask = np.isin(raw_label, [1]).astype("float32")
        project_image = project_item["image"].detach().cpu().numpy()
        project_mask = project_item["mask"].detach().cpu().numpy()[0]
        np.save(native_dir / f"{index:03d}_{row['sample_id']}.npy", native_image)
        np.save(project_dir / f"{index:03d}_{row['sample_id']}.npy", project_image)
        image_diff = np.abs(native_image - project_image)
        mask_diff = np.abs(native_mask - project_mask)
        image_max = float(image_diff.max()) if image_diff.size else 0.0
        image_mean = float(image_diff.mean()) if image_diff.size else 0.0
        rmse = float(np.sqrt(np.mean(np.square(image_diff)))) if image_diff.size else 0.0
        record = {"sample_id": row["sample_id"], "split": row["split"], "shape_equal": list(native_image.shape) == list(project_image.shape), "dtype_equal": str(native_image.dtype) == str(project_image.dtype), "channel_equal": native_image.shape[0] == 6 and project_image.shape[0] == 6, "normalization_equal": image_max <= 1e-4 and image_mean <= 1e-6, "image_max_abs_diff": image_max, "image_mean_abs_diff": image_mean, "image_rmse": rmse, "mask_shape_equal": list(native_mask.shape) == list(project_mask.shape), "mask_disagreement_pixels": int(np.count_nonzero(mask_diff)), "native_positive_fraction": float(native_mask.mean()), "project_positive_fraction": float(project_mask.mean()), "test_pixel_read": False}
        result_rows.append(record)
        if index < 8:
            # Inverse-normalize the first RGB copy for an auditable visual
            # comparison; this is a diagnostic image, never a label source.
            disp = np.transpose(native_image[:3] * std[:3] + mean[:3], (1, 2, 0))
            disp = np.clip(disp / 255.0, 0, 1)
            rgb = (disp * 255).astype("uint8")
            canvas = Image.new("RGB", (1536, 512), "white")
            canvas.paste(Image.fromarray(rgb), (0, 0))
            canvas.paste(Image.fromarray((project_mask * 255).astype("uint8")).convert("RGB"), (512, 0))
            canvas.paste(Image.fromarray((native_mask * 255).astype("uint8")).convert("RGB"), (1024, 0))
            draw = ImageDraw.Draw(canvas); draw.text((10, 10), "raw/native image", fill=(255, 0, 0)); draw.text((522, 10), "project mask", fill=(255, 0, 0)); draw.text((1034, 10), "native mask", fill=(255, 0, 0))
            canvas.save(preview_dir / f"{index:03d}_{row['sample_id']}.png")
    with (run / "02_pipeline_equivalence" / "INPUT_PIPELINE_EQUIVALENCE.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]) if result_rows else ["sample_id"]); writer.writeheader(); writer.writerows(result_rows)
    passed = all(row["shape_equal"] and row["channel_equal"] and row["normalization_equal"] and row["mask_disagreement_pixels"] == 0 and row["image_max_abs_diff"] <= 1e-4 and row["image_mean_abs_diff"] <= 1e-6 for row in result_rows)
    summary = {"status": "pass" if passed else "blocked_input_pipeline_not_equivalent", "sample_count": len(result_rows), "train_samples": sum(row["split"] == "train" for row in result_rows), "validation_samples": sum(row["split"] == "validation" for row in result_rows), "gates": {"shape": all(row["shape_equal"] for row in result_rows), "channel": all(row["channel_equal"] for row in result_rows), "normalization": all(row["normalization_equal"] for row in result_rows), "mask_disagreement_zero": all(row["mask_disagreement_pixels"] == 0 for row in result_rows), "image_max_abs_diff_le_1e4": all(row["image_max_abs_diff"] <= 1e-4 for row in result_rows), "image_mean_abs_diff_le_1e6": all(row["image_mean_abs_diff"] <= 1e-6 for row in result_rows)}, "tolerances": {"image_max_abs_diff": 1e-4, "image_mean_abs_diff": 1e-6}, "native_contract": contract, "test_pixel_reads": 0, "test_used": False}
    write_json(run / "02_pipeline_equivalence" / "INPUT_PIPELINE_EQUIVALENCE.json", summary)
    write_json(run / "02_pipeline_equivalence" / "INPUT_PIPELINE_DIFFERENCES.json", {"rows_with_difference": [row for row in result_rows if row["image_max_abs_diff"] > 1e-4 or row["mask_disagreement_pixels"]], "test_pixel_reads": 0})
    return summary


def create_governance(run: Path, native_python: Path) -> dict[str, Any]:
    full_candidates = [FULL20] + sorted((PROJECT_ROOT / "outputs" / "pretraining" / "change_v2").glob("*/checkpoints/best_v2_multisource_change.pth"))
    full = next((path for path in full_candidates if path.is_file()), None)
    registry = sorted((PROJECT_ROOT / "outputs" / "governance").glob("*/01_model_registry/CURRENT_BEST_MODEL_REGISTRY.json"))
    registry_path = registry[-1] if registry else None
    assets = {
        "iter2000": {"path": str(ITER2000.resolve()), "sha256": sha256(ITER2000), "expected_sha256": ITER2000_SHA},
        "full20": {"path": str(full.resolve()) if full else None, "sha256": sha256(full), "expected_sha256": FULL20_SHA},
        "current_best_registry": {"path": str(registry_path.resolve()) if registry_path else None, "sha256": sha256(registry_path)},
        "whu_train_validation": {"root": str(WHU_ROOT.resolve()), "registered_splits": ["train", "validation"], "pixel_reads": "train/validation only"},
        "whu_test": {"root": str((WHU_ROOT / "test").resolve()), "registered_only": True, "pixel_reads": 0},
        "second_test": {"registered_only": True, "pixel_reads": 0},
        "globfp_source": {"root": str(Path(r"D:\Data Set\3D-GloBFP").resolve()), "pixel_training": False},
        "nanjing_tongzhou": {"diagnostic_only_after_candidate_freeze": True},
        "old_recovery_run": {"path": str(OLD_RUN.resolve()), "read_only": True, "sha256": sha256(OLD_RUN / "RUN_MANIFEST.json")},
    }
    payload = {"run_id": run.name, "created_at_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(PROJECT_ROOT.resolve()), "git": git_snapshot(), "protected_assets": assets, "runtime": {"native_python": str(native_python), "native_python_exists": native_python.exists(), "gpu_lock": str((PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock").resolve())}, "flags": {"registry_modified": False, "iter2000_modified": False, "full20_modified": False, "construction_modified": False, "source_dataset_modified": False, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "tfr_training_started": False, "mti_training_started": False, "construction_training_started": False, "globfp_pixel_training_started": False, "height_training_started": False, "github_push_performed": False}, "scope": {"dataset": "WHU_Building", "train_count": 4736, "validation_count": 1036, "test": "registered_only_not_read", "candidate": "B1_NATIVE_RESUME_CONTROL"}, "status": "governance_created"}
    write_json(run / "00_governance" / "RUN_GOVERNANCE.json", payload)
    write_json(run / "00_governance" / "PROTECTED_ASSET_INVENTORY.json", assets)
    write_json(run / "00_governance" / "RUNTIME_AUDIT.json", payload["runtime"] | {"created_at_utc": payload["created_at_utc"]})
    write_json(run / "RUN_MANIFEST.json", {"run_id": run.name, "status": "in_progress", "scientific_status": "native_iter2000_baseline_reproduced_pending_training_pipeline_validation", "test_used": False, "test_pixel_reads": 0, "created_at_utc": payload["created_at_utc"]})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-run", type=Path, default=OLD_RUN)
    parser.add_argument("--native-python", type=Path, default=Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"))
    parser.add_argument("--skip-equivalence", action="store_true", help="only for dry governance tests; leaves formal_training_allowed=false")
    args = parser.parse_args()
    run = new_run(args.run_dir)
    source = args.source_run.resolve()
    governance = create_governance(run, args.native_python)
    source_signoff = source / "01_label_contract" / "WHU_LABEL_HUMAN_SIGNOFF.json"
    valid, signoff, errors = validate_signoff(source_signoff)
    write_json(run / "01_label_gate" / "LABEL_SIGNOFF_GATE.json", {"status": "pass" if valid else "awaiting_whu_label_human_signoff", "source_signoff": str(source_signoff), "signoff_imported_read_only": True, "checks": REQUIRED_SIGNOFF, "errors": errors, "test_pixel_reads": 0, "test_used": False})
    effective = {"status": "approved" if valid else "blocked", "source_signoff": str(source_signoff), "label_polarity": "nonzero_is_building", "building_values": [1], "background_values": [0], "ignore_values": [], "label_contract": {"label_polarity": "nonzero_is_building", "building_values": [1], "background_values": [0], "ignore_values": []}, "human_signoff": signoff if valid else None, "formal_training_label_gate": valid, "test_pixel_reads": 0, "test_used": False}
    write_json(run / "01_label_gate" / "LABEL_CONTRACT_EFFECTIVE.json", effective)
    write_json(run / "00_governance" / "SCIENTIFIC_STATUS_CORRECTION.json", {"old_run": str(source), "old_status_not_inherited": True, "new_status": "native_iter2000_baseline_reproduced_pending_training_pipeline_validation", "reason": "new isolated B1 pilot has independent gates", "test_pixel_reads": 0})
    old_contract = read_json(source / "02_native_recovery" / "NATIVE_CONFIG_AUDIT.json", {})
    contract = native_contract(source)
    # Include the exact values consumed by the project adapter.  The source
    # audit stores these under data_preprocessor; keeping both forms makes the
    # contract self-contained for later re-runs.
    contract["native_mean"] = contract.get("mean")
    contract["native_std"] = contract.get("std")
    write_json(run / "02_pipeline_equivalence" / "NATIVE_PIPELINE_CONTRACT.json", contract)
    rows = fixed_sample_rows()
    write_registry(run / "02_pipeline_equivalence" / "PIPELINE_EQUIVALENCE_SAMPLE_REGISTRY.csv", rows)
    if not valid:
        summary = {"status": "awaiting_whu_label_human_signoff", "formal_training_allowed": False, "sample_count": 0, "test_pixel_reads": 0, "test_used": False, "reason": "human label signoff is a hard gate; no pixel equivalence or GPU work started"}
    elif args.skip_equivalence:
        summary = {"status": "not_run", "formal_training_allowed": False, "sample_count": 0, "test_pixel_reads": 0, "test_used": False, "reason": "explicit dry governance mode"}
    else:
        summary = compare_pipeline(run, rows, contract, run / "01_label_gate" / "LABEL_CONTRACT_EFFECTIVE.json")
    write_json(run / "02_pipeline_equivalence" / "PIPELINE_EQUIVALENCE_RECEIPT.json", {**summary, "native_config_audit": str((source / "02_native_recovery" / "NATIVE_CONFIG_AUDIT.json").resolve()), "source_run": str(source), "test_pixel_reads": 0, "test_used": False})
    formal_allowed = valid and summary.get("status") == "pass"
    write_json(run / "03_training_sanity" / "TRAINING_SANITY_RECEIPT.json", {"status": "ready" if formal_allowed else "not_started", "one_sample": "not_started", "eight_sample": "not_started", "reason": None if formal_allowed else "pipeline/signoff gate not passed", "test_pixel_reads": 0, "test_used": False})
    write_json(run / "04_b1_training" / "B1_TRAINING_RECEIPT.json", {"status": "not_started", "candidate": "B1_NATIVE_RESUME_CONTROL", "initialization_checkpoint": str(ITER2000.resolve()), "initialization_checkpoint_sha256": sha256(ITER2000), "formal_training_allowed": formal_allowed, "reason": "sanity runner is a separate explicit stage", "test_pixel_reads": 0, "test_used": False})
    write_json(run / "06_object_metrics" / "OBJECT_METRICS_CONTRACT.json", {"status": "implemented", "matching": "Hungarian one-to-one max IoU", "thresholds": [0.3, 0.5, 0.7], "size_bands": "fixed train quantiles/recorded thresholds", "test_pixel_reads": 0})
    write_json(run / "06_object_metrics" / "CANDIDATE_MATRIX.json", {"B0_ITER2000_NATIVE": {"status": "available_read_only", "checkpoint_sha256": sha256(ITER2000)}, "B1_NATIVE_RESUME_CONTROL": {"status": "ready_for_sanity" if formal_allowed else "blocked", "reason": None if formal_allowed else "awaiting_whu_label_human_signoff_or_pipeline_equivalence"}, "B2_BOUNDARY_AUX": {"status": "not_allowed_this_round"}, "B3_GEOMETRY_AUX": {"status": "not_allowed_this_round"}, "H1_HEIGHT": {"status": "not_allowed_3d_globfp_has_no_image_pairs"}, "TFR": {"status": "not_started"}, "MTI": {"status": "not_started"}, "Construction": {"status": "not_started"}, "test_pixel_reads": 0, "test_used": False})
    write_json(run / "11_acceptance" / "PROTECTED_ASSET_INVARIANCE.json", {"status": "before_run_snapshot", "assets": governance["protected_assets"], "test_pixel_reads": 0, "test_used": False})
    final_status = "native_baseline_retained_training_pipeline_validated" if formal_allowed and args.skip_equivalence else ("blocked_input_pipeline_not_equivalent" if valid and summary.get("status") == "blocked_input_pipeline_not_equivalent" else "awaiting_whu_label_human_signoff" if not valid else "native_iter2000_baseline_reproduced_pending_training_pipeline_validation")
    print(json.dumps({"run_dir": str(run), "status": final_status, "label_signoff": "pass" if valid else "awaiting", "pipeline": summary.get("status"), "formal_training_allowed": formal_allowed, "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
    return 0 if valid and summary.get("status") == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
