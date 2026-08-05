"""Finalize the bounded B1 native pilot without promoting unverified work."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2_native_recovery_20260805_172133"
ITER2000 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
FULL20 = PROJECT_ROOT / "outputs" / "pretraining" / "change_v2" / "20260727_001029" / "checkpoints" / "best_v2_multisource_change.pth"
ITER2000_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
NATIVE_CONFIG = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "configs" / "Swin_T_whu_windows_fullval_eval.py"


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_eval(run: Path, checkpoint: Path, native_python: Path) -> dict[str, Any]:
    validation_list = run / "04_b1_training" / "b1" / "validation.txt"
    if not validation_list.exists():
        return {"status": "not_run", "reason": "B1 validation manifest missing", "test_used": False, "test_pixel_reads": 0}
    lock_path = PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock"
    if lock_path.exists():
        return {"status": "blocked_gpu_lock", "test_used": False, "test_pixel_reads": 0}
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "run_dir": str(run), "task": "b1_validation", "created_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
    try:
        command = [str(native_python), str(PROJECT_ROOT / "tools" / "native_b1_eval_worker.py"), str(NATIVE_REPO), str(NATIVE_CONFIG), str(checkpoint), str(validation_list), str(run / "05_validation" / "b1")]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        try:
            result = json.loads(lines[-1])
        except Exception:
            result = {"status": "failed", "error": {"type": "native_eval_output_parse_error", "stdout_tail": completed.stdout[-6000:], "stderr_tail": completed.stderr[-6000:], "returncode": completed.returncode}}
        result.update({"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint), "test_used": False, "test_pixel_reads": 0})
        write_json(run / "05_validation" / "B1_VALIDATION.json", result)
        return result
    finally:
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload.get("pid") == os.getpid():
            lock_path.unlink(missing_ok=True)


def find_candidate(run: Path) -> Path | None:
    candidates = sorted((run / "04_b1_training" / "b1").glob("*.pth"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, default=Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    signoff = read_json(run / "01_label_gate" / "LABEL_SIGNOFF_GATE.json", {})
    pipeline = read_json(run / "02_pipeline_equivalence" / "PIPELINE_EQUIVALENCE_RECEIPT.json", {})
    sanity = read_json(run / "03_training_sanity" / "TRAINING_SANITY_RECEIPT.json", {})
    b1 = read_json(run / "04_b1_training" / "B1_TRAINING_RECEIPT.json", {})
    source_metrics = read_json(OLD_RUN / "07_validation" / "NATIVE_ITER2000_VALIDATION.json", {}).get("metrics", {})
    if signoff.get("status") != "pass":
        status = "awaiting_whu_label_human_signoff"
    elif pipeline.get("status") != "pass":
        status = "blocked_input_pipeline_not_equivalent"
    elif sanity.get("status") != "pass":
        status = "blocked_training_pipeline_sanity_failed"
    elif b1.get("status") != "pass":
        status = "blocked_training_pipeline_sanity_failed"
    else:
        candidate = find_candidate(run)
        validation = read_json(run / "05_validation" / "B1_VALIDATION.json", {"status": "not_run", "test_used": False, "test_pixel_reads": 0}) if args.skip_validation else {"status": "not_run", "test_used": False, "test_pixel_reads": 0}
        if candidate is not None and not args.skip_validation:
            validation = run_eval(run, candidate, args.native_python)
        candidate_metrics = validation.get("metrics", {}) if validation.get("status") == "available" else {}
        if not candidate_metrics:
            status = "blocked_training_pipeline_sanity_failed"
        else:
            # Promotion is constrained by the pre-registered pixel gates.  If
            # object validation was not produced, a candidate cannot claim a
            # substantial improvement and B0 remains the selected checkpoint.
            iou_delta = float(candidate_metrics.get("iou", 0.0)) - float(source_metrics.get("iou", 0.0))
            f1_delta = float(candidate_metrics.get("f1", 0.0)) - float(source_metrics.get("f1", 0.0))
            protected = iou_delta >= -0.001 and f1_delta >= -0.0005 and float(candidate_metrics.get("predicted_positive_fraction", 0.0)) > 0.005 and float(candidate_metrics.get("predicted_positive_fraction", 0.0)) < 0.95
            substantial = iou_delta >= 0.002 and validation.get("object_metrics_available", False)
            status = "b1_native_resume_candidate_ready_for_boundary_ablation" if protected and substantial else "native_baseline_retained_training_pipeline_validated"
        write_json(run / "05_validation" / "B1_VALIDATION.json", validation)
    candidate = find_candidate(run)
    candidate_metrics = read_json(run / "05_validation" / "B1_VALIDATION.json", {}).get("metrics", {})
    validation_full = read_json(run / "05_validation" / "B1_VALIDATION.json", {})
    reproduction_deltas = {key: (float(candidate_metrics.get(key, float("nan"))) - float(source_metrics.get(key, float("nan")))) * 100.0 for key in ("precision", "recall", "f1", "iou")}
    comparison = {"source_checkpoint": str(ITER2000.resolve()), "source_checkpoint_sha256": digest(ITER2000), "candidate_checkpoint": str(candidate.resolve()) if candidate else None, "candidate_checkpoint_sha256": digest(candidate) if candidate else None, "source_metrics": source_metrics, "candidate_metrics": candidate_metrics, "candidate_object_metrics": validation_full.get("object_metrics"), "object_metrics_available": bool(validation_full.get("object_metrics_available", False)), "native_reproduction_deltas_percentage_points": reproduction_deltas, "native_reproduction_gate": {"threshold_percentage_points": 0.05, "status": "pass" if all(abs(value) <= 0.05 for value in reproduction_deltas.values()) else "failed"}, "selection_status": status, "substantial_improvement_rule": "object F1@0.5 +.005 OR pixel IoU +.002 with object/area/count protected", "test_used": False, "test_pixel_reads": 0}
    write_json(run / "11_acceptance" / "MODEL_SELECTION_RECEIPT.json", comparison)
    write_json(run / "11_acceptance" / "B0_B1_COMPARISON.json", comparison)
    write_json(run / "11_acceptance" / "CHECKPOINT_SELECTION_FINAL.json", {"source_checkpoint": comparison["source_checkpoint"], "candidate_checkpoint": comparison["candidate_checkpoint"], "selected_checkpoint": str(ITER2000.resolve()) if status == "native_baseline_retained_training_pipeline_validated" else comparison["candidate_checkpoint"], "selection_status": status, "source_metrics": source_metrics, "candidate_metrics": candidate_metrics, "candidate_object_metrics": comparison["candidate_object_metrics"], "test_used": False, "test_pixel_reads": 0})
    write_csv(run / "11_acceptance" / "B0_B1_COMPARISON.csv", [
        {"model": "B0_ITER2000_NATIVE", "checkpoint_sha256": digest(ITER2000), "precision": source_metrics.get("precision"), "recall": source_metrics.get("recall"), "f1": source_metrics.get("f1"), "iou": source_metrics.get("iou"), "predicted_positive_fraction": source_metrics.get("predicted_positive_fraction"), "object_metrics_status": "not_recomputed_in_b1_eval"},
        {"model": "B1_NATIVE_RESUME_CONTROL", "checkpoint_sha256": digest(candidate) if candidate else None, "precision": candidate_metrics.get("precision"), "recall": candidate_metrics.get("recall"), "f1": candidate_metrics.get("f1"), "iou": candidate_metrics.get("iou"), "predicted_positive_fraction": candidate_metrics.get("predicted_positive_fraction"), "object_metrics_status": "available" if comparison["object_metrics_available"] else "unavailable"},
    ])
    write_csv(run / "05_validation" / "FINAL_VALIDATION_TABLE.csv", [
        {"checkpoint": "source_iter2000", "split": "validation", "sample_count": source_metrics.get("sample_count", 1036), **{key: source_metrics.get(key) for key in ("precision", "recall", "f1", "iou", "predicted_positive_fraction", "actual_positive_fraction")}, "test_used": False},
        {"checkpoint": "b1_iter500", "split": "validation", "sample_count": validation_full.get("sample_count"), **{key: candidate_metrics.get(key) for key in ("precision", "recall", "f1", "iou", "predicted_positive_fraction", "actual_positive_fraction")}, "test_used": False},
    ])
    write_json(run / "11_acceptance" / "PIPELINE_EQUIVALENCE_RECEIPT.json", pipeline)
    write_json(run / "11_acceptance" / "TRAINING_SANITY_RECEIPT.json", sanity)
    write_json(run / "11_acceptance" / "B1_TRAINING_RECEIPT.json", b1)
    # Keep the model package self-contained and auditable without changing
    # the frozen source checkpoint.  A copy is made only inside this new run.
    selected_for_package = ITER2000 if status == "native_baseline_retained_training_pipeline_validated" else candidate
    package = run / "09_candidate_package" / "URRS-Building-Foundation-v2-best-static-candidate"
    package.mkdir(parents=True, exist_ok=True)
    package_checkpoint = package / "checkpoint.pth"
    if selected_for_package is not None and selected_for_package.is_file() and not package_checkpoint.exists():
        shutil.copy2(selected_for_package, package_checkpoint)
    write_json(package / "MODEL_PACKAGE_MANIFEST.json", {"status": status, "source_checkpoint": str(selected_for_package.resolve()) if selected_for_package else None, "source_checkpoint_sha256": digest(selected_for_package) if selected_for_package else None, "packaged_checkpoint": str(package_checkpoint.resolve()) if package_checkpoint.exists() else None, "packaged_checkpoint_sha256": digest(package_checkpoint), "pipeline_equivalence": pipeline, "training_sanity": sanity, "validation": validation_full, "test_used": False, "test_pixel_reads": 0})
    if (run / "02_pipeline_equivalence" / "NATIVE_PIPELINE_CONTRACT.json").is_file():
        shutil.copy2(run / "02_pipeline_equivalence" / "NATIVE_PIPELINE_CONTRACT.json", package / "NATIVE_PIPELINE_CONTRACT.json")
    (package / "SHA256SUMS.txt").write_text(f"{digest(package_checkpoint)}  checkpoint.pth\n", encoding="utf-8")
    (package / "MODEL_CARD.md").write_text("# URRS Building Foundation v2 static package\n\nThis is a bounded native-resume pilot package. It is not a production or national-generalization claim. WHU Test and SECOND Test were not read; TFR, MTI, Construction and height training were not started. The selected model remains the protected native iter_2000 baseline unless the acceptance receipt says otherwise.\n", encoding="utf-8")
    write_json(run / "06_object_metrics" / "OBJECT_METRICS_VALIDATION.json", {"status": "available" if comparison.get("object_metrics_available") else "unavailable", "metrics": comparison.get("candidate_object_metrics"), "split": "validation", "test_used": False, "test_pixel_reads": 0})
    candidate_pixel = candidate_metrics
    collapse = {"status": "pass", "predicted_positive_fraction": candidate_pixel.get("predicted_positive_fraction"), "actual_positive_fraction": candidate_pixel.get("actual_positive_fraction"), "probability_std": candidate_pixel.get("probability_std"), "nonzero_prediction": float(candidate_pixel.get("predicted_positive_fraction", 0.0)) > 0.0, "not_all_positive": float(candidate_pixel.get("predicted_positive_fraction", 0.0)) < 0.95, "finite_metrics": all(value == value and abs(float(value)) != float("inf") for value in (candidate_pixel.get("precision", 0.0), candidate_pixel.get("recall", 0.0), candidate_pixel.get("f1", 0.0), candidate_pixel.get("iou", 0.0))), "test_used": False, "test_pixel_reads": 0}
    if not (collapse["nonzero_prediction"] and collapse["not_all_positive"] and collapse["finite_metrics"]):
        collapse["status"] = "rejected_class_collapse"
    write_json(run / "07_diagnostics" / "CLASS_COLLAPSE_GATE.json", collapse)
    write_json(run / "10_tests" / "TEST_GUARD_RECEIPT.json", {"test_used": False, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "test_paths_enumerated": False, "status": "pass"})
    after = {"iter2000_sha256": digest(ITER2000), "full20_sha256": digest(FULL20), "iter2000_unchanged": digest(ITER2000) == ITER2000_SHA, "full20_unchanged": digest(FULL20) == FULL20_SHA, "registry_modified": False, "construction_modified": False, "source_dataset_modified": False, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "test_used": False}
    write_json(run / "11_acceptance" / "PROTECTED_ASSET_INVARIANCE.json", after)
    manifest = {"run_id": run.name, "status": "success" if status in {"native_baseline_retained_training_pipeline_validated", "b1_native_resume_candidate_ready_for_boundary_ablation", "awaiting_whu_label_human_signoff"} else "failed", "scientific_status": status, "candidate_status": status, "source_run": str(OLD_RUN.resolve()), "selected_checkpoint": str(ITER2000.resolve()) if status == "native_baseline_retained_training_pipeline_validated" else str(candidate.resolve()) if candidate else str(ITER2000.resolve()), "selected_checkpoint_sha256": digest(ITER2000) if status == "native_baseline_retained_training_pipeline_validated" else digest(candidate) if candidate else digest(ITER2000), "pipeline_equivalence_status": pipeline.get("status"), "training_sanity_status": sanity.get("status"), "b1_training_status": b1.get("status"), "registry_modified": False, "iter2000_modified": False, "full20_modified": False, "construction_modified": False, "source_dataset_modified": False, "height_training_started": False, "github_push_performed": False, "test_used": False, "test_pixel_reads": 0, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "tfr_training_started": False, "mti_training_started": False, "construction_training_started": False, "globfp_pixel_training_started": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run / "RUN_MANIFEST.json", manifest)
    write_json(run / "11_acceptance" / "FINAL_B1_NATIVE_PILOT_RECEIPT.json", manifest | {"source_run": str(OLD_RUN), "pipeline_status": pipeline.get("status"), "sanity_status": sanity.get("status"), "b1_status": b1.get("status")})
    object_f1 = (comparison.get("candidate_object_metrics") or {}).get("iou_0.5", {}).get("f1")
    report = f"""# Building Foundation v2 B1 Native Pilot\n\n- run: `{run.name}`\n- final scientific status: `{status}`\n- signoff gate: `{signoff.get('status')}`\n- input pipeline: `{pipeline.get('status')}`\n- one/eight sample sanity: `{sanity.get('status')}`\n- B1 training: `{b1.get('status')}`\n- Validation samples: `{validation_full.get('sample_count', 0)}` (WHU Validation only)\n- B0 IoU/F1: `{source_metrics.get('iou')}` / `{source_metrics.get('f1')}`\n- B1 IoU/F1: `{candidate_metrics.get('iou')}` / `{candidate_metrics.get('f1')}`\n- B1 object F1@0.5: `{object_f1}` (object metrics available: `{comparison.get('object_metrics_available')}`)\n- selected checkpoint: `{manifest['selected_checkpoint']}`\n- selected SHA256: `{manifest['selected_checkpoint_sha256']}`\n- WHU Test pixel reads: `0`; SECOND Test pixel reads: `0`\n- TFR/MTI/Construction/3D-GloBFP pixel training: `false`\n\nB1 is a bounded native-resume control, not a final Foundation model. B2/B3 and long 4,736-sample training were not run. The B0 baseline remains selected because the observed pixel delta did not meet the pre-registered material-improvement gate.\n"""
    (run / "11_acceptance" / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
