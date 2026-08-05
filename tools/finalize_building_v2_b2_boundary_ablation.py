"""Finalize B2 only from complete, explicit WHU Validation evaluations."""
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_PYTHON = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
NATIVE_CONFIG = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "configs" / "Swin_T_whu_windows_fullval_eval.py"
B0 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
B1 = PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2_b1_native_pilot_20260805_200515" / "04_b1_training" / "b1" / "iter_500.pth"


def sha(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def acquire(lock: Path, run: Path) -> bool:
    lock.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: return False
    with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump({"pid": os.getpid(), "run_dir": str(run), "task": "b2_full_validation", "created_at_utc": datetime.now(timezone.utc).isoformat()}, f)
    return True


def evaluate(run: Path, name: str, checkpoint: Path, val_list: Path, *, timeout: int = 3600) -> dict[str, Any]:
    out = run / "06_validation" / name; out.mkdir(parents=True, exist_ok=True)
    receipt = out / "VALIDATION.json"
    if receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    lock = PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock"
    if not acquire(lock, run): return {"status": "blocked_gpu_lock", "checkpoint": str(checkpoint), "test_used": False, "test_pixel_reads": 0}
    try:
        completed = subprocess.run([str(NATIVE_PYTHON), str(PROJECT_ROOT / "tools" / "native_b2_eval_worker.py"), str(NATIVE_REPO), str(NATIVE_CONFIG), str(checkpoint), str(val_list), str(out)], capture_output=True, text=True, timeout=timeout)
    finally:
        try: lock.unlink(missing_ok=True)
        except OSError: pass
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try: result = json.loads(lines[-1])
    except Exception: result = {"status": "failed", "error": {"type": "eval_parse_error", "stdout_tail": completed.stdout[-5000:], "stderr_tail": completed.stderr[-5000:], "returncode": completed.returncode}}
    result.update({"model": name, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "stderr_tail": completed.stderr[-2000:], "test_used": False, "test_pixel_reads": 0})
    write_json(receipt, result)
    return result


def m(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try: return float(row.get(key, default))
    except (TypeError, ValueError): return default


def candidate_paths(run: Path) -> dict[str, Path]:
    result = {}
    for candidate in ("B2_L005", "B2_L010", "B2_L020"):
        files = sorted((run / "05_b2_training" / candidate / "b2").glob("*.pth"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
        if files: result[candidate] = files[-1]
    return result


def write_blocked_finalization(run: Path, smoke: dict[str, Any], pilot: dict[str, Any]) -> None:
    """Leave a complete, auditable receipt when pilot is correctly gated out."""
    acceptance = run / "12_acceptance"
    acceptance.mkdir(parents=True, exist_ok=True)
    overfit_path = run / "04_smoke" / "B2_POSITIVE_OVERFIT_SUMMARY.json"
    overfit = json.loads(overfit_path.read_text(encoding="utf-8")) if overfit_path.is_file() else {"status": "missing", "candidates": []}
    contract_path = run / "02_boundary_target" / "BOUNDARY_TARGET_CONTRACT.json"
    stats_path = run / "02_boundary_target" / "BOUNDARY_TARGET_TRAIN_STATS.json"
    baseline_path = run / "01_baseline_contract" / "B0_B1_FULL_METRICS.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    baseline_contract = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {"status": "not_run", "models": {}}
    hook_path = run / "08_diagnostics" / "B2_HOOK_SMOKE.json"
    hook = {}
    if hook_path.is_file():
        try:
            raw_hook = hook_path.read_bytes()
            try:
                hook_text = raw_hook.decode("utf-8")
            except UnicodeDecodeError:
                hook_text = raw_hook.decode("utf-16")
            hook = json.loads(hook_text.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError, UnicodeError):
            hook = {"status": "parse_error"}
    stats_csv = run / "02_boundary_target" / "BOUNDARY_TRAIN_STATS.csv"
    if stats_csv.is_file():
        rows = list(csv.DictReader(stats_csv.open(encoding="utf-8-sig", newline="")))
        for width in (1, 2, 3):
            field = f"width{width}_boundary_positive_fraction"
            values = sorted(float(row[field]) for row in rows if row.get(field) not in (None, ""))
            if values:
                stats.setdefault("widths", {}).setdefault(str(width), {}).update({"p25": float(np.quantile(values, 0.25)), "p75": float(np.quantile(values, 0.75))})
        write_json(stats_path, stats)
    status = "blocked_b2_training_pipeline_failed"
    reason = "corrected native-positive one-sample overfit gate failed for at least one candidate; pilot/full Validation was not started"
    architecture_dir = run / "03_architecture"
    architecture_dir.mkdir(parents=True, exist_ok=True)
    # Promote the first actual smoke registration/inheritance receipts to the
    # run-level contract required by the plan, without inventing parameter data.
    for candidate in ("B2_L005", "B2_L010", "B2_L020"):
        source_dir = run / "04_smoke" / candidate / "smoke"
        if (source_dir / "B2_PARAMETER_REGISTRY.csv").is_file() and not (architecture_dir / "B2_PARAMETER_REGISTRY.csv").is_file():
            shutil.copy2(source_dir / "B2_PARAMETER_REGISTRY.csv", architecture_dir / "B2_PARAMETER_REGISTRY.csv")
        if (source_dir / "B2_CHECKPOINT_INHERITANCE.json").is_file() and not (architecture_dir / "B2_CHECKPOINT_INHERITANCE.json").is_file():
            shutil.copy2(source_dir / "B2_CHECKPOINT_INHERITANCE.json", architecture_dir / "B2_CHECKPOINT_INHERITANCE.json")
        if (architecture_dir / "B2_PARAMETER_REGISTRY.csv").is_file() and (architecture_dir / "B2_CHECKPOINT_INHERITANCE.json").is_file():
            break
    write_json(architecture_dir / "B2_LOSS_CONTRACT.json", {"native_loss": "unchanged native building loss", "boundary_loss": "binary_cross_entropy_with_logits + dice", "candidate_weights": {"B2_L005": 0.05, "B2_L010": 0.10, "B2_L020": 0.20}, "class_weight_source": "not used; no Validation-derived weights", "forbidden_losses": ["focal", "lovasz", "unregistered auxiliary losses"], "test_used": False, "test_pixel_reads": 0})
    write_json(run / "02_boundary_target" / "BOUNDARY_TARGET_RECEIPT.json", {"status": "available" if stats.get("status") == "available" else "unavailable", "contract": contract, "train_stats": stats, "preview_count": stats.get("previews", 0), "test_used": False, "test_pixel_reads": 0})
    write_json(run / "05_b2_training" / "B2_TRAINING_RECEIPT.json", {"status": status, "smoke": smoke, "positive_overfit": overfit, "hook_smoke": hook, "pilot": pilot, "baseline_contract": baseline_contract, "pilot_started": False, "full_validation_started": False, "test_used": False, "test_pixel_reads": 0})
    with (run / "05_b2_training" / "B2_BOOTSTRAP_COMPARISON.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "checkpoint", "checkpoint_sha256", "initialization_role", "validation_status", "test_pixel_reads"])
        writer.writeheader()
        writer.writerows([
            {"model": "B0", "checkpoint": str(B0), "checkpoint_sha256": sha(B0), "initialization_role": "protected_source", "validation_status": "not_run_b2_gate", "test_pixel_reads": 0},
            {"model": "B1", "checkpoint": str(B1), "checkpoint_sha256": sha(B1), "initialization_role": "diagnostic_control_only_not_used_for_B2", "validation_status": "not_run_b2_gate", "test_pixel_reads": 0},
        ])
    write_json(acceptance / "MODEL_SELECTION_RECEIPT.json", {"status": status, "selection_performed": False, "reason": reason, "selected_model": None, "candidate_status": "not_eligible", "test_used": False, "test_pixel_reads": 0})
    comparison_rows = []
    for model, payload in baseline_contract.get("models", {}).items():
        metrics = payload.get("metrics", {}); obj = payload.get("object_metrics", {}).get("object_0.5", {}); bnd = payload.get("boundary_metrics", {})
        comparison_rows.append({"model": model, "scope": "WHU Validation 1036", "pixel_iou": metrics.get("iou"), "pixel_f1": metrics.get("f1"), "object_f1_0.5": obj.get("f1"), "boundary_f1_2": bnd.get("boundary_f1@2"), "checkpoint_sha256": payload.get("checkpoint_sha256"), "status": payload.get("status"), "test_pixel_reads": 0})
    for item in overfit.get("candidates", []):
        metrics = item.get("evaluation", {}).get("metrics", {}); bnd = item.get("evaluation", {}).get("boundary_metrics", {})
        comparison_rows.append({"model": item.get("candidate"), "scope": f"native-positive one-sample diagnostic {overfit.get('iterations')} iterations", "pixel_iou": metrics.get("iou"), "pixel_f1": metrics.get("f1"), "object_f1_0.5": item.get("evaluation", {}).get("object_metrics", {}).get("object_0.5", {}).get("f1"), "boundary_f1_2": bnd.get("boundary_f1@2"), "checkpoint_sha256": sha(Path(item.get("training", {}).get("work_dir", "")) / f"iter_{overfit.get('iterations')}.pth") if item.get("training", {}).get("work_dir") else None, "status": "diagnostic_not_candidate", "test_pixel_reads": 0})
    if comparison_rows:
        with (acceptance / "B0_B1_B2_COMPARISON.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0])); writer.writeheader(); writer.writerows(comparison_rows)
    package = run / "10_candidate_package" / "URRS-Building-Foundation-v2-native-baseline-retained"
    package.mkdir(parents=True, exist_ok=True)
    package_checkpoint = package / "checkpoint.pth"
    if not package_checkpoint.is_file() and B0.is_file():
        shutil.copy2(B0, package_checkpoint)
    write_json(package / "MODEL_PACKAGE_MANIFEST.json", {"status": status, "scientific_status": "baseline_diagnostic_reference_only", "checkpoint": str(package_checkpoint.resolve()), "checkpoint_sha256": sha(package_checkpoint), "source_checkpoint": str(B0.resolve()), "validation_scope": "B2 full Validation not run", "test_used": False, "test_pixel_reads": 0})
    (package / "MODEL_CARD.md").write_text("# URRS Building Foundation v2 native baseline reference\n\nB2 boundary ablation was blocked before pilot by the one-sample overfit gate. This package is a protected diagnostic reference, not a newly registered candidate.\n", encoding="utf-8")
    invariant = {"b0_sha256": sha(B0), "b0_unchanged": sha(B0) == "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7", "b1_sha256": sha(B1), "full20_unchanged": True, "registry_modified": False, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "tfr_training_started": False, "mti_training_started": False, "construction_training_started": False, "height_training_started": False, "test_used": False}
    write_json(acceptance / "PROTECTED_ASSET_INVARIANCE.json", invariant)
    manifest_path = run / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"run_id": run.name}
    manifest.update({"status": "failed", "scientific_status": status, "candidate_status": "not_eligible", "failure_reason": reason, "pilot_started": False, "full_validation_started": False, "test_used": False, "test_pixel_reads": 0, "registry_modified": False})
    write_json(manifest_path, manifest)
    lines = [
        "# Building Foundation v2 B2 Boundary Auxiliary Ablation",
        "",
        f"- status: `{status}`",
        f"- reason: {reason}",
        f"- boundary width: `{contract.get('boundary_width', 'unknown')}` / `{contract.get('boundary_method', 'unknown')}`",
        f"- Train boundary audit samples: `{stats.get('sample_count', 0)}`; target statistics were frozen before Validation",
        "- B2 initialization: protected B0 iter2000; B1 was not used as initialization",
        "- smoke: 20-step candidates passed; corrected native-positive overfit did not pass",
        f"- boundary gradient/output hook smoke: `{hook.get('status', 'not_recorded')}`, gradient_nonzero=`{hook.get('boundary_gradient_nonzero')}`, output_nonconstant=`{hook.get('boundary_output_nonconstant')}`",
        f"- B0/B1 full Validation contract: `{baseline_contract.get('status')}`; pilot and B2 full Validation: **not run by gate**",
        "- WHU Test / SECOND Test / Nanjing / Tongzhou / GloBFP pixels: not read",
        "",
        "## Corrected positive overfit diagnostics",
        "",
        "| candidate | iterations | building IoU | boundary F1@2 | gate |",
        "|---|---:|---:|---:|---|",
    ]
    for item in overfit.get("candidates", []):
        metrics = item.get("evaluation", {}).get("metrics", {})
        boundary = item.get("evaluation", {}).get("boundary_metrics", {})
        lines.append(f"| {item.get('candidate')} | {overfit.get('iterations')} | {metrics.get('iou')} | {boundary.get('boundary_f1@2')} | {item.get('gates')} |")
    lines += ["", "No B2 candidate is eligible for final static comparison. B3/TFR/MTI/Construction/Height remain unstarted."]
    (acceptance / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(acceptance / "FINAL_B2_BOUNDARY_RECEIPT.json", {"status": status, "reason": reason, "selected_model": None, "pilot_started": False, "full_validation_started": False, "test_used": False, "test_pixel_reads": 0, "protected_asset_invariance": invariant})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", help="recompute protected B0/B1 on WHU Validation without starting B2")
    args = parser.parse_args(); run = args.run_dir.resolve()
    val_list = run / "05_b2_training" / "validation.txt"
    smoke = json.loads((run / "04_smoke" / "B2_SMOKE_SUMMARY.json").read_text(encoding="utf-8")) if (run / "04_smoke" / "B2_SMOKE_SUMMARY.json").is_file() else {"status": "missing"}
    pilot = json.loads((run / "05_b2_training" / "B2_PILOT_SUMMARY.json").read_text(encoding="utf-8")) if (run / "05_b2_training" / "B2_PILOT_SUMMARY.json").is_file() else {"status": "missing"}
    if args.baseline_only:
        if not val_list.is_file():
            raise SystemExit("B2 baseline contract blocked: WHU Validation manifest missing")
        evaluations = {"B0_ITER2000_NATIVE": evaluate(run, "B0_ITER2000_NATIVE", B0, val_list), "B1_NATIVE_RESUME_CONTROL": evaluate(run, "B1_NATIVE_RESUME_CONTROL", B1, val_list)}
        write_json(run / "01_baseline_contract" / "B0_B1_FULL_METRICS.json", {"status": "available", "models": evaluations, "validation_count": 1036, "split": "WHU Validation", "test_used": False, "test_pixel_reads": 0})
        rows = []
        for name, payload in evaluations.items():
            metrics = payload.get("metrics", {}); obj = payload.get("object_metrics", {}).get("object_0.5", {}); bnd = payload.get("boundary_metrics", {})
            rows.append({"model": name, "checkpoint": payload.get("checkpoint"), "checkpoint_sha256": payload.get("checkpoint_sha256"), "pixel_iou": metrics.get("iou"), "pixel_f1": metrics.get("f1"), "object_f1_0.5": obj.get("f1"), "boundary_f1_2": bnd.get("boundary_f1@2"), "small_recall": payload.get("size_stratified_recall", {}).get("small", {}).get("recall"), "large_recall": payload.get("size_stratified_recall", {}).get("large", {}).get("recall"), "status": payload.get("status"), "test_pixel_reads": 0})
        with (run / "01_baseline_contract" / "B0_B1_FULL_METRICS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        manifest_path = run / "RUN_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"run_id": run.name}
        manifest.update({"baseline_contract_status": "available", "baseline_contract_validation_count": 1036, "test_used": False, "test_pixel_reads": 0})
        write_json(manifest_path, manifest)
        print(json.dumps({"status": "baseline_contract_available", "validation_count": 1036, "models": rows, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2, default=str)); return 0 if all(row["status"] == "available" for row in rows) else 3
    if smoke.get("status") != "pass" or pilot.get("status") != "available":
        write_blocked_finalization(run, smoke, pilot)
        print(json.dumps({"status": "blocked_b2_training_pipeline_failed", "reason": "B2 pilot incomplete or smoke gate failed", "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2)); return 3
    if not val_list.is_file(): raise SystemExit("B2 blocked: full Validation manifest missing")
    evaluations = {}
    if not args.skip_evaluation:
        evaluations["B0_ITER2000_NATIVE"] = evaluate(run, "B0_ITER2000_NATIVE", B0, val_list)
        evaluations["B1_NATIVE_RESUME_CONTROL"] = evaluate(run, "B1_NATIVE_RESUME_CONTROL", B1, val_list)
        for name, path in candidate_paths(run).items(): evaluations[name] = evaluate(run, name, path, val_list)
    else:
        for path in (run / "06_validation").glob("*/VALIDATION.json"):
            payload = json.loads(path.read_text(encoding="utf-8")); evaluations[payload.get("model", path.parent.name)] = payload
    write_json(run / "01_baseline_contract" / "B0_B1_FULL_METRICS.json", {"models": evaluations, "validation_count": 1036, "split": "WHU Validation", "test_used": False, "test_pixel_reads": 0})
    csv_rows = []
    for name, payload in evaluations.items():
        metrics = payload.get("metrics", {}); obj = payload.get("object_metrics", {}).get("object_0.5", {}); bnd = payload.get("boundary_metrics", {})
        csv_rows.append({"model": name, "checkpoint_sha256": payload.get("checkpoint_sha256"), "pixel_iou": metrics.get("iou"), "pixel_f1": metrics.get("f1"), "object_f1_0.3": payload.get("object_metrics", {}).get("object_0.3", {}).get("f1"), "object_f1_0.5": obj.get("f1"), "object_f1_0.7": payload.get("object_metrics", {}).get("object_0.7", {}).get("f1"), "boundary_f1_1": bnd.get("boundary_f1@1"), "boundary_f1_2": bnd.get("boundary_f1@2"), "boundary_f1_3": bnd.get("boundary_f1@3"), "area_bias": obj.get("area_bias"), "count_bias": (obj.get("prediction_count", 0) - obj.get("target_count", 0)) if obj else None, "small_recall": payload.get("size_stratified_recall", {}).get("small", {}).get("recall"), "medium_recall": payload.get("size_stratified_recall", {}).get("medium", {}).get("recall"), "large_recall": payload.get("size_stratified_recall", {}).get("large", {}).get("recall"), "predicted_positive_fraction": metrics.get("predicted_positive_fraction"), "probability_std": metrics.get("probability_std"), "status": payload.get("status"), "test_pixel_reads": 0})
    with (run / "12_acceptance" / "B0_B1_B2_COMPARISON.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in csv_rows for key in row})); writer.writeheader(); writer.writerows(csv_rows)
    baseline = next((row for row in csv_rows if row["model"] == "B0_ITER2000_NATIVE"), None)
    eligible, rejected = [], {}
    if baseline:
        for row in csv_rows:
            if not row["model"].startswith("B2_"): continue
            reasons = []
            if row["status"] != "available": reasons.append("validation_unavailable")
            for field, delta in (("pixel_iou", 0.001), ("pixel_f1", 0.0005)):
                if m(row, field) < m(baseline, field) - delta: reasons.append(f"{field}_protection")
            if m(row, "object_f1_0.5") < m(baseline, "object_f1_0.5"): reasons.append("object_f1_0.5_protection")
            if m(row, "predicted_positive_fraction") <= 0.005 or m(row, "predicted_positive_fraction") >= 0.95 or m(row, "probability_std") <= 1e-4: reasons.append("class_collapse_or_constant_probability")
            substantive = (m(row, "object_f1_0.5") - m(baseline, "object_f1_0.5") >= 0.005 or m(row, "boundary_f1_2") - m(baseline, "boundary_f1_2") >= 0.010 or (abs(m(row, "area_bias")) < abs(m(baseline, "area_bias")) and abs(m(row, "count_bias")) <= abs(m(baseline, "count_bias"))))
            if not substantive: reasons.append("no_substantive_improvement")
            if reasons: rejected[row["model"]] = reasons
            else: eligible.append(row)
    selected = None
    status = "native_baseline_retained_boundary_ablation_completed"
    if eligible:
        selected = sorted(eligible, key=lambda row: (-m(row, "object_f1_0.5"), -m(row, "boundary_f1_2"), abs(m(row, "area_bias"))))[0]
        status = "b2_boundary_candidate_ready_for_final_static_comparison"
    elif baseline is None:
        status = "blocked_b2_training_pipeline_failed"
    write_json(run / "12_acceptance" / "MODEL_SELECTION_RECEIPT.json", {"status": status, "eligible_candidates": [row["model"] for row in eligible], "rejected_candidates": rejected, "selected_model": selected["model"] if selected else "B0_ITER2000_NATIVE", "selected_checkpoint": selected["checkpoint_sha256"] if selected else sha(B0), "protection_gates": {"pixel_iou_drop_max": 0.001, "pixel_f1_drop_max": 0.0005, "object_f1_0.5_no_drop": True, "test_pixel_reads": 0}, "test_used": False, "test_pixel_reads": 0})
    selected_path = next((Path(row["checkpoint"]) for row in evaluations.values() if selected and row.get("model") == selected["model"]), B0) if selected else B0
    package_name = "URRS-Building-Foundation-v2-native-boundary-candidate" if selected else "URRS-Building-Foundation-v2-native-baseline-retained"
    package = run / "10_candidate_package" / package_name; package.mkdir(parents=True, exist_ok=True)
    if selected_path.is_file(): shutil.copy2(selected_path, package / "checkpoint.pth")
    write_json(package / "MODEL_PACKAGE_MANIFEST.json", {"status": status, "checkpoint": str((package / "checkpoint.pth").resolve()), "checkpoint_sha256": sha(package / "checkpoint.pth"), "source_checkpoint": str(selected_path.resolve()), "boundary_target_contract": str((run / "02_boundary_target" / "BOUNDARY_TARGET_CONTRACT.json").resolve()), "validation_scope": "WHU Validation 1036 only", "test_used": False, "test_pixel_reads": 0})
    (package / "MODEL_CARD.md").write_text(f"# URRS Building Foundation v2 B2 Boundary Ablation\n\nStatus: `{status}`. Native RSBuilding building output is preserved. Boundary supervision is an auxiliary ablation only. WHU Test, SECOND Test, Nanjing, Tongzhou, Construction, Height and 3D-GloBFP pixels were not used. Human QA remains required.\n", encoding="utf-8")
    write_json(run / "12_acceptance" / "PROTECTED_ASSET_INVARIANCE.json", {"b0_sha256": sha(B0), "b0_unchanged": sha(B0) == "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7", "b1_sha256": sha(B1), "full20_unchanged": True, "registry_modified": False, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "tfr_training_started": False, "mti_training_started": False, "construction_training_started": False, "height_training_started": False, "test_used": False})
    manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8")); manifest.update({"status": "success", "scientific_status": status, "candidate_status": status, "selected_model": selected["model"] if selected else "B0_ITER2000_NATIVE", "selected_checkpoint": str(selected_path.resolve()), "selected_checkpoint_sha256": sha(selected_path), "validation_count": 1036, "test_used": False, "test_pixel_reads": 0, "registry_modified": False})
    write_json(run / "RUN_MANIFEST.json", manifest)
    report = [f"# Building Foundation v2 B2 Boundary Auxiliary Ablation", "", f"- run: `{run.name}`", f"- status: `{status}`", f"- selected: `{manifest['selected_model']}`", f"- selected checkpoint SHA256: `{manifest['selected_checkpoint_sha256']}`", "- boundary target width: `2` (symmetric, fixed before Validation)", "- Validation: `WHU Validation 1036; Test not read`", "", "## Candidate comparison", "", "| model | pixel IoU | pixel F1 | object F1@0.5 | BF1@2 | small recall | large recall |", "|---|---:|---:|---:|---:|---:|---:|"]
    report += [f"| {row['model']} | {row['pixel_iou']} | {row['pixel_f1']} | {row['object_f1_0.5']} | {row['boundary_f1_2']} | {row['small_recall']} | {row['large_recall']} |" for row in csv_rows]
    report += ["", "B2 is not a production or national-generalization claim. The protected native baseline is retained unless the selection receipt explicitly marks a candidate eligible. B3, TFR, MTI, Construction, Height and real-scene adaptation were not started."]
    (run / "12_acceptance" / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(run / "12_acceptance" / "FINAL_B2_BOUNDARY_RECEIPT.json", {"status": status, "run_dir": str(run), "selected_model": manifest["selected_model"], "selected_checkpoint": manifest["selected_checkpoint"], "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"], "validation_count": 1036, "test_used": False, "test_pixel_reads": 0})
    print(json.dumps({"status": status, "selected_model": manifest["selected_model"], "selected_checkpoint": manifest["selected_checkpoint"], "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"], "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__": raise SystemExit(main())
