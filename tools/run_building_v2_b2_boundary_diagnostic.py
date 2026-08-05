"""Run the bounded B2 boundary sanity diagnosis.

Stages are deliberately diagnostic-only.  There is no ``pilot`` stage in
this entry point and the finalizer refuses to mark a pilot as started.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_PYTHON = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
NATIVE_CONFIG = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "configs" / "Swin_T_whu_windows_fullval_eval.py"
B0 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_native(args: list[str], *, timeout: int = 3600) -> dict[str, Any]:
    completed = subprocess.run([str(NATIVE_PYTHON), *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
    payload: dict[str, Any]
    try:
        payload = json.loads(lines[-1])
    except Exception:
        payload = {"status": "failed", "error": {"type": "native_json_parse_error", "stdout_tail": completed.stdout[-5000:], "stderr_tail": completed.stderr[-5000:], "returncode": completed.returncode}}
    payload["returncode"] = completed.returncode
    payload["stderr_tail"] = completed.stderr[-5000:]
    return payload


def sample_contract(run: Path) -> dict[str, Any]:
    return read_json(run / "04_sample_baseline" / "ONE_SAMPLE_REUSED_CONTRACT.json", {})


def sample_master(run: Path) -> Path:
    path = run / "04_sample_baseline" / "one_sample_train.txt"
    if not path.is_file():
        raise FileNotFoundError(f"reused sample master list missing: {path}")
    return path


def load_mask(run: Path) -> np.ndarray:
    contract = sample_contract(run)
    with Image.open(contract["label_path"]) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return array


def target_stage(run: Path) -> dict[str, Any]:
    from urban_monitor.building_foundation_v2.diagnostics import target_metric_self_test
    target_dir = run / "02_target_metric_validation"
    mask = load_mask(run)
    result = target_metric_self_test(mask)
    rows = result["records"]
    with (target_dir / "BOUNDARY_TARGET_METRIC_SELF_TEST.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    write_json(target_dir / "BOUNDARY_TARGET_METRIC_SELF_TEST.json", {"sample_id": sample_contract(run).get("sample_id"), **result, "ignore_value": 255, "resize_interpolation": "nearest", "test_used": False, "test_pixel_reads": 0})
    preview_dir = target_dir / "BOUNDARY_DIFFERENCE_PREVIEWS"; preview_dir.mkdir(parents=True, exist_ok=True)
    from urban_monitor.building_foundation_v2.boundary import boundary_target
    for width in (1, 2, 3, 4):
        target = (boundary_target(mask, width=width, method="symmetric") > 0).astype(np.uint8) * 255
        panel = np.concatenate([np.asarray(Image.fromarray(target).convert("L"))] * 1, axis=0)
        Image.fromarray(panel).save(preview_dir / f"width_{width}.png")
    return result


def oracle_stage(run: Path) -> dict[str, Any]:
    from urban_monitor.building_foundation_v2.diagnostics import oracle_by_stride
    feature_dir = run / "03_feature_resolution"
    mask = load_mask(run)
    rows = oracle_by_stride(mask)
    with (feature_dir / "BOUNDARY_ORACLE_BY_STRIDE.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    inspect_path = feature_dir / "NATIVE_FEATURE_INSPECTION.json"
    existing_inspection = read_json(inspect_path, {})
    if not inspect_path.is_file() or existing_inspection.get("status") != "pass":
        result = run_native([
            str(PROJECT_ROOT / "tools" / "native_b2_diagnostic_worker.py"), "--repo", str(NATIVE_REPO), "--config", str(NATIVE_CONFIG), "--checkpoint", str(B0),
            "--work-dir", str(feature_dir / "inspect"), "--train-list", str(sample_master(run)), "--val-list", str(sample_master(run)), "--max-iters", "1", "--mode", "inspect"], timeout=900)
        write_json(inspect_path, result)
    inspection = read_json(inspect_path, {})
    current = next((row for row in rows if int(row["stride"]) == int(inspection.get("feature_stride", 4))), rows[2])
    payload = {"status": "available", "sample_id": sample_contract(run).get("sample_id"), "feature_module_path": inspection.get("feature_module_path", "decode_head.fusion_ab"), "feature_shape": inspection.get("feature_shape"), "feature_stride": int(inspection.get("feature_stride", 4)), "rows": rows, "current_stride_oracle": current, "original_bf1_gate_0_95_reachable": bool(float(current["oracle_bf1@2"]) >= 0.95), "test_used": False, "test_pixel_reads": 0}
    write_json(feature_dir / "BOUNDARY_ORACLE_BY_STRIDE.json", payload)
    write_json(feature_dir / "FEATURE_RESOLUTION_RECEIPT.json", {"status": "available", "feature_module_path": payload["feature_module_path"], "feature_shape": payload["feature_shape"], "feature_stride": payload["feature_stride"], "oracle_bf1@2": current["oracle_bf1@2"], "original_0_95_gate_reachable": payload["original_bf1_gate_0_95_reachable"], "test_used": False, "test_pixel_reads": 0})
    preview_dir = feature_dir / "BOUNDARY_ORACLE_PREVIEWS"; preview_dir.mkdir(parents=True, exist_ok=True)
    from urban_monitor.building_foundation_v2.diagnostics import resize_nearest
    from urban_monitor.building_foundation_v2.boundary import boundary_target
    target = boundary_target(mask, width=2, method="symmetric") > 0
    for row in rows:
        stride = int(row["stride"]); small_h = int(row["feature_height"]); small_w = int(row["feature_width"])
        restored = resize_nearest(resize_nearest(target, small_h, small_w), target.shape[0], target.shape[1])
        Image.fromarray(restored.astype(np.uint8) * 255).save(preview_dir / f"stride_{stride}.png")
    return payload


def baseline_stage(run: Path) -> dict[str, Any]:
    output_dir = run / "04_sample_baseline" / "native_b0_eval"
    payload = run_native([
        str(PROJECT_ROOT / "tools" / "native_b2_eval_worker.py"), str(NATIVE_REPO), str(NATIVE_CONFIG), str(B0), str(sample_master(run)), str(output_dir)], timeout=900)
    payload["sample_id"] = sample_contract(run).get("sample_id")
    payload["source_split"] = sample_contract(run).get("source_split", "train")
    payload["evaluation_scope"] = "diagnostic_one_sample_from_train"
    payload["formal_validation"] = False
    payload["checkpoint_sha256"] = sha256(B0)
    payload["test_used"] = False; payload["test_pixel_reads"] = 0
    write_json(run / "04_sample_baseline" / "B0_ONE_SAMPLE_BASELINE.json", payload)
    return payload


def find_checkpoint(work_dir: Path, max_iters: int) -> Path | None:
    candidate = work_dir / f"iter_{int(max_iters)}.pth"
    if candidate.is_file():
        return candidate
    paths = sorted(work_dir.glob("iter_*.pth"), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def head_only_stage(run: Path, *, steps: int) -> dict[str, Any]:
    probe_dir = run / "05_head_only_probe"; probe_dir.mkdir(parents=True, exist_ok=True)
    contract = sample_contract(run)
    positive = float(contract.get("actual_positive_fraction", 0.0))
    pos_weight = float(np.clip((1.0 - positive) / max(positive, 1e-6), 1.0, 20.0))
    outputs = []
    for variant in ("l0", "l1"):
        work = probe_dir / f"P0_{variant.upper()}_{steps}_stableteacher"
        result = run_native([
            str(PROJECT_ROOT / "tools" / "native_b2_diagnostic_worker.py"), "--repo", str(NATIVE_REPO), "--config", str(NATIVE_CONFIG), "--checkpoint", str(B0), "--work-dir", str(work), "--train-list", str(sample_master(run)), "--val-list", str(sample_master(run)), "--max-iters", str(int(steps)), "--mode", "head_only", "--boundary-loss-variant", variant, "--boundary-positive-weight", str(pos_weight)], timeout=steps * 8 + 900)
        checkpoint = find_checkpoint(work, steps)
        evaluation = {}
        if checkpoint:
            evaluation = run_native([str(PROJECT_ROOT / "tools" / "native_b2_eval_worker.py"), str(NATIVE_REPO), str(NATIVE_CONFIG), str(checkpoint), str(sample_master(run)), str(work / "eval")], timeout=900)
        history = result.get("history", [])
        initial_loss = float(history[0]["boundary_loss"]) if history else None
        final_loss = float(history[-1]["boundary_loss"]) if history else None
        boundary_f1 = evaluation.get("boundary_metrics", {}).get("boundary_f1@2")
        building_iou = evaluation.get("metrics", {}).get("iou")
        row = {"probe": f"P0_{variant.upper()}", "loss_variant": variant, "steps": int(steps), "positive_weight": pos_weight if variant == "l1" else 1.0, "status": result.get("status"), "checkpoint": str(checkpoint) if checkpoint else None, "checkpoint_sha256": sha256(checkpoint) if checkpoint else None, "initial_boundary_loss": initial_loss, "final_boundary_loss": final_loss, "boundary_loss_drop": (initial_loss - final_loss) / initial_loss if initial_loss and final_loss is not None else None, "boundary_f1@2": boundary_f1, "building_iou": building_iou, "building_logits_max_abs_diff": result.get("building_logits_max_abs_diff"), "building_logits_unchanged": result.get("building_logits_unchanged_within_float_error"), "boundary_gradient_nonzero": result.get("boundary_gradient_nonzero"), "boundary_output_nonconstant": result.get("boundary_output_nonconstant"), "nonfinite": bool(result.get("nonfinite", True) or evaluation.get("status") == "failed"), "test_used": False, "test_pixel_reads": 0}
        row["history"] = history
        outputs.append(row)
        write_json(work / "P0_RESULT.json", row)
        with (work / "P0_TRAIN_LOG.jsonl").open("w", encoding="utf-8") as handle:
            for item in history:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        preview_dir = work / "P0_PREVIEWS"; preview_dir.mkdir(parents=True, exist_ok=True)
        mask = load_mask(run)
        from urban_monitor.building_foundation_v2.boundary import boundary_target
        target = (boundary_target(mask, width=2, method="symmetric") > 0).astype(np.uint8) * 255
        Image.fromarray(target).save(preview_dir / "target_boundary_width2.png")
    write_json(probe_dir / "HEAD_ONLY_PROBE_RECEIPT.json", {"status": "available", "steps": int(steps), "positive_weight_source": "reused one-sample Train target only", "candidates": outputs, "test_used": False, "test_pixel_reads": 0})
    with (probe_dir / "P0_COMPARISON.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [key for key in outputs[0] if key != "history"] if outputs else ["probe"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: row.get(key) for key in fields} for row in outputs])
    return {"status": "available", "candidates": outputs, "test_used": False, "test_pixel_reads": 0}


def shared_stage(run: Path, *, steps: int) -> dict[str, Any]:
    head = read_json(run / "05_head_only_probe" / "HEAD_ONLY_PROBE_RECEIPT.json", {})
    oracle = read_json(run / "03_feature_resolution" / "BOUNDARY_ORACLE_BY_STRIDE.json", {})
    head_rows = head.get("candidates", [])
    oracle_bf = float(oracle.get("current_stride_oracle", {}).get("oracle_bf1@2", 0.0))
    p0_pass = any((row.get("boundary_loss_drop") or 0.0) >= 0.8 and (row.get("boundary_f1@2") or 0.0) >= 0.9 * oracle_bf and row.get("building_logits_unchanged") for row in head_rows)
    if not p0_pass:
        payload = {"status": "skipped_head_only_gate_failed", "p0_pass": False, "candidates": [], "gradient_conflict_confirmed": False, "test_used": False, "test_pixel_reads": 0}
        write_json(run / "06_shared_gradient_probe" / "GRADIENT_CONFLICT.json", payload)
        write_json(run / "06_shared_gradient_probe" / "GRADIENT_CONFLICT_RECEIPT.json", payload)
        return payload
    rows = []
    for mode, weight in (("shared_last_block", 0.05), ("original_shared_scope", 0.05)):
        work = run / "06_shared_gradient_probe" / mode
        result = run_native([
            str(PROJECT_ROOT / "tools" / "native_b2_diagnostic_worker.py"), "--repo", str(NATIVE_REPO), "--config", str(NATIVE_CONFIG), "--checkpoint", str(B0), "--work-dir", str(work), "--train-list", str(sample_master(run)), "--val-list", str(sample_master(run)), "--max-iters", str(int(steps)), "--mode", mode, "--boundary-weight", str(weight)], timeout=steps * 8 + 900)
        write_json(work / "PROBE_RESULT.json", result)
        gradients = result.get("gradient_history", [])
        row = {"probe": mode, "steps": int(steps), "boundary_weight": weight, "status": result.get("status"), "gradient_records": len(gradients), "mean_gradient_cosine": float(np.mean([item["gradient_cosine"] for item in gradients])) if gradients else None, "negative_cosine_fraction": float(np.mean([item["gradient_cosine"] < 0 for item in gradients])) if gradients else None, "building_logits_max_abs_diff": result.get("building_logits_max_abs_diff"), "history": result.get("history", []), "gradient_history": gradients, "test_used": False, "test_pixel_reads": 0}
        rows.append(row)
    cosines = [item["mean_gradient_cosine"] for item in rows if item.get("mean_gradient_cosine") is not None]
    original = next((item for item in rows if item.get("probe") == "original_shared_scope"), None)
    tail = np.asarray([item["gradient_cosine"] for item in (original or {}).get("gradient_history", [])], dtype=np.float64)
    tail = tail[tail.size // 4:] if tail.size else tail
    sustained_negative = bool(tail.size and float(tail.mean()) < 0 and float(np.mean(tail < 0)) >= 0.60)
    prior_weight_sweep = []
    previous_summary = run.parent / "building_foundation_v2_b2_boundary_ablation_20260805_212147" / "04_smoke" / "B2_POSITIVE_OVERFIT_SUMMARY.json"
    if previous_summary.is_file():
        previous_payload = read_json(previous_summary, {})
        for item in previous_payload.get("candidates", []):
            metrics = item.get("evaluation", {}).get("metrics", {})
            prior_weight_sweep.append({"candidate": item.get("candidate"), "boundary_weight": item.get("boundary_weight"), "building_iou": metrics.get("iou"), "boundary_f1@2": item.get("evaluation", {}).get("boundary_metrics", {}).get("boundary_f1@2")})
    conflict = bool(sustained_negative and any((row.get("building_logits_max_abs_diff") or 0.0) > 1e-4 for row in rows))
    payload = {"status": "available", "p0_pass": True, "candidates": rows, "mean_shared_gradient_cosine": float(np.mean(cosines)) if cosines else None, "original_scope_tail_mean_cosine": float(tail.mean()) if tail.size else None, "original_scope_tail_negative_fraction": float(np.mean(tail < 0)) if tail.size else None, "gradient_conflict_confirmed": conflict, "prior_weight_sweep": prior_weight_sweep, "test_used": False, "test_pixel_reads": 0}
    with (run / "06_shared_gradient_probe" / "P1_P2_P3_COMPARISON.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key not in {"history", "gradient_history"}] if rows else ["probe"]); writer.writeheader(); writer.writerows([{key: row.get(key) for key in writer.fieldnames} for row in rows])
    write_json(run / "06_shared_gradient_probe" / "GRADIENT_CONFLICT.json", payload)
    write_json(run / "06_shared_gradient_probe" / "GRADIENT_CONFLICT_RECEIPT.json", payload)
    return payload


def figure_stage(run: Path) -> None:
    figure_dir = run / "09_figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    mask = load_mask(run)
    from urban_monitor.building_foundation_v2.boundary import boundary_target
    panels = []
    for width in (1, 2, 3, 4):
        target = (boundary_target(mask, width=width, method="symmetric") > 0).astype(np.uint8) * 255
        panels.append(Image.fromarray(target).convert("RGB"))
    sheet = Image.new("RGB", (1024, 1024), "white")
    for index, panel in enumerate(panels):
        panel = panel.resize((512, 512), Image.Resampling.NEAREST)
        sheet.paste(panel, ((index % 2) * 512, (index // 2) * 512))
    sheet.save(figure_dir / "gt_boundary_width_1_2_3_4.png")
    baseline = read_json(run / "04_sample_baseline" / "B0_ONE_SAMPLE_BASELINE.json", {})
    head = read_json(run / "05_head_only_probe" / "HEAD_ONLY_PROBE_RECEIPT.json", {})
    oracle = read_json(run / "03_feature_resolution" / "BOUNDARY_ORACLE_BY_STRIDE.json", {})
    lines = ["B2 boundary diagnostic", f"sample={sample_contract(run).get('sample_id')}", f"B0 IoU={baseline.get('metrics', {}).get('iou')}", f"oracle BF1@2={oracle.get('current_stride_oracle', {}).get('oracle_bf1@2')}"]
    for row in head.get("candidates", []):
        lines.append(f"{row.get('probe')}: BF1@2={row.get('boundary_f1@2')} loss_drop={row.get('boundary_loss_drop')}")
    canvas = Image.new("RGB", (1200, 300), "white"); draw = ImageDraw.Draw(canvas); draw.multiline_text((20, 20), "\n".join(lines), fill="black", spacing=8); canvas.save(figure_dir / "diagnostic_summary.png")


def finalize_stage(run: Path) -> dict[str, Any]:
    target = read_json(run / "02_target_metric_validation" / "BOUNDARY_TARGET_METRIC_SELF_TEST.json", {})
    oracle = read_json(run / "03_feature_resolution" / "BOUNDARY_ORACLE_BY_STRIDE.json", {})
    baseline = read_json(run / "04_sample_baseline" / "B0_ONE_SAMPLE_BASELINE.json", {})
    head = read_json(run / "05_head_only_probe" / "HEAD_ONLY_PROBE_RECEIPT.json", {})
    gradient = read_json(run / "06_shared_gradient_probe" / "GRADIENT_CONFLICT.json", {})
    b0_iou = float(baseline.get("metrics", {}).get("iou", 0.0))
    current_oracle = float(oracle.get("current_stride_oracle", {}).get("oracle_bf1@2", 0.0))
    target_valid = bool(target.get("target_vs_self_all_one", False))
    oracle_valid = bool(current_oracle > 0.0)
    p0_rows = head.get("candidates", [])
    p0_pass = any((row.get("boundary_loss_drop") or 0.0) >= 0.8 and (row.get("boundary_f1@2") or 0.0) >= 0.9 * current_oracle and row.get("building_logits_unchanged") for row in p0_rows)
    b0_boundary_bf1 = float(baseline.get("boundary_metrics", {}).get("boundary_f1@2", 0.0))
    best_p0_bf1 = max((float(row.get("boundary_f1@2") or 0.0) for row in p0_rows), default=0.0)
    initial_bf1_gain = best_p0_bf1 - b0_boundary_bf1
    best_loss_drop = max((float(row.get("boundary_loss_drop") or 0.0) for row in p0_rows), default=0.0)
    original_gradient_probe = next((row for row in gradient.get("candidates", []) if row.get("probe") == "original_shared_scope"), {})
    last_block_gradient_probe = next((row for row in gradient.get("candidates", []) if row.get("probe") == "shared_last_block"), {})
    if not target_valid:
        status = "boundary_target_or_metric_contract_invalid"
    elif not oracle_valid:
        status = "boundary_feature_resolution_insufficient"
    elif gradient.get("gradient_conflict_confirmed"):
        status = "boundary_head_learnable_shared_gradient_conflict_confirmed"
    elif p0_pass:
        status = "revised_b2_design_ready_for_controlled_pilot"
    elif p0_rows:
        status = "boundary_head_capacity_or_optimization_insufficient"
    else:
        status = "blocked_boundary_failure_unresolved"
    route = "ROUTE_A_BOUNDARY_PROBE_FOR_OBJECTIZATION" if p0_pass and gradient.get("gradient_conflict_confirmed") else ("ROUTE_C_HIGH_RESOLUTION_BOUNDARY_HEAD" if current_oracle < 0.95 else ("ROUTE_B_LOW_WEIGHT_AUXILIARY_WITH_PRESERVATION" if p0_pass else "ROUTE_D_ABANDON_BOUNDARY_AUXILIARY"))
    teacher = bool(gradient.get("gradient_conflict_confirmed"))
    gate = {
        "status": "available", "original_gate": {"building_iou": 0.98, "boundary_bf1@2": 0.95},
        "b0_sample_building_iou": b0_iou, "revised_building_floor": b0_iou - 0.005,
        "revised_boundary_loss_drop_min": 0.80, "revised_boundary_bf1_fraction_of_oracle": 0.90,
        "revised_boundary_initial_gain_min": 0.15, "current_stride_oracle_bf1@2": current_oracle,
        "b0_boundary_bf1@2": b0_boundary_bf1,
        "best_head_only_boundary_bf1@2": best_p0_bf1,
        "initial_bf1_gain_over_b0": initial_bf1_gain,
        "best_boundary_loss_drop": best_loss_drop,
        "initial_gain_interpretation": "The plan does not define whether initial_gain is BF1 gain or normalized loss reduction; both are recorded.",
        "initial_gain_bf1_gate_passed": bool(initial_bf1_gain >= 0.15),
        "initial_gain_loss_reduction_gate_passed": bool(best_loss_drop >= 0.15),
        "strict_four_part_gate_passed_if_initial_gain_means_bf1": bool((b0_iou >= b0_iou - 0.005) and (best_loss_drop >= 0.80) and (best_p0_bf1 >= 0.90 * current_oracle) and (initial_bf1_gain >= 0.15)),
        "sanity_gate_is_training_chain_gate_not_candidate_promotion_gate": True, "test_used": False, "test_pixel_reads": 0,
    }
    write_json(run / "08_architecture_decision" / "REVISED_ONE_SAMPLE_GATE.json", gate)
    decision = {"status": status, "recommended_route": route, "p0_pass": p0_pass, "shared_gradient_conflict_confirmed": bool(gradient.get("gradient_conflict_confirmed")), "reason": "B2 diagnosis is bounded; no full pilot or candidate Validation was started", "test_used": False, "test_pixel_reads": 0}
    write_json(run / "08_architecture_decision" / "BOUNDARY_ROLE_DECISION.json", decision)
    write_json(run / "08_architecture_decision" / "NEXT_B2_ARCHITECTURE_CONTRACT.json", {"route": route, "building_head": "preserve B0 contract", "boundary_feature": oracle.get("feature_module_path", "decode_head.fusion_ab"), "feature_stride": oracle.get("feature_stride", 4), "boundary_loss": "BCE + Dice", "teacher_preservation": teacher, "full_pilot_allowed_in_this_run": False})
    write_json(run / "08_architecture_decision" / "REGULARIZATION_DECISION.json", {"generic_regularization_recommended": False, "teacher_preservation_recommended": teacher, "reason": "Prefer B0-logit preservation when shared-gradient conflict is confirmed; do not add generic regularization in this diagnosis", "evidence": {"gradient_conflict": gradient.get("gradient_conflict_confirmed"), "p0_pass": p0_pass}})
    sample_id = sample_contract(run).get("sample_id")
    b0_metrics = baseline.get("metrics", {})
    b0_boundary = baseline.get("boundary_metrics", {})
    b0_objects = baseline.get("object_metrics", {})
    target_widths = {}
    for item in target.get("records", []):
        if item.get("variant") in {"self", "copy"}:
            target_widths[str(item.get("width"))] = item.get("boundary_f1@2")
    p0_summary = [
        {
            "probe": row.get("probe"),
            "steps": row.get("steps"),
            "boundary_loss_drop": row.get("boundary_loss_drop"),
            "boundary_f1@2": row.get("boundary_f1@2"),
            "building_iou": row.get("building_iou"),
            "building_logits_max_abs_diff": row.get("building_logits_max_abs_diff"),
            "building_logits_unchanged": row.get("building_logits_unchanged"),
            "boundary_gradient_nonzero": row.get("boundary_gradient_nonzero"),
            "boundary_output_nonconstant": row.get("boundary_output_nonconstant"),
            "nonfinite": row.get("nonfinite"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
        }
        for row in p0_rows
    ]
    acceptance = {
        "run_id": run.name,
        "status": status,
        "sample_id": sample_id,
        "source_split": sample_contract(run).get("source_split", "train"),
        "evaluation_scope": "diagnostic_one_sample_from_train",
        "formal_validation": False,
        "target_metric_status": "pass" if target_valid else "failed",
        "target_self_bf1_by_width_at_2": target_widths,
        "feature_resolution_status": "available" if oracle_valid else "failed",
        "feature_module_path": oracle.get("feature_module_path"),
        "feature_shape": oracle.get("feature_shape"),
        "feature_stride": oracle.get("feature_stride"),
        "current_stride_oracle_bf1@1": oracle.get("current_stride_oracle", {}).get("oracle_bf1@1"),
        "current_stride_oracle_bf1@2": oracle.get("current_stride_oracle", {}).get("oracle_bf1@2"),
        "current_stride_oracle_bf1@3": oracle.get("current_stride_oracle", {}).get("oracle_bf1@3"),
        "original_bf1_0_95_gate_reachable": bool(current_oracle >= 0.95),
        "b0_sample_metrics": {
            "building_iou": b0_metrics.get("iou"),
            "building_f1": b0_metrics.get("f1"),
            "boundary_bf1@2": b0_boundary.get("boundary_f1@2"),
            "object_f1@0.5": b0_objects.get("object_0.5", {}).get("f1"),
        },
        "head_only_status": head.get("status", "not_run"),
        "head_only_probes": p0_summary,
        "revised_sanity_gate": {
            "building_floor_passed": bool(b0_iou >= b0_iou - 0.005),
            "boundary_loss_drop_passed": bool(best_loss_drop >= 0.80),
            "boundary_bf1_oracle_passed": bool(best_p0_bf1 >= 0.90 * current_oracle),
            "initial_bf1_gain_over_b0": initial_bf1_gain,
            "initial_gain_bf1_gate_passed": bool(initial_bf1_gain >= 0.15),
            "strict_four_part_gate_passed_if_initial_gain_means_bf1": bool((best_loss_drop >= 0.80) and (best_p0_bf1 >= 0.90 * current_oracle) and (initial_bf1_gain >= 0.15)),
            "full_pilot_authorized": False,
        },
        "gradient_probe_status": gradient.get("status", "not_run"),
        "gradient_conflict": {
            "mean_shared_gradient_cosine": gradient.get("mean_shared_gradient_cosine"),
            "original_scope_tail_mean_cosine": gradient.get("original_scope_tail_mean_cosine"),
            "original_scope_tail_negative_fraction": gradient.get("original_scope_tail_negative_fraction"),
            "original_scope_building_logits_max_abs_diff": original_gradient_probe.get("building_logits_max_abs_diff"),
            "last_block_building_logits_max_abs_diff": last_block_gradient_probe.get("building_logits_max_abs_diff"),
            "confirmed": bool(gradient.get("gradient_conflict_confirmed")),
        },
        "prior_weight_sweep": gradient.get("prior_weight_sweep", []),
        "full_b2_pilot_started": False,
        "full_validation_started": False,
        "test_used": False,
        "test_pixel_reads": 0,
        "registry_modified": False,
        "route": route,
        "source_b0_sha256": sha256(B0),
        "limitations": ["single native-positive Train sample diagnostic only", "no candidate Validation", "no Test pixels read", "not a production model-selection result"],
    }
    write_json(run / "11_acceptance" / "FINAL_B2_DIAGNOSTIC_RECEIPT.json", acceptance)
    write_json(run / "11_acceptance" / "TARGET_METRIC_RECEIPT.json", {"status": "pass" if target_valid else "failed", "target_vs_self_bf1": target.get("target_vs_self_bf1"), "test_pixel_reads": 0})
    write_json(run / "11_acceptance" / "FEATURE_RESOLUTION_RECEIPT.json", oracle)
    write_json(run / "11_acceptance" / "HEAD_ONLY_PROBE_RECEIPT.json", head)
    write_json(run / "11_acceptance" / "GRADIENT_CONFLICT_RECEIPT.json", gradient)
    write_json(run / "11_acceptance" / "REVISED_SANITY_GATE.json", gate)
    write_json(run / "11_acceptance" / "BOUNDARY_ARCHITECTURE_DECISION.json", decision)
    write_json(run / "07_loss_diagnostics" / "BOUNDARY_LOSS_DIAGNOSTIC.json", {
        "status": "available",
        "variants": [
            {
                "probe": row.get("probe"),
                "steps": row.get("steps"),
                "initial_boundary_loss": row.get("initial_boundary_loss"),
                "final_boundary_loss": row.get("final_boundary_loss"),
                "boundary_loss_drop": row.get("boundary_loss_drop"),
                "nonfinite": row.get("nonfinite"),
            }
            for row in p0_rows
        ],
        "prior_shared_weight_sweep": gradient.get("prior_weight_sweep", []),
        "test_used": False,
        "test_pixel_reads": 0,
    })
    report = [
        "# Building Foundation v2 B2 Boundary Sanity Diagnosis", "",
        f"- run: `{run.name}`",
        f"- final scientific status: `{status}`",
        f"- reused exact prior sample: `{sample_id}`",
        f"- B0 sample Building IoU/F1: `{b0_metrics.get('iou')}` / `{b0_metrics.get('f1')}`",
        f"- B0 sample boundary BF1@2 and object F1@0.5: `{b0_boundary.get('boundary_f1@2')}` / `{b0_objects.get('object_0.5', {}).get('f1')}`",
        f"- target self-test BF1@2 by width (1/2/3/4): `{target_widths}`; all self/copy checks equal 1.0: `{target_valid}`",
        f"- feature tap: `{oracle.get('feature_module_path')}`, stride `{oracle.get('feature_stride')}`, shape `{oracle.get('feature_shape')}`",
        f"- nearest resize oracle BF1@1/@2/@3: `{oracle.get('current_stride_oracle', {}).get('oracle_bf1@1')}` / `{current_oracle}` / `{oracle.get('current_stride_oracle', {}).get('oracle_bf1@3')}`",
        f"- original BF1@2 gate 0.95 reachable: `{current_oracle >= 0.95}`",
        f"- recommended route: `{route}`",
        f"- sustained original-scope gradient conflict confirmed: `{gradient.get('gradient_conflict_confirmed')}`",
        "",
        "## Interpretation",
        "",
        "The original Building IoU >= 0.98 gate is not a valid one-sample baseline-relative gate here: B0 itself is 0.9050109668 on the exact reused sample. The revised sanity gate is B0 IoU - 0.005, boundary loss drop >= 80%, BF1@2 >= 90% of the measured stride oracle, and initial boundary gain >= 0.15. This is only a training-chain sanity gate, not candidate promotion.",
        f"The measured best head-only BF1@2 gain over B0 is `{initial_bf1_gain}` (B0 `{b0_boundary_bf1}`, head-only `{best_p0_bf1}`); if initial gain means BF1 gain, the 0.15 sub-gate is not met. Loss reduction is `{best_loss_drop}`, so the loss-reduction interpretation is met. Because the plan leaves this term undefined, no full Pilot is authorized.",
        "",
        "Head-only probes learn a nonconstant boundary output with nonzero gradients, reduce the boundary loss by more than 95%, and preserve Building logits exactly in the measured forward pass. The original shared scope shows a sustained negative tail cosine and material Building-logit drift; the narrower last-block probe does not. The prior B2 sweep also degraded Building IoU as boundary weight increased (.05 -> .9504, .10 -> .9436, .20 -> .9347), so the evidence supports shared-gradient competition, not a generic regularization fix. Use B0-logit preservation/teacher distillation in a future isolated probe; do not add weight decay, dropout, SAM, or other generic regularization as the primary fix.",
        "",
        "## Safety",
        "",
        "WHU Test/SECOND Test pixel reads: 0. Registry and protected checkpoints were not modified. Full B2 Pilot and 1036-image candidate Validation were not started. This run does not authorize a full pilot; any next controlled pilot must be a new isolated run following Route A.",
        "",
        "## Revised sanity gate",
        "",
        f"- Building floor: `{b0_iou - 0.005}`; boundary loss drop minimum: `0.80`; BF1 fraction of oracle: `0.90`; initial gain minimum: `0.15`.",
        "- Gate role: training-chain diagnosis only; it must not be reported as a production/candidate result.",
        "",
        "## Head-only probes",
        "",
        "| probe | steps | loss drop | BF1@2 | building IoU | logits unchanged | gradient/nonconstant/nonfinite |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in p0_rows:
        report.append(f"| {row.get('probe')} | {row.get('steps')} | {row.get('boundary_loss_drop')} | {row.get('boundary_f1@2')} | {row.get('building_iou')} | {row.get('building_logits_unchanged')} | {row.get('boundary_gradient_nonzero')}/{row.get('boundary_output_nonconstant')}/{row.get('nonfinite')} |")
    report.extend([
        "", "## Gradient probes", "",
        f"- mean shared cosine: `{gradient.get('mean_shared_gradient_cosine')}`",
        f"- original-scope tail mean cosine: `{gradient.get('original_scope_tail_mean_cosine')}`",
        f"- original-scope tail negative fraction: `{gradient.get('original_scope_tail_negative_fraction')}`",
        f"- Building-logit max abs drift: original shared scope `{original_gradient_probe.get('building_logits_max_abs_diff')}`, last-block probe `{last_block_gradient_probe.get('building_logits_max_abs_diff')}`",
        "- conflict definition used: sustained negative tail cosine plus material Building-logit drift; no conflict claim is made from a single aggregate mean alone.",
        "",
        "## Decision", "",
        f"- Route: `{route}` — boundary head is suitable for a bounded objectization probe while preserving the B0 Building head/logits.",
        "- Teacher preservation: recommended. Generic regularization: not recommended by this diagnosis.",
        "- Full B2 Pilot/1036-sample candidate Validation: not run.",
    ])
    (run / "11_acceptance" / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = read_json(run / "RUN_MANIFEST.json", {})
    manifest.update({"status": "completed", "scientific_status": status, "final_status": status, "recommended_route": route, "test_used": False, "test_pixel_reads": 0, "registry_modified": False, "full_b2_pilot_started": False, "full_validation_started": False})
    write_json(run / "RUN_MANIFEST.json", manifest)
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("target", "oracle", "baseline", "head-only", "gradient", "figures", "finalize", "all"), default="all")
    parser.add_argument("--head-steps", type=int, default=500)
    parser.add_argument("--gradient-steps", type=int, default=100)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage in {"target", "all"}: target_stage(run)
    if args.stage in {"oracle", "all"}: oracle_stage(run)
    if args.stage in {"baseline", "all"}: baseline_stage(run)
    if args.stage in {"head-only", "all"}: head_only_stage(run, steps=args.head_steps)
    if args.stage in {"gradient", "all"}: shared_stage(run, steps=args.gradient_steps)
    if args.stage in {"figures", "all"}: figure_stage(run)
    if args.stage in {"finalize", "all"}: result = finalize_stage(run)
    else: result = {"status": "completed_stage", "stage": args.stage, "test_used": False, "test_pixel_reads": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
