"""Run isolated C-A/C-B0/C-B1 and record Tier-C blocked C-B2/C-B3.

The existing domain-adaptation preparation run is treated as read-only input.
Only its small receipts and manifests are copied into a new experiment root;
the image and pseudo-sample paths inside those manifests remain references to
the original artifacts.  No GloBFP pseudo-label is created when the matching
gate is Tier C.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_support_files(source_root: Path, target_root: Path) -> None:
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    files = [
        Path("00_acceptance") / "FINAL_ACCEPTANCE_RECEIPT.json",
        Path("02_training_data") / "DATA_PREPARATION_MANIFEST.json",
        Path("02_training_data") / "PSEUDO_LABEL_NUMERICAL_AUDIT.json",
        Path("02_training_data") / "MIXED_TRAIN_MANIFEST.csv",
        Path("02_training_data") / "MIXED_VALIDATION_MANIFEST.csv",
    ]
    for relative in files:
        source = source_root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def metric(payload: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def c_b1_gate(worker: Any, baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    gate = worker._gate(baseline, candidate)
    au = baseline["validation"]["ucs_validation"]
    ap = baseline["validation"]["nanjing_validation_proxy"]
    bv = candidate.get("best_validation", {})
    bu = bv.get("ucs_validation", {})
    bp = bv.get("nanjing_validation_proxy", {})
    checks = dict(gate.get("checks", {}))
    checks["boundary_f1_drop_le_0.02"] = metric(bu, "boundary_f1", -1.0) >= metric(au, "boundary_f1", 0.0) - 0.02
    baseline_candidate_fraction = metric(ap, "candidate_area_fraction", 0.0)
    candidate_fraction = metric(bp, "candidate_area_fraction", 0.0)
    checks["high_confidence_candidate_not_large_scale_suppression"] = (
        candidate_fraction >= baseline_candidate_fraction * 0.50
        if baseline_candidate_fraction > 0
        else candidate_fraction >= 0
    )
    checks["uncertainty_not_abnormally_zero"] = metric(bp, "mean_uncertainty", 0.0) > 0.01 and metric(bp, "mean_uncertainty", 0.0) < 0.99
    return {
        "status": "pass" if all(bool(value) for value in checks.values()) else "rejected",
        "checks": checks,
        "baseline_A": {"ucs": au, "proxy": ap},
        "candidate_C_B1": {"ucs": bu, "proxy": bp},
        "selection_used_holdout": False,
        "selection_used_locked_test": False,
        "created_at": now(),
    }


def pointer_experiment(target_root: Path, name: str, status: str, source: str | None = None, extra: Mapping[str, Any] | None = None) -> None:
    directory = target_root / "03_experiments" / name
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "experiment": name, "source_experiment": source, "created_at": now()}
    if extra:
        payload.update(extra)
    write_json(directory / "RUN_MANIFEST.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--match-audit", type=Path, required=True)
    parser.add_argument("--height-invariance", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_run_root.resolve()
    output_root = args.output_root.resolve()
    package_dir = args.package_dir.resolve()
    match_audit = read_json(args.match_audit.resolve())
    tier = str(match_audit.get("tier", "")).upper()
    if tier != "C":
        raise RuntimeError(f"this guarded runner only handles the observed Tier C path; received tier={tier!r}")
    copy_support_files(source_root, output_root)

    import sys

    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import construction_nanjing_domain_adaptation_worker as worker

    pseudo_audit = read_json(output_root / "02_training_data" / "PSEUDO_LABEL_NUMERICAL_AUDIT.json")
    config = worker._load_config(output_root)
    worker._set_seed()
    baseline = worker.run_A(output_root, package_dir, config, pseudo_audit)
    pointer_experiment(output_root, "C-A", "completed_frozen_baseline", "A", {
        "training_started": False,
        "validation_metrics": str(output_root / "03_experiments" / "A" / "VALIDATION_METRICS.json"),
        "checkpoint_sha256": baseline["context"]["checkpoint_sha256"],
    })

    pointer_experiment(output_root, "C-B0", "NO_OP_REJECTED_TIER_C", None, {
        "training_started": False,
        "prior_only": True,
        "stable_building_count": 0,
        "globfp_temporal_conflict_count": int(match_audit.get("temporal_conflict_count", 0)),
        "probability_rewrite": False,
        "reason": "Tier C matching gate; no stable GloBFP-supported building is eligible for prior-only suppression",
    })
    write_json(output_root / "03_experiments" / "C-B0" / "PRIOR_ONLY_AUDIT.json", {
        "status": "NO_OP_REJECTED_TIER_C",
        "stable_building_hard_negative_count": 0,
        "building_auxiliary_label_count": 0,
        "raw_construction_probability_modified": False,
        "construction_threshold_modified": False,
        "created_at": now(),
    })

    candidate = None
    candidate_error = None
    try:
        candidate = worker._run_train_experiment(output_root, package_dir, config, pseudo_audit, "B", baseline["validation"])
    except Exception as exc:  # preserve the failure as an auditable state
        candidate_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output_root / "03_experiments" / "B" / "RUN_FAILURE.json", candidate_error)
    if candidate is not None:
        gate = c_b1_gate(worker, baseline, candidate)
        write_json(output_root / "03_experiments" / "C-B1_GATE.json", gate)
        pointer_experiment(output_root, "C-B1", "completed_candidate_screen", "B", {
            "training_started": True,
            "source_worker_experiment": str(output_root / "03_experiments" / "B"),
            "gate_status": gate["status"],
            "best_checkpoint": candidate["manifest"].get("best_checkpoint"),
            "best_checkpoint_sha256": candidate["manifest"].get("best_checkpoint_sha256"),
        })
    else:
        gate = {"status": "failed", "checks": {}, "failure": candidate_error, "created_at": now()}
        write_json(output_root / "03_experiments" / "C-B1_GATE.json", gate)
        pointer_experiment(output_root, "C-B1", "FAILED_DETERMINISTIC_OR_RUNTIME_ERROR", "B", {"training_started": False, "failure": candidate_error})

    pointer_experiment(output_root, "C-B2", "BLOCKED_TIER_C_NO_GLOBFP_PSEUDO_LABELS", None, {
        "training_started": False,
        "reason": "Tier C explicitly forbids GloBFP pseudo-labels, hard negatives and auxiliary labels",
    })
    pointer_experiment(output_root, "C-B3", "BLOCKED_TIER_C", None, {
        "training_started": False,
        "reason": "C-B3 requires an accepted C-B2 and therefore cannot run",
    })

    b1_passed = candidate is not None and gate.get("status") == "pass"
    selected_experiment = "C-B1" if b1_passed else "C-A"
    selected_checkpoint = candidate["manifest"].get("best_checkpoint") if b1_passed and candidate else str((package_dir / "model.pth").resolve())
    selected_checkpoint_sha256 = candidate["manifest"].get("best_checkpoint_sha256") if b1_passed and candidate else baseline["context"]["checkpoint_sha256"]
    write_json(output_root / "04_selection" / "FINAL_MODEL_SELECTION.json", {
        "status": "COMPLETE_CONSERVATIVE_CONSTRUCTION_SELECTION",
        "selected_experiment": selected_experiment,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": selected_checkpoint_sha256,
        "adaptation_rejected": not b1_passed,
        "scientific_status": "domain_adaptation_candidate_or_rejected",
        "production_ready": False,
        "nanjing_ground_truth_available": False,
        "nanjing_true_accuracy_measured": False,
        "c_b1_gate": gate,
        "c_b2_status": "blocked_tier_c",
        "c_b3_status": "blocked_tier_c",
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
        "created_at": now(),
    })

    comparison_rows = [
        {"experiment": "C-A", "status": "completed_frozen_baseline", "training_started": False, "selected": selected_experiment == "C-A", "reason": "existing UCS386 checkpoint"},
        {"experiment": "C-B0", "status": "NO_OP_REJECTED_TIER_C", "training_started": False, "selected": False, "reason": "no stable GloBFP-supported building"},
        {"experiment": "C-B1", "status": "completed_candidate_screen" if candidate is not None else "FAILED", "training_started": candidate is not None, "selected": selected_experiment == "C-B1", "reason": gate.get("status")},
        {"experiment": "C-B2", "status": "BLOCKED_TIER_C_NO_GLOBFP_PSEUDO_LABELS", "training_started": False, "selected": False, "reason": "pseudo-label gate closed"},
        {"experiment": "C-B3", "status": "BLOCKED_TIER_C", "training_started": False, "selected": False, "reason": "C-B2 blocked"},
    ]
    write_csv(output_root / "CONSTRUCTION_EXPERIMENT_COMPARISON.csv", comparison_rows, ["experiment", "status", "training_started", "selected", "reason"])
    ucs_rows = []
    proxy_rows = []
    a_ucs = baseline["validation"]["ucs_validation"]
    a_proxy = baseline["validation"]["nanjing_validation_proxy"]
    ucs_rows.append({"experiment": "C-A", "status": "complete", "macro_miou": a_ucs.get("macro_miou"), "non_background_miou": a_ucs.get("non_background_miou"), "macro_f1": a_ucs.get("macro_f1"), "non_background_macro_f1": a_ucs.get("non_background_macro_f1"), "boundary_f1": a_ucs.get("boundary_f1"), "selection_used": True})
    proxy_rows.append({"experiment": "C-A", "status": "complete_proxy_only", **a_proxy})
    if candidate is not None:
        b_ucs = candidate.get("best_validation", {}).get("ucs_validation", {})
        b_proxy = candidate.get("best_validation", {}).get("nanjing_validation_proxy", {})
        ucs_rows.append({"experiment": "C-B1", "status": "complete", "macro_miou": b_ucs.get("macro_miou"), "non_background_miou": b_ucs.get("non_background_miou"), "macro_f1": b_ucs.get("macro_f1"), "non_background_macro_f1": b_ucs.get("non_background_macro_f1"), "boundary_f1": b_ucs.get("boundary_f1"), "selection_used": b1_passed})
        proxy_rows.append({"experiment": "C-B1", "status": "complete_proxy_only", **b_proxy})
    write_csv(output_root / "UCS_VALIDATION_COMPARISON.csv", ucs_rows, ["experiment", "status", "macro_miou", "non_background_miou", "macro_f1", "non_background_macro_f1", "boundary_f1", "selection_used"])
    proxy_fields = sorted({key for row in proxy_rows for key in row})
    write_csv(output_root / "NANJING_CONSTRUCTION_PROXY_COMPARISON.csv", proxy_rows, proxy_fields)

    write_json(output_root / "GLOBFP_PSEUDO_LABEL_AUDIT.json", {
        "status": "CLOSED_TIER_C",
        "tier": "C",
        "stable_building_hard_negative_count": 0,
        "building_auxiliary_label_count": 0,
        "globfp_only_label_count": 0,
        "temporal_conflict_excluded_count": int(match_audit.get("temporal_conflict_count", 0)),
        "pseudo_label_gate": "CLOSED_TIER_C",
        "current_nanjing_split_reused": True,
        "current_split_source": str(source_root / "02_training_data"),
        "current_mixed_train_manifest_sha256": sha256(output_root / "02_training_data" / "MIXED_TRAIN_MANIFEST.csv"),
        "training_inventory_augmented": False,
        "reason": "Tier C match quality does not permit GloBFP pseudo-labels or Construction C-B2/C-B3",
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
        "created_at": now(),
    })
    write_json(output_root / "BASELINE_POST_GLOBFP_INVARIANCE_AUDIT.json", read_json(args.height_invariance.resolve()))

    package_out = output_root / "06_model_package"
    package_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(selected_checkpoint), package_out / "construction_selected_checkpoint.pth")
    write_json(package_out / "MODEL_PACKAGE_MANIFEST.json", {
        "status": "SELECTED_CONSTRUCTION_CANDIDATE",
        "selected_experiment": selected_experiment,
        "source_checkpoint": selected_checkpoint,
        "checkpoint_sha256": sha256(package_out / "construction_selected_checkpoint.pth"),
        "gloBFP_prior_used": False,
        "scientific_status": "domain_adaptation_candidate_or_rejected",
        "production_ready": False,
        "created_at": now(),
    })
    write_json(output_root / "CONSTRUCTION_STAGE_RECEIPT.json", {
        "status": "COMPLETE_TIER_C_DEGRADED_CONSTRUCTION_SCREEN",
        "created_at": now(),
        "selected_experiment": selected_experiment,
        "selected_checkpoint_sha256": sha256(package_out / "construction_selected_checkpoint.pth"),
        "c_a_status": "completed_frozen_baseline",
        "c_b0_status": "NO_OP_REJECTED_TIER_C",
        "c_b1_status": "completed_candidate_screen" if candidate is not None else "FAILED",
        "c_b2_status": "BLOCKED_TIER_C_NO_GLOBFP_PSEUDO_LABELS",
        "c_b3_status": "BLOCKED_TIER_C",
        "stable_building_hard_negative_count": 0,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
        "nanjing_ground_truth_available": False,
    })
    print(json.dumps(read_json(output_root / "CONSTRUCTION_STAGE_RECEIPT.json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
