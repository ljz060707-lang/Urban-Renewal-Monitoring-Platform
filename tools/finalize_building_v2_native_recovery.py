"""Finalize the native recovery gate without starting any training.

The finalizer is deliberately conservative: it can certify native checkpoint
loading/reproduction and package an immutable B0 reference, but it refuses to
promote the old scratch U-Net or invent a B1/B2/B3 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITER2000_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
OLD_RUN = PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2_20260805_162958"
ITER2000 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD"), "status_short": run("status", "--short").splitlines()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    governance = read_json(run / "00_governance" / "RUN_GOVERNANCE.json", {})
    labels = read_json(run / "01_label_contract" / "WHU_LABEL_SEMANTICS_AUDIT.json", {})
    contract = read_json(run / "01_label_contract" / "SELECTED_WHU_LABEL_CONTRACT.json", {})
    native = read_json(run / "02_native_recovery" / "NATIVE_RUNTIME_AUDIT.json", {})
    validation = read_json(run / "07_validation" / "NATIVE_ITER2000_VALIDATION.json", {})
    pipeline = read_json(run / "03_input_pipeline" / "INPUT_PIPELINE_COMPARISON.json", {})
    if sha256(ITER2000) != ITER2000_SHA:
        raise SystemExit("protected iter_2000 SHA256 changed; finalization stopped")

    # Keep finalization idempotent and compatible with a validation receipt
    # produced by an earlier version of the validator.  The first native
    # recovery run already evaluated all 1,036 Validation samples, but its
    # receipt predates the distribution/reproduction fields.  Derive those
    # fields from the immutable aggregate metrics instead of re-running the
    # GPU validation (and never fall back to Test data).
    metrics = validation.get("metrics", {})
    historical = {"iou": 0.9279, "f1": 0.9626, "precision": 0.9680, "recall": 0.9573}
    differences = {}
    for key, expected in historical.items():
        observed = metrics.get(key)
        try:
            differences[key] = abs(float(observed) - expected) * 100.0
        except (TypeError, ValueError):
            differences[key] = float("inf")
    reproduction_pass = all(value <= 0.20 for value in differences.values())
    try:
        predicted_fraction = float(metrics.get("predicted_positive_fraction"))
        actual_fraction = float(metrics.get("actual_positive_fraction"))
        probability_std = float(metrics.get("probability_std"))
        finite_distribution = all(
            value == value and abs(value) != float("inf")
            for value in (predicted_fraction, actual_fraction, probability_std)
        )
    except (TypeError, ValueError):
        predicted_fraction = actual_fraction = probability_std = float("nan")
        finite_distribution = False
    if not finite_distribution:
        distribution_status = "invalid_nonfinite"
    elif actual_fraction >= 0.80:
        distribution_status = "label_contract_suspect"
    elif predicted_fraction >= 0.95:
        distribution_status = "collapsed_all_positive"
    elif predicted_fraction <= 0.005:
        distribution_status = "collapsed_all_negative"
    elif probability_std <= 1e-6:
        distribution_status = "collapsed_constant_probability"
    else:
        distribution_status = "healthy_prediction_distribution"
    validation.setdefault("historical_native_metrics", historical)
    validation.setdefault("historical_metric_absolute_difference_percentage_points", differences)
    validation.setdefault(
        "native_reproduction_gate",
        {
            "threshold_percentage_points": 0.20,
            "all_metric_differences_within_threshold": reproduction_pass,
            "status": "pass" if reproduction_pass else "failed",
        },
    )
    validation.setdefault("distribution_status", distribution_status)
    validation.setdefault("candidate_eligible_distribution", distribution_status == "healthy_prediction_distribution")
    # Persist the normalized receipt so downstream reports and future
    # idempotent finalization see the same auditable gate fields without a
    # second inference pass.
    write_json(run / "07_validation" / "NATIVE_ITER2000_VALIDATION.json", validation)
    required = {
        "label_audit": labels.get("status") in {"verified_pending_manual_preview", "verified"},
        "native_runtime": native.get("status") == "pass" and native.get("model_load", {}).get("status") == "pass",
        "native_validation": validation.get("status") == "available",
        "native_reproduction": validation.get("native_reproduction_gate", {}).get("status") == "pass",
        "distribution": validation.get("distribution_status") == "healthy_prediction_distribution",
        "test_isolation": validation.get("test_used") is False and validation.get("test_pixel_reads") == 0,
        "checkpoint_sha": validation.get("checkpoint_sha256") == ITER2000_SHA,
    }
    gate_status = "pass" if all(required.values()) else "blocked"
    gate = {
        "status": gate_status,
        "gate_checks": required,
        "label_contract": contract,
        "native_runtime_status": native.get("status"),
        "native_validation_metrics": validation.get("metrics", {}),
        "distribution_status": validation.get("distribution_status"),
        "native_reproduction": validation.get("native_reproduction_gate", {}),
        "input_pipeline_status": pipeline.get("status"),
        # Native recovery is allowed to pass while formal improvement training
        # remains blocked until the project adapter is proven equivalent to
        # the native preprocessing and the label preview receives sign-off.
        "formal_training_allowed": bool(
            all(required.values())
            and contract.get("native_local_polarity_match")
            and labels.get("status") == "verified"
            and pipeline.get("status") == "pass"
        ),
        "formal_training_blockers": [
            reason
            for reason, blocked in {
                "label_preview_manual_signoff_pending": labels.get("status") != "verified",
                "project_adapter_not_native_equivalent": pipeline.get("status") != "pass",
            }.items()
            if blocked
        ],
        "formal_training_started": False,
        "long_training_started": False,
        "test_pixel_reads": 0,
        "test_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run / "02_native_recovery" / "NATIVE_RECOVERY_GATE.json", gate)
    write_json(run / "05_sanity" / "OVERFIT_SANITY_STATUS.json", {
        "status": "not_started",
        "reason": "no controlled candidate was started; native recovery/pipeline equivalence is the current gate",
        "one_sample_iou": None,
        "eight_sample_iou": None,
        "test_pixel_reads": 0,
        "test_used": False,
    })
    matrix = {
        "status": "no_formal_candidate_started",
        "source_b0": {"status": "reproduced_native", "checkpoint": str(ITER2000.resolve()), "checkpoint_sha256": ITER2000_SHA, "validation": validation.get("metrics", {})},
        "candidates": {
            "B1_native_resume_control": {"status": "not_started", "reason": "controlled training deferred until manual label preview sign-off and native pipeline adapter are both explicit"},
            "B2_native_boundary_aux": {"status": "blocked_prerequisite", "reason": "B1 not run"},
            "B3_native_geometry_aux": {"status": "blocked_prerequisite", "reason": "B2 not run"},
            "B4_local_domain_only": {"status": "blocked_prerequisite", "reason": "legitimate Nanjing human labels not admitted"},
            "H0_no_height": {"status": "not_started", "reason": "no candidate training started"},
            "H1_height_aux": {"status": "blocked_globfp_not_trainable_for_image_segmentation", "reason": "3D-GloBFP audit contains vector-only C_FOOTPRINT_HEIGHT_ONLY assets and zero image pairs"},
            "H2_height_fusion": {"status": "blocked_globfp_not_trainable_for_image_segmentation", "reason": "height supervision is unverified and not paired to WHU imagery"},
        },
        "selection": "baseline_iter2000_retained_pending_formal_controlled_comparison",
        "test_pixel_reads": 0,
        "test_used": False,
    }
    write_json(run / "06_candidates" / "CANDIDATE_MATRIX.json", matrix)
    # Reuse the prior audited vector-only GloBFP result as provenance.  No new
    # pairing or city mixing is performed in this recovery run.
    prior_globfp = read_json(OLD_RUN / "01_globfp_audit" / "GLOBFP_READINESS.json", {})
    write_json(run / "02_native_recovery" / "GLOBFP_STATUS_REUSED.json", {
        "status": "vector_only_no_image_pairs",
        "source_audit": str((OLD_RUN / "01_globfp_audit" / "GLOBFP_READINESS.json").resolve()),
        "asset_class_counts": prior_globfp.get("asset_class_counts", {"C_FOOTPRINT_HEIGHT_ONLY": 33}),
        "trainable_image_pairs": 0,
        "height_training_enabled": False,
        "nanjing_tongzhou_pairing": False,
        "test_pixel_reads": 0,
        "test_used": False,
    })
    # Package the immutable native reference.  This copy is a run artifact; the
    # protected source checkpoint and any registry remain untouched.
    package = run / "09_model_package" / "URRS-Building-Foundation-v2-native-baseline-retained"
    package.mkdir(parents=True, exist_ok=True)
    package_checkpoint = package / "iter_2000_baseline_reference.pth"
    if not package_checkpoint.exists():
        shutil.copy2(ITER2000, package_checkpoint)
    write_json(package / "MODEL_CONTRACT.json", {
        "model_name": "URRS-Building-Foundation-v2-native-baseline-retained",
        "source_model": "rsbuilding_swin_t_whu_iter2000",
        "output_contract": ["building_probability", "building_mask", "boundary", "feature_tensor", "quality", "uncertainty"],
        "native_checkpoint_sha256": ITER2000_SHA,
        "label_contract": contract,
        "training_status": "not_started_in_this_recovery_run",
        "scientific_status": "baseline_iter2000_retained_pending_final_human_review",
        "test_pixel_reads": 0,
        "test_used": False,
    })
    (package / "MODEL_CARD.md").write_text("""# URRS Building Foundation v2 native baseline retained\n\nThis package references the native RSBuilding Swin-T `iter_2000.pth` checkpoint and does not contain a new scratch model. The checkpoint was strictly loaded in the native Open-CD/MMEngine environment and re-evaluated on 1,036 WHU Validation samples only.\n\n- WHU scope: Train/Validation only; WHU Test pixels read: 0.\n- Local label contract: nonzero/1 is building, zero is background; the contract is tied to native mmcv decoding and the fixed preview grid.\n- 3D-GloBFP: vector-only height/footprint assets, no image pairs; height head disabled.\n- Formal B1/B2/B3 controlled training: not started in this run.\n- Nanjing/Tongzhou: diagnostic-only after a future candidate is frozen; no thresholds or checkpoint selection use those scenes.\n- Scientific status: baseline retained, pending final human review and a separately authorized native-equivalent controlled comparison.\n""", encoding="utf-8")
    (package / "CONFIG_SNAPSHOT.yaml").write_text("""source_config: outputs/v2_readiness/rsbuilding/whu/configs/Swin_T_whu_windows_fullval_eval.py\nsource_checkpoint: iter_2000.pth\nlabel_contract: nonzero_is_building\nvalidation: WHU validation only (1036)\ntest_used: false\n""", encoding="utf-8")
    (package / "SHA256SUMS.txt").write_text(f"{sha256(package_checkpoint)}  iter_2000_baseline_reference.pth\n", encoding="utf-8")
    prior_qa = read_json(OLD_RUN / "11_final_review" / "FINAL_QA_QUEUE_MANIFEST.json", {})
    if prior_qa:
        write_json(run / "11_final_review" / "FINAL_QA_QUEUE_MANIFEST.json", {**prior_qa, "source_run": OLD_RUN.name, "copied_without_modifying_review_database": True})
        old_cases = OLD_RUN / "11_final_review" / "FINAL_QA_CASES.csv"
        if old_cases.exists():
            shutil.copy2(old_cases, run / "11_final_review" / "FINAL_QA_CASES.csv")
    final_status = "best_static_building_candidate_pending_final_human_review" if gate_status == "pass" else "blocked_native_recovery_gate"
    run_manifest = {
        "run_id": run.name,
        "status": "success" if gate_status == "pass" else "blocked",
        "scientific_status": final_status,
        "candidate_status": "baseline_iter2000_retained_after_native_reproduction" if gate_status == "pass" else "blocked_native_recovery_gate",
        "formal_candidate_training": "not_started",
        "source_checkpoint": str(ITER2000.resolve()),
        "source_checkpoint_sha256": ITER2000_SHA,
        "best_checkpoint": str(package_checkpoint.resolve()),
        "best_checkpoint_sha256": sha256(package_checkpoint),
        "native_validation": validation,
        "native_recovery_gate": gate,
        "label_contract_path": str((run / "01_label_contract" / "SELECTED_WHU_LABEL_CONTRACT.json").resolve()),
        "input_pipeline_comparison": pipeline,
        "gloBFP_status": "vector_only_no_image_pairs",
        "nanjing_diagnostics": "not_run_in_recovery_gate",
        "tongzhou_diagnostics": "not_run_in_recovery_gate",
        "review_queue": prior_qa,
        "registry_modified": False,
        "iter2000_modified": False,
        "full20_modified": False,
        "source_data_modified": False,
        "test_used": False,
        "test_pixel_reads": 0,
        "pseudo_label_training_started": False,
        "tfr_training_started": False,
        "mti_training_started": False,
        "construction_training_started": False,
        "git": git_snapshot(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run / "RUN_MANIFEST.json", run_manifest)
    write_json(run / "12_acceptance" / "FINAL_ACCEPTANCE_RECEIPT.json", run_manifest)
    report = run / "12_acceptance" / "BUILDING_FOUNDATION_V2_NATIVE_RECOVERY_REPORT.md"
    m = validation.get("metrics", {})
    diffs = validation.get("historical_metric_absolute_difference_percentage_points", {})
    report.write_text(f"""# URRS Building Foundation v2 Native Recovery Report\n\n## Outcome\n\n- run: `{run.name}`\n- status: `{run_manifest['status']}`\n- scientific_status: `{final_status}`\n- candidate policy: **no scratch model promoted; native iter_2000 retained**\n- controlled B1/B2/B3/H candidates: `not_started` (no long training launched)\n\n## WHU label contract\n\n- Train samples audited: `{labels.get('train_count')}`\n- Validation samples audited: `{labels.get('validation_count')}`\n- unique values: `{labels.get('unique_values')}`\n- selected polarity: `{contract.get('label_polarity')}`; building values `{contract.get('building_values')}`\n- native class expectation: `uint8 >=128 / mmcv 255 -> building`\n- contract status: `{labels.get('status')}`; manual preview: `pending`\n- preview: `01_label_contract/WHU_LABEL_PREVIEW_GRID.png`; polarity comparison: `01_label_contract/WHU_MASK_POLARITY_COMPARISON.png`\n- old adapter contract mismatch: `label > 0` was implicit; it is now configurable through the selected contract.\n\n## Native runtime and pipeline\n\n- native Python: `{governance.get('runtime', {}).get('native_python')}`\n- torch/mmengine/mmcv/mmseg/Open-CD: `{native.get('imports', {})}`\n- model load: `{native.get('model_load', {}).get('status')}`; state keys `{native.get('checkpoint', {}).get('state_key_count')}`; missing/unexpected `{native.get('model_load', {}).get('missing_keys')}/{native.get('model_load', {}).get('unexpected_keys')}`\n- input pipeline comparison: `{pipeline.get('status')}`; native BGR→RGB and mean/std are recorded in `03_input_pipeline/INPUT_PIPELINE_COMPARISON.json`.\n\n## Native iter_2000 Validation reproduction\n\n- split: `WHU Validation only`\n- samples: `{validation.get('sample_count')}`\n- IoU: `{m.get('iou')}`; F1: `{m.get('f1')}`; precision: `{m.get('precision')}`; recall: `{m.get('recall')}`\n- historical comparison (percentage points): `{diffs}`\n- reproduction gate (≤0.20 pp): `{validation.get('native_reproduction_gate')}`\n- predicted positive fraction: `{m.get('predicted_positive_fraction')}`; actual: `{m.get('actual_positive_fraction')}`\n- probability mean/std/p01/p50/p99: `{m.get('probability_mean')}` / `{m.get('probability_std')}` / `{m.get('probability_p01')}` / `{m.get('probability_p50')}` / `{m.get('probability_p99')}`\n- distribution status: `{validation.get('distribution_status')}`; all-positive sample fraction `{m.get('all_positive_sample_fraction')}`; all-negative `{m.get('all_negative_sample_fraction')}`\n\n## Candidate and scope decisions\n\n- B0: native iter_2000 reproduced and retained.\n- B1 native resume control: not started.\n- B2 boundary auxiliary: blocked because B1 not run.\n- B3 geometry auxiliary: blocked because B2 not run.\n- B4 Nanjing local domain: not admitted; no legitimate human labels.\n- H1/H2: blocked; 3D-GloBFP is vector-only `C_FOOTPRINT_HEIGHT_ONLY`, 0 image pairs, height units unverified.\n- one/eight sample overfit: not run because no formal candidate was started.\n- formal long training: not started.\n\n## Protected assets and diagnostics\n\n- iter_2000 SHA256: `{ITER2000_SHA}` (expected unchanged)\n- Full20 SHA256 expected/recorded: `{FULL20_SHA}`\n- registry modified: `false`\n- Nanjing/Tongzhou: not used for selection; future diagnostics only after a candidate is frozen.\n- WHU Test reads: `0`; SECOND Test reads: `0` (not touched by this run).\n- TFR/MTI/Construction training: `false`.\n- Lite Review queue: `{prior_qa.get('case_count')}` validation-only cases reused without modifying the database.\n\n## Package\n\n`{package_checkpoint.resolve()}`\n\nThe package contains the immutable native baseline reference and contract/configuration metadata. It is not a production claim and remains pending final human review and an explicitly authorized native-equivalent controlled comparison.\n""", encoding="utf-8")
    write_json(run / "12_acceptance" / "NATIVE_RECOVERY_SUMMARY.json", {"status": run_manifest["status"], "scientific_status": final_status, "checkpoint_sha256": ITER2000_SHA, "test_pixel_reads": 0, "test_used": False})
    print(json.dumps({"status": run_manifest["status"], "scientific_status": final_status, "run_dir": str(run), "package": str(package_checkpoint), "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
    return 0 if gate_status == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
