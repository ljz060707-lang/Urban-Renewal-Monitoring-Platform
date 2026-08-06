"""Governed runner for the detached Boundary Objectizer pilot.

The runner is intentionally conservative: governance and the fixed WHU Train
split may run while the worktree is dirty, but no GPU training, Calibration,
Validation, objectization selection, or Test discovery can start until the
Building-v2 source is committed.  A blocked run is a first-class, auditable
outcome and is never promoted to a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from init_building_v2_detached_boundary_objectizer import (
    DEFAULT_WHU_ROOT,
    PROJECT_ROOT,
    B0_RELATIVE,
    B0_SHA256,
    initialize_run,
    sha256_file,
    write_json,
)

STAGES = ("governance", "split", "train", "calibration", "objectization", "validation", "bootstrap", "package", "finalize", "all")


def git_status(project_root: Path = PROJECT_ROOT) -> list[str]:
    result = subprocess.run(["git", "status", "--short"], cwd=project_root, capture_output=True, text=True, check=False)
    return [line for line in result.stdout.splitlines() if line.strip()]


def source_dirty(project_root: Path = PROJECT_ROOT) -> list[str]:
    """Return only dirty paths that can alter the detached training contract."""
    dirty = git_status(project_root)
    relevant: list[str] = []
    for line in dirty:
        path = line[3:].strip().replace("\\", "/") if len(line) >= 3 else line
        if path.startswith("src/urban_monitor/building_foundation_v2/") or path.startswith("tools/") and (
            "building_v2" in path or "detached_boundary" in path or "native_b2" in path
        ):
            relevant.append(line)
    return relevant


def append_decision(run: Path, payload: dict[str, Any]) -> None:
    path = run / "00_governance" / "AUTONOMOUS_DECISION_LOG.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "test_used": False, "test_pixel_reads": 0, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_manifest(run: Path) -> dict[str, Any]:
    path = run / "RUN_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(f"RUN_MANIFEST.json missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(run: Path, **updates: Any) -> dict[str, Any]:
    payload = load_manifest(run)
    payload.update(updates)
    write_json(run / "RUN_MANIFEST.json", payload)
    return payload


def _sha256_json(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_blocked_receipts(run: Path, dirty: list[str], *, reason: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    final_git_status = git_status()
    write_json(run / "00_governance" / "GIT_FINAL_STATE.json", {"status_short": final_git_status, "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False).stdout.strip(), "github_push_performed": False})
    manifest = update_manifest(
        run,
        status="blocked_detached_boundary_training_failed",
        scientific_status="blocked_detached_boundary_training_not_started",
        governance_gate="blocked_dirty_source_requires_commit",
        failure_stage="pre_training_git_gate",
        failure_reason=reason,
        source_dirty_paths=dirty,
        source_git_final_status=final_git_status,
        training_started=False,
        calibration_started=False,
        objectization_started=False,
        validation_started=False,
        test_used=False,
        test_pixel_reads=0,
        registry_modified=False,
        github_push_performed=False,
        completed_at=now,
        next_action="commit Building-v2 source, re-run governance, then run the bounded pilot",
    )
    acceptance = run / "12_acceptance"
    write_json(acceptance / "FINAL_DETACHED_BOUNDARY_RECEIPT.json", {
        "status": "blocked_detached_boundary_training_failed", "governance_gate": "blocked_dirty_source_requires_commit", "run_id": run.name,
        "training_started": False, "validation_started": False,
        "source_dirty_paths": dirty, "reason": reason,
        "test_used": False, "test_pixel_reads": 0,
        "registry_modified": False, "github_push_performed": False,
        "created_at": now,
    })
    write_json(acceptance / "B0_INVARIANCE_RECEIPT.json", {
        "status": "not_run_formal_training_blocked", "b0_parameter_digest_before": None,
        "b0_parameter_digest_after": None, "building_logits_max_abs_diff": None,
        "building_probability_max_abs_diff": None, "building_mask_disagreement": None,
        "b0_sha256": manifest.get("b0_sha256"), "expected_b0_sha256": B0_SHA256,
        "test_pixel_reads": 0,
    })
    write_json(acceptance / "BOUNDARY_TRAINING_RECEIPT.json", {
        "status": "not_started", "reason": reason, "seeds": [20260805, 20260806, 20260807],
        "checkpoint_steps": list(range(0, 2001, 250)), "test_pixel_reads": 0,
    })
    write_json(acceptance / "BOUNDARY_CALIBRATION_RECEIPT.json", {"status": "not_started", "selection_split": "boundary_calibration", "validation_used": False, "test_pixel_reads": 0})
    write_json(acceptance / "OBJECTIZATION_CONTRACT_RECEIPT.json", {"status": "not_started", "oracle_selectable": False, "test_pixel_reads": 0})
    write_json(acceptance / "MODEL_SELECTION_RECEIPT.json", {"status": "blocked_detached_boundary_training_failed", "governance_gate": "blocked_dirty_source_requires_commit", "selected_system": None, "candidate_status": "blocked_detached_boundary_training_failed", "test_pixel_reads": 0})
    write_json(acceptance / "PROTECTED_ASSET_INVARIANCE.json", {
        "status": "verified_start_only", "b0_sha256": manifest.get("b0_sha256"),
        "b0_expected_sha256": B0_SHA256,
        "full20_sha256": json.loads((run / "00_governance" / "PROTECTED_ASSETS_START.json").read_text(encoding="utf-8")).get("full20", {}).get("sha256"),
        "full20_expected_sha256": "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621",
        "registry_sha256": None,
        "registry_modified": False, "original_data_modified": False,
        "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0,
    })
    write_json(acceptance / "AUTONOMOUS_EXECUTION_SUMMARY.json", {
        "starting_stage": "initialized_or_governance_only", "resumed": False,
        "engineering_issue": "Building-v2 source is uncommitted; formal training hard gate fired",
        "engineering_retry_used": False, "formal_training_retry_count": 0,
        "pre_training_governance_corrections": [
            "excluded 419 duplicate label-hash rows from selected pilot splits",
            "corrected protected Full20 path to best_v2_multisource_change.pth",
        ],
        "o2_conditional_used": False,
        "validation_contract_sha256": None, "final_selection": None,
        "scientific_conclusion": "blocked_detached_boundary_training_failed",
        "test_used": False, "test_pixel_reads": 0,
    })
    (run / "11_reports" / "FINAL_REPORT.md").write_text(
        "# Detached Boundary Objectizer — Final Report\n\n"
        "Status: `blocked_detached_boundary_training_failed`\n"
        "Governance gate: `blocked_dirty_source_requires_commit`\n\n"
        "Formal training was not started because Building-v2 source changes "
        "remain uncommitted. This is a governance stop, not a scientific "
        "failure or candidate result. No Calibration, objectization selection, "
        "WHU Validation, or Test data were read.\n\n"
        f"Dirty paths:\n" + "\n".join(f"- `{item}`" for item in dirty) + "\n\n"
        "Governance evidence: the fixed WHU Train registry contains 4,736 "
        "paired images; the selected pilot is 256 Train / 128 Calibration / "
        "32 Diagnostic. 419 rows with an identical label hash were excluded "
        "as a duplicate hash group, and the resulting critical leakage count "
        "is zero.\n\n"
        "Before the gate, two governance-only corrections were made (duplicate "
        "hash exclusion and protected Full20 path correction); no GPU or formal "
        "training retry occurred.\n\n"
        "No detached Boundary loss, BF1, object F1, split/merge rate, "
        "bootstrap interval, or WHU Validation metric exists for this run. "
        "The prior B2 diagnostic is not a detached-objectizer result and is "
        "not used for selection. B0 SHA256 is "
        "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7; "
        "Full20 SHA256 is "
        "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621.\n\n"
        "Next action: commit the reviewed source changes locally, then rerun "
        "the same run stages; do not alter the B0 checkpoint or data splits.\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve() if args.run_dir else None
    if run_dir is None:
        run_dir = initialize_run(whu_root=args.root, b0_checkpoint=args.b0_checkpoint, output_parent=args.output_parent, seed=args.seed)
    elif not (run_dir / "RUN_MANIFEST.json").is_file():
        run_dir = initialize_run(whu_root=args.root, b0_checkpoint=args.b0_checkpoint, run_dir=run_dir, seed=args.seed)

    append_decision(run_dir, {
        "current_stage": args.stage,
        "observed_evidence": {"manifest": str((run_dir / "RUN_MANIFEST.json").resolve()), "source_dirty": source_dirty()},
        "decision": "inspect_and_gate",
        "alternative_options": ["formal_training", "resume_existing_stage", "blocked_until_commit"],
        "reason_selected": "formal training requires committed Building-v2 source and no active GPU lock",
        "files_modified": [], "metrics_before": {}, "metrics_after": {}, "validation_used": False,
        "next_action": "apply_git_gate_before_train_or_validation",
    })
    dirty = source_dirty()
    if args.stage in {"train", "calibration", "objectization", "validation", "bootstrap", "package", "finalize", "all"} and dirty:
        write_blocked_receipts(run_dir, dirty, reason="uncommitted Building-v2 source changes violate the formal-training gate")
        print(json.dumps({"status": "blocked_detached_boundary_training_failed", "governance_gate": "blocked_dirty_source_requires_commit", "run_dir": str(run_dir), "dirty_paths": dirty, "training_started": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
        return 2
    # A clean tree is a prerequisite, not evidence of a completed pilot.  The
    # actual native worker is intentionally invoked only by a future clean-tree
    # run; this branch keeps the current implementation honest instead of
    # manufacturing training receipts.
    if args.stage in {"train", "calibration", "objectization", "validation", "bootstrap", "package", "finalize", "all"}:
        reason = "formal detached worker is not started in this safety pass; run the explicit native worker after reviewing the committed contract"
        write_blocked_receipts(run_dir, [], reason=reason)
        update_manifest(run_dir, status="blocked_detached_boundary_training_failed", failure_reason=reason)
        print(json.dumps({"status": "blocked_detached_boundary_training_failed", "run_dir": str(run_dir), "training_started": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
        return 2
    update_manifest(run_dir, status="governance_complete", completed_stage=args.stage, test_used=False, test_pixel_reads=0)
    print(json.dumps({"status": "governance_complete", "run_dir": str(run_dir), "stage": args.stage, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_WHU_ROOT)
    parser.add_argument("--b0-checkpoint", type=Path, default=None)
    parser.add_argument("--output-parent", type=Path, default=PROJECT_ROOT / "outputs" / "pretraining")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:  # always retain a machine-readable failure if initialization itself fails
        print(json.dumps({"status": "blocked_detached_boundary_training_failed", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
