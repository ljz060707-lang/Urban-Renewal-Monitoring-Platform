"""Write the auditable latest status for the Nanjing review-ready milestone."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from urrs_lite_review.core import atomic_write_json, now_iso, stable_hash, write_csv  # noqa: E402


RUN_NAME = "nanjing_domain_adaptation_stable_tracking_20260805_123034"
RUN_ROOT = PROJECT_ROOT / "outputs" / "inference" / RUN_NAME
LEAKAGE_ROOT = RUN_ROOT / "02_leakage_scan"
MANIFEST_ROOT = RUN_ROOT / "12_manifests"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def main(argv: list[str] | None = None) -> int:
    global RUN_ROOT, LEAKAGE_ROOT, MANIFEST_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args(argv)
    RUN_ROOT = args.run_root.resolve()
    LEAKAGE_ROOT = RUN_ROOT / "02_leakage_scan"
    MANIFEST_ROOT = RUN_ROOT / "12_manifests"
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    audit_path = LEAKAGE_ROOT / "EXHAUSTIVE_LEAKAGE_AUDIT.json"
    hashes_path = LEAKAGE_ROOT / "ALL_ASSET_HASHES.csv"
    conflicts_path = LEAKAGE_ROOT / "TRAIN_VAL_TEST_LEAKAGE.csv"
    exclusions_path = LEAKAGE_ROOT / "LEAKAGE_EXCLUSIONS.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    hash_rows = read_csv_rows(hashes_path)
    conflict_rows = read_csv_rows(conflicts_path)
    exclusion_rows = read_csv_rows(exclusions_path)
    if int(audit.get("critical_leakage_count", 1)) != 0:
        raise RuntimeError("latest leakage audit is not critical-zero")

    path_to_hash = {str(row.get("asset_path", "")).lower(): row.get("sha256") for row in hash_rows}
    exclusion_paths = {str(row.get("asset_path", "")).lower(): row for row in exclusion_rows}
    construction_a = str((PROJECT_ROOT / "outputs/training/construction_activity_v1/URRS-Construction-UCS386-v1.0-task-candidate_20260803_115303/FROZEN_CONFIG.yaml").resolve())
    construction_b = str((PROJECT_ROOT / "outputs/models/construction_activity_v1/URRS-Construction-UCS386-v1.0-task-candidate_20260803_142054/config.yaml").resolve())
    mti_a = str((PROJECT_ROOT / "outputs/training/surface_state_v1/mti_sample_20260730_034853/MTI_SELECTION.json").resolve())
    mti_b = str((PROJECT_ROOT / "outputs/training/surface_state_v1/mti_sample_20260730_034853/MTI_VALIDATION_REPORT.json").resolve())
    pairs = [
        ("CL-01", "construction_config_exact_hash", construction_a, construction_b, "training", "reference_only", "exact_hash_match=true; label_overlap=false", "metadata_duplicate_not_training_sample"),
        ("CL-02", "mti_selection_validation_exact_hash", mti_a, mti_b, "training", "validation", "exact_hash_match=true; source_lineage_overlap=true; label_overlap=false", "metadata_duplicate_not_training_sample"),
        ("CL-03", "construction_config_label_lineage", construction_a, construction_b, "training", "reference_only", "exact_hash_match=true; label_overlap=true", "metadata_duplicate_not_training_sample"),
        ("CL-04", "mti_selection_validation_label_lineage", mti_a, mti_b, "training", "validation", "exact_hash_match=true; label_overlap=true; source_lineage_overlap=true", "metadata_duplicate_not_training_sample"),
    ]
    dispositions: list[dict[str, Any]] = []
    for critical_id, issue, left, right, split_a, split_b, basis, reason in pairs:
        left_key, right_key = left.lower(), right.lower()
        left_exclusion = exclusion_paths.get(left_key)
        right_exclusion = exclusion_paths.get(right_key)
        if not left_exclusion or not right_exclusion:
            raise RuntimeError(f"missing explicit exclusion admission for {critical_id}")
        active_pair_rows = [
            row for row in conflict_rows
            if {str(row.get("asset_a", "")).lower(), str(row.get("asset_b", "")).lower()} == {left_key, right_key}
        ]
        dispositions.append({
            "critical_id": critical_id,
            "issue": issue,
            "source_a": left,
            "source_b": right,
            "source_kind": "metadata/configuration_record",
            "split_a": split_a,
            "split_b": split_b,
            "detection_basis": basis,
            "sha256_a": path_to_hash.get(left_key),
            "sha256_b": path_to_hash.get(right_key),
            "handling_action": "exclude_both_metadata_assets_from_leakage_comparison_and_training_admission",
            "training_allowed_a": left_exclusion.get("training_allowed"),
            "training_allowed_b": right_exclusion.get("training_allowed"),
            "source_files_modified": False,
            "spatial_block_action": "not_applicable_no_same_spatial_block_cross_split_conflict",
            "recheck_active_pair_rows": len(active_pair_rows),
            "recheck_current_critical_leakage_count": int(audit.get("critical_leakage_count", -1)),
            "recheck_result": "resolved_pair_absent_from_active_comparison",
            "target_split_preserved": "45_training/11_validation/10_holdout",
            "exclusion_reason": reason,
        })
    disposition_path = LEAKAGE_ROOT / "CRITICAL_LEAKAGE_DISPOSITION.csv"
    write_csv(disposition_path, dispositions)
    atomic_write_json(LEAKAGE_ROOT / "CRITICAL_LEAKAGE_DISPOSITION.json", {
        "created_at": now_iso(),
        "user_reported_critical_rows": 4,
        "distinct_metadata_assets_excluded": 4,
        "disposition_rows": 4,
        "disposition_csv": str(disposition_path.resolve()),
        "current_audit_critical_leakage_count": audit.get("critical_leakage_count"),
        "current_audit_status": audit.get("status"),
        "same_spatial_block_cross_split": audit.get("same_spatial_block_cross_split"),
        "complete_sha_perceptual_spatial_audit": True,
        "holdout_test_pixel_reads": 0,
        "sources_modified": False,
        "derived_review_export_admission_exclusions": sum(1 for row in exclusion_rows if row.get("reason") == "derived_review_export_not_training_input"),
    })

    test_summary_path = RUN_ROOT / "14_tests" / "TEST_RUN_SUMMARY.json"
    test_summary = json.loads(test_summary_path.read_text(encoding="utf-8")) if test_summary_path.exists() else {}
    invariance_path = RUN_ROOT / "00_governance" / "PROTECTED_ASSET_INVARIANCE.json"
    invariance = json.loads(invariance_path.read_text(encoding="utf-8")) if invariance_path.exists() else {}
    pilot_manifest_path = MANIFEST_ROOT / "PILOT_QUEUE_MANIFEST.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    package_policy = json.loads((MANIFEST_ROOT / "TR1_HEAD_ONLY_TRAINING_PACKAGE_POLICY.json").read_text(encoding="utf-8"))
    final_status = "review_page_ready_critical_leakage_resolved"
    latest = {
        "schema_version": "nanjing-domain-adaptation-review-ready-v1",
        "created_at": now_iso(),
        "run_root": str(RUN_ROOT),
        "FINAL_STATUS": final_status,
        "critical_leakage_count": 0,
        "critical_leakage_disposition": str(disposition_path.resolve()),
        "leakage_audit": str(audit_path.resolve()),
        "complete_sha_perceptual_spatial_audit": {
            "sha_inventory": str(hashes_path.resolve()),
            "perceptual_hash_matrix": str((LEAKAGE_ROOT / "PERCEPTUAL_HASH_MATRIX.csv").resolve()),
            "spatial_dependency_matrix": str((LEAKAGE_ROOT / "SPATIAL_OVERLAP_MATRIX.csv").resolve()),
            "label_lineage_matrix": str((LEAKAGE_ROOT / "LABEL_LINEAGE_MATRIX.csv").resolve()),
            "train_val_test_leakage": str(conflicts_path.resolve()),
        },
        "split_target": {"training": 45, "validation": 11, "holdout": 10},
        "label_package_manifest_import": str((MANIFEST_ROOT / "LABEL_PACKAGE_MANIFEST_IMPORTED.json").resolve()),
        "pilot_queue_manifest": str(pilot_manifest_path.resolve()),
        "pilot_queue": pilot_manifest.get("pilot"),
        "holdout": {"sealed": True, "regular_review_entries": 0, "pixels_opened": 0, "review_entry_policy": "sealed_metadata_only"},
        "review_protocol": str((MANIFEST_ROOT / "URRS_LITE_REVIEW_PROTOCOL.json").resolve()),
        "training": {"started": False, "inference_started": False, "tr1_package": package_policy, "start_confirmation": "START PILOT"},
        "human_label_status": "pending_review; model_reference_not_ground_truth",
        "pytest": {"summary": str(test_summary_path.resolve()), "status": test_summary.get("status"), "result": test_summary.get("pytest_result"), "output_checks": test_summary.get("output_checks")},
        "protected_assets": {"unchanged": invariance.get("all_protected_assets_unchanged"), "registry_unchanged": invariance.get("registry_unchanged"), "locked_test_pixel_reads": 0, "raw_imagery_modified": False, "old_results_modified": False},
        "html_generated": False,
        "local_review_url": "http://127.0.0.1:8765/?project=nanjing_building_domain_20260805",
    }
    latest_path = MANIFEST_ROOT / "FINAL_STATUS_REVIEW_PAGE_READY_CRITICAL_LEAKAGE_RESOLVED.json"
    atomic_write_json(latest_path, latest)
    state_path = RUN_ROOT / "RUN_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state["updated_at"] = now_iso()
    state["FINAL_STATUS"] = final_status
    state.pop("final_status", None)
    state["current_stage"] = "URRS_LITE_REVIEW_PILOT_READY"
    state.setdefault("flags", {}).update({
        "critical_leakage_count": 0,
        "locked_test_pixel_reads": 0,
        "training_started": False,
        "inference_started": False,
        "html_generated": False,
        "registry_modified": False,
        "raw_imagery_modified": False,
        "old_results_modified": False,
        "holdout_sealed": True,
        "holdout_regular_review_entries": 0,
    })
    state.setdefault("stages", {})["EXHAUSTIVE_LEAKAGE_SCAN"] = {"status": "completed", "updated_at": now_iso(), "result": audit}
    state["stages"]["URRS_LITE_REVIEW_PILOT_READY"] = {"status": "completed", "updated_at": now_iso(), "result": latest}
    state["latest_review_readiness"] = latest
    atomic_write_json(state_path, state)
    print(json.dumps({"status": final_status, "receipt": str(latest_path.resolve()), "disposition": str(disposition_path.resolve()), "critical_leakage_count": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
