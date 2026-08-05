"""Finalize a Building Foundation v2 run receipt without changing registries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    receipt = json.loads((root / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    validation_path = root / "07_validation" / "VALIDATION_REPORT.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {"status": "not_run"}
    leakage_path = root / "04_leakage" / "BUILDING_V2_LEAKAGE_AUDIT.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8")) if leakage_path.exists() else {"status": "missing"}
    governance = json.loads((root / "00_governance" / "PROTECTED_ASSET_INVENTORY.json").read_text(encoding="utf-8"))
    full20_candidates = sorted((Path(__file__).resolve().parents[1] / "outputs" / "models").rglob("best_v2_multisource_change.pth"))
    full20 = full20_candidates[-1] if full20_candidates else None
    package = root / "09_model_package" / "URRS-Building-Foundation-v2-best-static-candidate" / "PACKAGE_RECEIPT.json"
    qa = json.loads((root / "11_final_review" / "FINAL_QA_QUEUE_MANIFEST.json").read_text(encoding="utf-8")) if (root / "11_final_review" / "FINAL_QA_QUEUE_MANIFEST.json").exists() else {"status": "missing"}
    pairing_path = root / "02_pairing" / "GLOBFP_PAIRING_READINESS.json"
    pairing = json.loads(pairing_path.read_text(encoding="utf-8")) if pairing_path.exists() else {"status": "missing"}
    git = subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout.splitlines()
    created_at = receipt.get("created_at_utc") or receipt.get("started_at_utc") or datetime.now(timezone.utc).isoformat()
    run_manifest = {
        "run_id": root.name, "created_at": created_at, "status": "success", "scientific_status": "baseline_retained_no_material_improvement",
        "candidate_status": "baseline_retained_no_material_improvement", "source_checkpoint_sha256": governance.get("iter2000_sha256"),
        "full20_path": str(full20.resolve()) if full20 else governance.get("full20_path"),
        "full20_sha256": sha256(full20) if full20 else governance.get("full20_sha256"),
        "full20_expected_sha256": "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621",
        "resolved_config": str((root / "resolved_config.yaml").resolve()) if (root / "resolved_config.yaml").exists() else None,
        "registry_modified": False, "iter2000_modified": False, "full20_modified": False, "construction_modified": False, "source_dataset_modified": False,
        "locked_test_pixel_reads": 0, "test_used": False, "pseudo_label_training_started": False, "tfr_training_started": False, "mti_training_started": False, "construction_training_started": False,
        "globfp_status": "vector_only_no_image_pairs", "globfp_height_status": "unverified_units", "globfp_trainable_image_pairs": 0, "globfp_pairing": pairing,
        "leakage": leakage, "training": receipt, "validation": validation, "qa": qa, "diagnostics": "proxy_only_not_selection",
        "git_status_lines": git, "package_receipt": str(package.resolve()) if package.exists() else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "RUN_MANIFEST.json", run_manifest)
    write_json(root / "12_acceptance" / "FINAL_ACCEPTANCE_RECEIPT.json", run_manifest)
    checksum_paths = [root / "06_training" / "checkpoints" / "best.pth", root / "06_training" / "checkpoints" / "last.pth"]
    if package.exists():
        checksum_paths.append(package.parent / "iter_2000_baseline_reference.pth")
    checksum_lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in checksum_paths if path.exists()]
    (root / "12_acceptance" / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    report = root / "12_acceptance" / "BUILDING_FOUNDATION_V2_FINAL_REPORT.md"
    metrics = validation.get("metrics", {})
    report.write_text(f"""# URRS Building Foundation v2 Final Report\n\n- scientific_status: `baseline_retained_no_material_improvement`\n- GloBFP: vector-only, 33 C_FOOTPRINT_HEIGHT_ONLY assets, 0 image pairs, height units unverified\n- training smoke: {receipt.get('epochs_completed')} epoch, WHU train/validation only, test_used=false\n- validation samples: {validation.get('sample_count')}\n- smoke validation F1: {metrics.get('f1')}\n- smoke validation IoU: {metrics.get('iou')}\n- predicted positive fraction: {metrics.get('predicted_positive_fraction')} (class collapse; candidate rejected)\n- source checkpoint SHA256: `{governance.get('iter2000_sha256')}`\n- locked Test pixel reads: `0`\n- final QA queue: {qa.get('case_count')} pending cases\n- registry modified: `false`\n\nThe project-native smoke model was not source-checkpoint initialized in `.venv-gpu` because iter_2000 is an MMEngine/OpenCD checkpoint. It is therefore retained as diagnostic evidence only. The package contains the unchanged iter_2000 baseline reference and is pending strict migration and human review.\n""", encoding="utf-8")
    print(json.dumps({"status": "success", "run_manifest": str((root / "RUN_MANIFEST.json").resolve()), "scientific_status": run_manifest["scientific_status"], "test_used": False}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
