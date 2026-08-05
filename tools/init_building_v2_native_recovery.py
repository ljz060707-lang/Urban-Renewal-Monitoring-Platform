"""Create a non-destructive Building Foundation v2 native-recovery run.

The run is deliberately created before any model work.  It records protected
assets, source control state, runtime paths, and the hard Test/pseudo-training
guards used by the subsequent audit tools.  It never traverses a Test folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITER2000_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
ITER2000 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return result.stdout.strip()

    return {
        "repository": str(PROJECT_ROOT),
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short").splitlines(),
        "diff_stat": run("diff", "--stat").splitlines(),
    }


def find_full20() -> Path | None:
    candidates = sorted((PROJECT_ROOT / "outputs" / "pretraining").glob("change_v2/*/checkpoints/best_v2_multisource_change.pth"))
    return candidates[-1] if candidates else None


def create_run(root: Path | None) -> Path:
    parent = PROJECT_ROOT / "outputs" / "pretraining"
    if root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = parent / f"building_foundation_v2_native_recovery_{stamp}"
        suffix = 1
        while root.exists():
            root = parent / f"building_foundation_v2_native_recovery_{stamp}_{suffix:02d}"
            suffix += 1
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing run: {root}")
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "00_governance", "01_label_contract", "02_native_recovery", "03_input_pipeline",
        "04_leakage", "05_sanity", "06_candidates", "07_validation", "08_diagnostics",
        "09_model_package", "10_figures", "11_final_review", "12_acceptance",
    ):
        (root / name).mkdir()
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--native-python", type=Path, default=Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe"))
    parser.add_argument("--project-python", type=Path, default=PROJECT_ROOT / ".venv-gpu" / "Scripts" / "python.exe")
    args = parser.parse_args()
    run = create_run(args.run_dir)
    full20 = find_full20()
    payload = {
        "run_id": run.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "git": git_snapshot(),
        "protected_assets": {
            "iter_2000_path": str(ITER2000.resolve()),
            "iter_2000_sha256": sha256(ITER2000),
            "iter_2000_expected_sha256": ITER2000_SHA,
            "iter_2000_unchanged_before_run": sha256(ITER2000) == ITER2000_SHA,
            "full20_path": str(full20.resolve()) if full20 else None,
            "full20_sha256": sha256(full20) if full20 else None,
            "full20_expected_sha256": FULL20_SHA,
            "full20_unchanged_before_run": bool(full20 and sha256(full20) == FULL20_SHA),
            "registry_modified": False,
            "source_data_modified": False,
            "native_environment_modified": False,
        },
        "runtime": {
            "project_python": str(args.project_python.resolve()),
            "project_python_exists": args.project_python.exists(),
            "native_python": str(args.native_python.resolve()),
            "native_python_exists": args.native_python.exists(),
            "gpu_lock": str((PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock").resolve()),
            "gpu_lock_required_for_gpu_stages": True,
        },
        "guards": {
            "test_pixel_reads": 0,
            "test_used": False,
            "pseudo_label_training_started": False,
            "tfr_training_started": False,
            "mti_training_started": False,
            "construction_training_started": False,
            "long_training_started": False,
        },
        "scope": {
            "dataset": "WHU_Building",
            "allowed_splits": ["train", "validation"],
            "expected_train_count": 4736,
            "expected_validation_count": 1036,
            "locked_test_count": "registered_only_not_read",
            "native_checkpoint": "RSBuilding iter_2000",
        },
        "status": "governance_created_label_and_native_audit_pending",
    }
    write_json(run / "00_governance" / "RUN_GOVERNANCE.json", payload)
    write_json(run / "00_governance" / "PROTECTED_ASSET_INVENTORY.json", payload["protected_assets"] | {"created_at_utc": payload["created_at_utc"]})
    write_json(run / "00_governance" / "RUNTIME_AUDIT.json", payload["runtime"] | {"created_at_utc": payload["created_at_utc"]})
    write_json(run / "RUN_MANIFEST.json", {"run_id": run.name, "status": "in_progress", "test_used": False, "test_pixel_reads": 0, "created_at_utc": payload["created_at_utc"]})
    print(json.dumps({"run_dir": str(run.resolve()), **payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
