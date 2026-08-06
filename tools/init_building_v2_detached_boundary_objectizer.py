"""Create an auditable detached-boundary objectizer run.

This initializer is deliberately CPU/read-only with respect to protected
assets.  It discovers only ``WHU_Building/train`` and writes explicit,
deterministic pilot registries.  Formal GPU training is gated by the
orchestrator when Building-v2 source changes are uncommitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHU_ROOT = Path(r"D:\Data Set\WHU_Building")
B0_SHA256 = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
B0_RELATIVE = Path("outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth")
FULL20_RELATIVE = Path("outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth")
B1_RELATIVE = Path("outputs/pretraining/building_foundation_v2_b1_native_pilot_20260805_200515/04_b1_training/b1/iter_500.pth")

RUN_DIRS = (
    "00_governance", "01_boundary_data_audit", "02_boundary_split",
    "03_b0_contract", "04_boundary_training", "05_calibration",
    "06_objectization", "07_validation", "08_bootstrap",
    "09_model_package", "10_diagnostics", "11_reports", "12_acceptance",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_state(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=False)
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "status_short": status,
        "diff_stat": run("diff", "--stat"),
        "clean": not bool(status),
        "github_push_performed": False,
    }


def make_run(output_parent: Path, timestamp: str | None = None) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run = output_parent / f"building_foundation_v2_detached_boundary_objectizer_{stamp}"
    suffix = 1
    while run.exists():
        run = output_parent / f"building_foundation_v2_detached_boundary_objectizer_{stamp}_{suffix:02d}"
        suffix += 1
    for name in RUN_DIRS:
        (run / name).mkdir(parents=True, exist_ok=True)
    return run


def write_native_list(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the four-column native Open-CD list without discovering Test."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['image_path']}\t**\t**\t{row['label_path']}\t**\n")


def initialize_run(
    *,
    whu_root: Path = DEFAULT_WHU_ROOT,
    b0_checkpoint: Path | None = None,
    output_parent: Path | None = None,
    run_dir: Path | None = None,
    seed: int = 20260805,
) -> Path:
    # Prefer this checkout over any older editable install in .venv-gpu.
    source_root = PROJECT_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from urban_monitor.building_foundation_v2.detached_split import (
        build_boundary_split,
        discover_whu_train,
        split_leakage_audit,
        write_registry,
    )

    root = Path(whu_root).resolve()
    b0 = Path(b0_checkpoint or (PROJECT_ROOT / B0_RELATIVE)).resolve()
    parent = Path(output_parent or (PROJECT_ROOT / "outputs" / "pretraining")).resolve()
    run = Path(run_dir).resolve() if run_dir else make_run(parent)
    for name in RUN_DIRS:
        (run / name).mkdir(parents=True, exist_ok=True)

    state = git_state()
    write_json(run / "00_governance" / "GIT_START_STATE.json", state)
    protected = {
        "b0": {"path": str(b0), "sha256": sha256_file(b0), "expected_sha256": B0_SHA256,
               "hash_matches_expected": sha256_file(b0) == B0_SHA256},
        "b1": {"path": str((PROJECT_ROOT / B1_RELATIVE).resolve()), "sha256": sha256_file(PROJECT_ROOT / B1_RELATIVE)},
        "full20": {"path": str((PROJECT_ROOT / FULL20_RELATIVE).resolve()), "sha256": sha256_file(PROJECT_ROOT / FULL20_RELATIVE)},
        "registry_modified": False,
        "original_data_modified": False,
        "github_push_performed": False,
    }
    write_json(run / "00_governance" / "PROTECTED_ASSETS_START.json", protected)
    write_json(run / "00_governance" / "GPU_LOCK.json", {"required": True, "lock_path": str((PROJECT_ROOT / "outputs/runtime_locks/urrs_gpu.lock").resolve()), "acquired": False})
    write_json(run / "00_governance" / "RUN_FLAGS.json", {
        "test_pixel_reads": 0, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0,
        "pseudo_label_training_started": False, "tfr_training_started": False,
        "mti_training_started": False, "construction_training_started": False,
        "height_training_started": False, "globfp_training_started": False,
        "registry_modified": False, "github_push_performed": False,
    })

    rows = discover_whu_train(root)
    split = build_boundary_split(rows, seed=int(seed), train_count=256, calibration_count=128, diagnostic_count=32)
    audit = split_leakage_audit(split)
    audit.update({
        "total_whu_train_pairs": len(rows),
        "boundary_train": len(split["boundary_train"]),
        "boundary_calibration": len(split["boundary_calibration"]),
        "boundary_diagnostic": len(split["boundary_diagnostic"]),
        "unused_train_pairs": len(split["unused"]),
        "seed": int(seed),
        "split_scope": "WHU Train only",
        "validation_used_for_training": False,
        "test_used": False,
        "test_pixel_reads": 0,
    })
    write_registry(split["all"], run / "02_boundary_split" / "BOUNDARY_SPLIT_REGISTRY.csv")
    for key, filename in (("boundary_train", "boundary_train.txt"), ("boundary_calibration", "boundary_calibration.txt"), ("boundary_diagnostic", "boundary_diagnostic.txt")):
        write_native_list(split[key], run / "02_boundary_split" / filename)
    write_json(run / "02_boundary_split" / "BOUNDARY_SPLIT_AUDIT.json", audit)
    write_json(run / "01_boundary_data_audit" / "WHU_TRAIN_DISCOVERY.json", {
        "root": str(root), "pair_count": len(rows), "source_split": "train",
        "label_reads": len(rows), "test_pixel_reads": 0,
        "positive_fraction_min": min(float(row["positive_fraction"]) for row in rows),
        "positive_fraction_max": max(float(row["positive_fraction"]) for row in rows),
        "positive_fraction_mean": sum(float(row["positive_fraction"]) for row in rows) / len(rows),
    })
    contract = {
        "status": "pre_validation_contract_pending",
        "b0_checkpoint": str(b0), "b0_sha256": sha256_file(b0), "expected_b0_sha256": B0_SHA256,
        "checkpoint_state_keys_expected": 407, "b0_trainable_parameter_count": 0,
        "boundary_feature_path": "decode_head.fusion_ab", "boundary_feature_channels": 256,
        "boundary_feature_stride": 4, "boundary_head": "Conv256-GroupNorm-GELU-Conv1",
        "boundary_width": 2, "boundary_target_method": "symmetric", "ignore_value": 255,
        "loss": "BCE + Dice", "loss_variant": "P0_L0", "seeds": [20260805, 20260806, 20260807],
        "max_steps": 2000, "checkpoint_steps": list(range(0, 2001, 250)),
        "selection_split": "boundary_calibration", "validation_parameters_frozen_before_validation": True,
        "oracle_selectable": False, "test_pixel_reads": 0,
    }
    write_json(run / "03_b0_contract" / "B0_FROZEN_CONTRACT.json", contract)
    manifest = {
        "run_id": run.name, "status": "initialized", "scientific_status": "detached_boundary_objectizer_in_progress",
        "run_dir": str(run), "whu_root": str(root), "b0_checkpoint": str(b0), "b0_sha256": sha256_file(b0),
        "source_git": state, "source_dirty": not state["clean"], "training_started": False,
        "validation_started": False, "test_used": False, "test_pixel_reads": 0,
        "registry_modified": False, "original_data_modified": False,
        "split_audit": str((run / "02_boundary_split" / "BOUNDARY_SPLIT_AUDIT.json").resolve()),
        "next_action": "commit Building-v2 source before formal GPU training" if not state["clean"] else "run bounded detached training",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run / "RUN_MANIFEST.json", manifest)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_WHU_ROOT)
    parser.add_argument("--b0-checkpoint", type=Path, default=None)
    parser.add_argument("--output-parent", type=Path, default=PROJECT_ROOT / "outputs" / "pretraining")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    run = initialize_run(whu_root=args.root, b0_checkpoint=args.b0_checkpoint, output_parent=args.output_parent, run_dir=args.run_dir, seed=args.seed)
    print(json.dumps({"status": "initialized", "run_dir": str(run), "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
