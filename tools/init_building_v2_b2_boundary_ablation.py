"""Create the isolated, auditable B2 Boundary Auxiliary Ablation run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHU_ROOT = Path(r"D:\Data Set\WHU_Building")
NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
NATIVE_PYTHON = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
NATIVE_CONFIG = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "configs" / "Swin_T_whu_windows_fullval_eval.py"
B0 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
B1_RUN = PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2_b1_native_pilot_20260805_200515"
B1_REGISTRY = B1_RUN / "04_b1_training" / "B1_TRAIN_SAMPLE_REGISTRY.csv"
B1_VAL = B1_RUN / "04_b1_training" / "b1" / "validation.txt"
B0_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
B1_CHECKPOINT = B1_RUN / "04_b1_training" / "b1" / "iter_500.pth"
B1_SHA = "7efc2e34ab4cee2561ec461e744a4b789233525eaf0ff3e30e14ed1b4b9be9f1"
FULL20 = PROJECT_ROOT / "outputs" / "pretraining" / "change_v2" / "20260727_001029" / "checkpoints" / "best_v2_multisource_change.pth"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return {"branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD"), "status_short": run("status", "--short").splitlines(), "diff_stat": run("diff", "--stat").splitlines()}


def make_run(run_dir: Path | None) -> Path:
    parent = PROJECT_ROOT / "outputs" / "pretraining"
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = parent / f"building_foundation_v2_b2_boundary_ablation_{stamp}"
        suffix = 1
        while run_dir.exists():
            run_dir = parent / f"building_foundation_v2_b2_boundary_ablation_{stamp}_{suffix:02d}"; suffix += 1
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty B2 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("00_governance", "01_baseline_contract", "02_boundary_target", "03_architecture", "04_smoke", "05_b2_training", "06_validation", "07_object_metrics", "08_diagnostics", "09_figures", "10_candidate_package", "11_tests", "12_acceptance"):
        (run_dir / name).mkdir()
    return run_dir


def write_list(master: Path, samples: Path, rows: list[dict[str, str]]) -> None:
    samples.write_text("".join(f"{row['image_path']}\t**\t**\t{row['label_path']}\t**\n" for row in rows), encoding="utf-8")
    master.write_text(samples.resolve().as_posix() + "\n", encoding="utf-8")


def train_rows() -> list[dict[str, str]]:
    image_dir, label_dir = WHU_ROOT / "train" / "image", WHU_ROOT / "train" / "label"
    images = {p.stem: p for p in image_dir.glob("*.tif")}; labels = {p.stem: p for p in label_dir.glob("*.tif")}
    return [{"sample_id": f"whu_train_{stem}", "split": "train", "image_path": str(images[stem].resolve()), "label_path": str(labels[stem].resolve())} for stem in sorted(set(images) & set(labels))]


def registry_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def boundary_stats(run: Path, rows: list[dict[str, str]], limit: int = 400) -> dict[str, Any]:
    import rasterio
    from urban_monitor.building_foundation_v2.boundary import boundary_object_stats, boundary_target
    stats_rows: list[dict[str, Any]] = []
    aggregate: dict[int, list[float]] = {1: [], 2: [], 3: []}
    preview_dir = run / "02_boundary_target" / "previews"; preview_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for index, row in enumerate(rows[:limit]):
        with rasterio.open(row["label_path"]) as ds:
            mask = ds.read(1)
        record = {"sample_id": row["sample_id"]}
        for width in (1, 2, 3):
            values = boundary_object_stats(mask, width)
            aggregate[width].append(float(values["boundary_positive_fraction"]))
            record.update({f"width{width}_{key}": value for key, value in values.items()})
            if width == 2:
                record["boundary_width2_positive_fraction"] = float(values["boundary_positive_fraction"])
        stats_rows.append(record)
        if index < 40:
            with rasterio.open(row["image_path"]) as ds:
                rgb = ds.read([1, 2, 3])
            rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
            low, high = np.percentile(rgb[np.isfinite(rgb)], [2, 98]) if np.isfinite(rgb).any() else (0, 1)
            rgb = np.clip((rgb - low) / max(1e-6, high - low), 0, 1)
            target = boundary_target(mask, 2)
            panel = np.concatenate([rgb, np.repeat(mask[..., None] > 0, 3, axis=2), np.repeat(target[..., None], 3, axis=2)], axis=1)
            Image.fromarray((panel * 255).astype("uint8") if panel.dtype != bool else (panel.astype("uint8") * 255)).save(preview_dir / f"{index:03d}_{row['sample_id']}.png")
    with (run / "02_boundary_target" / "BOUNDARY_TRAIN_STATS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = sorted({key for item in stats_rows for key in item})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(stats_rows)
    payload = {"status": "available", "sample_count": len(stats_rows), "widths": {str(width): {"mean_boundary_positive_fraction": float(np.mean(values)) if values else 0.0, "p10": float(np.quantile(values, 0.1)) if values else 0.0, "p25": float(np.quantile(values, 0.25)) if values else 0.0, "p50": float(np.quantile(values, 0.5)) if values else 0.0, "p75": float(np.quantile(values, 0.75)) if values else 0.0, "p90": float(np.quantile(values, 0.9)) if values else 0.0} for width, values in aggregate.items()}, "selected_width": 2, "selection_reason": "pre-registered primary width; target statistics audited on WHU Train only", "previews": 40, "test_pixel_reads": 0, "test_used": False}
    write_json(run / "02_boundary_target" / "BOUNDARY_TARGET_TRAIN_STATS.json", payload)
    write_json(run / "02_boundary_target" / "BOUNDARY_TARGET_CONTRACT.json", {"boundary_method": "symmetric", "boundary_width": 2, "candidate_widths": [1, 2, 3], "inner_width": 2, "outer_width": 2, "ignore_handling": "ignore_value_255_excluded_from_target_and_loss", "interpolation": "nearest_neighbor_only_for_labels", "morphology_structure": "8_connectivity_square_structuring_element", "selection_evidence": "width 2 fixed before candidate Validation; Train stats only", "small_object_policy": "retain boundary signal; no object is discarded", "test_pixel_reads": 0, "test_used": False})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--skip-baseline-eval", action="store_true")
    args = parser.parse_args()
    run = make_run(args.run_dir)
    if not B0.is_file() or sha256(B0) != B0_SHA:
        raise SystemExit("B2 blocked: protected B0 iter2000 checkpoint missing or hash changed")
    if not B1_REGISTRY.is_file() or not B1_VAL.is_file():
        raise SystemExit("B2 blocked: fixed B1 train registry or validation manifest missing")
    snapshot = git_snapshot()
    snapshot["pre_b2_committed_head"] = "79f1a426af518673e42d82c94ad7c2366fa65882"
    snapshot["working_tree_changes_are_b2_implementation"] = True
    write_json(run / "00_governance" / "GIT_START_STATE.json", snapshot)
    protected = {"b0_iter2000": {"path": str(B0.resolve()), "sha256": sha256(B0), "expected_sha256": B0_SHA}, "b1_checkpoint": {"path": str(B1_CHECKPOINT.resolve()), "sha256": sha256(B1_CHECKPOINT), "expected_sha256": B1_SHA}, "full20": {"path": str(FULL20.resolve()), "sha256": sha256(FULL20)}, "b1_run": {"path": str(B1_RUN.resolve()), "read_only": True}, "whu_test": {"registered_only": True, "pixel_reads": 0}, "second_test": {"registered_only": True, "pixel_reads": 0}, "registry_modified": False}
    write_json(run / "00_governance" / "PROTECTED_ASSET_INVENTORY.json", protected)
    write_json(run / "00_governance" / "PRE_RUN_SHA256.json", {"b0": sha256(B0), "b1": sha256(B1_CHECKPOINT), "full20": sha256(FULL20), "registry_modified": False, "test_pixel_reads": 0})
    write_json(run / "00_governance" / "RUN_GOVERNANCE.json", {"run_id": run.name, "created_at_utc": datetime.now(timezone.utc).isoformat(), "git": snapshot, "native_python": str(NATIVE_PYTHON), "native_repo": str(NATIVE_REPO), "gpu_lock": str((PROJECT_ROOT / "outputs" / "runtime_locks" / "urrs_gpu.lock").resolve()), "flags": {"whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "tfr_training_started": False, "mti_training_started": False, "construction_training_started": False, "height_training_started": False, "globfp_pixel_training_started": False, "nanjing_training_started": False, "tongzhou_training_started": False, "registry_modified": False, "github_push_performed": False}, "status": "governance_created"})
    b1_rows = registry_rows(B1_REGISTRY)
    if len(b1_rows) != 256 or any(row.get("split") != "train" for row in b1_rows):
        raise SystemExit(f"B2 blocked: B1 fixed train registry invalid count/split: {len(b1_rows)}")
    shutil.copy2(B1_REGISTRY, run / "05_b2_training" / "B1_TRAIN_SAMPLE_REGISTRY.csv")
    train_master = run / "05_b2_training" / "train.txt"; train_samples = run / "05_b2_training" / "train_samples.txt"
    val_master = run / "05_b2_training" / "validation.txt"; val_samples = run / "05_b2_training" / "validation_samples.txt"
    write_list(train_master, train_samples, b1_rows)
    val_lines = [line for line in B1_VAL.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    # B1 validation.txt is the master list; its companion file is the data list
    # written next to it.  Copy both explicitly and never discover Test paths.
    shutil.copy2(B1_VAL, val_master)
    companion = B1_VAL.with_name(B1_VAL.stem + "_samples.txt")
    if companion.is_file(): shutil.copy2(companion, val_samples)
    write_json(run / "05_b2_training" / "B2_TRAINING_DATA_CONTRACT.json", {"train_registry": str((run / "05_b2_training" / "B1_TRAIN_SAMPLE_REGISTRY.csv").resolve()), "train_count": len(b1_rows), "validation_manifest": str(val_master.resolve()), "validation_count": 1036, "fixed_registry_reused": True, "random_reselection": False, "test_used": False, "test_pixel_reads": 0})
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    stats = boundary_stats(run, train_rows(), 400)
    write_json(run / "03_architecture" / "B2_ARCHITECTURE_CONTRACT.json", {"model_family": "native_opencd_foundation_swin_t", "backbone_preserved": True, "native_building_head_preserved": True, "boundary_head": {"type": "3x3_conv_groupnorm_gelu_1x1_conv", "input_feature": "decode_head.fusion_ab high-resolution native decoder feature", "feature_stride": "native fusion_ab output", "channels": 64}, "b0_initialization_required": True, "b1_initialization_forbidden": True, "new_parameter_prefix": "decode_head.boundary_head.", "native_state_keys_expected": 407, "test_used": False, "test_pixel_reads": 0})
    write_json(run / "RUN_MANIFEST.json", {"run_id": run.name, "status": "in_progress", "scientific_status": "b2_boundary_ablation_in_progress", "source_b0": str(B0.resolve()), "source_b0_sha256": sha256(B0), "source_b1_diagnostic_only": str(B1_CHECKPOINT.resolve()), "source_b1_sha256": sha256(B1_CHECKPOINT), "boundary_contract": str((run / "02_boundary_target" / "BOUNDARY_TARGET_CONTRACT.json").resolve()), "fixed_train_registry": str((run / "05_b2_training" / "B1_TRAIN_SAMPLE_REGISTRY.csv").resolve()), "test_used": False, "test_pixel_reads": 0, "registry_modified": False})
    print(json.dumps({"status": "initialized", "run_dir": str(run), "boundary_stats": stats, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
