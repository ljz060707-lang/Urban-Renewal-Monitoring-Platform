"""Initialize an isolated B2 boundary-failure diagnostic run.

The initializer only reads the prior B2 artifacts and the already frozen WHU
Train sample.  It never discovers or opens WHU/SECOND Test data and never
changes a registry or checkpoint.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_RUN = PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2_b2_boundary_ablation_20260805_212147"
B0 = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
B1 = PROJECT_ROOT / "outputs" / "pretraining" / "building_foundation_v2_b1_native_pilot_20260805_200515" / "04_b1_training" / "b1" / "iter_500.pth"
FULL20 = PROJECT_ROOT / "outputs" / "pretraining" / "change_v2" / "20260727_001029" / "checkpoints" / "best_directional_change.pth"
B0_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status_short": run("status", "--short"), "diff_stat": run("diff", "--stat")}


def previous_sample() -> dict[str, Any]:
    candidates = sorted(PREVIOUS_RUN.glob("04_smoke/B2_L005/one_sample_positive*/train_samples.txt"), key=lambda path: ("constantlr" not in path.name, str(path)))
    if not candidates:
        raise FileNotFoundError("previous B2 one-sample manifest not found")
    path = candidates[0]
    line = next(line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    parts = line.split("\t")
    image_path, label_path = parts[0], parts[3]
    registry = PREVIOUS_RUN / "05_b2_training" / "B1_TRAIN_SAMPLE_REGISTRY.csv"
    sample_id = None
    with registry.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if Path(row["image_path"]).resolve() == Path(image_path).resolve():
                sample_id = row["sample_id"]
                break
    if not sample_id:
        raise RuntimeError(f"sample path from previous run is not in fixed registry: {image_path}")
    from urban_monitor.building_foundation_v2.diagnostics import load_mask, object_size_counts
    mask = load_mask(label_path)
    stats = object_size_counts(mask)
    return {
        "sample_id": sample_id,
        "source_split": "train",
        "image_path": str(Path(image_path).resolve()),
        "label_path": str(Path(label_path).resolve()),
        "previous_manifest": str(path.resolve()),
        "actual_positive_fraction": float((mask > 0).mean()),
        "gt_object_count": stats["object_count"],
        "small_object_count": stats["small"],
        "medium_object_count": stats["medium"],
        "large_object_count": stats["large"],
        "original_image_shape": list(mask.shape),
        "training_crop_size": [512, 512],
        "padding_used": False,
        "mosaic_used": False,
        "random_augmentation_used": False,
        "diagnostic_pipeline": "native load + annotations + deterministic pack; no crop/flip/photo/mosaic/mixup",
    }


def make_run(parent: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = parent / f"building_foundation_v2_b2_boundary_diagnostic_{stamp}"
    suffix = 1
    while run.exists():
        run = parent / f"building_foundation_v2_b2_boundary_diagnostic_{stamp}_{suffix:02d}"
        suffix += 1
    for name in ("00_governance", "01_previous_run_contract", "02_target_metric_validation", "03_feature_resolution", "04_sample_baseline", "05_head_only_probe", "06_shared_gradient_probe", "07_loss_diagnostics", "08_architecture_decision", "09_figures", "10_tests", "11_acceptance"):
        (run / name).mkdir(parents=True, exist_ok=True)
    return run


def main() -> int:
    global PREVIOUS_RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, default=PREVIOUS_RUN)
    parser.add_argument("--output-parent", type=Path, default=PROJECT_ROOT / "outputs" / "pretraining")
    args = parser.parse_args()
    PREVIOUS_RUN = args.previous_run.resolve()
    run = make_run(args.output_parent.resolve())
    sample = previous_sample()
    b2_shas = {}
    for candidate in ("B2_L005", "B2_L010", "B2_L020"):
        checkpoint = next(iter((PREVIOUS_RUN / "04_smoke" / candidate).glob("one_sample_positive_constantlr_500/iter_500.pth")), None)
        if checkpoint and checkpoint.is_file():
            b2_shas[candidate] = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}
    state = git_state()
    protected = {
        "b0": {"path": str(B0.resolve()), "sha256": sha256(B0), "expected_sha256": B0_SHA},
        "b1": {"path": str(B1.resolve()), "sha256": sha256(B1)},
        "b2_failed_checkpoints": b2_shas,
        "full20": {"path": str(FULL20.resolve()), "sha256": sha256(FULL20) if FULL20.is_file() else None},
        "previous_run": str(PREVIOUS_RUN),
        "registry_modified": False,
        "source_data_modified": False,
        "whu_test_pixel_reads": 0,
        "second_test_pixel_reads": 0,
    }
    write_json(run / "00_governance" / "GIT_START_STATE.json", state)
    write_json(run / "00_governance" / "PROTECTED_ASSETS.json", protected)
    write_json(run / "00_governance" / "RUN_FLAGS.json", {
        "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "registry_modified": False,
        "source_data_modified": False, "b3_training_started": False, "tfr_training_started": False,
        "mti_training_started": False, "construction_training_started": False, "height_training_started": False,
        "full_b2_pilot_started": False, "test_used": False, "test_pixel_reads": 0,
    })
    for source, target in ((PREVIOUS_RUN / "RUN_MANIFEST.json", run / "01_previous_run_contract" / "PREVIOUS_RUN_MANIFEST.json"),
                           (PREVIOUS_RUN / "04_smoke" / "B2_POSITIVE_OVERFIT_SUMMARY.json", run / "01_previous_run_contract" / "PREVIOUS_POSITIVE_OVERFIT_SUMMARY.json"),
                           (PREVIOUS_RUN / "04_smoke" / "B2_SMOKE_SUMMARY.json", run / "01_previous_run_contract" / "PREVIOUS_SMOKE_SUMMARY.json")):
        if source.is_file():
            shutil.copy2(source, target)
    write_json(run / "04_sample_baseline" / "ONE_SAMPLE_REUSED_CONTRACT.json", sample)
    previous_master = Path(sample["previous_manifest"]).with_name("train.txt")
    previous_samples = Path(sample["previous_manifest"])
    if previous_master.is_file():
        shutil.copy2(previous_master, run / "04_sample_baseline" / "one_sample_train.txt")
    shutil.copy2(previous_samples, run / "04_sample_baseline" / "one_sample_train_samples.txt")
    write_json(run / "RUN_MANIFEST.json", {
        "run_id": run.name, "status": "in_progress", "scientific_status": "b2_boundary_diagnostic_in_progress",
        "previous_run": str(PREVIOUS_RUN), "sample_id": sample["sample_id"], "sample_contract": str((run / "04_sample_baseline" / "ONE_SAMPLE_REUSED_CONTRACT.json").resolve()),
        "test_used": False, "test_pixel_reads": 0, "registry_modified": False,
        "full_b2_pilot_started": False, "training_started": False,
    })
    print(json.dumps({"status": "initialized", "run_dir": str(run), "sample": sample, "test_used": False, "test_pixel_reads": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
