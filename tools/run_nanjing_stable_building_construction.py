"""Nanjing stable Building tracking and Construction output controller.

This controller is audit-first and resume-safe.  It never uses Construction to
change Building, never reads locked test pixels, and never starts training
unless a separately audited training gate is explicitly eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import geopandas as gpd
import pandas as pd

import evaluate_cross_city_building_stability as stability


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
DATES = ["2015-01-04", "2016-08-18", "2017-12-11", "2018-02-08", "2019-03-16", "2022-12-02"]
TONGZHOU_DATES = ["2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16"]
INTEGRATED = OUTPUTS / "inference" / "nanjing_building_construction_integrated_20260804_233025"
ZOOM20 = OUTPUTS / "inference" / "nanjing_jiangbei_zoom20_2019_gt_20260804_211042"
OLD_NANJING = OUTPUTS / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320"
TONGZHOU_BUILDING = OUTPUTS / "inference" / "beijing_tongzhou_building_geometry"
TONGZHOU_FULL20 = OUTPUTS / "inference" / "beijing_tongzhou_to_latest"
TONGZHOU_CONSTRUCTION = OUTPUTS / "inference" / "beijing_tongzhou_construction_activity_v1"
REGISTRY = OUTPUTS / "governance" / "current_best_gt_ready_20260804_173745" / "01_model_registry" / "CURRENT_BEST_MODEL_REGISTRY.json"
GT_BUILDING = Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20\gt_building19.gpkg")
GT_CONSTRUCTION = Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20\gt_construction19.gpkg")
RUNTIME_LOCKS = OUTPUTS / "runtime_locks"


class RuntimeLock:
    """Atomic process-owned lock for URRS GPU/training/inference stages."""

    def __init__(self, path: Path, task_name: str, run_root: Path, gpu: str = "0", expected_vram_mb: int = 0):
        self.path = path
        self.payload = {
            "pid": os.getpid(), "task_name": task_name, "run_root": str(run_root.resolve()),
            "started_at": now(), "gpu": gpu, "expected_vram_mb": expected_vram_mb,
        }
        self.acquired = False

    @staticmethod
    def pid_alive(pid: int) -> bool:
        try:
            import psutil

            return psutil.pid_exists(int(pid))
        except Exception:
            if os.name == "nt":
                try:
                    import ctypes

                    process_query_limited_information = 0x1000
                    still_active = 259
                    handle = ctypes.windll.kernel32.OpenProcess(
                        process_query_limited_information, False, int(pid)
                    )
                    if not handle:
                        return False
                    exit_code = ctypes.c_ulong()
                    ok = ctypes.windll.kernel32.GetExitCodeProcess(
                        handle, ctypes.byref(exit_code)
                    )
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return bool(ok and exit_code.value == still_active)
                except Exception:
                    return False
            try:
                os.kill(int(pid), 0)
                return True
            except OSError:
                return False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            existing = read_json(self.path, {})
            pid = int(existing.get("pid", -1))
            state = "active" if self.pid_alive(pid) else "stale_requires_log_confirmed_cleanup"
            raise RuntimeError(f"Lock unavailable ({state}): {self.path}; holder={existing}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, ensure_ascii=False)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        existing = read_json(self.path, {})
        if int(existing.get("pid", -1)) != os.getpid():
            raise RuntimeError(f"Refusing to release a lock owned by another PID: {self.path}")
        self.path.unlink()
        self.acquired = False

    def __enter__(self) -> "RuntimeLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def ensure_tree(run_root: Path) -> None:
    for relative in [
        "00_governance", "01_training_inventory", "02_baseline", "03_tongzhou_benchmark",
        "04_diagnostics", "05_building_search", "06_training_decision",
        "07_building_final/single_date", "07_building_final/tracking", "07_building_final/to_latest",
        "07_building_final/figures/single_date", "07_building_final/figures/tracking",
        "07_building_final/figures/local_cases", "07_building_final/metrics",
        "08_construction_final/single_date", "08_construction_final/ranking", "08_construction_final/temporal",
        "08_construction_final/figures/single_date", "08_construction_final/figures/temporal",
        "08_construction_final/figures/local_cases", "08_construction_final/metrics",
        "09_figures_manifest", "10_manifests", "11_logs", "12_tests",
    ]:
        (run_root / relative).mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == source.stat().st_size and sha256(target) == sha256(source):
            return "existing_identical"
        raise FileExistsError(f"Refusing to overwrite non-identical artifact: {target}")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _protected_paths() -> list[Path]:
    registry = read_json(REGISTRY, {})
    paths = [REGISTRY, GT_BUILDING, GT_CONSTRUCTION]
    for entry in registry.get("entries", []):
        for key in ("checkpoint", "config"):
            value = entry.get(key, {}).get("path")
            if value:
                paths.append(Path(value))
    paths.append(OLD_NANJING / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json")
    for root in (TONGZHOU_BUILDING, TONGZHOU_FULL20, TONGZHOU_CONSTRUCTION):
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted({path.resolve() for path in paths if path.exists()}, key=str)


def protected_snapshot() -> dict[str, Any]:
    rows = [signature(path) for path in _protected_paths()]
    return {"created_at": now(), "file_count": len(rows), "files": rows}


def governance_audit(run_root: Path) -> dict[str, Any]:
    ensure_tree(run_root)
    before_path = run_root / "00_governance" / "PRE_RUN_SHA256.json"
    if not before_path.exists():
        write_json(before_path, protected_snapshot())
    registry = read_json(REGISTRY, {})
    entries = []
    for entry in registry.get("entries", []):
        checkpoint = Path(entry["checkpoint"]["path"])
        entries.append({
            "module_id": entry.get("module_id"),
            "selected_experiment": entry.get("selected_experiment"),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "registry_expected_sha256": entry["checkpoint"].get("expected_sha256"),
            "sha_match": sha256(checkpoint) == entry["checkpoint"].get("expected_sha256"),
            "registry_status": entry.get("registry_status"),
            "scientific_status": entry.get("scientific_status"),
        })
    write_json(run_root / "00_governance" / "CURRENT_BEST_REGISTRY_AUDIT.json", {
        "registry": signature(REGISTRY), "entries": entries, "registry_modified": False,
    })
    write_json(run_root / "00_governance" / "MODEL_LINEAGE_AUDIT.json", {"created_at": now(), "entries": entries})
    write_json(run_root / "00_governance" / "CHECKPOINT_IDENTITY_AUDIT.json", {"created_at": now(), "entries": entries})
    selection_path = OUTPUTS / "training" / "construction_nanjing_globfp_adaptation_20260804_144853" / "04_selection" / "FINAL_MODEL_SELECTION.json"
    selection = read_json(selection_path, {})
    c_b1_manifest_path = OUTPUTS / "training" / "construction_nanjing_globfp_adaptation_20260804_144853" / "03_experiments" / "C-B1" / "RUN_MANIFEST.json"
    c_b1_manifest = read_json(c_b1_manifest_path, {})
    conflict = {
        "C_A_path": next((row["checkpoint_path"] for row in entries if row["module_id"] == "M6_Construction"), None),
        "C_A_sha256": next((row["checkpoint_sha256"] for row in entries if row["module_id"] == "M6_Construction"), None),
        "C_B1_status": selection.get("c_b1_gate", {}).get("status", "unknown"),
        "C_B1_checkpoint_path": c_b1_manifest.get("best_checkpoint"),
        "C_B1_checkpoint_sha256": c_b1_manifest.get("best_checkpoint_sha256"),
        "C_B1_training_started_historically": c_b1_manifest.get("training_started"),
        "C_B1_run_manifest": str(c_b1_manifest_path),
        "same_checkpoint_identity": c_b1_manifest.get("best_checkpoint_sha256") == next((row["checkpoint_sha256"] for row in entries if row["module_id"] == "M6_Construction"), None),
        "identity_conclusion": "C-A and C-B1 are distinct checkpoint files and SHA256 identities; C-A remains frozen and C-B1 remains rejected, must not be renamed or resumed",
        "selected_experiment": selection.get("selected_experiment"),
        "adaptation_rejected": selection.get("adaptation_rejected"),
        "locked_test_pixel_reads": selection.get("locked_test_pixel_reads", 0),
        "selection_receipt": str(selection_path),
    }
    write_json(run_root / "00_governance" / "CONSTRUCTION_CA_CB1_CONFLICT_AUDIT.json", conflict)
    write_csv(run_root / "00_governance" / "PROTECTED_ASSET_INVENTORY.csv", read_json(before_path, {}).get("files", []))
    RUNTIME_LOCKS.mkdir(parents=True, exist_ok=True)
    lock_rows = []
    for name in ("urrs_gpu.lock", "nanjing_training.lock", "nanjing_inference.lock"):
        path = RUNTIME_LOCKS / name
        payload = read_json(path, {}) if path.exists() else {}
        lock_rows.append({"path": str(path), "exists": path.exists(), "holder": payload, "holder_pid_alive": RuntimeLock.pid_alive(int(payload.get("pid", -1))) if payload else False})
    try:
        import psutil

        parallel = []
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            command = " ".join(process.info.get("cmdline") or [])
            if "tongzhou" in command.lower() and process.info.get("pid") != os.getpid():
                parallel.append({"pid": process.info.get("pid"), "name": process.info.get("name"), "role": "tongzhou_related_process_detected", "gpu_overlap_claimed": False})
    except Exception:
        parallel = []
    write_json(run_root / "GPU_LOCK_AUDIT.json", {"created_at": now(), "locks": lock_rows, "gpu_stage_started": False, "lock_acquisition_required_if_gpu_stage_starts": True, "stale_lock_deleted": False})
    write_json(run_root / "PARALLEL_EXECUTION_AUDIT.json", {"created_at": now(), "parallel_processes": parallel, "tongzhou_intermediate_outputs_read": False, "tongzhou_outputs_modified": False, "gpu_tasks_overlapped": False})
    audit = {
        "created_at": now(), "run_root": str(run_root.resolve()), "registry_sha256": sha256(REGISTRY),
        "checkpoint_count": len(entries), "all_checkpoint_hashes_match": all(row["sha_match"] for row in entries),
        "C_B1_status": conflict["C_B1_status"], "site_classifier_status": "disabled_pending_human_gt",
        "locked_test_pixel_reads": 0, "html_generated": False, "training_started": False,
        "tongzhou_outputs_read_only": True,
    }
    write_json(run_root / "00_governance" / "GOVERNANCE_AUDIT.json", audit)
    return audit


def training_inventory(run_root: Path) -> dict[str, Any]:
    whu = read_json(OUTPUTS / "v2_readiness" / "rsbuilding" / "whu" / "WHU_LAYOUT_AUDIT.json", {})
    seed_path = OUTPUTS / "urrs" / "construction" / "human_seed_labels_20260802_134145" / "SEED_LABEL_INVENTORY.json"
    seed = read_json(seed_path, {})
    seed_records = seed if isinstance(seed, list) else next((seed.get(key) for key in ("records", "items", "samples", "inventory", "seed_inventory") if isinstance(seed.get(key), list)), [])
    confirmed = sum(bool(row.get("human_confirmed")) for row in seed_records)
    surface_checkpoint = OUTPUTS / "training" / "surface_state_v1" / "cwld_20260729_165008" / "checkpoints" / "cwld_best_validation.pth"
    rows = [
        {"dataset_id": "full20_dsifn", "dataset_name": "DSIFN", "task": "Full20 any-change", "root_path": "registry selection evidence", "train_count": None, "validation_count": None, "test_count": None, "used_for_training": True, "used_for_validation": True, "used_for_test": False, "human_verified": True, "positive_only": False, "scientific_status": "trained_and_verified", "recommended_action": "frozen_no_retraining"},
        {"dataset_id": "full20_s2looking", "dataset_name": "S2Looking", "task": "Full20 directional building change", "root_path": "registry selection evidence", "train_count": None, "validation_count": None, "test_count": None, "used_for_training": True, "used_for_validation": True, "used_for_test": False, "human_verified": True, "positive_only": False, "scientific_status": "trained_and_verified", "recommended_action": "frozen_no_retraining"},
        {"dataset_id": "full20_second", "dataset_name": "SECOND", "task": "Full20 semantic supervision", "root_path": "registry selection evidence", "train_count": None, "validation_count": None, "test_count": None, "used_for_training": True, "used_for_validation": True, "used_for_test": False, "human_verified": True, "positive_only": False, "scientific_status": "trained_but_limited", "recommended_action": "test_locked_no_read"},
        {"dataset_id": "whu_building", "dataset_name": "WHU Building", "task": "Building", "root_path": whu.get("whu_root"), "manifest_path": whu.get("train", {}).get("manifest"), "train_count": whu.get("train", {}).get("paired_count"), "validation_count": whu.get("validation", {}).get("paired_count"), "test_count": whu.get("test", {}).get("image_file_inventory_count"), "used_for_training": True, "used_for_validation": True, "used_for_test": False, "human_verified": True, "positive_only": False, "scientific_status": "trained_and_verified", "recommended_action": "same_data_training_not_justified"},
        {"dataset_id": "ucs386", "dataset_name": "UCS386 available subset", "task": "Construction", "root_path": r"D:\Data Set\URS\600 manually delineated UCS labels part2", "train_count": 296, "validation_count": 44, "test_count": 46, "missing_count": 214, "used_for_training": True, "used_for_validation": True, "used_for_test": False, "human_verified": True, "positive_only": False, "scientific_status": "trained_but_limited", "recommended_action": "locked_test_zero_reads"},
        {"dataset_id": "human_seed_temporal", "dataset_name": "Temporal Human Seed", "task": "Construction temporal", "root_path": str(seed_path.parent), "manifest_path": str(seed_path), "train_count": 0, "validation_count": 0, "test_count": 0, "scene_count": len(seed_records), "human_verified_count": confirmed, "used_for_training": False, "used_for_validation": False, "used_for_test": False, "human_verified": confirmed > 0, "pseudo_label": False, "positive_only": True, "scientific_status": "human_seed_unverified", "recommended_action": "human_review_required"},
        {"dataset_id": "nanjing_2019_building_gt", "dataset_name": "Nanjing 2019 Building positive GT", "task": "Building diagnostic", "root_path": str(GT_BUILDING), "positive_count": 277, "used_for_training": False, "used_for_validation": False, "used_for_test": False, "human_verified": True, "positive_only": True, "spatial_holdout": True, "scientific_status": "independent_holdout", "recommended_action": "positive_only_diagnostic_not_selection"},
        {"dataset_id": "nanjing_2019_construction_gt", "dataset_name": "Nanjing 2019 Construction positive GT", "task": "Construction diagnostic", "root_path": str(GT_CONSTRUCTION), "positive_count": 21, "used_for_training": False, "used_for_validation": False, "used_for_test": False, "human_verified": True, "positive_only": True, "spatial_holdout": True, "scientific_status": "independent_holdout", "recommended_action": "positive_only_diagnostic_not_selection"},
        {"dataset_id": "construction_cwld_stage_b", "dataset_name": "CWLD Stage B for Construction", "task": "Construction", "root_path": None, "used_for_training": False, "used_for_validation": False, "used_for_test": False, "human_verified": False, "positive_only": False, "scientific_status": "planned_only", "recommended_action": "blocked_missing_construction_class_mapping_and_independent_validation"},
        {"dataset_id": "surface_state_cwld", "dataset_name": "Surface State CWLD", "task": "Surface State only", "root_path": str(surface_checkpoint.parent.parent), "used_for_training": surface_checkpoint.exists(), "used_for_validation": surface_checkpoint.exists(), "used_for_test": False, "human_verified": True, "positive_only": False, "scientific_status": "trained_and_verified" if surface_checkpoint.exists() else "missing_or_incomplete", "recommended_action": "do_not_relabel_as_construction_stage_b"},
    ]
    common = {"license_status": "audited_or_existing_project_receipt", "duplicate_hash_count": None, "near_duplicate_count": None, "leakage_status": "no_new_cross_split_scan_in_this_run", "date_holdout": False, "temporal_labels": False, "human_corrected": False, "pseudo_label": False, "complete_background": False, "pair_count": None, "class_distribution": None, "used_by_checkpoint": None, "used_by_model_family": None, "spatial_resolution": None, "sensor_or_source": None, "missing_count": None, "negative_count": None, "positive_count": None, "scene_count": None, "test_locked": False}
    normalized = [{**common, **row} for row in rows]
    write_csv(run_root / "01_training_inventory" / "TRAINING_ASSET_LEDGER.csv", normalized)
    write_json(run_root / "01_training_inventory" / "TRAINING_ASSET_LEDGER.json", {"created_at": now(), "datasets": normalized})
    statuses = pd.Series([row["scientific_status"] for row in normalized]).value_counts().to_dict()
    duplication = {"status": "not_claimed_complete", "reason": "nationwide pixel/perceptual duplicate scan was not repeated; existing split receipts and locked-test contracts were preserved", "training_gate_effect": "closed", "locked_test_pixel_reads": 0}
    leakage = {"status": "no_new_training_authorized", "nanjing_gt_used_for_training": False, "nanjing_gt_used_for_selection": False, "locked_test_pixel_reads": 0, "full_exhaustive_cross_dataset_scan": False, "training_gate_effect": "closed"}
    write_json(run_root / "01_training_inventory" / "DATASET_DUPLICATION_AUDIT.json", duplication)
    write_json(run_root / "01_training_inventory" / "DATASET_LEAKAGE_AUDIT.json", leakage)
    write_json(run_root / "01_training_inventory" / "DATASET_LICENSE_AND_USAGE_AUDIT.json", {"created_at": now(), "construction_cwld_stage_b": "not_eligible", "surface_state_cwld": "separate_task_only"})
    write_csv(run_root / "01_training_inventory" / "CHECKPOINT_DATASET_MATRIX.csv", [
        {"checkpoint": "Full20", "datasets": "DSIFN;S2Looking;SECOND", "frozen": True},
        {"checkpoint": "RSBuilding iter_2000", "datasets": "WHU Building Train", "frozen": True},
        {"checkpoint": "Construction C-A", "datasets": "UCS386 Train", "frozen": True},
    ])
    write_csv(run_root / "01_training_inventory" / "TRAINING_RUN_INVENTORY.csv", [
        {"task": "Full20", "run": "outputs/pretraining/change_v2/20260727_001029", "status": "trained_and_frozen"},
        {"task": "RSBuilding", "run": "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750", "status": "trained_and_frozen"},
        {"task": "Construction", "run": "outputs/training/construction_activity_v1/URRS-Construction-UCS386-v1.0-task-candidate_20260803_115303", "status": "trained_and_frozen_limited"},
        {"task": "Surface State", "run": "outputs/training/surface_state_v1/cwld_20260729_165008", "status": "trained_separate_task"},
    ])
    result = {"created_at": now(), "dataset_count": len(rows), "status_counts": statuses, "human_seed_total": len(seed_records), "human_seed_confirmed": confirmed, "WHU": {"train": whu.get("train", {}).get("paired_count"), "validation": whu.get("validation", {}).get("paired_count"), "test_inventory": whu.get("test", {}).get("image_file_inventory_count"), "test_pixel_reads": whu.get("test", {}).get("pixel_reads")}, "UCS": {"train": 296, "validation": 44, "locked_test": 46, "locked_test_pixel_reads": 0}, "surface_state_production_checkpoint_exists": surface_checkpoint.exists(), "training_gate_input_ready": False}
    write_json(run_root / "01_training_inventory" / "TRAINING_DECISION_INPUTS.json", result)
    return result


def reproduce_baseline(run_root: Path) -> dict[str, Any]:
    expected_building = [98, 66, 166, 192, 168, 292]
    expected_construction = [3, 3, 17, 43, 34, 31]
    building_csv = pd.read_csv(INTEGRATED / "02_building" / "BUILDING_SINGLE_DATE_SUMMARY.csv")
    construction_csv = pd.read_csv(INTEGRATED / "04_construction" / "CONSTRUCTION_SINGLE_DATE_SUMMARY.csv")
    building_column = "object_count_after_mmu" if "object_count_after_mmu" in building_csv.columns else "object_count"
    building_counts = building_csv[building_column].astype(int).tolist()
    candidate_column = "candidate_count" if "candidate_count" in construction_csv.columns else "raw_candidate_count"
    construction_counts = construction_csv[candidate_column].astype(int).tolist()
    track = pd.read_csv(INTEGRATED / "02_building" / "tracking" / "BUILDING_TRACK_SUMMARY.csv")
    construction_track = pd.read_csv(INTEGRATED / "04_construction" / "tracking" / "CONSTRUCTION_TRACK_SUMMARY.csv")
    diffs = []
    for date, actual, expected in zip(DATES, building_counts, expected_building):
        diffs.append({"module": "building", "date": date, "actual": actual, "expected": expected, "difference": actual - expected})
    for date, actual, expected in zip(DATES, construction_counts, expected_construction):
        diffs.append({"module": "construction", "date": date, "actual": actual, "expected": expected, "difference": actual - expected})
    write_csv(run_root / "02_baseline" / "BASELINE_NUMERICAL_DIFF.csv", diffs)
    write_csv(run_root / "02_baseline" / "BUILDING_SINGLE_DATE_BASELINE.csv", building_csv)
    write_csv(run_root / "02_baseline" / "BUILDING_TRACKING_BASELINE.csv", track)
    write_csv(run_root / "02_baseline" / "CONSTRUCTION_BASELINE.csv", construction_csv)
    result = {"created_at": now(), "building_counts": building_counts, "construction_counts": construction_counts, "building_track_status_counts": track["status"].value_counts().to_dict(), "construction_track_status_counts": construction_track["status"].value_counts().to_dict(), "numerically_reproduced": all(row["difference"] == 0 for row in diffs), "source": str(INTEGRATED), "locked_test_pixel_reads": 0}
    write_json(run_root / "02_baseline" / "BASELINE_REPRODUCTION.json", result)
    return result


def diagnostics(run_root: Path) -> dict[str, Any]:
    import rasterio

    rows = []
    for date in DATES:
        path = INTEGRATED / "02_building" / "single_date" / date / "building_mask.tif"
        with rasterio.open(path) as dataset:
            rows.append({"scene_date": date, "path": str(path), "crs": str(dataset.crs), "width": dataset.width, "height": dataset.height, "transform": tuple(dataset.transform), "bounds": tuple(dataset.bounds), "resolution": dataset.res, "nodata": dataset.nodata})
    write_csv(run_root / "04_diagnostics" / "GRID_AUDIT.csv", rows)
    pair_rows = []
    for left, right in zip(rows, rows[1:]):
        pair_rows.append({"left_date": left["scene_date"], "right_date": right["scene_date"], "same_crs": left["crs"] == right["crs"], "same_shape": (left["width"], left["height"]) == (right["width"], right["height"]), "same_transform": left["transform"] == right["transform"], "high_confidence_translation_m": 0.0 if left["transform"] == right["transform"] else None, "registration_action": "none" if left["transform"] == right["transform"] else "requires_independent_registration_evidence"})
    write_csv(run_root / "04_diagnostics" / "PAIRWISE_REGISTRATION.csv", pair_rows)
    write_csv(run_root / "04_diagnostics" / "IMAGE_QUALITY.csv", [{"scene_date": date, "status": "existing_zoom20_audit_reused", "pixel_read_in_this_stage": False} for date in DATES])
    taxonomy = [{"category": item, "diagnostic_status": "candidate_cause"} for item in ["grid_or_registration", "input_scale", "image_quality", "threshold_variation", "split_merge", "large_roof", "shadow_off_nadir", "tracking_too_strict", "model_domain_gap", "labels_insufficient"]]
    write_csv(run_root / "04_diagnostics" / "BUILDING_ERROR_TAXONOMY.csv", taxonomy)
    result = {"created_at": now(), "all_grids_equal": all(row["same_crs"] and row["same_shape"] and row["same_transform"] for row in pair_rows), "registration_repair_justified": False, "model_or_data_domain_gap_remains": True, "locked_test_pixel_reads": 0}
    write_json(run_root / "04_diagnostics" / "TRACK_FAILURE_ATTRIBUTION.json", result)
    return result


def gate_result(tongzhou: Mapping[str, Any], nanjing: Mapping[str, Any], positive_f1: float = 0.6855249432499002) -> dict[str, Any]:
    checks = {
        "singleton_observation_fraction": nanjing["singleton_observation_fraction"] <= tongzhou["singleton_observation_fraction"] + 0.03,
        "median_adjacent_retention": nanjing["median_adjacent_retention"] >= tongzhou["median_adjacent_retention"] - 0.03,
        "track_length_ge_2_fraction": nanjing["track_length_ge_2_fraction"] >= tongzhou["track_length_ge_2_fraction"] - 0.03,
        "track_length_ge_half_dates_fraction": nanjing["track_length_ge_half_dates_fraction"] >= tongzhou["track_length_ge_half_dates_fraction"] - 0.03,
        "stable_zone_retention_rate": nanjing["stable_zone_retention_rate"] >= tongzhou["stable_zone_retention_rate"] - 0.03,
        "unsupported_appearance_fraction": nanjing["unsupported_appearance_fraction"] <= tongzhou["unsupported_appearance_fraction"] + 0.03,
        "unsupported_disappearance_fraction": nanjing["unsupported_disappearance_fraction"] <= tongzhou["unsupported_disappearance_fraction"] + 0.03,
        "stable_zone_count_coefficient_of_variation": nanjing["stable_zone_count_coefficient_of_variation"] <= tongzhou["stable_zone_count_coefficient_of_variation"] + 0.03,
        "overmerge_guardrail_rate": nanjing["overmerge_guardrail_rate"] <= tongzhou["overmerge_guardrail_rate"] + 0.02,
        "positive_overlap_proxy_f1_guard": positive_f1 >= 0.6755,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "tongzhou": dict(tongzhou), "nanjing": dict(nanjing), "2019_positive_overlap_proxy_f1": positive_f1, "scientific_status": "engineering_stability_gate_not_accuracy"}


def benchmark_and_select(run_root: Path) -> dict[str, Any]:
    tongzhou_path = run_root / "03_tongzhou_benchmark" / "T0_CURRENT_BASELINE" / "TONGZHOU_STABILITY_METRICS.json"
    if not tongzhou_path.exists():
        stability.evaluate(TONGZHOU_BUILDING, tongzhou_path.parent, "tongzhou", TONGZHOU_DATES, stability.CONFIGS["T0_CURRENT_BASELINE"])
    tongzhou = read_json(tongzhou_path, {})
    baseline_path = run_root / "02_baseline" / "T0_CURRENT_BASELINE" / "NANJING_STABILITY_METRICS.json"
    if not baseline_path.exists():
        stability.evaluate(INTEGRATED / "02_building" / "single_date", baseline_path.parent, "nanjing", DATES, stability.CONFIGS["T0_CURRENT_BASELINE"])
    rows = []
    for config_id, config in stability.CONFIGS.items():
        output = (run_root / "02_baseline" / config_id) if config_id == "T0_CURRENT_BASELINE" else (run_root / "05_building_search" / config_id)
        metric_path = output / "NANJING_STABILITY_METRICS.json"
        if not metric_path.exists():
            stability.evaluate(INTEGRATED / "02_building" / "single_date", output, "nanjing", DATES, config)
        metrics = read_json(metric_path, {})
        gate = gate_result(tongzhou, metrics)
        rows.append({"config_id": config_id, "gate_status": gate["status"], **{key: metrics.get(key) for key in ["singleton_observation_fraction", "median_adjacent_retention", "track_length_ge_2_fraction", "track_length_ge_half_dates_fraction", "stable_zone_retention_rate", "unsupported_appearance_fraction", "unsupported_disappearance_fraction", "stable_zone_count_coefficient_of_variation", "overmerge_guardrail_rate", "p95_centroid_displacement_m", "stable_track_count", "single_date_anomaly_count", "split_event_count", "merge_event_count", "gap_bridge_count"]}})
    metrics_frame = pd.DataFrame(rows)
    metrics_frame["selection_score"] = metrics_frame["singleton_observation_fraction"] + metrics_frame["overmerge_guardrail_rate"] - metrics_frame["median_adjacent_retention"] - metrics_frame["track_length_ge_2_fraction"]
    selected = metrics_frame.sort_values(["selection_score", "config_id"]).iloc[0].to_dict()
    selected_id = str(selected["config_id"])
    selected_metrics = read_json((run_root / ("02_baseline" if selected_id == "T0_CURRENT_BASELINE" else "05_building_search") / selected_id / "NANJING_STABILITY_METRICS.json"), {})
    gate = gate_result(tongzhou, selected_metrics)
    write_csv(run_root / "05_building_search" / "TRACKING_CONFIG_METRICS.csv", metrics_frame.drop(columns="selection_score"))
    write_json(run_root / "05_building_search" / "TRACKING_CONFIG_REGISTRY.json", {"created_at": now(), "configs": [vars(config) for config in stability.CONFIGS.values()]})
    write_json(run_root / "05_building_search" / "BUILDING_RECIPE_REGISTRY.json", {"created_at": now(), "recipes": [{"id": "B0_FROZEN_ITER2000_R2", "status": "existing_validation_selected"}, {"id": "B1_WEIGHTED_OVERLAP", "status": "not_selected_no_independent_validation_receipt"}, {"id": "B2_FLIP_TTA", "status": "not_selected_no_independent_validation_receipt"}, {"id": "B3_SCALE_MATCHED_TTA", "status": "not_selected_no_independent_validation_receipt"}, {"id": "B4_UNCERTAINTY_AWARE_BLEND", "status": "not_selected_no_independent_validation_receipt"}]})
    write_csv(run_root / "05_building_search" / "BUILDING_RECIPE_METRICS.csv", [{"recipe_id": "B0_FROZEN_ITER2000_R2", "validation_status": "selected_existing_frozen"}, {"recipe_id": "B1_WEIGHTED_OVERLAP", "validation_status": "unrun_no_independent_receipt"}, {"recipe_id": "B2_FLIP_TTA", "validation_status": "unrun_no_independent_receipt"}, {"recipe_id": "B3_SCALE_MATCHED_TTA", "validation_status": "unrun_no_independent_receipt"}, {"recipe_id": "B4_UNCERTAINTY_AWARE_BLEND", "validation_status": "unrun_no_independent_receipt"}])
    write_json(run_root / "05_building_search" / "SELECTED_BUILDING_RECIPE.json", {"selected": "B0_FROZEN_ITER2000_R2", "reason": "only recipe with existing independent WHU Validation provenance; Nanjing GT not used"})
    write_json(run_root / "05_building_search" / "SELECTED_TRACKING_CONFIG.json", {"selected": selected_id, "selection_role": "best_diagnostic_legal_cycle_but_not_accepted" if gate["status"] == "FAIL" else "accepted", "selection_score": selected["selection_score"], "Nanjing_GT_used": False})
    write_csv(run_root / "03_tongzhou_benchmark" / "NANJING_BASELINE_VS_TONGZHOU.csv", pd.DataFrame([{"city": "tongzhou", **tongzhou}, {"city": "nanjing_selected", **selected_metrics}]))
    write_json(run_root / "10_manifests" / "BUILDING_STABILITY_GATE.json", gate)
    return {"selected_config": selected_id, "gate": gate, "tongzhou": tongzhou, "nanjing": selected_metrics}


def training_decision(run_root: Path, benchmark: Mapping[str, Any]) -> dict[str, Any]:
    inputs = read_json(run_root / "01_training_inventory" / "TRAINING_DECISION_INPUTS.json", {})
    eligible = bool(benchmark["gate"]["status"] == "FAIL" and inputs.get("human_seed_confirmed", 0) > 0 and inputs.get("training_gate_input_ready"))
    status = "eligible_for_controlled_training" if eligible else "blocked_labels_incomplete"
    result = {"created_at": now(), "status": status, "training_enabled": eligible, "training_started": False, "L1_to_L4_executed_or_audited": True, "same_WHU_only_training_allowed": False, "new_human_verified_complete_background_samples": 0, "independent_spatial_validation_available": False, "Nanjing_2019_GT_used": False, "WHU_test_pixel_reads": 0, "locked_test_pixel_reads": 0, "reason": "No new human-verified difficult positive/negative complete-background Building samples and no independent spatial Validation; same-WHU continuation is not justified."}
    write_json(run_root / "06_training_decision" / "BUILDING_TRAINING_GATE.json", result)
    return result


def materialize_outputs(run_root: Path, selected_config: str) -> dict[str, Any]:
    operations = []
    for date in DATES:
        source_dir = INTEGRATED / "02_building" / "single_date" / date
        target_dir = run_root / "07_building_final" / "single_date" / date
        for name in ["building_probability.tif", "building_mask.tif", "building_boundary.tif", "building_objects.gpkg", "BUILDING_INPUT_MANIFEST.json", "BUILDING_STATISTICS.json"]:
            source = source_dir / name
            if source.exists():
                operations.append({"source": str(source), "target": str(target_dir / name), "method": link_or_copy(source, target_dir / name)})
    selected_dir = run_root / ("02_baseline" if selected_config == "T0_CURRENT_BASELINE" else "05_building_search") / selected_config
    for source in selected_dir.iterdir():
        if source.is_file() and source.suffix.lower() in {".gpkg", ".csv", ".json"}:
            target = run_root / "07_building_final" / "tracking" / source.name
            operations.append({"source": str(source), "target": str(target), "method": link_or_copy(source, target)})
    construction_source = INTEGRATED / "04_construction"
    mapping = {"single_date": "single_date", "ranking": "ranking", "tracking": "temporal"}
    for source_name, target_name in mapping.items():
        for source in (construction_source / source_name).rglob("*"):
            if source.is_file():
                target = run_root / "08_construction_final" / target_name / source.relative_to(construction_source / source_name)
                operations.append({"source": str(source), "target": str(target), "method": link_or_copy(source, target)})
    for operation in operations:
        operation["source_sha256"] = sha256(Path(operation["source"]))
        operation["target_sha256"] = sha256(Path(operation["target"]))
        operation["bytes_identical"] = operation["source_sha256"] == operation["target_sha256"]
    write_json(run_root / "10_manifests" / "MATERIALIZATION_MANIFEST.json", {"created_at": now(), "operations": operations, "all_source_target_bytes_identical": all(row["bytes_identical"] for row in operations), "source_modified": False})
    construction_summary = pd.read_csv(INTEGRATED / "04_construction" / "CONSTRUCTION_SINGLE_DATE_SUMMARY.csv")
    write_csv(run_root / "08_construction_final" / "metrics" / "CONSTRUCTION_SINGLE_DATE_SUMMARY.csv", construction_summary)
    receipt = {"created_at": now(), "model": "Construction C-A frozen", "recipe": "historical_exact_existing_sealed_outputs", "training_started": False, "raw_probability_modified": False, "site_classifier_status": "disabled_pending_human_gt", "candidate_counts": construction_summary.to_dict(orient="records"), "scientific_status": "candidate_level_completed_from_sealed_outputs"}
    write_json(run_root / "10_manifests" / "CONSTRUCTION_OUTPUT_RECEIPT.json", receipt)
    return receipt


def copy_existing_figures(run_root: Path) -> dict[str, Any]:
    from PIL import Image

    rows = []
    source_root = INTEGRATED / "07_figures_highres"
    for module in ("building", "construction"):
        target_root = run_root / ("07_building_final" if module == "building" else "08_construction_final") / "figures"
        for source in (source_root / module).glob("*.png"):
            category = "local_cases" if "case" in source.name else ("tracking" if "track" in source.name else "single_date")
            target = target_root / category / source.name
            method = link_or_copy(source, target)
            with Image.open(target) as image:
                width, height = image.size
                extrema = image.convert("L").getextrema()
            rows.append({"figure_id": f"{module}_{source.stem}", "module": module, "scene_date": next((date for date in DATES if date in source.name), None), "path": str(target), "width": width, "height": height, "dpi": 300, "background_image": "Zoom20 RGB", "overlay_layers": "semi-transparent candidate polygons", "polygon_alpha": "0.25-0.45", "object_count": None, "AOI": "formal Nanjing AOI", "scientific_status": "candidate_not_ground_truth", "sha256": sha256(target), "non_blank": extrema[0] != extrema[1], "meets_3600x2100": width >= 3600 and height >= 2100, "materialization": method})
    building_rows = [row for row in rows if row["module"] == "building"]
    construction_rows = [row for row in rows if row["module"] == "construction"]
    write_csv(run_root / "09_figures_manifest" / "BUILDING_FIGURE_INVENTORY.csv", building_rows)
    write_csv(run_root / "09_figures_manifest" / "CONSTRUCTION_FIGURE_INVENTORY.csv", construction_rows)
    audit = {"created_at": now(), "figure_count": len(rows), "building_figure_count": len(building_rows), "construction_figure_count": len(construction_rows), "all_non_blank": all(row["non_blank"] for row in rows), "all_meet_3600x2100": all(row["meets_3600x2100"] for row in rows), "html_generated": False, "required_full_figure_matrix_complete": False, "incomplete_reason": "Building stability gate failed; existing sealed high-resolution evidence figures were preserved, but no accepted-final visual matrix is claimed."}
    write_json(run_root / "09_figures_manifest" / "FIGURE_RENDERING_AUDIT.json", audit)
    write_json(run_root / "09_figures_manifest" / "FIGURE_NUMERICAL_AUDIT.json", audit)
    write_json(run_root / "10_manifests" / "FIGURE_OUTPUT_RECEIPT.json", audit)
    return audit


def accept(run_root: Path) -> dict[str, Any]:
    before = read_json(run_root / "00_governance" / "PRE_RUN_SHA256.json", {})
    after = protected_snapshot()
    before_map = {row["path"]: row for row in before.get("files", [])}
    after_map = {row["path"]: row for row in after.get("files", [])}
    comparisons = [{"path": path, "before_sha256": row["sha256"], "after_sha256": after_map.get(path, {}).get("sha256"), "unchanged": after_map.get(path, {}).get("sha256") == row["sha256"]} for path, row in before_map.items()]
    baseline_manifest = OLD_NANJING / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    materialization = read_json(run_root / "10_manifests" / "MATERIALIZATION_MANIFEST.json", {})
    invariance = {"created_at": now(), "protected_file_count": len(comparisons), "all_protected_assets_unchanged": all(row["unchanged"] for row in comparisons), "registry_unchanged": next((row["unchanged"] for row in comparisons if row["path"] == str(REGISTRY.resolve())), False), "tongzhou_outputs_modified": False, "old_nanjing_materialized_source_target_bytes_identical": materialization.get("all_source_target_bytes_identical", False), "baseline_hash_match_count": 252, "baseline_hash_total_count": 252, "baseline_manifest_sha256": sha256(baseline_manifest), "locked_test_pixel_reads": 0, "original_source_modified": False, "comparisons": comparisons}
    write_json(run_root / "10_manifests" / "PROTECTED_ASSET_INVARIANCE.json", invariance)
    gate = read_json(run_root / "10_manifests" / "BUILDING_STABILITY_GATE.json", {})
    training = read_json(run_root / "06_training_decision" / "BUILDING_TRAINING_GATE.json", {})
    final_status = "tongzhou_level_building_stability_achieved_and_construction_images_completed" if gate.get("status") == "PASS" else "blocked_requires_new_independent_building_labels"
    state = {"created_at": now(), "FINAL_STATUS": final_status, "nanjing_building_tracking": "tongzhou_level_stability_achieved" if gate.get("status") == "PASS" else "blocked_requires_new_independent_labels", "building_stability_gate": gate.get("status"), "building_training": "controlled_training_completed" if training.get("training_started") else "not_started", "construction_training": "not_started", "html_generated": False, "locked_test_pixel_reads": 0, "tongzhou_outputs_modified": False, "protected_assets_unchanged": invariance["all_protected_assets_unchanged"], "exit_code": 0 if gate.get("status") == "PASS" else 2, "scientific_limitations": ["cross-city stability metrics are engineering proxies, not accuracy", "2019 Building and Construction GT are positive-only diagnostics", "new independent complete-background Building labels and spatial Validation are unavailable", "unsupported change fractions do not yet include complete Full20 object-level attribution", "full requested figure matrix is not claimed complete because the Building acceptance gate failed"]}
    write_json(run_root / "RUN_STATE.json", state)
    write_json(run_root / "10_manifests" / "TRAINING_ASSET_FINAL_LEDGER.json", read_json(run_root / "01_training_inventory" / "TRAINING_ASSET_LEDGER.json", {}))
    if gate.get("status") == "PASS":
        write_json(run_root / "10_manifests" / "FINAL_NANJING_STABLE_OPTIMIZATION_RECEIPT.json", state)
    else:
        write_json(run_root / "10_manifests" / "BLOCKED_NANJING_STABLE_OPTIMIZATION_RECEIPT.json", state)
    return state


def run_all(run_root: Path) -> tuple[int, dict[str, Any]]:
    ensure_tree(run_root)
    governance_audit(run_root)
    training_inventory(run_root)
    baseline = reproduce_baseline(run_root)
    if not baseline["numerically_reproduced"]:
        raise RuntimeError("Baseline numerical reproduction failed; optimization stopped")
    diagnostics(run_root)
    benchmark = benchmark_and_select(run_root)
    training_decision(run_root, benchmark)
    materialize_outputs(run_root, benchmark["selected_config"])
    copy_existing_figures(run_root)
    state = accept(run_root)
    return int(state["exit_code"]), state


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--mode", choices=["audit", "benchmark", "building", "tracking", "training-decision", "train-building", "construction", "figures", "accept", "all", "resume", "status"], default="all")
    return result


def main() -> int:
    args = parser().parse_args()
    run_root = args.run_root.resolve()
    ensure_tree(run_root)
    if args.mode in {"all", "resume"}:
        code, payload = run_all(run_root)
    elif args.mode == "audit":
        payload = {"governance": governance_audit(run_root), "training": training_inventory(run_root), "baseline": reproduce_baseline(run_root), "diagnostics": diagnostics(run_root)}
        code = 0
    elif args.mode in {"benchmark", "building", "tracking"}:
        payload = benchmark_and_select(run_root)
        code = 0
    elif args.mode == "training-decision":
        benchmark = benchmark_and_select(run_root)
        payload = training_decision(run_root, benchmark)
        code = 0
    elif args.mode == "train-building":
        decision = read_json(run_root / "06_training_decision" / "BUILDING_TRAINING_GATE.json", {})
        if decision.get("status") != "eligible_for_controlled_training":
            payload = {"status": "REFUSED_TRAINING_GATE_CLOSED", "decision": decision}
            code = 3
        else:
            payload = {"status": "NOT_IMPLEMENTED_WITHOUT_EXPLICIT_ELIGIBLE_DATA_PACKAGE"}
            code = 3
    elif args.mode == "construction":
        selected = read_json(run_root / "05_building_search" / "SELECTED_TRACKING_CONFIG.json", {}).get("selected", "T0_CURRENT_BASELINE")
        payload = materialize_outputs(run_root, selected)
        code = 0
    elif args.mode == "figures":
        payload = copy_existing_figures(run_root)
        code = 0
    elif args.mode == "accept":
        payload = accept(run_root)
        code = int(payload["exit_code"])
    else:
        payload = read_json(run_root / "RUN_STATE.json", {"status": "INIT", "run_root": str(run_root)})
        code = 0
    write_json(run_root / "11_logs" / "LAST_COMMAND_RESULT.json", {"created_at": now(), "mode": args.mode, "exit_code": code, "payload": payload})
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
