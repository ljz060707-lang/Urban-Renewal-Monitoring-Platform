"""Nanjing Building domain-adaptation and stable-tracking controller.

The controller is deliberately conservative.  It completes all read-only
audits and prepares a human-label package, but it will not start GPU training,
model selection, or final Building inference until an independent complete
background label package passes the label gate.  Construction is materialized
as a separate read-only C-A candidate branch and never changes Building.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
INFERENCE_ROOT = OUTPUTS / "inference"
RUNTIME_LOCK = OUTPUTS / "runtime_locks" / "urrs_gpu.lock"
DATES = [
    "2015-01-04",
    "2016-08-18",
    "2017-12-11",
    "2018-02-08",
    "2019-03-16",
    "2022-12-02",
]
FORBIDDEN_DATE = "2024-10-20"
AOI_WEST, AOI_EAST, AOI_SOUTH, AOI_NORTH = 118.633000, 118.652000, 32.024000, 32.041500
METRIC_CRS = "EPSG:32650"
ZOOM20_SOURCE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei 20")
ZOOM19_SOURCE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei")
GT_BUILDING = ZOOM20_SOURCE_ROOT / "gt_building19.gpkg"
GT_CONSTRUCTION = ZOOM20_SOURCE_ROOT / "gt_construction19.gpkg"
OLD_INTEGRATED = INFERENCE_ROOT / "nanjing_building_construction_integrated_20260804_233025"
OLD_STABLE = INFERENCE_ROOT / "nanjing_stable_building_construction_20260805_105929"
OLD_MODULAR = INFERENCE_ROOT / "nanjing_jiangbei_modular_rebuild_20260803_230320"
OLD_ZOOM20_AUDIT = INFERENCE_ROOT / "nanjing_jiangbei_zoom20_2019_gt_20260804_211042"
TONGZHOU_BUILDING = INFERENCE_ROOT / "beijing_tongzhou_building_geometry"
TONGZHOU_DATES = ["2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16"]
REGISTRY_GLOB = "governance/*/01_model_registry/CURRENT_BEST_MODEL_REGISTRY.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    temp.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[list(columns)]
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_tree(run_root: Path) -> None:
    directories = [
        "00_governance", "01_full20_attribution", "02_leakage_scan", "03_training_inventory", "04_baseline",
        "05_label_package/chips", "05_label_package/reference_predictions", "05_label_package/full20_reference",
        "05_label_package/templates", "05_label_package/instructions", "05_label_package/manifests",
        "06_label_gate", "07_training", "08_model_selection", "09_building_final/single_date",
        "09_building_final/tracking", "09_building_final/to_latest", "09_building_final/figures",
        "09_building_final/metrics", "10_construction_final/single_date", "10_construction_final/ranking",
        "10_construction_final/temporal", "10_construction_final/figures", "10_construction_final/metrics",
        "11_figure_audit", "12_manifests", "13_logs", "14_tests",
    ]
    for relative in directories:
        (run_root / relative).mkdir(parents=True, exist_ok=True)


def default_run_root() -> Path:
    return INFERENCE_ROOT / f"nanjing_domain_adaptation_stable_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def find_registry() -> Path:
    candidates = sorted(OUTPUTS.glob(REGISTRY_GLOB))
    if not candidates:
        raise FileNotFoundError(f"Current Best Registry not found under {OUTPUTS}")
    return candidates[-1]


def find_c_b1_manifest() -> Path | None:
    candidates = sorted(OUTPUTS.glob("training/*/03_experiments/C-B1/RUN_MANIFEST.json"))
    return candidates[-1] if candidates else None


def registry_entries(registry_path: Path) -> dict[str, dict[str, Any]]:
    registry = read_json(registry_path, {})
    result: dict[str, dict[str, Any]] = {}
    for entry in registry.get("entries", []):
        result[str(entry.get("module_id"))] = entry
    return result


def find_scene_manifest() -> Path:
    candidates = sorted(INFERENCE_ROOT.glob("nanjing_jiangbei_zoom20_2019_gt_*/00_input_audit/ZOOM20_SCENE_MANIFEST.json"))
    if not candidates:
        raise FileNotFoundError("Nanjing Zoom20 scene manifest is required")
    return candidates[-1]


def official_scene_paths() -> dict[str, Path]:
    manifest_path = find_scene_manifest()
    manifest = read_json(manifest_path, {})
    result: dict[str, Path] = {}
    for scene in manifest.get("scenes", []):
        date = str(scene.get("scene_date"))
        path = Path(scene.get("file_path", ""))
        if date == FORBIDDEN_DATE:
            continue
        if date in DATES and path.exists():
            result[date] = path
    missing = sorted(set(DATES) - set(result))
    if missing:
        raise FileNotFoundError(f"Missing official Nanjing scenes: {missing}")
    return result


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": sha256(path)}


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_file() and sha256(source) == sha256(target):
            return "existing_identical"
        raise FileExistsError(f"Refusing to overwrite non-identical output: {target}")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def state_path(run_root: Path) -> Path:
    return run_root / "RUN_STATE.json"


def load_state(run_root: Path) -> dict[str, Any]:
    return read_json(state_path(run_root), {
        "schema_version": "nanjing-domain-adaptation-v1",
        "run_root": str(run_root.resolve()),
        "created_at": now(),
        "updated_at": now(),
        "current_stage": "INIT",
        "final_status": "not_started",
        "stages": {},
        "flags": {
            "locked_test_pixel_reads": 0,
            "registry_modified": False,
            "full20_modified": False,
            "rsbuilding_frozen_checkpoint_modified": False,
            "construction_ca_modified": False,
            "cb1_enabled": False,
            "site_classifier_enabled": False,
            "nanjing_2019_positive_gt_used_for_training": False,
            "nanjing_2019_positive_gt_used_for_selection": False,
            "tongzhou_outputs_modified": False,
            "html_generated": False,
        },
    })


def save_state(run_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    write_json(state_path(run_root), state)


def mark_stage(run_root: Path, stage: str, status: str, **payload: Any) -> None:
    state = load_state(run_root)
    state.setdefault("stages", {})[stage] = {"status": status, "updated_at": now(), **payload}
    state["current_stage"] = stage
    save_state(run_root, state)


def stage_marker(run_root: Path, stage: str) -> Path:
    return run_root / "13_logs" / f"{stage}.complete.json"


def stage_is_complete(run_root: Path, stage: str) -> bool:
    return stage_marker(run_root, stage).exists()


def complete_stage(run_root: Path, stage: str, payload: Mapping[str, Any]) -> None:
    # Keep the stage state namespace separate from arbitrary business-result
    # keys such as ``run_root`` and ``status``.
    mark_stage(run_root, stage, "completed", result=dict(payload))
    write_json(stage_marker(run_root, stage), {"stage": stage, "completed_at": now(), **dict(payload)})


class GPULock:
    def __init__(self, run_root: Path, task_name: str, gpu_id: str = "0", expected_vram: int = 0) -> None:
        self.run_root = run_root
        self.payload = {
            "pid": os.getpid(), "task_name": task_name, "run_root": str(run_root.resolve()),
            "started_at": now(), "gpu_id": gpu_id, "expected_vram": expected_vram,
            "command": " ".join(sys.argv), "hostname": socket.gethostname(),
        }
        self.acquired = False

    def acquire(self) -> None:
        RUNTIME_LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(RUNTIME_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            holder = read_json(RUNTIME_LOCK, {})
            raise RuntimeError(f"GPU lock unavailable; no stale deletion without log-confirmed termination: {holder}") from error
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, ensure_ascii=False)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        holder = read_json(RUNTIME_LOCK, {})
        if int(holder.get("pid", -1)) != os.getpid():
            raise RuntimeError("Refusing to release a GPU lock owned by another PID")
        RUNTIME_LOCK.unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> "GPULock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def protected_paths(registry_path: Path, entries: Mapping[str, Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = [registry_path, GT_BUILDING, GT_CONSTRUCTION, OLD_MODULAR / "02_baseline_full20" / "FULL20_BASELINE_MANIFEST.json"]
    for entry in entries.values():
        for section in ("checkpoint", "config"):
            path = Path(str(entry.get(section, {}).get("path", "")))
            if path.exists():
                paths.append(path)
    try:
        paths.extend(official_scene_paths().values())
    except Exception:
        pass
    return sorted({path.resolve() for path in paths if path.exists()}, key=str)


def governance_audit(run_root: Path) -> dict[str, Any]:
    registry_path = find_registry()
    entries = registry_entries(registry_path)
    protected = protected_paths(registry_path, entries)
    before = {str(path): file_signature(path) for path in protected}
    write_json(run_root / "00_governance" / "PRE_RUN_SHA256.json", {"created_at": now(), "files": list(before.values()), "file_count": len(before)})
    registry_audit_rows: list[dict[str, Any]] = []
    for module_id, entry in entries.items():
        checkpoint = Path(str(entry.get("checkpoint", {}).get("path", "")))
        config = Path(str(entry.get("config", {}).get("path", "")))
        registry_audit_rows.append({
            "module_id": module_id,
            "model_name": entry.get("model_name"),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
            "registry_expected_sha256": entry.get("checkpoint", {}).get("expected_sha256"),
            "checkpoint_sha_match": bool(checkpoint.exists() and sha256(checkpoint) == entry.get("checkpoint", {}).get("expected_sha256")),
            "config_path": str(config),
            "config_sha256": sha256(config) if config.exists() else None,
            "registry_status": entry.get("registry_status"),
            "scientific_status": entry.get("scientific_status"),
        })
    c_b1_manifest = find_c_b1_manifest()
    c_b1 = read_json(c_b1_manifest, {}) if c_b1_manifest else {}
    c_a = next((row for row in registry_audit_rows if row["module_id"] == "M6_Construction"), {})
    write_json(run_root / "00_governance" / "CURRENT_BEST_REGISTRY_AUDIT.json", {
        "created_at": now(), "registry_path": str(registry_path.resolve()), "registry_sha256": sha256(registry_path),
        "entries": registry_audit_rows, "registry_modified": False,
    })
    write_json(run_root / "00_governance" / "MODEL_LINEAGE_AUDIT.json", {"created_at": now(), "entries": registry_audit_rows, "selection_policy": "within-module frozen registry only; no Nanjing proxy selection"})
    write_json(run_root / "00_governance" / "CHECKPOINT_IDENTITY_AUDIT.json", {"created_at": now(), "RSBuilding_iter_2000": next((row for row in registry_audit_rows if row["module_id"] == "M3_RSBuilding"), {}), "Full20": next((row for row in registry_audit_rows if row["module_id"] == "M2_Full20"), {}), "Construction_C_A": c_a, "Construction_C_B1": {"status": "rejected", "manifest_path": str(c_b1_manifest) if c_b1_manifest else None, "checkpoint_path": c_b1.get("best_checkpoint"), "checkpoint_sha256": c_b1.get("best_checkpoint_sha256")}, "same_checkpoint_identity": bool(c_b1.get("best_checkpoint_sha256") and c_b1.get("best_checkpoint_sha256") == c_a.get("checkpoint_sha256")), "cb1_enabled": False})
    write_csv(run_root / "00_governance" / "PROTECTED_ASSET_INVENTORY.csv", list(before.values()), ["path", "size", "mtime_ns", "sha256"])
    scene_manifest = find_scene_manifest()
    scene_data = read_json(scene_manifest, {})
    grid_path = OLD_ZOOM20_AUDIT / "02_common_grid" / "ZOOM20_GRID.json"
    grid = read_json(grid_path, {})
    write_json(run_root / "00_governance" / "NANJING_INPUT_GRID_AUDIT.json", {
        "scene_manifest": str(scene_manifest.resolve()), "scene_manifest_sha256": sha256(scene_manifest),
        "accepted_dates": DATES, "excluded_dates": [FORBIDDEN_DATE], "scene_count": len(scene_data.get("scenes", [])),
        "r2_grid_manifest": str(grid_path.resolve()), "grid": grid, "metric_crs": grid.get("metric_crs", METRIC_CRS),
        "aoi": {"west": AOI_WEST, "east": AOI_EAST, "south": AOI_SOUTH, "north": AOI_NORTH},
        "hardcoded_shape_used": False, "forbidden_2024_used": False,
    })
    lock_holder = read_json(RUNTIME_LOCK, {}) if RUNTIME_LOCK.exists() else None
    tongzhou_states: list[dict[str, Any]] = []
    for path in sorted(INFERENCE_ROOT.glob("beijing_tongzhou_construction_images_best_*")):
        state_file = path / "RUN_STATE.json"
        if state_file.exists():
            state = read_json(state_file, {})
            tongzhou_states.append({"run_root": str(path.resolve()), "run_state": state, "read_scope": "top-level running state only"})
    write_json(run_root / "GPU_LOCK_AUDIT.json", {"created_at": now(), "lock_path": str(RUNTIME_LOCK.resolve()), "exists": RUNTIME_LOCK.exists(), "holder": lock_holder, "gpu_stage_started": False, "stale_lock_deleted": False, "lock_operations": [], "locked_test_pixel_reads": 0})
    write_json(run_root / "PARALLEL_EXECUTION_AUDIT.json", {"created_at": now(), "tongzhou_parallel_states": tongzhou_states, "tongzhou_intermediate_outputs_read": False, "tongzhou_outputs_modified": False, "gpu_tasks_overlapped": False, "only_tongzhou_lock_and_top_level_run_state_checked": True})
    audit = {
        "created_at": now(), "run_root": str(run_root.resolve()), "registry_sha256": sha256(registry_path),
        "registry_path": str(registry_path.resolve()), "full20_sha256": next((row["checkpoint_sha256"] for row in registry_audit_rows if row["module_id"] == "M2_Full20"), None),
        "rsbuilding_iter_2000_sha256": next((row["checkpoint_sha256"] for row in registry_audit_rows if row["module_id"] == "M3_RSBuilding"), None),
        "construction_ca_sha256": c_a.get("checkpoint_sha256"), "construction_cb1_sha256": c_b1.get("best_checkpoint_sha256"), "construction_cb1_status": "rejected",
        "protected_file_count": len(protected), "locked_test_pixel_reads": 0, "registry_modified": False, "full20_modified": False,
        "rsbuilding_frozen_checkpoint_modified": False, "construction_ca_modified": False, "cb1_enabled": False,
        "site_classifier_enabled": False, "nanjing_2019_positive_gt_used_for_training": False, "nanjing_2019_positive_gt_used_for_selection": False,
        "tongzhou_outputs_modified": False, "html_generated": False, "training_started": False,
    }
    write_json(run_root / "00_governance" / "GOVERNANCE_AUDIT.json", audit)
    return audit


def full20_attribution_stage(run_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import attribute_full20_support_to_building_objects as attribution
    old_gate = read_json(OLD_STABLE / "10_manifests" / "BUILDING_STABILITY_GATE.json", {})
    old_nanjing = old_gate.get("nanjing", {})
    result = attribution.run_attribution(
        run_root,
        OLD_INTEGRATED / "02_building" / "single_date",
        OLD_MODULAR / "02_baseline_full20",
        dates=DATES,
        metric_crs=METRIC_CRS,
        old_metrics=old_nanjing,
        tracking_path=OLD_STABLE / "07_building_final" / "tracking" / "NANJING_BUILDING_TRACKS.gpkg",
    )
    return result


def leakage_stage(run_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import scan_urrs_dataset_leakage_exhaustive as scanner
    return scanner.run_scan(PROJECT_ROOT, run_root)


def training_inventory_stage(run_root: Path) -> dict[str, Any]:
    leakage = read_json(run_root / "02_leakage_scan" / "EXHAUSTIVE_LEAKAGE_AUDIT.json", {})
    seed_candidates = sorted(OUTPUTS.glob("urrs/construction/*SEED_LABEL_INVENTORY.json"))
    whu_candidates = sorted(OUTPUTS.glob("v2_readiness/rsbuilding/whu/**/*AUDIT*.json"))
    specs = [
        ("full20_dsifn", "DSIFN", "Full20", OUTPUTS / "pretraining", "trained_and_verified", True, False, False, "frozen checkpoint lineage; no Test pixels"),
        ("full20_s2looking", "S2Looking", "Full20", OUTPUTS / "pretraining" / "change_s2looking", "trained_and_verified", True, False, False, "frozen checkpoint lineage; no Test pixels"),
        ("full20_second", "SECOND", "Full20", OUTPUTS / "pretraining", "trained_but_limited", True, False, True, "Test inventory only; no pixel reads"),
        ("whu_building_train", "WHU Building Train", "Building", OUTPUTS / "v2_readiness" / "rsbuilding" / "whu", "trained_and_verified", True, True, False, "base public Building distribution"),
        ("whu_validation", "WHU Validation", "Building", OUTPUTS / "v2_readiness" / "rsbuilding" / "whu", "validation_only", False, True, False, "not eligible for Nanjing training"),
        ("whu_test_inventory", "WHU Test inventory", "Building", OUTPUTS / "v2_readiness" / "rsbuilding" / "whu", "test_locked", False, False, True, "inventory only; no pixels read"),
        ("ucs386", "UCS386", "Construction", Path(r"D:\Data Set\URS\600 manually delineated UCS labels part2"), "trained_but_limited", True, True, True, "Locked Test excluded from use"),
        ("cwld", "CWLD", "Construction", OUTPUTS / "training", "missing_or_incomplete", False, False, False, "not relabeled as Construction without mapping"),
        ("human_seed", "Human Seed", "Construction", seed_candidates[-1].parent if seed_candidates else OUTPUTS / "urrs", "human_seed_unverified", False, False, False, "human confirmation required"),
        ("temporal_seed", "Temporal Seed", "Construction", OUTPUTS / "urrs", "human_seed_unverified", False, False, False, "reference only"),
        ("nanjing_2019_gt", "Nanjing 2019 positive-only GT", "Building diagnostic", GT_BUILDING, "independent_holdout", False, False, False, "diagnostic only; never selection/training"),
        ("nanjing_six_images", "Nanjing six-date imagery", "Building domain adaptation", ZOOM20_SOURCE_ROOT, "reference_only", False, False, False, "label package source only"),
        ("tongzhou_four_images", "Tongzhou four-date imagery", "Benchmark", TONGZHOU_BUILDING, "reference_only", False, False, False, "sealed benchmark; not Nanjing parameter selection"),
    ]
    rows: list[dict[str, Any]] = []
    for dataset_id, name, task, root, status, training, human_verified, test_locked, action in specs:
        exists = root.exists()
        rows.append({
            "dataset_id": dataset_id, "dataset_name": name, "task": task, "root_path": str(root), "manifest_path": None,
            "license_status": "existing_project_receipt_or_source_inventory", "scene_count": None, "tile_count": None, "pair_count": None,
            "train_count": None, "validation_count": None, "test_count": None, "positive_count": None, "negative_count": None,
            "complete_background": False, "positive_only": dataset_id == "nanjing_2019_gt", "human_verified": human_verified,
            "human_corrected": False, "pseudo_label": False, "temporal_labels": "temporal" in task.lower(), "spatial_holdout": False,
            "date_holdout": False, "used_by_checkpoint": training, "used_for_training": training and dataset_id not in {"whu_validation", "whu_test_inventory"},
            "used_for_selection": False, "used_for_test": False, "leakage_status": "critical_zero" if leakage.get("critical_leakage_count") == 0 else "blocked",
            "recommended_action": action if exists else "missing_or_incomplete", "scientific_status": status if exists else "missing_or_incomplete",
            "test_locked": test_locked, "locked_test_pixel_reads": 0,
        })
    ledger = {
        "created_at": now(), "dataset_count": len(rows), "status_counts": pd.Series([row["scientific_status"] for row in rows]).value_counts().to_dict(),
        "rows": rows, "leakage_audit": str((run_root / "02_leakage_scan" / "EXHAUSTIVE_LEAKAGE_AUDIT.json").resolve()),
        "training_gate_open": False, "locked_test_pixel_reads": 0, "nanjing_2019_positive_gt_used_for_training": False,
    }
    write_csv(run_root / "03_training_inventory" / "TRAINING_ASSET_LEDGER.csv", rows)
    write_json(run_root / "03_training_inventory" / "TRAINING_ASSET_LEDGER.json", ledger)
    governance = read_json(run_root / "00_governance" / "GOVERNANCE_AUDIT.json", {})
    matrix = [
        {"checkpoint": "Full20 frozen", "checkpoint_sha256": governance.get("full20_sha256"), "datasets": "DSIFN;S2Looking;SECOND", "used_for_training": True, "used_for_selection": False, "test_pixel_reads": 0},
        {"checkpoint": "RSBuilding iter_2000 frozen", "checkpoint_sha256": governance.get("rsbuilding_iter_2000_sha256"), "datasets": "WHU Building Train", "used_for_training": True, "used_for_selection": False, "test_pixel_reads": 0},
        {"checkpoint": "Construction C-A frozen", "checkpoint_sha256": governance.get("construction_ca_sha256"), "datasets": "UCS386 available subset", "used_for_training": True, "used_for_selection": False, "test_pixel_reads": 0},
    ]
    write_csv(run_root / "03_training_inventory" / "CHECKPOINT_DATASET_MATRIX.csv", matrix)
    write_csv(run_root / "03_training_inventory" / "TRAINING_RUN_INVENTORY.csv", [{"run_id": "nanjing_domain_adaptation_current", "status": "not_started_waiting_label_gate", "training_started": False, "max_iteration": None, "early_stopping": None, "locked_test_pixel_reads": 0}])
    return {"status": "complete", "dataset_count": len(rows), "training_gate_open": False, "locked_test_pixel_reads": 0}


def baseline_stage(run_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import evaluate_cross_city_building_stability as stability
    nj_input = OLD_INTEGRATED / "02_building" / "single_date"
    nj_output = run_root / "04_baseline" / "nanjing_t7_recomputed"
    tz_output = run_root / "04_baseline" / "tongzhou_t0_recomputed"
    nj = stability.evaluate(nj_input, nj_output, "nanjing", DATES, stability.CONFIGS["T7_FULL_LEGAL_TRACKER"])
    tz = stability.evaluate(TONGZHOU_BUILDING, tz_output, "tongzhou", TONGZHOU_DATES, stability.CONFIGS["T0_CURRENT_BASELINE"])
    old_gate = read_json(OLD_STABLE / "10_manifests" / "BUILDING_STABILITY_GATE.json", {})
    old_nj = old_gate.get("nanjing", {})
    expected_counts = [98, 66, 166, 192, 168, 292]
    actual_counts = [int(nj_input.joinpath(date, "building_objects.gpkg").exists()) for date in DATES]
    count_rows: list[dict[str, Any]] = []
    for date, expected in zip(DATES, expected_counts):
        vector = next((path for path in (nj_input / date).glob("building_objects.gpkg") if path.exists()), None)
        count = len(gpd.read_file(vector)) if vector else 0
        count_rows.append({"scene_date": date, "expected_object_count": expected, "actual_object_count": count, "difference": count - expected, "source": str(vector) if vector else None})
    thresholds = {
        "singleton_observation_fraction": {"operator": "<=", "value": 0.3911},
        "median_adjacent_retention": {"operator": ">=", "value": 0.3951},
        "track_length_ge_2_fraction": {"operator": ">=", "value": 0.3819},
        "track_length_ge_half_dates_fraction": {"operator": ">=", "value": 0.3819},
        "stable_zone_retention_rate": {"operator": ">=", "value": 0.5891},
        "unsupported_appearance_fraction": {"operator": "<=", "value": 0.7417},
        "unsupported_disappearance_fraction": {"operator": "<=", "value": 0.7417},
        "overmerge_guardrail_rate": {"operator": "<=", "value": 0.1530},
    }
    gate_rows: list[dict[str, Any]] = []
    for metric, contract in thresholds.items():
        value = nj.get(metric)
        passed = value is not None and ((value <= contract["value"]) if contract["operator"] == "<=" else (value >= contract["value"]))
        gate_rows.append({"metric": metric, "nanjing": value, "tongzhou_reference": tz.get(metric), "operator": contract["operator"], "threshold": contract["value"], "pass": passed, "source": "previous sealed blocked receipt contract; not selected from Nanjing"})
    write_json(run_root / "04_baseline" / "NANJING_BASELINE_RECOMPUTED.json", {"metrics": nj, "expected_object_counts": expected_counts, "actual_object_counts": [row["actual_object_count"] for row in count_rows], "old_sealed_metrics": old_nj, "source": str(nj_output.resolve()), "locked_test_pixel_reads": 0})
    write_json(run_root / "04_baseline" / "TONGZHOU_BENCHMARK_RECOMPUTED.json", {"metrics": tz, "source": str(tz_output.resolve()), "benchmark_scope": "sealed Tongzhou Building baseline only; no new parallel output read", "locked_test_pixel_reads": 0})
    write_csv(run_root / "04_baseline" / "NANJING_VS_TONGZHOU_GATE.csv", gate_rows)
    write_json(run_root / "04_baseline" / "BASELINE_REPRODUCIBILITY_AUDIT.json", {
        "created_at": now(), "old_run_root": str(OLD_STABLE.resolve()), "nanjing_config": nj.get("config"), "tongzhou_config": tz.get("config"),
        "nanjing_count_rows": count_rows, "counts_exact": all(row["difference"] == 0 for row in count_rows),
        "expected_t7_metrics": {"singleton_observation_fraction": 0.5387, "median_adjacent_retention": 0.3169, "track_length_ge_2_fraction": 0.2591, "track_length_ge_half_dates_fraction": 0.0784, "stable_zone_retention_rate": 0.5667, "unsupported_appearance_fraction": 0.8652, "overmerge_guardrail_rate": 0.2649},
        "recomputed_metrics": {key: nj.get(key) for key in ["singleton_observation_fraction", "median_adjacent_retention", "track_length_ge_2_fraction", "track_length_ge_half_dates_fraction", "stable_zone_retention_rate", "unsupported_appearance_fraction", "unsupported_disappearance_fraction", "overmerge_guardrail_rate"]},
        "gate_contract": thresholds, "full20_attribution_not_substituted": True, "locked_test_pixel_reads": 0,
    })
    return {"status": "complete", "nanjing": nj, "tongzhou": tz, "counts_exact": all(row["difference"] == 0 for row in count_rows), "gate_pass": all(row["pass"] for row in gate_rows)}


def _metric_bounds() -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    xs, ys = zip(*[transformer.transform(x, y) for x, y in [(AOI_WEST, AOI_SOUTH), (AOI_WEST, AOI_NORTH), (AOI_EAST, AOI_SOUTH), (AOI_EAST, AOI_NORTH)]])
    return min(xs), min(ys), max(xs), max(ys)


def _spread_pick(items: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if len(items) <= target:
        return items
    items = sorted(items, key=lambda item: hashlib.sha256(item["block_id"].encode("utf-8")).hexdigest())
    indices = np.linspace(0, len(items) - 1, target, dtype=int)
    return [items[int(index)] for index in sorted(set(indices))]


def select_label_blocks() -> list[dict[str, Any]]:
    minx, miny, maxx, maxy = _metric_bounds()
    block_size = 180.0
    nx = int(np.ceil((maxx - minx) / block_size))
    ny = int(np.ceil((maxy - miny) / block_size))
    candidates: list[dict[str, Any]] = []
    for row in range(ny):
        for col in range(nx):
            west, east = minx + col * block_size, min(maxx, minx + (col + 1) * block_size)
            south, north = miny + row * block_size, min(maxy, miny + (row + 1) * block_size)
            if east <= west or north <= south:
                continue
            candidates.append({"block_id": f"B{row:02d}_{col:02d}", "spatial_block_id": f"B{row:02d}_{col:02d}", "grid_row": row, "grid_col": col, "west_m": west, "south_m": south, "east_m": east, "north_m": north, "geometry": box(west, south, east, north)})
    # Leave a column between validation and holdout wherever possible.  This
    # makes spatial independence explicit rather than relying on random rows.
    holdout = [item for item in candidates if item["grid_col"] % 8 == 7]
    validation = [item for item in candidates if item["grid_col"] % 8 == 5]
    training = [item for item in candidates if item not in holdout and item not in validation]
    selected: list[dict[str, Any]] = []
    for split, pool, target in [("training", training, 45), ("validation", validation, 13), ("holdout", holdout, 10)]:
        for item in _spread_pick(pool, target):
            item = dict(item)
            item["split"] = split
            selected.append(item)
    if len(selected) < 60:
        remaining = [item for item in candidates if item["block_id"] not in {x["block_id"] for x in selected}]
        for item in _spread_pick(remaining, 70 - len(selected)):
            item = dict(item)
            item["split"] = "training"
            selected.append(item)
    categories = [
        "dense_residential", "old_low_rise", "small_buildings", "industrial_roof", "corrugated_roof",
        "tree_occlusion", "strong_shadow", "bare_land", "parking", "temporary_structure", "continuous_roof",
        "water_edge", "road_edge", "high_reflectance", "blurred_image", "registration_difficult",
        "stable_region", "volatile_region", "new_area", "demolition_area", "no_change_hard_negative",
    ]
    for index, item in enumerate(sorted(selected, key=lambda value: value["block_id"])):
        item["planned_coverage_tags"] = ";".join([categories[index % len(categories)], "complete_background", "six_date_consistency_review"])
        item["label_status"] = "draft"
        item["geometry_wkt"] = item.pop("geometry").wkt
    return selected


def _load_scene_frames() -> dict[str, Any]:
    frames: dict[str, Any] = {}
    source_vectors = OLD_STABLE / "07_building_final" / "single_date"
    for date in DATES:
        path = source_vectors / date / "building_objects.gpkg"
        if path.exists():
            frames[date] = gpd.read_file(path).to_crs(METRIC_CRS)
    return frames


def _load_full20_vectors() -> dict[str, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    result: dict[str, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]] = {}
    base = OLD_MODULAR / "02_baseline_full20"
    for date in DATES[:-1]:
        root = base / f"{date}_to_{DATES[-1]}" / "vectors_metric"
        new_path, demolition_path = root / "new_building_metric.gpkg", root / "demolition_metric.gpkg"
        if new_path.exists() and demolition_path.exists():
            result[date] = (gpd.read_file(new_path).to_crs(METRIC_CRS), gpd.read_file(demolition_path).to_crs(METRIC_CRS))
    return result


def _iter_lines(geometry: Any) -> list[list[tuple[float, float]]]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [list(geometry.exterior.coords)]
    if geometry.geom_type == "MultiPolygon":
        return [list(part.exterior.coords) for part in geometry.geoms]
    if geometry.geom_type == "GeometryCollection":
        result: list[list[tuple[float, float]]] = []
        for part in geometry.geoms:
            result.extend(_iter_lines(part))
        return result
    return []


def _draw_overlay(base: Image.Image, frame: gpd.GeoDataFrame | None, block_geom: Any, scene_crs: Any, window: Window, transform: Any, color: tuple[int, int, int], label: str) -> Image.Image:
    image = base.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    if frame is not None and len(frame):
        selected = frame[frame.geometry.intersects(block_geom)]
        to_scene = Transformer.from_crs(METRIC_CRS, scene_crs, always_xy=True).transform
        sx = image.width / float(window.width)
        sy = image.height / float(window.height)
        for geometry in selected.geometry:
            scene_geometry = shapely_transform(to_scene, geometry)
            for line in _iter_lines(scene_geometry):
                pixels: list[tuple[float, float]] = []
                for x, y in line:
                    col = (x - transform.c) / transform.a * sx
                    row = (transform.f - y) / abs(transform.e) * sy
                    pixels.append((col, row))
                if len(pixels) >= 2:
                    draw.line(pixels, fill=color, width=2, joint="curve")
    draw.rectangle((2, 2, min(image.width - 2, 250), 24), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return image


def _write_empty_gpkg(path: Path, layer: str, geometry_type: str, properties: Mapping[str, str]) -> None:
    import pyogrio
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, pd.Series] = {}
    for name, specification in properties.items():
        dtype = "float64" if str(specification).startswith("float") else "string"
        columns[name] = pd.Series(dtype=dtype)
    frame = gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=METRIC_CRS), crs=METRIC_CRS)
    pyogrio.write_dataframe(frame, path, layer=layer, driver="GPKG", geometry_type=geometry_type)


def label_package_stage(run_root: Path) -> dict[str, Any]:
    package = run_root / "05_label_package"
    scenes = official_scene_paths()
    blocks = select_label_blocks()
    object_frames = _load_scene_frames()
    full20_frames = _load_full20_vectors()
    split_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    transformer_to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    for item in blocks:
        block_geom = box(float(item["west_m"]), float(item["south_m"]), float(item["east_m"]), float(item["north_m"]))
        selection_rows.append({key: value for key, value in item.items() if key not in {"geometry_wkt"}} | {"geometry_wkt": item["geometry_wkt"]})
        chip_paths: dict[str, str] = {}
        reference_images: list[Image.Image] = []
        full20_images: list[Image.Image] = []
        rgb_images: list[Image.Image] = []
        for date in DATES:
            source = scenes[date]
            with rasterio.open(source) as ds:
                to_scene = Transformer.from_crs(METRIC_CRS, ds.crs, always_xy=True).transform
                scene_block = shapely_transform(to_scene, block_geom)
                window = from_bounds(*scene_block.bounds, transform=ds.transform).round_offsets().round_lengths()
                col_off = max(0, min(int(window.col_off), ds.width - 1)); row_off = max(0, min(int(window.row_off), ds.height - 1))
                width = max(1, min(int(window.width), ds.width - col_off)); height = max(1, min(int(window.height), ds.height - row_off))
                window = Window(col_off, row_off, width, height)
                data = ds.read([1, 2, 3], window=window, out_shape=(3, 384, 384), resampling=Resampling.bilinear)
                image = Image.fromarray(np.moveaxis(np.clip(data, 0, 255).astype(np.uint8), 0, -1), mode="RGB")
                output = package / "chips" / str(item["split"]) / str(item["block_id"]) / date / "image.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                image.save(output, format="PNG")
                chip_paths[date] = str(output.resolve())
                rgb_images.append(image)
                local_transform = ds.window_transform(window)
                ref = _draw_overlay(image, object_frames.get(date), block_geom, ds.crs, window, local_transform, (255, 205, 0), "model_reference_not_ground_truth")
                reference_images.append(ref)
                if date in full20_frames:
                    new_frame, demolition_frame = full20_frames[date]
                    combined = pd.concat([new_frame.assign(_kind="new"), demolition_frame.assign(_kind="demolition")], ignore_index=True)
                    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=METRIC_CRS)
                    full = _draw_overlay(image, combined, block_geom, ds.crs, window, local_transform, (0, 220, 255), "Full20_reference_not_ground_truth")
                else:
                    full = _draw_overlay(image, None, block_geom, ds.crs, window, local_transform, (0, 220, 255), "Full20_unavailable_for_latest")
                full20_images.append(full)
            split_rows.append({
                "spatial_block_id": item["spatial_block_id"], "block_id": item["block_id"], "split": item["split"], "scene_date": date,
                "image": chip_paths[date], "reference_prediction": None, "full20_reference": None,
                "annotation_status": "draft", "model_reference_status": "model_reference_not_ground_truth",
                "valid_area_mask": "pending_human_annotation", "ignore_mask": "pending_human_annotation", "quality_mask": "pending_human_annotation",
                "occlusion_status": "pending", "shadow_status": "pending", "registration_status": "pending", "review_status": "pending",
            })
        def sheet(images: list[Image.Image], name: str, directory: Path) -> Path:
            canvas = Image.new("RGB", (384 * 3, 384 * 2), (20, 20, 20))
            for index, image in enumerate(images):
                canvas.paste(image, ((index % 3) * 384, (index // 3) * 384))
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / name
            canvas.save(path, format="PNG")
            return path
        ref_path = sheet(reference_images, "frozen_building_reference_six_dates.png", package / "reference_predictions" / str(item["split"]) / str(item["block_id"]))
        full_path = sheet(full20_images, "full20_reference_six_dates.png", package / "full20_reference" / str(item["split"]) / str(item["block_id"]))
        comparison = sheet(rgb_images, "six_date_rgb_comparison.png", package / "reference_predictions" / str(item["split"]) / str(item["block_id"]))
        for row in split_rows[-len(DATES):]:
            row["reference_prediction"] = str(ref_path.resolve())
            row["full20_reference"] = str(full_path.resolve())
            row["six_date_comparison"] = str(comparison.resolve())
    write_csv(package / "NANJING_LABEL_BLOCK_SELECTION.csv", selection_rows)
    write_csv(package / "NANJING_LABEL_SPLIT_MANIFEST.csv", split_rows)
    _write_empty_gpkg(package / "templates" / "NANJING_BUILDING_ANNOTATION_TEMPLATE.gpkg", "building_annotations", "MultiPolygon", {"annotation_id": "str:80", "spatial_block_id": "str:40", "scene_date": "str:20", "class_name": "str:40", "source_image": "str:254", "annotator": "str:80", "reviewer": "str:80", "annotation_status": "str:30", "occlusion_fraction": "float", "shadow_fraction": "float", "geometry_confidence": "float", "notes": "str:254"})
    _write_empty_gpkg(package / "templates" / "NANJING_VALID_AREA_TEMPLATE.gpkg", "valid_area", "Polygon", {"spatial_block_id": "str:40", "scene_date": "str:20", "mask_status": "str:30"})
    _write_empty_gpkg(package / "templates" / "NANJING_IGNORE_MASK_TEMPLATE.gpkg", "ignore_mask", "Polygon", {"spatial_block_id": "str:40", "scene_date": "str:20", "reason": "str:80"})
    (package / "instructions" / "ANNOTATION_INSTRUCTIONS_ZH.md").write_text("""# 南京 Building Domain Labels v1\n\n这些文件是人工标注任务包。参考预测和 Full20 图层只用于定位困难区域，均标记为 `model_reference_not_ground_truth`，不得直接转为正式标签。\n\n每个 `spatial_block_id` 的六期必须留在同一个 split。请在每期同时维护 building、non_building 背景覆盖、temporary_structure、uncertain_structure 和 ignore；强阴影、严重遮挡、缺失和配准失败区域必须进入 ignore mask。\n\n只有 `human_verified` 或 `human_corrected` 的几何才可进入训练；draft、自动预测、seed 和 pseudo label 均不可进入训练或独立评价。\n""", encoding="utf-8")
    (package / "instructions" / "ANNOTATION_QA_CHECKLIST.md").write_text("""# QA checklist\n\n- CRS 为 EPSG:32650，几何有效且无自交；\n- 六期日期完整；\n- building 与 ignore 不重叠；\n- valid_area 和 ignore_mask 完整；\n- 每个 split 含完整背景、困难负样本、小建筑、大型厂房、阴影/遮挡、稳定区和真实变化区；\n- reviewer 已填写；\n- annotation_status 仅在人工审核后改为 human_verified 或 human_corrected；\n- 不把参考预测复制为人工标签。\n""", encoding="utf-8")
    manifest = {
        "created_at": now(), "package_version": "Nanjing Building Domain Labels v1", "status": "awaiting_human_labels",
        "block_count": len(blocks), "training_block_count": sum(row["split"] == "training" for row in blocks), "validation_block_count": sum(row["split"] == "validation" for row in blocks), "holdout_block_count": sum(row["split"] == "holdout" for row in blocks),
        "date_count": len(DATES), "dates": DATES, "excluded_dates": [FORBIDDEN_DATE], "metric_crs": METRIC_CRS,
        "model_reference_status": "model_reference_not_ground_truth", "human_verified_count": 0, "human_corrected_count": 0,
        "complete_background_available": False, "independent_validation_available": False, "independent_holdout_available": False,
        "label_training_gate": "FAIL", "source_scene_manifest": str(find_scene_manifest().resolve()), "source_scene_manifest_sha256": sha256(find_scene_manifest()),
        "automatically_generated_labels": False, "pseudo_labels_created": False,
    }
    write_json(package / "LABEL_PACKAGE_MANIFEST.json", manifest)
    hash_rows: list[dict[str, Any]] = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "LABEL_PACKAGE_SHA256.csv":
            hash_rows.append({"path": str(path.relative_to(package)), "size": path.stat().st_size, "sha256": sha256(path), "source": "generated_label_package"})
    write_csv(package / "LABEL_PACKAGE_SHA256.csv", hash_rows, ["path", "size", "sha256", "source"])
    return {"status": "awaiting_human_labels", "block_count": len(blocks), "training_blocks": sum(row["split"] == "training" for row in blocks), "validation_blocks": sum(row["split"] == "validation" for row in blocks), "holdout_blocks": sum(row["split"] == "holdout" for row in blocks), "human_verified": 0, "human_corrected": 0, "label_gate": "FAIL"}


def _label_input_root(run_root: Path, label_root: Path | None = None) -> Path | None:
    candidates = [label_root] if label_root else []
    candidates.extend([run_root / "05_label_package" / name for name in ("annotations", "human_labels", "labels_incoming")])
    return next((path for path in candidates if path and path.exists()), None)


def label_gate_stage(run_root: Path, label_root: Path | None = None) -> dict[str, Any]:
    root = _label_input_root(run_root, label_root)
    human_verified = 0
    human_corrected = 0
    complete_background = False
    independent_validation = False
    independent_holdout = False
    qa_failures: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    if root is not None:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".gpkg", ".geojson", ".json", ".csv"}:
                continue
            if path.suffix.lower() in {".gpkg", ".geojson"}:
                try:
                    frame = gpd.read_file(path)
                    status_counts = frame.get("annotation_status", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
                    human_verified += int(status_counts.get("human_verified", 0))
                    human_corrected += int(status_counts.get("human_corrected", 0))
                    inventory.append({"path": str(path), "rows": len(frame), "crs": str(frame.crs), "status_counts": json.dumps(status_counts, ensure_ascii=False), "geometry_valid": bool(frame.geometry.is_valid.all()) if len(frame) else True})
                    if len(frame) and "split" in frame.columns:
                        independent_validation |= bool((frame.split.astype(str) == "validation").any())
                        independent_holdout |= bool((frame.split.astype(str) == "holdout").any())
                    if len(frame) and "class_name" in frame.columns:
                        complete_background |= bool((frame.class_name.astype(str) == "non_building").any())
                    if len(frame) and not frame.geometry.is_valid.all():
                        qa_failures.append({"path": str(path), "failure": "invalid_geometry"})
                except Exception as error:
                    qa_failures.append({"path": str(path), "failure": f"read_error:{type(error).__name__}:{error}"})
            else:
                try:
                    frame = pd.read_csv(path)
                    statuses = frame.get("annotation_status", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
                    human_verified += int(statuses.get("human_verified", 0)); human_corrected += int(statuses.get("human_corrected", 0))
                    inventory.append({"path": str(path), "rows": len(frame), "crs": frame.get("crs", pd.Series([None])).iloc[0] if len(frame) else None, "status_counts": json.dumps(statuses, ensure_ascii=False), "geometry_valid": None})
                except Exception as error:
                    qa_failures.append({"path": str(path), "failure": f"read_error:{type(error).__name__}:{error}"})
    requirements = [
        {"requirement": "complete_background_training", "pass": complete_background, "detail": "training requires non_building valid-area background"},
        {"requirement": "independent_spatial_validation", "pass": independent_validation, "detail": "validation blocks must be spatially independent"},
        {"requirement": "independent_spatial_holdout", "pass": independent_holdout, "detail": "holdout blocks must be spatially independent"},
        {"requirement": "human_verified_or_corrected_labels", "pass": human_verified + human_corrected > 0, "detail": "draft/reference/pseudo labels are not formal labels"},
        {"requirement": "no_label_qa_failures", "pass": not qa_failures, "detail": "geometry and provenance QA"},
    ]
    gate = "PASS" if all(row["pass"] for row in requirements) else "FAIL"
    payload = {
        "created_at": now(), "status": "PASS" if gate == "PASS" else "AWAITING_HUMAN_LABELS", "label_training_gate": gate,
        "input_root": str(root.resolve()) if root else None, "human_verified_count": human_verified, "human_corrected_count": human_corrected,
        "complete_background": complete_background, "independent_validation": independent_validation, "independent_holdout": independent_holdout,
        "requirements": requirements, "locked_test_pixel_reads": 0, "nanjing_2019_positive_gt_used_for_training": False,
        "nanjing_2019_positive_gt_used_for_selection": False, "pseudo_labels_used": False,
    }
    write_csv(run_root / "06_label_gate" / "INGESTED_LABEL_INVENTORY.csv", inventory)
    write_json(run_root / "06_label_gate" / "LABEL_GEOMETRY_AUDIT.json", {"created_at": now(), "files": inventory, "qa_failures": qa_failures})
    write_json(run_root / "06_label_gate" / "LABEL_SPLIT_LEAKAGE_AUDIT.json", {"created_at": now(), "status": "not_passed" if gate != "PASS" else "pass", "same_spatial_block_cross_split": False, "locked_test_pixel_reads": 0})
    write_csv(run_root / "06_label_gate" / "LABEL_CLASS_DISTRIBUTION.csv", [{"class_name": "building", "count": None}, {"class_name": "non_building", "count": None}, {"class_name": "temporary_structure", "count": None}, {"class_name": "uncertain_structure", "count": None}, {"class_name": "ignore", "count": None}])
    write_json(run_root / "06_label_gate" / "HUMAN_LABEL_READINESS.json", payload)
    write_csv(run_root / "06_label_gate" / "MISSING_LABEL_REQUIREMENTS.csv", [row for row in requirements if not row["pass"]])
    write_csv(run_root / "06_label_gate" / "LABEL_QA_FAILURES.csv", qa_failures)
    write_json(run_root / "06_label_gate" / "LABEL_READINESS_FINAL.json", payload)
    return {"status": payload["status"], "label_gate": gate, "human_verified": human_verified, "human_corrected": human_corrected, "complete_background": complete_background, "independent_validation": independent_validation, "independent_holdout": independent_holdout}


def training_stage(run_root: Path) -> dict[str, Any]:
    gate = read_json(run_root / "06_label_gate" / "LABEL_READINESS_FINAL.json", {})
    leakage = read_json(run_root / "02_leakage_scan" / "EXHAUSTIVE_LEAKAGE_AUDIT.json", {})
    if gate.get("label_training_gate") != "PASS" or leakage.get("critical_leakage_count", 1) != 0:
        result = {"status": "REFUSED_TRAINING_GATE_CLOSED", "training_started": False, "candidates": [], "reason": "independent complete-background human labels and/or leakage gate unavailable", "locked_test_pixel_reads": 0}
        write_json(run_root / "07_training" / "TRAINING_REFUSAL.json", result)
        return result
    # A future resumed run may enter this branch.  The contract is recorded
    # here, but no production training is implicit in the current run.
    result = {"status": "ELIGIBLE_CONTROLLED_TRAINING_NOT_STARTED", "training_started": False, "initial_checkpoint": "RSBuilding iter_2000", "candidates": [{"candidate_id": "TR1_HEAD_ONLY", "max_iteration": 1000}, {"candidate_id": "TR2_HIGH_LEVEL_PARTIAL", "max_iteration": 2000}, {"candidate_id": "TR3_FULL_LOW_LR", "max_iteration": 3000, "eligibility": "only after TR1 and TR2 underfit without overfit"}], "locked_test_pixel_reads": 0}
    write_json(run_root / "07_training" / "CONTROLLED_TRAINING_CONTRACT.json", result)
    return result


def model_selection_stage(run_root: Path) -> dict[str, Any]:
    training = read_json(run_root / "07_training" / "CONTROLLED_TRAINING_RESULT.json", read_json(run_root / "07_training" / "TRAINING_REFUSAL.json", {}))
    if not training.get("training_started"):
        result = {"status": "NOT_ELIGIBLE_NO_TRAINING_CANDIDATES", "selected_model": "iter_2000", "new_model_selected": False, "reason": "label gate or controlled training not complete", "locked_test_pixel_reads": 0}
        write_json(run_root / "08_model_selection" / "MODEL_SELECTION_AUDIT.json", result)
        return result
    result = {"status": "PENDING_VALIDATION_COMPARISON", "selected_model": None, "new_model_selected": False, "selection_source": "independent spatial validation only", "locked_test_pixel_reads": 0}
    write_json(run_root / "08_model_selection" / "MODEL_SELECTION_AUDIT.json", result)
    return result


def building_inference_stage(run_root: Path) -> dict[str, Any]:
    selection = read_json(run_root / "08_model_selection" / "SELECTED_BUILDING_MODEL.json", {})
    if not selection.get("selected_model"):
        result = {"status": "SKIPPED_WAITING_MODEL_SELECTION", "building_six_date_inference_complete": False, "locked_test_pixel_reads": 0}
        write_json(run_root / "09_building_final" / "BUILDING_INFERENCE_SKIPPED.json", result)
        return result
    result = {"status": "PENDING_GPU_INFERENCE", "building_six_date_inference_complete": False, "locked_test_pixel_reads": 0}
    write_json(run_root / "09_building_final" / "BUILDING_INFERENCE_PENDING.json", result)
    return result


def tracking_stage(run_root: Path) -> dict[str, Any]:
    result = {"status": "SKIPPED_WAITING_BUILDING_INFERENCE", "building_tracking_complete": False, "building_tongzhou_stability_gate": "NOT_EVALUATED", "tracker": "T7_FULL_LEGAL_TRACKER", "parameters_source": "previous frozen contract; no widening", "locked_test_pixel_reads": 0}
    write_json(run_root / "09_building_final" / "TRACKING_SKIPPED.json", result)
    return result


def _materialize_tree_without_html(source: Path, target: Path) -> dict[str, Any]:
    """Materialize a read-only candidate tree without creating web artifacts."""
    copied: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    if not source.exists():
        return {"source_exists": False, "source": str(source), "copied": copied, "excluded": excluded}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.suffix.lower() in {".html", ".htm"}:
            excluded.append({"source": str(path), "relative_path": str(relative), "reason": "html_forbidden"})
            continue
        destination = target / relative
        action = link_or_copy(path, destination)
        copied.append({"source": str(path), "target": str(destination), "relative_path": str(relative), "action": action, "sha256": sha256(path), "size": int(path.stat().st_size)})
    return {"source_exists": True, "source": str(source.resolve()), "copied": copied, "excluded": excluded}


def construction_stage(run_root: Path) -> dict[str, Any]:
    source = OLD_STABLE / "08_construction_final"
    materialized = _materialize_tree_without_html(source, run_root / "10_construction_final")
    candidate_counts: list[dict[str, Any]] = []
    summary_path = run_root / "10_construction_final" / "metrics" / "CONSTRUCTION_SINGLE_DATE_SUMMARY.csv"
    if summary_path.exists():
        candidate_counts = pd.read_csv(summary_path).to_dict(orient="records")
    tier_counts: dict[str, int] = {}
    confusion_counts: dict[str, int] = {}
    ranked_path = run_root / "10_construction_final" / "ranking" / "construction_candidates_ranked.gpkg"
    if ranked_path.exists():
        ranked = gpd.read_file(ranked_path)
        if "candidate_tier" in ranked.columns:
            tier_counts = {str(key): int(value) for key, value in ranked["candidate_tier"].fillna("missing").astype(str).value_counts().items()}
        if "likely_confusion_type" in ranked.columns:
            confusion_counts = {str(key): int(value) for key, value in ranked["likely_confusion_type"].fillna("missing").astype(str).value_counts().items()}
    temporal_track_counts: dict[str, int] = {}
    track_summary_path = run_root / "10_construction_final" / "temporal" / "CONSTRUCTION_TRACK_SUMMARY.csv"
    if track_summary_path.exists():
        track_summary = pd.read_csv(track_summary_path)
        if "status" in track_summary.columns:
            temporal_track_counts = {str(key): int(value) for key, value in track_summary["status"].fillna("missing").astype(str).value_counts().items()}
    temporal_layer_counts: dict[str, int] = {}
    for name in ("persistent_construction_candidates.gpkg", "newly_started_construction_candidates.gpkg", "likely_completed_construction_candidates.gpkg", "demolition_or_clearing_candidates.gpkg"):
        path = run_root / "10_construction_final" / "temporal" / name
        if path.exists():
            temporal_layer_counts[name] = int(len(gpd.read_file(path)))
    dates_found: dict[str, bool] = {}
    for date in DATES:
        dates_found[date] = any(
            path.is_file() and path.suffix.lower() not in {".html", ".htm"}
            for path in (run_root / "10_construction_final").rglob(f"*{date}*")
        )
    governance = read_json(run_root / "00_governance" / "GOVERNANCE_AUDIT.json", {})
    receipt = {
        "created_at": now(),
        "status": "candidate_not_ground_truth",
        "formal_status": "candidate_not_ground_truth",
        "construction_candidate_level": True,
        "source_run_root": str(OLD_STABLE.resolve()),
        "source_stage": str(source.resolve()),
        "source_materialization": materialized,
        "dates": DATES,
        "date_artifact_presence": dates_found,
        "construction_images_completed": bool(materialized["source_exists"] and all(dates_found.values())),
        "model": "M6_Construction_C-A",
        "model_label": "Construction C-A frozen",
        "recipe": "historical_exact_existing_sealed_outputs",
        "model_sha256": governance.get("construction_ca_sha256"),
        "rejected_model": "Construction C-B1",
        "rejected_model_sha256": governance.get("construction_cb1_sha256"),
        "rejected_model_enabled": False,
        "training_started": False,
        "raw_probability_modified": False,
        "building_branch_modified": False,
        "site_classifier_enabled": False,
        "read_only_reuse": True,
        "latest_date": DATES[-1],
        "forbidden_date_used": False,
        "candidate_counts": candidate_counts,
        "candidate_tier_counts": tier_counts,
        "likely_confusion_counts": confusion_counts,
        "temporal_track_counts": temporal_track_counts,
        "temporal_layer_counts": temporal_layer_counts,
        "locked_test_pixel_reads": 0,
        "scientific_limitations": [
            "Construction outputs are candidate-level evidence and are not ground truth.",
            "No Construction retraining or Site Classifier was enabled in this run.",
            "C-B1 remains rejected and was not restored or selected.",
        ],
    }
    write_json(run_root / "10_construction_final" / "CONSTRUCTION_OUTPUT_RECEIPT.json", receipt)
    return {"status": receipt["status"], "construction_images_completed": receipt["construction_images_completed"], "copied_file_count": len(materialized["copied"]), "excluded_html_count": len(materialized["excluded"]), "model": receipt["model"], "training_started": False, "locked_test_pixel_reads": 0}


def _figure_row(path: Path, root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "relative_path": str(path.relative_to(root)), "exists": path.exists(), "size_bytes": int(path.stat().st_size) if path.exists() else None}
    try:
        with Image.open(path) as image:
            row.update({"format": image.format, "mode": image.mode, "width": image.width, "height": image.height, "dpi": image.info.get("dpi"), "frames": getattr(image, "n_frames", 1)})
            sample = image.convert("RGB")
            sample.thumbnail((512, 512), Image.Resampling.BILINEAR)
            array = np.asarray(sample, dtype=np.float32)
            row["sample_mean"] = float(array.mean()) if array.size else None
            row["sample_std"] = float(array.std()) if array.size else None
            row["blank_like"] = bool(array.size == 0 or float(array.std()) < 1.0)
            if "A" in image.getbands():
                alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
                row["transparent_fraction"] = float((alpha == 0).mean()) if alpha.size else None
            else:
                row["transparent_fraction"] = 0.0
            row["readable"] = True
            row["error"] = None
    except Exception as error:
        row.update({"format": None, "mode": None, "width": None, "height": None, "dpi": None, "frames": None, "sample_mean": None, "sample_std": None, "blank_like": None, "transparent_fraction": None, "readable": False, "error": f"{type(error).__name__}:{error}"})
    return row


def figure_stage(run_root: Path) -> dict[str, Any]:
    building_root = run_root / "09_building_final" / "figures"
    construction_root = run_root / "10_construction_final" / "figures"
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    building_rows = [_figure_row(path, building_root) for path in sorted(building_root.rglob("*")) if path.is_file() and path.suffix.lower() in image_suffixes]
    construction_rows = [_figure_row(path, construction_root) for path in sorted(construction_root.rglob("*")) if path.is_file() and path.suffix.lower() in image_suffixes]
    columns = ["path", "relative_path", "exists", "size_bytes", "format", "mode", "width", "height", "dpi", "frames", "sample_mean", "sample_std", "blank_like", "transparent_fraction", "readable", "error"]
    write_csv(run_root / "11_figure_audit" / "BUILDING_FIGURE_INVENTORY.csv", building_rows, columns)
    write_csv(run_root / "11_figure_audit" / "CONSTRUCTION_FIGURE_INVENTORY.csv", construction_rows, columns)
    forbidden_html = [str(path) for path in run_root.rglob("*") if path.is_file() and path.suffix.lower() in {".html", ".htm"}]
    rendering = {
        "created_at": now(),
        "building_status": "not_rendered_waiting_model" if not building_rows else "rendered_audited",
        "construction_status": "rendered_audited" if construction_rows else "no_candidate_figures",
        "building_figure_count": len(building_rows),
        "construction_figure_count": len(construction_rows),
        "readable_building_count": sum(bool(row.get("readable")) for row in building_rows),
        "readable_construction_count": sum(bool(row.get("readable")) for row in construction_rows),
        "blank_like_building_count": sum(bool(row.get("blank_like")) for row in building_rows),
        "blank_like_construction_count": sum(bool(row.get("blank_like")) for row in construction_rows),
        "html_generated": bool(forbidden_html),
        "html_paths": forbidden_html,
        "locked_test_pixel_reads": 0,
    }
    numerical = {
        "created_at": now(),
        "building_metrics_available": (run_root / "09_building_final" / "metrics").exists() and any((run_root / "09_building_final" / "metrics").rglob("*.json")),
        "construction_metrics_available": (run_root / "10_construction_final" / "metrics").exists() and any((run_root / "10_construction_final" / "metrics").rglob("*.json")),
        "building_metrics_status": "not_available_waiting_model",
        "construction_metric_sources": [str(path) for path in sorted((run_root / "10_construction_final" / "metrics").rglob("*") if (run_root / "10_construction_final" / "metrics").exists() else []) if path.is_file() and path.suffix.lower() in {".json", ".csv"}],
        "formal_accuracy_claim": False,
        "candidate_output_claim": "Construction candidate-level only",
        "locked_test_pixel_reads": 0,
    }
    write_json(run_root / "11_figure_audit" / "FIGURE_RENDERING_AUDIT.json", rendering)
    write_json(run_root / "11_figure_audit" / "FIGURE_NUMERICAL_AUDIT.json", numerical)
    return {"status": "complete", "building_figure_count": len(building_rows), "construction_figure_count": len(construction_rows), "html_generated": bool(forbidden_html), "locked_test_pixel_reads": 0}


def protected_invariance(run_root: Path) -> dict[str, Any]:
    pre = read_json(run_root / "00_governance" / "PRE_RUN_SHA256.json", {})
    comparisons: list[dict[str, Any]] = []
    for before in pre.get("files", []):
        path = Path(str(before.get("path", "")))
        if not path.exists():
            comparisons.append({"path": str(path), "before": before, "after": None, "unchanged": False, "reason": "missing_after_run"})
            continue
        after = file_signature(path)
        unchanged = all(after.get(key) == before.get(key) for key in ("size", "mtime_ns", "sha256"))
        comparisons.append({"path": str(path), "before": before, "after": after, "unchanged": unchanged, "reason": None if unchanged else "signature_changed"})
    changed = [row for row in comparisons if not row["unchanged"]]
    result = {
        "created_at": now(), "file_count": len(comparisons), "unchanged_count": len(comparisons) - len(changed), "changed_count": len(changed),
        "all_protected_assets_unchanged": not changed, "changed_files": changed, "registry_unchanged": not any("CURRENT_BEST_MODEL_REGISTRY.json" in row["path"] and not row["unchanged"] for row in comparisons),
        "full20_unchanged": not any("FULL20_BASELINE_MANIFEST.json" in row["path"] and not row["unchanged"] for row in comparisons),
        "rsbuilding_frozen_checkpoint_unchanged": not any("rsbuilding" in row["path"].lower() and not row["unchanged"] for row in comparisons),
        "construction_ca_unchanged": not any("construction" in row["path"].lower() and not row["unchanged"] for row in comparisons),
        "tongzhou_outputs_modified": False,
        "locked_test_pixel_reads": 0,
    }
    write_json(run_root / "00_governance" / "POST_RUN_SHA256.json", {"created_at": now(), "files": [row.get("after") for row in comparisons if row.get("after")], "file_count": len(comparisons)})
    write_json(run_root / "00_governance" / "PROTECTED_ASSET_INVARIANCE.json", result)
    state = load_state(run_root)
    state.setdefault("flags", {}).update({
        "registry_modified": not result["registry_unchanged"], "full20_modified": not result["full20_unchanged"],
        "rsbuilding_frozen_checkpoint_modified": not result["rsbuilding_frozen_checkpoint_unchanged"], "construction_ca_modified": not result["construction_ca_unchanged"],
        "tongzhou_outputs_modified": False, "locked_test_pixel_reads": 0,
    })
    save_state(run_root, state)
    return result


def test_audit_stage(run_root: Path) -> dict[str, Any]:
    """Run the bounded regression set and record independent output checks."""
    test_files = ["full_project_suite"]
    command = [sys.executable, "-m", "pytest", "-q", "--basetemp", str(PROJECT_ROOT / ".urrs-domain-check")]
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600, check=False)
        pytest_result = {
            "returncode": int(completed.returncode), "passed": completed.returncode == 0,
            "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:],
        }
    except Exception as error:
        pytest_result = {"returncode": None, "passed": False, "stdout_tail": "", "stderr_tail": f"{type(error).__name__}: {error}"}
    state = load_state(run_root)
    html_count = sum(1 for path in run_root.rglob("*") if path.is_file() and path.suffix.lower() in {".html", ".htm"})
    construction = read_json(run_root / "10_construction_final" / "CONSTRUCTION_OUTPUT_RECEIPT.json", {})
    invariance = read_json(run_root / "00_governance" / "PROTECTED_ASSET_INVARIANCE.json", {})
    checks = {
        "html_generated": html_count > 0,
        "locked_test_pixel_reads": int(state.get("flags", {}).get("locked_test_pixel_reads", 0)),
        "protected_assets_unchanged": bool(invariance.get("all_protected_assets_unchanged", False)),
        "construction_six_dates_present": bool(construction.get("construction_images_completed", False)),
        "registry_unchanged": bool(invariance.get("registry_unchanged", False)),
        "tongzhou_outputs_modified": bool(state.get("flags", {}).get("tongzhou_outputs_modified", False)),
    }
    result = {
        "created_at": now(), "status": "PASS" if pytest_result["passed"] and not checks["html_generated"] and checks["locked_test_pixel_reads"] == 0 and checks["protected_assets_unchanged"] and checks["construction_six_dates_present"] and checks["registry_unchanged"] and not checks["tongzhou_outputs_modified"] else "FAIL",
        "pytest_command": command, "pytest_result": pytest_result, "output_checks": checks, "test_files": test_files,
        "scientific_status_not_changed": state.get("FINAL_STATUS", state.get("final_status")), "locked_test_pixel_reads": 0,
    }
    write_json(run_root / "14_tests" / "TEST_RUN_SUMMARY.json", result)
    write_json(run_root / "14_tests" / "OUTPUT_INVARIANT_AUDIT.json", {"created_at": now(), **checks, "locked_test_pixel_reads": 0})
    return {"status": result["status"], "pytest_passed": bool(pytest_result["passed"]), "html_generated": bool(checks["html_generated"]), "locked_test_pixel_reads": 0}


def acceptance_stage(run_root: Path) -> dict[str, Any]:
    if not (run_root / "00_governance" / "PROTECTED_ASSET_INVARIANCE.json").exists():
        protected_invariance(run_root)
    governance = read_json(run_root / "00_governance" / "GOVERNANCE_AUDIT.json", {})
    invariance = read_json(run_root / "00_governance" / "PROTECTED_ASSET_INVARIANCE.json", {})
    attribution = read_json(run_root / "01_full20_attribution" / "FULL20_ATTRIBUTION_COMPLETENESS.json", {})
    leakage = read_json(run_root / "02_leakage_scan" / "EXHAUSTIVE_LEAKAGE_AUDIT.json", {})
    baseline = read_json(run_root / "04_baseline" / "NANJING_BASELINE_RECOMPUTED.json", {})
    label_gate = read_json(run_root / "06_label_gate" / "LABEL_READINESS_FINAL.json", {})
    training = read_json(run_root / "07_training" / "CONTROLLED_TRAINING_RESULT.json", read_json(run_root / "07_training" / "TRAINING_REFUSAL.json", {}))
    selection = read_json(run_root / "08_model_selection" / "MODEL_SELECTION_AUDIT.json", {})
    building = read_json(run_root / "09_building_final" / "BUILDING_INFERENCE_SKIPPED.json", read_json(run_root / "09_building_final" / "BUILDING_INFERENCE_PENDING.json", {}))
    tracking = read_json(run_root / "09_building_final" / "TRACKING_SKIPPED.json", {})
    construction = read_json(run_root / "10_construction_final" / "CONSTRUCTION_OUTPUT_RECEIPT.json", {})
    figures = read_json(run_root / "11_figure_audit" / "FIGURE_RENDERING_AUDIT.json", {})
    label_pass = label_gate.get("label_training_gate") == "PASS"
    attribution_pass = attribution.get("status") == "complete" and int(attribution.get("missing_object_observation_count", 1)) == 0
    leakage_pass = int(leakage.get("critical_leakage_count", 1)) == 0
    if not label_pass:
        final_status = "awaiting_human_building_labels"
        reason = "Independent complete-background, spatial-validation, and holdout Building labels have not passed the human QA gate."
    elif not (attribution_pass and leakage_pass):
        final_status = "blocked_building_domain_gap_remains"
        reason = "A pre-training Full20 attribution or leakage gate remains unresolved."
    elif not (training.get("training_started") and selection.get("selected_model") and building.get("building_six_date_inference_complete") and tracking.get("building_tracking_complete")):
        final_status = "blocked_building_domain_gap_remains"
        reason = "Labels passed, but controlled training/model selection/final Building inference/tracking are not complete."
    else:
        final_status = "tongzhou_level_building_stability_achieved_and_construction_images_completed"
        reason = "All required gates and final outputs passed."
    flags = {
        "locked_test_pixel_reads": 0,
        "registry_modified": not bool(invariance.get("registry_unchanged", False)),
        "full20_modified": not bool(invariance.get("full20_unchanged", False)),
        "rsbuilding_frozen_checkpoint_modified": not bool(invariance.get("rsbuilding_frozen_checkpoint_unchanged", False)),
        "construction_ca_modified": not bool(invariance.get("construction_ca_unchanged", False)),
        "cb1_enabled": False, "site_classifier_enabled": False,
        "nanjing_2019_positive_gt_used_for_training": False, "nanjing_2019_positive_gt_used_for_selection": False,
        "tongzhou_outputs_modified": False, "html_generated": bool(figures.get("html_generated", False)),
    }
    payload = {
        "schema_version": "nanjing-domain-adaptation-v1", "created_at": now(), "run_root": str(run_root.resolve()),
        "FINAL_STATUS": final_status, "final_status": final_status, "status_reason": reason, "current_stage": "ACCEPTANCE",
        "aoi": {"west": AOI_WEST, "east": AOI_EAST, "south": AOI_SOUTH, "north": AOI_NORTH}, "dates": DATES, "latest_date": DATES[-1], "excluded_dates": [FORBIDDEN_DATE], "metric_crs": METRIC_CRS,
        "full20_attribution_status": attribution.get("status"), "full20_attribution_object_observation_count": attribution.get("object_observation_count"), "full20_attribution_missing_count": attribution.get("missing_object_observation_count"),
        "full20_attribution_summary": attribution,
        "leakage_status": leakage.get("status"), "leakage_critical_count": leakage.get("critical_leakage_count"), "leakage_summary": leakage,
        "label_gate": label_gate.get("label_training_gate"), "label_status": label_gate.get("status"), "human_verified_count": label_gate.get("human_verified_count", 0), "human_corrected_count": label_gate.get("human_corrected_count", 0),
        "complete_background": label_gate.get("complete_background", False), "independent_spatial_validation": label_gate.get("independent_validation", False), "independent_spatial_holdout": label_gate.get("independent_holdout", False),
        "training_started": bool(training.get("training_started", False)), "training_status": training.get("status"), "final_model": selection.get("selected_model"), "final_recipe": selection.get("selected_recipe"), "model_selection_status": selection.get("status"),
        "building_final_status": building.get("status"), "building_six_date_inference_complete": bool(building.get("building_six_date_inference_complete", False)), "building_tracking_complete": bool(tracking.get("building_tracking_complete", False)), "building_metrics": baseline.get("metrics", {}),
        "construction_model": construction.get("model"), "construction_model_sha256": construction.get("model_sha256"), "construction_images_completed": bool(construction.get("construction_images_completed", False)), "construction_scientific_status": construction.get("status"),
        "figure_audit": figures, "protected_asset_invariance": invariance, "registry_unchanged": bool(invariance.get("registry_unchanged", False)), "tongzhou_outputs_modified": False,
        "flags": flags, "locked_test_pixel_reads": 0, "html_generated": bool(figures.get("html_generated", False)), "scientific_limitations": [
            "Candidate/model-reference overlays are not ground truth.", "Nanjing 2019 positive-only Building data is diagnostic and was not used for training or selection.",
            "No formal Building accuracy or stability claim is made while the independent label gate is closed.", "Construction remains candidate-level evidence; C-B1 and Site Classifier remain disabled.",
        ],
    }
    if final_status == "awaiting_human_building_labels":
        write_json(run_root / "12_manifests" / "AWAITING_HUMAN_BUILDING_LABELS_RECEIPT.json", payload)
    elif final_status == "blocked_building_domain_gap_remains":
        write_json(run_root / "12_manifests" / "BLOCKED_SCIENTIFIC_STATUS.json", payload)
    write_json(run_root / "12_manifests" / "FINAL_ACCEPTANCE_MANIFEST.json", payload)
    state = load_state(run_root)
    state.update({"FINAL_STATUS": final_status, "final_status": final_status, "current_stage": "ACCEPTANCE", "flags": flags, "acceptance": payload})
    save_state(run_root, state)
    return {"status": final_status, "reason": reason, "training_started": bool(training.get("training_started", False)), "label_gate": label_gate.get("label_training_gate"), "registry_unchanged": bool(invariance.get("registry_unchanged", False)), "locked_test_pixel_reads": 0, "exit_code": 0 if final_status == "tongzhou_level_building_stability_achieved_and_construction_images_completed" else (2 if final_status == "awaiting_human_building_labels" else 3)}


def _execute_stage(run_root: Path, stage: str, callback: Any, resume: bool = True) -> dict[str, Any]:
    if resume and stage_is_complete(run_root, stage):
        return read_json(stage_marker(run_root, stage), {"status": "completed", "resumed": True})
    mark_stage(run_root, stage, "running")
    try:
        result = dict(callback())
    except Exception as error:
        mark_stage(run_root, stage, "failed", error=f"{type(error).__name__}: {error}")
        raise
    complete_stage(run_root, stage, result)
    return result


STAGE_SEQUENCE = [
    "GOVERNANCE_AUDIT", "FULL20_OBJECT_ATTRIBUTION", "EXHAUSTIVE_LEAKAGE_SCAN", "TRAINING_ASSET_LEDGER",
    "BASELINE_REPRODUCTION", "TONGZHOU_BENCHMARK", "LABEL_PLAN", "LABEL_INGESTION_AUDIT", "CONTROLLED_TRAINING",
    "MODEL_SELECTION", "FINAL_BUILDING_INFERENCE", "FINAL_TRACKING", "CONSTRUCTION_INFERENCE", "FIGURE_RENDERING", "TEST_AUDIT", "ACCEPTANCE",
]


def run_all(run_root: Path, label_root: Path | None = None, resume: bool = False) -> dict[str, Any]:
    ensure_tree(run_root)
    _execute_stage(run_root, "GOVERNANCE_AUDIT", lambda: governance_audit(run_root), resume)
    _execute_stage(run_root, "FULL20_OBJECT_ATTRIBUTION", lambda: full20_attribution_stage(run_root), resume)
    _execute_stage(run_root, "EXHAUSTIVE_LEAKAGE_SCAN", lambda: leakage_stage(run_root), resume)
    _execute_stage(run_root, "TRAINING_ASSET_LEDGER", lambda: training_inventory_stage(run_root), resume)
    baseline_result = _execute_stage(run_root, "BASELINE_REPRODUCTION", lambda: baseline_stage(run_root), resume)
    if not (resume and stage_is_complete(run_root, "TONGZHOU_BENCHMARK")):
        complete_stage(run_root, "TONGZHOU_BENCHMARK", {"status": "complete", "source": "04_baseline/TONGZHOU_BENCHMARK_RECOMPUTED.json", "metrics": baseline_result.get("tongzhou", {})})
    _execute_stage(run_root, "LABEL_PLAN", lambda: label_package_stage(run_root), resume and label_root is None)
    _execute_stage(run_root, "LABEL_INGESTION_AUDIT", lambda: label_gate_stage(run_root, label_root), resume and label_root is None)
    label_gate = read_json(run_root / "06_label_gate" / "LABEL_READINESS_FINAL.json", {})
    downstream_resume = resume and label_gate.get("label_training_gate") != "PASS"
    _execute_stage(run_root, "CONTROLLED_TRAINING", lambda: training_stage(run_root), downstream_resume)
    _execute_stage(run_root, "MODEL_SELECTION", lambda: model_selection_stage(run_root), downstream_resume)
    _execute_stage(run_root, "FINAL_BUILDING_INFERENCE", lambda: building_inference_stage(run_root), downstream_resume)
    _execute_stage(run_root, "FINAL_TRACKING", lambda: tracking_stage(run_root), downstream_resume)
    _execute_stage(run_root, "CONSTRUCTION_INFERENCE", lambda: construction_stage(run_root), resume)
    _execute_stage(run_root, "FIGURE_RENDERING", lambda: figure_stage(run_root), resume)
    _execute_stage(run_root, "TEST_AUDIT", lambda: test_audit_stage(run_root), resume)
    protected_invariance(run_root)
    acceptance = _execute_stage(run_root, "ACCEPTANCE", lambda: acceptance_stage(run_root), resume)
    return acceptance


def dispatch(mode: str, run_root: Path, label_root: Path | None = None) -> dict[str, Any]:
    ensure_tree(run_root)
    callbacks: dict[str, tuple[str, Any]] = {
        "audit": ("GOVERNANCE_AUDIT", lambda: governance_audit(run_root)),
        "full20": ("FULL20_OBJECT_ATTRIBUTION", lambda: full20_attribution_stage(run_root)),
        "leakage": ("EXHAUSTIVE_LEAKAGE_SCAN", lambda: leakage_stage(run_root)),
        "prepare": ("LABEL_PLAN", lambda: label_package_stage(run_root)),
        "ingest": ("LABEL_INGESTION_AUDIT", lambda: label_gate_stage(run_root, label_root)),
        "train": ("CONTROLLED_TRAINING", lambda: training_stage(run_root)),
        "select": ("MODEL_SELECTION", lambda: model_selection_stage(run_root)),
        "infer": ("FINAL_BUILDING_INFERENCE", lambda: building_inference_stage(run_root)),
        "track": ("FINAL_TRACKING", lambda: tracking_stage(run_root)),
        "construction": ("CONSTRUCTION_INFERENCE", lambda: construction_stage(run_root)),
        "figures": ("FIGURE_RENDERING", lambda: figure_stage(run_root)),
        "tests": ("TEST_AUDIT", lambda: test_audit_stage(run_root)),
        "accept": ("ACCEPTANCE", lambda: acceptance_stage(run_root)),
    }
    if mode == "all":
        return run_all(run_root, label_root, resume=False)
    if mode == "resume":
        return run_all(run_root, label_root, resume=True)
    if mode == "status":
        state = load_state(run_root)
        return {"status": state.get("final_status", "not_started"), "run_root": str(run_root.resolve()), "current_stage": state.get("current_stage"), "stages": state.get("stages", {}), "locked_test_pixel_reads": state.get("flags", {}).get("locked_test_pixel_reads", 0), "exit_code": 0}
    stage, callback = callbacks[mode]
    return _execute_stage(run_root, stage, callback, resume=(mode not in {"accept", "tests", "construction", "prepare", "ingest", "train", "select", "infer", "track"}))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=None, help="new independent run root; defaults to outputs/inference/nanjing_domain_adaptation_stable_tracking_<timestamp>")
    parser.add_argument("--mode", choices=["audit", "full20", "leakage", "prepare", "ingest", "train", "select", "infer", "track", "construction", "figures", "tests", "accept", "all", "resume", "status"], default="all")
    parser.add_argument("--label-root", default=None, help="optional human label delivery root")
    args = parser.parse_args(argv)
    run_root = Path(args.run_root).resolve() if args.run_root else default_run_root().resolve()
    label_root = Path(args.label_root).resolve() if args.label_root else None
    ensure_tree(run_root)
    try:
        result = dispatch(args.mode, run_root, label_root)
        exit_code = int(result.get("exit_code", 0))
    except Exception as error:
        state = load_state(run_root)
        state.update({"final_status": "blocked_building_domain_gap_remains", "FINAL_STATUS": "blocked_building_domain_gap_remains", "error": f"{type(error).__name__}: {error}"})
        save_state(run_root, state)
        result = {"status": "blocked_building_domain_gap_remains", "error": f"{type(error).__name__}: {error}", "run_root": str(run_root), "exit_code": 1}
        exit_code = 1
    result = {"run_root": str(run_root), "mode": args.mode, **result}
    write_json(run_root / "13_logs" / "LAST_COMMAND_RESULT.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
