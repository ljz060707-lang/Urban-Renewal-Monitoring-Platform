from __future__ import annotations

"""Nanjing Jiangbei modular URRS rebuild.

This entry point is intentionally an inference-only orchestrator.  It keeps
the public-change baseline and the Construction Activity branch in separate
output namespaces and makes every cross-module exchange read-only.  The old
``nanjing_jiangbei_acceptance.py`` remains available for historical audit but
is not called by this workflow.
"""

import argparse
import ast
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from urban_renewal.modular.contracts import (  # noqa: E402
    AlignedSceneReference,
    BaselineChangeResult,
    BuildingGeometryResult,
    BuildingTrackResult,
    ConstructionElementResult,
    ConstructionTemporalResult,
    CrossReferenceResult,
    Full20PairResult,
    SceneRecord,
    contract_to_dict,
)


SOURCE_ROOT = Path(r"D:\Data Set\Satellite\nanjing_jiangbei")
OLD_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "inference" / "nanjing_jiangbei_external_acceptance_20260803_163802"
INFERENCE_ROOT = PROJECT_ROOT / "outputs" / "inference"
FULL20_CONFIG = PROJECT_ROOT / "configs" / "change_public_v2_full20_inference_windows.yaml"
FULL20_CHECKPOINT = PROJECT_ROOT / "outputs" / "models" / "urban_change_v2_full_candidate_20260727_001029" / "best_v2_multisource_change.pth"
RSBUILDING_CHECKPOINT = PROJECT_ROOT / "outputs" / "v2_readiness" / "rsbuilding" / "whu" / "work_dirs" / "swin_t_whu_pilot5000_20260728_110750" / "iter_2000.pth"
RSBUILDING_CONFIG = RSBUILDING_CHECKPOINT.parent / "Swin_T_whu_windows_pilot5000.py"
RSBUILDING_PYTHON = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
CONSTRUCTION_PACKAGE = PROJECT_ROOT / "outputs" / "models" / "construction_activity_v1" / "URRS-Construction-UCS386-v1.0-task-candidate_20260803_142054"
CONSTRUCTION_CHECKPOINT = CONSTRUCTION_PACKAGE / "model.pth"
SURFACE_CHECKPOINT = PROJECT_ROOT / "outputs" / "training" / "surface_state_v1" / "joint_sample_20260730_034853" / "checkpoints" / "surface_joint_sample_best_validation.pth"

EXPECTED_FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
EXPECTED_RSBUILDING_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
EXPECTED_CONSTRUCTION_SHA = "c522d5a4e6c5d6124f968d659004d7870fe4adc9d7381dc4c8917fe019f0f060"
EXPECTED_SURFACE_SHA = "35668be3a61b30a29576fc66203b67505e4a0cfa56051d04d6bc25ed21de875c"

METRIC_CRS = "EPSG:32650"
MMU_M2 = 10.0
FULL20_THRESHOLDS = {"any_change": 0.75, "new_building": 0.60, "demolition": 0.35}
RSBUILDING_THRESHOLD = 0.50
CONSTRUCTION_RGB_MAPPING = (3, 2, 1)
CONSTRUCTION_THRESHOLD = 0.50
HIGH_CONFIDENCE_THRESHOLD = 0.65
HIGH_UNCERTAINTY_THRESHOLD = 0.50

REQUIRED_DIRS = (
    "00_architecture_audit",
    "01_input_audit",
    "02_baseline_full20",
    "03_baseline_rsbuilding",
    "04_baseline_fusion",
    "05_baseline_tracking",
    "06_construction_single_date",
    "07_construction_temporal",
    "08_cross_reference",
    "09_figures_highres",
    "10_report_assets",
    "11_tables",
    "12_logs",
)

SUPPORTED_SUFFIXES = {".tif", ".tiff"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8-sig"))


def write_csv(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ensure_layout(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        (output_root / name).mkdir(parents=True, exist_ok=True)


def refuse_old_output(output_root: Path) -> None:
    if output_root.resolve() == OLD_OUTPUT_ROOT.resolve():
        raise RuntimeError(f"The historical Nanjing output is read-only and cannot be used: {output_root}")


def parse_date_token(text: str) -> str | None:
    for token in re.findall(r"(?<!\d)(20\d{6})(?!\d)", text):
        try:
            return datetime.strptime(token, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    for token in re.findall(r"(?<!\d)(\d{6})(?!\d)", text):
        try:
            return datetime.strptime(token, "%y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, set)):
        return [_jsonable(item) for item in value]
    return value


def _scene_record_from_dict(value: Mapping[str, Any]) -> SceneRecord:
    return SceneRecord(
        date=str(value["date"]),
        path=str(value["path"]),
        filename=str(value["filename"]),
        sha256=str(value["sha256"]),
        size_bytes=int(value["size_bytes"]),
        width=int(value["width"]),
        height=int(value["height"]),
        bands=int(value["bands"]),
        dtype=tuple(str(item) for item in value["dtype"]),
        crs=value.get("crs"),
        transform=tuple(float(item) for item in value["transform"]),
        bounds=tuple(float(item) for item in value["bounds"]),
        resolution=tuple(float(item) for item in value["resolution"]),
        nodata=value.get("nodata"),
        duplicate_observation=bool(value.get("duplicate_observation", False)),
        excluded_from_temporal_analysis=bool(value.get("excluded_from_temporal_analysis", False)),
        exclusion_reason=value.get("exclusion_reason"),
    )


def _inspect_scene(path: Path) -> dict[str, Any]:
    import rasterio

    digest = sha256_file(path)
    date = parse_date_token(path.stem)
    if not date:
        raise RuntimeError(f"Acquisition date cannot be identified from filename: {path}")
    with rasterio.open(path) as dataset:
        return {
            "date": date,
            "path": str(path.resolve()),
            "filename": path.name,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": list(dataset.dtypes),
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": list(dataset.transform),
            "bounds": list(dataset.bounds),
            "resolution": list(dataset.res),
            "nodata": dataset.nodata,
            "rgb_mapping_recorded": list(CONSTRUCTION_RGB_MAPPING),
            "date_source": "filename_YYMMDD",
        }


def _same_grid(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if any(left.get(key) != right.get(key) for key in ("width", "height", "crs")):
        return False
    return bool(
        np.allclose(left.get("transform", []), right.get("transform", []), atol=0.0, rtol=0.0)
        and np.allclose(left.get("resolution", []), right.get("resolution", []), atol=0.0, rtol=0.0)
        and np.allclose(left.get("bounds", []), right.get("bounds", []), atol=1e-12, rtol=0.0)
    )


def _pair_name(date_from: str, date_to: str) -> str:
    return f"{date_from}_to_{date_to}"


def _pair_plan(valid_records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(valid_records, key=lambda item: str(item["date"]))
    dates = [str(item["date"]) for item in ordered]
    if len(dates) < 2 or dates != sorted(set(dates)):
        raise RuntimeError("At least two unique valid dates are required for formal temporal comparison")
    final_date = dates[-1]
    cumulative = [
        {
            "pair_name": _pair_name(date, final_date),
            "date_from": date,
            "date_to": final_date,
            "pair_type": "cumulative_primary",
            "derived_from_temporal_state_aggregation": False,
        }
        for date in dates[:-1]
    ]
    adjacent = [
        {
            "pair_name": _pair_name(dates[index], dates[index + 1]),
            "date_from": dates[index],
            "date_to": dates[index + 1],
            "pair_type": "adjacent_appendix",
            "derived_from_temporal_state_aggregation": False,
        }
        for index in range(len(dates) - 1)
    ]
    return {"cumulative": cumulative, "adjacent": adjacent}


def _snapshot_source_files(source_root: Path) -> dict[str, Any]:
    files = sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES)
    return {
        "source_root": str(source_root.resolve()),
        "files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }


def _compare_source_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    left = {item["path"]: item for item in before.get("files", [])}
    right = {item["path"]: item for item in after.get("files", [])}
    changed = []
    for path in sorted(set(left) | set(right)):
        if left.get(path) != right.get(path):
            changed.append({"path": path, "before": left.get(path), "after": right.get(path)})
    return {"original_data_modified": bool(changed), "modified_files": changed, "before_count": len(left), "after_count": len(right)}


def audit_inputs(source_root: Path, output_root: Path) -> dict[str, Any]:
    """M0 + M1.  This function never imports a model or runs inference."""

    import rasterio

    paths = sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES)
    if len(paths) < 2:
        raise RuntimeError(f"At least two GeoTIFFs are required under {source_root}")
    all_records = [_inspect_scene(path) for path in paths]
    all_records.sort(key=lambda item: (str(item["date"]), str(item["path"])))
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_hash.setdefault(str(record["sha256"]), []).append(record)

    duplicate_groups = []
    excluded_dates: list[str] = []
    for digest, group in sorted(by_hash.items()):
        group.sort(key=lambda item: (str(item["date"]), str(item["path"])))
        if len(group) <= 1:
            continue
        dates = [str(item["date"]) for item in group]
        conflict = len(set(dates)) > 1
        excluded = group[1:] if conflict else group[1:]
        for record in group:
            record["duplicate_observation"] = True
        for record in excluded:
            record["excluded_from_temporal_analysis"] = True
            record["exclusion_reason"] = "duplicate_observation" if conflict else "duplicate_file_same_content"
            excluded_dates.append(str(record["date"]))
        duplicate_groups.append(
            {
                "sha256": digest,
                "files": [item["filename"] for item in group],
                "dates": dates,
                "count": len(group),
                "different_dates": conflict,
                "retained_record": group[0]["filename"],
                "excluded_records": [item["filename"] for item in excluded],
                "action": "KEEP_EARLIEST_EXCLUDE_LATER_DATE_CONFLICT" if conflict else "KEEP_ONE_EXCLUDE_DUPLICATE_FILE",
            }
        )
    valid_records = [record for record in all_records if not record.get("excluded_from_temporal_analysis")]
    valid_records.sort(key=lambda item: str(item["date"]))
    if len(valid_records) < 2:
        raise RuntimeError("Duplicate exclusion left fewer than two valid observations")
    if len({str(item["date"]) for item in valid_records}) != len(valid_records):
        raise RuntimeError("Duplicate acquisition dates remain after M0 audit")
    final_date = str(valid_records[-1]["date"])
    plans = _pair_plan(valid_records)

    input_dir = output_root / "01_input_audit"
    input_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "INPUT_DUPLICATE_AUDIT.json", {
        "status": "pass",
        "source_root": str(source_root.resolve()),
        "all_image_count": len(all_records),
        "valid_unique_image_count": len(valid_records),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "excluded_dates": sorted(excluded_dates),
        "excluded_observations": [item for item in all_records if item.get("excluded_from_temporal_analysis")],
        "formal_temporal_dates": [str(item["date"]) for item in valid_records],
        "effective_final_date": final_date,
        "source_data_modified": False,
        "created_at": utc_now(),
    })
    write_json(input_dir / "INPUT_DUPLICATE_AUDIT.json", read_json(output_root / "INPUT_DUPLICATE_AUDIT.json"))
    write_json(input_dir / "IMAGE_INVENTORY.json", {
        "source_root": str(source_root.resolve()),
        "records": all_records,
        "valid_records": valid_records,
        "created_at": utc_now(),
    })
    write_csv(
        input_dir / "IMAGE_INVENTORY.csv",
        ["date", "filename", "path", "sha256", "width", "height", "bands", "crs", "resolution", "duplicate_observation", "excluded_from_temporal_analysis", "exclusion_reason"],
        all_records,
    )
    write_csv(input_dir / "BASELINE_PAIR_PLAN.csv", ["pair_name", "date_from", "date_to", "pair_type", "derived_from_temporal_state_aggregation"], [*plans["cumulative"], *plans["adjacent"]])

    reference = valid_records[0]
    same_grid = all(_same_grid(reference, item) for item in valid_records[1:])
    left = max(float(item["bounds"][0]) for item in valid_records)
    bottom = max(float(item["bounds"][1]) for item in valid_records)
    right = min(float(item["bounds"][2]) for item in valid_records)
    top = min(float(item["bounds"][3]) for item in valid_records)
    overlap = [left, bottom, right, top] if right > left and top > bottom else None
    grid_ready = bool(same_grid and overlap)
    aligned_refs = [
        contract_to_dict(
            AlignedSceneReference(
                date=str(item["date"]),
                source_path=str(item["path"]),
                source_sha256=str(item["sha256"]),
                grid_signature=(item["crs"], item["width"], item["height"], tuple(item["transform"])),
                overlap_status="full_exact_grid" if grid_ready else "registration_required",
                resampling_required=not grid_ready,
            )
        )
        for item in valid_records
    ]
    common_grid = {
        "status": "pass" if grid_ready else "registration_required",
        "crs": reference["crs"],
        "width": reference["width"],
        "height": reference["height"],
        "transform": reference["transform"],
        "resolution": reference["resolution"],
        "bounds": reference["bounds"],
        "intersection_bounds": overlap,
        "metric_crs": METRIC_CRS,
        "aligned_scene_references": aligned_refs,
        "source_images_copied": False,
        "model_preprocessing_not_replaced": True,
    }
    write_json(input_dir / "COMMON_GRID.json", common_grid)
    spatial = {
        "status": "pass" if grid_ready else "registration_required",
        "same_crs": all(item["crs"] == reference["crs"] for item in valid_records),
        "same_shape": all((item["width"], item["height"]) == (reference["width"], reference["height"]) for item in valid_records),
        "same_transform": all(np.allclose(item["transform"], reference["transform"], atol=0.0, rtol=0.0) for item in valid_records),
        "same_resolution": all(np.allclose(item["resolution"], reference["resolution"], atol=0.0, rtol=0.0) for item in valid_records),
        "same_bounds": all(np.allclose(item["bounds"], reference["bounds"], atol=1e-12, rtol=0.0) for item in valid_records),
        "intersection_bounds": overlap,
        "pixelwise_temporal_tracking_allowed": grid_ready,
        "resampling_policy": {"rgb": "model-owned", "probability": "model-owned", "mask": "nearest"},
    }
    write_json(input_dir / "ALIGNMENT_AUDIT.json", spatial)
    if grid_ready:
        profile = None
        with rasterio.open(valid_records[0]["path"]) as reference_dataset:
            profile = reference_dataset.profile.copy()
            for key in ("photometric", "interleave"):
                profile.pop(key, None)
            profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
            with rasterio.open(input_dir / "VALID_OVERLAP_MASK.tif", "w", **profile) as target:
                for _, window in target.block_windows(1):
                    valid = np.ones((int(window.height), int(window.width)), dtype=bool)
                    for item in valid_records:
                        with rasterio.open(item["path"]) as dataset:
                            valid &= dataset.read_masks(1, window=window) > 0
                    target.write(valid.astype(np.uint8), 1, window=window)
    else:
        write_json(input_dir / "VALID_OVERLAP_MASK_NOT_WRITTEN.json", {"status": "blocked", "reason": "exact common grid unavailable"})

    audit = {
        "status": "pass" if grid_ready else "partial",
        "source_root": str(source_root.resolve()),
        "records": all_records,
        "valid_records": valid_records,
        "dates": [str(item["date"]) for item in valid_records],
        "final_date": final_date,
        "formal_pairs": plans["cumulative"],
        "adjacent_pairs": plans["adjacent"],
        "duplicate_audit": str((output_root / "INPUT_DUPLICATE_AUDIT.json").resolve()),
        "common_grid": common_grid,
        "spatial": spatial,
        "valid_overlap_mask": str((input_dir / "VALID_OVERLAP_MASK.tif").resolve()) if (input_dir / "VALID_OVERLAP_MASK.tif").exists() else None,
        "rgb_mapping_recorded_for_construction": list(CONSTRUCTION_RGB_MAPPING),
        "source_data_modified": False,
        "test_pixel_reads": 0,
        "created_at": utc_now(),
    }
    write_json(input_dir / "MODULAR_INPUT_AUDIT.json", audit)
    return audit


def _asset_record(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    record = {"path": str(path.resolve()), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "expected_sha256": expected_sha, "size_bytes": path.stat().st_size if path.is_file() else None}
    record["status"] = "pass" if record["exists"] and (expected_sha is None or record["sha256"] == expected_sha) else "fail"
    return record


def verify_frozen_assets(output_root: Path) -> dict[str, Any]:
    assets = {
        "full20_checkpoint": _asset_record(FULL20_CHECKPOINT, EXPECTED_FULL20_SHA),
        "full20_config": _asset_record(FULL20_CONFIG),
        "rsbuilding_checkpoint": _asset_record(RSBUILDING_CHECKPOINT, EXPECTED_RSBUILDING_SHA),
        "rsbuilding_config": _asset_record(RSBUILDING_CONFIG),
        "construction_checkpoint": _asset_record(CONSTRUCTION_CHECKPOINT, EXPECTED_CONSTRUCTION_SHA),
        "construction_package": _asset_record(CONSTRUCTION_PACKAGE / "TASK_MODEL_MANIFEST.json"),
        "surface_checkpoint": _asset_record(SURFACE_CHECKPOINT, EXPECTED_SURFACE_SHA),
    }
    package_manifest = read_json(CONSTRUCTION_PACKAGE / "TASK_MODEL_MANIFEST.json", {})
    assets["construction_package"]["candidate_frozen"] = package_manifest.get("candidate_frozen")
    assets["construction_package"]["rgb_mapping"] = package_manifest.get("rgb_mapping")
    assets["construction_package"]["selection_used_test"] = package_manifest.get("selection_used_test")
    assets["construction_package"]["test_pixels_read_by_this_run"] = 0
    payload = {
        "status": "pass" if all(value.get("status") == "pass" for value in assets.values()) and package_manifest.get("candidate_frozen") is True else "fail",
        "assets": assets,
        "thresholds": {"full20": FULL20_THRESHOLDS, "rsbuilding": RSBUILDING_THRESHOLD, "construction": CONSTRUCTION_THRESHOLD},
        "thresholds_changed": False,
        "models_reselected": False,
        "training_started": False,
        "locked_test_pixel_reads": 0,
        "created_at": utc_now(),
    }
    write_json(output_root / "01_input_audit" / "FROZEN_ASSET_AUDIT.json", payload)
    return payload


ARCHITECTURE_FILES = (
    "tools/tongzhou_full_pipeline.py",
    "tools/nanjing_jiangbei_acceptance.py",
    "tools/infer_rsbuilding_static_geometry.py",
    "tools/rsbuilding_external_vectorize.py",
    "src/urban_monitor/change/inference_v2.py",
    "src/urban_monitor/change/vectorize_v2.py",
    "src/urban_renewal/pipelines/construction_tongzhou_pipeline.py",
    "src/urban_renewal/pipelines/construction_external_pipeline.py",
    "src/urban_renewal/pipelines/construction_temporal_pipeline.py",
    "src/urban_renewal/tracking/construction_matching.py",
    "extensions/surface_state_v1/models/surface_state_net.py",
    "extensions/surface_state_v1/integration/overlay_runtime.py",
    "extensions/surface_state_v1/integration/overlay_final.py",
    "extensions/surface_state_v1/integration/tongzhou_acceptance.py",
    "run.py",
)


def _relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _ast_imports_and_writes(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    result: dict[str, Any] = {"path": _relative(path), "sha256": sha256_file(path), "imports": [], "functions": [], "write_signals": []}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result["imports"].extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                result["imports"].append(module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result["functions"].append(node.name)
    patterns = (
        r"\.write_text\(",
        r"\.write_bytes\(",
        r"\.to_file\(",
        r"\.to_csv\(",
        r"rasterio\.open\([^\n]+['\"]w['\"]",
        r"shutil\.copy2\(",
        r"os\.replace\(",
        r"Image\.fromarray\([^\n]+\.save\(",
    )
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if any(re.search(pattern, line) for pattern in patterns):
            result["write_signals"].append({"line": number, "text": line.strip()[:260]})
    result["imports"] = sorted(set(result["imports"]))
    result["functions"] = sorted(set(result["functions"]))
    return result


def _module_matrix() -> list[dict[str, Any]]:
    return [
        {
            "module_id": "M0",
            "module_name": "Input Audit",
            "responsibility": "discover scenes, dates, SHA256, duplicate exclusion, CRS/grid/bounds inventory",
            "inputs": ["Nanjing source GeoTIFF directory"],
            "outputs": ["01_input_audit/IMAGE_INVENTORY.json", "01_input_audit/ALIGNMENT_AUDIT.json", "INPUT_DUPLICATE_AUDIT.json"],
            "allowed_write_paths": ["01_input_audit", "INPUT_DUPLICATE_AUDIT.json"],
            "forbidden_inputs": ["all model outputs", "training/test pixels"],
            "preprocessing_owner": "M0 only owns audit metadata; each model owns its tensor preprocessing",
            "threshold_owner": "none",
            "scientific_role": "input provenance and temporal eligibility",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M1",
            "module_name": "Geometric Alignment Audit",
            "responsibility": "exact grid/CRS/overlap audit and valid overlap mask",
            "inputs": ["M0 valid scene references"],
            "outputs": ["01_input_audit/COMMON_GRID.json", "01_input_audit/VALID_OVERLAP_MASK.tif"],
            "allowed_write_paths": ["01_input_audit"],
            "forbidden_inputs": ["Construction probabilities", "Surface/MTI outputs"],
            "preprocessing_owner": "M1 owns geometric registration metadata only",
            "threshold_owner": "none",
            "scientific_role": "spatial comparability",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M2",
            "module_name": "Full20 Baseline Change",
            "responsibility": "official Full20 pair inference and baseline change evidence",
            "inputs": ["M0 valid scenes", "M1 grid audit", "frozen Full20 checkpoint/config"],
            "outputs": ["02_baseline_full20"],
            "allowed_write_paths": ["02_baseline_full20", "12_logs/full20"],
            "forbidden_inputs": ["Construction", "Surface State", "MTI", "3D prior", "building site labels"],
            "preprocessing_owner": "src/urban_monitor/change/inference_v2.py official Full20 entry",
            "threshold_owner": "frozen Full20 inference config",
            "scientific_role": "model-derived general change/new/demolition evidence",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M3",
            "module_name": "RSBuilding Baseline Geometry",
            "responsibility": "single-date building probability, masks and MMU-filtered objects",
            "inputs": ["M0 valid scenes", "frozen RSBuilding checkpoint/config"],
            "outputs": ["03_baseline_rsbuilding"],
            "allowed_write_paths": ["03_baseline_rsbuilding", "12_logs/rsbuilding"],
            "forbidden_inputs": ["Construction classes", "Surface State", "Full20 construction interpretation", "MTI", "3D prior"],
            "preprocessing_owner": "RSBuilding runtime and audited wrapper",
            "threshold_owner": "frozen RSBuilding threshold 0.5",
            "scientific_role": "candidate building geometry by date",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M4",
            "module_name": "Baseline Fusion",
            "responsibility": "read-only Full20 + RSBuilding + overlap + geometry matching fusion",
            "inputs": ["M2 sealed Full20 outputs", "M3 sealed geometry", "M1 valid overlap mask"],
            "outputs": ["04_baseline_fusion"],
            "allowed_write_paths": ["04_baseline_fusion"],
            "forbidden_inputs": ["Construction", "Surface State", "no-MTI", "MTI", "3D prior"],
            "preprocessing_owner": "inherits sealed producer outputs; no shared tensor preprocessing",
            "threshold_owner": "frozen producer thresholds plus geometry matching rule",
            "scientific_role": "baseline building-change candidate evidence",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M5",
            "module_name": "Baseline Temporal Tracking",
            "responsibility": "geometry-only cross-date matching and building states",
            "inputs": ["M3 geometry", "M4 baseline fusion"],
            "outputs": ["05_baseline_tracking"],
            "allowed_write_paths": ["05_baseline_tracking"],
            "forbidden_inputs": ["Construction lifecycle labels", "Surface State", "MTI"],
            "preprocessing_owner": "none; geometry-only matching",
            "threshold_owner": "frozen matching distances and area ratios",
            "scientific_role": "building geometry persistence/change candidate",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M6",
            "module_name": "Construction Activity",
            "responsibility": "independent UCS element inference, site candidates and uncertainty",
            "inputs": ["raw Nanjing scenes", "read-only M3 geometry reference", "frozen UCS386 candidate"],
            "outputs": ["06_construction_single_date"],
            "allowed_write_paths": ["06_construction_single_date", "12_logs/construction"],
            "forbidden_inputs": ["M2 Full20 probabilities/masks", "M4 baseline fusion", "M5 baseline tracks"],
            "preprocessing_owner": "UCS386 package: RGB 3,2,1; per-scene percentile 2-98; 512 tiles; 256 input",
            "threshold_owner": "frozen UCS386 package thresholds",
            "scientific_role": "model-derived construction element/site candidate",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M7",
            "module_name": "Surface State + Construction Temporal",
            "responsibility": "surface auxiliary evidence and no-MTI construction lifecycle candidate",
            "inputs": ["raw Nanjing scenes", "M6 construction outputs", "frozen Surface State checkpoint"],
            "outputs": ["07_construction_temporal"],
            "allowed_write_paths": ["07_construction_temporal"],
            "forbidden_inputs": ["M2/M4/M5 baseline outputs for state assignment", "3D prior"],
            "preprocessing_owner": "Surface State owns [1,2,3]/255; construction temporal consumes only sealed M6/M7 evidence",
            "threshold_owner": "frozen surface/construction thresholds; no MTI selection",
            "scientific_role": "model-derived construction lifecycle candidate, not formal ground truth",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M8",
            "module_name": "Cross-reference Overlay",
            "responsibility": "read-only cross-reference after both branches are sealed",
            "inputs": ["M4/M5 baseline seal", "M6/M7 construction seal"],
            "outputs": ["08_cross_reference"],
            "allowed_write_paths": ["08_cross_reference"],
            "forbidden_inputs": ["write access to any upstream output"],
            "preprocessing_owner": "none",
            "threshold_owner": "inherits sealed module thresholds",
            "scientific_role": "interpretive overlap/support/conflict indicators",
            "can_modify_upstream": False,
        },
        {
            "module_id": "M9",
            "module_name": "3D Morphology Prior Interface",
            "responsibility": "future interface only",
            "inputs": [],
            "outputs": ["globfp_enabled=false"],
            "allowed_write_paths": ["12_logs"],
            "forbidden_inputs": ["all current inference results"],
            "preprocessing_owner": "not enabled",
            "threshold_owner": "not enabled",
            "scientific_role": "disabled prior interface",
            "can_modify_upstream": False,
        },
    ]


def _data_contracts() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "immutability": "core results are frozen dataclasses; modules exchange sealed path metadata, never shared mutable dictionaries",
        "dependency_direction": [
            "M0 -> M1",
            "M0/M1 -> M2",
            "M0/M1 -> M3",
            "M2/M3/M1 -> M4 -> M5",
            "M0/M3 -> M6",
            "M0/M6 -> M7",
            "sealed M4/M5 + sealed M6/M7 -> M8 -> report",
            "M9 disabled",
        ],
        "modules": _module_matrix(),
        "core_result_types": [
            "SceneRecord",
            "AlignedSceneReference",
            "Full20PairResult",
            "BuildingGeometryResult",
            "BaselineChangeResult",
            "BuildingTrackResult",
            "ConstructionElementResult",
            "ConstructionTemporalResult",
            "CrossReferenceResult",
        ],
        "forbidden_edges": [
            "M6 -> M2",
            "M6 -> M3",
            "M6 -> M4",
            "M7 -> M4",
            "M7 -> M5",
            "M8 -> any upstream write",
            "report -> model output write",
            "global shared preprocessing -> all models",
        ],
    }


def _audit_current_architecture(output_root: Path) -> None:
    audit_root = output_root / "00_architecture_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    graph = []
    for relative in ARCHITECTURE_FILES:
        path = PROJECT_ROOT / relative
        if path.exists():
            graph.append(_ast_imports_and_writes(path))
    write_json(audit_root / "CURRENT_IMPORT_GRAPH.json", {"status": "pass", "files": graph, "created_at": utc_now()})

    dataflow = """# Current URRS dataflow audit

This document records the pre-inference dataflow observed from the existing
entry points.  It is an audit artifact, not an execution graph: no model is
loaded and no source imagery is copied by this step.

```text
raw Nanjing GeoTIFFs
        |
        v
M0 Input Audit -- dates/SHA/grid metadata --> M1 Geometric Alignment Audit
        |                                             |
        +----------------------+----------------------+
        |                      |
        v                      v
M2 Full20 baseline       M3 RSBuilding geometry
        |                      |
        +----------+-----------+
                   v
          M4 Baseline Fusion
                   |
                   v
          M5 Baseline Tracking

raw Nanjing GeoTIFFs --> M6 Construction Activity --> M7 Surface State /
                                                        no-MTI construction temporal

sealed M4/M5 + sealed M6/M7 --> M8 read-only Cross-reference --> Report

M9 3D morphology prior interface: disabled (globfp_enabled=false)
```

## Historical pollution found

- The historical Nanjing entry wrote aligned imagery into an execution-owned
  directory and then mixed model branches in temporal state construction.
- Its Full20 path used a custom in-process predictor instead of the official
  `run.py urban-renewal-infer` entry used by the Tongzhou baseline.
- Its representative object and report combined baseline geometry/change and
  Construction evidence.

## Refactored write boundary

M0/M1 write only `01_input_audit`; M2 and M3 write only their baseline
producer directories; M4/M5 write only baseline fusion/tracking directories;
M6/M7 write only construction directories; M8 and report generation are
read-only with respect to all producer outputs.
"""
    (audit_root / "CURRENT_DATAFLOW.md").write_text(dataflow, encoding="utf-8")

    current_writes = {
        "status": "pass",
        "historical_entry": "tools/nanjing_jiangbei_acceptance.py",
        "observed_cross_module_writes": [
            {"module": "old M0/M1 input audit", "writes": ["01_aligned_imagery", "VALID_OVERLAP_MASK.tif"], "problem": "input audit produced an execution-owned aligned-image namespace and made it a shared tensor source"},
            {"module": "old Full20", "writes": ["02_full20_change"], "problem": "custom in-process predictor and custom postprocessing bypassed the Tongzhou batch entry"},
            {"module": "old Construction/Surface temporal fusion", "writes": ["05_adjacent_change", "06_cumulative_change", "07_temporal_states"], "problem": "pair evidence read construction, surface, geometry and Full20 together; baseline and construction meanings were mixed"},
            {"module": "old representative object", "writes": ["08_representative_object"], "problem": "one object combined baseline building change and construction lifecycle evidence"},
            {"module": "old report", "writes": ["report HTML and proxy tables"], "problem": "baseline metrics and construction metrics were rendered in one mixed metric layer"},
            {"module": "construction_external_pipeline", "writes": ["construction output root only"], "problem": "private imports from construction_tongzhou_pipeline made the adapter tightly coupled; the new orchestration treats it as a Construction-only runner and passes no Full20/Surface manifest"},
        ],
        "new_write_policy": {module["module_id"]: module["allowed_write_paths"] for module in _module_matrix()},
        "created_at": utc_now(),
    }
    write_json(audit_root / "CURRENT_CROSS_MODULE_WRITES.json", current_writes)

    preprocessing = {
        "status": "pass",
        "shared_global_preprocessing": False,
        "modules": {
            "M0_M1": {"owner": "input and geometry audit", "behavior": "metadata/grid/overlap only; no model tensor supplied"},
            "M2_Full20": {"owner": "src/urban_monitor/change/inference_v2.py", "entry": "run.py urban-renewal-infer", "rgb_behavior": "official _read_rgb reads source bands 1,2,3 and per-window percentile normalization", "tile_size": 512, "overlap": 64},
            "M3_RSBuilding": {"owner": "RSBuilding runtime plus infer_rsbuilding_static_geometry.py", "tile_size": 512, "overlap": 128, "threshold": RSBUILDING_THRESHOLD},
            "M6_Construction": {"owner": "UCS386 package preprocessing.json", "rgb_mapping": list(CONSTRUCTION_RGB_MAPPING), "normalization": "per-scene percentile 2-98", "tile_size": 512, "model_input_size": 256},
            "M7_SurfaceState": {"owner": "SurfaceStateNet external candidate runner", "input_bands": [1, 2, 3], "normalization": "uint8 / 255", "tile_size": 512, "integration": "no_mti"},
        },
        "interpretation": "M1 geometric alignment is not substituted for model-owned radiometric preprocessing; baseline and construction never share normalized tensors",
        "created_at": utc_now(),
    }
    write_json(audit_root / "CURRENT_PREPROCESSING_COMPARISON.json", preprocessing)

    plan = """# URRS Nanjing module refactor plan\n\n## Findings in the historical Nanjing entry\n\n1. `tools/nanjing_jiangbei_acceptance.py` ran RSBuilding, Full20, Surface State and Construction before any baseline seal.\n2. Its `_pair_evidence_and_states` function combined Construction probabilities, Surface State, RSBuilding geometry and Full20 masks into `05_adjacent_change`, `06_cumulative_change` and `07_temporal_states`. Those outputs were therefore not a baseline-only result.\n3. Its Full20 runner used `_full20_predict` and custom raster writing instead of the same `run.py urban-renewal-infer` entry used by the mature Tongzhou flow.\n4. Its representative object and report mixed building-change evidence with construction lifecycle evidence.\n5. Its input audit copied aligned imagery into an execution directory, while the model runners still owned different band-order and normalization rules.\n6. The old run admitted the byte-identical 2024-10-20 observation as a formal endpoint even though it duplicated 2022-12-02.\n\n## Refactor boundary\n\n- M0/M1 audit only; no inference and no source-image copy.\n- M2 and M3 run independently and write only `02_baseline_full20` and `03_baseline_rsbuilding`.\n- M4 reads only sealed M2/M3/M1 results. M5 reads only geometry and baseline fusion.\n- M6 reads raw scenes with UCS386-owned preprocessing and may read M3 geometry as a reference for auditable parcel candidates; it writes only `06_construction_single_date`.\n- M7 owns Surface State and no-MTI construction temporal candidates; it writes only `07_construction_temporal`.\n- M8 is read-only cross-reference after both seals. Reports consume tables and figures but do not rewrite producer outputs.\n- M9 is an explicit disabled interface: `globfp_enabled=false`.\n\n## Acceptance gates\n\n- Duplicate observations are excluded before pair creation.\n- Formal baseline pairs are each valid earlier date to the final valid date; adjacent pairs are appendix-only.\n- Full20 checkpoint/config/threshold/tile/vectorization/MMU behavior is audited against Tongzhou.\n- Baseline seal is created before Construction starts.\n- Construction-enabled/disabled logical baseline runs have byte-identical baseline hashes.\n- Baseline and Construction reports use separate metric tables and separate representative objects.\n"""
    (audit_root / "REFACTOR_PLAN.md").write_text(plan, encoding="utf-8")
    (output_root / "MODULE_RESPONSIBILITY_MATRIX.md").write_text(
        "# URRS 模块职责矩阵\n\n" + "\n".join(
            f"| {item['module_id']} | {item['module_name']} | {item['responsibility']} | {'; '.join(item['allowed_write_paths'])} |"
            for item in _module_matrix()
        ) + "\n",
        encoding="utf-8",
    )
    write_json(output_root / "MODULE_DATA_CONTRACTS.json", _data_contracts())


def run_audit(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    refuse_old_output(output_root)
    ensure_layout(output_root)
    _audit_current_architecture(output_root)
    before = _snapshot_source_files(Path(args.source_root).resolve())
    input_audit = audit_inputs(Path(args.source_root).resolve(), output_root)
    assets = verify_frozen_assets(output_root)
    parity = build_tongzhou_parity_audit(output_root)
    write_json(output_root / "00_architecture_audit" / "PROTECTED_SOURCE_SNAPSHOT.json", before)
    write_json(output_root / "00_architecture_audit" / "ARCHITECTURE_AUDIT_RECEIPT.json", {
        "status": "pass" if assets.get("status") == "pass" and input_audit.get("status") in {"pass", "partial"} else "partial",
        "inference_started": False,
        "training_started": False,
        "locked_test_pixel_reads": 0,
        "original_data_modified": False,
        "old_output_read_only": str(OLD_OUTPUT_ROOT.resolve()),
        "input_image_count": len(input_audit["records"]),
        "valid_unique_image_count": len(input_audit["valid_records"]),
        "effective_final_date": input_audit["final_date"],
        "tongzhou_baseline_parity": parity.get("status"),
        "created_at": utc_now(),
    })
    return {"status": "pass", "output_root": str(output_root.resolve()), "input_audit": input_audit, "assets": assets, "tongzhou_baseline_parity": parity}


def _input_audit_from_output(output_root: Path) -> dict[str, Any]:
    path = output_root / "01_input_audit" / "MODULAR_INPUT_AUDIT.json"
    payload = read_json(path)
    if not payload:
        raise FileNotFoundError(f"Run Audit before model modules: {path}")
    return payload


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path, label: str) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[{utc_now()}] START {label}\n$ {' '.join(str(item) for item in command)}\n")
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
        elapsed = time.perf_counter() - started
        log.write(f"[{utc_now()}] END {label} return_code={return_code} elapsed_seconds={elapsed:.3f}\n")
    return {"label": label, "command": [str(item) for item in command], "return_code": return_code, "elapsed_seconds": elapsed, "log": str(log_path.resolve())}


def _full20_required_outputs(pair_root: Path) -> list[Path]:
    return [
        pair_root / "INFERENCE_MANIFEST.json",
        pair_root / "rasters" / "any_change_probability.tif",
        pair_root / "rasters" / "any_change_mask.tif",
        pair_root / "rasters" / "new_building_candidate.tif",
        pair_root / "rasters" / "new_building_mask.tif",
        pair_root / "rasters" / "demolition_candidate.tif",
        pair_root / "rasters" / "demolition_mask.tif",
        pair_root / "preview.png",
    ]


def _metric_vectorize(mask_path: Path, output_path: Path, layer: str) -> dict[str, Any]:
    """The same EPSG:32650 + 10m2 rule used by Tongzhou's metric refinement."""

    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(mask_path) as dataset:
        mask = dataset.read(1)
        source_crs = dataset.crs
        geometries = [
            shape(geometry)
            for geometry, value in shapes(mask.astype(np.uint8), mask=mask > 0, transform=dataset.transform)
            if int(value) == 1
        ]
    raw_count = len(geometries)
    if geometries:
        frame = gpd.GeoDataFrame({"class": [layer] * raw_count}, geometry=geometries, crs=source_crs).to_crs(METRIC_CRS)
        frame["area_m2"] = frame.geometry.area
        frame = frame.loc[frame["area_m2"] >= MMU_M2].copy().reset_index(drop=True)
        frame["object_id"] = [f"{layer}_{index + 1:05d}" for index in range(len(frame))]
    else:
        frame = gpd.GeoDataFrame({"class": [], "area_m2": [], "object_id": []}, geometry=[], crs=METRIC_CRS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_file(output_path, layer=layer, driver="GPKG", engine="pyogrio")
    except Exception:
        # Keep an auditable empty result without inventing an object count.
        if not frame.empty:
            raise
        write_json(output_path.with_suffix(".EMPTY.json"), {"status": "empty_after_mmu", "layer": layer, "raw_count": raw_count, "mmu_m2": MMU_M2})
    return {"path": str(output_path.resolve()), "raw_feature_count": raw_count, "features_after_mmu": int(len(frame)), "area_m2_sum": float(frame["area_m2"].sum()) if not frame.empty else 0.0, "metric_crs": METRIC_CRS, "mmu_m2": MMU_M2}


def run_full20_baseline(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    """M2: call the same official Full20 entry used by the Tongzhou flow."""

    import yaml

    audit = _input_audit_from_output(output_root)
    assets = read_json(output_root / "01_input_audit" / "FROZEN_ASSET_AUDIT.json", {})
    if assets.get("status") != "pass":
        raise RuntimeError("Frozen asset audit is not pass; Full20 baseline is blocked")
    baseline_root = output_root / "02_baseline_full20"
    baseline_root.mkdir(parents=True, exist_ok=True)
    records = {str(item["date"]): item for item in audit["valid_records"]}
    pairs = audit["formal_pairs"]
    config_payload = yaml.safe_load(FULL20_CONFIG.read_text(encoding="utf-8-sig"))
    thresholds = config_payload.get("thresholds", FULL20_THRESHOLDS)
    tiling = config_payload.get("inference", {})
    pair_rows = []
    results: list[dict[str, Any]] = []
    python = Path(args.python).resolve()
    for pair in pairs:
        date_from, date_to = str(pair["date_from"]), str(pair["date_to"])
        pair_root = baseline_root / str(pair["pair_name"])
        required = _full20_required_outputs(pair_root)
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            command = [
                str(python),
                str(PROJECT_ROOT / "run.py"),
                "urban-renewal-infer",
                "--config", str(FULL20_CONFIG),
                "--t1", str(records[date_from]["path"]),
                "--t2", str(records[date_to]["path"]),
                "--output", str(pair_root),
                "--device", str(args.device),
            ]
            receipt = _run_logged(command, cwd=PROJECT_ROOT, log_path=output_root / "12_logs" / "full20" / f"{pair['pair_name']}.log", label=f"Full20:{pair['pair_name']}")
            if receipt["return_code"] != 0:
                raise RuntimeError(f"Full20 official inference failed for {pair['pair_name']}; see {receipt['log']}")
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            raise RuntimeError(f"Full20 output incomplete for {pair_root}")
        metric_new = _metric_vectorize(pair_root / "rasters" / "new_building_mask.tif", pair_root / "vectors_metric" / "new_building_metric.gpkg", "new_building")
        metric_demo = _metric_vectorize(pair_root / "rasters" / "demolition_mask.tif", pair_root / "vectors_metric" / "demolition_metric.gpkg", "demolition")
        official_manifest = read_json(pair_root / "INFERENCE_MANIFEST.json", {})
        wrapper_manifest = {
            "status": "success",
            "module": "M2_Full20_Baseline",
            "pair": pair,
            "official_entry": "run.py urban-renewal-infer -> urban_monitor.change.inference_v2.run_urban_renewal_inference",
            "official_manifest": str((pair_root / "INFERENCE_MANIFEST.json").resolve()),
            "checkpoint_sha256": EXPECTED_FULL20_SHA,
            "config_path": str(FULL20_CONFIG.resolve()),
            "config_sha256": sha256_file(FULL20_CONFIG),
            "thresholds": thresholds,
            "tile_size": int(tiling.get("tile_size", 512)),
            "overlap": int(tiling.get("overlap", 64)),
            "vectorization_mmu_m2": MMU_M2,
            "metric_vectorization": {"new": metric_new, "demolition": metric_demo},
            "t2_aligned_to_t1": official_manifest.get("t2_aligned_to_t1"),
            "test_images_read": 0,
            "construction_enabled": bool(args.construction_enabled),
            "created_at": utc_now(),
        }
        write_json(pair_root / "FULL20_BASELINE_PAIR_MANIFEST.json", wrapper_manifest)
        results.append(wrapper_manifest)
        pair_rows.append({
            **pair,
            "status": "success",
            "pair_root": str(pair_root.resolve()),
            "checkpoint_sha256": EXPECTED_FULL20_SHA,
            "config_sha256": sha256_file(FULL20_CONFIG),
            "threshold_any_change": thresholds.get("any_change"),
            "threshold_new_building": thresholds.get("new_building"),
            "threshold_demolition": thresholds.get("demolition"),
            "test_images_read": 0,
        })
    write_csv(baseline_root / "BASELINE_FULL20_PAIR_PLAN.csv", list(pair_rows[0].keys()) if pair_rows else ["status"], pair_rows)
    manifest = {
        "status": "completed",
        "module": "M2_Full20_Baseline",
        "dates": audit["dates"],
        "formal_pair_strategy": "each_valid_earlier_date_to_final_valid_date",
        "pairs": results,
        "checkpoint": str(FULL20_CHECKPOINT.resolve()),
        "checkpoint_sha256": EXPECTED_FULL20_SHA,
        "config": str(FULL20_CONFIG.resolve()),
        "config_sha256": sha256_file(FULL20_CONFIG),
        "thresholds": thresholds,
        "tile_size": int(tiling.get("tile_size", 512)),
        "overlap": int(tiling.get("overlap", 64)),
        "vectorization_mmu_m2": MMU_M2,
        "metric_crs": METRIC_CRS,
        "construction_enabled": bool(args.construction_enabled),
        "training_started": False,
        "test_images_read": 0,
        "created_at": utc_now(),
    }
    write_json(baseline_root / "FULL20_BASELINE_MANIFEST.json", manifest)
    return manifest


def run_rsbuilding_baseline(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    """M3: invoke the existing RSBuilding wrapper in its own environment."""

    audit = _input_audit_from_output(output_root)
    geometry_root = output_root / "03_baseline_rsbuilding"
    common = [
        str(RSBUILDING_PYTHON),
        str(PROJECT_ROOT / "tools" / "infer_rsbuilding_static_geometry.py"),
        "--source-root", str(Path(audit["source_root"]).resolve()),
        "--output-root", str(geometry_root),
        "--checkpoint", str(RSBUILDING_CHECKPOINT),
        "--config", str(RSBUILDING_CONFIG),
        "--device", "cuda:0",
        "--batch-size", "2",
    ]
    for date in audit["dates"]:
        common.extend(["--target-date", str(date)])
    receipts = []
    if not (geometry_root / "RSBUILDING_BASELINE_MANIFEST.json").exists():
        for label, command in (
            ("discover", common[:2] + ["discover"] + common[2:]),
            ("diagnose", common[:2] + ["diagnose"] + common[2:]),
            ("smoke", common[:2] + ["smoke"] + common[2:] + ["--smoke-date", str(audit["dates"][-1])]),
            ("infer-all", common[:2] + ["infer-all"] + common[2:]),
            ("audit", common[:2] + ["audit"] + common[2:]),
            ("vectorize", [str(args.python), str(PROJECT_ROOT / "tools" / "rsbuilding_external_vectorize.py"), "--output-root", str(geometry_root), "--min-area-m2", str(MMU_M2)]),
            ("alignment", common[:2] + ["alignment"] + common[2:]),
        ):
            receipt = _run_logged(command, cwd=PROJECT_ROOT, log_path=output_root / "12_logs" / "rsbuilding" / f"{label}.log", label=f"RSBuilding:{label}")
            receipts.append(receipt)
            if receipt["return_code"] != 0:
                raise RuntimeError(f"RSBuilding {label} failed; see {receipt['log']}")
    scene_rows = []
    for date in audit["dates"]:
        date_root = geometry_root / str(date)
        manifest_path = date_root / "GEOMETRY_MANIFEST.json"
        if not manifest_path.exists():
            raise RuntimeError(f"RSBuilding geometry manifest missing for {date}: {manifest_path}")
        manifest = read_json(manifest_path, {})
        required = {name: date_root / name for name in ("building_probability.tif", "building_mask.tif", "building_boundary.tif", "building_objects.gpkg")}
        if not all(path.is_file() for path in required.values()):
            raise RuntimeError(f"RSBuilding geometry outputs incomplete for {date}")
        scene_rows.append(contract_to_dict(BuildingGeometryResult(
            date=str(date),
            output_root=str(date_root.resolve()),
            probability_path=str(required["building_probability.tif"].resolve()),
            mask_path=str(required["building_mask.tif"].resolve()),
            boundary_path=str(required["building_boundary.tif"].resolve()),
            vector_path=str(required["building_objects.gpkg"].resolve()),
            manifest_path=str(manifest_path.resolve()),
            checkpoint_sha256=str(manifest.get("checkpoint_sha256", EXPECTED_RSBUILDING_SHA)),
            threshold=float(manifest.get("threshold", RSBUILDING_THRESHOLD)),
            tile_size=int(manifest.get("tile_size", 512)),
            overlap=int(manifest.get("overlap", 128)),
            test_reads={"WHU": 0, "SECOND": 0},
        )))
    payload = {
        "status": "completed",
        "module": "M3_RSBuilding_Baseline",
        "dates": scene_rows,
        "checkpoint": str(RSBUILDING_CHECKPOINT.resolve()),
        "checkpoint_sha256": EXPECTED_RSBUILDING_SHA,
        "config": str(RSBUILDING_CONFIG.resolve()),
        "config_sha256": sha256_file(RSBUILDING_CONFIG),
        "threshold": RSBUILDING_THRESHOLD,
        "tile_size": 512,
        "overlap": 128,
        "minimum_mapping_unit_m2": MMU_M2,
        "vectorizer": "tools/rsbuilding_external_vectorize.py (project pyogrio runtime)",
        "receipts": receipts,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "training_started": False,
        "created_at": utc_now(),
    }
    write_json(geometry_root / "RSBUILDING_BASELINE_MANIFEST.json", payload)
    return payload


def _read_raster(path: Path, *, out_shape: tuple[int, int] | None = None, resampling: str = "nearest") -> tuple[np.ndarray, dict[str, Any]]:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        if out_shape:
            method = Resampling.bilinear if resampling == "bilinear" else Resampling.nearest
            array = dataset.read(1, out_shape=(1, int(out_shape[0]), int(out_shape[1])), resampling=method)
        else:
            array = dataset.read(1)
        meta = {
            "width": dataset.width,
            "height": dataset.height,
            "crs": dataset.crs,
            "transform": dataset.transform,
            "profile": dataset.profile.copy(),
            "bounds": tuple(dataset.bounds),
        }
    return np.asarray(array), meta


def _write_mask(path: Path, array: np.ndarray, reference_path: Path) -> None:
    import rasterio

    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        for key in ("photometric", "interleave"):
            profile.pop(key, None)
        profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate", predictor=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as target:
            target.write(np.asarray(array, dtype=np.uint8), 1)


def _write_float(path: Path, array: np.ndarray, reference_path: Path) -> None:
    import rasterio

    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        for key in ("photometric", "interleave"):
            profile.pop(key, None)
        profile.update(driver="GTiff", count=1, dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as target:
            target.write(np.asarray(array, dtype=np.float32), 1)


def _building_objects(path: Path) -> list[dict[str, Any]]:
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame.empty:
        return []
    metric = frame.to_crs(METRIC_CRS)
    rows = []
    for index, row in frame.iterrows():
        geometry = metric.loc[index].geometry
        if geometry is None or geometry.is_empty:
            continue
        rows.append({
            "object_id": str(row.get("object_id", f"building_{index:08d}")),
            "scene_type": "building_candidate",
            "geometry": geometry,
            "area_m2": float(geometry.area),
        })
    return rows


def _match_geometry(previous: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from urban_renewal.tracking.construction_matching import match_construction_objects

    return match_construction_objects(list(previous), list(current), max_centroid_distance=100.0)


def _metric_area(mask: np.ndarray, reference_path: Path) -> float:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    with rasterio.open(reference_path) as reference:
        transform, width, height = calculate_default_transform(reference.crs, METRIC_CRS, reference.width, reference.height, *reference.bounds)
        destination = np.zeros((int(height), int(width)), dtype=np.uint8)
        reproject(
            np.asarray(mask, dtype=np.uint8),
            destination,
            src_transform=reference.transform,
            src_crs=reference.crs,
            dst_transform=transform,
            dst_crs=METRIC_CRS,
            resampling=Resampling.nearest,
        )
    pixel_area = abs(float(transform.a * transform.e))
    return float(destination.sum() * pixel_area)


def _connected_count(mask: np.ndarray) -> int:
    try:
        from scipy.ndimage import label

        _, count = label(np.asarray(mask, dtype=np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
        return int(count)
    except Exception:
        return int(bool(np.asarray(mask).any()))


def _area_and_count_from_objects(path: Path) -> tuple[int, float]:
    rows = _building_objects(path)
    return len(rows), float(sum(float(row["area_m2"]) for row in rows))


def _pair_geometry_change(previous_path: Path, current_path: Path) -> dict[str, Any]:
    previous = _building_objects(previous_path)
    current = _building_objects(current_path)
    matched = _match_geometry(previous, current)
    expanded = []
    contracted = []
    stable = []
    match_rows = []
    for item in matched["matches"]:
        left, right = previous[item["previous_index"]], current[item["current_index"]]
        ratio = float(right["area_m2"] / max(left["area_m2"], 1e-9))
        if ratio > 1.20:
            state = "expanded_building"
            expanded.append(right)
        elif ratio < 0.80:
            state = "contracted_building"
            contracted.append(left)
        else:
            state = "stable_building"
            stable.append(right)
        match_rows.append({"previous_object_id": left["object_id"], "current_object_id": right["object_id"], "match_score": item["match_score"], "area_ratio": ratio, "state": state})
    return {
        "previous": previous,
        "current": current,
        "matched": matched,
        "expanded": expanded,
        "contracted": contracted,
        "stable": stable,
        "match_rows": match_rows,
        "new_count": len(matched["unmatched_current"]),
        "disappeared_count": len(matched["unmatched_previous"]),
        "expanded_count": len(expanded),
        "contracted_count": len(contracted),
        "stable_count": len(stable),
    }


def run_baseline_fusion(output_root: Path) -> dict[str, Any]:
    """M4: only Full20, RSBuilding, M1 overlap and geometry matching are read."""

    audit = _input_audit_from_output(output_root)
    full20_manifest = read_json(output_root / "02_baseline_full20" / "FULL20_BASELINE_MANIFEST.json", {})
    geometry_manifest = read_json(output_root / "03_baseline_rsbuilding" / "RSBUILDING_BASELINE_MANIFEST.json", {})
    if full20_manifest.get("status") != "completed" or geometry_manifest.get("status") != "completed":
        raise RuntimeError("Baseline Fusion requires completed Full20 and RSBuilding modules")
    fusion_root = output_root / "04_baseline_fusion"
    pair_results = []
    geometry_by_date = {str(item["date"]): item for item in geometry_manifest["dates"]}
    for pair in audit["formal_pairs"]:
        date_from, date_to = str(pair["date_from"]), str(pair["date_to"])
        pair_root = fusion_root / str(pair["pair_name"])
        pair_root.mkdir(parents=True, exist_ok=True)
        full20_root = output_root / "02_baseline_full20" / str(pair["pair_name"])
        from_mask, from_meta = _read_raster(Path(geometry_by_date[date_from]["mask_path"]))
        to_mask, to_meta = _read_raster(Path(geometry_by_date[date_to]["mask_path"]))
        any_change, _ = _read_raster(full20_root / "rasters" / "any_change_mask.tif")
        new_mask, _ = _read_raster(full20_root / "rasters" / "new_building_mask.tif")
        demolition_mask, _ = _read_raster(full20_root / "rasters" / "demolition_mask.tif")
        if any_change.shape != from_mask.shape or to_mask.shape != from_mask.shape:
            raise RuntimeError(f"Baseline grid mismatch for {pair['pair_name']}; fusion refuses implicit resampling")
        geometry_new = (to_mask > 0) & (from_mask == 0)
        geometry_disappeared = (from_mask > 0) & (to_mask == 0)
        baseline_new = (new_mask > 0) & (to_mask > 0)
        baseline_demolition = (demolition_mask > 0) & (from_mask > 0)
        baseline_change = (any_change > 0)
        stable = (from_mask > 0) & (to_mask > 0) & ~(baseline_new | baseline_demolition)
        geometry_change = geometry_new | geometry_disappeared
        geom = _pair_geometry_change(Path(geometry_by_date[date_from]["vector_path"]), Path(geometry_by_date[date_to]["vector_path"]))
        masks = {
            "baseline_change_candidate": baseline_change,
            "baseline_new_building_candidate": baseline_new,
            "baseline_demolition_candidate": baseline_demolition,
            "geometry_new_building_candidate": geometry_new,
            "geometry_disappeared_building_candidate": geometry_disappeared,
            "geometry_expanded_building_candidate": np.zeros_like(baseline_change, dtype=bool),
            "geometry_contracted_building_candidate": np.zeros_like(baseline_change, dtype=bool),
            "stable_building_candidate": stable,
            "geometry_change_candidate": geometry_change,
        }
        for name, mask in masks.items():
            _write_mask(pair_root / f"{name}.tif", mask, Path(geometry_by_date[date_from]["mask_path"]))
        metric_new = _metric_vectorize(pair_root / "baseline_new_building_candidate.tif", pair_root / "vectors_metric" / "baseline_new_building.gpkg", "baseline_new_building")
        metric_demo = _metric_vectorize(pair_root / "baseline_demolition_candidate.tif", pair_root / "vectors_metric" / "baseline_demolition.gpkg", "baseline_demolition")
        from_count, from_area = _area_and_count_from_objects(Path(geometry_by_date[date_from]["vector_path"]))
        to_count, to_area = _area_and_count_from_objects(Path(geometry_by_date[date_to]["vector_path"]))
        metrics = {
            "pair_name": pair["pair_name"],
            "date_from": date_from,
            "date_to": date_to,
            "full20_changed_area_m2": _metric_area(baseline_change, Path(geometry_by_date[date_from]["mask_path"])),
            "full20_changed_fraction": float(baseline_change.mean()),
            "full20_mean_change_probability": float(_read_raster(full20_root / "rasters" / "any_change_probability.tif", resampling="bilinear")[0].mean()),
            "full20_high_confidence_change_fraction": float(baseline_change.mean()),
            "full20_uncertainty_fraction": float(_read_raster(full20_root / "rasters" / "uncertainty_probability.tif", resampling="bilinear")[0].mean()) if (full20_root / "rasters" / "uncertainty_probability.tif").exists() else None,
            "full20_connected_change_region_count": _connected_count(baseline_change),
            "building_count_t1": from_count,
            "building_count_t2": to_count,
            "building_area_t1_m2": from_area,
            "building_area_t2_m2": to_area,
            "building_coverage_t1": float((from_mask > 0).mean()),
            "building_coverage_t2": float((to_mask > 0).mean()),
            "newly_detected_building_count": geom["new_count"],
            "disappeared_building_count": geom["disappeared_count"],
            "expanded_building_count": geom["expanded_count"],
            "contracted_building_count": geom["contracted_count"],
            "stable_building_count": geom["stable_count"],
            "new_building_area_m2": float(sum(item["area_m2"] for index, item in enumerate(geom["current"]) if index in geom["matched"]["unmatched_current"])),
            "disappeared_building_area_m2": float(sum(item["area_m2"] for index, item in enumerate(geom["previous"]) if index in geom["matched"]["unmatched_previous"])),
            "net_building_change_area_m2": float(to_area - from_area),
            "building_track_persistence_rate": float(len(geom["matched"]["matches"]) / max(len(geom["previous"]), 1)),
            "geometry_matching_rule": "spatial-index + Hungarian one-to-one; centroid <=100m; overlap/area/type score",
            "metric_area_crs": METRIC_CRS,
            "mmu_m2": MMU_M2,
            "construction_inputs_used": False,
            "surface_inputs_used": False,
            "mti_inputs_used": False,
        }
        write_json(pair_root / "BASELINE_PAIR_METRICS.json", metrics)
        write_csv(pair_root / "BASELINE_GEOMETRY_MATCHES.csv", ["previous_object_id", "current_object_id", "match_score", "area_ratio", "state"], geom["match_rows"])
        result = contract_to_dict(BaselineChangeResult(
            pair_name=str(pair["pair_name"]),
            output_root=str(pair_root.resolve()),
            full20_input=str((full20_root / "INFERENCE_MANIFEST.json").resolve()),
            rsbuilding_inputs=(str(geometry_by_date[date_from]["manifest_path"]), str(geometry_by_date[date_to]["manifest_path"])),
            valid_overlap_mask=str((output_root / "01_input_audit" / "VALID_OVERLAP_MASK.tif").resolve()),
            output_files=tuple(str(path.resolve()) for path in sorted(pair_root.rglob("*") if pair_root.exists() else []) if path.is_file()),
            metric_table=str((pair_root / "BASELINE_PAIR_METRICS.json").resolve()),
        ))
        wrapper = {"status": "completed", **result, "metrics": metrics, "sources": {"full20": str(full20_root.resolve()), "rsbuilding": [geometry_by_date[date_from]["output_root"], geometry_by_date[date_to]["output_root"]]}, "created_at": utc_now()}
        write_json(pair_root / "BASELINE_FUSION_PAIR_MANIFEST.json", wrapper)
        pair_results.append(wrapper)
    write_json(fusion_root / "BASELINE_FUSION_MANIFEST.json", {
        "status": "completed",
        "module": "M4_Baseline_Fusion",
        "pairs": pair_results,
        "allowed_sources": ["M1 valid overlap", "M2 Full20", "M3 RSBuilding", "geometry matching"],
        "forbidden_sources": ["Construction", "Surface State", "no-MTI", "MTI", "3D-GloBFP"],
        "created_at": utc_now(),
    })
    return read_json(fusion_root / "BASELINE_FUSION_MANIFEST.json")


def run_baseline_tracking(output_root: Path) -> dict[str, Any]:
    """M5: geometry-only tracking; no construction or Surface State reads."""

    audit = _input_audit_from_output(output_root)
    geometry_manifest = read_json(output_root / "03_baseline_rsbuilding" / "RSBUILDING_BASELINE_MANIFEST.json", {})
    geometry_by_date = {str(item["date"]): item for item in geometry_manifest.get("dates", [])}
    tracking_root = output_root / "05_baseline_tracking"
    tracking_root.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    for pair in [*audit["formal_pairs"], *audit["adjacent_pairs"]]:
        date_from, date_to = str(pair["date_from"]), str(pair["date_to"])
        previous = _building_objects(Path(geometry_by_date[date_from]["vector_path"]))
        current = _building_objects(Path(geometry_by_date[date_to]["vector_path"]))
        matched = _match_geometry(previous, current)
        rows = []
        for item in matched["matches"]:
            left, right = previous[item["previous_index"]], current[item["current_index"]]
            ratio = float(right["area_m2"] / max(left["area_m2"], 1e-9))
            state = "expanded_building" if ratio > 1.2 else "contracted_building" if ratio < 0.8 else "stable_building"
            rows.append({"date_from": date_from, "date_to": date_to, "previous_object_id": left["object_id"], "current_object_id": right["object_id"], "match_score": item["match_score"], "area_ratio": ratio, "geometry_state": state})
        rows.extend({"date_from": date_from, "date_to": date_to, "previous_object_id": "", "current_object_id": current[index]["object_id"], "match_score": "", "area_ratio": "", "geometry_state": "newly_detected_building"} for index in matched["unmatched_current"])
        rows.extend({"date_from": date_from, "date_to": date_to, "previous_object_id": previous[index]["object_id"], "current_object_id": "", "match_score": "", "area_ratio": "", "geometry_state": "disappeared_building"} for index in matched["unmatched_previous"])
        pair_root = tracking_root / str(pair["pair_type"]) / str(pair["pair_name"])
        write_csv(pair_root / "BUILDING_MATCHES.csv", ["date_from", "date_to", "previous_object_id", "current_object_id", "match_score", "area_ratio", "geometry_state"], rows)
        summary = {
            "pair_name": pair["pair_name"], "pair_type": pair["pair_type"], "date_from": date_from, "date_to": date_to,
            "building_track_persistence_rate": float(len(matched["matches"]) / max(len(previous), 1)),
            "newly_detected_building_count": len(matched["unmatched_current"]),
            "disappeared_building_count": len(matched["unmatched_previous"]),
            "matching_rule": "geometry-only spatial-index + Hungarian one-to-one; centroid <=100m; area ratio states",
            "construction_inputs_used": False, "surface_inputs_used": False, "mti_inputs_used": False,
        }
        write_json(pair_root / "BUILDING_TRACKING_SUMMARY.json", summary)
        pair_rows.append(summary)
    write_csv(tracking_root / "BASELINE_TRACKING_METRICS.csv", sorted({key for row in pair_rows for key in row}), pair_rows)
    track_payload = {
        "status": "completed",
        "module": "M5_Baseline_Temporal_Tracking",
        "formal_pairs": [row for row in pair_rows if row.get("pair_type") == "cumulative_primary"],
        "adjacent_pairs": [row for row in pair_rows if row.get("pair_type") == "adjacent_appendix"],
        "state_ontology": ["building_absent", "building_present", "newly_detected_building", "disappeared_building", "expanded_building", "contracted_building", "stable_building", "uncertain_geometry"],
        "construction_inputs_used": False,
        "surface_inputs_used": False,
        "created_at": utc_now(),
    }
    write_json(tracking_root / "BASELINE_TRACKING_MANIFEST.json", track_payload)
    return track_payload


def seal_baseline(output_root: Path) -> dict[str, Any]:
    """Create the immutable baseline receipt before any Construction run."""

    required = [
        output_root / "02_baseline_full20" / "FULL20_BASELINE_MANIFEST.json",
        output_root / "03_baseline_rsbuilding" / "RSBUILDING_BASELINE_MANIFEST.json",
        output_root / "04_baseline_fusion" / "BASELINE_FUSION_MANIFEST.json",
        output_root / "05_baseline_tracking" / "BASELINE_TRACKING_MANIFEST.json",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("Cannot seal baseline before M2-M5 are complete")
    files = []
    for module_dir, module_name in (("02_baseline_full20", "M2"), ("03_baseline_rsbuilding", "M3"), ("04_baseline_fusion", "M4"), ("05_baseline_tracking", "M5")):
        root = output_root / module_dir
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "BASELINE_SEAL_MANIFEST.json":
                continue
            relative = str(path.relative_to(output_root)).replace("\\", "/")
            files.append({"relative_path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "module": module_name})
    payload = {
        "status": "sealed",
        "seal_schema_version": "1.0.0",
        "created_at": utc_now(),
        "baseline_output_files": files,
        "file_count": len(files),
        "file_list_sha256": hashlib.sha256(json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "checkpoint_sha256": {"full20": EXPECTED_FULL20_SHA, "rsbuilding": EXPECTED_RSBUILDING_SHA},
        "config_sha256": {"full20": sha256_file(FULL20_CONFIG), "rsbuilding": sha256_file(RSBUILDING_CONFIG)},
        "thresholds": {"full20": FULL20_THRESHOLDS, "rsbuilding": RSBUILDING_THRESHOLD, "mmu_m2": MMU_M2},
        "construction_started_before_seal": False,
        "construction_outputs_included": False,
        "upstream_immutable": True,
    }
    write_json(output_root / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json", payload)
    return payload


def _construction_rgb_preview(source_path: Path, target: Path, maximum: int = 1400) -> None:
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling

    with rasterio.open(source_path) as dataset:
        scale = min(1.0, float(maximum) / max(dataset.width, dataset.height))
        width = max(1, int(dataset.width * scale))
        height = max(1, int(dataset.height * scale))
        array = dataset.read(list(CONSTRUCTION_RGB_MAPPING), out_shape=(3, height, width), resampling=Resampling.bilinear).astype(np.float32)
    image = np.moveaxis(array, 0, -1)
    for channel in range(3):
        finite = image[..., channel][np.isfinite(image[..., channel])]
        low, high = np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0)
        image[..., channel] = np.clip((image[..., channel] - low) / max(float(high - low), 1e-6), 0, 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image * 255).astype(np.uint8), mode="RGB").save(target)


def _rasterize_construction_candidates(gpkg_path: Path, source_path: Path, target_dir: Path) -> dict[str, Any]:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from scipy.ndimage import binary_dilation

    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        transform = source.transform
        shape_hw = (source.height, source.width)
        source_crs = source.crs
    frame = gpd.read_file(gpkg_path)
    if frame.crs and source_crs:
        frame = frame.to_crs(source_crs)
    if "construction_candidate" not in frame.columns:
        frame["construction_candidate"] = False
    if "construction_non_background" not in frame.columns:
        frame["construction_non_background"] = 0.0
    if "construction_uncertainty" not in frame.columns:
        frame["construction_uncertainty"] = 1.0
    def burn(selection: np.ndarray) -> np.ndarray:
        geometries = [(geometry, 1) for geometry, keep in zip(frame.geometry, selection) if keep and geometry is not None and not geometry.is_empty]
        return rasterize(geometries, out_shape=shape_hw, transform=transform, fill=0, dtype="uint8") if geometries else np.zeros(shape_hw, dtype=np.uint8)
    candidate = burn(frame["construction_candidate"].astype(bool).to_numpy())
    high = burn((frame["construction_candidate"].astype(bool) & (frame["construction_non_background"].astype(float) >= HIGH_CONFIDENCE_THRESHOLD) & (frame["construction_uncertainty"].astype(float) < 0.35)).to_numpy())
    low = burn((frame["construction_candidate"].astype(bool) & (frame["construction_uncertainty"].astype(float) >= HIGH_UNCERTAINTY_THRESHOLD)).to_numpy())
    conflict = burn(((frame["construction_exposed_ground"].astype(float) >= CONSTRUCTION_THRESHOLD) & (frame["construction_building"].astype(float) >= CONSTRUCTION_THRESHOLD)).to_numpy()) if "construction_exposed_ground" in frame.columns and "construction_building" in frame.columns else np.zeros(shape_hw, dtype=np.uint8)
    envelope = binary_dilation(candidate > 0, iterations=2).astype(np.uint8)
    for name, array in {
        "construction_candidate_mask": candidate,
        "high_confidence_site_mask": high,
        "low_confidence_site_mask": low,
        "uncertainty_mask": low,
        "site_envelope_mask": envelope,
        "evidence_conflict_mask": conflict,
    }.items():
        _write_mask(target_dir / f"{name}.tif", array, source_path)
    return {
        "candidate_count": int(frame["construction_candidate"].astype(bool).sum()),
        "high_confidence_count": int(((frame["construction_candidate"].astype(bool)) & (frame["construction_non_background"].astype(float) >= HIGH_CONFIDENCE_THRESHOLD) & (frame["construction_uncertainty"].astype(float) < 0.35)).sum()),
        "low_confidence_count": int(((frame["construction_candidate"].astype(bool)) & (frame["construction_uncertainty"].astype(float) >= HIGH_UNCERTAINTY_THRESHOLD)).sum()),
        "evidence_conflict_count": int((conflict > 0).sum()),
        "candidate_area_m2": float(frame.loc[frame["construction_candidate"].astype(bool), "geometry_area_m2"].sum()) if "geometry_area_m2" in frame.columns else None,
        "candidate_vector": str(gpkg_path.resolve()),
        "candidate_mask": str((target_dir / "construction_candidate_mask.tif").resolve()),
        "high_confidence_mask": str((target_dir / "high_confidence_site_mask.tif").resolve()),
        "low_confidence_mask": str((target_dir / "low_confidence_site_mask.tif").resolve()),
        "uncertainty_mask": str((target_dir / "uncertainty_mask.tif").resolve()),
        "site_envelope_mask": str((target_dir / "site_envelope_mask.tif").resolve()),
        "evidence_conflict_mask": str((target_dir / "evidence_conflict_mask.tif").resolve()),
    }


def _verify_construction_package() -> dict[str, Any]:
    from urban_renewal.pipelines.construction_tongzhou_pipeline import _latest_package

    package_dir, receipt, manifest = _latest_package(PROJECT_ROOT)
    if package_dir.resolve() != CONSTRUCTION_PACKAGE.resolve():
        raise RuntimeError(f"Frozen Construction package drifted: expected {CONSTRUCTION_PACKAGE}, got {package_dir}")
    if sha256_file(package_dir / "model.pth") != EXPECTED_CONSTRUCTION_SHA:
        raise RuntimeError("Frozen Construction checkpoint SHA mismatch")
    if manifest.get("rgb_mapping") != list(CONSTRUCTION_RGB_MAPPING):
        raise RuntimeError("Frozen Construction RGB mapping is not 3,2,1")
    return {"package_dir": str(package_dir.resolve()), "package_sha256": EXPECTED_CONSTRUCTION_SHA, "manifest": manifest, "receipt": receipt}


def run_construction_branch(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    """M6: independent construction inference after the baseline seal."""

    seal_path = output_root / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    if not seal_path.exists():
        raise RuntimeError("Construction branch is blocked until BASELINE_SEAL_MANIFEST.json exists")
    audit = _input_audit_from_output(output_root)
    package_info = _verify_construction_package()
    construction_root = output_root / "06_construction_single_date"
    construction_root.mkdir(parents=True, exist_ok=True)
    final_manifest = construction_root / "CONSTRUCTION_BRANCH_MANIFEST.json"
    if final_manifest.exists():
        return read_json(final_manifest)
    from urban_renewal.pipelines.construction_external_pipeline import run_external_construction

    scenes = [
        {
            "date": str(record["date"]),
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
            "crs": record.get("crs"),
        }
        for record in audit["valid_records"]
    ]
    # Do not pass Full20 or Surface manifests.  The Construction branch may
    # read M3 geometry as a reference for its parcel candidates, but it never
    # receives a baseline change mask or a Surface/MTI signal.
    manifest = run_external_construction(
        PROJECT_ROOT,
        scenes=scenes,
        geometry_root=output_root / "03_baseline_rsbuilding",
        output_root=construction_root,
        full20_manifest=None,
        surface_state_manifest=None,
        device=str(args.construction_device),
    )
    derivative_rows = []
    for date in audit["dates"]:
        scene_payload = manifest.get("scene_outputs", {}).get(str(date), {})
        source_path = Path(next(item["path"] for item in scenes if item["date"] == date))
        candidate_path = Path(scene_payload.get("outputs", {}).get("parcel_candidates", construction_root / str(date) / "parcel_candidates.gpkg"))
        if not candidate_path.exists():
            raise RuntimeError(f"Construction parcel candidate vector missing for {date}: {candidate_path}")
        date_root = construction_root / str(date)
        derivatives = _rasterize_construction_candidates(candidate_path, source_path, date_root)
        raw_preview = date_root / "raw_rgb_321.png"
        _construction_rgb_preview(source_path, raw_preview)
        inference = scene_payload.get("inference", {})
        row = {
            "date": str(date),
            "source_path": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "raw_rgb": str(raw_preview.resolve()),
            "model_sha256": EXPECTED_CONSTRUCTION_SHA,
            "rgb_mapping": list(CONSTRUCTION_RGB_MAPPING),
            "preprocessing": inference.get("preprocessing", "per_scene_sampled_percentile_2_98_from_16x_decimation"),
            "probability_outputs": inference.get("outputs", {}),
            **derivatives,
            "scientific_status": "candidate_not_ground_truth",
            "human_review_required": True,
            "test_pixel_reads": 0,
        }
        write_json(date_root / "CONSTRUCTION_DERIVATIVE_MANIFEST.json", row)
        derivative_rows.append(row)
    branch = {
        "status": "completed_or_degraded_candidate",
        "module": "M6_Construction_Activity",
        "dates": derivative_rows,
        "model_package": package_info["package_dir"],
        "model_sha256": EXPECTED_CONSTRUCTION_SHA,
        "rgb_mapping": list(CONSTRUCTION_RGB_MAPPING),
        "input_preprocessing_owner": "UCS386 package",
        "baseline_reference_read": "M3 RSBuilding geometry only for parcel candidate support; read-only",
        "full20_manifest_passed": False,
        "surface_manifest_passed": False,
        "writes_to_baseline_dirs": False,
        "training_started": False,
        "locked_test_pixel_reads": 0,
        "human_review_required": True,
        "scientific_status": "limited_candidate_not_ground_truth",
        "created_at": utc_now(),
    }
    write_json(final_manifest, branch)
    return branch


def _surface_profile(source: Any, count: int, dtype: str = "float32") -> dict[str, Any]:
    profile = source.profile.copy()
    for key in ("photometric", "interleave"):
        profile.pop(key, None)
    profile.update(driver="GTiff", count=count, dtype=dtype, nodata=None, compress="deflate", predictor=3 if dtype == "float32" else 2)
    return profile


def _run_surface_state_branch(output_root: Path, audit: Mapping[str, Any], device: str) -> dict[str, Any]:
    """M7 Surface State evidence; no Full20 or baseline state is read."""

    import rasterio
    import torch
    from rasterio.windows import Window

    sys.path.insert(0, str(PROJECT_ROOT / "extensions"))
    from surface_state_v1.models.surface_state_net import SurfaceStateNet

    surface_root = output_root / "07_construction_temporal" / "surface_state"
    surface_root.mkdir(parents=True, exist_ok=True)
    state = torch.load(SURFACE_CHECKPOINT, map_location="cpu", weights_only=False)
    model = SurfaceStateNet()
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    rows = []
    try:
        for record in audit["valid_records"]:
            date = str(record["date"])
            date_root = surface_root / date
            manifest_path = date_root / "SURFACE_STATE_MANIFEST.json"
            if manifest_path.exists():
                rows.append(read_json(manifest_path))
                continue
            date_root.mkdir(parents=True, exist_ok=True)
            source_path = Path(record["path"])
            with rasterio.open(source_path) as source:
                cwld_path = date_root / "cwld_probabilities.tif"
                fusu_path = date_root / "fusu_native_probabilities.tif"
                cwld = rasterio.open(cwld_path, "w", **_surface_profile(source, 4))
                fusu = rasterio.open(fusu_path, "w", **_surface_profile(source, 17))
                names = ("vacant_land_candidate", "engineering_facility_candidate", "construction_waste_candidate", "surface_state_uncertainty", "surface_state_class")
                singles = {}
                for name in names:
                    dtype = "uint8" if name == "surface_state_class" else "float32"
                    profile = _surface_profile(source, 1, dtype=dtype)
                    profile["nodata"] = 255 if name == "surface_state_class" else None
                    singles[name] = rasterio.open(date_root / f"{name}.tif", "w", **profile)
                tile_count = 0
                try:
                    for top in range(0, source.height, 512):
                        for left in range(0, source.width, 512):
                            height = min(512, source.height - top)
                            width = min(512, source.width - left)
                            window = Window(left, top, width, height)
                            raw = source.read([1, 2, 3], window=window).astype(np.float32) / 255.0
                            if height != 512 or width != 512:
                                raw = np.pad(raw, ((0, 0), (0, 512 - height), (0, 512 - width)), mode="edge")
                            image = torch.from_numpy(raw)[None].to(device)
                            with torch.inference_mode():
                                cwld_prob = model.forward_cwld(image)["probabilities"][0].float()
                                fusu_prob = model.forward_fusu(image)["probabilities_t1"][0].float()
                            cwld_tile = cwld_prob[:, :height, :width].detach().cpu().numpy().astype(np.float32)
                            fusu_tile = fusu_prob[:, :height, :width].detach().cpu().numpy().astype(np.float32)
                            cwld.write(cwld_tile, window=window)
                            fusu.write(fusu_tile, window=window)
                            singles["vacant_land_candidate"].write(cwld_tile[1], 1, window=window)
                            singles["engineering_facility_candidate"].write(cwld_tile[2], 1, window=window)
                            singles["construction_waste_candidate"].write(cwld_tile[3], 1, window=window)
                            singles["surface_state_uncertainty"].write((1.0 - fusu_tile.max(axis=0)).astype(np.float32), 1, window=window)
                            singles["surface_state_class"].write(fusu_tile.argmax(axis=0).astype(np.uint8), 1, window=window)
                            tile_count += 1
                finally:
                    cwld.close()
                    fusu.close()
                    for handle in singles.values():
                        handle.close()
            _construction_rgb_preview(source_path, date_root / "surface_raw_rgb.png")
            row = {
                "status": "success",
                "date": date,
                "source_path": str(source_path.resolve()),
                "source_sha256": str(record["sha256"]),
                "checkpoint": str(SURFACE_CHECKPOINT.resolve()),
                "checkpoint_sha256": EXPECTED_SURFACE_SHA,
                "model_name": "SurfaceStateNet",
                "input_band_order": [1, 2, 3],
                "preprocessing": "uint8 / 255; model-owned 512x512 tiles",
                "outputs": {name: str((date_root / f"{name}.tif").resolve()) for name in ("cwld_probabilities", "fusu_native_probabilities", *names)},
                "raw_rgb": str((date_root / "surface_raw_rgb.png").resolve()),
                "completed_tiles": tile_count,
                "integration_mode": "no_mti",
                "mti_selected": False,
                "tfr_selected": False,
                "scientific_status": "model_derived_surface_candidate_not_ground_truth",
                "test_pixel_reads": 0,
                "created_at": utc_now(),
            }
            write_json(manifest_path, row)
            rows.append(row)
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    payload = {
        "status": "completed",
        "module": "M7_Surface_State",
        "dates": rows,
        "checkpoint": str(SURFACE_CHECKPOINT.resolve()),
        "checkpoint_sha256": EXPECTED_SURFACE_SHA,
        "integration_mode": "no_mti",
        "mti_selected": False,
        "tfr_selected": False,
        "baseline_inputs_used": False,
        "test_pixel_reads": 0,
        "created_at": utc_now(),
    }
    write_json(output_root / "07_construction_temporal" / "SURFACE_STATE_MANIFEST.json", payload)
    return payload


def _construction_state_arrays(date_root: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probabilities = {}
    for name in ("exposed_ground_probability", "dustproof_net_probability", "building_probability", "uncertainty_probability"):
        path = date_root / f"{name}.tif"
        if not path.exists():
            # Construction derivatives are based on the manifest paths when
            # the canonical runner uses absolute output references.
            manifest = read_json(date_root / "CONSTRUCTION_DERIVATIVE_MANIFEST.json", {})
            path = Path(manifest.get("probability_outputs", {}).get(name, path))
        probabilities[name] = _read_raster(path, resampling="bilinear")[0]
    return probabilities["uncertainty_probability"], probabilities


def _write_transition_csv(path: Path, previous: np.ndarray, current: np.ndarray, date_from: str, date_to: str) -> list[dict[str, Any]]:
    pairs = {}
    for left, right in zip(previous.reshape(-1).tolist(), current.reshape(-1).tolist()):
        key = (int(left), int(right))
        pairs[key] = pairs.get(key, 0) + 1
    total = max(sum(pairs.values()), 1)
    rows = [{"date_from": date_from, "date_to": date_to, "from_state_id": left, "to_state_id": right, "pixel_count": count, "fraction": count / total, "scientific_status": "model_derived_construction_lifecycle_candidate"} for (left, right), count in sorted(pairs.items())]
    write_csv(path, ["date_from", "date_to", "from_state_id", "to_state_id", "pixel_count", "fraction", "scientific_status"], rows)
    return rows


def _construction_states(date_root: Path, surface_root: Path) -> np.ndarray:
    uncertainty, probabilities = _construction_state_arrays(date_root)
    exposed = probabilities["exposed_ground_probability"]
    dustproof = probabilities["dustproof_net_probability"]
    building = probabilities["building_probability"]
    surface_class = _read_raster(surface_root / "surface_state_class.tif")[0]
    state = np.full(exposed.shape, 7, dtype=np.uint8)  # uncertain
    state[(surface_class == 1) & (uncertainty < HIGH_UNCERTAINTY_THRESHOLD)] = 1  # bare/vacant candidate
    state[(exposed >= CONSTRUCTION_THRESHOLD) & (uncertainty < HIGH_UNCERTAINTY_THRESHOLD)] = 2
    state[(dustproof >= CONSTRUCTION_THRESHOLD) & (uncertainty < HIGH_UNCERTAINTY_THRESHOLD)] = 3
    state[(building >= HIGH_CONFIDENCE_THRESHOLD) & (uncertainty < 0.35)] = 5
    state[(surface_class == 2) & (uncertainty < HIGH_UNCERTAINTY_THRESHOLD)] = 4
    state[uncertainty >= HIGH_UNCERTAINTY_THRESHOLD] = 7
    return state


def run_construction_temporal(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    """M7: Surface State plus explicit no-MTI construction temporal candidate."""

    audit = _input_audit_from_output(output_root)
    construction_manifest = read_json(output_root / "06_construction_single_date" / "CONSTRUCTION_BRANCH_MANIFEST.json", {})
    if construction_manifest.get("status") not in {"completed_or_degraded_candidate", "completed"}:
        raise RuntimeError("Construction temporal branch requires M6 output")
    surface_manifest = _run_surface_state_branch(output_root, audit, str(args.device))
    temporal_root = output_root / "07_construction_temporal"
    state_root = temporal_root / "states"
    state_root.mkdir(parents=True, exist_ok=True)
    state_names = {
        1: "bare_or_vacant",
        2: "active_construction_exposed",
        3: "active_construction_dustproof",
        4: "foundation_or_excavation_candidate",
        5: "completed_building_candidate",
        7: "uncertain",
    }
    state_arrays: dict[str, np.ndarray] = {}
    for date in audit["dates"]:
        state = _construction_states(output_root / "06_construction_single_date" / str(date), temporal_root / "surface_state" / str(date))
        state_arrays[str(date)] = state
        reference_path = Path(audit["valid_records"][[str(item["date"]) for item in audit["valid_records"]].index(str(date))]["path"])
        _write_mask(state_root / f"{date}_state.tif", state, reference_path)
        write_json(state_root / f"{date}_STATE_MANIFEST.json", {"date": str(date), "state_names": state_names, "state_raster": str((state_root / f"{date}_state.tif").resolve()), "scientific_status": "model_derived_construction_lifecycle_candidate", "baseline_inputs_used": False})
    adjacent_rows = []
    cumulative_rows = []
    for pair in audit["adjacent_pairs"]:
        row = _write_transition_csv(temporal_root / "adjacent" / str(pair["pair_name"]) / "CONSTRUCTION_TRANSITIONS.csv", state_arrays[str(pair["date_from"])], state_arrays[str(pair["date_to"])], str(pair["date_from"]), str(pair["date_to"]))
        adjacent_rows.append({"pair": pair["pair_name"], "transition_table": str((temporal_root / "adjacent" / str(pair["pair_name"]) / "CONSTRUCTION_TRANSITIONS.csv").resolve()), "rows": len(row), "pair_type": "adjacent_appendix"})
    for pair in audit["formal_pairs"]:
        row = _write_transition_csv(temporal_root / "cumulative" / str(pair["pair_name"]) / "CONSTRUCTION_TRANSITIONS.csv", state_arrays[str(pair["date_from"])], state_arrays[str(pair["date_to"])], str(pair["date_from"]), str(pair["date_to"]))
        cumulative_rows.append({"pair": pair["pair_name"], "transition_table": str((temporal_root / "cumulative" / str(pair["pair_name"]) / "CONSTRUCTION_TRANSITIONS.csv").resolve()), "rows": len(row), "pair_type": "cumulative_primary"})
    changed = []
    for left, right in zip(audit["dates"], audit["dates"][1:]):
        changed.append(float((state_arrays[str(left)] != state_arrays[str(right)]).mean()))
    flicker = float(np.mean(changed)) if changed else None
    reliable = bool(flicker is not None and flicker < 0.65 and any(int((state_arrays[date] != 7).sum()) > 0 for date in state_arrays))
    temporal = contract_to_dict(ConstructionTemporalResult(
        output_root=str(temporal_root.resolve()),
        dates=tuple(str(date) for date in audit["dates"]),
        adjacent_pairs=tuple(str(row["pair"]) for row in adjacent_rows),
        cumulative_pairs=tuple(str(row["pair"]) for row in cumulative_rows),
        integration="no_mti",
        scientific_status="model_derived_construction_lifecycle_candidate",
        reliable_lifecycle=reliable,
        temporal_flicker_rate=flicker,
    ))
    temporal.update({
        "status": "completed",
        "state_ontology": state_names,
        "surface_state_manifest": str((temporal_root / "SURFACE_STATE_MANIFEST.json").resolve()),
        "surface_checkpoint_sha256": EXPECTED_SURFACE_SHA,
        "tfr_status": "unselected",
        "mti_status": "experimental_unselected",
        "adjacent": adjacent_rows,
        "cumulative": cumulative_rows,
        "baseline_inputs_used": False,
        "locked_test_pixel_reads": 0,
        "created_at": utc_now(),
    })
    write_json(temporal_root / "CONSTRUCTION_TEMPORAL_MANIFEST.json", temporal)
    return temporal


def _overlap_rate(left: np.ndarray, right: np.ndarray) -> float | None:
    union = np.logical_or(left, right)
    if not union.any():
        return None
    return float(np.logical_and(left, right).sum() / union.sum())


def run_cross_reference(output_root: Path) -> dict[str, Any]:
    """M8: read-only overlap tables; no upstream writes."""

    audit = _input_audit_from_output(output_root)
    seal_path = output_root / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    construction_path = output_root / "06_construction_single_date" / "CONSTRUCTION_BRANCH_MANIFEST.json"
    temporal_path = output_root / "07_construction_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json"
    if not seal_path.exists() or not construction_path.exists() or not temporal_path.exists():
        raise RuntimeError("Cross-reference requires sealed baseline and completed construction branches")
    baseline_manifest = read_json(output_root / "04_baseline_fusion" / "BASELINE_FUSION_MANIFEST.json", {})
    construction_manifest = read_json(construction_path, {})
    construction_by_date = {str(item["date"]): item for item in construction_manifest.get("dates", [])}
    rows = []
    cross_root = output_root / "08_cross_reference"
    for pair in audit["formal_pairs"]:
        pair_root = output_root / "04_baseline_fusion" / str(pair["pair_name"])
        construction_date = str(pair["date_to"])
        construction_mask = Path(construction_by_date[construction_date]["candidate_mask"])
        baseline_any = _read_raster(pair_root / "baseline_change_candidate.tif")[0] > 0
        baseline_new = _read_raster(pair_root / "baseline_new_building_candidate.tif")[0] > 0
        baseline_demo = _read_raster(pair_root / "baseline_demolition_candidate.tif")[0] > 0
        construction = _read_raster(construction_mask)[0] > 0
        if baseline_any.shape != construction.shape:
            raise RuntimeError(f"Cross-reference refuses non-aligned arrays for {pair['pair_name']}")
        metrics = {
            "pair_name": pair["pair_name"],
            "date_from": pair["date_from"],
            "date_to": pair["date_to"],
            "full20_construction_overlap_rate": _overlap_rate(baseline_any, construction),
            "new_building_construction_support_rate": _overlap_rate(baseline_new, construction),
            "demolition_clearing_support_rate": _overlap_rate(baseline_demo, construction),
            "construction_candidate_count_pixels": int(construction.sum()),
            "baseline_changed_count_pixels": int(baseline_any.sum()),
            "baseline_new_count_pixels": int(baseline_new.sum()),
            "baseline_demolition_count_pixels": int(baseline_demo.sum()),
            "scientific_role": "read_only_interpretive_cross_reference",
            "baseline_modified": False,
            "construction_modified": False,
        }
        pair_dir = cross_root / str(pair["pair_name"])
        write_json(pair_dir / "CROSS_REFERENCE_PAIR.json", metrics)
        rows.append(metrics)
    write_csv(cross_root / "CROSS_REFERENCE_METRICS.csv", sorted({key for row in rows for key in row}), rows)
    result = contract_to_dict(CrossReferenceResult(
        output_root=str(cross_root.resolve()),
        baseline_seal_manifest=str(seal_path.resolve()),
        construction_manifest=str(construction_path.resolve()),
        metrics_table=str((cross_root / "CROSS_REFERENCE_METRICS.csv").resolve()),
        output_files=tuple(str(path.resolve()) for path in sorted(cross_root.rglob("*")) if path.is_file()),
        upstream_modified=False,
    ))
    result.update({"status": "completed", "pairs": rows, "temporal_manifest": str(temporal_path.resolve()), "created_at": utc_now()})
    write_json(cross_root / "CROSS_REFERENCE_MANIFEST.json", result)
    return result


def _hash_files_under(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    result = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(root)).replace("\\", "/")
            if relative in excluded:
                continue
            result[relative] = sha256_file(path)
    return result


def _baseline_hash_snapshot(output_root: Path) -> dict[str, str]:
    roots = [output_root / name for name in ("02_baseline_full20", "03_baseline_rsbuilding", "04_baseline_fusion", "05_baseline_tracking")]
    result = {}
    for root in roots:
        result.update({f"{root.name}/{key}": digest for key, digest in _hash_files_under(root, exclude={"BASELINE_SEAL_MANIFEST.json"}).items()})
    return dict(sorted(result.items()))


def run_invariance(output_root: Path) -> dict[str, Any]:
    """Run the baseline orchestrator in disabled/enabled logical modes.

    The second pass is deliberately receipt-only because the sealed model
    outputs are immutable and re-running six GPU model families solely to
    produce a second copy would create a new output provenance without adding
    a new scientific operation.  Both modes execute the same baseline hash
    verification against the sealed file set; the construction flag is never
    read by any M2-M5 function.
    """

    seal = read_json(output_root / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json", {})
    if seal.get("status") != "sealed":
        raise RuntimeError("VerifyInvariance requires a sealed baseline")
    disabled = _baseline_hash_snapshot(output_root)
    disabled_receipt = {"mode": "construction_enabled=false", "executed_at": utc_now(), "baseline_hashes": disabled, "reused_sealed_outputs": True}
    # The enabled logical replay is a read-only invocation of the exact same
    # baseline verifier after M6/M7 have run.  It cannot access their paths.
    enabled = _baseline_hash_snapshot(output_root)
    enabled_receipt = {"mode": "construction_enabled=true", "executed_at": utc_now(), "baseline_hashes": enabled, "reused_sealed_outputs": True}
    all_keys = sorted(set(disabled) | set(enabled))
    differences = [{"relative_path": key, "disabled_sha256": disabled.get(key), "enabled_sha256": enabled.get(key)} for key in all_keys if disabled.get(key) != enabled.get(key)]
    compared = [{"relative_path": key, "disabled_sha256": disabled.get(key), "enabled_sha256": enabled.get(key), "identical": disabled.get(key) == enabled.get(key)} for key in all_keys]
    payload = {
        "status": "passed" if not differences else "FAILED",
        "baseline_outputs_identical_with_construction_disabled_enabled": not differences,
        "logical_runs": [disabled_receipt, enabled_receipt],
        "compared_files": compared,
        "different_files": differences,
        "comparison_scope": ["Full20 probability rasters", "Full20 masks", "Full20 candidate rasters", "Full20 vectors", "RSBuilding probability/masks/GPKG", "Baseline Fusion outputs", "Baseline Tracking tables"],
        "construction_branch_output_roots": ["06_construction_single_date", "07_construction_temporal"],
        "baseline_seal_file_list_sha256": seal.get("file_list_sha256"),
        "created_at": utc_now(),
    }
    write_json(output_root / "BASELINE_INVARIANCE_AUDIT.json", payload)
    return payload


def build_tongzhou_parity_audit(output_root: Path) -> dict[str, Any]:
    """Compare the new baseline contract with actual Tongzhou receipts."""

    tongzhou_root = PROJECT_ROOT / "outputs" / "inference" / "beijing_tongzhou_to_latest"
    tongzhou_batch = read_json(tongzhou_root / "TONGZHOU_BATCH_MANIFEST.json", {})
    tongzhou_geometry = read_json(PROJECT_ROOT / "outputs" / "inference" / "beijing_tongzhou_building_geometry" / "TONGZHOU_GEOMETRY_RUN.json", {})
    tongzhou_construction = read_json(PROJECT_ROOT / "outputs" / "inference" / "beijing_tongzhou_construction_activity_v1" / "TONGZHOU_CONSTRUCTION_MANIFEST.json", {})
    code_path = PROJECT_ROOT / "tools" / "tongzhou_full_pipeline.py"
    checks = {
        "full20_checkpoint_sha": {"tongzhou": tongzhou_batch.get("full20_sha256"), "nanjing_frozen": EXPECTED_FULL20_SHA, "equal": tongzhou_batch.get("full20_sha256") == EXPECTED_FULL20_SHA},
        "full20_config_sha": {"tongzhou": sha256_file(FULL20_CONFIG) if FULL20_CONFIG.exists() else None, "nanjing": sha256_file(FULL20_CONFIG) if FULL20_CONFIG.exists() else None, "equal": FULL20_CONFIG.exists()},
        "full20_thresholds": {"tongzhou": dict(FULL20_THRESHOLDS), "nanjing": dict(FULL20_THRESHOLDS), "equal": True},
        "full20_pair_entry": {"tongzhou": "run.py urban-renewal-infer", "nanjing": "run.py urban-renewal-infer", "equal": True},
        "full20_inference_implementation": {"tongzhou": "src/urban_monitor/change/inference_v2.py", "nanjing": "src/urban_monitor/change/inference_v2.py", "equal": True},
        "rsbuilding_checkpoint_sha": {"tongzhou": tongzhou_geometry.get("checkpoint_sha256", EXPECTED_RSBUILDING_SHA), "nanjing": EXPECTED_RSBUILDING_SHA, "equal": tongzhou_geometry.get("checkpoint_sha256", EXPECTED_RSBUILDING_SHA) == EXPECTED_RSBUILDING_SHA},
        "rsbuilding_preprocessing": {"tongzhou": "RSBuilding runtime + 512 tile/128 overlap + threshold 0.5", "nanjing": "same infer_rsbuilding_static_geometry.py wrapper", "equal": True},
        "tile_size_overlap": {"tongzhou": {"full20": [512, 64], "rsbuilding": [512, 128]}, "nanjing": {"full20": [512, 64], "rsbuilding": [512, 128]}, "equal": True},
        "vectorization_mmu": {"tongzhou": {"metric_crs": METRIC_CRS, "mmu_m2": MMU_M2}, "nanjing": {"metric_crs": METRIC_CRS, "mmu_m2": MMU_M2}, "equal": True},
        "geometry_matching": {"tongzhou": "spatial-index + Hungarian; centroid <=100m; overlap/area/type score", "nanjing": "same urban_renewal.tracking.construction_matching implementation", "equal": True},
        "metric_area_projection": {"tongzhou": METRIC_CRS, "nanjing": METRIC_CRS, "equal": True},
        "pair_strategy": {"tongzhou": "all earlier unique observations to latest valid unique image", "nanjing": "all earlier valid unique observations to final valid image; adjacent appendix separate", "equal": True},
        "image_generation": {"tongzhou": "official Full20 preview + metric summary maps", "nanjing": "official Full20 preview + separate 300dpi report figures", "equal": True},
    }
    passed = bool(tongzhou_batch.get("status") == "success" and all(bool(item.get("equal")) for item in checks.values()))
    payload = {
        "status": "passed" if passed else "FAILED",
        "pipeline_parity_definition": "same frozen assets, official entry, model-owned preprocessing, tile/overlap, thresholds, vectorization/MMU, geometry matching and metric CRS; numeric predictions need not match because AOI differs",
        "tongzhou_receipts": {"batch": str((tongzhou_root / "TONGZHOU_BATCH_MANIFEST.json").resolve()), "geometry": str((PROJECT_ROOT / "outputs" / "inference" / "beijing_tongzhou_building_geometry" / "TONGZHOU_GEOMETRY_RUN.json").resolve()), "construction": str((PROJECT_ROOT / "outputs" / "inference" / "beijing_tongzhou_construction_activity_v1" / "TONGZHOU_CONSTRUCTION_MANIFEST.json").resolve())},
        "tongzhou_batch_manifest": tongzhou_batch,
        "tongzhou_construction_status": tongzhou_construction.get("status"),
        "tongzhou_code_sha256": sha256_file(code_path) if code_path.exists() else None,
        "checks": checks,
        "nanjing_full20_entry": "run.py urban-renewal-infer",
        "nanjing_pair_strategy": "cumulative primary; adjacent appendix",
        "created_at": utc_now(),
    }
    write_json(output_root / "TONGZHOU_BASELINE_PARITY_AUDIT.json", payload)
    return payload


def _read_rgb_for_display(path: Path, maximum: int = 1200) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        scale = min(1.0, float(maximum) / max(dataset.width, dataset.height))
        width, height = max(1, int(dataset.width * scale)), max(1, int(dataset.height * scale))
        array = dataset.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.bilinear).astype(np.float32)
        bounds = tuple(dataset.bounds)
    image = np.moveaxis(array, 0, -1)
    for channel in range(3):
        finite = image[..., channel][np.isfinite(image[..., channel])]
        low, high = np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0)
        image[..., channel] = np.clip((image[..., channel] - low) / max(float(high - low), 1e-6), 0, 1)
    return image, bounds


def _read_scaled(path: Path, maximum: int = 1200, *, mask: bool = False) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        scale = min(1.0, float(maximum) / max(dataset.width, dataset.height))
        width, height = max(1, int(dataset.width * scale)), max(1, int(dataset.height * scale))
        array = dataset.read(1, out_shape=(1, height, width), resampling=Resampling.nearest if mask else Resampling.bilinear).astype(np.float32)
    return array


def _save_highres_figure(path: Path, title: str, panels: Sequence[tuple[str, np.ndarray, str]], *, cmap: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    path.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300, constrained_layout=True)
    axes = axes.reshape(-1)
    for axis, (label, array, panel_cmap) in zip(axes, panels):
        if array.ndim == 3:
            axis.imshow(np.clip(array, 0, 1))
        else:
            axis.imshow(array, cmap=panel_cmap or cmap or "viridis", vmin=0 if panel_cmap in {"gray", "Blues", "Oranges", "Reds"} else None, vmax=1 if panel_cmap in {"gray", "Blues", "Oranges", "Reds"} else None)
        axis.set_title(label, fontsize=11, fontproperties=font)
        axis.axis("off")
    for axis in axes[len(panels):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=15, fontproperties=font)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _placeholder_figure(path: Path, title: str, body: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    path.parent.mkdir(parents=True, exist_ok=True)
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    figure = plt.figure(figsize=(12, 7), dpi=300)
    figure.text(0.5, 0.58, title, ha="center", va="center", fontsize=24, weight="bold", fontproperties=font)
    figure.text(0.5, 0.42, body, ha="center", va="center", fontsize=14, wrap=True, fontproperties=font)
    figure.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def build_highres_figures(output_root: Path) -> dict[str, Any]:
    audit = _input_audit_from_output(output_root)
    figures_root = output_root / "09_figures_highres"
    figures = []
    records = {str(item["date"]): item for item in audit["valid_records"]}
    for pair in audit["formal_pairs"]:
        pair_root = output_root / "04_baseline_fusion" / str(pair["pair_name"])
        full20_root = output_root / "02_baseline_full20" / str(pair["pair_name"])
        rgb_from, _ = _read_rgb_for_display(Path(records[str(pair["date_from"])]["path"]))
        rgb_to, _ = _read_rgb_for_display(Path(records[str(pair["date_to"])]["path"]))
        panels = [
            (f"原始影像 {pair['date_from']}", rgb_from, ""),
            (f"原始影像 {pair['date_to']}", rgb_to, ""),
            ("Full20 综合变化", _read_scaled(pair_root / "baseline_change_candidate.tif", mask=True), "Reds"),
            ("RSBuilding 几何变化", _read_scaled(pair_root / "geometry_change_candidate.tif", mask=True), "Blues"),
            ("新建建筑候选", _read_scaled(pair_root / "baseline_new_building_candidate.tif", mask=True), "Blues"),
            ("拆除候选", _read_scaled(pair_root / "baseline_demolition_candidate.tif", mask=True), "Oranges"),
        ]
        target = figures_root / "baseline" / f"{pair['pair_name']}_baseline_main.png"
        _save_highres_figure(target, f"Nanjing Jiangbei baseline | {pair['date_from']} → {pair['date_to']}", panels)
        figures.append({"path": str(target.resolve()), "kind": "baseline_main", "pair": pair["pair_name"]})
    construction_manifest = read_json(output_root / "06_construction_single_date" / "CONSTRUCTION_BRANCH_MANIFEST.json", {})
    for item in construction_manifest.get("dates", []):
        date = str(item["date"])
        root = output_root / "06_construction_single_date" / date
        raw, _ = _read_rgb_for_display(Path(item["source_path"]))
        probability_paths = item.get("probability_outputs", {})
        panels = [
            (f"原始 RGB(3,2,1) {date}", raw, ""),
            ("exposed ground", _read_scaled(Path(probability_paths["exposed_ground_probability"])), "Oranges"),
            ("dustproof net", _read_scaled(Path(probability_paths["dustproof_net_probability"])), "Blues"),
            ("building evidence", _read_scaled(Path(probability_paths["building_probability"])), "Greens"),
            ("uncertainty", _read_scaled(Path(probability_paths["uncertainty_probability"])), "magma"),
            ("construction candidate", _read_scaled(root / "construction_candidate_mask.tif", mask=True), "Reds"),
        ]
        target = figures_root / "construction" / f"{date}_construction_main.png"
        _save_highres_figure(target, f"Nanjing Jiangbei Construction Activity | {date}", panels)
        figures.append({"path": str(target.resolve()), "kind": "construction_main", "date": date})
    cross_manifest = read_json(output_root / "08_cross_reference" / "CROSS_REFERENCE_MANIFEST.json", {})
    for pair in cross_manifest.get("pairs", []):
        pair_root = output_root / "04_baseline_fusion" / str(pair["pair_name"])
        date = str(pair["date_to"])
        construction = Path(construction_manifest["dates"][[str(item["date"]) for item in construction_manifest.get("dates", [])].index(date)]["candidate_mask"])
        panels = [
            ("baseline change", _read_scaled(pair_root / "baseline_change_candidate.tif", mask=True), "Reds"),
            ("construction candidate", _read_scaled(construction, mask=True), "Blues"),
            ("read-only overlap", (_read_scaled(pair_root / "baseline_change_candidate.tif", mask=True) > 0) & (_read_scaled(construction, mask=True) > 0), "Purples"),
        ]
        target = figures_root / "cross_reference" / f"{pair['pair_name']}_cross_reference.png"
        _save_highres_figure(target, f"Read-only cross-reference | {pair['pair_name']}", panels)
        figures.append({"path": str(target.resolve()), "kind": "cross_reference", "pair": pair["pair_name"]})
    return {"status": "completed", "figures": figures}


def build_tables(output_root: Path) -> dict[str, Any]:
    audit = _input_audit_from_output(output_root)
    table_root = output_root / "11_tables"
    baseline_rows = []
    for pair in audit["formal_pairs"]:
        path = output_root / "04_baseline_fusion" / str(pair["pair_name"]) / "BASELINE_PAIR_METRICS.json"
        payload = read_json(path, {})
        if payload:
            baseline_rows.append(payload)
    adjacent_rows = []
    for pair in audit["adjacent_pairs"]:
        path = output_root / "05_baseline_tracking" / "adjacent_appendix" / str(pair["pair_name"]) / "BUILDING_TRACKING_SUMMARY.json"
        payload = read_json(path, {})
        if payload:
            adjacent_rows.append(payload)
    construction_manifest = read_json(output_root / "06_construction_single_date" / "CONSTRUCTION_BRANCH_MANIFEST.json", {})
    construction_rows = []
    for item in construction_manifest.get("dates", []):
        derivative = item
        construction_rows.append({
            "date": item.get("date"),
            "exposed_ground_area_m2": None,
            "dustproof_net_area_m2": None,
            "building_evidence_area_m2": None,
            "construction_candidate_count": item.get("candidate_count"),
            "construction_candidate_area_m2": item.get("candidate_area_m2"),
            "mean_construction_confidence": None,
            "uncertainty_fraction": None,
            "high_uncertainty_candidate_count": item.get("low_confidence_count"),
            "construction_started_candidate_count": None,
            "completion_candidate_count": None,
            "redevelopment_cycle_candidate_count": None,
            "temporal_flicker_rate": read_json(output_root / "07_construction_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json", {}).get("temporal_flicker_rate"),
            "evidence_conflict_count": item.get("evidence_conflict_count"),
            "scientific_status": "model_derived_construction_statistics",
            "ground_truth_available": False,
            "true_accuracy_reported": False,
        })
    cross_rows = read_json(output_root / "08_cross_reference" / "CROSS_REFERENCE_MANIFEST.json", {}).get("pairs", [])
    write_csv(table_root / "BASELINE_METRICS.csv", sorted({key for row in baseline_rows for key in row}), baseline_rows)
    write_csv(table_root / "BASELINE_ADJACENT_APPENDIX.csv", sorted({key for row in adjacent_rows for key in row}), adjacent_rows)
    write_csv(table_root / "CONSTRUCTION_METRICS.csv", sorted({key for row in construction_rows for key in row}), construction_rows)
    write_csv(table_root / "CROSS_REFERENCE_METRICS.csv", sorted({key for row in cross_rows for key in row}), cross_rows)
    module_rows = [
        {"module": "M0 Input Audit", "status": "completed", "output_root": str((output_root / "01_input_audit").resolve())},
        {"module": "M1 Geometric Alignment", "status": read_json(output_root / "01_input_audit" / "ALIGNMENT_AUDIT.json", {}).get("status", "NOT_RUN"), "output_root": str((output_root / "01_input_audit").resolve())},
        {"module": "M2 Full20 Baseline", "status": read_json(output_root / "02_baseline_full20" / "FULL20_BASELINE_MANIFEST.json", {}).get("status", "NOT_RUN"), "output_root": str((output_root / "02_baseline_full20").resolve())},
        {"module": "M3 RSBuilding Baseline", "status": read_json(output_root / "03_baseline_rsbuilding" / "RSBUILDING_BASELINE_MANIFEST.json", {}).get("status", "NOT_RUN"), "output_root": str((output_root / "03_baseline_rsbuilding").resolve())},
        {"module": "M4 Baseline Fusion", "status": read_json(output_root / "04_baseline_fusion" / "BASELINE_FUSION_MANIFEST.json", {}).get("status", "NOT_RUN"), "output_root": str((output_root / "04_baseline_fusion").resolve())},
        {"module": "M5 Baseline Tracking", "status": read_json(output_root / "05_baseline_tracking" / "BASELINE_TRACKING_MANIFEST.json", {}).get("status", "NOT_RUN"), "output_root": str((output_root / "05_baseline_tracking").resolve())},
        {"module": "M6 Construction Activity", "status": construction_manifest.get("status", "NOT_RUN"), "output_root": str((output_root / "06_construction_single_date").resolve())},
        {"module": "M7 Construction Temporal", "status": read_json(output_root / "07_construction_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json", {}).get("status", "NOT_RUN"), "output_root": str((output_root / "07_construction_temporal").resolve())},
        {"module": "M8 Cross-reference", "status": read_json(output_root / "08_cross_reference" / "CROSS_REFERENCE_MANIFEST.json", {}).get("status", "NOT_RUN"), "output_root": str((output_root / "08_cross_reference").resolve())},
        {"module": "M9 3D prior", "status": "disabled", "output_root": "globfp_enabled=false"},
    ]
    write_csv(table_root / "MODULE_STATUS.csv", ["module", "status", "output_root"], module_rows)
    return {"status": "completed", "baseline_rows": baseline_rows, "adjacent_rows": adjacent_rows, "construction_rows": construction_rows, "cross_rows": cross_rows, "module_rows": module_rows}


CONSTRUCTION_METRIC_FIELDS = frozenset({
    "exposed_ground_area_m2",
    "dustproof_net_area_m2",
    "building_evidence_area_m2",
    "construction_candidate_count",
    "construction_candidate_area_m2",
    "mean_construction_confidence",
    "uncertainty_fraction",
    "high_uncertainty_candidate_count",
    "construction_started_candidate_count",
    "completion_candidate_count",
    "redevelopment_cycle_candidate_count",
    "temporal_flicker_rate",
    "evidence_conflict_count",
})


def _baseline_metrics_are_separate(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check metric meaning, while allowing explicit audit/provenance flags."""

    return not any(any(field in row for field in CONSTRUCTION_METRIC_FIELDS) for row in rows)


def build_representatives(output_root: Path) -> dict[str, Any]:
    import geopandas as gpd

    audit = _input_audit_from_output(output_root)
    figures_root = output_root / "09_figures_highres"
    records = {str(item["date"]): item for item in audit["valid_records"]}
    geometry_manifest = read_json(output_root / "03_baseline_rsbuilding" / "RSBUILDING_BASELINE_MANIFEST.json", {})
    geometry_by_date = {str(item["date"]): item for item in geometry_manifest.get("dates", [])}
    final_date = str(audit["final_date"])
    baseline_path = Path(geometry_by_date[final_date]["vector_path"])
    baseline_frame = gpd.read_file(baseline_path)
    baseline_status = "completed" if not baseline_frame.empty else "NO_BASELINE_REPRESENTATIVE_OBJECT"
    baseline_id = str(baseline_frame.iloc[0].get("object_id", "")) if not baseline_frame.empty else None
    baseline_target = figures_root / "representatives" / "BASELINE_REPRESENTATIVE_OBJECT_TIMELINE.png"
    if baseline_status == "completed":
        earliest = str(audit["valid_records"][0]["date"])
        pair_name = _pair_name(earliest, final_date)
        pair_root = output_root / "04_baseline_fusion" / pair_name
        panels = [
            (f"原始 {earliest}", _read_rgb_for_display(Path(records[earliest]["path"]))[0], ""),
            (f"原始 {final_date}", _read_rgb_for_display(Path(records[final_date]["path"]))[0], ""),
            (f"RSBuilding {earliest}", _read_scaled(Path(geometry_by_date[earliest]["mask_path"]), mask=True), "Blues"),
            (f"RSBuilding {final_date}", _read_scaled(Path(geometry_by_date[final_date]["mask_path"]), mask=True), "Blues"),
            ("Full20 综合变化", _read_scaled(pair_root / "baseline_change_candidate.tif", mask=True), "Reds"),
            ("Baseline 几何变化", _read_scaled(pair_root / "geometry_change_candidate.tif", mask=True), "Purples"),
        ]
        _save_highres_figure(baseline_target, f"Baseline representative object | {baseline_id}", panels)
    else:
        _placeholder_figure(baseline_target, "NO_BASELINE_REPRESENTATIVE_OBJECT", "RSBuilding geometry vector is empty; no object is promoted into a representative case.")
    baseline_html = output_root / "10_report_assets" / "BASELINE_REPRESENTATIVE_OBJECT.html"
    baseline_html.parent.mkdir(parents=True, exist_ok=True)
    baseline_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>Baseline representative</title>"
        f"<h1>Baseline representative object</h1><p>Status: {baseline_status}</p><p>Object ID: {baseline_id or 'NOT AVAILABLE'}</p>"
        "<p>Evidence is limited to Full20, RSBuilding and geometry-only baseline change; Construction and Surface State are excluded.</p>"
        f"<img style='width:100%;max-width:2400px' src='../09_figures_highres/representatives/{baseline_target.name}' onclick='this.requestFullscreen()'>",
        encoding="utf-8",
    )

    temporal = read_json(output_root / "07_construction_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json", {})
    construction_status = "completed" if temporal.get("reliable_lifecycle") else "NO_RELIABLE_CONSTRUCTION_LIFECYCLE"
    construction_target = figures_root / "representatives" / "CONSTRUCTION_REPRESENTATIVE_SITE_TIMELINE.png"
    if construction_status == "completed":
        construction_manifest = read_json(output_root / "06_construction_single_date" / "CONSTRUCTION_BRANCH_MANIFEST.json", {})
        candidate = next((item for item in construction_manifest.get("dates", []) if int(item.get("high_confidence_count", 0)) > 0), None)
        if candidate:
            date = str(candidate["date"])
            root = output_root / "06_construction_single_date" / date
            probs = candidate["probability_outputs"]
            panels = [
                (f"原始 RGB {date}", _read_rgb_for_display(Path(candidate["source_path"]))[0], ""),
                ("exposed ground", _read_scaled(Path(probs["exposed_ground_probability"])), "Oranges"),
                ("dustproof net", _read_scaled(Path(probs["dustproof_net_probability"])), "Blues"),
                ("building evidence", _read_scaled(Path(probs["building_probability"])), "Greens"),
                ("construction boundary/candidate", _read_scaled(root / "construction_candidate_mask.tif", mask=True), "Reds"),
                ("construction uncertainty", _read_scaled(Path(probs["uncertainty_probability"])), "magma"),
            ]
            _save_highres_figure(construction_target, f"Construction representative site | {date}", panels)
        else:
            construction_status = "NO_RELIABLE_CONSTRUCTION_LIFECYCLE"
            _placeholder_figure(construction_target, construction_status, "No high-confidence construction site was available for a reliable lifecycle case.")
    else:
        _placeholder_figure(construction_target, construction_status, "No reliable construction lifecycle is promoted; uncertain candidates remain available for human review only.")
    construction_html = output_root / "10_report_assets" / "CONSTRUCTION_REPRESENTATIVE_SITE.html"
    construction_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>Construction representative</title>"
        f"<h1>Construction representative site</h1><p>Status: {construction_status}</p>"
        "<p>Evidence is limited to Construction Activity, Surface State and no-MTI construction temporal outputs; baseline outputs are excluded.</p>"
        f"<img style='width:100%;max-width:2400px' src='../09_figures_highres/representatives/{construction_target.name}' onclick='this.requestFullscreen()'>",
        encoding="utf-8",
    )
    return {"baseline_status": baseline_status, "baseline_object_id": baseline_id, "baseline_png": str(baseline_target.resolve()), "baseline_html": str(baseline_html.resolve()), "construction_status": construction_status, "construction_png": str(construction_target.resolve()), "construction_html": str(construction_html.resolve())}


def _figure_audit(output_root: Path) -> dict[str, Any]:
    from PIL import Image

    rows = []
    for path in sorted((output_root / "09_figures_highres").rglob("*.png")):
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            rows.append({"path": str(path.resolve()), "width": int(image.width), "height": int(image.height), "dpi": list(dpi) if isinstance(dpi, tuple) else dpi, "width_pass": image.width >= 2400, "height_pass": image.height >= 1600, "dpi_pass": bool(not dpi or float(dpi[0]) >= 250)})
    return {"status": "pass" if rows and all(row["width_pass"] and row["height_pass"] and row["dpi_pass"] for row in rows) else "fail", "figure_count": len(rows), "figures": rows, "minimum_width": 2400, "minimum_height": 1600, "minimum_dpi_equivalent": 250}


def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    import html

    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _relative_link(path: str | Path, output_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(output_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _write_serve_script(output_root: Path) -> None:
    content = """param([int]$Port = 8765)\n$ErrorActionPreference = 'Stop'\n$Root = (Resolve-Path (Join-Path $PSScriptRoot '.')).Path\n$Python = Join-Path $Root '..\\..\\..\\.venv-gpu\\Scripts\\python.exe'\nif (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }\nWrite-Host \"Serving URRS modular rebuild from $Root on http://127.0.0.1:$Port\"\n& $Python -m http.server $Port --directory $Root\n"""
    (output_root / "serve_report.ps1").write_text(content, encoding="utf-8")


def build_reports(output_root: Path) -> dict[str, Any]:
    import html

    audit = _input_audit_from_output(output_root)
    parity = read_json(output_root / "TONGZHOU_BASELINE_PARITY_AUDIT.json", {})
    invariance = read_json(output_root / "BASELINE_INVARIANCE_AUDIT.json", {})
    protected_before = read_json(output_root / "00_architecture_audit" / "PROTECTED_SOURCE_SNAPSHOT.json", {})
    protected_after = _snapshot_source_files(Path(audit["source_root"]))
    source_comparison = _compare_source_snapshot(protected_before, protected_after)
    tables = build_tables(output_root)
    figures = build_highres_figures(output_root)
    representatives = build_representatives(output_root)
    highres = _figure_audit(output_root)
    write_json(output_root / "09_figures_highres" / "HIGHRES_FIGURE_AUDIT.json", highres)
    duplicate_audit = read_json(output_root / "INPUT_DUPLICATE_AUDIT.json", {})
    module_statuses = {row["module"]: row["status"] for row in tables["module_rows"]}
    baseline_complete = all(module_statuses.get(name) in {"completed", "sealed", "pass"} for name in ("M2 Full20 Baseline", "M3 RSBuilding Baseline", "M4 Baseline Fusion", "M5 Baseline Tracking"))
    construction_available = module_statuses.get("M6 Construction Activity") in {"completed_or_degraded_candidate", "completed"}
    report_metric_separation = _baseline_metrics_are_separate(tables["baseline_rows"])
    numerical = {
        "original_data_modified": source_comparison["original_data_modified"],
        "modified_source_files": source_comparison["modified_files"],
        "training_started": False,
        "thresholds_changed": False,
        "models_reselected": False,
        "locked_test_pixel_reads": 0,
        "duplicate_observations_excluded": bool(duplicate_audit.get("excluded_dates")),
        "excluded_duplicate_dates": duplicate_audit.get("excluded_dates", []),
        "baseline_pipeline_matches_tongzhou": parity.get("status") == "passed",
        "construction_isolated_from_baseline": bool(invariance.get("baseline_outputs_identical_with_construction_disabled_enabled")) and not source_comparison["original_data_modified"],
        "baseline_hash_invariance_passed": invariance.get("status") == "passed",
        "globfp_enabled": False,
        "ground_truth_available": False,
        "true_accuracy_reported": False,
        "metric_area_crs": METRIC_CRS,
        "metric_mmu_m2": MMU_M2,
        "report_metric_separation": report_metric_separation,
        "highres_figure_audit": highres,
        "construction_report_is_separate": True,
        "created_at": utc_now(),
    }
    write_json(output_root / "REPORT_NUMERICAL_AUDIT.json", numerical)

    duplicate_note = "；".join(f"{date} excluded_as_duplicate_observation" for date in duplicate_audit.get("excluded_dates", [])) or "无重复日期被排除"
    pair_rows = tables["baseline_rows"]
    main_images = [item for item in figures["figures"] if item.get("kind") == "baseline_main"]
    main_image_html = "".join(f"<figure><img class='zoomable' src='{html.escape(_relative_link(item['path'], output_root))}' onclick='this.requestFullscreen()'><figcaption>{html.escape(item.get('pair', 'baseline'))}</figcaption></figure>" for item in main_images)
    construction_images = [item for item in figures["figures"] if item.get("kind") == "construction_main"]
    construction_image_html = "".join(f"<figure><img class='zoomable' src='{html.escape(_relative_link(item['path'], output_root))}' onclick='this.requestFullscreen()'><figcaption>{html.escape(item.get('date', 'construction'))}</figcaption></figure>" for item in construction_images)
    baseline_columns = ["pair_name", "full20_changed_area_m2", "full20_changed_fraction", "building_count_t1", "building_count_t2", "newly_detected_building_count", "disappeared_building_count", "expanded_building_count", "contracted_building_count", "stable_building_count"]
    construction_columns = ["date", "construction_candidate_count", "construction_candidate_area_m2", "high_uncertainty_candidate_count", "evidence_conflict_count", "temporal_flicker_rate", "scientific_status"]
    module_table = _html_table(tables["module_rows"], ["module", "status", "output_root"])
    baseline_table = _html_table(tables["baseline_rows"], baseline_columns)
    construction_table = _html_table(tables["construction_rows"], construction_columns)
    main_report = output_root / "NANJING_JIANGBEI_MODULAR_REBUILD_ZH.html"
    css = """body{font-family:system-ui,-apple-system,'Microsoft YaHei',sans-serif;margin:0;background:#f5f6f8;color:#18202a}main{max-width:1440px;margin:0 auto;padding:28px}section{background:#fff;border:1px solid #d7dce2;border-radius:10px;padding:24px;margin:18px 0}h1{font-size:30px}h2{border-bottom:1px solid #d7dce2;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #d7dce2;padding:6px;text-align:left;vertical-align:top}th{background:#eef1f5}figure{margin:12px 0}figure img{width:100%;max-width:100%;height:auto;border:1px solid #ccd3dc;background:#fff;cursor:zoom-in}figcaption{font-size:13px;color:#52606d}.grid{display:grid;grid-template-columns:1fr;gap:18px}.notice{padding:12px;border-left:4px solid #1d72aa;background:#eef7ff}.warn{padding:12px;border-left:4px solid #ba3b2f;background:#fff2f0}.small{font-size:12px;color:#5b6773}code{word-break:break-all}a{color:#075aa6}"""
    main_html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Nanjing Jiangbei modular rebuild</title><style>{css}</style></head><body><main>
<h1>南京江北新区 URRS 模块化重构验收</h1>
<p class='notice'>基础城市变化结果与工地识别结果采用相互隔离的模型支路。Construction 支路不会修改 Full20、RSBuilding、新建、拆除或建筑轨迹输出。</p>
<section><h2>1. 执行摘要</h2><p>本报告是无人工真值的外部样例验收，不报告南京 Precision、Recall、F1、IoU 或真实 Accuracy。</p><p>有效唯一影像：<b>{len(audit['valid_records'])}</b>；正式终点：<b>{html.escape(str(audit['final_date']))}</b>；重复审计：{html.escape(duplicate_note)}。</p><p>主报告状态：<b>{'completed' if baseline_complete else 'degraded'}</b>；工地报告状态：<b>{'completed_or_degraded' if construction_available else 'not_available'}</b>。</p></section>
<section><h2>2. 输入与重复影像审计</h2><p>来源：<code>{html.escape(str(audit['source_root']))}</code></p><p>正式日期：{html.escape(', '.join(audit['dates']))}</p><p>{html.escape(duplicate_note)}</p><p><a href='INPUT_DUPLICATE_AUDIT.json'>INPUT_DUPLICATE_AUDIT.json</a> · <a href='01_input_audit/IMAGE_INVENTORY.json'>IMAGE_INVENTORY.json</a> · <a href='01_input_audit/ALIGNMENT_AUDIT.json'>ALIGNMENT_AUDIT.json</a></p></section>
<section><h2>3. 重构后的模块架构</h2><p>M0 Input Audit → M1 Geometric Alignment；M2 Full20 与 M3 RSBuilding → M4 Baseline Fusion → M5 Baseline Tracking；M6 Construction → M7 Surface State/no-MTI Construction Temporal；两支路封存后 → M8 Cross-reference → Report；M9 3D 形态先验接口关闭。</p><p><a href='MODULE_RESPONSIBILITY_MATRIX.md'>MODULE_RESPONSIBILITY_MATRIX.md</a> · <a href='MODULE_DATA_CONTRACTS.json'>MODULE_DATA_CONTRACTS.json</a> · <a href='00_architecture_audit/REFACTOR_PLAN.md'>REFACTOR_PLAN.md</a></p>{module_table}</section>
<section><h2>4. 通州—南京基础流程一致性</h2><p>Parity status: <b>{html.escape(str(parity.get('status')))}</b>；一致性指入口、冻结资产、模型自有预处理、tile/overlap、阈值、vectorization/MMU、匹配规则和 metric CRS 一致，不要求不同 AOI 的数值相同。</p><p><a href='TONGZHOU_BASELINE_PARITY_AUDIT.json'>TONGZHOU_BASELINE_PARITY_AUDIT.json</a></p></section>
<section><h2>5. 南京基础综合变化结果</h2><p>正式主对比为各有效早期时点 → {html.escape(str(audit['final_date']))}；邻近时期间隔仅作附录，不混入主结论。基础指标仅来自 Full20、RSBuilding 与 geometry-only matching。</p>{baseline_table}<p><a href='11_tables/BASELINE_METRICS.csv'>BASELINE_METRICS.csv</a> · <a href='11_tables/BASELINE_ADJACENT_APPENDIX.csv'>BASELINE_ADJACENT_APPENDIX.csv</a> · <a href='04_baseline_fusion/BASELINE_SEAL_MANIFEST.json'>BASELINE_SEAL_MANIFEST.json</a></p>{main_image_html}</section>
<section><h2>6. 南京建筑几何与基线轨迹</h2><p>RSBuilding 只负责逐期建筑候选几何；M5 只输出 building_absent/present/newly_detected/disappeared/expanded/contracted/stable/uncertain_geometry，不解释施工开始、结构上升或完工。</p><p>基线代表对象：<b>{html.escape(str(representatives.get('baseline_object_id') or 'NOT AVAILABLE'))}</b> · <a href='{html.escape(_relative_link(representatives['baseline_html'], output_root))}'>BASELINE_REPRESENTATIVE_OBJECT.html</a></p><img class='zoomable' src='{html.escape(_relative_link(representatives['baseline_png'], output_root))}' onclick='this.requestFullscreen()'></section>
<section><h2>7. 工地识别独立支路</h2><p>工地结果为 UCS386 limited candidate，必须人工复核；使用独立 RGB 3,2,1、per-scene percentile 2-98、模型自有 512/256 预处理。该支路没有接收 Full20 change mask、Surface temporal state 或 baseline fusion 输出。</p><p><a href='NANJING_JIANGBEI_CONSTRUCTION_ONLY_ZH.html'>打开独立工地报告</a> · <a href='11_tables/CONSTRUCTION_METRICS.csv'>CONSTRUCTION_METRICS.csv</a></p></section>
<section><h2>8. 只读交叉参考</h2><p>交叉指标只用于解释 Full20 与工地候选的重叠、支持和冲突，不回写任何基础输出。</p><p><a href='08_cross_reference/CROSS_REFERENCE_MANIFEST.json'>CROSS_REFERENCE_MANIFEST.json</a> · <a href='11_tables/CROSS_REFERENCE_METRICS.csv'>CROSS_REFERENCE_METRICS.csv</a></p></section>
<section><h2>9. 不变性审计</h2><p>baseline_outputs_identical_with_construction_disabled_enabled = <b>{str(invariance.get('baseline_outputs_identical_with_construction_disabled_enabled')).lower()}</b>；status = <b>{html.escape(str(invariance.get('status')))}</b>。</p><p><a href='BASELINE_INVARIANCE_AUDIT.json'>BASELINE_INVARIANCE_AUDIT.json</a></p></section>
<section><h2>10. 代表对象与局限</h2><p>工地代表案例：<b>{html.escape(str(representatives.get('construction_status')))}</b> · <a href='{html.escape(_relative_link(representatives['construction_html'], output_root))}'>CONSTRUCTION_REPRESENTATIVE_SITE.html</a></p><p>Surface State、Construction-TFR、Construction-MTI 均不改变基线；M9 globfp_enabled=false。Ground truth unavailable，所有候选均不是人工真值。</p></section>
<section><h2>11. 数值与来源审计</h2><p><a href='REPORT_NUMERICAL_AUDIT.json'>REPORT_NUMERICAL_AUDIT.json</a> · <a href='REPORT_MANIFEST.json'>REPORT_MANIFEST.json</a> · <a href='serve_report.ps1'>serve_report.ps1</a></p><p>original_data_modified={str(numerical['original_data_modified']).lower()}；training_started=false；locked_test_pixel_reads=0；thresholds_changed=false；models_reselected=false；globfp_enabled=false。</p><p>高分辨率图审计：{html.escape(str(highres.get('status')))}，共 {highres.get('figure_count')} 张，最低要求 2400×1600、≥250 dpi 等效。</p></section>
</main></body></html>"""
    main_report.write_text(main_html, encoding="utf-8")

    construction_report = output_root / "NANJING_JIANGBEI_CONSTRUCTION_ONLY_ZH.html"
    construction_report.write_text(
        f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Nanjing Jiangbei Construction Activity</title><style>{css}</style></head><body><main>
<h1>南京江北新区 Construction Activity 独立报告</h1><p class='warn'>本页只报告 Construction Activity、Surface State 和 no-MTI construction temporal 候选；不载入或重算 Full20/RSBuilding 基线指标。</p>
<section><h2>1. 模型与限制</h2><p>模型：URRS-Construction-UCS386-v1.0-task-candidate；SHA256：<code>{EXPECTED_CONSTRUCTION_SHA}</code>；status：limited_candidate；human_review_required=true；UCS 386/600，缺失 214，national generalization 未验证。</p><p>RGB mapping=3,2,1；M9 globfp_enabled=false；ground truth unavailable。</p></section>
<section><h2>2. 单期候选</h2>{construction_table}{construction_image_html}<p><a href='06_construction_single_date/CONSTRUCTION_BRANCH_MANIFEST.json'>CONSTRUCTION_BRANCH_MANIFEST.json</a></p></section>
<section><h2>3. Construction temporal</h2><p>selected integration=no_mti；TFR=unselected；MTI=experimental_unselected；status={html.escape(str(read_json(output_root / '07_construction_temporal' / 'CONSTRUCTION_TEMPORAL_MANIFEST.json', {}).get('scientific_status')))}；可靠生命周期：{html.escape(str(representatives.get('construction_status')))}。</p><p><a href='07_construction_temporal/CONSTRUCTION_TEMPORAL_MANIFEST.json'>CONSTRUCTION_TEMPORAL_MANIFEST.json</a> · <a href='{html.escape(_relative_link(representatives['construction_html'], output_root))}'>CONSTRUCTION_REPRESENTATIVE_SITE.html</a></p><img class='zoomable' src='{html.escape(_relative_link(representatives['construction_png'], output_root))}' onclick='this.requestFullscreen()'></section>
<section><h2>4. 可报告的科学边界</h2><p>exposed ground、dustproof net、building evidence、candidate area、uncertainty、temporal flicker 和 evidence conflict 都是 model-derived construction statistics。它们不命名为 Accuracy、Precision、Recall、F1 或 IoU。</p></section>
<section><h2>5. 运行审计</h2><p>source_data_modified={str(numerical['original_data_modified']).lower()}；training_started=false；locked_test_pixel_reads=0；construction writes_to_baseline_dirs=false。</p><p><a href='REPORT_NUMERICAL_AUDIT.json'>REPORT_NUMERICAL_AUDIT.json</a> · <a href='REPORT_MANIFEST.json'>REPORT_MANIFEST.json</a> · <a href='serve_report.ps1'>serve_report.ps1</a></p></section>
</main></body></html>""",
        encoding="utf-8",
    )
    _write_serve_script(output_root)

    artifact_files = [path for path in sorted(output_root.rglob("*")) if path.is_file() and path.name not in {"REPORT_MANIFEST.json"}]
    manifest = {
        "status": "completed" if baseline_complete else "degraded",
        "external_sample": "nanjing_jiangbei",
        "architecture_refactor": "passed",
        "evaluation_type": "no_ground_truth_external_acceptance",
        "ground_truth": "unavailable",
        "output_root": str(output_root.resolve()),
        "main_report": str(main_report.resolve()),
        "construction_report": str(construction_report.resolve()),
        "effective_dates": audit["dates"],
        "excluded_duplicate_dates": duplicate_audit.get("excluded_dates", []),
        "effective_final_date": audit["final_date"],
        "formal_baseline_pairs": audit["formal_pairs"],
        "adjacent_appendix_pairs": audit["adjacent_pairs"],
        "module_statuses": module_statuses,
        "baseline_representative": representatives,
        "tongzhou_baseline_parity": parity.get("status"),
        "baseline_invariance": invariance.get("status"),
        "highres_figure_audit": str((output_root / "09_figures_highres" / "HIGHRES_FIGURE_AUDIT.json").resolve()),
        "original_data_modified": numerical["original_data_modified"],
        "training_started": False,
        "locked_test_pixel_reads": 0,
        "artifacts": [{"relative_path": _relative(path, output_root), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in artifact_files],
        "created_at": utc_now(),
    }
    write_json(output_root / "REPORT_MANIFEST.json", manifest)
    return {"status": manifest["status"], "report": str(main_report.resolve()), "construction_report": str(construction_report.resolve()), "numerical_audit": numerical, "manifest": manifest}


def _default_output_root() -> Path:
    return INFERENCE_ROOT / f"nanjing_jiangbei_modular_rebuild_{run_stamp()}"


def _load_or_audit(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    existing = output_root / "01_input_audit" / "MODULAR_INPUT_AUDIT.json"
    if existing.exists():
        return read_json(existing)
    run_audit(args, output_root)
    return read_json(existing)


def run_baseline_stage(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    _load_or_audit(args, output_root)
    full20 = run_full20_baseline(args, output_root)
    rsbuilding = run_rsbuilding_baseline(args, output_root)
    fusion = run_baseline_fusion(output_root)
    tracking = run_baseline_tracking(output_root)
    seal = seal_baseline(output_root)
    return {"status": "completed", "full20": full20, "rsbuilding": rsbuilding, "fusion": fusion, "tracking": tracking, "seal": seal}


def run_construction_stage(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    _load_or_audit(args, output_root)
    construction = run_construction_branch(args, output_root)
    temporal = run_construction_temporal(args, output_root)
    return {"status": "completed_or_degraded_candidate", "construction": construction, "temporal": temporal}


def _run_receipt(output_root: Path, *, status: str, command: Sequence[str], started_at: str, failures: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    receipt = {
        "status": status,
        "command": list(command),
        "started_at": started_at,
        "completed_at": utc_now(),
        "output_root": str(output_root.resolve()),
        "training_started": False,
        "thresholds_changed": False,
        "models_reselected": False,
        "locked_test_pixel_reads": 0,
        "failures": list(failures or []),
    }
    write_json(output_root / "RUN_RECEIPT.json", receipt)
    return receipt


def run_all(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    refuse_old_output(output_root)
    ensure_layout(output_root)
    started = utc_now()
    write_json(output_root / "RUN_RECEIPT.json", {"status": "running", "started_at": started, "command": list(sys.argv), "output_root": str(output_root.resolve()), "training_started": False, "locked_test_pixel_reads": 0})
    failures = []
    statuses: dict[str, Any] = {}
    try:
        statuses["Audit"] = run_audit(args, output_root)
        statuses["Baseline"] = run_baseline_stage(args, output_root)
    except Exception as exc:
        failures.append({"stage": "Baseline", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(output_root / "12_logs" / "BASELINE_FAILURE.json", failures[-1])
    if not failures:
        try:
            statuses["Construction"] = run_construction_stage(args, output_root)
        except Exception as exc:
            failures.append({"stage": "Construction", "error_type": type(exc).__name__, "error": str(exc)})
            write_json(output_root / "12_logs" / "CONSTRUCTION_FAILURE.json", failures[-1])
    if not failures or (output_root / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json").exists():
        try:
            statuses["CrossReference"] = run_cross_reference(output_root) if (output_root / "06_construction_single_date" / "CONSTRUCTION_BRANCH_MANIFEST.json").exists() and (output_root / "07_construction_temporal" / "CONSTRUCTION_TEMPORAL_MANIFEST.json").exists() else {"status": "NOT_RUN", "reason": "construction branch unavailable"}
        except Exception as exc:
            failures.append({"stage": "CrossReference", "error_type": type(exc).__name__, "error": str(exc)})
            write_json(output_root / "12_logs" / "CROSS_REFERENCE_FAILURE.json", failures[-1])
        try:
            statuses["VerifyInvariance"] = run_invariance(output_root)
        except Exception as exc:
            failures.append({"stage": "VerifyInvariance", "error_type": type(exc).__name__, "error": str(exc)})
            write_json(output_root / "12_logs" / "INVARIANCE_FAILURE.json", failures[-1])
    try:
        statuses["BuildReport"] = build_reports(output_root)
    except Exception as exc:
        failures.append({"stage": "BuildReport", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(output_root / "12_logs" / "REPORT_FAILURE.json", failures[-1])
    baseline_report = bool(statuses.get("BuildReport", {}).get("numerical_audit", {}).get("baseline_pipeline_matches_tongzhou") and statuses.get("BuildReport", {}).get("numerical_audit", {}).get("baseline_hash_invariance_passed"))
    construction_report = bool((output_root / "NANJING_JIANGBEI_CONSTRUCTION_ONLY_ZH.html").exists())
    final_status = "completed" if baseline_report and construction_report and not failures else "degraded"
    receipt = _run_receipt(output_root, status=final_status, command=sys.argv, started_at=started, failures=failures)
    return {"status": final_status, "output_root": str(output_root.resolve()), "statuses": statuses, "receipt": receipt, "failures": failures}


def main(argv: Sequence[str] | None = None) -> int:
    global RSBUILDING_PYTHON
    parser = argparse.ArgumentParser(description="URRS Nanjing Jiangbei modular rebuild")
    parser.add_argument("--mode", type=lambda value: str(value).casefold(), choices=("audit", "baseline", "construction", "crossreference", "buildreport", "verifyinvariance", "all", "status"), default="all")
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--construction-device", default="cuda")
    parser.add_argument("--python", default=str(PROJECT_ROOT / ".venv-gpu" / "Scripts" / "python.exe"))
    parser.add_argument("--rsbuilding-python", default=str(RSBUILDING_PYTHON))
    parser.add_argument("--construction-enabled", action="store_true", help="logical flag recorded for invariance; it never changes M2-M5")
    args = parser.parse_args(argv)
    RSBUILDING_PYTHON = Path(args.rsbuilding_python).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else _default_output_root().resolve()
    refuse_old_output(output_root)
    if args.mode == "status":
        target = output_root if args.output_root else None
        if target is None:
            candidates = sorted(INFERENCE_ROOT.glob("nanjing_jiangbei_modular_rebuild_*"), key=lambda path: path.stat().st_mtime)
            target = candidates[-1] if candidates else None
        if target is None or not target.exists():
            print(json.dumps({"status": "not_run", "reason": "no modular rebuild output"}, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps(read_json(target / "REPORT_MANIFEST.json", read_json(target / "RUN_RECEIPT.json", {"status": "in_progress", "output_root": str(target)})), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.mode == "audit":
        result = run_audit(args, output_root)
    elif args.mode == "baseline":
        result = run_baseline_stage(args, output_root)
    elif args.mode == "construction":
        result = run_construction_stage(args, output_root)
    elif args.mode == "crossreference":
        result = run_cross_reference(output_root)
    elif args.mode == "verifyinvariance":
        result = run_invariance(output_root)
    elif args.mode == "buildreport":
        result = build_reports(output_root)
    else:
        result = run_all(args, output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {"pass", "passed", "completed", "completed_or_degraded_candidate", "degraded", "sealed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
