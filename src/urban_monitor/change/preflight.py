from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from .adapters.base import read_binary, read_s2looking_binary
from .adapters.second import semantic_ids
from .registry import enabled_adapters, load_change_config
from .second_dev_split import build_or_load_second_development_split
from .splits import resolve_train_validation_test_splits
from .checkpoint_merge_v2 import merge_v2_checkpoints
from .supervision_v2 import supervision_matrix


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_shape(path: Path) -> tuple[int, ...]:
    with Image.open(path) as image:
        return tuple(np.asarray(image).shape)


def _gpu():
    result = {"python": platform.python_version(), "executable": sys.executable, "virtual_env": os.environ.get("VIRTUAL_ENV", "")}
    try:
        import torch
        import torchvision
        result.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            result.update(gpu_name=prop.name, gpu_memory_bytes=prop.total_memory)
    except Exception as exc:
        result["torch_error"] = str(exc)
    try:
        command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        output = subprocess.run(command, capture_output=True, text=True, timeout=10)
        result["nvidia_smi"] = output.stdout.strip() if output.returncode == 0 else output.stderr.strip()
    except Exception as exc:
        result["nvidia_smi"] = f"unavailable: {exc}"
    return result


def _report(path: Path, result: dict) -> None:
    lines = ["# Public change preflight", "", f"READY_TO_TRAIN = **{result['READY_TO_TRAIN']}**", ""]
    for name in ("blockers", "warnings", "passed_checks"):
        lines.append(f"## {name}")
        values = result[name] or ["none"]
        lines.extend(f"- {value}" for value in values)
        lines.append("")
    lines += ["## Dataset summary", "", "| dataset | train | validation | test | binary | semantic | directional | status |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for item in result["datasets"]:
        lines.append("| {name} | {train} | {validation} | {test} | {binary} | {semantic} | {directional} | {status} |".format(**item))
    lines += ["", "## Recommended command", "", f"`{result['recommended_command']}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_change_preflight(config_path: str | Path, *, output_root: str | Path | None = None, require_cuda: bool = True) -> dict:
    """Validate raw public datasets without copying or modifying them."""
    config = load_change_config(config_path)
    v2_mode = int(config.get("version", 0) or 0) == 2
    root = Path(output_root or config.get("output_root", "outputs/pretraining/change")) / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "READY_TO_TRAIN": False, "blockers": [], "warnings": [], "passed_checks": [], "datasets": [],
        "enabled_datasets": [], "detected_layouts": {}, "selected_validation_splits": {}, "supervision_by_stage": {},
        "gpu": _gpu(), "free_disk_bytes": shutil.disk_usage(root).free,
        "recommended_batch_size": config.get("training", {}).get("batch_size", "auto"),
        "recommended_command": f'python -u run.py change-all --config "{Path(config_path)}" --device cuda:0 --quick',
    }
    if require_cuda and not result["gpu"].get("cuda_available"):
        result["blockers"].append("CUDA Torch is unavailable; formal public training will not fall back to CPU")
    all_rows = []
    names = [name for name, values in config.get("datasets", {}).items() if values.get("enabled")]
    if not names:
        result["blockers"].append("No public change dataset is enabled")
    for name in names:
        values = config.get("datasets", {}).get(name, {})
        configured_root = str(values.get("root", ""))
        item = {"name": name, "root": configured_root, "train": 0, "validation": 0, "test": 0,
                "binary": 0, "semantic": 0, "directional": 0, "status": "blocked", "blockers": [], "warnings": []}
        result["enabled_datasets"].append(name)
        try:
            isolated = dict(config)
            isolated["datasets"] = {key: {**value, "enabled": key == name} for key, value in config.get("datasets", {}).items()}
            adapter = enabled_adapters(isolated)[0]
            item["root"] = str(adapter.root.resolve())
            result["detected_layouts"][name] = {
                "adapter": adapter.__class__.__name__,
                "root": item["root"],
                "expected_splits": ["train", "validation", "test"],
            }
            records = adapter.discover()
            if name == "second" and v2_mode and values.get("development_validation", {}).get("enabled", False):
                frozen = build_or_load_second_development_split(records, adapter)
                records = frozen.records
                item["development_validation"] = {
                    "status": "ready", "source": "frozen_train_subset", "train": frozen.audit["development_train_count"],
                    "validation": frozen.audit["actual_validation_count"], "seed": frozen.audit["seed"],
                    "manifest": frozen.audit["manifest"], "manifest_sha256": frozen.manifest_sha256,
                    "test_images_read": frozen.audit["test_images_read"],
                }
                result["second_development_validation"] = item["development_validation"]
            # Discover validates matching stems, supported aliases, dimensions and readable inputs.
            for record in records:
                split = record.split.casefold()
                if split == "train": item["train"] += 1
                elif split in {"val", "valid", "validation"}: item["validation"] += 1
                elif split == "test": item["test"] += 1
                item["binary"] += int(record.has_binary); item["semantic"] += int(record.has_semantic); item["directional"] += int(record.has_directional)
                all_rows.append(record.to_row())
            if name == "s2looking":
                mismatches, new_pixels, dem_pixels = [], 0, 0
                for record in records:
                    cd, new, dem = read_s2looking_binary(record.gt_cd_path), read_s2looking_binary(record.gt_new_path), read_s2looking_binary(record.gt_dem_path)
                    mismatch = float(np.mean(cd != np.logical_or(new, dem)))
                    mismatches.append(mismatch); new_pixels += int(new.sum()); dem_pixels += int(dem.sum())
                threshold = config.get("audit", {})
                maximum = max(mismatches, default=0.0)
                item["s2looking_max_mismatch"] = maximum
                if maximum > float(threshold.get("s2looking_mismatch_failure_ratio", 0.01)):
                    item["blockers"].append(f"S2Looking mismatch {maximum:.6f} exceeds failure threshold")
                elif maximum > float(threshold.get("s2looking_mismatch_warning_ratio", 0.001)):
                    item["warnings"].append(f"S2Looking mismatch {maximum:.6f} exceeds warning threshold")
                if not new_pixels: item["warnings"].append("S2Looking new-building labels are all empty")
                if not dem_pixels: item["warnings"].append("S2Looking demolition labels are all empty")
                item["direction_definition"] = "T1 -> T2: label1/new building, label2/demolition"
            if name == "dsifn":
                ratios = [float(read_binary(record.gt_cd_path).mean()) for record in records]
                item["change_pixel_ratio_mean"] = float(np.mean(ratios)) if ratios else 0.0
                item["supervision"] = {"has_binary": True, "has_directional": False, "has_semantic": False}
            if name == "second":
                ids = set()
                ignore_index = int(values.get("semantic_ignore_index", values.get("ignore_index", 255)))
                for record in records:
                    ids.update(np.unique(semantic_ids(record.gt_t1_path, adapter.color_mapping)).astype(int).tolist())
                    ids.update(np.unique(semantic_ids(record.gt_t2_path, adapter.color_mapping)).astype(int).tolist())
                item["semantic_class_ids"] = sorted(ids)
                trainable_ids = sorted(class_id for class_id in ids if class_id != ignore_index)
                building = adapter.building_class_id
                if v2_mode and not values.get("mapping_verified", False):
                    # 255 is an explicitly configured ignore value.  It is not
                    # remapped and it must never be guessed to mean building.
                    item["semantic_supervision"] = "partial_ready"
                    item["building_mapping"] = "pending"
                    item["warnings"].append("SECOND building mapping pending: directional building losses are disabled")
                else:
                    if building not in trainable_ids: item["blockers"].append(f"SECOND building_class_id={building} is absent from labels")
                    expected = list(range(max(trainable_ids) + 1)) if trainable_ids else []
                    if trainable_ids != expected: item["blockers"].append(f"SECOND class IDs are not contiguous 0..C-1 after ignore={ignore_index}: {trainable_ids}")
                    if trainable_ids and max(trainable_ids) >= int(config.get("model", {}).get("semantic_classes", 0)):
                        item["blockers"].append("SECOND class IDs exceed model.semantic_classes")
            item["status"] = "ready" if not item["blockers"] else "blocked"
            result["passed_checks"].append(f"{name}: discovered {len(records)} paired samples")
        except Exception as exc:
            item["blockers"].append(str(exc))
        result["datasets"].append(item)
        result["blockers"].extend(f"{name}: {value}" for value in item["blockers"])
        result["warnings"].extend(f"{name}: {value}" for value in item["warnings"])
    if all_rows:
        import pandas as pd
        frame = pd.DataFrame(all_rows)
        try:
            _, _, _, selection = resolve_train_validation_test_splits(frame, config)
            result["selected_validation_splits"] = selection
            for stage, flag in (("binary", "has_binary"), ("semantic", "has_semantic"), ("directional", "has_directional")):
                result["supervision_by_stage"][stage] = int(frame[frame[flag].astype(str).str.casefold().isin({"true", "1", "yes"})].shape[0])
        except Exception as exc:
            result["blockers"].append(str(exc))
    else:
        result["blockers"].append("No valid enabled public dataset samples")
    if v2_mode and not result["blockers"]:
        # This is a real construction/merge check, not a dry-run.  It opens
        # only declared train/validation samples; adapters never enumerate test.
        try:
            init = config.get("initialization", {})
            merged = root / "v2_merged_initial.pth"
            merge_report = merge_v2_checkpoints(
                config,
                init["dsifn_checkpoint"],
                init["s2looking_checkpoint"],
                merged,
                root / "CHECKPOINT_MERGE_REPORT.json",
            )
            result["checkpoint_merge"] = merge_report
            result["supervision_router"] = supervision_matrix(
                second_mapping_verified=bool(config.get("datasets", {}).get("second", {}).get("mapping_verified", False))
            )
            result["test_images_read"] = 0
            result["passed_checks"].append("V2 model instantiated and checkpoint merge verified")
        except Exception as exc:
            result["blockers"].append(f"V2 checkpoint merge: {exc}")
    result["READY_TO_TRAIN"] = not result["blockers"]
    (root / "PRETRAIN_CHECKLIST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(root / "PRETRAIN_CHECKLIST.md", result)
    return result
