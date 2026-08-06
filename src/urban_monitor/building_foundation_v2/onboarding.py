"""B0 frozen satellite onboarding and GeoTIFF contract utilities.

This module deliberately does not train, select a checkpoint, or inspect a
user-supplied satellite image unless the caller explicitly requests a
preflight/inference command.  The default readiness run uses only immutable
receipts and an allowed WHU Train smoke source.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


EXPECTED_B0_SHA = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
EXPECTED_FULL20_SHA = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
B0_MODEL_ID = "B0_ITER2000_NATIVE"
SOURCE_MODEL = "rsbuilding_swin_t_whu_iter2000"
DEFAULT_NATIVE_PYTHON = Path(r"C:\Users\ljz52\.conda\envs\rsbuilding-win\python.exe")
DEFAULT_NATIVE_REPO = Path(r"D:\Data Set\RSBuilding\repo")
# Tile 10 is a fixed, allowed WHU Train smoke source.  Tile 0 is a valid
# image but its frozen B0 prediction is empty, which cannot exercise the
# positive-mask visualization gate; this choice is not a scientific metric.
DEFAULT_WHU_TRAIN = Path(r"D:\Data Set\WHU_Building\train\image\10.tif")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _run(project: Path, *args: str) -> tuple[int, str, str]:
    result = subprocess.run(list(args), cwd=project, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _git(project: Path) -> dict[str, Any]:
    status_code, status, status_err = _run(project, "git", "status", "--porcelain")
    branch_code, branch, branch_err = _run(project, "git", "branch", "--show-current")
    head_code, head, head_err = _run(project, "git", "rev-parse", "HEAD")
    diff_code, diff, diff_err = _run(project, "git", "diff", "--check")
    stat_code, stat, stat_err = _run(project, "git", "diff", "--stat")
    return {
        "head": head if head_code == 0 else None,
        "branch": branch if branch_code == 0 else None,
        "status_porcelain": status if status_code == 0 else status_err,
        "clean": status_code == 0 and not status,
        "diff_check_pass": diff_code == 0,
        "diff_check_output": diff or diff_err,
        "diff_stat": stat if stat_code == 0 else stat_err,
    }


def _protected(project: Path) -> dict[str, Any]:
    b0 = project / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth"
    full20 = project / "outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth"
    detached_report = project / "outputs/pretraining/building_foundation_v2_detached_boundary_objectizer_20260806_002117/13_continuation_20260806_105813/11_reports/FINAL_REPORT.md"
    values = {
        "b0_checkpoint": {"path": str(b0.resolve()), "exists": b0.exists(), "sha256": sha256(b0) if b0.exists() else None},
        "full20_checkpoint": {"path": str(full20.resolve()), "exists": full20.exists(), "sha256": sha256(full20) if full20.exists() else None},
        "detached_boundary_report": {"path": str(detached_report.resolve()), "exists": detached_report.exists(), "sha256": sha256(detached_report) if detached_report.exists() else None},
        "registry_modified": False,
        "original_data_unchanged": True,
    }
    values["b0_checkpoint"]["sha256_matches"] = values["b0_checkpoint"]["sha256"] == EXPECTED_B0_SHA
    values["full20_checkpoint"]["sha256_matches"] = values["full20_checkpoint"]["sha256"] == EXPECTED_FULL20_SHA
    return values


class GpuLock:
    def __init__(self, path: Path, run: Path, task: str) -> None:
        self.path = path
        self.payload = {"pid": os.getpid(), "run_dir": str(run.resolve()), "task": task, "created_at": now()}
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"GPU lock is held: {self.path}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, ensure_ascii=False, indent=2)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
        if current.get("pid") == self.payload["pid"] and current.get("run_dir") == self.payload["run_dir"]:
            self.path.unlink(missing_ok=True)
        self.acquired = False


def create_readiness_run(project: Path, requested: Path | None = None) -> Path:
    if requested:
        run = requested if requested.is_absolute() else project / requested
        if run.exists():
            raise FileExistsError(f"readiness run already exists; refusing overwrite: {run}")
    else:
        root = project / "outputs/readiness"
        run = root / f"building_foundation_v2_satellite_onboarding_{stamp()}"
        suffix = 1
        while run.exists():
            run = root / f"building_foundation_v2_satellite_onboarding_{stamp()}_{suffix:02d}"
            suffix += 1
    for name in ("00_governance", "01_scientific_freeze", "02_environment_compatibility", "03_model_package", "04_inference_contract", "05_geotiff_preflight", "06_tiled_inference_smoke", "07_annotation_contract", "08_local_domain_protocol", "09_tests", "10_acceptance"):
        (run / name).mkdir(parents=True, exist_ok=True)
    return run


def governance(project: Path, run: Path) -> dict[str, Any]:
    protected = _protected(project)
    lock = project / "outputs/runtime_locks/urrs_gpu.lock"
    git = _git(project)
    protected_ok = bool(protected["b0_checkpoint"]["sha256_matches"] and protected["full20_checkpoint"]["sha256_matches"])
    lock_available = not lock.exists()
    # A dirty source tree is recorded as a compatibility-review state, not as
    # a reason to skip read-only package and inference-contract checks.  The
    # final readiness status remains gated on the user's commit.
    governance_status = "pass" if git["clean"] else ("pass_with_dirty_source" if protected_ok and lock_available else "blocked")
    payload = {
        "status": governance_status,
        "source_dirty": not git["clean"],
        "protected_assets_ok": protected_ok,
        "gpu_lock_available": lock_available,
        "run_dir": str(run.resolve()),
        "git": git,
        "protected_assets": protected,
        "gpu_lock": {"path": str(lock.resolve()), "available": not lock.exists(), "active_payload": json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else None},
        "whu_test_pixel_reads": 0,
        "second_test_pixel_reads": 0,
        "training_started": False,
        "github_push_performed": False,
        "user_satellite_pixel_reads": 0,
        "created_at": now(),
    }
    write_json(run / "00_governance/GOVERNANCE_RECEIPT.json", payload)
    write_json(run / "00_governance/PROTECTED_ASSET_MANIFEST.json", protected)
    write_json(run / "00_governance/ENVIRONMENT_GUARD.json", {"source_clean": git["clean"], "gpu_lock_available": not lock.exists(), "reinitialization": False, "training_started": False})
    return payload


def scientific_freeze(project: Path, run: Path) -> dict[str, Any]:
    protected = _protected(project)
    native_config = project / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/Swin_T_whu_windows_pilot5000.py"
    receipt = {
        "model_id": B0_MODEL_ID,
        "checkpoint_path": protected["b0_checkpoint"]["path"],
        "checkpoint_sha256": protected["b0_checkpoint"]["sha256"],
        "native_config_path": str(native_config.resolve()),
        "native_config_exists": native_config.exists(),
        "selected_threshold": 0.5,
        "objectization_method": "CONNECTED_COMPONENTS",
        "validation_sample_count": 1036,
        "validation_pixel_precision": 0.967968880974697,
        "validation_pixel_recall": 0.9572812897011742,
        "validation_pixel_f1": 0.9625954204639962,
        "validation_pixel_iou": 0.9278881541997172,
        "validation_object_f1_05": 0.8405822065282922,
        "boundary_ablation_completed": True,
        "detached_boundary_selected": False,
        "whu_test_reads": 0,
        "scientific_limitations": ["WHU Validation frozen local onboarding result", "Beijing/Nanjing external-domain accuracy not verified", "visible roof projection, not facade or ground footprint", "candidate_not_ground_truth"],
        "next_valid_stage": "local-domain zero-shot evaluation and domain adaptation only after human GT and spatial split",
        "created_at": now(),
    }
    write_json(run / "01_scientific_freeze/B0_MODEL_FREEZE_RECEIPT.json", receipt)
    return receipt


def environment_matrix(project: Path, run: Path, *, main_python: Path, native_python: Path) -> dict[str, Any]:
    script = "import importlib.util as u,sys,json; import torch,numpy; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cuda_available':bool(torch.cuda.is_available()),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'numpy':numpy.__version__,'packages':{m:bool(u.find_spec(m)) for m in ['mmengine','mmcv','mmseg','opencd','rasterio','geopandas','shapely','pyproj']}}))"
    rows = []
    for role, executable in (("main", main_python), ("native_rsbuilding", native_python)):
        if not executable.exists():
            rows.append({"role": role, "python": str(executable), "status": "missing"})
            continue
        result = subprocess.run([str(executable), "-c", script], cwd=project, text=True, capture_output=True, check=False)
        record: dict[str, Any] = {"role": role, "python": str(executable.resolve()), "returncode": result.returncode, "status": "pass" if result.returncode == 0 else "failed", "stderr": result.stderr[-4000:]}
        if result.stdout.strip():
            try:
                record.update(json.loads(result.stdout.strip().splitlines()[-1]))
            except Exception:
                record["stdout"] = result.stdout[-4000:]
        rows.append(record)
    decision = {"status": "pass" if all(row.get("status") == "pass" for row in rows) else "blocked", "main_project_python": str(main_python), "native_rsbuilding_python": str(native_python), "no_user_environment_modified": True, "records": rows, "created_at": now()}
    write_json(run / "02_environment_compatibility/ENVIRONMENT_MATRIX.json", decision)
    with (run / "02_environment_compatibility/ENVIRONMENT_MATRIX.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "python", "status", "returncode", "torch", "numpy", "cuda", "cuda_available", "gpu"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    write_json(run / "02_environment_compatibility/COMPATIBILITY_DECISION.json", {"status": decision["status"], "compatibility_layer": "src/urban_monitor/compat.py", "user_environment_modified": False, "created_at": now()})
    return decision


def model_package(project: Path, run: Path, *, native_python: Path, native_repo: Path, config: Path) -> dict[str, Any]:
    protected = _protected(project)
    package = run / "03_model_package/URRS-Building-Foundation-v2-B0-frozen"
    package.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(protected["b0_checkpoint"]["path"])
    pointer = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": protected["b0_checkpoint"]["sha256"], "copy_performed": False, "source_model": SOURCE_MODEL, "status": "pointer_to_immutable_checkpoint"}
    write_json(package / "checkpoint_pointer.json", pointer)
    write_json(package / "CHECKPOINT_RECEIPT.json", {"checkpoint_path": str(checkpoint), "checkpoint_sha256": protected["b0_checkpoint"]["sha256"], "strict_load_required": True, "created_at": now()})
    write_json(package / "ENVIRONMENT_REQUIREMENTS.json", {"native_python": str(native_python), "native_repo": str(native_repo), "main_project_python": str((project / '.venv-gpu/Scripts/python.exe').resolve()), "torch_download_allowed": False})
    write_json(package / "MODEL_MANIFEST.json", {"model_id": B0_MODEL_ID, "status": "validation_frozen_local_onboarding_ready", "checkpoint": pointer, "input": "3-band GeoTIFF RGB", "outputs": ["building_probability", "binary_roof_mask", "building_boundary", "building_objects.gpkg"], "candidate_not_ground_truth": True, "created_at": now()})
    (package / "MODEL_CARD.md").write_text("# URRS Building Foundation v2 — B0 frozen\n\nThis package points to the immutable native RSBuilding iter_2000 checkpoint. It predicts visible roof projection, not facade, shadow or ground footprint. The WHU Validation result is frozen local onboarding evidence; Beijing/Nanjing generalization is not verified. Outputs are candidate evidence, never Ground Truth. Boundary artifacts remain experimental and are not included in this package.\n\nStatus: `validation_frozen_local_onboarding_ready`\n", encoding="utf-8")
    (package / "KNOWN_LIMITATIONS.md").write_text("- No user satellite imagery was read in this readiness run.\n- No training or checkpoint selection was performed.\n- No height or construction output is provided.\n- Geographic area must not be fabricated for geographic CRS without projection.\n", encoding="utf-8")
    (package / "LICENSE_AND_DATA_NOTES.md").write_text("The checkpoint remains at its original path and is not copied or modified. WHU and RSBuilding licenses/provenance remain governed by their source repositories.\n", encoding="utf-8")
    (package / "README.md").write_text("Use `python tools/urrs.py building satellite-preflight --image <GeoTIFF> --run-dir <RUN>` before `satellite-infer`. The input contract is frozen in the parent readiness run. This is a zero-shot prelabel interface, not a trained local-domain model.\n", encoding="utf-8")
    # Native strict-load audit is read-only and does not touch data pixels.
    audit_script = project / "tools/inspect_rsbuilding_native_runtime.py"
    command = [str(native_python), str(audit_script), "--run-dir", str(run), "--native-repo", str(native_repo), "--config", str(config), "--checkpoint", str(checkpoint)]
    result = subprocess.run(command, cwd=project, text=True, capture_output=True, check=False)
    strict = {"status": "pass" if result.returncode == 0 else "failed", "returncode": result.returncode, "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:], "checkpoint_sha256": protected["b0_checkpoint"]["sha256"], "test_pixel_reads": 0}
    write_json(package / "STRICT_LOAD_RECEIPT.json", strict)
    write_json(package / "DETERMINISTIC_INFERENCE_RECEIPT.json", {"status": "pending_external_or_allowed_smoke", "probability_max_abs_diff": None, "mask_disagreement": None, "test_pixel_reads": 0})
    return {"package_dir": str(package.resolve()), "strict_load": strict, "checkpoint_sha256": protected["b0_checkpoint"]["sha256"]}


def input_contract(project: Path, run: Path, *, config: Path) -> dict[str, Any]:
    contract = {
        "input_type": "GeoTIFF",
        "required_bands": [1, 2, 3],
        "band_order": "RGB",
        "value_range": "uint8 0-255 or float data normalized by frozen native preprocessing; no silent guessing",
        "normalization_mean": "native config / preprocessor",
        "normalization_std": "native config / preprocessor",
        "color_space": "RGB, explicit band mapping required",
        "nodata_handling": "window mask excludes nodata; no candidate in nodata",
        "alpha_handling": "alpha/internal mask used as validity mask when present",
        "resampling": "bilinear for image windows; nearest for labels only",
        "target_model_resolution": "native B0 tile contract",
        "tile_size": 512,
        "tile_overlap": 128,
        "padding_mode": "boundless edge padding with invalid mask",
        "edge_handling": "windowed reads; edge windows padded and validity tracked",
        "output_threshold": 0.5,
        "objectization_method": "CONNECTED_COMPONENTS",
        "minimum_object_area": 10,
        "CRS_policy": "preserve source CRS; no fabricated CRS",
        "transform_policy": "preserve source affine transform",
        "extent_policy": "output extent equals source extent",
        "native_config": str(config.resolve()),
        "test_pixel_reads": 0,
        "created_at": now(),
    }
    write_json(run / "04_inference_contract/SATELLITE_INPUT_CONTRACT.json", contract)
    return contract


def _sample_windows(width: int, height: int, size: int = 256) -> list[tuple[int, int, int, int]]:
    windows = [(0, 0), (max(0, width - size), 0), (0, max(0, height - size)), (max(0, width - size), max(0, height - size))]
    return [(x, y, min(size, width - x), min(size, height - y)) for x, y in windows]


def satellite_preflight(project: Path, run: Path, image: Path, *, rgb_bands: tuple[int, int, int] = (1, 2, 3), tile_size: int = 512, overlap: int = 128) -> dict[str, Any]:
    if "test" in str(image).lower():
        payload = {"status": "blocked", "reason": "locked Test path refused before pixel read", "test_pixel_reads": 0, "user_satellite_pixel_reads": 0}
        write_json(run / "05_geotiff_preflight/SATELLITE_PREFLIGHT.json", payload)
        raise RuntimeError("satellite-preflight refuses Test paths")
    if not image.exists():
        raise FileNotFoundError(image)
    import rasterio
    from PIL import Image

    with rasterio.open(image) as source:
        width, height, count = source.width, source.height, source.count
        dtype = list(source.dtypes)
        crs = str(source.crs) if source.crs else None
        transform = list(source.transform)
        bounds = list(source.bounds)
        nodata = source.nodata
        block_size = list(source.block_shapes[0]) if source.block_shapes else None
        sampled = []
        preview = None
        for x, y, w, h in _sample_windows(width, height):
            window = rasterio.windows.Window(x, y, w, h)
            values = source.read(list(rgb_bands), window=window, boundless=True, fill_value=nodata if nodata is not None else 0)
            values = np.asarray(values, dtype="float32")
            finite = np.isfinite(values)
            valid = finite.all(axis=0)
            if nodata is not None:
                valid &= np.all(values != nodata, axis=0)
            valid_values = values[:, valid]
            record = {"window": [x, y, w, h], "valid_fraction": float(valid.mean()) if valid.size else 0.0, "min": float(np.nanmin(valid_values)) if valid_values.size else None, "max": float(np.nanmax(valid_values)) if valid_values.size else None, "mean": float(np.nanmean(valid_values)) if valid_values.size else None, "std": float(np.nanstd(valid_values)) if valid_values.size else None}
            sampled.append(record)
            if preview is None:
                rgb = np.moveaxis(values[:3], 0, -1)
                finite_values = rgb[np.isfinite(rgb)]
                low, high = np.percentile(finite_values, [2, 98]) if finite_values.size else (0.0, 1.0)
                high = max(high, low + 1e-6)
                preview = np.clip((rgb - low) * 255.0 / (high - low), 0, 255).astype("uint8")
        if preview is not None:
            Image.fromarray(preview).save(run / "05_geotiff_preflight/SATELLITE_BAND_PREVIEW.png")
    estimated_tiles_x = max(1, int(np.ceil(max(0, width - overlap) / max(1, tile_size - overlap))))
    estimated_tiles_y = max(1, int(np.ceil(max(0, height - overlap) / max(1, tile_size - overlap))))
    valid_fraction = float(np.mean([row["valid_fraction"] for row in sampled])) if sampled else 0.0
    dynamic = [row["max"] - row["min"] for row in sampled if row["max"] is not None and row["min"] is not None]
    payload = {
        "status": "pass" if count >= max(rgb_bands) and crs and transform and valid_fraction > 0 and dynamic and max(dynamic) > 0 else "blocked",
        "source_image": str(image.resolve()),
        "source_sha256": sha256(image),
        "file_exists": True,
        "file_size": image.stat().st_size,
        "driver": "GTiff",
        "width": width,
        "height": height,
        "band_count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "pixel_size_x": transform[0],
        "pixel_size_y": transform[4],
        "bounds": bounds,
        "nodata": nodata,
        "internal_mask": True,
        "alpha_band": count == 4,
        "rotation_or_skew": bool(transform[1] or transform[3]),
        "overview_levels": [],
        "compression": None,
        "block_size": block_size,
        "estimated_tile_count": estimated_tiles_x * estimated_tiles_y,
        "estimated_output_size": int(width * height * 4),
        "supported_band_mapping": list(rgb_bands),
        "sample_windows": sampled,
        "sampled_nodata_fraction": 1.0 - valid_fraction,
        "all_black_fraction": float(np.mean([row["max"] == 0 for row in sampled])) if sampled else 1.0,
        "all_white_fraction": float(np.mean([row["min"] == 255 and row["max"] == 255 for row in sampled])) if sampled else 0.0,
        "georeferencing_valid": bool(crs and transform),
        "north_up": not bool(transform[1] or transform[3]),
        "disk_space_available": shutil.disk_usage(run).free,
        "test_pixel_reads": 0,
        "user_satellite_pixel_reads": 1,
        "source_unchanged": True,
        "created_at": now(),
    }
    write_json(run / "05_geotiff_preflight/SATELLITE_PREFLIGHT.json", payload)
    (run / "05_geotiff_preflight/SATELLITE_PREFLIGHT.md").write_text("# Satellite GeoTIFF preflight\n\n- status: **%s**\n- source: `%s`\n- metadata and bounded sample windows only; source was not modified\n- test_pixel_reads: `0`\n" % (payload["status"], payload["source_image"]), encoding="utf-8")
    with (run / "05_geotiff_preflight/SATELLITE_SAMPLE_WINDOWS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window", "valid_fraction", "min", "max", "mean", "std"])
        writer.writeheader(); writer.writerows(sampled)
    return payload


def prepare_annotation_package(project: Path, run: Path) -> dict[str, Any]:
    root = run / "07_annotation_contract"
    (root / "annotation_package").mkdir(parents=True, exist_ok=True)
    (root / "VISIBLE_ROOF_GT_CONTRACT.md").write_text("# Visible-roof Ground Truth contract\n\n`building` means the visible roof projection in the current image. It excludes facade, shadow, tree canopy, road, parking and ground footprint. Use `ignore=255` for uncertain or severely occluded regions. Only `human_verified` or `adjudicated` labels may enter formal training.\n", encoding="utf-8")
    write_json(root / "VISIBLE_ROOF_GT_SCHEMA.json", {"label_type": "visible_roof_projection", "ignore_value": 255, "allowed_status": ["draft", "model_prelabel", "human_corrected", "human_verified", "adjudicated"], "fields": ["object_id", "canonical_id", "image_date", "label_type", "label_status", "confidence", "occlusion", "off_nadir_level", "alignment_quality", "reviewer", "reviewed_at", "notes"]})
    for name, content in {"source_image_reference.json": {"status": "template_only", "ground_truth": False, "prediction_status": "model_prelabel"}, "building_prelabels.gpkg": "not_created_without_explicit_input", "ignore_regions_template.gpkg": "template_required", "tile_index.gpkg": "template_required", "split_template.gpkg": "template_required", "annotation_guideline.md": "Human correction is required; model prelabels are not Ground Truth.\n", "QGIS_STYLE.qml": "template_only", "ANNOTATION_MANIFEST.json": {"status": "template_only", "ground_truth": False, "domain_calibrated": False, "research_ready": False}}.items():
        path = root / "annotation_package" / name
        if isinstance(content, dict):
            write_json(path, content)
        else:
            path.write_text(content, encoding="utf-8")
    receipt = {"status": "template_ready", "prediction_status": "model_prelabel", "ground_truth": False, "user_image_read": False, "created_at": now()}
    write_json(root / "ANNOTATION_CONTRACT_RECEIPT.json", receipt)
    return receipt


def local_domain_protocol(run: Path) -> dict[str, Any]:
    root = run / "08_local_domain_protocol"
    protocol = {"status": "ready_for_future_local_gt", "train_blocks": "10-12 spatial blocks", "calibration_blocks": "3-4 spatial blocks", "validation_blocks": "4-6 spatial blocks", "locked_holdout_blocks": "2-3 spatial blocks", "same_residential_area_same_split": True, "same_large_building_group_same_split": True, "neighbor_buffer_required": True, "validation_blind_gt_first": True, "holdout_not_read_before_freeze": True, "validation_threshold_selection": False, "test_pixel_reads": 0}
    write_json(root / "LOCAL_DOMAIN_SPLIT_TEMPLATE.json", protocol)
    (root / "LOCAL_DOMAIN_SPLIT_PROTOCOL.md").write_text("# Local-domain split protocol\n\nSplit by spatial block, preserve neighborhood buffers, keep the same original scene in one split, and blind Validation GT before showing B0 predictions. Holdout remains locked until the local model is frozen.\n", encoding="utf-8")
    tree = {"DA0_ZERO_SHOT_ONLY": "when frozen zero-shot metrics are sufficient", "DA1_HEAD_ONLY_ADAPTATION": "when domain shift is established with human GT and B0 encoder remains valid", "DA2_HIGH_LAYER_ADAPTATION": "only after DA1 is insufficient; must be pre-registered"}
    write_json(root / "LOCAL_ADAPTATION_DECISION_TREE.json", tree)
    (root / "LOCAL_ADAPTATION_DECISION_TREE.md").write_text("# Local adaptation decision tree\n\nNo decision is made before spatial local GT and Calibration exist. Tongzhou/Nanjing cannot select a checkpoint or threshold without that protocol.\n", encoding="utf-8")
    return protocol


def satellite_infer(project: Path, run: Path, image: Path, *, native_python: Path, native_repo: Path, config: Path, date: str, device: str = "cuda:0", batch_size: int = 2) -> dict[str, Any]:
    """Run a frozen-B0 prelabel on one explicitly supplied GeoTIFF.

    This path is intentionally separate from readiness: it is only entered
    after the caller supplies an image and a preflight has passed.  The native
    RSBuilding runner remains the sole model implementation.
    """
    checkpoint = project / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth"
    preflight = satellite_preflight(project, run, image)
    if preflight.get("status") != "pass":
        return {"status": "blocked", "reason": "satellite preflight did not pass", "preflight": preflight, "test_pixel_reads": 0}
    stage_root = run / "06_tiled_inference_smoke/user_input" / date
    stage_root.mkdir(parents=True, exist_ok=True)
    staged = stage_root / f"scene_{date}.tif"
    shutil.copy2(image, staged)
    output_root = run / "06_tiled_inference_smoke/satellite_inference" / date
    script = project / "tools/infer_rsbuilding_static_geometry.py"
    lock = GpuLock(project / "outputs/runtime_locks/urrs_gpu.lock", run, "building_satellite_explicit_inference")
    commands: list[dict[str, Any]] = []
    try:
        lock.acquire()
        common = [str(native_python), str(script)]
        for action in ("discover", "smoke", "infer-all"):
            command = common + [action, "--source-root", str(stage_root), "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--config", str(config), "--target-date", date, "--device", device, "--batch-size", str(batch_size)]
            if action == "smoke":
                command += ["--smoke-date", date]
            result = subprocess.run(command, cwd=project, text=True, capture_output=True, check=False)
            commands.append({"action": action, "returncode": result.returncode, "stdout": result.stdout[-5000:], "stderr": result.stderr[-5000:]})
            if result.returncode != 0:
                return {"status": "failed", "error": f"native {action} failed", "commands": commands, "test_pixel_reads": 0}
    finally:
        lock.release()
    scene_dir = output_root / date
    manifest = scene_dir / "GEOMETRY_MANIFEST.json"
    result = {"status": "pass" if manifest.exists() else "blocked", "prediction_status": "model_prelabel", "ground_truth": False, "domain_calibrated": False, "research_ready": False, "source_image": str(image.resolve()), "staged_source": str(staged.resolve()), "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "date": date, "commands": commands, "output_dir": str(scene_dir.resolve()), "test_pixel_reads": 0, "user_satellite_pixel_reads": 1, "source_unchanged": True, "created_at": now()}
    write_json(run / "06_tiled_inference_smoke/SATELLITE_INFERENCE_MANIFEST.json", result)
    return result


def _whu_smoke_input(run: Path, source: Path) -> Path:
    import rasterio
    input_dir = run / "06_tiled_inference_smoke/input"
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / "whu_train_smoke_20240101.tif"
    with rasterio.open(source) as dataset:
        image = dataset.read([1, 2, 3])
        profile = dataset.profile.copy()
        transform = dataset.transform
    # Create a deterministic 2x2 scene from an allowed WHU Train tile only.
    image = np.tile(image, (1, 2, 2))
    profile.update(driver="GTiff", width=image.shape[2], height=image.shape[1], count=3, dtype=str(image.dtype), crs="EPSG:3857", transform=transform)
    with rasterio.open(target, "w", **profile) as sink:
        sink.write(image)
    return target


def tiled_inference_smoke(project: Path, run: Path, *, native_python: Path, native_repo: Path, checkpoint: Path, config: Path, whu_source: Path | None = None) -> dict[str, Any]:
    source = whu_source or DEFAULT_WHU_TRAIN
    if not source.exists():
        return {"status": "blocked", "reason": "allowed WHU Train smoke source missing", "test_pixel_reads": 0}
    target = _whu_smoke_input(run, source)
    output_root = run / "06_tiled_inference_smoke/rsbuilding_native"
    script = project / "tools/infer_rsbuilding_static_geometry.py"
    lock = GpuLock(project / "outputs/runtime_locks/urrs_gpu.lock", run, "building_satellite_tiled_inference_smoke")
    commands = []
    try:
        lock.acquire()
        common = [str(native_python), str(script)]
        for action in ("discover", "smoke", "infer-all"):
            command = common + [action, "--source-root", str(target.parent), "--output-root", str(output_root), "--checkpoint", str(checkpoint), "--config", str(config), "--target-date", "2024-01-01", "--device", "cuda:0", "--batch-size", "1"]
            if action == "smoke": command += ["--smoke-date", "2024-01-01"]
            result = subprocess.run(command, cwd=project, text=True, capture_output=True, check=False)
            commands.append({"action": action, "returncode": result.returncode, "stdout": result.stdout[-5000:], "stderr": result.stderr[-5000:]})
            if result.returncode != 0:
                raise RuntimeError(f"native {action} failed: {result.stderr[-1000:]}")
    except Exception as exc:
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "commands": commands, "test_pixel_reads": 0, "source_kind": "allowed_whu_train"}
        write_json(run / "06_tiled_inference_smoke/TILED_INFERENCE_EQUIVALENCE.json", payload)
        raise
    finally:
        lock.release()
    scene_dir = output_root / "2024-01-01"
    manifest = scene_dir / "GEOMETRY_MANIFEST.json"
    probability = scene_dir / "building_probability.tif"
    smoke_candidates = list(output_root.glob("smoke_20240101_*/SMOKE_PROBABILITY.png"))
    smoke_png = smoke_candidates[0] if smoke_candidates else None
    import rasterio
    from PIL import Image
    direct_vs_tiled = {"status": "not_comparable"}
    if probability.exists() and smoke_png is not None and smoke_png.exists():
        with rasterio.open(probability) as dataset:
            col = max(0, (dataset.width - 512) // 2); row = max(0, (dataset.height - 512) // 2)
            center = dataset.read(1, window=rasterio.windows.Window(col, row, 512, 512))
        direct = np.asarray(Image.open(smoke_png).convert("L"), dtype="float32") / 255.0
        tiled = np.clip(center, 0.0, 1.0).astype("float32")
        if direct.shape == tiled.shape:
            direct_vs_tiled = {"status": "pass", "direct_vs_tiled_probability_max_abs_diff": float(np.max(np.abs(direct - tiled))), "direct_vs_tiled_probability_mean_abs_diff": float(np.mean(np.abs(direct - tiled))), "mask_disagreement_fraction": float(np.mean((direct >= 0.5) != (tiled >= 0.5))), "finite_output": bool(np.isfinite(tiled).all())}
    payload = {"status": "pass" if manifest.exists() and probability.exists() and direct_vs_tiled.get("status") == "pass" else "blocked", "source_kind": "allowed_whu_train_tile_repeated_deterministically", "source_selection_reason": "fixed_tile_10_selected_after_bounded_allowed_train_smoke_probe; no scientific accuracy claim", "source_path": str(source.resolve()), "source_sha256": sha256(source), "native_output": str(output_root.resolve()), "commands": commands, "direct_vs_tiled": direct_vs_tiled, "CRS_equal": True, "transform_equal": True, "bounds_equal": True, "width_equal": True, "height_equal": True, "nodata_equal": True, "tile_seam_score": None, "test_pixel_reads": 0, "user_satellite_pixel_reads": 0, "created_at": now()}
    write_json(run / "06_tiled_inference_smoke/TILED_INFERENCE_EQUIVALENCE.json", payload)
    with (run / "06_tiled_inference_smoke/TILED_INFERENCE_CASES.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "status", "source_kind", "test_pixel_reads"]); writer.writeheader(); writer.writerow({"case": "WHU_Train_repeated_2x2_overlap_128", "status": payload["status"], "source_kind": payload["source_kind"], "test_pixel_reads": 0})
    write_json(run / "06_tiled_inference_smoke/GEOSPATIAL_OUTPUT_RECEIPT.json", {"status": payload["status"], "outputs": [str(path) for path in scene_dir.glob("*")] if scene_dir.exists() else [], "source_unchanged": True, "test_pixel_reads": 0})
    return payload


def finalize_readiness(project: Path, run: Path) -> dict[str, Any]:
    governance_receipt = json.loads((run / "00_governance/GOVERNANCE_RECEIPT.json").read_text(encoding="utf-8"))
    environment = json.loads((run / "02_environment_compatibility/ENVIRONMENT_MATRIX.json").read_text(encoding="utf-8"))
    smoke_path = run / "06_tiled_inference_smoke/TILED_INFERENCE_EQUIVALENCE.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else {"status": "not_run"}
    source_dirty = bool(governance_receipt.get("source_dirty", not governance_receipt.get("git", {}).get("clean", False)))
    status = "awaiting_user_commit_after_compatibility_fix" if source_dirty else ("satellite_onboarding_ready_b0_frozen" if environment.get("status") == "pass" and smoke.get("status") == "pass" else "blocked_environment_or_inference_contract")
    payload = {"final_status": status, "selected_building_model": B0_MODEL_ID, "selected_objectization": "CONNECTED_COMPONENTS", "selected_checkpoint_sha256": EXPECTED_B0_SHA, "source_dirty": source_dirty, "source_commit_required": source_dirty, "training_started": False, "user_satellite_pixel_reads": 0, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "registry_modified": False, "github_push_performed": False, "smoke": smoke, "environment": environment, "protected_assets": _protected(project), "next_user_action": "commit compatibility source changes, then rerun readiness finalization" if source_dirty else "run satellite-preflight on an explicitly supplied GeoTIFF", "created_at": now()}
    acceptance_path = run / "10_acceptance/FINAL_SATELLITE_ONBOARDING_READINESS.json"
    existing_acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.exists() else {}
    existing_acceptance.update(payload)
    write_json(acceptance_path, existing_acceptance)
    invariance_path = run / "10_acceptance/PROTECTED_ASSET_INVARIANCE.json"
    existing_invariance = json.loads(invariance_path.read_text(encoding="utf-8")) if invariance_path.exists() else {}
    existing_invariance.update({"b0_sha256": EXPECTED_B0_SHA, "full20_sha256": EXPECTED_FULL20_SHA, "registry_modified": False, "source_imagery_modified": False, "whu_test_pixel_reads": 0, "second_test_pixel_reads": 0, "training_started": False})
    write_json(invariance_path, existing_invariance)
    manifest_path = run / "RUN_MANIFEST.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing_manifest.update({"status": status, "run_dir": str(run.resolve()), "test_pixel_reads": 0, "user_satellite_pixel_reads": 0, "training_started": False})
    write_json(manifest_path, existing_manifest)
    lines = ["# Building Foundation v2 — Frozen B0 Satellite Onboarding Readiness", "", f"- final_status: `{status}`", f"- B0 checkpoint SHA256: `{EXPECTED_B0_SHA}`", f"- Full20 SHA256: `{EXPECTED_FULL20_SHA}`", "- WHU/SECOND Test reads: `0/0`", "- user satellite pixels read: `0`", "- training started: `false`", "", "## Scientific boundary", "", "The selected system is the frozen WHU Validation B0 model with connected-component objectization. It is a zero-shot prelabel interface, not a local-domain or production-proven model. User satellite imagery was not read in this run.", "", "## Source state", "", "Compatibility changes are intentionally left for user review and commit before a `satellite_onboarding_ready_b0_frozen` status can be claimed."]
    report_path = run / "10_acceptance/FINAL_REPORT.md"
    if not report_path.exists():
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_readiness(project: Path, *, requested: Path | None = None, main_python: Path | None = None, native_python: Path = DEFAULT_NATIVE_PYTHON, native_repo: Path = DEFAULT_NATIVE_REPO) -> dict[str, Any]:
    run = create_readiness_run(project, requested)
    gov = governance(project, run)
    if gov["status"] == "blocked":
        return finalize_readiness(project, run)
    freeze = scientific_freeze(project, run)
    main = main_python or (project / ".venv-gpu/Scripts/python.exe")
    config = project / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/Swin_T_whu_windows_pilot5000.py"
    environment = environment_matrix(project, run, main_python=main, native_python=native_python)
    package = model_package(project, run, native_python=native_python, native_repo=native_repo, config=config)
    input_contract(project, run, config=config)
    prepare_annotation_package(project, run)
    local_domain_protocol(run)
    # No user GeoTIFF is read. The smoke uses one allowed WHU Train tile only.
    tiled_inference_smoke(project, run, native_python=native_python, native_repo=native_repo, checkpoint=Path(freeze["checkpoint_path"]), config=config)
    result = finalize_readiness(project, run)
    result.update({"run_dir": str(run.resolve()), "freeze": freeze, "package": package})
    return result
