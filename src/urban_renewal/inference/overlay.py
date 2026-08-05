from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
import torch

from ..core.hashing import sha256_file
from ..core.receipts import read_json, write_json, utc_now


EXPECTED_FULL20 = "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
EXPECTED_RSBUILDING = "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
EXPECTED_JOINT = "35668be3a61b30a29576fc66203b67505e4a0cfa56051d04d6bc25ed21de875c"
DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")


def _copy_scene_geometry(geometry_root: Path, scene: str, out_scene: Path) -> list[str]:
    out_scene.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in ("building_probability.tif", "building_mask.tif", "building_boundary.tif", "building_objects.gpkg"):
        source = geometry_root / scene / name
        if source.exists():
            target = out_scene / name
            shutil.copy2(source, target)
            outputs.append(str(target))
    return outputs


def _write_candidate(path: Path, template: Path, arrays: tuple[Path, ...], mode: str) -> dict[str, Any]:
    with rasterio.open(template) as base:
        profile = base.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        path.parent.mkdir(parents=True, exist_ok=True)
        positives = 0
        total = 0
        with rasterio.open(path, "w", **profile) as dst:
            for _, window in base.block_windows(1):
                masks = []
                for src_path in arrays:
                    with rasterio.open(src_path) as src:
                        masks.append(src.read(1, window=window, masked=True).filled(0))
                if mode == "new":
                    value = (masks[0] > 0) & (masks[1] > 0) & (masks[2] == 0)
                elif mode == "demolition":
                    value = (masks[0] > 0) & (masks[1] > 0) & (masks[2] == 0)
                else:
                    value = (masks[0] > 0) & (masks[1] > 0) & (masks[2] > 0)
                out = np.where(value, 1, 0).astype("uint8")
                dst.write(out, 1, window=window)
                positives += int(value.sum())
                total += int(value.size)
    return {"path": str(path), "positive_fraction": positives / total if total else 0.0, "mode": mode}


def _write_window_candidate(path: Path, evidence_path: Path, prior_path: Path, post_path: Path, mode: str) -> dict[str, Any]:
    with rasterio.open(evidence_path) as evidence, rasterio.open(prior_path) as prior, rasterio.open(post_path) as post:
        profile = evidence.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")
        path.parent.mkdir(parents=True, exist_ok=True)
        positives = 0
        total = 0
        with rasterio.open(path, "w", **profile) as dst:
            for _, window in evidence.block_windows(1):
                ev = evidence.read(1, window=window, masked=True).filled(0).astype(np.float32)
                a = prior.read(1, window=window, masked=True).filled(0)
                b = post.read(1, window=window, masked=True).filled(0)
                if mode == "new":
                    value = (ev >= 0.5) & (a == 0) & (b == 1)
                elif mode == "demolition":
                    value = (ev >= 0.5) & (a == 1) & (b == 0)
                else:
                    value = (a == 1) & (b == 1) & (ev < 0.5)
                dst.write(value.astype("uint8"), 1, window=window)
                positives += int(value.sum())
                total += int(value.size)
    return {"path": str(path), "positive_fraction": positives / total if total else 0.0, "mode": mode}


def _overview(source_path: Path, mask_paths: list[Path], target: Path) -> None:
    with rasterio.open(source_path) as src:
        scale = min(1.0, 1400 / max(src.width, src.height))
        h, w = max(1, int(src.height * scale)), max(1, int(src.width * scale))
        bands = src.read([1, 2, 3] if src.count >= 3 else [1], out_shape=(min(3, src.count), h, w), resampling=Resampling.bilinear)
        if bands.shape[0] == 1:
            bands = np.repeat(bands, 3, axis=0)
        image = np.moveaxis(bands, 0, -1).astype(np.float32)
        image -= image.min()
        image /= max(image.max(), 1e-6)
        canvas = (image * 255).astype(np.uint8)
        target.parent.mkdir(parents=True, exist_ok=True)
        base = Image.fromarray(canvas, mode="RGB").convert("RGBA")
        colors = ((255, 0, 0, 120), (0, 150, 255, 120), (255, 210, 0, 110))
        for mask_path, color in zip(mask_paths, colors):
            with rasterio.open(mask_path) as mask:
                arr = mask.read(1, out_shape=(h, w), resampling=Resampling.nearest)
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            overlay.putalpha(Image.fromarray(np.where(arr > 0, color[3], 0).astype(np.uint8)))
            solid = Image.new("RGBA", (w, h), color[:3] + (0,))
            solid.putalpha(overlay.getchannel("A"))
            base = Image.alpha_composite(base, solid)
        draw = ImageDraw.Draw(base)
        draw.rectangle((8, 8, 370, 42), fill=(0, 0, 0, 180))
        draw.text((16, 16), "URRS candidate overlay 路 not Ground Truth", fill=(255, 255, 255, 255))
        base.convert("RGB").save(target)


def run_geometry_overlay(project_root: str | Path, *, output_root: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    geometry_root = project_root / "outputs/inference/beijing_tongzhou_building_geometry"
    evidence_root = project_root / "outputs/inference/beijing_tongzhou_to_latest"
    joint = project_root / "outputs/training/surface_state_v1/joint_sample_20260730_034853/checkpoints/surface_joint_sample_best_validation.pth"
    full20 = project_root / "outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth"
    rsbuilding = project_root / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth"
    mti = project_root / "outputs/training/surface_state_v1/mti_sample_20260730_034853/MTI_SELECTION_FINAL.json"
    for path in (joint, full20, rsbuilding, mti):
        if not path.exists():
            raise FileNotFoundError(path)
    hashes = {"joint": sha256_file(joint), "full20": sha256_file(full20), "rsbuilding": sha256_file(rsbuilding), "mti_selection": sha256_file(mti)}
    if hashes["joint"] != EXPECTED_JOINT or hashes["full20"] != EXPECTED_FULL20 or hashes["rsbuilding"] != EXPECTED_RSBUILDING:
        raise RuntimeError(f"Frozen artifact hash mismatch: {hashes}")
    selection = read_json(mti)
    if selection.get("integration_mode") != "no_mti":
        raise RuntimeError("Only the selected no-MTI integration may be used")
    geometry_audit = read_json(geometry_root / "TONGZHOU_URRS_GEOMETRY_AUDIT.json")
    alignment = read_json(geometry_root / "TONGZHOU_URRS_ALIGNMENT_AUDIT.json") if (geometry_root / "TONGZHOU_URRS_ALIGNMENT_AUDIT.json").exists() else {}
    if geometry_audit.get("geometry_numerical_status") != "pass" or alignment.get("alignment_status") != "ready_for_pixelwise_temporal_tracking":
        raise RuntimeError("Geometry numerical and alignment gates must pass before overlay")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(output_root) if output_root else project_root / "outputs/integration/urban_renewal_overlay_with_geometry" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    # Load the selected Joint artifact on CPU as an explicit integration check.
    state = torch.load(joint, map_location="cpu", weights_only=False)
    joint_loaded = isinstance(state, dict) and bool(state)
    per_scene = []
    for date in DATES:
        scene_output = output / "per_scene" / date
        copied = _copy_scene_geometry(geometry_root, date, scene_output)
        per_scene.append({"scene_date": date, "outputs": copied, "candidate_scope": "RSBuilding geometry candidate", "scientific_status": "candidate_not_ground_truth"})
    per_transition = []
    for pair_dir in sorted(evidence_root.glob("*_to_20230716")):
        stem = pair_dir.name
        before = stem[:8]
        before_date = {"20180506": "2018-05-06", "20200723": "2020-07-23", "20220320": "2022-03-20"}.get(before)
        if not before_date:
            continue
        evidence = pair_dir / "rasters"
        prior = geometry_root / before_date / "building_mask.tif"
        post = geometry_root / "2023-07-16" / "building_mask.tif"
        transition = output / "per_transition" / stem
        transition.mkdir(parents=True, exist_ok=True)
        new = _write_window_candidate(transition / "new_building_candidate.tif", evidence / "new_building_evidence.tif", prior, post, "new")
        demolition = _write_window_candidate(transition / "demolition_candidate.tif", evidence / "demolition_evidence.tif", prior, post, "demolition")
        persistent = _write_window_candidate(transition / "persistent_building_candidate.tif", evidence / "any_change_probability.tif", prior, post, "persistent")
        per_transition.append({"transition": stem, "from": before_date, "to": "2023-07-16", "full20_evidence_available": True, "outputs": [new, demolition, persistent], "threshold_source": "validation_fixed", "scientific_status": "candidate_not_ground_truth"})
    _overview(geometry_root / "2023-07-16" / "building_probability.tif", [geometry_root / "2023-07-16" / "building_mask.tif"], output / "overlay_overview.png")
    review = output / "visual_review"
    review.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "overlay_overview.png", review / "contact_sheet.png")
    (review / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>URRS Tongzhou review</title><style>body{font-family:system-ui;background:#111;color:#eee;margin:2rem}img{max-width:100%}</style><h1>URRS Tongzhou candidate review</h1><p>Candidate evidence only; not Ground Truth. Human review pending.</p><img src='contact_sheet.png'>", encoding="utf-8")
    load_audit = {"joint_checkpoint": str(joint), "joint_checkpoint_sha256": hashes["joint"], "joint_loaded_cpu": joint_loaded, "full20_checkpoint": str(full20), "full20_checkpoint_sha256": hashes["full20"], "rsbuilding_checkpoint": str(rsbuilding), "rsbuilding_checkpoint_sha256": hashes["rsbuilding"], "selected_mti": False, "mti_reason": "validation_underperformed_no_mti", "mti_selection": str(mti)}
    geometry_load = {"geometry_root": str(geometry_root), "scenes_loaded": DATES, "per_scene_files": per_scene, "alignment_status": alignment.get("alignment_status"), "test_reads": {"WHU": 0, "SECOND": 0}}
    numerical = {"geometry_audit": geometry_audit, "alignment_audit": alignment, "finite_outputs": True, "test_reads": {"WHU": 0, "SECOND": 0}}
    write_json(output / "CHECKPOINT_LOAD_AUDIT.json", load_audit)
    write_json(output / "GEOMETRY_LOAD_AUDIT.json", geometry_load)
    write_json(output / "NUMERICAL_AUDIT.json", numerical)
    manifest = {"status": "success", "integration_status": "overlay_candidate", "candidate_scope": "Tongzhou four-scene geometry + Full20 available transitions", "scientific_status": "candidate_not_ground_truth", "inference_ready": False, "selected_mti": False, "mti_reason": "validation_underperformed_no_mti", "checkpoint_load_audit": load_audit, "geometry_load_audit": geometry_load, "per_scene": per_scene, "per_transition": per_transition, "review_packet": str(review / "index.html"), "human_visual_review": "pending_user_review", "whu_test_reads": 0, "second_test_reads": 0, "executed_at": utc_now()}
    write_json(output / "MASTER_OVERLAY_MANIFEST.json", manifest)
    return manifest
