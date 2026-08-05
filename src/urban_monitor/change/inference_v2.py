from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .fusion_v2 import fuse_teacher_outputs
from .geospatial import (
    aligned_pair,
    blending_weight,
    grid_from_dataset,
    resolution_interpretation,
    tile_windows,
    write_float_raster,
    write_mask_raster,
)
from .models.siamese_unet import build_change_model
from .mudi_v2 import write_mudi_features
from .preview_v2 import build_preview
from .teacher_registry import build_teacher_registry, sha256_file
from .vectorize_v2 import vectorize_mask


def _checkpoint_state(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def _read_rgb(dataset, window) -> np.ndarray:
    bands = list(range(1, min(3, dataset.count) + 1))
    array = dataset.read(bands, window=window, boundless=False).astype(np.float32)
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    elif array.shape[0] == 2:
        array = np.concatenate([array, array[-1:]], axis=0)
    for channel in range(3):
        values = array[channel]
        finite = values[np.isfinite(values)]
        if not finite.size:
            array[channel] = 0
            continue
        low, high = np.percentile(finite, (2, 98))
        array[channel] = np.clip((values - low) / max(float(high - low), 1e-6), 0, 1)
    return array


def _load_model(config: dict, teacher: str, checkpoint: Path, device: str):
    import torch

    teacher_values = config.get("teachers", {}).get(teacher, {})
    model_values = teacher_values.get("model", config.get("model", {}))
    model = build_change_model({"model": model_values})
    state = _checkpoint_state(checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval().to(torch.device(device))
    return model


def _predict_checkpoint(
    t1,
    t2,
    *,
    checkpoint: Path,
    teacher: str,
    output_names: dict[str, str],
    config: dict,
    device: str,
    tile_size: int,
    overlap: int,
) -> dict[str, np.ndarray]:
    import torch

    model = _load_model(config, teacher, checkpoint, device)
    sums = {name: np.zeros((t1.height, t1.width), dtype=np.float32) for name in output_names}
    weights = np.zeros((t1.height, t1.width), dtype=np.float32)
    with torch.inference_mode():
        for window in tile_windows(t1.width, t1.height, tile_size, overlap):
            first = torch.from_numpy(_read_rgb(t1, window)[None]).to(device)
            second = torch.from_numpy(_read_rgb(t2, window)[None]).to(device)
            result = model(first, second)
            height, width = int(window.height), int(window.width)
            weight = blending_weight(height, width)
            row, col = int(window.row_off), int(window.col_off)
            weights[row : row + height, col : col + width] += weight
            for output_name, result_name in output_names.items():
                probability = torch.sigmoid(result[result_name])[0, 0].float().cpu().numpy()
                sums[output_name][row : row + height, col : col + width] += probability[:height, :width] * weight
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {name: value / np.maximum(weights, 1e-6) for name, value in sums.items()}


def _quicklook(dataset) -> Image.Image:
    from rasterio.enums import Resampling

    scale = min(1.0, 1024.0 / max(dataset.width, dataset.height))
    width, height = max(1, int(dataset.width * scale)), max(1, int(dataset.height * scale))
    bands = list(range(1, min(3, dataset.count) + 1))
    array = dataset.read(bands, out_shape=(len(bands), height, width), resampling=Resampling.bilinear).astype(np.float32)
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    elif array.shape[0] == 2:
        array = np.concatenate([array, array[-1:]], axis=0)
    for channel in range(3):
        low, high = np.percentile(array[channel][np.isfinite(array[channel])], (2, 98))
        array[channel] = np.clip((array[channel] - low) / max(float(high - low), 1e-6), 0, 1)
    return Image.fromarray((np.moveaxis(array, 0, -1) * 255).astype(np.uint8))


def run_urban_renewal_inference(
    config: dict,
    *,
    t1_path: str | Path,
    t2_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
) -> dict:
    output = Path(output_dir)
    rasters, vectors = output / "rasters", output / "vectors"
    output.mkdir(parents=True, exist_ok=True)
    registry = build_teacher_registry(config)
    statuses = registry.statuses()
    tiling = config.get("inference", {})
    tile_size, overlap = int(tiling.get("tile_size", 256)), int(tiling.get("overlap", 64))
    thresholds = config.get("thresholds", {"any_change": 0.75, "new_building": 0.6, "demolition": 0.35})
    warnings: list[str] = []
    with aligned_pair(t1_path, t2_path) as (t1, t2, reprojected):
        grid = grid_from_dataset(t1)
        probabilities: dict[str, np.ndarray] = {}

        # ---------------------------------------------------------
        # Formal V2 Full20 inference
        #
        # Development/source checkpoints (DSIFN/S2Looking) are
        # initialization/training sources. For external imagery,
        # prefer the frozen validation-selected V2 multitask model.
        # ---------------------------------------------------------

        inference_cfg = config.get("inference", {})
        full20_value = inference_cfg.get("checkpoint")

        if full20_value:
            checkpoint = Path(full20_value)

            if not checkpoint.is_absolute():
                checkpoint = checkpoint.resolve()

            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"V2 Full20 inference checkpoint not found: "
                    f"{checkpoint}"
                )

            probabilities.update(
                _predict_checkpoint(
                    t1,
                    t2,
                    checkpoint=checkpoint,

                    # Current loader's s2looking branch builds the
                    # full config semantic dimension (7 classes).
                    # This is a loader hint only; checkpoint is V2.
                    teacher="s2looking",

                    output_names={
                        "any_change": "binary_change",
                        "new_building": "new_building",
                        "demolition": "demolition",
                    },

                    config=config,
                    device=device,
                    tile_size=tile_size,
                    overlap=overlap,
                )
            )

        else:
            # Legacy source-teacher mode retained only as fallback.
            if registry.get("dsifn").available():
                checkpoint = Path(
                    registry
                    .get("dsifn")
                    .metadata()
                    .checkpoint
                )

                probabilities.update(
                    _predict_checkpoint(
                        t1,
                        t2,
                        checkpoint=checkpoint,
                        teacher="dsifn",
                        output_names={
                            "any_change": "binary_change",
                        },
                        config=config,
                        device=device,
                        tile_size=tile_size,
                        overlap=overlap,
                    )
                )

            if registry.get("s2looking").available():
                checkpoint = Path(
                    registry
                    .get("s2looking")
                    .metadata()
                    .checkpoint
                )

                probabilities.update(
                    _predict_checkpoint(
                        t1,
                        t2,
                        checkpoint=checkpoint,
                        teacher="s2looking",
                        output_names={
                            "building_change": "binary_change",
                            "new_building": "new_building",
                            "demolition": "demolition",
                        },
                        config=config,
                        device=device,
                        tile_size=tile_size,
                        overlap=overlap,
                    )
                )

        fused = fuse_teacher_outputs(
            any_change=probabilities.get("any_change"),
            new_building=probabilities.get("new_building"),
            demolition=probabilities.get("demolition"),
        )
        written = {}
        for name, array in fused.outputs.items():
            written[name] = str(write_float_raster(rasters / f"{name}.tif", array, t1).resolve())
        masks = {}
        for name, source_name, threshold_name in (
            ("any_change_mask", "any_change_probability", "any_change"),
            ("new_building_mask", "new_building_candidate", "new_building"),
            ("demolition_mask", "demolition_candidate", "demolition"),
        ):
            if source_name not in fused.outputs:
                continue
            mask = fused.outputs[source_name] >= float(thresholds[threshold_name])
            path = write_mask_raster(rasters / f"{name}.tif", mask, t1)
            masks[name] = str(path.resolve())
        vector_reports = {}
        for mask_name, class_name in (("new_building_mask", "new_building"), ("demolition_mask", "demolition")):
            if mask_name in masks:
                vector_reports[class_name] = vectorize_mask(
                    masks[mask_name],
                    vectors / f"{class_name}.gpkg",
                    layer=class_name,
                    class_name=class_name,
                    min_mapping_unit_m2=float(config.get("vectorization", {}).get("min_mapping_unit_m2", 10)),
                )
        preview, index = build_preview(
            output,
            t1=_quicklook(t1),
            t2=_quicklook(t2),
            outputs=fused.outputs,
            pending=fused.pending_outputs,
            teacher_status=statuses,
        )
        resolution = max(grid.resolution)
        mudi = write_mudi_features(
            output / "MUDI_BASE_FEATURES.csv",
            pixel_area_m2=grid.resolution[0] * grid.resolution[1],
            pixel_count=grid.width * grid.height,
            outputs=fused.outputs,
            teacher_status=statuses,
        )
    manifest = {
        "status": "success",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "input_t1": str(Path(t1_path).resolve()),
        "input_t2": str(Path(t2_path).resolve()),
        "input_t1_sha256": sha256_file(t1_path),
        "input_t2_sha256": sha256_file(t2_path),
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "spatial_resolution": grid.resolution,
        "recommended_interpretation": resolution_interpretation(resolution),
        "t2_aligned_to_t1": reprojected,
        "teachers": statuses,
        "thresholds": thresholds,
        "fusion_version": "v2",
        "outputs": {**written, **masks, "preview": str(preview.resolve()), "index": str(index.resolve()), "mudi": str(mudi.resolve())},
        "vectors": vector_reports,
        "pending_outputs": fused.pending_outputs,
        "warnings": warnings + fused.warnings,
        "test_images_read": 0,
    }
    manifest_path = output / "INFERENCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
