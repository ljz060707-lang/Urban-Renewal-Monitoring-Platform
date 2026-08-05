from __future__ import annotations

"""Load-only inference adapter for the frozen Construction task candidate.

This module deliberately has no training, resume, optimizer, scheduler, or
dataset-selection code.  It consumes a packaged candidate and an explicit
image manifest, and writes auditable probability rasters for one image.
"""

import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from urban_renewal.construction.json_io import dump_json, load_json
from urban_renewal.models.construction import ConstructionElementNet


CLASS_NAMES = (
    "ucs_background",
    "ucs_exposed_area",
    "ucs_dustproof_net",
    "ucs_building",
)
OUTPUT_NAMES = {
    "ucs_background": "background_probability.tif",
    "ucs_exposed_area": "exposed_ground_probability.tif",
    "ucs_dustproof_net": "dustproof_net_probability.tif",
    "ucs_building": "building_probability.tif",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_payload(package_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    package_dir = package_dir.resolve()
    manifest = load_json(package_dir / "TASK_MODEL_MANIFEST.json")
    config_path = package_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    ontology = load_json(package_dir / "ontology.json")
    preprocessing = load_json(package_dir / "preprocessing.json")
    thresholds = load_json(package_dir / "thresholds.json")
    return manifest, config, ontology, preprocessing, thresholds


def load_task_model(package_dir: Path, device: str | torch.device = "cpu") -> tuple[ConstructionElementNet, dict[str, Any]]:
    """Load the packaged model with strict architecture and ontology checks."""

    package_dir = package_dir.resolve()
    manifest, config, ontology, preprocessing, thresholds = _package_payload(package_dir)
    model_meta = config.get("model", {})
    if model_meta.get("architecture") != "ConstructionElementNet":
        raise ValueError(f"unsupported task-model architecture: {model_meta.get('architecture')!r}")
    width = int(model_meta.get("width", 24))
    class_names = tuple(ontology.get("class_names", []))
    if class_names != CLASS_NAMES:
        raise ValueError(f"packaged class order mismatch: {class_names!r}")
    rgb_mapping = tuple(int(value) for value in preprocessing.get("rgb_mapping", []))
    if rgb_mapping != (3, 2, 1):
        raise ValueError(f"packaged RGB mapping mismatch: {rgb_mapping!r}")
    if preprocessing.get("normalization") != "per_image_percentile_2_98":
        raise ValueError("packaged preprocessing normalization is not the frozen UCS preprocessing")
    if not thresholds.get("frozen", False):
        raise ValueError("task-model thresholds are not frozen")

    checkpoint = package_dir / "model.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("packaged model.pth has no model_state_dict")
    model = ConstructionElementNet(width=width)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "packaged model state mismatch: "
            f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}"
        )
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    context = {
        "manifest": manifest,
        "config": config,
        "ontology": ontology,
        "preprocessing": preprocessing,
        "thresholds": thresholds,
        "checkpoint_sha256": _sha256(checkpoint),
        "package_dir": str(package_dir),
        "device": str(target_device),
    }
    return model, context


def _read_manifest_row(manifest_path: Path, sample_id: str | None) -> dict[str, str]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty inference manifest: {manifest_path}")
    row = next((item for item in rows if sample_id is None or item.get("sample_id") == str(sample_id)), None)
    if row is None:
        raise KeyError(f"sample_id={sample_id!r} not found in {manifest_path}")
    if str(row.get("split", "")).casefold() == "test":
        raise RuntimeError("UCS Locked Test pixels are not available to task-model inference")
    if not row.get("image_path"):
        raise ValueError(f"inference row has no image_path: {row}")
    return row


def _prepare_image(image_path: Path, rgb_mapping: tuple[int, int, int], image_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    import rasterio

    with rasterio.open(image_path) as source:
        if max(rgb_mapping) > source.count:
            raise ValueError(f"rgb_mapping={rgb_mapping} exceeds source band count={source.count}")
        raw = source.read(rgb_mapping).astype(np.float32)
        source_profile = source.profile.copy()
        source_transform = source.transform
        source_crs = source.crs.to_string() if source.crs else None
        source_shape = [int(source.height), int(source.width)]
    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError(f"image contains no finite pixels: {image_path}")
    lo = float(np.nanpercentile(raw[finite], 2.0))
    hi = float(np.nanpercentile(raw[finite], 98.0))
    normalized = np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    image = F.interpolate(torch.from_numpy(normalized)[None], size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
    gray = (image * image.new_tensor([0.2989, 0.5870, 0.1140])[:, None, None]).sum(0, keepdim=True)
    metadata = {
        "source_path": str(image_path.resolve()),
        "source_shape": source_shape,
        "output_shape": [int(image_size), int(image_size)],
        "source_profile": source_profile,
        "source_transform": source_transform,
        "source_crs": source_crs,
        "normalization": "per_image_percentile_2_98",
        "normalization_low": lo,
        "normalization_high": hi,
    }
    return torch.cat([image, gray], dim=0), metadata


def _write_probability(path: Path, array: np.ndarray, metadata: Mapping[str, Any]) -> None:
    import rasterio
    from rasterio.transform import Affine

    source_profile = dict(metadata["source_profile"])
    source_height, source_width = metadata["source_shape"]
    output_height, output_width = metadata["output_shape"]
    transform = metadata["source_transform"] * Affine.scale(source_width / output_width, source_height / output_height)
    profile = {
        "driver": "GTiff",
        "height": output_height,
        "width": output_width,
        "count": 1,
        "dtype": "float32",
        "crs": source_profile.get("crs"),
        "transform": transform,
        "compress": "deflate",
        "predictor": 3,
        "nodata": None,
    }
    finite = np.asarray(array, dtype=np.float32)
    if not np.isfinite(finite).all():
        raise ValueError(f"non-finite probability output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as target:
        target.write(finite, 1)


def infer_task_model(package_dir: Path, manifest: Path, output_dir: Path, *, sample_id: str | None = None, device: str = "cuda") -> dict[str, Any]:
    """Run inference for one non-Test image and write the standard six outputs."""

    package_dir = package_dir.resolve()
    manifest = manifest.resolve()
    output_dir = output_dir.resolve()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for task-model inference but CUDA is unavailable")
    model, context = load_task_model(package_dir, device=device)
    preprocessing = context["preprocessing"]
    rgb_mapping = tuple(int(value) for value in preprocessing["rgb_mapping"])
    image_size = int(preprocessing.get("image_size", 256))
    row = _read_manifest_row(manifest, sample_id)
    input_tensor, image_metadata = _prepare_image(Path(row["image_path"]), rgb_mapping, image_size)
    image = input_tensor[0:3][None].to(device)
    texture = input_tensor[3:4][None].to(device)
    with torch.no_grad():
        output = model(image, "ucs", texture)
        class_probabilities = output["ucs_probabilities"].float()
        boundary_probability = output["boundary_logits"].float().sigmoid()
        uncertainty_probability = output["uncertainty_logits"].float().sigmoid()
    outputs = {
        "background_probability": class_probabilities[0, 0].detach().cpu().numpy(),
        "exposed_ground_probability": class_probabilities[0, 1].detach().cpu().numpy(),
        "dustproof_net_probability": class_probabilities[0, 2].detach().cpu().numpy(),
        "building_probability": class_probabilities[0, 3].detach().cpu().numpy(),
        "boundary_probability": boundary_probability[0, 0].detach().cpu().numpy(),
        "uncertainty_probability": uncertainty_probability[0, 0].detach().cpu().numpy(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for key, array in outputs.items():
        filename = OUTPUT_NAMES.get(key, f"{key}.tif")
        path = output_dir / filename
        _write_probability(path, array, image_metadata)
        output_paths[key] = str(path)
    all_arrays_finite = all(np.isfinite(array).all() for array in outputs.values())
    receipt = {
        "status": "success" if all_arrays_finite else "failed_nonfinite_output",
        "model_name": context["manifest"].get("model_name"),
        "package_dir": str(package_dir),
        "package_model_sha256": context["checkpoint_sha256"],
        "input_manifest": str(manifest),
        "sample_id": row.get("sample_id"),
        "scene_id": row.get("scene_id", row.get("sample_id")),
        "input_image": str(Path(row["image_path"]).resolve()),
        "class_order": list(CLASS_NAMES),
        "rgb_mapping": list(rgb_mapping),
        "preprocessing": {
            "image_size": image_size,
            "normalization": "per_image_percentile_2_98",
        },
        "output_paths": output_paths,
        "output_shape": image_metadata["output_shape"],
        "all_outputs_finite": bool(all_arrays_finite),
        "test_pixel_reads": 0,
        "locked_test_sample_reads": 0,
        "human_review_required": True,
        "generated_by": "urban_renewal.construction.task_model",
    }
    dump_json(output_dir / "CONSTRUCTION_ELEMENT_MANIFEST.json", receipt)
    return receipt
