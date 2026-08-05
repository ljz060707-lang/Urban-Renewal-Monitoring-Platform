from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.surface_state_net import SurfaceStateNet
from ..models.tfr import TemporalFeatureRefinement
from ..training.checkpointing import sha256


def _read(path):
    path = Path(path)
    metadata = {"width": None, "height": None, "crs": None, "transform": None, "nodata": None}
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as source:
            array = source.read([1, 2, 3]).transpose(1, 2, 0)
            metadata.update(width=source.width, height=source.height, crs=str(source.crs) if source.crs else None, transform=tuple(source.transform), nodata=source.nodata)
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
            metadata.update(width=image.width, height=image.height)
    tensor = torch.from_numpy(array.astype(np.float32).transpose(2, 0, 1) / 255)
    return tensor, metadata


def alignment_guard(left: dict, right: dict) -> dict:
    keys = ("width", "height", "crs", "transform")
    differences = {key: [left.get(key), right.get(key)] for key in keys if left.get(key) != right.get(key)}
    return {
        "alignment_status": "registration_required_before_temporal_inference" if differences else "aligned",
        "differences": differences,
        "spatial_status": "pixel_coordinates_only" if left.get("crs") is None else "georeferenced",
    }


def _write_tif(path, data, metadata):
    import rasterio
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[None]
    kwargs = {"driver": "GTiff", "height": data.shape[1], "width": data.shape[2], "count": data.shape[0], "dtype": "float32"}
    if metadata.get("crs"):
        kwargs.update(crs=metadata["crs"], transform=metadata["transform"])
    with rasterio.open(path, "w", **kwargs) as target:
        target.write(data)


@torch.no_grad()
def overlay_infer(config: dict, *, repo: str | Path) -> dict:
    repo = Path(repo)
    output = Path(config["output"])
    if not output.is_absolute():
        output = repo / output
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(config["surface_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = repo / checkpoint
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SurfaceStateNet()
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    image_t1, metadata_t1 = _read(config["input_t1"])
    image_t2 = metadata_t2 = None
    guard = {"alignment_status": "single_image", "spatial_status": "pixel_coordinates_only" if metadata_t1["crs"] is None else "georeferenced", "differences": {}}
    if config.get("input_t2"):
        image_t2, metadata_t2 = _read(config["input_t2"])
        guard = alignment_guard(metadata_t1, metadata_t2)
        if guard["alignment_status"] != "aligned":
            manifest = {"status": "failed_alignment_guard", **guard}
            (output / "MASTER_OVERLAY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return manifest
    cwld = model.forward_cwld(image_t1[None])["probabilities"][0].numpy()
    fusu = model.forward_fusu(image_t1[None], image_t2[None] if image_t2 is not None else None)
    fusu_t1 = fusu["probabilities_t1"][0].numpy()
    _write_tif(output / "cwld_probabilities.tif", cwld, metadata_t1)
    _write_tif(output / "fusu_native_probabilities.tif", fusu_t1, metadata_t1)
    _write_tif(output / "vacant_land_candidate.tif", cwld[1], metadata_t1)
    _write_tif(output / "engineering_facility_candidate.tif", cwld[2], metadata_t1)
    _write_tif(output / "construction_waste_candidate.tif", cwld[3], metadata_t1)
    outputs = ["cwld_probabilities.tif", "fusu_native_probabilities.tif", "vacant_land_candidate.tif", "engineering_facility_candidate.tif", "construction_waste_candidate.tif"]
    if image_t2 is not None:
        fusu_t2 = fusu["probabilities_t2"][0].numpy()
        semantic_change = 1 - (fusu_t1 * fusu_t2).sum(0)
        _write_tif(output / "semantic_change_probability.tif", semantic_change, metadata_t1)
        _write_tif(output / "temporal_consistency.tif", 1 - semantic_change, metadata_t1)
        renewal = np.maximum.reduce([semantic_change, cwld[1], cwld[2], cwld[3]])
        _write_tif(output / "renewal_activity_candidate.tif", renewal, metadata_t1)
        outputs += ["semantic_change_probability.tif", "temporal_consistency.tif", "renewal_activity_candidate.tif"]
    overview = Image.fromarray((np.clip(cwld[[3, 2, 1]].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
    overview.save(output / "overlay_overview.png")
    outputs.append("overlay_overview.png")
    manifest = {
        "status": "success",
        "integration_status": "overlay_candidate",
        "candidate_scope": "fusu_500_sample_domain",
        "scientific_status": "candidate_not_ground_truth",
        "inference_ready": False,
        "surface_checkpoint": str(checkpoint.resolve()),
        "surface_checkpoint_sha256": sha256(checkpoint),
        "mti_selection": config.get("mti_selection"),
        "full20_evidence": config.get("full20_prediction"),
        "rsbuilding_evidence": config.get("rsbuilding_probability"),
        "new_building_output": "unsupported_without_verified_building_semantics",
        "demolition_output": "unsupported_without_verified_building_semantics",
        "outputs": outputs,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "whu_test_reads": 0,
        "second_test_reads": 0,
        **guard,
    }
    (output / "MASTER_OVERLAY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
