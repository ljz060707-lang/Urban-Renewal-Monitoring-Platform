"""Run auditable static Building Foundation v2 inference on one raster.

The command never traverses a Test directory and does not invent a CRS.  For
non-georeferenced WHU tiles it writes pixel-coordinate NPZ/PNG outputs and
marks vectorization as skipped instead of manufacturing geographic objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if "test" in str(args.image).lower():
        raise SystemExit("Building v2 inference refuses paths containing Test")
    if not args.checkpoint.exists() or not args.image.exists():
        raise SystemExit("checkpoint and image must exist")
    if not 0.0 < args.threshold < 1.0:
        raise SystemExit("threshold must be in (0,1)")

    import rasterio  # type: ignore
    from urban_monitor.building_foundation_v2 import BuildingFoundationNet

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model = BuildingFoundationNet(base_channels=16, height_enabled=bool(config.get("height_enabled", False))).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with rasterio.open(args.image) as source:
        image = source.read(out_shape=(3, source.height, source.width), resampling=rasterio.enums.Resampling.bilinear)
        profile = source.profile.copy()
        crs = source.crs
        transform = source.transform
        width, height = source.width, source.height
        nodata = source.nodata
    image = np.asarray(image, dtype="float32")
    finite = np.isfinite(image)
    valid = finite.all(axis=0)
    if nodata is not None:
        valid &= np.all(image != nodata, axis=0)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    scale = 255.0 if float(np.nanmax(image)) > 1.5 else 1.0
    tensor = torch.from_numpy(np.clip(image / scale, 0.0, 1.0)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    arrays: dict[str, np.ndarray] = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            arrays[key] = value.detach().cpu().numpy()[0, 0].astype("float32")
    probability = np.clip(arrays["building_probability"], 0.0, 1.0)
    mask = ((probability >= args.threshold) & valid).astype("uint8")
    boundary = (arrays["boundary_probability"] >= args.threshold).astype("uint8") * valid.astype("uint8")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "static_outputs.npz", probability=probability, mask=mask, boundary=boundary, distance=arrays.get("distance"), quality=arrays.get("quality_logits"), uncertainty=arrays.get("uncertainty_logits"))
    Image.fromarray((probability * 255).astype("uint8")).save(args.output / "building_probability.png")
    Image.fromarray(mask * 255).save(args.output / "building_mask.png")
    Image.fromarray(boundary * 255).save(args.output / "building_boundary.png")

    geotiff_outputs: list[str] = []
    vectorization_status = "skipped_no_crs"
    if crs:
        profile.update(driver="GTiff", count=1, dtype="float32", height=height, width=width, transform=transform, crs=crs, nodata=None)
        for name, values, dtype in (("building_probability.tif", probability, "float32"), ("building_mask.tif", mask, "uint8"), ("building_boundary.tif", boundary, "uint8")):
            local = profile.copy(); local.update(dtype=dtype, nodata=255 if dtype == "uint8" else None)
            data = values.copy()
            if dtype == "uint8": data[~valid] = 255
            with rasterio.open(args.output / name, "w", **local) as target:
                target.write(data.astype(dtype), 1)
            geotiff_outputs.append(name)
        try:
            import geopandas as gpd  # type: ignore
            from rasterio.features import shapes  # type: ignore
            from shapely.geometry import shape  # type: ignore
            geometries = [shape(geom) for geom, value in shapes(mask.astype("uint8"), mask=mask == 1, transform=transform) if value == 1]
            if geometries:
                frame = gpd.GeoDataFrame({"object_id": [f"building_{i:06d}" for i in range(len(geometries))], "state": "building_candidate", "scientific_status": "candidate_not_ground_truth"}, geometry=geometries, crs=crs)
                frame["area_m2"] = frame.geometry.area
                frame["perimeter_m"] = frame.geometry.length
                frame.to_file(args.output / "building_objects.gpkg", layer="buildings", driver="GPKG")
                vectorization_status = "written"
            else:
                vectorization_status = "empty_mask"
        except Exception as exc:
            vectorization_status = f"failed:{type(exc).__name__}"
    manifest = {
        "status": "success", "source_image": str(args.image.resolve()), "source_sha256": sha256(args.image),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "threshold": args.threshold, "width": width, "height": height, "crs": str(crs) if crs else None,
        "transform": list(transform) if crs else None, "nodata": nodata, "valid_pixel_fraction": float(valid.mean()),
        "positive_fraction": float(mask.mean()), "finite": bool(np.isfinite(probability).all()),
        "spatial_status": "georeferenced" if crs else "pixel_coordinates_only", "geotiff_outputs": geotiff_outputs,
        "vectorization_status": vectorization_status,
        "scientific_status": "candidate_not_ground_truth", "test_used": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output / "STATIC_OUTPUT_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
