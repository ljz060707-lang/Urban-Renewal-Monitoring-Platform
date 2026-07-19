from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape

from .geometry import greedy_polygon_nms, regularize_building_polygon


def _scale_to_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def regularize_prediction_frame(frame: gpd.GeoDataFrame, config: dict | None = None) -> gpd.GeoDataFrame:
    config = config or {}
    if frame.empty:
        return frame
    out = frame.copy()
    out["geometry_raw_area"] = out.geometry.area.astype(float)
    out.geometry = [
        regularize_building_polygon(
            geom,
            simplify_tolerance=float(config.get("simplify_tolerance_m", 0.35)),
            orthogonalize=bool(config.get("orthogonalize", True)),
            rectangularity_threshold=float(config.get("rectangularity_threshold", 0.82)),
            min_area=float(config.get("min_area_m2", 4.0)),
            max_area_change_ratio=float(config.get("max_area_change_ratio", 1.75)),
            max_centroid_shift=float(config["max_centroid_shift_m"]) if config.get("max_centroid_shift_m") is not None else None,
            max_hausdorff_distance=float(config["max_hausdorff_distance_m"]) if config.get("max_hausdorff_distance_m") is not None else None,
        )
        for geom in out.geometry
    ]
    out["geometry_regularized"] = True
    out["regularized_area"] = out.geometry.area.astype(float)
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    return out


def infer_raster(
    image_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint: str | Path,
    chip_size: int,
    overlap: int,
    score_threshold: float,
    mask_threshold: float,
    model_version: str,
    regularization: dict | None = None,
) -> Path:
    # Geometry-only utilities in this module must remain usable in the CPU smoke
    # workflow without installing the optional teacher-model stack.
    import torch
    from torchvision.transforms.functional import to_tensor

    from .model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint=checkpoint).to(device).eval()
    stride = chip_size - overlap
    rows: list[dict] = []
    with rasterio.open(image_path) as src:
        for row_off in range(0, src.height, stride):
            for col_off in range(0, src.width, stride):
                width = min(chip_size, src.width - col_off)
                height = min(chip_size, src.height - row_off)
                if width < chip_size // 2 or height < chip_size // 2:
                    continue
                window = Window(col_off, row_off, width, height)
                data = src.read(window=window)
                rgb = np.moveaxis(data[:3] if data.shape[0] >= 3 else np.repeat(data[:1], 3, axis=0), 0, -1)
                rgb = _scale_to_uint8(rgb)
                tensor = to_tensor(rgb).to(device)
                with torch.inference_mode():
                    pred = model([tensor])[0]
                scores = pred["scores"].detach().cpu().numpy()
                masks = pred["masks"].detach().cpu().numpy()[:, 0]
                transform = src.window_transform(window)
                for score, mask in zip(scores, masks):
                    if float(score) < score_threshold:
                        continue
                    binary = mask >= mask_threshold
                    if binary.sum() < 8:
                        continue
                    for geom_json, value in shapes(binary.astype(np.uint8), mask=binary, transform=transform):
                        if int(value) != 1:
                            continue
                        geom = regularize_building_polygon(
                            shape(geom_json),
                            simplify_tolerance=float((regularization or {}).get("simplify_tolerance_m", 0.35)),
                            orthogonalize=bool((regularization or {}).get("orthogonalize", True)),
                            rectangularity_threshold=float((regularization or {}).get("rectangularity_threshold", 0.82)),
                            min_area=float((regularization or {}).get("min_area_m2", 4.0)),
                            max_area_change_ratio=float((regularization or {}).get("max_area_change_ratio", 1.75)),
                            max_centroid_shift=float((regularization or {})["max_centroid_shift_m"]) if (regularization or {}).get("max_centroid_shift_m") is not None else None,
                            max_hausdorff_distance=float((regularization or {})["max_hausdorff_distance_m"]) if (regularization or {}).get("max_hausdorff_distance_m") is not None else None,
                        )
                        if geom.is_empty or geom.area <= 0:
                            continue
                        rows.append(
                            {
                                "score": float(score),
                                "model_version": model_version,
                                "geometry_regularized": True,
                                "geometry": geom,
                            }
                        )
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=src.crs)
    if not frame.empty:
        keep = greedy_polygon_nms(frame.geometry, frame["score"], iou_threshold=0.5)
        frame = frame.iloc[keep].reset_index(drop=True)
        frame["prediction_id"] = [f"p{i:07d}" for i in range(len(frame))]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(output, driver="GPKG")
    return output
