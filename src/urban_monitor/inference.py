from __future__ import annotations

from pathlib import Path
import hashlib

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape
import pandas as pd

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
    rgb_bands: list[int] | tuple[int, ...] = (1, 2, 3),
    nms_iou_threshold: float = 0.4,
    source_date: str = "",
    min_size: int = 512,
    max_size: int = 1024,
    apply_regularization: bool = True,
) -> Path:
    # Geometry-only utilities in this module must remain usable in the CPU smoke
    # workflow without installing the optional teacher-model stack.
    import torch
    from torchvision.transforms.functional import to_tensor

    from .model import build_model, checkpoint_sha256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint=checkpoint, min_size=min_size, max_size=max_size).to(device).eval()
    stride = chip_size - overlap
    rows: list[dict] = []
    with rasterio.open(image_path) as src:
        raster_crs = src.crs
        for row_off in range(0, src.height, stride):
            for col_off in range(0, src.width, stride):
                width = min(chip_size, src.width - col_off)
                height = min(chip_size, src.height - row_off)
                if width < chip_size // 2 or height < chip_size // 2:
                    continue
                window = Window(col_off, row_off, width, height)
                if any(int(band) < 1 or int(band) > src.count for band in rgb_bands):
                    raise ValueError(f"RGB bands {list(rgb_bands)} exceed image band count {src.count}: {image_path}")
                data = src.read(list(rgb_bands), window=window)
                rgb = np.moveaxis(data, 0, -1)
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
                        raw_geometry = shape(geom_json)
                        geom = regularize_building_polygon(
                            raw_geometry,
                            simplify_tolerance=float((regularization or {}).get("simplify_tolerance_m", 0.35)),
                            orthogonalize=bool((regularization or {}).get("orthogonalize", True)),
                            rectangularity_threshold=float((regularization or {}).get("rectangularity_threshold", 0.82)),
                            min_area=float((regularization or {}).get("min_area_m2", 4.0)),
                            max_area_change_ratio=float((regularization or {}).get("max_area_change_ratio", 1.75)),
                            max_centroid_shift=float((regularization or {})["max_centroid_shift_m"]) if (regularization or {}).get("max_centroid_shift_m") is not None else None,
                            max_hausdorff_distance=float((regularization or {})["max_hausdorff_distance_m"]) if (regularization or {}).get("max_hausdorff_distance_m") is not None else None,
                        ) if apply_regularization else raw_geometry
                        if geom.is_empty or geom.area <= 0:
                            continue
                        rows.append(
                            {
                                "score": float(score),
                                "model_version": model_version,
                                "source_date": source_date,
                                "schema_version": "0.4.0",
                                "checkpoint_hash": checkpoint_sha256(checkpoint),
                                "geometry_regularized": apply_regularization,
                                "geometry": geom,
                            }
                        )
    if rows:
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=raster_crs)
        keep = greedy_polygon_nms(frame.geometry, frame["score"], iou_threshold=nms_iou_threshold)
        frame = frame.iloc[keep].reset_index(drop=True)
        frame["prediction_id"] = [
            "pred_" + hashlib.sha1(f"{source_date}|{geom.normalize().wkb_hex}|{score:.8f}".encode("utf-8")).hexdigest()[:16]
            for geom, score in zip(frame.geometry, frame["score"])
        ]
    else:
        frame = gpd.GeoDataFrame(
            columns=["prediction_id", "score", "source_date", "model_version", "checkpoint_hash", "schema_version", "geometry"],
            geometry="geometry",
            crs=raster_crs,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(output, driver="GPKG")
    return output


def infer_full_timeseries(
    prepared_dir: str | Path,
    dates: list[str],
    output_dir: str | Path,
    *,
    checkpoint: str | Path,
    chip_size: int,
    overlap: int,
    score_threshold: float,
    mask_threshold: float,
    nms_iou_threshold: float,
    model_version: str,
    rgb_bands: list[int] | tuple[int, ...] = (1, 2, 3),
    regularization: dict | None = None,
    min_size: int = 512,
    max_size: int = 1024,
) -> Path:
    from .model import checkpoint_sha256

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.name != "best.pth":
        # Explicit checkpoints are allowed by the CLI; automatic callers must pass best.pth.
        pass
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    raw_dir = output / "raw"
    regularized_dir = output / "regularized"
    exchange_dir = output / "exchange_wgs84"
    rows = []
    for date in sorted(dates):
        image_path = prepared / "images" / f"{date}.tif"
        raw_path = raw_dir / f"{date}.gpkg"
        infer_raster(
            image_path,
            raw_path,
            checkpoint=checkpoint_path,
            chip_size=chip_size,
            overlap=overlap,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            model_version=model_version,
            regularization={},
            rgb_bands=rgb_bands,
            nms_iou_threshold=nms_iou_threshold,
            source_date=date,
            min_size=min_size,
            max_size=max_size,
            apply_regularization=False,
        )
        raw = gpd.read_file(raw_path)
        regularized = regularize_prediction_frame(raw, regularization)
        regularized_path = regularized_dir / f"{date}.gpkg"
        regularized_path.parent.mkdir(parents=True, exist_ok=True)
        regularized.to_file(regularized_path, driver="GPKG")
        exchange = regularized.to_crs("EPSG:4326") if regularized.crs else regularized
        exchange_path = exchange_dir / f"{date}.gpkg"
        exchange_path.parent.mkdir(parents=True, exist_ok=True)
        exchange.to_file(exchange_path, driver="GPKG")
        rows.append(
            {
                "date": date,
                "raw_predictions": len(raw),
                "regularized_predictions": len(regularized),
                "checkpoint": str(checkpoint_path),
                "checkpoint_hash": checkpoint_sha256(checkpoint_path),
                "raw_path": str(raw_path),
                "regularized_path": str(regularized_path),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "inference_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False, encoding="utf-8-sig")
    return summary
