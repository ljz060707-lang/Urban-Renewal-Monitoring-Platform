from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from .crs import ensure_metric_crs, ensure_same_crs


BANDS = ["new_fraction", "demolition_fraction", "construction_fraction", "any_change_fraction"]


def _aligned_grid(bounds, resolution: float):
    minx, miny, maxx, maxy = bounds
    minx = np.floor(minx / resolution) * resolution
    miny = np.floor(miny / resolution) * resolution
    maxx = np.ceil(maxx / resolution) * resolution
    maxy = np.ceil(maxy / resolution) * resolution
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)
    return transform, width, height


def _reference_aligned_grid(reference, resolution: float):
    left, top = reference.transform.c, reference.transform.f
    width_m = reference.width * abs(reference.transform.a)
    height_m = reference.height * abs(reference.transform.e)
    width = int(np.ceil(width_m / resolution))
    height = int(np.ceil(height_m / resolution))
    return from_origin(left, top, resolution, resolution), width, height


def _fractional_raster(geometries: Iterable, transform, width: int, height: int, supersample: int) -> np.ndarray:
    geoms = [g for g in geometries if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros((height, width), dtype=np.float32)
    hi_transform = from_origin(transform.c, transform.f, transform.a / supersample, -transform.e / supersample)
    hi = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=(height * supersample, width * supersample),
        transform=hi_transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return hi.reshape(height, supersample, width, supersample).mean(axis=(1, 3)).astype(np.float32)


def export_student_change_labels(
    summary_path: str | Path,
    intervals_path: str | Path,
    reference_image_path: str | Path,
    output_dir: str | Path,
    *,
    target_resolution_m: float = 10.0,
    min_mapping_unit_m2: float = 200.0,
    supersample: int = 4,
    min_fraction_to_keep: float = 0.05,
    require_reviewed_or_high_confidence: bool = True,
    model_version: str = "",
    schema_version: str = "0.3.0",
    run_id: str = "",
    source_aoi: str = "",
) -> list[Path]:
    """Export MMU-filtered, fractional 10 m labels for a later student model."""
    summary = gpd.read_file(summary_path)
    intervals = pd.read_csv(intervals_path, dtype=str, encoding="utf-8-sig").fillna("")
    frame = summary.merge(intervals[["track_id", "interval_end", "earliest_absent_date"]], on="track_id", how="left")
    ensure_metric_crs(frame.crs)
    frame["event_area_m2"] = frame.geometry.area.astype(float)
    frame["event_date"] = np.where(
        frame["state"].eq("demolished"),
        frame["earliest_absent_date"],
        frame["interval_end"],
    )
    frame = frame[frame["state"].isin(["new_building", "demolished", "construction_candidate"])]
    excluded_by_mmu = frame[frame["event_area_m2"] < min_mapping_unit_m2].copy()
    eligible = frame[frame["event_area_m2"] >= min_mapping_unit_m2].copy()
    excluded_missing_date = eligible[eligible["event_date"].astype(str).str.len() < 10].copy()
    eligible = eligible[eligible["event_date"].astype(str).str.len() >= 10].copy()
    if require_reviewed_or_high_confidence:
        confidence = eligible.get("confidence", pd.Series("low", index=eligible.index)).astype(str)
        review = eligible.get("review_status", pd.Series("", index=eligible.index)).astype(str)
        accepted = confidence.isin({"high", "medium"}) | review.eq("reviewed")
        excluded_low_confidence = eligible[~accepted].copy()
        frame = eligible[accepted].copy()
    else:
        excluded_low_confidence = eligible.iloc[0:0].copy()
        frame = eligible

    with rasterio.open(reference_image_path) as ref:
        crs = ref.crs
        ensure_metric_crs(crs)
        transform, width, height = _reference_aligned_grid(ref, target_resolution_m)
    if frame.crs != crs:
        frame = frame.to_crs(crs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    inventory_rows: list[dict] = []
    for event_date, group in frame.groupby("event_date"):
        arrays = []
        for state in ["new_building", "demolished", "construction_candidate"]:
            arr = _fractional_raster(group.loc[group["state"] == state, "geometry"], transform, width, height, supersample)
            arr[arr < min_fraction_to_keep] = 0.0
            arrays.append(arr)
        any_change = np.clip(np.maximum.reduce(arrays), 0.0, 1.0)
        stack = np.stack(arrays + [any_change]).astype(np.float32)
        out_path = output_dir / f"{event_date}.tif"
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 4,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "nodata": 0.0,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(stack)
            for idx, name in enumerate(BANDS, start=1):
                dst.set_band_description(idx, name)
            dst.update_tags(
                model_version=model_version,
                schema_version=schema_version,
                mmu_m2=str(min_mapping_unit_m2),
                target_resolution_m=str(target_resolution_m),
                source_aoi=source_aoi,
                analysis_crs=str(crs),
                creation_time=pd.Timestamp.now(tz="UTC").isoformat(),
                run_id=run_id,
            )
        outputs.append(out_path)
        inventory_rows.append(
            {
                "event_date": event_date,
                "n_events": len(group),
                "event_area_m2": float(group["event_area_m2"].sum()),
                "target_resolution_m": target_resolution_m,
                "min_mapping_unit_m2": min_mapping_unit_m2,
                "path": str(out_path),
            }
        )
    columns = ["track_id", "state", "event_area_m2", "event_date", "confidence", "review_status"]
    for excluded, name in [
        (excluded_by_mmu, "excluded_by_mmu.csv"),
        (excluded_missing_date, "excluded_missing_date.csv"),
        (excluded_low_confidence, "excluded_low_confidence.csv"),
    ]:
        excluded[[column for column in columns if column in excluded.columns]].to_csv(
            output_dir / name, index=False, encoding="utf-8-sig"
        )
    pd.DataFrame(inventory_rows).to_csv(output_dir / "label_inventory.csv", index=False, encoding="utf-8-sig")
    return outputs
