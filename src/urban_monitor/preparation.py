from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from shapely.geometry import box

from .crs import select_local_utm_crs, validate_source_crs
from .ground_truth import GroundTruthResult, normalize_label_file, write_ground_truth_report
from .project_discovery import DiscoveredProject
from .coco import build_spatial_coco
from .splitting import build_spatial_blocks, write_split_report


def _image_info(date: str, path: Path, analysis_crs: CRS) -> dict[str, Any]:
    with rasterio.open(path) as source:
        source_crs = validate_source_crs(source.crs)
        bounds = transform_bounds(source_crs, analysis_crs, *source.bounds, densify_pts=21)
        mask = source.dataset_mask(out_shape=(min(source.height, 256), min(source.width, 256)))
        nodata_fraction = float(np.mean(mask == 0))
        resolution = float(np.sqrt(abs(source.transform.a * source.transform.e)))
        return {
            "date": date,
            "path": path,
            "source_crs": source_crs,
            "analysis_bounds": bounds,
            "coverage_area": box(*bounds).area,
            "nodata_fraction": nodata_fraction,
            "resolution": resolution,
            "count": source.count,
            "dtype": source.dtypes[0],
        }


def _analysis_crs(project: DiscoveredProject) -> CRS:
    footprints = []
    for path in project.images.values():
        with rasterio.open(path) as source:
            source_crs = validate_source_crs(source.crs)
            bounds = transform_bounds(source_crs, "EPSG:4326", *source.bounds, densify_pts=21)
            footprints.append(box(*bounds))
    aoi = gpd.GeoSeries(footprints, crs="EPSG:4326").union_all()
    return select_local_utm_crs(aoi, "EPSG:4326")


def _reference_grid(reference: dict[str, Any], analysis_crs: CRS) -> dict[str, Any]:
    with rasterio.open(reference["path"]) as source:
        transform, width, height = calculate_default_transform(
            source.crs,
            analysis_crs,
            source.width,
            source.height,
            *source.bounds,
        )
    return {
        "transform": transform,
        "width": int(width),
        "height": int(height),
        "crs": analysis_crs,
        "reference_date": reference["date"],
        "reference_source": str(reference["path"]),
    }


def _warp(source_path: Path, output_path: Path, grid: dict[str, Any], resampling: Resampling) -> Path:
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        profile.update(
            crs=grid["crs"],
            transform=grid["transform"],
            width=grid["width"],
            height=grid["height"],
            compress="deflate",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as target:
            for band in range(1, source.count + 1):
                reproject(
                    source=rasterio.band(source, band),
                    destination=rasterio.band(target, band),
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=grid["transform"],
                    dst_crs=grid["crs"],
                    resampling=resampling,
                )
    return output_path


def prepare_project(project: DiscoveredProject, run_dir: str | Path) -> dict[str, Any]:
    """Create aligned working copies and normalized labels without changing originals."""
    run = Path(run_dir)
    prepared = run / "prepared"
    data_report = run / "data_report"
    analysis_crs = _analysis_crs(project)

    for date, path in project.labels.items():
        label_crs = gpd.read_file(path, rows=1).crs
        if label_crs is None:
            raise ValueError(f"Ground Truth CRS is missing: {path}")

    infos = [_image_info(date, path, analysis_crs) for date, path in sorted(project.images.items())]
    common = box(*infos[0]["analysis_bounds"])
    for info in infos[1:]:
        common = common.intersection(box(*info["analysis_bounds"]))
    if common.is_empty or common.area <= 0:
        raise ValueError("Images have no common coverage area after CRS transformation")

    # Largest coverage, least nodata, highest resolution, then earliest date.
    reference = sorted(
        infos,
        key=lambda item: (-item["coverage_area"], item["nodata_fraction"], item["resolution"], item["date"]),
    )[0]
    grid = _reference_grid(reference, analysis_crs)
    gt_results: dict[str, GroundTruthResult] = {}
    warnings: list[str] = []
    resolutions = [info["resolution"] for info in infos]
    if max(resolutions) / max(min(resolutions), 1e-12) > 2.0:
        warnings.append("Image GSD differs by more than 2x; inspect alignment and model scale robustness")

    for date, image_path in sorted(project.images.items()):
        _warp(image_path, prepared / "images" / f"{date}.tif", grid, Resampling.bilinear)
        image_bounds = next(info["analysis_bounds"] for info in infos if info["date"] == date)
        result = normalize_label_file(
            project.labels[date],
            source_date=date,
            target_crs=analysis_crs,
            image_bounds=image_bounds,
            min_building_area_m2=float(project.config["data"]["min_building_area_m2"]),
            classes=project.config["data"].get("classes", ["building"]),
        )
        label_output = prepared / "labels" / f"{date}.gpkg"
        label_output.parent.mkdir(parents=True, exist_ok=True)
        result.accepted.to_file(label_output, driver="GPKG")
        gt_results[date] = result

    quality_dir = project.project_dir / "quality"
    if quality_dir.is_dir():
        for path in sorted(quality_dir.glob("*.tif*")):
            from .project_discovery import parse_date_from_filename

            date = parse_date_from_filename(path)
            if date in project.images:
                _warp(path, prepared / "quality" / f"{date}.tif", grid, Resampling.nearest)

    report_paths = write_ground_truth_report(gt_results, data_report)
    grid_bounds = array_bounds(grid["height"], grid["width"], grid["transform"])
    extent_width = grid_bounds[2] - grid_bounds[0]
    extent_height = grid_bounds[3] - grid_bounds[1]
    configured_block_size = float(project.config["split"]["block_size_m"])
    block_size = min(configured_block_size, max(min(extent_width, extent_height) / 2.0, 1.0))
    if block_size < configured_block_size:
        warnings.append(
            f"split.block_size_m adjusted from {configured_block_size:g} to {block_size:.2f} for this small AOI"
        )
    blocks = build_spatial_blocks(
        grid_bounds,
        analysis_crs,
        block_size_m=block_size,
        validation_fraction=float(project.config["split"]["validation_fraction"]),
        seed=int(project.config["split"]["seed"]),
    )
    split_result = write_split_report(blocks, report_paths["accepted_labels.gpkg"], data_report)
    warnings.extend(split_result["warnings"])
    tiling = project.config["tiling"]
    coco_summary = build_spatial_coco(
        prepared,
        blocks,
        sorted(project.images),
        tile_size=int(tiling["tile_size"]),
        overlap=int(tiling["overlap"]),
        min_object_visibility=float(tiling["min_object_visibility"]),
        include_empty_tiles=bool(tiling["include_empty_tiles"]),
        max_empty_to_positive_ratio=float(tiling["max_empty_to_positive_ratio"]),
        rgb_bands=project.config["data"]["rgb_bands"],
    )
    split_summary = pd.read_csv(split_result["summary_path"])
    for split in ("train", "val"):
        values = coco_summary["splits"][split]
        split_summary.loc[split_summary["split"] == split, "positive_tiles"] = values["positive_tiles"]
        split_summary.loc[split_summary["split"] == split, "empty_tiles"] = values["empty_tiles"]
    split_summary.to_csv(split_result["summary_path"], index=False, encoding="utf-8-sig")
    grid_payload = {
        "reference_date": grid["reference_date"],
        "reference_source": grid["reference_source"],
        "analysis_crs": analysis_crs.to_string(),
        "width": grid["width"],
        "height": grid["height"],
        "transform": list(grid["transform"])[:6],
        "common_coverage_area_m2": common.area,
    }
    (prepared / "reference_grid.json").write_text(json.dumps(grid_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        "strategy": "local_wgs84_utm",
        "analysis_crs": analysis_crs.to_string(),
        "exchange_crs": "EPSG:4326",
        "display_crs": "EPSG:3857",
        "images": [
            {
                "date": info["date"],
                "source_path": str(info["path"]),
                "source_crs": info["source_crs"].to_string(),
                "working_path": str(prepared / "images" / f"{info['date']}.tif"),
                "resampling": "bilinear",
            }
            for info in infos
        ],
        "labels": [
            {
                "date": date,
                "source_path": str(project.labels[date]),
                "working_path": str(prepared / "labels" / f"{date}.gpkg"),
                "operation": "GeoPandas.to_crs + validation + image clipping",
            }
            for date in sorted(project.labels)
        ],
    }
    (prepared / "crs_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Alignment report",
        "",
        f"- Analysis CRS: `{analysis_crs.to_string()}`",
        f"- Reference date: `{grid['reference_date']}`",
        f"- Common coverage: `{common.area:.2f} m²`",
        "- RGB/continuous images: bilinear resampling",
        "- Quality/category masks: nearest resampling",
        "- Source files were not modified.",
    ]
    lines.extend(f"- WARNING: {warning}" for warning in warnings)
    (prepared / "alignment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "prepared_dir": prepared,
        "data_report_dir": data_report,
        "analysis_crs": analysis_crs,
        "reference_date": grid["reference_date"],
        "warnings": warnings,
        "coco_summary": coco_summary,
    }
