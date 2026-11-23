#!/usr/bin/env python3
"""Measure local image-frame ties among annual 10 m base rasters in one case.

This is a bounded, read-only diagnostic. The annual TIFFs have no verified
per-pixel acquisition dates; candidate shifts describe image-texture alignment,
not accepted georeferencing or ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import fiona
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import Polygon, mapping, shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment.base_tie import (estimate_local_gradient_field,
                                        image_shift_to_projected_correction)


BASE_DIR = Path(r"F:\xian\xian底图\xian搴曞浘")
RECONSTRUCTION_DIR = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction")
DEFAULT_INPUT = RECONSTRUCTION_DIR / "base_tie_field_subpx_B04_01" / "BASE_TIE_DIAGNOSTIC.json"
DEFAULT_OUTPUT = RECONSTRUCTION_DIR / "base_mosaic_pair_B04_01"
BASE_YEARS = tuple(range(2017, 2023))


def _validate_fixed_case_diagnostic(diagnostic: dict[str, Any]) -> str:
    """Accept only a named fixed-case receipt, not a citywide or anonymous run."""
    case_id = diagnostic.get("case_id")
    if (diagnostic.get("scope") != "FIXED_CASE_LOCAL_ONLY"
            or not isinstance(case_id, str)
            or not re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", case_id)):
        raise ValueError("Input must be a named fixed-case local base-tie receipt")
    return case_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_polygon(transform: Affine, window: list[int]) -> Polygon:
    x0, y0, x1, y1 = window
    return Polygon([
        transform * (x0, y0), transform * (x1, y0),
        transform * (x1, y1), transform * (x0, y1),
    ])


def _load_r4_exclusion(
    diagnostic: dict[str, Any],
    transform: Affine,
    shape_hw: tuple[int, int],
    buffer_m: float,
) -> tuple[np.ndarray, dict[str, str], int]:
    sources: dict[str, str] = {}
    for epoch in diagnostic["epochs"].values():
        for item in epoch["r4_roof_exclusion"]["mask_geometry_sources"].values():
            path = Path(item["path"])
            expected = str(item["sha256"])
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"Frozen R4 geometry hash mismatch: {path}")
            sources[str(path)] = expected

    raster_bounds = window_bounds(
        Window(0, 0, shape_hw[1], shape_hw[0]), transform,
    )
    raster_bounds = (
        raster_bounds[0] - buffer_m, raster_bounds[1] - buffer_m,
        raster_bounds[2] + buffer_m, raster_bounds[3] + buffer_m,
    )
    mask = np.zeros(shape_hw, dtype=np.uint8)
    geometry_batch = []
    feature_count = 0
    invalid_count = 0

    def burn_batch() -> None:
        if not geometry_batch:
            return
        burned = rasterize(
            [(mapping(geom.buffer(buffer_m)), 1) for geom in geometry_batch],
            out_shape=shape_hw, transform=transform, fill=0,
            all_touched=True, dtype="uint8",
        )
        np.maximum(mask, burned, out=mask)
        geometry_batch.clear()

    for path_text in sorted(sources):
        with fiona.open(path_text, layer="buildings") as source:
            if source.crs.to_epsg() != 32649:
                raise ValueError(f"R4 geometry is not EPSG:32649: {path_text}")
            for feature in source.filter(bbox=raster_bounds):
                if not feature.get("geometry"):
                    invalid_count += 1
                    continue
                geom = shape(feature["geometry"])
                if geom.is_empty or not geom.is_valid:
                    invalid_count += 1
                    continue
                geometry_batch.append(geom)
                feature_count += 1
                if len(geometry_batch) >= 256:
                    burn_batch()
        burn_batch()
    if invalid_count:
        raise ValueError(f"Invalid or empty R4 geometries in the local window: {invalid_count}")

    return mask.astype(bool), sources, feature_count


def _read_base_chips(
    window_values: list[int],
    expected_grid: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, str], dict[int, dict[str, Any]]]:
    col, row, width, height = (int(value) for value in window_values)
    window = Window(col, row, width, height)
    expected_transform = Affine(*expected_grid["full_transform"])
    expected_crs = "EPSG:32649"
    expected_resolution = (10.0, 10.0)
    images: dict[int, np.ndarray] = {}
    hashes: dict[int, str] = {}
    metadata: dict[int, dict[str, Any]] = {}
    for year in BASE_YEARS:
        path = BASE_DIR / f"Xian_{year}_TrueColor.tif"
        if not path.is_file():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            if (str(src.crs) != expected_crs or tuple(src.res) != expected_resolution or
                    tuple(src.transform) != tuple(expected_transform)):
                raise ValueError(f"Annual base grid differs from the fixed reference: {path}")
            if (col < 0 or row < 0 or col + width > src.width or row + height > src.height):
                raise ValueError(f"Local comparison window is outside the base raster: {path}")
            chip = np.moveaxis(src.read(window=window), 0, 2)
            if chip.shape != (height, width, 3):
                raise ValueError(f"Expected an RGB base raster: {path} -> {chip.shape}")
            images[year] = chip
            hashes[year] = _sha256(path)
            metadata[year] = {
                "path": str(path), "sha256": hashes[year], "crs": str(src.crs),
                "resolution": list(src.res), "width": src.width, "height": src.height,
                "bands": src.count, "window": [col, row, width, height],
            }
    return images, hashes, metadata


def run(
    input_diagnostic: Path,
    output_dir: Path,
    *,
    grid_size: int = 3,
    search_radius_px: int = 5,
    r4_buffer_m: float = 20.0,
) -> dict[str, Any]:
    input_diagnostic = Path(input_diagnostic)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite base-pair diagnostic: {output_dir}")
    if grid_size < 1 or search_radius_px < 1 or r4_buffer_m < 0:
        raise ValueError("Grid size/radius must be positive and R4 buffer non-negative")
    diagnostic_hash = _sha256(input_diagnostic)
    diagnostic = json.loads(input_diagnostic.read_text(encoding="utf-8"))
    case_id = _validate_fixed_case_diagnostic(diagnostic)
    epochs = diagnostic.get("epochs")
    if not isinstance(epochs, dict) or not epochs:
        raise ValueError("Fixed-case base-tie receipt contains no actual-date epochs")
    reference_epoch_label, reference_epoch = sorted(epochs.items())[0]
    grid = reference_epoch["base_grid"]
    if grid.get("crs") != "EPSG:32649" or grid.get("resolution") != [10.0, 10.0]:
        raise ValueError("Reference window must use the verified 10 m EPSG:32649 grid")
    window_values = [int(value) for value in grid["window"]]
    window = Window(*window_values)
    reference_match = re.search(
        r"Xian_(20\d{2})_TrueColor\.tif$",
        Path(str(reference_epoch.get("base_path") or "")).name,
    )
    if reference_match is None:
        raise ValueError(
            f"Could not resolve the reference raster for epoch {reference_epoch_label}")
    reference_year = int(reference_match.group(1))
    reference_path = BASE_DIR / f"Xian_{reference_year}_TrueColor.tif"
    if Path(reference_epoch["base_path"]).resolve() != reference_path.resolve():
        raise ValueError("Reference epoch does not point to the configured local 10 m base stack")
    with rasterio.open(reference_path) as reference:
        local_transform = reference.window_transform(window)
        base_grid = {
            "reference_path": str(reference_path),
            "reference_sha256": reference_epoch["base_sha256"],
            "crs": str(reference.crs), "resolution": list(reference.res),
            "full_transform": list(reference.transform),
            "local_transform": list(local_transform),
            "window": window_values,
            "shape": [int(window.height), int(window.width)],
            "bounds": list(window_bounds(window, reference.transform)),
        }
    images, base_hashes_before, base_metadata = _read_base_chips(
        window_values, base_grid,
    )
    for epoch, item in diagnostic["epochs"].items():
        match = re.search(r"Xian_(20\d{2})_TrueColor\.tif$", Path(item["base_path"]).name)
        if not match:
            raise ValueError(f"Could not resolve the referenced annual base raster: {item['base_path']}")
        year = int(match.group(1))
        if base_hashes_before.get(year) != item.get("base_sha256"):
            raise ValueError(f"Referenced annual base hash differs from the fixed diagnostic: {epoch}")
    r4_exclusion, r4_hashes_before, r4_feature_count = _load_r4_exclusion(
        diagnostic, local_transform, (int(window.height), int(window.width)),
        r4_buffer_m,
    )

    pair_records = []
    vector_rows = []
    csv_rows = []
    years = list(BASE_YEARS)
    for i, reference_year in enumerate(years):
        for moving_year in years[i + 1:]:
            reference = images[reference_year]
            moving = images[moving_year]
            reference_valid = np.any(reference > 0, axis=2) & ~r4_exclusion
            moving_valid = np.any(moving > 0, axis=2) & ~r4_exclusion
            common_valid = reference_valid & moving_valid
            field = estimate_local_gradient_field(
                reference, moving, reference_valid, grid_size=grid_size,
                radius=search_radius_px, min_samples=100,
                moving_valid_mask=moving_valid,
            )
            cells = []
            candidate_vectors = []
            for cell in field["cells"]:
                best = cell["best"]
                refined = best.get("subpixel_peak") if best else None
                observed_dx = (refined["dx_px"] if refined else best["dx_px"]) if best else None
                observed_dy = (refined["dy_px"] if refined else best["dy_px"]) if best else None
                if best:
                    correction_dx_m, correction_dy_m = image_shift_to_projected_correction(
                        observed_dx, observed_dy, local_transform)
                    candidate_vectors.append([observed_dx, observed_dy])
                else:
                    correction_dx_m = correction_dy_m = None
                geometry = _cell_polygon(local_transform, cell["base_pixel_window"])
                result_cell = {
                    **cell,
                    "observed_moving_shift_px": (
                        [float(observed_dx), float(observed_dy)] if best else None
                    ),
                    "correction_to_reference_m_candidate": (
                        [float(correction_dx_m), float(correction_dy_m)] if best else None
                    ),
                    "reviewed": False,
                    "applied": False,
                    "polygon_epsg32649": mapping(geometry),
                }
                cells.append(result_cell)
                csv_row = {
                    "reference_year_label": reference_year,
                    "moving_year_label": moving_year,
                    "cell_row": cell["row"], "cell_col": cell["col"],
                    "status": cell["status"], "valid_pixels": cell["valid_pixel_count"],
                    "best_correlation": best["corr"] if best else None,
                    "peak_margin": cell["peak_margin"],
                    "observed_dx_px": observed_dx, "observed_dy_px": observed_dy,
                    "candidate_correction_dx_m": correction_dx_m,
                    "candidate_correction_dy_m": correction_dy_m,
                    "reviewed": False, "applied": False,
                }
                csv_rows.append(csv_row)
                vector_rows.append((geometry, csv_row))
            values = np.asarray(candidate_vectors, dtype=np.float64)
            if len(values):
                center = np.median(values, axis=0)
                residual = np.linalg.norm(values - center, axis=1)
                summary = {
                    "candidate_cell_count": int(len(values)),
                    "median_observed_moving_shift_px": center.tolist(),
                    "median_cell_deviation_px": float(np.median(residual)),
                    "field_interpolated": False,
                    "field_applied": False,
                }
            else:
                summary = {
                    "candidate_cell_count": 0,
                    "median_observed_moving_shift_px": None,
                    "median_cell_deviation_px": None,
                    "field_interpolated": False,
                    "field_applied": False,
                }
            pair_records.append({
                "reference_year_label": reference_year,
                "moving_year_label": moving_year,
                "base_acquisition_date_status": "UNKNOWN_ANNUAL_RASTER_LABEL_ONLY",
                "common_valid_pixel_count_before_shift_search": int(common_valid.sum()),
                "field": field,
                "summary": summary,
                "cells": cells,
            })

    if _sha256(input_diagnostic) != diagnostic_hash:
        raise RuntimeError("Input diagnostic changed during base-pair analysis")
    base_hashes_after = {
        year: _sha256(Path(base_metadata[year]["path"])) for year in years
    }
    r4_hashes_after = {path: _sha256(Path(path)) for path in r4_hashes_before}
    if base_hashes_after != base_hashes_before or r4_hashes_after != r4_hashes_before:
        raise RuntimeError("A base raster or frozen R4 input changed during analysis")

    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "BASE_MOSAIC_PAIR_LOCAL_TIES.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    vector_path = output_dir / "BASE_MOSAIC_PAIR_LOCAL_TIES.gpkg"
    schema = {"geometry": "Polygon", "properties": {
        "ref_year": "int", "mov_year": "int", "cell_id": "str:16",
        "status": "str:40", "valid_n": "int", "corr": "float",
        "margin": "float", "obs_dx_px": "float", "obs_dy_px": "float",
        "corr_dx_m": "float", "corr_dy_m": "float", "reviewed": "int",
        "applied": "int",
    }}
    with fiona.open(
        vector_path, "w", driver="GPKG", layer="base_mosaic_pair_local_ties",
        schema=schema, crs="EPSG:32649",
    ) as sink:
        for geometry, row in vector_rows:
            cell_id = f"r{int(row['cell_row']):02d}c{int(row['cell_col']):02d}"
            sink.write({"geometry": mapping(geometry), "properties": {
                "ref_year": row["reference_year_label"],
                "mov_year": row["moving_year_label"], "cell_id": cell_id,
                "status": row["status"], "valid_n": row["valid_pixels"],
                "corr": row["best_correlation"], "margin": row["peak_margin"],
                "obs_dx_px": row["observed_dx_px"], "obs_dy_px": row["observed_dy_px"],
                "corr_dx_m": row["candidate_correction_dx_m"],
                "corr_dy_m": row["candidate_correction_dy_m"],
                "reviewed": 0, "applied": 0,
            }})

    base_hashes_final = {
        year: _sha256(Path(base_metadata[year]["path"])) for year in years
    }
    r4_hashes_final = {path: _sha256(Path(path)) for path in r4_hashes_before}
    if (base_hashes_final != base_hashes_before or r4_hashes_final != r4_hashes_before or
            _sha256(input_diagnostic) != diagnostic_hash):
        raise RuntimeError("A read-only input changed before receipt publication")

    receipt = {
        "schema_version": "xian_r5.base_mosaic_pair_local_tie_diagnostic.v1",
        "case_id": case_id,
        "scope": "FIXED_CASE_LOCAL_ONLY",
        "input_diagnostic": str(input_diagnostic),
        "input_diagnostic_sha256": diagnostic_hash,
        "base_year_labels": list(BASE_YEARS),
        "base_year_date_semantics": "ANNUAL_FILE_LABELS_ONLY;PIXEL_ACQUISITION_DATES_UNKNOWN",
        "method": "PAIRWISE_GRADIENT_NCC_ON_SHARED_10M_GRID",
        "mask": {
            "r4_building_buffer_m": r4_buffer_m,
            "r4_features_in_local_window": r4_feature_count,
            "semantics": "UNREVIEWED_CANDIDATE_GROUND;NOT_A_STABLE_GROUND_LABEL",
            "both_raster_valid_masks_checked_at_each_candidate_shift": True,
        },
        "grid_size": grid_size,
        "search_radius_px": search_radius_px,
        "base_grid": base_grid,
        "base_rasters": base_metadata,
        "pairwise_fields": pair_records,
        "outputs": {
            "candidate_csv": str(csv_path),
            "candidate_vector": str(vector_path),
            "local_tie_records": len(csv_rows),
        },
        "base_sha256_before": base_hashes_before,
        "base_sha256_after": base_hashes_final,
        "r4_geometry_sha256_before": r4_hashes_before,
        "r4_geometry_sha256_after": r4_hashes_final,
        "field_reviewed": False,
        "field_interpolated": False,
        "field_applied": False,
        "registration_promoted": False,
        "formal_ground_footprint_count": 0,
        "interpretation": "RELATIVE_ANNUAL_BASE_IMAGE_TEXTURE_TIES_ONLY;NOT_GROUND_CONTROL_OR_CORRECTION",
    }
    receipt_path = output_dir / "BASE_MOSAIC_PAIR_DIAGNOSTIC.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-diagnostic", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--search-radius-px", type=int, default=5)
    parser.add_argument("--r4-buffer-m", type=float, default=20.0)
    args = parser.parse_args()
    result = run(
        args.input_diagnostic, args.output, grid_size=args.grid_size,
        search_radius_px=args.search_radius_px, r4_buffer_m=args.r4_buffer_m,
    )
    print(json.dumps({
        "case_id": result["case_id"], "output": str(args.output),
        "pair_count": len(result["pairwise_fields"]),
        "record_count": result["outputs"]["local_tie_records"],
        "registration_promoted": result["registration_promoted"],
    }, ensure_ascii=False))
