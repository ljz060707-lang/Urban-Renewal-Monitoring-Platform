#!/usr/bin/env python3
"""Build a bounded, date-aware ALIKED/LightGlue local-tile field diagnostic.

This tool estimates separate tie candidates in small neighboring-tile cells.
It does not interpolate between cells, warp roofs, or produce formal footprints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

try:
    import fiona
except ImportError:  # the isolated GPU runtime uses pyogrio instead
    fiona = None
try:
    import pyogrio
except ImportError:  # production runtimes may provide Fiona only
    pyogrio = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

STAGE2 = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2")
LIGHTGLUE_SOURCE = STAGE2 / "06_lightglue_ab" / "third_party" / "LightGlue"
MODEL_RECEIPT = STAGE2 / "06_lightglue_ab" / "ALIGNMENT_PROXY_DIAGNOSTIC_RECEIPT_V2.json"
TORCH_HOME = STAGE2 / "06_lightglue_ab" / "torch_cache"

from xian_r5.alignment.local_field import (
    LOCAL_TRANSFORM_MODEL, candidate_cycle_diagnostics, estimate_local_tile_shift,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_local_vector_geometries(path: Path, *, layer: str, bbox: tuple[float, ...]):
    """Read a bounded vector window through the runtime's available adapter."""
    if fiona is not None:
        with fiona.open(path, layer=layer) as source:
            crs = source.crs.to_epsg()
            features = [feature.get("geometry") for feature in source.filter(bbox=bbox)]
    elif pyogrio is not None:
        frame = pyogrio.read_dataframe(path, layer=layer, bbox=bbox)
        crs = frame.crs.to_epsg() if frame.crs is not None else None
        features = list(frame.geometry)
    else:
        raise RuntimeError("Vector I/O requires Fiona or pyogrio in this runtime")
    return crs, features


def _write_local_field_candidates(path: Path, pair_records: list[dict[str, Any]]) -> None:
    """Persist local candidate cells using Fiona or the isolated pyogrio adapter."""
    if fiona is not None:
        schema = {"geometry": "Polygon", "properties": {
            "from_date": "str:10", "to_date": "str:10", "cell_id": "str:16",
            "status": "str:40", "transform_model": "str:64",
            "raw_n": "int", "eligible_n": "int", "inlier_n": "int",
            "ratio": "float", "coverage": "float", "p90_px": "float", "p90_m": "float",
            "corr_dx_m": "float", "corr_dy_m": "float", "reviewed": "int", "applied": "int",
        }}
        with fiona.open(path, "w", driver="GPKG", layer="local_tile_field_candidates",
                        schema=schema, crs="EPSG:32649") as sink:
            for pair in pair_records:
                for row in pair["cells"]:
                    sink.write({"geometry": row["polygon_epsg32649"], "properties": {
                        "from_date": row["from_date"], "to_date": row["to_date"],
                        "cell_id": row["cell_id"], "status": row["status"],
                        "transform_model": row["transform_model"],
                        "raw_n": row["matched_features_raw"], "eligible_n": row["eligible_matches"],
                        "inlier_n": row["inliers"], "ratio": row["inlier_ratio"],
                        "coverage": row["coverage_4x4"], "p90_px": row["p90_residual_px"],
                        "p90_m": row["p90_residual_m"],
                        "corr_dx_m": row["correction_dx_m_candidate"],
                        "corr_dy_m": row["correction_dy_m_candidate"],
                        "reviewed": 0, "applied": 0,
                    }})
        return
    if pyogrio is None:
        raise RuntimeError("Vector I/O requires Fiona or pyogrio in this runtime")

    import geopandas as gpd
    import pandas as pd

    rows = []
    geometries = []
    for pair in pair_records:
        for row in pair["cells"]:
            rows.append({
                "from_date": row["from_date"], "to_date": row["to_date"],
                "cell_id": row["cell_id"], "status": row["status"],
                "transform_model": row["transform_model"],
                "raw_n": row["matched_features_raw"], "eligible_n": row["eligible_matches"],
                "inlier_n": row["inliers"], "ratio": row["inlier_ratio"],
                "coverage": row["coverage_4x4"], "p90_px": row["p90_residual_px"],
                "p90_m": row["p90_residual_m"],
                "corr_dx_m": row["correction_dx_m_candidate"],
                "corr_dy_m": row["correction_dy_m_candidate"],
                "reviewed": 0, "applied": 0,
            })
            geometries.append(shape(row["polygon_epsg32649"]))
    frame = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:32649")
    for name in ("raw_n", "eligible_n", "inlier_n", "reviewed", "applied"):
        frame[name] = pd.Series(frame[name], dtype="int64")
    for name in ("ratio", "coverage", "p90_px", "p90_m", "corr_dx_m", "corr_dy_m"):
        frame[name] = pd.Series(frame[name], dtype="float64")
    pyogrio.write_dataframe(frame, path, layer="local_tile_field_candidates", driver="GPKG")


def _load_model(model_receipt_path: Path, *, device_preference: str = "cpu"):
    if not LIGHTGLUE_SOURCE.is_dir():
        raise FileNotFoundError(f"Official LightGlue source is unavailable: {LIGHTGLUE_SOURCE}")
    if not model_receipt_path.is_file():
        raise FileNotFoundError(f"ALIKED/LightGlue model receipt is unavailable: {model_receipt_path}")
    model_receipt = json.loads(model_receipt_path.read_text(encoding="utf-8"))
    model_hashes = {}
    for item in model_receipt.get("weights", []):
        path = Path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"Cached model weight missing or hash-mismatched: {path}")
        model_hashes[str(path)] = item["sha256"]
    if len(model_hashes) < 2:
        raise ValueError("Model receipt does not identify both ALIKED and LightGlue weights")

    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    sys.path.insert(0, str(LIGHTGLUE_SOURCE))
    import torch
    from lightglue import ALIKED, LightGlue

    if device_preference not in {"cpu", "cuda", "auto"}:
        raise ValueError(f"Unsupported inference device preference: {device_preference}")
    if device_preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if device_preference == "cpu":
        torch.set_num_threads(1)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    extractor = ALIKED(max_num_keypoints=1200).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    return torch, extractor, matcher, device, model_receipt, model_hashes


def _match(extractor: Any, matcher: Any, image0: np.ndarray, image1: np.ndarray,
           torch: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:
    tensor0 = torch.from_numpy(np.ascontiguousarray(image0.transpose(2, 0, 1))).float().div_(255).unsqueeze(0).to(device)
    tensor1 = torch.from_numpy(np.ascontiguousarray(image1.transpose(2, 0, 1))).float().div_(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        features0 = extractor.extract(tensor0)
        features1 = extractor.extract(tensor1)
        matched = matcher({"image0": features0, "image1": features1})["matches"][0]
    points0 = features0["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    points1 = features1["keypoints"][0].detach().cpu().numpy().astype(np.float32)
    pairs = matched.detach().cpu().numpy().astype(np.int64)
    if pairs.ndim != 2 or pairs.shape[-1] != 2 or not len(pairs):
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty
    return points0[pairs[:, 0]], points1[pairs[:, 1]]


def _color_candidate_mask(rgb: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 40)
    dark = hsv[:, :, 2] < 28
    return ~(green | dark)


def _candidate_mask(image: np.ndarray, valid_mask: np.ndarray, excluded: np.ndarray,
                    *, color_filter_profile: str) -> np.ndarray:
    """Build an explicitly diagnostic feature mask; neither profile proves stable ground."""
    if color_filter_profile not in {"green-dark-exclusion", "none"}:
        raise ValueError(f"Unsupported color-filter profile: {color_filter_profile}")
    mask = valid_mask.astype(bool) & (excluded == 0)
    if color_filter_profile == "green-dark-exclusion":
        mask &= _color_candidate_mask(image)
    return mask.astype(np.uint8)


def _preview_image(rgb: np.ndarray, candidate_mask: np.ndarray) -> Image.Image:
    """Tint pixels excluded from the candidate tie mask; this is not ground truth."""
    display = rgb.copy()
    excluded = candidate_mask == 0
    if excluded.any():
        pixels = display[excluded].astype(np.float32)
        display[excluded] = (pixels * 0.45 + np.array([185, 35, 35]) * 0.55).astype(np.uint8)
    return Image.fromarray(display, mode="RGB")


def _write_match_contact_sheet(path: Path, pair: dict[str, Any]) -> None:
    """Render eligible RANSAC inliers for human review, without promoting shifts."""
    cells = pair["cells"]
    if not cells:
        raise ValueError("Cannot render an empty pairwise match contact sheet")
    cell_height, cell_width = cells[0]["image_from"].shape[:2]
    panel_width = cell_width * 2 + 32
    panel_height = cell_height + 92
    margin = 16
    columns = 2
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (margin * 2 + columns * panel_width,
                               margin * 2 + rows * panel_height), (28, 31, 36))
    draw = ImageDraw.Draw(sheet)

    for index, cell in enumerate(cells):
        x0 = margin + (index % columns) * panel_width
        y0 = margin + (index // columns) * panel_height
        record = cell["record"]
        correction = record["correction_dx_m_candidate"], record["correction_dy_m_candidate"]
        shift_label = ("n/a" if any(value is None for value in correction)
                       else f"({correction[0]:.2f}, {correction[1]:.2f}) m")
        title = (f"{record['cell_id']}  {record['status']}  eligible={record['eligible_matches']}  "
                 f"inliers={record['inliers']} ({record['inlier_ratio']:.2f})")
        detail = (f"candidate correction={shift_label}; ground mask reviewed="
                  f"{record['ground_mask_reviewed']}; applied={record['applied_to_image_or_geometry']}")
        draw.text((x0 + 4, y0 + 2), title, fill=(245, 245, 245))
        draw.text((x0 + 4, y0 + 22), detail, fill=(206, 211, 218))

        image_y = y0 + 48
        left_x = x0 + 4
        right_x = left_x + cell_width + 20
        left = _preview_image(cell["image_from"], cell["mask_from"])
        right = _preview_image(cell["image_to"], cell["mask_to"])
        sheet.paste(left, (left_x, image_y))
        sheet.paste(right, (right_x, image_y))
        draw.text((left_x + 4, image_y + 3), pair["from_date"], fill=(255, 255, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0))
        draw.text((right_x + 4, image_y + 3), pair["to_date"], fill=(255, 255, 255),
                  stroke_width=1, stroke_fill=(0, 0, 0))

        for source, target in zip(cell["inlier_points_from_px"], cell["inlier_points_to_px"]):
            sx, sy = left_x + float(source[0]), image_y + float(source[1])
            tx, ty = right_x + float(target[0]), image_y + float(target[1])
            draw.line((sx, sy, tx, ty), fill=(255, 214, 35), width=2)
            radius = 3
            draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius),
                         fill=(54, 255, 88), outline=(0, 0, 0))
            draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius),
                         fill=(255, 66, 232), outline=(0, 0, 0))

    draw.text((margin, sheet.height - 13),
              "Yellow: RANSAC inlier match | Green/pink: endpoints | Red tint: outside candidate mask",
              fill=(225, 225, 225))
    sheet.save(path, format="PNG", optimize=True)


def _read_mosaics(diagnostic: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    tile_window = diagnostic["grid"]["tile_window"]
    x0, y0, nx, ny = (int(tile_window[key]) for key in ("x0", "y0", "nx", "ny"))
    dates = sorted(diagnostic["submeter_tiles"], key=date.fromisoformat)
    images: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    tile_hashes: dict[str, str] = {}
    for acquisition in dates:
        image = np.zeros((ny * 256, nx * 256, 3), dtype=np.uint8)
        mask = np.zeros((ny * 256, nx * 256), dtype=np.uint8)
        seen_tiles = set()
        for record in diagnostic["submeter_tiles"][acquisition]:
            tile_id = str(record["tile_id"])
            if tile_id in seen_tiles:
                raise ValueError(f"Duplicate tile in exact-date mosaic: {acquisition}/{tile_id}")
            seen_tiles.add(tile_id)
            tile_x, tile_y = (int(value) for value in tile_id.split(":"))
            path = Path(record["path"])
            digest_before = _sha256(path)
            expected = record.get("sha256")
            if expected and digest_before != expected:
                raise ValueError(f"Input tile differs from mosaic diagnostic: {path}")
            with Image.open(path) as source:
                rgb = np.asarray(source.convert("RGB"))
            if rgb.shape != (256, 256, 3):
                raise ValueError(f"Unexpected z20 tile shape: {path} -> {rgb.shape}")
            digest_after = _sha256(path)
            if digest_after != digest_before or (record.get("sha256_after_read") and digest_after != record["sha256_after_read"]):
                raise RuntimeError(f"Tile changed while being read: {path}")
            col, row = tile_x - x0, y0 + ny - 1 - tile_y
            if not (0 <= col < nx and 0 <= row < ny):
                raise ValueError(f"Tile outside declared local grid: {tile_id}")
            ys, xs = row * 256, col * 256
            image[ys:ys + 256, xs:xs + 256] = rgb
            mask[ys:ys + 256, xs:xs + 256] = 1
            tile_hashes[str(path)] = digest_before
        images[acquisition], valid[acquisition] = image, mask
    return images, valid, tile_hashes


def _load_roof_union(diagnostic: dict, buffer_m: float) -> tuple[Any | None, dict[str, str], int, int]:
    bounds = diagnostic["grid"]["bounds"]
    query = (bounds[0] - buffer_m - 2, bounds[1] - buffer_m - 2,
             bounds[2] + buffer_m + 2, bounds[3] + buffer_m + 2)
    geometries = []
    source_hashes = {}
    invalid_count = 0
    for item in diagnostic["frozen_r4_building_exclusion"]["sources"].values():
        path = Path(item["path"])
        digest_before = _sha256(path)
        if digest_before != item["sha256"]:
            raise ValueError(f"Frozen R4 geometry hash mismatch: {path}")
        crs_epsg, features = _read_local_vector_geometries(path, layer="buildings", bbox=query)
        if crs_epsg != 32649:
            raise ValueError(f"Frozen R4 geometry CRS is not EPSG:32649: {path}")
        for feature_geometry in features:
            if feature_geometry is None:
                invalid_count += 1
                continue
            geometry = (feature_geometry if hasattr(feature_geometry, "geom_type")
                        else shape(feature_geometry))
            if geometry.is_empty or not geometry.is_valid:
                invalid_count += 1
                continue
            geometries.append(geometry)
        digest_after = _sha256(path)
        if digest_after != digest_before:
            raise RuntimeError(f"Frozen R4 geometry changed while being read: {path}")
        source_hashes[str(path)] = digest_before
    if invalid_count:
        raise ValueError(f"Invalid or empty R4 features intersect local query: {invalid_count}")
    if not geometries:
        return None, source_hashes, 0, invalid_count
    union = unary_union(geometries)
    exclusion = union.buffer(buffer_m) if buffer_m else union
    return exclusion, source_hashes, len(geometries), invalid_count


def _pixel_polygon(pixel_transform: tuple[float, ...], x0: int, y0: int,
                   x1: int, y1: int) -> Polygon:
    affine = Affine(*pixel_transform[:6])
    corners = [affine * point for point in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return Polygon(corners)


def _tile_patch_windows(tile_cols: int, tile_rows: int, patch_tiles: int):
    """Partition a local tile mosaic into bounded patches, clipping edge cells."""
    if tile_cols < 1 or tile_rows < 1 or patch_tiles < 1:
        raise ValueError("Tile dimensions and patch_tiles must be positive")
    if patch_tiles > max(tile_cols, tile_rows):
        raise ValueError("patch_tiles cannot exceed both local mosaic dimensions")
    for row, tile_row0 in enumerate(range(0, tile_rows, patch_tiles)):
        tile_row1 = min(tile_rows, tile_row0 + patch_tiles)
        for col, tile_col0 in enumerate(range(0, tile_cols, patch_tiles)):
            tile_col1 = min(tile_cols, tile_col0 + patch_tiles)
            yield (row, col, tile_row0, tile_col0,
                   tile_row1 - tile_row0, tile_col1 - tile_col0)


def _local_quality_window(base_path: Path, bounds: tuple[float, ...]) -> tuple[Affine, tuple[int, int], dict]:
    with rasterio.open(base_path) as base:
        if str(base.crs) != "EPSG:32649" or tuple(base.res) != (10.0, 10.0):
            raise ValueError("Local quality TIF requires the verified 10 m EPSG:32649 base grid")
        window = from_bounds(*bounds, transform=base.transform)
        col0 = max(0, math.floor(window.col_off))
        row0 = max(0, math.floor(window.row_off))
        col1 = min(base.width, math.ceil(window.col_off + window.width))
        row1 = min(base.height, math.ceil(window.row_off + window.height))
        if col1 <= col0 or row1 <= row0:
            raise ValueError("Fixed-case field does not intersect the 10 m base raster")
        transform = base.transform * Affine.translation(col0, row0)
        info = {"path": str(base_path), "sha256": _sha256(base_path),
                "crs": str(base.crs), "resolution": list(base.res),
                "window": [col0, row0, col1 - col0, row1 - row0],
                "transform": list(transform)}
        return transform, (row1 - row0, col1 - col0), info


def run(input_diagnostic: Path, output_dir: Path, *, patch_tiles: int = 3,
        roof_exclusion_buffer_m: float = 20.0,
        max_p90_residual_m: float = 2.0,
        color_filter_profile: str = "green-dark-exclusion",
        device_preference: str = "cpu",
        model_receipt_path: Path = MODEL_RECEIPT,
        base_raster: Path | None = None) -> dict:
    input_diagnostic = Path(input_diagnostic)
    output_dir = Path(output_dir)
    if base_raster is None:
        raise ValueError("An explicit 10 m EPSG:32649 base raster is required for local-field QA")
    base_raster = Path(base_raster)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing local field output: {output_dir}")
    diagnostic_hash_before = _sha256(input_diagnostic)
    diagnostic = json.loads(input_diagnostic.read_text(encoding="utf-8"))
    if diagnostic.get("scope") != "FIXED_CASE_LOCAL_ONLY":
        raise ValueError("Input mosaic diagnostic is not fixed-case local evidence")
    tile_window = diagnostic["grid"]["tile_window"]
    nx, ny = int(tile_window["nx"]), int(tile_window["ny"])
    patch_windows = tuple(_tile_patch_windows(nx, ny, patch_tiles))
    if roof_exclusion_buffer_m < 0:
        raise ValueError("Roof exclusion buffer cannot be negative")
    if not math.isfinite(max_p90_residual_m) or max_p90_residual_m <= 0:
        raise ValueError("Maximum p90 projected residual must be a positive finite number of metres")
    if color_filter_profile not in {"green-dark-exclusion", "none"}:
        raise ValueError(f"Unsupported color-filter profile: {color_filter_profile}")

    input_hashes_before = {}
    for name, path_value in diagnostic["input_paths"].items():
        path = Path(path_value)
        actual = _sha256(path)
        expected = diagnostic["input_sha256"].get(name)
        if expected and actual != expected:
            raise ValueError(f"Mosaic diagnostic input hash mismatch: {name}")
        input_hashes_before[name] = actual

    images, valid_masks, tile_hashes_before = _read_mosaics(diagnostic)
    roof_union, roof_hashes_before, roof_count, invalid_roofs = _load_roof_union(
        diagnostic, roof_exclusion_buffer_m)
    height, width = next(iter(valid_masks.values())).shape
    pixel_transform = tuple(float(value) for value in diagnostic["grid"]["pixel_transform"])
    affine = Affine(*pixel_transform[:6])
    if roof_union is None:
        excluded = np.zeros((height, width), dtype=np.uint8)
    else:
        excluded = rasterize(
            [(mapping(roof_union), 1)], out_shape=(height, width), transform=affine,
            fill=0, all_touched=True, dtype="uint8",
        )
    candidate_masks = {}
    for acquisition, image in images.items():
        candidate_masks[acquisition] = _candidate_mask(
            image, valid_masks[acquisition], excluded,
            color_filter_profile=color_filter_profile,
        )

    torch, extractor, matcher, device, model_receipt, model_hashes = _load_model(
        model_receipt_path, device_preference=device_preference,
    )
    model_receipt_hash_before = _sha256(model_receipt_path)
    model_hashes_before = dict(model_hashes)
    ordered_dates = sorted(images, key=date.fromisoformat)
    pair_records = []
    match_preview_pairs = []
    pair_rows: dict[tuple[str, str], dict[str, dict]] = {}
    for from_date, to_date in combinations(ordered_dates, 2):
        common_fraction = float((valid_masks[from_date].astype(bool) & valid_masks[to_date].astype(bool)).mean())
        rows = []
        fields = {}
        preview_cells = []
        for (row, col, tile_row0, tile_col0,
             tile_rows_in_cell, tile_cols_in_cell) in patch_windows:
            py0 = tile_row0 * 256
            py1 = (tile_row0 + tile_rows_in_cell) * 256
            px0 = tile_col0 * 256
            px1 = (tile_col0 + tile_cols_in_cell) * 256
            cell_id = f"r{row:02d}c{col:02d}"
            image0 = images[from_date][py0:py1, px0:px1]
            image1 = images[to_date][py0:py1, px0:px1]
            mask0 = candidate_masks[from_date][py0:py1, px0:px1]
            mask1 = candidate_masks[to_date][py0:py1, px0:px1]
            points0, points1 = _match(extractor, matcher, image0, image1, torch, device)
            result = estimate_local_tile_shift(
                points0, points1, mask0, mask1, pixel_transform,
                max_p90_residual_m=max_p90_residual_m,
            )
            geometry = _pixel_polygon(pixel_transform, px0, py0, px1, py1)
            record = {
                "from_date": from_date, "to_date": to_date, "cell_id": cell_id,
                "tile_col0": tile_col0,
                "tile_row0_north_to_south": tile_row0,
                "tile_rows_in_cell": tile_rows_in_cell,
                "tile_cols_in_cell": tile_cols_in_cell,
                "patch_tiles": patch_tiles,
                "common_tile_coverage_fraction": common_fraction,
                "candidate_ground_mask_fraction_from": float(mask0.mean()),
                "candidate_ground_mask_fraction_to": float(mask1.mean()),
                "matched_features_raw": int(len(points0)),
                **result.as_dict(),
                "polygon_epsg32649": mapping(geometry),
            }
            rows.append(record)
            fields[cell_id] = record
            preview_cells.append({
                "record": record,
                "image_from": image0,
                "image_to": image1,
                "mask_from": mask0,
                "mask_to": mask1,
                "inlier_points_from_px": result.inlier_points_from_px,
                "inlier_points_to_px": result.inlier_points_to_px,
            })
        pair_rows[(from_date, to_date)] = fields
        status_counts = Counter(row["status"] for row in rows)
        supported = [row for row in rows if row["status"] == "LOCAL_CANDIDATE_NOT_REVIEWED"]
        vectors = [[row["correction_dx_m_candidate"], row["correction_dy_m_candidate"]]
                   for row in supported]
        pair_records.append({
            "from_date": from_date, "to_date": to_date,
            "cell_count": len(rows), "status_counts": dict(status_counts),
            "candidate_cells_not_reviewed": len(supported),
            "median_candidate_correction_m": np.median(vectors, axis=0).tolist() if vectors else None,
            "field_interpolated": False, "field_applied": False,
            "cells": rows,
        })
        match_preview_pairs.append({
            "from_date": from_date, "to_date": to_date, "cells": preview_cells,
        })

    cycle_rows = candidate_cycle_diagnostics(ordered_dates, pair_rows)

    base_transform, base_shape, base_info = _local_quality_window(base_raster, tuple(diagnostic["grid"]["bounds"]))
    base_hash_before = base_info["sha256"]
    if input_hashes_before != {name: _sha256(Path(path)) for name, path in diagnostic["input_paths"].items()}:
        raise RuntimeError("A frozen date/map input changed during local-field estimation")
    if tile_hashes_before != {path: _sha256(Path(path)) for path in tile_hashes_before}:
        raise RuntimeError("A source image tile changed during local-field estimation")
    if roof_hashes_before != {path: _sha256(Path(path)) for path in roof_hashes_before}:
        raise RuntimeError("A frozen R4 source changed during local-field estimation")
    model_hashes_after = {path: _sha256(Path(path)) for path in model_hashes_before}
    if model_hashes_after != model_hashes_before or _sha256(model_receipt_path) != model_receipt_hash_before:
        raise RuntimeError("ALIKED/LightGlue model evidence changed during local-field estimation")
    if _sha256(input_diagnostic) != diagnostic_hash_before:
        raise RuntimeError("Input mosaic diagnostic changed during local-field estimation")

    output_dir.mkdir(parents=True, exist_ok=False)
    vector_path = output_dir / "LOCAL_TILE_FIELD_CANDIDATES.gpkg"
    _write_local_field_candidates(vector_path, pair_records)

    match_preview_outputs = []
    for pair in match_preview_pairs:
        preview_path = output_dir / f"{pair['from_date']}_to_{pair['to_date']}_MATCH_REVIEW.png"
        _write_match_contact_sheet(preview_path, pair)
        match_preview_outputs.append({
            "path": str(preview_path), "sha256": _sha256(preview_path),
            "from_date": pair["from_date"], "to_date": pair["to_date"],
            "cell_count": len(pair["cells"]),
            "evidence": "ELIGIBLE_LIGHTGLUE_MATCHES_RANSAC_INLIERS;CANDIDATE_MASK_ONLY;NO_GROUND_TRUTH_OR_SHIFT_PROMOTION",
        })

    csv_path = output_dir / "LOCAL_TILE_FIELD_CANDIDATES.csv"
    csv_rows = []
    for pair in pair_records:
        for row in pair["cells"]:
            csv_rows.append({key: value for key, value in row.items() if key != "polygon_epsg32649"})
    if csv_rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    quality_tifs = []
    for pair in pair_records:
        shapes = []
        for row in pair["cells"]:
            code = {"INSUFFICIENT_GROUND_CANDIDATE_MATCHES": 1,
                    "NO_LOCAL_SIMILARITY_INLIERS": 1,
                    "NO_LOCAL_AFFINE_INLIERS": 1,
                    "INSUFFICIENT_GROUND_CANDIDATE_INLIERS": 1,
                    "WEAK_LOCAL_CANDIDATE": 2,
                    "LOCAL_CANDIDATE_NOT_REVIEWED": 3}.get(row["status"], 1)
            shapes.append((shape(row["polygon_epsg32649"]), code))
        quality = rasterize(shapes, out_shape=base_shape, transform=base_transform,
                            fill=255, all_touched=True, dtype="uint8")
        tif_path = output_dir / f"{pair['from_date']}_to_{pair['to_date']}_tile_quality_DIAGNOSTIC.tif"
        with rasterio.open(tif_path, "w", driver="GTiff", width=base_shape[1], height=base_shape[0],
                           count=1, dtype="uint8", crs="EPSG:32649", transform=base_transform,
                           nodata=255, compress="deflate") as target:
            target.write(quality, 1)
            target.set_band_description(1, "tile-local tie quality; candidate only; no geometry correction")
        with rasterio.open(tif_path) as check:
            if str(check.crs) != "EPSG:32649" or check.res != (10.0, 10.0):
                raise RuntimeError(f"Local diagnostic TIF lost the 10 m base grid: {tif_path}")
            col_offset = (check.transform.c - base_info["transform"][2]) / 10.0
            row_offset = (check.transform.f - base_info["transform"][5]) / -10.0
            if abs(col_offset - round(col_offset)) > 1e-8 or abs(row_offset - round(row_offset)) > 1e-8:
                raise RuntimeError(f"Local diagnostic TIF is not pixel-aligned to the 10 m base: {tif_path}")
        quality_tifs.append({"path": str(tif_path), "sha256": _sha256(tif_path),
                             "from_date": pair["from_date"], "to_date": pair["to_date"],
                             "quality_codes": {"1": "INSUFFICIENT_SUPPORT", "2": "WEAK_LOCAL_CANDIDATE",
                                               "3": "NUMERIC_CANDIDATE_NOT_REVIEWED", "255": "OUTSIDE_LOCAL_CELL_GRID"}})

    input_hashes_after = {name: _sha256(Path(path)) for name, path in diagnostic["input_paths"].items()}
    tile_hashes_after = {path: _sha256(Path(path)) for path in tile_hashes_before}
    roof_hashes_after = {path: _sha256(Path(path)) for path in roof_hashes_before}
    base_hash_after = _sha256(base_raster)
    if (input_hashes_after != input_hashes_before or tile_hashes_after != tile_hashes_before or
            roof_hashes_after != roof_hashes_before or base_hash_after != base_hash_before):
        raise RuntimeError("A read-only source changed before local-field receipt publication")

    receipt = {
        "schema_version": "xian_r5.local_tile_lightglue_field_diagnostic.v1",
        "case_id": diagnostic["case_id"], "scope": "FIXED_CASE_LOCAL_ONLY",
        "input_mosaic_diagnostic": str(input_diagnostic),
        "input_mosaic_diagnostic_sha256": diagnostic_hash_before,
        "actual_dates_in_order": ordered_dates,
        "grid": diagnostic["grid"],
        "method": "ALIKED_LIGHTGLUE_MATCHES_FILTERED_BY_FROZEN_R4_BUFFER_AND_COLOR_PROFILE",
        "transform_model": LOCAL_TRANSFORM_MODEL,
        "model": {"receipt_path": str(model_receipt_path), "receipt_sha256": model_receipt_hash_before,
                  "lightglue_commit": model_receipt.get("lightglue_commit"),
                  "official_aliked_source": model_receipt.get("official_aliked_source"),
                  "device": str(device), "device_preference": device_preference,
                  "inference_reproducibility": (
                      "CPU_SINGLE_THREAD_PREFERENCE;DETERMINISTIC_RANSAC_SEED_PER_CORRESPONDENCE_SET"
                      if device.type == "cpu" else
                      "CUDA_SEEDED_BUT_BITWISE_MATCH_REPRODUCIBILITY_NOT_GUARANTEED;DETERMINISTIC_RANSAC_SEED_PER_CORRESPONDENCE_SET"),
                  "weights_sha256": model_hashes_before},
        "mask": {"roof_exclusion_buffer_m": roof_exclusion_buffer_m,
                 "R4_features_in_local_query": roof_count, "invalid_R4_features": invalid_roofs,
                 "color_filter_profile": color_filter_profile,
                 "ground_mask_reviewed": False,
                 "semantics": ("UNREVIEWED_CANDIDATE_FEATURES;R4_BUFFERED;GREEN_AND_DARK_PIXELS_EXCLUDED;NOT_VERIFIED_STABLE_GROUND"
                               if color_filter_profile == "green-dark-exclusion" else
                               "UNREVIEWED_CANDIDATE_FEATURES;R4_BUFFERED;NO_COLOR_FILTER;NOT_VERIFIED_STABLE_GROUND")},
        "cell_size_tiles": patch_tiles,
        "cell_partition": {
            "tile_cols": nx, "tile_rows": ny,
            "cell_count": len(patch_windows),
            "edge_cells_clipped": any(
                rows != patch_tiles or cols != patch_tiles
                for _, _, _, _, rows, cols in patch_windows),
            "semantics": "NORTH_WEST_ANCHORED_PATCHES;RIGHT_AND_BOTTOM_EDGE_CELLS_CLIPPED_TO_SOURCE_TILE_EXTENT",
        },
        "candidate_gate": {"min_matches": 6, "min_inliers": 8, "min_inlier_ratio": 0.25,
                           "min_coverage_4x4": 0.25,
                           "max_p90_residual_m": max_p90_residual_m,
                           "ransac_threshold_px": 3.0,
                           "residual_gate_semantics": "P90_INLIER_RESIDUAL_AFTER_AFFINE_CONVERSION_TO_PROJECTED_METRES"},
        "pairwise_fields": pair_records,
        "three_date_cycle": {
            "cells": cycle_rows,
            "interpretation": (
                "ARITHMETIC_ONLY_NOT_VALIDATION;WEAK_OR_UNREVIEWED_VECTORS_NEVER_APPLIED"),
        },
        "outputs": {"candidate_vector": str(vector_path), "candidate_csv": str(csv_path),
                    "quality_tifs": quality_tifs,
                    "match_review_contact_sheets": match_preview_outputs},
        "base_raster": {**base_info, "sha256_after": base_hash_after,
                         "grid_anchor_only": True},
        "input_sha256_before": input_hashes_before,
        "input_sha256_after": input_hashes_after,
        "roof_geometry_sha256_before": roof_hashes_before,
        "roof_geometry_sha256_after": roof_hashes_after,
        "image_tile_sha256_before": tile_hashes_before,
        "image_tile_sha256_after": tile_hashes_after,
        "registration_promoted": False, "field_interpolated": False,
        "registration_applied": False,
        "formal_ground_footprint_count": 0,
        "interpretation": "ENGINEERING_DIAGNOSTIC_NOT_GROUND_CORRECTION_OR_FORMAL_CHANGE_PRODUCT",
    }
    (output_dir / "LOCAL_TILE_FIELD_DIAGNOSTIC.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-diagnostic", type=Path, required=True,
                        help="Existing actual-date mosaic diagnostic JSON for one fixed case")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-tiles", type=int, default=3,
                        help="Square local registration cells measured in z20 tiles")
    parser.add_argument("--roof-exclusion-buffer-m", type=float, default=20.0)
    parser.add_argument("--max-p90-residual-m", type=float, default=2.0,
                        help="Maximum inlier p90 residual in the projected CRS metres; affects candidate strength only")
    parser.add_argument("--color-filter-profile", choices=("green-dark-exclusion", "none"),
                        default="green-dark-exclusion",
                        help="Candidate feature mask only; 'none' keeps all valid pixels outside the frozen R4 buffer, not verified ground")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="Inference device; CPU single-thread is the reproducible default; CUDA remains diagnostic and may vary across runs")
    parser.add_argument("--model-receipt", type=Path, default=MODEL_RECEIPT)
    parser.add_argument("--base-raster", type=Path, required=True,
                        help="10 m EPSG:32649 reference mosaic used for local-field QA")
    args = parser.parse_args()
    result = run(args.input_diagnostic, args.output, patch_tiles=args.patch_tiles,
                  roof_exclusion_buffer_m=args.roof_exclusion_buffer_m,
                  max_p90_residual_m=args.max_p90_residual_m,
                  color_filter_profile=args.color_filter_profile,
                  device_preference=args.device,
                  model_receipt_path=args.model_receipt, base_raster=args.base_raster)
    print(json.dumps({"case_id": result["case_id"], "output": str(args.output),
                      "pairwise_cells": [pair["status_counts"] for pair in result["pairwise_fields"]],
                      "registered": result["registration_applied"]}, ensure_ascii=False))
