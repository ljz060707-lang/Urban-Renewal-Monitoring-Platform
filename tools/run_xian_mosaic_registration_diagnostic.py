#!/usr/bin/env python3
"""Estimate bounded, date-coherent high-resolution mosaic ties (candidate only).

The tile-date registry and frozen R4 geometries are read-only. Roof exclusion
and color screening create a ground-feature candidate mask, not verified stable
ground. No fitted transform is applied to imagery, geometries, or products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

try:
    import cv2
except (ImportError, OSError):  # bounded mosaic assembly does not require OpenCV
    cv2 = None
import fiona
import numpy as np
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.reconstruction.registration import (
    estimate_mosaic_phase_registration,
    estimate_mosaic_registration,
)


STAGE2 = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\17_stage2")
OUTPUT_ROOT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction")
FIXED_MAP = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv"
FIXED_GEOMETRY = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_GEOMETRY.gpkg"
R4_FREEZE = STAGE2 / "13_receipts" / "R4_FREEZE_STAGE2.json"
RAW_ROOT = Path(r"F:\xian\xian")
DATE_CSV = RAW_ROOT / "date_audit" / "tables" / "tile_actual_date.csv"
SCENE = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
SCENE_TO_EPOCH = {scene: epoch for epoch, scene in SCENE.items()}
GRID_ORIGIN_X, GRID_ORIGIN_Y = 420534, 311895
GRID_ORIGIN_LON, GRID_ORIGIN_LAT = 108.75778198242188, 34.16130065917969
GRID_STEP_DEG = 0.0006866455078125
TO_UTM = Transformer.from_crs(4326, 32649, always_xy=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_duplicate_date_sources(sources: list[tuple[str, Path]]):
    """Choose one source only when same-date duplicate tiles are byte-identical."""
    ordered = sorted(sources, key=lambda item: item[0])
    if not ordered:
        return None, {"status": "NO_EXACT_DATE_TILE_SOURCE", "sources": []}
    records = [{"epoch": epoch, "path": str(path), "sha256": _sha256(path)}
               for epoch, path in ordered]
    if len({item["sha256"] for item in records}) > 1:
        return None, {"status": "DUPLICATE_ACTUAL_DATE_SOURCE_CONFLICT_SKIPPED",
                      "sources": records}
    selected = (ordered[0][0], ordered[0][1], records[0]["sha256"])
    status = ("DUPLICATE_ACTUAL_DATE_IDENTICAL_SOURCE_COLLAPSED"
              if len(ordered) > 1 else "SINGLE_EXACT_DATE_SOURCE")
    return selected, {"status": status, "sources": records,
                      "selected_epoch": selected[0]}


def _group_actual_date_rows(rows: list[dict[str, str]]) -> tuple[dict[str, list[str]], dict[str, set[tuple[int, int]]], int]:
    epochs_by_date: dict[str, set[str]] = defaultdict(set)
    centers_by_date: dict[str, set[tuple[int, int]]] = defaultdict(set)
    unresolved_date_rows = 0
    for row in rows:
        epoch = row.get("epoch_label", "")
        actual_date = row.get("actual_date", "")
        try:
            date.fromisoformat(actual_date)
        except (TypeError, ValueError):
            unresolved_date_rows += 1
            continue
        if epoch not in SCENE:
            unresolved_date_rows += 1
            continue
        epochs_by_date[actual_date].add(epoch)
        if row.get("tile_x") and row.get("tile_y"):
            centers_by_date[actual_date].add((int(row["tile_x"]), int(row["tile_y"])))
    return ({actual_date: sorted(epochs) for actual_date, epochs in epochs_by_date.items()},
            centers_by_date, unresolved_date_rows)


def _fixed_case(case_id: str) -> tuple[dict[str, list[str]], dict[str, set[tuple[int, int]]], int]:
    rows = []
    with FIXED_MAP.open("r", encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row.get("case_id") == case_id
                and row.get("pixel_available", "").lower() == "true"]
    if not rows:
        raise ValueError(f"No pixel-available fixed-case rows for {case_id}")
    epochs_by_date, centers_by_date, unresolved_date_rows = _group_actual_date_rows(rows)
    if not epochs_by_date:
        raise ValueError(f"No actual-date-resolved fixed-case rows for {case_id}")
    return epochs_by_date, centers_by_date, unresolved_date_rows


def _wanted_tiles(centers: dict[str, set[tuple[int, int]]], radius_tiles: int) -> set[tuple[int, int]]:
    return {(x + dx, y + dy)
            for epoch_centers in centers.values()
            for x, y in epoch_centers
            for dx in range(-radius_tiles, radius_tiles + 1)
            for dy in range(-radius_tiles, radius_tiles + 1)}


def _tile_catalog(epochs: set[str], wanted: set[tuple[int, int]]) -> dict[tuple[str, int, int], str | None]:
    xs = [x for x, _ in wanted]
    ys = [y for _, y in wanted]
    bounds = (min(xs), max(xs), min(ys), max(ys))
    catalog: dict[tuple[str, int, int], str | None] = {}
    with DATE_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            epoch = SCENE_TO_EPOCH.get(row["task_name"])
            if epoch not in epochs:
                continue
            x, y = int(row["tile_x"]), int(row["tile_y"])
            if not (bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]):
                continue
            pixel = RAW_ROOT / row["task_name"] / "20" / str(x) / f"gesh_{x}_{y}_20.jpg"
            actual_date = row.get("actual_capture_date") or None
            value = (actual_date if row.get("tile_present") == "1" and actual_date
                     and pixel.is_file() else None)
            key = (epoch, x, y)
            if key in catalog and catalog[key] != value:
                catalog[key] = None
            else:
                catalog[key] = value
    return catalog


def _grid_transform(x0: int, y0: int, nx: int, ny: int) -> Affine:
    lon_w = GRID_ORIGIN_LON + (x0 - GRID_ORIGIN_X) * GRID_STEP_DEG
    lat_s = GRID_ORIGIN_LAT + (y0 - GRID_ORIGIN_Y) * GRID_STEP_DEG
    lon_e, lat_n = lon_w + nx * GRID_STEP_DEG, lat_s + ny * GRID_STEP_DEG
    tl = TO_UTM.transform(lon_w, lat_n)
    tr = TO_UTM.transform(lon_e, lat_n)
    bl = TO_UTM.transform(lon_w, lat_s)
    return Affine((tr[0] - tl[0]) / (nx * 256), (bl[0] - tl[0]) / (ny * 256), tl[0],
                  (tr[1] - tl[1]) / (nx * 256), (bl[1] - tl[1]) / (ny * 256), tl[1])


def _load_building_exclusion(freeze: dict, epochs: list[str], bounds: tuple[float, ...], buffer_m: float):
    geometries = []
    receipts = {}
    for epoch in epochs:
        entry = freeze["files"][f"geometry_{epoch}"]
        path = Path(entry["path"])
        actual_hash = _sha256(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Frozen R4 geometry hash mismatch: {path}")
        receipts[epoch] = {"path": str(path), "sha256": actual_hash,
                           "freeze_status": "FROZEN_SOURCE_HASH_VERIFIED"}
        with fiona.open(path, layer="buildings") as source:
            if source.crs.to_epsg() != 32649:
                raise ValueError(f"Unexpected R4 geometry CRS: {path}")
            for feature in source.filter(bbox=bounds):
                if feature.get("geometry"):
                    geom = shape(feature["geometry"])
                    if not geom.is_empty and geom.is_valid:
                        geometries.append(geom)
    if geometries:
        exclusion = unary_union(geometries).buffer(buffer_m)
    else:
        exclusion = None
    return exclusion, len(geometries), receipts


def _assemble_mosaics(epochs_by_date: dict[str, list[str]],
                      catalog: dict[tuple[str, int, int], str | None],
                      x0: int, y0: int, nx: int, ny: int):
    mosaics = {}
    all_records = {}
    summaries = {}
    duplicate_source_audits = {}
    for actual_date, aliases in sorted(epochs_by_date.items(), key=lambda item: date.fromisoformat(item[0])):
        image = np.zeros((ny * 256, nx * 256, 3), dtype=np.uint8)
        valid = np.zeros((ny * 256, nx * 256), dtype=np.uint8)
        records = []
        duplicate_audits = []
        selected_epochs = Counter()
        histograms = {epoch: Counter() for epoch in aliases}
        for tile_y in range(y0 + ny - 1, y0 - 1, -1):
            for tile_x in range(x0, x0 + nx):
                tile_sources = []
                for epoch in aliases:
                    tile_date = catalog.get((epoch, tile_x, tile_y))
                    histograms[epoch][str(tile_date or "MISSING_OR_DATE_UNKNOWN")] += 1
                    if tile_date != actual_date:
                        continue
                    path = RAW_ROOT / SCENE[epoch] / "20" / str(tile_x) / f"gesh_{tile_x}_{tile_y}_20.jpg"
                    if path.is_file():
                        tile_sources.append((epoch, path))
                selected, source_audit = _resolve_duplicate_date_sources(tile_sources)
                if selected is None:
                    if source_audit["status"] != "NO_EXACT_DATE_TILE_SOURCE":
                        duplicate_audits.append({"tile_id": f"{tile_x}:{tile_y}", **source_audit})
                    continue
                epoch, path, digest_before = selected
                try:
                    with Image.open(path) as source:
                        rgb = np.asarray(source.convert("RGB"))
                except OSError:
                    continue
                if rgb.shape != (256, 256, 3):
                    continue
                digest_after = _sha256(path)
                if digest_after != digest_before:
                    duplicate_audits.append({"tile_id": f"{tile_x}:{tile_y}",
                                             "status": "TILE_CHANGED_DURING_READ_SKIPPED",
                                             "path": str(path), "sha256_before": digest_before,
                                             "sha256_after": digest_after})
                    continue
                col, row = tile_x - x0, (y0 + ny - 1) - tile_y
                y1, x1 = row * 256, col * 256
                image[y1:y1 + 256, x1:x1 + 256] = rgb
                valid[y1:y1 + 256, x1:x1 + 256] = 1
                selected_epochs[epoch] += 1
                records.append({"tile_id": f"{tile_x}:{tile_y}", "path": str(path),
                                "selected_source_epoch": epoch, "sha256": digest_before,
                                "sha256_after_read": digest_after,
                                "duplicate_source_status": source_audit["status"],
                                "candidate_source_epochs": [item["epoch"] for item in source_audit["sources"]]})
                if source_audit["status"] == "DUPLICATE_ACTUAL_DATE_IDENTICAL_SOURCE_COLLAPSED":
                    duplicate_audits.append({"tile_id": f"{tile_x}:{tile_y}", **source_audit})
        mosaics[actual_date] = {"rgb": image, "valid": valid}
        all_records[actual_date] = records
        duplicate_source_audits[actual_date] = duplicate_audits
        summaries[actual_date] = {
            "nominal_epoch_labels_collapsed": aliases,
            "selected_source_epoch_counts": dict(selected_epochs),
            "wanted_tiles": nx * ny,
            "exact_date_tiles_read": len(records),
            "exact_date_tile_coverage_fraction": len(records) / (nx * ny),
            "tile_date_histogram_by_epoch": {epoch: dict(counts) for epoch, counts in histograms.items()},
            "duplicate_source_audit_count": len(duplicate_audits),
        }
    return mosaics, all_records, summaries, duplicate_source_audits


def _ground_candidate(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 40)
    dark = hsv[:, :, 2] < 28
    return ~(green | dark)


def _local_grid(source: np.ndarray, target: np.ndarray, mask: np.ndarray, *,
                from_date: str, to_date: str, pixel_to_projected: tuple[float, float, float, float],
                grid_size: int, max_features: int, ratio_test: float,
                include_match_points: bool = False,
                feature_method: str = "SIFT") -> dict:
    height, width = mask.shape
    cells = []
    supported = []
    for row in range(grid_size):
        y0, y1 = row * height // grid_size, (row + 1) * height // grid_size
        for col in range(grid_size):
            x0, x1 = col * width // grid_size, (col + 1) * width // grid_size
            result = estimate_mosaic_registration(
                source[y0:y1, x0:x1], target[y0:y1, x0:x1], mask[y0:y1, x0:x1],
                from_date=from_date, to_date=to_date,
                pixel_to_projected=pixel_to_projected, max_features=max_features,
                ratio_test=ratio_test, include_match_points=include_match_points,
                feature_method=feature_method)
            item = {"row": row, "col": col, "pixel_window": [x0, y0, x1, y1], **result.as_dict()}
            cells.append(item)
            if result.translation_dx_m is not None and result.translation_dy_m is not None:
                supported.append([result.translation_dx_m, result.translation_dy_m])
    if supported:
        values = np.asarray(supported, dtype=np.float64)
        center = np.median(values, axis=0)
        summary = {"supported_cells": len(values),
                   "median_translation_m": [float(center[0]), float(center[1])],
                   "mad_east_m": float(np.median(np.abs(values[:, 0] - center[0]))),
                   "mad_north_m": float(np.median(np.abs(values[:, 1] - center[1]))),
                   "field_interpolated": False, "field_applied": False}
    else:
        summary = {"supported_cells": 0, "median_translation_m": None,
                   "mad_east_m": None, "mad_north_m": None,
                   "field_interpolated": False, "field_applied": False}
    return {"grid_rows": grid_size, "grid_cols": grid_size,
            "summary": summary, "cells": cells}


def _write_local_match_previews(output_dir: Path, pair_records: list[dict],
                                mosaics: dict[str, dict]) -> list[dict]:
    """Render inlier correspondence previews, prominently marked candidate-only."""
    preview_dir = output_dir / "match_previews"
    preview_dir.mkdir()
    records = []
    for pair in pair_records:
        source = mosaics[pair["from_date"]]["rgb"]
        target = mosaics[pair["to_date"]]["rgb"]
        for cell in pair["local_field_candidate"]["cells"]:
            model = cell.get("models", {}).get("translation", {})
            correspondences = model.get("inlier_correspondences_px", [])
            if cell.get("status") != "GROUND_MASKED_CANDIDATE_NOT_REVIEWED" or len(correspondences) < 4:
                continue
            x0, y0, x1, y1 = (int(value) for value in cell["pixel_window"])
            source_crop = cv2.cvtColor(source[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
            target_crop = cv2.cvtColor(target[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
            keypoints_source = [cv2.KeyPoint(float(row[0]), float(row[1]), 5.0)
                                for row in correspondences]
            keypoints_target = [cv2.KeyPoint(float(row[2]), float(row[3]), 5.0)
                                for row in correspondences]
            matches = [cv2.DMatch(index, index, 0.0)
                       for index in range(len(correspondences))]
            preview = cv2.drawMatches(
                source_crop, keypoints_source, target_crop, keypoints_target, matches,
                None, matchColor=(0, 220, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            header = np.full((58, preview.shape[1], 3), 32, dtype=np.uint8)
            cell_id = f"r{int(cell['row']):02d}c{int(cell['col']):02d}"
            cv2.putText(header, f"{pair['from_date']} -> {pair['to_date']} | {cell_id} | translation inliers={len(correspondences)}",
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(header, "UNREVIEWED CANDIDATE - feature type/stability unknown; no correction applied",
                        (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
            image = np.vstack((header, preview))
            path = preview_dir / f"{pair['from_date']}_to_{pair['to_date']}_{cell_id}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Could not write candidate match preview: {path}")
            records.append({"path": str(path), "sha256": _sha256(path),
                            "from_date": pair["from_date"], "to_date": pair["to_date"],
                            "cell_id": cell_id, "status": cell["status"],
                            "translation_inliers": len(correspondences),
                            "ground_mask_reviewed": False, "applied": False})
    return records


def _local_phase_grid(source: np.ndarray, target: np.ndarray, mask: np.ndarray, *,
                      from_date: str, to_date: str,
                      pixel_to_projected: tuple[float, float, float, float],
                      grid_size: int) -> dict:
    height, width = mask.shape
    cells = []
    supported = []
    responses = []
    for row in range(grid_size):
        y0, y1 = row * height // grid_size, (row + 1) * height // grid_size
        for col in range(grid_size):
            x0, x1 = col * width // grid_size, (col + 1) * width // grid_size
            result = estimate_mosaic_phase_registration(
                source[y0:y1, x0:x1], target[y0:y1, x0:x1], mask[y0:y1, x0:x1],
                from_date=from_date, to_date=to_date,
                pixel_to_projected=pixel_to_projected)
            cells.append({"row": row, "col": col, "pixel_window": [x0, y0, x1, y1],
                          **result.as_dict()})
            if result.translation_dx_m is not None and result.translation_dy_m is not None:
                supported.append([result.translation_dx_m, result.translation_dy_m])
                if result.phase_response is not None:
                    responses.append(result.phase_response)
    if supported:
        values = np.asarray(supported, dtype=np.float64)
        center = np.median(values, axis=0)
        summary = {"candidate_cells": len(values),
                   "median_translation_m": [float(center[0]), float(center[1])],
                   "mad_east_m": float(np.median(np.abs(values[:, 0] - center[0]))),
                   "mad_north_m": float(np.median(np.abs(values[:, 1] - center[1]))),
                   "median_phase_response": float(np.median(responses)) if responses else None,
                   "field_interpolated": False, "field_applied": False}
    else:
        summary = {"candidate_cells": 0, "median_translation_m": None,
                   "mad_east_m": None, "mad_north_m": None,
                   "median_phase_response": None,
                   "field_interpolated": False, "field_applied": False}
    return {"grid_rows": grid_size, "grid_cols": grid_size,
            "summary": summary, "cells": cells}


def run(case_id: str, output: Path, *, radius_tiles: int = 5,
        tile_window: tuple[int, int, int, int] | None = None,
        assembly_only: bool = False,
        roof_exclusion_buffer_m: float = 20.0, local_grid_size: int = 3,
        ratio_test: float = 0.72,
        color_filter_profile: str = "green-dark-exclusion",
        include_match_points: bool = False,
        feature_method: str = "SIFT") -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not 1 <= radius_tiles <= 10:
        raise ValueError("radius_tiles must be between 1 and 10 for bounded fixed-case runs")
    if roof_exclusion_buffer_m < 0 or local_grid_size < 1 or not 0.5 <= ratio_test < 1.0:
        raise ValueError("buffer/grid must be valid and ratio_test must be in [0.5, 1.0)")
    if color_filter_profile not in {"green-dark-exclusion", "none"}:
        raise ValueError(f"Unsupported color-filter profile: {color_filter_profile}")
    if assembly_only and include_match_points:
        raise ValueError("Match-point previews require registration, not assembly-only mode")
    feature_method = feature_method.upper()
    if feature_method not in {"SIFT", "ORB"}:
        raise ValueError(f"Unsupported registration feature method: {feature_method}")

    epochs_by_date, centers_by_date, unresolved_date_rows = _fixed_case(case_id)
    if not centers_by_date:
        raise ValueError(f"No dated tile centers are available for {case_id}")
    if tile_window is None:
        wanted = _wanted_tiles(centers_by_date, radius_tiles)
    else:
        if (len(tile_window) != 4
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in tile_window)):
            raise ValueError("tile_window must be integer (x0, y0, nx, ny)")
        x_start, y_start, tile_width, tile_height = tile_window
        if (tile_width < 1 or tile_height < 1
                or tile_width * tile_height > 625):
            raise ValueError("Explicit tile window must be positive and no larger than 625 tiles")
        wanted = [(x, y) for y in range(y_start, y_start + tile_height)
                  for x in range(x_start, x_start + tile_width)]
    x_values, y_values = [x for x, _ in wanted], [y for _, y in wanted]
    x0, y0 = min(x_values), min(y_values)
    nx, ny = max(x_values) - x0 + 1, max(y_values) - y0 + 1
    if nx * ny > 625:
        raise ValueError(f"Window too large for a fixed-case diagnostic: {nx}x{ny} tiles")

    candidate_epochs = {epoch for aliases in epochs_by_date.values() for epoch in aliases}
    catalog = _tile_catalog(candidate_epochs, wanted)
    aliases_by_date = {actual_date: sorted(aliases)
                       for actual_date, aliases in epochs_by_date.items()}
    mosaics, tile_records, mosaic_summaries, duplicate_source_audits = _assemble_mosaics(
        aliases_by_date, catalog, x0, y0, nx, ny)

    transform = _grid_transform(x0, y0, nx, ny)
    width, height = nx * 256, ny * 256
    corners = [transform * (0, 0), transform * (width, 0),
               transform * (0, height), transform * (width, height)]
    bounds = (min(p[0] for p in corners), min(p[1] for p in corners),
              max(p[0] for p in corners), max(p[1] for p in corners))
    freeze = json.loads(R4_FREEZE.read_text(encoding="utf-8"))
    exclusion, n_source_geometries, exclusion_sources = _load_building_exclusion(
        freeze, sorted(SCENE), bounds, roof_exclusion_buffer_m)
    if exclusion is not None:
        building_mask = rasterize([(exclusion, 1)], out_shape=(height, width),
                                  transform=transform, fill=0, dtype="uint8", all_touched=True).astype(bool)
    else:
        building_mask = np.zeros((height, width), dtype=bool)

    mask_semantics = ("COMMON_EXACT_DATE_TILE_COVERAGE_INTERSECTION;BUFFERED_UNION_OF_FROZEN_R4_BUILDINGS;"
                      + ("GREEN_AND_DARK_PIXELS_EXCLUDED;" if color_filter_profile == "green-dark-exclusion" else "NO_COLOR_FILTER;")
                      + "UNREVIEWED_CANDIDATE_FEATURES;NOT_VERIFIED_STABLE_GROUND")
    pixel_to_projected = (transform.a, transform.b, transform.d, transform.e)
    ordered_dates = sorted(mosaics, key=date.fromisoformat)
    pair_results = []
    pair_translation = {}
    phase_translation = {}
    date_pairs = (() if assembly_only else (
        (from_date, to_date)
        for source_index, from_date in enumerate(ordered_dates)
        for to_date in ordered_dates[source_index + 1:]
    ))
    for from_date, to_date in date_pairs:
            source = mosaics[from_date]["rgb"]
            target = mosaics[to_date]["rgb"]
            candidate_mask = (mosaics[from_date]["valid"].astype(bool)
                              & mosaics[to_date]["valid"].astype(bool)
                              & ~building_mask)
            if color_filter_profile == "green-dark-exclusion":
                candidate_mask &= _ground_candidate(source) & _ground_candidate(target)
            result = estimate_mosaic_registration(
                source, target, candidate_mask,
                from_date=from_date, to_date=to_date,
                pixel_to_projected=pixel_to_projected,
                max_features=12000, ratio_test=ratio_test,
                include_match_points=include_match_points,
                feature_method=feature_method)
            phase_result = estimate_mosaic_phase_registration(
                source, target, candidate_mask,
                from_date=from_date, to_date=to_date,
                pixel_to_projected=pixel_to_projected)
            local = _local_grid(
                source, target, candidate_mask,
                from_date=from_date, to_date=to_date,
                pixel_to_projected=pixel_to_projected,
                grid_size=local_grid_size, max_features=2500, ratio_test=ratio_test,
                include_match_points=include_match_points,
                feature_method=feature_method)
            local_phase = _local_phase_grid(
                source, target, candidate_mask,
                from_date=from_date, to_date=to_date,
                pixel_to_projected=pixel_to_projected,
                grid_size=local_grid_size)
            method_consistency = None
            if (result.translation_dx_m is not None and result.translation_dy_m is not None
                    and phase_result.translation_dx_m is not None
                    and phase_result.translation_dy_m is not None):
                method_consistency = float(np.hypot(
                    result.translation_dx_m - phase_result.translation_dx_m,
                    result.translation_dy_m - phase_result.translation_dy_m))
            pair_results.append({
                "from_date": from_date,
                "to_date": to_date,
                "interval_days": (date.fromisoformat(to_date) - date.fromisoformat(from_date)).days,
                "status": result.status,
                "global": result.as_dict(),
                "global_phase_correlation_candidate": phase_result.as_dict(),
                "global_sift_phase_difference_m": method_consistency,
                "local_field_candidate": local,
                "local_phase_field_candidate": local_phase,
                "candidate_mask_pixels": int(candidate_mask.sum()),
                "candidate_mask_fraction": float(candidate_mask.mean()),
                "mask_semantics": mask_semantics,
                "transform_applied": False,
            })
            if result.translation_dx_m is not None and result.translation_dy_m is not None:
                pair_translation[(from_date, to_date)] = np.array(
                    [result.translation_dx_m, result.translation_dy_m], dtype=np.float64)
            if (phase_result.translation_dx_m is not None
                    and phase_result.translation_dy_m is not None):
                phase_translation[(from_date, to_date)] = np.array(
                    [phase_result.translation_dx_m, phase_result.translation_dy_m], dtype=np.float64)

    cycle = None
    phase_cycle = None
    if assembly_only:
        cycle = {"status": "NOT_RUN_ASSEMBLY_ONLY", "field_applied": False}
        phase_cycle = {"status": "NOT_RUN_ASSEMBLY_ONLY", "field_applied": False}
    elif len(ordered_dates) == 3:
        first, middle, last = ordered_dates
        d01, d12, d02 = (pair_translation.get((first, middle)),
                         pair_translation.get((middle, last)),
                         pair_translation.get((first, last)))
        if d01 is not None and d12 is not None and d02 is not None:
            residual = d01 + d12 - d02
            cycle = {"status": "DIAGNOSTIC_ONLY_NOT_VALIDATION",
                     "closure_residual_m": [float(residual[0]), float(residual[1])],
                     "closure_norm_m": float(np.linalg.norm(residual)),
                     "field_applied": False}
        else:
            cycle = {"status": "NOT_COMPUTED_INSUFFICIENT_PAIR_TIES", "field_applied": False}
        p01, p12, p02 = (phase_translation.get((first, middle)),
                         phase_translation.get((middle, last)),
                         phase_translation.get((first, last)))
        if p01 is not None and p12 is not None and p02 is not None:
            phase_residual = p01 + p12 - p02
            phase_cycle = {"status": "DIAGNOSTIC_ONLY_NOT_VALIDATION",
                           "closure_residual_m": [float(phase_residual[0]), float(phase_residual[1])],
                           "closure_norm_m": float(np.linalg.norm(phase_residual)),
                           "field_applied": False}
        else:
            phase_cycle = {"status": "NOT_COMPUTED_INSUFFICIENT_PAIR_TIES", "field_applied": False}

    report = {
        "schema_version": "xian_r5.mosaic_registration_diagnostic.v1",
        "case_id": case_id,
        "scope": "FIXED_CASE_LOCAL_ONLY",
        "input_paths": {"fixed_case_map": str(FIXED_MAP),
                        "fixed_case_geometry": str(FIXED_GEOMETRY),
                        "actual_date_registry": str(DATE_CSV),
                        "r4_freeze_receipt": str(R4_FREEZE)},
        "input_sha256": {"fixed_case_map": _sha256(FIXED_MAP),
                         "fixed_case_geometry": _sha256(FIXED_GEOMETRY),
                         "actual_date_registry": _sha256(DATE_CSV),
                         "r4_freeze_receipt": _sha256(R4_FREEZE)},
        "grid": {"crs": "EPSG:32649", "tile_pixel_size": [256, 256],
                 "tile_window": {"x0": x0, "y0": y0, "nx": nx, "ny": ny},
                 "pixel_transform": list(transform), "bounds": bounds},
        "parameters": {"radius_tiles": radius_tiles,
                       "explicit_tile_window": list(tile_window) if tile_window else None,
                       "assembly_only": assembly_only,
                       "registration_status": (
                           "NOT_RUN_ASSEMBLY_ONLY" if assembly_only else "DIAGNOSTIC_CANDIDATES_WRITTEN"),
                       "roof_exclusion_buffer_m": roof_exclusion_buffer_m,
                       "local_grid_size": local_grid_size,
                       "ratio_test": ratio_test,
                       "sift_lowe_ratio_test": ratio_test if feature_method == "SIFT" else None,
                       "color_filter_profile": color_filter_profile,
                       "include_match_points": include_match_points,
                       "feature_method": feature_method,
                       "global_max_features": 12000,
                       "local_max_features": 2500,
                       "nominal_epochs_collapsed_by_actual_date": True,
                       "ground_mask_reviewed": False,
                       "registration_promoted": False,
                       "registration_applied": False},
        "actual_date_association": {
            "nominal_epoch_labels_by_actual_date": {
                actual_date: aliases for actual_date, aliases in sorted(
                    epochs_by_date.items(), key=lambda item: date.fromisoformat(item[0]))},
            "unresolved_fixed_case_rows_omitted_from_registration": unresolved_date_rows,
            "duplicate_acquisitions_collapsed_by_actual_date": True,
            "candidate_source_epochs_by_actual_date": aliases_by_date,
            "source_selection_policy": "PER_TILE_EXACT_DATE;COLLAPSE_ONLY_BYTE_IDENTICAL_DUPLICATES;SKIP_CONFLICTS",
        },
        "mask_semantics": mask_semantics,
        "frozen_r4_building_exclusion": {
            "feature_count_before_union": n_source_geometries,
            "buffer_m": roof_exclusion_buffer_m,
            "sources": exclusion_sources,
        },
        "epochs": mosaic_summaries,
        "submeter_tiles": tile_records,
        "duplicate_date_source_audits": duplicate_source_audits,
        "pairwise_candidates": pair_results,
        "match_point_coordinate_semantics": (
            "When enabled, each model's inlier_correspondences_px rows are "
            "[source_x, source_y, target_x, target_y] in input-array pixels; "
            "they are diagnostic matches, not reviewed ground controls."
            if include_match_points else None),
        "match_point_previews": [],
        "three_date_cycle_candidate": cycle,
        "three_date_phase_cycle_candidate": phase_cycle,
        "limitations": ([
            "Registration was not run; this receipt only records bounded exact-date mosaic assembly."
            if assembly_only else
            f"{feature_method} model fits are diagnostic candidates and are not applied to imagery or geometry.",
            "Candidate pixels outside buffered frozen R4 polygons, with optional colour exclusions, do not prove stable ground.",
            "No roof-to-ground parallax correction or confirmed ground footprint is produced by this run.",
        ]),
        "interpretation": (
            "ASSEMBLY_ONLY_EXACT_DATE_MOSAIC_INPUT;NO_REGISTRATION_ESTIMATED"
            if assembly_only else
            "ENGINEERING_DIAGNOSTIC_ONLY_NOT_GROUND_CORRECTION_OR_FORMAL_CHANGE_PRODUCT"),
    }
    output.mkdir(parents=True)
    if include_match_points:
        report["match_point_previews"] = _write_local_match_previews(
            output, pair_results, mosaics)
    (output / "MOSAIC_REGISTRATION_DIAGNOSTIC.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-tiles", type=int, default=5)
    parser.add_argument("--tile-window", type=int, nargs=4,
                        metavar=("X0", "Y0", "NX", "NY"),
                        help="Optional explicit bounded source-tile window; used instead of radius_tiles")
    parser.add_argument("--assembly-only", action="store_true",
                        help="Assemble and hash a bounded exact-date mosaic receipt without running SIFT/phase registration")
    parser.add_argument("--roof-exclusion-buffer-m", type=float, default=20.0)
    parser.add_argument("--local-grid-size", type=int, default=3)
    parser.add_argument("--color-filter-profile", choices=("green-dark-exclusion", "none"),
                        default="green-dark-exclusion",
                        help="Diagnostic-only pixel filtering; 'none' retains vegetation/shadow candidates outside the frozen R4 buffer")
    parser.add_argument("--ratio-test", type=float, default=0.72,
                        help="SIFT Lowe ratio threshold; diagnostic sensitivity only, never acceptance")
    parser.add_argument("--include-match-points", action="store_true",
                        help="Store local model inlier coordinates and write candidate-only review PNGs")
    parser.add_argument("--feature-method", choices=("SIFT", "ORB"), default="SIFT",
                        help="Local feature detector/descriptor used for diagnostic matching")
    args = parser.parse_args()
    result = run(args.case_id, args.output, radius_tiles=args.radius_tiles,
                 tile_window=tuple(args.tile_window) if args.tile_window else None,
                 assembly_only=args.assembly_only,
                 roof_exclusion_buffer_m=args.roof_exclusion_buffer_m,
                 local_grid_size=args.local_grid_size,
                 ratio_test=args.ratio_test,
                 color_filter_profile=args.color_filter_profile,
                 include_match_points=args.include_match_points,
                 feature_method=args.feature_method)
    print(json.dumps({
        "output": str(args.output),
        "feature_method": result["parameters"]["feature_method"],
        "match_point_previews": len(result["match_point_previews"]),
        "epochs": {key: {"tiles": value["exact_date_tiles_read"],
                         "coverage": value["exact_date_tile_coverage_fraction"]}
                   for key, value in result["epochs"].items()},
        "pairs": [{"from": row["from_date"], "to": row["to_date"],
                   "status": row["status"],
                   "translation_m": [row["global"]["translation_dx_m"],
                                      row["global"]["translation_dy_m"]],
                   "local_supported": row["local_field_candidate"]["summary"]["supported_cells"],
                   "phase_translation_m": [row["global_phase_correlation_candidate"]["translation_dx_m"],
                                            row["global_phase_correlation_candidate"]["translation_dy_m"]],
                   "phase_response": row["global_phase_correlation_candidate"]["phase_response"],
                   "phase_local_candidates": row["local_phase_field_candidate"]["summary"]["candidate_cells"]}
                  for row in result["pairwise_candidates"]],
        "cycle": result["three_date_cycle_candidate"],
    }, ensure_ascii=False))
