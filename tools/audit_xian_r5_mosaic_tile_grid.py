"""Audit fixed-case source JPEG windows against stitched and 10m raster grids.

This is a read-only, bounded audit: it opens raster metadata and the exact
source tiles already listed by a mosaic-registration diagnostic. It does not
scan or rewrite full-city imagery and does not apply a positional correction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xian_r5.alignment import MosaicTileGrid, axis_aligned_window_for_bounds
from xian_r5.alignment.mosaic_grid import sha256_file


SOURCE_ROOT = Path(r"F:\xian\xian")
BASE_ROOT = Path(r"F:\xian\xian底图\xian搴曞浘")
DEFAULT_DIAGNOSTIC = Path(
    r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction"
    r"\B04-TQA10_mosaic_registration_diag_20260925_01"
    r"\MOSAIC_REGISTRATION_DIAGNOSTIC.json"
)
DEFAULT_OUTPUT = Path(
    r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction"
    r"\B04_TQA10_MOSAIC_COMPONENT_WINDOW_AUDIT_20260926.json"
)
SCENES = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
TILE_ID_RE = re.compile(r"(\d+):(\d+)")


def _raster_profile(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as raster:
        if raster.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        return {
            "path": str(path),
            "crs": raster.crs.to_string(),
            "width": raster.width,
            "height": raster.height,
            "transform": list(tuple(float(value) for value in tuple(raster.transform)[:6])),
            "bounds": [float(value) for value in raster.bounds],
            "resolution": [float(value) for value in raster.res],
            "size_bytes": path.stat().st_size,
            "sha256_status": "NOT_COMPUTED_FULL_RASTER_SCAN",
            "tags": raster.tags(),
        }


def _base10m_profiles() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles = [_raster_profile(BASE_ROOT / f"Xian_{year}_TrueColor.tif")
                for year in range(2017, 2023)]
    anchor = profiles[0]
    signature = (anchor["crs"], anchor["width"], anchor["height"],
                 tuple(anchor["transform"]), tuple(anchor["bounds"]))
    for profile in profiles[1:]:
        current = (profile["crs"], profile["width"], profile["height"],
                   tuple(profile["transform"]), tuple(profile["bounds"]))
        if current != signature:
            raise ValueError("The six annual 10m rasters do not share one exact spatial grid")
    if anchor["crs"] != "EPSG:32649" or anchor["resolution"] != [10.0, 10.0]:
        raise ValueError("Expected shared 10m base rasters in EPSG:32649")
    return profiles, {
        "crs": anchor["crs"],
        "width": anchor["width"],
        "height": anchor["height"],
        "transform": anchor["transform"],
        "bounds": anchor["bounds"],
        "resolution": anchor["resolution"],
        "role": "SPATIAL_OVERLAY_GRID_ONLY_NOT_GROUND_CONTROL",
    }


def _project_component(receipt: dict[str, Any], transformer: Transformer,
                       base_profile: dict[str, Any]) -> dict[str, Any]:
    west, south, east, north = receipt["window_bounds_epsg4326"]
    lonlat_corners = ((west, north), (east, north), (east, south), (west, south))
    projected = [transformer.transform(lon, lat) for lon, lat in lonlat_corners]
    bounds = [min(x for x, _ in projected), min(y for _, y in projected),
              max(x for x, _ in projected), max(y for _, y in projected)]
    base_left, base_bottom, base_right, base_top = base_profile["bounds"]
    left, bottom = max(bounds[0], base_left), max(bounds[1], base_bottom)
    right, top = min(bounds[2], base_right), min(bounds[3], base_top)
    window_record = None
    if left < right and bottom < top:
        window_record = axis_aligned_window_for_bounds(
            [left, bottom, right, top], base_profile["transform"],
            base_profile["width"], base_profile["height"],
        )
    return {
        "source_component_corners_epsg32649": [[float(x), float(y)] for x, y in projected],
        "source_component_bounds_epsg32649": [float(value) for value in bounds],
        "base10m_bounding_pixel_window_col_row_width_height": window_record,
        "base10m_spatial_grid_role": "OVERLAY_ONLY_NOT_POSITIONAL_ACCURACY_EVIDENCE",
    }


def run(diagnostic_path: Path, output_path: Path) -> dict[str, Any]:
    diagnostic_path, output_path = Path(diagnostic_path), Path(output_path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output_path}")
    diagnostic_sha_before = sha256_file(diagnostic_path)
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic.get("schema_version") != "xian_r5.mosaic_registration_diagnostic.v1":
        raise ValueError("Unsupported source mosaic diagnostic schema")
    if diagnostic.get("scope") != "FIXED_CASE_LOCAL_ONLY":
        raise ValueError("This audit accepts fixed-case mosaic diagnostics only")

    grid_values = diagnostic["grid"]
    tile_window = grid_values["tile_window"]
    x0, y0 = int(tile_window["x0"]), int(tile_window["y0"])
    nx, ny = int(tile_window["nx"]), int(tile_window["ny"])
    if min(nx, ny) < 1:
        raise ValueError("Fixed-case source tile window must be non-empty")
    expected_tile_ids = {
        (x, y) for y in range(y0, y0 + ny) for x in range(x0, x0 + nx)
    }

    date_epoch_map = diagnostic["actual_date_association"][
        "candidate_source_epochs_by_actual_date"
    ]
    submeter_tiles = diagnostic.get("submeter_tiles")
    if not isinstance(submeter_tiles, dict) or not submeter_tiles:
        raise ValueError("Fixed-case diagnostic has no per-date source tile receipts")

    base_profiles, base_profile = _base10m_profiles()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    grids: dict[str, MosaicTileGrid] = {}
    scene_summaries: dict[str, dict[str, Any]] = {}
    component_records: list[dict[str, Any]] = []
    date_summaries: list[dict[str, Any]] = []
    union_points: list[tuple[float, float]] = []

    for actual_date, entries in sorted(submeter_tiles.items()):
        epoch_values = {str(entry.get("selected_source_epoch") or "") for entry in entries}
        if len(epoch_values) != 1 or "" in epoch_values:
            raise ValueError(f"Source epoch is not uniquely resolved for {actual_date}")
        source_epoch = next(iter(epoch_values))
        if source_epoch not in SCENES:
            raise ValueError(f"Unknown GEID task source epoch: {source_epoch}")
        if source_epoch not in date_epoch_map.get(actual_date, []):
            raise ValueError(f"Per-tile source epoch conflicts with actual-date receipt: {actual_date}")
        scene = SCENES[source_epoch]
        grid = grids.get(scene)
        if grid is None:
            grid = MosaicTileGrid.open(
                SOURCE_ROOT / f"{scene}.geid",
                SOURCE_ROOT / f"{scene}_combined" / f"{scene}_zoom_20.tif",
                SOURCE_ROOT / scene / "20",
            )
            grids[scene] = grid
            scene_summaries[scene] = grid.summary()

        seen: set[tuple[int, int]] = set()
        records_for_date = []
        for entry in entries:
            match = TILE_ID_RE.fullmatch(str(entry.get("tile_id", "")))
            if match is None:
                raise ValueError(f"Malformed tile ID for {actual_date}: {entry.get('tile_id')}")
            tile_x, tile_y = (int(value) for value in match.groups())
            coord = (tile_x, tile_y)
            if coord not in expected_tile_ids or coord in seen:
                raise ValueError(f"Unexpected or duplicate component tile {coord} for {actual_date}")
            seen.add(coord)
            tile_receipt = grid.component_receipt(
                tile_x,
                tile_y,
                actual_date=actual_date,
                actual_date_source_ref=(
                    f"{diagnostic_path}#submeter_tiles/{actual_date}/{tile_x}:{tile_y}"
                ),
                expected_sha256=entry.get("sha256"),
                expected_path=entry.get("path"),
            )
            after_read_sha = entry.get("sha256_after_read")
            if after_read_sha and tile_receipt["source_tile_sha256"] != after_read_sha:
                raise ValueError(f"Tile changed after registration diagnostic read: {coord}")
            tile_receipt["selected_source_epoch"] = source_epoch
            tile_receipt["duplicate_source_status"] = entry.get("duplicate_source_status")
            tile_receipt.update(_project_component(tile_receipt, to_utm, base_profile))
            records_for_date.append(tile_receipt)

        if seen != expected_tile_ids:
            missing = sorted(expected_tile_ids - seen)
            raise ValueError(f"Date {actual_date} lacks fixed-case source tiles: {missing}")
        records_for_date.sort(
            key=lambda record: (record["pixel_window_col_row_width_height"][1],
                                record["pixel_window_col_row_width_height"][0])
        )
        component_records.extend(records_for_date)
        for record in records_for_date:
            west, south, east, north = record["source_component_bounds_epsg32649"]
            union_points.extend(((west, south), (east, north)))
        date_summaries.append({
            "actual_date": actual_date,
            "selected_source_epoch": source_epoch,
            "scene_id": scene,
            "expected_components": len(expected_tile_ids),
            "verified_components": len(records_for_date),
            "missing_components": 0,
        })

    if sha256_file(diagnostic_path) != diagnostic_sha_before:
        raise RuntimeError("Mosaic registration diagnostic changed during the audit")
    signatures = {
        (grid.mosaic_crs, grid.mosaic_width_px, grid.mosaic_height_px,
         grid.mosaic_transform, grid.mosaic_bounds, grid.min_tile_x,
         grid.min_tile_y, grid.max_tile_x, grid.max_tile_y,
         grid.tile_width_px, grid.tile_height_px)
        for grid in grids.values()
    }
    same_source_grid = len(signatures) == 1
    if not same_source_grid:
        raise ValueError("Actual-date source mosaics do not share one exact source pixel grid")

    projected_union = [min(x for x, _ in union_points), min(y for _, y in union_points),
                       max(x for x, _ in union_points), max(y for _, y in union_points)]
    diagnostic_bounds = [float(value) for value in grid_values["bounds"]]
    bounds_delta = [projected_union[i] - diagnostic_bounds[i] for i in range(4)]
    max_bounds_delta = max(abs(value) for value in bounds_delta)
    if max_bounds_delta > 0.02:
        raise ValueError(
            "Source TIFF tile windows do not reproduce the fixed-case projected grid "
            f"(max bounds residual {max_bounds_delta:.6f} m)"
        )

    report = {
        "schema_version": "xian_r5.mosaic_component_window_audit.v1",
        "case_id": diagnostic.get("case_id"),
        "scope": "FIXED_CASE_ONLY_NO_CITYWIDE_SCAN_OR_OUTPUT",
        "source_diagnostic": {
            "path": str(diagnostic_path.resolve()),
            "sha256": diagnostic_sha_before,
            "sha256_unchanged_after_read": True,
        },
        "geid_and_stitched_mosaics_by_scene": scene_summaries,
        "actual_date_component_counts": date_summaries,
        "common_actual_date_source_grid_verified": same_source_grid,
        "fixed_case_tile_window": {
            "x0": x0,
            "y0": y0,
            "nx": nx,
            "ny": ny,
            "tile_size_px": [int(value) for value in grid_values["tile_pixel_size"]],
            "component_count_per_date": len(expected_tile_ids),
        },
        "reference_10m_rasters": base_profiles,
        "reference_10m_common_grid": base_profile,
        "projected_fixed_case_bounds_epsg32649": projected_union,
        "diagnostic_fixed_case_bounds_epsg32649": diagnostic_bounds,
        "projected_bounds_residual_m": bounds_delta,
        "max_projected_bounds_residual_m": max_bounds_delta,
        "component_windows": component_records,
        "formal_ground_correction": "NOT_APPLIED_NO_REVIEWED_GROUND_EVIDENCE",
        "roof_to_ground_footprint_correction": "NOT_ESTIMATED_BY_THIS_AUDIT",
        "positional_accuracy_claimed": False,
        "output_role": "SOURCE_TRACEABILITY_AND_COMMON_GRID_WINDOW_AUDIT_ONLY",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as sink:
        json.dump(report, sink, ensure_ascii=False, indent=2)
        sink.write("\n")
    return {
        "output": str(output_path.resolve()),
        "actual_dates": len(date_summaries),
        "components_verified": len(component_records),
        "max_projected_bounds_residual_m": max_bounds_delta,
        "formal_ground_correction": report["formal_ground_correction"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.diagnostic, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
