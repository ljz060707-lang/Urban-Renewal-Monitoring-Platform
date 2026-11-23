#!/usr/bin/env python3
"""Write candidate-only, spatially composed B04 alignment evidence.

The outputs are local displacement candidates on the verified 10 m grid, not
warped imagery or ground footprints. The annual reference's capture date and
per-pixel mosaic source provenance remain unknown.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import fiona
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import mapping
from shapely.strtree import STRtree

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment.field import LocalDisplacementField
from xian_r5.alignment.field_io import (
    read_direct_base_tie_fields,
    read_lightglue_local_field_candidates,
)
from xian_r5.alignment.spatial_composition import compose_spatial_alignment_fields


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_nonoverlapping_fields(fields: Iterable[LocalDisplacementField]) -> None:
    for field in fields:
        cells = list(field.cells)
        tree = STRtree([cell.support_geometry for cell in cells])
        for left, cell in enumerate(cells):
            for raw_right in tree.query(cell.support_geometry):
                right = int(raw_right)
                if right <= left:
                    continue
                overlap_area = cell.support_geometry.intersection(
                    cells[right].support_geometry).area
                if overlap_area > 1e-6:
                    raise ValueError(
                        f"Overlapping composed supports cannot be rasterized safely: "
                        f"{field.moving_date}/{cell.cell_id}/{cells[right].cell_id}")


def _local_grid(base_path: Path, fields: list[LocalDisplacementField]) -> dict[str, object]:
    cells = [cell for field in fields for cell in field.cells]
    if not cells:
        raise ValueError("No spatially composed candidate cells are available for raster output")
    xmin = min(cell.support_geometry.bounds[0] for cell in cells)
    ymin = min(cell.support_geometry.bounds[1] for cell in cells)
    xmax = max(cell.support_geometry.bounds[2] for cell in cells)
    ymax = max(cell.support_geometry.bounds[3] for cell in cells)
    with rasterio.open(base_path) as base:
        if (base.crs is None or base.crs.to_epsg() != 32649
                or base.res != (10.0, 10.0)):
            raise ValueError("Candidate alignment TIFs require the 10 m EPSG:32649 reference raster")
        transform = base.transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise ValueError("Reference raster must be north-up for pixel-aligned candidate products")
        col0 = max(0, int(np.floor((xmin - transform.c) / transform.a)))
        col1 = min(base.width, int(np.ceil((xmax - transform.c) / transform.a)))
        row0 = max(0, int(np.floor((transform.f - ymax) / -transform.e)))
        row1 = min(base.height, int(np.ceil((transform.f - ymin) / -transform.e)))
        if col1 <= col0 or row1 <= row0:
            raise ValueError("Composed support lies outside the reference raster extent")
        local_transform = transform * Affine.translation(col0, row0)
        return {
            "crs": "EPSG:32649",
            "resolution_m": [10.0, 10.0],
            "width": col1 - col0,
            "height": row1 - row0,
            "window": [col0, row0, col1 - col0, row1 - row0],
            "transform": list(local_transform)[:6],
            "reference_transform": list(transform)[:6],
            "reference_shape": [base.height, base.width],
        }


def _write_vectors(path: Path, fields: list[LocalDisplacementField]) -> int:
    schema = {"geometry": "Unknown", "properties": {
        "field_id": "str:96", "cell_id": "str:96", "actual_date": "str:16",
        "reference_label": "str:80", "status": "str:72",
        "reference_date_status": "str:48",
        "candidate_dx_m": "float", "candidate_dy_m": "float",
        "support_area_m2": "float", "cycle_dof": "int",
        "closure_max_m": "float", "reviewed": "int",
        "ground_evidence": "int", "applied": "int", "source_refs": "str:32767"}}
    count = 0
    with fiona.open(path, "w", driver="GPKG", layer="spatial_alignment_candidates",
                    schema=schema, crs="EPSG:32649") as sink:
        for field in fields:
            for cell in field.cells:
                metadata = cell.metadata
                sink.write({"geometry": mapping(cell.support_geometry), "properties": {
                    "field_id": field.field_id,
                    "cell_id": cell.cell_id,
                    "actual_date": field.moving_date,
                    "reference_label": field.reference_label,
                    "status": cell.status,
                    "reference_date_status": field.reference_date_status,
                    "candidate_dx_m": cell.correction_dx_m_candidate,
                    "candidate_dy_m": cell.correction_dy_m_candidate,
                    "support_area_m2": cell.support_geometry.area,
                    "cycle_dof": metadata.get("cycle_degrees_of_freedom"),
                    "closure_max_m": metadata.get("closure_max_m"),
                    "reviewed": int(cell.reviewed),
                    "ground_evidence": int(cell.independent_ground_evidence),
                    "applied": int(cell.applied),
                    "source_refs": json.dumps(
                        metadata.get("source_candidate_cell_refs", []), ensure_ascii=False),
                }})
                count += 1
    return count


def _write_candidate_rasters(
    output: Path,
    fields: list[LocalDisplacementField],
    grid: dict[str, object],
) -> dict[str, dict[str, object]]:
    transform = Affine(*grid["transform"])
    shape = (int(grid["height"]), int(grid["width"]))
    profiles: dict[str, dict[str, object]] = {}
    quality_code_by_status = {
        "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY": 1,
        "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED": 2,
        "SPATIALLY_COMPOSED_CANDIDATE_TREE_NO_CYCLE_CHECK": 3,
    }
    for field in fields:
        offset_path = output / f"{field.moving_date}_to_2019_base_offset_CANDIDATE.tif"
        quality_path = output / f"{field.moving_date}_to_2019_base_quality_CANDIDATE.tif"
        dx_shapes = []
        dy_shapes = []
        quality_shapes = []
        status_counts: dict[str, int] = {}
        for cell in field.cells:
            status_counts[cell.status] = status_counts.get(cell.status, 0) + 1
            quality_code = quality_code_by_status.get(cell.status)
            if quality_code is None:
                raise ValueError(f"Unsupported spatial-composition status for TIF: {cell.status}")
            dx_shapes.append((cell.support_geometry, float(cell.correction_dx_m_candidate)))
            dy_shapes.append((cell.support_geometry, float(cell.correction_dy_m_candidate)))
            quality_shapes.append((cell.support_geometry, quality_code))

        dx = rasterize(dx_shapes, out_shape=shape, transform=transform,
                       fill=np.nan, dtype="float32", all_touched=False)
        dy = rasterize(dy_shapes, out_shape=shape, transform=transform,
                       fill=np.nan, dtype="float32", all_touched=False)
        quality = rasterize(quality_shapes, out_shape=shape, transform=transform,
                            fill=0, dtype="uint8", all_touched=False)
        common = {
            "driver": "GTiff", "width": int(grid["width"]),
            "height": int(grid["height"]), "crs": "EPSG:32649",
            "transform": transform, "compress": "deflate",
        }
        with rasterio.open(offset_path, "w", count=2, dtype="float32",
                           nodata=np.nan, **common) as target:
            target.write(dx, 1)
            target.write(dy, 2)
            target.set_band_description(1, "candidate_correction_east_m")
            target.set_band_description(2, "candidate_correction_north_m")
            target.update_tags(
                actual_date=field.moving_date,
                reference_label=field.reference_label or "",
                reference_acquisition_date_status=field.reference_date_status,
                correction_semantics="IMAGE_TO_ANNUAL_REFERENCE_GRID_CANDIDATE_ONLY",
                positional_accuracy="NOT_ESTABLISHED",
                roof_to_ground_parallax="NOT_RESOLVED",
                field_interpolated="false", applied="false",
            )
        with rasterio.open(quality_path, "w", count=1, dtype="uint8",
                           nodata=0, **common) as target:
            target.write(quality, 1)
            target.set_band_description(1, "spatial_alignment_candidate_quality")
            target.update_tags(
                quality_codes=json.dumps({
                    "0": "outside_composed_candidate_support",
                    "1": "cycle_screen_only_unreviewed_candidate",
                    "2": "cycle_screen_failed_unreviewed_candidate",
                    "3": "tree_no_cycle_check_unreviewed_candidate",
                }),
                field_interpolated="false", applied="false",
            )
        with rasterio.open(offset_path) as check:
            if check.crs.to_epsg() != 32649 or check.res != (10.0, 10.0):
                raise RuntimeError(f"Candidate offset TIF lost its 10 m CRS/grid: {offset_path}")
            if check.transform != transform or (check.height, check.width) != shape:
                raise RuntimeError(f"Candidate offset TIF transform/shape changed: {offset_path}")
        profiles[field.moving_date] = {
            "offset_tif": offset_path.name,
            "quality_tif": quality_path.name,
            "status_counts": status_counts,
            "numeric_candidate_cells": len(field.cells),
            "quality_pixel_counts": {
                str(int(code)): int(count)
                for code, count in zip(*np.unique(quality, return_counts=True))
            },
            "applied": False,
            "formal_ground_footprint": False,
        }
    return profiles


def run(
    direct_base_tie_diagnostic: Path,
    relative_diagnostics: list[Path],
    base_raster: Path,
    output: Path,
) -> dict[str, object]:
    direct_path = Path(direct_base_tie_diagnostic).resolve()
    relative_paths = [Path(path).resolve() for path in relative_diagnostics]
    base_path = Path(base_raster).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite spatial-composition output: {output}")
    if not direct_path.is_file() or not base_path.is_file():
        raise FileNotFoundError(direct_path if not direct_path.is_file() else base_path)
    if not relative_paths or any(not path.is_file() for path in relative_paths):
        raise FileNotFoundError("All actual-date relative diagnostics must exist")
    if len({str(path).casefold() for path in relative_paths}) != len(relative_paths):
        raise ValueError("Relative diagnostic inputs must be unique")

    input_case_ids = {
        json.loads(path.read_text(encoding="utf-8")).get("case_id")
        for path in [direct_path, *relative_paths]
    }
    if len(input_case_ids) != 1 or None in input_case_ids:
        raise ValueError("Direct and relative alignment diagnostics must share one fixed case ID")
    case_id = next(iter(input_case_ids))

    input_paths = [direct_path, *relative_paths]
    input_hashes_before = {str(path): _sha256(path) for path in input_paths}
    base_hash_before = _sha256(base_path)
    direct_fields = read_direct_base_tie_fields(
        direct_path,
        verified_reference_raster_hashes={str(base_path): base_hash_before},
    )
    relative_fields = []
    for path in relative_paths:
        relative_fields.extend(read_lightglue_local_field_candidates(
            path, expected_base_raster=base_path,
            expected_base_sha256=base_hash_before,
        ))

    fields, composition = compose_spatial_alignment_fields(direct_fields, relative_fields)
    composition["case_id"] = case_id
    _assert_nonoverlapping_fields(fields)
    composition["candidate_cells"] = [{
        "field_id": field.field_id,
        "actual_date": field.moving_date,
        "reference_label": field.reference_label,
        "reference_date": field.reference_date,
        "reference_date_status": field.reference_date_status,
        "cell_id": cell.cell_id,
        "status": cell.status,
        "candidate_correction_m": [
            cell.correction_dx_m_candidate, cell.correction_dy_m_candidate],
        "support_geometry_epsg32649": mapping(cell.support_geometry),
        "metadata": cell.metadata,
        "reviewed": cell.reviewed,
        "independent_ground_evidence": cell.independent_ground_evidence,
        "positional_uncertainty_m": cell.positional_uncertainty_m,
        "applied": cell.applied,
    } for field in fields for cell in field.cells]
    grid = _local_grid(base_path, fields) if fields else None

    output.mkdir(parents=True)
    gpkg_path = output / "spatial_alignment_candidates.gpkg"
    feature_count = _write_vectors(gpkg_path, fields)
    raster_profiles = _write_candidate_rasters(output, fields, grid) if grid else {}

    csv_path = output / "spatial_alignment_candidates.csv"
    columns = ["actual_date", "field_id", "cell_id", "status",
               "candidate_dx_m", "candidate_dy_m", "support_area_m2",
               "cycle_degrees_of_freedom", "closure_rmse_m", "closure_p90_m",
               "closure_max_m", "source_candidate_cell_refs", "reviewed",
               "independent_ground_evidence", "applied"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for field in fields:
            for cell in field.cells:
                row = {
                    "actual_date": field.moving_date,
                    "field_id": field.field_id,
                    "cell_id": cell.cell_id,
                    "status": cell.status,
                    "candidate_dx_m": cell.correction_dx_m_candidate,
                    "candidate_dy_m": cell.correction_dy_m_candidate,
                    "support_area_m2": cell.support_geometry.area,
                    "reviewed": cell.reviewed,
                    "independent_ground_evidence": cell.independent_ground_evidence,
                    "applied": cell.applied,
                    **cell.metadata,
                }
                writer.writerow({
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list)) else row.get(key)
                    for key in columns
                })

    source_hashes_after = {str(path): _sha256(path) for path in input_paths}
    base_hash_after = _sha256(base_path)
    if source_hashes_after != input_hashes_before or base_hash_after != base_hash_before:
        raise RuntimeError("An input diagnostic or the 10 m base raster changed during composition")

    composition["source_inputs"] = {
        "direct_base_tie_diagnostic": {
            "path": str(direct_path),
            "sha256_before": input_hashes_before[str(direct_path)],
            "sha256_after": source_hashes_after[str(direct_path)],
        },
        "relative_diagnostics": [{
            "path": str(path),
            "sha256_before": input_hashes_before[str(path)],
            "sha256_after": source_hashes_after[str(path)],
        } for path in relative_paths],
        "annual_reference_raster": {
            "path": str(base_path),
            "sha256_before": base_hash_before,
            "sha256_after": base_hash_after,
            "acquisition_date_status": "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN",
            "per_pixel_mosaic_component_provenance": "NOT_RECORDED",
        },
    }
    composition["products"] = {
        "geopackage": gpkg_path.name,
        "geopackage_feature_count": feature_count,
        "csv": csv_path.name,
        "candidate_rasters_by_actual_date": raster_profiles,
        "raster_grid": grid,
        "field_interpolated": False,
        "field_applied": False,
        "formal_ground_footprint_product": False,
    }
    (output / "SPATIAL_ALIGNMENT_COMPOSITION_DIAGNOSTIC.json").write_text(
        json.dumps(composition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return composition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-base-tie-diagnostic", type=Path, required=True)
    parser.add_argument("--relative-diagnostic", type=Path, action="append", required=True,
                        help="Actual-date LightGlue diagnostic; pass once per local mosaic window")
    parser.add_argument("--base-raster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.direct_base_tie_diagnostic, args.relative_diagnostic,
                 args.base_raster, args.output)
    print(json.dumps({
        "output": str(args.output),
        "composed_atoms": result["spatially_composed_atom_count"],
        "output_cells_by_date": result["output_cell_counts_by_actual_date"],
        "cycle_status_counts": result["cycle_status_counts"],
        "fields_applied": result["field_applied"],
        "ground_position_resolved": result["absolute_ground_position_resolved"],
    }, ensure_ascii=False))
