#!/usr/bin/env python3
"""Compose B04 actual-date-to-base ties with the six-year base pose graph.

The runner writes local displacement candidates on the existing 10 m grid.
It does not resample imagery, change geometry, infer annual-base capture dates,
or promote any candidate into a corrected product.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fiona
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment.composition import compose_actual_date_fields
from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField


RECONSTRUCTION_DIR = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction")
DEFAULT_BASE_PAIR = RECONSTRUCTION_DIR / "base_mosaic_pair_B04_02" / "BASE_MOSAIC_PAIR_DIAGNOSTIC.json"
DEFAULT_BASE_TIE = RECONSTRUCTION_DIR / "base_tie_field_subpx_B04_01" / "BASE_TIE_DIAGNOSTIC.json"
DEFAULT_OUTPUT = RECONSTRUCTION_DIR / "composed_registration_field_B04_01"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_objects(composition: dict) -> list[LocalDisplacementField]:
    fields: list[LocalDisplacementField] = []
    anchor_path = str(composition["anchor_path"])
    anchor_label = f"Xian_{composition['anchor_year_label']}_TrueColor"
    for actual_date in sorted({row["actual_date"] for row in composition["date_fields"]}):
        rows = [row for row in composition["date_fields"] if row["actual_date"] == actual_date]
        cells = []
        for row in rows:
            candidate = row["composed_correction_m_candidate"]
            bounds = row["support_bounds_epsg32649"]
            cells.append(DisplacementCell(
                cell_id=row["cell_id"],
                support_geometry=box(*bounds),
                correction_dx_m_candidate=float(candidate[0]) if candidate is not None else None,
                correction_dy_m_candidate=float(candidate[1]) if candidate is not None else None,
                fit_dispersion_m=None,
                status=row["status"],
                method="SUBMETER_TO_ANNUAL_BASE_PLUS_ANNUAL_BASE_POSE_GRAPH",
                source_ref=f"{row['source_epoch_label']}:{actual_date}:{row['cell_id']}",
                reviewed=False,
                independent_ground_evidence=False,
                positional_uncertainty_m=None,
                applied=False,
                metadata={key: value for key, value in row.items()
                          if key not in {"support_bounds_epsg32649"}},
            ))
        fields.append(LocalDisplacementField(
            field_id=f"{actual_date}_to_{anchor_label}",
            reference_date=None,
            moving_date=actual_date,
            cells=tuple(cells),
            reference_label=anchor_label,
            reference_source=anchor_path,
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
        ))
    return fields


def _write_field_vectors(path: Path, fields: list[LocalDisplacementField]) -> int:
    schema = {"geometry": "Polygon", "properties": {
        "field_id": "str:64", "cell_id": "str:24", "moving_date": "str:16",
        "reference_label": "str:80", "reference_date": "str:16",
        "reference_date_status": "str:48", "status": "str:64",
        "candidate_dx_m": "float", "candidate_dy_m": "float",
        "base_dx_m": "float", "base_dy_m": "float",
        "highres_dx_m": "float", "highres_dy_m": "float",
        "cycle_rmse_m": "float", "cycle_max_m": "float", "cycle_dof": "int",
        "peak_corr": "float", "peak_margin": "float", "eligible_px": "int",
        "reviewed": "int", "ground_evidence": "int", "applied": "int"}}
    count = 0
    with fiona.open(path, "w", driver="GPKG", layer="composed_alignment_candidates",
                    schema=schema, crs="EPSG:32649") as sink:
        for field in fields:
            for cell in field.cells:
                meta = cell.metadata
                base = meta.get("source_base_to_anchor_correction_m_candidate")
                highres = meta.get("highres_to_source_base_correction_m_candidate")
                sink.write({"geometry": mapping(cell.support_geometry), "properties": {
                    "field_id": field.field_id, "cell_id": cell.cell_id,
                    "moving_date": field.moving_date, "reference_label": field.reference_label,
                    "reference_date": field.reference_date,
                    "reference_date_status": field.reference_date_status,
                    "status": cell.status,
                    "candidate_dx_m": cell.correction_dx_m_candidate,
                    "candidate_dy_m": cell.correction_dy_m_candidate,
                    "base_dx_m": base[0] if base else None,
                    "base_dy_m": base[1] if base else None,
                    "highres_dx_m": highres[0] if highres else None,
                    "highres_dy_m": highres[1] if highres else None,
                    "cycle_rmse_m": meta.get("base_pose_closure_rmse_m"),
                    "cycle_max_m": meta.get("base_pose_closure_max_m"),
                    "cycle_dof": meta.get("base_pose_cycle_degrees_of_freedom"),
                    "peak_corr": meta.get("peak_correlation"),
                    "peak_margin": meta.get("peak_margin"),
                    "eligible_px": meta.get("eligible_pixels", 0),
                    "reviewed": int(cell.reviewed),
                    "ground_evidence": int(cell.independent_ground_evidence),
                    "applied": int(cell.applied)}})
                count += 1
    return count


def _write_field_rasters(output: Path, composition: dict) -> dict[str, dict]:
    transform = Affine(*composition["grid"]["transform"][:6])
    height, width = (int(v) for v in composition["grid"]["shape"])
    output_profiles = {}
    rows_by_date: dict[str, list[dict]] = defaultdict(list)
    for row in composition["date_fields"]:
        rows_by_date[row["actual_date"]].append(row)

    for actual_date, rows in sorted(rows_by_date.items()):
        dx = np.full((height, width), np.nan, dtype=np.float32)
        dy = np.full((height, width), np.nan, dtype=np.float32)
        quality = np.zeros((height, width), dtype=np.uint8)
        occupied = np.zeros((height, width), dtype=bool)
        ambiguous = np.zeros((height, width), dtype=bool)
        status_counts: Counter[str] = Counter()
        for row in rows:
            col0, row0, col1, row1 = row["support_pixel_window"]
            if not (0 <= col0 < col1 <= width and 0 <= row0 < row1 <= height):
                raise ValueError(f"Composed support cell leaves the local grid: {actual_date}/{row['cell_id']}")
            region = np.s_[row0:row1, col0:col1]
            vector = row["composed_correction_m_candidate"]
            status = row["status"]
            status_counts[status] += 1
            if vector is None:
                code = 4
            elif status == "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY":
                code = 1
            elif status == "COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED":
                code = 2
            else:
                code = 3
            overlap = occupied[region]
            quality_region = quality[region]
            dx_region = dx[region]
            dy_region = dy[region]
            if np.any(overlap):
                ambiguous[region] |= overlap
                dx_region[overlap] = np.nan
                dy_region[overlap] = np.nan
                quality_region[overlap] = 255
            fresh = ~overlap
            quality_region[fresh] = code
            if vector is not None:
                dx_region[fresh] = float(vector[0])
                dy_region[fresh] = float(vector[1])
            occupied[region] = True
        quality[ambiguous] = 255
        offset_path = output / f"{actual_date}_to_2019_base_candidate_offset.tif"
        quality_path = output / f"{actual_date}_to_2019_base_candidate_quality.tif"
        common = {"driver": "GTiff", "width": width, "height": height, "crs": "EPSG:32649",
                  "transform": transform, "tiled": True, "blockxsize": 256,
                  "blockysize": 256, "compress": "deflate"}
        with rasterio.open(offset_path, "w", count=2, dtype="float32", nodata=np.nan,
                           **common) as sink:
            sink.write(dx, 1)
            sink.write(dy, 2)
            sink.set_band_description(1, "candidate_correction_east_m")
            sink.set_band_description(2, "candidate_correction_north_m")
            sink.update_tags(
                direction=composition["direction"],
                reference_actual_date_status=composition["anchor_acquisition_date_status"],
                interpolated="false", applied="false", positional_accuracy="not_established",
            )
        with rasterio.open(quality_path, "w", count=1, dtype="uint8", nodata=None,
                           **common) as sink:
            sink.write(quality, 1)
            sink.set_band_description(1, "candidate_field_internal_status")
            sink.update_tags(
                codes=json.dumps({"0": "outside_field", "1": "cycle_screen_only_candidate",
                                  "2": "cycle_screen_failed_candidate", "3": "no_cycle_check_candidate",
                                  "4": "covered_without_composed_vector", "255": "overlapping_support_cells"}),
                applied="false", interpolated="false",
            )
        with rasterio.open(offset_path) as raster:
            if raster.crs.to_epsg() != 32649 or raster.res != (10.0, 10.0):
                raise RuntimeError(f"Unexpected composed displacement grid: {offset_path}")
            if raster.transform != transform or raster.width != width or raster.height != height:
                raise RuntimeError(f"Composed displacement grid changed during writing: {offset_path}")
        output_profiles[actual_date] = {
            "offset_tif": offset_path.name,
            "quality_tif": quality_path.name,
            "cell_status_counts": dict(status_counts),
            "numeric_candidate_cells": sum(row["composed_correction_m_candidate"] is not None
                                             for row in rows),
            "quality_counts": {str(int(code)): int(count)
                               for code, count in zip(*np.unique(quality, return_counts=True))},
        }
    return output_profiles


def run(base_pair_path: Path, base_tie_path: Path, output: Path,
        *, anchor_year: int = 2019) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite composed-field output: {output}")
    for path in (base_pair_path, base_tie_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path): _sha256(path) for path in (base_pair_path, base_tie_path)}
    base_pair = json.loads(base_pair_path.read_text(encoding="utf-8"))
    base_tie = json.loads(base_tie_path.read_text(encoding="utf-8"))
    composition = compose_actual_date_fields(base_pair, base_tie, anchor_year=anchor_year)
    fields = _field_objects(composition)
    output.mkdir(parents=True)
    gpkg_path = output / "composed_alignment_candidates.gpkg"
    feature_count = _write_field_vectors(gpkg_path, fields)
    raster_manifest = _write_field_rasters(output, composition)
    csv_fields = [
        "actual_date", "source_epoch_label", "source_base_year_label", "reference_year_label",
        "cell_id", "status", "highres_observed_shift_px",
        "highres_to_source_base_correction_m_candidate",
        "source_base_to_anchor_correction_m_candidate", "composed_correction_m_candidate",
        "peak_correlation", "peak_margin", "eligible_pixels", "base_pose_edge_count",
        "base_pose_cycle_degrees_of_freedom", "base_pose_closure_rmse_m",
        "base_pose_closure_max_m", "reviewed", "independent_ground_evidence", "applied",
    ]
    with (output / "composed_alignment_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in composition["date_fields"]:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False)
                             if isinstance(row.get(key), (list, dict)) else row.get(key)
                             for key in csv_fields})
    source_hashes_after = {str(path): _sha256(path) for path in (base_pair_path, base_tie_path)}
    if hashes_before != source_hashes_after:
        raise RuntimeError("Diagnostic source JSON changed during composition")
    composition["source_diagnostics"] = {
        "base_pair_path": str(base_pair_path),
        "base_pair_sha256_before": hashes_before[str(base_pair_path)],
        "base_pair_sha256_after": source_hashes_after[str(base_pair_path)],
        "base_tie_path": str(base_tie_path),
        "base_tie_sha256_before": hashes_before[str(base_tie_path)],
        "base_tie_sha256_after": source_hashes_after[str(base_tie_path)],
    }
    composition["products"] = {
        "geopackage": gpkg_path.name,
        "geopackage_feature_count": feature_count,
        "rasters_by_actual_date": raster_manifest,
        "position_accuracy_claimed": False,
        "accepted_cells": 0,
    }
    (output / "COMPOSED_ALIGNMENT_DIAGNOSTIC.json").write_text(
        json.dumps(composition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return composition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pair-diagnostic", type=Path, default=DEFAULT_BASE_PAIR)
    parser.add_argument("--base-tie-diagnostic", type=Path, default=DEFAULT_BASE_TIE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-year", type=int, default=2019)
    args = parser.parse_args()
    result = run(args.base_pair_diagnostic, args.base_tie_diagnostic,
                 args.output, anchor_year=args.anchor_year)
    print(json.dumps({"output": str(args.output),
                      "actual_dates": sorted({row["actual_date"] for row in result["date_fields"]}),
                      "candidate_cells": sum(row["composed_correction_m_candidate"] is not None
                                             for row in result["date_fields"]),
                      "fields_applied": result["field_applied"]}, ensure_ascii=False))
