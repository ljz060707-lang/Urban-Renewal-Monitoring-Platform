from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.field_io import read_multi_anchor_base_tie_fields


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_base_raster(path: Path) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=10, count=3,
        dtype="uint8", crs="EPSG:32649",
        transform=Affine(10, 0, 100, 0, -10, 200),
    ) as sink:
        for band in range(1, 4):
            sink.write(np.full((10, 10), band * 20, dtype=np.uint8), band)


def _write_report(root: Path, year: int, dx_m: float, dy_m: float) -> Path:
    raster = root / f"Xian_{year}_TrueColor.tif"
    _write_base_raster(raster)
    peak = {
        "dx_px": -dx_m / 10.0,
        "dy_px": dy_m / 10.0,
        "status": "SUBPIXEL_CORRELATION_PEAK_CANDIDATE_NOT_APPLIED",
    }
    report = {
        "schema_version": "xian_r5.base_tie_diagnostic.v1",
        "case_id": "B04-TQA10",
        "scope": "FIXED_CASE_LOCAL_ONLY",
        "automatic_registration_promoted": False,
        "parameters": {
            "base_reference_year": year,
            "radius_tiles": 5,
            "roof_exclusion_buffer_m": 5.0,
            "local_grid_size": 1,
        },
        "inputs_sha256": {"case_map": "map-hash", "case_geometry": "geometry-hash"},
        "epochs": {"2018": {
            "actual_date": "2018-04-29",
            "base_path": str(raster),
            "base_sha256": _sha256(raster),
            "base_grid": {
                "crs": "EPSG:32649", "resolution": [10.0, 10.0],
                "window": [0, 0, 10, 10],
                "transform": [10, 0, 100, 0, -10, 200],
            },
            "submeter_tiles": {
                "exact_date_read": 1, "exact_date_coverage_fraction": 1.0,
                "tile_ids": [{"tile_id": "420000:312000", "sha256": "tile-hash"}],
            },
            "mosaic_bounds_epsg32649": [100, 100, 200, 200],
            "r4_roof_exclusion": {
                "mask_geometry_sources": {"2018": {"sha256": "frozen-roof-hash"}},
            },
            "local_spatial_tie_diagnostic": {
                "grid_rows": 1, "grid_cols": 1, "search_radius_10m_pixels": 3,
                "field_interpolated": False, "field_applied": False,
                "cells": [{
                    "row": 0, "col": 0, "base_pixel_window": [0, 0, 10, 10],
                    "maximum_masked_samples_available": 140,
                    "minimum_samples_required": 100,
                    "sample_support_status": "NUMERIC_PEAK_AVAILABLE",
                    "best": {
                        "corr": 0.8, "dx_px": 0, "dy_px": 0, "n": 120,
                        "subpixel_peak": peak,
                    },
                    "runner_up": {"corr": 0.7, "dx_px": 1, "dy_px": 0, "n": 120},
                    "peak_margin": 0.1,
                    "status": "COARSE_TIE_CANDIDATE_NOT_APPLIED",
                }],
            },
        }},
    }
    path = root / f"base_tie_{year}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class MultiAnchorBaseTieTests(unittest.TestCase):
    def test_aggregates_co_gridded_annual_candidates_without_accepting_them(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = [
                _write_report(root, 2017, 2.0, 3.0),
                _write_report(root, 2018, 4.0, 3.0),
                _write_report(root, 2019, 3.0, 4.8),
            ]

            fields, receipt = read_multi_anchor_base_tie_fields(
                reports, canonical_reference_raster=root / "Xian_2019_TrueColor.tif")

            self.assertEqual(len(fields), 1)
            cell = fields[0].cells[0]
            self.assertEqual(cell.status, "MULTI_ANCHOR_BASE_TIE_CANDIDATE_NOT_REVIEWED")
            self.assertAlmostEqual(cell.correction_dx_m_candidate, 3.0)
            self.assertAlmostEqual(cell.correction_dy_m_candidate, 3.0)
            self.assertAlmostEqual(cell.fit_dispersion_m, 1.0)
            self.assertIsNone(cell.positional_uncertainty_m)
            self.assertFalse(cell.accepted)
            self.assertFalse(cell.applied)
            self.assertFalse(cell.independent_ground_evidence)
            self.assertEqual(cell.metadata["anchor_summary"]["reference_years"],
                             [2017, 2018, 2019])
            window_evidence = cell.metadata["reference_window_evidence"]
            self.assertEqual([row["reference_year_label"] for row in window_evidence],
                             [2017, 2018, 2019])
            self.assertEqual(
                [row["cell_window_full_reference_raster"] for row in window_evidence],
                [[0, 0, 10, 10]] * 3)
            self.assertTrue(all(row["reference_raster_sha256"]
                                == row["reference_raster_grid"]["sha256"]
                                for row in window_evidence))
            self.assertTrue(all(row["reference_component_id_status"]
                                == "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"
                                for row in window_evidence))
            self.assertEqual(len(receipt["field_cell_summaries"][0]
                                 ["reference_window_evidence"]), 3)
            self.assertFalse(receipt["dispersion_is_positional_uncertainty"])
            self.assertEqual(receipt["accepted_cells"], 0)
            self.assertFalse(receipt["field_applied"])
            self.assertEqual(len(receipt["integrity_files_sha256"]), 6)

    def test_requires_multiple_diagnostics_and_verifies_reference_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = _write_report(root, 2019, 1.0, 1.0)
            with self.assertRaisesRegex(ValueError, "at least two"):
                read_multi_anchor_base_tie_fields(
                    [report], canonical_reference_raster=root / "Xian_2019_TrueColor.tif")

            second = _write_report(root, 2018, 1.0, 1.0)
            reference = root / "Xian_2018_TrueColor.tif"
            reference.write_bytes(reference.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_multi_anchor_base_tie_fields(
                    [report, second], canonical_reference_raster=root / "Xian_2019_TrueColor.tif")


if __name__ == "__main__":
    unittest.main()
