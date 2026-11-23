from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import fiona
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.composition import compose_actual_date_fields
from tools.run_xian_composed_alignment_candidates import (_field_objects,
                                                          _write_field_rasters,
                                                          _write_field_vectors)


TRANSFORM = [10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0, 0.0, 0.0, 1.0]
WINDOW = [0, 0, 10, 10]


def _pair(reference: int, moving: int, correction: tuple[float, float], *,
          coarse=(0, 0)) -> dict:
    return {
        "reference_year_label": reference,
        "moving_year_label": moving,
        "cells": [{
            "row": 0, "col": 0, "base_pixel_window": WINDOW,
            "best": {"dx_px": coarse[0], "dy_px": coarse[1], "corr": 0.8, "n": 100},
            "correction_to_reference_m_candidate": list(correction),
        }],
    }


def _epoch(actual_date: str, base_year: int, shift: tuple[float, float] | None) -> dict:
    best = None
    if shift is not None:
        best = {"dx_px": 1, "dy_px": -1, "corr": 0.75, "n": 80,
                "subpixel_peak": {"dx_px": shift[0], "dy_px": shift[1]}}
    return {
        "actual_date": actual_date,
        "base_path": f"F:/xian/xian底图/Xian_{base_year}_TrueColor.tif",
        "base_sha256": f"sha-{base_year}",
        "base_grid": {"crs": "EPSG:32649", "transform": TRANSFORM.copy(),
                      "window": [100, 200, 10, 10]},
        "local_spatial_tie_diagnostic": {
            "status": "DIAGNOSTIC_ONLY_NOT_ACCEPTED",
            "search_radius_px": 5,
            "field_interpolated": False, "field_applied": False,
            "cells": [{"row": 0, "col": 0, "base_pixel_window": WINDOW,
                       "valid_pixel_count": 100, "best": best,
                       "peak_margin": 0.05 if best else None}],
        },
    }


def _reports() -> tuple[dict, dict]:
    hashes = {str(year): f"sha-{year}" for year in (2018, 2019, 2020)}
    base_report = {
        "base_year_labels": [2018, 2019, 2020],
        "base_year_date_semantics": "ANNUAL_FILE_LABELS_ONLY;PIXEL_ACQUISITION_DATES_UNKNOWN",
        "search_radius_px": 5,
        "field_reviewed": False, "field_applied": False,
        "registration_promoted": False,
        "base_sha256_before": hashes, "base_sha256_after": hashes.copy(),
        "base_rasters": {str(year): {
            "path": f"F:/xian/xian底图/Xian_{year}_TrueColor.tif",
            "sha256": f"sha-{year}"} for year in (2018, 2019, 2020)},
        "base_grid": {
            "reference_path": "F:/xian/xian底图/Xian_2018_TrueColor.tif",
            "crs": "EPSG:32649", "resolution": [10.0, 10.0],
            "local_transform": TRANSFORM.copy(), "shape": [10, 10],
            "window": [100, 200, 10, 10], "bounds": [1000, 1900, 1100, 2000],
        },
        "pairwise_fields": [
            _pair(2018, 2019, (-2, 3)),
            _pair(2019, 2020, (5, 4)),
            _pair(2018, 2020, (3, 7), coarse=(5, 0)),
        ],
    }
    tie_report = {
        "automatic_registration_promoted": False,
        "epochs": {
            "2018": _epoch("2018-04-29", 2018, (1.25, -0.5)),
            "2019": _epoch("2018-11-01", 2018, (0.0, 0.0)),
            "2020": _epoch("2019-12-02", 2019, (0.2, 0.3)),
        },
    }
    return base_report, tie_report


class CompositionTests(unittest.TestCase):
    def test_composes_real_date_tie_with_anchor_relative_base_pose(self):
        base_report, tie_report = _reports()

        result = compose_actual_date_fields(base_report, tie_report)

        self.assertEqual([row["actual_date"] for row in result["date_fields"]], [
            "2018-04-29", "2018-11-01", "2019-12-02"])
        first = next(row for row in result["date_fields"] if row["actual_date"] == "2018-04-29")
        self.assertEqual(first["highres_to_source_base_correction_m_candidate"], [-12.5, -5.0])
        self.assertEqual(first["source_base_to_anchor_correction_m_candidate"], [2.0, -3.0])
        self.assertEqual(first["composed_correction_m_candidate"], [-10.5, -8.0])
        self.assertEqual(first["status"], "COMPOSED_CANDIDATE_TREE_HAS_NO_CYCLE_CHECK")
        self.assertFalse(first["reviewed"])
        self.assertFalse(first["applied"])
        self.assertIsNone(first["positional_uncertainty_m"])
        self.assertTrue(result["base_year_acquisition_dates_used"] is False)

    def test_search_boundary_tie_is_excluded_and_reference_date_stays_unknown(self):
        base_report, tie_report = _reports()

        result = compose_actual_date_fields(base_report, tie_report)

        pose = result["pose_graph_cells"][0]
        self.assertEqual(pose["search_boundary_edges_excluded"], 1)
        anchor_date = next(row for row in result["date_fields"]
                           if row["actual_date"] == "2019-12-02")
        self.assertIsNone(anchor_date["reference_date"])
        self.assertEqual(anchor_date["reference_date_status"],
                         "REFERENCE_ACQUISITION_DATE_UNKNOWN")
        self.assertEqual(anchor_date["composed_correction_m_candidate"], [-2.0, 3.0])

    def test_missing_subpixel_tie_never_falls_back_to_coarse_correction(self):
        base_report, tie_report = _reports()
        tie_report["epochs"]["2018"]["local_spatial_tie_diagnostic"]["cells"][0]["best"]["subpixel_peak"] = None

        result = compose_actual_date_fields(base_report, tie_report)

        row = next(item for item in result["date_fields"] if item["actual_date"] == "2018-04-29")
        self.assertIsNone(row["highres_to_source_base_correction_m_candidate"])
        self.assertIsNone(row["composed_correction_m_candidate"])
        self.assertEqual(row["status"], "HIGHRES_BASE_TIE_NO_USABLE_SUBPIXEL_CANDIDATE")

    def test_rejects_wrong_grids_or_promoted_inputs(self):
        base_report, tie_report = _reports()
        tie_report["epochs"]["2018"]["base_grid"]["transform"][2] += 10
        with self.assertRaises(ValueError):
            compose_actual_date_fields(base_report, tie_report)

    def test_field_adapters_write_candidate_only_geopackage_and_common_grid_tifs(self):
        base_report, tie_report = _reports()
        composed = compose_actual_date_fields(base_report, tie_report)
        fields = _field_objects(composed)

        self.assertEqual(len(fields), 3)
        self.assertTrue(all(field.reference_date is None for field in fields))
        self.assertTrue(all(field.reference_date_status == "REFERENCE_ACQUISITION_DATE_UNKNOWN"
                            for field in fields))
        self.assertTrue(all(not cell.accepted and not cell.applied
                            for field in fields for cell in field.cells))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gpkg = root / "fields.gpkg"
            self.assertEqual(_write_field_vectors(gpkg, fields), 3)
            manifest = _write_field_rasters(root, composed)
            self.assertEqual(set(manifest), {"2018-04-29", "2018-11-01", "2019-12-02"})
            with fiona.open(gpkg, layer="composed_alignment_candidates") as layer:
                features = list(layer)
            self.assertEqual(len(features), 3)
            self.assertTrue(all(item["properties"]["applied"] == 0 for item in features))
            self.assertTrue(all(item["properties"]["reference_date_status"] ==
                                "REFERENCE_ACQUISITION_DATE_UNKNOWN" for item in features))
            with rasterio.open(root / "2018-04-29_to_2019_base_candidate_offset.tif") as raster:
                self.assertEqual(raster.crs.to_epsg(), 32649)
                self.assertEqual(raster.res, (10.0, 10.0))
                self.assertEqual((raster.width, raster.height), (10, 10))
                np.testing.assert_allclose(raster.read()[:, 0, 0], [-10.5, -8.0])
                self.assertEqual(raster.tags()["applied"], "false")
            with rasterio.open(root / "2018-04-29_to_2019_base_candidate_quality.tif") as raster:
                self.assertIn(3, np.unique(raster.read(1)))

        base_report, tie_report = _reports()
        base_report["field_applied"] = True
        with self.assertRaises(ValueError):
            compose_actual_date_fields(base_report, tie_report)


if __name__ == "__main__":
    unittest.main()
