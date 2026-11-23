from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment import (export_component_tie_consensus_candidates,
                               read_component_tie_consensus_fields)


ACTUAL_DATE = "2019-12-02"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ComponentTieConsensusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.grids = {}
        for year in (2018, 2019):
            path = self.root / f"reference_{year}.tif"
            with rasterio.open(
                path, "w", driver="GTiff", width=7, height=2, count=3,
                dtype="uint8", crs="EPSG:32649",
                transform=from_origin(0, 20, 10, 10),
            ) as target:
                target.write(np.full((3, 2, 7), year % 256, dtype="uint8"))
            self.grids[year] = path
        self.tiles = {}
        self.tile_ids = {
            "left": "420790:312184", "disagree": "420791:312184",
            "low": "420792:312184", "few": "420793:312184",
            "none": "420794:312184", "outside": "420795:312184",
        }
        for label, tile_id in self.tile_ids.items():
            tile = self.root / f"{label}.jpg"
            tile.write_bytes(f"tile:{label}".encode())
            self.tiles[tile_id] = {"path": str(tile), "sha256": _sha256(tile)}
        self.case_geometry = self.root / "fixed_case.gpkg"
        with fiona.open(
            self.case_geometry, "w", driver="GPKG", layer="fixed_case_polygons",
            schema={"geometry": "Polygon", "properties": {
                "case_id": "str:32", "role": "str:16"}}, crs="EPSG:32649",
        ) as sink:
            sink.write({"geometry": mapping(box(0, 0, 50, 20)),
                        "properties": {"case_id": "B04-TQA10", "role": "TARGET"}})
        self.diagnostics = [self._write_report(year) for year in (2018, 2019)]

    def tearDown(self):
        self.temp.cleanup()

    def _write_report(self, year: int) -> Path:
        components = []
        definitions = [
            ("left", (0, 0, 10, 20), 1, 0, 0.88, 0.2, 20, "NUMERIC_PEAK_AVAILABLE"),
            ("disagree", (10, 0, 20, 20), 0 if year == 2018 else 1, 0,
             0.85, 0.2, 20, "NUMERIC_PEAK_AVAILABLE"),
            ("low", (20, 0, 30, 20), 1, 0, 0.72, 0.05, 20,
             "NUMERIC_PEAK_AVAILABLE"),
            ("few", (30, 0, 40, 20), None, None, None, None, None,
             "INSUFFICIENT_MASKED_SAMPLES"),
            ("none", (40, 0, 50, 20), None, None, None, None, None,
             "NO_VALID_CORRELATION_SCORE"),
            ("outside", (50, 0, 60, 20), 1, 0, 0.9, 0.3, 20,
             "NUMERIC_PEAK_AVAILABLE"),
        ]
        for label, bounds, dx, dy, corr, margin, n, sample_status in definitions:
            xmin, ymin, xmax, ymax = bounds
            tile_id = self.tile_ids[label]
            cell = {
                "tile_id": tile_id,
                "actual_date": ACTUAL_DATE,
                "source_tile_path": self.tiles[tile_id]["path"],
                "source_tile_sha256": self.tiles[tile_id]["sha256"],
                "support_polygon_epsg32649": {
                    "type": "Polygon",
                    "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax],
                                     [xmin, ymax], [xmin, ymin]]],
                },
                "sample_support_status": sample_status,
                "minimum_samples_required": 4,
                "maximum_masked_samples_available": (
                    n if n is not None else
                    4 if sample_status == "NO_VALID_CORRELATION_SCORE" else 2),
                "peak_margin": margin,
                "best": ({"dx_px": dx, "dy_px": dy, "corr": corr, "n": n}
                         if dx is not None else None),
            }
            components.append(cell)

        report = {
            "schema_version": "xian_r5.base_tie_diagnostic.v1",
            "case_id": "B04-TQA10",
            "scope": "FIXED_CASE_LOCAL_ONLY",
            "automatic_registration_promoted": False,
            "inputs_sha256": {"fixed_case_geometry": _sha256(self.case_geometry)},
            "parameters": {
                "base_reference_year": year,
                "local_support_mode": "EXACT_SOURCE_TILE_COMPONENTS",
                "local_field_interpolated_or_applied": False,
            },
            "epochs": {
                "2020": {
                    "actual_date": ACTUAL_DATE,
                    "base_path": str(self.grids[year]),
                    "base_sha256": _sha256(self.grids[year]),
                    "base_grid": {
                        "crs": "EPSG:32649", "resolution": [10.0, 10.0],
                        "window": [0, 0, 7, 2],
                        "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 20.0,
                                      0.0, 0.0, 1.0],
                    },
                    "local_spatial_tie_diagnostic": {
                        "support_mode": "EXACT_SOURCE_TILE_COMPONENTS",
                        "field_interpolated": False,
                        "field_applied": False,
                        "cells": components,
                    },
                },
            },
        }
        path = self.root / f"diagnostic_{year}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_exports_only_high_confidence_same_shift_candidates(self):
        output = self.root / "products"
        output.mkdir()
        manifest = export_component_tie_consensus_candidates(
            output, self.diagnostics,
            fixed_case_geometry_path=self.case_geometry)

        self.assertEqual(manifest["reference_years"], [2018, 2019])
        self.assertEqual(manifest["actual_dates"], [ACTUAL_DATE])
        self.assertEqual(manifest["source_component_images_revalidated"], 6)
        self.assertFalse(manifest["absolute_ground_position_resolved"])
        self.assertFalse(manifest["ground_footprints_written"])
        self.assertFalse(manifest["formal_change_products_written"])

        date_products = manifest["products_by_actual_date"][ACTUAL_DATE]
        self.assertEqual(date_products["high_confidence_same_shift_candidates"], 1)
        self.assertEqual(date_products["high_confidence_candidates_outside_case_roi"], 1)
        self.assertEqual(date_products["case_roi_component_count"], 5)
        field_path = Path(date_products["field_path"])
        quality_path = Path(date_products["quality_path"])
        with rasterio.open(field_path) as field, rasterio.open(quality_path) as quality:
            self.assertEqual(field.crs.to_epsg(), 32649)
            self.assertEqual(field.transform, from_origin(0, 20, 10, 10))
            self.assertEqual(quality.transform, field.transform)
            self.assertEqual(field.nodata, -9999.0)
            self.assertEqual(quality.nodata, 255)
            q = quality.read(1)
            self.assertEqual(int(q[0, 0]), 1)
            self.assertEqual(int(q[0, 1]), 3)
            self.assertEqual(int(q[0, 2]), 2)
            self.assertEqual(int(q[0, 3]), 5)
            self.assertEqual(int(q[0, 4]), 6)
            self.assertEqual(int(q[0, 5]), 255)
            self.assertEqual(int(q[0, 6]), 255)
            self.assertAlmostEqual(float(field.read(1)[0, 0]), -10.0)
            self.assertAlmostEqual(float(field.read(2)[0, 0]), 0.0)
            self.assertEqual(float(field.read(1)[0, 1]), -9999.0)
            self.assertEqual(float(field.read(1)[0, 2]), -9999.0)

        with fiona.open(manifest["candidate_vector_path"],
                        layer="source_component_tie_consensus") as source:
            rows = {feature["properties"]["tile_id"]: feature for feature in source}
        self.assertEqual(len(rows), 5)
        self.assertNotIn(self.tile_ids["outside"], rows)
        self.assertEqual(rows[self.tile_ids["left"]]["properties"]["decision"],
                         "MULTI_REFERENCE_HIGH_CONFIDENCE_SAME_SHIFT_CANDIDATE")
        self.assertEqual(rows[self.tile_ids["left"]]["properties"]["candidate_dx_m"], -10.0)
        self.assertEqual(rows[self.tile_ids["disagree"]]["properties"]["decision"],
                         "MULTI_REFERENCE_SHIFT_DISAGREEMENT")
        fields, field_receipt = read_component_tie_consensus_fields(
            manifest["manifest_path"],
            canonical_reference_raster=self.grids[2019])
        self.assertEqual(len(fields), 1)
        self.assertEqual(len(fields[0].cells), 5)
        self.assertEqual(field_receipt["numeric_candidate_cell_count"], 1)
        self.assertEqual(field_receipt["accepted_cell_count"], 0)
        self.assertFalse(field_receipt["field_applied"])
        self.assertTrue(all(cell.source_component_scoped for cell in fields[0].cells))
        self.assertTrue(all(not cell.accepted for cell in fields[0].cells))
        insufficient = next(cell for cell in fields[0].cells
                            if cell.cell_id == self.tile_ids["few"])
        self.assertEqual(insufficient.metadata["maximum_masked_samples_available"], 2)
        self.assertEqual(insufficient.metadata["minimum_samples_required"], 4)

    def test_rejects_non_component_or_reused_output(self):
        bad_report = json.loads(self.diagnostics[0].read_text(encoding="utf-8"))
        bad_report["parameters"]["local_support_mode"] = "UNIFORM_GRID"
        bad_path = self.root / "bad.json"
        bad_path.write_text(json.dumps(bad_report), encoding="utf-8")
        output = self.root / "products"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "exact source-tile"):
            export_component_tie_consensus_candidates(
                output, [bad_path, self.diagnostics[1]],
                fixed_case_geometry_path=self.case_geometry)

        existing = output / f"B04-TQA10_{ACTUAL_DATE}_component_tie_quality.tif"
        existing.touch()
        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            export_component_tie_consensus_candidates(
                output, self.diagnostics,
                fixed_case_geometry_path=self.case_geometry)


if __name__ == "__main__":
    unittest.main()
