"""Contract tests for grid-aligned, explicitly relative roof candidates."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment import write_relative_roof_candidate_products


ANCHOR = "2018-04-29"
MOVING = "2019-12-02"


def pose_graph() -> dict:
    def cell(row, column, bounds, status, offsets, sample_count=5):
        edges = ([{"reference_id": ANCHOR, "moving_id": MOVING,
                   "sample_count": sample_count}]
                 if status != "INSUFFICIENT_RELATIVE_SUPPORT" else [])
        return {
            "row": row, "column": column,
            "bounds_epsg32649": list(bounds), "status": status,
            "candidate_corrections_to_anchor_m": offsets,
            "pairwise_candidate_edges": edges,
            "closure_max_m": 0.25,
        }

    return {
        "schema_version": "xian_r5.local_relative_pose_graph_candidates.v1",
        "anchor_id": ANCHOR,
        "grid_shape_rows_columns": [2, 2],
        "minimum_support_per_pairwise_edge": 5,
        "source_semantics": "RELATIVE_TO_UNVERIFIED_STATIC_REFERENCE_ONLY",
        "absolute_ground_position_resolved": False,
        "applied": False,
        "interpolated": False,
        "cells": [
            cell(0, 0, (0, 20, 20, 40),
                 "RELATIVE_POSE_CYCLE_DIAGNOSTIC_CANDIDATE",
                 {ANCHOR: [0.0, 0.0], MOVING: [-2.0, 3.0]}),
            cell(0, 1, (20, 20, 40, 40),
                 "INSUFFICIENT_RELATIVE_SUPPORT", {}),
            cell(1, 0, (0, 0, 20, 20),
                 "RELATIVE_POSE_TREE_NO_CYCLE_CHECK",
                 {ANCHOR: [0.0, 0.0], MOVING: [-1.0, 1.0]}, 4),
            cell(1, 1, (20, 0, 40, 20),
                 "INSUFFICIENT_RELATIVE_SUPPORT", {}),
        ],
    }


class RelativeRoofCandidateProductTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "reference.tif"
        with rasterio.open(
            self.base, "w", driver="GTiff", width=4, height=4, count=3,
            dtype="uint8", crs="EPSG:32649", transform=from_origin(0, 40, 10, 10),
        ) as target:
            target.write(np.zeros((3, 4, 4), dtype=np.uint8))
        self.roofs = {
            "e0": [{"object_id": "anchor-roof", "geometry": box(4, 30, 8, 34)}],
            "e1": [
                {"object_id": "supported-roof", "geometry": box(4, 30, 8, 34)},
                {"object_id": "cross-cell-roof", "geometry": box(18, 28, 21, 32)},
                {"object_id": "unsupported-roof", "geometry": box(25, 28, 30, 32)},
            ],
        }
        self.common = {
            "reference_raster": self.base,
            "reference_raster_stack": [self.base],
            "roi_bounds": (0, 0, 40, 40),
            "local_roofs_by_epoch": self.roofs,
            "geometry_source_paths": {"e0": "e0.gpkg", "e1": "e1.gpkg"},
            "geometry_source_sha256": {"e0": "a" * 64, "e1": "b" * 64},
            "actual_dates": {"e0": ANCHOR, "e1": MOVING},
            "pose_graph": pose_graph(),
            "outline_source_path": "2019_roof_outline.shp",
            "outline_source_component_sha256": {"2019_roof_outline.shp": "c" * 64},
            "outline_source_epoch_label": "2019",
            "outline_source_geometry_semantics": "ROOF_BOUNDARY_FROM_IMAGE",
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_exports_relative_fields_masks_and_lineage_on_reference_grid(self):
        output = self.root / "products"
        output.mkdir()
        manifest = write_relative_roof_candidate_products(output, **self.common)

        self.assertFalse(manifest["absolute_ground_position_resolved"])
        self.assertFalse(manifest["ground_footprint_products_written"])
        self.assertFalse(manifest["formal_change_products_written"])
        self.assertEqual(
            manifest["support_sample_count_semantics"],
            "SUM_OF_PAIRWISE_ROOF_TIES_INCIDENT_ON_ACTUAL_DATE_NOT_ACCURACY")
        self.assertEqual(manifest["outline_source"]["source_epoch_label"], "2019")
        self.assertEqual(manifest["outline_source"]["source_geometry_semantics_user_reported"],
                         "IMAGE_DRAWN_ROOF_BOUNDARY")
        self.assertEqual(manifest["outline_source"]["component_sha256_revalidated"],
                         {"2019_roof_outline.shp": "c" * 64})
        self.assertIsNone(manifest["outline_source"]["capture_date"])
        self.assertIsNone(manifest["outline_source"]["accuracy_m"])
        self.assertTrue(Path(manifest["manifest_path"]).is_file())
        disk_manifest = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(disk_manifest["schema_version"],
                         "xian_r5.relative_roof_candidate_products.v1")

        field_path = Path(next(item["path"] for item in manifest["field_products"]
                               if item["actual_date"] == MOVING))
        quality_path = Path(next(item["path"] for item in manifest["quality_products"]
                                 if item["actual_date"] == MOVING))
        mask_path = Path(next(item["path"] for item in manifest["roof_candidate_mask_products"]
                              if item["actual_date"] == MOVING))
        with rasterio.open(field_path) as field, rasterio.open(quality_path) as quality, \
                rasterio.open(mask_path) as mask:
            for dataset in (field, quality, mask):
                self.assertEqual(dataset.crs.to_epsg(), 32649)
                self.assertEqual(dataset.transform, from_origin(0, 40, 10, 10))
                self.assertEqual((dataset.width, dataset.height), (4, 4))
            self.assertEqual(field.nodata, -9999.0)
            self.assertAlmostEqual(float(field.read(1)[0, 0]), -2.0)
            self.assertAlmostEqual(float(field.read(2)[0, 0]), 3.0)
            self.assertEqual(float(field.read(1)[0, 2]), -9999.0)
            self.assertTrue(set(np.unique(quality.read(1))).issubset({1, 2, 3, 255}))
            self.assertEqual(mask.nodata, 255)
            self.assertTrue(set(np.unique(mask.read(1))).issubset({0, 1, 255}))
            self.assertEqual(int(mask.read(1)[0, 0]), 1)
            self.assertEqual(int(mask.read(1)[0, 2]), 255)

        gpkg = Path(manifest["candidate_vector_path"])
        with fiona.open(gpkg, layer="candidate_decisions") as decisions:
            by_id = {item["properties"]["candidate_uid"].split(":", 1)[1]: item
                     for item in decisions}
        self.assertEqual(by_id["cross-cell-roof"]["properties"]["decision"],
                         "ROOF_CROSSES_RELATIVE_CELL_BOUNDARY")
        self.assertEqual(by_id["unsupported-roof"]["properties"]["decision"],
                         "INSUFFICIENT_RELATIVE_SUPPORT")
        accepted = []
        with fiona.open(gpkg, layer="relative_registered_roof_candidates") as candidates:
            for feature in candidates:
                properties = feature["properties"]
                accepted.append(properties["candidate_uid"])
                self.assertEqual(properties["source_outline_epoch"], "2019")
                self.assertEqual(properties["source_outline_semantics"],
                                 "ROOF_BOUNDARY_FROM_IMAGE")
                self.assertEqual(properties["source_outline_semantics_user_reported"],
                                 "IMAGE_DRAWN_ROOF_BOUNDARY")
                self.assertEqual(properties["source_outline_epoch_provenance"],
                                 "USER_REPORTED_NOT_EXACT_ACQUISITION_DATE")
                self.assertEqual(properties["source_outline_ref"],
                                 "2019_roof_outline.shp")
                self.assertEqual(properties["source_outline_ground_truth"], 0)
                self.assertEqual(properties["absolute_ground"], 0)
                self.assertEqual(properties["formal_ground_use"], 0)
                self.assertIsNone(properties["position_sigma_m"])
                if properties["actual_date"] == MOVING:
                    self.assertEqual(properties["support_sample_count"], 5)
        self.assertCountEqual(accepted, [f"{ANCHOR}:anchor-roof", f"{MOVING}:supported-roof"])

    def test_rejects_applied_or_non_roof_reference(self):
        applied_output = self.root / "applied"
        applied_output.mkdir()
        applied_graph = {**pose_graph(), "applied": True}
        with self.assertRaisesRegex(ValueError, "candidate-only"):
            write_relative_roof_candidate_products(
                applied_output, **{**self.common, "pose_graph": applied_graph})

        wrong_semantics_output = self.root / "wrong-semantics"
        wrong_semantics_output.mkdir()
        with self.assertRaisesRegex(ValueError, "image-derived roof boundary"):
            write_relative_roof_candidate_products(
                wrong_semantics_output, **{
                    **self.common,
                    "outline_source_geometry_semantics": "GROUND_FOOTPRINT",
                })


if __name__ == "__main__":
    unittest.main()
