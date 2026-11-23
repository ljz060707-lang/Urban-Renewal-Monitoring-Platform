"""Strict input-contract tests; synthetic records are not Xi'an accuracy evidence."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fiona
import numpy as np
import rasterio
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.reconstruction import (Observation, attach_reviewed_ground_footprints,
                                    reconstruct, recheck_review_sources)
from xian_r5.reconstruction.products import ID_UNKNOWN, STATE_CODES, write_products


class ReviewedGroundFootprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.evidence_path = self.root / "human_review_note.txt"
        self.evidence_path.write_text("synthetic reviewed source evidence\n", encoding="utf-8")
        self.evidence_sha256 = hashlib.sha256(self.evidence_path.read_bytes()).hexdigest()
        self.layer_path = self.root / "reviewed.gpkg"
        self.uid = "B04-TQA10:2018:roof-01"
        self.actual_date = "2018-04-29"

    def tearDown(self):
        self.temp.cleanup()

    def observation(self, *, actual_date="DEFAULT", uid=None):
        if actual_date == "DEFAULT":
            actual_date = self.actual_date
        return Observation(
            uid=uid or self.uid, epoch_label="2018", actual_date=actual_date,
            tile_id=None, image_path=None, roof=box(315200, 3799200, 315215, 3799210),
            source_ref="synthetic-roof-source", confirmed_identity_key="physical-a",
            identity_reviewed=True)

    def feature(self, **overrides):
        properties = {
            "uid": self.uid,
            "actual_date": self.actual_date,
            "source_ref": "review-session:case-b04/object-01",
            "method": "VISIBLE_GROUND_BOUNDARY",
            "independent_ground_evidence": 1,
            "reviewed": 1,
            "positional_sigma_m": 0.75,
            "roof_to_ground_dx_m": -2.5,
            "roof_to_ground_dy_m": 0.25,
            "reviewer_id": "reviewer-test-01",
            "reviewed_at": "2026-09-20T11:35:00+08:00",
            "evidence_path": self.evidence_path.name,
            "evidence_sha256": self.evidence_sha256,
            "physical_identity_key": None,
            "identity_reviewed": 0,
        }
        properties.update(overrides)
        return {"geometry": mapping(box(315198, 3799201, 315213, 3799211)),
                "properties": properties}

    def write_layer(self, features=None, *, crs="EPSG:32649"):
        schema = {"geometry": "Polygon", "properties": {
            "uid": "str:96", "actual_date": "str:16", "source_ref": "str:254",
            "method": "str:40", "independent_ground_evidence": "int", "reviewed": "int",
            "positional_sigma_m": "float", "roof_to_ground_dx_m": "float",
            "roof_to_ground_dy_m": "float", "reviewer_id": "str:96",
            "reviewed_at": "str:40", "evidence_path": "str:512",
            "evidence_sha256": "str:64", "physical_identity_key": "str:96",
            "identity_reviewed": "int"}}
        with fiona.open(self.layer_path, "w", driver="GPKG", layer="reviewed_ground_footprints",
                        schema=schema, crs=crs) as sink:
            for feature in features or []:
                sink.write(feature)
        return self.layer_path

    def test_valid_record_attaches_and_product_preserves_review_lineage(self):
        observation = self.observation()
        self.write_layer([self.feature(physical_identity_key="physical-a", identity_reviewed=1)])

        report = attach_reviewed_ground_footprints([observation], self.layer_path)
        result = reconstruct([observation])

        self.assertEqual(report["schema_version"], "xian_r5.reviewed_ground_footprints.v1")
        self.assertEqual(report["attached_uids"], [self.uid])
        self.assertEqual(report["source_hash_recheck"], "PASS")
        self.assertEqual(report["evidence_hash_recheck"], "PASS")
        self.assertEqual(observation.footprint_status, "GROUND_VERIFIED")
        self.assertEqual(observation.ground_footprint.bounds, (315198.0, 3799201.0, 315213.0, 3799211.0))
        self.assertIsNotNone(observation.physical_id)
        products = self.root / "products"
        write_products(products, result)
        with fiona.open(products / "building_lineage.gpkg", layer="ground_footprints") as layer:
            properties = dict(next(iter(layer))["properties"])
        self.assertEqual(properties["source_ref"], "review-session:case-b04/object-01")
        self.assertEqual(properties["reviewer_id"], "reviewer-test-01")
        self.assertEqual(properties["reviewed_at"], "2026-09-20T11:35:00+08:00")
        self.assertEqual(properties["evidence_sha256"], self.evidence_sha256)
        self.assertEqual(properties["independent_ground_evidence"], 1)
        self.assertEqual(properties["reviewed"], 1)
        self.assertAlmostEqual(properties["roof_to_ground_dx_m"], -2.5)
        self.assertEqual(recheck_review_sources(report), report["integrity_files_sha256"])

    def test_reviewed_identity_key_yields_stable_id_and_change_state(self):
        first = self.observation()
        first.confirmed_identity_key = None
        first.identity_reviewed = False
        second = self.observation(uid="B04-TQA10:2019:roof-01", actual_date="2018-11-01")
        second.epoch_label = "2019"
        second.confirmed_identity_key = None
        second.identity_reviewed = False
        self.write_layer([
            self.feature(physical_identity_key="physical-a", identity_reviewed=1),
            self.feature(uid=second.uid, actual_date=second.actual_date,
                         physical_identity_key="physical-a", identity_reviewed=1),
        ])

        attach_reviewed_ground_footprints([first, second], self.layer_path)
        result = reconstruct([first, second])
        self.assertEqual(first.physical_id, second.physical_id)
        self.assertIsNotNone(first.physical_id)
        self.assertEqual([row["state"] for row in result.transitions], ["STABLE"])
        output = self.root / "products_with_reviewed_identity"
        write_products(output, result)
        with rasterio.open(output / "2018-04-29_to_2018-11-01_change.tif") as change:
            self.assertTrue(np.any(change.read(1) == STATE_CODES["STABLE"]))

    def test_empty_layer_is_a_valid_unresolved_input(self):
        observation = self.observation()
        self.write_layer([])

        report = attach_reviewed_ground_footprints([observation], self.layer_path)
        result = reconstruct([observation])

        self.assertEqual(report["attached_observation_count"], 0)
        self.assertIsNone(observation.ground_footprint)
        self.assertIsNone(observation.physical_id)
        self.assertEqual(result.observations[0].footprint_status, "ROOF_ONLY_UNRESOLVED")

    def test_shifted_roofs_without_ground_evidence_do_not_become_real_change(self):
        first = self.observation()
        second = self.observation(uid="B04-TQA10:2019:roof-01",
                                  actual_date="2018-11-01")
        second.epoch_label = "2019"
        second.roof = box(315206, 3799200, 315221, 3799210)

        # Even a candidate identity reviewed independently of geometry cannot
        # turn image-roof polygons into ground footprints or change evidence.
        result = reconstruct([first, second])

        self.assertEqual([row["presence"] for row in result.presence], ["U", "U"])
        self.assertEqual(result.transitions, [])
        self.assertTrue(all(obs.ground_footprint is None for obs in result.observations))
        self.assertTrue(all(obs.physical_id is None for obs in result.observations))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchor = root / "base10m.tif"
            with rasterio.open(anchor, "w", driver="GTiff", width=100, height=100,
                               count=1, dtype="uint8", crs="EPSG:32649",
                               transform=rasterio.transform.from_origin(
                                   315000, 3799400, 10, 10)) as sink:
                sink.write(np.zeros((1, 100, 100), dtype="uint8"))

            output = root / "products"
            write_products(output, result, anchor_raster=anchor)

            for actual_date in ("2018-04-29", "2018-11-01"):
                with rasterio.open(output / f"{actual_date}_ground_footprint.tif") as raster:
                    values = set(np.unique(raster.read(1)).tolist())
                    self.assertIn(2, values)  # U: roof location unresolved
                    self.assertNotIn(1, values)  # no confirmed ground footprint
                with rasterio.open(output / f"{actual_date}_physical_id.tif") as raster:
                    self.assertIn(ID_UNKNOWN, raster.read(1))

            interval = "2018-04-29_to_2018-11-01"
            with rasterio.open(output / f"{interval}_change.tif") as raster:
                values = set(np.unique(raster.read(1)).tolist())
                self.assertIn(4, values)
                self.assertTrue(values.isdisjoint({1, 2, 3}))

            aligned_products = [
                rasterio.open(output / f"{actual_date}_ground_footprint_BASE10M.tif")
                for actual_date in ("2018-04-29", "2018-11-01")
            ]
            try:
                self.assertEqual(aligned_products[0].crs, aligned_products[1].crs)
                self.assertEqual(aligned_products[0].transform,
                                 aligned_products[1].transform)
                self.assertEqual(aligned_products[0].res, (10.0, 10.0))
            finally:
                for raster in aligned_products:
                    raster.close()

    def test_confirmed_ground_geometry_outputs_without_confirmed_identity(self):
        first = self.observation()
        first.confirmed_identity_key = None
        first.identity_reviewed = False
        second = self.observation(uid="B04-TQA10:2019:roof-01", actual_date="2018-11-01")
        second.epoch_label = "2019"
        second.confirmed_identity_key = None
        second.identity_reviewed = False
        self.write_layer([
            self.feature(),
            self.feature(uid=second.uid, actual_date=second.actual_date),
        ])

        attach_reviewed_ground_footprints([first, second], self.layer_path)
        result = reconstruct([first, second])
        self.assertEqual(result.transitions, [])
        self.assertEqual([row["presence"] for row in result.presence], ["U", "U"])
        output = self.root / "products_without_confirmed_identity"
        manifest = write_products(output, result)

        with rasterio.open(output / "2018-04-29_ground_footprint.tif") as footprints:
            self.assertTrue(np.any(footprints.read(1) == 1))
        with rasterio.open(output / "2018-04-29_physical_id.tif") as ids:
            self.assertTrue(np.any(ids.read(1) == ID_UNKNOWN))
        with rasterio.open(output / "2018-04-29_to_2018-11-01_change.tif") as change:
            array = change.read(1)
            self.assertTrue(np.any(array == STATE_CODES["UNRESOLVED"]))
            self.assertFalse(np.any(np.isin(array, [STATE_CODES["STABLE"],
                                                    STATE_CODES["ADDITION"],
                                                    STATE_CODES["DEMOLITION"]])))
        self.assertEqual(manifest["confirmed_ground_geometry_observations"], 2)
        self.assertEqual(manifest["confirmed_physical_identity_observations"], 0)

    def test_unreviewed_or_inaccurate_records_are_rejected(self):
        cases = [
            ({"reviewed": 0}, "completed human review"),
            ({"independent_ground_evidence": 0}, "independent ground evidence"),
            ({"positional_sigma_m": 2.01}, "between 0 and 2 m"),
            ({"reviewed_at": "2026-09-20T11:35:00"}, "include a timezone"),
            ({"reviewer_id": ""}, "reviewer_id is required"),
            ({"physical_identity_key": "unreviewed-key"}, "supplied together"),
            ({"identity_reviewed": 1}, "supplied together"),
            ({"evidence_path": self.layer_path.name}, "cannot be independent evidence"),
            ({"method": "SPATIAL_NEAREST_NEIGHBOR"}, "Unsupported reviewed footprint method"),
            ({"evidence_sha256": "0" * 64}, "Evidence SHA-256 mismatch"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                self.write_layer([self.feature(**overrides)])
                with self.assertRaisesRegex(ValueError, message):
                    attach_reviewed_ground_footprints([self.observation()], self.layer_path)

    def test_exact_uid_and_actual_date_are_mandatory(self):
        for feature, observation, message in [
            (self.feature(uid="unknown"), self.observation(), "unknown observation UID"),
            (self.feature(actual_date="2018-11-01"), self.observation(), "does not match observation"),
            (self.feature(), self.observation(actual_date=None), "date is unresolved"),
        ]:
            with self.subTest(message=message):
                self.write_layer([feature])
                with self.assertRaisesRegex(ValueError, message):
                    attach_reviewed_ground_footprints([observation], self.layer_path)

    def test_duplicate_records_and_non_metric_crs_are_rejected(self):
        self.write_layer([self.feature(), self.feature()])
        with self.assertRaisesRegex(ValueError, "More than one reviewed footprint"):
            attach_reviewed_ground_footprints([self.observation()], self.layer_path)

        self.write_layer([self.feature()], crs="EPSG:4326")
        with self.assertRaisesRegex(ValueError, "must use EPSG:32649"):
            attach_reviewed_ground_footprints([self.observation()], self.layer_path)

    def test_source_hash_change_and_post_load_evidence_mutation_are_detected(self):
        self.write_layer([self.feature()])
        observation = self.observation()
        with patch("xian_r5.reconstruction.reviewed_footprints._source_component_hashes",
                   side_effect=[{"source": "a"}, {"source": "b"}]):
            with self.assertRaisesRegex(RuntimeError, "source changed while being read"):
                attach_reviewed_ground_footprints([observation], self.layer_path)

        report = attach_reviewed_ground_footprints([observation], self.layer_path)
        self.evidence_path.write_text("changed after the geometry review was loaded\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source changed during reconstruction"):
            recheck_review_sources(report)


if __name__ == "__main__":
    unittest.main()
