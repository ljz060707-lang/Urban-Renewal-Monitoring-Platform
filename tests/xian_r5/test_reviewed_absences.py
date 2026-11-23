"""Absence evidence interface tests; synthetic inputs are not Xi'an accuracy evidence."""

from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import fiona
import rasterio
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.reconstruction import (AbsenceEvidence, FootprintEvidence, Observation,
                                    attach_reviewed_physical_identities,
                                    load_reviewed_absences, reconstruct,
                                    recheck_review_sources)
from xian_r5.reconstruction.products import STATE_CODES, write_products


class ReviewedAbsenceTests(unittest.TestCase):
    fields = [
        "confirmed_identity_key", "epoch_label", "actual_date", "image_ref",
        "full_footprint_coverage", "reviewed", "reviewer_id", "reviewed_at",
        "evidence_path", "evidence_sha256",
    ]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.support = self.root / "absence_review_source.txt"
        self.support.write_text("synthetic fully-covered reviewed absence evidence\n", encoding="utf-8")
        self.support_sha = hashlib.sha256(self.support.read_bytes()).hexdigest()
        self.csv_path = self.root / "reviewed_absences.csv"

    def tearDown(self):
        self.temp.cleanup()

    def positive_observation(self):
        evidence = FootprintEvidence(
            geometry=box(315200, 3799200, 315215, 3799210),
            source_ref="survey:building-a", method="SURVEY",
            independent_ground_evidence=True, reviewed=True, positional_sigma_m=0.5,
            reviewer_id="reviewer-01", reviewed_at="2026-09-20T10:00:00+00:00",
            evidence_path="positive-ground-source", evidence_sha256="a" * 64,
            evidence_integrity_verified=True)
        return Observation(
            uid="B04:2018:building-a", epoch_label="2018", actual_date="2018-04-29",
            tile_id=None, image_path=None, roof=box(315203, 3799200, 315218, 3799210),
            source_ref="roof:building-a", footprint_evidence=evidence,
            confirmed_identity_key="building-a", identity_reviewed=True,
            identity_evidence_type="REVIEWED_GROUND_FOOTPRINT_RECORD",
            identity_evidence_ref=evidence.source_ref,
            identity_evidence_path=evidence.evidence_path,
            identity_evidence_sha256=evidence.evidence_sha256,
            identity_reviewer_id=evidence.reviewer_id,
            identity_reviewed_at=evidence.reviewed_at,
            identity_evidence_integrity_verified=True)

    def second_epoch_observation(self):
        return Observation(
            uid="B04:2019:neighbor", epoch_label="2019", actual_date="2018-11-01",
            tile_id=None, image_path=None, roof=box(315300, 3799200, 315315, 3799210),
            source_ref="roof:neighbor")

    def absence_row(self, **overrides):
        row = {
            "confirmed_identity_key": "building-a",
            "epoch_label": "2019",
            "actual_date": "2018-11-01",
            "image_ref": "dated-image:2018-11-01:tile-01",
            "full_footprint_coverage": "1",
            "reviewed": "1",
            "reviewer_id": "reviewer-02",
            "reviewed_at": "2026-09-21T10:00:00+00:00",
            "evidence_path": self.support.name,
            "evidence_sha256": self.support_sha,
        }
        row.update(overrides)
        return row

    def write_csv(self, rows):
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)
        return self.csv_path

    def test_reviewed_full_coverage_absence_produces_demolition(self):
        first = self.positive_observation()
        second = self.second_epoch_observation()
        self.write_csv([self.absence_row()])

        absences, report = load_reviewed_absences([first, second], self.csv_path)
        result = reconstruct([first, second], absence_evidence=absences)

        self.assertEqual(len(absences), 1)
        self.assertTrue(absences[0].accepted)
        self.assertEqual(report["reviewed_absence_count"], 1)
        self.assertEqual(report["source_hash_recheck"], "PASS")
        self.assertEqual(recheck_review_sources(report), report["integrity_files_sha256"])
        absence_presence = next(row for row in result.presence if row["presence"] == "0")
        self.assertEqual(absence_presence["reviewer_id"], "reviewer-02")
        self.assertEqual(absence_presence["image_ref"], "dated-image:2018-11-01:tile-01")
        self.assertEqual(absence_presence["evidence_sha256"], self.support_sha)
        self.assertEqual([row["state"] for row in result.transitions], ["DEMOLITION"])
        output = self.root / "demolition_products"
        manifest = write_products(output, result)
        with rasterio.open(output / "2018-04-29_to_2018-11-01_change.tif") as change:
            self.assertTrue(np.any(change.read(1) == STATE_CODES["DEMOLITION"]))
        self.assertEqual(manifest["schema_version"], "xian_r5.grounded_products.v17")
        self.assertEqual(manifest["presence_states_layer"],
                         "building_lineage.gpkg#building_presence")
        self.assertEqual(manifest["presence_state_counts"], {"1": 1, "0": 1, "U": 1})
        with fiona.open(output / "building_lineage.gpkg", layer="building_presence") as source:
            rows = [dict(feature["properties"]) for feature in source]
        self.assertEqual(len(rows), 3)
        absent = next(row for row in rows if row["presence_state"] == "0")
        self.assertIsNone(absent["observation_id"])
        self.assertEqual(absent["actual_date"], "2018-11-01")
        self.assertEqual(absent["confirmed_identity_key"], "building-a")
        self.assertEqual(absent["presence_evidence_type"], "REVIEWED_FULL_FOOTPRINT_COVERAGE")
        self.assertEqual(absent["coverage_complete"], 1)
        self.assertEqual(absent["evidence_sha256"], self.support_sha)
        unresolved = next(row for row in rows if row["presence_state"] == "U")
        self.assertEqual(unresolved["observation_id"], "B04:2019:neighbor")
        self.assertIn("not a reviewed absence", unresolved["notes"])

    def test_zero_coverage_unknown_key_and_wrong_actual_date_are_rejected(self):
        first, second = self.positive_observation(), self.second_epoch_observation()
        invalid_rows = [
            (self.absence_row(full_footprint_coverage="0"), "requires full-footprint coverage"),
            (self.absence_row(confirmed_identity_key="not-reviewed"), "identity without reviewed identity evidence"),
            (self.absence_row(actual_date="2019-12-02"), "not an observed actual date"),
            (self.absence_row(reviewed="0"), "requires full-footprint coverage"),
            (self.absence_row(reviewed_at="2026-09-21T10:00:00"), "must include a timezone"),
            (self.absence_row(evidence_sha256="0" * 64), "SHA-256 mismatch"),
        ]
        for row, message in invalid_rows:
            with self.subTest(row=row):
                self.write_csv([row])
                with self.assertRaisesRegex(ValueError, message):
                    load_reviewed_absences([first, second], self.csv_path)

    def test_duplicate_key_epoch_and_post_load_evidence_change_are_rejected(self):
        first, second = self.positive_observation(), self.second_epoch_observation()
        self.write_csv([self.absence_row(), self.absence_row()])
        with self.assertRaisesRegex(ValueError, "Duplicate reviewed absence"):
            load_reviewed_absences([first, second], self.csv_path)

        self.write_csv([self.absence_row()])
        _, report = load_reviewed_absences([first, second], self.csv_path)
        self.support.write_text("changed after review load\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source changed during reconstruction"):
            recheck_review_sources(report)

    def test_absence_is_validated_against_exact_date_and_source_label(self):
        first = self.positive_observation()
        second = self.second_epoch_observation()
        duplicate_date_source = Observation(
            uid="B04:2020:other", epoch_label="2020", actual_date="2018-11-01",
            tile_id=None, image_path=None, roof=box(315400, 3799200, 315415, 3799210),
            source_ref="roof:other")
        row = self.absence_row(epoch_label="2020")
        self.write_csv([row])

        absences, _ = load_reviewed_absences(
            [first, second, duplicate_date_source], self.csv_path)

        self.assertEqual(len(absences), 1)
        self.assertEqual(absences[0].actual_date, "2018-11-01")
        self.assertEqual(absences[0].epoch_label, "2020")

        self.write_csv([self.absence_row(epoch_label="2020", actual_date="2018-04-29")])
        with self.assertRaisesRegex(ValueError, "not observed under the supplied epoch label"):
            load_reviewed_absences([first, second, duplicate_date_source], self.csv_path)

    def test_direct_absence_object_needs_review_provenance(self):
        absent = AbsenceEvidence("building-a", "2019", "2018-11-01", "image", True, True)
        self.assertFalse(absent.accepted)

    def test_reviewed_identity_is_independent_of_ground_footprint(self):
        evidence_path = self.root / "identity_review_evidence.txt"
        evidence_path.write_text("synthetic reviewed cross-date identity evidence\n",
                                 encoding="utf-8")
        evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        identity_gpkg = self.root / "reviewed_identities.gpkg"
        schema = {"geometry": "None", "properties": {
            "uid": "str:96", "actual_date": "str:16",
            "physical_identity_key": "str:96", "reviewed": "int",
            "evidence_type": "str:64", "evidence_ref": "str:1024",
            "evidence_path": "str:1024", "evidence_sha256": "str:64",
            "reviewer_id": "str:96", "reviewed_at": "str:40"}}
        with fiona.open(identity_gpkg, "w", driver="GPKG",
                        layer="reviewed_physical_identities", schema=schema) as sink:
            sink.write({"geometry": None, "properties": {
                "uid": "B04:2018:building-a", "actual_date": "2018-04-29",
                "physical_identity_key": "building-a", "reviewed": 1,
                "evidence_type": "CROSS_DATE_MANUAL_REVIEW",
                "evidence_ref": "review:building-a",
                "evidence_path": evidence_path.name,
                "evidence_sha256": evidence_sha,
                "reviewer_id": "reviewer-03",
                "reviewed_at": "2026-09-22T10:00:00+00:00"}})
        observation = Observation(
            uid="B04:2018:building-a", epoch_label="2018",
            actual_date="2018-04-29", tile_id=None, image_path=None,
            roof=box(315200, 3799200, 315215, 3799210), source_ref="roof:building-a")
        report = attach_reviewed_physical_identities([observation], identity_gpkg)
        result = reconstruct([observation])
        self.assertEqual(report["reviewed_identity_observation_count"], 1)
        self.assertIsNone(observation.ground_footprint)
        self.assertIsNotNone(observation.physical_id)
        self.assertEqual(result.presence[0]["presence"], "1")
        self.assertEqual(result.presence[0]["identity_evidence_sha256"], evidence_sha)
        output = self.root / "identity_only_products"
        manifest = write_products(output, result)
        self.assertEqual(manifest["confirmed_ground_geometry_observations"], 0)
        with fiona.open(output / "building_lineage.gpkg", layer="ground_footprints") as source:
            self.assertEqual(len(source), 0)
        with fiona.open(output / "building_lineage.gpkg", layer="identity_graph_nodes") as source:
            self.assertIsNotNone(next(iter(source))["properties"]["physical_id"])
        with fiona.open(output / "building_lineage.gpkg", layer="building_presence") as source:
            presence = list(source)[0]["properties"]
        self.assertEqual(presence["presence_state"], "1")
        self.assertEqual(presence["identity_evidence_sha256"], evidence_sha)


if __name__ == "__main__":
    unittest.main()
