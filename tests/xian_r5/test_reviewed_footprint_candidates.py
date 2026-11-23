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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.reconstruction.core import Observation, reconstruct
from xian_r5.reconstruction.products import write_products
from xian_r5.reconstruction.reviewed_candidates import (
    LAYER_NAME,
    attach_reviewed_ground_footprint_candidates,
    recheck_reviewed_candidate_sources,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReviewedFootprintCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scene = self.root / "2018-04-29_RGB_UTM.tif"
        with rasterio.open(self.scene, "w", driver="GTiff", width=300,
                           height=300, count=3, dtype="uint8",
                           crs="EPSG:32649", transform=from_origin(0, 300, 1, 1)) as sink:
            sink.write(np.zeros((3, 300, 300), dtype=np.uint8))
            sink.update_tags(
                PRODUCT_KIND="RAW_ACTUAL_DATE_SCENE_REVIEW_CONTEXT",
                ACTUAL_DATE="2018-04-29", ORTHORECTIFIED="false")
        self.gpkg = self.root / "review.gpkg"
        self.observation = Observation(
            uid="B04:2018-04-29:roof-1",
            epoch_label="2018",
            actual_date="2018-04-29",
            tile_id="420798:312173",
            image_path=str(self.scene),
            roof=box(100, 100, 112, 110),
            source_ref="roof:1",
            covered_tiles=("MAIN_Z20:420798:312173",),
            date_status="DATE_COHERENT_FULL_ROOF_COVERAGE",
        )

    def tearDown(self):
        self.temp.cleanup()

    def write_layer(self, *, sigma=None, evidence_digest=None, uncertainty_basis=None,
                    evidence_path=None):
        schema = {
            "geometry": "Polygon",
            "properties": {
                "uid": "str:96", "actual_date": "str:16",
                "candidate_id": "str:80", "source_ref": "str:254",
                "review_status": "str:48", "reviewer_id": "str:96",
                "reviewed_at": "str:40", "evidence_path": "str:512",
                "evidence_sha256": "str:64", "uncertainty_basis": "str:40",
                "positional_sigma_m": "float", "notes": "str:512",
            },
        }
        with fiona.open(self.gpkg, "w", driver="GPKG", layer=LAYER_NAME,
                        schema=schema, crs="EPSG:32649") as sink:
            sink.write({
                "geometry": mapping(box(101, 100, 113, 110)),
                "properties": {
                    "uid": self.observation.uid,
                    "actual_date": self.observation.actual_date,
                    "candidate_id": "manual-1",
                    "source_ref": "review:manual-1",
                    "review_status": "VISIBLE_GROUND_BOUNDARY_CANDIDATE",
                    "reviewer_id": "reviewer-1",
                    "reviewed_at": "2026-09-28T10:30:00+08:00",
                    "evidence_path": evidence_path or self.scene.name,
                    "evidence_sha256": evidence_digest or sha256(self.scene),
                    "uncertainty_basis": uncertainty_basis or "UNKNOWN_ABSOLUTE_GEOREF",
                    "positional_sigma_m": sigma,
                    "notes": "Visible edge candidate; no absolute control.",
                },
            })

    def test_candidate_is_hash_verified_but_never_promoted_to_ground_or_change(self):
        self.write_layer()
        report = attach_reviewed_ground_footprint_candidates(
            [self.observation], self.gpkg)

        self.assertEqual(report["attached_candidate_count"], 1)
        self.assertEqual(report["attached_observation_count"], 1)
        self.assertEqual(report["ground_footprints_confirmed"], 0)
        self.assertFalse(report["physical_identity_or_change_state_created"])
        candidate = self.observation.footprint_candidates[0]
        self.assertEqual(candidate.source_geometry_semantics,
                         "VISIBLE_GROUND_BOUNDARY_CANDIDATE")
        self.assertEqual(candidate.source_capture_date, "2018-04-29")
        self.assertIsNone(candidate.source_accuracy_m)
        self.assertIsNone(self.observation.ground_footprint)

        reconstruction = reconstruct([self.observation])
        self.assertIsNone(reconstruction.observations[0].physical_id)
        self.assertEqual(reconstruction.presence[0]["presence"], "U")
        self.assertEqual(reconstruction.transitions, [])

        output = self.root / "products"
        manifest = write_products(output, reconstruction)
        products = manifest["ground_footprint_candidate_products"]
        self.assertEqual(len(products), 1)
        self.assertFalse(products[0]["formal_ground_footprint_product"])
        with rasterio.open(output / products[0]["candidate_tif"]) as raster:
            self.assertIn(1, np.unique(raster.read(1)))
            self.assertEqual(raster.tags()["FORMAL_GROUND_FOOTPRINT_PRODUCT"], "NO")
        with fiona.open(output / "building_lineage.gpkg",
                        layer="footprint_candidates") as layer:
            rows = list(layer)
        self.assertEqual(len(rows), 1)
        properties = dict(rows[0]["properties"])
        self.assertEqual(properties["association"], "MANUAL_VISUAL_CANDIDATE_ONLY")
        review_provenance = json.loads(properties["source_properties_json"])
        self.assertFalse(review_provenance["formal_ground_footprint_confirmed"])

        final_hashes = recheck_reviewed_candidate_sources(report)
        self.assertEqual(final_hashes, report["integrity_files_sha256"])

    def test_candidate_layer_rejects_numeric_accuracy_claims_and_bad_evidence_hashes(self):
        self.write_layer(sigma=1.5)
        with self.assertRaisesRegex(ValueError, "must remain null"):
            attach_reviewed_ground_footprint_candidates([self.observation], self.gpkg)

        self.write_layer(evidence_digest="0" * 64)
        with self.assertRaisesRegex(ValueError, "Evidence SHA-256 mismatch"):
            attach_reviewed_ground_footprint_candidates([self.observation], self.gpkg)

        self.write_layer(evidence_path="2019-12-02_RGB_UTM.tif")
        with self.assertRaisesRegex(ValueError, "exact-date scene raster"):
            attach_reviewed_ground_footprint_candidates([self.observation], self.gpkg)


if __name__ == "__main__":
    unittest.main()
