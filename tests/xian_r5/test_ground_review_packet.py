from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import fiona
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.reconstruction.ground_review_packet import (
    REVIEW_FIELDS,
    REVIEW_LAYER,
    REVIEW_CANDIDATE_FIELDS,
    REVIEW_CANDIDATE_LAYER,
    export_ground_footprint_review_packet,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroundReviewPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "source_run"
        self.run_dir.mkdir()
        self.dates = ["2018-04-29", "2018-11-01"]
        self.tile_root = self.root / "tiles"
        self.tile_root.mkdir()
        self.base_path = self.root / "base10m.tif"
        with rasterio.open(self.base_path, "w", driver="GTiff", width=100,
                           height=100, count=3, dtype="uint8", crs="EPSG:32649",
                           transform=from_origin(0, 1000, 10, 10)) as sink:
            sink.write(np.zeros((3, 100, 100), dtype=np.uint8))

        with fiona.open(self.run_dir / "building_lineage.gpkg", "w", driver="GPKG",
                        layer="roof_candidates",
                        schema={"geometry": "Polygon", "properties": {
                            "uid": "str:96", "epoch": "str:16", "actual_date": "str:16",
                            "physical_id": "int", "source_ref": "str:254",
                            "status": "str:40", "area_m2": "float", "sigma_m": "float"}},
                        crs="EPSG:32649") as sink:
            for i, actual_date in enumerate(self.dates):
                uid = f"B04:{actual_date}:roof-{i}"
                geom = box(100 + 8 * i, 390, 120 + 8 * i, 410)
                sink.write({"geometry": mapping(geom), "properties": {
                    "uid": uid, "epoch": str(2018 + i), "actual_date": actual_date,
                    "physical_id": None, "source_ref": f"roof:{uid}",
                    "status": "ROOF_ONLY_UNRESOLVED", "area_m2": geom.area,
                    "sigma_m": None}})

        candidate_schema = {"geometry": "Polygon", "properties": {
            "uid": "str:96", "actual_date": "str:16", "candidate_id": "str:80",
            "source_ref": "str:254", "source_date": "str:16",
            "source_epoch_label": "str:32", "source_geometry_semantics": "str:48",
            "source_date_status": "str:48", "source_date_candidates_json": "str:256",
            "source_scene_linkage_status": "str:80", "association": "str:40",
            "status": "str:40"}}
        with fiona.open(self.run_dir / "building_lineage.gpkg", "w", driver="GPKG",
                        layer="footprint_candidates", schema=candidate_schema,
                        crs="EPSG:32649") as sink:
            for i, actual_date in enumerate(self.dates):
                sink.write({"geometry": mapping(box(102 + 8 * i, 392, 122 + 8 * i, 412)),
                            "properties": {
                                "uid": f"B04:{actual_date}:roof-{i}",
                                "actual_date": actual_date, "candidate_id": str(i + 1),
                                "source_ref": "candidate:roof-boundary", "source_date": None,
                                "source_epoch_label": "2019",
                                "source_geometry_semantics": "ROOF_BOUNDARY_FROM_IMAGE",
                                "source_date_status": "SINGLE_ACTUAL_DATE_CANDIDATE",
                                "source_date_candidates_json": '["2018-11-01"]',
                                "source_scene_linkage_status": "UNVERIFIED",
                                "association": "POSITIVE_AREA_OVERLAP_ONLY",
                                "status": "UNVERIFIED_CANDIDATE"}})

        tile_inventory = {}
        for date_index, actual_date in enumerate(self.dates):
            records = []
            for tile_x, tile_y, color in [
                    (10, 20, (30 + date_index, 0, 0)),
                    (11, 20, (0, 50 + date_index, 0)),
                    (10, 21, (0, 0, 70 + 40 * date_index)),
                    (11, 21, (80 + date_index, 80, 0))]:
                tile_path = self.tile_root / f"{actual_date}_{tile_x}_{tile_y}.jpg"
                Image.new("RGB", (256, 256), color).save(tile_path, quality=100)
                records.append({"tile_id": f"{tile_x}:{tile_y}", "path": str(tile_path),
                                "sha256": sha256(tile_path), "sha256_after_read": sha256(tile_path),
                                "selected_source_epoch": str(2018 + date_index)})
            tile_inventory[actual_date] = records

        self.diagnostic_path = self.root / "MOSAIC_REGISTRATION_DIAGNOSTIC.json"
        self.diagnostic_path.write_text(json.dumps({
            "schema_version": "xian_r5.mosaic_registration_diagnostic.v1",
            "case_id": "B04-TQA10",
            "parameters": {"registration_applied": False},
            "grid": {"crs": "EPSG:32649", "tile_window": {
                "x0": 10, "y0": 20, "nx": 2, "ny": 2},
                "pixel_transform": [1, 0, 0, 0, -1, 512]},
            "epochs": {actual_date: {
                "wanted_tiles": 4, "exact_date_tiles_read": 4,
                "exact_date_tile_coverage_fraction": 1.0} for actual_date in self.dates},
            "submeter_tiles": tile_inventory,
        }, indent=2), encoding="utf-8")

        (self.run_dir / "ENGINEERING_QA.json").write_text(json.dumps({
            "schema_version": "xian_r5.reconstruction_engineering_qa.v1",
            "case_id": "B04-TQA10", "structural_qa": "PASS", "errors": {},
            "r4_status_sha256_unchanged": True,
            "tracked_input_hashes_unchanged": True,
            "pixel_input_hashes_unchanged": True,
            "base_raster_hashes_unchanged": True,
        }), encoding="utf-8")
        (self.run_dir / "PRODUCT_MANIFEST.json").write_text(json.dumps({
            "schema_version": "xian_r5.grounded_products.v13",
            "crs": "EPSG:32649", "image_alignment_field_applied": False,
            "actual_date_order": self.dates,
            "reference_grid": {"path": str(self.base_path)},
        }), encoding="utf-8")
        (self.run_dir / "EVIDENCE_RECEIPT.json").write_text(json.dumps({
            "schema_version": "xian_r5.grounded_fixed_case_run.v1",
            "case_id": "B04-TQA10", "image_grid_alignment_applied_count": 0,
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_packet_preserves_exact_date_scene_grid_and_empty_review_gate(self):
        output = self.root / "review_packet"
        manifest = export_ground_footprint_review_packet(
            self.run_dir, self.diagnostic_path, output, patch_size_px=256)

        self.assertEqual(manifest["schema_version"],
                         "xian_r5.ground_footprint_review_packet.v1")
        self.assertEqual(manifest["actual_date_order"], self.dates)
        self.assertEqual(manifest["roof_candidate_count"], 2)
        self.assertEqual(manifest["external_footprint_candidate_association_count"], 2)
        self.assertEqual(manifest["reviewed_ground_footprints"]["template_feature_count"], 0)
        self.assertEqual(manifest["reviewed_ground_footprint_candidates"]["template_feature_count"], 0)
        self.assertFalse(manifest["reviewed_ground_footprint_candidates"][
            "formal_ground_footprint_confirmed"])
        self.assertFalse(manifest["ground_footprint_correction_applied"])
        self.assertFalse(manifest["positional_accuracy_claimed"])
        self.assertEqual(manifest["source_hash_recheck"], "PASS")

        instructions = (output / "REVIEW_INSTRUCTIONS.md").read_text(encoding="utf-8")
        self.assertNotIn("scenes/scenes/", instructions)
        self.assertIn(REVIEW_CANDIDATE_LAYER, instructions)
        self.assertIn("leave `positional_sigma_m` null", instructions)
        for actual_date in self.dates:
            self.assertIn(f"`scenes/{actual_date}_RGB_UTM.tif`", instructions)

        for actual_date in self.dates:
            with rasterio.open(output / "scenes" / f"{actual_date}_RGB_UTM.tif") as raster:
                self.assertEqual(raster.crs.to_epsg(), 32649)
                self.assertEqual(raster.transform, Affine(1, 0, 0, 0, -1, 512))
                self.assertEqual(raster.tags()["ORTHORECTIFIED"], "false")
                self.assertEqual(raster.tags()["POSITIONAL_ACCURACY_CLAIMED"], "false")
                self.assertEqual(raster.read(3)[0, 0],
                                 70 + 40 * self.dates.index(actual_date))
            self.assertTrue((output / "cards" / f"B04_{actual_date}_roof-"
                             f"{self.dates.index(actual_date)}.png").is_file())

        review_gpkg = output / "ground_footprint_review.gpkg"
        self.assertEqual(set(fiona.listlayers(review_gpkg)), {
            "roof_candidates", "footprint_candidates", REVIEW_LAYER,
            REVIEW_CANDIDATE_LAYER})
        with fiona.open(review_gpkg, layer=REVIEW_LAYER) as layer:
            self.assertEqual(len(layer), 0)
            self.assertEqual(set(layer.schema["properties"]), set(REVIEW_FIELDS))
        with fiona.open(review_gpkg, layer=REVIEW_CANDIDATE_LAYER) as layer:
            self.assertEqual(len(layer), 0)
            self.assertEqual(set(layer.schema["properties"]), set(REVIEW_CANDIDATE_FIELDS))
        with fiona.open(review_gpkg, layer="roof_candidates") as layer:
            self.assertEqual(len(layer), 2)
            self.assertTrue(all(row["properties"]["review_state"] == "NOT_REVIEWED"
                                for row in layer))
        with (output / "review_index.csv").open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["review_state"] == "NOT_REVIEWED" for row in rows))
        for relative_path, digest in manifest["output_files_sha256"].items():
            self.assertEqual(sha256(output / relative_path), digest)

    def test_explicit_qa_receipt_is_hashed_without_rewriting_source_run(self):
        alternate_qa = self.root / "named_engineering_qa.json"
        alternate_qa.write_text(
            (self.run_dir / "ENGINEERING_QA.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        (self.run_dir / "ENGINEERING_QA.json").unlink()
        output = self.root / "review_packet_with_named_qa"

        manifest = export_ground_footprint_review_packet(
            self.run_dir, self.diagnostic_path, output,
            patch_size_px=256, qa_receipt_path=alternate_qa)

        self.assertEqual(
            manifest["source_inputs"]["engineering_qa"]["path"],
            str(alternate_qa.resolve()))
        self.assertEqual(
            manifest["source_inputs"]["engineering_qa"]["sha256"],
            sha256(alternate_qa))
        self.assertFalse((self.run_dir / "ENGINEERING_QA.json").exists())
        self.assertEqual(manifest["source_hash_recheck"], "PASS")

    def test_incomplete_tile_coverage_fails_without_leaving_output(self):
        diagnostic = json.loads(self.diagnostic_path.read_text(encoding="utf-8"))
        diagnostic["epochs"][self.dates[0]]["exact_date_tiles_read"] = 3
        self.diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")
        output = self.root / "should_not_exist"
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            export_ground_footprint_review_packet(
                self.run_dir, self.diagnostic_path, output, patch_size_px=256)
        self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export_ground_footprint_review_packet(
                self.run_dir, self.diagnostic_path, output, patch_size_px=256)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
