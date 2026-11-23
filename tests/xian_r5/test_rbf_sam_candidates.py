"""Contract tests for unverified RBF-SAM roof-to-footprint proposals."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import fiona
from PIL import Image
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.affinity import affine_transform
from shapely.geometry import box, mapping
from shapely.ops import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.reconstruction import (  # noqa: E402
    Observation,
    attach_rbf_sam_footprint_candidates,
    polygonize_rbf_sam_roof_mask,
    reconstruct,
)
from xian_r5.reconstruction.products import write_products  # noqa: E402
from xian_r5.alignment.mosaic_grid import MosaicTileGrid  # noqa: E402
from xian_r5.reconstruction.rbf_sam_native import (  # noqa: E402
    NATIVE_MANIFEST_SCHEMA,
)
from xian_r5.reconstruction.rbf_sam_handoff import prepare_rbf_sam_handoff  # noqa: E402


class RbfSamCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scene_id = "scene-2018"
        self.tile_x, self.tile_y = 420796, 312172
        self.source_root = self.root / self.scene_id / "20"
        self.image_path = self.source_root / str(self.tile_x) / (
            f"gesh_{self.tile_x}_{self.tile_y}_20.jpg")
        self.image_path.parent.mkdir(parents=True)
        Image.new("RGB", (256, 256), (31, 72, 128)).save(self.image_path, format="JPEG")
        self.image_sha256 = hashlib.sha256(self.image_path.read_bytes()).hexdigest()
        self.geid_path = self.root / f"{self.scene_id}.geid"
        self.geid_path.write_text(
            "[HistoryDate]\nHistoryDate=2018-01-01\n\n"
            "[Zoom]\nZoom1=20\nZoom2=20\n\n"
            f"[XY]\nMinX_20={self.tile_x}\nMinY_20={self.tile_y}\n"
            f"MaxX_20={self.tile_x}\nMaxY_20={self.tile_y}\n\n"
            "[AREA]\nLeft_Longitude_download_20=110\n"
            "Right_Longitude_download_20=110.001\n"
            "Top_Latitude_download_20=34.001\n"
            "Bottom_Latitude_download_20=34\n",
            encoding="utf-8",
        )
        self.mosaic_path = self.root / "mosaic.tif"
        with rasterio.open(
            self.mosaic_path, "w", driver="GTiff", width=256, height=256,
            count=3, dtype="uint8", crs="EPSG:4326",
            transform=from_origin(110, 34.001, 0.001 / 256, 0.001 / 256),
        ) as mosaic:
            mosaic.write(np.zeros((3, 256, 256), dtype=np.uint8))
        self.grid = MosaicTileGrid.open(
            self.geid_path, self.mosaic_path, self.source_root)
        self.mosaic_diagnostic_path = self.root / "mosaic_diagnostic.json"
        self.mosaic_diagnostic_path.write_text(json.dumps({
            "schema_version": "xian_r5.mosaic_registration_diagnostic.v1",
            "scope": "FIXED_CASE_LOCAL_ONLY",
            "submeter_tiles": {"2018-04-29": [{
                "tile_id": f"{self.tile_x}:{self.tile_y}",
                "selected_source_epoch": "2018",
                "path": str(self.image_path.resolve()),
                "sha256": self.image_sha256,
            }]},
        }), encoding="utf-8")
        self.grids = {"2018": self.grid}
        self.uid = "case-a:2018:roof-01"
        model_grid = self.grid.model_input_grid_for_tile(
            self.tile_x, self.tile_y,
            model_input_width_px=256, model_input_height_px=256)
        pixel_affine = Affine(*model_grid.pixel_to_crs_affine)
        geographic_roof = affine_transform(
            box(1, 1, 3, 3),
            [pixel_affine.a, pixel_affine.b, pixel_affine.d, pixel_affine.e,
             pixel_affine.c, pixel_affine.f],
        )
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
        roof_utm = transform(to_utm.transform, geographic_roof)
        self.observation = Observation(
            uid=self.uid,
            epoch_label="2018",
            actual_date="2018-04-29",
            tile_id="420796:312172",
            image_path=str(self.image_path),
            roof=roof_utm,
            source_ref="fixed-case observation roof",
        )
        self.bundle_path = self.root / "rbf_predictions.json"
        self.bundle = {
            "schema_version": "xian_r5.rbf_sam_prediction_bundle.v2",
            "scope": "FIXED_CASE_LOCAL_ONLY",
            "offset_semantics": "RBF_SAM_BONAI_FINAL_OFFSET;APPLY_NEGATIVE_TO_ROOF_MASK",
            "model": {
                "name": "RBF-SAM",
                "source": "https://github.com/Xuaner624/RBF-SAM",
                "revision": "test-revision",
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "b" * 64,
            },
            "predictions": [self.prediction()],
        }
        self.write_bundle()

    def tearDown(self):
        self.temp.cleanup()

    def prediction(self, **overrides):
        row = {
            "prediction_id": "p-001",
            "observation_uid": self.uid,
            "actual_date": "2018-04-29",
            "tile_id": "420796:312172",
            "source_image_path": str(self.image_path),
            "source_image_sha256": self.image_sha256,
            "model_input_sha256": "c" * 64,
            "model_input_width": 256,
            "model_input_height": 256,
            "roof_geometry_pixel": mapping(box(1, 1, 3, 3)),
            "predicted_offset_xy_px": [1, -1],
            "prompt_source": "OBSERVATION_ROOF",
            "prompt_type": "roof_box",
            "prompt_box_xyxy_pixel": [1, 1, 3, 3],
        }
        row.update(overrides)
        return row

    def write_bundle(self):
        self.bundle_path.write_text(json.dumps(self.bundle), encoding="utf-8")

    def attach(self):
        return attach_rbf_sam_footprint_candidates(
            [self.observation], self.bundle_path,
            mosaic_diagnostic_path=self.mosaic_diagnostic_path,
            mosaic_grids_by_source_epoch=self.grids,
        )

    def test_import_derives_grid_from_real_tile_receipt_and_keeps_candidate_unverified(self):
        report = self.attach()

        self.assertEqual(report["status"], "UNVERIFIED_CANDIDATES_ATTACHED")
        self.assertEqual(report["attached_candidate_count"], 1)
        self.assertEqual(report["schema_version"], "xian_r5.rbf_sam_candidate_import.v2")
        self.assertEqual(report["mosaic_diagnostic_sha256"],
                         hashlib.sha256(self.mosaic_diagnostic_path.read_bytes()).hexdigest())
        candidate = self.observation.footprint_candidates[0]
        self.assertIsNotNone(candidate.source_roof_geometry)
        self.assertNotEqual(candidate.geometry.bounds, candidate.source_roof_geometry.bounds)
        self.assertEqual(candidate.source_accuracy_m, None)
        self.assertEqual(candidate.source_properties["applied_mask_translation_xy_px"], [-1.0, 1.0])
        self.assertEqual(
            candidate.source_properties["pixel_to_crs_affine_status"],
            "DERIVED_FROM_VALIDATED_GEID_MOSAIC_TILE_GRID",
        )
        self.assertEqual(candidate.source_properties["prompt_type"], "roof_box")
        self.assertEqual(candidate.source_properties["prompt_box_xyxy_pixel"], [1.0, 1.0, 3.0, 3.0])
        self.assertEqual(candidate.source_properties["source_pixel_window_col_row_width_height"],
                         [0, 0, 256, 256])
        self.assertEqual(candidate.source_properties["source_grid_scene_id"], self.scene_id)
        self.assertIn("#submeter_tiles/2018-04-29/420796:312172",
                      candidate.source_properties["actual_date_source_ref"])
        self.assertEqual(candidate.source_properties["pixel_crs"], "EPSG:4326")
        self.assertEqual(candidate.source_properties["pixel_to_crs_affine"],
                         [0.001 / 256, 0.0, 110.0, 0.0, -0.001 / 256, 34.001])
        self.assertFalse(candidate.source_properties["independent_ground_evidence"])
        self.assertIsNone(self.observation.ground_footprint)

    def test_native_output_bridge_bundle_runs_through_candidate_importer(self):
        native_output_path = self.root / "native_rbf_sam.json"
        manifest_path = self.root / "native_binding.json"
        run_output_path = self.root / "native_run"
        checkpoint_path = self.root / "checkpoint.pth"
        config_path = self.root / "model_config.py"
        checkpoint_path.write_bytes(b"synthetic checkpoint fixture")
        config_path.write_text("model = 'RBF-SAM'\n", encoding="utf-8")
        manifest = {
            "schema_version": NATIVE_MANIFEST_SCHEMA,
            "scope": "FIXED_CASE_LOCAL_ONLY",
            "model": {
                "name": "RBF-SAM",
                "source": "https://github.com/Xuaner624/RBF-SAM",
                "revision": "test-revision",
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            },
            "images": [{
                "native_file_name": self.image_path.name,
                "source_image_path": str(self.image_path),
                "source_image_sha256": self.image_sha256,
                "model_input_path": str(self.image_path),
                "model_input_sha256": self.image_sha256,
                "model_input_width": 256,
                "model_input_height": 256,
                "preprocess_semantics": "FULL_TILE_ASPECT_PRESERVING_RESIZE_NO_CROP",
                "actual_date": "2018-04-29",
                "tile_id": "420796:312172",
                "prompts": [{
                    "native_annotation_id": 41,
                    "observation_uid": self.uid,
                    "prompt_source": "OBSERVATION_ROOF",
                    "prompt_type": "roof_box",
                    "prompt_box_xyxy_pixel": [20, 20, 24, 24],
                }],
            }],
        }
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[20:24, 20:24] = 1
        native_output_path.write_text(json.dumps([{
            "file_name": self.image_path.name,
            "id": 41,
            "roof_mask": mask.tolist(),
            "offset": [1.0, -1.0],
        }]), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        self.bundle_path, bridge_report = prepare_rbf_sam_handoff(
            run_output_path,
            prediction_bundle=None,
            mosaic_diagnostic=self.mosaic_diagnostic_path,
            native_output=native_output_path,
            inference_manifest=manifest_path,
        )
        self.assertEqual(bridge_report["status"], "UNVERIFIED_CANDIDATE_BUNDLE_WRITTEN")
        self.assertEqual(
            self.bundle_path,
            run_output_path.resolve().with_name(
                run_output_path.name + "_rbf_sam_prediction_bundle.json"),
        )
        report = self.attach()

        self.assertEqual(report["status"], "UNVERIFIED_CANDIDATES_ATTACHED")
        self.assertEqual(report["attached_candidate_count"], 1)
        self.assertEqual(
            self.observation.footprint_candidates[0].source_properties["prompt_type"],
            "roof_box",
        )
        self.assertEqual(
            self.observation.footprint_candidates[0].source_properties[
                "prompt_box_xyxy_pixel"
            ],
            [20.0, 20.0, 24.0, 24.0],
        )
        self.assertIsNone(self.observation.ground_footprint)

    def test_binary_mask_polygonization_preserves_pixel_grid_and_rejects_probabilities(self):
        import numpy as np
        from shapely.geometry import shape

        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        roof = shape(polygonize_rbf_sam_roof_mask(mask))
        self.assertEqual(roof.bounds, (2, 1, 4, 3))
        self.assertEqual(roof.area, 4)
        with self.assertRaisesRegex(ValueError, "already be binarized"):
            polygonize_rbf_sam_roof_mask(mask.astype(float) * 0.5)
        with self.assertRaisesRegex(ValueError, "no positive pixels"):
            polygonize_rbf_sam_roof_mask(np.zeros((4, 5), dtype=np.uint8))

    def test_import_rejects_benchmark_prompt_without_partial_attachment(self):
        self.bundle["predictions"] = [self.prediction(prompt_source="BENCHMARK_GT")]
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "current roof-instance source"):
            self.attach()
        self.assertEqual(self.observation.footprint_candidates, [])

    def test_import_rejects_actual_date_or_tile_mismatch(self):
        for override, message in (({"actual_date": "2018-11-01"}, "actual date"),
                                  ({"tile_id": "420797:312172"}, "tile ID")):
            with self.subTest(override=override):
                self.bundle["predictions"] = [self.prediction(**override)]
                self.write_bundle()
                with self.assertRaisesRegex(ValueError, message):
                    self.attach()
                self.assertEqual(self.observation.footprint_candidates, [])

    def test_import_rechecks_source_hash_and_rejects_untraceable_pixels(self):
        self.bundle["predictions"] = [self.prediction(source_image_sha256="d" * 64)]
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "source image hash mismatch"):
            self.attach()
        self.assertEqual(self.observation.footprint_candidates, [])

        self.bundle["predictions"] = [self.prediction(pixel_to_crs_affine=[1, 0, 0, 0, 0, 0])]
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "must not supply a pixel affine/CRS"):
            self.attach()
        self.assertEqual(self.observation.footprint_candidates, [])

    def test_import_rejects_mosaic_receipt_mismatch_and_non_full_tile_model_grid(self):
        self.bundle["predictions"] = [self.prediction(model_input_width=256,
                                                       model_input_height=128)]
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            self.attach()
        self.assertEqual(self.observation.footprint_candidates, [])

        self.bundle["predictions"] = [self.prediction()]
        report = json.loads(self.mosaic_diagnostic_path.read_text(encoding="utf-8"))
        report["submeter_tiles"]["2018-04-29"][0]["sha256"] = "0" * 64
        self.mosaic_diagnostic_path.write_text(json.dumps(report), encoding="utf-8")
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "receipt hash"):
            self.attach()
        self.assertEqual(self.observation.footprint_candidates, [])

    def test_products_keep_candidate_and_paired_source_roof_separate(self):
        self.attach()
        result = reconstruct([self.observation])
        output = self.root / "products"
        manifest = write_products(output, result)

        self.assertEqual(manifest["footprint_candidate_roof_source_count"], 1)
        self.assertIn("footprint_candidate_roof_sources", manifest["footprint_candidate_roof_source_layer"])
        self.assertEqual(manifest["formal_ground_observations"], 0)
        lineage_path = output / "building_lineage.gpkg"
        with fiona.open(lineage_path, layer="footprint_candidates") as layer:
            candidate = next(iter(layer))
            self.assertEqual(candidate["properties"]["status"], "UNVERIFIED_CANDIDATE")
            provenance = json.loads(candidate["properties"]["source_properties_json"])
            self.assertEqual(provenance["prediction_id"], "p-001")
            self.assertEqual(provenance["actual_date"], "2018-04-29")
        with fiona.open(lineage_path, layer="footprint_candidate_roof_sources") as layer:
            source_roof = next(iter(layer))
            self.assertEqual(source_roof["properties"]["candidate_id"], "RBF-p-001")


if __name__ == "__main__":
    unittest.main()
