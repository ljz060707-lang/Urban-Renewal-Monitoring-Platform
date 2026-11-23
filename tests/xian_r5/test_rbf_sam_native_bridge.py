from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.reconstruction.rbf_sam_native import (
    NATIVE_MANIFEST_SCHEMA,
    convert_native_output_files,
)
from xian_r5.reconstruction.rbf_sam_handoff import prepare_rbf_sam_handoff


class RbfSamNativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_image = self.root / "source.png"
        self.model_input = self.root / "model_input.png"
        self.native_output = self.root / "native.json"
        self.manifest_path = self.root / "binding.json"
        self.bundle_path = self.root / "bundle.json"
        self.checkpoint_path = self.root / "checkpoint.pth"
        self.config_path = self.root / "model_config.py"
        Image.new("RGB", (8, 6), (20, 40, 60)).save(self.source_image)
        Image.new("RGB", (4, 3), (20, 40, 60)).save(self.model_input)
        self.checkpoint_path.write_bytes(b"synthetic checkpoint fixture")
        self.config_path.write_text("model = 'RBF-SAM'\n", encoding="utf-8")
        self.manifest = {
            "schema_version": NATIVE_MANIFEST_SCHEMA,
            "scope": "FIXED_CASE_LOCAL_ONLY",
            "model": {
                "name": "RBF-SAM",
                "source": "https://github.com/Xuaner624/RBF-SAM",
                "revision": "test-revision",
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": self._hash(self.checkpoint_path),
                "config_path": str(self.config_path),
                "config_sha256": self._hash(self.config_path),
            },
            "images": [{
                "native_file_name": "tile_420797_312173.png",
                "source_image_path": str(self.source_image),
                "source_image_sha256": self._hash(self.source_image),
                "model_input_path": str(self.model_input),
                "model_input_sha256": self._hash(self.model_input),
                "model_input_width": 4,
                "model_input_height": 3,
                "preprocess_semantics": "FULL_TILE_ASPECT_PRESERVING_RESIZE_NO_CROP",
                "actual_date": "2018-04-29",
                "tile_id": "420797:312173",
                "prompts": [{
                    "native_annotation_id": 41,
                    "observation_uid": "B04-TQA10:2018:roof-1",
                    "prompt_source": "DETECTED_ROOF_INSTANCE",
                    "prompt_type": "roof_box",
                    "prompt_box_xyxy_pixel": [0.5, 0.25, 3.5, 2.75],
                }],
            }],
        }
        self.native_rows = [{
            "file_name": "tile_420797_312173.png",
            "id": 41,
            "roof_mask": [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
            ],
            "offset": [0.5, -1.25],
        }]
        self._write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_inputs(self):
        self.native_output.write_text(json.dumps(self.native_rows), encoding="utf-8")
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_converts_native_mask_offset_and_exact_bindings(self):
        result = convert_native_output_files(
            self.native_output, self.manifest_path, self.bundle_path)
        bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "UNVERIFIED_CANDIDATE_BUNDLE_WRITTEN")
        self.assertEqual(result["ground_footprints_confirmed"], 0)
        self.assertEqual(bundle["schema_version"], "xian_r5.rbf_sam_prediction_bundle.v2")
        prediction = bundle["predictions"][0]
        self.assertEqual(prediction["observation_uid"], "B04-TQA10:2018:roof-1")
        self.assertEqual(prediction["actual_date"], "2018-04-29")
        self.assertEqual(prediction["tile_id"], "420797:312173")
        self.assertEqual(prediction["predicted_offset_xy_px"], [0.5, -1.25])
        self.assertEqual(prediction["prompt_source"], "DETECTED_ROOF_INSTANCE")
        self.assertEqual(prediction["prompt_type"], "roof_box")
        self.assertEqual(prediction["prompt_box_xyxy_pixel"], [0.5, 0.25, 3.5, 2.75])
        self.assertEqual(bundle["model"]["checkpoint_sha256"], self._hash(self.checkpoint_path))
        self.assertEqual(bundle["native_inference"]["predictions_are_ground_truth"], False)

    def test_runner_handoff_converts_and_protects_generated_bundle(self):
        run_output = self.root / "fixed_case_run"
        bundle_path, report = prepare_rbf_sam_handoff(
            run_output,
            prediction_bundle=None,
            mosaic_diagnostic=self.root / "mosaic_diagnostic.json",
            native_output=self.native_output,
            inference_manifest=self.manifest_path,
        )

        self.assertEqual(
            bundle_path,
            run_output.resolve().with_name(
                run_output.name + "_rbf_sam_prediction_bundle.json"),
        )
        self.assertEqual(report["status"], "UNVERIFIED_CANDIDATE_BUNDLE_WRITTEN")
        self.assertEqual(
            json.loads(bundle_path.read_text(encoding="utf-8"))["schema_version"],
            "xian_r5.rbf_sam_prediction_bundle.v2",
        )
        with self.assertRaises(FileExistsError):
            prepare_rbf_sam_handoff(
                run_output,
                prediction_bundle=None,
                mosaic_diagnostic=self.root / "mosaic_diagnostic.json",
                native_output=self.native_output,
                inference_manifest=self.manifest_path,
            )

    def test_runner_handoff_rejects_ambiguous_or_incomplete_modes(self):
        run_output = self.root / "fixed_case_run"
        mosaic = self.root / "mosaic_diagnostic.json"
        bundle = self.root / "existing_bundle.json"
        with self.assertRaisesRegex(ValueError, "requires both"):
            prepare_rbf_sam_handoff(
                run_output, prediction_bundle=None, mosaic_diagnostic=mosaic,
                native_output=self.native_output, inference_manifest=None)
        with self.assertRaisesRegex(ValueError, "exact-date mosaic"):
            prepare_rbf_sam_handoff(
                run_output, prediction_bundle=None, mosaic_diagnostic=None,
                native_output=self.native_output, inference_manifest=self.manifest_path)
        with self.assertRaisesRegex(ValueError, "not both"):
            prepare_rbf_sam_handoff(
                run_output, prediction_bundle=bundle, mosaic_diagnostic=mosaic,
                native_output=self.native_output, inference_manifest=self.manifest_path)
        with self.assertRaisesRegex(ValueError, "both the prediction bundle"):
            prepare_rbf_sam_handoff(
                run_output, prediction_bundle=bundle, mosaic_diagnostic=None,
                native_output=None, inference_manifest=None)
        self.assertEqual(
            prepare_rbf_sam_handoff(
                run_output, prediction_bundle=bundle, mosaic_diagnostic=mosaic,
                native_output=None, inference_manifest=None),
            (bundle, None),
        )

    def test_rejects_unbound_native_prediction(self):
        self.native_rows[0]["id"] = 99
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "no exact prompt binding"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_crop_or_letterbox_preprocessing(self):
        self.manifest["images"][0]["preprocess_semantics"] = "CENTER_CROP"
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "FULL_TILE_ASPECT_PRESERVING_RESIZE_NO_CROP"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_mask_dimensions_that_do_not_match_model_input(self):
        self.native_rows[0]["roof_mask"] = [[0, 1], [1, 0]]
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_aspect_ratio_change_even_when_manifest_claims_full_tile(self):
        Image.new("RGB", (4, 4), (20, 40, 60)).save(self.model_input)
        self.manifest["images"][0]["model_input_width"] = 4
        self.manifest["images"][0]["model_input_height"] = 4
        self.manifest["images"][0]["model_input_sha256"] = self._hash(self.model_input)
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "aspect ratio differs"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_nonbinary_native_mask(self):
        self.native_rows[0]["roof_mask"][1][1] = 0.5
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "must already be binarized"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_empty_native_output(self):
        self.native_rows.clear()
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_missing_declared_prompt_result(self):
        self.manifest["images"][0]["prompts"].append({
            "native_annotation_id": 42,
            "observation_uid": "B04-TQA10:2018:roof-2",
            "prompt_source": "DETECTED_ROOF_INSTANCE",
            "prompt_type": "roof_box",
            "prompt_box_xyxy_pixel": [1, 1, 3, 2],
        })
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "missing 1 declared prompt"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_unknown_prompt_type(self):
        self.manifest["images"][0]["prompts"][0]["prompt_type"] = ["roof_box"]
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "prompt_type must explicitly"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_changed_source_input_hash(self):
        self.source_image.write_bytes(b"changed-source-tile")
        with self.assertRaisesRegex(ValueError, "source_image SHA-256 mismatch"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_changed_checkpoint_hash(self):
        self.checkpoint_path.write_bytes(b"different checkpoint")
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_rejects_prompt_box_outside_model_input(self):
        self.manifest["images"][0]["prompts"][0]["prompt_box_xyxy_pixel"] = [0, 0, 5, 2]
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "inside the model input"):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)

    def test_refuses_to_overwrite_bundle(self):
        self.bundle_path.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            convert_native_output_files(
                self.native_output, self.manifest_path, self.bundle_path)
        self.assertEqual(self.bundle_path.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
