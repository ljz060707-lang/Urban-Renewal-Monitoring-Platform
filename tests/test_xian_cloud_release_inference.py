"""Synthetic torch/geometric input-contract QA. No real inference or GT reads."""
import copy
import importlib.util
import io
from pathlib import Path
import types
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
from affine import Affine
from shapely.geometry import Polygon, box, mapping, shape
import torch

_path = Path(__file__).resolve().parents[1] / "tools/xian_cloud_release_inference.py"
_spec = importlib.util.spec_from_file_location("release_infer_tested", _path)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
torch.set_num_threads(2)


def npz(extra=None):
    stream = io.BytesIO()
    np.savez(stream, hr_rgb=np.zeros((3, 256, 256), np.uint8), lr_rgb=np.zeros((3, 64, 64), np.uint8),
             source_support=np.ones((256, 256), np.uint8), affine=np.array([2.5, 0, 0, 0, -2.5, 640]), **(extra or {}))
    stream.seek(0)
    return stream


def row():
    return {"region_id": "SYNTHETIC", "native_image": "data/native/a.png", "feature_input": "data/xian/TRAIN/a_input.npz",
            "source_window": [0, 0, 1026, 40], "source_raster_affine": [2.68e-6, 0, 108.8, 0, -2.68e-6, 34.3],
            "source_raster_crs": "EPSG:4326", "source_observed_date": "2018-11-01",
            "native_image_sha256": "a"*64, "exact_support": mapping(box(0, 0, 640, 640))}


def instance(geometry):
    return {"original_geometry": mapping(geometry), "candidate_geometry": None, "model_dx_m": None,
            "model_dy_m": None, "risks": [], "physical_id": None, "validated_dx_m": None, "applied_dx_m": None}


class InferenceTests(unittest.TestCase):
    def test_feature_npz_has_only_allowed_arrays_no_gt(self):
        with np.load(npz(), allow_pickle=False) as data:
            hr, lr, valid, aff, lr_valid = m.feature_arrays(data)
        self.assertEqual(hr.shape, (3, 256, 256))
        self.assertEqual(lr.shape, (3, 64, 64))
        self.assertIsNone(lr_valid)
        with np.load(npz({"reference_roof": np.zeros((256, 256), np.uint8)}), allow_pickle=False) as data:
            with self.assertRaises(ValueError):
                m.feature_arrays(data)

    def test_gtfree_view_rejects_manual_or_supervision_even_if_unused(self):
        root = Path("synthetic_clean_view").resolve()
        cfg = {"records": [row()]}
        paths = [root / row()["native_image"], root / row()["feature_input"], root / "config/MANUAL_REFERENCE_GEOMETRY.json"]
        with patch.object(Path, "rglob", return_value=iter(paths)), patch.object(Path, "is_symlink", return_value=False), \
             patch.object(Path, "is_file", return_value=True), patch.object(Path, "stat", return_value=types.SimpleNamespace(st_size=1)), \
             patch.object(m.g, "sha", return_value="a"*64):
            with self.assertRaisesRegex(ValueError, "nonallowlisted"):
                m.verify_gtfree_view(root, cfg)

    def test_parent_joint_experiments_and_frozen_weight_hashes(self):
        files = {name: name for name in ("S_state.pth", "A_state.pth", "S_parent_state.pth", "A_parent_state.pth")}
        manifest = {"release_scope": "JOINT_RESEARCH_CANDIDATE_NOT_PRODUCTION_ADMITTED", "files": files,
                    "GT_required_for_inference": False, "frozen_before_FINAL_test": True}
        with patch.object(m.g, "read_json", return_value=manifest), patch.object(m.g, "sha", side_effect=lambda p: Path(p).name):
            plan, _ = m.release_plan(Path("release"), ["E0", "E1", "E2", "E3"])
            self.assertEqual(plan["E1"]["S"], "S_parent_state.pth")
            self.assertEqual(plan["E2"]["A"], "A_parent_state.pth")
            self.assertTrue(plan["E3"]["joint_training_executed"])
            manifest["release_scope"] = "S_AND_A_INDEPENDENT_TRAINED_PARENTS_NOT_JOINT_MODEL"
            plan, _ = m.release_plan(Path("release"), ["E1", "E3"])
            self.assertFalse(plan["E3"]["joint_training_executed"])
            self.assertIn("NOT_JOINT", plan["E3"]["scope"])
        with patch.object(m.g, "read_json", return_value=manifest), patch.object(m.g, "sha", return_value="WRONG"):
            with self.assertRaises(ValueError):
                m.release_plan(Path("release"), ["E1", "E3"])

    def test_native_affine_uses_source_window_offset_and_pixel_edges(self):
        r = row()
        r["source_window"][:2] = [100, 200]
        aff = m.native_affine(r)
        expected = Affine(*r["source_raster_affine"]) * (100, 200)
        self.assertEqual(aff * (0, 0), expected)
        geometry = m.pixel_geometry_to_metric(box(0, 0, 10, 10), aff, "EPSG:4326")
        self.assertTrue(geometry.is_valid)
        self.assertGreater(geometry.area, 0.)

    def test_components_cross512_seams_new_ids_and_strict_threshold(self):
        p = np.zeros((40, 1026), np.float32)
        p[10:20, 510:515] = .9
        p[30:32, 20:22] = .5  # strict>.5 excludes equality
        rows = m.native_components(p, row(), "b"*64)
        self.assertEqual(len(rows), 1)
        result = rows[0]
        self.assertEqual(shape(result["native_pixel_edge_geometry"]).area, 50.)
        self.assertIn("a"*64 + "_" + "b"*64, result["observation_id"])
        self.assertIsNone(result["physical_id"])
        self.assertFalse(result["legacy53_id_inherited"])
        self.assertIsNone(result["model_dx_m"])
        self.assertEqual(result["camera_parallax"], "UNKNOWN")

    def test_partial_empty_invalid_proxy_not_extrapolated(self):
        objects = [instance(box(20, 20, 30, 30)), instance(box(639, 10, 641, 20)),
                   instance(box(.01, .01, .02, .02)),
                   instance(Polygon([(0, 0), (2, 2), (0, 2), (2, 0)]))]
        before = copy.deepcopy([r["original_geometry"] for r in objects])
        masks, indices = m.instance_proxies(objects, box(0, 0, 640, 640), np.ones((256, 256), np.uint8),
                                           Affine(2.5, 0, 0, 0, -2.5, 640))
        self.assertEqual(indices, [0])
        self.assertEqual(masks.shape, (1, 256, 256))
        self.assertIn("INCOMPLETE", objects[1]["status"])
        self.assertIn("EMPTY", objects[2]["status"])
        self.assertIn("INVALID", objects[3]["status"])
        self.assertEqual(before, [r["original_geometry"] for r in objects])
        self.assertTrue(all(objects[i]["model_dx_m"] is None for i in (1, 2, 3)))

    def test_affine_linear_metres_sign_and_no_application_or_shape_change(self):
        objects = [instance(box(20, 20, 30, 30))]
        flags = {k: np.array([[False]]) for k in ("source_support_incomplete", "reference_support_missing", "support_overflow", "near_training_bound")}
        prediction = {**flags, "total_shift_px": np.array([[[1., 2.]]]), "unmatched_probability": np.array([[.4]])}
        original = copy.deepcopy(objects[0]["original_geometry"])
        m.apply_model_candidates(objects, [0], prediction, Affine(2.5, 0, 0, 0, -2.5, 640), box(0, 0, 640, 640))
        self.assertEqual((objects[0]["model_dx_m"], objects[0]["model_dy_m"]), (2.5, -5.))
        self.assertEqual(objects[0]["original_geometry"], original)
        self.assertEqual(shape(objects[0]["candidate_geometry"]).area, shape(original).area)
        self.assertIsNone(objects[0]["applied_dx_m"])
        self.assertIsNone(objects[0]["validated_dx_m"])
        self.assertTrue(objects[0]["status"].startswith("UNRESOLVED_"))
        objects = [instance(box(20, 20, 30, 30))]
        prediction["source_support_incomplete"][:] = True
        m.apply_model_candidates(objects, [0], prediction, Affine(2.5, 0, 0, 0, -2.5, 640), box(0, 0, 640, 640))
        self.assertIsNone(objects[0]["candidate_geometry"])
        self.assertIsNone(objects[0]["model_dx_m"])
        prediction["total_shift_px"][:] = np.nan
        m.apply_model_candidates(objects, [0], prediction, Affine(2.5, 0, 0, 0, -2.5, 640), box(0, 0, 640, 640))
        self.assertIsNone(objects[0]["raw_model_total_shift_px"])

    def test_nonoverlap512_stitching_replay_inputs_without_real_model(self):
        calls = []
        class Model(torch.nn.Module):
            def forward(self, image):
                calls.append(tuple(image.shape))
                return {"roof_logits": (image[:, :1]-.5)*10, "facade_logits": (image[:, 1:2]-.5)*10}
        api = types.SimpleNamespace(normalize_rgb=lambda image: image.float()/255)
        rgb = np.zeros((513, 513, 3), np.uint8)
        rgb[510:513, 510:513, 0] = 255
        roof, facade, replay = m.predict_native(Model(), api, rgb, torch.device("cpu"))
        self.assertEqual(calls, [(1, 3, 512, 512)]*4)
        self.assertEqual((roof > .5).sum(), 9)
        self.assertEqual((facade > .5).sum(), 0)
        self.assertEqual(replay["native_rgb512"].dtype, np.uint8)
        self.assertEqual(replay["native_rgb256"].shape, (1, 3, 256, 256))
        self.assertEqual(replay["expected_S_roof_logits"].shape, (1, 1, 512, 512))

    def test_frozen_aggregate_affine24_adapter_no_core_patch(self):
        api = m.g.transfer_api(Path("synthetic_payload"))
        grid = m.g.grid_with_affine(api, Affine(10, 0, 0, 0, -10, 10), 1, 1, "EPSG:32649")
        result = m.g.exact_fractions(api, grid, {"roof": box(0, 0, 4.3, 10)}, box(0, 0, 10, 10))
        self.assertAlmostEqual(result["fractions"][0, 0, 0], .43)

    def test_infer_orchestration_E1_E3_same_inputs_and_real_candidate_mapping(self):
        from PIL import Image
        from pyproj import Transformer
        root, release, output = map(lambda v: Path(v).resolve(), ("memory_view", "memory_release", "memory_inference"))
        r = row()
        r["source_window"] = [0, 0, 16, 16]
        r["source_raster_affine"] = [.0001, 0, 108.8, 0, -.0001, 34.3]
        cx, cy = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True).transform(108.8008, 34.2992)
        support = box(cx-320, cy-320, cx+320, cy+320)
        r["exact_support"] = mapping(support)
        aff = np.array([2.5, 0, cx-320, 0, -2.5, cy+320])
        feature = io.BytesIO()
        np.savez(feature, hr_rgb=np.zeros((3, 256, 256), np.uint8), lr_rgb=np.zeros((3, 64, 64), np.uint8),
                 source_support=np.ones((256, 256), np.uint8), affine=aff)
        feature_bytes = feature.getvalue()
        image = np.zeros((16, 16, 3), np.uint8)
        image[5:10, 5:10, 0] = 255
        class S(torch.nn.Module):
            def forward(self, x):
                return {"roof_logits": (x[:, :1]-.5)*10, "facade_logits": (x[:, 1:2]-.5)*10}
        class A(torch.nn.Module):
            def forward(self, hr, lr, masks, **kwargs):
                n = masks.shape[1]
                self_test.assertEqual(tuple(hr.shape), (1, 3, 256, 256))
                self_test.assertEqual(tuple(lr.shape), (1, 3, 64, 64))
                result = {key: torch.zeros((1, n), dtype=torch.bool) for key in (
                    "source_support_incomplete", "reference_support_missing", "support_overflow", "near_training_bound")}
                return {**result, "total_shift_px": torch.tensor([[[.5, 1.]]]).expand(1, n, 2),
                        "unmatched_probability": torch.full((1, n), .4)}
        self_test = self
        files = {"S_parent_state.pth": "b"*64, "A_parent_state.pth": "c"*64,
                 "S_state.pth": "b"*64, "A_state.pth": "c"*64}
        manifest = {"files": files, "frozen_before_FINAL_test": True, "GT_required_for_inference": False,
                    "release_scope": "S_AND_A_INDEPENDENT_TRAINED_PARENTS_NOT_JOINT_MODEL"}
        inputs = {r["feature_input"]: {"sha256": "d"*64}, r["native_image"]: {"sha256": "a"*64},
                  "weights/B0.pth": {"sha256": "f"*64}}
        writes, snapshots, tif_calls = {}, {}, []
        original_load = np.load
        def read_npz(_path, **kwargs):
            return original_load(io.BytesIO(feature_bytes), **kwargs)
        with ExitStack() as stack:
            stack.enter_context(patch.object(m, "load_native_config", return_value={"records": [r], "target_grid": m.g.TARGET_GRID}))
            stack.enter_context(patch.object(m, "verify_gtfree_view", return_value=inputs))
            stack.enter_context(patch.object(m, "_torch_modules", return_value=(
                types.SimpleNamespace(make_s_model=lambda *_: S()),
                types.SimpleNamespace(normalize_rgb=lambda x: x.float()/255),
                types.SimpleNamespace(load_alignment_checkpoint=lambda _: A()))))
            stack.enter_context(patch.object(m.g, "read_json", return_value=manifest))
            stack.enter_context(patch.object(m.g, "sha", side_effect=lambda p: files.get(Path(p).name, "frozen")))
            stack.enter_context(patch.object(m.g, "write_json", side_effect=lambda p, d: writes.__setitem__(str(p), copy.deepcopy(d))))
            stack.enter_context(patch.object(m.g, "verify_artifacts"))
            stack.enter_context(patch.object(m.g, "artifacts", return_value={}))
            stack.enter_context(patch.object(m.g, "write_tif", side_effect=lambda *args, **kwargs: tif_calls.append(args)))
            stack.enter_context(patch.object(Path, "mkdir"))
            stack.enter_context(patch.object(Image, "open", side_effect=lambda *_: Image.fromarray(image)))
            stack.enter_context(patch.object(np, "load", side_effect=read_npz))
            stack.enter_context(patch.object(np, "savez_compressed", side_effect=lambda p, **data: snapshots.__setitem__(str(p), data)))
            stack.enter_context(patch.object(torch, "load", return_value={}))
            result = m.infer(root, release, output, device="cpu")
            # Replay the serialized synthetic schema, not a real weight/image.
            result["artifacts"] = {"E3/" + name: {} for name in (
                "GT_FREE_REPLAY.json", "GT_FREE_REPLAY_INPUTS.npz", "GT_FREE_REPLAY_EXPECTED.npz")}
            def replay_read(path):
                if Path(path).name == "INFERENCE_MANIFEST.json":
                    return result
                return writes[str(output / "E3/GT_FREE_REPLAY.json")]
            def replay_npz(path, **kwargs):
                stream = io.BytesIO()
                np.savez(stream, **snapshots[str(path)])
                stream.seek(0)
                return original_load(stream, **kwargs)
            with patch.object(m.g, "read_json", side_effect=replay_read), patch.object(np, "load", side_effect=replay_npz):
                replayed = m.replay(root, release, output / "E3", Path("memory_replayed"), device="cpu")
                self.assertEqual(replayed["status"], "PASS")
                self.assertTrue(replayed["A_forward_executed"])
                self.assertEqual(set(replayed["comparisons"]), {"S_roof_logits", "S_facade_logits", "A_total_shift_px"})
                snapshots[str(output / "E3/GT_FREE_REPLAY_EXPECTED.npz")]["A_total_shift_px"] += .02
                failed = m.replay(root, release, output / "E3", Path("memory_replayed_fail"), device="cpu")
                self.assertEqual(failed["status"], "FAIL_NUMERICAL_REPLAY")
                self.assertFalse(failed["comparisons"]["A_total_shift_px"]["pass"])
        self.assertEqual(result["status"], "COMPLETE_GT_FREE_PREDICTIONS")
        self.assertFalse(result["GT_read"])
        self.assertEqual(len(tif_calls), 4)
        E1 = writes[str(output / "E1/instances.json")]["instances"][0]
        E3 = writes[str(output / "E3/instances.json")]["instances"][0]
        self.assertEqual(E1["original_geometry"], E3["original_geometry"])
        self.assertIsNone(E1["model_dx_m"])
        self.assertEqual((E3["model_dx_m"], E3["model_dy_m"]), (1.25, -2.5))
        self.assertIsNone(E3["applied_dx_m"])
        self.assertFalse(result["experiments"]["E3"]["joint_training_executed"])
        saved = snapshots[str(output / "E3/GT_FREE_REPLAY_INPUTS.npz")]
        self.assertEqual(saved["instance_masks"].shape, (1, 1, 256, 256))
        self.assertTrue(set(saved) <= {"native_rgb512", "native_rgb256", "hr_rgb", "lr_rgb", "instance_masks", "source_valid", "feature_affine", "lr_valid"})
        self.assertEqual(m.REPLAY_TOLERANCE, {"S_atol": .005, "S_rtol": .005,
                                            "A_shift_atol_px": .001, "A_shift_rtol": .001})


if __name__ == "__main__":
    unittest.main()
