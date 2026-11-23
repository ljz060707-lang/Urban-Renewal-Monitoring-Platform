"""Synthetic geometry/GTiff/GPKG QA; use geo env, no torch or real GT reads.

All binary artifact tests use GDAL /vsimem. No disk dataset/weight outputs.
Run: geo_python -B -m unittest discover -s tests -p test_xian_cloud_release_geo.py
"""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
from affine import Affine
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

_path = Path(__file__).resolve().parents[1] / "tools/xian_cloud_release_geo.py"
_spec = importlib.util.spec_from_file_location("release_geo_tested", _path)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def instance(geometry=None):
    return {"observation_id": "OBS_" + "a"*64 + "_" + "b"*64 + "_C00000001",
            "original_geometry": mapping(geometry if geometry is not None else box(1, 1, 4, 4)),
            "candidate_geometry": mapping(box(2, 2, 5, 5)), "physical_id": None,
            "status": "UNRESOLVED_MODEL_CANDIDATE_NOT_ADMITTED", "source_id": "SYNTHETIC",
            "source_date": "2018-11-01", "rgb_sha256": "a"*64, "S_sha256": "b"*64,
            "A_sha256": "c"*64, "model_dx_m": 1., "model_dy_m": 1.}


class GeoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = g.transfer_api(Path("synthetic_payload"))

    def test_torch_free_import_and_confined_paths(self):
        self.assertNotIn("torch", sys.modules)
        for value in ("../private", "/absolute", "C:/GT", "data\\labels"):
            with self.assertRaises(ValueError):
                g.path_in(Path("memory"), value)

    def test_exact_frozen_union_and_block_equivalence(self):
        grid = self.api.Grid(Affine(10, 0, 0, 0, -10, 20), 2, 2, "EPSG:32649")
        valid = box(0, 0, 20, 20)
        merged = unary_union([box(0, 0, 10, 10), box(5, 0, 15, 10)])
        classes = {"raw": merged, "candidate": box(0, 0, 4.3, 10)}
        a = g.exact_fractions(self.api, grid, classes, valid, block=1)
        b = self.api.aggregate(grid, classes, valid, events=True)
        np.testing.assert_allclose(a["fractions"], b["fractions"], atol=1e-12, rtol=0)
        self.assertEqual(a["denominator_m2"], 100)
        self.assertAlmostEqual(a["fractions"][0].sum()*100, 150.)  # not200 summed overlap
        self.assertAlmostEqual(a["fractions"][1].sum()*100, 43.)
        self.assertLess(a["area_conservation_error_m2"], 1e-8)

    def test_unknown_qualification_and_strict_candidate_threshold(self):
        candidate = np.array([[.43, .430001], [0., 1.]])
        valid = np.array([[1., 1.], [0., 1.]])
        layers = g.candidate_layers(candidate, candidate, valid)
        self.assertEqual(layers["candidate_label"].tolist(), [[0, 1], [255, 1]])
        self.assertTrue((layers["formal_label"] == 255).all())
        self.assertTrue((layers["fraction"][2] == -9999).all())
        self.assertEqual(layers["fraction"][0, 1, 0], -9999)
        self.assertEqual(layers["weight"].sum(), 0.)
        self.assertEqual(layers["quality"].sum(), 0)

    def test_invalid_geometry_retained_not_buffer_repaired(self):
        invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        row = instance(invalid)
        before = copy.deepcopy(row)
        raw, candidate, risks = g.geometry_sets([row])
        self.assertTrue(raw.is_empty and candidate.is_empty)
        self.assertEqual(len(risks), 1)
        self.assertEqual(row, before)
        row = instance(box(-1, 1, 4, 4))
        row["source_exact_support"] = mapping(box(0, 0, 10, 10))
        raw, _, _ = g.geometry_sets([row])
        self.assertEqual(raw.area, 12.)  # integral uses observed fragment only
        self.assertEqual(row["original_geometry"], mapping(box(-1, 1, 4, 4)))

    def test_reference_polygons_original_fid_not_layer_fid_or_raster_pool(self):
        polygon = box(0, 0, 4.3, 10)
        document = {"records": [{"GT_original_FID": "ORIGINAL_1042", "derived_gpkg_FID": "8",
            "geometry": mapping(polygon), "geometry_sha256": g.sha_bytes(polygon.wkb),
            "year": 2019, "crs": "EPSG:32649", "actual_date": None}]}
        rows = g.manual_polygons(document)
        self.assertEqual(rows[0][:2], ("ORIGINAL_1042", "8"))
        grid = self.api.Grid(Affine(10, 0, 0, 0, -10, 10), 1, 1, "EPSG:32649")
        ref = g.exact_fractions(self.api, grid, {"reference": rows[0][2]}, box(0, 0, 10, 10))
        self.assertAlmostEqual(ref["fractions"][0, 0, 0], .43)  # no2.5m raster rounding
        metric = g.reference_metrics(np.array([[.5]]), ref["fractions"][0], ref["valid"])
        self.assertAlmostEqual(metric["reference-polygon-set_MAE"], .07)
        self.assertIn("BACKGROUND_COMPLETENESS_NOT_CERTIFIED", metric["limitations"])
        self.assertIsNone(metric["physical_instance_metrics"])
        bad = copy.deepcopy(document)
        bad["records"][0]["geometry_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            g.manual_polygons(bad)

    def test_reference_summary_pixel_denominator_weighting(self):
        def metric(n, mean):
            return {"reference_positive_cells": n, "full_roi_cells": n,
                    "reference-polygon-set_MAE": mean,
                    "full_roi_fraction_MAE_DIAGNOSTIC_UNVERIFIED_BACKGROUND": mean}
        rows = [{"split": "FINAL_TEST", "experiments": {"E3": {"raw": metric(1, .1)}}},
                {"split": "FINAL_TEST", "experiments": {"E3": {"raw": metric(3, .3)}}}]
        result = g.reference_summary(rows)["FINAL_TEST"]["E3"]["raw"]
        self.assertAlmostEqual(result["reference-polygon-set_MAE"], .25)
        self.assertEqual(result["reference_positive_cells"], 4)

    def test_actual_geotiff_in_memory_reopen_grid_nodata_bands(self):
        from rasterio.io import MemoryFile
        grid = self.api.Grid(Affine(10, 0, 308460, 0, -10, 3794750), 2, 2, "EPSG:32649")
        values = np.stack([np.full((2, 2), .43), np.full((2, 2), .5), np.full((2, 2), -9999)]).astype("float32")
        with MemoryFile() as mem:
            g.write_tif(mem.name, values, grid, ["raw", "candidate", "qualified"], -9999)
            with mem.open() as src:
                np.testing.assert_array_equal(src.read(), values)
                self.assertEqual(src.transform, grid.affine)
                self.assertEqual(str(src.crs), "EPSG:32649")
                self.assertEqual(src.nodata, -9999)
                self.assertEqual(src.descriptions, ("raw", "candidate", "qualified"))

    def test_actual_gpkg_in_memory_both_geometry_layers_and_null_ids(self):
        import fiona
        from fiona.io import MemoryFile
        row = instance()
        with MemoryFile(ext=".gpkg") as mem:
            errors = g.write_vectors(mem.name, [row])
            self.assertEqual(errors, [])
            self.assertEqual(set(fiona.listlayers(mem.name)), {"original_geometry", "model_candidate_geometry"})
            with fiona.open(mem.name, layer="model_candidate_geometry") as src:
                result = next(iter(src))
                self.assertIsNone(result.properties["physical_id"])
                self.assertIsNone(result.properties["validated_dx_m"])
                self.assertIsNone(result.properties["applied_dx_m"])
                self.assertEqual(result.properties["model_dx_m"], 1.)
                self.assertEqual(result.properties["observation_id"], row["observation_id"])
                provenance = json.loads(result.properties["provenance"])
                self.assertEqual(provenance["status"], "UNRESOLVED_MODEL_CANDIDATE_NOT_ADMITTED")

    def test_fixed_submeter_profile_and_bounded_windows(self):
        writes = []
        class Dataset:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def set_band_description(self, *_):
                pass
            def update_tags(self, **_):
                pass
            def write(self, values, band, window):
                writes.append((values.shape, window.width, window.height, int(values.min())))
        with patch("rasterio.open", return_value=Dataset()) as opened:
            g.write_submeter("memory_only.tif", [], Polygon())
        profile = opened.call_args.kwargs
        self.assertEqual((profile["width"], profile["height"]), (10240, 10240))
        self.assertEqual(profile["transform"], Affine(.25, 0, 308460, 0, -.25, 3794750))
        self.assertEqual(len(writes), 1600)
        self.assertTrue(all(row == ((256, 256), 256, 256, 255) for row in writes))

    def test_geo_orchestration_manual_file_once_no_supervision_or_raster_gt(self):
        payload, predictions, output = map(lambda v: Path(v).resolve(), ("memory_payload", "memory_predictions", "memory_geo_output"))
        sources, refs, manual, objects = [], [], [], []
        for index in range(4):
            split = "CALIBRATION" if index < 2 else "FINAL_TEST"
            left, top = 308460 + index*20, 3794750
            support = box(left, top-20, left+20, top)
            geometry = box(left+1, top-5, left+5, top-1)
            identifier = "memory_roi" + str(index)
            feature = f"data/xian/{split}/{identifier}_input.npz"
            sources.append({"region_id": identifier, "feature_input": feature, "exact_support": mapping(support),
                            "feature_affine": [.078125, 0, left, 0, -.078125, top]})
            refs.append({"id": identifier, "split": split, "input": feature, "support": mapping(support),
                         "reference_GT_year": 2019, "source_actual_date": "2018-11-01",
                         "training_supervision": "THIS_MUST_NEVER_BE_OPENED_supervision.npz"})
            manual.append({"GT_original_FID": str(1000+index), "derived_gpkg_FID": str(index),
                "geometry": mapping(geometry), "geometry_sha256": g.sha_bytes(geometry.wkb),
                "year": 2019, "crs": "EPSG:32649", "actual_date": None})
            obj = instance(geometry)
            obj.update(source_id=identifier, source_exact_support=mapping(support), candidate_geometry=mapping(geometry))
            objects.append(obj)
        frozen = {"status": "COMPLETE_GT_FREE_PREDICTIONS", "target_grid": g.TARGET_GRID, "artifacts": {},
                  "experiments": {e: {"instances": e + "/instances.json"} for e in ("E1", "E3")}}
        writes, opens = {}, []
        def read(path):
            if str(path).endswith("INFERENCE_MANIFEST.json"):
                return frozen
            if str(path).endswith("instances.json"):
                return {"sources": sources, "instances": objects}
            if str(path).endswith("alignment_data.json"):
                return {"records": refs}
            self.fail("unexpected metadata read: " + str(path))
        def opened(path, *_, **__):
            self.assertEqual(str(path), str(payload / "config/MANUAL_REFERENCE_GEOMETRY.json"))
            opens.append(str(path))
            return io.BytesIO(json.dumps({"records": manual}).encode())
        with ExitStack() as stack:
            stack.enter_context(patch.object(g, "read_json", side_effect=read))
            stack.enter_context(patch.object(g, "write_json", side_effect=lambda p, d: writes.__setitem__(str(p), copy.deepcopy(d))))
            stack.enter_context(patch.object(g, "sha", return_value="frozen_sha"))
            stack.enter_context(patch.object(g, "verify_artifacts"))
            stack.enter_context(patch.object(g, "transfer_api", return_value=self.api))
            stack.enter_context(patch.object(g, "artifacts", return_value={}))
            for name in ("write_tif", "write_submeter"):
                stack.enter_context(patch.object(g, name))
            stack.enter_context(patch.object(g, "write_vectors", return_value=[]))
            stack.enter_context(patch.object(Path, "mkdir"))
            stack.enter_context(patch.object(Path, "open", opened))
            result = g.geo(payload, predictions, output)
            opens_before = len(opens)
            def export_read(path):
                if "alignment_data" in str(path) or "MANUAL_REFERENCE" in str(path):
                    self.fail("exports-only must not access reference metadata or GT")
                return read(path)
            with patch.object(g, "read_json", side_effect=export_read):
                exported = g.geo(payload, predictions, Path("memory_export_only").resolve(), exports_only=True)
            self.assertEqual(len(opens), opens_before)
            self.assertEqual(exported["status"], "CANDIDATE_EXPORT_ONLY_GT_NOT_READ")
            self.assertEqual(exported["reference_evaluation"], "NOT_RUN_EXPORTS_ONLY")
        self.assertEqual(len(opens), 1)
        self.assertEqual(result["public_FINAL57"], "NOT_RUN")
        self.assertEqual(result["qualified_pixel_count"], 0)
        evaluation = writes[str(output / "REFERENCE_ONLY_EVALUATION.json")]
        self.assertEqual(evaluation["reference_method"], "EXACT_POLYGON_UNION_CELL_INTERSECTION")
        self.assertEqual(evaluation["reference_polygon_count"], 4)
        self.assertEqual(len(evaluation["records"]), 4)
        self.assertEqual(evaluation["records"][0]["GT_original_FIDs"], ["1000"])
        self.assertFalse(evaluation["records"][0]["NPZ_reference_FIDs_used"])
        self.assertEqual(evaluation["by_role_experiment"]["FINAL_TEST"]["E3"]["raw"]["reference-polygon-set_MAE"], 0.)

    def test_geo_export_comparison_real_in_memory_files_and_changed_value(self):
        import rasterio
        import fiona
        from rasterio.io import MemoryFile as RasterMemory
        from fiona.io import MemoryFile as VectorMemory
        expected, actual = Path("golden_exports"), Path("replayed_exports")
        names = ("BUILDING_FRACTION_10M.tif", "BUILDING_ALIGNED_QUALITY.tif", "SUPERVISION_WEIGHT_10M.tif",
                 "BUILDING_LABEL_10M.tif", "CANDIDATE_LABEL_10M_AT_0P43.tif", "SOURCE_VALID_FRACTION_10M.tif",
                 "BUILDING_ALIGNED_SUBMETER.tif")
        manifest = {"experiments": {"E3": {}}, "artifacts": {
            "E3/" + name: {"sha256": "synthetic", "bytes": 0}
            for name in (*names, "BUILDING_ALIGNMENT_VECTORS.gpkg")}}
        grid = self.api.Grid(Affine(10, 0, 308460, 0, -10, 3794750), 2, 2, "EPSG:32649")
        with RasterMemory() as raster_a, RasterMemory() as raster_b, VectorMemory(ext=".gpkg") as vector:
            for mem in (raster_a, raster_b):
                g.write_tif(mem.name, np.full((2, 2), .43, np.float32), grid, ["MODEL_ONLY"], -9999)
            self.assertEqual(g.write_vectors(vector.name, [instance()]), [])
            raster_open, vector_open = rasterio.open, fiona.open
            def open_raster(path, *args, **kwargs):
                mem = raster_a if expected in Path(path).parents else raster_b
                return raster_open(mem.name, *args, **kwargs)
            def open_vector(path, *args, **kwargs):
                return vector_open(vector.name, *args, **kwargs)
            with patch.object(g, "read_json", return_value=manifest), patch.object(g, "write_json"), \
                 patch.object(g, "verify_artifacts") as verified, \
                 patch("rasterio.open", side_effect=open_raster), patch("fiona.open", side_effect=open_vector):
                result = g.compare_geo_exports(expected, actual)
                self.assertEqual(result["status"], "PASS")
                self.assertFalse(result["GT_read"])
                self.assertEqual(verified.call_count, 8)
                self.assertFalse(result["regenerated_file_byte_equality_claimed"])
                with raster_open(raster_b.name, "r+") as changed:
                    changed.write(np.full((1, 2, 2), .44, np.float32))
                result = g.compare_geo_exports(expected, actual)
                self.assertEqual(result["status"], "FAIL_GEO_REPLAY")
                self.assertGreater(result["experiments"]["E3"][names[0]]["maximum_absolute_value_difference"], .009)


if __name__ == "__main__":
    unittest.main()
