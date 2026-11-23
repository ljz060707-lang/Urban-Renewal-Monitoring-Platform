"""Synthetic-only partition32/8 differential QA, no GT/models/cloud or disk data.

geo_python -B -m unittest discover -s tests -p test_xian_cloud_geo_partition_recovery.py
Also compatible with pytest. Timing is reported, not asserted across machines;
the deterministic performance gate asserts every original aggregate window<=8.
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
from affine import Affine
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("partition8_tested", ROOT / "tools/xian_cloud_geo_partition_recovery.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fast, cls.provenance = r.load_partition8(Path("synthetic_no_payload"))
        spec = importlib.util.spec_from_file_location("unmodified_frozen_geo_for_comparison", ROOT / "tools/xian_cloud_release_geo.py")
        cls.old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.old)
        cls.api = cls.old.transfer_api(Path("synthetic_no_payload"))

    def compare(self, grid, classes, support, name):
        start = time.perf_counter()
        old = self.old.exact_fractions(self.api, grid, classes, support)
        old_seconds = time.perf_counter()-start
        start = time.perf_counter()
        new = self.fast.exact_fractions(self.api, grid, classes, support)
        new_seconds = time.perf_counter()-start
        np.testing.assert_allclose(new["fractions"], old["fractions"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(new["valid"], old["valid"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(new["unknown"], old["unknown"], rtol=0, atol=1e-12)
        self.assertEqual(new["denominator_m2"], 100)
        self.assertEqual(old["denominator_m2"], 100)
        self.assertLess(old["area_conservation_error_m2"], 1e-5)
        self.assertLess(new["area_conservation_error_m2"], 1e-5)
        if len(classes) >= 2:
            # Fraction tolerance is NOT permission to change strict>.43 labels.
            before = self.old.candidate_layers(*old["fractions"][:2], old["valid"])
            after = self.old.candidate_layers(*new["fractions"][:2], new["valid"])
            for key in ("candidate_label", "formal_label", "quality", "weight"):
                np.testing.assert_array_equal(after[key], before[key])
            self.assertTrue((after["fraction"][2] == -9999).all())
            self.assertTrue((after["formal_label"] == 255).all())
            self.assertEqual(float(after["weight"].sum()), 0)
            self.assertEqual(int(after["quality"].sum()), 0)
        report = {"synthetic_case": name, "old_block32_seconds": old_seconds,
                  "new_block8_seconds": new_seconds,
                  "fraction_maxdiff": float(np.max(np.abs(new["fractions"]-old["fractions"]))),
                  "coverage_maxdiff": float(np.max(np.abs(new["valid"]-old["valid"]))),
                  "old_conservation_error_m2": old["area_conservation_error_m2"],
                  "new_conservation_error_m2": new["area_conservation_error_m2"]}
        print(json.dumps(report), flush=True)
        return old, new

    def test_frozen_hashes_explicit_helper_and_isolated_import(self):
        self.assertEqual(self.provenance["frozen_helper_sha256"], r.FROZEN_GEO_SHA)
        self.assertEqual(self.provenance["frozen_label_transfer_sha256"], r.FROZEN_TRANSFER_SHA)
        self.assertEqual(self.old.exact_fractions.__defaults__, (32,))
        self.assertEqual(self.fast.exact_fractions.__defaults__, (8,))
        other, _ = r.load_partition8(Path("synthetic_no_payload"), ROOT / "tools/xian_cloud_release_geo.py")
        self.assertIsNot(other, self.fast)
        self.assertEqual(self.old.exact_fractions.__defaults__, (32,))
        with patch.object(r, "sha", return_value="BAD_SHA"):
            with self.assertRaisesRegex(ValueError, "Geo helper SHA"):
                r.load_partition8(Path("synthetic_no_payload"))
        real_sha = r.sha
        with patch.object(r, "sha", side_effect=lambda p: "BAD_SHA" if Path(p).name == "label_transfer.py" else real_sha(p)):
            with self.assertRaisesRegex(ValueError, "label_transfer SHA"):
                r.load_partition8(Path("synthetic_no_payload"))
        self.assertNotIn("torch", sys.modules)

    def test_holes_multipolygons_overlap_union_and_partial_support(self):
        x, y = 308460., 3794750.
        grid = self.old.grid_with_affine(self.api, Affine(10, 0, x, 0, -10, y), 33, 17, "EPSG:32649")
        outer, hole = box(x+3.2, y-163.3, x+326.1, y-1.7), box(x+72.3, y-91.2, x+160.7, y-12.6)
        with_hole = Polygon(outer.exterior.coords, [hole.exterior.coords])
        raw = unary_union([with_hole, box(x+100.1, y-80.7, x+185.6, y-4.1),
                           box(x+328., y-169., x+329., y-165.)])
        candidate = unary_union([box(x+10.2, y-161.7, x+180.4, y-3.1),
                                 box(x+140.1, y-150.6, x+320.9, y-21.7)])
        support = box(x+1.1, y-166.7, x+322.8, y-2.2).difference(box(x+200.2, y-70.4, x+245.3, y-20.8))
        self.compare(grid, {"raw": raw, "candidate": candidate}, support, "holes_multipolygon_overlap_partial_UTM")

    def test_full100m2_denominator_nodata_and_strict_threshold(self):
        grid = self.old.grid_with_affine(self.api, Affine(10, 0, 0, 0, -10, 10), 17, 1, "EPSG:32649")
        duplicate_union = unary_union([box(0, 0, 4.3, 10), box(0, 0, 4.3, 10)])
        _, result = self.compare(grid, {"raw": duplicate_union, "candidate": duplicate_union}, box(0, 0, 5, 10), "partial50m2_not_denominator50")
        self.assertAlmostEqual(result["fractions"][0, 0, 0], .43)
        self.assertAlmostEqual(result["valid"][0, 0], .5)
        self.assertAlmostEqual(result["unknown"][0, 0], .5)
        layers = self.old.candidate_layers(*result["fractions"], result["valid"])
        self.assertEqual(layers["candidate_label"][0, 0], 0)
        self.assertTrue((layers["candidate_label"][0, 1:] == 255).all())
        self.assertTrue((layers["fraction"][:, 0, 1:] == -9999).all())

    def test_sheared_grid_exact100m2_and_empty_support(self):
        grid = self.old.grid_with_affine(self.api, Affine(10, 2, 0, 0, -10, 190), 17, 19, "EPSG:32649")
        self.compare(grid, {"raw": box(2.4, 3.2, 183.4, 182.6), "candidate": box(20.1, 1.8, 200.8, 172.3)},
                     box(5.3, 10.2, 180.7, 180.9), "sheared100m2")
        _, empty = self.compare(grid, {"raw": Polygon(), "candidate": Polygon()}, Polygon(), "empty_support")
        self.assertEqual(float(empty["fractions"].sum()), 0.)
        self.assertTrue((empty["unknown"] == 1).all())

    def test_invalid_geometry_rejected_without_repair(self):
        grid = self.old.grid_with_affine(self.api, Affine(10, 0, 0, 0, -10, 10), 1, 1, "EPSG:32649")
        invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        for module in (self.old, self.fast):
            with self.assertRaises(Exception):
                module.exact_fractions(self.api, grid, {"raw": invalid}, box(0, 0, 10, 10))

    def test_deterministic_performance_gate_bounds_original_aggregate_to8(self):
        grid = self.old.grid_with_affine(self.api, Affine(10, 0, 0, 0, -10, 320), 32, 32, "EPSG:32649")
        support = box(0, 0, 320, 320)
        with patch.object(self.api, "aggregate", wraps=self.api.aggregate) as called:
            self.old.exact_fractions(self.api, grid, {"raw": support}, support)
            self.assertEqual(called.call_count, 1)
            self.assertEqual(called.call_args.args[0].width, 32)
        with patch.object(self.api, "aggregate", wraps=self.api.aggregate) as called:
            self.fast.exact_fractions(self.api, grid, {"raw": support}, support)
            self.assertEqual(called.call_count, 16)
            self.assertTrue(all(c.args[0].width <= 8 and c.args[0].height <= 8 for c in called.call_args_list))
        with self.assertRaisesRegex(ValueError, "frozen to partition8"):
            self.fast.exact_fractions(self.api, grid, {"raw": support}, support, block=32)

    def test_dense4096_component_synthetic_performance_differential(self):
        grid = self.old.grid_with_affine(self.api, Affine(10, 0, 0, 0, -10, 320), 32, 32, "EPSG:32649")
        geometry = unary_union([box(x+.4, y+.3, x+3.7, y+3.6)
                                for x in range(0, 320, 5) for y in range(0, 320, 5)])
        self.compare(grid, {"raw": geometry, "candidate": geometry}, box(0, 0, 320, 320), "dense4096_equal_channels_no_dedup_optimization")

    def test_new_output_delegation_and_evidence_without_real_flow(self):
        geo, provenance = r.load_partition8(Path("synthetic_no_payload"))
        holder = {}
        def load(payload, frozen_helper=None, *, on_first_call=None):
            holder["callback"] = on_first_call
            return geo, provenance
        def delegate(payload, predictions, output, *, exports_only):
            self.assertTrue(exports_only)
            holder["callback"]()
            return {"status": "CANDIDATE_EXPORT_ONLY_GT_NOT_READ"}
        fake_file = io.StringIO()
        with patch.object(r, "load_partition8", side_effect=load), patch.object(geo, "geo", side_effect=delegate), \
             patch.object(geo, "compare_geo_exports", return_value={"status": "PASS"}) as compare, \
             patch.object(Path, "exists", return_value=False), patch.object(Path, "open") as opened, \
             contextlib.redirect_stdout(io.StringIO()):
            opened.return_value.__enter__.return_value = fake_file
            result = r.run_recovery("memory_payload", "memory_predictions", "memory_new_output", exports_only=True,
                                    expected_geo="memory_golden", frozen_helper="same_original_helper.py")
        self.assertEqual(result["export_replay"]["status"], "PASS")
        self.assertEqual(compare.call_count, 1)
        self.assertEqual(opened.call_args.args, ("x",))
        evidence = json.loads(fake_file.getvalue())
        self.assertEqual(evidence["recovery_fixed_block"], 8)
        self.assertTrue(evidence["exports_only"])
        self.assertFalse(evidence["reference_branch_entered_at_evidence_write"])
        self.assertFalse(evidence["scientific_validation"])
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "must be NEW"):
                r.run_recovery("memory_payload", "memory_predictions", "existing_output")


if __name__ == "__main__":
    unittest.main()
