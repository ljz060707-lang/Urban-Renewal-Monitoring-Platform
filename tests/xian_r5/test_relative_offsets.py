"""Contract tests for spatially binned relative-shift candidates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment import (solve_local_relative_pose_graphs,
                               summarize_local_relative_shifts)


class LocalRelativeShiftTests(unittest.TestCase):
    def test_separates_local_vectors_and_keeps_ground_unresolved(self):
        samples = [
            {"sample_id": f"west-{index}", "x_m": 10 + index,
             "y_m": 90 - index, "dx_m": 8.0 + index * 0.1,
             "dy_m": -4.0 - index * 0.1}
            for index in range(5)
        ] + [
            {"sample_id": f"east-{index}", "x_m": 60 + index,
             "y_m": 90 - index, "dx_m": -2.0 + index * 0.1,
             "dy_m": 3.0 + index * 0.1}
            for index in range(5)
        ]

        result = summarize_local_relative_shifts(
            samples, (0, 0, 100, 100), grid_shape=(1, 2), minimum_support=5)

        self.assertEqual(result["sample_count"], 10)
        self.assertEqual(result["numeric_candidate_cell_count"], 2)
        west, east = result["cells"]
        self.assertAlmostEqual(west["median_dx_m"], 8.2)
        self.assertAlmostEqual(west["median_dy_m"], -4.2)
        self.assertAlmostEqual(east["median_dx_m"], -1.8)
        self.assertAlmostEqual(east["median_dy_m"], 3.2)
        self.assertFalse(result["absolute_ground_position_resolved"])
        self.assertFalse(result["applied"])
        self.assertTrue(all(not cell["absolute_ground_position_resolved"]
                            and not cell["applied"] for cell in result["cells"]))

    def test_sparse_cells_are_explicit_and_not_interpolated(self):
        result = summarize_local_relative_shifts(
            [{"sample_id": "one", "x_m": 10, "y_m": 90, "dx_m": 2, "dy_m": -1}],
            (0, 0, 100, 100), grid_shape=(2, 2), minimum_support=2)

        self.assertEqual(result["numeric_candidate_cell_count"], 0)
        self.assertFalse(result["interpolated"])
        self.assertEqual(result["cells"][0]["status"],
                         "INSUFFICIENT_RELATIVE_SHIFT_SUPPORT")
        self.assertIsNone(result["cells"][0]["median_dx_m"])
        self.assertEqual(sum(cell["sample_count"] for cell in result["cells"]), 1)

    def test_rejects_duplicate_ids_and_out_of_bounds_samples(self):
        duplicate = [
            {"sample_id": "same", "x_m": 10, "y_m": 90, "dx_m": 2, "dy_m": -1},
            {"sample_id": "same", "x_m": 11, "y_m": 89, "dx_m": 2, "dy_m": -1},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            summarize_local_relative_shifts(duplicate, (0, 0, 100, 100))
        with self.assertRaisesRegex(ValueError, "outside bounds"):
            summarize_local_relative_shifts(
                [{"sample_id": "outside", "x_m": 101, "y_m": 90,
                  "dx_m": 2, "dy_m": -1}], (0, 0, 100, 100))

    def test_pose_graph_composes_three_dates_and_reports_cycle_closure(self):
        pairs = [
            ("2018-04-29", "2018-11-01", (4.0, 3.0)),
            ("2018-11-01", "2019-12-02", (-1.0, 2.0)),
            ("2018-04-29", "2019-12-02", (3.0, 5.0)),
        ]
        samples = [
            {"sample_id": f"building-{index}", "reference_id": reference,
             "moving_id": moving, "x_m": 10 + index, "y_m": 90 - index,
             "dx_m": motion[0], "dy_m": motion[1]}
            for reference, moving, motion in pairs
            for index in range(5)
        ]

        result = solve_local_relative_pose_graphs(
            samples, (0, 0, 100, 100), anchor_id="2018-04-29",
            grid_shape=(1, 1), minimum_support=5)

        cell = result["cells"][0]
        self.assertEqual(cell["status"], "RELATIVE_POSE_CYCLE_DIAGNOSTIC_CANDIDATE")
        self.assertEqual(cell["cycle_degrees_of_freedom"], 1)
        self.assertAlmostEqual(
            cell["candidate_corrections_to_anchor_m"]["2018-11-01"][0], -4.0)
        self.assertAlmostEqual(
            cell["candidate_corrections_to_anchor_m"]["2019-12-02"][1], -5.0)
        self.assertAlmostEqual(cell["closure_max_m"], 0.0, places=9)
        self.assertFalse(result["absolute_ground_position_resolved"])
        self.assertFalse(result["applied"])

    def test_pose_graph_exposes_inconsistent_cycle_without_accepting_it(self):
        pairs = [
            ("a", "b", (4.0, 3.0)),
            ("b", "c", (-1.0, 2.0)),
            ("a", "c", (3.0, 8.0)),
        ]
        samples = [
            {"sample_id": f"building-{index}", "reference_id": reference,
             "moving_id": moving, "x_m": 10 + index, "y_m": 90 - index,
             "dx_m": motion[0], "dy_m": motion[1]}
            for reference, moving, motion in pairs
            for index in range(5)
        ]

        result = solve_local_relative_pose_graphs(
            samples, (0, 0, 100, 100), anchor_id="a",
            grid_shape=(1, 1), minimum_support=5)

        cell = result["cells"][0]
        self.assertEqual(cell["status"], "RELATIVE_POSE_CYCLE_DIAGNOSTIC_CANDIDATE")
        self.assertGreater(cell["closure_max_m"], 0.5)
        self.assertFalse(cell["applied"])
        self.assertFalse(cell["absolute_ground_position_resolved"])

    def test_pose_graph_leaves_under_supported_cells_disconnected(self):
        samples = [
            {"sample_id": f"building-{index}", "reference_id": "a",
             "moving_id": "b", "x_m": 10 + index, "y_m": 90 - index,
             "dx_m": 2.0, "dy_m": -1.0}
            for index in range(4)
        ]

        result = solve_local_relative_pose_graphs(
            samples, (0, 0, 100, 100), anchor_id="a",
            grid_shape=(1, 1), minimum_support=5)

        cell = result["cells"][0]
        self.assertEqual(cell["status"], "INSUFFICIENT_RELATIVE_SUPPORT")
        self.assertEqual(cell["connected_actual_dates"], ["a"])
        self.assertNotIn("b", cell["candidate_corrections_to_anchor_m"])


if __name__ == "__main__":
    unittest.main()
