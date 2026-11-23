from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.pose_graph import (TranslationEdge,
                                          compose_moving_to_anchor_corrections,
                                          solve_translation_pose_graph)


class PoseGraphTests(unittest.TestCase):
    def test_moving_to_anchor_corrections_compose_additively(self):
        self.assertEqual(
            compose_moving_to_anchor_corrections((1.5, -2.0), (4.0, 3.0)),
            (5.5, 1.0),
        )

    def test_composes_pairwise_moving_to_reference_corrections(self):
        # Anchor pose is zero; A is (3,-4), B is (7,2). Each edge is
        # correction(moving -> reference) = pose(moving) - pose(reference).
        edges = [
            TranslationEdge("R", "A", 3, -4, "R<-A"),
            TranslationEdge("R", "B", 7, 2, "R<-B"),
            TranslationEdge("A", "B", 4, 6, "A<-B"),
        ]

        result = solve_translation_pose_graph(edges, "R")

        self.assertEqual(result.status, "OVERDETERMINED_CYCLE_DIAGNOSTIC_CANDIDATE")
        self.assertEqual(result.offsets_m["R"], (0.0, 0.0))
        self.assertAlmostEqual(result.offsets_m["A"][0], 3.0)
        self.assertAlmostEqual(result.offsets_m["A"][1], -4.0)
        self.assertAlmostEqual(result.offsets_m["B"][0], 7.0)
        self.assertAlmostEqual(result.offsets_m["B"][1], 2.0)
        self.assertEqual(result.cycle_degrees_of_freedom, 1)
        self.assertAlmostEqual(result.closure_max_m, 0.0, places=10)

    def test_outlier_is_exposed_as_cycle_residual_not_silently_dropped(self):
        edges = [
            TranslationEdge("R", "A", 3, -4, "R<-A"),
            TranslationEdge("R", "B", 7, 2, "R<-B"),
            TranslationEdge("A", "B", 4, 6, "A<-B"),
            TranslationEdge("R", "B", 30, 2, "R<-B-outlier"),
        ]

        result = solve_translation_pose_graph(edges, "R")

        self.assertEqual(len(result.edge_residuals_m), 4)
        self.assertGreater(result.closure_max_m, 1.0)
        self.assertEqual(result.cycle_degrees_of_freedom, 2)

    def test_disconnected_nodes_and_anchor_only_remain_unresolved(self):
        result = solve_translation_pose_graph([
            TranslationEdge("A", "B", 1, 2, "A<-B"),
            TranslationEdge("X", "Y", 4, 5, "X<-Y"),
        ], "A")

        self.assertEqual(result.connected_ids, ("A", "B"))
        self.assertNotIn("X", result.offsets_m)
        self.assertEqual(result.status, "CONNECTED_TREE_NO_CYCLE_CHECK")

        isolated = solve_translation_pose_graph([], "R")
        self.assertEqual(isolated.status, "ANCHOR_ONLY_NO_CONNECTED_TIES")
        self.assertEqual(isolated.offsets_m, {"R": (0.0, 0.0)})

    def test_invalid_edge_and_empty_anchor_are_rejected(self):
        with self.assertRaises(ValueError):
            TranslationEdge("R", "A", float("nan"), 1, "bad")
        with self.assertRaises(ValueError):
            solve_translation_pose_graph([], "")


if __name__ == "__main__":
    unittest.main()
