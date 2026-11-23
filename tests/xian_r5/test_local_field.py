from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.local_field import (
    LOCAL_TRANSFORM_MODEL, _residuals_in_projected_units,
    candidate_cycle_diagnostics, estimate_local_tile_shift,
)


class LocalFieldTests(unittest.TestCase):
    def test_three_date_cycle_reports_weak_numeric_candidates_without_promotion(self):
        dates = ("2018-04-29", "2018-11-01", "2019-12-02")
        pairwise = {
            (dates[0], dates[1]): {"r00c00": {
                "status": "WEAK_LOCAL_CANDIDATE",
                "correction_dx_m_candidate": 1.0,
                "correction_dy_m_candidate": -7.0,
                "ground_mask_reviewed": False,
            }},
            (dates[1], dates[2]): {"r00c00": {
                "status": "WEAK_LOCAL_CANDIDATE",
                "correction_dx_m_candidate": -2.0,
                "correction_dy_m_candidate": 3.0,
                "ground_mask_reviewed": False,
            }},
            (dates[0], dates[2]): {"r00c00": {
                "status": "LOCAL_CANDIDATE_NOT_REVIEWED",
                "correction_dx_m_candidate": -1.0,
                "correction_dy_m_candidate": -4.0,
                "ground_mask_reviewed": False,
            }},
        }

        result = candidate_cycle_diagnostics(dates, pairwise)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cycle_closure_m_candidate"], [0.0, 0.0])
        self.assertEqual(result[0]["status"], "ARITHMETIC_ONLY_NOT_VALIDATION")
        self.assertFalse(result[0]["ground_position_validated"])
        self.assertFalse(result[0]["field_applied"])
        self.assertEqual(
            [item["status"] for item in result[0]["pairwise_evidence"]],
            ["WEAK_LOCAL_CANDIDATE", "WEAK_LOCAL_CANDIDATE",
             "LOCAL_CANDIDATE_NOT_REVIEWED"],
        )

    def test_cycle_diagnostics_require_three_dates_and_common_cells(self):
        self.assertEqual(candidate_cycle_diagnostics(("a", "b"), {}), [])
        dates = ("a", "b", "c")
        pairwise = {
            ("a", "b"): {"only_ab": {"correction_dx_m_candidate": 1,
                                       "correction_dy_m_candidate": 2}},
            ("b", "c"): {"only_bc": {"correction_dx_m_candidate": 3,
                                       "correction_dy_m_candidate": 4}},
            ("a", "c"): {"only_ac": {"correction_dx_m_candidate": 4,
                                       "correction_dy_m_candidate": 6}},
        }
        self.assertEqual(candidate_cycle_diagnostics(dates, pairwise), [])

    def test_similarity_candidate_is_local_unreviewed_and_not_applied(self):
        xx, yy = np.meshgrid(np.linspace(20, 236, 6), np.linspace(20, 236, 6))
        source = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
        target = source + np.array([3.25, -1.75], dtype=np.float32)
        target[:4] += np.array([80, -60], dtype=np.float32)
        mask = np.full((256, 256), 255, dtype=np.uint8)

        result = estimate_local_tile_shift(
            source, target, mask, mask,
            (0.25, 0.0, 500000.0, 0.0, -0.30, 3800000.0, 0.0, 0.0, 1.0),
        )

        self.assertEqual(result.status, "LOCAL_CANDIDATE_NOT_REVIEWED")
        self.assertGreaterEqual(result.inliers, 8)
        self.assertAlmostEqual(result.correction_dx_m_candidate, -0.8125, places=2)
        self.assertAlmostEqual(result.correction_dy_m_candidate, -0.525, places=2)
        self.assertFalse(result.ground_mask_reviewed)
        self.assertFalse(result.field_interpolated)
        self.assertFalse(result.applied_to_image_or_geometry)
        self.assertEqual(result.transform_model, LOCAL_TRANSFORM_MODEL)
        self.assertEqual(len(result.inlier_points_from_px), result.inliers)
        self.assertEqual(len(result.inlier_points_to_px), result.inliers)
        self.assertNotIn("inlier_points_from_px", result.as_dict())
        self.assertNotIn("inlier_points_to_px", result.as_dict())

        repeated = estimate_local_tile_shift(
            source, target, mask, mask,
            (0.25, 0.0, 500000.0, 0.0, -0.30, 3800000.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(result.ransac_seed, repeated.ransac_seed)
        self.assertEqual(result.as_dict(), repeated.as_dict())
        self.assertEqual(result.inlier_points_from_px, repeated.inlier_points_from_px)

    def test_projected_residual_uses_rotated_anisotropic_affine_basis(self):
        residuals = _residuals_in_projected_units(
            np.array([[2.0, 0.0], [0.0, 4.0]]),
            (0.0, -0.5, 100.0, 0.25, 0.0, 200.0),
        )

        np.testing.assert_allclose(residuals, [0.5, 2.0])

    def test_candidate_residual_gate_is_in_projected_metres(self):
        xx, yy = np.meshgrid(np.linspace(24, 232, 8), np.linspace(24, 232, 8))
        source = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
        noise = np.random.default_rng(417).uniform(-0.8, 0.8, size=source.shape).astype(np.float32)
        target = source + np.array([2.0, -1.0], dtype=np.float32) + noise
        mask = np.full((256, 256), 255, dtype=np.uint8)
        transform = (0.25, 0.0, 500000.0, 0.0, -0.25, 3800000.0)

        permissive = estimate_local_tile_shift(
            source, target, mask, mask, transform, max_p90_residual_m=2.0,
        )
        strict = estimate_local_tile_shift(
            source, target, mask, mask, transform, max_p90_residual_m=0.01,
        )

        self.assertEqual(permissive.status, "LOCAL_CANDIDATE_NOT_REVIEWED")
        self.assertEqual(strict.status, "WEAK_LOCAL_CANDIDATE")
        self.assertAlmostEqual(permissive.p90_residual_m,
                               0.25 * permissive.p90_residual_px, places=5)
        self.assertAlmostEqual(strict.p90_residual_m, permissive.p90_residual_m, places=8)
        self.assertFalse(permissive.ground_mask_reviewed)
        self.assertFalse(permissive.applied_to_image_or_geometry)

    def test_both_ends_of_match_must_be_in_candidate_ground_mask(self):
        source = np.array([[20, 20], [60, 20], [100, 20], [140, 20],
                           [20, 100], [60, 100], [100, 100], [140, 100]], dtype=np.float32)
        target = source + np.array([2, -1], dtype=np.float32)
        mask_from = np.full((256, 256), 255, dtype=np.uint8)
        mask_to = np.full((256, 256), 255, dtype=np.uint8)
        mask_to[:80, :] = 0

        result = estimate_local_tile_shift(
            source, target, mask_from, mask_to,
            (0.25, 0.0, 0.0, 0.0, -0.30, 0.0),
        )

        self.assertEqual(result.eligible_matches, 4)
        self.assertEqual(result.status, "INSUFFICIENT_GROUND_CANDIDATE_MATCHES")
        self.assertIsNone(result.correction_dx_m_candidate)
        self.assertEqual(result.inlier_points_from_px, ())
        self.assertEqual(result.inlier_points_to_px, ())

    def test_inadequate_spatial_support_is_weak_not_promoted(self):
        source = np.array([[20, 20], [60, 20], [100, 20], [140, 20],
                           [20, 100], [60, 100], [100, 100]], dtype=np.float32)
        target = source + np.array([1.5, 2.25], dtype=np.float32)
        mask = np.full((256, 256), 255, dtype=np.uint8)

        result = estimate_local_tile_shift(
            source, target, mask, mask,
            (0.25, 0.0, 0.0, 0.0, -0.30, 0.0),
        )

        self.assertEqual(result.status, "WEAK_LOCAL_CANDIDATE")
        self.assertFalse(result.ground_mask_reviewed)
        self.assertIsNotNone(result.correction_dx_m_candidate)


if __name__ == "__main__":
    unittest.main()
