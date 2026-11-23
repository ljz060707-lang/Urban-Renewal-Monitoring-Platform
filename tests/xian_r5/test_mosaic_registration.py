from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from xian_r5.reconstruction.registration import (
    estimate_mosaic_phase_registration,
    estimate_mosaic_registration,
)
from tools.run_xian_mosaic_registration_diagnostic import (
    _group_actual_date_rows,
    _resolve_duplicate_date_sources,
)


def _textured_rgb(seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.55)
    for _ in range(45):
        center = tuple(int(v) for v in rng.integers(12, 244, size=2))
        radius = int(rng.integers(3, 12))
        color = int(rng.integers(40, 240))
        cv2.circle(gray, center, radius, color, 1, cv2.LINE_AA)
    return np.repeat(gray[:, :, None], 3, axis=2)


class MosaicRegistrationTests(unittest.TestCase):
    def test_real_date_grouping_splits_mixed_epoch_and_collapses_duplicate_date(self) -> None:
        rows = [
            {"epoch_label": "2018", "actual_date": "2018-04-29", "tile_x": "10", "tile_y": "20"},
            {"epoch_label": "2018", "actual_date": "2018-11-01", "tile_x": "11", "tile_y": "20"},
            {"epoch_label": "2019", "actual_date": "2018-11-01", "tile_x": "12", "tile_y": "20"},
            {"epoch_label": "2020", "actual_date": "DATE_UNKNOWN", "tile_x": "13", "tile_y": "20"},
        ]

        epochs_by_date, centers_by_date, unresolved = _group_actual_date_rows(rows)

        self.assertEqual(epochs_by_date, {
            "2018-04-29": ["2018"],
            "2018-11-01": ["2018", "2019"],
        })
        self.assertEqual(centers_by_date["2018-04-29"], {(10, 20)})
        self.assertEqual(centers_by_date["2018-11-01"], {(11, 20), (12, 20)})
        self.assertEqual(unresolved, 1)

    def test_diagnostic_mosaic_skips_conflicting_duplicate_source_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            same_a, same_b = root / "same-a.jpg", root / "same-b.jpg"
            conflict_a, conflict_b = root / "conflict-a.jpg", root / "conflict-b.jpg"
            same_a.write_bytes(b"identical")
            same_b.write_bytes(b"identical")
            conflict_a.write_bytes(b"first")
            conflict_b.write_bytes(b"second")

            selected, collapsed = _resolve_duplicate_date_sources([("2019", same_b), ("2018", same_a)])
            conflict, rejected = _resolve_duplicate_date_sources([("2018", conflict_a), ("2019", conflict_b)])

        self.assertEqual(selected[0], "2018")
        self.assertEqual(collapsed["status"], "DUPLICATE_ACTUAL_DATE_IDENTICAL_SOURCE_COLLAPSED")
        self.assertIsNone(conflict)
        self.assertEqual(rejected["status"], "DUPLICATE_ACTUAL_DATE_SOURCE_CONFLICT_SKIPPED")

    def test_fractional_translation_is_reported_as_candidate_without_application(self) -> None:
        source = _textured_rgb()
        moved = cv2.warpAffine(
            source,
            np.float32([[1, 0, 2.5], [0, 1, -1.25]]),
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = estimate_mosaic_registration(
            source,
            moved,
            np.ones((256, 256), dtype=np.uint8),
            from_date="2018-04-29",
            to_date="2018-11-01",
            pixel_to_projected=(0.5, 0.0, 0.0, -0.5),
        )

        self.assertEqual(result.status, "GROUND_MASKED_CANDIDATE_NOT_REVIEWED")
        self.assertAlmostEqual(result.translation_dx_m, -1.25, delta=0.2)
        self.assertAlmostEqual(result.translation_dy_m, -0.625, delta=0.2)
        self.assertFalse(result.ground_mask_reviewed)
        self.assertFalse(result.applied_to_image_or_geometry)
        self.assertEqual(set(result.models), {"translation", "similarity", "affine", "homography"})
        self.assertGreaterEqual(result.models["translation"]["n_inliers"], 12)

        point_export = estimate_mosaic_registration(
            source,
            moved,
            np.ones((256, 256), dtype=np.uint8),
            from_date="2018-04-29",
            to_date="2018-11-01",
            pixel_to_projected=(0.5, 0.0, 0.0, -0.5),
            include_match_points=True,
        )
        tie_points = point_export.models["translation"]["inlier_correspondences_px"]
        self.assertEqual(len(tie_points), point_export.models["translation"]["n_inliers"])
        self.assertTrue(all(len(row) == 4 for row in tie_points))

    def test_identical_actual_dates_are_not_treated_as_two_epochs(self) -> None:
        image = _textured_rgb()

        result = estimate_mosaic_registration(
            image,
            image.copy(),
            np.ones((256, 256), dtype=np.uint8),
            from_date="2018-11-01",
            to_date="2018-11-01",
            pixel_to_projected=(1.0, 0.0, 0.0, -1.0),
        )

        self.assertEqual(result.status, "DUPLICATE_EPOCH")
        self.assertIsNone(result.translation_dx_m)
        self.assertFalse(result.applied_to_image_or_geometry)

    def test_orb_reports_translation_candidate_without_application(self) -> None:
        source = _textured_rgb(seed=83)
        moved = cv2.warpAffine(
            source,
            np.float32([[1, 0, 2.5], [0, 1, -1.25]]),
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = estimate_mosaic_registration(
            source,
            moved,
            np.ones((256, 256), dtype=np.uint8),
            from_date="2018-04-29",
            to_date="2018-11-01",
            pixel_to_projected=(0.5, 0.0, 0.0, -0.5),
            ratio_test=0.85,
            feature_method="ORB",
        )

        self.assertIn(result.status, {"GROUND_MASKED_CANDIDATE_NOT_REVIEWED",
                                      "WEAK_GROUND_MASKED_MATCH_CANDIDATE"})
        self.assertFalse(result.applied_to_image_or_geometry)
        if result.translation_dx_m is not None:
            self.assertAlmostEqual(result.translation_dx_m, -1.25, delta=0.3)
            self.assertAlmostEqual(result.translation_dy_m, -0.625, delta=0.3)

    def test_empty_mask_returns_insufficient_support(self) -> None:
        image = _textured_rgb()

        result = estimate_mosaic_registration(
            image,
            image.copy(),
            np.zeros((256, 256), dtype=np.uint8),
            from_date="2018-04-29",
            to_date="2018-11-01",
            pixel_to_projected=(1.0, 0.0, 0.0, -1.0),
        )

        self.assertEqual(result.status, "INSUFFICIENT_GROUND_CANDIDATE_FEATURES")
        self.assertEqual(result.valid_mask_fraction, 0.0)
        self.assertFalse(result.applied_to_image_or_geometry)

    def test_phase_correlation_recovers_fractional_candidate_without_application(self) -> None:
        source = _textured_rgb(seed=57)
        moved = cv2.warpAffine(
            source,
            np.float32([[1, 0, 2.5], [0, 1, -1.25]]),
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = estimate_mosaic_phase_registration(
            source,
            moved,
            np.ones((256, 256), dtype=np.uint8),
            from_date="2018-04-29",
            to_date="2018-11-01",
            pixel_to_projected=(0.5, 0.0, 0.0, -0.5),
        )

        self.assertEqual(result.status, "PHASE_CORRELATION_CANDIDATE_NOT_REVIEWED")
        self.assertAlmostEqual(result.translation_dx_px, -2.5, delta=0.5)
        self.assertAlmostEqual(result.translation_dy_px, 1.25, delta=0.5)
        self.assertAlmostEqual(result.translation_dx_m, -1.25, delta=0.25)
        self.assertAlmostEqual(result.translation_dy_m, -0.625, delta=0.25)
        self.assertGreater(result.phase_response, 0.0)
        self.assertFalse(result.applied_to_image_or_geometry)


if __name__ == "__main__":
    unittest.main()
