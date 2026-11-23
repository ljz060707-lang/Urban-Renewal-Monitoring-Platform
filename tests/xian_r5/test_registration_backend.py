from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.reconstruction import registration


class RegistrationBackendTests(unittest.TestCase):
    def test_missing_opencv_is_reported_explicitly(self) -> None:
        with patch.object(registration, "cv2", None):
            self.assertEqual(registration.registration_backend_status()["status"],
                             "BLOCKED_NO_OPENCV")
            self.assertFalse(registration.registration_backend_status()["available"])

    def test_tile_registration_does_not_open_inputs_without_backend(self) -> None:
        with patch.object(registration, "cv2", None):
            result = registration.estimate_tile_registration(
                Path("missing-left.jpg"), Path("missing-right.jpg"),
                tile_x=1, tile_y=2, from_date="2018-04-29",
                to_date="2018-11-01")

        self.assertEqual(result.status, "BLOCKED_NO_OPENCV")
        self.assertIsNone(result.dx_m)
        self.assertIsNone(result.dy_m)

    def test_mosaic_estimators_return_blocked_without_synthetic_candidates(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.uint8)
        common = {
            "from_date": "2018-04-29",
            "to_date": "2018-11-01",
            "pixel_to_projected": (0.5, 0.0, 0.0, -0.5),
        }
        with patch.object(registration, "cv2", None):
            feature = registration.estimate_mosaic_registration(
                image, image.copy(), mask, **common)
            phase = registration.estimate_mosaic_phase_registration(
                image, image.copy(), mask, **common)

        self.assertEqual(feature.status, "BLOCKED_NO_OPENCV")
        self.assertIsNone(feature.translation_dx_m)
        self.assertIsNone(feature.translation_dy_m)
        self.assertEqual(phase.status, "BLOCKED_NO_OPENCV")
        self.assertIsNone(phase.translation_dx_m)
        self.assertIsNone(phase.translation_dy_m)
        self.assertFalse(feature.applied_to_image_or_geometry)
        self.assertFalse(phase.applied_to_image_or_geometry)


if __name__ == "__main__":
    unittest.main()
