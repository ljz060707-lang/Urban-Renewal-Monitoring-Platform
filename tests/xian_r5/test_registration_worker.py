from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.reconstruction.registration_worker import SubprocessRegistrationBackend


class SubprocessRegistrationBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.python = root / "python.exe"
        self.worker = root / "worker.py"
        self.python.write_bytes(b"test executable placeholder")
        self.worker.write_text("# test worker placeholder\n", encoding="utf-8")
        self.backend = SubprocessRegistrationBackend(self.python, self.worker)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_health_reports_subprocess_adapter(self) -> None:
        reply = {"ok": True, "backend": {
            "name": "OPENCV_SIFT_GROUND_MASKED_TRANSLATION",
            "available": True, "status": "AVAILABLE", "import_error": None,
        }}
        completed = type("Completed", (), {
            "returncode": 0, "stdout": json.dumps(reply), "stderr": "",
        })()
        with patch("xian_r5.reconstruction.registration_worker.subprocess.run",
                   return_value=completed):
            status = self.backend.status()

        self.assertTrue(status["available"])
        self.assertEqual(status["status"], "AVAILABLE")
        self.assertEqual(status["adapter"], "PYTHON_SUBPROCESS")
        self.assertEqual(status["python_executable"], str(self.python.resolve()))

    def test_estimation_transports_images_and_roof_geometry_through_interface(self) -> None:
        registration = {
            "tile_x": 10, "tile_y": 20,
            "from_date": "2018-04-29", "to_date": "2018-11-01",
            "dx_m": -1.0, "dy_m": 0.5, "sigma_m": 0.2,
            "n_matches": 30, "n_inliers": 24, "coverage_4x4": 0.5,
            "median_residual_px": 0.4,
            "status": "GROUND_MASKED_CANDIDATE_NOT_REVIEWED",
            "anchor_reviewed": False,
        }
        completed = type("Completed", (), {
            "returncode": 0,
            "stdout": json.dumps({"ok": True, "registration": registration}),
            "stderr": "",
        })()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("xian_r5.reconstruction.registration_worker.subprocess.run",
                   return_value=completed) as run_process:
            result = self.backend.estimate_tile_registration(
                None, None, tile_x=10, tile_y=20,
                from_date="2018-04-29", to_date="2018-11-01",
                left_roofs=(box(0, 0, 1, 1),), left_image=image,
                right_image=image.copy(), tile_span=1)

        request = json.loads(run_process.call_args.kwargs["input"])
        self.assertEqual(result.status, "GROUND_MASKED_CANDIDATE_NOT_REVIEWED")
        self.assertIsNotNone(request["left"]["png_base64"])
        self.assertEqual(request["left_roofs_wkb_hex"], [box(0, 0, 1, 1).wkb_hex])
        self.assertEqual(request["tile_span"], 1)
        self.assertFalse(result.anchor_reviewed)


if __name__ == "__main__":
    unittest.main()
