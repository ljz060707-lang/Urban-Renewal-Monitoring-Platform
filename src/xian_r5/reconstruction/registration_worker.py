"""Subprocess adapter for the optional OpenCV registration implementation."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from shapely.geometry.base import BaseGeometry

from .registration import Registration


class SubprocessRegistrationBackend:
    """Run the existing registration implementation in another Python runtime.

    The adapter transports only read-only image paths or in-memory PNG bytes
    plus roof WKB. The worker returns the same Registration value as the
    in-process implementation; it never accepts or applies a correction.
    """

    def __init__(self, python_executable: Path, worker_script: Path,
                 timeout_seconds: int = 180) -> None:
        self.python_executable = Path(python_executable).resolve()
        self.worker_script = Path(worker_script).resolve()
        self.timeout_seconds = int(timeout_seconds)

    def _request(self, payload: dict) -> dict:
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"Registration worker Python not found: {self.python_executable}")
        if not self.worker_script.is_file():
            raise FileNotFoundError(f"Registration worker script not found: {self.worker_script}")
        completed = subprocess.run(
            [str(self.python_executable), str(self.worker_script)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=self.timeout_seconds,
            cwd=str(self.worker_script.parents[1]),
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "worker exited without details").strip()
            raise RuntimeError(f"Registration worker failed ({completed.returncode}): {detail[-4000:]}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Registration worker returned invalid JSON") from exc
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error") or "Registration worker rejected the request"))
        return response

    def status(self) -> dict:
        try:
            response = self._request({"operation": "health"})
            worker_status = response.get("backend", {})
            available = worker_status.get("available") is True
            import_error = worker_status.get("import_error")
            if not available and not import_error:
                import_error = "Worker did not report an available OpenCV backend"
        except Exception as exc:
            available = False
            import_error = f"{type(exc).__name__}: {exc}"
        return {
            "name": "OPENCV_SIFT_GROUND_MASKED_TRANSLATION",
            "adapter": "PYTHON_SUBPROCESS",
            "available": available,
            "status": "AVAILABLE" if available else "BLOCKED_NO_OPENCV",
            "python_executable": str(self.python_executable),
            "worker_script": str(self.worker_script),
            "import_error": import_error,
        }

    @staticmethod
    def _image_png_base64(image: np.ndarray) -> str:
        buffer = BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
            buffer, format="PNG", compress_level=1)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def estimate_tile_registration(
        self,
        left_path: Path | None,
        right_path: Path | None,
        *,
        tile_x: int,
        tile_y: int,
        from_date: str,
        to_date: str,
        left_roofs=(),
        right_roofs=(),
        anchor_reviewed: bool = False,
        left_image: np.ndarray | None = None,
        right_image: np.ndarray | None = None,
        tile_span: int = 1,
    ) -> Registration:
        def image_source(path: Path | None, array: np.ndarray | None) -> dict:
            if array is not None:
                return {"png_base64": self._image_png_base64(array)}
            if path is None:
                raise ValueError("Each registration input needs an image path or array")
            return {"path": str(Path(path).resolve())}

        response = self._request({
            "operation": "estimate_tile_registration",
            "left": image_source(left_path, left_image),
            "right": image_source(right_path, right_image),
            "left_roofs_wkb_hex": [geometry.wkb_hex for geometry in left_roofs],
            "right_roofs_wkb_hex": [geometry.wkb_hex for geometry in right_roofs],
            "tile_x": int(tile_x),
            "tile_y": int(tile_y),
            "from_date": from_date,
            "to_date": to_date,
            "anchor_reviewed": bool(anchor_reviewed),
            "tile_span": int(tile_span),
        })
        return Registration(**response["registration"])
