#!/usr/bin/env python3
"""One-request JSON worker for an OpenCV-capable Python runtime."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image
from shapely import wkb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.reconstruction.registration import (estimate_tile_registration,
                                                registration_backend_status)


def _image(source: dict) -> tuple[Path | None, np.ndarray | None]:
    if "path" in source:
        return Path(source["path"]), None
    raw = base64.b64decode(source["png_base64"], validate=True)
    with Image.open(BytesIO(raw)) as image:
        return None, np.asarray(image.convert("RGB"))


def handle(request: dict) -> dict:
    operation = request.get("operation")
    if operation == "health":
        return {"ok": True, "backend": registration_backend_status()}
    if operation != "estimate_tile_registration":
        raise ValueError(f"Unsupported operation: {operation}")
    left_path, left_image = _image(request["left"])
    right_path, right_image = _image(request["right"])
    left_roofs = [wkb.loads(bytes.fromhex(value))
                  for value in request.get("left_roofs_wkb_hex", [])]
    right_roofs = [wkb.loads(bytes.fromhex(value))
                   for value in request.get("right_roofs_wkb_hex", [])]
    result = estimate_tile_registration(
        left_path, right_path,
        tile_x=int(request["tile_x"]),
        tile_y=int(request["tile_y"]),
        from_date=str(request["from_date"]),
        to_date=str(request["to_date"]),
        left_roofs=left_roofs,
        right_roofs=right_roofs,
        anchor_reviewed=bool(request.get("anchor_reviewed", False)),
        left_image=left_image,
        right_image=right_image,
        tile_span=int(request.get("tile_span", 1)),
    )
    return {"ok": True, "registration": result.as_dict()}


if __name__ == "__main__":
    try:
        print(json.dumps(handle(json.load(sys.stdin)), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False,
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}, ensure_ascii=False))
        raise SystemExit(1)
