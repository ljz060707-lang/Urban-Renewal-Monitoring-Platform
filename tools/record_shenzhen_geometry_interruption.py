"""Record an interrupted, resumable full-raster geometry attempt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", type=Path, required=True)
    args = parser.parse_args()
    states = []
    for path in sorted(args.geometry_root.rglob("_STREAM_STATE.json")):
        try:
            states.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            states.append({"path": str(path), "read_error": str(exc)})
    payload = {
        "schema_version": "urrs.shenzhen.geometry.interruption_receipt.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "INTERRUPTED_FOR_STORAGE_SAFETY",
        "reason": "full probability/mask/boundary rasters projected to exceed available E: space for four full Shenzhen mosaics",
        "resumable": True,
        "preserved_partial_outputs": True,
        "states": states,
        "raw_tifs_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "scientific_status": "candidate_not_ground_truth",
    }
    target = args.geometry_root / f"GEOMETRY_INFERENCE_INTERRUPTION_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(target), "status": payload["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
