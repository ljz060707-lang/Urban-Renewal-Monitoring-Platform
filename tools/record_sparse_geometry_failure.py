"""Record a resumable sparse-geometry failure without touching raw inputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-date", required=True)
    parser.add_argument("--error-type", required=True)
    parser.add_argument("--error", required=True)
    parser.add_argument("--traceback", default="")
    args = parser.parse_args()
    scene_dir = args.output_root / args.scene_date
    state_path = scene_dir / "_SPARSE_STATE.json"
    state = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            state = {"state_read_error": str(exc)}
    payload = {
        "schema_version": "urrs.shenzhen.sparse_geometry_failure.v1",
        "status": "FAILURE_RECEIPT",
        "scene_date": args.scene_date,
        "error_type": args.error_type,
        "error": args.error,
        "traceback": args.traceback,
        "state_snapshot": state,
        "resume_safe": True,
        "raw_tifs_modified": False,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "scientific_status": "candidate_not_ground_truth",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / f"SPARSE_GEOMETRY_FAILURE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(target), "status": payload["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
