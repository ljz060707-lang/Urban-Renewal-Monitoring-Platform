from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.integration.overlay_mti_runtime import calibrated_overlay_infer


def main() -> int:
    parser = argparse.ArgumentParser(description="SurfaceState v1 calibrated MTI overlay")
    parser.add_argument("--config", required=True)
    parser.add_argument("--surface-checkpoint")
    parser.add_argument("--mti-selection")
    parser.add_argument("--input-t1")
    parser.add_argument("--input-t2")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    for key, value in (
        ("surface_checkpoint", args.surface_checkpoint),
        ("mti_selection", args.mti_selection),
        ("input_t1", args.input_t1),
        ("input_t2", args.input_t2),
        ("output", args.output),
    ):
        if value is not None:
            config[key] = value
    missing = [
        key
        for key in ("surface_checkpoint", "mti_selection", "input_t1", "input_t2", "output")
        if not config.get(key)
    ]
    if missing:
        raise ValueError("Missing calibrated overlay configuration: " + ", ".join(missing))
    result = calibrated_overlay_infer(config, repo=REPO)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
