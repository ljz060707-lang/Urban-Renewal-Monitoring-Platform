from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.integration.overlay_final import final_overlay_infer


def main() -> int:
    parser = argparse.ArgumentParser(description="SurfaceState v1 final candidate overlay")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mti-selection", required=True)
    parser.add_argument("--input-t1", required=True)
    parser.add_argument("--input-t2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
    config.update(
        surface_checkpoint=args.checkpoint,
        mti_selection=args.mti_selection,
        input_t1=args.input_t1,
        input_t2=args.input_t2,
        output=args.output,
    )
    result = final_overlay_infer(config, repo=REPO)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
