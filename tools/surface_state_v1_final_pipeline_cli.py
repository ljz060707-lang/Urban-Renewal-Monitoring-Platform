from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.final_pipeline import run_final_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="SurfaceState v1 final training pipeline")
    parser.add_argument(
        "--cwld-run",
        default="outputs/training/surface_state_v1/cwld_20260729_165008",
    )
    parser.add_argument("--pipeline-dir")
    args = parser.parse_args()
    pipeline = args.pipeline_dir or (
        "outputs/pipeline/surface_state_v1_final/" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    result = run_final_pipeline(args.cwld_run, pipeline, repo=REPO)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
