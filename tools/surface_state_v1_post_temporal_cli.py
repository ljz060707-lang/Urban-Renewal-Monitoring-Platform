from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.formal_runtime_guards import formal_training_runtime
from surface_state_v1.training.post_temporal_pipeline import run_post_temporal_pipeline
from surface_state_v1.training.scaler_runtime_guard import allow_empty_disabled_scaler_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume SurfaceState V1 after TFR")
    parser.add_argument("--pipeline-dir")
    args = parser.parse_args()
    output = args.pipeline_dir or (
        "outputs/pipeline/surface_state_v1_post_temporal/"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    with formal_training_runtime(), allow_empty_disabled_scaler_state():
        result = run_post_temporal_pipeline(
            repo=REPO,
            cwld_run="outputs/training/surface_state_v1/cwld_20260729_165008",
            baseline_run="outputs/training/surface_state_v1/fusu_sample_20260730_024737",
            tfr_run="outputs/training/surface_state_v1/tfr_sample_20260730_024737",
            selection_dir="outputs/training/surface_state_v1/fusu_branch_selection_20260730_024737",
            pipeline_dir=output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

