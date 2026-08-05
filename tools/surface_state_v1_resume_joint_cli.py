from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.formal_runtime_guards import formal_training_runtime
from surface_state_v1.training.resume_joint_pipeline import resume_joint_and_finish
from surface_state_v1.training.scaler_runtime_guard import allow_empty_disabled_scaler_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely resume Joint and finish Surface State V1")
    parser.add_argument("--joint-run", required=True)
    parser.add_argument("--pipeline-dir", required=True)
    args = parser.parse_args()
    with formal_training_runtime(), allow_empty_disabled_scaler_state():
        result = resume_joint_and_finish(repo=REPO, joint_run=args.joint_run, pipeline_dir=args.pipeline_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())