from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.experiment_control import (
    run_joint_from_selection,
    run_joint_probes,
    select_temporal_branch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="SurfaceState v1 experiment control")
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select-temporal")
    select.add_argument("--baseline-run", required=True)
    select.add_argument("--tfr-run", required=True)
    select.add_argument("--output", required=True)
    probes = commands.add_parser("joint-probes")
    probes.add_argument("--config", required=True)
    probes.add_argument("--cwld-checkpoint", required=True)
    probes.add_argument("--fusu-checkpoint", required=True)
    probes.add_argument("--use-tfr", action="store_true")
    probes.add_argument("--output", required=True)
    joint = commands.add_parser("joint-train-selected")
    joint.add_argument("--config", required=True)
    joint.add_argument("--selection", required=True)
    joint.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "select-temporal":
        result = select_temporal_branch(args.baseline_run, args.tfr_run, args.output)
    elif args.command == "joint-probes":
        result = run_joint_probes(
            REPO / args.config,
            args.cwld_checkpoint,
            args.fusu_checkpoint,
            args.use_tfr,
            args.output,
            repo=REPO,
        )
    else:
        result = run_joint_from_selection(
            REPO / args.config, args.selection, args.output, repo=REPO
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
