from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.training.cwld_finalization import finalize_cwld
from surface_state_v1.training.safe_resume import resume_cwld


def main() -> int:
    parser = argparse.ArgumentParser(description="SurfaceState v1 continuation control")
    commands = parser.add_subparsers(dest="command", required=True)
    resume = commands.add_parser("resume-cwld")
    resume.add_argument("--config", required=True)
    resume.add_argument("--run-dir", required=True)
    finalize = commands.add_parser("cwld-finalize")
    finalize.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "resume-cwld":
        result = resume_cwld(REPO / args.config, args.run_dir, repo=REPO)
    else:
        result = finalize_cwld(args.run_dir, repo=REPO)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

