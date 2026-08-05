"""Unified local CLI for the Building Foundation v2 audit/training flow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "pair", "leakage", "train", "validate", "package", "qa", "diagnostics", "finalize"):
        item = sub.add_parser(name); item.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = {
        "audit": "audit_globfp_dataset.py", "pair": "pair_globfp_with_local_imagery.py", "leakage": "audit_building_v2_leakage.py",
        "train": "train_building_foundation_v2.py", "validate": "validate_building_foundation_v2.py", "package": "package_building_foundation_v2.py",
        "qa": "create_building_v2_qa_queue.py", "diagnostics": "build_building_v2_diagnostics.py",
        "finalize": "finalize_building_v2_run.py",
    }[args.command]
    return subprocess.call([sys.executable, str(ROOT / "tools" / script), *args.args], cwd=str(ROOT))


if __name__ == "__main__": raise SystemExit(main())
