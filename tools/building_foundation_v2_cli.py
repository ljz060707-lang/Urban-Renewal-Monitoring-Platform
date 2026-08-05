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
    for name in (
        "audit", "pair", "leakage", "train", "validate", "package", "qa", "diagnostics", "finalize",
        "init-native-recovery", "audit-labels", "inspect-native", "compare-pipeline", "validate-native", "finalize-native",
        "init-b1-pilot", "run-b1-pilot", "finalize-b1-pilot",
        "init-b2-boundary", "run-b2-boundary", "finalize-b2-boundary",
        "init-b2-diagnostic", "run-b2-diagnostic",
    ):
        item = sub.add_parser(name); item.add_argument("args", nargs=argparse.REMAINDER)
    # The delegated tools own their option schemas.  ``parse_known_args`` is
    # intentional here: it keeps the unified help/dispatch command from
    # rejecting tool-specific options such as ``--run-dir`` before the target
    # script can validate them.
    args, unknown = parser.parse_known_args()
    script = {
        "audit": "audit_globfp_dataset.py", "pair": "pair_globfp_with_local_imagery.py", "leakage": "audit_building_v2_leakage.py",
        "train": "train_building_foundation_v2.py", "validate": "validate_building_foundation_v2.py", "package": "package_building_foundation_v2.py",
        "qa": "create_building_v2_qa_queue.py", "diagnostics": "build_building_v2_diagnostics.py",
        "finalize": "finalize_building_v2_run.py",
        "init-native-recovery": "init_building_v2_native_recovery.py",
        "audit-labels": "audit_whu_label_semantics.py",
        "inspect-native": "inspect_rsbuilding_native_runtime.py",
        "compare-pipeline": "compare_rsbuilding_input_pipelines.py",
        "validate-native": "validate_rsbuilding_iter2000_native.py",
        "finalize-native": "finalize_building_v2_native_recovery.py",
        "init-b1-pilot": "init_building_v2_b1_native_pilot.py",
        "run-b1-pilot": "run_building_v2_b1_native_pilot.py",
        "finalize-b1-pilot": "finalize_building_v2_b1_native_pilot.py",
        "init-b2-boundary": "init_building_v2_b2_boundary_ablation.py",
        "run-b2-boundary": "run_building_v2_b2_boundary_ablation.py",
        "finalize-b2-boundary": "finalize_building_v2_b2_boundary_ablation.py",
        "init-b2-diagnostic": "init_building_v2_b2_boundary_diagnostic.py",
        "run-b2-diagnostic": "run_building_v2_b2_boundary_diagnostic.py",
    }[args.command]
    forwarded = [*unknown, *getattr(args, "args", [])]
    return subprocess.call([sys.executable, str(ROOT / "tools" / script), *forwarded], cwd=str(ROOT))


if __name__ == "__main__": raise SystemExit(main())
