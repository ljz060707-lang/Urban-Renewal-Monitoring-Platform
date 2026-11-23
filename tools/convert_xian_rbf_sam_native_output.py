"""Convert official RBF-SAM JSON output to the fixed-case R5 candidate bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.reconstruction.rbf_sam_native import convert_native_output_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert native RBF-SAM JSON predictions; output remains unverified.")
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True,
                        help="Exact file/date/tile/prompt bindings and input hashes.")
    parser.add_argument("--output", type=Path, required=True,
                        help="New path for xian_r5.rbf_sam_prediction_bundle.v2.")
    args = parser.parse_args()
    result = convert_native_output_files(
        args.native_output, args.inference_manifest, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
