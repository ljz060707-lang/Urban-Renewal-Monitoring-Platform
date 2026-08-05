"""Validate the user-owned Nanjing manual GT GeoPackage, read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .nanjing_gt_common import GT_DEFAULT, inspect_gt, write_json
except ImportError:
    from nanjing_gt_common import GT_DEFAULT, inspect_gt, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=GT_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="check path/contract only; never reads an existing GT")
    args = parser.parse_args(argv)
    if args.dry_run:
        result = {"status": "dry_run_ready", "gt_path": str(args.gt.resolve()), "gt_modified": False, "training_started": False, "inference_started": False, "locked_test_pixel_reads": 0}
    else:
        result = inspect_gt(args.gt)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"awaiting_manual_annotation", "GT_VALIDATION_PASS", "dry_run_ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
