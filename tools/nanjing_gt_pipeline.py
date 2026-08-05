"""Unified explicit manual-GT validation/evaluation/export entry point."""

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
    parser.add_argument("--mode", choices=["status", "validate", "evaluate-building", "evaluate-construction", "evaluate-all", "export-building", "export-construction"], default="status")
    parser.add_argument("--gt", type=Path, default=GT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        result = {"status": "dry_run_ready", "mode": args.mode, "training_started": False, "inference_started": False, "locked_test_pixel_reads": 0, "human_gt_modified": False, "true_accuracy_reported": False}
    else:
        audit = inspect_gt(args.gt)
        result = {"status": audit["status"], "mode": args.mode, "gt_path": str(args.gt.resolve()), "gt_audit_status": audit["status"], "metrics": {}, "training_started": False, "inference_started": False, "locked_test_pixel_reads": 0, "human_gt_modified": False, "true_accuracy_reported": False}
        if args.mode == "status":
            result["status"] = audit["status"]
        elif audit["status"] == "GT_VALIDATION_PASS":
            if args.mode.startswith("evaluate") and not args.predictions:
                result["status"] = "predictions_required"
            elif args.mode.startswith("export"):
                result["status"] = "export_entry_ready"
            else:
                result["status"] = "validation_passed_no_model_execution"
    write_json(args.output_dir / "NANJING_GT_PIPELINE_STATUS.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] not in {"GT_VALIDATION_ERROR", "PREDICTION_MANIFEST_MISSING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
