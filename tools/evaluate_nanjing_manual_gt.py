"""Future explicit evaluation entry point; never trains or runs inference."""

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
    parser.add_argument("--module", choices=["building", "construction", "all"], default="all")
    parser.add_argument("--predictions", type=Path, help="explicit existing prediction manifest; no inference is launched")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        result = {"status": "dry_run_ready", "module": args.module, "metrics": {}, "true_accuracy_reported": False, "training_started": False, "inference_started": False, "locked_test_pixel_reads": 0}
    else:
        gt_audit = inspect_gt(args.gt)
        result = {
            "status": gt_audit["status"] if gt_audit["status"] != "GT_VALIDATION_PASS" else "predictions_required",
            "module": args.module,
            "gt_audit_status": gt_audit["status"],
            "gt_path": str(args.gt.resolve()),
            "metrics": {},
            "prediction_manifest": str(args.predictions.resolve()) if args.predictions else "",
            "true_accuracy_reported": False,
            "training_started": False,
            "inference_started": False,
            "locked_test_pixel_reads": 0,
            "human_ground_truth_available": gt_audit.get("human_ground_truth_available", False),
        }
        if gt_audit["status"] == "GT_VALIDATION_PASS" and gt_audit.get("metrics_allowed") and args.predictions:
            if not args.predictions.exists():
                result["status"] = "PREDICTION_MANIFEST_MISSING"
            else:
                result["status"] = "PREDICTIONS_READY_FOR_EXPLICIT_METRIC_IMPLEMENTATION"
                result["note"] = "No checkpoint selection or inference is performed by this entry point."
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"awaiting_manual_annotation", "GT_VALIDATION_PASS", "predictions_required", "dry_run_ready", "PREDICTIONS_READY_FOR_EXPLICIT_METRIC_IMPLEMENTATION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
