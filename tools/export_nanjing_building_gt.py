"""Export accepted building GT attributes without modifying the source GT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .nanjing_gt_common import GT_DEFAULT, accepted_features, inspect_gt, write_json
except ImportError:
    from nanjing_gt_common import GT_DEFAULT, accepted_features, inspect_gt, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=GT_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run or not args.gt.exists():
        result = {"status": "dry_run_ready" if args.dry_run else "awaiting_manual_annotation", "output": str(args.output.resolve()), "source_gt_modified": False, "records_exported": 0}
        write_json(args.output.with_suffix(".json"), result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    audit = inspect_gt(args.gt)
    if audit["status"] != "GT_VALIDATION_PASS":
        write_json(args.output.with_suffix(".json"), {"status": "GT_VALIDATION_ERROR", "audit": audit, "source_gt_modified": False})
        return 2
    records = accepted_features(args.gt, "gt_building")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in records for key in row}) if records else ["gt_id", "date", "split", "label_status"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    result = {"status": "exported_read_only_derivative", "output": str(args.output.resolve()), "source_gt_modified": False, "records_exported": len(records)}
    write_json(args.output.with_suffix(".json"), result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
