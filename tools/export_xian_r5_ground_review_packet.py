#!/usr/bin/env python3
"""Export an actual-date, candidate-only ground-footprint review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.reconstruction.ground_review_packet import (
    export_ground_footprint_review_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True,
                        help="Existing structurally passing fixed-case reconstruction run")
    parser.add_argument("--mosaic-diagnostic", type=Path, required=True,
                        help="Actual-date z20 mosaic registration diagnostic")
    parser.add_argument("--qa-receipt", type=Path, default=None,
                        help="Optional explicitly named engineering QA JSON; defaults to source-run/ENGINEERING_QA.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New review packet directory; existing paths are refused")
    parser.add_argument("--patch-size-px", type=int, default=512,
                        help="Square source-image context size for per-observation cards")
    args = parser.parse_args()
    manifest = export_ground_footprint_review_packet(
        args.source_run_dir, args.mosaic_diagnostic, args.output_dir,
        patch_size_px=args.patch_size_px,
        qa_receipt_path=args.qa_receipt,
    )
    print(json.dumps({
        "status": manifest["review_state"],
        "output_dir": str(args.output_dir.resolve()),
        "actual_dates": manifest["actual_date_order"],
        "roof_candidate_count": manifest["roof_candidate_count"],
        "review_card_count": manifest["review_card_count"],
        "template_feature_count": manifest["reviewed_ground_footprints"]["template_feature_count"],
        "source_hash_recheck": manifest["source_hash_recheck"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
