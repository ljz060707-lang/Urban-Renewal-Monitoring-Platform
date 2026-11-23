#!/usr/bin/env python3
"""Export a read-only-source, fail-closed image-alignment review package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment.review_packet import (
    R5_RELATIVE_REVIEW_SCOPE_ID,
    export_alignment_review_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True,
                        help="Existing fixed-case reconstruction output directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New review-package directory; existing paths are refused")
    parser.add_argument(
        "--review-scope-allowlist",
        choices=(R5_RELATIVE_REVIEW_SCOPE_ID,),
        help=("Strictly limit the base-grid review layer to the frozen nine "
              "B04 date-cell keys; candidate fallback is disabled"))
    args = parser.parse_args()
    manifest = export_alignment_review_packet(
        args.source_run_dir, args.output_dir,
        review_scope_allowlist=args.review_scope_allowlist)
    print(json.dumps({
        "status": "EXPORTED_UNREVIEWED_CANDIDATES",
        "output_dir": str(args.output_dir.resolve()),
        "absolute_base_candidate_count": manifest["absolute_base_review"]["numeric_candidate_count"],
        "relative_date_pair_candidate_count": manifest["relative_date_pair_diagnostics"]["numeric_candidate_count"],
        "source_hash_recheck": manifest["source_hash_recheck"],
        "reference_hash_recheck": manifest["reference_hash_recheck"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
