#!/usr/bin/env python3
"""Compile bounded B04 multi-reference, exact-source-tile tie candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xian_r5.alignment import export_component_tie_consensus_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic", action="append", type=Path, required=True,
        help="B04 exact-source-component BASE_TIE_DIAGNOSTIC.json; repeat for each reference.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="New, not-yet-existing B04 output directory under the reconstruction root.",
    )
    parser.add_argument(
        "--fixed-case-geometry", type=Path, required=True,
        help="Frozen fixed-case GeoPackage used by the source diagnostics.",
    )
    parser.add_argument("--minimum-correlation", type=float, default=0.6)
    parser.add_argument("--minimum-peak-margin", type=float, default=0.1)
    parser.add_argument("--agreement-tolerance-m", type=float, default=0.01)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to reuse existing output directory: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output parent directory does not exist: {args.output.parent}")
    try:
        case_ids = {
            json.loads(path.read_text(encoding="utf-8")).get("case_id")
            for path in args.diagnostic
        }
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"could not read a diagnostic JSON: {exc}")
    if case_ids != {"B04-TQA10"}:
        parser.error(f"this bounded compiler accepts only B04-TQA10, got {sorted(case_ids)}")
    args.output.mkdir()
    manifest = export_component_tie_consensus_candidates(
        args.output, args.diagnostic,
        fixed_case_geometry_path=args.fixed_case_geometry,
        minimum_correlation=args.minimum_correlation,
        minimum_peak_margin=args.minimum_peak_margin,
        agreement_tolerance_m=args.agreement_tolerance_m,
    )
    summary = {
        "manifest_path": manifest["manifest_path"],
        "case_id": manifest["case_id"],
        "reference_years": manifest["reference_years"],
        "actual_dates": manifest["actual_dates"],
        "source_component_images_revalidated": manifest[
            "source_component_images_revalidated"],
        "products_by_actual_date": manifest["products_by_actual_date"],
        "absolute_ground_position_resolved": manifest[
            "absolute_ground_position_resolved"],
        "formal_change_products_written": manifest[
            "formal_change_products_written"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
