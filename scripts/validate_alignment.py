#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.alignment import validate_alignment
from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_manifest(cfg["manifest"], cfg["data_root"])
    report = validate_alignment(records, require_projected=cfg["alignment"].get("require_projected_crs", True))
    output = prepare_output(Path(cfg["output_root"]) / "alignment_report.csv", force=args.force)
    report.to_csv(output, index=False, encoding="utf-8-sig")
    print(report[["date", "matches_reference", "mismatch_reason"]].to_string(index=False))
    if not report["matches_reference"].all():
        raise SystemExit(f"Alignment failed. See {output}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
