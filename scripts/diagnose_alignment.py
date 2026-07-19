#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.alignment_diagnostics import diagnose_alignment, write_alignment_report
from urban_monitor.config import load_config
from urban_monitor.manifest import load_manifest
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Write adjacent-date ground/roof alignment diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tracking = cfg.get("tracking", {})
    report = diagnose_alignment(
        load_manifest(cfg["manifest"], cfg["data_root"]),
        base_buffer_m=float(tracking.get("base_buffer_m", tracking.get("buffer_m", 3.0))),
        off_nadir_scale_m_per_degree=float(tracking.get("off_nadir_scale_m_per_degree", 0.35)),
        max_dynamic_buffer_m=float(tracking.get("max_dynamic_buffer_m", 20.0)),
    )
    root = Path(cfg["output_root"])
    prepare_output(root / "alignment_report.csv", force=args.force)
    prepare_output(root / "alignment_report.md", force=args.force)
    write_alignment_report(report, root / "alignment_report.csv", root / "alignment_report.md")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
