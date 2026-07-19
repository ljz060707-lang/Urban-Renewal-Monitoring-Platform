#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.downscale import export_student_change_labels
from urban_monitor.manifest import load_manifest
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MMU-filtered fractional labels for a 10 m student model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_manifest(cfg["manifest"], cfg["data_root"])
    d = cfg.get("student_export", {})
    prepare_output(Path(cfg["output_root"]) / "student_labels" / "label_inventory.csv", force=args.force)
    tracking_dir = Path(cfg["output_root"]) / "tracking"
    reviewed = tracking_dir / "tracks_reviewed.gpkg"
    summary = reviewed if reviewed.exists() else tracking_dir / "tracks_summary.gpkg"
    print(f"student_export_source={'reviewed' if reviewed.exists() else 'auto'}:{summary}")
    outputs = export_student_change_labels(
        summary,
        Path(cfg["output_root"]) / "tracking" / "state_intervals.csv",
        records[0].image_path,
        Path(cfg["output_root"]) / "student_labels",
        target_resolution_m=float(d.get("target_resolution_m", 10.0)),
        min_mapping_unit_m2=float(d.get("min_mapping_unit_m2", 200.0)),
        supersample=int(d.get("supersample", 4)),
        min_fraction_to_keep=float(d.get("min_fraction_to_keep", 0.05)),
        require_reviewed_or_high_confidence=bool(d.get("require_reviewed_or_high_confidence", True)),
        model_version=str(cfg.get("model_version", "")),
        schema_version=str(cfg.get("project", {}).get("schema_version", "0.4.0")),
        source_aoi=str(cfg.get("aoi_id", "")),
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
