#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.evaluation import evaluate_phase1
from urban_monitor.manifest import load_manifest
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate available Phase-1 Ground Truth without inventing metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepare_output(Path(cfg["output_root"]) / "evaluation_summary.json", force=args.force)
    prepare_output(Path(cfg["output_root"]) / "evaluation_summary.md", force=args.force)
    records = load_manifest(cfg["manifest"], cfg["data_root"])
    regularized = Path(cfg["output_root"]) / "predictions_regularized"
    predictions = {record.date: regularized / f"{record.date}.gpkg" for record in records}
    truth = {record.date: record.label_path for record in records if record.label_path}
    summary = evaluate_phase1(predictions, truth, cfg["output_root"])
    print(summary)


if __name__ == "__main__":
    main()
