#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import rasterio
import yaml
from shapely.geometry import box

from urban_monitor.crs import select_local_utm_crs
from urban_monitor.provenance import utc_now, write_run_manifest


LOG = logging.getLogger("phase1-smoke")


def _run(repo: Path, *arguments: str) -> None:
    command = [sys.executable, *arguments]
    LOG.info("run: %s", " ".join(command))
    subprocess.run(command, cwd=repo, check=True)


def _copy_required_aliases(run_dir: Path) -> None:
    shutil.copytree(run_dir / "predictions", run_dir / "mock_predictions", dirs_exist_ok=True)
    shutil.copytree(run_dir / "predictions_regularized", run_dir / "regularized_predictions", dirs_exist_ok=True)
    for name in [
        "tracks_long.gpkg", "tracks_summary.gpkg", "observation_matrix.csv", "state_intervals.csv",
        "lineage_events.gpkg", "lineage_events.csv",
    ]:
        source = run_dir / "tracking" / name
        if source.exists():
            shutil.copy2(source, run_dir / name)
    for name in ["excluded_by_mmu.csv", "excluded_missing_date.csv", "excluded_low_confidence.csv"]:
        shutil.copy2(run_dir / "student_labels" / name, run_dir / name)


def _required_outputs(run_dir: Path) -> list[Path]:
    return [
        run_dir / "alignment_report.csv",
        run_dir / "alignment_report.md",
        run_dir / "composite_quality_masks",
        run_dir / "coco",
        run_dir / "mock_predictions",
        run_dir / "regularized_predictions",
        run_dir / "tracks_long.gpkg",
        run_dir / "tracks_summary.gpkg",
        run_dir / "observation_matrix.csv",
        run_dir / "state_intervals.csv",
        run_dir / "lineage_events.gpkg",
        run_dir / "review" / "index.html",
        run_dir / "student_labels",
        run_dir / "excluded_by_mmu.csv",
        run_dir / "excluded_missing_date.csv",
        run_dir / "excluded_low_confidence.csv",
        run_dir / "evaluation_summary.md",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the legacy-compatible CPU-only synthetic workflow.")
    parser.add_argument("--config", default="configs/example.yaml", help="Base configuration to copy into the run")
    parser.add_argument("--output-root", default="outputs", help="Parent output directory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo = Path(__file__).resolve().parents[1]
    base_config = (repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    aoi_id = str(cfg.get("aoi_id", "demo_aoi"))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("smoke_%Y%m%dT%H%M%SZ")
    output_root = (repo / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    run_dir = output_root / aoi_id / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Run directory exists; use --force or a new --run-id: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = run_dir / "data" / aoi_id
    config_path = run_dir / "smoke_config.yaml"
    cfg["data_root"] = str(data_root)
    cfg["manifest"] = str(data_root / "manifest.csv")
    cfg["output_root"] = str(run_dir)
    cfg.setdefault("review", {})["review_csv"] = str(run_dir / "review" / "reviewed_states.csv")
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    start_time = utc_now()
    status = "failed"
    warnings: list[str] = []
    try:
        _run(repo, "scripts/make_synthetic_example.py", "--output", str(data_root), "--force")
        for script in [
            "validate_schemas.py",
            "validate_crs.py",
            "validate_alignment.py",
            "diagnose_alignment.py",
            "generate_quality_masks.py",
            "build_coco.py",
            "make_mock_predictions.py",
            "regularize_predictions.py",
            "track_states.py",
            "build_review_queue.py",
            "export_student_labels.py",
            "evaluate_phase1.py",
        ]:
            _run(repo, f"scripts/{script}", "--config", str(config_path), "--force")
        _copy_required_aliases(run_dir)
        missing = [str(path) for path in _required_outputs(run_dir) if not path.exists()]
        if missing:
            raise RuntimeError(f"Smoke workflow missing required outputs: {missing}")
        status = "success"
    finally:
        manifest = data_root / "manifest.csv"
        source_crs = analysis_crs = ""
        first_image = data_root / "images" / "2022-04-01.tif"
        if first_image.exists():
            with rasterio.open(first_image) as source:
                source_crs = str(source.crs)
                analysis_crs = select_local_utm_crs(box(*source.bounds), source.crs).to_string()
        files = [path for path in run_dir.rglob("*") if path.is_file() and path.name != "RUN_MANIFEST.json"]
        write_run_manifest(
            run_dir / "RUN_MANIFEST.json",
            run_id=run_id,
            aoi_id=aoi_id,
            command=" ".join(sys.argv),
            config_path=config_path,
            status=status,
            input_files=[manifest, *sorted((data_root / "images").glob("*.tif"))] if manifest.exists() else [],
            output_files=files,
            warnings=warnings,
            start_time=start_time,
            model_version=str(cfg.get("model_version", "")),
            source_crs=source_crs,
            analysis_crs=analysis_crs,
        )
    LOG.info("smoke status=%s run_dir=%s", status, run_dir)
    if status != "success":
        raise SystemExit(1)
    print(run_dir)


if __name__ == "__main__":
    main()
