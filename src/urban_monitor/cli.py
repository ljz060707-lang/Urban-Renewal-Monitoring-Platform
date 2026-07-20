from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_data_hash(project) -> str:
    files = [*project.images.values(), *project.labels.values()]
    project_yaml = project.project_dir / "project.yaml"
    if project_yaml.exists():
        files.append(project_yaml)
    quality = project.project_dir / "quality"
    if quality.exists():
        files.extend(path for path in quality.iterdir() if path.is_file())
    payload = {
        "files": {str(path): file_sha256(path) for path in sorted(files)},
        "config": {key: project.config[key] for key in ("data", "tiling", "split", "crs")},
    }
    return _json_hash(payload)


def _cache(run: Path) -> dict:
    path = run / "CACHE.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_cache(run: Path, payload: dict) -> None:
    (run / "CACHE.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest(aoi_id: str) -> dict | None:
    path = REPO_ROOT / "outputs" / aoi_id / "latest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = Path(payload.get("run_directory", ""))
    return payload if payload.get("status") == "success" and run.is_dir() else None


def _unique_run(aoi_id: str) -> Path:
    root = REPO_ROOT / "outputs" / aoi_id / "runs"
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / base
    suffix = 1
    while path.exists():
        path = root / f"{base}_{suffix:02d}"
        suffix += 1
    for name in ("logs", "prepared", "training", "evaluation", "predictions", "tracking", "review", "reports"):
        (path / name).mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": path.name, "aoi_id": aoi_id, "created_at": _now(), "status": "running",
        "schema_version": "0.4.0", "stages": {}, "cache": {}, "review_source": "auto",
    }
    (path / "RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _update_manifest(run: Path, **updates) -> dict:
    path = run / "RUN_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _stage(run: Path, name: str, status: str, cache_status: str = "", **details) -> None:
    path = run / "RUN_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("stages", {})[name] = {"status": status, "cache": cache_status, "time": _now(), **details}
    if cache_status:
        payload.setdefault("cache", {})[name] = cache_status
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run / "logs" / "workflow.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} {name} status={status} cache={cache_status}\n")


def complete_run(run: Path, best_checkpoint: str | Path | None = None) -> None:
    best = str(best_checkpoint or "")
    output_files = [str(path) for path in sorted(run.rglob("*")) if path.is_file()]
    payload = _update_manifest(run, status="success", completed_at=_now(), best_checkpoint=best, output_files=output_files)
    latest = {
        "run_id": run.name,
        "status": "success",
        "best_checkpoint": best,
        "run_directory": str(run.resolve()),
        "created_at": payload["created_at"],
    }
    path = REPO_ROOT / "outputs" / payload["aoi_id"] / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")


def fail_run(run: Path, stage: str, exc: BaseException) -> None:
    traceback_path = run / "logs" / "traceback.txt"
    traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
    _update_manifest(
        run, status="failed", failure_stage=stage, error_message=str(exc),
        traceback_path=str(traceback_path), completed_at=_now(),
    )


def _clone_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _resolved_config(project, run: Path, manifest: Path) -> dict:
    config = deepcopy(project.config)
    config.update(
        aoi_id=project.aoi_id,
        data_root=str(project.project_dir),
        manifest=str(manifest),
        output_root=str(run),
        model_version=f"maskrcnn-v0.4-{run.name}",
    )
    config["chips"] = {"size": config["tiling"]["tile_size"], "overlap": config["tiling"]["overlap"]}
    config["review"]["review_csv"] = str(run / "review" / "reviewed_states.csv")
    return config


def doctor(data_root: str | Path = "data") -> int:
    required = ["yaml", "numpy", "pandas", "rasterio", "geopandas", "shapely", "cv2", "torch", "torchvision"]
    missing = []
    versions = {}
    for name in required:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except Exception:
            missing.append(name)
    issues = []
    if sys.version_info < (3, 10):
        issues.append(f"Python {platform.python_version()} is too old; require >=3.10")
    if missing:
        issues.append("Missing packages: " + ", ".join(missing))
    data = (REPO_ROOT / data_root).resolve() if not Path(data_root).is_absolute() else Path(data_root)
    data.mkdir(parents=True, exist_ok=True)
    try:
        probe = data / "中文路径_写权限检查.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        issues.append(f"Data directory is not writable or Chinese path failed: {exc}")
    free_gb = shutil.disk_usage(data).free / (1024 ** 3)
    if free_gb < 2:
        issues.append(f"Available disk is only {free_gb:.2f} GiB")
    cuda = False
    cuda_info = "Torch unavailable"
    if "torch" not in missing:
        import torch

        cuda = torch.cuda.is_available()
        cuda_info = torch.version.cuda or "CPU build"
    print(f"Python: {platform.python_version()}")
    print("Packages: " + json.dumps(versions, ensure_ascii=False))
    print(f"CUDA: {cuda} ({cuda_info})")
    print(f"Data directory: {data}")
    print(f"Free disk: {free_gb:.2f} GiB")
    if issues:
        print("环境异常")
        for issue in issues:
            print(f"- {issue}")
        print('Install: pip install -e ".[train]"')
        return 1
    print("环境正常")
    return 0


def _prepare(aoi_id: str, manifest_arg: str | None, force: bool, *, new_run: bool) -> tuple[Any, Path, dict, Path | None]:
    import yaml

    from .preparation import prepare_project
    from .project_discovery import discover_project, resolve_manifest

    project = discover_project(aoi_id, REPO_ROOT / "data")
    manifest, manifest_source = resolve_manifest(project, manifest_arg)
    data_hash = compute_data_hash(project)
    latest = _latest(aoi_id)
    source_run = Path(latest["run_directory"]) if latest else None
    source_cache = _cache(source_run) if source_run else {}
    if not force and not new_run and source_run and source_cache.get("data_hash") == data_hash and (source_run / "prepared" / "coco").exists():
        _stage(source_run, "prepare", "success", "reused", manifest_source=manifest_source)
        config = _resolved_config(project, source_run, manifest)
        return project, source_run, config, source_run
    run = _unique_run(aoi_id)
    config = _resolved_config(project, run, manifest)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        if not force and source_run and source_cache.get("data_hash") == data_hash and (source_run / "prepared" / "coco").exists():
            _clone_tree(source_run / "prepared", run / "prepared")
            _clone_tree(source_run / "data_report", run / "data_report")
            cache_status = "reused"
        else:
            prepare_project(project, run)
            cache_status = "invalidated" if source_run and not force else "recomputed"
        cache = {"data_hash": data_hash, "source_run": str(source_run or "")}
        _write_cache(run, cache)
        input_files = [*project.images.values(), *project.labels.values()]
        project_yaml = project.project_dir / "project.yaml"
        if project_yaml.exists():
            input_files.append(project_yaml)
        _update_manifest(
            run,
            manifest_path=str(manifest),
            manifest_source=manifest_source,
            input_files=[str(path) for path in input_files],
            input_hashes={str(path): file_sha256(path) for path in input_files},
            resolved_config_hash=file_sha256(run / "resolved_config.yaml"),
        )
        _stage(run, "prepare", "success", cache_status, manifest=str(manifest), manifest_source=manifest_source)
        return project, run, config, source_run
    except Exception as exc:
        fail_run(run, "prepare", exc)
        raise


def _training_hash(config: dict, run: Path) -> str:
    coco = run / "prepared" / "coco"
    return _json_hash({
        "model": config["model"], "training": config["training"],
        "train_coco": file_sha256(coco / "annotations_train.json"),
        "val_coco": file_sha256(coco / "annotations_val.json"),
    })


def _train(project, run: Path, config: dict, source_run: Path | None, *, force: bool, resume: bool) -> Path:
    from .preflight import run_preflight
    from .training import train_teacher

    run_preflight(project, run)
    cache = _cache(run)
    source_cache = _cache(source_run) if source_run else {}
    training_hash = _training_hash(config, run)
    if not force and not resume and source_run and source_cache.get("training_hash") == training_hash and (source_run / "training" / "checkpoints" / "best.pth").exists():
        _clone_tree(source_run / "training", run / "training")
        _clone_tree(source_run / "evaluation", run / "evaluation")
        best = run / "training" / "checkpoints" / "best.pth"
        status = "reused"
    else:
        training = config["training"]
        model = config["model"]
        should_resume = resume or bool(training.get("resume", False))
        if should_resume and source_run and (source_run / "training" / "checkpoints" / "last.pth").exists():
            _copy_tree(source_run / "training", run / "training")
        best = train_teacher(
            run / "prepared" / "coco", run / "training",
            epochs=int(training["epochs"]), batch_size=training["batch_size"],
            learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
            num_workers=training["num_workers"], num_classes=int(model["num_classes"]),
            pretrained=bool(model["pretrained"]), weights_path=model.get("weights_path"),
            allow_pretrained_download=bool(model.get("allow_pretrained_download", False)),
            amp=bool(training["amp"]), gradient_clip_norm=float(training.get("gradient_clip_norm", 5.0)),
            early_stopping_patience=int(training["early_stopping_patience"]),
            save_every_epochs=int(training["save_every_epochs"]), resume=should_resume,
            seed=int(training["seed"]), min_size=int(model["min_size"]), max_size=int(model["max_size"]),
            freeze_backbone=bool(model.get("freeze_backbone", False)),
        )
        status = "invalidated" if source_run else "recomputed"
    cache["training_hash"] = training_hash
    if status == "reused" and source_cache.get("evaluation_hash"):
        cache["evaluation_hash"] = source_cache["evaluation_hash"]
    _write_cache(run, cache)
    _stage(run, "train", "success", status, best_checkpoint=str(best))
    return best


def _evaluate(config: dict, run: Path, best: Path, *, force: bool) -> None:
    from .validation import evaluate_checkpoint

    cache = _cache(run)
    evaluation_hash = _json_hash({
        "checkpoint": file_sha256(best),
        "score_threshold": config["inference"]["score_threshold"],
        "mask_threshold": config["inference"]["mask_threshold"],
    })
    if not force and cache.get("evaluation_hash") == evaluation_hash and (run / "evaluation" / "metrics.json").exists():
        status = "reused"
    else:
        evaluate_checkpoint(
            run / "prepared" / "coco", run / "prepared", best, run / "evaluation",
            num_classes=int(config["model"]["num_classes"]), min_size=int(config["model"]["min_size"]),
            max_size=int(config["model"]["max_size"]), score_threshold=float(config["inference"]["score_threshold"]),
            mask_threshold=float(config["inference"]["mask_threshold"]),
        )
        status = "recomputed"
    cache["evaluation_hash"] = evaluation_hash
    _write_cache(run, cache)
    _stage(run, "evaluate", "success", status, checkpoint=str(best))


def _predict(project, config: dict, run: Path, best: Path, source_run: Path | None, *, force: bool) -> None:
    from .inference import infer_full_timeseries

    cache = _cache(run)
    source_cache = _cache(source_run) if source_run else {}
    inference_hash = _json_hash({"checkpoint": file_sha256(best), "inference": config["inference"], "rgb_bands": config["data"]["rgb_bands"]})
    if not force and source_run and source_cache.get("inference_hash") == inference_hash and (source_run / "predictions" / "inference_summary.csv").exists():
        _clone_tree(source_run / "predictions", run / "predictions")
        status = "reused"
    else:
        infer_full_timeseries(
            run / "prepared", sorted(project.images), run / "predictions", checkpoint=best,
            chip_size=int(config["tiling"]["tile_size"]), overlap=int(config["tiling"]["overlap"]),
            score_threshold=float(config["inference"]["score_threshold"]),
            mask_threshold=float(config["inference"]["mask_threshold"]),
            nms_iou_threshold=float(config["inference"]["nms_iou_threshold"]),
            model_version=config["model_version"], rgb_bands=config["data"]["rgb_bands"],
            regularization=config["regularization"], min_size=int(config["model"]["min_size"]), max_size=int(config["model"]["max_size"]),
        )
        status = "invalidated" if source_run else "recomputed"
    cache["inference_hash"] = inference_hash
    _write_cache(run, cache)
    _stage(run, "predict", "success", status, checkpoint=str(best))


def _track(project, config: dict, run: Path, source_run: Path | None, *, force: bool) -> None:
    from .manifest import load_manifest
    from .review import build_review_queue

    cache = _cache(run)
    source_cache = _cache(source_run) if source_run else {}
    tracking_hash = _json_hash({
        "tracking": config["tracking"], "confidence": config["confidence"],
        "predictions": file_sha256(run / "predictions" / "inference_summary.csv"),
    })
    if not force and source_run and source_cache.get("tracking_hash") == tracking_hash and (source_run / "review" / "index.html").exists():
        _clone_tree(source_run / "tracking", run / "tracking")
        _clone_tree(source_run / "review", run / "review")
        status = "reused"
    else:
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "track_states.py"), "--config", str(run / "resolved_config.yaml"), "--force"], cwd=REPO_ROOT, check=True)
        summary = run / "tracking" / "tracks_summary.gpkg"
        build_review_queue(
            summary,
            load_manifest(config["manifest"], config["data_root"]),
            run / "review",
            padding_m=float(config["review"]["chip_padding_m"]),
            long_path=run / "tracking" / "tracks_long.gpkg",
            lineage_path=run / "tracking" / "lineage_events.gpkg",
        )
        status = "invalidated" if source_run else "recomputed"
    cache["tracking_hash"] = tracking_hash
    _write_cache(run, cache)
    reviewed = (run / "tracking" / "tracks_reviewed.gpkg").exists()
    _update_manifest(run, review_source="reviewed" if reviewed else "auto")
    _stage(run, "track", "success", status, review_source="reviewed" if reviewed else "auto")


def _report(run: Path) -> Path:
    best = run / "training" / "checkpoints" / "best.pth"
    report = run / "reports" / "RUN_REPORT.md"
    lines = [
        "# v0.4 run report", "", f"- Run: `{run.name}`", f"- Best checkpoint: `{best}`",
        f"- Evaluation: `{run / 'evaluation' / 'evaluation_summary.md'}`",
        f"- Predictions: `{run / 'predictions'}`", f"- Tracking: `{run / 'tracking'}`",
        f"- Review: `{run / 'review' / 'index.html'}`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _stage(run, "report", "success", "recomputed")
    return report


def _demo_data(force: bool = False, aoi_id: str = "demo_aoi") -> str:
    import geopandas as gpd
    import numpy as np
    import rasterio
    import yaml
    from rasterio.transform import from_origin
    from shapely.geometry import box

    root = REPO_ROOT / "data" / aoi_id
    marker = root / ".generated_demo"
    if marker.exists() and not force:
        return aoi_id
    if root.exists() and any(root.iterdir()) and not marker.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite non-demo data: {root}; use demo --force")
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    dates = ["2022-04-01", "2023-04-01", "2024-04-01"]
    for date_index, date in enumerate(dates):
        data = np.full((3, 64, 64), 40, dtype=np.uint8)
        geometries = [
            box(500004, 4400040, 500012, 4400048), box(500036, 4400040, 500044, 4400048),
            box(500004, 4400008, 500012, 4400016), box(500036, 4400008, 500044 + date_index * 2, 4400016),
        ]
        for geometry in geometries:
            minx, miny, maxx, maxy = geometry.bounds
            row0, row1 = int(64 - maxy + 4400000), int(64 - miny + 4400000)
            col0, col1 = int(minx - 500000), int(maxx - 500000)
            data[:, row0:row1, col0:col1] = 180
        with rasterio.open(
            root / "images" / f"{date}.tif", "w", driver="GTiff", height=64, width=64,
            count=3, dtype="uint8", crs="EPSG:32650", transform=from_origin(500000, 4400064, 1, 1),
        ) as target:
            target.write(data)
        gpd.GeoDataFrame(geometry=geometries, crs="EPSG:32650").to_file(root / "labels" / f"{date}.gpkg", driver="GPKG")
    project_yaml = {
        "project": {"name": "v0.4 CPU demo"},
        "data": {"min_building_area_m2": 1.0, "rgb_bands": [1, 2, 3]},
        "tiling": {"tile_size": 32, "overlap": 0, "min_object_visibility": 0.2, "include_empty_tiles": True, "max_empty_to_positive_ratio": 0.0},
        "split": {"block_size_m": 32, "validation_fraction": 0.25, "seed": 42},
        "model": {"pretrained": False, "min_size": 64, "max_size": 64, "freeze_backbone": True},
        "training": {"epochs": 1, "batch_size": 1, "num_workers": 0, "amp": False, "resume": False, "early_stopping_patience": 1},
        "inference": {"score_threshold": 0.99, "mask_threshold": 0.5, "nms_iou_threshold": 0.4},
        "regularization": {"min_area_m2": 1.0},
        "review": {"chip_padding_m": 8.0},
    }
    (root / "project.yaml").write_text(yaml.safe_dump(project_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    marker.write_text("generated by python run.py demo\n", encoding="utf-8")
    return aoi_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Urban renewal v0.4 plug-and-train workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--data-root", default="data")
    for name in ("prepare", "train", "evaluate", "predict", "track", "all"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--aoi", required=True)
        sub.add_argument("--manifest")
        sub.add_argument("--force", action="store_true")
        if name == "train":
            sub.add_argument("--resume", action="store_true")
        if name == "predict":
            sub.add_argument("--checkpoint")
    demo = subparsers.add_parser("demo")
    demo.add_argument("--force", action="store_true")
    demo.add_argument("--aoi", default="demo_aoi")
    return parser


def main(argv: list[str] | None = None) -> int:
    routed = list(sys.argv[1:] if argv is None else argv)
    if routed and routed[0].startswith("change-"):
        from .change.cli import main as change_main
        return change_main(routed)
    args = build_parser().parse_args(routed)
    if args.command == "doctor":
        return doctor(args.data_root)
    if args.command == "demo":
        aoi = _demo_data(args.force, args.aoi)
        args = argparse.Namespace(command="all", aoi=aoi, manifest=None, force=args.force)
    if args.command == "all" and doctor() != 0:
        return 1
    run = None
    stage = "prepare"
    try:
        new_run = args.command != "prepare"
        project, run, config, source_run = _prepare(args.aoi, getattr(args, "manifest", None), args.force, new_run=new_run)
        if args.command in {"evaluate", "predict", "track"} and source_run:
            _clone_tree(source_run / "training", run / "training")
            _clone_tree(source_run / "evaluation", run / "evaluation")
        if args.command == "track" and source_run:
            _clone_tree(source_run / "predictions", run / "predictions")
        if args.command == "prepare":
            if source_run != run or json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))["status"] != "success":
                complete_run(run)
            print(run)
            return 0
        best = run / "training" / "checkpoints" / "best.pth"
        if args.command in {"train", "all"}:
            stage = "train"
            best = _train(project, run, config, source_run, force=args.force, resume=getattr(args, "resume", False))
            stage = "evaluate"
            _evaluate(config, run, best, force=args.force)
            if args.command == "train":
                complete_run(run, best)
                print(f"Best model:\n{best}")
                return 0
        elif not best.exists():
            raise FileNotFoundError(f"No best.pth in latest successful run: {best}; run train first")
        if args.command == "evaluate":
            stage = "evaluate"
            _evaluate(config, run, best, force=args.force)
            complete_run(run, best)
            print(run / "evaluation" / "evaluation_summary.md")
            return 0
        if args.command in {"predict", "all"}:
            stage = "predict"
            explicit = getattr(args, "checkpoint", None)
            if explicit:
                best = Path(explicit).resolve()
            _predict(project, config, run, best, source_run, force=args.force)
            if args.command == "predict":
                complete_run(run, best)
                print(run / "predictions" / "inference_summary.csv")
                return 0
        if args.command in {"track", "all"}:
            stage = "track"
            _track(project, config, run, source_run, force=args.force)
            if args.command == "track":
                complete_run(run, best)
                print(run / "review" / "index.html")
                return 0
        stage = "report"
        _report(run)
        complete_run(run, best)
        if "demo" in sys.argv:
            print("Demo completed successfully.")
            print(f"\nBest model:\n{best}")
            print(f"\nEvaluation:\n{run / 'evaluation' / 'evaluation_summary.md'}")
            print(f"\nReview page:\n{run / 'review' / 'index.html'}")
        else:
            print(run)
        return 0
    except Exception as exc:
        if run is not None and (run / "RUN_MANIFEST.json").exists():
            fail_run(run, stage, exc)
        print(f"ERROR [{stage}]: {exc}", file=sys.stderr)
        return 1
