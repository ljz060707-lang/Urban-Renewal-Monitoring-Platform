from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import yaml

from .corrections import apply_review_corrections
from .evaluation import evaluate_change_checkpoint
from .local_labels import generate_local_change_candidates
from .preparation import create_change_run, prepare_public_change
from .registry import enabled_adapters, load_change_config
from .review import build_static_review
from .training import STAGE_FILES, train_change_pipeline, train_change_stage
from .visualization import build_training_visualizations


REPO_ROOT = Path(__file__).resolve().parents[3]


def _now(): return datetime.now(timezone.utc).isoformat()


def _pointer_root(config: dict) -> Path:
    root = Path(config.get("output_root", "outputs/pretraining/change"))
    return root if root.is_absolute() else REPO_ROOT / root


def _write_pointer(config: dict, run: Path, status: str, best: Path | None = None):
    payload = {"run_id": run.name, "run_directory": str(run.resolve()), "status": status, "best_checkpoint": str(best.resolve()) if best else "", "updated_at": _now()}
    root = _pointer_root(config); root.mkdir(parents=True, exist_ok=True)
    filename = "latest.json" if status == "success" else "latest_prepared.json"
    (root / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest(config: dict) -> Path:
    pointer = _pointer_root(config) / "latest.json"
    if not pointer.is_file():
        pointer = _pointer_root(config) / "latest_prepared.json"
    if not pointer.is_file(): raise FileNotFoundError(f"No successful/prepared change run pointer under {_pointer_root(config)}; run change-prepare first")
    run = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_directory"])
    if not run.is_dir(): raise FileNotFoundError(run)
    return run


def _resolve_run(value: str, config: dict) -> Path:
    return _latest(config) if value == "latest" else Path(value).resolve()


def change_doctor(config_path: str | Path, *, require_gpu: bool = True) -> int:
    checks, errors, warnings = {}, [], []
    try:
        config = load_change_config(config_path)
    except Exception as exc:
        print(f"ERROR: cannot load config: {exc}"); return 1
    for name in ("torch", "torchvision", "numpy", "pandas", "PIL", "rasterio", "geopandas"):
        try:
            module = importlib.import_module(name); checks[name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc: errors.append(f"missing package {name}: {exc}")
    gpu = {"available": False, "torch_version": checks.get("torch"), "cuda_runtime": None, "cudnn": None, "devices": []}
    if "torch" in checks:
        import torch
        gpu.update(available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(index); gpu["devices"].append({"index": index, "name": prop.name, "memory_bytes": prop.total_memory})
        elif require_gpu: errors.append("CUDA unavailable for formal training; CPU fallback is disabled")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        checks["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except Exception as exc: warnings.append(f"nvidia-smi not available: {exc}")
    datasets = []
    try:
        for adapter in enabled_adapters(config):
            records = adapter.discover(); datasets.append({"name": adapter.dataset_name, "root": str(adapter.root), "samples": len(records)})
    except Exception as exc: errors.append(f"dataset check failed: {exc}")
    output = _pointer_root(config); output.mkdir(parents=True, exist_ok=True)
    try:
        probe = output / ".write_test"; probe.write_text("ok"); probe.unlink()
    except OSError as exc: errors.append(f"output not writable: {exc}")
    free = shutil.disk_usage(output).free
    if free < 5 * 1024**3: warnings.append(f"free disk below 5 GiB: {free/1024**3:.2f} GiB")
    report = {"status": "failed" if errors else "ok", "python": platform.python_version(), "packages": checks, "gpu": gpu,
              "datasets": datasets, "free_disk_bytes": free, "errors": errors, "warnings": warnings}
    (output / "CHANGE_DOCTOR.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors and require_gpu:
        print("Install a CUDA-enabled Torch build using the command generated at https://pytorch.org/get-started/locally/")
    return 1 if errors else 0


def _prepare(config_path, run_id=None):
    config = load_change_config(config_path)
    run = prepare_public_change(config_path, run_id=run_id)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({key:value for key,value in config.items() if key != "_config_path"}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_pointer(config, run, "prepared")
    return config, run


def _report(run: Path, results: dict):
    lines = ["# Change teacher run report", "", f"- run: `{run.name}`"]
    for stage, checkpoint in results.items(): lines.append(f"- {stage}: `{checkpoint}`")
    lines += [f"- metrics: `{run / 'metrics'}`", f"- dashboard: `{run / 'training_dashboard.html'}`", f"- review: `{run / 'review' / 'index.html'}`"]
    path = run / "reports" / "training_report.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return path


def _mark_failed_run(run: Path, command: str, exc: Exception) -> None:
    """Keep an auditable failed run without advancing the success pointer."""
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    traceback_path = logs / "traceback.txt"
    traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
    manifest_path = run / "RUN_MANIFEST.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        status="failed",
        failure_stage=command,
        error_message=str(exc),
        traceback_path=str(traceback_path.resolve()),
        failed_at=_now(),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _synthetic_config(force: bool = False):
    import numpy as np
    from PIL import Image
    root = REPO_ROOT / "data" / "change_demo"
    if root.exists() and force: shutil.rmtree(root)
    def save(path, value): path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(value.astype(np.uint8)).save(path)
    for dataset in ("s2looking", "second", "dsifn"):
        for split in ("train", "val"):
            image1 = np.full((64,64,3), 35, np.uint8); image2 = image1.copy(); image2[20:44,20:44] = 190
            cd = np.zeros((64,64), np.uint8); cd[20:44,20:44] = 255
            base = root / dataset / split
            if dataset == "s2looking":
                for name,value in (("Image1",image1),("Image2",image2),("label",cd),("label1",cd),("label2",np.zeros_like(cd))): save(base/name/"sample.png",value)
            elif dataset == "second":
                gt1=np.zeros_like(cd); gt2=(cd>0).astype(np.uint8)
                for name,value in (("T1",image1),("T2",image2),("GT_T1",gt1),("GT_T2",gt2)): save(base/name/"sample.png",value)
            else:
                for name,value in (("T1",image1),("T2",image2),("GT",cd)): save(base/name/"sample.png",value)
    config = {
        "schema_version":"0.5.0", "output_root":str(REPO_ROOT/"outputs"/"pretraining"/"change_demo"),
        "datasets":{"s2looking":{"enabled":True,"root":str(root/"s2looking")},
                    "second":{"enabled":True,"root":str(root/"second"),"building_class_id":1,"mapping_verified":True},
                    "dsifn":{"enabled":True,"root":str(root/"dsifn")}},
        "audit":{"preview_count":2,"second_preview_count":2},
        "model":{"architecture":"multitask_siamese_unet","semantic_classes":2,"pretrained":False,"dropout":0},
        "training":{"epochs":1,"batch_size":1,"num_workers":0,"patch_size":64,"samples_per_epoch":3,"amp":False,"learning_rate":1e-4,"early_stopping_patience":2,"seed":4,
                    "dataset_sampling":{"s2looking":2.0,"second":1.0,"dsifn":1.0}},
    }
    path = root / "change_demo.yaml"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def build_parser():
    parser = argparse.ArgumentParser(description="v0.5 multi-teacher change training")
    subs = parser.add_subparsers(dest="command", required=True)
    doctor = subs.add_parser("change-doctor"); doctor.add_argument("--config", default="configs/change_public.yaml")
    prepare = subs.add_parser("change-prepare"); prepare.add_argument("--config", default="configs/change_public.yaml"); prepare.add_argument("--run-id")
    train = subs.add_parser("change-train"); train.add_argument("--config", default="configs/change_public.yaml"); train.add_argument("--stage", choices=["binary","semantic","directional","all"], default="all"); train.add_argument("--device", default="cuda:0"); train.add_argument("--resume", action="store_true"); train.add_argument("--distributed", action="store_true"); train.add_argument("--run", default="latest")
    all_parser = subs.add_parser("change-all"); all_parser.add_argument("--config", default="configs/change_public.yaml"); all_parser.add_argument("--device", default="cuda:0"); all_parser.add_argument("--resume", action="store_true"); all_parser.add_argument("--distributed", action="store_true"); all_parser.add_argument("--quick", action="store_true")
    demo = subs.add_parser("change-demo"); demo.add_argument("--device", default="cpu"); demo.add_argument("--force", action="store_true")
    review = subs.add_parser("change-review"); review.add_argument("--config", default="configs/change_public.yaml"); review.add_argument("--run", default="latest"); review.add_argument("--interactive", action="store_true")
    retrain = subs.add_parser("change-retrain"); retrain.add_argument("--config", default="configs/change_public.yaml"); retrain.add_argument("--run", default="latest"); retrain.add_argument("--from-review", action="store_true", required=True); retrain.add_argument("--device", default="cuda:0")
    labels = subs.add_parser("change-labels"); labels.add_argument("--aoi", required=True); labels.add_argument("--run")
    finetune = subs.add_parser("change-finetune"); finetune.add_argument("--aoi", required=True); finetune.add_argument("--config", default="configs/change_public.yaml"); finetune.add_argument("--checkpoint", default="latest-directional"); finetune.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    active_run = None
    try:
        if args.command == "change-doctor": return change_doctor(args.config, require_gpu=True)
        if args.command == "change-prepare":
            _, run = _prepare(args.config, args.run_id); active_run = run; print(run); return 0
        if args.command == "change-demo":
            if args.device != "cpu": raise ValueError("change-demo is the only CPU path and must use --device cpu")
            config_path = _synthetic_config(args.force); config, run = _prepare(config_path); active_run = run
            index = run / "prepared" / "public_change_index.csv"
            results = train_change_pipeline(config, index, run, device_name="cpu", allow_cpu=True)
            evaluate_change_checkpoint(results["directional"], index, run, device_name="cpu")
            build_training_visualizations(run); review = build_static_review(run); _report(run, results)
            manifest = json.loads((run/"RUN_MANIFEST.json").read_text(encoding="utf-8")); manifest.update(status="success",completed_at=_now(),checkpoints={k:str(v) for k,v in results.items()}); (run/"RUN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            _write_pointer(config, run, "success", results["directional"])
            print(f"Change demo completed successfully.\nBest model:\n{results['directional']}\nDashboard:\n{run/'training_dashboard.html'}\nReview page:\n{review}"); return 0
        if args.command == "change-labels":
            output = generate_local_change_candidates(args.aoi, repo_root=REPO_ROOT, run=args.run); print(output); return 0
        config = load_change_config(args.config)
        if args.command == "change-all":
            if change_doctor(args.config, require_gpu=True): return 1
            config, run = _prepare(args.config); active_run = run; index=run/"prepared"/"public_change_index.csv"
            if args.quick:
                config.setdefault("training", {})["epochs"] = 1
            results=train_change_pipeline(config,index,run,device_name=args.device,resume=args.resume,distributed=args.distributed)
            evaluate_change_checkpoint(results["directional"],index,run,device_name=args.device); build_training_visualizations(run); build_static_review(run); _report(run,results)
            manifest=json.loads((run/"RUN_MANIFEST.json").read_text(encoding="utf-8"));manifest.update(status="success",completed_at=_now(),checkpoints={k:str(v) for k,v in results.items()});(run/"RUN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8");_write_pointer(config,run,"success",results["directional"]);print(run);return 0
        run = _resolve_run(args.run if hasattr(args,"run") else "latest", config)
        index = run / "prepared" / "public_change_index.csv"
        if args.command == "change-train":
            stages=["binary","semantic","directional"] if args.stage=="all" else [args.stage]
            results=train_change_pipeline(config,index,run,stages=stages,device_name=args.device,resume=args.resume,distributed=args.distributed);print("\n".join(str(path) for path in results.values()));return 0
        if args.command == "change-review":
            path=build_static_review(run);print(path)
            if args.interactive:
                from .review_app import launch_review_app; launch_review_app(run)
            return 0
        if args.command == "change-retrain":
            decisions=run/"review"/"review_decisions.csv"; reviewed=apply_review_corrections(index,decisions,run)
            new_run=create_change_run(_pointer_root(config)); previous=run/"checkpoints"/STAGE_FILES["directional"]
            best=train_change_stage(config,reviewed,new_run,"directional",device_name=args.device,initial_checkpoint=previous);print(best);return 0
        if args.command == "change-finetune":
            pointer = REPO_ROOT/"outputs"/args.aoi/"latest.json"; local_run=Path(json.loads(pointer.read_text(encoding="utf-8"))["run_directory"])
            local_index=local_run/"change_labels"/"local_change_index.reviewed.csv"
            if not local_index.is_file(): raise FileNotFoundError(f"No human-confirmed local change index: {local_index}")
            checkpoint=run/"checkpoints"/STAGE_FILES["directional"] if args.checkpoint=="latest-directional" else Path(args.checkpoint)
            target=create_change_run(REPO_ROOT/"outputs"/args.aoi/"change_finetune")
            best=train_change_stage(config,local_index,target,"local_finetune",device_name=args.device,initial_checkpoint=checkpoint);print(best);return 0
        raise ValueError(args.command)
    except Exception as exc:
        if active_run is not None:
            _mark_failed_run(active_run, args.command, exc)
        print(f"ERROR [{args.command}]: {exc}", file=sys.stderr)
        return 1
