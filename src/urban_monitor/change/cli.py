from __future__ import annotations

import argparse
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

import pandas as pd

import yaml

from .corrections import apply_review_corrections
from .compare import compare_runs
from .diagnostics import binary_stage_diagnostic
from .evaluation import evaluate_change_checkpoint
from .hard_examples import build_hard_examples, build_demolition_head_sampling_index
from .ai_review import run_ai_review
from .local_labels import generate_local_change_candidates
from .optimization import audit_gradients, audit_training_signal, benchmark_loader, experiment_summary, profile_training
from .preparation import create_change_run, prepare_public_change
from .preflight import run_change_preflight
from .registry import enabled_adapters, load_change_config
from .review import build_static_review
from .train_review import generate_train_review
from .training import STAGE_FILES, train_change_pipeline, train_change_stage
from .splits import quick_subset
from .final_test import run_final_test
from .manual_diagnostic_review import build_manual_diagnostic_review
from .thresholds import scan_thresholds
from .visualization import build_training_visualizations
from .preflight_v2 import format_v2_preflight, run_v2_preflight
from .inference_v2 import run_urban_renewal_inference
from .hard_examples_v2 import mine_v2_hard_examples


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
    if value == "latest":
        return _latest(config)
    path = Path(value)
    if path.is_absolute() or len(path.parts) > 1:
        return path.resolve()
    candidate = _pointer_root(config) / value
    return candidate.resolve() if candidate.is_dir() else path.resolve()


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
            datasets.append({"name": adapter.dataset_name, "root": str(adapter.root), "samples": "preflight_required"})
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


def _prepare(config_path, run_id=None, quick: bool = False):
    config = load_change_config(config_path)
    run = prepare_public_change(config_path, run_id=run_id, quick=quick)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({key:value for key,value in config.items() if key != "_config_path"}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _write_pointer(config, run, "prepared")
    return config, run


def _report(run: Path, results: dict):
    lines = ["# Change teacher run report", "", f"- run: `{run.name}`"]
    for stage, checkpoint in results.items(): lines.append(f"- {stage}: `{checkpoint}`")
    lines += [f"- metrics: `{run / 'metrics'}`", f"- dashboard: `{run / 'training_dashboard.html'}`", f"- review: `{run / 'review' / 'index.html'}`"]
    path = run / "reports" / "training_report.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return path


def _final_checkpoint(results: dict[str, Path]) -> Path:
    if not results:
        raise ValueError("No public training stage completed")
    return next(reversed(results.values()))


def _record_stage_manifest(run: Path, results: dict[str, Path]) -> None:
    path = run / "RUN_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    states_path = run / "training_stage_status.json"
    states = json.loads(states_path.read_text(encoding="utf-8")) if states_path.is_file() else {}
    payload["stages"] = states
    payload["checkpoints"] = {stage: str(value) for stage, value in results.items()}
    payload["final_checkpoint"] = str(_final_checkpoint(results))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_completed(run: Path, index: Path, checkpoint: Path, device: str) -> None:
    evaluate_change_checkpoint(checkpoint, index, run, device_name=device)
    frame = pd.read_csv(index, dtype=str).fillna("")
    if (frame["split"].astype(str).str.casefold() == "test").any():
        evaluate_change_checkpoint(checkpoint, index, run, device_name=device, split_scope="test")


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
    log_path = run / "metrics" / "train_log_directional.csv"
    last_epoch = None
    if log_path.is_file():
        try:
            last_epoch = int(pd.read_csv(log_path).iloc[-1]["epoch"])
        except Exception:
            pass
    manifest.update(
        status="failed",
        failure_stage=command,
        error_message=str(exc),
        traceback_path=str(traceback_path.resolve()),
        last_completed_epoch=last_epoch,
        failed_at=_now(),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_complete_run(run: Path, command: str) -> dict:
    manifest_path = run / "RUN_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{command} refuses incomplete run without RUN_MANIFEST.json: {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "success":
        raise RuntimeError(f"{command} requires status=success, got {manifest.get('status')!r}: {run}")
    return manifest


def _inherit_candidate_index(source_run: Path, candidate_run: Path) -> dict:
    """Copy the immutable split index; candidates must never re-split data."""
    source = source_run / "prepared" / "public_change_index.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Baseline index required for retrain: {source}")
    target = candidate_run / "prepared" / "public_change_index.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash, target_hash = _sha256(source), _sha256(target)
    if source_hash != target_hash:
        raise RuntimeError("Candidate index hash differs from baseline; refusing retrain")
    frame = pd.read_csv(target, dtype=str).fillna("")
    counts = frame["split"].astype(str).str.casefold().value_counts().to_dict()
    return {"source_run_id": source_run.name, "source_index_path": str(source.resolve()),
            "candidate_index_path": str(target.resolve()), "index_sha256": target_hash, "split_counts": counts}


def _resolve_checkpoint(run: Path, manifest: dict, kind: str, path: str | None) -> Path:
    if path:
        checkpoint = Path(path)
    elif kind == "last":
        checkpoint = run / "checkpoints" / "last_directional.pth"
    else:
        checkpoint = Path(manifest["final_checkpoint"])
        if not checkpoint.is_absolute(): checkpoint = REPO_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Requested {kind} checkpoint does not exist: {checkpoint}")
    return checkpoint.resolve()


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
    preflight = subs.add_parser("change-preflight"); preflight.add_argument("--config", default="configs/change_public.yaml")
    prepare = subs.add_parser("change-prepare"); prepare.add_argument("--config", default="configs/change_public.yaml"); prepare.add_argument("--run-id")
    train = subs.add_parser("change-train"); train.add_argument("--config", default="configs/change_public.yaml"); train.add_argument("--stage", choices=["binary","semantic","directional","all"], default="all"); train.add_argument("--device", default="cuda:0"); train.add_argument("--resume", action="store_true"); train.add_argument("--distributed", action="store_true"); train.add_argument("--run", default="latest")
    all_parser = subs.add_parser("change-all"); all_parser.add_argument("--config", default="configs/change_public.yaml"); all_parser.add_argument("--device", default="cuda:0"); all_parser.add_argument("--resume", action="store_true"); all_parser.add_argument("--distributed", action="store_true"); all_parser.add_argument("--quick", action="store_true")
    demo = subs.add_parser("change-demo"); demo.add_argument("--device", default="cpu"); demo.add_argument("--force", action="store_true")
    review = subs.add_parser("change-review"); review.add_argument("--config", default="configs/change_public.yaml"); review.add_argument("--run", default="latest"); review.add_argument("--interactive", action="store_true"); review.add_argument("--ai-assist", action="store_true"); review.add_argument("--split", choices=["train", "validation"], default="validation"); review.add_argument("--selection", choices=["hard", "random", "all"], default="all"); review.add_argument("--max-samples", type=int, default=100); review.add_argument("--device", default="cuda:0"); review.add_argument("--output-dir"); review.add_argument("--force", action="store_true")
    scan = subs.add_parser("change-threshold-scan"); scan.add_argument("--config", default="configs/change_public.yaml"); scan.add_argument("--run", default="latest"); scan.add_argument("--split", default="validation", choices=["validation"]); scan.add_argument("--device", default="cuda:0"); scan.add_argument("--checkpoint-kind", choices=["best", "last"], default="best"); scan.add_argument("--checkpoint-path")
    final_test = subs.add_parser("change-final-test", help="One-time held-out test evaluation with fixed thresholds")
    final_test.add_argument("--config", default="configs/change_public.yaml")
    final_test.add_argument("--run", default="latest")
    final_test.add_argument("--checkpoint-kind", choices=["best", "last"], default="best")
    final_test.add_argument("--device", default="cuda:0")
    final_test.add_argument("--binary-threshold", type=float, required=True)
    final_test.add_argument("--new-threshold", type=float, default=None,
                            help="Required only when the held-out split has directional supervision")
    final_test.add_argument("--demolition-threshold", type=float, default=None,
                            help="Required only when the held-out split has directional supervision")
    final_test.add_argument("--confirm-final-test", action="store_true", help="Acknowledge this one-time read of split=test")
    diagnostic = subs.add_parser("change-manual-diagnostic-review", help="Build offline DSIFN test/validation diagnostic page from existing assets")
    diagnostic.add_argument("--config", default="configs/change_public_dsifn_windows.yaml")
    diagnostic.add_argument("--run", default="20260725_232317")
    diagnostic.add_argument("--validation-limit", type=int, default=80)
    hard = subs.add_parser("change-hard-examples"); hard.add_argument("--config", default="configs/change_public.yaml"); hard.add_argument("--run", default="latest"); hard.add_argument("--review-csv")
    ai = subs.add_parser("change-ai-review"); ai.add_argument("--config", default="configs/change_public.yaml"); ai.add_argument("--run", default="latest"); ai.add_argument("--model", default="glm-4.6v-flash"); ai.add_argument("--split", default="validation"); ai.add_argument("--max-samples", type=int, default=100); ai.add_argument("--dry-run", action="store_true"); ai.add_argument("--force-refresh", action="store_true"); ai.add_argument("--resume", action="store_true")
    retrain = subs.add_parser("change-retrain"); retrain.add_argument("--config", default="configs/change_public.yaml"); retrain.add_argument("--run", default="latest"); retrain.add_argument("--from-review", action="store_true"); retrain.add_argument("--device", default="cuda:0")
    compare = subs.add_parser("change-compare"); compare.add_argument("--config", default="configs/change_public.yaml"); compare.add_argument("--baseline-run", required=True); compare.add_argument("--candidate-run", default="latest"); compare.add_argument("--split", default="validation", choices=["validation"]); compare.add_argument("--device", default="cuda:0"); compare.add_argument("--checkpoint-kind", choices=["best", "last"], default="best"); compare.add_argument("--checkpoint-path")
    labels = subs.add_parser("change-labels"); labels.add_argument("--aoi", required=True); labels.add_argument("--run"); labels.add_argument("--data-root", default="data")
    finetune = subs.add_parser("change-finetune"); finetune.add_argument("--aoi", required=True); finetune.add_argument("--config", default="configs/change_public.yaml"); finetune.add_argument("--checkpoint", default="latest-directional"); finetune.add_argument("--device", default="cuda:0"); finetune.add_argument("--data-root", default="data")
    profile = subs.add_parser("change-profile-training"); profile.add_argument("--config", default="configs/change_public_s2looking_windows.yaml"); profile.add_argument("--run", default="20260721_232528"); profile.add_argument("--device", default="cuda:0"); profile.add_argument("--batches", type=int, default=100)
    signal = subs.add_parser("change-signal-audit"); signal.add_argument("--config", default="configs/change_public_s2looking_demolition_head_smoke.yaml"); signal.add_argument("--run", default="20260721_232528"); signal.add_argument("--patches", type=int, default=500)
    gradient = subs.add_parser("change-gradient-audit"); gradient.add_argument("--config", default="configs/change_public_s2looking_windows.yaml"); gradient.add_argument("--run", default="20260721_232528"); gradient.add_argument("--device", default="cuda:0"); gradient.add_argument("--batches", type=int, default=50)
    benchmark = subs.add_parser("change-benchmark-training"); benchmark.add_argument("--config", default="configs/change_public_s2looking_windows.yaml"); benchmark.add_argument("--run", default="20260721_232528"); benchmark.add_argument("--device", default="cuda:0"); benchmark.add_argument("--batches", type=int, default=20)
    summary = subs.add_parser("change-experiment-summary"); summary.add_argument("--config", default="configs/change_public_s2looking_windows.yaml"); summary.add_argument("--baseline-run", default="20260721_232528")
    v2_preflight = subs.add_parser("urban-renewal-preflight")
    v2_preflight.add_argument("--config", default="configs/change_public_v2_multisource_windows.yaml")
    v2_preflight.add_argument("--output", default="outputs/v2_readiness/URBAN_RENEWAL_PREFLIGHT.json")
    v2_infer = subs.add_parser("urban-renewal-infer")
    v2_infer.add_argument("--config", default="configs/change_public_v2_multisource_windows.yaml")
    v2_infer.add_argument("--t1", required=True)
    v2_infer.add_argument("--t2", required=True)
    v2_infer.add_argument("--output", required=True)
    v2_infer.add_argument("--device", default="cuda:0")
    v2_hard = subs.add_parser("change-mine-v2-hard-examples")
    v2_hard.add_argument("--config", default="configs/change_public_v2_multisource_windows.yaml")
    v2_hard.add_argument("--index", required=True)
    v2_hard.add_argument("--output", default="outputs/pretraining/change_v2/hard_examples_train.csv")
    v2_hard.add_argument("--split", choices=["train"], default="train")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    active_run = None
    try:
        if args.command == "urban-renewal-preflight":
            config = load_change_config(args.config)
            report = run_v2_preflight(config, args.output)
            print(format_v2_preflight(report))
            return 0 if report["readiness"]["READY_FOR_PARTIAL_INFERENCE"] else 1
        if args.command == "urban-renewal-infer":
            config = load_change_config(args.config)
            report = run_urban_renewal_inference(
                config, t1_path=args.t1, t2_path=args.t2, output_dir=args.output, device=args.device
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "change-mine-v2-hard-examples":
            report = mine_v2_hard_examples(args.index, args.output, split=args.split)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "change-doctor": return change_doctor(args.config, require_gpu=True)
        if args.command == "change-preflight":
            report = run_change_preflight(args.config, require_cuda=True)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["READY_TO_TRAIN"] else 1
        if args.command == "change-prepare":
            _, run = _prepare(args.config, args.run_id); active_run = run; print(run); return 0
        if args.command == "change-demo":
            if args.device != "cpu": raise ValueError("change-demo is the only CPU path and must use --device cpu")
            config_path = _synthetic_config(args.force); config, run = _prepare(config_path); active_run = run
            index = run / "prepared" / "public_change_index.csv"
            results = train_change_pipeline(config, index, run, device_name="cpu", allow_cpu=True)
            best = _final_checkpoint(results); _evaluate_completed(run, index, best, "cpu")
            build_training_visualizations(run); review = build_static_review(run); _report(run, results)
            _record_stage_manifest(run, results); manifest = json.loads((run/"RUN_MANIFEST.json").read_text(encoding="utf-8")); manifest.update(status="success",completed_at=_now()); (run/"RUN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            _write_pointer(config, run, "success", best)
            print(f"Change demo completed successfully.\nBest model:\n{best}\nDashboard:\n{run/'training_dashboard.html'}\nReview page:\n{review}"); return 0
        if args.command == "change-labels":
            output = generate_local_change_candidates(args.aoi, repo_root=REPO_ROOT, run=args.run, data_root=args.data_root); print(output); return 0
        config = load_change_config(args.config)
        if args.command in {"change-profile-training", "change-signal-audit", "change-gradient-audit", "change-benchmark-training"}:
            run = _resolve_run(args.run, config); index = run / "prepared" / "public_change_index.csv"
            if not index.is_file(): raise FileNotFoundError(index)
            if args.command == "change-profile-training":
                print(json.dumps(profile_training(config, index, run, device_name=args.device, batches=args.batches), ensure_ascii=False, indent=2)); return 0
            if args.command == "change-signal-audit":
                if bool(config.get("training", {}).get("demolition_head_only", False)):
                    hard_path = run / "review" / "hard_examples.csv"
                    if hard_path.is_file():
                        sampling = config.get("sampling", {})
                        index, _ = build_demolition_head_sampling_index(
                            run, hard_path, demolition_only_weight=float(sampling.get("demolition_only_weight", 2.0)),
                            both_weight=float(sampling.get("both_weight", 1.25)), ordinary_weight=float(sampling.get("ordinary_sample_weight", 1.0)),
                            max_hard_fraction=float(sampling.get("max_hard_sample_fraction", .10)),
                            max_both_fraction_within_hard=float(sampling.get("max_both_fraction_within_hard", .40)),
                        )
                print(json.dumps(audit_training_signal(config, index, run, sample_patches=args.patches), ensure_ascii=False, indent=2)); return 0
            if args.command == "change-gradient-audit":
                print(audit_gradients(config, index, run, device_name=args.device, batches=args.batches)); return 0
            print(benchmark_loader(config, index, run, device_name=args.device, batches=args.batches)); return 0
        if args.command == "change-experiment-summary":
            print(experiment_summary(_pointer_root(config), args.baseline_run)); return 0
        if args.command == "change-threshold-scan":
            run = _resolve_run(args.run, config); manifest = _require_complete_run(run, "change-threshold-scan"); checkpoint = _resolve_checkpoint(run, manifest, args.checkpoint_kind, args.checkpoint_path)
            candidate_index = run / "prepared" / "public_change_index.csv"
            if not candidate_index.is_file():
                source = manifest.get("source_run_id") or manifest.get("baseline_run") or "unknown"
                raise FileNotFoundError(f"Candidate index is missing: {candidate_index}. This retrain run should inherit it from source run {source}; rerun change-retrain after the inheritance fix.")
            output = scan_thresholds(checkpoint, candidate_index, run, device_name=args.device, split=args.split)
            if args.split == "validation": binary_stage_diagnostic(run)
            print(output); return 0
        if args.command == "change-final-test":
            run = _resolve_run(args.run, config)
            manifest = _require_complete_run(run, "change-final-test")
            checkpoint = _resolve_checkpoint(run, manifest, args.checkpoint_kind, None)
            candidate_index = run / "prepared" / "public_change_index.csv"
            if not candidate_index.is_file():
                source = manifest.get("source_run_id") or manifest.get("baseline_run") or "unknown"
                raise FileNotFoundError(f"Final-test index is missing: {candidate_index}. Expected inherited source run: {source}.")
            output = run_final_test(
                checkpoint, candidate_index, run, device_name=args.device,
                thresholds={"binary": args.binary_threshold, "new": args.new_threshold, "demolition": args.demolition_threshold},
                confirm_final_test=args.confirm_final_test, cli_config=config,
                expected_checkpoint_sha256=manifest.get("best_checkpoint_sha256") if args.checkpoint_kind == "best" else manifest.get("last_checkpoint_sha256"),
            )
            print(output); return 0
        if args.command == "change-manual-diagnostic-review":
            run = _resolve_run(args.run, config)
            print(json.dumps(build_manual_diagnostic_review(run, validation_limit=args.validation_limit), ensure_ascii=False, indent=2)); return 0
        if args.command == "change-hard-examples":
            run = _resolve_run(args.run, config); sampling = config.get("sampling", {})
            print(build_hard_examples(run, args.review_csv, false_negative_weight=float(sampling.get("hard_false_negative_weight", 3.0)), positive_weight=float(sampling.get("positive_sample_weight", 2.0)), ordinary_weight=float(sampling.get("ordinary_sample_weight", 1.0)))); return 0
        if args.command == "change-ai-review":
            run = _resolve_run(args.run, config); print(run_ai_review(run, model=args.model, split=args.split, max_samples=args.max_samples, dry_run=args.dry_run, force_refresh=args.force_refresh)); return 0
        if args.command == "change-compare":
            baseline, candidate = _resolve_run(args.baseline_run, config), _resolve_run(args.candidate_run, config)
            baseline_manifest = _require_complete_run(baseline, "change-compare baseline"); candidate_manifest = _require_complete_run(candidate, "change-compare candidate")
            candidate_checkpoint = _resolve_checkpoint(candidate, candidate_manifest, args.checkpoint_kind, args.checkpoint_path)
            baseline_checkpoint = _resolve_checkpoint(baseline, baseline_manifest, "best", None)
            print(compare_runs(baseline, candidate, _sha256(baseline_checkpoint), _sha256(candidate_checkpoint))); return 0
        if args.command == "change-all":
            if change_doctor(args.config, require_gpu=True): return 1
            if args.quick and str(config.get("schema_version", "")).startswith("2"):
                preflight = run_v2_preflight(config, _pointer_root(config) / "preflight" / "V2_QUICK_PREFLIGHT.json")
                if not preflight["readiness"]["READY_FOR_V2_QUICK"]:
                    return 1
            else:
                preflight = run_change_preflight(args.config, require_cuda=True)
                if not preflight["READY_TO_TRAIN"]: return 1
            config, run = _prepare(args.config, quick=args.quick); active_run = run; index=run/"prepared"/"public_change_index.csv"
            if args.quick:
                config.setdefault("training", {})["epochs"] = int(config.get("quick", {}).get("epochs", 1))
                quick = quick_subset(pd.read_csv(index, dtype=str).fillna(""), config)
                index = run / "prepared" / "public_change_index.quick.csv"; quick.to_csv(index, index=False, encoding="utf-8-sig")
                (run / "resolved_config.yaml").write_text(yaml.safe_dump({key:value for key,value in config.items() if key != "_config_path"}, allow_unicode=True, sort_keys=False), encoding="utf-8")
            results=train_change_pipeline(config,index,run,device_name=args.device,resume=args.resume,distributed=args.distributed)
            best = _final_checkpoint(results)
            # Quick is a supervised train/validation smoke: it deliberately
            # avoids the prediction-style evaluator, which reads held-out test
            # samples and writes prediction artifacts.
            if not args.quick:
                _evaluate_completed(run,index,best,args.device)
            build_training_visualizations(run)
            if not args.quick:
                build_static_review(run)
            _report(run,results)
            _record_stage_manifest(run, results); manifest=json.loads((run/"RUN_MANIFEST.json").read_text(encoding="utf-8"));manifest.update(status="success",completed_at=_now());(run/"RUN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8");_write_pointer(config,run,"success",best);print(run);return 0
        run = _resolve_run(args.run if hasattr(args,"run") else "latest", config)
        index = run / "prepared" / "public_change_index.csv"
        if args.command == "change-train":
            preflight = run_change_preflight(args.config, require_cuda=args.device.startswith("cuda"))
            if not preflight["READY_TO_TRAIN"]: return 1
            stages=["binary","semantic","directional"] if args.stage=="all" else [args.stage]
            results=train_change_pipeline(config,index,run,stages=stages,device_name=args.device,resume=args.resume,distributed=args.distributed);print("\n".join(str(path) for path in results.values()));return 0
        if args.command == "change-review":
            if args.ai_assist:
                # This is deliberately non-fatal: static human review remains usable
                # when the optional credential/dependency is not installed.
                if os.environ.get("ZAI_API_KEY") or os.environ.get("GLM_API_KEY"):
                    run_ai_review(run, max_samples=100)
                else:
                    print("GLM auxiliary review is not enabled (no API key); opening human review only.")
            if args.split == "train":
                manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
                checkpoint = Path(manifest["final_checkpoint"])
                if not checkpoint.is_absolute(): checkpoint = REPO_ROOT / checkpoint
                destination = Path(args.output_dir) if args.output_dir else run / "review_train"
                samples = generate_train_review(checkpoint, index, run, device_name=args.device, selection=args.selection,
                                                max_samples=args.max_samples, output_dir=destination, force=args.force)
                path = build_static_review(run, samples, output_dir=destination)
            else:
                path=build_static_review(run, output_dir=args.output_dir)
            if args.interactive:
                from .review_app import launch_review_app; launch_review_app(run)
            return 0
        if args.command == "change-retrain":
            decisions=run/"review_train"/"review_decisions_train.csv"
            demolition_head_only = bool(config.get("training", {}).get("demolition_head_only", False))
            if args.from_review:
                if not decisions.is_file(): raise FileNotFoundError("--from-review requires review_train/review_decisions_train.csv")
                sampling=config.get("sampling", {})
                hard_path = build_hard_examples(run, decisions, false_negative_weight=float(sampling.get("hard_false_negative_weight", 3.0)), positive_weight=float(sampling.get("positive_sample_weight", 2.0)), ordinary_weight=float(sampling.get("ordinary_sample_weight", 1.0)))
                hard = pd.read_csv(hard_path, dtype=str).fillna("")
                if not hard.empty and not (hard["sample_id"].isin(pd.read_csv(index, dtype=str).fillna("").query("split == 'train'")["sample_id"])).all():
                    raise RuntimeError("Hard-example gate rejected validation/test IDs")
                if demolition_head_only:
                    training_index, demolition_sampling = build_demolition_head_sampling_index(
                        run, hard_path,
                        demolition_only_weight=float(sampling.get("demolition_only_weight", 2.0)),
                        both_weight=float(sampling.get("both_weight", 1.25)),
                        ordinary_weight=float(sampling.get("ordinary_sample_weight", 1.0)),
                        max_hard_fraction=float(sampling.get("max_hard_sample_fraction", 0.10)),
                        max_both_fraction_within_hard=float(sampling.get("max_both_fraction_within_hard", 0.40)),
                    )
                else:
                    training_index = run / "prepared" / "public_change_index.hard.csv"; demolition_sampling = {}
            else:
                training_index=run/("prepared/public_change_index.demolition_head.csv" if demolition_head_only else "prepared/public_change_index.hard.csv")
                if not training_index.is_file():
                    sampling=config.get("sampling", {}); hard_path = build_hard_examples(run, false_negative_weight=float(sampling.get("hard_false_negative_weight", 3.0)), positive_weight=float(sampling.get("positive_sample_weight", 2.0)), ordinary_weight=float(sampling.get("ordinary_sample_weight", 1.0)))
                    if demolition_head_only:
                        training_index, demolition_sampling = build_demolition_head_sampling_index(run, hard_path, demolition_only_weight=float(sampling.get("demolition_only_weight", 2.0)), both_weight=float(sampling.get("both_weight", 1.25)), ordinary_weight=float(sampling.get("ordinary_sample_weight", 1.0)), max_hard_fraction=float(sampling.get("max_hard_sample_fraction", 0.10)), max_both_fraction_within_hard=float(sampling.get("max_both_fraction_within_hard", 0.40)))
                    else:
                        training_index=run/"prepared"/"public_change_index.hard.csv"; demolition_sampling = {}
                else:
                    demolition_sampling = {}
            new_run=create_change_run(_pointer_root(config)); active_run = new_run
            index_provenance = _inherit_candidate_index(run, new_run)
            initial=config.get("training", {}).get("initialization_checkpoint") or (run/"checkpoints"/STAGE_FILES["directional"])
            initial=Path(initial); initial=initial if initial.is_absolute() else REPO_ROOT/initial
            if not initial.is_file(): raise FileNotFoundError(f"initialization checkpoint not found: {initial}")
            config.setdefault("training", {})["finetune_state"] = "demolition_head_finetune" if demolition_head_only else "hard_example_finetune"
            weighted = pd.read_csv(training_index, dtype=str).fillna("")
            hard_count = (int(demolition_sampling.get("demolition_hard_pool", demolition_sampling.get("hard_selected", 0))) if demolition_head_only
                          else int((pd.to_numeric(weighted.get("sample_weight", pd.Series("1", index=weighted.index)), errors="coerce").fillna(1.0) >= float(config.get("sampling", {}).get("hard_false_negative_weight", 3.0))).sum()))
            train_only = int((weighted["split"].str.casefold() == "train").sum())
            rejected = int(len(hard) - hard_count) if args.from_review else 0
            manifest = {"status": "running", "command": "change-retrain", "baseline_run": str(run), "initialization_checkpoint": str(initial), "hard_examples_path": str(hard_path) if args.from_review else "", "hard_examples_loaded": hard_count, "train_only_count": train_only, "validation_test_rejected_count": rejected, "sampling_weights": config.get("sampling", {}), "demolition_head_only": demolition_head_only, "demolition_sampling": demolition_sampling, "finetune_state": config["training"]["finetune_state"], "test_used": False, "created_at": _now(), **index_provenance}
            (new_run/"RUN_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            (new_run/"resolved_config.yaml").write_text(yaml.safe_dump({key:value for key,value in config.items() if key != "_config_path"}, allow_unicode=True, sort_keys=False), encoding="utf-8")
            (new_run/"reports"/"HARD_EXAMPLE_SUMMARY.md").write_text("\n".join(["# Hard-example finetune input", "", f"- source: `{manifest['hard_examples_path']}`", f"- loaded train-only hard examples: {hard_count}", f"- validation/test decisions rejected: {rejected}", f"- sampling weights: `{json.dumps(config.get('sampling', {}), ensure_ascii=False)}`", "- test used: false"]) + "\n", encoding="utf-8")
            print(json.dumps({key: manifest[key] for key in ("hard_examples_path", "hard_examples_loaded", "train_only_count", "validation_test_rejected_count", "sampling_weights")}, ensure_ascii=False))
            best=train_change_stage(config,training_index,new_run,"directional",device_name=args.device,initial_checkpoint=initial)
            if not bool(config.get("training", {}).get("smoke_training_only", False)):
                evaluate_change_checkpoint(best,training_index,new_run,device_name=args.device); build_training_visualizations(new_run); build_static_review(new_run)
            last = new_run / "checkpoints" / "last_directional.pth"
            selection_path = new_run / "reports" / "model_selection_directional.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
            manifest=json.loads((new_run/"RUN_MANIFEST.json").read_text(encoding="utf-8")); manifest.update(status="success", completed_at=_now(), baseline_run=str(run), initialization_checkpoint=str(initial), final_checkpoint=str(best), best_checkpoint_sha256=_sha256(best), last_checkpoint=str(last), last_checkpoint_sha256=_sha256(last), model_selection_status=selection.get("model_selection_status", "unknown"), source_validation_metrics=selection.get("source_validation_metrics"), best_finetune_epoch_metrics=selection.get("best_finetune_epoch_metrics"), smoke_training_only=bool(config.get("training", {}).get("smoke_training_only", False))); (new_run/"RUN_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            print(best);return 0
        if args.command == "change-finetune":
            pointer = REPO_ROOT/"outputs"/args.aoi/"latest.json"; local_run=Path(json.loads(pointer.read_text(encoding="utf-8"))["run_directory"])
            local_index=local_run/"change_labels"/"local_change_index.reviewed.csv"
            if not local_index.is_file(): raise FileNotFoundError(f"No human-confirmed local change index: {local_index}")
            checkpoint=run/"checkpoints"/STAGE_FILES["directional"] if args.checkpoint=="latest-directional" else Path(args.checkpoint)
            target=create_change_run(REPO_ROOT/"outputs"/args.aoi/"change_finetune")
            best=train_change_stage(config,local_index,target,"local_finetune",device_name=args.device,initial_checkpoint=checkpoint);print(best);return 0
        raise ValueError(args.command)
    except BaseException as exc:
        if active_run is not None:
            _mark_failed_run(active_run, args.command, exc)
        if isinstance(exc, KeyboardInterrupt):
            print(f"INTERRUPTED [{args.command}]", file=sys.stderr)
            return 130
        traceback.print_exc(file=sys.stderr)
        print(f"ERROR [{args.command}]: {exc}", file=sys.stderr)
        return 1
