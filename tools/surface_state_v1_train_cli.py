"""Training CLI for the existing non-invasive SurfaceState v1 extension."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "extensions"))

from surface_state_v1.datasets.fusu_split import split_fusu_sample
from surface_state_v1.integration.overlay_runtime import overlay_infer
from surface_state_v1.integration.registration import register_candidate
from surface_state_v1.training.full_trainer import train
from surface_state_v1.training.mti_calibration import calibrate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SurfaceState v1 training and integration")
    commands = root.add_subparsers(dest="command", required=True)

    split = commands.add_parser("fusu-split")
    split.add_argument("--manifest", default="outputs/data_audit/fusu/FUSU_MANIFEST.csv")
    split.add_argument("--output", default="outputs/data_audit/fusu_sample")
    split.add_argument("--seed", type=int, default=20260729)
    split.add_argument("--fold", type=int, default=0)
    split.add_argument("--num-folds", type=int, default=5)

    for name in ("cwld-train", "fusu-train", "joint-train", "temporal-train"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--run-dir")

    mti = commands.add_parser("mti-calibrate")
    mti.add_argument("--config", required=True)
    mti.add_argument("--checkpoint", required=True)
    mti.add_argument("--output", required=True)

    register = commands.add_parser("register-candidate")
    register.add_argument("--run-dir", required=True)

    overlay = commands.add_parser("overlay-infer")
    overlay.add_argument("--config", required=True)
    overlay.add_argument("--surface-checkpoint")
    overlay.add_argument("--input-t1")
    overlay.add_argument("--input-t2")
    overlay.add_argument("--output")

    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("--stage", choices=["split", "cwld", "fusu", "temporal", "joint"], required=True)
    pipeline.add_argument("--config")
    pipeline.add_argument("--run-dir")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "fusu-split":
        result = split_fusu_sample(REPO / args.manifest, REPO / args.output, seed=args.seed, fold=args.fold, num_folds=args.num_folds)
    elif args.command in {"cwld-train", "fusu-train", "joint-train", "temporal-train"}:
        result = train(REPO / args.config, REPO, run_dir=args.run_dir)
    elif args.command == "mti-calibrate":
        result = calibrate(REPO / args.config, args.checkpoint, args.output, repo=REPO)
    elif args.command == "register-candidate":
        result = register_candidate(args.run_dir, repo=REPO)
    elif args.command == "overlay-infer":
        config = yaml.safe_load((REPO / args.config).read_text(encoding="utf-8"))
        for key, value in (("surface_checkpoint", args.surface_checkpoint), ("input_t1", args.input_t1), ("input_t2", args.input_t2), ("output", args.output)):
            if value is not None:
                config[key] = value
        missing = [key for key in ("surface_checkpoint", "input_t1", "output") if not config.get(key)]
        if missing:
            raise ValueError("Missing overlay configuration: " + ", ".join(missing))
        result = overlay_infer(config, repo=REPO)
    else:
        if args.stage == "split":
            result = split_fusu_sample(REPO / "outputs/data_audit/fusu/FUSU_MANIFEST.csv", REPO / "outputs/data_audit/fusu_sample")
        else:
            defaults = {
                "cwld": "extensions/surface_state_v1/configs/cwld_train.yaml",
                "fusu": "extensions/surface_state_v1/configs/fusu_sample_train.yaml",
                "temporal": "extensions/surface_state_v1/configs/temporal_tfr_sample_train.yaml",
                "joint": "extensions/surface_state_v1/configs/surface_joint_sample_train.yaml",
            }
            result = train(REPO / (args.config or defaults[args.stage]), REPO, run_dir=args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
