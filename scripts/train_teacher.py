#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.training import train_teacher
from urban_monitor.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    t = cfg["training"]
    training_root = Path(cfg["output_root"]) / "training"
    resume_requested = args.resume or bool(t.get("resume", False))
    if args.force:
        resume_requested = False
        for name in ("best.pth", "last.pth"):
            prepare_output(training_root / "checkpoints" / name, force=True)
    elif (training_root / "checkpoints" / "best.pth").exists() and not resume_requested:
        prepare_output(training_root / "checkpoints" / "best.pth", force=False)
    result = train_teacher(
        Path(cfg["output_root"]) / "prepared" / "coco" if (Path(cfg["output_root"]) / "prepared" / "coco").exists() else Path(cfg["output_root"]) / "coco",
        training_root,
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        num_workers=int(t["num_workers"]),
        num_classes=int(t.get("num_classes", 2)),
        checkpoint=t.get("checkpoint"),
        pretrained=bool(cfg.get("model", {}).get("pretrained", False)),
        weights_path=cfg.get("model", {}).get("weights_path"),
        allow_pretrained_download=bool(cfg.get("model", {}).get("allow_pretrained_download", False)),
        amp=bool(t.get("amp", True)),
        gradient_clip_norm=float(t.get("gradient_clip_norm", 5.0)),
        early_stopping_patience=int(t.get("early_stopping_patience", 7)),
        save_every_epochs=int(t.get("save_every_epochs", 1)),
        resume=resume_requested,
        seed=int(t.get("seed", cfg.get("seed", 42))),
        min_size=int(cfg.get("model", {}).get("min_size", 512)),
        max_size=int(cfg.get("model", {}).get("max_size", 1024)),
    )
    print(f"Checkpoint: {result}")


if __name__ == "__main__":
    main()
