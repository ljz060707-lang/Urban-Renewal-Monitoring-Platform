#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from urban_monitor.config import load_config
from urban_monitor.training import train_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    t = cfg["training"]
    result = train_teacher(
        Path(cfg["output_root"]) / "coco",
        Path(cfg["output_root"]) / "checkpoints",
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        num_workers=int(t["num_workers"]),
        num_classes=int(t.get("num_classes", 2)),
        checkpoint=t.get("checkpoint"),
    )
    print(f"Checkpoint: {result}")


if __name__ == "__main__":
    main()
