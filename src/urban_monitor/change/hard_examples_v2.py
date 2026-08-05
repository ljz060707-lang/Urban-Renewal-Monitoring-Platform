from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .supervision_v2 import TestReadGuard


def mine_v2_hard_examples(index_path: str | Path, output_path: str | Path, *, split: str = "train") -> dict:
    if split.casefold() != "train":
        raise ValueError("change-mine-v2-hard-examples only permits --split train")
    guard = TestReadGuard("change-mine-v2-hard-examples")
    frame = pd.read_csv(index_path, dtype=str).fillna("")
    split_values = frame["split"].astype(str).str.casefold()
    train = frame.loc[split_values == "train"].copy()
    if train.empty:
        raise ValueError("No train samples in change index")
    score_columns = [
        name for name in ("false_negative_pixels", "uncertainty", "edge_density", "component_count")
        if name in train.columns
    ]
    if score_columns:
        for name in score_columns:
            train[name] = pd.to_numeric(train[name], errors="coerce").fillna(0)
        train["hard_score"] = train[score_columns].rank(pct=True).mean(axis=1)
        train = train.sort_values(["hard_score", "sample_id"], ascending=[False, True])
    else:
        train["hard_score"] = 0.0
        train = train.sort_values("sample_id")
    train["hard_reason"] = "train_signal_candidate"
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(target, index=False, encoding="utf-8-sig")
    report = {
        "status": "success",
        "split": "train",
        "sample_count": len(train),
        "test_images_read": guard.receipt().test_images_read,
        "output": str(target.resolve()),
    }
    target.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

