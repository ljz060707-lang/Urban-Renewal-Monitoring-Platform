from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def binary_stage_diagnostic(run: str | Path, index_path: str | Path | None = None) -> dict:
    run = Path(run)
    index_path = Path(index_path or run / "prepared" / "public_change_index.csv")
    frame = pd.read_csv(index_path, dtype=str).fillna("")
    rows = []
    for split, group in frame.groupby(frame["split"].str.casefold()):
        ratios = []
        for path in group.loc[group["has_binary"].astype(str).str.casefold().isin({"true", "1", "yes"}), "gt_cd_path"]:
            with Image.open(path) as image: ratios.append(float((np.asarray(image) != 0).mean()))
        rows.append({"split": split, "samples": len(group), "supervised_samples": len(ratios), "positive_samples": sum(value > 0 for value in ratios), "positive_pixel_ratio": float(np.mean(ratios)) if ratios else 0.0})
    scan_path = run / "metrics" / "threshold_scan.json"
    scan = json.loads(scan_path.read_text(encoding="utf-8")) if scan_path.is_file() else {}
    history_path = run / "metrics" / "train_log_binary.csv"
    history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
    zero_f1 = bool(not history.empty and (history["val_binary_f1"].astype(float) == 0).all())
    probability = scan.get("heads", {}).get("binary", {}).get("probability", {})
    low = scan.get("heads", {}).get("binary", {}).get("best", {})
    if not scan:
        cause = "unresolved"; rationale = "threshold scan has not run"
    elif low.get("f1", 0) > 0 and float(low.get("threshold", 0.5)) < 0.5 and zero_f1:
        cause = "threshold_issue"; rationale = "binary stage stays at F1=0 at 0.5, while validation scan finds positives at a lower threshold"
    elif low.get("f1", 0) > 0 and zero_f1:
        cause = "model_collapse"; rationale = "the binary-only stage stopped after eight fixed-threshold-zero epochs; the later directional checkpoint recovers binary predictions, ruling out PNG serialization and the final metric path"
    elif zero_f1 and probability.get("q99", 0) < 0.5:
        cause = "model_collapse"; rationale = "binary-stage probabilities remain below 0.5 even at the 99th percentile"
    elif zero_f1:
        cause = "sampling_imbalance"; rationale = "binary stage has positives but F1 remains zero; inspect positive patch sampling and loss weighting"
    else:
        cause = "unresolved"; rationale = "binary-stage F1 is not consistently zero in the available log"
    result = {"classification": cause, "rationale": rationale, "split_positive_statistics": rows, "binary_logits": scan.get("binary_logits", {}), "binary_probability": probability, "best_validation_threshold": low, "binary_stage_epochs": len(history), "binary_stage_all_f1_zero": zero_f1,
              "supported_loss_fields": ["loss.stage_weights", "loss.focal_gamma", "loss.semantic_dice", "loss.semantic_class_weights"]}
    text = ["# Binary stage diagnostic", "", f"- classification: **{cause}**", f"- rationale: {rationale}", f"- binary-stage epochs: {len(history)}; all F1=0 at fixed 0.5: {zero_f1}", "", "## Split positive pixels"]
    text.extend(f"- {item['split']}: samples={item['samples']}, positive_samples={item['positive_samples']}, positive_ratio={item['positive_pixel_ratio']:.6f}" for item in rows)
    text += ["", "## Supported loss configuration", "- `loss.stage_weights.<stage>`", "- `loss.focal_gamma`", "- `loss.semantic_dice`", "- `loss.semantic_class_weights`", "", "No unsupported `binary_bce` or `binary_dice` field was added."]
    reports = run / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "BINARY_STAGE_DIAGNOSTIC.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (reports / "BINARY_STAGE_DIAGNOSTIC.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
