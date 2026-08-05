from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .evaluation_cache import validation_probabilities


THRESHOLDS = [round(value / 100, 2) for value in range(5, 100, 5)]
HEADS = {"binary": ("binary_change", "binary"), "new": ("new_building", "new"), "demolition": ("demolition", "demolition")}


def _curve(path: Path, title: str, series: list[tuple[str, list[tuple[float, float]], str]]) -> None:
    width, height, margin = 760, 380, 54
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 14), title, fill="black")
    draw.rectangle((margin, 42, width - 24, height - margin), outline="#607080")
    for label, points, color in series:
        if not points:
            continue
        mapped = [(margin + x * (width - margin - 24), height - margin - y * (height - margin - 42)) for x, y in points]
        draw.line(mapped, fill=color, width=3)
        draw.text((width - 210, 46 + 18 * series.index((label, points, color))), label, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _histogram(path: Path, title: str, values: np.ndarray) -> None:
    """Write a dependency-light probability histogram for the audit report."""
    width, height, margin = 760, 360, 42
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    counts, _ = np.histogram(values, bins=20, range=(0.0, 1.0))
    maximum = int(counts.max()) if len(counts) else 1
    for index, count in enumerate(counts):
        left = margin + index * (width - 2 * margin) / len(counts)
        right = margin + (index + 1) * (width - 2 * margin) / len(counts) - 2
        top = height - margin - (count / max(1, maximum) * (height - 2 * margin))
        draw.rectangle((left, top, right, height - margin), fill="#2774ae")
    draw.text((14, 14), title, fill="black")
    draw.text((margin, height - margin + 8), "0.0", fill="black")
    draw.text((width - margin - 25, height - margin + 8), "1.0", fill="black")
    image.save(path)


def _pr_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    target = target.astype(bool).ravel()
    if not target.any():
        return None
    order = np.argsort(-probability.ravel(), kind="stable")
    truth = target[order]
    hits = np.cumsum(truth)
    precision = hits / np.arange(1, len(truth) + 1)
    return float(precision[truth].sum() / truth.sum())


def _metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    predicted, truth = probability >= threshold, target.astype(bool)
    tp, fp, fn = int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum())
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {
        "threshold": threshold, "precision": precision, "recall": recall,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
        "predicted_positive_pixel_ratio": float(predicted.mean()), "actual_positive_pixel_ratio": float(truth.mean()),
        "false_positive_pixels": fp, "false_negative_pixels": fn, "true_positive_pixels": tp,
    }


def scan_thresholds(checkpoint: str | Path, index_path: str | Path, run: str | Path, *, device_name: str = "cuda:0",
                    split: str = "validation", thresholds: list[float] | None = None) -> pd.DataFrame:
    """Scan validation thresholds from one inference pass or its raw cache."""
    if split != "validation":
        raise ValueError("Threshold selection is validation-only; test is isolated")
    checkpoint, run = Path(checkpoint), Path(run)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cached, cache_metadata = validation_probabilities(checkpoint, index_path, run.parent / "cache" / "evaluation", device_name=device_name, split=split)
    values = {name: [[pair[0]], [pair[1]]] for name, pair in cached.items()}
    selected_count = int(cache_metadata["validation_samples"])
    thresholds = thresholds or THRESHOLDS
    rows, summary = [], {"split": split, "selection": cache_metadata.get("selection"), "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash, "cache": cache_metadata, "heads": {}}
    for head, (probs, targets) in values.items():
        if not probs or not probs[0].size or not targets[0].size:
            summary["heads"][head] = {"status": "skipped_no_supervision"}
            continue
        probability, target = np.concatenate(probs), np.concatenate(targets)
        auc = _pr_auc(probability, target)
        head_rows = []
        for threshold in thresholds:
            row = {"head": head, **_metrics(probability, target, threshold), "pr_auc": auc, "sample_count": selected_count, "checkpoint_sha256": checkpoint_hash}
            rows.append(row); head_rows.append(row)
        best = max(head_rows, key=lambda item: (item["f1"], item["recall"], -item["threshold"]))
        summary["heads"][head] = {"status": "available", "best": best, "probability": {"min": float(probability.min()), "max": float(probability.max()), "mean": float(probability.mean()), "q01": float(np.quantile(probability, .01)), "q50": float(np.quantile(probability, .5)), "q99": float(np.quantile(probability, .99))}}
    output = pd.DataFrame(rows)
    root = run / "metrics" / "threshold_scans" / checkpoint_hash
    root.mkdir(parents=True, exist_ok=True)
    output.to_csv(root / "threshold_scan.csv", index=False, encoding="utf-8-sig")
    (root / "threshold_scan.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    visual = run / "visualizations" / "threshold_scans" / checkpoint_hash; visual.mkdir(parents=True, exist_ok=True)
    _curve(visual / "threshold_f1_curve.png", "Validation threshold vs F1", [(head, list(zip(group.threshold, group.f1)), color) for (head, group), color in zip(output.groupby("head"), ["#1976d2", "#20a050", "#d94343"])])
    _curve(visual / "precision_recall_curve.png", "Validation precision-recall", [(head, list(zip(group.recall, group.precision)), color) for (head, group), color in zip(output.groupby("head"), ["#1976d2", "#20a050", "#d94343"])])
    binary_probs, binary_targets = values["binary"]
    if binary_probs and binary_probs[0].size:
        probabilities = np.concatenate(binary_probs)
        targets = np.concatenate(binary_targets).astype(bool)
        _histogram(visual / "positive_probability_histogram.png", "Binary probability: GT-positive pixels", probabilities[targets])
        _histogram(visual / "negative_probability_histogram.png", "Binary probability: GT-negative pixels", probabilities[~targets])
    binary = summary["heads"].get("binary", {}).get("probability", {})
    report = ["# Threshold diagnostic", "", f"- split: `{split}` (test is not used for threshold selection)", f"- checkpoint: `{checkpoint.resolve()}`", f"- checkpoint SHA256: `{checkpoint_hash}`", f"- raw probability cache hit: `{cache_metadata['cache_hit']}`", f"- inference seconds: `{cache_metadata['inference_seconds']:.3f}`", ""]
    for head, value in summary["heads"].items():
        status, best = value.get("status", "available"), value.get("best")
        if status == "skipped_no_supervision" or not best:
            report.append(f"- {head}: skipped_no_supervision")
            continue
        report.append(f"- {head}: threshold={best['threshold']:.2f}, F1={best['f1']:.4f}, recall={best['recall']:.4f}, IoU={best['iou']:.4f}, PR-AUC={best['pr_auc']:.4f}")
    if binary: report.extend(["", f"- binary probability mean={binary['mean']:.6f}, q50={binary['q50']:.6f}"])
    (root / "THRESHOLD_DIAGNOSTIC.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output
