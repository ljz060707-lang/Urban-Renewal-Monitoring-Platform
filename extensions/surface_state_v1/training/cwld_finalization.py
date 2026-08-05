from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..datasets.cwld import CWLDManifestDataset
from ..datasets.cwld_codec import CLASS_NAMES, CWLDCodec
from ..models.surface_state_net import SurfaceStateNet
from .checkpointing import sha256, write_hashes
from .metrics import Confusion


PALETTE = np.asarray(
    [[0, 0, 0], [255, 255, 255], [0, 102, 255], [255, 0, 0], [255, 215, 0]],
    dtype=np.uint8,
)


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _read_epochs(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("CWLD finalization requires at least one complete epoch")
    return rows


def _best_epoch(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            float(row["metrics"]["cwld"]["macro_mIoU"]),
            float(row["metrics"]["cwld"]["macro_F1"]),
            -int(row["epoch"]),
        ),
    )


def _class_gate(metrics: dict) -> dict:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"Invalid CWLD confusion matrix shape: {matrix.shape}")
    total = int(matrix.sum())
    predicted = matrix.sum(0) / max(1, total)
    actual = matrix.sum(1) / max(1, total)
    classes = {}
    for index, name in enumerate(CLASS_NAMES):
        base = dict(metrics["per_class"][name])
        base.update(
            predicted_pixel_fraction=float(predicted[index]),
            ground_truth_pixel_fraction=float(actual[index]),
        )
        classes[name] = base
    non_background = list(CLASS_NAMES)[1:]
    no_prediction = [
        name for name in non_background if classes[name]["predicted_pixel_fraction"] <= 0
    ]
    dominant_fraction = float(predicted.max())
    naive_accuracy = float(actual.max())
    model_accuracy = float(metrics["pixel_accuracy"])
    passed = (
        not no_prediction
        and dominant_fraction < 0.995
        and model_accuracy > naive_accuracy
        and all(
            math.isfinite(classes[name][metric])
            for name in non_background
            for metric in ("iou", "f1", "precision", "recall")
        )
    )
    return {
        "status": "pass" if passed else "rejected_class_collapse",
        "passed": passed,
        "classes": classes,
        "maximum_predicted_class_fraction": dominant_fraction,
        "naive_majority_pixel_accuracy": naive_accuracy,
        "model_pixel_accuracy": model_accuracy,
        "model_beats_naive_majority": model_accuracy > naive_accuracy,
        "non_background_classes_without_prediction": no_prediction,
        "ignore_pixels_excluded": True,
    }


def _sample_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    confusion = Confusion(4)
    confusion.update(prediction, target)
    report = confusion.report(list(CLASS_NAMES))
    return float(report["macro_mIoU"]), float(report["macro_F1"])


@torch.no_grad()
def _visual_fixed32(run: Path, checkpoint: dict, repo: Path) -> dict:
    config = checkpoint["config_snapshot"]
    manifest = Path(config["cwld_manifest"])
    if not manifest.is_absolute():
        manifest = repo / manifest
    codec = CWLDCodec(
        **config.get(
            "codec",
            {"max_anchor_distance": 128, "min_distance_margin": 16, "ignore_index": 255},
        )
    )
    dataset = CWLDManifestDataset(manifest, "validation", codec, config["image_size"])
    indices = np.linspace(0, len(dataset) - 1, 32, dtype=int).tolist()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SurfaceStateNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    root = run / "visualizations" / "final_fixed32"
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    cards = []
    html_rows = []
    for order, index in enumerate(indices):
        item = dataset[index]
        image = item["image"]
        target = item["target"].numpy()
        prediction = (
            model.forward_cwld(image[None].to(device))["logits"]
            .argmax(1)[0]
            .cpu()
            .numpy()
        )
        miou, f1 = _sample_metrics(prediction, target)
        rgb = (image.numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        gt_index = np.where(target == 255, 4, target)
        gt = PALETTE[gt_index]
        pred = PALETTE[prediction]
        error = rgb.copy()
        mismatch = (target != 255) & (prediction != target)
        error[mismatch] = (0.35 * error[mismatch] + 0.65 * np.array([255, 0, 255])).astype(
            np.uint8
        )
        error[target == 255] = np.array([255, 215, 0], dtype=np.uint8)
        panels = [Image.fromarray(value).resize((256, 256)) for value in (rgb, gt, pred, error)]
        card = Image.new("RGB", (1024, 286), "white")
        for panel_index, panel in enumerate(panels):
            card.paste(panel, (panel_index * 256, 30))
        filename = f"{order:02d}_{item['sample_id']}.jpg"
        card.save(samples / filename, quality=90)
        cards.append(card.resize((512, 143)))
        html_rows.append(
            f"<article><h2>{html.escape(item['sample_id'])}</h2>"
            f"<p>macro IoU={miou:.4f}; macro F1={f1:.4f}</p>"
            f"<img src='samples/{html.escape(filename)}'></article>"
        )
    sheet = Image.new("RGB", (1024, 8 * 286), "white")
    for order, card in enumerate(cards):
        sheet.paste(card, ((order % 2) * 512, (order // 2) * 143))
    sheet.save(root / "contact_sheet.png")
    page = """<!doctype html><meta charset="utf-8"><title>CWLD fixed Validation 32</title>
<style>body{font-family:system-ui;margin:24px;background:#111;color:#eee}article{margin:24px 0}
img{max-width:100%;height:auto;border:1px solid #555}</style>
<h1>CWLD official Validation fixed 32</h1>
<p>Panels: image / ground truth / prediction / error overlay. Gold pixels are ignore.</p>
""" + "\n".join(html_rows)
    (root / "index.html").write_text(page, encoding="utf-8")
    return {
        "automatic_visual_export": "pass",
        "human_visual_review": "pending",
        "sample_count": 32,
        "indices": indices,
        "index_html": str((root / "index.html").resolve()),
        "contact_sheet": str((root / "contact_sheet.png").resolve()),
    }


def finalize_cwld(run_dir: str | Path, *, repo: str | Path) -> dict:
    repo, run = Path(repo), Path(run_dir)
    if not run.is_absolute():
        run = repo / run
    receipt_path = run / "TRAINING_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "success":
        raise RuntimeError(
            f"CWLD run is not complete: status={receipt.get('status')!r}; resume before finalization"
        )
    rows = _read_epochs(run / "logs" / "epochs.jsonl")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(epochs[0], epochs[-1] + 1)):
        raise RuntimeError(f"Non-contiguous epoch log: {epochs}")
    steps = [int(row["global_step"]) for row in rows]
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise RuntimeError(f"Non-increasing global_step sequence: {steps}")
    if any(int(row["metrics"]["cwld"]["sample_count"]) != 731 for row in rows):
        raise RuntimeError("Not every CWLD epoch evaluated all 731 Validation samples")
    if not _finite(rows):
        raise RuntimeError("NaN or Inf found in CWLD epoch metrics")
    if any(int(row["metrics"]["cwld"].get("test_read_count", -1)) != 0 for row in rows):
        raise RuntimeError("Test read guard failed")

    best_row = _best_epoch(rows)
    best_path = run / "checkpoints" / "cwld_best_validation.pth"
    last_path = run / "checkpoints" / "last.pth"
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    last_state = torch.load(last_path, map_location="cpu", weights_only=False)
    required_last = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "epoch",
        "global_step",
    )
    missing = [name for name in required_last if name not in last_state]
    if missing:
        raise RuntimeError(f"last checkpoint missing fields: {missing}")
    if int(best_state["epoch"]) != int(best_row["epoch"]):
        raise RuntimeError(
            "Best checkpoint epoch disagrees with recomputed Validation selection: "
            f"checkpoint={best_state['epoch']} recomputed={best_row['epoch']}"
        )

    gate = _class_gate(best_row["metrics"]["cwld"])
    visual = _visual_fixed32(run, best_state, repo)
    table_path = run / "FINAL_VALIDATION_TABLE.csv"
    with table_path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "epoch",
                "global_step",
                "train_loss",
                "macro_mIoU",
                "macro_F1",
                "validation_loss",
                "sample_count",
                "test_read_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]["cwld"]
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "global_step": row["global_step"],
                    "train_loss": row["train_loss"],
                    "macro_mIoU": metrics["macro_mIoU"],
                    "macro_F1": metrics["macro_F1"],
                    "validation_loss": metrics["validation_loss"],
                    "sample_count": metrics["sample_count"],
                    "test_read_count": metrics["test_read_count"],
                }
            )
    candidate_status = (
        "cwld_validation_candidate" if gate["passed"] else "rejected_class_collapse"
    )
    selection = {
        "candidate_status": candidate_status,
        "best_epoch": int(best_row["epoch"]),
        "best_global_step": int(best_row["global_step"]),
        "primary_macro_mIoU": best_row["metrics"]["cwld"]["macro_mIoU"],
        "secondary_macro_F1": best_row["metrics"]["cwld"]["macro_F1"],
        "tie_breaker": "earlier_epoch",
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": sha256(best_path),
        "last_checkpoint_path": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256(last_path),
        "class_collapse_gate": gate,
        "visual_review": visual,
        "test_reads": 0,
    }
    (run / "CHECKPOINT_SELECTION_FINAL.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_receipt = {
        "status": "success",
        "candidate_status": candidate_status,
        "epochs_completed": len(rows),
        "last_epoch": epochs[-1],
        "planned_epochs": int(best_state["config_snapshot"]["epochs"]),
        "early_stopping_or_plan_complete": True,
        "all_validation_sample_count": 731,
        "all_metrics_finite": True,
        "global_step_contiguous": True,
        "checkpoint_load": "pass",
        "test_reads": 0,
        **selection,
    }
    (run / "CWLD_FINALIZATION_RECEIPT.json").write_text(
        json.dumps(final_receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run / "MODEL_CARD_FINAL.md").write_text(
        "# CWLD Surface State Validation Candidate\n\n"
        "- Validation: CWLD official Validation, 731 samples.\n"
        "- Codec: four official anchors; mixed/uncertain colours use ignore=255.\n"
        "- Engineering facility is not construction_site.\n"
        "- Scope: Beijing-oriented construction-waste surface-state evidence.\n"
        "- This is not urban-renewal event Ground Truth and does not provide building geometry.\n"
        f"- Candidate status: `{candidate_status}`.\n"
        "- Test reads: `0`.\n",
        encoding="utf-8",
    )
    write_hashes(run, [best_path, last_path])
    return final_receipt

