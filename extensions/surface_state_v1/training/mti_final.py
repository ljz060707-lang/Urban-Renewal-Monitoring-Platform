from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..datasets.fusu import FUSUSampleDataset
from ..models.mti import MultiTaskIntegration
from ..models.surface_state_net import SurfaceStateNet
from ..models.temporal_change_head import TemporalChangeHead
from ..models.tfr import TemporalFeatureRefinement
from .checkpointing import sha256
from .full_trainer import _fusu_forward
from .metrics import BinaryCounts, Confusion


@torch.no_grad()
def recalibrate_mti(
    config_path: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    repo: str | Path,
    provisional_selection: str | Path | None = None,
) -> dict:
    started = time.perf_counter()
    repo, output = Path(repo), Path(output)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path(config["fusu_validation_manifest"])
    if not manifest.is_absolute():
        manifest = repo / manifest
    if "test" in str(manifest).casefold():
        raise RuntimeError("MTI recalibration may not read Test")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SurfaceStateNet()
    model.load_state_dict(state["model_state_dict"], strict=True)
    use_tfr = state.get("tfr_state_dict") is not None
    tfr = None
    if use_tfr:
        snapshot = state.get("config_snapshot", {})
        tfr = TemporalFeatureRefinement(
            24,
            heads=int(snapshot.get("temporal", {}).get("num_heads", 4)),
            dropout=float(snapshot.get("temporal", {}).get("dropout", 0.1)),
        )
        tfr.load_state_dict(state["tfr_state_dict"], strict=True)
        tfr.eval()
    change_head = None
    if state.get("change_head_state_dict") is not None:
        change_head = TemporalChangeHead(24)
        change_head.load_state_dict(state["change_head_state_dict"], strict=True)
        change_head.eval()
    model.eval()
    loader = DataLoader(
        FUSUSampleDataset(manifest, int(state["config_snapshot"].get("image_size", 512))),
        batch_size=1,
        num_workers=0,
    )
    observations = []
    no_mti_counts = BinaryCounts()
    semantic_confusions = []
    no_mti_flicker = []
    for batch in loader:
        t1, t2 = batch["image_t1"], batch["image_t2"]
        y1, y2 = batch["mask_t1"].numpy(), batch["mask_t2"].numpy()
        logits1, logits2, _ = _fusu_forward(
            model, tfr, change_head, t1, t2, use_tfr
        )
        p1, p2 = logits1.softmax(1), logits2.softmax(1)
        confidence = torch.stack([p1.max(1).values, p2.max(1).values], 1)
        change_probability = (1 - (p1 * p2).sum(1)).unsqueeze(1)
        observations.append((confidence, change_probability, y1, y2))
        predicted_change = p1.argmax(1).numpy() != p2.argmax(1).numpy()
        target_change = y1 != y2
        no_mti_counts.update(predicted_change, target_change)
        no_mti_flicker.append(float(predicted_change.mean()))
        for prediction, target in ((p1.argmax(1).numpy(), y1), (p2.argmax(1).numpy(), y2)):
            confusion = Confusion(17)
            confusion.update(prediction, target)
            semantic_confusions.append(confusion.report([str(i) for i in range(17)])["macro_mIoU"])

    rows = []
    best = None
    for unary_weight, change_weight, persistence_penalty, impossible_penalty in itertools.product(
        config["unary_weight"],
        config["change_weight"],
        config["persistence_penalty"],
        config["impossible_transition_penalty"],
    ):
        counts = BinaryCounts()
        flicker = []
        for confidence, change_probability, y1, y2 in observations:
            unary = (
                torch.logit(confidence.clamp(1e-6, 1 - 1e-6)) * unary_weight
            ).sigmoid()
            adjusted = (
                change_probability * change_weight
                - persistence_penalty * (1 - change_probability)
            ).clamp(0, 1)
            result = MultiTaskIntegration(change_weight=1.0)(unary, adjusted)
            predicted = (
                result["optimal_building_sequence"][:, 1]
                != result["optimal_building_sequence"][:, 0]
            )
            counts.update(predicted.numpy(), y1 != y2)
            flicker.append(float(predicted.float().mean()))
        change_metrics = counts.report()
        semantic_miou = float(np.mean(semantic_confusions))
        flicker_rate = float(np.mean(flicker))
        score = (
            0.50 * change_metrics["f1"]
            + 0.30 * semantic_miou
            + 0.20 * (1 - flicker_rate)
        )
        row = {
            "unary_weight": unary_weight,
            "change_weight": change_weight,
            "persistence_penalty": persistence_penalty,
            "impossible_transition_penalty": impossible_penalty,
            "temporal_change_F1": change_metrics["f1"],
            "semantic_macro_mIoU": semantic_miou,
            "flicker_rate": flicker_rate,
            "temporal_consistency_violation": flicker_rate,
            "prediction_coverage": flicker_rate,
            "mti_selection_score": score,
        }
        rows.append(row)
        if best is None or row["mti_selection_score"] > best["mti_selection_score"]:
            best = row
    with (output / "MTI_CALIBRATION_GRID.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provisional = None
    if provisional_selection and Path(provisional_selection).exists():
        provisional = json.loads(Path(provisional_selection).read_text(encoding="utf-8"))
    no_mti = no_mti_counts.report()
    no_mti.update(
        semantic_macro_mIoU=float(np.mean(semantic_confusions)),
        flicker_rate=float(np.mean(no_mti_flicker)),
    )
    selection = {
        "mti_status": "mti_sample_validation_calibrated_candidate",
        "upstream_checkpoint": str(Path(checkpoint).resolve()),
        "upstream_checkpoint_sha256": sha256(checkpoint),
        "upstream_tfr_loaded": use_tfr,
        "selected_parameters": best,
        "no_mti": no_mti,
        "old_provisional_mti": provisional,
        "calibration_rows": len(rows),
        "calibration_seconds": time.perf_counter() - started,
        "validation_scope": "FUSU 500-pair grouped sample Validation; NOT full FUSU",
        "test_reads": 0,
    }
    (output / "MTI_SELECTION.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "MTI_VALIDATION_REPORT.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selection

