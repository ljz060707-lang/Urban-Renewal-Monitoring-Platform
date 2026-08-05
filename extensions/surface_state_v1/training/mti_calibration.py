from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..datasets.fusu import FUSUSampleDataset
from ..models.mti import MultiTaskIntegration
from ..models.surface_state_net import SurfaceStateNet
from .checkpointing import sha256
from .metrics import BinaryCounts


@torch.no_grad()
def calibrate(config_path, checkpoint, output, *, repo):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repo, output = Path(repo), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path(config["fusu_validation_manifest"])
    if not manifest.is_absolute():
        manifest = repo / manifest
    if "test" in str(manifest).casefold():
        raise RuntimeError("MTI calibration may not read Test")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SurfaceStateNet().to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    dataset = FUSUSampleDataset(manifest, int(state["config_snapshot"].get("image_size", 512)))
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    observations = []
    semantic_confusions = []
    for batch in loader:
        t1, t2 = batch["image_t1"].to(device), batch["image_t2"].to(device)
        y1, y2 = batch["mask_t1"].numpy(), batch["mask_t2"].numpy()
        output_model = model.forward_fusu(t1, t2)
        p1, p2 = output_model["probabilities_t1"], output_model["probabilities_t2"]
        semantic_confidence = torch.stack([p1.max(1).values, p2.max(1).values], 1)
        change_probability = (1 - (p1 * p2).sum(1)).unsqueeze(1)
        observations.append((semantic_confidence.cpu(), change_probability.cpu(), y1, y2))
        semantic_confusions.append(float((p1.argmax(1).cpu().numpy() == y1).mean() + (p2.argmax(1).cpu().numpy() == y2).mean()) / 2)

    rows = []
    best = None
    for unary_weight, change_weight, persistence_penalty, impossible_penalty in itertools.product(
        config["unary_weight"], config["change_weight"], config["persistence_penalty"], config["impossible_transition_penalty"]
    ):
        counts = BinaryCounts()
        flicker = []
        for confidence, change_probability, y1, y2 in observations:
            logits = torch.logit(confidence.clamp(1e-6, 1 - 1e-6)) * unary_weight
            unary = logits.sigmoid()
            adjusted_change = (change_probability * change_weight - persistence_penalty * (1 - change_probability)).clamp(0, 1)
            result = MultiTaskIntegration(change_weight=1.0)(unary, adjusted_change)
            predicted_change = result["optimal_building_sequence"][:, 1] != result["optimal_building_sequence"][:, 0]
            target_change = y1 != y2
            counts.update(predicted_change.numpy(), target_change)
            flicker.append(float(predicted_change.float().mean()))
        change_metrics = counts.report()
        macro_miou_proxy = float(np.mean(semantic_confusions))
        flicker_rate = float(np.mean(flicker))
        score = 0.50 * change_metrics["f1"] + 0.30 * macro_miou_proxy + 0.20 * (1 - flicker_rate)
        row = {
            "unary_weight": unary_weight,
            "change_weight": change_weight,
            "persistence_penalty": persistence_penalty,
            "impossible_transition_penalty": impossible_penalty,
            "temporal_change_F1": change_metrics["f1"],
            "semantic_accuracy_proxy": macro_miou_proxy,
            "flicker_rate": flicker_rate,
            "mti_selection_score": score,
        }
        rows.append(row)
        if best is None or row["mti_selection_score"] > best["mti_selection_score"]:
            best = row

    with (output / "MTI_CALIBRATION_GRID.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "mti_status": "sample_validation_calibrated_candidate",
        "selected_parameters": best,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "validation_scope": "FUSU 500-pair sample spatial holdout — NOT full FUSU validation",
        "full_fusu_validated": False,
        "national_generalization_verified": False,
        "test_read_count": 0,
    }
    (output / "MTI_SELECTION.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (output / "MTI_VALIDATION_REPORT.json").write_text(json.dumps({"best": best, "grid_rows": len(rows), **selection}, indent=2), encoding="utf-8")
    return selection
