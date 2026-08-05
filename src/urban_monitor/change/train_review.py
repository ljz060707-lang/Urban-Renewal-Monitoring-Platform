from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .augmentations import ChangePairTransform
from .datasets import PublicChangeDataset
from .evaluation import _array, _save_prediction
from .models import build_change_model
from .splits import resolve_train_validation_test_splits


def _sample_metrics(probability: np.ndarray, target: np.ndarray) -> dict:
    prediction, truth = probability >= .5, target.astype(bool)
    tp, fp, fn = int((prediction & truth).sum()), int((prediction & ~truth).sum()), int((~prediction & truth).sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
            "false_negative_pixels": fn, "false_positive_pixels": fp,
            "gt_positive_ratio": float(truth.mean()), "prediction_positive_ratio": float(prediction.mean())}


def _rank(frame: pd.DataFrame, selection: str, max_samples: int) -> pd.DataFrame:
    if selection == "random":
        return frame.sample(min(max_samples, len(frame)), random_state=42).reset_index(drop=True)
    if selection == "all":
        return frame.head(max_samples).reset_index(drop=True)
    # A boolean first key ensures "GT changed but almost empty prediction" is
    # always surfaced before ordinary low-recall examples.
    ranked = frame.assign(
        _near_empty=((frame.gt_positive_ratio > 0) & (frame.prediction_positive_ratio < .001)).astype(int),
        _directional_fn=frame.new_false_negative_pixels + frame.demolition_false_negative_pixels,
    ).sort_values(["_near_empty", "recall", "false_negative_pixels", "uncertainty", "_directional_fn"],
                  ascending=[False, True, False, False, False], kind="stable")
    return ranked.head(max_samples).drop(columns=["_near_empty", "_directional_fn"]).reset_index(drop=True)


def generate_train_review(checkpoint: str | Path, index_path: str | Path, run: str | Path, *, device_name: str = "cuda:0",
                          selection: str = "hard", max_samples: int = 150, output_dir: str | Path | None = None,
                          force: bool = False) -> Path:
    """Infer original train only, rank difficult samples, then materialise a separate review page input."""
    import torch

    if selection not in {"hard", "random", "all"}:
        raise ValueError("selection must be hard, random, or all")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    run, checkpoint = Path(run), Path(checkpoint)
    review = Path(output_dir) if output_dir else run / "review_train"
    review = review.resolve()
    samples_path = review / "review_samples.csv"
    if samples_path.is_file() and not force:
        return samples_path
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA train review requested but unavailable")
    all_frame = pd.read_csv(index_path, dtype=str).fillna("")
    train_frame, _, _, split_summary = resolve_train_validation_test_splits(all_frame, config)
    if train_frame.empty or not (train_frame["split"].astype(str).str.casefold() == "train").all():
        raise ValueError("Train review requires an explicit non-empty train split")
    # Deliberate audit guard: test dataframe is never passed to the Dataset.
    dataset = PublicChangeDataset(train_frame, transform=ChangePairTransform(int(config.get("training", {}).get("patch_size", 256)), training=False))
    model = build_change_model(config).to(device).eval(); model.load_state_dict(payload["model"])
    ranking_path = review / "train_review_ranking.csv"
    rows: list[dict] = []
    batch_size = 16
    if ranking_path.is_file() and not force:
        selected = pd.read_csv(ranking_path, dtype=str).fillna("")
        selected = selected.head(max_samples)
        for column in selected.columns:
            if column not in {"sample_id", "dataset_name", "split", "stage"}:
                numeric = pd.to_numeric(selected[column], errors="coerce")
                if not numeric.isna().all():
                    selected[column] = numeric
    else:
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                output = model(torch.stack([item["t1"] for item in batch]).to(device), torch.stack([item["t2"] for item in batch]).to(device))
                for offset, sample in enumerate(batch):
                    binary, new, demolition = (torch.sigmoid(output[key][offset, 0]).detach().cpu().numpy()
                                                for key in ("binary_change", "new_building", "demolition"))
                    uncertainty = float(torch.sigmoid(output["uncertainty"][offset, 0]).mean().detach().cpu())
                    target = _array(sample["binary"].squeeze(0))
                    new_target, dem_target = _array(sample["new"].squeeze(0)), _array(sample["demolition"].squeeze(0))
                    item = _sample_metrics(binary, target)
                    item.update(sample_id=sample["sample_id"], dataset_name=sample["dataset_name"], split="train", stage="directional",
                                uncertainty=uncertainty, new_false_negative_pixels=_sample_metrics(new, new_target)["false_negative_pixels"],
                                demolition_false_negative_pixels=_sample_metrics(demolition, dem_target)["false_negative_pixels"])
                    rows.append(item)
        selected = _rank(pd.DataFrame(rows), selection, max_samples)
    review.mkdir(parents=True, exist_ok=True)
    selected.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    assets: list[dict] = []
    # Do not perform a second full 3500-image pass.  The first pass already
    # ranked all train samples; materialise only the selected IDs now.
    positions = {str(row.sample_id): position for position, row in train_frame.reset_index(drop=True).iterrows()}
    selected_positions = [positions[sample_id] for sample_id in selected.sample_id]
    with torch.no_grad():
        for start in range(0, len(selected_positions), batch_size):
            current_positions = selected_positions[start:start + batch_size]
            batch = [dataset[position] for position in current_positions]
            output = model(torch.stack([item["t1"] for item in batch]).to(device), torch.stack([item["t2"] for item in batch]).to(device))
            for offset, sample in enumerate(batch):
                safe = sample["sample_id"].replace(":", "_")
                source = train_frame.iloc[current_positions[offset]]
                pred_root = review / "predictions" / sample["dataset_name"]
                paths = {}
                for label, key in (("binary", "binary_change"), ("new", "new_building"), ("demolition", "demolition")):
                    path = pred_root / f"{safe}_{label}.png"; _save_prediction(path, torch.sigmoid(output[key][offset, 0]).detach().cpu().numpy()); paths[f"{label}_probability_path"] = str(path)
                record = selected[selected.sample_id == sample["sample_id"]].iloc[0].to_dict()
                record.update(t1_path=str(source.t1_path), t2_path=str(source.t2_path), gt_binary_path=str(source.get("gt_cd_path", "")),
                              gt_new_path=str(source.get("gt_new_path", "")), gt_demolition_path=str(source.get("gt_dem_path", "")), **paths)
                assets.append(record)
    output = pd.DataFrame(assets).sort_values(["recall", "false_negative_pixels"], ascending=[True, False])
    output.to_csv(samples_path, index=False, encoding="utf-8-sig")
    (review / "split_guard.json").write_text(__import__("json").dumps({**split_summary, "review_split": "train", "test_dataset_opened": False}, indent=2), encoding="utf-8")
    return samples_path
