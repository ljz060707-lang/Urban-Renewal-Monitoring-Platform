from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .augmentations import ChangePairTransform
from .datasets import PublicChangeDataset
from .metrics import ChangeMetricAccumulator
from .models import build_change_model
from .splits import resolve_train_validation_test_splits


def _array(tensor):
    return tensor.detach().cpu().numpy()


def _rgb(tensor):
    return Image.fromarray(np.clip(_array(tensor).transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))


def _mask(array, color):
    value = np.asarray(array) != 0
    output = np.zeros((*value.shape, 3), dtype=np.uint8); output[value] = color
    return Image.fromarray(output)


def _heat(array):
    value = np.clip(np.asarray(array), 0, 1)
    output = np.stack([value * 180 + 40, np.zeros_like(value) + 35, value * 255], axis=-1)
    return Image.fromarray(output.astype(np.uint8))


def _error(prediction, target):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    output = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    output[prediction & target] = (46, 204, 113)
    output[prediction & ~target] = (255, 67, 78)
    output[~prediction & target] = (255, 207, 61)
    return Image.fromarray(output)


def _panel(path: Path, panels):
    width, height = 192, 192
    canvas = Image.new("RGB", (width * 5, (height + 24) * 2), "#11151b")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(panels):
        image = image.resize((width, height), Image.Resampling.NEAREST if "GT" in title or "Pred" in title else Image.Resampling.BILINEAR)
        x, y = (index % 5) * width, (index // 5) * (height + 24)
        canvas.paste(image, (x, y + 24)); draw.text((x + 6, y + 5), title, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)


def _save_prediction(path: Path, probability):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(probability * 255, 0, 255).astype(np.uint8)).save(path)


def evaluate_change_checkpoint(checkpoint: str | Path, index_path: str | Path, run: str | Path, *,
                               device_name: str = "cpu", stage: str | None = None, threshold: float = 0.5,
                               preview_limit: int = 100, split_scope: str = "validation") -> dict:
    import torch
    run, checkpoint = Path(run), Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    stage = stage or payload.get("stage", "directional")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    model = build_change_model(config); model.load_state_dict(payload["model"]); model.to(device).eval()
    all_frame = pd.read_csv(index_path, dtype=str).fillna("")
    _, validation_frame, test_frame, split_summary = resolve_train_validation_test_splits(all_frame, config)
    frame = test_frame if split_scope == "test" else validation_frame
    if frame.empty:
        raise ValueError(f"Evaluation requires {split_scope} samples with GT")
    size = int(config.get("training", {}).get("patch_size", 256))
    dataset = PublicChangeDataset(frame, transform=ChangePairTransform(size, training=False))
    names = sorted(set(frame["dataset_name"]))
    semantic_classes = int(config.get("model", config.get("change_model", {})).get("semantic_classes", 7))
    has_semantic = frame["has_semantic"].astype(str).str.casefold().isin({"true", "1", "yes"}).any()
    building_value = config.get("datasets", {}).get("second", {}).get("building_class_id")
    if has_semantic and building_value is None:
        raise ValueError("datasets.second.building_class_id is required when semantic supervision is enabled")
    building = int(building_value) if building_value is not None else None
    accumulators = {name: ChangeMetricAccumulator(semantic_classes, building) for name in [*names, "combined"]}
    error_roots = ["false_positive", "false_negative", "low_confidence", "new_dem_conflict", "possible_misalignment", "label_inconsistency", "model_disagreement"]
    for name in error_roots: (run / "errors" / name).mkdir(parents=True, exist_ok=True)
    review_rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]; name = sample["dataset_name"]
            outputs = model(sample["t1"].unsqueeze(0).to(device), sample["t2"].unsqueeze(0).to(device))
            binary = _array(torch.sigmoid(outputs["binary_change"])[0, 0]); new = _array(torch.sigmoid(outputs["new_building"])[0, 0])
            demolition = _array(torch.sigmoid(outputs["demolition"])[0, 0]); uncertainty = _array(torch.sigmoid(outputs["uncertainty"])[0, 0])
            semantic_t1 = _array(outputs["semantic_t1"].argmax(1)[0]); semantic_t2 = _array(outputs["semantic_t2"].argmax(1)[0])
            targets = {key: (_array(sample[key].squeeze(0)) if sample[key] is not None and sample[key].ndim == 3 else (_array(sample[key]) if sample[key] is not None else None))
                       for key in ("binary", "new", "demolition", "semantic_t1", "semantic_t2")}
            kwargs = dict(binary_probability=binary, binary_target=targets["binary"], new_probability=new, new_target=targets["new"],
                          demolition_probability=demolition, demolition_target=targets["demolition"], semantic_t1=semantic_t1, semantic_t2=semantic_t2,
                          semantic_target_t1=targets["semantic_t1"], semantic_target_t2=targets["semantic_t2"], threshold=threshold)
            accumulators[name].update(**kwargs); accumulators["combined"].update(**kwargs)
            safe = sample["sample_id"].replace(":", "_")
            prediction_root = run / "predictions" / name / stage
            for label, probability in (("binary", binary), ("new", new), ("demolition", demolition), ("uncertainty", uncertainty)):
                _save_prediction(prediction_root / f"{safe}_{label}.png", probability)
            if index < preview_limit:
                zero = np.zeros_like(binary)
                preview = run / "previews" / name / stage / f"{safe}.jpg"
                _panel(preview, [
                    ("T1", _rgb(sample["t1"])), ("T2", _rgb(sample["t2"])),
                    ("GT binary", _mask(targets["binary"] if targets["binary"] is not None else zero, (230, 230, 230))),
                    ("Pred binary", _mask(binary >= threshold, (230, 230, 230))),
                    ("GT new", _mask(targets["new"] if targets["new"] is not None else zero, (64, 143, 255))),
                    ("Pred new", _mask(new >= threshold, (64, 143, 255))),
                    ("GT demolition", _mask(targets["demolition"] if targets["demolition"] is not None else zero, (255, 153, 51))),
                    ("Pred demolition", _mask(demolition >= threshold, (255, 153, 51))),
                    ("Uncertainty", _heat(uncertainty)),
                    ("Error", _error(binary >= threshold, targets["binary"] if targets["binary"] is not None else zero)),
                ])
                if targets["binary"] is not None:
                    pred, gt = binary >= threshold, targets["binary"] != 0
                    fp, fn = int((pred & ~gt).sum()), int((~pred & gt).sum())
                    if fp: shutil.copy2(preview, run / "errors" / "false_positive" / preview.name)
                    if fn: shutil.copy2(preview, run / "errors" / "false_negative" / preview.name)
                if float(uncertainty.mean()) > 0.5: shutil.copy2(preview, run / "errors" / "low_confidence" / preview.name)
                if np.logical_and(new >= threshold, demolition >= threshold).any(): shutil.copy2(preview, run / "errors" / "new_dem_conflict" / preview.name)
            gt = targets["binary"] != 0 if targets["binary"] is not None else None
            pred = binary >= threshold
            if gt is not None:
                tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
                sample_f1 = 2 * tp / max(1, 2 * tp + fp + fn); sample_iou = tp / max(1, tp + fp + fn)
            else:
                sample_f1 = sample_iou = None
            review_rows.append({
                "sample_id": sample["sample_id"], "dataset_name": name, "stage": stage,
                "f1": sample_f1, "iou": sample_iou, "uncertainty": float(uncertainty.mean()),
                "precision": tp / max(1, tp + fp) if gt is not None else None,
                "recall": tp / max(1, tp + fn) if gt is not None else None,
                "false_positive_pixels": fp if gt is not None else None,
                "false_negative_pixels": fn if gt is not None else None,
                "gt_positive_ratio": float(gt.mean()) if gt is not None else None,
                "prediction_positive_ratio": float(pred.mean()),
                "t1_path": str(frame.iloc[index]["t1_path"]), "t2_path": str(frame.iloc[index]["t2_path"]),
                "gt_binary_path": str(frame.iloc[index].get("gt_cd_path", "")),
                "gt_new_path": str(frame.iloc[index].get("gt_new_path", "")),
                "gt_demolition_path": str(frame.iloc[index].get("gt_dem_path", "")),
                "binary_probability_path": str((run / "predictions" / name / stage / f"{safe}_binary.png").resolve()),
                "new_probability_path": str((run / "predictions" / name / stage / f"{safe}_new.png").resolve()),
                "demolition_probability_path": str((run / "predictions" / name / stage / f"{safe}_demolition.png").resolve()),
                "preview_path": str((run / "previews" / name / stage / f"{safe}.jpg").resolve()),
            })
    results = {name: accumulator.result() for name, accumulator in accumulators.items()}
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    metrics_root = run / "metrics" if split_scope == "validation" else run / "metrics" / split_scope
    metrics_root.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        (metrics_root / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_root / "split_selection.json").write_text(json.dumps({**split_summary, "scope": split_scope}, ensure_ascii=False, indent=2), encoding="utf-8")
    review_name = "review_samples.csv" if split_scope == "validation" else f"review_samples_{split_scope}.csv"
    pd.DataFrame(review_rows).to_csv(run / "review" / review_name, index=False, encoding="utf-8-sig")
    return results
