from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape

from .geometry import polygon_iou


def _polygons(mask: np.ndarray, transform) -> list:
    return [
        shape(geometry)
        for geometry, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform)
        if int(value) == 1
    ]


def _match(predicted: list, truth: list, threshold: float = 0.5):
    candidates = sorted(
        ((polygon_iou(predicted[i], truth[j]), i, j) for i in range(len(predicted)) for j in range(len(truth))),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for iou, pred_index, truth_index in candidates:
        if iou < threshold or pred_index in used_pred or truth_index in used_truth:
            continue
        used_pred.add(pred_index)
        used_truth.add(truth_index)
        matches.append((pred_index, truth_index, iou))
    return matches, sorted(set(range(len(predicted))) - used_pred), sorted(set(range(len(truth))) - used_truth)


def _empty_frame(crs, columns):
    return gpd.GeoDataFrame({column: [] for column in [*columns, "geometry"]}, geometry="geometry", crs=crs)


def _write_layer(rows: list[dict], path: Path, crs, columns: list[str]) -> None:
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else _empty_frame(crs, columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GPKG")


def _preview(image_path: Path, truth_masks: np.ndarray, pred_masks: np.ndarray, output: Path) -> None:
    image = Image.open(image_path).convert("RGBA")
    height, width = image.height, image.width
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for mask in truth_masks:
        overlay[mask.astype(bool)] = (0, 220, 80, 90)
    for mask in pred_masks:
        existing = overlay[mask.astype(bool), :3].sum(axis=1) > 0
        overlay[mask.astype(bool)] = (255, 40, 40, 100)
        indices = np.argwhere(mask.astype(bool))
        if len(indices) and existing.any():
            overlap_indices = indices[existing]
            overlay[overlap_indices[:, 0], overlap_indices[:, 1]] = (255, 220, 0, 130)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, Image.fromarray(overlay, "RGBA")).convert("RGB").save(output, quality=90)


def evaluate_checkpoint(
    coco_root: str | Path,
    prepared_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    num_classes: int = 2,
    min_size: int = 512,
    max_size: int = 1024,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    match_iou_threshold: float = 0.5,
    device_name: str | None = None,
) -> dict[str, float]:
    import torch

    from .dataset import CocoBuildingDataset
    from .model import build_model, checkpoint_sha256

    root = Path(coco_root)
    payload = json.loads((root / "annotations_val.json").read_text(encoding="utf-8"))
    if not payload.get("annotations"):
        raise ValueError("Validation Ground Truth is empty; metrics are not computable")
    dataset = CocoBuildingDataset(root, "annotations_val.json")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(num_classes=num_classes, checkpoint=checkpoint, min_size=min_size, max_size=max_size).to(device).eval()
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    predictions_rows = []
    fp_rows = []
    fn_rows = []
    error_rows = []
    object_rows = []
    preview_rows = []
    all_ious = []
    total_tp = total_fp = total_fn = 0
    predicted_area = truth_area = 0.0
    analysis_crs = None
    info_by_id = {int(item["id"]): item for item in payload["images"]}
    for index in range(len(dataset)):
        image, target = dataset[index]
        image_id = int(target["image_id"].item())
        info = info_by_id[image_id]
        date = str(info["date"])
        with rasterio.open(prepared / "images" / f"{date}.tif") as source:
            analysis_crs = analysis_crs or source.crs
            transform = source.window_transform(
                Window(int(info["col_off"]), int(info["row_off"]), int(info["width"]), int(info["height"]))
            )
        with torch.inference_mode():
            prediction = model([image.to(device)])[0]
        keep = prediction["scores"].detach().cpu() >= score_threshold
        scores = prediction["scores"].detach().cpu()[keep].numpy()
        pred_masks = (prediction["masks"].detach().cpu()[keep, 0] >= mask_threshold).numpy()
        truth_masks = target["masks"].numpy().astype(bool)
        pred_polygons = [polygon for mask in pred_masks for polygon in _polygons(mask, transform)]
        truth_polygons = [polygon for mask in truth_masks for polygon in _polygons(mask, transform)]
        expanded_scores = [float(score) for score, mask in zip(scores, pred_masks) for _ in _polygons(mask, transform)]
        matches, fp_indices, fn_indices = _match(pred_polygons, truth_polygons, match_iou_threshold)
        matched_pred = {pred_index: (truth_index, iou) for pred_index, truth_index, iou in matches}
        matched_truth = {truth_index: (pred_index, iou) for pred_index, truth_index, iou in matches}
        total_tp += len(matches)
        total_fp += len(fp_indices)
        total_fn += len(fn_indices)
        all_ious.extend(iou for _, _, iou in matches)
        predicted_area += sum(geometry.area for geometry in pred_polygons)
        truth_area += sum(geometry.area for geometry in truth_polygons)
        for pred_index, geometry in enumerate(pred_polygons):
            row = {
                "prediction_id": f"val_{image_id}_{pred_index}", "image_id": image_id, "source_date": date,
                "score": expanded_scores[pred_index], "matched": pred_index in matched_pred,
                "matched_iou": matched_pred.get(pred_index, (None, 0.0))[1],
                "checkpoint_hash": checkpoint_sha256(checkpoint), "schema_version": "0.4.0", "geometry": geometry,
            }
            predictions_rows.append(row)
            if pred_index in fp_indices:
                fp_rows.append({**row, "error_type": "false_positive"})
                error_rows.append({**row, "error_type": "false_positive"})
        for truth_index, geometry in enumerate(truth_polygons):
            matched = truth_index in matched_truth
            iou = matched_truth.get(truth_index, (None, 0.0))[1]
            object_rows.append(
                {"image_id": image_id, "source_date": date, "truth_index": truth_index, "area_m2": geometry.area, "matched": matched, "iou": iou}
            )
            if truth_index in fn_indices:
                row = {"truth_id": f"truth_{image_id}_{truth_index}", "image_id": image_id, "source_date": date, "error_type": "false_negative", "geometry": geometry}
                fn_rows.append(row)
                error_rows.append(row)
        precision_tile = len(matches) / max(len(matches) + len(fp_indices), 1)
        recall_tile = len(matches) / max(len(matches) + len(fn_indices), 1)
        f1_tile = 2 * precision_tile * recall_tile / max(precision_tile + recall_tile, 1e-12)
        preview_rows.append((f1_tile, root / info["file_name"], truth_masks, pred_masks, image_id))

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    metrics = {
        "precision": precision, "recall": recall, "F1": f1, "mean_matched_IoU": mean_iou,
        "Dice": 2 * mean_iou / (1 + mean_iou) if mean_iou else 0.0,
        "object_count_error": float((total_tp + total_fp) - (total_tp + total_fn)),
        "building_area_error": float(predicted_area - truth_area),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "checkpoint": str(checkpoint), "checkpoint_hash": checkpoint_sha256(checkpoint),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    objects = pd.DataFrame(object_rows)
    objects.to_csv(output / "object_metrics.csv", index=False, encoding="utf-8-sig")
    quantiles = objects["area_m2"].quantile([1 / 3, 2 / 3]).tolist()
    q1, q2 = quantiles
    objects["size"] = objects["area_m2"].map(lambda area: "small" if area <= q1 else "medium" if area <= q2 else "large")
    size_rows = []
    for size in ("small", "medium", "large"):
        subset = objects[objects["size"] == size]
        size_rows.append({
            "size": size, "method": f"truth area terciles: q33={q1:.3f}, q67={q2:.3f}",
            "truth_objects": len(subset), "matched_objects": int(subset["matched"].sum()),
            "recall": float(subset["matched"].mean()) if len(subset) else np.nan,
            "mean_iou": float(subset.loc[subset["matched"], "iou"].mean()) if subset["matched"].any() else 0.0,
        })
    pd.DataFrame(size_rows).to_csv(output / "size_stratified_metrics.csv", index=False, encoding="utf-8-sig")
    _write_layer(predictions_rows, output / "validation_predictions.gpkg", analysis_crs, ["prediction_id", "image_id", "source_date", "score", "matched", "matched_iou", "checkpoint_hash", "schema_version"])
    _write_layer(error_rows, output / "validation_errors.gpkg", analysis_crs, ["error_type", "image_id", "source_date"])
    _write_layer(fp_rows, output / "false_positives.gpkg", analysis_crs, ["prediction_id", "image_id", "source_date", "score", "error_type"])
    _write_layer(fn_rows, output / "false_negatives.gpkg", analysis_crs, ["truth_id", "image_id", "source_date", "error_type"])
    preview_rows.sort(key=lambda item: item[0], reverse=True)
    for group, candidates in (("good", preview_rows), ("hard", list(reversed(preview_rows)))):
        for index in range(10):
            _, image_path, truths, predictions, image_id = candidates[index % len(candidates)]
            _preview(image_path, truths, predictions, output / "previews" / f"{group}_{index + 1:02d}_{image_id}.jpg")
    lines = ["# Evaluation summary", "", f"- Checkpoint: `{checkpoint}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in metrics.items() if key not in {"checkpoint", "checkpoint_hash"})
    (output / "evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
