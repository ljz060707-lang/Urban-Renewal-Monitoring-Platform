from __future__ import annotations

import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import rasterio


class PreflightError(RuntimeError):
    pass


def run_preflight(project, run_dir: str | Path) -> dict[str, object]:
    run = Path(run_dir)
    blockers: list[str] = []
    warnings: list[str] = []
    image_rows = []
    for date, path in sorted(project.images.items()):
        try:
            with rasterio.open(path) as source:
                if source.crs is None:
                    blockers.append(f"CRS missing: {path}")
                bands = project.config["data"]["rgb_bands"]
                if any(int(band) < 1 or int(band) > source.count for band in bands):
                    blockers.append(f"RGB bands {bands} exceed band count {source.count}: {path}")
                sample = source.dataset_mask(out_shape=(min(source.height, 256), min(source.width, 256)))
                image_rows.append({
                    "date": date, "bands": source.count, "crs": str(source.crs),
                    "gsd": (abs(source.transform.a) + abs(source.transform.e)) / 2,
                    "nodata_fraction": float(np.mean(sample == 0)),
                })
        except Exception as exc:
            blockers.append(f"Image cannot be read: {path}: {exc}")
    if not image_rows:
        blockers.append("No training images")
    if len(image_rows) < 3:
        warnings.append("Fewer than 3 image dates")
    if image_rows:
        gsd_values = [row["gsd"] for row in image_rows]
        if max(gsd_values) / max(min(gsd_values), 1e-12) > 2:
            warnings.append("Image GSD differs by more than 2x")
    if not project.labels:
        blockers.append("No Ground Truth labels")
    accepted_path = run / "data_report" / "accepted_labels.gpkg"
    accepted = gpd.read_file(accepted_path) if accepted_path.exists() else gpd.GeoDataFrame()
    if accepted.empty:
        blockers.append("All Ground Truth labels are empty after validation")
    label_counts = accepted.groupby("source_date").size().to_dict() if not accepted.empty else {}
    for date in project.images:
        if int(label_counts.get(date, 0)) == 0:
            warnings.append(f"No accepted buildings on {date}; it remains a hard-negative date")
    rejected_path = run / "data_report" / "rejected_labels.gpkg"
    rejected = gpd.read_file(rejected_path) if rejected_path.exists() else gpd.GeoDataFrame()
    invalid_count = int(rejected.get("validation_action", []).astype(str).str.contains("invalid").sum()) if not rejected.empty else 0
    outside_count = int(rejected.get("validation_action", []).astype(str).str.contains("outside").sum()) if not rejected.empty else 0
    coco = run / "prepared" / "coco"
    split_payloads = {}
    for split in ("train", "val"):
        path = coco / f"annotations_{split}.json"
        if not path.exists():
            blockers.append(f"Missing COCO file: {path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            image_ids = {int(item["id"]) for item in payload.get("images", [])}
            if not image_ids:
                blockers.append(f"{split} COCO has no images")
            if not payload.get("annotations"):
                blockers.append(f"{split} COCO has no Ground Truth annotations")
            for annotation in payload.get("annotations", []):
                if int(annotation.get("image_id", -1)) not in image_ids or len(annotation.get("bbox", [])) != 4 or not annotation.get("segmentation"):
                    blockers.append(f"Invalid COCO annotation in {path}")
                    break
            for item in payload.get("images", []):
                if not (coco / item["file_name"]).is_file():
                    blockers.append(f"Missing COCO tile: {item['file_name']}")
                    break
            split_payloads[split] = payload
        except Exception as exc:
            blockers.append(f"COCO annotations invalid: {path}: {exc}")
    if set(split_payloads) == {"train", "val"}:
        train_blocks = {item.get("block_id") for item in split_payloads["train"]["images"]}
        val_blocks = {item.get("block_id") for item in split_payloads["val"]["images"]}
        if (train_blocks - {None}) & (val_blocks - {None}):
            blockers.append("Train/validation spatial block leakage detected")
        val_buildings = len(split_payloads["val"]["annotations"])
        if val_buildings < 20:
            warnings.append(f"Validation has only {val_buildings} annotations (<20)")
        total_tiles = sum(len(payload["images"]) for payload in split_payloads.values())
        positive_tiles = sum(len({ann["image_id"] for ann in payload["annotations"]}) for payload in split_payloads.values())
        if total_tiles < 20:
            warnings.append("Dataset has fewer than 20 tiles")
        if total_tiles and (total_tiles - positive_tiles) / total_tiles > 0.6:
            warnings.append("Empty tile fraction is above 60%")
    else:
        val_buildings = total_tiles = positive_tiles = 0
    if all(not (project.project_dir / "quality" / f"{date}.tif").exists() for date in project.images):
        warnings.append("No quality mask supplied")
    warnings.append("Off-nadir metadata is missing unless supplied in project.yaml/manifest")
    free_gb = shutil.disk_usage(run).free / (1024 ** 3)
    if free_gb < 2:
        warnings.append(f"Only {free_gb:.2f} GiB free disk space")
    try:
        probe = run / "reports" / ".write_test"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        blockers.append(f"Output directory is not writable: {exc}")
    model = project.config["model"]
    weight_source = str(model.get("weights_path") or ("torchvision download/cache" if model.get("pretrained") else "random initialization"))
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "unavailable"
        blockers.append("Torch is not installed; run pip install -e \".[train]\"")
    area_summary = {}
    if not accepted.empty:
        areas = accepted.geometry.area
        area_summary = {"min": float(areas.min()), "median": float(areas.median()), "max": float(areas.max())}
    report = {
        "blockers": blockers, "warnings": warnings, "images": image_rows,
        "image_count": len(project.images), "label_file_count": len(project.labels),
        "accepted_buildings": len(accepted), "invalid_geometries": invalid_count,
        "outside_geometries": outside_count, "building_area_m2": area_summary,
        "train_images": len(split_payloads.get("train", {}).get("images", [])),
        "val_images": len(split_payloads.get("val", {}).get("images", [])),
        "train_annotations": len(split_payloads.get("train", {}).get("annotations", [])),
        "val_annotations": len(split_payloads.get("val", {}).get("annotations", [])),
        "positive_tiles": positive_tiles, "total_tiles": total_tiles,
        "weight_source": weight_source, "device": device, "free_disk_gb": free_gb,
    }
    lines = ["# PREFLIGHT REPORT", "", f"- Status: `{'BLOCKED' if blockers else 'PASS'}`"]
    for key, value in report.items():
        if key not in {"blockers", "warnings", "images"}:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blockers", *([f"- {item}" for item in blockers] or ["- None"]), "", "## Warnings", *([f"- {item}" for item in warnings] or ["- None"]), ""])
    path = run / "reports" / "PREFLIGHT_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    if blockers:
        raise PreflightError("Preflight failed: " + "; ".join(blockers))
    return report
