from __future__ import annotations

import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

from .ground_truth import read_building_labels
from .project_discovery import discover_project


class PreflightError(RuntimeError):
    pass


def run_personal_preflight(aoi_id: str, data_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Validate external or repository-local AOI inputs without modifying them."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    blockers, warnings, rows = [], [], []
    try:
        project = discover_project(aoi_id, data_root)
    except Exception as exc:
        report = {"READY_TO_PREPARE": False, "blockers": [str(exc)], "warnings": [], "report_path": str(output / "PERSONAL_PREPARE_CHECKLIST.md")}
        (output / "PERSONAL_PREPARE_CHECKLIST.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "PERSONAL_PREPARE_CHECKLIST.md").write_text(f"# Personal preflight\n\nREADY_TO_PREPARE = **False**\n\n- {exc}\n", encoding="utf-8")
        return report
    footprints, resolutions = [], []
    layer = project.config["data"].get("building_layer", "buildings")
    for date, image_path in sorted(project.images.items()):
        try:
            with rasterio.open(image_path) as image:
                if image.crs is None: blockers.append(f"{date}: image CRS is missing")
                footprint = box(*image.bounds); footprints.append(footprint)
                resolution = (abs(image.transform.a) + abs(image.transform.e)) / 2; resolutions.append(resolution)
                labels = read_building_labels(project.labels[date], building_layer=layer)
                if labels.crs is None: blockers.append(f"{date}: label CRS is missing")
                types = set(labels.geom_type.dropna())
                if not types <= {"Polygon", "MultiPolygon"}: blockers.append(f"{date}: labels are not Polygon/MultiPolygon: {sorted(types)}")
                invalid = int((~labels.geometry.is_valid).sum()) if not labels.empty else 0
                empty = int(labels.geometry.is_empty.sum()) if not labels.empty else 0
                if labels.empty: blockers.append(f"{date}: buildings layer is empty")
                if labels.crs is not None and image.crs is not None:
                    projected = labels.to_crs(image.crs)
                    if not projected.empty and not projected.geometry.union_all().intersects(footprint): blockers.append(f"{date}: labels do not intersect image extent")
                rows.append({"date": date, "image": str(image_path), "label": str(project.labels[date]), "building_layer": layer,
                             "buildings": len(labels), "invalid_geometries": invalid, "empty_geometries": empty, "gsd": resolution,
                             "image_crs": str(image.crs), "label_crs": str(labels.crs)})
        except Exception as exc:
            blockers.append(f"{date}: cannot validate image/label pair: {exc}")
    if len(footprints) > 1:
        common = footprints[0]
        for footprint in footprints[1:]: common = common.intersection(footprint)
        if common.is_empty: blockers.append("Images have no common coverage area")
    if resolutions and max(resolutions) / max(min(resolutions), 1e-12) > 2: warnings.append("Image GSD differs by more than 2x")
    report = {"READY_TO_PREPARE": not blockers, "aoi_id": aoi_id, "data_root": str(Path(data_root).resolve()),
              "building_layer": layer, "pairs": rows, "blockers": blockers, "warnings": warnings,
              "report_path": str(output / "PERSONAL_PREPARE_CHECKLIST.md")}
    (output / "PERSONAL_PREPARE_CHECKLIST.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Personal AOI preflight", "", f"READY_TO_PREPARE = **{report['READY_TO_PREPARE']}**", "", "## Pairs"]
    lines.extend(f"- {row['date']}: buildings={row['buildings']}, layer={row['building_layer']}, gsd={row['gsd']}" for row in rows)
    lines += ["", "## Blockers", *([f"- {value}" for value in blockers] or ["- none"]), "", "## Warnings", *([f"- {value}" for value in warnings] or ["- none"]), ""]
    (output / "PERSONAL_PREPARE_CHECKLIST.md").write_text("\n".join(lines), encoding="utf-8")
    return report


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
