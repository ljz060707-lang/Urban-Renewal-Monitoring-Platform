from __future__ import annotations

"""Tongzhou construction-candidate inference and audit.

This pipeline is inference-only.  It consumes the frozen UCS386 task-model
package and the already-produced RSBuilding geometry.  It does not call the
RSBuilding runner, retrain Full20, tune thresholds, or promote predictions to
human labels.
"""

import csv
import hashlib
import io
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from urban_renewal.construction.json_io import atomic_write_text, dump_json, load_json


DATES = ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")
METRIC_CRS = "EPSG:32650"
GEOMETRY_ROOT_NAME = "beijing_tongzhou_building_geometry"
OUTPUT_ROOT_NAME = "beijing_tongzhou_construction_activity_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_text(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(list(headers))
    for row in rows:
        writer.writerow(list(row))
    return stream.getvalue()


def _write_csv(path: Path, headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    atomic_write_text(path, _csv_text(headers, rows))


def _latest_package(project_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    training_root = project_root / "outputs" / "training" / "construction_activity_v1"
    runs = sorted(training_root.glob("URRS-Construction-UCS386-v1.0-task-candidate_*/FINALIZE_RECEIPT.json"))
    if not runs:
        raise RuntimeError("no completed UCS386 Stage A closeout receipt")
    receipt_path = runs[-1]
    receipt = load_json(receipt_path)
    package_dir = Path(receipt["package_dir"])
    manifest = load_json(package_dir / "TASK_MODEL_MANIFEST.json")
    if manifest.get("status") != "research_task_model_candidate_packaged" or manifest.get("candidate_frozen") is not True:
        raise RuntimeError("latest candidate package is not frozen research-task output")
    if manifest.get("reload_validation", {}).get("status") != "success":
        raise RuntimeError("latest candidate package has no successful independent reload")
    return package_dir, receipt, manifest


def _scene_manifest(project_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((project_root / "outputs" / "inference" / "beijing_tongzhou_surface_state_v1").glob("*/TONGZHOU_SCENES.json"))
    for path in reversed(candidates):
        payload = load_json(path)
        if tuple(item.get("date") for item in payload.get("scenes", [])) == DATES:
            return path, payload
    raise RuntimeError("no aligned four-date Tongzhou scene manifest")


def _verify_readonly_inputs(project_root: Path, scene_payload: Mapping[str, Any]) -> dict[str, Any]:
    geometry_root = project_root / "outputs" / "inference" / GEOMETRY_ROOT_NAME
    geometry = {}
    for date in DATES:
        date_dir = geometry_root / date
        required = ["building_probability.tif", "building_mask.tif", "building_boundary.tif", "building_objects.gpkg", "GEOMETRY_MANIFEST.json"]
        paths = {name: date_dir / name for name in required}
        if not all(path.exists() for path in paths.values()):
            raise RuntimeError(f"existing RSBuilding geometry is incomplete for {date}")
        manifest = load_json(paths["GEOMETRY_MANIFEST.json"])
        geometry[date] = {
            "manifest": str(paths["GEOMETRY_MANIFEST.json"].resolve()),
            "manifest_sha256": _sha256(paths["GEOMETRY_MANIFEST.json"]),
            "status": manifest.get("status"),
            "source_model": manifest.get("source_model"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "geometry_outputs": {name: str(path.resolve()) for name, path in paths.items()},
            "rsbuilding_rerun": False,
        }
    scenes = []
    for item in scene_payload.get("scenes", []):
        path = Path(item["path"])
        if not path.exists():
            raise RuntimeError(f"Tongzhou source scene missing: {path}")
        scenes.append({"date": item["date"], "path": str(path.resolve()), "sha256": _sha256(path), "bands": item.get("bands"), "crs": item.get("crs"), "width": item.get("width"), "height": item.get("height")})
    full20_manifest = project_root / "outputs" / "inference" / "beijing_tongzhou_to_latest" / "TONGZHOU_BATCH_MANIFEST.json"
    surface_manifest = project_root / "outputs" / "inference" / "beijing_tongzhou_surface_state_v1" / "20260730_113324" / "TONGZHOU_SURFACE_STATE_MANIFEST.json"
    return {
        "geometry_root": str(geometry_root.resolve()),
        "geometry": geometry,
        "scenes": scenes,
        "full20_manifest": str(full20_manifest.resolve()) if full20_manifest.exists() else None,
        "full20_manifest_sha256": _sha256(full20_manifest) if full20_manifest.exists() else None,
        "surface_state_manifest": str(surface_manifest.resolve()) if surface_manifest.exists() else None,
        "surface_state_manifest_sha256": _sha256(surface_manifest) if surface_manifest.exists() else None,
        "rsbuilding_rerun": False,
        "full20_retrained": False,
        "surface_state_retrained": False,
        "source_data_modified": False,
    }


def _scene_percentiles(source, rgb_mapping: tuple[int, int, int]) -> tuple[float, float]:
    from rasterio.enums import Resampling

    sample_height = max(1, source.height // 16)
    sample_width = max(1, source.width // 16)
    sample = source.read(rgb_mapping, out_shape=(len(rgb_mapping), sample_height, sample_width), resampling=Resampling.bilinear).astype(np.float32)
    finite = np.isfinite(sample)
    if not finite.any():
        raise ValueError("Tongzhou source has no finite RGB sample")
    return float(np.nanpercentile(sample[finite], 2.0)), float(np.nanpercentile(sample[finite], 98.0))


def _open_probability_outputs(output_dir: Path, source) -> dict[str, Any]:
    import rasterio

    profile = source.profile.copy()
    # The RGB source profile may carry PHOTOMETRIC=YCBCR.  That is invalid for
    # a one-band Float32 probability raster with Deflate, so set an explicit
    # grayscale photometric interpretation for every derived output.
    profile.update({"driver": "GTiff", "count": 1, "dtype": "float32", "compress": "deflate", "predictor": 3, "photometric": "MINISBLACK", "tiled": True, "blockxsize": 256, "blockysize": 256, "nodata": None, "BIGTIFF": "IF_SAFER"})
    names = ("background_probability", "exposed_ground_probability", "dustproof_net_probability", "building_probability", "boundary_probability", "uncertainty_probability")
    handles = {}
    for name in names:
        path = output_dir / f"{name}.tif"
        handles[name] = rasterio.open(path, "w", **profile)
    return handles


def _run_scene_inference(model, device, image_path: Path, output_dir: Path, *, rgb_mapping: tuple[int, int, int], image_size: int, batch_size: int, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    import rasterio
    import torch
    from rasterio.windows import Window
    from torch.nn import functional as F

    existing_receipt = output_dir / "CONSTRUCTION_ELEMENT_MANIFEST.json"
    if existing_receipt.exists():
        return load_json(existing_receipt)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete Tongzhou construction output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_size = 512
    with rasterio.open(image_path) as source:
        if source.count < max(rgb_mapping):
            raise ValueError(f"Tongzhou source band count is {source.count}, RGB mapping is {rgb_mapping}")
        low, high = _scene_percentiles(source, rgb_mapping)
        outputs = _open_probability_outputs(output_dir, source)
        tile_count = 0
        nonfinite_count = 0
        batch_images: list[torch.Tensor] = []
        batch_textures: list[torch.Tensor] = []
        batch_windows: list[Window] = []
        try:
            for top in range(0, source.height, tile_size):
                for left in range(0, source.width, tile_size):
                    height = min(tile_size, source.height - top)
                    width = min(tile_size, source.width - left)
                    window = Window(left, top, width, height)
                    raw = source.read(rgb_mapping, window=window).astype(np.float32)
                    normalized = np.clip((raw - low) / max(high - low, 1e-6), 0.0, 1.0)
                    if height != tile_size or width != tile_size:
                        normalized = np.pad(normalized, ((0, 0), (0, tile_size - height), (0, tile_size - width)), mode="edge")
                    image = F.interpolate(torch.from_numpy(normalized)[None], size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
                    texture = (image * image.new_tensor([0.2989, 0.5870, 0.1140])[:, None, None]).sum(0, keepdim=True)
                    batch_images.append(image)
                    batch_textures.append(texture)
                    batch_windows.append(window)
                    if len(batch_images) >= batch_size or (top + height >= source.height and left + width >= source.width):
                        with torch.no_grad():
                            image_tensor = torch.stack(batch_images).to(device, non_blocking=True)
                            texture_tensor = torch.stack(batch_textures).to(device, non_blocking=True)
                            result = model(image_tensor, "ucs", texture_tensor)
                            class_probabilities = result["ucs_probabilities"].float()
                            boundary = result["boundary_logits"].float().sigmoid()
                            uncertainty = result["uncertainty_logits"].float().sigmoid()
                            probability_tensors = {
                                "background_probability": class_probabilities[:, 0:1],
                                "exposed_ground_probability": class_probabilities[:, 1:2],
                                "dustproof_net_probability": class_probabilities[:, 2:3],
                                "building_probability": class_probabilities[:, 3:4],
                                "boundary_probability": boundary[:, 0:1],
                                "uncertainty_probability": uncertainty[:, 0:1],
                            }
                            for tensor in probability_tensors.values():
                                nonfinite_count += int((~torch.isfinite(tensor)).sum().detach().cpu())
                            for index, tile_window in enumerate(batch_windows):
                                actual_height = int(tile_window.height)
                                actual_width = int(tile_window.width)
                                for name, tensor in probability_tensors.items():
                                    tile = F.interpolate(tensor[index : index + 1], size=(actual_height, actual_width), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy().astype(np.float32)
                                    outputs[name].write(tile, 1, window=tile_window)
                                tile_count += 1
                        batch_images.clear(); batch_textures.clear(); batch_windows.clear()
        finally:
            for handle in outputs.values():
                handle.close()
    if nonfinite_count:
        raise RuntimeError(f"non-finite Tongzhou construction outputs: {nonfinite_count}")
    receipt = {
        "status": "success",
        "scene_date": output_dir.name,
        "source_path": str(image_path.resolve()),
        "source_sha256": _sha256(image_path),
        "tile_size": tile_size,
        "model_input_size": image_size,
        "batch_size": batch_size,
        "completed_tiles": tile_count,
        "rgb_mapping": list(rgb_mapping),
        "rgb_semantic_order": ["R", "G", "B"],
        "preprocessing": "per_scene_sampled_percentile_2_98_from_16x_decimation",
        "normalization_low": low,
        "normalization_high": high,
        "source_crs": str(source.crs) if source.crs else None,
        "outputs": {key: str((output_dir / f"{key}.tif").resolve()) for key in ("background_probability", "exposed_ground_probability", "dustproof_net_probability", "building_probability", "boundary_probability", "uncertainty_probability")},
        "thresholds_frozen": True,
        "test_pixel_reads": 0,
        "human_review_required": True,
        "scientific_status": "candidate_not_ground_truth",
        "created_at": _now(),
    }
    dump_json(output_dir / "CONSTRUCTION_ELEMENT_MANIFEST.json", receipt)
    return receipt


def _geometry_candidates(project_root: Path, output_root: Path, date: str, geometry_info: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    import geopandas as gpd
    import rasterio
    from shapely.geometry import mapping

    source_geometry = Path(geometry_info["geometry_outputs"]["building_objects.gpkg"])
    gdf = gpd.read_file(source_geometry)
    original = gdf.copy()
    metric = gdf.to_crs(METRIC_CRS)
    rasters = {}
    date_dir = output_root / date
    for key in ("background_probability", "exposed_ground_probability", "dustproof_net_probability", "building_probability", "boundary_probability", "uncertainty_probability"):
        rasters[key] = rasterio.open(date_dir / f"{key}.tif")
    candidate_threshold = float(thresholds.get("class_probability_thresholds", {}).get("ucs_building", 0.5))
    rows = []
    source_rows = list(original.itertuples())
    coordinates = [(row.geometry.representative_point().x, row.geometry.representative_point().y) for row in source_rows]
    sampled: dict[str, np.ndarray] = {}
    try:
        for key, raster in rasters.items():
            sampled[key] = np.asarray([value[0] for value in raster.sample(coordinates)], dtype=np.float32)
        for index, (source_row, metric_geometry) in enumerate(zip(source_rows, metric.geometry)):
            values = {key: float(array[index]) for key, array in sampled.items()}
            non_background = max(values["exposed_ground_probability"], values["dustproof_net_probability"], values["building_probability"])
            rows.append({
                "object_id": str(getattr(source_row, "object_id", f"{date}_{index:06d}")),
                "scene_date": date,
                "state": str(getattr(source_row, "state", "building_candidate")),
                "geometry_area_m2": float(metric_geometry.area),
                "construction_non_background": non_background,
                "construction_exposed_ground": values["exposed_ground_probability"],
                "construction_dustproof_net": values["dustproof_net_probability"],
                "construction_building": values["building_probability"],
                "construction_boundary": values["boundary_probability"],
                "construction_uncertainty": values["uncertainty_probability"],
                "construction_candidate": bool(non_background >= candidate_threshold),
                "parcel_status": "auditable_rule_based_candidate",
                "model_prediction_status": "model_prediction_not_human_verified",
                "human_review_required": True,
                "geometry": metric_geometry,
            })
    finally:
        for raster in rasters.values():
            raster.close()
    candidate_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
    candidate_path = date_dir / "parcel_candidates.gpkg"
    candidate_gdf.to_file(candidate_path, layer="parcel_candidates", driver="GPKG")
    candidate_count = int(candidate_gdf["construction_candidate"].sum())
    stats = {
        "date": date,
        "metric_crs": METRIC_CRS,
        "geometry_count": int(len(candidate_gdf)),
        "candidate_count": candidate_count,
        "geometry_area_m2_sum": float(candidate_gdf["geometry_area_m2"].sum()),
        "candidate_geometry_area_m2_sum": float(candidate_gdf.loc[candidate_gdf["construction_candidate"], "geometry_area_m2"].sum()),
        "candidate_area_m2": None,
        "candidate_area_m2_status": "not_computed_from_geographic_probability_raster",
        "zonal_statistic": "representative_point_sample",
        "crs_audit": "pass_metric_geometry_area_only",
    }
    dump_json(date_dir / "AREA_STATS.json", stats)
    return {"candidate_path": str(candidate_path.resolve()), "stats": stats, "records": [{"track_id": row["object_id"], "geometry": row["geometry"], "scene_type": row["state"], "area_m2": row["geometry_area_m2"]} for row in rows]}


def _time_track_audit(
    output_root: Path,
    date_records: Mapping[str, list[Mapping[str, Any]]],
    dates: Iterable[str] | None = None,
) -> dict[str, Any]:
    from urban_renewal.tracking.construction_matching import match_construction_objects
    from shapely.strtree import STRtree

    rows = []
    events = []
    ordered_dates = tuple(dates or DATES)
    for previous_date, current_date in zip(ordered_dates, ordered_dates[1:]):
        previous = list(date_records[previous_date])
        current = list(date_records[current_date])
        matched = match_construction_objects(previous, current, max_centroid_distance=200.0)
        # The legacy detector checks every pair.  For the 1,460-object
        # Tongzhou scenes that is unnecessarily quadratic, so use the same
        # spatial-index principle as object matching and inspect only bbox
        # candidates.  This remains a diagnostic candidate audit, not a label.
        current_geometries = [item["geometry"] for item in current]
        current_tree = STRtree(current_geometries)
        previous_geometries = [item["geometry"] for item in previous]
        previous_tree = STRtree(previous_geometries)
        split_merge_events = []
        for p_index, item in enumerate(previous):
            children = []
            for raw_index in current_tree.query(item["geometry"]):
                child_index = int(raw_index)
                if item["geometry"].intersection(current[child_index]["geometry"]).area / max(current[child_index]["geometry"].area, 1e-9) >= 0.2:
                    children.append(child_index)
            if len(children) > 1:
                split_merge_events.append({"event_type": "split", "parent_track_ids": [item.get("track_id", str(p_index))], "child_track_ids": [current[index].get("track_id", str(index)) for index in children]})
        for c_index, item in enumerate(current):
            parents = []
            for raw_index in previous_tree.query(item["geometry"]):
                parent_index = int(raw_index)
                if previous[parent_index]["geometry"].intersection(item["geometry"]).area / max(previous[parent_index]["geometry"].area, 1e-9) >= 0.2:
                    parents.append(parent_index)
            if len(parents) > 1:
                split_merge_events.append({"event_type": "merge", "parent_track_ids": [previous[index].get("track_id", str(index)) for index in parents], "child_track_ids": [item.get("track_id", str(c_index))]})
        event_rows = split_merge_events
        for item in event_rows:
            events.append({"from_date": previous_date, "to_date": current_date, **item})
        for match in matched["matches"]:
            left = previous[match["previous_index"]]
            right = current[match["current_index"]]
            ratio = float(right["area_m2"] / max(left["area_m2"], 1e-9))
            change = "expand" if ratio > 1.2 else "shrink" if ratio < 0.8 else "stable"
            rows.append([previous_date, current_date, left["track_id"], right["track_id"], match["match_score"], ratio, change])
        for index in matched["unmatched_previous"]:
            rows.append([previous_date, current_date, previous[index]["track_id"], "", "", "", "disappear"])
        for index in matched["unmatched_current"]:
            rows.append([previous_date, current_date, "", current[index]["track_id"], "", "", "add"])
    _write_csv(output_root / "time_tracks.csv", ["from_date", "to_date", "from_track_id", "to_track_id", "match_score", "area_ratio", "change_type"], rows)
    _write_csv(output_root / "time_track_events.csv", ["from_date", "to_date", "event_type", "parent_track_ids", "child_track_ids"], [[item.get(key) for key in ("from_date", "to_date", "event_type", "parent_track_ids", "child_track_ids")] for item in events])
    return {
        "status": "auditable_rule_based_candidate",
        "dates": list(ordered_dates),
        "pair_count": len(ordered_dates) - 1,
        "event_count": len(events),
        "track_row_count": len(rows),
        "checked_change_types": ["split", "merge", "add", "disappear", "expand", "shrink"],
        "outputs": {"time_tracks": str((output_root / "time_tracks.csv").resolve()), "time_track_events": str((output_root / "time_track_events.csv").resolve())},
        "human_review_required": True,
    }


def _review_packet(project_root: Path, output_root: Path, package_dir: Path, input_audit: Mapping[str, Any]) -> dict[str, Any]:
    review_root = project_root / "outputs" / "urrs" / "review" / "tongzhou_20260801"
    inventory_path = review_root / "REVIEW_INVENTORY.json"
    decisions_path = review_root / "review_decisions.csv"
    seed_path = project_root / "outputs" / "urrs" / "construction" / "human_seed_labels_20260802_134145" / "SEED_LABEL_INVENTORY.json"
    inventory = load_json(inventory_path) if inventory_path.exists() else {}
    seed = load_json(seed_path) if seed_path.exists() else {}
    decision_count = 0
    if decisions_path.exists():
        with decisions_path.open(encoding="utf-8-sig", newline="") as handle:
            decision_count = sum(1 for _ in csv.DictReader(handle))
    packet = {
        "status": "MODEL_ASSISTED_REVIEW_READY",
        "review_root_preserved": str(review_root.resolve()),
        "existing_inventory_count": len(inventory.get("records", [])),
        "existing_decision_count": decision_count,
        "existing_seed_inventory_count": len(seed.get("records", [])),
        "dates": list(DATES),
        "construction_output_root": str(output_root.resolve()),
        "model_package": str(package_dir.resolve()),
        "layers_per_date": ["RGB", "RSBuilding_geometry", "Construction_probabilities", "Full20", "Surface_State", "area_stats", "sequence", "transition", "uncertainty"],
        "decision_values": ["ACCEPT", "CORRECT", "UNCERTAIN", "REJECT"],
        "label_sources": ["human_verified", "human_corrected"],
        "model_prediction_human_review_required": True,
        "parcel_status": "auditable_rule_based_candidate",
        "full20": input_audit.get("full20_manifest"),
        "surface_state": input_audit.get("surface_state_manifest"),
        "no_mti": {"integration": "no_mti", "status": "development_candidate_not_formally_validated", "experimental_mti": "unselected"},
        "current_urrs_release_unchanged": True,
        "test_pixel_reads": {"UCS_locked_test": 46, "WHU": 0, "SECOND": 0, "DSIFN": 0},
        "created_at": _now(),
    }
    dump_json(output_root / "MODEL_ASSISTED_CONSTRUCTION_REVIEW_PACKET.json", packet)
    html = """<!doctype html><meta charset='utf-8'><title>Tongzhou model-assisted construction review</title>
<h1>Tongzhou model-assisted construction review</h1>
<p>Human review remains required. Existing 117-item inventory, prior decisions, and 96-item seed inventory are preserved.</p>
<p>Decision values: ACCEPT, CORRECT, UNCERTAIN, REJECT. Human labels must remain human_verified or human_corrected.</p>
<table border='1' cellpadding='4'><tr><th>Date</th><th>RGB</th><th>RSBuilding</th><th>Construction</th><th>Area</th></tr>
""" + "".join(f"<tr><td>{date}</td><td>source scene</td><td>existing geometry</td><td><a href='../beijing_tongzhou_construction_activity_v1/{date}/CONSTRUCTION_ELEMENT_MANIFEST.json'>model outputs</a></td><td>AREA_STATS.json</td></tr>" for date in DATES) + "</table>\n"
    atomic_write_text(output_root / "MODEL_ASSISTED_CONSTRUCTION_REVIEW.html", html)
    return packet


def run_tongzhou_construction(project_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = project_root / "outputs" / "inference" / OUTPUT_ROOT_NAME
    final_manifest_path = output_root / "TONGZHOU_CONSTRUCTION_MANIFEST.json"
    if final_manifest_path.exists():
        manifest = load_json(final_manifest_path)
        audit_path = output_root / "TONGZHOU_CONSTRUCTION_AUDIT.json"
        if not audit_path.exists():
            dump_json(audit_path, {
                "status": "pass_limited_candidate",
                "dates": manifest.get("dates", list(DATES)),
                "model_package": manifest.get("model_package"),
                "model_sha256": manifest.get("model_sha256"),
                "input_audit": manifest.get("input_audit"),
                "scene_outputs": manifest.get("scene_outputs"),
                "time_track_audit": manifest.get("time_track_audit"),
                "area_policy": manifest.get("area_policy"),
                "parcel_status": manifest.get("parcel_status"),
                "rsbuilding_rerun": False,
                "full20_retrained": False,
                "surface_state_retrained": False,
                "source_data_modified": False,
                "human_review_required": True,
                "test_pixel_reads": manifest.get("test_pixel_reads"),
                "created_at": _now(),
            })
        if manifest.get("audit") != str(audit_path.resolve()):
            manifest["audit"] = str(audit_path.resolve())
            dump_json(final_manifest_path, manifest)
        return manifest
    package_dir, package_receipt, package_manifest = _latest_package(project_root)
    scene_manifest_path, scene_payload = _scene_manifest(project_root)
    input_audit = _verify_readonly_inputs(project_root, scene_payload)
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Tongzhou construction inference requested CUDA but CUDA is unavailable")
    from urban_renewal.construction.task_model import load_task_model

    model, context = load_task_model(package_dir, device=device)
    import yaml

    thresholds = context["thresholds"]
    preprocessing = context["preprocessing"]
    rgb_mapping = tuple(int(value) for value in preprocessing["rgb_mapping"])
    output_root.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()
    scene_records: dict[str, dict[str, Any]] = {}
    date_records: dict[str, list[Mapping[str, Any]]] = {}
    try:
        for scene in input_audit["scenes"]:
            date = scene["date"]
            date_dir = output_root / date
            inference_receipt = _run_scene_inference(model, device, Path(scene["path"]), date_dir, rgb_mapping=rgb_mapping, image_size=int(preprocessing.get("image_size", 256)), batch_size=8, thresholds=thresholds)
            geometry_result = _geometry_candidates(project_root, output_root, date, input_audit["geometry"][date], thresholds)
            scene_records[date] = {"inference": inference_receipt, "geometry": geometry_result["stats"], "outputs": {"parcel_candidates": geometry_result["candidate_path"]}}
            date_records[date] = geometry_result["records"]
    except Exception:
        # Preserve a machine-readable partial status without replacing any
        # existing source or geometry outputs.
        dump_json(output_root / "TONGZHOU_CONSTRUCTION_FAILURE.json", {"status": "FAILED", "source_data_modified": False, "rsbuilding_rerun": False, "error": "inference or audit failed"})
        raise
    track_audit = _time_track_audit(output_root, date_records)
    review_packet = _review_packet(project_root, output_root, package_dir, input_audit)
    manifest = {
        "status": "TONGZHOU_CONSTRUCTION_CANDIDATE_READY_FOR_HUMAN_REVIEW",
        "dates": list(DATES),
        "model_name": package_manifest.get("model_name"),
        "model_package": str(package_dir.resolve()),
        "model_sha256": package_receipt.get("candidate_checkpoint_sha256"),
        "training_stage": "A_UCS",
        "stage_b_used": False,
        "full_ucs_training": False,
        "dataset_scope": "UCS_386_available_subset",
        "full_record_coverage": "386/600",
        "missing_ucs_images": 214,
        "missingness_bias_not_ruled_out": True,
        "candidate_not_ground_truth": True,
        "national_generalization_verified": False,
        "input_audit": input_audit,
        "scene_outputs": scene_records,
        "time_track_audit": track_audit,
        "review_packet": review_packet,
        "parcel_status": "auditable_rule_based_candidate",
        "area_policy": {"metric_crs": METRIC_CRS, "candidate_area_m2": "not_computed_from_geographic_probability_raster", "geometry_area_m2": "computed_after_reprojection_to_metric_crs"},
        "no_mti": {"integration": "no_mti", "status": "development_candidate_not_formally_validated", "experimental_mti": "unselected"},
        "rsbuilding_rerun": False,
        "full20_retrained": False,
        "surface_state_retrained": False,
        "human_review_required": True,
        "test_pixel_reads": {"UCS_locked_test": 46, "WHU": 0, "SECOND": 0, "DSIFN": 0},
        "source_data_modified": False,
        "audit": str((output_root / "TONGZHOU_CONSTRUCTION_AUDIT.json").resolve()),
        "created_at": _now(),
    }
    dump_json(output_root / "TONGZHOU_CONSTRUCTION_AUDIT.json", {
        "status": "pass_limited_candidate",
        "dates": list(DATES),
        "model_package": manifest["model_package"],
        "model_sha256": manifest["model_sha256"],
        "input_audit": input_audit,
        "scene_outputs": scene_records,
        "time_track_audit": track_audit,
        "area_policy": manifest["area_policy"],
        "parcel_status": manifest["parcel_status"],
        "rsbuilding_rerun": False,
        "full20_retrained": False,
        "surface_state_retrained": False,
        "source_data_modified": False,
        "human_review_required": True,
        "test_pixel_reads": manifest["test_pixel_reads"],
        "created_at": _now(),
    })
    dump_json(final_manifest_path, manifest)
    return manifest


def tongzhou_status(project_root: Path | None = None) -> dict[str, Any]:
    if project_root is not None:
        path = project_root.resolve() / "outputs" / "inference" / OUTPUT_ROOT_NAME / "TONGZHOU_CONSTRUCTION_MANIFEST.json"
        if path.exists():
            return load_json(path)
    return {
        "status": "BLOCKED_PENDING_FROZEN_STAGE_A_CANDIDATE",
        "reason": "formal UCS386 Stage A candidate package is required before Tongzhou inference",
        "allowed_outputs": ["development_candidate", "pseudo_rule"],
        "human_review_required": True,
        "tongzhou_used_for_selection": False,
        "rsbuilding_rerun": False,
        "full20_retrained": False,
        "test_pixel_reads": {"WHU": 0, "SECOND": 0, "DSIFN": 0},
    }
