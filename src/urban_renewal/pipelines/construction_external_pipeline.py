from __future__ import annotations

"""Inference-only Construction Activity orchestration for external scenes.

The model and tile runner remain the canonical implementations in
``construction_tongzhou_pipeline``.  This module only supplies a scene-set
adapter so an independently audited external AOI can be processed without
reusing or mutating Tongzhou outputs or human-review records.
"""

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

from urban_renewal.construction.json_io import dump_json, load_json

from .construction_tongzhou_pipeline import (
    METRIC_CRS,
    _geometry_candidates,
    _latest_package,
    _run_scene_inference,
    _sha256,
    _time_track_audit,
)


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _readonly_geometry_inputs(geometry_root: Path, dates: Iterable[str]) -> dict[str, dict[str, Any]]:
    geometry: dict[str, dict[str, Any]] = {}
    required_names = (
        "building_probability.tif",
        "building_mask.tif",
        "building_boundary.tif",
        "building_objects.gpkg",
        "GEOMETRY_MANIFEST.json",
    )
    for date in dates:
        date_dir = geometry_root / date
        paths = {name: date_dir / name for name in required_names}
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise RuntimeError(f"RSBuilding geometry is incomplete for {date}: {missing}")
        manifest = load_json(paths["GEOMETRY_MANIFEST.json"])
        if manifest.get("status") != "success":
            raise RuntimeError(f"RSBuilding geometry is not successful for {date}")
        geometry[date] = {
            "manifest": str(paths["GEOMETRY_MANIFEST.json"].resolve()),
            "manifest_sha256": _sha256(paths["GEOMETRY_MANIFEST.json"]),
            "status": manifest.get("status"),
            "source_model": manifest.get("source_model"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "geometry_outputs": {name: str(path.resolve()) for name, path in paths.items()},
            "rsbuilding_rerun": False,
        }
    return geometry


def _review_record_snapshot(project_root: Path) -> dict[str, Any]:
    """Read existing review artefacts without writing to their directories."""
    review_root = project_root / "outputs" / "urrs" / "review" / "tongzhou_20260801"
    inventory_path = review_root / "REVIEW_INVENTORY.json"
    decisions_path = review_root / "review_decisions.csv"
    seed_path = (
        project_root
        / "outputs"
        / "urrs"
        / "construction"
        / "human_seed_labels_20260802_134145"
        / "SEED_LABEL_INVENTORY.json"
    )
    inventory = load_json(inventory_path) if inventory_path.exists() else {}
    seed = load_json(seed_path) if seed_path.exists() else {}
    return {
        "review_root": str(review_root.resolve()),
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _sha256(inventory_path) if inventory_path.exists() else None,
        "inventory_count": len(inventory.get("records", [])),
        "decision_path": str(decisions_path.resolve()),
        "decision_sha256": _sha256(decisions_path) if decisions_path.exists() else None,
        "decision_count": _count_csv_rows(decisions_path),
        "seed_inventory_path": str(seed_path.resolve()),
        "seed_inventory_sha256": _sha256(seed_path) if seed_path.exists() else None,
        "seed_inventory_count": len(seed.get("records", [])),
        "preserved": True,
    }


def run_external_construction(
    project_root: Path,
    *,
    scenes: list[Mapping[str, Any]],
    geometry_root: Path,
    output_root: Path,
    full20_manifest: str | None,
    surface_state_manifest: str | None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the frozen UCS386 candidate on an external scene set."""
    project_root = project_root.resolve()
    geometry_root = geometry_root.resolve()
    output_root = output_root.resolve()
    dates = tuple(str(scene["date"]) for scene in scenes)
    if len(dates) < 1 or dates != tuple(sorted(dates)) or len(set(dates)) != len(dates):
        raise ValueError("External construction scenes must be unique and date-sorted")

    final_manifest_path = output_root / "CONSTRUCTION_EXTERNAL_MANIFEST.json"
    if final_manifest_path.exists():
        return load_json(final_manifest_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete external construction output: {output_root}")

    package_dir, package_receipt, package_manifest = _latest_package(project_root)
    if package_manifest.get("rgb_mapping") != [3, 2, 1]:
        raise RuntimeError("Frozen UCS386 package RGB mapping is not the recorded 3,2,1 mapping")
    geometry = _readonly_geometry_inputs(geometry_root, dates)
    input_scenes = []
    for scene in scenes:
        path = Path(str(scene["path"])).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        input_scenes.append(
            {
                "date": str(scene["date"]),
                "path": str(path),
                "sha256": _sha256(path),
                "width": scene.get("width"),
                "height": scene.get("height"),
                "crs": scene.get("crs"),
            }
        )

    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("External construction inference requested CUDA but CUDA is unavailable")

    from urban_renewal.construction.task_model import load_task_model

    model, context = load_task_model(package_dir, device=device)
    model.to(device)
    model.eval()
    thresholds = context["thresholds"]
    preprocessing = context["preprocessing"]
    rgb_mapping = tuple(int(value) for value in preprocessing["rgb_mapping"])
    output_root.mkdir(parents=True, exist_ok=True)
    scene_records: dict[str, dict[str, Any]] = {}
    date_records: dict[str, list[Mapping[str, Any]]] = {}
    try:
        for scene in input_scenes:
            date = scene["date"]
            date_dir = output_root / date
            inference = _run_scene_inference(
                model,
                device,
                Path(scene["path"]),
                date_dir,
                rgb_mapping=rgb_mapping,
                image_size=int(preprocessing.get("image_size", 256)),
                batch_size=8,
                thresholds=thresholds,
            )
            geometry_result = _geometry_candidates(
                project_root,
                output_root,
                date,
                geometry[date],
                thresholds,
            )
            scene_records[date] = {
                "inference": inference,
                "geometry": geometry_result["stats"],
                "outputs": {"parcel_candidates": geometry_result["candidate_path"]},
            }
            date_records[date] = geometry_result["records"]
    except Exception as exc:
        dump_json(
            output_root / "CONSTRUCTION_EXTERNAL_FAILURE.json",
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_data_modified": False,
                "rsbuilding_rerun": False,
                "full20_retrained": False,
                "surface_state_retrained": False,
                "test_pixel_reads": {"UCS_locked_test": 0, "WHU": 0, "SECOND": 0, "DSIFN": 0},
            },
        )
        raise

    track_audit = _time_track_audit(output_root, date_records, dates)
    review_snapshot = _review_record_snapshot(project_root)
    review_packet = {
        "status": "MODEL_ASSISTED_EXTERNAL_REVIEW_READY",
        "external_sample": "nanjing_jiangbei",
        "dates": list(dates),
        "construction_output_root": str(output_root.resolve()),
        "model_package": str(package_dir.resolve()),
        "model_prediction_human_review_required": True,
        "candidate_not_ground_truth": True,
        "review_records_preserved": review_snapshot,
        "no_mti": {
            "integration": "no_mti",
            "status": "development_candidate_not_formally_validated",
            "experimental_mti": "unselected",
        },
        "test_pixel_reads": {"UCS_locked_test": 0, "WHU": 0, "SECOND": 0, "DSIFN": 0},
    }
    dump_json(output_root / "MODEL_ASSISTED_EXTERNAL_REVIEW_PACKET.json", review_packet)
    manifest = {
        "status": "EXTERNAL_CONSTRUCTION_CANDIDATE_READY_FOR_HUMAN_REVIEW",
        "external_sample": "nanjing_jiangbei",
        "dates": list(dates),
        "model_name": package_manifest.get("model_name"),
        "model_package": str(package_dir.resolve()),
        "model_sha256": package_receipt.get("candidate_checkpoint_sha256"),
        "training_stage": "A_UCS",
        "stage_b_used": False,
        "dataset_scope": "UCS_386_available_subset",
        "full_record_coverage": "386/600",
        "missing_ucs_images": 214,
        "missingness_bias_not_ruled_out": True,
        "candidate_not_ground_truth": True,
        "national_generalization_verified": False,
        "rgb_mapping": list(rgb_mapping),
        "input_scenes": input_scenes,
        "geometry_root": str(geometry_root),
        "geometry": geometry,
        "scene_outputs": scene_records,
        "time_track_audit": track_audit,
        "review_packet": str((output_root / "MODEL_ASSISTED_EXTERNAL_REVIEW_PACKET.json").resolve()),
        "full20_manifest": full20_manifest,
        "surface_state_manifest": surface_state_manifest,
        "parcel_status": "auditable_rule_based_candidate",
        "area_policy": {
            "metric_crs": METRIC_CRS,
            "candidate_area_m2": "not_computed_from_geographic_probability_raster",
            "geometry_area_m2": "computed_after_reprojection_to_metric_crs",
        },
        "no_mti": {
            "integration": "no_mti",
            "status": "development_candidate_not_formally_validated",
            "experimental_mti": "unselected",
        },
        "rsbuilding_rerun": False,
        "full20_retrained": False,
        "surface_state_retrained": False,
        "human_review_required": True,
        "test_pixel_reads": {"UCS_locked_test": 0, "WHU": 0, "SECOND": 0, "DSIFN": 0},
        "source_data_modified": False,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    audit = {
        "status": "pass_limited_candidate",
        "external_sample": "nanjing_jiangbei",
        "dates": list(dates),
        "model_package": manifest["model_package"],
        "model_sha256": manifest["model_sha256"],
        "input_scenes": input_scenes,
        "geometry": geometry,
        "scene_outputs": scene_records,
        "time_track_audit": track_audit,
        "area_policy": manifest["area_policy"],
        "source_data_modified": False,
        "human_review_required": True,
        "test_pixel_reads": manifest["test_pixel_reads"],
    }
    dump_json(output_root / "CONSTRUCTION_EXTERNAL_AUDIT.json", audit)
    dump_json(final_manifest_path, manifest)
    return manifest

