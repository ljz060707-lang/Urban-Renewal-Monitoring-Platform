from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy
import warnings

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "urban-renewal-phase1", "version": "0.4.0", "schema_version": "0.4.0"},
    "data": {
        "image_glob": ["*.tif", "*.tiff"],
        "label_glob": ["*.gpkg", "*.geojson", "*.shp"],
        "rgb_bands": [1, 2, 3],
        "min_building_area_m2": 10.0,
        "classes": ["building"],
    },
    "tiling": {
        "tile_size": 512,
        "overlap": 64,
        "min_object_visibility": 0.30,
        "include_empty_tiles": True,
        "max_empty_to_positive_ratio": 1.0,
    },
    "split": {
        "strategy": "spatial_blocks",
        "validation_fraction": 0.20,
        "block_size_m": 500,
        "seed": 42,
    },
    "model": {
        "architecture": "maskrcnn_resnet50_fpn",
        "num_classes": 2,
        "pretrained": False,
        "weights_path": None,
        "allow_pretrained_download": False,
        "min_size": 512,
        "max_size": 1024,
    },
    "training": {
        "epochs": 30,
        "batch_size": "auto",
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "num_workers": "auto",
        "amp": True,
        "gradient_clip_norm": 5.0,
        "early_stopping_patience": 7,
        "save_every_epochs": 1,
        "resume": True,
        "seed": 42,
    },
    "inference": {
        "score_threshold": 0.50,
        "mask_threshold": 0.50,
        "nms_iou_threshold": 0.40,
    },
    "confidence": {
        "high_threshold": 0.75,
        "medium_threshold": 0.50,
        "weights": {
            "detection_score": 0.20,
            "valid_observations": 0.12,
            "interval_width": 0.10,
            "usable_fraction": 0.12,
            "match_score": 0.12,
            "ambiguity_margin": 0.10,
            "registration": 0.08,
            "off_nadir_consistency": 0.05,
            "canonical_geometry": 0.04,
            "lineage_clarity": 0.04,
            "manual_review": 0.03,
        },
    },
    "regularization": {
        "simplify_tolerance_m": 0.35,
        "orthogonalize": True,
        "rectangularity_threshold": 0.82,
        "min_area_m2": 4.0,
        "max_area_change_ratio": 1.20,
        "max_centroid_shift_m": 2.0,
        "max_hausdorff_distance_m": 4.0,
        "preserve_holes": True,
        "min_hole_area_m2": 5.0,
    },
    "quality": {"min_quality_fraction": 0.70},
    "tracking": {
        "min_iou": 0.08,
        "min_buffer_iou": 0.12,
        "min_bbox_iou": 0.05,
        "max_centroid_distance_m": 25.0,
        "buffer_m": 4.0,
        "off_nadir_scale_m_per_degree": 0.35,
        "max_dynamic_buffer_m": 20.0,
        "ambiguity_margin_threshold": 0.08,
        "max_identity_area_ratio": 1.8,
        "match_weights": {"iou": 0.15, "buffer_iou": 0.35, "bbox_iou": 0.10, "centroid": 0.25, "area": 0.10, "shape": 0.05},
        "min_persistence": 2,
        "max_track_gap": 2,
        "max_track_gap_days": 1100,
        "max_soft_gap_epochs": 2,
        "max_soft_gap_days": 1100,
        "min_quality_fraction": 0.80,
        "isolated_gap_repair": True,
        "construction_area_ratio": 1.50,
        "construction_gap_buffer_m": 4.0,
        "construction_gap_max_buffer_iou": 0.35,
        "low_confidence_threshold": 0.55,
        "force_new_track_on_merge": True,
        "lineage": {"min_component_overlap": 0.25, "min_event_coverage": 0.70, "min_counterpart_coverage": 0.25, "buffer_m": 2.0},
    },
    "review": {"chip_padding_m": 25.0, "review_version": "0.4.0"},
    "student_export": {
        "target_resolution_m": 10.0,
        "min_mapping_unit_m2": 200.0,
        "supersample": 4,
        "min_fraction_to_keep": 0.05,
        "require_reviewed_or_high_confidence": True,
    },
    "crs": {
        "source_policy": "preserve",
        "analysis_strategy": "local_utm",
        "exchange_crs": "EPSG:4326",
        "display_crs": "EPSG:3857",
        "reject_missing_crs": True,
        "reject_geographic_metric_operations": True,
        "reject_web_mercator_metric_operations": True,
        "warn_cross_zone_width_km": 100,
    },
    "provenance": {"write_run_manifest": True, "hash_inputs": True, "record_git_commit": True},
}


def merge_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one resolved config; all v0.4 defaults live in this module."""
    resolved = deepcopy(DEFAULT_CONFIG)

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(resolved, overrides or {})
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    for key in ("project", "crs", "provenance"):
        if key not in cfg:
            warnings.warn(f"Legacy config: using default for {key}", stacklevel=2)
    resolved = merge_config(cfg)
    resolved["_config_path"] = str(config_path.resolve())
    return resolved


def load_project_config(project_dir: str | Path) -> dict[str, Any]:
    """Load optional data/<aoi_id>/project.yaml over the central defaults."""
    path = Path(project_dir) / "project.yaml"
    if not path.exists():
        return merge_config()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"project.yaml must be a mapping: {path}")
    resolved = merge_config(payload)
    resolved["_config_path"] = str(path.resolve())
    return resolved


def resolve_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if base is None:
        return path.resolve()
    return (Path(base) / path).resolve()
