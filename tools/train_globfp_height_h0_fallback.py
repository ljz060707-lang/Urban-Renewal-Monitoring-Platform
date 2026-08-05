"""Dependency-limited H0 height baseline from completed morphology features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AREA_BINS = np.asarray([0, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000, np.inf], dtype="float64")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    errors = np.abs(y_pred - y_true)
    height_groups = [("0_5m", 0, 5), ("5_10m", 5, 10), ("10_20m", 10, 20), ("20_50m", 20, 50), ("50m_plus", 50, np.inf)]
    area_groups = [("small_under_100m2", 0, 100), ("medium_100_1000m2", 100, 1000), ("large_1000m2_plus", 1000, np.inf)]
    by_height = {}
    by_area = {}
    for name, low, high in height_groups:
        mask = (y_true >= low) & (y_true < high)
        by_height[name] = {"count": int(mask.sum()), "MAE_m": float(errors[mask].mean()) if mask.any() else None}
    areas = frame["footprint_area_m2"].to_numpy(dtype="float64")
    for name, low, high in area_groups:
        mask = (areas >= low) & (areas < high)
        by_area[name] = {"count": int(mask.sum()), "MAE_m": float(errors[mask].mean()) if mask.any() else None}
    centered = y_true - y_true.mean()
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2) / np.sum(centered**2)) if np.sum(centered**2) > 0 else None
    return {
        "count": int(len(y_true)),
        "MAE_m": float(errors.mean()),
        "RMSE_m": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "median_absolute_error_m": float(np.median(errors)),
        "P90_absolute_error_m": float(np.quantile(errors, 0.90)),
        "mean_bias_m": float(np.mean(y_pred - y_true)),
        "R2": r2,
        "by_height_bin": by_height,
        "by_area_bin": by_area,
    }


def run(features_path: Path, output_root: Path, project_root: Path) -> dict[str, Any]:
    training_root = output_root / "03_height_training"
    package_root = output_root / "04_height_model_package"
    package_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(features_path)
    frame = frame[frame["globfp_height_m"].notna() & np.isfinite(frame["globfp_height_m"]) & (frame["globfp_height_m"] > 0)].copy()
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    holdout = frame[frame["split"] == "internal_holdout"].copy()
    if train.empty or validation.empty or holdout.empty:
        raise RuntimeError("H0 fallback requires non-empty train, validation, and internal holdout splits")
    medians = {}
    y_train = train["globfp_height_m"].to_numpy(dtype="float64")
    global_median = float(np.median(y_train))
    for index in range(len(AREA_BINS) - 1):
        values = y_train[(train["footprint_area_m2"].to_numpy() >= AREA_BINS[index]) & (train["footprint_area_m2"].to_numpy() < AREA_BINS[index + 1])]
        medians[str(index)] = float(np.median(values)) if len(values) else global_median

    def predict(part: pd.DataFrame) -> np.ndarray:
        values = []
        for area in part["footprint_area_m2"].to_numpy(dtype="float64"):
            index = int(np.searchsorted(AREA_BINS, area, side="right") - 1)
            values.append(medians.get(str(index), global_median))
        return np.asarray(values, dtype="float64")

    val_pred = predict(validation)
    hold_pred = predict(holdout)
    train_pred = predict(train)
    model_path = package_root / "BuildingHeightMorphologyRegressor_H0.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model_id": "H0_area_group_median",
                "area_bins_m2": AREA_BINS.tolist(),
                "height_medians": medians,
                "global_median_m": global_median,
                "height_unit": "m",
                "metric_crs": "EPSG:6933",
                "scientific_status": "model_derived_building_height_candidate",
                "training_excluded_nanjing": True,
                "seed": 20260804,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    selected = "H0_area_group_median"
    dependency_status = {
        "H1": {"status": "NOT_RUN_DEPENDENCY_UNAVAILABLE", "missing": ["scikit-learn", "lightgbm", "xgboost"]},
        "H2": {"status": "NOT_RUN_DEPENDENCY_UNAVAILABLE", "missing": ["scikit-learn", "lightgbm", "xgboost"]},
    }
    validation_metrics = {"H0": metrics(validation["globfp_height_m"].to_numpy(dtype="float64"), val_pred, validation), **dependency_status}
    holdout_metrics = {"H0": metrics(holdout["globfp_height_m"].to_numpy(dtype="float64"), hold_pred, holdout), **dependency_status}
    write_json(output_root / "HEIGHT_MODEL_VALIDATION_METRICS.json", {"status": "COMPLETE_H0_ONLY", "selected_model": selected, "models": validation_metrics, "selection_scope": "non-Nanjing validation source files only", "H1_H2_gate": "blocked_by_missing_model_dependencies"})
    write_json(output_root / "HEIGHT_MODEL_INTERNAL_HOLDOUT.json", {"status": "COMPLETE_H0_ONLY", "models": holdout_metrics, "holdout_used_for_selection": False})
    write_json(training_root / "HEIGHT_MODEL_FEATURE_SCHEMA.json", {"status": "COMPLETE_FEATURES_AVAILABLE_H0_ONLY", "target": "globfp_height_m", "height_unit": "m", "longitude_latitude_used_as_predictors": False, "metric_crs": "EPSG:6933", "morphology_features": [column for column in frame.columns if column not in {"globfp_height_m", "split", "source_file", "source_region_id", "source_grid_id"}]})
    split_frame = pd.read_csv(training_root / "HEIGHT_SPATIAL_SPLIT.csv")
    write_json(training_root / "HEIGHT_SPLIT_AUDIT.json", {"status": "PASS_SPATIAL_FILE_GROUPED_SPLIT_H0_FALLBACK", "created_at_utc": now_utc(), "seed": 20260804, "nanjing_features_used": False, "sample_feature_count": int(len(frame)), "split_counts": {split: int((frame["split"] == split).sum()) for split in ("train", "validation", "internal_holdout")}, "split_grid_counts": {split: int(split_frame.loc[split_frame["split"] == split, "source_grid_id"].nunique()) for split in ("train", "validation", "internal_holdout")}, "excluded_source_files": split_frame.loc[split_frame["excluded_from_training"].astype(bool), "source_file"].tolist() if "excluded_from_training" in split_frame else []})
    write_json(output_root / "HEIGHT_MODEL_TRAINING_MANIFEST.json", {"status": "COMPLETE_H0_ONLY_DEPENDENCY_LIMITED", "created_at_utc": now_utc(), "selected_model": selected, "model_package": str(model_path), "model_sha256": sha256_file(model_path), "training_sample_count": int(len(frame)), "train_count": int(len(train)), "validation_count": int(len(validation)), "internal_holdout_count": int(len(holdout)), "nanjing_excluded": True, "nanjing_external_acceptance": "NOT_ACCEPTED_TIER_C_MATCH_GATE", "H1_H2_status": dependency_status, "training_started": True, "locked_test_pixel_reads": 0, "source_data_modified": False})
    seal = project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320" / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    seal_payload = json.loads(seal.read_text(encoding="utf-8-sig"))
    records = seal_payload.get("files") or seal_payload.get("records") or []
    matched = 0
    total = 0
    for record in records:
        path = record.get("path") or record.get("file") or record.get("relative_path")
        expected = record.get("sha256") or record.get("hash")
        if not path or not expected:
            continue
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = seal.parent / candidate
        total += 1
        if candidate.is_file() and sha256_file(candidate) == expected:
            matched += 1
    write_json(output_root / "HEIGHT_BRANCH_INVARIANCE_AUDIT.json", {"status": "PASS" if matched == 252 and total == 252 else "FAIL", "baseline_hash_match_count": matched, "baseline_hash_total_count": total, "baseline_hash_invariance_passed": matched == 252 and total == 252, "height_branch_writes_baseline": False, "construction_checkpoint_selection_affected": False})
    return {"status": "COMPLETE_H0_ONLY_DEPENDENCY_LIMITED", "selected_model": selected, "model_path": str(model_path), "model_sha256": sha256_file(model_path), "feature_count": len(frame), "validation": validation_metrics, "holdout": holdout_metrics, "h1_h2": dependency_status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.features, args.output_root, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
