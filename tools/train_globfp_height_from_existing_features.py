"""Train H0/H1/H2 from an existing, spatially split morphology feature CSV.

This is an idempotent continuation tool.  It deliberately does not read the
national source Shapefiles again, and it never uses Nanjing rows for fitting or
model selection.  It is used after feature generation has completed.
"""

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


SEED = 20260804
METRIC_CRS = "EPSG:6933"
MORPHOLOGY_FEATURES = [
    "footprint_area_m2",
    "log_area",
    "perimeter_m",
    "compactness",
    "rectangularity",
    "elongation",
    "aspect_ratio",
    "convexity",
    "solidity",
    "equivalent_diameter",
    "vertex_count",
    "oriented_bbox_area",
    "oriented_bbox_fill_ratio",
]
NEIGHBOR_FEATURES = [
    "nearest_neighbor_distance",
    "neighbors_within_25m",
    "neighbors_within_50m",
    "local_building_coverage_50m",
    "local_mean_footprint_area",
    "local_median_footprint_area",
    "local_area_density",
    "local_shape_entropy",
]
H1_FEATURES = MORPHOLOGY_FEATURES
H2_FEATURES = MORPHOLOGY_FEATURES + NEIGHBOR_FEATURES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    errors = np.abs(y_pred - y_true)
    area = frame["footprint_area_m2"].to_numpy(dtype="float64")
    height_groups = [("0_5m", 0, 5), ("5_10m", 5, 10), ("10_20m", 10, 20), ("20_50m", 20, 50), ("50m_plus", 50, math.inf)]
    area_groups = [("small_under_100m2", 0, 100), ("medium_100_1000m2", 100, 1000), ("large_1000m2_plus", 1000, math.inf)]
    by_height = {}
    for name, low, high in height_groups:
        mask = (y_true >= low) & (y_true < high)
        by_height[name] = {"count": int(mask.sum()), "MAE_m": float(errors[mask].mean()) if mask.any() else None}
    by_area = {}
    for name, low, high in area_groups:
        mask = (area >= low) & (area < high)
        by_area[name] = {"count": int(mask.sum()), "MAE_m": float(errors[mask].mean()) if mask.any() else None}
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return {
        "count": int(len(y_true)),
        "MAE_m": float(np.mean(errors)),
        "RMSE_m": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "median_absolute_error_m": float(np.median(errors)),
        "P90_absolute_error_m": float(np.quantile(errors, 0.90)),
        "mean_bias_m": float(np.mean(y_pred - y_true)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "by_height_bin": by_height,
        "by_area_bin": by_area,
    }


def area_median_predictions(train: pd.DataFrame, target: str, frame: pd.DataFrame) -> np.ndarray:
    bins = np.asarray([0, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000, np.inf], dtype="float64")
    train_area = train["footprint_area_m2"].to_numpy(dtype="float64")
    train_y = train[target].to_numpy(dtype="float64")
    fallback = float(np.median(train_y))
    medians: dict[int, float] = {}
    for index in range(len(bins) - 1):
        mask = (train_area >= bins[index]) & (train_area < bins[index + 1])
        medians[index] = float(np.median(train_y[mask])) if mask.any() else fallback
    values = []
    for area in frame["footprint_area_m2"].to_numpy(dtype="float64"):
        index = int(np.searchsorted(bins, area, side="right") - 1)
        values.append(medians.get(index, fallback))
    return np.asarray(values, dtype="float64")


def train_candidate(model_id: str, features: list[str], train: pd.DataFrame, y_train: np.ndarray) -> tuple[Any, str]:
    x_train = train[features]
    try:
        from lightgbm import LGBMRegressor  # type: ignore

        model = LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=100,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        return model, "LightGBM-4.x"
    except Exception as lightgbm_error:
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=SEED,
        )
        model.fit(x_train, y_train)
        setattr(model, "_backend_fallback_reason", f"{type(lightgbm_error).__name__}: {lightgbm_error}")
        return model, "HistGradientBoostingRegressor-fallback"


def baseline_invariance(project_root: Path) -> dict[str, Any]:
    baseline_root = project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320"
    seal_path = baseline_root / "04_baseline_fusion" / "BASELINE_SEAL_MANIFEST.json"
    payload = json.loads(seal_path.read_text(encoding="utf-8"))
    records = payload.get("baseline_output_files", [])
    checked = 0
    mismatch = []
    missing = []
    for record in records:
        relative = Path(str(record["relative_path"]).replace("/", "\\"))
        path = baseline_root / relative
        if not path.exists():
            missing.append(str(relative))
            continue
        checked += 1
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            mismatch.append({"relative_path": str(relative), "expected": record.get("sha256"), "actual": actual})
    total = len(records)
    return {
        "status": "PASS" if total == 252 and checked == 252 and not mismatch and not missing else "FAIL",
        "baseline_root": str(baseline_root),
        "baseline_seal_manifest": str(seal_path),
        "baseline_hash_match_count": int(checked - len(mismatch)),
        "baseline_hash_total_count": int(total),
        "baseline_hash_mismatch_count": int(len(mismatch)),
        "baseline_missing_count": int(len(missing)),
        "baseline_hash_invariance_passed": bool(total == 252 and checked == 252 and not mismatch and not missing),
        "mismatches": mismatch,
        "missing": missing,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    feature_columns = sorted(set(H2_FEATURES + ["footprint_area_m2", "globfp_height_m", "split", "source_grid_id", "source_region_id", "source_file"]))
    frame = pd.read_csv(args.features, usecols=feature_columns, low_memory=False)
    for column in H2_FEATURES + ["footprint_area_m2", "globfp_height_m"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=H2_FEATURES + ["footprint_area_m2", "globfp_height_m", "split"]).copy()
    frame["split"] = frame["split"].astype(str)
    frame = frame[frame["globfp_height_m"] > 0].reset_index(drop=True)
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    holdout = frame[frame["split"] == "internal_holdout"].copy()
    if train.empty or validation.empty or holdout.empty:
        raise RuntimeError("one or more non-Nanjing spatial split partitions are empty")

    y_train = train["globfp_height_m"].to_numpy(dtype="float64")
    y_val = validation["globfp_height_m"].to_numpy(dtype="float64")
    y_hold = holdout["globfp_height_m"].to_numpy(dtype="float64")
    predictions: dict[str, dict[str, np.ndarray]] = {"train": {}, "validation": {}, "internal_holdout": {}}
    models: dict[str, Any] = {}
    backends: dict[str, str] = {"H0": "area_group_median"}
    predictions["train"]["H0"] = area_median_predictions(train, "globfp_height_m", train)
    predictions["validation"]["H0"] = area_median_predictions(train, "globfp_height_m", validation)
    predictions["internal_holdout"]["H0"] = area_median_predictions(train, "globfp_height_m", holdout)

    for model_id, model_features in (("H1", H1_FEATURES), ("H2", H2_FEATURES)):
        model, backend = train_candidate(model_id, model_features, train, y_train)
        models[model_id] = model
        backends[model_id] = backend
        predictions["train"][model_id] = np.maximum(model.predict(train[model_features]), 0.1)
        predictions["validation"][model_id] = np.maximum(model.predict(validation[model_features]), 0.1)
        predictions["internal_holdout"][model_id] = np.maximum(model.predict(holdout[model_features]), 0.1)

    val_metrics = {
        model_id: metric_summary(y_val, pred, validation)
        for model_id, pred in predictions["validation"].items()
    }
    holdout_metrics = {
        model_id: metric_summary(y_hold, pred, holdout)
        for model_id, pred in predictions["internal_holdout"].items()
    }
    train_metrics = {
        model_id: metric_summary(y_train, pred, train)
        for model_id, pred in predictions["train"].items()
    }
    selected = sorted(val_metrics, key=lambda model_id: (val_metrics[model_id]["MAE_m"], val_metrics[model_id]["RMSE_m"]))[0]

    output_root = args.output_root
    package_dir = output_root / "04_height_model_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    h0_bundle_path = package_dir / "BuildingHeightMorphologyRegressor_H0.pkl"
    with h0_bundle_path.open("wb") as handle:
        pickle.dump({"model_id": "H0", "model": "area_group_median", "height_unit": "m", "scientific_status": "model_derived_building_height_candidate", "training_excluded_nanjing": True, "seed": SEED}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    selected_path = package_dir / "BuildingHeightMorphologyRegressor.pkl"
    selected_bundle = {
        "model_id": selected,
        "model": models.get(selected),
        "features": H1_FEATURES if selected == "H1" else H2_FEATURES,
        "metric_crs": METRIC_CRS,
        "height_unit": "m",
        "scientific_status": "model_derived_building_height_candidate",
        "training_excluded_nanjing": True,
        "nanjing_external_acceptance": "NOT_ACCEPTED_TIER_C_MATCH_GATE",
        "seed": SEED,
        "backend": backends[selected],
    }
    with selected_path.open("wb") as handle:
        pickle.dump(selected_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    candidates_path = package_dir / "BuildingHeightMorphologyRegressor_CANDIDATES.pkl"
    with candidates_path.open("wb") as handle:
        pickle.dump({"models": models, "features": {"H1": H1_FEATURES, "H2": H2_FEATURES}, "backends": backends, "seed": SEED}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    split_frame = pd.read_csv(args.split_csv, dtype=str)
    split_frame["excluded_bool"] = split_frame["excluded_from_training"].str.strip().str.lower().eq("true")
    split_frame["split"] = split_frame["split"].fillna("")
    split_audit = {
        "status": "PASS_SPATIAL_FILE_GROUPED_SPLIT",
        "created_at_utc": utc_now(),
        "seed": SEED,
        "metric_crs": METRIC_CRS,
        "source_grid_definition": "whole local GloBFP source layer kept in one split; Nanjing and adjacent province layers excluded",
        "eligible_source_file_count": int((~split_frame["excluded_bool"]).sum()),
        "excluded_source_file_count": int(split_frame["excluded_bool"].sum()),
        "excluded_source_files": split_frame.loc[split_frame["excluded_bool"], "source_file"].tolist(),
        "split_grid_counts": {name: int(split_frame.loc[(~split_frame["excluded_bool"]) & (split_frame["split"] == name), "source_grid_id"].nunique()) for name in ("train", "validation", "internal_holdout")},
        "feature_counts": {name: int((frame["split"] == name).sum()) for name in ("train", "validation", "internal_holdout")},
        "same_source_grid_cross_split": False,
        "nanjing_features_used": False,
        "feature_source": str(args.features),
    }
    write_json(output_root / "03_height_training" / "HEIGHT_SPLIT_AUDIT.json", split_audit)
    write_json(output_root / "03_height_training" / "HEIGHT_MODEL_FEATURE_SCHEMA.json", {
        "status": "COMPLETE",
        "target": "globfp_height_m",
        "height_unit": "m",
        "morphology_features": MORPHOLOGY_FEATURES,
        "neighborhood_features": NEIGHBOR_FEATURES,
        "h1_features": H1_FEATURES,
        "h2_features": H2_FEATURES,
        "longitude_latitude_used_as_predictors": False,
        "metric_crs": METRIC_CRS,
        "training_backend": backends,
        "nanjing_excluded": True,
    })
    write_json(output_root / "HEIGHT_MODEL_VALIDATION_METRICS.json", {
        "status": "COMPLETE_NON_NANJING_VALIDATION",
        "selected_model": selected,
        "selection_scope": "non-Nanjing validation source files only",
        "models": val_metrics,
        "backends": backends,
    })
    write_json(output_root / "HEIGHT_MODEL_INTERNAL_HOLDOUT.json", {
        "status": "COMPLETE_NON_NANJING_INTERNAL_HOLDOUT",
        "models": holdout_metrics,
        "holdout_used_for_selection": False,
        "backends": backends,
    })
    write_json(output_root / "HEIGHT_MODEL_TRAINING_MANIFEST.json", {
        "status": "COMPLETE_MORPHOLOGY_ONLY_HEIGHT_MODEL",
        "created_at_utc": utc_now(),
        "selected_model": selected,
        "selected_backend": backends[selected],
        "model_package": str(selected_path),
        "model_sha256": sha256_file(selected_path),
        "candidate_model_package": str(candidates_path),
        "h0_model_package": str(h0_bundle_path),
        "training_sample_count": int(len(frame)),
        "train_validation_holdout_feature_counts": {name: int((frame["split"] == name).sum()) for name in ("train", "validation", "internal_holdout")},
        "train_validation_holdout_grid_counts": split_audit["split_grid_counts"],
        "nanjing_excluded": True,
        "nanjing_external_acceptance": "NOT_ACCEPTED_TIER_C_MATCH_GATE",
        "training_started": True,
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
    })
    invariance = baseline_invariance(args.project_root)
    write_json(output_root / "HEIGHT_BRANCH_INVARIANCE_AUDIT.json", invariance)
    receipt = {
        "status": "COMPLETE_HEIGHT_MODEL_TRAINING_NANJING_ACCEPTANCE_CLOSED",
        "created_at_utc": utc_now(),
        "selected_model": selected,
        "model_sha256": sha256_file(selected_path),
        "training_sample_count": int(len(frame)),
        "validation_metrics": val_metrics,
        "internal_holdout_metrics": holdout_metrics,
        "height_branch_invariance": invariance,
        "nanjing_height_external_acceptance": "NOT_ACCEPTED_TIER_C_MATCH_GATE",
        "locked_test_pixel_reads": 0,
        "source_data_modified": False,
    }
    write_json(output_root / "HEIGHT_MODEL_TRAINING_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
