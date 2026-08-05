"""Build a morphology-only GloBFP height model with conservative spatial splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
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
MODEL_FEATURES_H1 = MORPHOLOGY_FEATURES
MODEL_FEATURES_H2 = MORPHOLOGY_FEATURES + NEIGHBOR_FEATURES


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


def write_csv(path: Path, frame_or_rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frame_or_rows, pd.DataFrame):
        frame_or_rows.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        rows = list(frame_or_rows)
        fields = sorted({key for row in rows for key in row}) if rows else ["status"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_aoi_envelope(data_root: Path, buffer_m: float) -> dict[str, Any]:
    from pyproj import Transformer  # type: ignore
    from shapely.geometry import box  # type: ignore

    manifest = json.loads((data_root / "01_nanjing_extract" / "EXTRACTION_MANIFEST.json").read_text(encoding="utf-8"))
    bounds = manifest["audit"]["bounds"]["sealed_common_grid_bounds_epsg4326"]
    to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    to_source = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True)
    minx, miny = to_metric.transform(bounds[0], bounds[1])
    maxx, maxy = to_metric.transform(bounds[2], bounds[3])
    envelope_metric = box(minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)
    envelope_source = envelope_metric.__class__(
        [to_source.transform(x, y) for x, y in envelope_metric.exterior.coords]
    )
    return {
        "aoi_bounds_epsg4326": bounds,
        "exclusion_buffer_m": buffer_m,
        "exclusion_envelope_epsg4326": envelope_source.bounds,
        "exclusion_envelope_metric_crs": METRIC_CRS,
        "exclusion_reason": "sealed Nanjing AOI plus conservative neighboring-grid buffer; province files whose bounds intersect are excluded",
    }


def assign_split(grid_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"{SEED}:{grid_id}".encode()).digest()[:4], "big") % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "internal_holdout"


def discover_sources(source_root: Path, exclusion_bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    import pyogrio  # type: ignore
    from shapely.geometry import box  # type: ignore

    exclusion = box(*exclusion_bounds)
    rows = []
    for path in sorted(source_root.rglob("*.shp")):
        info = pyogrio.read_info(path)
        bounds = tuple(float(value) for value in info.get("total_bounds"))
        file_box = box(*bounds)
        grid_id = path.stem
        excluded = bool(file_box.intersects(exclusion))
        rows.append(
            {
                "source_grid_id": grid_id,
                "source_region_id": path.stem,
                "source_file": str(path),
                "source_directory": path.parent.name,
                "feature_count": int(info.get("features", 0)),
                "crs": str(info.get("crs")) if info.get("crs") else None,
                "geometry_type": str(info.get("geometry_type")),
                "minx": bounds[0],
                "miny": bounds[1],
                "maxx": bounds[2],
                "maxy": bounds[3],
                "excluded_from_training": excluded,
                "exclusion_reason": "Nanjing/adjacent conservative envelope intersects file bounds" if excluded else None,
                "split": "excluded" if excluded else assign_split(grid_id),
            }
        )
    return pd.DataFrame(rows)


def read_uniform_sample(path: Path, feature_count: int, target_count: int) -> Any:
    import pyogrio  # type: ignore

    if target_count >= feature_count:
        return pyogrio.read_dataframe(path, use_arrow=False)
    offsets = np.linspace(0, feature_count - 1, target_count, dtype=np.int64)
    # Read a small number of contiguous deterministic windows to avoid loading
    # an entire provincial layer while retaining spatially dispersed samples.
    windows = np.array_split(offsets, min(20, max(1, target_count // 2000)))
    frames = []
    for window in windows:
        if len(window) == 0:
            continue
        start = int(window[0])
        stop = int(window[-1]) + 1
        frame = pyogrio.read_dataframe(path, skip_features=start, max_features=stop - start, use_arrow=False)
        if len(frame) > len(window):
            positions = np.linspace(0, len(frame) - 1, len(window), dtype=np.int64)
            frame = frame.iloc[positions].copy()
        frames.append(frame)
    if not frames:
        return pyogrio.read_dataframe(path, max_features=0, use_arrow=False)
    return pd.concat(frames, ignore_index=True)


def vertex_count(geometry: Any) -> int:
    if geometry is None or geometry.is_empty:
        return 0
    if geometry.geom_type == "Polygon":
        return sum(len(ring.coords) for ring in [geometry.exterior, *geometry.interiors])
    if geometry.geom_type == "MultiPolygon":
        return sum(vertex_count(part) for part in geometry.geoms)
    return len(getattr(geometry, "coords", []))


def morphology_frame(frame: Any, source_grid_id: str, source_region_id: str) -> pd.DataFrame:
    from scipy.spatial import cKDTree  # type: ignore

    metric = frame.to_crs(METRIC_CRS)
    metric = metric[metric.geometry.notna() & ~metric.geometry.is_empty].copy()
    try:
        invalid = ~metric.geometry.is_valid
        if invalid.any():
            metric.loc[invalid, "geometry"] = metric.loc[invalid, "geometry"].make_valid()
    except AttributeError:
        invalid = ~metric.geometry.is_valid
        if invalid.any():
            metric.loc[invalid, "geometry"] = metric.loc[invalid, "geometry"].buffer(0)
    metric = metric[metric.geometry.notna() & ~metric.geometry.is_empty & metric.geometry.is_valid].copy()
    if metric.empty:
        return pd.DataFrame()
    geometries = list(metric.geometry)
    areas = np.asarray([max(float(geom.area), 0.0) for geom in geometries], dtype="float64")
    perimeters = np.asarray([max(float(geom.length), 0.0) for geom in geometries], dtype="float64")
    oriented_areas = []
    rectangularity = []
    elongation = []
    aspect_ratio = []
    convexity = []
    solidity = []
    equivalent_diameter = []
    vertices = []
    for geom, area in zip(geometries, areas):
        oriented = geom.minimum_rotated_rectangle
        oriented_area = float(oriented.area)
        oriented_areas.append(oriented_area)
        coords = list(oriented.exterior.coords) if not oriented.is_empty else []
        lengths = [float(coords[i].distance(coords[i + 1])) if False else float(((coords[i][0] - coords[i + 1][0]) ** 2 + (coords[i][1] - coords[i + 1][1]) ** 2) ** 0.5) for i in range(len(coords) - 1)]
        lengths = sorted([value for value in lengths if value > 0], reverse=True)
        major = lengths[0] if lengths else 0.0
        minor = lengths[1] if len(lengths) > 1 else major
        elongation.append(major / minor if minor else 1.0)
        aspect_ratio.append(major / minor if minor else 1.0)
        rectangularity.append(area / oriented_area if oriented_area else 0.0)
        hull_area = float(geom.convex_hull.area)
        convexity.append(area / hull_area if hull_area else 0.0)
        solidity.append(area / hull_area if hull_area else 0.0)
        equivalent_diameter.append(math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0)
        vertices.append(vertex_count(geom))
    compactness = np.divide(4.0 * math.pi * areas, perimeters**2, out=np.zeros_like(areas), where=perimeters > 0)
    centroids = np.asarray([[geom.centroid.x, geom.centroid.y] for geom in geometries], dtype="float64")
    tree = cKDTree(centroids)
    nearest = tree.query(centroids, k=min(2, len(centroids)))[0]
    nearest_distance = nearest[:, 1] if len(centroids) > 1 else np.zeros(len(centroids))
    compactness_array = compactness
    neighbors_25 = []
    neighbors_50 = []
    coverage_50 = []
    mean_area = []
    median_area = []
    density_50 = []
    entropy = []
    bins = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 10.0])
    for index, point in enumerate(centroids):
        ids25 = [candidate for candidate in tree.query_ball_point(point, 25.0) if candidate != index]
        ids50 = [candidate for candidate in tree.query_ball_point(point, 50.0) if candidate != index]
        local_areas = areas[ids50] if ids50 else np.asarray([], dtype="float64")
        neighbors_25.append(len(ids25))
        neighbors_50.append(len(ids50))
        coverage_50.append(float((areas[index] + local_areas.sum()) / (math.pi * 50.0**2)))
        mean_area.append(float(local_areas.mean()) if local_areas.size else float(areas[index]))
        median_area.append(float(np.median(local_areas)) if local_areas.size else float(areas[index]))
        density_50.append(float((len(ids50) + 1) / (math.pi * 50.0**2)))
        if ids50:
            counts, _ = np.histogram(compactness_array[ids50], bins=bins)
            probabilities = counts[counts > 0] / max(counts.sum(), 1)
            entropy.append(float(-(probabilities * np.log(probabilities)).sum() / math.log(len(bins))))
        else:
            entropy.append(0.0)
    output = pd.DataFrame(
        {
            "footprint_area_m2": areas,
            "log_area": np.log1p(areas),
            "perimeter_m": perimeters,
            "compactness": compactness,
            "rectangularity": rectangularity,
            "elongation": elongation,
            "aspect_ratio": aspect_ratio,
            "convexity": convexity,
            "solidity": solidity,
            "equivalent_diameter": equivalent_diameter,
            "vertex_count": vertices,
            "oriented_bbox_area": oriented_areas,
            "oriented_bbox_fill_ratio": rectangularity,
            "nearest_neighbor_distance": nearest_distance,
            "neighbors_within_25m": neighbors_25,
            "neighbors_within_50m": neighbors_50,
            "local_building_coverage_50m": coverage_50,
            "local_mean_footprint_area": mean_area,
            "local_median_footprint_area": median_area,
            "local_area_density": density_50,
            "local_shape_entropy": entropy,
            "source_region_id": source_region_id,
            "source_grid_id": source_grid_id,
            "globfp_height_m": metric["Height"].astype(float).to_numpy(),
            "geometry": geometries,
        }
    )
    return output.replace([np.inf, -np.inf], np.nan).dropna(subset=["globfp_height_m", "footprint_area_m2"])


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

    errors = np.abs(y_pred - y_true)
    height_groups = [("0_5m", 0, 5), ("5_10m", 5, 10), ("10_20m", 10, 20), ("20_50m", 20, 50), ("50m_plus", 50, math.inf)]
    height_metrics = {}
    for name, low, high in height_groups:
        mask = (y_true >= low) & (y_true < high)
        height_metrics[name] = {"count": int(mask.sum()), "MAE_m": float(errors[mask].mean()) if mask.any() else None}
    area_groups = [("small_under_100m2", 0, 100), ("medium_100_1000m2", 100, 1000), ("large_1000m2_plus", 1000, math.inf)]
    area_metrics = {}
    area = frame["footprint_area_m2"].to_numpy(dtype="float64")
    for name, low, high in area_groups:
        mask = (area >= low) & (area < high)
        area_metrics[name] = {"count": int(mask.sum()), "MAE_m": float(errors[mask].mean()) if mask.any() else None}
    return {
        "count": int(len(y_true)),
        "MAE_m": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_m": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "median_absolute_error_m": float(np.median(errors)),
        "P90_absolute_error_m": float(np.quantile(errors, 0.90)),
        "mean_bias_m": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "by_height_bin": height_metrics,
        "by_area_bin": area_metrics,
    }


def train_models(features: pd.DataFrame, split_frame: pd.DataFrame, output_root: Path, seed: int) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

    features = features.reset_index(drop=True)
    train = features[features["split"] == "train"].copy()
    validation = features[features["split"] == "validation"].copy()
    holdout = features[features["split"] == "internal_holdout"].copy()
    if train.empty or validation.empty or holdout.empty:
        raise RuntimeError("spatial split produced an empty training/validation/holdout feature set")
    y_train = train["globfp_height_m"].to_numpy(dtype="float64")
    y_val = validation["globfp_height_m"].to_numpy(dtype="float64")
    y_hold = holdout["globfp_height_m"].to_numpy(dtype="float64")
    train_predictions = {}
    val_predictions = {}
    hold_predictions = {}
    models: dict[str, Any] = {}

    area_bins = np.asarray([0, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000, np.inf], dtype="float64")
    medians = {}
    for index in range(len(area_bins) - 1):
        mask = (train["footprint_area_m2"].to_numpy() >= area_bins[index]) & (train["footprint_area_m2"].to_numpy() < area_bins[index + 1])
        medians[index] = float(np.median(y_train[mask])) if mask.any() else float(np.median(y_train))

    def h0_predict(frame: pd.DataFrame) -> np.ndarray:
        area = frame["footprint_area_m2"].to_numpy(dtype="float64")
        values = []
        for value in area:
            index = int(np.searchsorted(area_bins, value, side="right") - 1)
            values.append(medians.get(index, float(np.median(y_train))))
        return np.asarray(values, dtype="float64")

    train_predictions["H0"] = h0_predict(train)
    val_predictions["H0"] = h0_predict(validation)
    hold_predictions["H0"] = h0_predict(holdout)
    for model_id, model_features in (("H1", MODEL_FEATURES_H1), ("H2", MODEL_FEATURES_H2)):
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=seed,
        )
        estimator.fit(train[model_features], y_train)
        models[model_id] = estimator
        train_predictions[model_id] = np.maximum(estimator.predict(train[model_features]), 0.1)
        val_predictions[model_id] = np.maximum(estimator.predict(validation[model_features]), 0.1)
        hold_predictions[model_id] = np.maximum(estimator.predict(holdout[model_features]), 0.1)
    val_metrics = {model_id: metric_summary(y_val, predictions, validation) for model_id, predictions in val_predictions.items()}
    holdout_metrics = {model_id: metric_summary(y_hold, predictions, holdout) for model_id, predictions in hold_predictions.items()}
    train_metrics = {model_id: metric_summary(y_train, predictions, train) for model_id, predictions in train_predictions.items()}
    selected = sorted(val_metrics, key=lambda model_id: (val_metrics[model_id]["MAE_m"], val_metrics[model_id]["RMSE_m"]))[0]
    package_dir = output_root / "04_height_model_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = package_dir / "BuildingHeightMorphologyRegressor.pkl"
    bundle = {
        "model_id": selected,
        "model": models.get(selected),
        "features": MODEL_FEATURES_H1 if selected == "H1" else MODEL_FEATURES_H2,
        "metric_crs": METRIC_CRS,
        "height_unit": "m",
        "scientific_status": "model_derived_building_height_candidate",
        "training_excluded_nanjing": True,
        "seed": seed,
    }
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "selected_model": selected,
        "validation_metrics": val_metrics,
        "internal_holdout_metrics": holdout_metrics,
        "train_metrics": train_metrics,
        "models": models,
        "bundle_path": bundle_path,
        "bundle_sha256": sha256_file(bundle_path),
        "split_counts": {split: int((features["split"] == split).sum()) for split in ("train", "validation", "internal_holdout")},
        "grid_counts": {split: int(split_frame.loc[split_frame["split"] == split, "source_grid_id"].nunique()) for split in ("train", "validation", "internal_holdout")},
    }


def run_training(source_root: Path, data_root: Path, output_root: Path, max_samples: int, exclusion_buffer_m: float) -> dict[str, Any]:
    import geopandas as gpd  # type: ignore
    from shapely.geometry import box  # type: ignore

    output_root.mkdir(parents=True, exist_ok=True)
    training_root = output_root / "03_height_training"
    features_root = training_root / "features_by_grid"
    features_root.mkdir(parents=True, exist_ok=True)
    envelope = load_aoi_envelope(data_root, exclusion_buffer_m)
    sources = discover_sources(source_root, tuple(envelope["exclusion_envelope_epsg4326"]))
    write_csv(training_root / "HEIGHT_SPATIAL_SPLIT.csv", sources)
    split_map = gpd.GeoDataFrame(
        sources.copy(),
        geometry=[box(row.minx, row.miny, row.maxx, row.maxy) for row in sources.itertuples()],
        crs="EPSG:4326",
    )
    split_map.to_file(training_root / "HEIGHT_SPLIT_MAP.gpkg", layer="height_split_map", driver="GPKG", engine="pyogrio")
    eligible = sources[~sources["excluded_from_training"]].copy()
    total_eligible = int(eligible["feature_count"].sum())
    feature_frames = []
    per_file_cap = max(5_000, min(50_000, int(max_samples / max(len(eligible), 1) * 2)))
    for row in eligible.itertuples():
        target_count = min(per_file_cap, max(5_000, int(round(max_samples * row.feature_count / max(total_eligible, 1)))))
        target_count = min(target_count, row.feature_count)
        print(json.dumps({"stage": "height_feature_sampling", "source_grid_id": row.source_grid_id, "target_count": target_count, "feature_count": row.feature_count}, ensure_ascii=False), flush=True)
        raw = read_uniform_sample(Path(row.source_file), int(row.feature_count), int(target_count))
        if "Height" not in raw.columns:
            continue
        derived = morphology_frame(raw, row.source_grid_id, row.source_region_id)
        if derived.empty:
            continue
        derived["split"] = row.split
        derived["source_file"] = row.source_file
        write_csv(features_root / f"{row.source_grid_id}.csv", derived.drop(columns=["geometry"]))
        feature_frames.append(derived)
    if not feature_frames:
        raise RuntimeError("no national morphology features were generated")
    features = pd.concat(feature_frames, ignore_index=True)
    features = features[features["globfp_height_m"].notna() & np.isfinite(features["globfp_height_m"]) & (features["globfp_height_m"] > 0)].copy()
    combined_path = training_root / "MORPHOLOGY_FEATURE_SAMPLE.csv"
    write_csv(combined_path, features.drop(columns=["geometry"]))
    parquet_status = {"written": False, "reason": "pyarrow unavailable in project environment", "fallback": str(combined_path)}
    try:
        features.drop(columns=["geometry"]).to_parquet(training_root / "MORPHOLOGY_FEATURE_SAMPLE.parquet", index=False)
        parquet_status = {"written": True, "path": str(training_root / "MORPHOLOGY_FEATURE_SAMPLE.parquet")}
    except Exception as exc:
        parquet_status["error"] = f"{type(exc).__name__}: {exc}"
    training = train_models(features, sources, output_root, SEED)
    split_audit = {
        "status": "PASS_SPATIAL_FILE_GROUPED_SPLIT",
        "created_at_utc": utc_now(),
        "seed": SEED,
        "metric_crs": METRIC_CRS,
        "source_grid_definition": "province Shapefile layer because local GloBFP grid index was not exposed in schema; whole source files are kept in one split",
        "nanjing_exclusion": envelope,
        "excluded_source_files": sources.loc[sources["excluded_from_training"], "source_file"].tolist(),
        "eligible_source_file_count": int(len(eligible)),
        "sample_feature_count": int(len(features)),
        "split_counts": training["split_counts"],
        "split_grid_counts": training["grid_counts"],
        "same_source_grid_cross_split": False,
        "nanjing_features_used": False,
        "parquet": parquet_status,
    }
    write_json(training_root / "HEIGHT_SPLIT_AUDIT.json", split_audit)
    write_json(training_root / "HEIGHT_MODEL_FEATURE_SCHEMA.json", {"status": "COMPLETE", "target": "globfp_height_m", "height_unit": "m", "morphology_features": MORPHOLOGY_FEATURES, "neighborhood_features": NEIGHBOR_FEATURES, "h1_features": MODEL_FEATURES_H1, "h2_features": MODEL_FEATURES_H2, "longitude_latitude_used_as_predictors": False, "metric_crs": METRIC_CRS})
    write_json(output_root / "HEIGHT_MODEL_VALIDATION_METRICS.json", {"status": "COMPLETE_NON_NANJING_VALIDATION", "selected_model": training["selected_model"], "models": training["validation_metrics"], "selection_scope": "non-Nanjing validation source files only"})
    write_json(output_root / "HEIGHT_MODEL_INTERNAL_HOLDOUT.json", {"status": "COMPLETE_NON_NANJING_INTERNAL_HOLDOUT", "models": training["internal_holdout_metrics"], "holdout_used_for_selection": False})
    write_json(output_root / "HEIGHT_MODEL_TRAINING_MANIFEST.json", {"status": "COMPLETE_MORPHOLOGY_ONLY_HEIGHT_MODEL", "created_at_utc": utc_now(), "selected_model": training["selected_model"], "model_package": str(training["bundle_path"]), "model_sha256": training["bundle_sha256"], "training_sample_count": len(features), "train_validation_holdout_grid_counts": training["grid_counts"], "nanjing_excluded": True, "nanjing_external_acceptance": "NOT_ACCEPTED_TIER_C_MATCH_GATE", "training_started": True, "locked_test_pixel_reads": 0, "source_data_modified": False})
    return {"status": "COMPLETE_MORPHOLOGY_ONLY_HEIGHT_MODEL", "output_root": str(output_root), "selected_model": training["selected_model"], "model_sha256": training["bundle_sha256"], "training_sample_count": len(features), "split_audit": split_audit, "validation_metrics": training["validation_metrics"], "internal_holdout_metrics": training["internal_holdout_metrics"},


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Data Set\3D-GloBFP"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument("--exclusion-buffer-m", type=float, default=15_000.0)
    args = parser.parse_args()
    result = run_training(args.source_root, args.data_root, args.output_root, args.max_samples, args.exclusion_buffer_m)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
