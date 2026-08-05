from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid
import geopandas as gpd
from pyproj import Transformer

from ..core.hashing import sha256_file
from ..core.receipts import read_json, write_json, utc_now


REQUIRED = ("building_probability.tif", "building_mask.tif", "building_boundary.tif", "GEOMETRY_MANIFEST.json")


def _scene_dirs(root: Path) -> list[Path]:
    return [root / date for date in ("2018-05-06", "2020-07-23", "2022-03-20", "2023-07-16")]


def _source_meta(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": scene.get("width"),
        "height": scene.get("height"),
        "crs": scene.get("crs"),
        "transform": scene.get("transform"),
        "nodata": scene.get("nodata"),
        "resolution": scene.get("resolution"),
        "bounds": scene.get("bounds"),
    }


def audit_geometry(root: str | Path, *, checkpoint_sha256: str | None = None) -> dict[str, Any]:
    root = Path(root)
    scenes_path = root / "TONGZHOU_SCENES.json"
    scenes_payload = read_json(scenes_path) if scenes_path.exists() else {}
    scenes_by_date = {item.get("scene_date"): item for item in scenes_payload.get("scenes", [])}
    rows: list[dict[str, Any]] = []
    all_pass = True
    for scene_dir in _scene_dirs(root):
        date = scene_dir.name
        manifest_path = scene_dir / "GEOMETRY_MANIFEST.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        row: dict[str, Any] = {"scene_date": date, "status": "pass", "required_files": {name: (scene_dir / name).exists() for name in REQUIRED}}
        row["manifest_checkpoint_sha256"] = manifest.get("checkpoint_sha256")
        row["checkpoint_sha256_match"] = checkpoint_sha256 is None or manifest.get("checkpoint_sha256") == checkpoint_sha256
        try:
            with rasterio.open(scene_dir / "building_probability.tif") as prob, rasterio.open(scene_dir / "building_mask.tif") as mask, rasterio.open(scene_dir / "building_boundary.tif") as boundary:
                probability = prob.read(1, out_dtype="float32", masked=True)
                mask_values = np.unique(mask.read(1))
                boundary_values = np.unique(boundary.read(1))
                probability_array = np.asarray(probability.filled(np.nan), dtype=np.float64)
                source = scenes_by_date.get(date, {})
                row.update({
                    "probability_finite": bool(np.isfinite(probability_array).all()),
                    "probability_in_range": bool(np.nanmin(probability_array) >= 0 and np.nanmax(probability_array) <= 1),
                    "probability_min": float(np.nanmin(probability_array)),
                    "probability_max": float(np.nanmax(probability_array)),
                    "positive_fraction": float((mask.read(1) == 1).mean()),
                    "mask_values": [int(x) for x in mask_values],
                    "boundary_values": [int(x) for x in boundary_values],
                    "shape_match": prob.shape == mask.shape == boundary.shape == (source.get("height"), source.get("width")),
                    "crs_match": str(prob.crs) == str(source.get("crs")),
                    "transform_match": list(prob.transform) == list(source.get("transform", [])),
                    "nodata_recorded": manifest.get("nodata"),
                    "tiles_complete": manifest.get("completed_tiles") == manifest.get("expected_tiles"),
                })
                row["status"] = "pass" if all([
                    all(row["required_files"].values()), row["probability_finite"], row["probability_in_range"],
                    row["positive_fraction"] > 0, row["positive_fraction"] < 0.90,
                    set(row["mask_values"]).issubset({0, 1, 255}), set(row["boundary_values"]).issubset({0, 1, 255}),
                    row["shape_match"], row["crs_match"], row["transform_match"], row["tiles_complete"], row["checkpoint_sha256_match"],
                ]) else "fail"
        except Exception as exc:
            row["status"] = "fail"
            row["error"] = f"{type(exc).__name__}: {exc}"
        all_pass = all_pass and row["status"] == "pass"
        rows.append(row)
    result = {
        "geometry_numerical_status": "pass" if all_pass and len(rows) == 4 else "fail",
        "scenes": rows,
        "test_reads": {"WHU": 0, "SECOND": 0},
        "audited_at": utc_now(),
    }
    write_json(root / "TONGZHOU_URRS_GEOMETRY_AUDIT.json", result)
    return result


def vectorize_geometry(root: str | Path, *, min_area_m2: float = 10.0) -> dict[str, Any]:
    root = Path(root)
    audit = read_json(root / "TONGZHOU_URRS_GEOMETRY_AUDIT.json") if (root / "TONGZHOU_URRS_GEOMETRY_AUDIT.json").exists() else {}
    if audit.get("geometry_numerical_status") != "pass":
        raise RuntimeError("Geometry numerical audit must pass before vectorization")
    rows: list[dict[str, Any]] = []
    for scene_dir in _scene_dirs(root):
        date = scene_dir.name
        output = scene_dir / "building_objects.gpkg"
        with rasterio.open(scene_dir / "building_mask.tif") as src:
            data = src.read(1)
            geoms = []
            for geom_json, value in shapes(data, mask=data == 1, transform=src.transform, connectivity=8):
                if int(value) != 1:
                    continue
                geom = make_valid(shape(geom_json))
                if geom.is_empty:
                    continue
                geoms.append(geom)
            if not geoms:
                skip = {"status": "skipped_empty_mask", "scene_date": date, "source_model": "rsbuilding_swin_t_whu_iter2000", "scientific_status": "candidate_not_ground_truth"}
                write_json(scene_dir / "VECTORIZATION_SKIPPED.json", skip)
                rows.append({"scene_date": date, "status": "skipped_empty_mask", "objects": 0, "gpkg": None})
                continue
            crs = src.crs
            metric_crs = "EPSG:32650" if crs and crs.to_epsg() == 4326 else crs
            transformer = Transformer.from_crs(crs, metric_crs, always_xy=True) if crs and metric_crs else None
            records = []
            for idx, geom in enumerate(geoms, 1):
                metric_geom = shapely_transform(transformer.transform, geom) if transformer else geom
                area = float(metric_geom.area) if transformer else None
                if area is not None and area < min_area_m2:
                    continue
                records.append({"object_id": f"{date.replace('-', '')}_b{idx:06d}", "scene_date": date, "state": "building_candidate", "area_m2": area, "perimeter_m": float(metric_geom.length) if transformer else None, "source_model": "rsbuilding_swin_t_whu_iter2000", "scientific_status": "candidate_not_ground_truth", "geometry": geom})
            if not records:
                skip = {"status": "skipped_below_min_area", "scene_date": date, "min_area_m2": min_area_m2, "source_model": "rsbuilding_swin_t_whu_iter2000", "scientific_status": "candidate_not_ground_truth"}
                write_json(scene_dir / "VECTORIZATION_SKIPPED.json", skip)
                rows.append({"scene_date": date, "status": "skipped_below_min_area", "objects": 0, "gpkg": None})
                continue
            frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
            frame.to_file(output, layer="buildings", driver="GPKG", engine="pyogrio")
            rows.append({"scene_date": date, "status": "pass", "objects": int(len(frame)), "gpkg": str(output), "area_m2_total": float(frame["area_m2"].sum())})
    result = {"vectorization_status": "pass", "scenes": rows, "min_area_m2": min_area_m2, "test_reads": {"WHU": 0, "SECOND": 0}, "completed_at": utc_now()}
    write_json(root / "TONGZHOU_URRS_VECTORIZATION.json", result)
    return result


def audit_alignment(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    scenes = []
    for scene_dir in _scene_dirs(root):
        with rasterio.open(scene_dir / "building_probability.tif") as ds:
            scenes.append({"scene_date": scene_dir.name, "width": ds.width, "height": ds.height, "crs": str(ds.crs), "transform": list(ds.transform), "resolution": list(ds.res), "bounds": list(ds.bounds), "nodata": ds.nodata})
    baseline = scenes[0] if scenes else {}
    mismatches = []
    for item in scenes[1:]:
        for key in ("width", "height", "crs", "transform", "resolution", "bounds", "nodata"):
            left, right = item.get(key), baseline.get(key)
            both_nan = isinstance(left, float) and isinstance(right, float) and np.isnan(left) and np.isnan(right)
            if left != right and not both_nan:
                mismatches.append(f"{item['scene_date']}.{key}")
    result = {"alignment_status": "ready_for_pixelwise_temporal_tracking" if scenes and not mismatches else "registration_required_before_pixelwise_fusion", "scenes": scenes, "mismatches": mismatches, "checked_at": utc_now(), "test_reads": {"WHU": 0, "SECOND": 0}}
    write_json(root / "TONGZHOU_URRS_ALIGNMENT_AUDIT.json", result)
    return result


def geometry_status(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    result = {"root": str(root), "scene_count": len([p for p in _scene_dirs(root) if p.exists()]), "geometry_rasters_complete": all((p / name).exists() for p in _scene_dirs(root) for name in REQUIRED), "numerical_audit": str(root / "TONGZHOU_URRS_GEOMETRY_AUDIT.json") if (root / "TONGZHOU_URRS_GEOMETRY_AUDIT.json").exists() else None, "vectorization": str(root / "TONGZHOU_URRS_VECTORIZATION.json") if (root / "TONGZHOU_URRS_VECTORIZATION.json").exists() else None, "alignment": str(root / "TONGZHOU_URRS_ALIGNMENT_AUDIT.json") if (root / "TONGZHOU_URRS_ALIGNMENT_AUDIT.json").exists() else None, "test_reads": {"WHU": 0, "SECOND": 0}, "checked_at": utc_now()}
    return result
