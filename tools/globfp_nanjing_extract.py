"""Extract the Nanjing-area slice from the observed Jiangsu GloBFP layer.

The extraction bounds come from the sealed modular COMMON_GRID and a recorded
metric buffer.  The source Shapefile is never edited and the result is kept in
the timestamped data-preparation branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBLISHED_SOURCE = "https://essd.copernicus.org/articles/16/5357/2024/index.html"
HEIGHT_UNIT_PROVENANCE = {
    "height_field": "Height",
    "unit": "m",
    "status": "PUBLISHED_UNIT_PROVENANCE_CONFIRMED_LOCAL_SCHEMA_FIELD_READ",
    "source": PUBLISHED_SOURCE,
    "dataset_time_anchor": "2020",
    "interpretation": "GloBFP height is an estimated building-height attribute, not survey ground truth.",
}


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
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def derive_bounds(old_modular_root: Path, buffer_m: float) -> dict[str, Any]:
    from pyproj import Transformer  # type: ignore

    audit_path = old_modular_root / "01_input_audit" / "MODULAR_INPUT_AUDIT.json"
    audit = read_json(audit_path)
    common = audit["common_grid"]
    bounds4326 = common.get("intersection_bounds") or common.get("bounds")
    if not bounds4326 or len(bounds4326) != 4:
        raise RuntimeError("sealed COMMON_GRID has no four-value bounds")
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    to_source = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
    minx, miny = to_metric.transform(bounds4326[0], bounds4326[1])
    maxx, maxy = to_metric.transform(bounds4326[2], bounds4326[3])
    metric_bounds = [minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m]
    source_corners = [
        to_source.transform(metric_bounds[0], metric_bounds[1]),
        to_source.transform(metric_bounds[2], metric_bounds[3]),
    ]
    source_bounds = [source_corners[0][0], source_corners[0][1], source_corners[1][0], source_corners[1][1]]
    return {
        "sealed_common_grid_bounds_epsg4326": bounds4326,
        "sealed_common_grid_metric_crs": "EPSG:32650",
        "metric_bounds_epsg32650_with_buffer": metric_bounds,
        "source_query_bbox_epsg4326": source_bounds,
        "buffer_m": buffer_m,
        "bounds_source": str(audit_path),
        "valid_overlap_mask": audit.get("valid_overlap_mask"),
    }


def object_id_series(frame: Any, fields: list[str]) -> tuple[Any, str]:
    import pandas as pd  # type: ignore

    candidates = [field for field in ("id_all", "id", "ID", "fid", "FID", "object_id") if field in fields]
    if candidates:
        selected = candidates[0]
        values = frame[selected]
        return values.map(lambda value: None if pd.isna(value) else str(value)), selected
    return pd.Series([f"jiangsu_row_{index:09d}" for index in range(len(frame))], index=frame.index), "ROW_ORDINAL_FALLBACK"


def geometry_status(frame: Any) -> Any:
    import pandas as pd  # type: ignore

    status = pd.Series("valid", index=frame.index, dtype="string")
    status[frame.geometry.isna()] = "empty_geometry"
    status[frame.geometry.notna() & frame.geometry.is_empty.fillna(False)] = "empty_geometry"
    status[frame.geometry.notna() & ~frame.geometry.is_empty.fillna(False) & ~frame.geometry.is_valid.fillna(False)] = "invalid_geometry"
    return status


def run_extract(source_root: Path, old_modular_root: Path, output_root: Path, buffer_m: float) -> dict[str, Any]:
    import geopandas as gpd  # type: ignore
    import numpy as np  # type: ignore
    import pyogrio  # type: ignore
    import rasterio  # type: ignore
    from rasterio.features import rasterize  # type: ignore
    from rasterio.transform import from_origin  # type: ignore
    from shapely.geometry import box  # type: ignore
    from shapely.ops import transform as shapely_transform  # type: ignore
    from pyproj import Transformer  # type: ignore

    audit_root = output_root / "00_globfp_audit"
    extract_root = output_root / "01_nanjing_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    bounds = derive_bounds(old_modular_root, buffer_m)
    query_bbox = tuple(bounds["source_query_bbox_epsg4326"])
    source_file = source_root / "China_3" / "jiangsu.shp"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)

    info = pyogrio.read_info(source_file)
    source_crs = str(info.get("crs")) if info.get("crs") else None
    fields = [str(value) for value in info.get("fields", [])]
    height_field = "Height" if "Height" in fields else None
    if height_field is None:
        raise RuntimeError(f"Height field not present in real Jiangsu schema: {fields}")
    frame = pyogrio.read_dataframe(source_file, bbox=query_bbox, use_arrow=True)
    if frame.empty:
        raise RuntimeError("sealed COMMON_GRID-derived query returned zero GloBFP features")
    frame = frame.set_crs(source_crs or "EPSG:4326", allow_override=False)
    frame["globfp_object_id"], id_field = object_id_series(frame, fields)
    frame["source_file"] = str(source_file)
    frame["source_grid_id"] = "China_3"
    frame["original_crs"] = source_crs
    frame["footprint_area_m2"] = None
    frame["perimeter_m"] = None
    frame["actual_height_field_name"] = height_field
    frame["globfp_height_m"] = frame[height_field].astype(float)
    frame["height_valid"] = frame["globfp_height_m"].notna() & np.isfinite(frame["globfp_height_m"]) & (frame["globfp_height_m"] > 0)
    frame["geometry_quality"] = geometry_status(frame)
    frame["geometry_repaired"] = False
    frame["extraction_status"] = "raw_common_grid_buffer_intersection"
    frame["source_dataset_time_anchor"] = "2020"
    raw = frame.copy()

    metric = raw.to_crs("EPSG:32650")
    metric["footprint_area_m2"] = metric.geometry.area.astype(float)
    metric["perimeter_m"] = metric.geometry.length.astype(float)
    valid_geometry = metric.geometry.notna() & ~metric.geometry.is_empty.fillna(False)
    invalid = valid_geometry & ~metric.geometry.is_valid.fillna(False)
    repaired_count = 0
    if invalid.any():
        try:
            repaired = metric.loc[invalid, "geometry"].make_valid()
        except AttributeError:
            repaired = metric.loc[invalid, "geometry"].buffer(0)
        metric.loc[invalid, "geometry"] = repaired
        metric.loc[invalid, "geometry_repaired"] = metric.loc[invalid, "geometry"].notna() & ~metric.loc[invalid, "geometry"].is_empty
        repaired_count = int(metric.loc[invalid, "geometry_repaired"].sum())
    metric["geometry_quality"] = geometry_status(metric)
    metric["height_valid"] = metric["globfp_height_m"].notna() & np.isfinite(metric["globfp_height_m"]) & (metric["globfp_height_m"] > 0)
    metric["extraction_status"] = "valid_metric_candidate"
    excluded_mask = (
        metric.geometry.isna()
        | metric.geometry.is_empty.fillna(False)
        | ~metric["height_valid"]
        | ~metric.geometry.is_valid.fillna(False)
        | (metric["footprint_area_m2"] <= 0)
    )
    excluded = metric.loc[excluded_mask].copy()
    excluded["extraction_status"] = "excluded_geometry_or_height_quality"
    valid = metric.loc[~excluded_mask].copy()
    valid["extraction_status"] = "valid_metric_candidate"
    valid["scientific_status"] = "globfp_2020_estimated_building_height"
    valid["building_volume_proxy_m3"] = valid["footprint_area_m2"] * valid["globfp_height_m"]
    valid["height_unit"] = "m"
    valid["height_unit_provenance_status"] = HEIGHT_UNIT_PROVENANCE["status"]
    valid["geometry_quality"] = geometry_status(valid)
    volume = valid[
        [
            "globfp_object_id",
            "source_file",
            "source_grid_id",
            "original_crs",
            "footprint_area_m2",
            "perimeter_m",
            "actual_height_field_name",
            "globfp_height_m",
            "height_valid",
            "height_unit",
            "building_volume_proxy_m3",
            "geometry_repaired",
            "geometry_quality",
            "extraction_status",
            "scientific_status",
            "geometry",
        ]
    ].copy()

    raw_path = extract_root / "globfp_nanjing_raw.gpkg"
    valid_path = extract_root / "globfp_nanjing_valid.gpkg"
    excluded_path = extract_root / "globfp_nanjing_excluded.gpkg"
    volume_path = extract_root / "globfp_nanjing_volume_2020.gpkg"
    pyogrio.write_dataframe(raw, raw_path, layer="globfp_nanjing_raw", driver="GPKG")
    pyogrio.write_dataframe(valid, valid_path, layer="globfp_nanjing_valid", driver="GPKG")
    pyogrio.write_dataframe(excluded, excluded_path, layer="globfp_nanjing_excluded", driver="GPKG")
    pyogrio.write_dataframe(volume, volume_path, layer="globfp_nanjing_volume_2020", driver="GPKG")

    # A compact metric raster is only a spatial summary of the extracted slice.
    metric_bounds = bounds["metric_bounds_epsg32650_with_buffer"]
    resolution_m = 10.0
    width = int(math.ceil((metric_bounds[2] - metric_bounds[0]) / resolution_m))
    height = int(math.ceil((metric_bounds[3] - metric_bounds[1]) / resolution_m))
    transform = from_origin(metric_bounds[0], metric_bounds[3], resolution_m, resolution_m)
    raster_shapes = [(geom, 1) for geom in valid.geometry if geom is not None and not geom.is_empty]
    height_shapes = [(geom, float(value)) for geom, value in zip(valid.geometry, valid["globfp_height_m"]) if geom is not None and not geom.is_empty]
    footprint_arr = rasterize(raster_shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    height_arr = rasterize(height_shapes, out_shape=(height, width), transform=transform, fill=0, dtype="float32", merge_alg=rasterio.enums.MergeAlg.max)
    for path, array, dtype, nodata in (
        (extract_root / "globfp_nanjing_footprint_2020.tif", footprint_arr, "uint8", 0),
        (extract_root / "globfp_nanjing_height_2020.tif", height_arr, "float32", 0),
    ):
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype=dtype, crs="EPSG:32650", transform=transform, nodata=nodata, compress="deflate") as dataset:
            dataset.write(array, 1)

    source_sha = sha256_file(source_file)
    audit_payload = {
        "status": "PASS_TARGET_SLICE_EXTRACTED",
        "created_at_utc": utc_now(),
        "source_root_read_only": str(source_root),
        "source_file": str(source_file),
        "source_file_sha256": source_sha,
        "source_crs": source_crs,
        "source_geometry_type": str(info.get("geometry_type")),
        "source_fields": fields,
        "actual_height_field_name": height_field,
        "building_id_field_selected": id_field,
        "height_unit_provenance": HEIGHT_UNIT_PROVENANCE,
        "dataset_time_anchor": "2020",
        "bounds": bounds,
        "metric_crs": "EPSG:32650",
        "query_bbox_not_hand_drawn": True,
        "raw_feature_count": int(len(raw)),
        "valid_feature_count": int(len(valid)),
        "excluded_feature_count": int(len(excluded)),
        "geometry_repaired_count": repaired_count,
        "height_valid_count": int(valid["height_valid"].sum()),
        "height_invalid_count": int((~valid["height_valid"]).sum()),
        "empty_geometry_count_raw": int(raw.geometry.isna().sum() + raw.geometry.is_empty.fillna(False).sum()),
        "invalid_geometry_count_raw": int((~raw.geometry.is_valid.fillna(False)).sum()),
        "duplicate_object_id_count_raw": int(raw["globfp_object_id"].duplicated().sum()),
        "raster_resolution_m": resolution_m,
        "raster_shape": [height, width],
        "source_data_modified": False,
        "locked_test_pixel_reads": 0,
        "training_started": False,
    }
    manifest = {
        "status": audit_payload["status"],
        "created_at_utc": audit_payload["created_at_utc"],
        "outputs": {
            "raw": str(raw_path),
            "valid": str(valid_path),
            "excluded": str(excluded_path),
            "footprint_raster": str(extract_root / "globfp_nanjing_footprint_2020.tif"),
            "height_raster": str(extract_root / "globfp_nanjing_height_2020.tif"),
            "volume_proxy": str(volume_path),
        },
        "audit": audit_payload,
    }
    write_json(extract_root / "EXTRACTION_AUDIT.json", audit_payload)
    write_json(extract_root / "EXTRACTION_MANIFEST.json", manifest)
    write_json(extract_root / "HEIGHT_UNIT_PROVENANCE.json", HEIGHT_UNIT_PROVENANCE)
    write_json(audit_root / "HEIGHT_UNIT_PROVENANCE.json", HEIGHT_UNIT_PROVENANCE)
    return audit_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\Data Set\3D-GloBFP"))
    parser.add_argument("--old-modular-root", type=Path, default=project_root / "outputs" / "inference" / "nanjing_jiangbei_modular_rebuild_20260803_230320")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--buffer-m", type=float, default=250.0)
    args = parser.parse_args()
    result = run_extract(args.source_root, args.old_modular_root, args.output_root, args.buffer_m)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
