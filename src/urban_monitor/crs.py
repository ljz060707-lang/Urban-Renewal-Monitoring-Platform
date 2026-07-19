from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import CRS, Geod
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry.base import BaseGeometry


def validate_source_crs(value: Any) -> CRS:
    """Return a parsed CRS and never guess when metadata is missing."""
    if value is None or str(value).strip() in {"", "None"}:
        raise ValueError("Source CRS is missing; declare the verified CRS before processing")
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f"Invalid source CRS: {value!r}") from exc


def _geometry_and_crs(aoi: Any, source_crs: Any | None) -> tuple[BaseGeometry, CRS]:
    if isinstance(aoi, gpd.GeoDataFrame):
        if aoi.empty:
            raise ValueError("AOI is empty")
        return aoi.geometry.union_all(), validate_source_crs(aoi.crs)
    if isinstance(aoi, gpd.GeoSeries):
        if aoi.empty:
            raise ValueError("AOI is empty")
        return aoi.union_all(), validate_source_crs(aoi.crs)
    if isinstance(aoi, BaseGeometry):
        if aoi.is_empty:
            raise ValueError("AOI is empty")
        return aoi, validate_source_crs(source_crs)
    raise TypeError("AOI must be a GeoDataFrame, GeoSeries or Shapely geometry")


def get_aoi_centroid_lonlat(aoi: Any, source_crs: Any | None = None) -> tuple[float, float]:
    """Return the AOI centroid in WGS84 longitude/latitude."""
    geometry, crs = _geometry_and_crs(aoi, source_crs)
    series = gpd.GeoSeries([geometry], crs=crs).to_crs("EPSG:4326")
    centroid = series.iloc[0].centroid
    return float(centroid.x), float(centroid.y)


def select_local_utm_crs(
    aoi: Any,
    source_crs: Any | None = None,
    *,
    max_width_km: float = 100.0,
) -> CRS:
    """Select a northern WGS84 UTM zone from the AOI centroid.

    Phase 1 rejects AOIs wider than ``max_width_km`` and warns when a small AOI
    crosses a UTM zone boundary.
    """
    geometry, crs = _geometry_and_crs(aoi, source_crs)
    lonlat = gpd.GeoSeries([geometry], crs=crs).to_crs("EPSG:4326").iloc[0]
    minx, miny, maxx, maxy = lonlat.bounds
    lon, lat = float(lonlat.centroid.x), float(lonlat.centroid.y)
    if not (0.0 <= lat <= 84.0):
        raise ValueError(f"Phase 1 expects a northern-hemisphere AOI, got latitude={lat:.6f}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"AOI longitude is invalid: {lon}")
    _, _, width_m = Geod(ellps="WGS84").inv(minx, lat, maxx, lat)
    width_km = abs(width_m) / 1000.0
    if width_km > max_width_km:
        raise ValueError(
            f"AOI width {width_km:.1f} km exceeds Phase-1 limit {max_width_km:.1f} km; split the AOI"
        )
    zone_min = int((minx + 180.0) // 6.0) + 1
    zone_max = int((maxx + 180.0) // 6.0) + 1
    if zone_min != zone_max:
        warnings.warn(
            f"AOI crosses UTM zones {zone_min}/{zone_max}; using centroid zone because width is {width_km:.1f} km",
            stacklevel=2,
        )
    zone = min(60, max(1, int((lon + 180.0) // 6.0) + 1))
    result = CRS.from_epsg(32600 + zone)
    if result.axis_info[0].unit_name.lower() not in {"metre", "meter"}:
        raise ValueError(f"Selected CRS is not metric: {result.to_string()}")
    return result


def ensure_metric_crs(value: Any) -> CRS:
    """Reject geographic and Web Mercator CRS for formal metric operations."""
    crs = validate_source_crs(value)
    if crs.is_geographic:
        raise ValueError(f"Metric operation is forbidden in geographic CRS {crs.to_string()}")
    if crs.to_epsg() == 3857:
        raise ValueError("EPSG:3857 is display-only and forbidden for formal area/distance operations")
    units = {axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name}
    if not units.intersection({"metre", "meter"}):
        raise ValueError(f"CRS does not use metres: {crs.to_string()}")
    return crs


def ensure_same_crs(*values: Any) -> CRS:
    """Require all non-empty CRS values to be equivalent."""
    parsed = [validate_source_crs(value) for value in values]
    if not parsed:
        raise ValueError("At least one CRS is required")
    if any(item != parsed[0] for item in parsed[1:]):
        raise ValueError(f"CRS mismatch: {[item.to_string() for item in parsed]}")
    return parsed[0]


def reproject_vector(frame: gpd.GeoDataFrame, target_crs: Any) -> gpd.GeoDataFrame:
    """Return a reprojected copy while refusing missing source CRS."""
    validate_source_crs(frame.crs)
    target = validate_source_crs(target_crs)
    return frame.to_crs(target)


def reproject_raster_to_reference(
    source_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject a raster exactly onto a reference raster grid."""
    output = Path(output_path)
    with rasterio.open(source_path) as src, rasterio.open(reference_path) as ref:
        validate_source_crs(src.crs)
        validate_source_crs(ref.crs)
        profile = src.profile.copy()
        profile.update(crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    resampling=resampling,
                )
    return output


def write_crs_provenance(path: str | Path, **values: Any) -> Path:
    """Write CRS decisions as UTF-8 JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: (CRS.from_user_input(value).to_string() if "crs" in key and value else value) for key, value in values.items()}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
