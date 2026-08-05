from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AreaAudit:
    raster_area_m2: float | None
    vector_area_m2: float | None
    area_discrepancy_ratio: float | None
    area_audit: str
    spatial_status: str


def is_metric_crs(crs) -> bool:
    if crs is None:
        return False
    try:
        axis = crs.axis_info
        return bool(crs.is_projected and axis and all(item.unit_name.casefold() in {"metre", "meter"} for item in axis[:2]))
    except Exception:
        return False


def audit_area(mask, transform, geometries, crs, *, discrepancy_limit=0.05) -> AreaAudit:
    import numpy as np
    from pyproj import CRS

    if crs is None or not is_metric_crs(CRS.from_user_input(crs)):
        return AreaAudit(None, None, None, "not_applicable", "pixel_coordinates_only_or_non_metric_crs")
    pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d))
    raster_area = float(np.count_nonzero(np.asarray(mask) == 1) * pixel_area)
    vector_area = float(sum(geometry.area for geometry in geometries))
    discrepancy = abs(vector_area - raster_area) / max(raster_area, 1e-9)
    return AreaAudit(raster_area, vector_area, discrepancy, "pass" if discrepancy <= discrepancy_limit else "fail", "metric_projected")


def lifecycle_areas(site_geometries, initial_building=None, final_building=None):
    from shapely.ops import unary_union

    site_union = unary_union(site_geometries) if site_geometries else None
    initial = float(initial_building.area) if initial_building is not None else 0.0
    final = float(final_building.area) if final_building is not None else 0.0
    return {
        "track_union_area_m2": float(site_union.area) if site_union is not None else 0.0,
        "net_building_change_area_m2": final - initial,
        "affected_building_area_m2": float(initial_building.symmetric_difference(final_building).area) if initial_building is not None and final_building is not None else None,
    }
