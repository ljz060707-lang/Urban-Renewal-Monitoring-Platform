from __future__ import annotations

import numpy as np


def repair_geometry(geometry):
    if geometry.is_empty:
        return geometry
    if not geometry.is_valid:
        try:
            from shapely import make_valid
            geometry = make_valid(geometry)
        except ImportError:
            geometry = geometry.buffer(0)
    return geometry


def vectorize_mask(mask, transform, *, min_area=10.0, simplify_tolerance=0.0, max_area_loss=0.02):
    import rasterio.features
    from shapely.geometry import shape

    binary = np.asarray(mask, dtype=np.uint8)
    geometries = []
    for mapping, value in rasterio.features.shapes(binary, mask=binary == 1, transform=transform, connectivity=8):
        if value != 1:
            continue
        geometry = repair_geometry(shape(mapping))
        if geometry.is_empty or geometry.area < min_area:
            continue
        if simplify_tolerance > 0:
            simplified = repair_geometry(geometry.simplify(simplify_tolerance, preserve_topology=True))
            if not simplified.is_empty and abs(simplified.area - geometry.area) / max(geometry.area, 1e-9) <= max_area_loss:
                geometry = simplified
        geometries.append(geometry)
    return geometries
