from __future__ import annotations

from pathlib import Path

import numpy as np


def vectorize_mask(
    mask_path: str | Path,
    output_path: str | Path,
    *,
    layer: str,
    class_name: str,
    min_mapping_unit_m2: float = 10.0,
    source: str = "v2_fusion",
    confidence: float | None = None,
) -> dict:
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape

    with rasterio.open(mask_path) as dataset:
        values = dataset.read(1)
        valid = values > 0
        geometries = [
            shape(geometry)
            for geometry, value in shapes(values.astype(np.uint8), mask=valid, transform=dataset.transform)
            if int(value) == 1
        ]
        frame = gpd.GeoDataFrame(
            {
                "class_name": [class_name] * len(geometries),
                "source": [source] * len(geometries),
                "confidence": [confidence] * len(geometries),
                "status": ["candidate"] * len(geometries),
            },
            geometry=geometries,
            crs=dataset.crs,
        )
    before = len(frame)
    if not frame.empty:
        if frame.crs and frame.crs.is_geographic:
            area = frame.to_crs(frame.estimate_utm_crs()).area
        else:
            area = frame.area
        frame["area_m2"] = area.to_numpy()
        frame = frame.loc[frame["area_m2"] >= float(min_mapping_unit_m2)].copy()
    else:
        frame["area_m2"] = []
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(target, layer=layer, driver="GPKG")
    return {"features_before_mmu": before, "features_written": len(frame), "path": str(target.resolve())}

