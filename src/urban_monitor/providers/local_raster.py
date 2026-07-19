from __future__ import annotations

from pathlib import Path

import rasterio

from ..crs import validate_source_crs


def inspect_local_raster(path: str | Path) -> dict[str, object]:
    """Inspect a legally prepared local raster without provider-specific assumptions."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with rasterio.open(source) as dataset:
        crs = validate_source_crs(dataset.crs)
        return {
            "path": str(source),
            "crs": crs.to_string(),
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "bounds": tuple(dataset.bounds),
            "transform": tuple(dataset.transform)[:6],
            "nodata": dataset.nodata,
        }
