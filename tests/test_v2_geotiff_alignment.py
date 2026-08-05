import numpy as np
import rasterio
from rasterio.transform import from_origin

from urban_monitor.change.geospatial import aligned_pair, grid_from_dataset, write_float_raster


def _raster(path, transform):
    with rasterio.open(
        path, "w", driver="GTiff", width=8, height=8, count=3, dtype="uint8",
        crs="EPSG:32650", transform=transform,
    ) as sink:
        sink.write(np.zeros((3, 8, 8), dtype=np.uint8))


def test_alignment_and_output_preserve_reference_grid(tmp_path):
    t1, t2 = tmp_path / "t1.tif", tmp_path / "t2.tif"
    _raster(t1, from_origin(500000, 4000000, 1, 1))
    _raster(t2, from_origin(500001, 4000000, 1, 1))
    with aligned_pair(t1, t2) as (first, second, aligned):
        assert aligned
        output = write_float_raster(tmp_path / "prob.tif", np.ones((8, 8)), first)
        with rasterio.open(output) as result:
            assert grid_from_dataset(result).transform == grid_from_dataset(first).transform
            assert result.crs == first.crs

