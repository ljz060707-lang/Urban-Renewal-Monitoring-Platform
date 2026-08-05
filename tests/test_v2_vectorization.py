import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from urban_monitor.change.vectorize_v2 import vectorize_mask


def test_vectorization_applies_mmu_and_preserves_crs(tmp_path):
    raster = tmp_path / "mask.tif"
    values = np.zeros((12, 12), dtype=np.uint8)
    values[1:3, 1:3] = 1
    values[5:10, 5:10] = 1
    with rasterio.open(
        raster, "w", driver="GTiff", width=12, height=12, count=1, dtype="uint8",
        crs="EPSG:32650", transform=from_origin(500000, 4000000, 1, 1),
    ) as sink:
        sink.write(values, 1)
    output = tmp_path / "features.gpkg"
    report = vectorize_mask(raster, output, layer="new", class_name="new", min_mapping_unit_m2=10)
    frame = gpd.read_file(output, layer="new")
    assert report["features_written"] == 1
    assert frame.crs.to_epsg() == 32650
    assert frame.iloc[0]["area_m2"] >= 10

