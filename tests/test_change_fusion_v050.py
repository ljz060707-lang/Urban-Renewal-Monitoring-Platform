from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urban_monitor.change.fusion import fuse_teachers


def _raster(path: Path, left_value: float, right_value: float):
    data = np.zeros((20,20), dtype=np.float32); data[:, :10] = left_value; data[:, 10:] = right_value
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=1, dtype="float32", crs="EPSG:32650", transform=from_origin(0,20,1,1)) as target: target.write(data,1)


def test_double_teacher_fusion_flags_disagreement(tmp_path: Path):
    tracks = gpd.GeoDataFrame({"state_final":["stable","new_building"],"geometry":[box(1,1,8,8),box(12,1,18,8)]}, crs="EPSG:32650")
    tracks_path = tmp_path / "tracks.gpkg"; tracks.to_file(tracks_path, driver="GPKG")
    any_path,new_path,dem_path = tmp_path/"any.tif",tmp_path/"new.tif",tmp_path/"dem.tif"
    _raster(any_path,0.9,0.9); _raster(new_path,0.1,0.9); _raster(dem_path,0.1,0.1)
    output = fuse_teachers(tracks_path, any_path, new_path, dem_path, tmp_path/"fusion")
    result = gpd.read_file(output)
    assert result.loc[0,"review_priority"] == "high"
    assert "stable_vs_high_change" in result.loc[0,"fusion_issues"]
    assert bool(result.loc[1,"teacher_agreement"])
