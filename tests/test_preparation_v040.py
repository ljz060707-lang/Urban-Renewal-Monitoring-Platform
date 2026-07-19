from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urban_monitor.project_discovery import discover_project
from urban_monitor.provenance import file_sha256
from urban_monitor.preparation import prepare_project


def _write_image(path: Path, value: int):
    data = np.full((3, 32, 32), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(116.30, 39.90, 0.0001, 0.0001),
    ) as target:
        target.write(data)


def test_prepare_selects_utm_and_preserves_sources(tmp_path):
    project_dir = tmp_path / "data" / "中文片区"
    images = project_dir / "images"
    labels = project_dir / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for date, value in [("2023-01-01", 80), ("2024-01-01", 120)]:
        image = images / f"{date}.tif"
        _write_image(image, value)
        gpd.GeoDataFrame(geometry=[box(116.3005, 39.8975, 116.3015, 39.8985)], crs="EPSG:4326").to_file(
            labels / f"{date}.gpkg", driver="GPKG"
        )
    before = {path.name: file_sha256(path) for path in images.glob("*.tif")}
    result = prepare_project(discover_project("中文片区", tmp_path / "data"), tmp_path / "run")
    assert result["analysis_crs"].to_epsg() == 32650
    assert (tmp_path / "run" / "prepared" / "reference_grid.json").exists()
    assert (tmp_path / "run" / "prepared" / "crs_provenance.json").exists()
    assert (tmp_path / "run" / "data_report" / "accepted_labels.gpkg").exists()
    for path in images.glob("*.tif"):
        assert file_sha256(path) == before[path.name]
    with rasterio.open(tmp_path / "run" / "prepared" / "images" / "2023-01-01.tif") as source:
        assert source.crs.to_epsg() == 32650


def test_prepare_rejects_missing_label_crs(tmp_path):
    project_dir = tmp_path / "data" / "demo"
    (project_dir / "images").mkdir(parents=True)
    (project_dir / "labels").mkdir()
    _write_image(project_dir / "images" / "2024-01-01.tif", 80)
    # GeoJSON without an explicit crs is interpreted as WGS84, so use a Shapefile with cleared .prj.
    label = project_dir / "labels" / "2024-01-01.shp"
    gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(label)
    label.with_suffix(".prj").unlink()
    project = discover_project("demo", tmp_path / "data")
    try:
        prepare_project(project, tmp_path / "run")
    except ValueError as exc:
        assert "CRS is missing" in str(exc)
    else:
        raise AssertionError("missing label CRS should stop prepare")
