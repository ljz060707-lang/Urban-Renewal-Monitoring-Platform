from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urban_monitor.change.local_labels import generate_local_change_candidates


def _image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:32650", transform=from_origin(500000, 4400000, 1, 1)) as target:
        target.write(np.zeros((3,64,64), dtype=np.uint8))


def test_local_labels_are_candidates_requiring_review(tmp_path: Path):
    project = tmp_path / "data" / "aoi"; labels = project / "labels"; images = project / "images"; labels.mkdir(parents=True); images.mkdir()
    _image(images / "2023-01-01.tif"); _image(images / "2024-01-01.tif")
    gpd.GeoDataFrame({"geometry":[box(500005,4399950,500015,4399960)]}, crs="EPSG:32650").to_file(labels / "2023-01-01.gpkg", driver="GPKG")
    gpd.GeoDataFrame({"geometry":[box(500005,4399950,500015,4399960),box(500030,4399950,500040,4399960)]}, crs="EPSG:32650").to_file(labels / "2024-01-01.gpkg", driver="GPKG")
    run = tmp_path / "outputs" / "aoi" / "runs" / "r1"; run.mkdir(parents=True)
    latest = tmp_path / "outputs" / "aoi" / "latest.json"; latest.parent.mkdir(parents=True, exist_ok=True); latest.write_text(json.dumps({"status":"success","run_directory":str(run)}), encoding="utf-8")
    output = generate_local_change_candidates("aoi", repo_root=tmp_path)
    candidates = gpd.read_file(output / "2023-01-01_to_2024-01-01" / "GT_NEW_candidate.gpkg")
    assert len(candidates) == 1 and candidates.iloc[0]["candidate_status"] == "requires_human_review"
    page = run / "review" / "local_change_labels" / "index.html"
    assert "候选不是 Ground Truth" in page.read_text(encoding="utf-8")
