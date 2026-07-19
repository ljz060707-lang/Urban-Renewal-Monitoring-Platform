import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urban_monitor.downscale import export_student_change_labels


def _fixture(tmp_path):
    summary = tmp_path / "summary.gpkg"; intervals = tmp_path / "intervals.csv"; reference = tmp_path / "ref.tif"
    frame = gpd.GeoDataFrame([
        {"track_id": "small", "state": "new_building", "confidence": "high", "review_status": "auto", "geometry": box(0, 0, 19.9, 10)},
        {"track_id": "keep", "state": "new_building", "confidence": "high", "review_status": "auto", "geometry": box(20, 0, 40.1, 10)},
    ], crs="EPSG:32650")
    frame.to_file(summary, driver="GPKG")
    pd.DataFrame([{"track_id": "small", "interval_end": "2024-01-01", "earliest_absent_date": ""}, {"track_id": "keep", "interval_end": "2024-01-01", "earliest_absent_date": ""}]).to_csv(intervals, index=False, encoding="utf-8-sig")
    with rasterio.open(reference, "w", driver="GTiff", width=50, height=20, count=1, dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 20, 1, 1)) as dst:
        dst.write(np.ones((20, 50), dtype=np.uint8), 1)
    return summary, intervals, reference


def test_mmu_199_201_boundary_and_fraction_range(tmp_path):
    summary, intervals, reference = _fixture(tmp_path); output = tmp_path / "labels"
    paths = export_student_change_labels(summary, intervals, reference, output, min_mapping_unit_m2=200, run_id="run1")
    assert len(paths) == 1
    excluded = pd.read_csv(output / "excluded_by_mmu.csv", encoding="utf-8-sig")
    assert list(excluded.track_id) == ["small"]
    with rasterio.open(paths[0]) as src:
        data = src.read()
        assert float(data.min()) >= 0 and float(data.max()) <= 1
        assert src.tags()["run_id"] == "run1"


def test_missing_date_is_reported_not_invented(tmp_path):
    summary, intervals, reference = _fixture(tmp_path); output = tmp_path / "labels"
    data = pd.read_csv(intervals, encoding="utf-8-sig"); data.loc[data.track_id == "keep", "interval_end"] = ""; data.to_csv(intervals, index=False, encoding="utf-8-sig")
    export_student_change_labels(summary, intervals, reference, output, min_mapping_unit_m2=200)
    excluded = pd.read_csv(output / "excluded_missing_date.csv", encoding="utf-8-sig")
    assert list(excluded.track_id) == ["keep"]
