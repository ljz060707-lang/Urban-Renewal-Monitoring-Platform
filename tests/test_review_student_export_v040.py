import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urban_monitor.downscale import export_student_change_labels


def test_reviewed_rejected_new_building_never_enters_student_labels(tmp_path):
    summary = tmp_path / "tracks_reviewed.gpkg"
    intervals = tmp_path / "state_intervals.csv"
    reference = tmp_path / "reference.tif"
    frame = gpd.GeoDataFrame(
        [{
            "track_id": "trk1", "state": "new_building", "reviewed_state": "new_building",
            "state_final": "rejected", "review_status": "reviewed", "confidence": "high",
            "geometry": box(0, 0, 20, 20),
        }], crs="EPSG:32650"
    )
    frame.to_file(summary, driver="GPKG")
    pd.DataFrame([{
        "track_id": "trk1", "interval_end": "2024-01-01", "earliest_absent_date": "",
    }]).to_csv(intervals, index=False, encoding="utf-8-sig")
    with rasterio.open(
        reference, "w", driver="GTiff", height=32, width=32, count=1, dtype="uint8",
        crs="EPSG:32650", transform=from_origin(0, 32, 1, 1),
    ) as target:
        target.write(np.zeros((1, 32, 32), dtype=np.uint8))
    outputs = export_student_change_labels(
        summary, intervals, reference, tmp_path / "student", min_mapping_unit_m2=1,
        require_reviewed_or_high_confidence=False,
    )
    assert outputs == []
    inventory = pd.read_csv(tmp_path / "student" / "label_inventory.csv")
    assert inventory.empty


def test_effective_state_prefers_state_final(tmp_path):
    summary = tmp_path / "summary.gpkg"
    intervals = tmp_path / "intervals.csv"
    reference = tmp_path / "reference.tif"
    gpd.GeoDataFrame([{
        "track_id": "trk1", "state": "demolished", "reviewed_state": "stable",
        "state_final": "new_building", "review_status": "reviewed", "confidence": "high",
        "geometry": box(0, 0, 20, 20),
    }], crs="EPSG:32650").to_file(summary, driver="GPKG")
    pd.DataFrame([{"track_id": "trk1", "interval_end": "2024-01-01", "earliest_absent_date": "2024-02-01"}]).to_csv(
        intervals, index=False, encoding="utf-8-sig"
    )
    with rasterio.open(reference, "w", driver="GTiff", height=32, width=32, count=1, dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 32, 1, 1)) as target:
        target.write(np.zeros((1, 32, 32), dtype=np.uint8))
    outputs = export_student_change_labels(summary, intervals, reference, tmp_path / "student", min_mapping_unit_m2=1)
    assert len(outputs) == 1
    with rasterio.open(outputs[0]) as source:
        assert source.read(1).max() > 0 and source.read(2).max() == 0
