import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urban_monitor.manifest import DateRecord
from urban_monitor.review import apply_reviews, build_review_queue


def _summary(path):
    gpd.GeoDataFrame([{"track_id": "trk1", "state": "uncertain", "review_status": "pending", "confidence": "low", "mean_score": .8, "geometry": box(4, 4, 8, 8)}], crs="EPSG:32650").to_file(path, driver="GPKG")


def test_apply_reviews_preserves_auto_and_chinese_notes(tmp_path):
    summary = tmp_path / "summary.gpkg"; review = tmp_path / "review.csv"; output = tmp_path / "reviewed.gpkg"
    _summary(summary)
    pd.DataFrame([{"row_type": "track", "record_id": "trk1", "track_id": "trk1", "reviewed_state": "stable", "date_quality_json": "{}", "notes": "中文审核备注", "reviewer": "审核员"}]).to_csv(review, index=False, encoding="utf-8-sig")
    apply_reviews(summary, review, output)
    frame = gpd.read_file(output)
    assert frame.iloc[0].state == "uncertain" and frame.iloc[0].state_final == "stable"
    assert frame.iloc[0].notes == "中文审核备注" and frame.iloc[0].reviewer == "审核员"


def test_apply_reviews_is_idempotent(tmp_path):
    summary = tmp_path / "summary.gpkg"; review = tmp_path / "review.csv"; output = tmp_path / "reviewed.gpkg"
    _summary(summary)
    pd.DataFrame([{"row_type": "track", "record_id": "trk1", "track_id": "trk1", "reviewed_state": "stable"}]).to_csv(review, index=False, encoding="utf-8-sig")
    apply_reviews(summary, review, output); apply_reviews(summary, review, output)
    assert len(gpd.read_file(output)) == 1


def test_review_html_contains_chinese_ui(tmp_path):
    summary = tmp_path / "summary.gpkg"; image = tmp_path / "image.tif"; output = tmp_path / "review"
    _summary(summary)
    with rasterio.open(image, "w", driver="GTiff", width=16, height=16, count=3, dtype="uint8", crs="EPSG:32650", transform=from_origin(0, 16, 1, 1)) as dst:
        dst.write(np.ones((3, 16, 16), dtype=np.uint8) * 100)
    record = DateRecord("2024-01-01", image, None, None, "train")
    html = build_review_queue(summary, [record], output, padding_m=2)
    text = html.read_text(encoding="utf-8")
    assert "城市更新时序" in text and "id='reviewer'" in text
