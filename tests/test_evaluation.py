import geopandas as gpd
from shapely.geometry import box

from urban_monitor.evaluation import evaluate_phase1


def test_exact_object_match_scores_one(tmp_path):
    prediction = tmp_path / "pred.gpkg"; truth = tmp_path / "truth.gpkg"
    for path in [prediction, truth]:
        gpd.GeoDataFrame([{"geometry": box(0, 0, 10, 10)}], crs="EPSG:32650").to_file(path, driver="GPKG")
    summary = evaluate_phase1({"2024-01-01": prediction}, {"2024-01-01": truth}, tmp_path / "evaluation")
    assert summary["segmentation"]["f1"] == 1
    assert summary["tracking"].startswith("not computable")


def test_missing_ground_truth_is_not_computable(tmp_path):
    summary = evaluate_phase1({}, {}, tmp_path / "evaluation")
    assert summary["segmentation"] == "not computable"
    assert "not computable" in (tmp_path / "evaluation" / "evaluation_summary.md").read_text(encoding="utf-8")
