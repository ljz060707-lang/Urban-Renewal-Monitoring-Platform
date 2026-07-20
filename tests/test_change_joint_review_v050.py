from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from urban_monitor.change.joint_review import apply_joint_review, build_joint_review


def test_joint_review_updates_state_and_rejected_never_enters_student_labels(tmp_path: Path):
    fused = tmp_path / "fused.gpkg"
    gpd.GeoDataFrame({"fusion_id":["a","b"],"state_final":["stable","new_building"],"review_priority":["high","low"],"geometry":[box(0,0,1,1),box(2,2,3,3)]}, crs="EPSG:3857").to_file(fused, driver="GPKG")
    page = build_joint_review(fused, tmp_path / "review")
    assert "joint_review_decisions.csv" in page.read_text(encoding="utf-8")
    decisions = tmp_path / "decisions.csv"
    pd.DataFrame([{"fusion_id":"a","final_state":"rejected","teacher_decision":"both_wrong","notes":"bad","review_time":"now"}]).to_csv(decisions,index=False)
    reviewed, student = apply_joint_review(fused, decisions, tmp_path / "out")
    assert gpd.read_file(reviewed).set_index("fusion_id").loc["a","state_final"] == "rejected"
    assert gpd.read_file(student)["fusion_id"].tolist() == ["b"]
