import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from urrs_lite_review.core import ReviewDB, frozen_split


def test_split_is_deterministic_and_locked(tmp_path):
    db = ReviewDB(tmp_path / "review.sqlite")
    db.add_project("demo", "Demo", ["2020-01-01"])
    g = {"type": "Point", "coordinates": [0, 0]}
    db.add_sample(sample_id="a", project="demo", module="m", scene_date="2020-01-01", block="demo_b", source_asset=None, chip_path=None, source_geometry=g)
    db.add_sample(sample_id="b", project="demo", module="m", scene_date="2020-01-02", block="demo_b", source_asset=None, chip_path=None, source_geometry=g)
    assert db.conn.execute("select split from samples where sample_id='a'").fetchone()[0] == db.conn.execute("select split from samples where sample_id='b'").fetchone()[0]
    assert db.conn.execute("select locked from split_registry where spatial_block_id='demo_b'").fetchone()[0] == 1
    assert frozen_split("demo_b", "demo") == frozen_split("demo_b", "demo")
    db.close()

