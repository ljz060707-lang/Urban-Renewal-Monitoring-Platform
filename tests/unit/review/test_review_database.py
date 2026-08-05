import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from urrs_lite_review import core


def _db(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "EVENTS_PATH", tmp_path / "review_events.jsonl")
    monkeypatch.setattr(core, "DATA_ROOT", tmp_path)
    return core.ReviewDB(tmp_path / "review.sqlite")


def test_new_database_and_autosave(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    db.add_project("demo", "Demo", ["2020-01-01"])
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    db.add_sample(sample_id="s1", project="demo", module="demo", scene_date="2020-01-01", block="demo_b00", source_asset=None, chip_path=None, source_geometry=geom)
    result = db.record_review("s1", "accept", {"notes": "human checked"}, "tester")
    assert result["success"] and result["database_revision"] == 1
    assert json.loads((tmp_path / "review_events.jsonl").read_text(encoding="utf-8"))["sample_id"] == "s1"
    db.close()


def test_repair_from_jsonl(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    db.add_project("demo", "Demo", ["2020-01-01"])
    geom = {"type": "Point", "coordinates": [0, 0]}
    db.add_sample(sample_id="s1", project="demo", module="demo", scene_date="2020-01-01", block="demo_b00", source_asset=None, chip_path=None, source_geometry=geom)
    db.record_review("s1", "accept", {}, "tester")
    db.conn.execute("DELETE FROM review_decisions"); db.conn.commit()
    repaired = core.rebuild_review_decisions_from_jsonl(db)
    assert repaired["repaired"] == 1
    db.close()


def test_human_draw_and_delete_reason(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    db.add_project("demo", "Demo", ["2020-01-01"])
    point = {"type": "Point", "coordinates": [0, 0]}
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    db.add_sample(sample_id="s1", project="demo", module="demo", scene_date="2020-01-01", block="demo_b00", source_asset=None, chip_path=None, source_geometry=point)
    db.record_review("s1", "add", {"geometry": polygon}, "tester")
    assert db.conn.execute("select source from human_annotations where sample_id='s1'").fetchone()[0] == "human_drawn"
    try:
        db.record_review("s1", "delete", {}, "tester")
    except ValueError:
        pass
    else:
        raise AssertionError("delete without a reason must be rejected")
    db.close()
