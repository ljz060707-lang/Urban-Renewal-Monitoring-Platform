import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from urrs_lite_review.core import write_geojson


def test_geojson_export_is_atomic(tmp_path):
    p = tmp_path / "labels.geojson"
    n = write_geojson(p, [{"geometry": {"type": "Point", "coordinates": [1, 2]}, "sample_id": "s1"}])
    assert n == 1 and p.exists() and not p.with_name(p.name + ".part").exists()

