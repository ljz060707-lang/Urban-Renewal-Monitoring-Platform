import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from urrs_lite_review.core import ReviewDB


def test_holdout_setting_is_sealed(tmp_path):
    db = ReviewDB(tmp_path / "review.sqlite")
    assert db.setting("holdout_sealed") == "1"
    db.close()

