import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from urrs_lite_review.check_training_gate import _counts
from urrs_lite_review.core import ReviewDB


def test_empty_project_gate_counts_are_explicit(tmp_path):
    db = ReviewDB(tmp_path / "review.sqlite")
    got = _counts(db, "missing_project")
    assert got["rows"] == 0 and got["holdout_sealed"] is True
    db.close()

