import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from urrs_lite_review.core import validate_geometry


def test_self_intersection_blocked():
    bow = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    assert validate_geometry(bow)[0] is False


def test_closed_polygon_is_valid():
    square = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    assert validate_geometry(square)[0] is True

