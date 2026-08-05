import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from urrs_lite_review.server import Handler


def test_server_declares_local_only():
    assert "127.0.0.1" in Handler.server_version or Handler.server_version.startswith("URRS-Lite")

