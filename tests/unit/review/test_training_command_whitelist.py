import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from urrs_lite_review.launch_controlled_training import is_allowed_command


def test_arbitrary_shell_command_rejected():
    assert not is_allowed_command("TR1_HEAD_ONLY", ["cmd", "/c", "whoami"])


def test_whitelisted_token_only():
    assert is_allowed_command("TR1_HEAD_ONLY")
    assert not is_allowed_command("rm_rf")

