from pathlib import Path

import pytest

from surface_state_v1.training.checkpointing import sha256


def test_frozen_checkpoint_hashes():
    repo = Path(__file__).resolve().parents[3]
    full20 = repo / "outputs/pretraining/change_v2/20260727_001029/checkpoints/best_v2_multisource_change.pth"
    rsbuilding = repo / "outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth"
    if not full20.exists() or not rsbuilding.exists():
        pytest.skip("Frozen checkpoints are not available in this checkout")
    assert sha256(full20) == "70040ca0e688537fd63479589690d5be5eefef859f39763c6b45a09852bc2621"
    assert sha256(rsbuilding) == "f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7"
