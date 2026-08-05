from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_head_only_probe_has_bounded_steps_and_loss_variants():
    text = (ROOT / "tools" / "run_building_v2_b2_boundary_diagnostic.py").read_text(encoding="utf-8")
    assert "head-steps" in text
    assert '("l0", "l1")' in text
    assert "boundary_loss_drop" in text

