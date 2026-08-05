from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_diagnostic_runner_has_no_full_pilot_stage():
    text = (ROOT / "tools" / "run_building_v2_b2_boundary_diagnostic.py").read_text(encoding="utf-8")
    assert "full_b2_pilot_started" in text
    assert 'choices=("target", "oracle", "baseline", "head-only", "gradient", "figures", "finalize", "all")' in text

