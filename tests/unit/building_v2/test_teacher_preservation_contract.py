from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_conflict_decision_prefers_teacher_preservation_over_generic_regularization():
    text = (ROOT / "tools" / "run_building_v2_b2_boundary_diagnostic.py").read_text(encoding="utf-8")
    assert "teacher_preservation_recommended" in text
    assert "generic_regularization_recommended" in text
    assert "ROUTE_A_BOUNDARY_PROBE_FOR_OBJECTIZATION" in text

