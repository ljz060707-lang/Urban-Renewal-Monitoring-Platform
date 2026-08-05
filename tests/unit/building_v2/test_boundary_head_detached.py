from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_detached_probe_has_explicit_detach_and_no_pilot_entrypoint():
    text = (ROOT / "tools" / "native_b2_diagnostic_worker.py").read_text(encoding="utf-8")
    assert "self._feature.detach()" in text
    assert "head_only" in text
    runner = (ROOT / "tools" / "run_building_v2_b2_boundary_diagnostic.py").read_text(encoding="utf-8")
    assert '"pilot"' not in runner

