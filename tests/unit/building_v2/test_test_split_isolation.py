from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_diagnostic_tools_reject_test_paths_and_receipt_zero_reads():
    for name in ("native_b2_diagnostic_worker.py", "run_building_v2_b2_boundary_diagnostic.py", "init_building_v2_b2_boundary_diagnostic.py"):
        text = (ROOT / "tools" / name).read_text(encoding="utf-8")
        assert "test_pixel_reads" in text
        assert "test_used" in text
    assert "\\\\test\\\\" in (ROOT / "tools" / "native_b2_diagnostic_worker.py").read_text(encoding="utf-8")

