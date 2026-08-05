from pathlib import Path

from urban_monitor.change.teacher_registry import TeacherStatus, build_teacher_registry


def test_registry_reports_ready_and_missing(tmp_path):
    dataset = tmp_path / "dsifn"
    dataset.mkdir()
    (dataset / "marker.txt").write_text("ready")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    registry = build_teacher_registry(
        {
            "teachers": {
                "dsifn": {"enabled": True, "root": str(dataset), "checkpoint": str(checkpoint)},
                "s2looking": {"enabled": True, "root": str(tmp_path / "missing")},
                "second": {"enabled": False},
                "rsbuilding": {"enabled": False},
            }
        }
    )
    assert registry.get("dsifn").metadata().status is TeacherStatus.READY
    assert registry.get("s2looking").metadata().status is TeacherStatus.MISSING
    assert "any_change_probability" in registry.get("dsifn").available_outputs()

