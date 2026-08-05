from urban_monitor.change.preflight_v2 import run_v2_preflight


def test_missing_optional_teachers_do_not_block_partial(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "marker").write_text("ok")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"x")
    report = run_v2_preflight(
        {
            "teachers": {
                "dsifn": {"root": str(root), "checkpoint": str(checkpoint)},
                "s2looking": {"root": str(root), "checkpoint": str(checkpoint)},
                "second": {"root": str(tmp_path / "missing")},
                "rsbuilding": {"root": str(tmp_path / "missing-rs")},
            }
        }
    )
    assert report["readiness"]["READY_FOR_PARTIAL_INFERENCE"]
    assert report["readiness"]["READY_FOR_V2_QUICK"]
    assert not report["readiness"]["READY_FOR_FULL_FUSION"]
    assert report["test_images_read"] == 0

