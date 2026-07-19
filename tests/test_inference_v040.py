from pathlib import Path

def test_checkpoint_hash_is_stable(tmp_path):
    import pytest

    pytest.importorskip("torch")
    from urban_monitor.model import checkpoint_sha256

    path = tmp_path / "best.pth"
    path.write_bytes(b"trained-model")
    before = checkpoint_sha256(path)
    assert before == checkpoint_sha256(path)
    path.write_bytes(b"changed-model")
    assert before != checkpoint_sha256(path)


def test_legacy_inference_defaults_to_best_checkpoint():
    text = Path("scripts/infer_timeseries.py").read_text(encoding="utf-8")
    assert "best.pth" in text and "latest.pth" not in text
