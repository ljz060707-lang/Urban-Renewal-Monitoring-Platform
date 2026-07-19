from pathlib import Path

import pytest

from urban_monitor.outputs import prepare_output


def test_force_is_required_to_replace_output(tmp_path: Path):
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        prepare_output(target, force=False)
    assert target.read_text(encoding="utf-8") == "old"


def test_force_removes_only_explicit_output(tmp_path: Path):
    target = tmp_path / "result.txt"
    sibling = tmp_path / "keep.txt"
    target.write_text("old", encoding="utf-8")
    sibling.write_text("keep", encoding="utf-8")
    assert prepare_output(target, force=True) == target
    assert not target.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"
