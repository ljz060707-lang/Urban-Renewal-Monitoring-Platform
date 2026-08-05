from urban_monitor.change.adapters.base import split_directories


def test_allowed_splits_never_returns_test(tmp_path):
    for name in ("train", "validation", "test"):
        (tmp_path / name).mkdir()
    result = split_directories(tmp_path, ["train", "validation"])
    assert [name for name, _ in result] == ["train", "val"]
    assert all(path.name.casefold() != "test" for _, path in result)
