import pandas as pd
import pytest

from urban_monitor.change.hard_examples_v2 import mine_v2_hard_examples
from urban_monitor.change.supervision_v2 import TestReadGuard


def test_test_guard_and_hard_mining_never_select_test(tmp_path):
    guard = TestReadGuard("dev-command")
    with pytest.raises(PermissionError):
        guard.check("test", "secret.png")
    index = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"sample_id": "train:1", "split": "train"},
            {"sample_id": "validation:1", "split": "validation"},
            {"sample_id": "test:1", "split": "test"},
        ]
    ).to_csv(index, index=False)
    report = mine_v2_hard_examples(index, tmp_path / "hard.csv")
    result = pd.read_csv(tmp_path / "hard.csv")
    assert result["sample_id"].tolist() == ["train:1"]
    assert report["test_images_read"] == 0

