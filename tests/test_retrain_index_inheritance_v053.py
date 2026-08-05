import json
from pathlib import Path

import pandas as pd

from urban_monitor.change.cli import _inherit_candidate_index, _sha256


def test_candidate_inherits_identical_baseline_index(tmp_path: Path):
    source = tmp_path / "baseline"; candidate = tmp_path / "candidate"
    (source / "prepared").mkdir(parents=True); (candidate / "prepared").mkdir(parents=True)
    frame = pd.DataFrame([
        {"sample_id": "a", "split": "train"}, {"sample_id": "b", "split": "val"}, {"sample_id": "c", "split": "test"},
    ])
    index = source / "prepared" / "public_change_index.csv"; frame.to_csv(index, index=False, encoding="utf-8-sig")
    provenance = _inherit_candidate_index(source, candidate)
    target = candidate / "prepared" / "public_change_index.csv"
    assert target.is_file() and _sha256(index) == _sha256(target) == provenance["index_sha256"]
    assert provenance["source_run_id"] == "baseline"
    assert provenance["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert pd.read_csv(index).equals(pd.read_csv(target))
