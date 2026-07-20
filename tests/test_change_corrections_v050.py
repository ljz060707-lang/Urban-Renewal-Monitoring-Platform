from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from urban_monitor.change.corrections import apply_review_corrections, save_correction


def test_human_correction_overrides_original_and_exclusions_are_removed(tmp_path: Path):
    run = tmp_path / "run"; (run / "review").mkdir(parents=True)
    original = tmp_path / "original.png"; corrected = tmp_path / "corrected.png"
    Image.fromarray(np.zeros((4,4), dtype=np.uint8)).save(original); Image.fromarray(np.ones((4,4), dtype=np.uint8)).save(corrected)
    index = tmp_path / "index.csv"
    pd.DataFrame([
        {"sample_id":"a","gt_cd_path":original,"gt_new_path":"","gt_dem_path":""},
        {"sample_id":"b","gt_cd_path":original,"gt_new_path":"","gt_dem_path":""},
        {"sample_id":"c","gt_cd_path":original,"gt_new_path":"","gt_dem_path":""},
    ]).to_csv(index, index=False)
    saved = save_correction(corrected, run, "a", "binary", reviewer="tester")
    decisions = run / "review" / "review_decisions.csv"
    pd.DataFrame([{"sample_id":"a","decision":"label_error","notes":"已修正"},
                  {"sample_id":"b","decision":"exclude","notes":""},
                  {"sample_id":"c","decision":"prediction_error","notes":"hard"}]).to_csv(decisions, index=False)
    output = apply_review_corrections(index, decisions, run)
    frame = pd.read_csv(output, dtype=str).fillna("")
    assert frame["sample_id"].tolist() == ["a", "c"]
    assert Path(frame.loc[frame["sample_id"]=="a", "gt_cd_path"].iloc[0]) == saved
    assert frame.loc[frame["sample_id"]=="a", "binary_label_source"].iloc[0] == "human_corrected_label"
    assert frame.loc[frame["sample_id"]=="c", "sample_weight"].iloc[0] == "2.0"
