from pathlib import Path

import pandas as pd
from PIL import Image

from urban_monitor.change.hard_examples import build_hard_examples
from urban_monitor.change.review import build_static_review
from urban_monitor.change.thresholds import _metrics
from urban_monitor.change.train_review import _rank


def test_threshold_metrics_select_expected_threshold():
    import numpy as np
    low = _metrics(np.array([.2, .6, .9]), np.array([0, 1, 1]), .5)
    high = _metrics(np.array([.2, .6, .9]), np.array([0, 1, 1]), .8)
    assert low["f1"] > high["f1"]


def test_static_review_uses_offscreen_overlay_and_keeps_gt(tmp_path: Path):
    run = tmp_path / "run"; (run / "review").mkdir(parents=True); (run / "prepared").mkdir()
    for name in ("t1", "t2", "gt", "binary", "new", "dem"):
        Image.new("L" if name in {"gt", "binary", "new", "dem"} else "RGB", (8, 8), 120).save(tmp_path / f"{name}.png")
    pd.DataFrame([{"sample_id": "x", "t1_path": str(tmp_path / "t1.png"), "t2_path": str(tmp_path / "t2.png"),
                   "binary_probability_path": str(tmp_path / "binary.png"), "new_probability_path": str(tmp_path / "new.png"),
                   "demolition_probability_path": str(tmp_path / "dem.png")}]).to_csv(run / "review" / "review_samples.csv", index=False)
    pd.DataFrame([{"sample_id": "x", "gt_cd_path": str(tmp_path / "gt.png"), "gt_new_path": str(tmp_path / "gt.png"), "gt_demolition_path": str(tmp_path / "gt.png")}]).to_csv(run / "prepared" / "public_change_index.csv", index=False)
    page = build_static_review(run).read_text(encoding="utf-8")
    assert "ctx.drawImage(base" in page and "ctx.drawImage(off" in page
    assert "ctx.putImageData(out" not in page
    assert "gt.png" in page


def test_hard_examples_never_promotes_validation_or_test(tmp_path: Path):
    run = tmp_path / "run"; (run / "review").mkdir(parents=True); (run / "prepared").mkdir()
    pd.DataFrame([{"sample_id": "val", "split": "val", "has_binary": "true"}, {"sample_id": "train", "split": "train", "has_binary": "true"}, {"sample_id": "test", "split": "test", "has_binary": "true"}]).to_csv(run / "prepared" / "public_change_index.csv", index=False)
    pd.DataFrame([{"sample_id": "val", "recall": "0", "gt_positive_ratio": "0.1", "false_negative_pixels": "9"}]).to_csv(run / "review" / "review_samples.csv", index=False)
    decisions = tmp_path / "decisions.csv"
    pd.DataFrame([{"sample_id": "val", "decision": "prediction_error", "notes": "false_negative"}, {"sample_id": "test", "decision": "prediction_error", "notes": "false_negative"}, {"sample_id": "train", "decision": "prediction_error", "notes": "false_negative"}]).to_csv(decisions, index=False)
    build_hard_examples(run, decisions)
    hard = pd.read_csv(run / "review" / "hard_examples.csv")
    assert set(hard.loc[hard["sample_id"].isin({"val", "test"}), "sample_id"]) == {"val"}
    weighted = pd.read_csv(run / "prepared" / "public_change_index.hard.csv")
    assert float(weighted.loc[weighted.sample_id == "train", "sample_weight"].iloc[0]) == 3.0


def test_hard_rank_and_max_samples_are_deterministic():
    frame = pd.DataFrame([
        {"sample_id": "ordinary", "gt_positive_ratio": .1, "prediction_positive_ratio": .1, "recall": .01, "false_negative_pixels": 20, "uncertainty": .2, "new_false_negative_pixels": 0, "demolition_false_negative_pixels": 0},
        {"sample_id": "empty", "gt_positive_ratio": .1, "prediction_positive_ratio": 0, "recall": .2, "false_negative_pixels": 10, "uncertainty": .1, "new_false_negative_pixels": 0, "demolition_false_negative_pixels": 0},
        {"sample_id": "third", "gt_positive_ratio": .1, "prediction_positive_ratio": .2, "recall": .3, "false_negative_pixels": 1, "uncertainty": .9, "new_false_negative_pixels": 9, "demolition_false_negative_pixels": 0},
    ])
    selected = _rank(frame, "hard", 2)
    assert list(selected.sample_id) == ["empty", "ordinary"]
    assert len(_rank(frame, "random", 2)) == 2
