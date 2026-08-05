from pathlib import Path

import pandas as pd
from PIL import Image

from urban_monitor.change.manual_diagnostic_review import build_manual_diagnostic_review


def test_manual_diagnostic_review_uses_existing_assets_only(tmp_path: Path):
    run = tmp_path / "20260725_232317"; review = run / "review"; review.mkdir(parents=True)
    images = tmp_path / "images"; images.mkdir()
    for name, color in (("t1.jpg", (20, 20, 20)), ("t2.jpg", (80, 80, 80)), ("gt.png", 255), ("prob.png", 220)):
        Image.new("L" if name.endswith("png") else "RGB", (8, 8), color).save(images / name)
    def row(split: str, index: int):
        return {"sample_id": f"dsifn:{split}:{index}", "dataset_name": "dsifn", "stage": "binary", "f1": index / 100,
                "iou": .2, "precision": .3, "recall": .4, "uncertainty": .5 + index / 1000,
                "false_positive_pixels": index + 1, "false_negative_pixels": index + 2,
                "t1_path": images / "t1.jpg", "t2_path": images / "t2.jpg", "gt_binary_path": images / "gt.png",
                "binary_probability_path": images / "prob.png"}
    pd.DataFrame([row("test", index) for index in range(48)]).to_csv(review / "review_samples_test.csv", index=False)
    pd.DataFrame([row("val", index) for index in range(90)]).to_csv(review / "review_samples.csv", index=False)

    result = build_manual_diagnostic_review(run)
    root = run / "manual_diagnostic_review"
    test = pd.read_csv(root / "TEST_ALL48_MANUAL_DIAGNOSTIC.csv")
    validation = pd.read_csv(root / "VALIDATION_MANUAL_REVIEW_QUEUE_80.csv")
    page = (root / "index.html").read_text(encoding="utf-8")

    assert result["test_inference_performed"] is False
    assert len(test) == 48 and len(validation) == 80
    assert not test["training_eligible"].any()
    assert not test["threshold_selection_eligible"].any()
    assert not test["model_selection_eligible"].any()
    assert "textarea" not in page.casefold()
    assert "localStorage" in page and "DSIFN_MANUAL_REVIEW_DECISIONS_20260725_232317.csv" in page
    assert result["asset_counts"] == {"t1": 128, "t2": 128, "gt": 128, "prediction": 128, "overlay": 128}
