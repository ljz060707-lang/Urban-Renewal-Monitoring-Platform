from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from urban_monitor.change.review import build_static_review


def test_static_review_contains_filters_local_storage_and_csv_export(tmp_path: Path):
    run = tmp_path / "run"; (run / "review").mkdir(parents=True); (run / "errors" / "false_positive").mkdir(parents=True)
    image = run / "sample.png"; Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
    mask = run / "mask.png"; Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(mask)
    preview = run / "preview.jpg"; Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(preview)
    (run / "errors" / "false_positive" / preview.name).write_bytes(preview.read_bytes())
    pd.DataFrame([{"sample_id":"second:val:x","dataset_name":"second","stage":"directional","f1":0.5,"iou":0.3,"uncertainty":0.7,
                   "t1_path":image,"t2_path":image,"binary_probability_path":mask,"new_probability_path":mask,
                   "demolition_probability_path":mask,"preview_path":preview}]).to_csv(run / "review" / "review_samples.csv", index=False)
    output = build_static_review(run)
    text = output.read_text(encoding="utf-8")
    assert "localStorage" in text and "review_decisions.csv" in text
    assert "prediction_error" in text and "needs_mask_edit" in text
    assert "threshold" in text and "false_positive" in text
