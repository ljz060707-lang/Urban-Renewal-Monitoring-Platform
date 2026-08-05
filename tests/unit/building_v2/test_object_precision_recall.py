import numpy as np

from urban_monitor.building_foundation_v2.object_metrics import match_objects


def test_object_precision_recall_counts_unmatched_objects():
    target = np.zeros((12, 12), dtype=np.uint8)
    pred = np.zeros_like(target)
    target[1:3, 1:3] = 1
    target[7:9, 7:9] = 1
    pred[1:3, 1:3] = 1
    pred[9:11, 9:11] = 1
    result = match_objects(pred, target, threshold=0.5)
    assert result["matched_count"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5

