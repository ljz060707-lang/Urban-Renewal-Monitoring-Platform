import numpy as np

from urban_monitor.building_foundation_v2.object_metrics import match_objects


def test_one_to_one_iou_matching_is_deterministic():
    truth = np.zeros((8, 8), dtype=np.uint8)
    truth[1:3, 1:3] = 1
    truth[5:7, 5:7] = 1
    result = match_objects(truth, truth, threshold=0.5)
    assert result["matched_count"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_threshold_discards_low_iou_pair():
    truth = np.zeros((8, 8), dtype=np.uint8)
    pred = np.zeros_like(truth)
    truth[1:5, 1:5] = 1
    pred[3:7, 3:7] = 1
    result = match_objects(pred, truth, threshold=0.7)
    assert result["matched_count"] == 0
    assert result["unmatched_prediction_count"] == 1
    assert result["unmatched_target_count"] == 1

