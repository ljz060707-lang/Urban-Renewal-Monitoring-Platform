import numpy as np

from urban_monitor.building_foundation_v2.object_metrics import area_bias_metrics, size_stratified_recall


def test_area_bias_and_size_bands():
    target = np.zeros((20, 20), dtype=np.uint8)
    pred = np.zeros_like(target)
    target[1:3, 1:3] = 1
    target[10:16, 10:16] = 1
    pred[1:3, 1:3] = 1
    result = area_bias_metrics(pred, target)
    assert result["area_bias"] < 0
    bands = size_stratified_recall(pred, target, area_thresholds=(8, 20))
    assert bands["small"]["target_count"] == 1
    assert bands["large"]["target_count"] == 1
