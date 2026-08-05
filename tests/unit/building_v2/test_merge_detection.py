import numpy as np

from urban_monitor.building_foundation_v2.object_metrics import match_objects


def test_merge_event_is_reported():
    target = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros_like(target)
    target[2:5, 2:5] = 1
    target[6:9, 6:9] = 1
    pred[2:9, 2:9] = 1
    result = match_objects(pred, target, threshold=0.15)
    assert result["merge_count"] == 1
