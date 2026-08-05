import numpy as np

from urban_monitor.building_foundation_v2.object_metrics import match_objects


def test_split_event_is_reported():
    target = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros_like(target)
    target[2:8, 2:8] = 1
    pred[2:8, 2:4] = 1
    pred[2:8, 5:8] = 1
    result = match_objects(pred, target, threshold=0.2)
    assert result["split_count"] == 1
