import numpy as np

from urban_monitor.building_foundation_v2.metrics import object_metrics


def test_objectization_counts_components():
    value = object_metrics(np.array([[1, 0], [0, 1]], dtype=float), np.array([[1, 0], [0, 1]], dtype=float))
    assert value["target_object_count"] == 2
