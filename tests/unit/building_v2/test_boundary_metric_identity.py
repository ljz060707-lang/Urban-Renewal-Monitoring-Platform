import numpy as np

from urban_monitor.building_foundation_v2.diagnostics import target_metric_self_test


def test_boundary_target_metric_identity_is_exact():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    result = target_metric_self_test(mask)
    assert result["target_vs_self_all_one"] is True
    assert all(abs(value - 1.0) < 1e-12 for value in result["target_vs_self_bf1"].values())

