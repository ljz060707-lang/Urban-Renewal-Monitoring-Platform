import numpy as np

from urban_monitor.building_foundation_v2.boundary import boundary_metrics, boundary_target
from urban_monitor.building_foundation_v2.diagnostics import shift


def test_controlled_shift_degrades_zero_tolerance_more_than_tolerant_match():
    mask = np.zeros((64, 64), dtype=np.uint8); mask[16:48, 16:48] = 1
    target = boundary_target(mask, width=2) > 0
    metrics = boundary_metrics(shift(target, 2, 2), target)
    assert metrics["boundary_f1@1"] <= metrics["boundary_f1@2"]
    assert np.isfinite(metrics["hausdorff95"])

