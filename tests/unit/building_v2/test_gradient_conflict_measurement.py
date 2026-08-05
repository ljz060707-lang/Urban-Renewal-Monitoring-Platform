import numpy as np

from urban_monitor.building_foundation_v2.diagnostics import gradient_cosine


def test_gradient_cosine_reports_direction_and_zero_norm():
    assert gradient_cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert gradient_cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == -1.0
    assert gradient_cosine(np.zeros(2), np.ones(2)) is None

