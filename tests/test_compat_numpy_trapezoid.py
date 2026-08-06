import numpy as np

from urban_monitor.compat import trapezoid_integral


def test_trapezoid_integral_matches_expected_area():
    x = np.array([0.0, 1.0, 2.0])
    y = x * x
    reference = getattr(np, "trapezoid", None)
    if reference is None:
        reference = getattr(np, "trapz", None)
    expected = float(reference(y, x)) if reference is not None else 2.5
    assert trapezoid_integral(y, x) == expected
