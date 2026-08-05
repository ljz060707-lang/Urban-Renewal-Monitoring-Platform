import numpy as np

from urban_monitor.building_foundation_v2.boundary import boundary_metrics, bootstrap_ci


def test_boundary_f1_tolerance_and_distance_are_finite_for_matching_edges():
    target = np.zeros((32, 32), dtype=bool)
    target[8:24, 8] = True
    target[8:24, 23] = True
    prediction = np.zeros_like(target)
    prediction[8:24, 10] = True
    prediction[8:24, 21] = True
    metrics = boundary_metrics(prediction, target)
    assert metrics["boundary_f1@2"] > metrics["boundary_f1@1"]
    assert np.isfinite(metrics["mean_boundary_distance"])
    assert np.isfinite(metrics["hausdorff95"])


def test_bootstrap_ci_is_deterministic():
    first = bootstrap_ci([0.1, 0.2, 0.3], seed=7, n_bootstrap=100)
    second = bootstrap_ci([0.1, 0.2, 0.3], seed=7, n_bootstrap=100)
    assert first == second
