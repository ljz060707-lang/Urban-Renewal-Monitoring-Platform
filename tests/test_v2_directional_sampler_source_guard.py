import pytest


def test_v2_directional_sampler_rejects_non_s2looking_directional_pool():
    torch = pytest.importorskip("torch")

    from urban_monitor.change.samplers import V2DirectionalSampler

    dataset_names = [
        "dsifn",
        "s2looking",
        "s2looking",
        "second",
    ]

    # index=3 belongs to SECOND and must never enter
    # an S2Looking directional pool.
    with pytest.raises(ValueError):
        V2DirectionalSampler(
            dataset_names,
            weights={
                "dsifn": 1.0,
                "s2looking": 2.0,
                "second": 1.5,
            },
            new_positive=[1, 3],
            demolition_positive=[2],
            num_samples=16,
            composition={
                "new_positive": 0.4,
                "demolition_positive": 0.2,
            },
            seed=42,
        )