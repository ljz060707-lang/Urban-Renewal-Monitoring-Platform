from collections import Counter

from urban_monitor.change.samplers import DatasetBalancedSampler


def test_dataset_sampler_is_deterministic_and_keeps_each_dataset():
    names = ["dsifn"] * 8 + ["s2looking"] * 4 + ["second"] * 2
    first = DatasetBalancedSampler(names, {"dsifn": 1, "s2looking": 2, "second": 1.5}, num_samples=30, seed=7)
    second = DatasetBalancedSampler(names, {"dsifn": 1, "s2looking": 2, "second": 1.5}, num_samples=30, seed=7)
    values = list(first)
    assert values == list(second)
    assert set(Counter(names[index] for index in values)) == {"dsifn", "s2looking", "second"}

