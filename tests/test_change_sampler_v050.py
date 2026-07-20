from __future__ import annotations

import pytest


def test_weighted_sampler_is_deterministic_and_balances_dataset_size():
    pytest.importorskip("torch")
    from urban_monitor.change.samplers import DatasetBalancedSampler
    names = ["s2looking"] * 100 + ["second"] * 10 + ["dsifn"] * 5
    first = DatasetBalancedSampler(names, {"s2looking": 2, "second": 1, "dsifn": 1}, num_samples=400, seed=7)
    second = DatasetBalancedSampler(names, {"s2looking": 2, "second": 1, "dsifn": 1}, num_samples=400, seed=7)
    assert list(first) == list(second)
    counts = first.last_counts
    assert all(counts[name] > 50 for name in ("s2looking", "second", "dsifn"))
    assert counts["s2looking"] > counts["second"]


def test_round_robin_and_distributed_views_cover_every_dataset():
    pytest.importorskip("torch")
    from urban_monitor.change.samplers import DatasetBalancedSampler, DistributedSamplerView
    names = ["a", "a", "b", "c"]
    sampler = DatasetBalancedSampler(names, mode="round_robin", num_samples=12)
    values = list(sampler)
    assert set(names[index] for index in values) == {"a", "b", "c"}
    rank0 = list(DistributedSamplerView(sampler, num_replicas=2, rank=0))
    rank1 = list(DistributedSamplerView(sampler, num_replicas=2, rank=1))
    assert len(rank0) == len(rank1) == 6
