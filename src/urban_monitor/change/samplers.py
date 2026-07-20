from __future__ import annotations

from collections import Counter, defaultdict
import math


def _torch():
    import torch
    from torch.utils.data import Sampler
    return torch, Sampler


class DatasetBalancedSamplerBase:
    pass


def _build_sampler_classes():
    torch, Sampler = _torch()

    class DatasetBalancedSampler(Sampler[int]):
        def __init__(self, dataset_names: list[str], weights: dict[str, float] | None = None, *,
                     num_samples: int | None = None, seed: int = 42, mode: str = "weighted",
                     minimum_per_dataset: int = 1):
            if not dataset_names:
                raise ValueError("dataset_names cannot be empty")
            self.dataset_names = list(dataset_names)
            self.indices: dict[str, list[int]] = defaultdict(list)
            for index, name in enumerate(dataset_names):
                self.indices[name].append(index)
            self.weights = {name: float((weights or {}).get(name, 1.0)) for name in self.indices}
            if any(value <= 0 for value in self.weights.values()):
                raise ValueError("dataset sampling weights must be positive")
            self.num_samples = int(num_samples or len(dataset_names))
            self.seed, self.epoch, self.mode = int(seed), 0, mode
            self.minimum_per_dataset = int(minimum_per_dataset)
            if self.num_samples < self.minimum_per_dataset * len(self.indices):
                raise ValueError("num_samples is too small to satisfy minimum_per_dataset")
            if mode not in {"weighted", "round_robin"}:
                raise ValueError(f"Unknown sampler mode: {mode}")
            self.last_counts: dict[str, int] = {}

        def set_epoch(self, epoch: int):
            self.epoch = int(epoch)

        def __len__(self):
            return self.num_samples

        def _weighted(self, generator):
            sample_weights = torch.tensor([
                self.weights[name] / len(self.indices[name]) for name in self.dataset_names
            ], dtype=torch.double)
            guaranteed = []
            for name in sorted(self.indices):
                choices = torch.randint(len(self.indices[name]), (self.minimum_per_dataset,), generator=generator).tolist()
                guaranteed.extend(self.indices[name][choice] for choice in choices)
            remaining = self.num_samples - len(guaranteed)
            drawn = (
                torch.multinomial(sample_weights, remaining, replacement=True, generator=generator).tolist()
                if remaining > 0 else []
            )
            result = guaranteed + drawn
            order = torch.randperm(len(result), generator=generator).tolist()
            return [result[index] for index in order]

        def _round_robin(self, generator):
            names = sorted(self.indices)
            pools = {}
            for name in names:
                order = torch.randperm(len(self.indices[name]), generator=generator).tolist()
                pools[name] = [self.indices[name][index] for index in order]
            cursors = Counter()
            result = []
            while len(result) < self.num_samples:
                for name in names:
                    repeats = max(1, round(self.weights[name]))
                    for _ in range(repeats):
                        if len(result) >= self.num_samples:
                            break
                        pool = pools[name]
                        result.append(pool[cursors[name] % len(pool)])
                        cursors[name] += 1
            return result

        def __iter__(self):
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            result = self._weighted(generator) if self.mode == "weighted" else self._round_robin(generator)
            self.last_counts = dict(Counter(self.dataset_names[index] for index in result))
            return iter(result)

    class DistributedSamplerView(Sampler[int]):
        """Rank-local deterministic view over the balanced global sample stream."""

        def __init__(self, sampler: DatasetBalancedSampler, *, num_replicas: int, rank: int):
            if rank < 0 or rank >= num_replicas:
                raise ValueError("invalid distributed rank")
            self.sampler, self.num_replicas, self.rank = sampler, int(num_replicas), int(rank)
            self.num_samples = math.ceil(len(sampler) / self.num_replicas)

        def set_epoch(self, epoch: int):
            self.sampler.set_epoch(epoch)

        def __len__(self):
            return self.num_samples

        def __iter__(self):
            values = list(iter(self.sampler))
            total = self.num_samples * self.num_replicas
            values += values[: total - len(values)]
            return iter(values[self.rank:total:self.num_replicas])

    return DatasetBalancedSampler, DistributedSamplerView


DatasetBalancedSampler, DistributedSamplerView = _build_sampler_classes()
