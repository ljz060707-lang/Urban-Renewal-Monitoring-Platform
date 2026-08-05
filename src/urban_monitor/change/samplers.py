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
                     minimum_per_dataset: int = 1, example_weights: list[float] | None = None):
            if not dataset_names:
                raise ValueError("dataset_names cannot be empty")
            self.dataset_names = list(dataset_names)
            self.indices: dict[str, list[int]] = defaultdict(list)
            for index, name in enumerate(dataset_names):
                self.indices[name].append(index)
            self.weights = {name: float((weights or {}).get(name, 1.0)) for name in self.indices}
            self.example_weights = [float(value) for value in (example_weights or [1.0] * len(dataset_names))]
            if len(self.example_weights) != len(dataset_names) or any(value <= 0 for value in self.example_weights):
                raise ValueError("example_weights must be positive and match dataset_names")
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
                self.weights[name] * self.example_weights[index] / len(self.indices[name]) for index, name in enumerate(self.dataset_names)
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


class DemolitionPatchCompositionSampler:
    """Deterministic epoch composition over all train images.

    Values are ``(row_index, sampling_category)`` tuples so the Dataset can
    direct the crop without duplicating source images.
    """
    def __init__(self, categories: dict[str, list[int]], *, num_samples: int, composition: dict[str, float], seed: int = 42):
        import torch
        self.torch = torch
        self.categories = {name: list(values) for name, values in categories.items() if values}
        self.num_samples, self.composition, self.seed, self.epoch = int(num_samples), dict(composition), int(seed), 0
        if "ordinary" not in self.categories:
            raise ValueError("ordinary sampling pool cannot be empty")

    def set_epoch(self, epoch: int): self.epoch = int(epoch)
    def __len__(self): return self.num_samples

    def __iter__(self):
        generator = self.torch.Generator().manual_seed(self.seed + self.epoch)
        allocated, output = 0, []
        ordered = list(self.composition)
        for position, category in enumerate(ordered):
            pool = self.categories.get(category) or self.categories["ordinary"]
            count = self.num_samples - allocated if position == len(ordered) - 1 else int(round(self.num_samples * float(self.composition[category])))
            allocated += count
            choices = self.torch.randint(len(pool), (count,), generator=generator).tolist()
            output.extend((pool[choice], category) for choice in choices)
        order = self.torch.randperm(len(output), generator=generator).tolist()
        return iter(output[index] for index in order)


# Backward-compatible name: the same deterministic tuple-index mechanism is
# used for V2 directional strata without manufacturing labels or masks.
DirectionalPatchCompositionSampler = DemolitionPatchCompositionSampler


class V2DirectionalSampler:
    """Preserve source-level weights, then stratify only S2Looking draws."""
    def __init__(self, dataset_names, *, weights, new_positive, demolition_positive,
                 num_samples, composition, seed=42):
        import torch
        self.torch, self.names = torch, list(dataset_names)
        self.by_dataset = defaultdict(list)
        for index, name in enumerate(self.names): self.by_dataset[name].append(index)
        self.weights = {name: float(weights.get(name, 1.0)) for name in self.by_dataset}
        self.new, self.demolition = list(new_positive), list(demolition_positive)
        invalid_new = [
            index
            for index in self.new
            if self.names[index].casefold() != "s2looking"
        ]

        invalid_demolition = [
            index
            for index in self.demolition
            if self.names[index].casefold() != "s2looking"
        ]

        if invalid_new or invalid_demolition:
            raise ValueError(
                "V2DirectionalSampler directional pools must contain "
                "S2Looking samples only: "
                f"invalid_new={len(invalid_new)}, "
                f"invalid_demolition={len(invalid_demolition)}"
            )
        self.num_samples, self.composition, self.seed, self.epoch = int(num_samples), dict(composition), int(seed), 0
        self.last_counts = {}
        if not self.demolition or not self.new:
            raise ValueError("directional_composition requires real S2Looking new and demolition samples")

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return self.num_samples

    def __iter__(self):
        generator = self.torch.Generator().manual_seed(self.seed + self.epoch)
        total_weight = sum(self.weights.values())
        quotas = {name: int(round(self.num_samples * weight / total_weight)) for name, weight in self.weights.items()}
        # deterministic rounding reconciliation goes to the highest-weight source.
        quotas["s2looking"] = quotas.get("s2looking", 0) + self.num_samples - sum(quotas.values())
        output = []
        for name, count in quotas.items():
            if name == "s2looking":
                demo_count = int(round(count * float(self.composition.get("demolition_positive", .2))))
                new_count = int(round(count * float(self.composition.get("new_positive", .4))))
                background_count = max(0, count - demo_count - new_count)
                for pool, category, amount in ((self.demolition, "demolition_positive", demo_count), (self.new, "new_positive", new_count), (self.by_dataset[name], "ordinary", background_count)):
                    picked = self.torch.randint(len(pool), (amount,), generator=generator).tolist()
                    output.extend((pool[item], category) for item in picked)
            else:
                pool = self.by_dataset[name]
                picked = self.torch.randint(len(pool), (count,), generator=generator).tolist()
                output.extend((pool[item], "ordinary") for item in picked)
        order = self.torch.randperm(len(output), generator=generator).tolist()
        result = [output[index] for index in order]
        self.last_counts = dict(Counter(self.names[index] for index, _ in result))
        return iter(result)
