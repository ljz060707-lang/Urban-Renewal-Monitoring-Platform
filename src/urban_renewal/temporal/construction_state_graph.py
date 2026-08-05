from __future__ import annotations

import math


class ConstructionStateGraph:
    def __init__(self, config: dict):
        self.states = tuple(config["states"])
        self.default_cost = float(config.get("default_transition_cost", 4.0))
        self.costs = {(item["from"], item["to"]): float(item["cost"]) for item in config.get("transitions", [])}

    @classmethod
    def from_yaml(cls, path):
        import yaml
        from pathlib import Path

        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def transition_cost(self, previous: str, current: str, evidence=None) -> float:
        cost = self.costs.get((previous, current), 0.0 if previous == current else self.default_cost)
        evidence = evidence or {}
        if previous == "completed_building" and current == "cleared_site" and evidence.get("demolition", 0.0) < 0.5:
            cost += 8.0
        if previous == "vacant_bare_land" and current == "completed_building" and evidence.get("construction", 0.0) < 0.5:
            cost += 8.0
        return float(cost)

    def validate_sequence(self, states, evidences=None):
        evidences = evidences or [{} for _ in states]
        costs = [self.transition_cost(states[index - 1], states[index], evidences[index]) for index in range(1, len(states))]
        return {"total_cost": float(sum(costs)), "transition_costs": costs, "impossible_transition_count": sum(cost >= 8 for cost in costs)}
