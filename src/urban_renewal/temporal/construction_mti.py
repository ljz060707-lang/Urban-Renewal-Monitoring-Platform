from __future__ import annotations

import numpy as np


class ConstructionMTI:
    available = True
    enabled = False
    selection_status = "experimental_unselected"

    def __init__(self, state_graph, transition_weight=1.0):
        self.state_graph = state_graph
        self.transition_weight = float(transition_weight)

    def __call__(self, state_probabilities, valid_mask=None, transition_evidence=None):
        probabilities = np.asarray(state_probabilities, dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError("ConstructionMTI expects [T,states]")
        timesteps, count = probabilities.shape
        valid = np.ones(timesteps, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        scores = np.full((timesteps, count), np.inf)
        back = np.zeros((timesteps, count), dtype=np.int64)
        scores[0] = -np.log(np.clip(probabilities[0], 1e-8, 1.0))
        for time in range(1, timesteps):
            if not valid[time]:
                scores[time] = scores[time - 1]; back[time] = np.arange(count); continue
            evidence = (transition_evidence or [{}] * timesteps)[time]
            for current in range(count):
                costs = np.asarray([
                    self.state_graph.transition_cost(self.state_graph.states[previous], self.state_graph.states[current], evidence)
                    for previous in range(count)
                ])
                candidate = scores[time - 1] + self.transition_weight * costs
                back[time, current] = int(candidate.argmin())
                scores[time, current] = float(candidate.min() - np.log(np.clip(probabilities[time, current], 1e-8, 1.0)))
        sequence = np.zeros(timesteps, dtype=np.int64); sequence[-1] = int(scores[-1].argmin())
        for time in range(timesteps - 1, 0, -1):
            sequence[time - 1] = back[time, sequence[time]]
        return {"state_sequence": sequence, "state_names": [self.state_graph.states[index] for index in sequence], "path_cost": float(scores[-1, sequence[-1]]), "integration": "construction_mti_experimental", "label_source": "model_prediction"}
