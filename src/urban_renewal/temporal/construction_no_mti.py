from __future__ import annotations

import numpy as np


class ConstructionNoMTI:
    selected = True
    selection_status = "default_until_human_temporal_validation"

    def __call__(self, state_probabilities, valid_mask=None):
        probabilities = np.asarray(state_probabilities, dtype=np.float64)
        states = probabilities.argmax(-1)
        confidence = probabilities.max(-1)
        if valid_mask is not None:
            states = np.where(valid_mask, states, -1)
            confidence = np.where(valid_mask, confidence, 0.0)
        return {"state_sequence": states, "confidence": confidence, "integration": "no_mti", "label_source": "model_prediction"}
