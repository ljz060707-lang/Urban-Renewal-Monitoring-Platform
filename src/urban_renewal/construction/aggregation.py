from __future__ import annotations

import numpy as np


def aggregate_site_mask(evidence: dict[str, np.ndarray], thresholds: dict[str, float], uncertainty_limit=0.5):
    keys = ("exposed_ground_evidence", "dustproof_net_evidence", "engineering_facility_evidence", "material_debris_evidence")
    shapes = {np.asarray(evidence[key]).shape for key in keys if key in evidence}
    if len(shapes) != 1:
        raise ValueError("construction evidence arrays must share one shape")
    mask = np.zeros(next(iter(shapes)), dtype=bool)
    for key in keys:
        if key in evidence:
            mask |= np.asarray(evidence[key]) >= float(thresholds.get(key, 0.5))
    if "uncertain_evidence" in evidence:
        mask &= np.asarray(evidence["uncertain_evidence"]) < float(uncertainty_limit)
    return mask
