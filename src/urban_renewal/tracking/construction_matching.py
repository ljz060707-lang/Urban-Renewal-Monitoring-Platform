from __future__ import annotations

import math


def _iou(left, right):
    union = left.union(right).area
    return float(left.intersection(right).area / union) if union else 0.0


def match_construction_objects(previous, current, max_centroid_distance=100.0):
    """Spatial-index candidates followed by Hungarian one-to-one assignment."""
    from scipy.optimize import linear_sum_assignment
    from shapely.strtree import STRtree
    import numpy as np

    previous = list(previous); current = list(current)
    if not previous or not current:
        return {"matches": [], "unmatched_previous": list(range(len(previous))), "unmatched_current": list(range(len(current))), "candidate_pairs": 0}
    tree = STRtree([item["geometry"] for item in current])
    candidate_costs = {}
    for left_index, left in enumerate(previous):
        query = left["geometry"].buffer(max_centroid_distance).envelope
        for right_index in tree.query(query):
            right_index = int(right_index); right = current[right_index]
            distance = left["geometry"].centroid.distance(right["geometry"].centroid)
            if distance > max_centroid_distance:
                continue
            overlap = _iou(left["geometry"], right["geometry"])
            area_ratio = min(left["geometry"].area, right["geometry"].area) / max(left["geometry"].area, right["geometry"].area, 1e-9)
            type_similarity = float(left.get("scene_type") == right.get("scene_type"))
            score = 0.55 * overlap + 0.20 * area_ratio + 0.15 * (1.0 - distance / max_centroid_distance) + 0.10 * type_similarity
            candidate_costs[(left_index, right_index)] = 1.0 - score
    cost = np.full((len(previous), len(current)), 1e6)
    for pair, value in candidate_costs.items(): cost[pair] = value
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for left_index, right_index in zip(rows.tolist(), cols.tolist()):
        if cost[left_index, right_index] < 0.95:
            matches.append({"previous_index": left_index, "current_index": right_index, "match_score": float(1.0 - cost[left_index, right_index])})
    used_left = {item["previous_index"] for item in matches}; used_right = {item["current_index"] for item in matches}
    return {"matches": matches, "unmatched_previous": [i for i in range(len(previous)) if i not in used_left], "unmatched_current": [i for i in range(len(current)) if i not in used_right], "candidate_pairs": len(candidate_costs)}
