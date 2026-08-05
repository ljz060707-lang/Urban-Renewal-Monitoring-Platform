from __future__ import annotations


def detect_split_merge(previous, current, overlap_threshold=0.2):
    events = []
    for p_index, p in enumerate(previous):
        children = [c_index for c_index, c in enumerate(current) if p["geometry"].intersection(c["geometry"]).area / max(c["geometry"].area, 1e-9) >= overlap_threshold]
        if len(children) > 1:
            events.append({"event_type": "split", "parent_track_ids": [p.get("track_id", str(p_index))], "child_track_ids": [current[index].get("track_id", str(index)) for index in children]})
    for c_index, c in enumerate(current):
        parents = [p_index for p_index, p in enumerate(previous) if p["geometry"].intersection(c["geometry"]).area / max(p["geometry"].area, 1e-9) >= overlap_threshold]
        if len(parents) > 1:
            events.append({"event_type": "merge", "parent_track_ids": [previous[index].get("track_id", str(index)) for index in parents], "child_track_ids": [c.get("track_id", str(c_index))]})
    return events
