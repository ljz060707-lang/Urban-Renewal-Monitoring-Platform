"""Object-level candidate matching with spatial indexing and explicit lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TrackedObject:
    object_id: str
    date: str
    geometry: object
    state: str
    state_probability: float
    area_m2: float
    change_type: str
    quality: str
    uncertainty: float


@dataclass(frozen=True)
class LineageEdge:
    parent_object_id: str
    child_object_id: str
    relation: str
    iou: float
    centroid_distance: float
    area_ratio: float


def _iou(left, right) -> float:
    intersection = left.intersection(right).area
    union = left.union(right).area
    return float(intersection / union) if union else 0.0


def match_objects(previous: Iterable[TrackedObject], current: Iterable[TrackedObject], *, search_distance: float = 20.0, min_iou: float = 0.05) -> list[LineageEdge]:
    from shapely.strtree import STRtree

    left = list(previous); right = list(current)
    if not left or not right:
        return []
    tree = STRtree([item.geometry for item in left])
    candidates: list[LineageEdge] = []
    parents_by_child: dict[str, int] = {}
    children_by_parent: dict[str, int] = {}
    for child in right:
        for index in tree.query(child.geometry.buffer(search_distance)):
            parent = left[int(index)]
            distance = float(parent.geometry.centroid.distance(child.geometry.centroid))
            overlap = _iou(parent.geometry, child.geometry)
            if distance > search_distance or overlap < min_iou:
                continue
            ratio = max(parent.area_m2, child.area_m2) / max(1e-9, min(parent.area_m2, child.area_m2))
            candidates.append(LineageEdge(parent.object_id, child.object_id, "one_to_one", overlap, distance, ratio))
            parents_by_child[child.object_id] = parents_by_child.get(child.object_id, 0) + 1
            children_by_parent[parent.object_id] = children_by_parent.get(parent.object_id, 0) + 1
    output = []
    for edge in candidates:
        relation = "many_to_one" if parents_by_child[edge.child_object_id] > 1 else ("one_to_many" if children_by_parent[edge.parent_object_id] > 1 else "one_to_one")
        output.append(LineageEdge(edge.parent_object_id, edge.child_object_id, relation, edge.iou, edge.centroid_distance, edge.area_ratio))
    return output
