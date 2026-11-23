"""Track-query and neighborhood features for the R5 no-training prototype."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


def _float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _optional_nonnegative_float(value: Any) -> float | None:
    """Parse measured uncertainty without turning missing/invalid values into zero."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


@dataclass
class BuildingObservation:
    object_id: str
    epoch: str
    x: float
    y: float
    area_m2: float
    perimeter_m: float = 0.0
    compactness: float = 0.0
    elongation: float = 0.0
    orientation: float = 0.0
    convexity: float = 0.0
    rectangularity: float = 0.0
    semantic: tuple[float, ...] = field(default_factory=tuple)
    boundary: tuple[float, ...] = field(default_factory=tuple)
    alignment_dx: float = 0.0
    alignment_dy: float = 0.0
    alignment_uncertainty: float | None = None
    roof_dx: float = 0.0
    roof_dy: float = 0.0
    offset_confidence: float = 0.0
    previous_identity_score: float = 0.0
    track_age: float = 0.0
    previous_match_type: str = "NONE"
    gap_history: float = 0.0
    split_merge_diagnostic: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, epoch: str | None = None) -> "BuildingObservation":
        semantic = row.get("semantic", ()) or ()
        boundary = row.get("boundary", ()) or ()
        if isinstance(semantic, str):
            semantic = tuple(_float(x) for x in semantic.split(",") if x.strip())
        if isinstance(boundary, str):
            boundary = tuple(_float(x) for x in boundary.split(",") if x.strip())
        return cls(
            object_id=str(row.get("object_id") or row.get("id") or row.get("object_uid") or ""),
            epoch=str(epoch or row.get("epoch") or row.get("scene_key") or ""),
            x=_float(row.get("x", row.get("cx", row.get("centroid_x")))),
            y=_float(row.get("y", row.get("cy", row.get("centroid_y")))),
            area_m2=max(0.0, _float(row.get("area_m2"))),
            perimeter_m=max(0.0, _float(row.get("perimeter_m", row.get("perimeter")))),
            compactness=_float(row.get("compactness")),
            elongation=_float(row.get("elongation")),
            orientation=_float(row.get("orientation")),
            convexity=_float(row.get("convexity")),
            rectangularity=_float(row.get("rectangularity")),
            semantic=tuple(semantic),
            boundary=tuple(boundary),
            alignment_dx=_float(row.get("alignment_dx", row.get("local_dx"))),
            alignment_dy=_float(row.get("alignment_dy", row.get("local_dy"))),
            alignment_uncertainty=_optional_nonnegative_float(
                row.get("alignment_uncertainty", row.get("sigma_alignment"))),
            roof_dx=_float(row.get("roof_dx")),
            roof_dy=_float(row.get("roof_dy")),
            offset_confidence=max(0.0, min(1.0, _float(row.get("offset_confidence")))),
            previous_identity_score=max(0.0, min(1.0, _float(row.get("previous_identity_score")))),
            track_age=max(0.0, _float(row.get("track_age"))),
            previous_match_type=str(row.get("previous_match_type") or "NONE"),
            gap_history=max(0.0, _float(row.get("gap_history"))),
            split_merge_diagnostic=_float(row.get("split_merge_diagnostic")),
            metadata=dict(row),
        )

    @property
    def offset_magnitude(self) -> float:
        return float(math.hypot(self.roof_dx, self.roof_dy))


class TrackQueryEncoder:
    """Encodes the six requested feature groups into a fixed-size vector.

    Semantic and boundary channels are empty unless a frozen feature provider
    supplies them; zero-filled channels are retained and reported through the
    feature mask instead of being mistaken for measured semantics.
    """

    group_names = ("semantic", "boundary", "shape", "offset", "neighborhood", "temporal")

    def __init__(self, nearest_n: int = 8, semantic_dim: int = 8, boundary_dim: int = 8) -> None:
        self.nearest_n = nearest_n
        self.semantic_dim = semantic_dim
        self.boundary_dim = boundary_dim

    def _pad(self, values: Iterable[float], size: int) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(list(values), dtype=np.float32)
        out = np.zeros(size, dtype=np.float32)
        mask = np.zeros(size, dtype=np.float32)
        length = min(size, len(raw))
        if length:
            out[:length] = raw[:length]
            mask[:length] = 1.0
        return out, mask

    def _shape(self, obs: BuildingObservation) -> tuple[np.ndarray, np.ndarray]:
        values = [
            math.log1p(obs.area_m2),
            math.log1p(obs.perimeter_m),
            obs.compactness,
            obs.elongation,
            math.sin(obs.orientation),
            math.cos(obs.orientation),
            obs.convexity,
            obs.rectangularity,
        ]
        return np.asarray(values, dtype=np.float32), np.ones(8, dtype=np.float32)

    def _neighborhood(self, obs: BuildingObservation, current: list[BuildingObservation]) -> tuple[np.ndarray, np.ndarray]:
        neighbors = [item for item in current if item.object_id != obs.object_id]
        neighbors.sort(key=lambda item: math.hypot(item.x - obs.x, item.y - obs.y))
        values: list[float] = []
        mask: list[float] = []
        for item in neighbors[: self.nearest_n]:
            dx, dy = item.x - obs.x, item.y - obs.y
            distance = math.hypot(dx, dy)
            values.extend([
                math.log1p(distance),
                math.atan2(dy, dx),
                math.log1p(max(0.0, item.area_m2)) - math.log1p(max(0.0, obs.area_m2)),
                float(item.area_m2 <= obs.area_m2),
            ])
            mask.extend([1.0] * 4)
        size = self.nearest_n * 4
        out = np.zeros(size, dtype=np.float32)
        out_mask = np.zeros(size, dtype=np.float32)
        length = min(size, len(values))
        if length:
            out[:length] = values[:length]
            out_mask[:length] = mask[:length]
        return out, out_mask

    def encode(self, obs: BuildingObservation, current: Iterable[BuildingObservation]) -> dict[str, Any]:
        current_list = list(current)
        semantic, semantic_mask = self._pad(obs.semantic, self.semantic_dim)
        boundary, boundary_mask = self._pad(obs.boundary, self.boundary_dim)
        shape, shape_mask = self._shape(obs)
        uncertainty = (obs.alignment_uncertainty
                       if obs.alignment_uncertainty is not None else 0.0)
        offset_values = [obs.alignment_dx, obs.alignment_dy, uncertainty,
                         obs.roof_dx, obs.roof_dy, obs.offset_confidence]
        offset = np.asarray(offset_values, dtype=np.float32)
        offset_fields = ("alignment_dx", "alignment_dy", "alignment_uncertainty", "roof_dx", "roof_dy", "offset_confidence")
        offset_mask = np.asarray([
            float(
                obs.alignment_uncertainty is not None
                if field == "alignment_uncertainty"
                else ((field in obs.metadata and obs.metadata[field] is not None)
                      or abs(value) > 0.0)
            )
            for field, value in zip(offset_fields, offset_values)
        ], dtype=np.float32)
        neighborhood, neighborhood_mask = self._neighborhood(obs, current_list)
        temporal_values = [
            obs.previous_identity_score,
            obs.track_age,
            float(obs.previous_match_type == "1:1"),
            float(obs.previous_match_type == "MERGE"),
            obs.gap_history,
            obs.split_merge_diagnostic,
        ]
        temporal = np.asarray(temporal_values, dtype=np.float32)
        temporal_mask = np.asarray([
            float(
                field in obs.metadata
                or (field == "previous_match_type" and obs.previous_match_type != "NONE")
                or (field != "previous_match_type" and abs(value) > 0.0)
            )
            for field, value in (
                ("previous_identity_score", temporal_values[0]),
                ("track_age", temporal_values[1]),
                ("previous_match_type", temporal_values[2] + temporal_values[3]),
                ("gap_history", temporal_values[4]),
                ("split_merge_diagnostic", temporal_values[5]),
            )
        ], dtype=np.float32)
        values = np.concatenate([semantic, boundary, shape, offset, neighborhood, temporal])
        feature_mask = np.concatenate([semantic_mask, boundary_mask, shape_mask, offset_mask, neighborhood_mask, temporal_mask])
        group_masks = {
            "semantic": semantic_mask,
            "boundary": boundary_mask,
            "shape": shape_mask,
            "offset": offset_mask,
            "neighborhood": neighborhood_mask,
            "temporal": temporal_mask,
        }
        return {
            "object_id": obs.object_id,
            "vector": values,
            "feature_mask": feature_mask,
            "groups": {
                "semantic": semantic,
                "boundary": boundary,
                "shape": shape,
                "offset": offset,
                "neighborhood": neighborhood,
                "temporal": temporal,
            },
            "group_masks": group_masks,
            "groups_available": {name: bool(mask.any()) for name, mask in group_masks.items()},
        }
