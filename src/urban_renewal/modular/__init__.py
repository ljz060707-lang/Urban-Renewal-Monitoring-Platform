"""Modular, inference-only URRS orchestration contracts.

The package deliberately contains only small immutable result contracts and
boundary helpers.  Heavy model runners remain owned by their existing model
modules; the Nanjing orchestration layer may read their sealed outputs but
does not mutate them.
"""

from .contracts import (
    AlignedSceneReference,
    BaselineChangeResult,
    BuildingGeometryResult,
    BuildingTrackResult,
    ConstructionElementResult,
    ConstructionTemporalResult,
    CrossReferenceResult,
    Full20PairResult,
    SceneRecord,
)

__all__ = [
    "AlignedSceneReference",
    "BaselineChangeResult",
    "BuildingGeometryResult",
    "BuildingTrackResult",
    "ConstructionElementResult",
    "ConstructionTemporalResult",
    "CrossReferenceResult",
    "Full20PairResult",
    "SceneRecord",
]
