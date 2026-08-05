from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SceneRecord:
    scene_date: str
    source_path: str
    source_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    crs: str | None = None
    transform: tuple[float, ...] | None = None
    nodata: float | None = None
    georeferenced: bool = False


@dataclass
class AlignmentReport:
    status: str
    scenes: list[dict[str, Any]] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)


@dataclass
class PipelineReceipt:
    stage: str
    status: str
    created_at: str
    outputs: dict[str, Any] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)


@dataclass
class SystemState:
    current_stage: str
    completed_gates: list[str]
    blocked_gates: list[str]
    blocking_reasons: list[str]
    next_allowed_actions: list[str]
    prohibited_actions: list[str]
    inference_ready: bool = False
