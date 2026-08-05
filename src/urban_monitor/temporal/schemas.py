from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class QualityState(str, Enum):
    USABLE = "usable"
    UNUSABLE = "unusable"
    OCCLUDED = "occluded"
    REGISTRATION_ISSUE = "registration_issue"
    UNKNOWN = "unknown_quality"


class TemporalState(str, Enum):
    STABLE_VACANT = "stable_vacant"
    CONSTRUCTION_CANDIDATE = "construction_candidate"
    CONSTRUCTION_ACTIVE = "construction_active"
    NEWLY_COMPLETED_BUILDING = "newly_completed_building"
    STABLE_BUILDING = "stable_building"
    DEMOLITION_CANDIDATE = "demolition_candidate"
    DEMOLISHED = "demolished"
    REDEVELOPMENT_CANDIDATE = "redevelopment_candidate"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class TemporalObservation:
    object_id: str
    date: date
    vacant_probability: float | None
    construction_probability: float | None
    building_probability: float | None
    quality: QualityState
    geometry: object | None = None
    area_m2: float | None = None
    uncertainty: float | None = None
    source: str = ""


@dataclass(frozen=True)
class StateSegment:
    state: str
    start_date: date
    end_date: date
    observation_count: int


@dataclass
class TemporalTrackResult:
    track_id: str
    state: TemporalState
    confidence: float
    observations_used: int
    segments: list[StateSegment] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
