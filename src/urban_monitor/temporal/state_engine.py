from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import yaml

from .schemas import QualityState, StateSegment, TemporalObservation, TemporalState, TemporalTrackResult
from .transitions import derive_transition


class TemporalStateEngine:
    def __init__(self, config: dict):
        self.config = config
        self.threshold = float(config.get("probability_threshold", 0.55))
        self.allowed_presence = {QualityState(value) for value in config.get("quality_states_allowed_for_presence", ["usable", "unknown_quality"])}
        self.allowed_absence = {QualityState(value) for value in config.get("quality_states_allowed_for_absence", ["usable"])}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TemporalStateEngine":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def _visual_state(self, observation: TemporalObservation) -> str | None:
        if observation.quality not in self.allowed_presence:
            return None
        scores = {
            "vacant_land": observation.vacant_probability,
            "construction_site": observation.construction_probability,
            "building": observation.building_probability,
        }
        available = {key: float(value) for key, value in scores.items() if value is not None}
        if not available:
            return None
        state, value = max(available.items(), key=lambda item: item[1])
        return state if value >= self.threshold else None

    def _segments(self, observations: list[TemporalObservation]) -> list[StateSegment]:
        segments: list[StateSegment] = []
        for observation in observations:
            state = self._visual_state(observation)
            if state is None:
                continue
            if segments and segments[-1].state == state and (observation.date - segments[-1].end_date).days <= int(self.config.get("max_gap_days", 550)):
                previous = segments[-1]
                segments[-1] = StateSegment(state, previous.start_date, observation.date, previous.observation_count + 1)
            else:
                segments.append(StateSegment(state, observation.date, observation.date, 1))
        return segments

    def infer(self, track_id: str, observations: Iterable[TemporalObservation]) -> TemporalTrackResult:
        ordered = sorted(observations, key=lambda item: item.date)
        segments = self._segments(ordered)
        if not segments:
            return TemporalTrackResult(track_id, TemporalState.UNCERTAIN, 0.0, 0, notes=["No usable surface-state observations"])
        min_count = int(self.config.get("min_consecutive_observations", 2))
        min_elapsed = int(self.config.get("min_elapsed_days", 180))
        events = []
        for before, after in zip(segments, segments[1:]):
            event = derive_transition(before.state, after.state)
            if event != "uncertain_transition":
                events.append({"event": event, "from": before.state, "to": after.state, "date": after.start_date.isoformat()})
        last = segments[-1]
        stable = last.observation_count >= min_count and (last.end_date - last.start_date).days >= min_elapsed
        state = TemporalState.UNCERTAIN
        if last.state == "building" and last.observation_count >= int(self.config.get("min_stable_building_observations", 2)):
            state = TemporalState.NEWLY_COMPLETED_BUILDING if any(event["event"] in {"completion", "newly_completed_building"} for event in events[-1:]) else TemporalState.STABLE_BUILDING
        elif last.state == "vacant_land" and last.observation_count >= int(self.config.get("min_stable_vacant_observations", 2)):
            state = TemporalState.DEMOLISHED if any(event["event"] == "demolition" for event in events[-1:]) else TemporalState.STABLE_VACANT
        elif last.state == "construction_site":
            state = TemporalState.CONSTRUCTION_ACTIVE if stable else TemporalState.CONSTRUCTION_CANDIDATE
        elif len(segments) >= 2 and segments[-2].state == "building":
            state = TemporalState.DEMOLITION_CANDIDATE
        if events and events[-1]["event"] == "redevelopment_start":
            state = TemporalState.REDEVELOPMENT_CANDIDATE
        usable = sum(self._visual_state(item) is not None for item in ordered)
        confidence = min(1.0, usable / max(1, min_count + 1))
        return TemporalTrackResult(track_id, state, confidence, usable, segments, events)

    def as_dict(self, result: TemporalTrackResult) -> dict:
        payload = asdict(result)
        payload["state"] = result.state.value
        return payload
