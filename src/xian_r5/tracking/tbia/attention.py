"""Handcrafted local cross-attention prototype for R5 Stage 1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .features import BuildingObservation, TrackQueryEncoder


@dataclass(frozen=True)
class TBIAConfig:
    base_radius_m: float = 24.0
    alpha_uncertainty: float = 2.0
    beta_offset: float = 1.5
    gamma_object_scale: float = 0.5
    top_k: int = 8
    min_affinity: float = 0.45
    temperature: float = 0.20
    # Search-only retrieval margin when positional uncertainty is unknown;
    # this is not stored or exposed as a measured positional sigma.
    unknown_alignment_search_buffer_m: float = 16.0


@dataclass
class MatchDecision:
    previous_id: str
    current_id: str | None
    affinity: float
    candidate_count: int
    search_radius_m: float
    relation: str
    resolver_stage: str
    evidence_source: str
    hard_negative_ids: list[str]
    best_candidate_id: str | None = None
    candidate_matches: list[tuple[str, float]] = field(default_factory=list)
    candidate_search_radii_m: dict[str, float] = field(default_factory=dict)


def _safe_cosine(left: np.ndarray, right: np.ndarray, mask_left: np.ndarray, mask_right: np.ndarray) -> float:
    mask = (mask_left > 0) & (mask_right > 0)
    if not mask.any():
        return 0.0
    a, b = left[mask], right[mask]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class TBIAPrototype:
    """A local, explainable attention analogue used before training.

    It preserves separate group affinities and uses alignment uncertainty in
    candidate retrieval.  It is not presented as a trained neural network.
    """

    def __init__(self, config: TBIAConfig | None = None, encoder: TrackQueryEncoder | None = None) -> None:
        self.config = config or TBIAConfig()
        self.encoder = encoder or TrackQueryEncoder()

    def search_radius(
        self,
        previous: BuildingObservation,
        current: BuildingObservation | None = None,
    ) -> float:
        """Return a pair-aware retrieval radius, or a one-observation envelope.

        Known endpoint sigmas combine in quadrature. If either endpoint is
        unknown, the configured value is added as a search-only floor; it is
        never written back as a measured uncertainty.
        """
        endpoints = (previous,) if current is None else (previous, current)
        known_sigmas = [item.alignment_uncertainty for item in endpoints
                        if item.alignment_uncertainty is not None]
        measured_pair_margin = self.config.alpha_uncertainty * math.sqrt(
            sum(float(sigma) ** 2 for sigma in known_sigmas))
        has_unknown_endpoint = any(item.alignment_uncertainty is None for item in endpoints)
        uncertainty_margin = (
            max(measured_pair_margin, self.config.unknown_alignment_search_buffer_m)
            if has_unknown_endpoint else measured_pair_margin
        )
        return (self.config.base_radius_m + uncertainty_margin
                + self.config.beta_offset * previous.offset_magnitude
                + self.config.gamma_object_scale * math.sqrt(max(0.0, previous.area_m2)))

    def _group_affinity(self, previous: BuildingObservation, current: BuildingObservation, current_epoch: list[BuildingObservation]) -> tuple[float, dict[str, float]]:
        q = self.encoder.encode(previous, [previous])
        k = self.encoder.encode(current, current_epoch)
        groups = q["groups"]
        groups_k = k["groups"]
        masks = q["group_masks"]
        masks_k = k["group_masks"]
        shape_sim = 1.0 / (1.0 + abs(math.log1p(previous.area_m2) - math.log1p(current.area_m2)))
        pair_radius = self.search_radius(previous, current)
        geometry_sim = 1.0 / (1.0 + math.hypot(previous.x - current.x, previous.y - current.y) / max(1.0, pair_radius))
        boundary_sim = _safe_cosine(groups["boundary"], groups_k["boundary"], masks["boundary"], masks_k["boundary"])
        offset_available = bool(masks["offset"].any() and masks_k["offset"].any())
        offset_sim = 1.0 / (1.0 + abs(previous.offset_magnitude - current.offset_magnitude)) if offset_available else 0.0
        neighborhood_sim = _safe_cosine(groups["neighborhood"], groups_k["neighborhood"], masks["neighborhood"], masks_k["neighborhood"])
        semantic_sim = _safe_cosine(groups["semantic"], groups_k["semantic"], masks["semantic"], masks_k["semantic"])
        values = {
            "semantic": max(0.0, semantic_sim),
            "geometry": geometry_sim,
            "boundary": max(0.0, boundary_sim),
            "neighborhood": max(0.0, neighborhood_sim),
            "offset": offset_sim,
            "shape": shape_sim,
        }
        weights = {"semantic": 0.10, "geometry": 0.30, "boundary": 0.15, "neighborhood": 0.15, "offset": 0.10, "shape": 0.20}
        return float(sum(values[key] * weights[key] for key in weights)), values

    def match(self, previous_epoch: Iterable[BuildingObservation], current_epoch: Iterable[BuildingObservation]) -> list[MatchDecision]:
        previous_list = list(previous_epoch)
        current_list = list(current_epoch)
        all_edges: list[tuple[str, str, float]] = []
        decisions: list[MatchDecision] = []
        for previous in previous_list:
            candidate_radii = {
                candidate.object_id: self.search_radius(previous, candidate)
                for candidate in current_list
            }
            radius = max(candidate_radii.values(), default=self.search_radius(previous))
            local = [candidate for candidate in current_list
                     if math.hypot(previous.x - candidate.x, previous.y - candidate.y)
                     <= candidate_radii[candidate.object_id]]
            scored = [(candidate, *self._group_affinity(previous, candidate, current_list)) for candidate in local]
            scored.sort(key=lambda item: item[1], reverse=True)
            hard_negatives = [item.object_id for item in sorted(local, key=lambda item: abs(math.log1p(item.area_m2) - math.log1p(previous.area_m2)))[: self.config.top_k]]
            if not scored:
                decisions.append(MatchDecision(
                    previous.object_id, None, 0.0, 0, radius, "0:0",
                    "R1_DIRECT_IDENTITY", "NO_LOCAL_CANDIDATE", hard_negatives))
                continue
            best, score, _ = scored[0]
            candidate_matches = [
                (item[0].object_id, float(item[1]))
                for item in scored if item[1] >= self.config.min_affinity
            ]
            all_edges.extend((previous.object_id, current_id, affinity)
                             for current_id, affinity in candidate_matches)
            relation = "1:1" if len(candidate_matches) == 1 else "1:N"
            decisions.append(MatchDecision(
                previous.object_id,
                best.object_id if score >= self.config.min_affinity else None,
                float(score), len(scored), radius, relation,
                "R2_TBIA_HANDCRAFTED",
                "semantic_boundary_shape_offset_neighborhood", hard_negatives,
                best_candidate_id=best.object_id,
                candidate_matches=candidate_matches,
                candidate_search_radii_m={item.object_id: candidate_radii[item.object_id]
                                           for item in local},
            ))
        incoming: dict[str, int] = {}
        for _, current_id, score in all_edges:
            if score >= self.config.min_affinity:
                incoming[current_id] = incoming.get(current_id, 0) + 1
        for decision in decisions:
            if decision.current_id and incoming.get(decision.current_id, 0) > 1:
                decision.relation = "N:1"
                decision.resolver_stage = "R2_TBIA_HANDCRAFTED_SEGMENTATION_MERGE_SUSPECT"
        return decisions
