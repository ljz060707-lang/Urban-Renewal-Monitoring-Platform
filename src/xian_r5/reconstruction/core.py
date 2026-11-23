"""Small public contract for date, geometry, identity and presence evidence.

All geometries are in EPSG:32649.  A rooftop is *never* a ground footprint by
default.  U means unresolved, not observed absence.  IDs in candidate links
are never exported as confirmed physical IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import math
import re
from typing import Any, Iterable

from shapely.affinity import translate
from shapely.geometry.base import BaseGeometry

from xian_r5.resolver.merged_mask_resolver import resolve_merged_mask
from xian_r5.tracking.tbia.attention import TBIAPrototype
from xian_r5.tracking.tbia.features import BuildingObservation


@dataclass(frozen=True)
class FootprintEvidence:
    geometry: BaseGeometry
    source_ref: str
    method: str
    independent_ground_evidence: bool
    reviewed: bool
    positional_sigma_m: float | None
    roof_to_ground_dx_m: float | None = None
    roof_to_ground_dy_m: float | None = None
    crs: str = "EPSG:32649"
    reviewer_id: str | None = None
    reviewed_at: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    evidence_integrity_verified: bool = False

    @property
    def accepted(self) -> bool:
        reviewed_at_valid = False
        if isinstance(self.reviewed_at, str) and self.reviewed_at:
            try:
                reviewed_at = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
                reviewed_at_valid = reviewed_at.tzinfo is not None and reviewed_at.utcoffset() is not None
            except ValueError:
                reviewed_at_valid = False
        evidence_hash_valid = bool(
            isinstance(self.evidence_sha256, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", self.evidence_sha256)
        )
        sigma_valid = bool(
            isinstance(self.positional_sigma_m, (int, float))
            and not isinstance(self.positional_sigma_m, bool)
            and math.isfinite(self.positional_sigma_m)
            and 0 <= self.positional_sigma_m <= 2.0
        )
        return (
            self.independent_ground_evidence is True
            and self.reviewed is True
            and isinstance(self.source_ref, str) and bool(self.source_ref.strip())
            and isinstance(self.reviewer_id, str) and bool(self.reviewer_id.strip())
            and reviewed_at_valid
            and isinstance(self.evidence_path, str) and bool(self.evidence_path)
            and evidence_hash_valid
            and self.evidence_integrity_verified is True
            and isinstance(self.method, str)
            and self.method in {"SURVEY", "VISIBLE_GROUND_BOUNDARY", "VALIDATED_3D_PROJECTION"}
            and sigma_valid
            and isinstance(self.geometry, BaseGeometry) and self.geometry.is_valid
            and not self.geometry.is_empty
            and self.crs == "EPSG:32649"
        )


@dataclass(frozen=True)
class ExternalFootprintCandidate:
    """Unverified external geometry; never sufficient for ground acceptance."""

    geometry: BaseGeometry
    candidate_id: str
    source_ref: str
    source_crs: str
    source_capture_date: str | None = None
    source_date_status: str = "NOT_ASSOCIATED"
    source_date_candidates: tuple[str, ...] = ()
    source_tile_ids: tuple[str, ...] = ()
    source_tile_actual_dates: tuple[tuple[str, str | None], ...] = ()
    source_scene_key: str | None = None
    source_scene_linkage_status: str = "UNVERIFIED"
    source_date_provenance: str | None = None
    source_epoch_label: str | None = None
    source_geometry_semantics: str | None = None
    source_geometry_semantics_provenance: str | None = None
    source_accuracy_m: float | None = None
    association_method: str = "UNVERIFIED_SPATIAL_MATCH"
    match_iou: float | None = None
    intersection_area_m2: float | None = None
    roof_overlap_fraction: float | None = None
    candidate_overlap_fraction: float | None = None
    candidate_centroid_dx_from_roof_m: float | None = None
    candidate_centroid_dy_from_roof_m: float | None = None
    centroid_distance_m: float | None = None
    source_properties: dict[str, Any] = field(default_factory=dict)
    candidate_polygon_distance_m: float | None = None
    candidate_search_radius_m: float | None = None
    centroid_aligned_roof_iou: float | None = None
    centroid_aligned_iou_gain: float | None = None
    source_roof_geometry: BaseGeometry | None = None


@dataclass
class Observation:
    uid: str
    epoch_label: str
    actual_date: str | None
    tile_id: str | None
    image_path: str | None
    roof: BaseGeometry
    source_ref: str
    covered_tiles: tuple[str, ...] = ()
    source_tile_ids: tuple[str, ...] = ()
    source_image_paths: tuple[str, ...] = ()
    source_hashes: tuple[str, ...] = ()
    date_status: str = "CENTROID_TILE_ONLY"
    date_candidates: tuple[str, ...] = ()
    footprint_evidence: FootprintEvidence | None = None
    footprint_candidates: list[ExternalFootprintCandidate] = field(default_factory=list)
    registration_dx_m: float | None = None
    registration_dy_m: float | None = None
    registration_sigma_m: float | None = None
    registration_status: str = "NOT_RUN"
    footprint_candidate: BaseGeometry | None = None
    footprint_status: str = "ROOF_ONLY_UNRESOLVED"
    ground_footprint: BaseGeometry | None = None
    candidate_identity: str | None = None
    confirmed_identity_key: str | None = None
    identity_reviewed: bool = False
    identity_evidence_type: str | None = None
    identity_evidence_ref: str | None = None
    identity_evidence_path: str | None = None
    identity_evidence_sha256: str | None = None
    identity_reviewer_id: str | None = None
    identity_reviewed_at: str | None = None
    identity_evidence_integrity_verified: bool = False
    physical_id: int | None = None
    evidence_notes: list[str] = field(default_factory=list)
    registered_roof: BaseGeometry | None = None
    image_alignment_status: str = "NOT_EVALUATED"
    image_alignment_field_id: str | None = None
    image_alignment_cell_id: str | None = None
    image_alignment_dx_m: float | None = None
    image_alignment_dy_m: float | None = None
    image_alignment_positional_uncertainty_m: float | None = None
    image_alignment_relative_uncertainty_m: float | None = None
    image_alignment_review_basis: str | None = None
    image_alignment_review_metadata: dict[str, Any] | None = None

    @property
    def tracking_geometry(self) -> BaseGeometry:
        """Use reviewed ground geometry, else aligned roof proxy, else raw roof."""
        if self.ground_footprint is not None:
            return self.ground_footprint
        if self.registered_roof is not None:
            return self.registered_roof
        return self.roof


@dataclass
class Reconstruction:
    observations: list[Observation]
    candidate_edges: list[dict[str, Any]]
    identity_graph: list[dict[str, Any]]
    presence: list[dict[str, Any]]
    transitions: list[dict[str, Any]]
    resolver_events: list[dict[str, Any]]
    repair_candidates: list[dict[str, Any]]
    epoch_order: list[str] = field(default_factory=list)
    actual_date_order: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AbsenceEvidence:
    confirmed_identity_key: str
    epoch_label: str
    actual_date: str
    image_ref: str
    full_footprint_coverage: bool
    reviewed: bool
    reviewer_id: str | None = None
    reviewed_at: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    evidence_integrity_verified: bool = False

    @property
    def accepted(self) -> bool:
        reviewed_at_valid = False
        if isinstance(self.reviewed_at, str) and self.reviewed_at:
            try:
                reviewed_at = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
                reviewed_at_valid = reviewed_at.tzinfo is not None and reviewed_at.utcoffset() is not None
            except ValueError:
                reviewed_at_valid = False
        evidence_hash_valid = bool(
            isinstance(self.evidence_sha256, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", self.evidence_sha256)
        )
        return bool(
            isinstance(self.confirmed_identity_key, str) and self.confirmed_identity_key.strip()
            and isinstance(self.image_ref, str) and self.image_ref.strip()
            and self.full_footprint_coverage is True and self.reviewed is True
            and _parse_actual(self.actual_date)
            and isinstance(self.reviewer_id, str) and self.reviewer_id.strip()
            and reviewed_at_valid and isinstance(self.evidence_path, str) and self.evidence_path
            and evidence_hash_valid and self.evidence_integrity_verified
        )


def _parse_actual(value: str | None) -> date | None:
    if value is None or value in {"", "DATE_UNKNOWN", "NO_TILE_DATE_RECORD"}:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def reviewed_identity_is_valid(observation: Observation) -> bool:
    """Return whether identity is bound to complete, hash-verified review provenance."""
    if not (observation.identity_reviewed is True
            and isinstance(observation.confirmed_identity_key, str)
            and observation.confirmed_identity_key.strip()
            and _parse_actual(observation.actual_date) is not None
            and observation.identity_evidence_integrity_verified is True):
        return False
    if not all(isinstance(value, str) and value.strip() for value in (
            observation.identity_evidence_type, observation.identity_evidence_ref,
            observation.identity_evidence_path, observation.identity_reviewer_id,
            observation.identity_reviewed_at, observation.identity_evidence_sha256)):
        return False
    if not re.fullmatch(r"[0-9a-fA-F]{64}", observation.identity_evidence_sha256):
        return False
    try:
        reviewed_at = datetime.fromisoformat(
            observation.identity_reviewed_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return reviewed_at.tzinfo is not None and reviewed_at.utcoffset() is not None


def _chronological_epoch_order(observations: list[Observation], absences: list[AbsenceEvidence]) -> list[str]:
    """Order nominal output groups by their observed dates, never label text.

    If a nominal group contains multiple acquisition dates, its median date is
    used only to order the output groups. Every identity transition still
    checks its own exact dates and remains unresolved when that order conflicts.
    """
    dates_by_epoch: dict[str, list[date]] = {}
    all_epochs = {obs.epoch_label for obs in observations}
    all_epochs.update(absence.epoch_label for absence in absences if absence.accepted)
    for obs in observations:
        observed_date = _parse_actual(obs.actual_date)
        if observed_date is not None:
            dates_by_epoch.setdefault(obs.epoch_label, []).append(observed_date)
    for absence in absences:
        observed_date = _parse_actual(absence.actual_date)
        if absence.accepted and observed_date is not None:
            dates_by_epoch.setdefault(absence.epoch_label, []).append(observed_date)

    def sort_key(epoch: str) -> tuple[date, str]:
        values = sorted(dates_by_epoch.get(epoch, []))
        if not values:
            return date.max, epoch
        return values[len(values) // 2], epoch

    return sorted(all_epochs, key=sort_key)


def _actual_date_order(observations: list[Observation], absences: list[AbsenceEvidence]) -> list[str]:
    """Return unique, verified acquisition dates in chronological order."""
    values = {
        observed_date.isoformat()
        for observed_date in (_parse_actual(obs.actual_date) for obs in observations)
        if observed_date is not None
    }
    values.update(
        observed_date.isoformat()
        for observed_date in (_parse_actual(absence.actual_date) for absence in absences if absence.accepted)
        if observed_date is not None
    )
    return sorted(values)


def recover_footprints(observations: Iterable[Observation]) -> None:
    """Accept only independent, reviewed ground geometry; preserve roof lineage."""
    for obs in observations:
        evidence = obs.footprint_evidence
        if evidence is not None and evidence.accepted:
            obs.ground_footprint = evidence.geometry
            obs.footprint_status = "GROUND_VERIFIED"
            obs.evidence_notes.append(f"GROUND_SOURCE:{evidence.source_ref}")
            if obs.identity_reviewed and obs.confirmed_identity_key:
                if obs.identity_evidence_type is None:
                    obs.identity_evidence_type = "REVIEWED_GROUND_FOOTPRINT_RECORD"
                    obs.identity_evidence_ref = evidence.source_ref
                    obs.identity_evidence_path = evidence.evidence_path
                    obs.identity_evidence_sha256 = evidence.evidence_sha256
                    obs.identity_reviewer_id = evidence.reviewer_id
                    obs.identity_reviewed_at = evidence.reviewed_at
                    obs.identity_evidence_integrity_verified = (
                        evidence.evidence_integrity_verified is True)
        elif evidence is not None:
            obs.ground_footprint = None
            obs.footprint_candidate = evidence.geometry
            obs.footprint_status = "GROUND_CANDIDATE_UNVERIFIED"
            obs.evidence_notes.append("FOOTPRINT_EVIDENCE_GATE_FAILED")
        elif obs.footprint_candidates:
            obs.ground_footprint = None
            if obs.footprint_candidate is None:
                # This only selects which unverified candidate is shown as the
                # primary review geometry. It never changes ground status.
                best = max(
                    obs.footprint_candidates,
                    key=lambda item: (
                        item.centroid_aligned_roof_iou
                        if item.centroid_aligned_roof_iou is not None
                        else item.match_iou or 0.0,
                        -(item.candidate_polygon_distance_m or 0.0),
                    ),
                )
                obs.footprint_candidate = best.geometry
            obs.footprint_status = "GROUND_CANDIDATE_UNVERIFIED"
            obs.evidence_notes.append("EXTERNAL_FOOTPRINT_SOURCE_UNVERIFIED")
        else:
            obs.ground_footprint = None
            obs.footprint_status = "ROOF_ONLY_UNRESOLVED"
        # Reviewed image-grid alignment is applied separately. It never
        # establishes the roof-to-ground parallax vector or ground geometry.


def _tbia_observation(obs: Observation) -> BuildingObservation:
    geom = obs.tracking_geometry
    centroid = geom.centroid
    ev = obs.footprint_evidence
    if obs.ground_footprint is not None:
        alignment_dx = alignment_dy = 0.0
        alignment_uncertainty = (ev.positional_sigma_m if ev is not None else None)
        uncertainty_source = "REVIEWED_GROUND_FOOTPRINT" if alignment_uncertainty is not None else "UNKNOWN"
    elif obs.registered_roof is not None:
        alignment_dx = alignment_dy = 0.0
        if obs.image_alignment_review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES":
            alignment_uncertainty = obs.image_alignment_relative_uncertainty_m
            uncertainty_source = ("RELATIVE_IMAGE_TO_BASE_GRID"
                                  if alignment_uncertainty is not None else "UNKNOWN")
        else:
            alignment_uncertainty = obs.image_alignment_positional_uncertainty_m
            uncertainty_source = ("REVIEWED_IMAGE_GRID_ALIGNMENT"
                                  if alignment_uncertainty is not None else "UNKNOWN")
    else:
        alignment_dx = obs.registration_dx_m or 0.0
        alignment_dy = obs.registration_dy_m or 0.0
        alignment_uncertainty = obs.registration_sigma_m
        uncertainty_source = ("REGISTRATION_CANDIDATE" if alignment_uncertainty is not None
                              else "UNKNOWN")
    return BuildingObservation(
        object_id=obs.uid,
        epoch=obs.actual_date or obs.epoch_label,
        x=centroid.x,
        y=centroid.y,
        area_m2=geom.area,
        perimeter_m=geom.length,
        compactness=(4 * 3.141592653589793 * geom.area / (geom.length * geom.length)) if geom.length else 0.0,
        alignment_dx=alignment_dx,
        alignment_dy=alignment_dy,
        alignment_uncertainty=alignment_uncertainty,
        roof_dx=ev.roof_to_ground_dx_m if ev and ev.roof_to_ground_dx_m is not None else 0.0,
        roof_dy=ev.roof_to_ground_dy_m if ev and ev.roof_to_ground_dy_m is not None else 0.0,
        offset_confidence=1.0 if ev and ev.accepted and ev.roof_to_ground_dx_m is not None else 0.0,
        metadata={"footprint_status": obs.footprint_status,
                  "date_known": _parse_actual(obs.actual_date) is not None,
                  "alignment_uncertainty": alignment_uncertainty,
                  "alignment_uncertainty_source": uncertainty_source},
    )


def _tracking_geometry_basis(previous: Observation, current: Observation) -> str:
    if previous.ground_footprint is not None and current.ground_footprint is not None:
        return "GROUND"
    if previous.registered_roof is not None or current.registered_roof is not None:
        return "ROOF_IMAGE_GRID"
    return "ROOF_PROXY"


def _valid_registered_alignment_uncertainty(observation: Observation) -> bool:
    """Validate relative grid precision separately from absolute ground accuracy."""
    metadata = observation.image_alignment_review_metadata or {}
    basis = observation.image_alignment_review_basis or metadata.get("review_basis")
    absolute = observation.image_alignment_positional_uncertainty_m
    relative = observation.image_alignment_relative_uncertainty_m
    if basis == "RELATIVE_REFERENCE_IMAGE_FEATURES":
        return bool(
            absolute is None
            and isinstance(relative, (int, float)) and not isinstance(relative, bool)
            and math.isfinite(float(relative)) and 0 <= relative <= 2.0
            and metadata.get("independent_ground_evidence") is False
        )
    if basis not in (None, "INDEPENDENT_GROUND_CONTROL"):
        return False
    if (not isinstance(absolute, (int, float)) or isinstance(absolute, bool)
            or not math.isfinite(float(absolute)) or not 0 <= absolute <= 2.0
            or relative is not None):
        return False
    if (basis == "INDEPENDENT_GROUND_CONTROL"
            and metadata.get("independent_ground_evidence") is not True):
        return False
    return True


def _shared_external_outline_reference_edges(
    previous: list[Observation], current: list[Observation],
) -> list[dict[str, Any]]:
    """Emit candidate-only temporal links from a shared unverified outline FID.

    The reference may help bridge roof shifts or split/merge candidates, but it
    is never ground truth and cannot promote an identity or presence state.
    """
    external_associations = {
        "POSITIVE_AREA_OVERLAP_ONLY", "NEARBY_WITHIN_SEARCH_RADIUS",
    }

    def by_reference(records: list[Observation]) -> dict[tuple[str, str], dict[str, ExternalFootprintCandidate]]:
        found: dict[tuple[str, str], dict[str, ExternalFootprintCandidate]] = {}
        for observation in records:
            for candidate in observation.footprint_candidates:
                if (candidate.association_method not in external_associations
                        or not candidate.source_ref or not candidate.candidate_id):
                    continue
                key = (candidate.source_ref, candidate.candidate_id)
                found.setdefault(key, {})[observation.uid] = candidate
        return found

    previous_by_reference = by_reference(previous)
    current_by_reference = by_reference(current)
    shared: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}
    for reference in sorted(set(previous_by_reference) & set(current_by_reference)):
        for from_uid, left in previous_by_reference[reference].items():
            for to_uid, right in current_by_reference[reference].items():
                # A reused reference key must still identify the same source
                # geometry and metadata in both acquisitions.
                if (left.source_crs != right.source_crs
                        or left.source_capture_date != right.source_capture_date
                        or left.source_epoch_label != right.source_epoch_label
                        or left.source_geometry_semantics != right.source_geometry_semantics
                        or left.source_geometry_semantics_provenance != right.source_geometry_semantics_provenance
                        or left.source_accuracy_m != right.source_accuracy_m
                        or not left.geometry.equals(right.geometry)):
                    continue
                pair = (from_uid, to_uid)
                shared.setdefault(pair, {})[reference] = {
                    "source_ref": reference[0],
                    "candidate_id": reference[1],
                    "source_crs": left.source_crs,
                    "source_capture_date": left.source_capture_date,
                    "source_epoch_label": left.source_epoch_label,
                    "source_geometry_semantics": left.source_geometry_semantics,
                    "source_geometry_semantics_provenance": left.source_geometry_semantics_provenance,
                    "source_accuracy_m": left.source_accuracy_m,
                    "verification_status": "UNVERIFIED_CANDIDATE",
                    "from_association_method": left.association_method,
                    "to_association_method": right.association_method,
                }
    if not shared:
        return []

    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for from_uid, to_uid in shared:
        outgoing.setdefault(from_uid, set()).add(to_uid)
        incoming.setdefault(to_uid, set()).add(from_uid)

    previous_by_uid = {item.uid: item for item in previous}
    current_by_uid = {item.uid: item for item in current}
    edges = []
    for from_uid, to_uid in sorted(shared):
        left = previous_by_uid[from_uid]
        right = current_by_uid[to_uid]
        out_count = len(outgoing[from_uid])
        in_count = len(incoming[to_uid])
        relation = ("1:1" if out_count == in_count == 1 else
                    "1:N" if out_count > 1 and in_count == 1 else
                    "N:1" if out_count == 1 and in_count > 1 else "N:M")
        references = [shared[(from_uid, to_uid)][key]
                      for key in sorted(shared[(from_uid, to_uid)])]
        edges.append({
            "from_uid": from_uid,
            "to_uid": to_uid,
            "from_date": left.actual_date,
            "to_date": right.actual_date,
            "temporal_order_valid": True,
            "relation": relation,
            "affinity": None,
            "search_radius_m": None,
            "candidate_count": len(references),
            "candidate_rank": None,
            "hard_negative_ids": [],
            "status": "CANDIDATE_ONLY",
            "geometry_basis": _tracking_geometry_basis(left, right),
            "resolver_stage": "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE",
            "edge_basis": "SHARED_UNVERIFIED_EXTERNAL_OUTLINE_REFERENCE",
            "candidate_references": references,
        })
    return edges


def _candidate_links(observations: list[Observation], segmenter: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_acquisition: dict[str, list[Observation]] = {}
    for obs in observations:
        observed_date = _parse_actual(obs.actual_date)
        if observed_date is not None:
            by_acquisition.setdefault(observed_date.isoformat(), []).append(obs)
    acquisitions = sorted(by_acquisition, key=date.fromisoformat)
    matcher = TBIAPrototype()
    edges: list[dict[str, Any]] = []
    resolver_events: list[dict[str, Any]] = []
    repair_candidates: list[dict[str, Any]] = []
    for previous_date, current_date in zip(acquisitions, acquisitions[1:]):
        previous = by_acquisition[previous_date]
        current = by_acquisition[current_date]
        decisions = matcher.match(map(_tbia_observation, previous), map(_tbia_observation, current))
        previous_by_uid = {o.uid: o for o in previous}
        current_by_uid = {o.uid: o for o in current}
        incoming: dict[str, set[str]] = {}
        candidates_by_source: dict[str, list[tuple[str, float, int]]] = {}
        for decision in decisions:
            prior = previous_by_uid[decision.previous_id]
            if decision.candidate_matches:
                candidates_by_source[prior.uid] = [
                    (uid, affinity, rank)
                    for rank, (uid, affinity) in enumerate(decision.candidate_matches, 1)
                ]
                for current_uid, _affinity in decision.candidate_matches:
                    incoming.setdefault(current_uid, set()).add(prior.uid)
            else:
                # Preserve the best sub-threshold candidate as a diagnostic,
                # without admitting it into the candidate identity graph.
                best_uid = decision.best_candidate_id
                candidates_by_source[prior.uid] = [
                    (best_uid, decision.affinity, 1)
                ]

        outgoing_counts = {uid: len(links) for uid, links in candidates_by_source.items()
                           if links and links[0][0] is not None
                           and any(decision.previous_id == uid and decision.candidate_matches
                                   for decision in decisions)}
        incoming_counts = {uid: len(prior_uids) for uid, prior_uids in incoming.items()}
        decision_by_uid = {decision.previous_id: decision for decision in decisions}
        for prior_uid, candidate_links in candidates_by_source.items():
            prior = previous_by_uid[prior_uid]
            decision = decision_by_uid[prior_uid]
            accepted_source = bool(decision.candidate_matches)
            for current_uid, affinity, rank in candidate_links:
                curr = current_by_uid.get(current_uid or "")
                temporal_ok = bool(curr and _parse_actual(prior.actual_date) == date.fromisoformat(previous_date)
                                    and _parse_actual(curr.actual_date) == date.fromisoformat(current_date)
                                    and date.fromisoformat(previous_date) < date.fromisoformat(current_date))
                if accepted_source and temporal_ok:
                    out_count = outgoing_counts.get(prior_uid, 0)
                    in_count = incoming_counts.get(curr.uid, 0)
                    relation = ("1:1" if out_count == in_count == 1 else
                                "1:N" if out_count > 1 and in_count == 1 else
                                "N:1" if out_count == 1 and in_count > 1 else "N:M")
                    status = "CANDIDATE_ONLY"
                else:
                    relation = "UNRESOLVED"
                    status = ("BELOW_AFFINITY_THRESHOLD" if curr is not None
                              else "NO_LOCAL_CANDIDATE")
                edges.append({
                    "from_uid": prior.uid, "to_uid": curr.uid if curr else None,
                    "from_date": prior.actual_date, "to_date": curr.actual_date if curr else None,
                    "temporal_order_valid": temporal_ok, "relation": relation,
                    "affinity": round(float(affinity), 5),
                    "search_radius_m": round(
                        decision.candidate_search_radii_m.get(
                            curr.uid if curr else "", decision.search_radius_m), 3),
                    "candidate_count": len(decision.candidate_matches),
                    "candidate_rank": rank,
                    "hard_negative_ids": decision.hard_negative_ids,
                    "status": status,
                    "geometry_basis": _tracking_geometry_basis(prior, curr) if curr else "ROOF_PROXY",
                    "resolver_stage": decision.resolver_stage,
                    "edge_basis": "TBIA_GEOMETRIC",
                    "candidate_references": [],
                })
        for current_id, prior_uid_set in incoming.items():
            prior_ids = sorted(prior_uid_set)
            if len(prior_ids) > 1:
                curr = current_by_uid[current_id]
                result = resolve_merged_mask(
                    current_id, prior_ids, current_area_m2=curr.tracking_geometry.area,
                    prior_area_m2=[previous_by_uid[uid].tracking_geometry.area for uid in prior_ids],
                    segmenter=segmenter, image=curr.image_path,
                    prompts=[{"uid": uid,
                              "x": previous_by_uid[uid].tracking_geometry.centroid.x,
                              "y": previous_by_uid[uid].tracking_geometry.centroid.y}
                             for uid in prior_ids],
                )
                valid_children = []
                if result.status == "INSTANCE_MERGE_ERROR_CANDIDATE":
                    for index, child in enumerate(result.child_masks):
                        if not isinstance(child, BaseGeometry) or child.is_empty or not child.is_valid:
                            break
                        if child.intersection(curr.tracking_geometry).area / max(child.area, 1e-9) < 0.8:
                            break
                        if any(child.intersection(other).area / max(min(child.area, other.area), 1e-9) > 0.05 for other in valid_children):
                            break
                        valid_children.append(child)
                    if len(valid_children) == len(prior_ids):
                        for index, child in enumerate(valid_children):
                            repair_candidates.append({"merged_uid": current_id, "prior_uid": prior_ids[index],
                                                      "child_index": index, "geometry": child,
                                                      "status": "INSTANCE_REPAIR_CANDIDATE_NOT_GROUND"})
                    else:
                        result.status = "SEGMENTER_GEOMETRY_REJECTED"
                resolver_events.append({"current_uid": current_id, "prior_uids": prior_ids, "status": result.status, "resolver_stage": result.resolver_stage, "child_count": len(result.child_masks), "promoted_to_ground": False})
    # Shared external outline references can bridge a missed intermediate
    # observation. Keep these links separate from adjacent-date TBIA edges and
    # candidate-only so they never imply presence or confirm a physical ID.
    for left_index, left_date in enumerate(acquisitions):
        for right_date in acquisitions[left_index + 1:]:
            edges.extend(_shared_external_outline_reference_edges(
                by_acquisition[left_date], by_acquisition[right_date]))
    return edges, resolver_events, repair_candidates


def _identity_and_state(observations: list[Observation], edges: list[dict[str, Any]],
                        absences: list[AbsenceEvidence], actual_date_order: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Only reviewed physical-identity evidence can yield IDs.

    No-match does not prove absence.  Missing epochs and date ties remain U.
    """
    # TBIA candidate links are not an identity verdict.  A reviewed physical
    # key is required and is deliberately distinct from frozen R4 track IDs.
    keys = sorted({
        o.confirmed_identity_key for o in observations
        if reviewed_identity_is_valid(o)
    })
    id_by_key = {key: int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big") for key in keys}
    if any(value in {0, 4294967295} for value in id_by_key.values()) or len(set(id_by_key.values())) != len(id_by_key):
        raise ValueError("Physical ID hash collision or reserved value; use an explicit ID registry")
    for key in keys:
        footprints_by_date: dict[str, list[Any]] = {}
        for obs in observations:
            if (obs.confirmed_identity_key == key and obs.identity_reviewed
                    and obs.ground_footprint is not None):
                observed_date = _parse_actual(obs.actual_date)
                if observed_date is not None:
                    footprints_by_date.setdefault(observed_date.isoformat(), []).append(obs.ground_footprint)
        for observed_date, geometries in footprints_by_date.items():
            if any(not geometries[0].equals(geometry) for geometry in geometries[1:]):
                raise ValueError(
                    f"Multiple reviewed footprints for {key} on actual date {observed_date}; "
                    "resolve instances first")
    graph: list[dict[str, Any]] = []
    for obs in sorted(observations, key=lambda item: (_parse_actual(item.actual_date) or date.max, item.uid)):
        accepted = reviewed_identity_is_valid(obs)
        obs.physical_id = id_by_key[obs.confirmed_identity_key] if accepted else None
        status = "REVIEWED_PHYSICAL_IDENTITY" if accepted else "UNRESOLVED_IDENTITY"
        graph.append({"uid": obs.uid, "actual_date": obs.actual_date, "epoch_label": obs.epoch_label, "physical_id": obs.physical_id, "status": status, "candidate_identity": obs.candidate_identity})
    presence = [
        {"uid": obs.uid, "epoch_label": obs.epoch_label, "actual_date": obs.actual_date,
         "candidate_identity": obs.candidate_identity,
         "physical_id": obs.physical_id, "presence": "1" if obs.physical_id is not None else "U",
         "confirmed_identity_key": obs.confirmed_identity_key if reviewed_identity_is_valid(obs) else None,
         "identity_reviewed": reviewed_identity_is_valid(obs),
         "identity_evidence_type": obs.identity_evidence_type,
         "identity_evidence_ref": obs.identity_evidence_ref,
         "identity_evidence_path": obs.identity_evidence_path,
         "identity_evidence_sha256": obs.identity_evidence_sha256,
         "identity_reviewer_id": obs.identity_reviewer_id,
         "identity_reviewed_at": obs.identity_reviewed_at,
         "footprint_source_ref": obs.footprint_evidence.source_ref if obs.footprint_evidence else None,
         "footprint_method": obs.footprint_evidence.method if obs.footprint_evidence else None,
         "footprint_sigma_m": obs.footprint_evidence.positional_sigma_m if obs.footprint_evidence else None,
         "presence_evidence_type": (
             "REVIEWED_PHYSICAL_IDENTITY_AND_OBSERVATION" if obs.physical_id is not None
             else "CANDIDATE_OBSERVATION_OR_IDENTITY_UNRESOLVED"),
         "presence_evidence_ref": (
             obs.footprint_evidence.source_ref if obs.footprint_evidence
             else obs.source_ref),
         "coverage_complete": (
             True if (obs.physical_id is not None
                      and obs.date_status == "DATE_COHERENT_FULL_ROOF_COVERAGE")
             else None),
         "review_status": "REVIEWED" if obs.physical_id is not None else "UNRESOLVED",
         "confidence": None,
         "notes": ("Reviewed physical identity and dated observation."
                   if obs.physical_id is not None else
                   "Unresolved candidate; this is not a reviewed absence."),
         "basis": "REVIEWED_PHYSICAL_IDENTITY" if obs.physical_id is not None else "CANDIDATE_ONLY"}
        for obs in observations
    ]
    accepted_absences = [a for a in absences if a.accepted]
    for absence in accepted_absences:
        key = absence.confirmed_identity_key
        if key not in id_by_key:
            raise ValueError(
                f"Reviewed absence identity has no dated reviewed identity observation: {key}")
        if any(o.actual_date == absence.actual_date and o.physical_id == id_by_key[key]
               for o in observations):
            raise ValueError(f"Conflicting presence and absence for {key}/{absence.actual_date}")
        presence.append({"uid": None, "epoch_label": absence.epoch_label,
                         "actual_date": absence.actual_date,
                         "physical_id": id_by_key[key], "confirmed_identity_key": key,
                         "identity_reviewed": True, "presence": "0",
                         "basis": "REVIEWED_FULL_FOOTPRINT_COVERAGE",
                         "presence_evidence_type": "REVIEWED_FULL_FOOTPRINT_COVERAGE",
                         "presence_evidence_ref": absence.image_ref,
                         "coverage_complete": True,
                         "review_status": "REVIEWED",
                         "confidence": None,
                         "notes": "Full-footprint coverage was reviewed; building absent.",
                         "image_ref": absence.image_ref,
                         "full_footprint_coverage": absence.full_footprint_coverage,
                         "reviewed": absence.reviewed, "reviewer_id": absence.reviewer_id,
                         "reviewed_at": absence.reviewed_at,
                         "evidence_path": absence.evidence_path,
                         "evidence_sha256": absence.evidence_sha256})
    transitions: list[dict[str, Any]] = []
    presence_by_identity_date: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in presence:
        observed_date = _parse_actual(row.get("actual_date"))
        physical_id = row.get("physical_id")
        if observed_date is not None and physical_id is not None:
            presence_by_identity_date.setdefault(
                (int(physical_id), observed_date.isoformat()), []).append(row)

    for left, right in zip(actual_date_order, actual_date_order[1:]):
        ids = sorted(set(id_by_key.values()))
        for physical_id in ids:
            a = presence_by_identity_date.get((physical_id, left), [])
            b = presence_by_identity_date.get((physical_id, right), [])
            a_states = {row["presence"] for row in a}
            b_states = {row["presence"] for row in b}
            if "1" in a_states and "0" in a_states:
                raise ValueError(f"Conflicting presence and absence for physical ID {physical_id}/{left}")
            if "1" in b_states and "0" in b_states:
                raise ValueError(f"Conflicting presence and absence for physical ID {physical_id}/{right}")
            pa = "1" if "1" in a_states else "0" if "0" in a_states else "U"
            pb = "1" if "1" in b_states else "0" if "0" in b_states else "U"
            state = {("1", "1"): "STABLE", ("0", "1"): "ADDITION", ("1", "0"): "DEMOLITION"}.get((pa, pb), "UNRESOLVED")
            transitions.append({"from_epoch": left, "to_epoch": right,
                                "from_output_group": left, "to_output_group": right,
                                "physical_id": physical_id, "state": state,
                                "from_presence": pa, "to_presence": pb,
                                "from_date": left if a else None,
                                "to_date": right if b else None})
    return graph, presence, transitions


def reconstruct(observations: Iterable[Observation], *, segmenter: Any = None, absence_evidence: Iterable[AbsenceEvidence] = ()) -> Reconstruction:
    records = list(observations)
    absences = list(absence_evidence)
    if len({o.uid for o in records}) != len(records):
        raise ValueError("Observation uid must be unique")
    observed_epoch_dates = {
        (obs.epoch_label, parsed.isoformat())
        for obs in records
        if (parsed := _parse_actual(obs.actual_date)) is not None
    }
    for absence in absences:
        if (absence.accepted
                and (absence.epoch_label, absence.actual_date) not in observed_epoch_dates):
            raise ValueError(
                "Reviewed absence actual date is not observed under its supplied epoch label: "
                f"{absence.epoch_label}/{absence.actual_date}")
    for obs in records:
        if obs.roof.is_empty or not obs.roof.is_valid:
            raise ValueError(f"Invalid roof: {obs.uid}")
        if obs.registered_roof is not None and (
                obs.registered_roof.is_empty or not obs.registered_roof.is_valid
                or obs.registered_roof.geom_type not in {"Polygon", "MultiPolygon"}
                or not obs.image_alignment_field_id or not obs.image_alignment_cell_id
                or obs.image_alignment_status != "ACCEPTED_IMAGE_GRID_SHIFT_ROOF_NOT_GROUND"
                or obs.image_alignment_dx_m is None or obs.image_alignment_dy_m is None
                or not math.isfinite(float(obs.image_alignment_dx_m))
                or not math.isfinite(float(obs.image_alignment_dy_m))
                or not isinstance(obs.image_alignment_review_metadata, dict)
                or any(not obs.image_alignment_review_metadata.get(name) for name in (
                    "reviewer_id", "reviewed_at", "review_basis", "review_evidence_type",
                    "review_evidence_path", "review_evidence_sha256",
                    "review_reference_sha256"))
                or not _valid_registered_alignment_uncertainty(obs)
                or obs.registered_roof.hausdorff_distance(translate(
                    obs.roof, xoff=float(obs.image_alignment_dx_m),
                    yoff=float(obs.image_alignment_dy_m))) > 1e-7):
            raise ValueError(f"Invalid or unproven registered roof geometry: {obs.uid}")
        for candidate in obs.footprint_candidates:
            if candidate.geometry.is_empty or not candidate.geometry.is_valid:
                raise ValueError(f"Invalid footprint candidate: {obs.uid}/{candidate.candidate_id}")
            if not candidate.source_ref or not candidate.candidate_id:
                raise ValueError(f"Missing footprint candidate provenance: {obs.uid}")
            if (candidate.source_roof_geometry is not None
                    and (candidate.source_roof_geometry.is_empty
                         or not candidate.source_roof_geometry.is_valid
                         or candidate.source_roof_geometry.geom_type not in {"Polygon", "MultiPolygon"})):
                raise ValueError(f"Invalid source roof geometry: {obs.uid}/{candidate.candidate_id}")
            if candidate.source_accuracy_m is not None and (not math.isfinite(candidate.source_accuracy_m) or candidate.source_accuracy_m < 0):
                raise ValueError(f"Invalid footprint candidate source accuracy: {obs.uid}/{candidate.candidate_id}")
            if candidate.match_iou is not None and (not math.isfinite(candidate.match_iou) or not 0 <= candidate.match_iou <= 1):
                raise ValueError(f"Invalid footprint candidate match score: {obs.uid}/{candidate.candidate_id}")
            fractions = (candidate.roof_overlap_fraction, candidate.candidate_overlap_fraction)
            if any(value is not None and (not math.isfinite(value) or not 0 <= value <= 1)
                   for value in fractions):
                raise ValueError(f"Invalid footprint candidate overlap fraction: {obs.uid}/{candidate.candidate_id}")
            finite_metrics = (
                candidate.intersection_area_m2,
                candidate.candidate_centroid_dx_from_roof_m,
                candidate.candidate_centroid_dy_from_roof_m,
                candidate.centroid_distance_m,
                candidate.candidate_polygon_distance_m,
                candidate.centroid_aligned_roof_iou,
                candidate.centroid_aligned_iou_gain,
            )
            if any(value is not None and not math.isfinite(value) for value in finite_metrics):
                raise ValueError(f"Invalid footprint candidate geometry metric: {obs.uid}/{candidate.candidate_id}")
            if candidate.intersection_area_m2 is not None and candidate.intersection_area_m2 < 0:
                raise ValueError(f"Negative footprint candidate intersection area: {obs.uid}/{candidate.candidate_id}")
            if candidate.centroid_distance_m is not None and candidate.centroid_distance_m < 0:
                raise ValueError(f"Negative footprint candidate centroid distance: {obs.uid}/{candidate.candidate_id}")
            if candidate.candidate_polygon_distance_m is not None and candidate.candidate_polygon_distance_m < 0:
                raise ValueError(f"Negative footprint candidate polygon distance: {obs.uid}/{candidate.candidate_id}")
            if candidate.candidate_search_radius_m is not None and candidate.candidate_search_radius_m < 0:
                raise ValueError(f"Negative footprint candidate search radius: {obs.uid}/{candidate.candidate_id}")
            if (candidate.centroid_aligned_roof_iou is not None
                    and not 0 <= candidate.centroid_aligned_roof_iou <= 1):
                raise ValueError(f"Invalid centroid-aligned candidate IoU: {obs.uid}/{candidate.candidate_id}")
            if (candidate.centroid_aligned_iou_gain is not None
                    and not -1 <= candidate.centroid_aligned_iou_gain <= 1):
                raise ValueError(f"Invalid centroid-aligned candidate IoU gain: {obs.uid}/{candidate.candidate_id}")
            if candidate.association_method in {
                    "POSITIVE_AREA_OVERLAP_ONLY", "NEARBY_WITHIN_SEARCH_RADIUS"}:
                if (any(value is None for value in (*fractions, *finite_metrics))
                        or candidate.match_iou is None):
                    raise ValueError(f"Incomplete external candidate geometry metrics: {obs.uid}/{candidate.candidate_id}")
                intersection_area = obs.roof.intersection(candidate.geometry).area
                roof_fraction = intersection_area / obs.roof.area if obs.roof.area > 0 else 0.0
                candidate_fraction = intersection_area / candidate.geometry.area if candidate.geometry.area > 0 else 0.0
                union_area = obs.roof.union(candidate.geometry).area
                expected_iou = intersection_area / union_area if union_area > 0 else 0.0
                dx = candidate.geometry.centroid.x - obs.roof.centroid.x
                dy = candidate.geometry.centroid.y - obs.roof.centroid.y
                if (not math.isclose(candidate.intersection_area_m2, intersection_area,
                                     rel_tol=1e-6, abs_tol=1e-4)
                        or not math.isclose(candidate.roof_overlap_fraction, roof_fraction,
                                            rel_tol=1e-6, abs_tol=1e-6)
                        or not math.isclose(candidate.candidate_overlap_fraction, candidate_fraction,
                                            rel_tol=1e-6, abs_tol=1e-6)
                        or not math.isclose(candidate.match_iou, expected_iou,
                                            rel_tol=1e-6, abs_tol=1e-6)
                        or not math.isclose(candidate.candidate_centroid_dx_from_roof_m, dx,
                                            rel_tol=1e-6, abs_tol=1e-4)
                        or not math.isclose(candidate.candidate_centroid_dy_from_roof_m, dy,
                                            rel_tol=1e-6, abs_tol=1e-4)):
                    raise ValueError(f"Footprint candidate metrics do not match geometries: {obs.uid}/{candidate.candidate_id}")
                expected_distance = math.hypot(
                    candidate.candidate_centroid_dx_from_roof_m,
                    candidate.candidate_centroid_dy_from_roof_m,
                )
                if not math.isclose(candidate.centroid_distance_m, expected_distance,
                                    rel_tol=1e-6, abs_tol=1e-4):
                    raise ValueError(f"Inconsistent footprint candidate centroid metrics: {obs.uid}/{candidate.candidate_id}")
                expected_polygon_distance = obs.roof.distance(candidate.geometry)
                if not math.isclose(candidate.candidate_polygon_distance_m,
                                    expected_polygon_distance, rel_tol=1e-6, abs_tol=1e-4):
                    raise ValueError(f"Inconsistent footprint candidate polygon distance: {obs.uid}/{candidate.candidate_id}")
                shifted_roof = translate(
                    obs.roof,
                    xoff=candidate.candidate_centroid_dx_from_roof_m,
                    yoff=candidate.candidate_centroid_dy_from_roof_m,
                )
                shifted_union_area = shifted_roof.union(candidate.geometry).area
                expected_shifted_iou = (
                    shifted_roof.intersection(candidate.geometry).area / shifted_union_area
                    if shifted_union_area > 0 else 0.0
                )
                if (not math.isclose(candidate.centroid_aligned_roof_iou,
                                     expected_shifted_iou, rel_tol=1e-6, abs_tol=1e-6)
                        or not math.isclose(candidate.centroid_aligned_iou_gain,
                                            expected_shifted_iou - expected_iou,
                                            rel_tol=1e-6, abs_tol=1e-6)):
                    raise ValueError(f"Inconsistent centroid-alignment diagnostic: {obs.uid}/{candidate.candidate_id}")
                if candidate.association_method == "POSITIVE_AREA_OVERLAP_ONLY":
                    if intersection_area <= 0.01 or candidate.candidate_search_radius_m is not None:
                        raise ValueError(f"Invalid positive-overlap association: {obs.uid}/{candidate.candidate_id}")
                elif (intersection_area > 0.01
                      or candidate.candidate_search_radius_m is None
                      or expected_polygon_distance > candidate.candidate_search_radius_m + 1e-6):
                    raise ValueError(f"Invalid nearby-candidate association: {obs.uid}/{candidate.candidate_id}")
                if (candidate.candidate_search_radius_m is not None
                        and not math.isfinite(candidate.candidate_search_radius_m)):
                    raise ValueError(f"Invalid footprint candidate search radius: {obs.uid}/{candidate.candidate_id}")
    recover_footprints(records)
    edges, resolver_events, repair_candidates = _candidate_links(records, segmenter)
    epoch_order = _chronological_epoch_order(records, absences)
    actual_date_order = _actual_date_order(records, absences)
    graph, presence, transitions = _identity_and_state(records, edges, absences, actual_date_order)
    return Reconstruction(records, edges, graph, presence, transitions,
                          resolver_events, repair_candidates, epoch_order, actual_date_order)
