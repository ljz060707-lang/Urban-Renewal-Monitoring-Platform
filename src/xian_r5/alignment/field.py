"""Traceable local image-registration fields with a fail-closed apply gate.

Image-to-reference registration and absolute ground accuracy are distinct
evidence claims. A reviewed, measured stable-feature tie may authorize a
relative grid correction while absolute ground uncertainty remains unknown.
Neither evidence basis estimates roof-to-ground parallax. Cells are never
interpolated or extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
import re
from typing import Iterable

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry


_TILE_ID_RE = re.compile(r"^\d+:\d+$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class DisplacementCell:
    """One spatial support cell and its moving-to-reference tie evidence."""

    cell_id: str
    support_geometry: BaseGeometry
    correction_dx_m_candidate: float | None
    correction_dy_m_candidate: float | None
    fit_dispersion_m: float | None
    status: str
    method: str
    source_ref: str
    reviewed: bool = False
    independent_ground_evidence: bool = False
    positional_uncertainty_m: float | None = None
    applied: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    alignment_review_basis: str = "UNREVIEWED"
    relative_registration_uncertainty_m: float | None = None

    def __post_init__(self) -> None:
        if not self.cell_id or not self.status or not self.method or not self.source_ref:
            raise ValueError("Displacement cells require IDs, status, method and provenance")
        if (self.support_geometry.is_empty or not self.support_geometry.is_valid
                or self.support_geometry.geom_type not in {"Polygon", "MultiPolygon"}):
            raise ValueError(f"Invalid displacement support geometry: {self.cell_id}")
        pair = (self.correction_dx_m_candidate, self.correction_dy_m_candidate)
        if (pair[0] is None) != (pair[1] is None):
            raise ValueError("A displacement candidate must contain both x and y")
        for label, value in (("correction dx", pair[0]), ("correction dy", pair[1]),
                             ("fit dispersion", self.fit_dispersion_m),
                             ("absolute positional uncertainty", self.positional_uncertainty_m),
                             ("relative registration uncertainty",
                              self.relative_registration_uncertainty_m)):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite {label}: {self.cell_id}")
        if self.fit_dispersion_m is not None and self.fit_dispersion_m < 0:
            raise ValueError("Fit dispersion cannot be negative")
        if self.positional_uncertainty_m is not None and self.positional_uncertainty_m < 0:
            raise ValueError("Absolute positional uncertainty cannot be negative")
        if (self.relative_registration_uncertainty_m is not None
                and self.relative_registration_uncertainty_m < 0):
            raise ValueError("Relative registration uncertainty cannot be negative")
        if self.alignment_review_basis not in {
                "UNREVIEWED", "INDEPENDENT_GROUND_CONTROL",
                "RELATIVE_REFERENCE_IMAGE_FEATURES"}:
            raise ValueError(f"Unsupported alignment review basis: {self.alignment_review_basis}")
        if (self.alignment_review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES"
                and (self.independent_ground_evidence
                     or self.positional_uncertainty_m is not None)):
            raise ValueError(
                "Relative image ties cannot assert independent ground evidence or absolute uncertainty")
        if (self.relative_registration_uncertainty_m is not None
                and self.alignment_review_basis != "RELATIVE_REFERENCE_IMAGE_FEATURES"):
            raise ValueError(
                "Relative registration uncertainty requires the relative image-feature review basis")
        if self.applied and not self.accepted:
            raise ValueError("An unaccepted displacement candidate cannot be marked applied")

    @property
    def accepted(self) -> bool:
        """Whether the evidence is sufficient to use this as a grid correction."""
        absolute_ground_review = (
            self.status == "ACCEPTED_REVIEWED_GROUND"
            and self.reviewed
            and self.independent_ground_evidence
            and self.positional_uncertainty_m is not None
            and self.positional_uncertainty_m <= 2.0
            and self.alignment_review_basis in {
                "UNREVIEWED", "INDEPENDENT_GROUND_CONTROL"}
        )
        relative_image_review = (
            self.status == "ACCEPTED_REVIEWED_RELATIVE_IMAGE_TIE"
            and self.reviewed
            and not self.independent_ground_evidence
            and self.alignment_review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES"
            and self.positional_uncertainty_m is None
            and self.relative_registration_uncertainty_m is not None
            and self.relative_registration_uncertainty_m <= 2.0
        )
        absolute_source_scope_ok = self.source_component_scoped
        relative_source_scope_ok = (
            self.source_component_scoped
            or self.relative_cell_source_scope_review_authorized
        )
        return bool(
            self.correction_dx_m_candidate is not None
            and ((absolute_ground_review and absolute_source_scope_ok)
                 or (relative_image_review and relative_source_scope_ok))
        )

    @property
    def relative_cell_review_candidate(self) -> bool:
        """Whether this exact composed cell may receive a relative-only review."""
        metadata = self.metadata
        candidate_status = metadata.get("candidate_status", self.status)
        source_refs = metadata.get("source_candidate_cell_refs")
        source_edge_count = metadata.get("source_edge_count")
        component_statuses = metadata.get("reference_component_id_statuses")
        return bool(
            self.method == "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE"
            and candidate_status == "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY"
            and metadata.get("composition_method") == "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH"
            and metadata.get("source_component_scope") == "SOURCE_COMPONENT_SCOPE_UNKNOWN"
            and metadata.get("anchor_acquisition_date_status")
                == "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"
            and metadata.get("reference_component_id_status")
                == "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"
            and isinstance(component_statuses, list) and bool(component_statuses)
            and all(value == "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"
                    for value in component_statuses)
            and metadata.get("cycle_limit_semantics")
                == "INTERNAL_CLOSURE_SCREEN_ONLY_NOT_POSITIONAL_ACCURACY"
            and metadata.get("roof_to_ground_parallax_resolved") is False
            and isinstance(source_refs, list) and len(source_refs) >= 2
            and all(isinstance(value, str) and value.strip() for value in source_refs)
            and isinstance(source_edge_count, int) and not isinstance(source_edge_count, bool)
            and source_edge_count == len(source_refs)
        )

    @property
    def relative_cell_source_scope_review_authorized(self) -> bool:
        """Whether a reviewer attested to ties/holdouts inside this exact cell."""
        return bool(
            self.alignment_review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES"
            and self.metadata.get("relative_cell_source_scope_review_authorized") is True
            and self.metadata.get("relative_ties_within_support_cell") is True
            and self.metadata.get("relative_cell_source_scope_review_semantics")
                == "EXACT_COMPOSED_CELL_RELATIVE_TIES_ONLY_SOURCE_COMPONENT_AND_ABSOLUTE_POSITION_UNKNOWN"
        )

    @property
    def source_component_scoped(self) -> bool:
        """Whether this cell is tied to one hash-identified source tile per date."""
        if self.metadata.get("source_component_scope") != "SINGLE_SOURCE_COMPONENT_PER_CELL":
            return False
        components = self.metadata.get("source_component_ids")
        if not isinstance(components, list) or not components:
            return False
        coverage = self.metadata.get("source_component_coverage_fraction")
        if (isinstance(coverage, bool) or not isinstance(coverage, (int, float))
                or not math.isfinite(float(coverage))
                or not math.isclose(float(coverage), 1.0, rel_tol=0.0, abs_tol=1e-8)):
            return False
        seen_dates: set[str] = set()
        for component in components:
            if not isinstance(component, dict):
                return False
            actual_date = component.get("actual_date")
            try:
                date.fromisoformat(str(actual_date))
            except (TypeError, ValueError):
                return False
            tile_id = component.get("tile_id")
            path = component.get("path")
            digest = component.get("sha256")
            if (not _TILE_ID_RE.fullmatch(str(tile_id or ""))
                    or not isinstance(path, str) or not path.strip()
                    or not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
                    or str(actual_date) in seen_dates):
                return False
            seen_dates.add(str(actual_date))
        return True

    @property
    def source_component_issue(self) -> str | None:
        """Return the fail-closed reason when the source tile scope is unresolved."""
        if self.source_component_scoped:
            return None
        scope = self.metadata.get("source_component_scope")
        if scope == "MULTIPLE_SOURCE_COMPONENTS_PER_CELL":
            return "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED"
        if scope == "INCOMPLETE_SOURCE_COMPONENT_COVERAGE":
            return "SOURCE_COMPONENT_COVERAGE_INCOMPLETE"
        if scope == "SOURCE_COMPONENT_ID_MISMATCH":
            return "SOURCE_COMPONENT_ID_MISMATCH"
        return "SOURCE_COMPONENT_SCOPE_UNKNOWN"

    def source_component_id_for_date(self, actual_date: str) -> str | None:
        if not self.source_component_scoped:
            return None
        for component in self.metadata["source_component_ids"]:
            if component.get("actual_date") == actual_date:
                return str(component["tile_id"])
        return None


@dataclass(frozen=True)
class DisplacementSample:
    status: str
    field_id: str
    cell_id: str | None
    correction_dx_m_candidate: float | None
    correction_dy_m_candidate: float | None
    fit_dispersion_m: float | None
    accepted_correction_m: tuple[float, float] | None
    interpolated: bool = False


@dataclass(frozen=True)
class LocalDisplacementField:
    """A date-ordered, EPSG:32649 field of local registration candidates."""

    field_id: str
    reference_date: str | None
    moving_date: str
    cells: tuple[DisplacementCell, ...]
    crs: str = "EPSG:32649"
    correction_scope: str = "IMAGE_TO_REFERENCE_GRID"
    reference_label: str | None = None
    reference_source: str | None = None
    reference_date_status: str = "ACTUAL_DATE_KNOWN"

    def __post_init__(self) -> None:
        if not self.field_id or not self.cells:
            raise ValueError("A displacement field needs an ID and at least one support cell")
        if self.crs != "EPSG:32649":
            raise ValueError("Local displacement fields must use EPSG:32649")
        if self.correction_scope != "IMAGE_TO_REFERENCE_GRID":
            raise ValueError("Image registration cannot be relabeled as roof-to-ground parallax")
        try:
            moving = date.fromisoformat(self.moving_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("Moving images require a real ISO acquisition date") from exc
        if self.reference_date is None:
            if (self.reference_date_status != "REFERENCE_ACQUISITION_DATE_UNKNOWN"
                    or not self.reference_label or not self.reference_source):
                raise ValueError("Unknown reference dates require an explicit label and source")
        else:
            try:
                reference = date.fromisoformat(self.reference_date)
            except (TypeError, ValueError) as exc:
                raise ValueError("Known reference dates must use ISO acquisition dates") from exc
            if self.reference_date_status != "ACTUAL_DATE_KNOWN":
                raise ValueError("Reference date status conflicts with the supplied actual date")
            if moving <= reference:
                raise ValueError("Moving-image date must follow its reference date")
        if len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("Displacement cell IDs must be unique within a field")
        if any(cell.applied for cell in self.cells):
            raise ValueError("This field contract only carries unapplied evidence")

    def sample(self, x: float, y: float) -> DisplacementSample:
        """Return an exact-cell sample; gaps and overlaps remain unresolved."""
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Sample coordinates must be finite")
        point = Point(float(x), float(y))
        matches = [cell for cell in self.cells if cell.support_geometry.covers(point)]
        if not matches:
            return DisplacementSample("OUTSIDE_FIELD", self.field_id, None,
                                      None, None, None, None)
        if len(matches) != 1:
            return DisplacementSample("AMBIGUOUS_OVERLAPPING_CELLS", self.field_id,
                                      None, None, None, None, None)
        cell = matches[0]
        accepted = ((cell.correction_dx_m_candidate, cell.correction_dy_m_candidate)
                    if cell.accepted else None)
        return DisplacementSample(
            cell.status, self.field_id, cell.cell_id,
            cell.correction_dx_m_candidate, cell.correction_dy_m_candidate,
            cell.fit_dispersion_m, accepted,
        )


def group_fields(cells: Iterable[tuple[str, str, DisplacementCell]]) -> list[LocalDisplacementField]:
    """Group ``(reference date, moving date, cell)`` rows by actual date pair."""
    grouped: dict[tuple[str, str], list[DisplacementCell]] = {}
    for reference_date, moving_date, cell in cells:
        grouped.setdefault((reference_date, moving_date), []).append(cell)
    result = []
    for (reference_date, moving_date), field_cells in sorted(grouped.items()):
        field_id = f"{reference_date}_to_{moving_date}"
        result.append(LocalDisplacementField(field_id, reference_date, moving_date,
                                             tuple(field_cells)))
    return result
