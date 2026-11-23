"""Image-grid roof previews and fail-closed reviewed alignment application."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from shapely.affinity import translate
from shapely.geometry.base import BaseGeometry

from .field import LocalDisplacementField


_MIN_DOMINANT_CELL_ROOF_AREA_FRACTION = 0.8


@dataclass(frozen=True)
class RoofGridCandidate:
    """A roof in the original image grid or its separate aligned candidate."""

    geometry: BaseGeometry
    status: str
    cell_id: str | None
    correction_dx_m_candidate: float | None
    correction_dy_m_candidate: float | None
    grid_positional_uncertainty_m: float | None
    candidate_shift_rendered: bool
    cell_support_fraction: float | None = None
    relative_registration_uncertainty_m: float | None = None


@dataclass(frozen=True)
class RoofGridApplication:
    """Reviewed image-grid correction result; never a roof-to-ground estimate."""

    geometry: BaseGeometry
    status: str
    field_id: str | None = None
    cell_id: str | None = None
    correction_dx_m: float | None = None
    correction_dy_m: float | None = None
    positional_uncertainty_m: float | None = None
    applied: bool = False
    review_metadata: dict[str, object] | None = None
    relative_registration_uncertainty_m: float | None = None
    review_basis: str | None = None


def apply_accepted_roof_grid_alignment(
    roof: BaseGeometry,
    actual_date: str | None,
    fields: Iterable[LocalDisplacementField],
    *,
    reference_sources: Iterable[str],
) -> RoofGridApplication:
    """Apply one reviewed cell from a verified co-gridded reference stack.

    Candidate fields, partial-cell coverage, overlaps, and competing accepted
    cells fail closed. The input roof is never mutated; a successful result is
    still an image-grid roof proxy, not a ground footprint.
    """
    if roof.is_empty or not roof.is_valid or roof.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Roof candidates must be valid, non-empty polygon geometries")
    def normalize_source(value: str) -> str:
        return str(value).replace("\\", "/").rstrip("/").casefold()

    target_references = {normalize_source(source) for source in reference_sources if source}
    if not target_references:
        raise ValueError("At least one verified image-grid reference source is required")
    if not actual_date:
        return RoofGridApplication(roof, "ACTUAL_DATE_UNKNOWN")

    eligible = [field for field in fields
                if field.moving_date == actual_date
                and field.reference_source
                and normalize_source(field.reference_source) in target_references
                and field.correction_scope == "IMAGE_TO_REFERENCE_GRID"]
    if not eligible:
        return RoofGridApplication(roof, "NO_MATCHING_REVIEWED_ALIGNMENT_FIELD")

    accepted_hits = []
    ambiguous_accepted_support = False
    unreviewed_hit = False
    ambiguous_unreviewed_support = False
    any_spatial_support = False
    component_issues: set[str] = set()
    for field in eligible:
        intersecting = [cell for cell in field.cells if cell.support_geometry.intersects(roof)]
        if not intersecting:
            continue
        any_spatial_support = True
        covering = [cell for cell in intersecting if cell.support_geometry.covers(roof)]
        if len(intersecting) != 1 or len(covering) != 1:
            if any(cell.accepted for cell in intersecting):
                ambiguous_accepted_support = True
            else:
                ambiguous_unreviewed_support = True
            continue
        cell = covering[0]
        component_issue = cell.source_component_issue
        if (component_issue is not None
                and not cell.relative_cell_source_scope_review_authorized):
            component_issues.add(component_issue)
            continue
        if cell.accepted:
            accepted_hits.append((field, cell))
        else:
            unreviewed_hit = True

    if ambiguous_accepted_support or len(accepted_hits) > 1:
        return RoofGridApplication(roof, "AMBIGUOUS_ACCEPTED_ALIGNMENT_SUPPORT")
    if component_issues:
        priority = (
            "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED",
            "SOURCE_COMPONENT_COVERAGE_INCOMPLETE",
            "SOURCE_COMPONENT_ID_MISMATCH",
            "SOURCE_COMPONENT_SCOPE_UNKNOWN",
        )
        status = next((item for item in priority if item in component_issues),
                      "SOURCE_COMPONENT_SCOPE_UNKNOWN")
        return RoofGridApplication(roof, status)
    if len(accepted_hits) == 1:
        field, cell = accepted_hits[0]
        required_review_metadata = (
            "reviewer_id", "reviewed_at", "review_basis", "review_evidence_type",
            "review_evidence_path", "review_evidence_sha256", "review_reference_sha256",
        )
        if any(not cell.metadata.get(name) for name in required_review_metadata):
            return RoofGridApplication(roof, "ACCEPTED_ALIGNMENT_PROVENANCE_INCOMPLETE")
        dx = float(cell.correction_dx_m_candidate)
        dy = float(cell.correction_dy_m_candidate)
        review_metadata = {name: cell.metadata[name] for name in required_review_metadata}
        review_metadata.update({
            "alignment_evidence_type": cell.metadata.get("alignment_evidence_type"),
            "independent_ground_evidence": cell.independent_ground_evidence,
            "absolute_positional_uncertainty_m": cell.positional_uncertainty_m,
            "relative_registration_uncertainty_m": cell.relative_registration_uncertainty_m,
            "source_component_scope": cell.metadata.get("source_component_scope"),
            "reference_component_id_status": cell.metadata.get("reference_component_id_status"),
            "relative_cell_source_scope_review_authorized": (
                cell.relative_cell_source_scope_review_authorized),
        })
        return RoofGridApplication(
            translate(roof, xoff=dx, yoff=dy),
            "ACCEPTED_IMAGE_GRID_SHIFT_ROOF_NOT_GROUND",
            field.field_id, cell.cell_id, dx, dy,
            cell.positional_uncertainty_m, True,
            review_metadata,
            cell.relative_registration_uncertainty_m,
            cell.alignment_review_basis,
        )
    if ambiguous_unreviewed_support:
        status = "CROSSES_OR_OVERLAPS_ALIGNMENT_CELLS"
    elif unreviewed_hit:
        status = "UNREVIEWED_ALIGNMENT_CANDIDATE"
    elif any_spatial_support:
        status = "NO_APPLICABLE_ALIGNMENT_CELL"
    else:
        status = "OUTSIDE_ALIGNMENT_FIELD"
    return RoofGridApplication(roof, status)


def align_roof_candidate_to_reference_grid(
    roof: BaseGeometry,
    actual_date: str | None,
    field: LocalDisplacementField,
) -> RoofGridCandidate:
    """Render one roof's unaccepted image-grid shift without calling it ground.

    A shift is rendered when one numeric cell fully contains the roof. For an
    unreviewed field only, a building crossing cell edges can also receive a
    candidate from the unique cell covering at least 80% of its area. That is
    a whole-building translation for review, not a piecewise warp or an
    accepted correction. Reviewed fields remain fail-closed for partial-cell
    coverage. This operation never estimates roof-to-ground parallax.
    """
    if roof.is_empty or not roof.is_valid or roof.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Roof candidates must be valid, non-empty polygon geometries")
    if field.correction_scope != "IMAGE_TO_REFERENCE_GRID":
        raise ValueError("Roof candidate preview only accepts image-to-reference fields")
    if not actual_date:
        return RoofGridCandidate(roof, "ACTUAL_DATE_UNKNOWN", None, None, None, None, False)
    if actual_date != field.moving_date:
        return RoofGridCandidate(roof, "FIELD_ACTUAL_DATE_MISMATCH", None, None, None, None, False)

    intersecting = [cell for cell in field.cells if cell.support_geometry.intersects(roof)]
    covering = [cell for cell in field.cells if cell.support_geometry.covers(roof)]
    if not intersecting:
        return RoofGridCandidate(roof, "OUTSIDE_ALIGNMENT_FIELD", None, None, None, None, False)
    dominant_cell = False
    if len(intersecting) == 1 and len(covering) == 1 and intersecting[0] is covering[0]:
        cell = covering[0]
        support_fraction = 1.0
    else:
        ranked_support = sorted(
            ((float(roof.intersection(candidate_cell.support_geometry).area / roof.area),
              candidate_cell)
             for candidate_cell in intersecting),
            key=lambda item: item[0], reverse=True)
        if not ranked_support:
            return RoofGridCandidate(roof, "CROSSES_OR_OVERLAPS_ALIGNMENT_CELLS", None,
                                     None, None, None, False)
        support_fraction, cell = ranked_support[0]
        tied_dominant_cells = (
            len(ranked_support) > 1 and
            math.isclose(support_fraction, ranked_support[1][0], rel_tol=0.0, abs_tol=1e-9))
        if (support_fraction < _MIN_DOMINANT_CELL_ROOF_AREA_FRACTION
                or tied_dominant_cells or cell.accepted):
            return RoofGridCandidate(roof, "CROSSES_OR_OVERLAPS_ALIGNMENT_CELLS", None,
                                     None, None, None, False)
        support_fraction = min(1.0, support_fraction)
        dominant_cell = True

    if dominant_cell:
        source_component_ids = {
            candidate_cell.source_component_id_for_date(actual_date)
            for candidate_cell in intersecting
        }
        if None in source_component_ids or len(source_component_ids) != 1:
            issues = {candidate_cell.source_component_issue
                      for candidate_cell in intersecting
                      if candidate_cell.source_component_issue is not None}
            if "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED" in issues:
                status = "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED"
            elif "SOURCE_COMPONENT_COVERAGE_INCOMPLETE" in issues:
                status = "SOURCE_COMPONENT_COVERAGE_INCOMPLETE"
            elif "SOURCE_COMPONENT_ID_MISMATCH" in issues:
                status = "SOURCE_COMPONENT_ID_MISMATCH"
            elif "SOURCE_COMPONENT_SCOPE_UNKNOWN" in issues:
                status = "SOURCE_COMPONENT_SCOPE_UNKNOWN"
            else:
                status = "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED"
            return RoofGridCandidate(roof, status, cell.cell_id, None, None,
                                     cell.positional_uncertainty_m, False,
                                     support_fraction,
                                     cell.relative_registration_uncertainty_m)

    dx, dy = cell.correction_dx_m_candidate, cell.correction_dy_m_candidate
    if dx is None or dy is None:
        return RoofGridCandidate(roof, "NO_NUMERIC_ALIGNMENT_CANDIDATE", cell.cell_id,
                                 None, None, cell.positional_uncertainty_m, False,
                                 support_fraction, cell.relative_registration_uncertainty_m)
    component_issue = cell.source_component_issue
    if component_issue is not None:
        return RoofGridCandidate(roof, component_issue, cell.cell_id,
                                 None, None, cell.positional_uncertainty_m, False,
                                 support_fraction, cell.relative_registration_uncertainty_m)
    candidate = translate(roof, xoff=float(dx), yoff=float(dy))
    status = ("UNACCEPTED_DOMINANT_CELL_IMAGE_GRID_SHIFT_ROOF_CANDIDATE"
              if dominant_cell else
              "ACCEPTED_IMAGE_GRID_SHIFT_ROOF_NOT_GROUND" if cell.accepted else
              "UNACCEPTED_IMAGE_GRID_SHIFT_ROOF_CANDIDATE")
    return RoofGridCandidate(candidate, status, cell.cell_id, float(dx), float(dy),
                             cell.positional_uncertainty_m, True, support_fraction,
                             cell.relative_registration_uncertainty_m)
