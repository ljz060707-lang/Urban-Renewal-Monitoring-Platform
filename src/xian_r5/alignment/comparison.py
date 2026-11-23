"""Candidate-only comparison of alternate registration paths into one grid."""

from __future__ import annotations

import math
from statistics import median

from .field import LocalDisplacementField


def compare_alignment_fields(
    direct: LocalDisplacementField,
    alternate: LocalDisplacementField,
) -> dict:
    """Compare same-date, same-reference cell vectors without accepting either.

    The result measures pathway consistency only; it is not positional accuracy
    because the paths may share image evidence and neither supplies ground control.
    """
    if (direct.moving_date != alternate.moving_date
            or direct.reference_date != alternate.reference_date
            or direct.reference_label != alternate.reference_label
            or direct.reference_date_status != alternate.reference_date_status
            or direct.crs != alternate.crs):
        raise ValueError("Alignment fields do not describe the same date and reference grid")
    direct_cells = {cell.cell_id: cell for cell in direct.cells}
    alternate_cells = {cell.cell_id: cell for cell in alternate.cells}
    if direct_cells.keys() != alternate_cells.keys():
        raise ValueError("Alignment fields have different spatial cell coverage")

    rows = []
    distances = []
    for cell_id in sorted(direct_cells):
        direct_cell, alternate_cell = direct_cells[cell_id], alternate_cells[cell_id]
        if not direct_cell.support_geometry.equals(alternate_cell.support_geometry):
            raise ValueError(f"Alignment support geometries differ: {cell_id}")
        direct_vector = (direct_cell.correction_dx_m_candidate,
                         direct_cell.correction_dy_m_candidate)
        alternate_vector = (alternate_cell.correction_dx_m_candidate,
                            alternate_cell.correction_dy_m_candidate)
        direct_numeric = direct_vector[0] is not None and direct_vector[1] is not None
        alternate_numeric = alternate_vector[0] is not None and alternate_vector[1] is not None
        distance = None
        if direct_numeric and alternate_numeric:
            distance = math.hypot(direct_vector[0] - alternate_vector[0],
                                  direct_vector[1] - alternate_vector[1])
            distances.append(distance)
            status = "BOTH_PATHS_HAVE_CANDIDATE"
        elif direct_numeric:
            status = "DIRECT_PATH_ONLY"
        elif alternate_numeric:
            status = "ALTERNATE_PATH_ONLY"
        else:
            status = "NO_NUMERIC_CANDIDATE"
        rows.append({
            "cell_id": cell_id,
            "direct_status": direct_cell.status,
            "alternate_status": alternate_cell.status,
            "direct_correction_m_candidate": list(direct_vector) if direct_numeric else None,
            "alternate_correction_m_candidate": list(alternate_vector) if alternate_numeric else None,
            "pathway_difference_m": distance,
            "status": status,
        })
    return {
        "schema_version": "xian_r5.alignment_pathway_comparison.v1",
        "moving_date": direct.moving_date,
        "reference_label": direct.reference_label,
        "direct_field_id": direct.field_id,
        "alternate_field_id": alternate.field_id,
        "cell_count": len(rows),
        "both_numeric_candidate_count": len(distances),
        "median_pathway_difference_m": median(distances) if distances else None,
        "max_pathway_difference_m": max(distances) if distances else None,
        "cells": rows,
        "interpretation": "PATHWAY_AGREEMENT_SCREEN_ONLY_NOT_POSITIONAL_ACCURACY",
        "correction_applied": False,
    }
