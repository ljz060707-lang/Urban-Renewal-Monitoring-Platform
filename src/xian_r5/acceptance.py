"""S4.14 read-only acceptance primitives; never fit or apply a transform.

This supplement can audit Shadow products without certifying their geometry.
Numerical diagnostics are deliberately not an authorization token. Existing V2
review/application adapters remain the only route to an applicable transform.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from .alignment.reviewed_field import (
    RELATIVE_UNCERTAINTY_RULE, attach_reviewed_alignment_cells,
    recheck_reviewed_alignment_sources,
)

CONTRACT = "XIAN_R5_S414_ACCEPTANCE_V1"
TASKS = ("IMAGE_TO_REFERENCE_REGISTRATION", "BUILDING_SPECIFIC_DISPLACEMENT",
         "ROOF_TO_GROUND_FOOTPRINT")
OBJECT_STATUSES = (
    "ACCEPTED_FOR_SHADOW_ALIGNMENT", "UNRESOLVED_INSUFFICIENT_TIES",
    "UNRESOLVED_HIGH_CHECK_RESIDUAL", "UNRESOLVED_SOURCE_SUPPORT",
    "UNRESOLVED_TEMPORAL_COMPATIBILITY", "UNRESOLVED_TRANSFORM_EXTRAPOLATION",
    "UNRESOLVED_BUILDING_PARALLAX", "REJECTED_INVALID_INPUT",
)
GATE_STATUSES = ("PASS", "FAIL", "UNRESOLVED", "NOT_RUN")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate(checks, *, executed=True):
    """Aggregate machine-derived checks, not reviewer-supplied PASS flags."""
    if not executed:
        return "NOT_RUN"
    if any(v is False for v in checks.values()):
        return "FAIL"
    if not checks or any(v is not True for v in checks.values()):
        return "UNRESOLVED"
    return "PASS"


def finite(value):
    return (not isinstance(value, (bool, np.bool_)) and
            isinstance(value, (int, float, np.number)) and math.isfinite(float(value)))


def vector(dx, dy):
    if dx is None and dy is None:
        return None
    if not finite(dx) or not finite(dy):
        raise ValueError("Vector must be a finite complete pair or two nulls")
    return [float(dx), float(dy)]


def tie_diagnostics(ties, dx, dy):
    """Recompute descriptive residuals and the additive V2 upper bound.

    This is NOT evidence admission: it does not interpret image pixels or
    verify localization claims. invoke checked_v2_review for that existing
    provenance adapter. Median/P90 never replace max adjusted bounds.
    """
    candidate = vector(dx, dy)
    empty = {"FIT": [], "HOLDOUT": []}
    if candidate is None and ties:
        raise ValueError("Ties cannot be scored against an unknown vector")
    groups, ids, bounds = set(), set(), {"FIT": [], "HOLDOUT": []}
    for tie in ties:
        role, uid, group = (tie.get(k) for k in ("FIT_or_HOLDOUT", "tie_id", "independence_group"))
        if role not in empty or not uid or uid in ids or not group:
            raise ValueError("Invalid role, duplicate tie, or missing independence group")
        ids.add(uid)
        if tie.get("precision_eligible") is not True:
            continue
        if group in groups:
            raise ValueError("FIT/CHECK spatial structure group reused")
        groups.add(group)
        if (tie.get("within_exact_support") is not True or
                tie.get("feature_class") != "STABLE_NONBUILDING_GROUND_FEATURE"):
            raise ValueError("Eligible tie needs exact support and non-building evidence")
        values = [tie.get(k) for k in ("ge_x", "ge_y", "reference_x", "reference_y",
                  "ge_localization_bound_m", "reference_localization_bound_m")]
        if not all(finite(v) for v in values) or min(values[4:]) < 0:
            raise ValueError("Coordinates/bounds must be finite; bounds nonnegative")
        x, y, rx, ry, gb, rb = values
        residual = math.hypot(x + dx - rx, y + dy - ry)
        adjusted = residual + gb + rb
        # Stored derived numbers, when present, must reproduce exactly.
        for key, result in (("residual_dx_m", x + dx - rx), ("residual_dy_m", y + dy - ry),
                            ("residual_m", residual), ("adjusted_bound_m", adjusted)):
            if key in tie and (not finite(tie[key]) or not math.isclose(tie[key], result, abs_tol=1e-9, rel_tol=0)):
                raise ValueError("Stored statistic recomputation mismatch: " + key)
        empty[role].append(residual)
        bounds[role].append(adjusted)
    fit, check = empty["FIT"], empty["HOLDOUT"]
    fit_max = max(bounds["FIT"]) if fit else None
    check_max = max(bounds["HOLDOUT"]) if check else None
    enough = len(fit) >= 3 and len(check) >= 3
    median = float(np.median(check)) if check else None
    p90 = float(np.percentile(check, 90, method="linear")) if check else None
    numerical = (enough and fit_max <= 2 and check_max <= 2 and median <= 2 and p90 <= 3)
    return dict(fit_count=len(fit), check_count=len(check), fit_residuals_m=fit,
                check_residuals_m=check, check_median_m=median, check_p90_m=p90,
                check_max_m=max(check) if check else None,
                fit_max_adjusted_bound_m=fit_max, holdout_max_adjusted_bound_m=check_max,
                relative_registration_uncertainty_m=check_max if enough else None,
                bound_rule=RELATIVE_UNCERTAINTY_RULE, bound_formula="norm(GE+candidate-reference)+GE_bound+reference_bound",
                uncertainty_semantics="RELATIVE_IMAGE_GRID_ONLY", ABSOLUTE_POSITION_ACCURACY="UNKNOWN",
                numerical_gate="PASS" if numerical else ("FAIL" if enough else "UNRESOLVED"),
                object_status=("UNRESOLVED_HIGH_CHECK_RESIDUAL" if enough and not numerical
                               else (None if enough else "UNRESOLVED_INSUFFICIENT_TIES")),
                authorization="NOT_AUTHORIZED_BY_DIAGNOSTICS", percentile_method="linear")


def checked_v2_review(fields, review_path):
    """Use original strict adapter, never a caller-written uncertainty/PASS."""
    reviewed, report = attach_reviewed_alignment_cells(fields, Path(review_path))
    recheck_reviewed_alignment_sources(report)
    return reviewed, report


def unresolved_building(record):
    """Audit an untransformed Shadow observation, without ID inheritance.

    Positive/application paths must use the existing transform adapter plus
    G2/G3 evidence, not this no-authority adapter. Fail if applied geometry is
    smuggled into its input.
    """
    uid = record.get("observation_id")
    geom = shape(record["geometry"])
    if (not uid or record.get("original_id") != uid or geom.is_empty or not geom.is_valid or
            sha256_geometry(geom) != record.get("geometry_sha256")):
        raise ValueError("Invalid original observation ID or geometry hash")
    if record.get("registered_geometry") is not None or record.get("ground_footprint") is not None:
        raise ValueError("Unverified input cannot contain applied geometry")
    reasons = []
    if record.get("source_support_complete") is not True:
        reasons.append("UNRESOLVED_SOURCE_SUPPORT")
    # Date continuity to reference is not proved by a coherent source date.
    reasons.append("UNRESOLVED_TEMPORAL_COMPATIBILITY")
    reasons.extend(["UNRESOLVED_INSUFFICIENT_TIES", "UNRESOLVED_BUILDING_PARALLAX"])
    return dict(observation_id=uid, original_id=uid, source_object_id=record.get("source_object_id"),
                original_geometry_sha256=record["geometry_sha256"],
                source_image=record.get("source_path"), source_image_sha256=record.get("source_RGB_sha256"),
                source_paths=record.get("source_paths", []), source_hashes=record.get("source_hashes", []),
                source_tile_ids=record.get("source_tile_ids", []),
                source_support_complete=record.get("source_support_complete"),
                source_support_uncovered_m2=record.get("source_support_uncovered_m2"),
                actual_date=record.get("actual_date"), temporal_compatibility="UNKNOWN",
                candidate_vector_m=None, validated_vector_m=None, applied_vector_m=None,
                displacement_source=None, transform_version=None, matching_evidence=None,
                uncertainty_m=None, uncertainty_method=None, extrapolation=None,
                crosses_local_field_boundary=None, involves_building_parallax="UNKNOWN",
                application_status=reasons[0], unresolved_reasons=reasons,
                building_displacement_evidence="BUILDING_PARALLAX_UNRESOLVED",
                physical_id=record.get("physical_id"), geometry_preserved=True,
                application_eligible=False)


def sha256_geometry(geometry):
    return hashlib.sha256(geometry.wkb).hexdigest()


def whole_object_field(geometry, fields):
    """Structural support check only, not field acceptance or transport.

    fields: id/support/vector tuples from already evidence-checked fields.
    Never split or blend a whole object. Any intersecting unknown/conflicting
    field prevents selection, even if another field covers it completely.
    """
    relevant = [(uid, support, v) for uid, support, v in fields
                if geometry.intersection(support).area > 0]
    if not relevant:
        return {"status": "UNRESOLVED_SOURCE_SUPPORT", "vector_m": None}
    vectors = [None if v is None else vector(*v) for _, _, v in relevant]
    if any(v is None for v in vectors) or any(v != vectors[0] for v in vectors):
        return {"status": "UNRESOLVED_TRANSFORM_EXTRAPOLATION", "vector_m": None}
    from shapely.ops import unary_union
    if not unary_union([s for _, s, _ in relevant]).covers(geometry):
        return {"status": "UNRESOLVED_SOURCE_SUPPORT", "vector_m": None}
    return {"status": "STRUCTURAL_SUPPORT_ONLY_NOT_AUTHORIZATION", "vector_m": vectors[0],
            "field_ids": sorted(uid for uid, _, _ in relevant)}


def hard_labels(fraction, eligible, *, threshold, operator, ignore=255):
    """Only materialize a pre-frozen hard rule on independently eligible pixels."""
    fraction, eligible = np.asarray(fraction), np.asarray(eligible)
    if eligible.dtype != bool or fraction.shape != eligible.shape or not finite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid label rule/eligibility")
    if operator not in (">", ">="):
        raise ValueError("Hard-label operator must be explicitly frozen")
    if np.any(eligible & (~np.isfinite(fraction) | (fraction < 0) | (fraction > 1))):
        raise ValueError("Eligible fraction outside [0,1]")
    labels = np.full(fraction.shape, ignore, dtype=np.uint8)
    labels[eligible] = ((fraction[eligible] > threshold) if operator == ">" else
                        (fraction[eligible] >= threshold)).astype(np.uint8)
    return labels
