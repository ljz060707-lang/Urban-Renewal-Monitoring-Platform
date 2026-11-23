"""Fail-closed human review adapter for local image-grid tie cells."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import fiona
from pyproj import CRS
from shapely.geometry import Point, shape

from .field import LocalDisplacementField


LAYER_NAME = "reviewed_image_alignment_cells"
REQUIRED_FIELDS = {
    "field_id", "cell_id", "moving_date", "reference_source",
    "reference_source_sha256", "candidate_source_ref", "correction_dx_m",
    "correction_dy_m", "independent_ground_evidence", "ground_evidence_type",
    "reviewed", "positional_uncertainty_m", "reviewer_id", "reviewed_at",
    "evidence_path", "evidence_sha256",
}
RELATIVE_REVIEW_FIELDS = {
    "review_basis", "alignment_evidence_type",
    "relative_registration_uncertainty_m", "nonbuilding_tie_pairs",
    "holdout_tie_pairs", "relative_ties_within_support_cell",
    "positional_uncertainty_semantics", "uncertainty_semantics",
    "fit_max_adjusted_bound_m",
    "holdout_max_adjusted_bound_m",
}
GROUND_EVIDENCE_TYPES = {
    "SURVEY_CONTROL", "INDEPENDENT_ORTHO", "AUTHORITATIVE_MAP",
    "OTHER_INDEPENDENT_GROUND_CONTROL",
}
ALIGNMENT_REVIEW_BASES = {
    "INDEPENDENT_GROUND_CONTROL", "RELATIVE_REFERENCE_IMAGE_FEATURES",
}
RELATIVE_ALIGNMENT_EVIDENCE_TYPE = "STABLE_NONBUILDING_IMAGE_FEATURE_TIES"
RELATIVE_REVIEW_CONTRACT_AMENDMENT = "R5_RELATIVE_ALIGNMENT_REVIEW_CONTRACT_AMENDMENT_V2"
RELATIVE_UNCERTAINTY_RULE = "R5_RELATIVE_ALIGNMENT_UNCERTAINTY_RULE_V1"
RELATIVE_TIE_SIDECAR_SCHEMA = "xian_r5.relative_alignment_tie_sidecar.v2"
RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS = "RELATIVE_IMAGE_GRID_ONLY"
TIE_NOT_PRECISION_ELIGIBLE = "TIE_NOT_PRECISION_ELIGIBLE"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_LOCALIZATION_DISTANCE_RE = re.compile(
    r"\b(?:within|at most|no more than|less than|bounded to|localized to)\s+"
    r"\d+(?:\.\d+)?\s*(?:m|met(?:er|re)s?)\b",
    re.IGNORECASE,
)
_CONFIDENCE_ONLY_BASIS = {
    "high", "medium", "low", "high confidence", "medium confidence",
    "low confidence", "confident", "strong", "weak",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_sha256(geometry) -> str:
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        raise ValueError("Support geometry must be non-empty and valid before hashing")
    return hashlib.sha256(geometry.wkb).hexdigest()


def _source_components(path: Path) -> dict[str, str]:
    components = [path]
    components.extend(Path(f"{path}{suffix}") for suffix in ("-wal", "-shm", "-journal")
                      if Path(f"{path}{suffix}").is_file())
    return {str(component.resolve()): _sha256(component)
            for component in sorted(components)}


def _flag(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be stored as 0 or 1")


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reviewed_at is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be an ISO timestamp with timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone")
    return value


def _resolved(value: str, *, parent: Path | None = None) -> Path:
    path = Path(value)
    if not path.is_absolute() and parent is not None:
        path = parent / path
    return path.resolve()


def _finite_number(value: object, name: str, *, context: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be finite numeric for {context}")
    return float(value)


def _same_number(value: object, expected: float, name: str, *, context: str) -> None:
    actual = _finite_number(value, name, context=context)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} recomputation mismatch for {context}")


def _evidence_basis(value: object, name: str, *, context: str) -> str:
    if not isinstance(value, str) or len(value.strip()) < 24:
        raise ValueError(f"{name} must explain the evidence basis for {context}")
    normalized = " ".join(value.casefold().split())
    if normalized in _CONFIDENCE_ONLY_BASIS:
        raise ValueError(f"{name} cannot be only a confidence label for {context}")
    return value.strip()


def _localization_basis(value: object, *, context: str) -> str:
    basis = _evidence_basis(value, "localization_basis", context=context)
    if not _LOCALIZATION_DISTANCE_RE.search(basis):
        raise ValueError(
            "localization_basis must state an explicit numeric metre range and "
            f"how the feature supports it for {context}")
    return basis


def _recompute_relative_ties(
    sidecar_path: Path,
    *,
    field_id: str,
    cell_id: str,
    moving_date: str,
    candidate_source_ref: str,
    candidate_dx_m: float,
    candidate_dy_m: float,
    support_geometry,
    reference_path: Path,
    review_layer_path: Path,
    evidence_hashes: dict[str, str],
    review_sidecar_digest: str,
) -> dict:
    """Validate tie provenance and independently recompute the frozen bounds."""
    context = f"{field_id}/{cell_id}/{moving_date}"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Relative tie sidecar is not readable JSON for {context}") from exc
    if not isinstance(sidecar, dict):
        raise ValueError(f"Relative tie sidecar root must be an object for {context}")
    reference_path = reference_path.resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Relative tie reference raster does not exist: {reference_path}")
    reference_digest = _sha256(reference_path)
    expected_root = {
        "schema_version": RELATIVE_TIE_SIDECAR_SCHEMA,
        "review_contract_amendment": RELATIVE_REVIEW_CONTRACT_AMENDMENT,
        "uncertainty_rule": RELATIVE_UNCERTAINTY_RULE,
        "coordinates_crs": "EPSG:32649",
        "field_id": field_id,
        "cell_id": cell_id,
        "moving_date": moving_date,
        "candidate_source_ref": candidate_source_ref,
        "reference_source": str(reference_path),
        "reference_source_sha256": reference_digest,
        "support_geometry_sha256": _geometry_sha256(support_geometry),
    }
    allowed_root_fields = set(expected_root) | {
        "candidate_dx_m", "candidate_dy_m", "fit_max_adjusted_bound_m",
        "holdout_max_adjusted_bound_m", "relative_registration_uncertainty_m",
        "positional_uncertainty_m", "uncertainty_semantics",
        "absolute_position_accuracy", "ties",
    }
    if set(sidecar) != allowed_root_fields:
        raise ValueError(
            f"Relative tie sidecar fields do not match the frozen schema for {context}")
    for name, expected in expected_root.items():
        if sidecar.get(name) != expected:
            raise ValueError(f"Relative tie sidecar {name} mismatch for {context}")
    _same_number(sidecar.get("candidate_dx_m"), candidate_dx_m,
                 "candidate_dx_m", context=context)
    _same_number(sidecar.get("candidate_dy_m"), candidate_dy_m,
                 "candidate_dy_m", context=context)
    if sidecar.get("absolute_position_accuracy") != "UNKNOWN":
        raise ValueError(f"Relative tie sidecar must preserve absolute accuracy as UNKNOWN for {context}")
    if sidecar.get("uncertainty_semantics") != RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS:
        raise ValueError(f"Relative tie sidecar has invalid uncertainty semantics for {context}")

    ties = sidecar.get("ties")
    if not isinstance(ties, list) or not ties:
        raise ValueError(f"Relative tie sidecar must contain tie records for {context}")
    required_tie_fields = {
        "tie_id", "field_id", "cell_id", "moving_date", "FIT_or_HOLDOUT",
        "feature_type", "precision_eligible", "precision_ineligibility_reason",
        "ge_x", "ge_y", "reference_x",
        "reference_y", "candidate_dx_m", "candidate_dy_m",
        "residual_dx_m", "residual_dy_m", "residual_m",
        "ge_localization_bound_m", "reference_localization_bound_m",
        "adjusted_bound_m", "localization_basis", "evidence_path",
        "evidence_sha256", "within_exact_support", "independence_group",
        "feature_class", "stable_nonbuilding_ground_evidence",
        "stability_basis",
    }
    allowed_tie_fields = required_tie_fields
    tie_ids: set[str] = set()
    eligible_groups: set[str] = set()
    eligible_adjusted: dict[str, list[float]] = {"FIT": [], "HOLDOUT": []}
    tie_evidence_before: dict[str, str] = {}

    for index, tie in enumerate(ties):
        tie_context = f"{context}/tie[{index}]"
        if not isinstance(tie, dict):
            raise ValueError(f"Relative tie record must be an object for {tie_context}")
        missing = required_tie_fields - set(tie)
        if missing:
            raise ValueError(f"Relative tie record is missing fields for {tie_context}: {sorted(missing)}")
        extra = set(tie) - allowed_tie_fields
        if extra:
            raise ValueError(f"Relative tie record has unsupported fields for {tie_context}: {sorted(extra)}")
        tie_id = tie.get("tie_id")
        if not isinstance(tie_id, str) or not tie_id.strip() or tie_id.strip() in tie_ids:
            raise ValueError(f"Relative tie_id must be non-empty and unique for {tie_context}")
        tie_ids.add(tie_id.strip())
        for name, expected in (
            ("field_id", field_id),
            ("cell_id", cell_id),
            ("moving_date", moving_date),
        ):
            if tie.get(name) != expected:
                raise ValueError(f"Tie {name} mismatch for {tie_context}")
        role = tie.get("FIT_or_HOLDOUT")
        if role not in {"FIT", "HOLDOUT"}:
            raise ValueError(f"FIT_or_HOLDOUT must be FIT or HOLDOUT for {tie_context}")
        independence_group = tie.get("independence_group")
        if not isinstance(independence_group, str) or not independence_group.strip():
            raise ValueError(f"independence_group is required for {tie_context}")
        independence_group = independence_group.strip()
        feature_type = tie.get("feature_type")
        if not isinstance(feature_type, str) or len(feature_type.strip()) < 3:
            raise ValueError(f"feature_type must identify the observed feature for {tie_context}")
        if tie.get("feature_class") != "STABLE_NONBUILDING_GROUND_FEATURE":
            raise ValueError(f"Tie is not identified as stable non-building ground evidence for {tie_context}")
        if tie.get("stable_nonbuilding_ground_evidence") is not True:
            raise ValueError(f"Tie lacks stable non-building ground evidence for {tie_context}")
        _evidence_basis(tie.get("stability_basis"), "stability_basis", context=tie_context)
        localization_basis = tie.get("localization_basis")

        evidence_raw = tie.get("evidence_path")
        evidence_digest = tie.get("evidence_sha256")
        if not isinstance(evidence_raw, str) or not evidence_raw.strip():
            raise ValueError(f"Tie evidence_path is required for {tie_context}")
        if not isinstance(evidence_digest, str) or not _SHA256_RE.fullmatch(evidence_digest):
            raise ValueError(f"Tie evidence_sha256 must be a SHA-256 for {tie_context}")
        tie_evidence = _resolved(evidence_raw, parent=sidecar_path.parent)
        if tie_evidence in {sidecar_path, review_layer_path, reference_path}:
            raise ValueError(f"Tie evidence must be independent of the sidecar, review layer and reference for {tie_context}")
        evidence_key = str(tie_evidence)
        if evidence_key not in tie_evidence_before:
            if not tie_evidence.is_file():
                raise FileNotFoundError(f"Tie evidence artifact does not exist: {tie_evidence}")
            tie_evidence_before[evidence_key] = _sha256(tie_evidence)
        if tie_evidence_before[evidence_key].casefold() != evidence_digest.casefold():
            raise ValueError(f"Tie evidence SHA-256 mismatch for {tie_context}")
        evidence_hashes[evidence_key] = tie_evidence_before[evidence_key]

        ge_x = _finite_number(tie.get("ge_x"), "ge_x", context=tie_context)
        ge_y = _finite_number(tie.get("ge_y"), "ge_y", context=tie_context)
        reference_x = _finite_number(tie.get("reference_x"), "reference_x", context=tie_context)
        reference_y = _finite_number(tie.get("reference_y"), "reference_y", context=tie_context)
        _same_number(tie.get("candidate_dx_m"), candidate_dx_m,
                     "candidate_dx_m", context=tie_context)
        _same_number(tie.get("candidate_dy_m"), candidate_dy_m,
                     "candidate_dy_m", context=tie_context)
        residual_dx = (ge_x + candidate_dx_m) - reference_x
        residual_dy = (ge_y + candidate_dy_m) - reference_y
        residual_m = math.hypot(residual_dx, residual_dy)
        _same_number(tie.get("residual_dx_m"), residual_dx,
                     "residual_dx_m", context=tie_context)
        _same_number(tie.get("residual_dy_m"), residual_dy,
                     "residual_dy_m", context=tie_context)
        _same_number(tie.get("residual_m"), residual_m,
                     "residual_m", context=tie_context)
        points_inside = all(support_geometry.covers(point) for point in (
            Point(ge_x, ge_y),
            Point(reference_x, reference_y),
            Point(ge_x + candidate_dx_m, ge_y + candidate_dy_m),
        ))
        if tie.get("within_exact_support") is not points_inside:
            raise ValueError(f"within_exact_support does not match exact support geometry for {tie_context}")
        if not points_inside:
            raise ValueError(f"All relative ties must lie inside exact support geometry for {tie_context}")

        precision_eligible = tie.get("precision_eligible")
        if not isinstance(precision_eligible, bool):
            raise ValueError(f"precision_eligible must be boolean for {tie_context}")
        ineligibility_reason = tie.get("precision_ineligibility_reason")
        if precision_eligible:
            _localization_basis(localization_basis, context=tie_context)
            if ineligibility_reason not in (None, ""):
                raise ValueError(f"Precision-eligible tie cannot have an ineligibility reason for {tie_context}")
            ge_bound = _finite_number(
                tie.get("ge_localization_bound_m"), "ge_localization_bound_m",
                context=tie_context)
            reference_bound = _finite_number(
                tie.get("reference_localization_bound_m"), "reference_localization_bound_m",
                context=tie_context)
            if ge_bound < 0 or reference_bound < 0:
                raise ValueError(f"Localization bounds must be non-negative for {tie_context}")
            adjusted_bound = residual_m + ge_bound + reference_bound
            _same_number(tie.get("adjusted_bound_m"), adjusted_bound,
                         "adjusted_bound_m", context=tie_context)
            if tie.get("precision_ineligibility_reason") not in (None, ""):
                raise ValueError(f"Eligible tie cannot have a precision-ineligibility reason for {tie_context}")
            if independence_group in eligible_groups:
                raise ValueError(f"FIT/HOLDOUT ties are not independent: {independence_group}")
            eligible_groups.add(independence_group)
            eligible_adjusted[role].append(adjusted_bound)
        else:
            _evidence_basis(localization_basis, "localization_basis", context=tie_context)
            if ineligibility_reason != TIE_NOT_PRECISION_ELIGIBLE:
                raise ValueError(
                    f"Ineligible tie reason must be {TIE_NOT_PRECISION_ELIGIBLE} for {tie_context}")
            ge_value = tie.get("ge_localization_bound_m")
            reference_value = tie.get("reference_localization_bound_m")
            if ge_value is not None:
                ge_value = _finite_number(ge_value, "ge_localization_bound_m", context=tie_context)
                if ge_value < 0:
                    raise ValueError(f"Localization bounds must be non-negative for {tie_context}")
            if reference_value is not None:
                reference_value = _finite_number(
                    reference_value, "reference_localization_bound_m", context=tie_context)
                if reference_value < 0:
                    raise ValueError(f"Localization bounds must be non-negative for {tie_context}")
            if ge_value is not None and reference_value is not None:
                adjusted_bound = residual_m + ge_value + reference_value
                _same_number(tie.get("adjusted_bound_m"), adjusted_bound,
                             "adjusted_bound_m", context=tie_context)
            elif tie.get("adjusted_bound_m") is not None:
                raise ValueError(f"Ineligible tie without both bounds cannot have adjusted_bound_m for {tie_context}")

    if len(eligible_adjusted["FIT"]) < 3 or len(eligible_adjusted["HOLDOUT"]) < 3:
        raise ValueError(
            f"Relative review requires at least 3 independent precision-eligible FIT and HOLDOUT ties for {context}")
    fit_max = max(eligible_adjusted["FIT"])
    holdout_max = max(eligible_adjusted["HOLDOUT"])
    if fit_max > 2.0 or holdout_max > 2.0:
        raise ValueError(f"FIT/HOLDOUT maximum adjusted bound exceeds 2 m for {context}")
    if holdout_max < 0.0:
        raise ValueError(f"Computed relative registration uncertainty is negative for {context}")
    relative_uncertainty = holdout_max
    for name, expected in (
        ("fit_max_adjusted_bound_m", fit_max),
        ("holdout_max_adjusted_bound_m", holdout_max),
        ("relative_registration_uncertainty_m", relative_uncertainty),
        ("positional_uncertainty_m", relative_uncertainty),
    ):
        _same_number(sidecar.get(name), expected, name, context=context)
    return {
        "fit_max_adjusted_bound_m": fit_max,
        "holdout_max_adjusted_bound_m": holdout_max,
        "relative_registration_uncertainty_m": relative_uncertainty,
        "positional_uncertainty_m": relative_uncertainty,
        "positional_uncertainty_semantics": RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS,
        "uncertainty_semantics": RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS,
        "fit_tie_count": len(eligible_adjusted["FIT"]),
        "holdout_tie_count": len(eligible_adjusted["HOLDOUT"]),
        "tie_count": len(ties),
        "relative_ties_within_support_cell": True,
        "absolute_position_accuracy": "UNKNOWN",
        "sidecar_schema_version": RELATIVE_TIE_SIDECAR_SCHEMA,
        "sidecar_sha256": review_sidecar_digest.lower(),
    }


def attach_reviewed_alignment_cells(
    fields: Iterable[LocalDisplacementField],
    source_path: Path,
    *,
    layer: str = LAYER_NAME,
) -> tuple[list[LocalDisplacementField], dict]:
    """Attach hash-verified review attestations to exact candidate field cells.

    A review can authorize only the unchanged candidate vector and support
    geometry it names. It does not itself apply a correction or certify a
    roof-to-ground offset; the evidence file and reviewer attestation remain
    explicit in the returned provenance report.
    """
    original_fields = list(fields)
    if not original_fields:
        raise ValueError("At least one candidate alignment field is required")
    source_path = Path(source_path).resolve()
    if source_path.suffix.casefold() != ".gpkg" or not source_path.is_file():
        raise FileNotFoundError(f"Reviewed alignment input must be an existing GeoPackage: {source_path}")
    source_hashes_before = _source_components(source_path)
    evidence_hashes_before: dict[str, str] = {}
    reference_hashes_before: dict[str, str] = {}
    field_cells: dict[tuple[str, str, str], tuple[int, int]] = {}
    for field_index, alignment_field in enumerate(original_fields):
        if not alignment_field.reference_source:
            continue
        reference = str(_resolved(alignment_field.reference_source))
        for cell_index, cell in enumerate(alignment_field.cells):
            key = (alignment_field.field_id, cell.cell_id, reference.casefold())
            if key in field_cells:
                raise ValueError(f"Candidate alignment field/cell key is not unique: {key}")
            field_cells[key] = (field_index, cell_index)

    replacements = {}
    review_rows = []
    source_feature_count = 0
    source_schema = None
    with fiona.open(source_path, layer=layer) as source:
        source_crs_input = source.crs_wkt or source.crs
        if not source_crs_input or CRS.from_user_input(source_crs_input).to_epsg() != 32649:
            raise ValueError("Reviewed alignment layer must use EPSG:32649")
        source_schema = {
            "geometry": source.schema.get("geometry"),
            "properties": dict(source.schema.get("properties", {})),
        }
        missing = REQUIRED_FIELDS - set(source_schema["properties"])
        if missing:
            raise ValueError(f"Reviewed alignment layer is missing fields: {sorted(missing)}")
        relative_fields_present = RELATIVE_REVIEW_FIELDS & set(source_schema["properties"])
        if relative_fields_present and relative_fields_present != RELATIVE_REVIEW_FIELDS:
            raise ValueError(
                "Reviewed alignment layer has an incomplete relative-review contract: "
                f"{sorted(RELATIVE_REVIEW_FIELDS - relative_fields_present)}")
        has_review_basis = relative_fields_present == RELATIVE_REVIEW_FIELDS
        if source_schema["geometry"] not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Reviewed alignment cells must have polygon support geometry")
        source_feature_count = len(source)
        seen = set()
        for feature in source:
            props = dict(feature["properties"])
            if "accepted" in props and not _flag(props["accepted"], "accepted"):
                raise ValueError("Accepted-only alignment layer cannot contain pending or rejected rows")
            if str(props.get("review_state", "")).upper() in {"PENDING", "REJECTED", "UNRESOLVED"}:
                raise ValueError("Accepted-only alignment layer requires a completed accepted review")
            field_id = props.get("field_id")
            cell_id = props.get("cell_id")
            reference_raw = props.get("reference_source")
            if not all(isinstance(value, str) and value.strip()
                       for value in (field_id, cell_id, reference_raw)):
                raise ValueError("Reviewed alignment rows require field, cell and reference IDs")
            reference = _resolved(reference_raw, parent=source_path.parent)
            key = (field_id, cell_id, str(reference).casefold())
            if key not in field_cells:
                raise ValueError(f"Reviewed alignment row references unknown field/cell: {key}")
            if key in seen:
                raise ValueError(f"Duplicate reviewed alignment row: {key}")
            seen.add(key)
            field_index, cell_index = field_cells[key]
            alignment_field = original_fields[field_index]
            cell = alignment_field.cells[cell_index]
            if alignment_field.moving_date != props.get("moving_date"):
                raise ValueError(f"Reviewed alignment actual date mismatch: {key}")
            if str(_resolved(alignment_field.reference_source)) != str(reference):
                raise ValueError(f"Reviewed alignment reference path mismatch: {key}")
            if cell.accepted:
                raise ValueError(f"Alignment cell is already reviewed/accepted: {key}")
            if cell.correction_dx_m_candidate is None or cell.correction_dy_m_candidate is None:
                raise ValueError(f"Cannot review an alignment cell without a numeric candidate: {key}")
            if (not cell.source_component_scoped
                    and not cell.relative_cell_review_candidate):
                raise ValueError(
                    f"Cannot review an alignment cell without single-source-component support or an eligible composed-cell relative candidate: {key}")
            candidate_ref = props.get("candidate_source_ref")
            if candidate_ref != cell.source_ref:
                raise ValueError(f"Reviewed alignment candidate provenance mismatch: {key}")
            for name, expected in (("correction_dx_m", cell.correction_dx_m_candidate),
                                   ("correction_dy_m", cell.correction_dy_m_candidate)):
                value = props.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite numeric for {key}")
                if not math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-5):
                    raise ValueError(f"Reviewed alignment vector differs from its candidate: {key}/{name}")

            independent = _flag(props.get("independent_ground_evidence"), "independent_ground_evidence")
            reviewed = _flag(props.get("reviewed"), "reviewed")
            if not reviewed:
                raise ValueError(f"Alignment cell requires completed human review: {key}")
            if has_review_basis:
                review_basis = props.get("review_basis")
                if review_basis not in ALIGNMENT_REVIEW_BASES:
                    raise ValueError(f"Unsupported or missing alignment review basis for {key}: {review_basis!r}")
            else:
                # v1 review layers are retained as an absolute-control-only adapter.
                review_basis = "INDEPENDENT_GROUND_CONTROL"
            if (not cell.source_component_scoped
                    and review_basis != "RELATIVE_REFERENCE_IMAGE_FEATURES"):
                raise ValueError(
                    f"Unknown source-component scope permits relative cell review only: {key}")
            alignment_evidence_type = (
                props.get("alignment_evidence_type") if has_review_basis else None)
            ground_evidence_type = props.get("ground_evidence_type")
            submitted_positional_uncertainty = props.get("positional_uncertainty_m")
            if submitted_positional_uncertainty is not None:
                if (isinstance(submitted_positional_uncertainty, bool)
                        or not isinstance(submitted_positional_uncertainty, (int, float))):
                    raise ValueError(f"positional_uncertainty_m must be numeric or null for {key}")
                submitted_positional_uncertainty = float(submitted_positional_uncertainty)
                if (not math.isfinite(submitted_positional_uncertainty)
                        or not 0 <= submitted_positional_uncertainty <= 2.0):
                    raise ValueError(f"positional_uncertainty_m must be between 0 and 2 m for {key}")

            absolute_uncertainty = None
            relative_uncertainty = None
            nonbuilding_tie_pairs = 0
            holdout_tie_pairs = 0
            ties_within_cell = False
            relative_metrics = None
            if review_basis == "INDEPENDENT_GROUND_CONTROL":
                if not independent:
                    raise ValueError(f"Ground-control review basis requires independent evidence: {key}")
                if ground_evidence_type not in GROUND_EVIDENCE_TYPES:
                    raise ValueError(
                        f"Unsupported independent ground evidence type for {key}: {ground_evidence_type!r}")
                if submitted_positional_uncertainty is None:
                    raise ValueError(f"Ground-control review requires positional_uncertainty_m for {key}")
                absolute_uncertainty = submitted_positional_uncertainty
                if has_review_basis and alignment_evidence_type != "INDEPENDENT_GROUND_CONTROL":
                    raise ValueError(f"Ground-control alignment_evidence_type mismatch for {key}")
                if has_review_basis and props.get("relative_registration_uncertainty_m") is not None:
                    raise ValueError(f"Ground-control review cannot populate relative-only uncertainty for {key}")
                if (has_review_basis
                        and props.get("positional_uncertainty_semantics")
                        not in (None, "", "ABSOLUTE_GROUND_CONTROL")):
                    raise ValueError(f"Ground-control positional uncertainty semantics mismatch for {key}")
                if (has_review_basis
                        and props.get("uncertainty_semantics")
                        not in (None, "", "ABSOLUTE_GROUND_CONTROL")):
                    raise ValueError(f"Ground-control uncertainty semantics mismatch for {key}")
            elif review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES":
                if independent:
                    raise ValueError(f"Relative image-tie review cannot claim independent ground evidence: {key}")
                if ground_evidence_type not in (None, ""):
                    raise ValueError(f"Relative image-tie review cannot name ground_evidence_type: {key}")
                if submitted_positional_uncertainty is None:
                    raise ValueError(
                        f"Relative image-tie review must populate positional_uncertainty_m from the tie sidecar: {key}")
                if alignment_evidence_type != RELATIVE_ALIGNMENT_EVIDENCE_TYPE:
                    raise ValueError(
                        f"Relative review requires {RELATIVE_ALIGNMENT_EVIDENCE_TYPE} for {key}")
                if (props.get("positional_uncertainty_semantics")
                        != RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS):
                    raise ValueError(
                        f"Relative review positional uncertainty must be {RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS}: {key}")
                if props.get("uncertainty_semantics") != RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS:
                    raise ValueError(
                        f"Relative review uncertainty must be {RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS}: {key}")
                relative_uncertainty = props.get("relative_registration_uncertainty_m")
                if (isinstance(relative_uncertainty, bool)
                        or not isinstance(relative_uncertainty, (int, float))):
                    raise ValueError(
                        f"relative_registration_uncertainty_m must be numeric for {key}")
                relative_uncertainty = float(relative_uncertainty)
                if not math.isfinite(relative_uncertainty) or not 0 <= relative_uncertainty <= 2.0:
                    raise ValueError(
                        f"relative_registration_uncertainty_m must be between 0 and 2 m for {key}")
                for name in ("nonbuilding_tie_pairs", "holdout_tie_pairs"):
                    value = props.get(name)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"{name} must be a non-negative integer for {key}")
                nonbuilding_tie_pairs = int(props["nonbuilding_tie_pairs"])
                holdout_tie_pairs = int(props["holdout_tie_pairs"])
                ties_within_cell = _flag(
                    props.get("relative_ties_within_support_cell"),
                    "relative_ties_within_support_cell")
                if not ties_within_cell:
                    raise ValueError(
                        f"Relative tie and holdout pairs must be reviewed inside the exact support cell: {key}")
            else:
                raise ValueError(f"Unsupported alignment review basis for {key}: {review_basis!r}")

            reviewer = props.get("reviewer_id")
            if not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError(f"reviewer_id is required for {key}")
            reviewed_at = _timestamp(props.get("reviewed_at"))
            reference_digest = props.get("reference_source_sha256")
            if not isinstance(reference_digest, str) or not _SHA256_RE.fullmatch(reference_digest):
                raise ValueError(f"reference_source_sha256 must be a SHA-256 for {key}")
            if not reference.is_file():
                raise FileNotFoundError(f"Reviewed alignment reference raster does not exist: {reference}")
            reference_key = str(reference)
            if reference_key not in reference_hashes_before:
                reference_hashes_before[reference_key] = _sha256(reference)
            if reference_hashes_before[reference_key].casefold() != reference_digest.casefold():
                raise ValueError(f"Reference raster SHA-256 mismatch for {key}")

            evidence_path_raw = props.get("evidence_path")
            evidence_digest = props.get("evidence_sha256")
            if not isinstance(evidence_path_raw, str) or not evidence_path_raw.strip():
                raise ValueError(f"evidence_path is required for {key}")
            if not isinstance(evidence_digest, str) or not _SHA256_RE.fullmatch(evidence_digest):
                raise ValueError(f"evidence_sha256 must be a 64-character SHA-256 for {key}")
            evidence_path = _resolved(evidence_path_raw, parent=source_path.parent)
            if str(evidence_path) in source_hashes_before or evidence_path == reference:
                raise ValueError(f"Review layer and reference raster cannot be independent evidence for {key}")
            if not evidence_path.is_file():
                raise FileNotFoundError(f"Alignment review evidence does not exist: {evidence_path}")
            evidence_key = str(evidence_path)
            if evidence_key not in evidence_hashes_before:
                evidence_hashes_before[evidence_key] = _sha256(evidence_path)
            if evidence_hashes_before[evidence_key].casefold() != evidence_digest.casefold():
                raise ValueError(f"Alignment review evidence SHA-256 mismatch for {key}")

            if not feature.get("geometry"):
                raise ValueError(f"Reviewed alignment support geometry is missing for {key}")
            reviewed_geometry = shape(feature["geometry"])
            if (reviewed_geometry.is_empty or not reviewed_geometry.is_valid
                    or reviewed_geometry.geom_type not in {"Polygon", "MultiPolygon"}
                    or not reviewed_geometry.equals_exact(cell.support_geometry, 1e-9)):
                raise ValueError(f"Reviewed alignment support differs from the candidate cell: {key}")

            if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES":
                relative_metrics = _recompute_relative_ties(
                    evidence_path,
                    field_id=field_id,
                    cell_id=cell_id,
                    moving_date=alignment_field.moving_date,
                    candidate_source_ref=cell.source_ref,
                    candidate_dx_m=float(cell.correction_dx_m_candidate),
                    candidate_dy_m=float(cell.correction_dy_m_candidate),
                    support_geometry=cell.support_geometry,
                    reference_path=reference,
                    review_layer_path=source_path,
                    evidence_hashes=evidence_hashes_before,
                    review_sidecar_digest=evidence_digest,
                )
                for name in (
                    "fit_max_adjusted_bound_m", "holdout_max_adjusted_bound_m",
                    "relative_registration_uncertainty_m",
                ):
                    _same_number(props.get(name), relative_metrics[name], name, context=str(key))
                _same_number(
                    submitted_positional_uncertainty,
                    relative_metrics["relative_registration_uncertainty_m"],
                    "positional_uncertainty_m", context=str(key))
                if (relative_metrics["positional_uncertainty_m"]
                        != relative_metrics["relative_registration_uncertainty_m"]):
                    raise ValueError(f"Relative positional uncertainty recomputation mismatch for {key}")
                if (nonbuilding_tie_pairs != relative_metrics["fit_tie_count"]
                        or holdout_tie_pairs != relative_metrics["holdout_tie_count"]):
                    raise ValueError(f"FIT/HOLDOUT tie counts differ from the sidecar for {key}")
                if not ties_within_cell or not relative_metrics["relative_ties_within_support_cell"]:
                    raise ValueError(f"Relative ties are not all within exact support for {key}")
                relative_uncertainty = relative_metrics["relative_registration_uncertainty_m"]

            relative_cell_scope_authorized = bool(
                not cell.source_component_scoped
                and cell.relative_cell_review_candidate
                and review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES"
                and ties_within_cell)
            metadata = {
                **cell.metadata,
                "reviewer_id": reviewer.strip(),
                "reviewed_at": reviewed_at,
                "review_basis": review_basis,
                "review_evidence_type": alignment_evidence_type or ground_evidence_type,
                "alignment_evidence_type": alignment_evidence_type or ground_evidence_type,
                "ground_evidence_type": ground_evidence_type or None,
                "review_evidence_path": str(evidence_path),
                "review_evidence_sha256": evidence_digest.lower(),
                "review_reference_sha256": reference_digest.lower(),
                "absolute_positional_uncertainty_m": absolute_uncertainty,
                "relative_grid_positional_uncertainty_m": (
                    relative_uncertainty
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "relative_grid_positional_uncertainty_semantics": (
                    RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "uncertainty_semantics": (
                    RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "absolute_position_accuracy": (
                    "UNKNOWN"
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else
                    ("SUPPORTED_BY_REVIEWED_GROUND_CONTROL"
                     if review_basis == "INDEPENDENT_GROUND_CONTROL" else "UNKNOWN")),
                "relative_uncertainty_rule": (
                    RELATIVE_UNCERTAINTY_RULE
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "relative_fit_max_adjusted_bound_m": (
                    relative_metrics["fit_max_adjusted_bound_m"] if relative_metrics else None),
                "relative_holdout_max_adjusted_bound_m": (
                    relative_metrics["holdout_max_adjusted_bound_m"] if relative_metrics else None),
                "relative_tie_sidecar_schema_version": (
                    relative_metrics["sidecar_schema_version"] if relative_metrics else None),
                "relative_registration_uncertainty_m": relative_uncertainty,
                "nonbuilding_tie_pairs": nonbuilding_tie_pairs,
                "holdout_tie_pairs": holdout_tie_pairs,
                "candidate_status": cell.metadata.get("candidate_status", cell.status),
                "relative_ties_within_support_cell": (
                    ties_within_cell if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else False),
                "relative_cell_source_scope_review_authorized": relative_cell_scope_authorized,
                "relative_cell_source_scope_review_semantics": (
                    "EXACT_COMPOSED_CELL_RELATIVE_TIES_ONLY_SOURCE_COMPONENT_AND_ABSOLUTE_POSITION_UNKNOWN"
                    if relative_cell_scope_authorized else None),
            }
            replacements[(field_index, cell_index)] = replace(
                cell,
                status=("ACCEPTED_REVIEWED_GROUND" if review_basis == "INDEPENDENT_GROUND_CONTROL"
                        else "ACCEPTED_REVIEWED_RELATIVE_IMAGE_TIE"),
                reviewed=True, independent_ground_evidence=independent,
                positional_uncertainty_m=absolute_uncertainty,
                alignment_review_basis=review_basis,
                relative_registration_uncertainty_m=relative_uncertainty,
                metadata=metadata)
            review_rows.append({
                "field_id": field_id, "cell_id": cell_id,
                "moving_date": alignment_field.moving_date,
                "reference_source": str(reference),
                "candidate_source_ref": cell.source_ref,
                "correction_dx_m": float(cell.correction_dx_m_candidate),
                "correction_dy_m": float(cell.correction_dy_m_candidate),
                "review_basis": review_basis,
                "alignment_evidence_type": alignment_evidence_type or "INDEPENDENT_GROUND_CONTROL",
                "positional_uncertainty_m": absolute_uncertainty,
                "relative_grid_positional_uncertainty_m": (
                    relative_uncertainty
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "relative_grid_positional_uncertainty_semantics": (
                    RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "uncertainty_semantics": (
                    RELATIVE_POSITIONAL_UNCERTAINTY_SEMANTICS
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "absolute_position_accuracy": (
                    "UNKNOWN"
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else
                    ("SUPPORTED_BY_REVIEWED_GROUND_CONTROL"
                     if review_basis == "INDEPENDENT_GROUND_CONTROL" else "UNKNOWN")),
                "relative_uncertainty_rule": (
                    RELATIVE_UNCERTAINTY_RULE
                    if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else None),
                "fit_max_adjusted_bound_m": (
                    relative_metrics["fit_max_adjusted_bound_m"] if relative_metrics else None),
                "holdout_max_adjusted_bound_m": (
                    relative_metrics["holdout_max_adjusted_bound_m"] if relative_metrics else None),
                "relative_registration_uncertainty_m": relative_uncertainty,
                "nonbuilding_tie_pairs": nonbuilding_tie_pairs,
                "holdout_tie_pairs": holdout_tie_pairs,
                "relative_ties_within_support_cell": (
                    ties_within_cell if review_basis == "RELATIVE_REFERENCE_IMAGE_FEATURES" else False),
                "relative_cell_source_scope_review_authorized": relative_cell_scope_authorized,
                "ground_evidence_type": ground_evidence_type or None,
                "reviewer_id": reviewer.strip(), "reviewed_at": reviewed_at,
                "evidence_path": evidence_key,
                "evidence_sha256": evidence_digest.lower(),
            })

    reviewed_fields = []
    for field_index, alignment_field in enumerate(original_fields):
        reviewed_cells = tuple(
            replacements.get((field_index, cell_index), cell)
            for cell_index, cell in enumerate(alignment_field.cells))
        reviewed_fields.append(replace(alignment_field, cells=reviewed_cells))

    source_hashes_after = _source_components(source_path)
    evidence_hashes_after = {path: _sha256(Path(path)) for path in evidence_hashes_before}
    reference_hashes_after = {path: _sha256(Path(path)) for path in reference_hashes_before}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(f"Reviewed alignment source changed while being read: {source_path}")
    if evidence_hashes_after != evidence_hashes_before:
        raise RuntimeError("Alignment review evidence changed while being read")
    if reference_hashes_after != reference_hashes_before:
        raise RuntimeError("Reference raster changed while alignment cells were reviewed")

    integrity = {**source_hashes_after, **evidence_hashes_after, **reference_hashes_after}
    report = {
        "schema_version": "xian_r5.reviewed_image_alignment_cells.v3",
        "source_path": str(source_path), "source_layer": layer,
        "source_crs": "EPSG:32649", "source_feature_count": source_feature_count,
        "source_schema": source_schema,
        "accepted_cell_count": len(review_rows),
        "absolute_ground_control_cell_count": sum(
            row["review_basis"] == "INDEPENDENT_GROUND_CONTROL" for row in review_rows),
        "relative_reference_feature_cell_count": sum(
            row["review_basis"] == "RELATIVE_REFERENCE_IMAGE_FEATURES" for row in review_rows),
        "relative_cell_source_scope_authorization_count": sum(
            row["relative_cell_source_scope_review_authorized"] for row in review_rows),
        "review_status": ("ACCEPTED_REVIEWED_CELLS" if review_rows else
                          "NO_REVIEWABLE_ALIGNMENT_EVIDENCE"),
        "absolute_position_accuracy_claimed": bool(review_rows) and all(
            row["review_basis"] == "INDEPENDENT_GROUND_CONTROL" for row in review_rows),
        "reviewed_cells": review_rows,
        "source_component_sha256": source_hashes_before,
        "independent_evidence_sha256": evidence_hashes_before,
        "reference_raster_sha256": reference_hashes_before,
        "integrity_files_sha256": integrity,
        "source_hash_recheck": "PASS", "evidence_hash_recheck": "PASS",
        "field_application_performed": False,
        "semantics": "HUMAN_REVIEW_ATTESTATION_AND_HASH_VERIFIED_LINEAGE_NOT_AUTOMATIC_GROUND_TRUTH",
    }
    return reviewed_fields, report


def recheck_reviewed_alignment_sources(report: dict) -> dict[str, str]:
    """Verify the review layer, reference raster and evidence stayed unchanged."""
    expected = report.get("integrity_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Reviewed alignment report has no integrity file inventory")
    actual = {path: _sha256(Path(path)) for path in expected}
    if actual != expected:
        raise RuntimeError("Reviewed alignment, reference raster or evidence changed during run")
    return actual
