"""Evidence-gated, actual-date building reconstruction.

Public exports are loaded on first use so importing a focused submodule such
as image registration does not require unrelated vector-I/O dependencies.
Candidate rooftop geometry and diagnostic image registration never become a
formal ground footprint without an independently verified footprint source.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AbsenceEvidence": (".core", "AbsenceEvidence"),
    "ExternalFootprintCandidate": (".core", "ExternalFootprintCandidate"),
    "FootprintEvidence": (".core", "FootprintEvidence"),
    "Observation": (".core", "Observation"),
    "Reconstruction": (".core", "Reconstruction"),
    "attach_overlapping_external_candidates": (
        ".external_candidates", "attach_overlapping_external_candidates"),
    "attach_rbf_sam_footprint_candidates": (
        ".rbf_sam_candidates", "attach_rbf_sam_footprint_candidates"),
    "polygonize_rbf_sam_roof_mask": (
        ".rbf_sam_candidates", "polygonize_rbf_sam_roof_mask"),
    "attach_reviewed_ground_footprints": (
        ".reviewed_footprints", "attach_reviewed_ground_footprints"),
    "attach_reviewed_ground_footprint_candidates": (
        ".reviewed_candidates", "attach_reviewed_ground_footprint_candidates"),
    "load_reviewed_absences": (".reviewed_absences", "load_reviewed_absences"),
    "attach_reviewed_physical_identities": (
        ".reviewed_identities", "attach_reviewed_physical_identities"),
    "recheck_reviewed_identity_sources": (
        ".reviewed_identities", "recheck_reviewed_identity_sources"),
    "recheck_review_sources": (".reviewed_footprints", "recheck_review_sources"),
    "recheck_reviewed_candidate_sources": (
        ".reviewed_candidates", "recheck_reviewed_candidate_sources"),
    "export_ground_footprint_review_packet": (
        ".ground_review_packet", "export_ground_footprint_review_packet"),
    "reconstruct": (".core", "reconstruct"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public export only when a caller requests it."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
