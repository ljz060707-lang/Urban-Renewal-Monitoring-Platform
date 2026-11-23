"""Image-source and registration evidence for R5.

Public symbols are loaded on first access so numerical modules such as
``local_field`` do not acquire vector-I/O or rendering dependencies merely
because they live in this package.  The exported interface stays compatible
with ``from xian_r5.alignment import ...``.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_observation_registry": (".registry", "build_observation_registry"),
    "classify_supplement_assets": (".registry", "classify_supplement_assets"),
    "image_shift_to_projected_correction": (".base_tie", "image_shift_to_projected_correction"),
    "DisplacementCell": (".field", "DisplacementCell"),
    "DisplacementSample": (".field", "DisplacementSample"),
    "LocalDisplacementField": (".field", "LocalDisplacementField"),
    "MosaicTileGrid": (".mosaic_grid", "MosaicTileGrid"),
    "TileRasterWindow": (".mosaic_grid", "TileRasterWindow"),
    "TileModelInputGrid": (".mosaic_grid", "TileModelInputGrid"),
    "axis_aligned_window_for_bounds": (
        ".mosaic_grid", "axis_aligned_window_for_bounds"),
    "load_mosaic_grids_by_source_epoch": (
        ".mosaic_grid", "load_mosaic_grids_by_source_epoch"),
    "group_fields": (".field", "group_fields"),
    "compose_actual_date_fields": (".composition", "compose_actual_date_fields"),
    "compose_spatial_alignment_fields": (
        ".spatial_composition", "compose_spatial_alignment_fields"),
    "direct_base_tie_field_from_epoch": (".field_io", "direct_base_tie_field_from_epoch"),
    "read_composed_alignment_fields": (".field_io", "read_composed_alignment_fields"),
    "read_direct_base_tie_fields": (".field_io", "read_direct_base_tie_fields"),
    "read_component_tie_consensus_fields": (
        ".field_io", "read_component_tie_consensus_fields"),
    "read_lightglue_local_field_candidates": (
        ".field_io", "read_lightglue_local_field_candidates"),
    "read_spatial_alignment_composition_fields": (
        ".field_io", "read_spatial_alignment_composition_fields"),
    "compare_alignment_fields": (".comparison", "compare_alignment_fields"),
    "CandidateAlignmentPreview": (".preview", "CandidateAlignmentPreview"),
    "render_candidate_alignment_preview": (".preview", "render_candidate_alignment_preview"),
    "RoofGridCandidate": (".roof_candidate", "RoofGridCandidate"),
    "align_roof_candidate_to_reference_grid": (".roof_candidate", "align_roof_candidate_to_reference_grid"),
    "summarize_local_relative_shifts": (
        ".relative_offsets", "summarize_local_relative_shifts"),
    "solve_local_relative_pose_graphs": (
        ".relative_offsets", "solve_local_relative_pose_graphs"),
    "write_relative_roof_candidate_products": (
        ".relative_roof_candidate_products", "write_relative_roof_candidate_products"),
    "export_component_tie_consensus_candidates": (
        ".component_tie_consensus", "export_component_tie_consensus_candidates"),
    "PoseGraphResult": (".pose_graph", "PoseGraphResult"),
    "TranslationEdge": (".pose_graph", "TranslationEdge"),
    "compose_moving_to_anchor_corrections": (".pose_graph", "compose_moving_to_anchor_corrections"),
    "solve_translation_pose_graph": (".pose_graph", "solve_translation_pose_graph"),
    "export_alignment_review_packet": (".review_packet", "export_alignment_review_packet"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
