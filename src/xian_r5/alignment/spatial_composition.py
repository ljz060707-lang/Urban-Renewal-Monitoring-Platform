"""Compose spatially overlapping actual-date ties into an annual-base grid.

This module joins direct dated-image/base-grid candidates to relative dated
image candidates only on the exact polygonal overlap of their evidence cells.
It does not interpolate, apply a warp, assert ground accuracy, or estimate
roof-to-ground parallax. The annual reference raster's acquisition date and
per-pixel source provenance remain explicitly unknown.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from .field import DisplacementCell, LocalDisplacementField
from .pose_graph import TranslationEdge, solve_translation_pose_graph


_DIRECT_CANDIDATE_STATUS = "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED"
_RELATIVE_CANDIDATE_STATUS = "LOCAL_CANDIDATE_NOT_REVIEWED"


def _source_token(source: str | None) -> str:
    return str(source or "").replace("\\", "/").casefold()


@dataclass(frozen=True)
class _SpatialEdge:
    field_id: str
    reference_id: str
    moving_id: str
    dx_m: float
    dy_m: float
    source_ref: str
    kind: str
    cell: DisplacementCell

    @property
    def geometry(self) -> BaseGeometry:
        return self.cell.support_geometry


def _positive_area_components(edges: list[_SpatialEdge]) -> list[list[int]]:
    """Partition candidate supports by positive-area spatial connectivity."""
    if not edges:
        return []
    tree = STRtree([edge.geometry for edge in edges])
    parents = list(range(len(edges)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left, edge in enumerate(edges):
        for raw_right in tree.query(edge.geometry):
            right = int(raw_right)
            if right <= left:
                continue
            intersection_area = edge.geometry.intersection(edges[right].geometry).area
            area_tolerance = max(
                1e-8,
                min(edge.geometry.area, edges[right].geometry.area) * 1e-12,
            )
            if intersection_area > area_tolerance:
                union(left, right)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(edges)):
        grouped[root(index)].append(index)
    components = list(grouped.values())
    components.sort(key=lambda component: (
        min(edges[index].geometry.bounds[0] for index in component),
        min(edges[index].geometry.bounds[1] for index in component),
        tuple(edges[index].source_ref for index in component),
    ))
    return components


def _covers_polygon(container: BaseGeometry, polygon: BaseGeometry) -> bool:
    """Test full-area coverage with a small overlay tolerance."""
    outside_area = polygon.difference(container).area
    return outside_area <= max(1e-6, polygon.area * 1e-10)


def _validate_unreviewed_cell(cell: DisplacementCell, *, field_id: str) -> None:
    if cell.reviewed or cell.applied or cell.independent_ground_evidence:
        raise ValueError(f"Spatial composition accepts unreviewed candidates only: {field_id}/{cell.cell_id}")
    if cell.metadata.get("field_interpolated") is True or cell.metadata.get("interpolated") is True:
        raise ValueError(f"Interpolated input cells cannot enter spatial composition: {field_id}/{cell.cell_id}")


def _collect_edges(
    direct_base_fields: Iterable[LocalDisplacementField],
    relative_fields: Iterable[LocalDisplacementField],
) -> tuple[list[_SpatialEdge], str, str, set[str]]:
    direct_fields = tuple(direct_base_fields)
    relative_field_rows = tuple(relative_fields)
    if not direct_fields:
        raise ValueError("At least one direct actual-date/base-grid field is required")
    if not relative_field_rows:
        raise ValueError("At least one actual-date relative field is required")

    anchor_label = direct_fields[0].reference_label
    anchor_source = direct_fields[0].reference_source
    if not anchor_label or not anchor_source:
        raise ValueError("Direct base fields must identify their annual reference label and path")
    anchor_signature = (str(anchor_label), _source_token(anchor_source))
    direct_dates: set[str] = set()
    reference_hashes: set[str] = set()
    field_ids: set[str] = set()
    edges: list[_SpatialEdge] = []
    anchor_id = f"ANNUAL_BASE:{anchor_label}"

    for field in direct_fields:
        edge_field_id = f"{field.field_id}|{_source_token(field.reference_source)}"
        signature = (str(field.reference_label or ""), _source_token(field.reference_source))
        if (signature != anchor_signature
                or field.reference_date is not None
                or field.reference_date_status != "REFERENCE_ACQUISITION_DATE_UNKNOWN"):
            raise ValueError("Direct fields must share one annual reference with unknown acquisition date")
        if edge_field_id in field_ids:
            raise ValueError(f"Duplicate alignment field ID: {field.field_id}")
        field_ids.add(edge_field_id)
        if field.moving_date in direct_dates:
            raise ValueError(f"Duplicate direct base fields for actual date: {field.moving_date}")
        direct_dates.add(field.moving_date)
        for cell in field.cells:
            _validate_unreviewed_cell(cell, field_id=field.field_id)
            reference_hash = cell.metadata.get("reference_raster_sha256")
            if reference_hash:
                reference_hashes.add(str(reference_hash).lower())
            if cell.correction_dx_m_candidate is None:
                continue
            if cell.status != _DIRECT_CANDIDATE_STATUS:
                continue
            edges.append(_SpatialEdge(
                field_id=edge_field_id,
                reference_id=anchor_id,
                moving_id=field.moving_date,
                dx_m=float(cell.correction_dx_m_candidate),
                dy_m=float(cell.correction_dy_m_candidate),
                source_ref=cell.source_ref,
                kind="DIRECT_BASE_TIE",
                cell=cell,
            ))

    if len(reference_hashes) > 1:
        raise ValueError("Direct fields identify conflicting annual reference raster hashes")

    for field in relative_field_rows:
        edge_field_id = f"{field.field_id}|{_source_token(field.reference_source)}"
        if (field.reference_date is None
                or field.reference_date_status != "ACTUAL_DATE_KNOWN"
                or field.reference_date >= field.moving_date
                or field.reference_label != field.reference_date):
            raise ValueError("Relative fields must preserve ordered actual acquisition dates")
        if edge_field_id in field_ids:
            raise ValueError(f"Duplicate alignment field ID: {field.field_id}")
        field_ids.add(edge_field_id)
        for cell in field.cells:
            _validate_unreviewed_cell(cell, field_id=field.field_id)
            if cell.correction_dx_m_candidate is None:
                continue
            if cell.status != _RELATIVE_CANDIDATE_STATUS:
                continue
            edges.append(_SpatialEdge(
                field_id=edge_field_id,
                reference_id=field.reference_date,
                moving_id=field.moving_date,
                dx_m=float(cell.correction_dx_m_candidate),
                dy_m=float(cell.correction_dy_m_candidate),
                source_ref=cell.source_ref,
                kind="RELATIVE_ACTUAL_DATE_TIE",
                cell=cell,
            ))

    edges.sort(key=lambda edge: (
        edge.kind, edge.reference_id, edge.moving_id, edge.geometry.bounds,
        edge.source_ref,
    ))
    return edges, anchor_id, str(anchor_label), direct_dates


def compose_spatial_alignment_fields(
    direct_base_fields: Iterable[LocalDisplacementField],
    relative_fields: Iterable[LocalDisplacementField],
    *,
    cycle_screening_limit_m: float = 5.0,
) -> tuple[list[LocalDisplacementField], dict[str, object]]:
    """Compose candidate ties into the common annual-base grid on exact overlap.

    Direct cells are date-to-annual-grid candidate edges. Relative cells are
    moving-date-to-reference-date candidate edges. Polygon boundaries from all
    overlapping candidate cells are overlaid into disjoint support atoms; a
    pose graph is solved only from edges whose polygons fully cover an atom.
    Regions without a path to the annual-grid anchor produce no output field.

    Returned fields are candidate-only and unapplied. A low graph-closure
    residual is an internal consistency screen, not ground-position accuracy.
    """
    if (not math.isfinite(cycle_screening_limit_m)
            or cycle_screening_limit_m <= 0):
        raise ValueError("Cycle screening limit must be finite and positive")

    direct_base_fields = tuple(direct_base_fields)
    relative_fields = tuple(relative_fields)
    edges, anchor_id, anchor_label, direct_dates = _collect_edges(
        direct_base_fields, relative_fields)
    components = _positive_area_components(edges)
    cells_by_date: dict[str, list[DisplacementCell]] = defaultdict(list)
    graph_rows: list[dict[str, object]] = []
    ambiguous_atom_count = 0
    unanchored_component_count = 0
    direct_only_atom_count = 0
    spatially_composed_atom_count = 0
    cycle_status_counts: Counter[str] = Counter()
    atom_index = 0

    for component_index, component_indices in enumerate(components):
        component_edges = [edges[index] for index in component_indices]
        if not any(edge.kind == "DIRECT_BASE_TIE" for edge in component_edges):
            unanchored_component_count += 1
            continue

        linework = unary_union([edge.geometry.boundary for edge in component_edges])
        faces = list(polygonize([linework]))
        faces.sort(key=lambda face: (
            round(face.bounds[1], 7), round(face.bounds[0], 7),
            round(face.bounds[3], 7), round(face.bounds[2], 7),
            round(face.area, 7),
        ))

        for face in faces:
            if face.is_empty or face.area <= 1e-8:
                continue
            covering = [edge for edge in component_edges
                        if _covers_polygon(edge.geometry, face)]
            if not any(edge.kind == "DIRECT_BASE_TIE" for edge in covering):
                continue

            per_field: dict[str, list[_SpatialEdge]] = defaultdict(list)
            per_date_pair: dict[tuple[str, str, str], list[_SpatialEdge]] = defaultdict(list)
            for edge in covering:
                per_field[edge.field_id].append(edge)
                per_date_pair[(edge.kind, edge.reference_id, edge.moving_id)].append(edge)
            if (any(len(rows) != 1 for rows in per_field.values())
                    or any(len(rows) != 1 for rows in per_date_pair.values())):
                ambiguous_atom_count += 1
                graph_rows.append({
                    "atom_id": f"c{component_index:04d}_a{atom_index:05d}",
                    "support_bounds_epsg32649": [float(value) for value in face.bounds],
                    "support_area_m2": float(face.area),
                    "status": "AMBIGUOUS_OVERLAPPING_CANDIDATE_CELLS",
                    "reviewed": False,
                    "applied": False,
                })
                atom_index += 1
                continue

            pose_edges = [TranslationEdge(
                reference_id=edge.reference_id,
                moving_id=edge.moving_id,
                correction_dx_m=edge.dx_m,
                correction_dy_m=edge.dy_m,
                source_ref=edge.source_ref,
            ) for edge in covering]
            solution = solve_translation_pose_graph(pose_edges, anchor_id)
            if solution.status == "CONNECTED_GRAPH_RANK_DEFICIENT":
                graph_rows.append({
                    "atom_id": f"c{component_index:04d}_a{atom_index:05d}",
                    "support_bounds_epsg32649": [float(value) for value in face.bounds],
                    "support_area_m2": float(face.area),
                    "status": solution.status,
                    "reviewed": False,
                    "applied": False,
                })
                atom_index += 1
                continue

            used_edges = [edge for edge in covering
                          if edge.source_ref in solution.used_edge_sources]
            if not any(edge.kind == "RELATIVE_ACTUAL_DATE_TIE" for edge in used_edges):
                direct_only_atom_count += 1
                graph_rows.append({
                    "atom_id": f"c{component_index:04d}_a{atom_index:05d}",
                    "support_bounds_epsg32649": [float(value) for value in face.bounds],
                    "support_area_m2": float(face.area),
                    "candidate_edge_count": len(solution.used_edge_sources),
                    "candidate_edge_sources": list(solution.used_edge_sources),
                    "connected_actual_dates": sorted(
                        value for value in solution.connected_ids if value != anchor_id),
                    "status": "DIRECT_BASE_TIE_ONLY_NO_RELATIVE_LINK",
                    "reviewed": False,
                    "applied": False,
                })
                atom_index += 1
                continue

            if solution.cycle_degrees_of_freedom == 0:
                composition_status = "SPATIALLY_COMPOSED_CANDIDATE_TREE_NO_CYCLE_CHECK"
            elif (solution.closure_max_m is not None
                  and solution.closure_max_m > cycle_screening_limit_m):
                composition_status = "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED"
            else:
                composition_status = "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY"
            cycle_status_counts[composition_status] += 1
            atom_id = f"c{component_index:04d}_a{atom_index:05d}"
            spatially_composed_atom_count += 1
            reference_component_statuses = sorted({
                str(edge.cell.metadata["reference_component_id_status"])
                for edge in used_edges
                if edge.cell.metadata.get("reference_component_id_status")
            })
            for actual_date, offset in sorted(solution.offsets_m.items()):
                if actual_date == anchor_id:
                    continue
                cell_id = f"{atom_id}_{actual_date}"
                metadata: dict[str, object] = {
                    "composition_method": "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH",
                    "support_atom_id": atom_id,
                    "support_area_m2": float(face.area),
                    "source_candidate_cell_refs": [edge.source_ref for edge in used_edges],
                    "source_candidate_field_ids": sorted({edge.field_id for edge in used_edges}),
                    "source_edge_count": len(solution.used_edge_sources),
                    "source_edge_kinds": sorted({edge.kind for edge in used_edges}),
                    "source_edge_corrections_m": {
                        edge.source_ref: [edge.dx_m, edge.dy_m] for edge in used_edges
                    },
                    "anchor_id": anchor_id,
                    "anchor_acquisition_date_status": (
                        "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"),
                    "reference_component_id_statuses": reference_component_statuses,
                    "cycle_degrees_of_freedom": solution.cycle_degrees_of_freedom,
                    "closure_rmse_m": solution.closure_rmse_m,
                    "closure_p90_m": solution.closure_p90_m,
                    "closure_max_m": solution.closure_max_m,
                    "cycle_screening_limit_m": cycle_screening_limit_m,
                    "cycle_limit_semantics": (
                        "INTERNAL_CLOSURE_SCREEN_ONLY_NOT_POSITIONAL_ACCURACY"),
                    "reviewed": False,
                    "independent_ground_evidence": False,
                    "positional_uncertainty_m": None,
                    "field_interpolated": False,
                    "applied": False,
                    "roof_to_ground_parallax_resolved": False,
                }
                cells_by_date[actual_date].append(DisplacementCell(
                    cell_id=cell_id,
                    support_geometry=face,
                    correction_dx_m_candidate=float(offset[0]),
                    correction_dy_m_candidate=float(offset[1]),
                    fit_dispersion_m=None,
                    status=composition_status,
                    method="SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE",
                    source_ref=f"spatial-composition:{atom_id}:{actual_date}",
                    reviewed=False,
                    independent_ground_evidence=False,
                    positional_uncertainty_m=None,
                    applied=False,
                    metadata=metadata,
                ))

            graph_rows.append({
                "atom_id": atom_id,
                "support_bounds_epsg32649": [float(value) for value in face.bounds],
                "support_area_m2": float(face.area),
                "candidate_edge_count": len(solution.used_edge_sources),
                "candidate_edge_sources": list(solution.used_edge_sources),
                "connected_actual_dates": sorted(
                    value for value in solution.connected_ids if value != anchor_id),
                "cycle_degrees_of_freedom": solution.cycle_degrees_of_freedom,
                "closure_rmse_m": solution.closure_rmse_m,
                "closure_p90_m": solution.closure_p90_m,
                "closure_max_m": solution.closure_max_m,
                "edge_residuals_m": list(solution.edge_residuals_m),
                "status": composition_status,
                "reviewed": False,
                "applied": False,
            })
            atom_index += 1

    fields: list[LocalDisplacementField] = []
    for actual_date, cells in sorted(cells_by_date.items()):
        fields.append(LocalDisplacementField(
            field_id=f"SPATIAL_{actual_date}_TO_{anchor_label}",
            reference_date=None,
            moving_date=actual_date,
            cells=tuple(sorted(cells, key=lambda cell: cell.cell_id)),
            crs="EPSG:32649",
            correction_scope="IMAGE_TO_REFERENCE_GRID",
            reference_label=anchor_label,
            reference_source=str(direct_base_fields[0].reference_source),
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
        ))

    summary: dict[str, object] = {
        "schema_version": "xian_r5.spatially_composed_alignment_candidates.v1",
        "anchor_label": anchor_label,
        "anchor_id": anchor_id,
        "anchor_acquisition_date_status": (
            "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN"),
        "crs": "EPSG:32649",
        "input_direct_candidate_edge_count": sum(
            edge.kind == "DIRECT_BASE_TIE" for edge in edges),
        "input_relative_candidate_edge_count": sum(
            edge.kind == "RELATIVE_ACTUAL_DATE_TIE" for edge in edges),
        "input_actual_dates_with_direct_fields": sorted(direct_dates),
        "spatial_component_count": len(components),
        "unanchored_component_count": unanchored_component_count,
        "candidate_atom_count": len(graph_rows),
        "spatially_composed_atom_count": spatially_composed_atom_count,
        "direct_base_tie_only_atom_count": direct_only_atom_count,
        "ambiguous_atom_count": ambiguous_atom_count,
        "cycle_status_counts": dict(sorted(cycle_status_counts.items())),
        "output_cell_counts_by_actual_date": {
            actual_date: len(cells) for actual_date, cells in sorted(cells_by_date.items())
        },
        "graph_atoms": graph_rows,
        "field_interpolated": False,
        "field_applied": False,
        "absolute_ground_position_resolved": False,
        "roof_to_ground_parallax_resolved": False,
        "formal_change_use": False,
        "interpretation": (
            "SPATIALLY_COMPOSED_IMAGE_GRID_CANDIDATES_ONLY;ANNUAL_BASE_DATE_AND_PER_PIXEL_"
            "MOSAIC_PROVENANCE_UNKNOWN;NO_GROUND_CONTROL;NO_ROOF_TO_GROUND_CORRECTION"),
    }
    return fields, summary
