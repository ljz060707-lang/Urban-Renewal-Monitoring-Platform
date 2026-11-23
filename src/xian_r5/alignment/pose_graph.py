"""Solve a local translation graph without promoting its coordinates.

Each edge is a candidate correction from a moving raster into a reference
raster. For an anchor-relative offset ``p[year]``, the convention is
``p[moving] - p[reference] = correction_moving_to_reference``. The solver
returns only anchor-connected offsets and internal cycle residuals; it does
not label them as positional accuracy or accepted ground control.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Iterable


def _least_squares_qr(
    design: list[list[float]],
    measurements: list[list[float]],
) -> tuple[list[tuple[float, float]] | None, int]:
    """Solve a small two-coordinate least-squares system without BLAS/LAPACK.

    The local pose graph has few raster nodes. Twice-reorthogonalized modified
    Gram-Schmidt keeps this solve self-contained and avoids runtime-dependent
    native linear-algebra backends at this module's seam.
    """
    row_count = len(design)
    column_count = len(design[0]) if design else 0
    if (row_count == 0 or column_count == 0
            or len(measurements) != row_count
            or any(len(row) != column_count for row in design)
            or any(len(row) != 2 for row in measurements)):
        raise ValueError("Pose-graph least-squares inputs have invalid dimensions")

    q_columns: list[list[float]] = []
    upper = [[0.0] * column_count for _ in range(column_count)]
    for column in range(column_count):
        vector = [float(design[row][column]) for row in range(row_count)]
        for _ in range(2):
            for prior, q_column in enumerate(q_columns):
                projection = sum(q_column[row] * vector[row]
                                 for row in range(row_count))
                upper[prior][column] += projection
                vector = [value - projection * q_column[row]
                          for row, value in enumerate(vector)]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return None, column
        upper[column][column] = norm
        q_columns.append([value / norm for value in vector])

    projected = [[sum(q_columns[column][row] * measurements[row][axis]
                       for row in range(row_count))
                   for axis in range(2)]
                  for column in range(column_count)]
    solution = [[0.0, 0.0] for _ in range(column_count)]
    for axis in range(2):
        for column in range(column_count - 1, -1, -1):
            remainder = projected[column][axis] - sum(
                upper[column][later] * solution[later][axis]
                for later in range(column + 1, column_count))
            solution[column][axis] = remainder / upper[column][column]
    return [(row[0], row[1]) for row in solution], column_count


def _linear_percentile(values: list[float], percentile: float) -> float:
    """Match NumPy's default linear percentile for a small value sequence."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Percentiles require at least one value")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("Percentile must be between 0 and 100")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True)
class TranslationEdge:
    reference_id: str
    moving_id: str
    correction_dx_m: float
    correction_dy_m: float
    source_ref: str

    def __post_init__(self) -> None:
        if not self.reference_id or not self.moving_id or self.reference_id == self.moving_id:
            raise ValueError("Translation edges require two different raster IDs")
        if not self.source_ref:
            raise ValueError("Translation edges require provenance")
        if not math.isfinite(self.correction_dx_m) or not math.isfinite(self.correction_dy_m):
            raise ValueError("Translation edge values must be finite")


@dataclass(frozen=True)
class PoseGraphResult:
    anchor_id: str
    offsets_m: dict[str, tuple[float, float]]
    connected_ids: tuple[str, ...]
    used_edge_sources: tuple[str, ...]
    edge_residuals_m: tuple[dict[str, float | str], ...]
    cycle_degrees_of_freedom: int
    closure_rmse_m: float | None
    closure_p90_m: float | None
    closure_max_m: float | None
    status: str


def compose_moving_to_anchor_corrections(
    moving_to_reference_m: tuple[float, float],
    reference_to_anchor_m: tuple[float, float],
) -> tuple[float, float]:
    """Add translations along ``moving -> reference -> anchor``."""
    values = (*moving_to_reference_m, *reference_to_anchor_m)
    if len(values) != 4 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Correction composition requires two finite x/y vectors")
    return (float(moving_to_reference_m[0] + reference_to_anchor_m[0]),
            float(moving_to_reference_m[1] + reference_to_anchor_m[1]))


def solve_translation_pose_graph(
    edges: Iterable[TranslationEdge], anchor_id: str,
) -> PoseGraphResult:
    """Least-squares anchor-relative translations for one local support cell.

    Edges outside the anchor's connected component are ignored. A tree has no
    independent cycle check and is reported as such. Redundant inconsistent
    edges remain in the solution and are exposed through edge residuals.
    """
    if not anchor_id:
        raise ValueError("An explicit reference-grid anchor is required")
    records = tuple(edges)
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in records:
        graph[edge.reference_id].add(edge.moving_id)
        graph[edge.moving_id].add(edge.reference_id)

    connected = {anchor_id}
    queue = deque([anchor_id])
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, set()) - connected:
            connected.add(neighbor)
            queue.append(neighbor)

    used = [edge for edge in records
            if edge.reference_id in connected and edge.moving_id in connected]
    variable_ids = sorted(connected - {anchor_id})
    index = {raster_id: i for i, raster_id in enumerate(variable_ids)}
    offsets: dict[str, tuple[float, float]] = {anchor_id: (0.0, 0.0)}
    if not used:
        return PoseGraphResult(anchor_id, offsets, (anchor_id,), (), (), 0,
                               None, None, None, "ANCHOR_ONLY_NO_CONNECTED_TIES")

    design = [[0.0] * len(variable_ids) for _ in used]
    measurements = [[0.0, 0.0] for _ in used]
    for row, edge in enumerate(used):
        if edge.moving_id != anchor_id:
            design[row][index[edge.moving_id]] += 1.0
        if edge.reference_id != anchor_id:
            design[row][index[edge.reference_id]] -= 1.0
        measurements[row] = [edge.correction_dx_m, edge.correction_dy_m]

    solved, rank = _least_squares_qr(design, measurements)
    if rank < len(variable_ids):
        # Connectivity should imply full rank for an incidence matrix with one
        # anchored node; retain the guard in case malformed input violates it.
        return PoseGraphResult(anchor_id, offsets, tuple(sorted(connected)),
                               tuple(edge.source_ref for edge in used), (), 0,
                               None, None, None, "CONNECTED_GRAPH_RANK_DEFICIENT")
    assert solved is not None
    offsets.update({raster_id: solved[index[raster_id]]
                    for raster_id in variable_ids})
    residual_vectors = []
    residual_norms = []
    for row, measurement in zip(design, measurements):
        residual = (
            sum(row[column] * solved[column][0]
                for column in range(len(variable_ids))) - measurement[0],
            sum(row[column] * solved[column][1]
                for column in range(len(variable_ids))) - measurement[1],
        )
        residual_vectors.append(residual)
        residual_norms.append(math.hypot(*residual))
    residual_rows = tuple({"source_ref": edge.source_ref,
                           "residual_dx_m": float(residual_vectors[i][0]),
                           "residual_dy_m": float(residual_vectors[i][1]),
                           "residual_norm_m": float(residual_norms[i])}
                          for i, edge in enumerate(used))
    cycle_dof = max(0, len(used) - max(0, len(connected) - 1))
    if cycle_dof == 0:
        status = "CONNECTED_TREE_NO_CYCLE_CHECK"
    else:
        status = "OVERDETERMINED_CYCLE_DIAGNOSTIC_CANDIDATE"
    return PoseGraphResult(
        anchor_id=anchor_id,
        offsets_m=offsets,
        connected_ids=tuple(sorted(connected)),
        used_edge_sources=tuple(edge.source_ref for edge in used),
        edge_residuals_m=residual_rows,
        cycle_degrees_of_freedom=cycle_dof,
        closure_rmse_m=math.sqrt(sum(value * value for value in residual_norms)
                                 / len(residual_norms)),
        closure_p90_m=_linear_percentile(residual_norms, 90.0),
        closure_max_m=max(residual_norms),
        status=status,
    )
