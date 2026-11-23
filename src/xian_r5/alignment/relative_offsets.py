"""Summarize spatially local, relative building-shift candidates.

These estimates describe roof motion relative to a shared candidate reference.
They do not resolve absolute ground position and are never applied as a warp.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Iterable, Mapping, Sequence

from .pose_graph import TranslationEdge, solve_translation_pose_graph


def summarize_local_relative_shifts(
    samples: Iterable[Mapping[str, object]],
    bounds: Sequence[float],
    *,
    grid_shape: tuple[int, int] = (4, 4),
    minimum_support: int = 5,
) -> dict[str, object]:
    """Bin feature-level relative shifts and return robust candidate summaries.

    Each sample must contain ``sample_id``, ``x_m``, ``y_m``, ``dx_m``, and
    ``dy_m``. Coordinates are projected metres in the same CRS as ``bounds``.
    Cells are indexed north-to-south, then west-to-east. Empty and weakly
    supported cells remain explicit; no interpolation or application occurs.
    """
    if len(bounds) != 4:
        raise ValueError("bounds must be (xmin, ymin, xmax, ymax)")
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError("bounds must be finite")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bounds must have positive width and height")
    if (len(grid_shape) != 2 or any(isinstance(value, bool)
                                    or not isinstance(value, int) or value < 1
                                    for value in grid_shape)):
        raise ValueError("grid_shape must contain two positive integers")
    if isinstance(minimum_support, bool) or not isinstance(minimum_support, int) or minimum_support < 1:
        raise ValueError("minimum_support must be a positive integer")

    rows, columns = grid_shape
    cell_width = (xmax - xmin) / columns
    cell_height = (ymax - ymin) / rows
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    seen_ids: set[str] = set()
    sample_count = 0
    for sample in samples:
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("each sample requires a nonempty string sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        try:
            x_m, y_m, dx_m, dy_m = (
                float(sample[name]) for name in ("x_m", "y_m", "dx_m", "dy_m"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"sample {sample_id} has incomplete numeric values") from exc
        if not all(math.isfinite(value) for value in (x_m, y_m, dx_m, dy_m)):
            raise ValueError(f"sample {sample_id} contains non-finite values")
        if not (xmin <= x_m <= xmax and ymin <= y_m <= ymax):
            raise ValueError(f"sample {sample_id} falls outside bounds")
        column = min(columns - 1, int((x_m - xmin) / cell_width))
        row = min(rows - 1, int((ymax - y_m) / cell_height))
        grouped[(row, column)].append({
            "sample_id": sample_id, "x_m": x_m, "y_m": y_m,
            "dx_m": dx_m, "dy_m": dy_m,
        })
        sample_count += 1

    cells = []
    for row in range(rows):
        for column in range(columns):
            x0 = xmin + column * cell_width
            x1 = xmin + (column + 1) * cell_width
            y1 = ymax - row * cell_height
            y0 = ymax - (row + 1) * cell_height
            cell_samples = grouped.get((row, column), [])
            dxs = [float(item["dx_m"]) for item in cell_samples]
            dys = [float(item["dy_m"]) for item in cell_samples]
            median_dx = median(dxs) if dxs else None
            median_dy = median(dys) if dys else None
            mad_dx = median([abs(value - median_dx) for value in dxs]) if dxs else None
            mad_dy = median([abs(value - median_dy) for value in dys]) if dys else None
            radial = (median([math.hypot(float(item["dx_m"]) - median_dx,
                                         float(item["dy_m"]) - median_dy)
                              for item in cell_samples]) if cell_samples else None)
            supported = len(cell_samples) >= minimum_support
            cells.append({
                "row": row, "column": column,
                "bounds_epsg32649": [x0, y0, x1, y1],
                "center_epsg32649": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
                "sample_count": len(cell_samples),
                "sample_ids": sorted(item["sample_id"] for item in cell_samples),
                "median_dx_m": median_dx if supported else None,
                "median_dy_m": median_dy if supported else None,
                "mad_dx_m": mad_dx if supported else None,
                "mad_dy_m": mad_dy if supported else None,
                "median_radial_deviation_m": radial if supported else None,
                "observed_median_dx_m": median_dx,
                "observed_median_dy_m": median_dy,
                "observed_mad_dx_m": mad_dx, "observed_mad_dy_m": mad_dy,
                "status": ("NUMERIC_RELATIVE_SHIFT_CANDIDATE"
                           if supported
                           else "INSUFFICIENT_RELATIVE_SHIFT_SUPPORT"),
                "absolute_ground_position_resolved": False,
                "applied": False,
            })

    return {
        "method": "SPATIAL_BINNED_PER_FEATURE_MEDIAN_MAD",
        "grid_shape_rows_columns": [rows, columns],
        "minimum_support": minimum_support,
        "sample_count": sample_count,
        "numeric_candidate_cell_count": sum(
            cell["status"] == "NUMERIC_RELATIVE_SHIFT_CANDIDATE" for cell in cells),
        "bounds_epsg32649": [xmin, ymin, xmax, ymax],
        "source_semantics": "RELATIVE_TO_UNVERIFIED_STATIC_REFERENCE_ONLY",
        "interpolated": False,
        "applied": False,
        "absolute_ground_position_resolved": False,
        "cells": cells,
    }


def solve_local_relative_pose_graphs(
    pairwise_samples: Iterable[Mapping[str, object]],
    bounds: Sequence[float],
    *,
    anchor_id: str,
    grid_shape: tuple[int, int] = (4, 4),
    minimum_support: int = 5,
) -> dict[str, object]:
    """Solve per-cell temporal alignment candidates from pairwise roof motion.

    Inputs use ``dx_m``/``dy_m`` for observed apparent roof motion from
    ``reference_id`` to ``moving_id``. The function negates those values to
    construct moving-to-reference correction edges for the existing pose
    graph solver. A closed graph is only an internal relative-consistency
    diagnostic; neither it nor a tree resolves absolute ground position.
    """
    if not isinstance(anchor_id, str) or not anchor_id.strip():
        raise ValueError("an explicit relative-pose anchor_id is required")
    samples_by_pair: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for sample in pairwise_samples:
        reference_id, moving_id = sample.get("reference_id"), sample.get("moving_id")
        if (not isinstance(reference_id, str) or not reference_id.strip()
                or not isinstance(moving_id, str) or not moving_id.strip()
                or reference_id == moving_id):
            raise ValueError("each relative sample requires distinct reference_id and moving_id")
        samples_by_pair[(reference_id, moving_id)].append(sample)

    pair_grids = []
    pairwise_sample_count = 0
    for (reference_id, moving_id), records in sorted(samples_by_pair.items()):
        summary = summarize_local_relative_shifts(
            ({"sample_id": item.get("sample_id"), "x_m": item.get("x_m"),
              "y_m": item.get("y_m"), "dx_m": item.get("dx_m"),
              "dy_m": item.get("dy_m")} for item in records),
            bounds, grid_shape=grid_shape, minimum_support=minimum_support)
        pair_grids.append({
            "reference_id": reference_id, "moving_id": moving_id,
            "summary": summary,
        })
        pairwise_sample_count += int(summary["sample_count"])

    grid_template = summarize_local_relative_shifts(
        (), bounds, grid_shape=grid_shape, minimum_support=minimum_support)
    edges_by_cell: dict[tuple[int, int], list[TranslationEdge]] = defaultdict(list)
    pair_metrics_by_cell: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for pair in pair_grids:
        reference_id, moving_id = pair["reference_id"], pair["moving_id"]
        for cell in pair["summary"]["cells"]:
            key = (int(cell["row"]), int(cell["column"]))
            if cell["status"] != "NUMERIC_RELATIVE_SHIFT_CANDIDATE":
                continue
            motion_dx, motion_dy = float(cell["median_dx_m"]), float(cell["median_dy_m"])
            source_ref = f"static-reference-relative:{reference_id}->{moving_id}:r{key[0]}c{key[1]}"
            edges_by_cell[key].append(TranslationEdge(
                reference_id=reference_id, moving_id=moving_id,
                correction_dx_m=-motion_dx, correction_dy_m=-motion_dy,
                source_ref=source_ref))
            pair_metrics_by_cell[key].append({
                "reference_id": reference_id, "moving_id": moving_id,
                "sample_count": cell["sample_count"],
                "apparent_roof_motion_m": [motion_dx, motion_dy],
                "candidate_correction_m": [-motion_dx, -motion_dy],
                "mad_dx_m": cell["mad_dx_m"], "mad_dy_m": cell["mad_dy_m"],
                "source_semantics": "RELATIVE_TO_UNVERIFIED_STATIC_REFERENCE_ONLY",
            })

    output_cells = []
    for template in grid_template["cells"]:
        key = (int(template["row"]), int(template["column"]))
        edges = edges_by_cell.get(key, [])
        solution = solve_translation_pose_graph(edges, anchor_id)
        if not edges:
            status = "INSUFFICIENT_RELATIVE_SUPPORT"
        elif solution.cycle_degrees_of_freedom > 0:
            status = "RELATIVE_POSE_CYCLE_DIAGNOSTIC_CANDIDATE"
        elif solution.status == "CONNECTED_TREE_NO_CYCLE_CHECK":
            status = "RELATIVE_POSE_TREE_NO_CYCLE_CHECK"
        else:
            status = solution.status
        output_cells.append({
            "row": key[0], "column": key[1],
            "bounds_epsg32649": template["bounds_epsg32649"],
            "pairwise_candidate_edges": pair_metrics_by_cell.get(key, []),
            "connected_actual_dates": list(solution.connected_ids),
            "candidate_corrections_to_anchor_m": {
                actual_date: [float(value) for value in offset]
                for actual_date, offset in solution.offsets_m.items()},
            "cycle_degrees_of_freedom": solution.cycle_degrees_of_freedom,
            "closure_rmse_m": solution.closure_rmse_m,
            "closure_p90_m": solution.closure_p90_m,
            "closure_max_m": solution.closure_max_m,
            "edge_residuals_m": list(solution.edge_residuals_m),
            "status": status,
            "absolute_ground_position_resolved": False,
            "applied": False,
        })

    return {
        "schema_version": "xian_r5.local_relative_pose_graph_candidates.v1",
        "method": "EXISTING_TRANSLATION_POSE_GRAPH_ON_SPATIAL_MEDIAN_RELATIVE_TIES",
        "anchor_id": anchor_id,
        "grid_shape_rows_columns": list(grid_shape),
        "minimum_support_per_pairwise_edge": minimum_support,
        "pairwise_edge_count": len(samples_by_pair),
        "pairwise_sample_count": pairwise_sample_count,
        "source_semantics": "RELATIVE_TO_UNVERIFIED_STATIC_REFERENCE_ONLY",
        "cycle_semantics": "INTERNAL_CONSISTENCY_NOT_ABSOLUTE_POSITION_ACCURACY",
        "interpolated": False,
        "absolute_ground_position_resolved": False,
        "applied": False,
        "cells": output_cells,
    }
