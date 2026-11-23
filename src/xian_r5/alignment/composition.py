"""Compose local actual-date imagery ties with annual-base pose candidates.

The output is a spatially explicit candidate field from a dated sub-meter
observation into a selected annual 10 m reference grid. Annual file labels are
not treated as acquisition dates, search-boundary ties are excluded, and no
field is interpolated, applied, or promoted to ground truth.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Any

from .base_tie import image_shift_to_projected_correction
from .pose_graph import TranslationEdge, solve_translation_pose_graph


def _year_from_base_path(path: str) -> int:
    match = re.search(r"Xian_(\d{4})_TrueColor\.tif$", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot resolve annual base label from path: {path}")
    return int(match.group(1))


def _cell_id(row: int, col: int) -> str:
    return f"r{row}c{col}"


def _support_bounds(transform: list[float], window: list[int]) -> list[float]:
    a, b, c, d, e, f = (float(value) for value in transform[:6])
    if abs(b) > 1e-12 or abs(d) > 1e-12 or a <= 0 or e >= 0:
        raise ValueError("Composition requires a north-up, east-right EPSG:32649 grid")
    col0, row0, col1, row1 = (int(value) for value in window)
    if col1 <= col0 or row1 <= row0:
        raise ValueError("Invalid local tie cell window")
    x0, x1 = c + col0 * a, c + col1 * a
    y0, y1 = f + row0 * e, f + row1 * e
    return [float(x0), float(min(y0, y1)), float(x1), float(max(y0, y1))]


def compose_actual_date_fields(
    base_pair_report: dict[str, Any],
    base_tie_report: dict[str, Any],
    *,
    anchor_year: int = 2019,
    cycle_screening_limit_m: float = 5.0,
) -> dict[str, Any]:
    """Return local correction candidates into the selected annual base grid.

    Pose graph equations and pixel-shift direction are centralized here so the
    caller only needs the two diagnostic receipts. The 5 m closure limit is a
    half-pixel screening label for a 10 m grid, not a position-accuracy claim.
    """
    if cycle_screening_limit_m <= 0:
        raise ValueError("Cycle screening limit must be positive")
    if base_pair_report.get("field_reviewed") or base_pair_report.get("field_applied"):
        raise ValueError("Base-pair input must remain an unreviewed, unapplied diagnostic")
    if base_pair_report.get("registration_promoted"):
        raise ValueError("Refusing an already promoted base-pair diagnostic")
    if base_tie_report.get("automatic_registration_promoted"):
        raise ValueError("High-resolution/base ties must remain unpromoted candidates")
    if base_pair_report.get("base_year_date_semantics") != "ANNUAL_FILE_LABELS_ONLY;PIXEL_ACQUISITION_DATES_UNKNOWN":
        raise ValueError("Annual-base date semantics are not the expected unknown-date state")

    years = tuple(int(value) for value in base_pair_report.get("base_year_labels", ()))
    if anchor_year not in years:
        raise ValueError(f"Anchor year {anchor_year} is absent from the base-pair graph")
    anchor_path = str(base_pair_report["base_rasters"][str(anchor_year)]["path"])
    base_grid = base_pair_report["base_grid"]
    transform = list(base_grid["local_transform"])
    height, width = (int(value) for value in base_grid["shape"])
    base_shape = (height, width)
    if base_grid.get("crs") != "EPSG:32649" or base_grid.get("resolution") != [10.0, 10.0]:
        raise ValueError("Annual-base pose graph must use the shared 10 m EPSG:32649 grid")

    base_hash_before = base_pair_report.get("base_sha256_before")
    base_hash_after = base_pair_report.get("base_sha256_after")
    if not isinstance(base_hash_before, dict) or base_hash_before != base_hash_after:
        raise ValueError("Annual base source hashes were not stable in the source diagnostic")

    cell_windows: dict[str, list[int]] = {}
    edges_by_cell: dict[str, list[TranslationEdge]] = {}
    boundary_exclusions: dict[str, int] = {}
    radius = int(base_pair_report["search_radius_px"])
    for pair in base_pair_report["pairwise_fields"]:
        reference_id = str(pair["reference_year_label"])
        moving_id = str(pair["moving_year_label"])
        for cell in pair["cells"]:
            row, col = int(cell["row"]), int(cell["col"])
            key = _cell_id(row, col)
            window = [int(v) for v in cell["base_pixel_window"]]
            if key in cell_windows and cell_windows[key] != window:
                raise ValueError(f"Base-pair cell grid changes between edges: {key}")
            cell_windows[key] = window
            correction = cell.get("correction_to_reference_m_candidate")
            best = cell.get("best")
            if correction is None or best is None:
                continue
            if (abs(int(best["dx_px"])) >= radius or
                    abs(int(best["dy_px"])) >= radius):
                boundary_exclusions[key] = boundary_exclusions.get(key, 0) + 1
                continue
            edges_by_cell.setdefault(key, []).append(TranslationEdge(
                reference_id=reference_id,
                moving_id=moving_id,
                correction_dx_m=float(correction[0]),
                correction_dy_m=float(correction[1]),
                source_ref=f"{reference_id}<-{moving_id}:{key}",
            ))
    if not cell_windows:
        raise ValueError("Annual-base pair diagnostic contains no local cells")

    pose_graphs = {}
    pose_rows = []
    for key in sorted(cell_windows):
        solution = solve_translation_pose_graph(edges_by_cell.get(key, ()), str(anchor_year))
        pose_graphs[key] = solution
        pose_rows.append({
            "cell_id": key,
            "support_pixel_window": cell_windows[key],
            "support_bounds_epsg32649": _support_bounds(transform, cell_windows[key]),
            "candidate_edge_count": len(solution.used_edge_sources),
            "search_boundary_edges_excluded": boundary_exclusions.get(key, 0),
            "connected_base_years": [int(value) for value in solution.connected_ids],
            "anchor_year": anchor_year,
            "anchor_relative_offsets_m_candidate": {
                int(year): [float(value) for value in offset]
                for year, offset in solution.offsets_m.items()
            },
            "cycle_degrees_of_freedom": solution.cycle_degrees_of_freedom,
            "closure_rmse_m": solution.closure_rmse_m,
            "closure_p90_m": solution.closure_p90_m,
            "closure_max_m": solution.closure_max_m,
            "status": solution.status,
            "edge_residuals_m": list(solution.edge_residuals_m),
            "reviewed": False,
            "applied": False,
        })

    date_fields = []
    for epoch_label, epoch in sorted(base_tie_report.get("epochs", {}).items()):
        actual_date = str(epoch.get("actual_date") or "")
        try:
            date.fromisoformat(actual_date)
        except ValueError as exc:
            raise ValueError(f"Missing real acquisition date for {epoch_label}") from exc
        source_base_year = _year_from_base_path(str(epoch["base_path"]))
        if source_base_year not in years:
            raise ValueError(f"Date tie points to unregistered annual base: {source_base_year}")
        if epoch.get("base_sha256") != base_pair_report["base_rasters"][str(source_base_year)]["sha256"]:
            raise ValueError(f"Actual-date tie and annual-base pose inputs differ: {actual_date}")
        tie_grid = epoch["base_grid"]
        if (tie_grid.get("crs") != base_grid["crs"] or
                list(tie_grid.get("transform", ())) != transform or
                [int(v) for v in tie_grid.get("window", ())] != [int(v) for v in base_grid["window"]]):
            raise ValueError(f"Actual-date tie grid does not match annual-base pose grid: {actual_date}")
        local = epoch.get("local_spatial_tie_diagnostic")
        if not isinstance(local, dict) or local.get("field_interpolated") or local.get("field_applied"):
            raise ValueError(f"Invalid or applied high-resolution tie field for {actual_date}")
        local_radius = int(local.get("search_radius_px",
                                     local.get("search_radius_10m_pixels", 0)))
        if local_radius < 1:
            raise ValueError(f"High-resolution tie field has no valid search radius: {actual_date}")
        local_by_cell = {}
        for cell in local.get("cells", []):
            key = _cell_id(int(cell["row"]), int(cell["col"]))
            if key not in cell_windows:
                raise ValueError(f"Actual-date tie cell is absent from the annual pose grid: {key}")
            window = [int(v) for v in cell["base_pixel_window"]]
            if window != cell_windows[key]:
                raise ValueError(f"Actual-date and annual-pose cell windows differ: {actual_date}/{key}")
            local_by_cell[key] = cell

        for key in sorted(cell_windows):
            cell = local_by_cell.get(key)
            solution = pose_graphs[key]
            base_offset = solution.offsets_m.get(str(source_base_year))
            is_anchor = source_base_year == anchor_year
            best = cell.get("best") if cell else None
            subpixel = best.get("subpixel_peak") if best else None
            tie_ok = bool(best and subpixel and subpixel.get("dx_px") is not None
                          and subpixel.get("dy_px") is not None)
            if best and (abs(int(best["dx_px"])) >= local_radius or
                         abs(int(best["dy_px"])) >= local_radius):
                tie_ok = False
            highres_correction = None
            composed = None
            if tie_ok and (is_anchor or base_offset is not None):
                highres_correction = image_shift_to_projected_correction(
                    float(subpixel["dx_px"]), float(subpixel["dy_px"]), transform)
                base_offset = (0.0, 0.0) if is_anchor else base_offset
                composed = (float(highres_correction[0] + base_offset[0]),
                            float(highres_correction[1] + base_offset[1]))

            if not tie_ok:
                status = "HIGHRES_BASE_TIE_NO_USABLE_SUBPIXEL_CANDIDATE"
            elif not is_anchor and base_offset is None:
                status = "ANNUAL_BASE_POSE_DISCONNECTED_FROM_ANCHOR"
            elif is_anchor:
                status = "COMPOSED_CANDIDATE_ANCHOR_BASE_DATE_UNKNOWN"
            elif solution.cycle_degrees_of_freedom == 0:
                status = "COMPOSED_CANDIDATE_TREE_HAS_NO_CYCLE_CHECK"
            elif (solution.closure_max_m is not None and
                  solution.closure_max_m > cycle_screening_limit_m):
                status = "COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED"
            else:
                status = "COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY"

            peak = float(best["corr"]) if best else None
            margin = float(cell["peak_margin"]) if cell and cell.get("peak_margin") is not None else None
            bounds = _support_bounds(transform, cell_windows[key])
            date_fields.append({
                "actual_date": actual_date,
                "source_epoch_label": str(epoch_label),
                "source_base_year_label": source_base_year,
                "source_base_path": str(epoch["base_path"]),
                "source_base_date_status": "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN",
                "reference_year_label": anchor_year,
                "reference_path": anchor_path,
                "reference_date": None,
                "reference_date_status": "REFERENCE_ACQUISITION_DATE_UNKNOWN",
                "cell_id": key,
                "row": int(cell["row"]) if cell else int(key[1:key.index("c")]),
                "col": int(cell["col"]) if cell else int(key[key.index("c") + 1:]),
                "support_pixel_window": cell_windows[key],
                "support_bounds_epsg32649": bounds,
                "highres_observed_shift_px": ([float(subpixel["dx_px"]), float(subpixel["dy_px"])]
                                               if tie_ok else None),
                "highres_to_source_base_correction_m_candidate": (
                    [float(v) for v in highres_correction] if highres_correction else None),
                "source_base_to_anchor_correction_m_candidate": (
                    [float(v) for v in base_offset] if base_offset is not None else None),
                "composed_correction_m_candidate": [float(v) for v in composed] if composed else None,
                "peak_correlation": peak,
                "peak_margin": margin,
                "eligible_pixels": int(best["n"]) if best else 0,
                "base_pose_edge_count": len(solution.used_edge_sources),
                "base_pose_cycle_degrees_of_freedom": solution.cycle_degrees_of_freedom,
                "base_pose_closure_rmse_m": solution.closure_rmse_m,
                "base_pose_closure_max_m": solution.closure_max_m,
                "cycle_screening_limit_m": cycle_screening_limit_m,
                "base_pose_status": ("ANCHOR_BY_DEFINITION_NOT_VALIDATED" if is_anchor
                                     else solution.status),
                "status": status,
                "reviewed": False,
                "independent_ground_evidence": False,
                "positional_uncertainty_m": None,
                "interpolated": False,
                "applied": False,
            })

    date_fields.sort(key=lambda row: (row["actual_date"], row["cell_id"]))
    return {
        "schema_version": "xian_r5.composed_alignment_candidates.v1",
        "anchor_year_label": anchor_year,
        "anchor_path": anchor_path,
        "anchor_acquisition_date_status": "ANNUAL_FILE_LABEL_ONLY_ACQUISITION_DATE_UNKNOWN",
        "crs": "EPSG:32649",
        "resolution_m": 10.0,
        "grid": {"transform": transform, "shape": list(base_shape),
                 "window": [int(v) for v in base_grid["window"]],
                 "bounds": list(base_grid["bounds"])},
        "direction": "ACTUAL_DATE_SUBMETER_IMAGE_TO_2019_ANNUAL_BASE;CORRECTIONS_ARE_MOVING_TO_REFERENCE",
        "composition": "SUBMETER_TO_NOMINAL_YEAR_BASE_CANDIDATE_PLUS_BASE_YEAR_TO_2019_POSE_CANDIDATE",
        "base_year_acquisition_dates_used": False,
        "search_boundary_ties_excluded": True,
        "cycle_screening_limit_m": cycle_screening_limit_m,
        "cycle_limit_semantics": "HALF_10M_PIXEL_INTERNAL_CLOSURE_SCREEN_ONLY_NOT_POSITIONAL_ACCURACY",
        "pose_graph_cells": pose_rows,
        "date_fields": date_fields,
        "field_interpolated": False,
        "field_applied": False,
        "imagery_resampled": False,
        "formal_ground_footprint_count": 0,
        "registration_promoted": False,
        "interpretation": "LOCAL_DIAGNOSTIC_CANDIDATE_ONLY_NOT_A_CORRECTED_IMAGE_OR_GROUND_FOOTPRINT",
    }
