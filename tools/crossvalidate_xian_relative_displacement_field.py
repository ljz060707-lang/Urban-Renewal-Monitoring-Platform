#!/usr/bin/env python3
"""LOO diagnostic: does a spatial plane explain static-FID relative shifts?"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import fiona
import numpy as np
from shapely.geometry import shape


ROOT = Path(r"F:\xian\XIAN_R5_MODEL_UPGRADE_20260923\18_reconstruction")
RUNS = {
    "B03-TEX04": ROOT / "B03-TEX04_full_mosaic_relative_offsets_01",
    "B01-TEX02": ROOT / "B01-TEX02_full_mosaic_relative_offsets_01",
    "B05-TEX01": ROOT / "B05-TEX01_full_mosaic_relative_offsets_01",
    "B04-TQA10": ROOT / "B04_full_mosaic_relative_offsets_02",
}
DEFAULT_OUTPUT = ROOT / "relative_displacement_field_cv_02"


def _mad(values: list[float]) -> float | None:
    if not values:
        return None
    center = median(values)
    return median([abs(value - center) for value in values])


def _errors(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = actual - predicted
    euclidean = np.sqrt(np.sum(residual * residual, axis=1))
    return {
        "cv_rmse_dx_m": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "cv_rmse_dy_m": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "cv_rmse_vector_m": float(np.sqrt(np.mean(euclidean ** 2))),
        "cv_median_abs_dx_m": float(np.median(np.abs(residual[:, 0]))),
        "cv_median_abs_dy_m": float(np.median(np.abs(residual[:, 1]))),
        "cv_p90_vector_error_m": float(np.quantile(euclidean, 0.9)),
    }


def _loo_models(samples: list[dict], roi_bounds: list[float]) -> dict:
    n = len(samples)
    if n < 3:
        return {"status": "INSUFFICIENT_MATCHES_FOR_FIELD", "n": n}
    coords = np.asarray([[row["x_m"], row["y_m"]] for row in samples], dtype=float)
    values = np.asarray([[row["apparent_dx_m"], row["apparent_dy_m"]] for row in samples], dtype=float)
    xmin, ymin, xmax, ymax = roi_bounds
    sx, sy = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    design = np.column_stack((np.ones(n), (coords[:, 0] - center_x) / sx, (coords[:, 1] - center_y) / sy))
    constant_prediction = np.zeros_like(values)
    plane_prediction = np.zeros_like(values)
    plane_cv_valid = True
    for holdout in range(n):
        train = np.arange(n) != holdout
        constant_prediction[holdout] = np.median(values[train], axis=0)
        beta, _, rank, _ = np.linalg.lstsq(design[train], values[train], rcond=None)
        if rank < 3:
            plane_cv_valid = False
            break
        plane_prediction[holdout] = design[holdout] @ beta
    if plane_cv_valid:
        plane_cv = _errors(values, plane_prediction)
        beta, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
        rank = int(rank)
        gradients = {
            "dx_per_km_x": float(beta[1, 0] * 1000 / sx),
            "dx_per_km_y": float(beta[2, 0] * 1000 / sy),
            "dy_per_km_x": float(beta[1, 1] * 1000 / sx),
            "dy_per_km_y": float(beta[2, 1] * 1000 / sy),
        }
    else:
        plane_cv, gradients, rank = None, None, None
    occupied = set()
    for x, y in coords:
        col = min(2, max(0, int((x - xmin) / max(xmax - xmin, 1.0) * 3)))
        row = min(2, max(0, int((y - ymin) / max(ymax - ymin, 1.0) * 3)))
        occupied.add((row, col))
    constant_cv = _errors(values, constant_prediction)
    supports_field = bool(plane_cv and len(occupied) >= 4 and plane_cv["cv_rmse_vector_m"] < 0.8 * constant_cv["cv_rmse_vector_m"])
    return {
        "status": "SPATIAL_PLANE_IMPROVES_LOO" if supports_field else "SPATIAL_PLANE_NOT_SUPPORTED_BY_LOO",
        "n": n, "distinct_fids": len({row["external_fid"] for row in samples}),
        "occupied_3x3_cells": len(occupied), "design_rank": rank,
        "constant_median_model_loo": constant_cv,
        "affine_plane_model_loo": plane_cv,
        "affine_plane_gradient_m_per_km": gradients,
        "all_sample_median_dx_m": float(np.median(values[:, 0])),
        "all_sample_median_dy_m": float(np.median(values[:, 1])),
        "all_sample_mad_dx_m": _mad(values[:, 0].tolist()),
        "all_sample_mad_dy_m": _mad(values[:, 1].tolist()),
        "model_applied": False,
    }


def _median_shift(samples: list[dict]) -> dict:
    if not samples:
        return {"n": 0, "median_dx_m": None, "median_dy_m": None,
                "mad_dx_m": None, "mad_dy_m": None}
    dx = [row["apparent_dx_m"] for row in samples]
    dy = [row["apparent_dy_m"] for row in samples]
    return {"n": len(samples), "median_dx_m": median(dx), "median_dy_m": median(dy),
            "mad_dx_m": _mad(dx), "mad_dy_m": _mad(dy)}


def _load_matches(run_dir: Path) -> tuple[dict, dict, dict]:
    diagnostic = json.loads((run_dir / "FULL_MOSAIC_RELATIVE_OFFSET_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    roofs, references = defaultdict(dict), defaultdict(dict)
    gpkg = run_dir / "LOCAL_STATIC_REFERENCE_MATCHES.gpkg"
    with fiona.open(gpkg, layer="r4_roof_matches") as source:
        for feature in source:
            props = dict(feature["properties"])
            key = (str(props["actual_date"]), str(props["external_fid"]))
            roofs[key] = props
    with fiona.open(gpkg, layer="external_polygon_matches") as source:
        for feature in source:
            props = dict(feature["properties"])
            key = (str(props["actual_date"]), str(props["external_fid"]))
            references[key] = shape(feature["geometry"])
    return diagnostic, roofs, references


def analyze(output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite displacement-field CV output: {output_dir}")
    case_results = {}
    for case_id, run_dir in RUNS.items():
        if not (run_dir / "FULL_MOSAIC_RELATIVE_OFFSET_DIAGNOSTIC.json").is_file():
            raise FileNotFoundError(run_dir / "FULL_MOSAIC_RELATIVE_OFFSET_DIAGNOSTIC.json")
        diagnostic, roofs, references = _load_matches(run_dir)
        actual_dates = [diagnostic["actual_dates"][epoch] for epoch in sorted(
            diagnostic["actual_dates"], key=lambda epoch: diagnostic["actual_dates"][epoch])]
        xmin, ymin, xmax, ymax = diagnostic["bounded_roi_epsg32649"]
        pair_results = []
        for earlier, later in zip(actual_dates, actual_dates[1:]):
            shared_fids = sorted({fid for date, fid in roofs if date == earlier}
                                 & {fid for date, fid in roofs if date == later})
            samples = []
            for fid in shared_fids:
                left, right = roofs[(earlier, fid)], roofs[(later, fid)]
                reference_geometry = references[(earlier, fid)]
                samples.append({
                    "external_fid": fid,
                    "x_m": reference_geometry.centroid.x,
                    "y_m": reference_geometry.centroid.y,
                    "apparent_dx_m": float(left["offset_dx_m"] - right["offset_dx_m"]),
                    "apparent_dy_m": float(left["offset_dy_m"] - right["offset_dy_m"]),
                    "left_r4_object_id": left["r4_object_id"],
                    "right_r4_object_id": right["r4_object_id"],
                    "left_iou": float(left["iou"]), "right_iou": float(right["iou"]),
                })
            iou_sensitivity = []
            for threshold in (0.05, 0.1, 0.15, 0.2, 0.3):
                selected = [row for row in samples if min(row["left_iou"], row["right_iou"]) >= threshold]
                iou_sensitivity.append({
                    "minimum_iou_each_date": threshold,
                    "shift_summary": _median_shift(selected),
                    "leave_one_out": _loo_models(selected, [xmin, ymin, xmax, ymax]),
                })
            strict = iou_sensitivity[-1]
            strict_summary = strict["shift_summary"]
            strict_loo = strict["leave_one_out"]
            strict_constant = strict_loo.get("constant_median_model_loo") or {}
            supported = bool(
                strict_summary["n"] >= 10
                and strict_loo.get("occupied_3x3_cells", 0) >= 4
                and (strict_summary["mad_dx_m"] or 0.0) <= 2.0
                and (strict_summary["mad_dy_m"] or 0.0) <= 2.0
                and strict_constant.get("cv_median_abs_dx_m", float("inf")) <= 2.0
                and strict_constant.get("cv_median_abs_dy_m", float("inf")) <= 2.0
            )
            translation_candidate = {
                "status": "RELATIVE_TRANSLATION_CANDIDATE_NOT_APPLIED" if supported else "INSUFFICIENT_OR_INCONSISTENT_SUPPORT",
                "minimum_iou_each_date": 0.3,
                "distinct_external_fid_count": strict_summary["n"],
                "occupied_3x3_cells": strict_loo.get("occupied_3x3_cells", 0),
                "observed_later_minus_earlier_roof_shift_m": {
                    "dx_m": strict_summary["median_dx_m"], "dy_m": strict_summary["median_dy_m"]},
                "candidate_translation_to_align_later_to_earlier_m": {
                    "dx_m": -strict_summary["median_dx_m"] if supported else None,
                    "dy_m": -strict_summary["median_dy_m"] if supported else None},
                "cross_fid_mad_m": {"dx": strict_summary["mad_dx_m"], "dy": strict_summary["mad_dy_m"]},
                "loo_median_absolute_error_m": {
                    "dx": strict_constant.get("cv_median_abs_dx_m"),
                    "dy": strict_constant.get("cv_median_abs_dy_m")},
                "ground_position_verified": False,
                "applied_to_geometry_or_raster": False,
            }
            pair_results.append({
                "earlier_actual_date": earlier, "later_actual_date": later,
                "match_count": len(samples),
                "median_displacement": {
                    "dx_m": median([row["apparent_dx_m"] for row in samples]) if samples else None,
                    "dy_m": median([row["apparent_dy_m"] for row in samples]) if samples else None,
                },
                "shift_mad": _median_shift(samples),
                "iou_threshold_sensitivity": iou_sensitivity,
                "relative_translation_candidate": translation_candidate,
                "leave_one_out_model_comparison": _loo_models(samples, [xmin, ymin, xmax, ymax]),
                "samples": samples,
            })
        all_pairs_supported = bool(pair_results) and all(
            row["relative_translation_candidate"]["status"] == "RELATIVE_TRANSLATION_CANDIDATE_NOT_APPLIED"
            for row in pair_results)
        cumulative = []
        if all_pairs_supported:
            cumulative.append({"actual_date": actual_dates[0],
                               "candidate_translation_to_reference_frame_m": {"dx_m": 0.0, "dy_m": 0.0}})
            total_dx = total_dy = 0.0
            for pair in pair_results:
                shift = pair["relative_translation_candidate"]["observed_later_minus_earlier_roof_shift_m"]
                total_dx += shift["dx_m"]
                total_dy += shift["dy_m"]
                cumulative.append({
                    "actual_date": pair["later_actual_date"],
                    "accumulated_observed_roof_shift_from_reference_m": {"dx_m": total_dx, "dy_m": total_dy},
                    "candidate_translation_to_reference_frame_m": {"dx_m": -total_dx, "dy_m": -total_dy},
                    "status": "RELATIVE_CANDIDATE_NOT_APPLIED",
                })
        case_results[case_id] = {
            "diagnostic_source": str(run_dir / "FULL_MOSAIC_RELATIVE_OFFSET_DIAGNOSTIC.json"),
            "source_accuracy_m": diagnostic["external_source"]["accuracy_m"],
            "external_source_capture_date": diagnostic["external_source"]["capture_date"],
            "external_source_ground_truth": diagnostic["external_source"]["ground_truth"],
            "epoch_pair_results": pair_results,
            "cumulative_relative_reference_frame_candidate": cumulative,
            "cumulative_candidate_status": "RELATIVE_TRANSLATIONS_SUPPORTED_NOT_APPLIED" if all_pairs_supported else "NOT_ESTABLISHED_FOR_ALL_DATES",
        }
    output = {
        "schema_version": "xian_r5.relative_displacement_field_cv.v1",
        "fit_basis": "same_static_external_FID_mutual_best_overlap_across_exact_tile_dates",
        "models": ["constant_coordinate_median", "affine_spatial_plane"],
        "cv_method": "leave_one_external_FID_out",
        "field_acceptance": "NONE; diagnostics do not authorize any correction",
        "case_results": case_results,
        "interpretation": "RELATIVE_IMAGE_OR_ROOF_SHIFT_ONLY; unknown vector-layer absolute bias does not cancel for ground positioning",
        "model_applied": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "RELATIVE_DISPLACEMENT_FIELD_CV.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output_dir)
    summaries = {}
    for case_id, case in result["case_results"].items():
        summaries[case_id] = [{
            "interval": f"{row['earlier_actual_date']}->{row['later_actual_date']}",
            "matches": row["match_count"],
            "median_dx_m": row["median_displacement"]["dx_m"],
            "median_dy_m": row["median_displacement"]["dy_m"],
            "constant_loo_rmse_m": (row["leave_one_out_model_comparison"].get("constant_median_model_loo") or {}).get("cv_rmse_vector_m"),
            "plane_loo_rmse_m": (row["leave_one_out_model_comparison"].get("affine_plane_model_loo") or {}).get("cv_rmse_vector_m"),
            "field_status": row["leave_one_out_model_comparison"].get("status"),
            "relative_translation_candidate": row["relative_translation_candidate"],
            "iou_sensitivity": [{
                "min_iou": sensitivity["minimum_iou_each_date"],
                "n": sensitivity["shift_summary"]["n"],
                "median_dx_m": sensitivity["shift_summary"]["median_dx_m"],
                "median_dy_m": sensitivity["shift_summary"]["median_dy_m"],
                "mad_dx_m": sensitivity["shift_summary"]["mad_dx_m"],
                "mad_dy_m": sensitivity["shift_summary"]["mad_dy_m"],
                "constant_loo_median_abs_dx": (sensitivity["leave_one_out"].get("constant_median_model_loo") or {}).get("cv_median_abs_dx_m"),
                "constant_loo_median_abs_dy": (sensitivity["leave_one_out"].get("constant_median_model_loo") or {}).get("cv_median_abs_dy_m"),
            } for sensitivity in row["iou_threshold_sensitivity"]],
        } for row in case["epoch_pair_results"]]
        summaries[case_id] = {"intervals": summaries[case_id],
                              "cumulative_candidate_status": case["cumulative_candidate_status"],
                              "cumulative_reference_frame": case["cumulative_relative_reference_frame_candidate"]}
    print(json.dumps(summaries, ensure_ascii=False))
