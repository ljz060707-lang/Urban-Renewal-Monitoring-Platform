#!/usr/bin/env python3
"""Expand a bounded fixed-site static-vector relative-shift check to its 11x12 AOI.

All vector sources remain candidate evidence. The tool reads only objects whose
bounds intersect the already-materialized local mosaic extent and never shifts
or rewrites source geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

import fiona
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xian_r5.alignment.relative_offsets import (
    solve_local_relative_pose_graphs,
    summarize_local_relative_shifts,
)
from xian_r5.alignment.relative_roof_candidate_products import (
    write_relative_roof_candidate_products,
)


ROOT = Path(r"F:\xian")
RUN_DIR = ROOT / "XIAN_R5_MODEL_UPGRADE_20260923" / "18_reconstruction" / "B04-TQA10_externalcandidate_02"
RECONSTRUCTION_DIR = ROOT / "XIAN_R5_MODEL_UPGRADE_20260923" / "18_reconstruction"
CASE_DIAGNOSTICS = {
    "B03-TEX04": "base_tie_B03_04_masked",
    "B01-TEX02": "base_tie_B01_02_masked",
    "B05-TEX01": "base_tie_B05_02_masked",
    "B04-TQA10": "base_tie_B04_02_masked",
}
FIXED_GEOMETRY = ROOT / "XIAN_R5_MODEL_UPGRADE_20260923" / "17_stage2" / "03_fixed_cases" / "R5_FIXED_CASE_GEOMETRY.gpkg"
TRANSFER_2018 = ROOT / "_xian_geom_transfer" / "2018-04-29_building_objects.gpkg"
FROZEN_GEOMETRY_DIR = ROOT / "西安_三期建筑追踪与三态最终正式候选版_20260916" / "02_三期建筑几何" / "remote_full_geometry"
R4_POSTCHECK = RUN_DIR / "R4_POSTCHECK.json"
EXTERNAL_LAYER = ROOT / "西安建筑轮廓数据" / "西安.shp"
DEFAULT_OUTPUT = RECONSTRUCTION_DIR / "B04_full_mosaic_relative_offsets_07"
BASE_RASTER_DIR = ROOT / "xian底图" / "xian搴曞浘"
REFERENCE_RASTER_STACK = tuple(
    BASE_RASTER_DIR / f"Xian_{year}_TrueColor.tif" for year in range(2017, 2023))
REFERENCE_RASTER = BASE_RASTER_DIR / "Xian_2019_TrueColor.tif"
MAX_MATCH_DISTANCE_M = 20.0
MIN_IOU = 0.05
MIN_INTERSECTION_M2 = 0.01
LOCAL_SHIFT_GRID_SHAPE = (4, 4)
LOCAL_SHIFT_MINIMUM_SUPPORT = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mad(values: list[float]) -> float | None:
    if not values:
        return None
    center = median(values)
    return median([abs(value - center) for value in values])


def _safe_properties(raw: dict) -> dict:
    return {str(key): (value if value is None or isinstance(value, (str, int, float, bool)) else str(value))
            for key, value in raw.items()}


def _load_local_roofs(path: Path, roi_bounds: tuple[float, float, float, float]) -> list[dict]:
    roofs = []
    with fiona.open(path, layer="buildings") as source:
        if CRS.from_user_input(source.crs_wkt or source.crs).to_epsg() != 32649:
            raise ValueError(f"Unexpected candidate geometry CRS: {path}")
        for feature in source.filter(bbox=roi_bounds):
            if not feature.get("geometry"):
                continue
            geometry = shape(feature["geometry"])
            if geometry.is_empty or not geometry.is_valid or not geometry.intersects(Polygon.from_bounds(*roi_bounds)):
                continue
            props = _safe_properties(dict(feature["properties"]))
            roofs.append({"object_id": str(props["object_id"]), "geometry": geometry, "properties": props})
    if len({item["object_id"] for item in roofs}) != len(roofs):
        raise ValueError(f"Duplicate object_id in local candidate geometry: {path}")
    return roofs


def _load_external(path: Path, roi_bounds: tuple[float, float, float, float], expected_hashes: dict) -> tuple[list[dict], str]:
    to_wgs = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    to_ground = None
    roi = Polygon.from_bounds(*roi_bounds)
    roi_wgs = transform(to_wgs.transform, roi.buffer(MAX_MATCH_DISTANCE_M))
    external = []
    with fiona.open(path) as source:
        crs_input = source.crs_wkt or source.crs
        if not crs_input:
            raise ValueError("External building layer has no CRS")
        source_crs = CRS.from_user_input(crs_input)
        crs_label = source_crs.to_string()
        to_ground = Transformer.from_crs(source_crs, "EPSG:32649", always_xy=True)
        for feature in source.filter(bbox=roi_wgs.bounds):
            if not feature.get("geometry"):
                continue
            raw_geometry = shape(feature["geometry"])
            geometry = transform(to_ground.transform, raw_geometry)
            if geometry.is_empty or not geometry.is_valid or not geometry.intersects(roi.buffer(MAX_MATCH_DISTANCE_M)):
                continue
            external.append({"fid": str(feature["id"]), "geometry": geometry,
                             "properties": _safe_properties(dict(feature["properties"]))})
    current_hashes = {}
    for name, expected in expected_hashes.items():
        component = path.with_name(name)
        actual = _sha256(component)
        if actual != expected:
            raise ValueError(f"External component hash changed: {component}")
        current_hashes[name] = actual
    return external, crs_label


def _fixed_targets(case_id: str, epoch: str) -> dict[str, object]:
    targets = {}
    with fiona.open(FIXED_GEOMETRY, layer="fixed_case_polygons") as source:
        for feature in source:
            props = dict(feature["properties"])
            if (props.get("case_id") == case_id and str(props.get("epoch_label")) == epoch
                    and props.get("role") == "TARGET"):
                targets[str(props["object_id"])] = shape(feature["geometry"])
    return targets


def _verify_fixed_subset(case_id: str, epoch: str, local_roofs: list[dict]) -> dict:
    targets = _fixed_targets(case_id, epoch)
    by_id = {item["object_id"]: item["geometry"] for item in local_roofs}
    exact = [object_id for object_id, geometry in targets.items()
             if object_id in by_id and geometry.equals(by_id[object_id])]
    if set(exact) != set(targets):
        raise ValueError(f"Local {epoch} candidate geometry fails fixed-case exact subset check")
    return {"fixed_target_count": len(targets), "exact_geometry_matches": len(exact),
            "verified_object_ids": sorted(exact), "status": "EXACT_SUBSET_MATCH"}


def _mutual_best_matches(external: list[dict], roofs: list[dict]) -> dict[str, dict]:
    edges_by_fid: dict[str, list[dict]] = defaultdict(list)
    edges_by_object: dict[str, list[dict]] = defaultdict(list)
    for ref in external:
        ext = ref["geometry"]
        ex0, ey0, ex1, ey1 = ext.bounds
        for roof in roofs:
            geom = roof["geometry"]
            rx0, ry0, rx1, ry1 = geom.bounds
            if (ex0 > rx1 + MAX_MATCH_DISTANCE_M or rx0 > ex1 + MAX_MATCH_DISTANCE_M
                    or ey0 > ry1 + MAX_MATCH_DISTANCE_M or ry0 > ey1 + MAX_MATCH_DISTANCE_M):
                continue
            distance = ext.distance(geom)
            if distance > MAX_MATCH_DISTANCE_M:
                continue
            intersection = ext.intersection(geom).area
            if intersection <= MIN_INTERSECTION_M2:
                continue
            union = ext.union(geom).area
            iou = intersection / union if union else 0.0
            if iou < MIN_IOU:
                continue
            edge = {"fid": ref["fid"], "external": ref, "roof": roof,
                    "distance_m": distance, "iou": iou,
                    "external_minus_roof_centroid_dx_m": ext.centroid.x - geom.centroid.x,
                    "external_minus_roof_centroid_dy_m": ext.centroid.y - geom.centroid.y}
            edges_by_fid[ref["fid"]].append(edge)
            edges_by_object[roof["object_id"]].append(edge)

    def rank(edge: dict) -> tuple[float, float, str, str]:
        return (edge["iou"], -edge["distance_m"], edge["fid"], edge["roof"]["object_id"])

    best_fid = {fid: max(edges, key=rank) for fid, edges in edges_by_fid.items()}
    best_object = {object_id: max(edges, key=rank) for object_id, edges in edges_by_object.items()}
    matches = {}
    for fid, edge in best_fid.items():
        reverse = best_object.get(edge["roof"]["object_id"])
        if reverse is edge:
            matches[fid] = edge
    return matches


def analyze(output_dir: Path, case_id: str) -> dict:
    if case_id not in CASE_DIAGNOSTICS:
        raise ValueError(f"Unsupported fixed case: {case_id}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite bounded AOI diagnostic: {output_dir}")
    receipt = json.loads((RUN_DIR / "EVIDENCE_RECEIPT.json").read_text(encoding="utf-8"))
    tie_diagnostic = RECONSTRUCTION_DIR / CASE_DIAGNOSTICS[case_id] / "BASE_TIE_DIAGNOSTIC.json"
    tie = json.loads(tie_diagnostic.read_text(encoding="utf-8"))
    postcheck = json.loads(R4_POSTCHECK.read_text(encoding="utf-8"))
    if receipt["external_footprint_candidate_source"].get("ground_truth") is not False:
        raise ValueError("External polygon source is not explicitly marked candidate-only")
    epochs = tie["epochs"]
    epoch_order = sorted(epochs, key=lambda label: epochs[label]["actual_date"])
    if any(epochs[label]["submeter_tiles"]["exact_date_coverage_fraction"] != 1.0 for label in epoch_order):
        raise ValueError(f"The {case_id} local imagery mosaic is not completely date-coherent")
    actual_dates = {label: epochs[label]["actual_date"] for label in epoch_order}
    roi_bounds = tuple(tie["epochs"][epoch_order[0]]["mosaic_bounds_epsg32649"])

    source_component_hashes = receipt["external_footprint_candidate_source"]["source_component_sha256"]
    external, external_crs = _load_external(EXTERNAL_LAYER, roi_bounds, source_component_hashes)
    postcheck_by_path = {row["path"].lower(): row for row in postcheck["files"].values()}
    geometry_paths = {
        "2018": TRANSFER_2018,
        "2019": FROZEN_GEOMETRY_DIR / "2019-01-01_building_objects.gpkg",
        "2020": FROZEN_GEOMETRY_DIR / "2020-12-31_building_objects.gpkg",
    }
    local_roofs, geometry_receipts, fixed_subset_checks = {}, {}, {}
    for epoch in epoch_order:
        path = geometry_paths[epoch]
        roofs = _load_local_roofs(path, roi_bounds)
        local_roofs[epoch] = roofs
        file_hash = _sha256(path)
        frozen_row = postcheck_by_path.get(str(path).lower())
        frozen_status = "NOT_IN_R4_FREEZE" if frozen_row is None else (
            "FROZEN_SHA_MATCH" if frozen_row["match"] and file_hash == frozen_row["expected_sha256"] else "FROZEN_SHA_MISMATCH")
        if frozen_status == "FROZEN_SHA_MISMATCH":
            raise ValueError(f"Frozen R4 geometry hash mismatch: {path}")
        geometry_receipts[epoch] = {"path": str(path), "sha256": file_hash,
                                    "features_in_b04_mosaic_bounds": len(roofs),
                                    "freeze_status": frozen_status}
        fixed_subset_checks[epoch] = _verify_fixed_subset(case_id, epoch, roofs)

    match_by_epoch = {}
    epoch_summary = {}
    for epoch in epoch_order:
        matches = _mutual_best_matches(external, local_roofs[epoch])
        match_by_epoch[epoch] = matches
        epoch_summary[epoch] = {
            "actual_date": actual_dates[epoch],
            "local_r4_candidate_roofs": len(local_roofs[epoch]),
            "external_features_in_mosaic_bounds": len(external),
            "mutual_best_overlap_matches": len(matches),
            "matched_external_fids": sorted(matches),
        }

    fids_all_dates = set.intersection(*(set(match_by_epoch[epoch]) for epoch in epoch_order))
    fids_by_epoch_pair = {}
    offset_rows = []
    relative_pose_samples = []
    anchor_epoch = epoch_order[0]
    roi_geometry = Polygon.from_bounds(*roi_bounds)
    for left_epoch, right_epoch in combinations(epoch_order, 2):
        left_matches, right_matches = match_by_epoch[left_epoch], match_by_epoch[right_epoch]
        shared = sorted(set(left_matches) & set(right_matches))
        changes = []
        for fid in shared:
            left, right = left_matches[fid], right_matches[fid]
            # The external layer is static in this comparison. Negating the
            # change in external-minus-roof offset gives roof motion relative
            # to that static candidate polygon, not an absolute ground vector.
            dx = left["external_minus_roof_centroid_dx_m"] - right["external_minus_roof_centroid_dx_m"]
            dy = left["external_minus_roof_centroid_dy_m"] - right["external_minus_roof_centroid_dy_m"]
            sample_geometry = left["roof"]["geometry"].intersection(
                Polygon.from_bounds(*roi_bounds))
            if sample_geometry.is_empty:
                raise ValueError(f"Matched roof has no support in the local mosaic ROI: {fid}")
            sample_point = sample_geometry.representative_point()
            row = {"external_fid": fid,
                   "source_properties": left["external"]["properties"],
                   "earlier_roof_roi_point_epsg32649": [
                       float(sample_point.x), float(sample_point.y)],
                   "left_object_id": left["roof"]["object_id"],
                   "right_object_id": right["roof"]["object_id"],
                   "left_iou": left["iou"], "right_iou": right["iou"],
                   "left_distance_m": left["distance_m"], "right_distance_m": right["distance_m"],
                   "apparent_roof_shift_relative_to_static_polygon_dx_m": dx,
                   "apparent_roof_shift_relative_to_static_polygon_dy_m": dy}
            changes.append(row)
        fids_by_epoch_pair[f"{actual_dates[left_epoch]}->{actual_dates[right_epoch]}"] = len(shared)
        local_shift_grid = summarize_local_relative_shifts(
            ({"sample_id": str(item["external_fid"]),
              "x_m": item["earlier_roof_roi_point_epsg32649"][0],
              "y_m": item["earlier_roof_roi_point_epsg32649"][1],
              "dx_m": item["apparent_roof_shift_relative_to_static_polygon_dx_m"],
              "dy_m": item["apparent_roof_shift_relative_to_static_polygon_dy_m"]}
             for item in changes),
            roi_bounds, grid_shape=LOCAL_SHIFT_GRID_SHAPE,
            minimum_support=LOCAL_SHIFT_MINIMUM_SUPPORT)
        offset_rows.append({"left_epoch": left_epoch, "right_epoch": right_epoch,
                            "earlier_actual_date": actual_dates[left_epoch],
                            "later_actual_date": actual_dates[right_epoch],
                            "shared_external_fid_count": len(shared), "fid_details": changes,
                            "local_relative_shift_grid": local_shift_grid})

        for fid in sorted(fids_all_dates):
            left, right = left_matches[fid], right_matches[fid]
            anchor_geometry = match_by_epoch[anchor_epoch][fid]["roof"]["geometry"]
            anchor_support = anchor_geometry.intersection(roi_geometry)
            if anchor_support.is_empty:
                raise ValueError(f"Three-date matched roof has no anchor-ROI support: {fid}")
            anchor_point = anchor_support.representative_point()
            relative_pose_samples.append({
                "sample_id": str(fid),
                "reference_id": actual_dates[left_epoch],
                "moving_id": actual_dates[right_epoch],
                "x_m": float(anchor_point.x), "y_m": float(anchor_point.y),
                "dx_m": (left["external_minus_roof_centroid_dx_m"]
                         - right["external_minus_roof_centroid_dx_m"]),
                "dy_m": (left["external_minus_roof_centroid_dy_m"]
                         - right["external_minus_roof_centroid_dy_m"]),
            })

    local_relative_pose_graph = solve_local_relative_pose_graphs(
        relative_pose_samples, roi_bounds,
        anchor_id=actual_dates[anchor_epoch],
        grid_shape=LOCAL_SHIFT_GRID_SHAPE,
        minimum_support=LOCAL_SHIFT_MINIMUM_SUPPORT)

    pair_summaries = []
    for row in offset_rows:
        dxs = [item["apparent_roof_shift_relative_to_static_polygon_dx_m"] for item in row["fid_details"]]
        dys = [item["apparent_roof_shift_relative_to_static_polygon_dy_m"] for item in row["fid_details"]]
        pair_summaries.append({
            "earlier_actual_date": row["earlier_actual_date"], "later_actual_date": row["later_actual_date"],
            "shared_external_fid_count": row["shared_external_fid_count"],
            "apparent_shift_median_dx_m": median(dxs) if dxs else None,
            "apparent_shift_median_dy_m": median(dys) if dys else None,
            "apparent_shift_mad_dx_m": _mad(dxs), "apparent_shift_mad_dy_m": _mad(dys),
            "numeric_local_candidate_cell_count": (
                row["local_relative_shift_grid"]["numeric_candidate_cell_count"]),
        })

    output_dir.mkdir(parents=True)
    relative_roof_candidate_products = write_relative_roof_candidate_products(
        output_dir=output_dir,
        reference_raster=REFERENCE_RASTER,
        reference_raster_stack=REFERENCE_RASTER_STACK,
        roi_bounds=roi_bounds,
        local_roofs_by_epoch=local_roofs,
        geometry_source_paths=geometry_paths,
        geometry_source_sha256={
            epoch: geometry_receipts[epoch]["sha256"] for epoch in epoch_order},
        actual_dates=actual_dates,
        pose_graph=local_relative_pose_graph,
        outline_source_path=str(EXTERNAL_LAYER),
        outline_source_component_sha256=source_component_hashes,
        outline_source_epoch_label="2019",
        outline_source_geometry_semantics="ROOF_BOUNDARY_FROM_IMAGE",
    )
    schema = {"geometry": "Unknown", "properties": {
        "case_id": "str:16",
        "epoch": "str:8", "actual_date": "str:16", "r4_object_id": "str:24",
        "external_fid": "str:24", "external_id": "int", "floor": "int",
        "iou": "float", "distance_m": "float", "offset_dx_m": "float", "offset_dy_m": "float",
        "status": "str:40"}}
    output_gpkg = output_dir / "LOCAL_STATIC_REFERENCE_MATCHES.gpkg"
    with fiona.open(output_gpkg, "w", driver="GPKG", layer="r4_roof_matches", schema=schema, crs="EPSG:32649") as sink:
        for epoch in epoch_order:
            for fid, edge in match_by_epoch[epoch].items():
                props = edge["external"]["properties"]
                sink.write({"geometry": mapping(edge["roof"]["geometry"]), "properties": {
                    "case_id": case_id,
                    "epoch": epoch, "actual_date": actual_dates[epoch], "r4_object_id": edge["roof"]["object_id"],
                    "external_fid": fid, "external_id": props.get("Id"), "floor": props.get("Floor"),
                    "iou": edge["iou"], "distance_m": edge["distance_m"],
                    "offset_dx_m": edge["external_minus_roof_centroid_dx_m"],
                    "offset_dy_m": edge["external_minus_roof_centroid_dy_m"],
                    "status": "MUTUAL_BEST_CANDIDATE_NOT_GROUND_VERIFIED"}})
    with fiona.open(output_gpkg, "w", driver="GPKG", layer="external_polygon_matches", schema=schema, crs="EPSG:32649") as sink:
        for epoch in epoch_order:
            for fid, edge in match_by_epoch[epoch].items():
                props = edge["external"]["properties"]
                sink.write({"geometry": mapping(edge["external"]["geometry"]), "properties": {
                    "case_id": case_id,
                    "epoch": epoch, "actual_date": actual_dates[epoch], "r4_object_id": edge["roof"]["object_id"],
                    "external_fid": fid, "external_id": props.get("Id"), "floor": props.get("Floor"),
                    "iou": edge["iou"], "distance_m": edge["distance_m"],
                    "offset_dx_m": edge["external_minus_roof_centroid_dx_m"],
                    "offset_dy_m": edge["external_minus_roof_centroid_dy_m"],
                    "status": "MUTUAL_BEST_CANDIDATE_NOT_GROUND_VERIFIED"}})

    local_cell_schema = {"geometry": "Polygon", "properties": {
        "case_id": "str:16", "earlier_date": "str:16", "later_date": "str:16",
        "row": "int", "column": "int", "sample_count": "int",
        "median_dx_m": "float", "median_dy_m": "float",
        "mad_dx_m": "float", "mad_dy_m": "float",
        "radial_m": "float", "status": "str:40",
        "absolute_ground": "int", "applied": "int"}}
    for pair in offset_rows:
        layer_name = ("relative_shift_" + pair["earlier_actual_date"].replace("-", "")
                      + "_" + pair["later_actual_date"].replace("-", ""))
        with fiona.open(output_gpkg, "w", driver="GPKG", layer=layer_name,
                        schema=local_cell_schema, crs="EPSG:32649") as sink:
            for cell in pair["local_relative_shift_grid"]["cells"]:
                sink.write({"geometry": mapping(Polygon.from_bounds(
                    *cell["bounds_epsg32649"])), "properties": {
                    "case_id": case_id,
                    "earlier_date": pair["earlier_actual_date"],
                    "later_date": pair["later_actual_date"],
                    "row": cell["row"], "column": cell["column"],
                    "sample_count": cell["sample_count"],
                    "median_dx_m": cell["median_dx_m"],
                    "median_dy_m": cell["median_dy_m"],
                    "mad_dx_m": cell["mad_dx_m"], "mad_dy_m": cell["mad_dy_m"],
                    "radial_m": cell["median_radial_deviation_m"],
                    "status": cell["status"], "absolute_ground": 0, "applied": 0}})

    pose_cell_schema = {"geometry": "Polygon", "properties": {
        "case_id": "str:16", "anchor_date": "str:16", "row": "int",
        "column": "int", "edge_count": "int", "connected_json": "str:128",
        "offsets_json": "str:1024", "cycle_dof": "int",
        "closure_rmse_m": "float", "closure_p90_m": "float",
        "closure_max_m": "float", "status": "str:48",
        "absolute_ground": "int", "applied": "int"}}
    with fiona.open(output_gpkg, "w", driver="GPKG", layer="relative_pose_graph",
                    schema=pose_cell_schema, crs="EPSG:32649") as sink:
        for cell in local_relative_pose_graph["cells"]:
            sink.write({"geometry": mapping(Polygon.from_bounds(
                *cell["bounds_epsg32649"])), "properties": {
                "case_id": case_id,
                "anchor_date": local_relative_pose_graph["anchor_id"],
                "row": cell["row"], "column": cell["column"],
                "edge_count": len(cell["pairwise_candidate_edges"]),
                "connected_json": json.dumps(cell["connected_actual_dates"]),
                "offsets_json": json.dumps(
                    cell["candidate_corrections_to_anchor_m"], sort_keys=True),
                "cycle_dof": cell["cycle_degrees_of_freedom"],
                "closure_rmse_m": cell["closure_rmse_m"],
                "closure_p90_m": cell["closure_p90_m"],
                "closure_max_m": cell["closure_max_m"],
                "status": cell["status"], "absolute_ground": 0, "applied": 0}})

    result = {
        "schema_version": "xian_r5.fixed_site_full_mosaic_relative_offset_diagnostic.v4",
        "case_id": case_id,
        "bounded_roi_epsg32649": list(roi_bounds),
        "roi_source": str(tie_diagnostic),
        "actual_dates": actual_dates,
        "submeter_exact_date_coverage": {epoch: epochs[epoch]["submeter_tiles"]["exact_date_coverage_fraction"] for epoch in epoch_order},
        "external_source": {"path": str(EXTERNAL_LAYER), "crs": external_crs,
                            "capture_date": None, "accuracy_m": None, "ground_truth": False,
                            "user_reported_epoch_label": "2019",
                            "user_reported_epoch_label_provenance": "USER_REPORTED_NOT_EXACT_ACQUISITION_DATE",
                            "user_reported_geometry_semantics": "IMAGE_DRAWN_ROOF_BOUNDARY",
                            "geometry_semantics_code": "ROOF_BOUNDARY_FROM_IMAGE",
                            "geometry_semantics_provenance": "USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED",
                            "component_sha256_revalidated": source_component_hashes},
        "r4_geometry_sources": geometry_receipts,
        "fixed_case_geometry_exact_subset_checks": fixed_subset_checks,
        "matching_parameters": {"max_distance_m": MAX_MATCH_DISTANCE_M,
                                 "min_iou": MIN_IOU, "min_intersection_m2": MIN_INTERSECTION_M2,
                                 "association": "mutual_best_overlap_per_epoch"},
        "epoch_match_summary": epoch_summary,
        "external_fids_matched_all_three_dates": sorted(fids_all_dates),
        "external_fid_count_all_three_dates": len(fids_all_dates),
        "epoch_pair_offset_details": offset_rows,
        "epoch_pair_offset_summaries": pair_summaries,
        "local_relative_pose_graph": local_relative_pose_graph,
        "relative_roof_candidate_products": relative_roof_candidate_products,
        "local_relative_shift_policy": {
            "grid_shape_rows_columns": list(LOCAL_SHIFT_GRID_SHAPE),
            "minimum_support": LOCAL_SHIFT_MINIMUM_SUPPORT,
            "robust_summary": "PER_CELL_MEDIAN_AND_MEDIAN_ABSOLUTE_DEVIATION",
            "source_semantics": "RELATIVE_TO_UNVERIFIED_STATIC_REFERENCE_ONLY",
            "interpolated": False, "applied": False,
            "absolute_ground_position_resolved": False,
        },
        "local_relative_pose_graph_policy": {
            "reference_date": actual_dates[anchor_epoch],
            "edge_observations": "SAME_EXTERNAL_FIDS_MATCHED_IN_ALL_THREE_ACTUAL_DATES",
            "apparent_motion_to_alignment_correction": "NEGATE_X_AND_Y",
            "cycle_diagnostic_is_positional_accuracy": False,
            "absolute_ground_position_resolved": False,
            "interpolated": False,
            "applied": False,
        },
        "matched_geometry_output": str(output_gpkg),
        "relative_roof_candidate_manifest": relative_roof_candidate_products["manifest_path"],
        "relative_roof_candidate_manifest_sha256": _sha256(
            Path(relative_roof_candidate_products["manifest_path"])),
        "interpretation": "DIFFERENTIAL_LOCAL_REGISTRATION_CANDIDATE_ONLY; source bias may cancel across dates but absolute ground position is unverified",
        "registration_or_footprint_translation_applied": False,
        "relative_roof_candidate_geometry_exported": True,
        "formal_ground_footprints_or_changes_generated": False,
    }
    (output_dir / "FULL_MOSAIC_RELATIVE_OFFSET_DIAGNOSTIC.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=sorted(CASE_DIAGNOSTICS), default="B04-TQA10")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output_dir, args.case_id)
    print(json.dumps({
        "case_id": result["case_id"],
        "roi_features": {epoch: {key: value for key, value in summary.items() if key != "matched_external_fids"}
                         for epoch, summary in result["epoch_match_summary"].items()},
        "all_three_dates_external_fids": result["external_fid_count_all_three_dates"],
        "epoch_pair_offset_summaries": result["epoch_pair_offset_summaries"],
        "local_pose_graph": {
            "anchor_id": result["local_relative_pose_graph"]["anchor_id"],
            "numeric_cells": sum(
                bool(cell["pairwise_candidate_edges"])
                for cell in result["local_relative_pose_graph"]["cells"]),
            "cycle_cells": sum(
                cell["status"] == "RELATIVE_POSE_CYCLE_DIAGNOSTIC_CANDIDATE"
                for cell in result["local_relative_pose_graph"]["cells"]),
            "status_counts": dict(Counter(
                cell["status"] for cell in result["local_relative_pose_graph"]["cells"])),
            "maximum_cycle_closure_m": max((
                cell["closure_max_m"] or 0.0
                for cell in result["local_relative_pose_graph"]["cells"]), default=0.0),
            "applied": result["local_relative_pose_graph"]["applied"],
        },
        "local_relative_shift_policy": result["local_relative_shift_policy"],
        "local_numeric_candidate_cells": [
            pair["local_relative_shift_grid"]["numeric_candidate_cell_count"]
            for pair in result["epoch_pair_offset_details"]],
        "translation_applied": result["registration_or_footprint_translation_applied"],
    }, ensure_ascii=False))
