#!/usr/bin/env python3
"""Bounded cross-date comparison to the undocumented local building layer.

All output remains candidate evidence: the shapefile has no capture date or
accuracy metadata in the supplied folder, so none of its polygons are promoted
to ground footprints.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

import fiona
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform


ROOT = Path(r"F:\xian")
STAGE2 = ROOT / "XIAN_R5_MODEL_UPGRADE_20260923" / "17_stage2"
OUT_ROOT = ROOT / "XIAN_R5_MODEL_UPGRADE_20260923" / "18_reconstruction"
SHAPEFILE = ROOT / "西安建筑轮廓数据" / "西安.shp"
FIXED_MAP = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv"
FIXED_GEOM = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_GEOMETRY.gpkg"
FREEZE = STAGE2 / "13_receipts" / "R4_FREEZE_STAGE2.json"
CASE_DIAG = {
    "B03-TEX04": "base_tie_B03_04_masked",
    "B01-TEX02": "base_tie_B01_02_masked",
    "B05-TEX01": "base_tie_B05_02_masked",
    "B04-TQA10": "base_tie_B04_02_masked",
}
SCENE = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
ORIGIN_X, ORIGIN_Y = 420534, 311895
ORIGIN_LON, ORIGIN_LAT = 108.75778198242188, 34.16130065917969
STEP = 0.0006866455078125
TO_UTM = Transformer.from_crs(4326, 32649, always_xy=True)
TO_WGS = Transformer.from_crs(32649, 4326, always_xy=True)


def _pixel_geometry(geometry, x0: int, y_top: int, tile_count: int):
    def convert(x, y, z=None):
        lon, lat = TO_WGS.transform(x, y)
        px = (lon - (ORIGIN_LON + (x0 - ORIGIN_X) * STEP)) / STEP * 256
        py = ((ORIGIN_LAT + (y_top - ORIGIN_Y) * STEP) - lat) / STEP * 256
        return (px, py) if z is None else (px, py, z)
    return transform(convert, geometry)


def _draw(draw, geometry, color, width=2):
    if geometry.geom_type == "Polygon":
        draw.line([(round(x), round(y)) for x, y in geometry.exterior.coords], fill=color, width=width)
        for ring in geometry.interiors:
            draw.line([(round(x), round(y)) for x, y in ring.coords], fill=color, width=width)
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            _draw(draw, part, color, width)


def _tile_bounds(tile_x: int, tile_y: int):
    west = ORIGIN_LON + (tile_x - ORIGIN_X) * STEP
    south = ORIGIN_LAT + (tile_y - ORIGIN_Y) * STEP
    east, north = west + STEP, south + STEP
    corners = [TO_UTM.transform(west, south), TO_UTM.transform(east, south),
               TO_UTM.transform(east, north), TO_UTM.transform(west, north)]
    return Polygon(corners).bounds


def _match_one(roof, external):
    if not external:
        return {"status": "NO_LOCAL_EXTERNAL_POLYGON"}
    ordered = sorted(((roof.distance(item["geometry"]), item) for item in external), key=lambda p: p[0])
    nearest_distance, nearest_item = ordered[0]
    choices = []
    for distance, item in ordered:
        if distance > 20:
            break
        ground = item["geometry"]
        intersection = roof.intersection(ground).area
        union = roof.union(ground).area
        iou = intersection / union if union else 0.0
        choices.append((iou, -distance, item, distance, intersection))
    if nearest_distance > 20.0:
        return {"status": "NO_EXTERNAL_POLYGON_WITHIN_20M",
                "nearest_polygon_distance_m": float(nearest_distance)}
    if choices:
        iou, _, selected, distance, intersection = max(choices)
    else:
        return {"status": "NO_OVERLAP_WITH_CANDIDATE_WITHIN_20M",
                "nearest_polygon_distance_m": float(nearest_distance)}
    if iou <= 0:
        return {"status": "NO_OVERLAP_WITH_CANDIDATE_WITHIN_20M",
                "nearest_polygon_distance_m": float(nearest_distance)}
    roof_centroid, external_centroid = roof.centroid, selected["geometry"].centroid
    return {
        "status": "OVERLAPPING_CANDIDATE_NOT_GROUND_VERIFIED",
        "external_fid": selected["fid"], "external_floor": selected["properties"].get("Floor"),
        "nearest_polygon_distance_m": float(nearest_distance),
        "selected_polygon_distance_m": float(distance), "intersection_over_union": float(iou),
        "roof_overlap_fraction": float(intersection / max(roof.area, 1e-9)),
        "external_overlap_fraction": float(intersection / max(selected["geometry"].area, 1e-9)),
        "external_to_roof_area_ratio": float(selected["geometry"].area / max(roof.area, 1e-9)),
        "external_minus_roof_centroid_dx_m": float(external_centroid.x - roof_centroid.x),
        "external_minus_roof_centroid_dy_m": float(external_centroid.y - roof_centroid.y),
    }


def run() -> dict:
    output = OUT_ROOT / "external_footprint_crossdate_02"
    if output.exists():
        raise FileExistsError(output)
    with FIXED_MAP.open("r", encoding="utf-8-sig", newline="") as source:
        map_rows = list(csv.DictReader(source))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    by_case_epoch = defaultdict(list)
    for row in map_rows:
        if row["case_id"] in CASE_DIAG and row.get("r4_polygon_object_id") and row.get("pixel_available", "").lower() == "true":
            by_case_epoch[(row["case_id"], row["epoch_label"])].append(row)
    fixed_geometries = defaultdict(dict)
    with fiona.open(FIXED_GEOM, layer="fixed_case_polygons") as layer:
        for feature in layer:
            props = dict(feature["properties"])
            if props.get("case_id") in CASE_DIAG and props.get("role") == "TARGET":
                fixed_geometries[(props["case_id"], str(props["epoch_label"]))][str(props["object_id"])] = (shape(feature["geometry"]), props)
    outputs, track_rows = [], defaultdict(list)
    for case_id, diag_dirname in CASE_DIAG.items():
        diagnostic = json.loads((OUT_ROOT / diag_dirname / "BASE_TIE_DIAGNOSTIC.json").read_text(encoding="utf-8"))
        for epoch, epoch_info in diagnostic["epochs"].items():
            rows = by_case_epoch.get((case_id, epoch), [])
            if not rows:
                continue
            actual_dates = sorted({row["actual_date"] for row in rows})
            if len(actual_dates) != 1:
                raise ValueError(f"Actual-date mixture in {case_id}/{epoch}: {actual_dates}")
            tile_files = {item["tile_id"]: Path(item["path"]) for item in epoch_info["submeter_tiles"]["tile_ids"]}
            centers = {(int(row["tile_x"]), int(row["tile_y"])) for row in rows if row.get("tile_x") and row.get("tile_y")}
            cx, cy = round(sum(x for x, _ in centers) / len(centers)), round(sum(y for _, y in centers) / len(centers))
            required = {(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            if any(f"{x}:{y}" not in tile_files for x, y in required):
                continue
            mosaic = np.zeros((768, 768, 3), dtype=np.uint8)
            for row_index, y in enumerate((cy + 1, cy, cy - 1)):
                for col_index, x in enumerate((cx - 1, cx, cx + 1)):
                    image = Image.open(tile_files[f"{x}:{y}"]).convert("RGB")
                    mosaic[row_index * 256:(row_index + 1) * 256, col_index * 256:(col_index + 1) * 256] = np.asarray(image)
            x0, y0, y_top = cx - 1, cy - 1, cy + 2
            local_bounds = Polygon([TO_UTM.transform(ORIGIN_LON + (x0 - ORIGIN_X) * STEP, ORIGIN_LAT + (y0 - ORIGIN_Y) * STEP),
                                    TO_UTM.transform(ORIGIN_LON + (x0 + 3 - ORIGIN_X) * STEP, ORIGIN_LAT + (y0 - ORIGIN_Y) * STEP),
                                    TO_UTM.transform(ORIGIN_LON + (x0 + 3 - ORIGIN_X) * STEP, ORIGIN_LAT + (y0 + 3 - ORIGIN_Y) * STEP),
                                    TO_UTM.transform(ORIGIN_LON + (x0 - ORIGIN_X) * STEP, ORIGIN_LAT + (y0 + 3 - ORIGIN_Y) * STEP)]).bounds
            local_wgs = [TO_WGS.transform(local_bounds[0], local_bounds[1]), TO_WGS.transform(local_bounds[2], local_bounds[3])]
            shapefile_bbox = (min(p[0] for p in local_wgs), min(p[1] for p in local_wgs), max(p[0] for p in local_wgs), max(p[1] for p in local_wgs))
            external = []
            with fiona.open(SHAPEFILE) as source:
                for feature in source.filter(bbox=shapefile_bbox):
                    if feature.get("geometry"):
                        polygon = transform(TO_UTM.transform, shape(feature["geometry"]))
                        if polygon.is_valid and not polygon.is_empty and polygon.intersects(Polygon([
                            TO_UTM.transform(shapefile_bbox[0], shapefile_bbox[1]),
                            TO_UTM.transform(shapefile_bbox[2], shapefile_bbox[1]),
                            TO_UTM.transform(shapefile_bbox[2], shapefile_bbox[3]),
                            TO_UTM.transform(shapefile_bbox[0], shapefile_bbox[3])])):
                            external.append({"fid": feature["id"], "geometry": polygon, "properties": dict(feature["properties"])})
            target_items, visual_targets = [], []
            seen = set()
            for row in rows:
                object_id = str(row["r4_polygon_object_id"])
                obs = fixed_geometries.get((case_id, epoch), {}).get(object_id)
                if obs is None:
                    continue
                geometry, properties = obs
                key = (object_id, row.get("physical_track_id"))
                if key in seen:
                    continue
                seen.add(key)
                comparison = _match_one(geometry, external)
                item = {"case_id": case_id, "epoch_label": epoch, "actual_date": actual_dates[0],
                        "object_id": object_id, "object_uid": row.get("object_uid"),
                        "case_track_id": row.get("case_track_id"), "physical_track_id": row.get("physical_track_id"),
                        "r4_roof_area_m2": geometry.area, **comparison}
                target_items.append(item)
                visual_targets.append(geometry)
                if item.get("physical_track_id"):
                    track_rows[(case_id, item["physical_track_id"])].append(item)
            image = Image.fromarray(mosaic)
            draw = ImageDraw.Draw(image)
            for candidate in visual_targets:
                _draw(draw, _pixel_geometry(candidate, x0, y_top, 3), (255, 40, 30), 2)
            for candidate in external:
                _draw(draw, _pixel_geometry(candidate["geometry"], x0, y_top, 3), (0, 255, 255), 2)
            draw.text((6, 4), f"{case_id} {actual_dates[0]} | red R4 roof | cyan local layer", fill=(255, 255, 0))
            overlay = output / f"{case_id}_{epoch}_{actual_dates[0]}_overlay.png"
            image.save(overlay)
            outputs.append({"case_id": case_id, "epoch_label": epoch, "actual_date": actual_dates[0],
                            "date_verified_3x3_tiles": 9, "external_features": len(external),
                            "target_observations": len(target_items), "overlay": str(overlay),
                            "targets": target_items})
    track_summary = []
    for (case_id, track_id), items in sorted(track_rows.items()):
        items = sorted(items, key=lambda item: item["actual_date"])
        observed_external_fids = [item.get("external_fid") for item in items
                                  if item.get("external_fid") is not None
                                  and item.get("status") == "OVERLAPPING_CANDIDATE_NOT_GROUND_VERIFIED"]
        paired_dates = {item["actual_date"] for item in items if item.get("external_fid") is not None
                        and item.get("status") == "OVERLAPPING_CANDIDATE_NOT_GROUND_VERIFIED"}
        track_summary.append({"case_id": case_id, "physical_track_id": track_id,
                              "observations": [{"actual_date": item["actual_date"], "epoch": item["epoch_label"],
                                                "object_id": item["object_id"], "external_fid": item.get("external_fid"),
                                                "offset_dx_m": item.get("external_minus_roof_centroid_dx_m"),
                                                "offset_dy_m": item.get("external_minus_roof_centroid_dy_m"),
                                                "iou": item.get("intersection_over_union"),
                                                "match_status": item["status"]} for item in items],
                              "overlap_candidate_dates": sorted(paired_dates),
                              "same_external_fid_across_dates": len(paired_dates) >= 2 and len(set(observed_external_fids)) == 1})
    result = {"schema_version": "xian_r5.external_footprint_crossdate.v1",
              "source_layer": str(SHAPEFILE), "source_crs": "EPSG:4326",
              "source_date": "UNKNOWN", "source_accuracy": "UNKNOWN", "ground_truth": False,
              "r4_geometry_sha256": {epoch: freeze["files"][f"geometry_{epoch}"]["sha256"] for epoch in ("2018", "2019", "2020")},
              "cases": outputs, "physical_track_candidate_comparison": track_summary,
              "registration_or_footprint_shift_promoted": False}
    (output / "CROSS_DATE_COMPARISON.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"case_epochs": len(report["cases"]), "target_observations": sum(case["target_observations"] for case in report["cases"]),
                      "candidate_tracks": len(report["physical_track_candidate_comparison"]),
                      "same_external_fid_tracks": sum(track["same_external_fid_across_dates"] for track in report["physical_track_candidate_comparison"]),
                      "output": str(OUT_ROOT / "external_footprint_crossdate_02")}, ensure_ascii=False))
