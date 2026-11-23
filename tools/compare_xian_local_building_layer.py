#!/usr/bin/env python3
"""Compare the local WGS84 building layer to R4 outlines on exact-date tiles.

The source has no declared capture date/source in its available sidecars. This
script reports geometric agreement only; it does not designate that layer as
ground truth.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import fiona
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform, unary_union


ROOT = Path(r"F:\xian")
STAGE2 = ROOT / "XIAN_R5_MODEL_UPGRADE_20260923" / "17_stage2"
OUT_ROOT = ROOT / "XIAN_R5_MODEL_UPGRADE_20260923" / "18_reconstruction"
SHAPEFILE = ROOT / "西安建筑轮廓数据" / "西安.shp"
FIXED_MAP = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_TILE_MAP.csv"
FIXED_GEOM = STAGE2 / "03_fixed_cases" / "R5_FIXED_CASE_GEOMETRY.gpkg"
FREEZE = STAGE2 / "13_receipts" / "R4_FREEZE_STAGE2.json"
DATE_DIRS = {
    "B03-TEX04": "base_tie_B03_04_masked",
    "B01-TEX02": "base_tie_B01_02_masked",
    "B05-TEX01": "base_tie_B05_02_masked",
    "B04-TQA10": "base_tie_B04_02_masked",
}
ORIGIN_X, ORIGIN_Y = 420534, 311895
ORIGIN_LON, ORIGIN_LAT = 108.75778198242188, 34.16130065917969
STEP = 0.0006866455078125
TO_UTM = Transformer.from_crs(4326, 32649, always_xy=True)
TO_WGS = Transformer.from_crs(32649, 4326, always_xy=True)


def _tile_ll(x: int, y: int) -> tuple[float, float]:
    return ORIGIN_LON + (x - ORIGIN_X) * STEP, ORIGIN_LAT + (y - ORIGIN_Y) * STEP


def _pixel_coords(geom, x0: int, y_top: int, width: int, height: int):
    def project_xy(x, y, z=None):
        lon, lat = TO_WGS.transform(x, y)
        px = (lon - (ORIGIN_LON + (x0 - ORIGIN_X) * STEP)) / (width * STEP) * width * 256
        py = ((ORIGIN_LAT + (y_top - ORIGIN_Y) * STEP) - lat) / (height * STEP) * height * 256
        return (px, py) if z is None else (px, py, z)
    return transform(project_xy, geom)


def _draw_geometry(draw, geom, fill, width=1):
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        rings = [geom.exterior, *geom.interiors]
        for ring in rings:
            draw.line([(round(x), round(y)) for x, y in ring.coords], fill=fill, width=width)
    elif geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geom.geoms:
            _draw_geometry(draw, part, fill, width)


def run(case_ids: list[str]) -> dict:
    output = OUT_ROOT / "independent_ground_comparison_04"
    if output.exists():
        raise FileExistsError(output)
    with FIXED_MAP.open("r", encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    summary = []
    output.mkdir(parents=True)
    for case_id in case_ids:
        case_rows = [row for row in rows if row["case_id"] == case_id and row["epoch_label"] == "2018"
                     and row.get("pixel_available", "").lower() == "true"]
        if not case_rows:
            summary.append({"case_id": case_id, "status": "NO_2018_PIXEL_MAPPED_ROW"})
            continue
        date = {row["actual_date"] for row in case_rows}
        if len(date) != 1:
            raise ValueError(f"Date heterogeneity for {case_id}: {date}")
        actual_date = next(iter(date))
        known = json.loads((OUT_ROOT / DATE_DIRS[case_id] / "BASE_TIE_DIAGNOSTIC.json").read_text(encoding="utf-8"))
        tile_paths = {item["tile_id"]: Path(item["path"]) for item in known["epochs"]["2018"]["submeter_tiles"]["tile_ids"]}
        tile_ids = set(tile_paths)
        centers = {(int(row["tile_x"]), int(row["tile_y"])) for row in case_rows}
        cx = round(sum(x for x, _ in centers) / len(centers))
        cy = round(sum(y for _, y in centers) / len(centers))
        center_tiles = {(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        missing = sorted(f"{x}:{y}" for x, y in center_tiles if f"{x}:{y}" not in tile_ids)
        if missing:
            raise ValueError(f"Center image window not date-confirmed for {case_id}: {missing}")
        x0, y0 = cx - 1, cy - 1
        mosaic = np.zeros((3 * 256, 3 * 256, 3), dtype=np.uint8)
        for row_index, y in enumerate((cy + 1, cy, cy - 1)):
            for col_index, x in enumerate((cx - 1, cx, cx + 1)):
                path = tile_paths[f"{x}:{y}"]
                tile = np.asarray(Image.open(path).convert("RGB"))
                if tile.shape != (256, 256, 3):
                    raise ValueError(f"Invalid tile size: {path}")
                mosaic[row_index * 256:(row_index + 1) * 256,
                       col_index * 256:(col_index + 1) * 256] = tile
        west, south = _tile_ll(x0, y0)
        east, north = _tile_ll(x0 + 3, y0 + 3)
        bbox_utm = Polygon([TO_UTM.transform(west, south), TO_UTM.transform(east, south),
                            TO_UTM.transform(east, north), TO_UTM.transform(west, north)]).bounds
        bbox_wgs_points = [TO_WGS.transform(bbox_utm[0], bbox_utm[1]), TO_WGS.transform(bbox_utm[2], bbox_utm[3])]
        query_bbox = (min(p[0] for p in bbox_wgs_points), min(p[1] for p in bbox_wgs_points),
                      max(p[0] for p in bbox_wgs_points), max(p[1] for p in bbox_wgs_points))
        external = []
        with fiona.open(SHAPEFILE) as source:
            if source.crs.to_epsg() != 4326:
                raise ValueError("Local independent layer CRS changed")
            for feature in source.filter(bbox=query_bbox):
                geom = shape(feature["geometry"]) if feature.get("geometry") else None
                if geom is None or geom.is_empty or not geom.is_valid:
                    continue
                ground_candidate = transform(TO_UTM.transform, geom)
                if ground_candidate.intersects(box(*bbox_utm)):
                    external.append({"geometry": ground_candidate, "properties": dict(feature["properties"]), "fid": feature["id"]})
        epoch_geometry_path = Path(freeze["files"]["geometry_2018"]["path"])
        roof_candidates = []
        with fiona.open(epoch_geometry_path, layer="buildings") as source:
            for feature in source.filter(bbox=bbox_utm):
                geom = shape(feature["geometry"]) if feature.get("geometry") else None
                if geom is not None and not geom.is_empty and geom.is_valid:
                    roof_candidates.append({"geometry": geom, "properties": dict(feature["properties"]), "fid": feature["id"]})
        pairs = []
        for roof in roof_candidates:
            nearest_distance = None
            best_iou = None
            for index, external_item in enumerate(external):
                distance = roof["geometry"].distance(external_item["geometry"])
                nearest_distance = distance if nearest_distance is None else min(nearest_distance, distance)
                if distance > 20.0:
                    continue
                ground = external_item["geometry"]
                inter = roof["geometry"].intersection(ground).area
                union = roof["geometry"].union(ground).area
                iou = inter / union if union else 0.0
                candidate = {"distance_m": distance, "iou": iou, "roof_index": roof["fid"],
                             "external_index": external_item["fid"],
                             "roof_area_m2": roof["geometry"].area, "external_area_m2": ground.area,
                             "centroid_dx_m": ground.centroid.x - roof["geometry"].centroid.x,
                             "centroid_dy_m": ground.centroid.y - roof["geometry"].centroid.y,
                             "external_to_roof_area_ratio": ground.area / max(roof["geometry"].area, 1e-9),
                             "external_floor": external_item["properties"].get("Floor")}
                if best_iou is None or candidate["iou"] > best_iou["iou"] or (candidate["iou"] == best_iou["iou"] and distance < best_iou["distance_m"]):
                    best_iou = candidate
            pairs.append({"roof_index": roof["fid"], "nearest_distance_m": nearest_distance,
                          "best_local_iou": best_iou["iou"] if best_iou else 0.0,
                          "best_iou_distance_m": best_iou["distance_m"] if best_iou else None,
                          "best_external_index": best_iou["external_index"] if best_iou else None,
                          "centroid_dx_m": best_iou["centroid_dx_m"] if best_iou else None,
                          "centroid_dy_m": best_iou["centroid_dy_m"] if best_iou else None,
                          "external_to_roof_area_ratio": best_iou["external_to_roof_area_ratio"] if best_iou else None,
                          "external_floor": best_iou["external_floor"] if best_iou else None})
        image = Image.fromarray(mosaic)
        draw = ImageDraw.Draw(image)
        for item in roof_candidates:
            _draw_geometry(draw, _pixel_coords(item["geometry"], x0, y0 + 3, 3, 3), (255, 30, 30), 2)
        for item in external:
            _draw_geometry(draw, _pixel_coords(item["geometry"], x0, y0 + 3, 3, 3), (0, 255, 255), 2)
        image_path = output / f"{case_id}_2018_roof_vs_local_layer.png"
        image.save(image_path)
        target_geometries = []
        with fiona.open(FIXED_GEOM, layer="fixed_case_polygons") as fixed_layer:
            for feature in fixed_layer:
                properties = dict(feature["properties"])
                if properties["case_id"] == case_id and properties["epoch_label"] == "2018" and properties["role"] == "TARGET":
                    target_geometries.append(shape(feature["geometry"]))
        target_pixels = [_pixel_coords(geom, x0, y0 + 3, 3, 3) for geom in target_geometries]
        if target_pixels:
            bounds = unary_union(target_pixels).bounds
            pad = 90
            crop = (max(0, int(bounds[0] - pad)), max(0, int(bounds[1] - pad)),
                    min(image.width, int(bounds[2] + pad)), min(image.height, int(bounds[3] + pad)))
            zoom = image.crop(crop).resize(((crop[2] - crop[0]) * 3, (crop[3] - crop[1]) * 3), Image.Resampling.NEAREST)
            zoom_path = output / f"{case_id}_2018_target_zoom.png"
            zoom.save(zoom_path)
        else:
            zoom_path = None
        matched_dx = [row["centroid_dx_m"] for row in pairs if row["centroid_dx_m"] is not None]
        matched_dy = [row["centroid_dy_m"] for row in pairs if row["centroid_dy_m"] is not None]
        area_ratios = [row["external_to_roof_area_ratio"] for row in pairs if row["external_to_roof_area_ratio"] is not None]
        summary.append({
            "case_id": case_id, "actual_date": actual_date,
            "tile_window_coverage": "9_OF_9_DATE_VERIFIED",
            "local_footprint_layer_features": len(external),
            "r4_roof_candidate_features": len(roof_candidates),
            "nearest_geometry_distance_median": float(np.median([row["nearest_distance_m"] for row in pairs if row["nearest_distance_m"] is not None])) if any(row["nearest_distance_m"] is not None for row in pairs) else None,
            "best_local_iou_median": float(np.median([row["best_local_iou"] for row in pairs])) if pairs else None,
            "best_local_centroid_dx_median": float(np.median(matched_dx)) if matched_dx else None,
            "best_local_centroid_dy_median": float(np.median(matched_dy)) if matched_dy else None,
            "best_local_area_ratio_median": float(np.median(area_ratios)) if area_ratios else None,
            "pair_count_nearest_within_5m": sum(row["nearest_distance_m"] is not None and row["nearest_distance_m"] <= 5 for row in pairs),
            "pair_count_best_iou_over_0_5": sum(row["best_local_iou"] >= 0.5 for row in pairs),
            "floor_value_counts": dict(Counter(str(item["properties"].get("Floor")) for item in external)),
            "candidate_overlay_png": str(image_path), "target_zoom_png": str(zoom_path) if zoom_path else None,
            "matches": pairs,
            "reference_geometry_sha256": freeze["files"]["geometry_2018"]["sha256"],
            "independent_source_crs": "EPSG:4326",
            "independent_source_capture_date": "UNKNOWN",
            "independent_source_accuracy": "UNKNOWN",
        })
    result = {"schema_version": "xian_r5.local_building_geometry_comparison.v1",
              "source_layer": str(SHAPEFILE),
              "source_semantics": "POLYGON_WITH_Id_AND_Floor_FIELDS; PROVENANCE_UNAVAILABLE",
              "source_is_ground_truth": False,
              "color_legend": {"red": "R4_roof_candidate", "cyan": "local_building_layer_candidate"},
              "cases": summary}
    (output / "LOCAL_GEOMETRY_COMPARISON.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run(["B03-TEX04", "B01-TEX02", "B05-TEX01", "B04-TQA10"])
    print(json.dumps([{"case": item["case_id"], "external_n": item["local_footprint_layer_features"],
                       "r4_n": item["r4_roof_candidate_features"], "median_iou": item["best_local_iou_median"],
                       "under5m": item["pair_count_nearest_within_5m"], "iou05": item["pair_count_best_iou_over_0_5"]}
                      for item in report["cases"]], ensure_ascii=False))
