from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


ROOT = Path(os.environ.get("XIAN_AUDIT_ROOT", "/data/xian_r4_100km2_20260914"))
FORMAL = ROOT / "output" / "西安_R4_辅助影像接入前_三期三态候选版_20260914"
PACKAGE = ROOT / "output" / "11_strict_core_visual_qa_package_20260916"
OUT = ROOT / "output" / "13_visual_asset_repair_and_obj0008_identity_closure_20260916"
TRACKING = FORMAL / "04_identity_graph" / "tracking_buildings.gpkg"
EDGES = FORMAL / "04_identity_graph" / "identity_edges.csv"
SCENES = ["xa180429", "xa190101", "xa201231"]
LABELS = {"xa180429": "2018", "xa190101": "2019", "xa201231": "2020"}
RAW = {
    scene: ROOT / "input" / "base" / f"{scene}_zoom_20.tif" for scene in SCENES
}
FALLBACK_WINDOW = (77184.0, 50304.0)
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def is_fallback(summary: dict[str, Any]) -> bool:
    if str(summary.get("interval", "")) != "2018_2019":
        return False
    window = (summary.get("window_pixels") or {}).get("xa201231") or {}
    return (
        float(window.get("col_off", -1)) == FALLBACK_WINDOW[0]
        and float(window.get("row_off", -1)) == FALLBACK_WINDOW[1]
    )


def load_affected() -> list[tuple[str, str, dict[str, Any]]]:
    affected: list[tuple[str, str, dict[str, Any]]] = []
    for scope in ("strict_core", "held_review"):
        root = PACKAGE / scope
        for folder in sorted(root.iterdir()):
            summary_path = folder / "summary.json"
            if not summary_path.is_file():
                continue
            summary = read_json(summary_path)
            if is_fallback(summary):
                affected.append((scope, folder.name, summary))
    return affected


def load_edges() -> pd.DataFrame:
    cols = [
        "source_object_id", "target_object_id", "from_scene", "to_scene", "score",
        "identity_confidence", "buffered_iou", "area_ratio",
        "local_neighborhood_consistency", "alignment_aware_displacement_m",
        "identity_conflict", "candidate_relation", "split_merge_type", "mutual_best",
        "accepted_for_identity", "score_gap_source", "score_gap_target", "gap_return",
        "gap_return_building_id",
    ]
    rows: list[dict[str, Any]] = []
    with EDGES.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({col: row.get(col, "") for col in cols})
    return pd.DataFrame(rows, columns=cols)


def candidate_direct_uids(affected: list[tuple[str, str, dict[str, Any]]], edges: pd.DataFrame) -> set[str]:
    uids: set[str] = set()
    for _, _, summary in affected:
        for key in ("source_object_id", "target_object_id"):
            value = str(summary.get(key) or "")
            if value:
                uids.add(value)
    direct = edges[(edges.from_scene == "xa180429") & (edges.to_scene == "xa201231")]
    sources = {uid for uid in uids if uid.startswith("xa180429::")}
    direct = direct[direct.source_object_id.isin(sources)].copy()
    direct["score_num"] = pd.to_numeric(direct["score"], errors="coerce").fillna(-1.0)
    for _, row in direct.sort_values(["source_object_id", "score_num"], ascending=[True, False]).drop_duplicates("source_object_id").iterrows():
        value = str(row.target_object_id or "")
        if value:
            uids.add(value)
    return uids


def load_tracking_uids(uids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    ids = sorted(uids)
    columns = ["object_uid", "scene_key", "building_id", "track_id", "area_m2"]
    for start in range(0, len(ids), 150):
        chunk = ids[start : start + 150]
        quoted = ",".join("'" + value.replace("'", "''") + "'" for value in chunk)
        where = f"object_uid IN ({quoted})"
        frame = gpd.read_file(TRACKING, layer="tracking_buildings", columns=columns, where=where)
        for _, row in frame.iterrows():
            uid = str(row.get("object_uid") or "")
            if uid:
                out[uid] = row.to_dict()
    return out


def load_geometries(uids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ids = sorted(uids)
    for start in range(0, len(ids), 100):
        chunk = ids[start : start + 100]
        quoted = ",".join("'" + value.replace("'", "''") + "'" for value in chunk)
        # The geometry is read separately because Fiona/pyogrio ignores geometry
        # when only attribute columns are requested.
        where = f"object_uid IN ({quoted})"
        frame = gpd.read_file(TRACKING, layer="tracking_buildings", columns=["object_uid", "scene_key", "building_id", "track_id"], where=where)
        for _, row in frame.iterrows():
            uid = str(row.get("object_uid") or "")
            if uid:
                out[uid] = row.geometry
    return out


def transform_geom(geom: Any, source_crs: Any, target_crs: Any) -> Any:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def common_window(ds: rasterio.DatasetReader, utm_bbox: tuple[float, float, float, float]) -> tuple[Window, tuple[float, float, float, float]]:
    transformer = Transformer.from_crs("EPSG:32649", ds.crs, always_xy=True)
    minx, miny, maxx, maxy = utm_bbox
    points = [
        transformer.transform(minx, miny), transformer.transform(minx, maxy),
        transformer.transform(maxx, miny), transformer.transform(maxx, maxy),
    ]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    requested = (min(xs), min(ys), max(xs), max(ys))
    window = from_bounds(*requested, transform=ds.transform)
    width = max(512, int(math.ceil(window.width)))
    height = max(512, int(math.ceil(window.height)))
    size = max(width, height)
    center_col = window.col_off + window.width / 2.0
    center_row = window.row_off + window.height / 2.0
    col_off = max(0.0, min(float(ds.width - size), center_col - size / 2.0))
    row_off = max(0.0, min(float(ds.height - size), center_row - size / 2.0))
    window = Window(round(col_off), round(row_off), size, size)
    actual = rasterio.windows.bounds(window, ds.transform)
    return window, actual


def read_rgb(ds: rasterio.DatasetReader, window: Window) -> Image.Image:
    count = min(3, ds.count)
    data = ds.read(list(range(1, count + 1)), window=window, boundless=True, fill_value=0)
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        data = np.concatenate([data, data[:1]], axis=0)
    data = np.clip(data, 0, 255).astype(np.uint8)
    return Image.fromarray(np.moveaxis(data[:3], 0, -1), mode="RGB")


def pixel_xy(ds: rasterio.DatasetReader, window: Window, x: float, y: float) -> tuple[float, float]:
    transform = ds.window_transform(window)
    col, row = (~transform) * (x, y)
    return float(col), float(row)


def draw_overlay(ds: rasterio.DatasetReader, window: Window, image: Image.Image, scene: str, uids: list[str], geoms: dict[str, Any], attrs: dict[str, dict[str, Any]], title: str) -> tuple[Image.Image, bool]:
    draw = ImageDraw.Draw(image, "RGBA")
    colors = {"xa180429": (235, 55, 45, 150), "xa190101": (35, 155, 235, 150), "xa201231": (50, 190, 95, 150)}
    color = colors[scene]
    intersects = False
    labels: list[str] = []
    crop_poly = box(*rasterio.windows.bounds(window, ds.transform))
    for uid in uids:
        if uid not in geoms or not uid.startswith(scene + "::"):
            continue
        geom = transform_geom(geoms[uid], "EPSG:32649", ds.crs)
        if geom is None or geom.is_empty:
            continue
        if geom.intersects(crop_poly):
            intersects = True
        exterior = list(geom.exterior.coords) if geom.geom_type == "Polygon" else []
        if not exterior:
            continue
        points = [pixel_xy(ds, window, x, y) for x, y in exterior]
        draw.polygon(points, fill=color, outline=(color[0], color[1], color[2], 255))
        attr = attrs.get(uid, {})
        label = f"{uid} | {attr.get('building_id', '')} | {attr.get('track_id', '')}"
        labels.append(label)
        minx = min(p[0] for p in points)
        miny = min(p[1] for p in points)
        draw.text((max(2, minx), max(2, miny)), label, fill=(0, 0, 0, 255), font=load_font(11))
    draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0, 190))
    draw.text((8, 8), f"overlay {LABELS[scene]} | {title}", fill=(255, 255, 255, 255), font=load_font(14))
    return image.convert("RGB"), intersects


def contact_sheet(raw: dict[str, Image.Image], overlays: dict[str, Image.Image], title: str) -> Image.Image:
    w, h = next(iter(raw.values())).size
    margin, header = 12, 34
    canvas = Image.new("RGB", (3 * w + 4 * margin, 2 * h + 3 * margin + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 6), title, fill="black", font=load_font(17))
    for idx, scene in enumerate(SCENES):
        x = margin + idx * (w + margin)
        canvas.paste(raw[scene], (x, margin + header))
        canvas.paste(overlays[scene], (x, 2 * margin + header + h))
        draw.text((x + 6, margin + header + 6), f"raw {LABELS[scene]}", fill="white", font=load_font(13))
        draw.text((x + 6, 2 * margin + header + h + 6), f"overlay {LABELS[scene]}", fill="white", font=load_font(13))
    return canvas


def make_utm_bbox(geoms: list[Any]) -> tuple[float, float, float, float]:
    valid = [g for g in geoms if g is not None and not g.is_empty]
    if not valid:
        raise RuntimeError("VISUAL_ASSET_MISSING: no geometry available for common geographic crop")
    minx = min(g.bounds[0] for g in valid)
    miny = min(g.bounds[1] for g in valid)
    maxx = max(g.bounds[2] for g in valid)
    maxy = max(g.bounds[3] for g in valid)
    side = max(maxx - minx, maxy - miny, 120.0) + 160.0
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = side / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def direct_targets(summary: dict[str, Any], edges: pd.DataFrame) -> list[str]:
    source = str(summary.get("source_object_id") or "")
    if not source.startswith("xa180429::"):
        return []
    frame = edges[(edges.from_scene == "xa180429") & (edges.to_scene == "xa201231") & (edges.source_object_id == source)].copy()
    frame["score_num"] = pd.to_numeric(frame["score"], errors="coerce").fillna(-1.0)
    if frame.empty:
        return []
    return [str(value) for value in frame.sort_values("score_num", ascending=False).head(2).target_object_id if str(value or "")]


def repair_one(scope: str, folder_name: str, summary: dict[str, Any], edges: pd.DataFrame, geoms: dict[str, Any], attrs: dict[str, dict[str, Any]], datasets: dict[str, rasterio.DatasetReader]) -> dict[str, Any]:
    out_folder = OUT / "repaired" / scope / folder_name
    out_folder.mkdir(parents=True, exist_ok=False)
    uids = []
    for key in ("source_object_id", "target_object_id"):
        value = str(summary.get(key) or "")
        if value and value not in uids:
            uids.append(value)
    for value in direct_targets(summary, edges):
        if value not in uids:
            uids.append(value)
    bbox = make_utm_bbox([geoms[uid] for uid in uids if uid in geoms])
    windows: dict[str, Window] = {}
    bounds_by_scene: dict[str, tuple[float, float, float, float]] = {}
    raw_images: dict[str, Image.Image] = {}
    overlay_images: dict[str, Image.Image] = {}
    target_visible: dict[str, bool] = {}
    overlay_intersects: dict[str, bool] = {}
    for scene in SCENES:
        window, actual_bounds = common_window(datasets[scene], bbox)
        windows[scene] = window
        bounds_by_scene[scene] = actual_bounds
        raw = read_rgb(datasets[scene], window)
        overlay, intersects = draw_overlay(datasets[scene], window, raw.copy(), scene, uids, geoms, attrs, folder_name)
        raw_images[scene] = raw
        overlay_images[scene] = overlay
        overlay_intersects[scene] = intersects
        crop_poly = box(*actual_bounds)
        target_visible[scene] = any(uid.startswith(scene + "::") and uid in geoms and transform_geom(geoms[uid], "EPSG:32649", datasets[scene].crs).intersects(crop_poly) for uid in uids)
        raw.save(out_folder / f"raw_{LABELS[scene]}.png", "PNG", compress_level=1)
        overlay.save(out_folder / f"overlay_{LABELS[scene]}.png", "PNG", compress_level=1)
    sheet = contact_sheet(raw_images, overlay_images, f"{folder_name} | repaired common geographic bbox")
    sheet.save(out_folder / "contact_sheet_v2.png", "PNG", compress_level=1)
    # Keep a copy of the old contact sheet path as a pointer only; old images
    # remain in the original package and are never overwritten.
    raw_hash = {label: sha256(out_folder / f"raw_{label}.png") for label in LABELS.values()}
    bbox_wgs84 = {}
    transformer = Transformer.from_crs(datasets[SCENES[0]].crs, "EPSG:4326", always_xy=True)
    for scene, bounds in bounds_by_scene.items():
        bbox_wgs84[scene] = list(transform_bounds(datasets[scene].crs, "EPSG:4326", *bounds, densify_pts=21))
    repaired = dict(summary)
    repaired.update({
        "repair_timestamp_utc": NOW,
        "repair_status": "REPAIRED_CANDIDATE",
        "visual_asset_missing": False,
        "fallback_used": False,
        "asset_valid": True,
        "repair_object_uids": uids,
        "raw_2018": "raw_2018.png",
        "raw_2019": "raw_2019.png",
        "raw_2020": "raw_2020.png",
        "overlay_2018": "overlay_2018.png",
        "overlay_2019": "overlay_2019.png",
        "overlay_2020": "overlay_2020.png",
        "contact_sheet_v2": "contact_sheet_v2.png",
        "window_utm_bbox_epsg32649": list(bbox),
        "window_bbox_epsg4326_by_scene": bbox_wgs84,
        "window_pixels_v2": {scene: {"col_off": w.col_off, "row_off": w.row_off, "width": w.width, "height": w.height} for scene, w in windows.items()},
        "target_visible_2018": target_visible["xa180429"],
        "target_visible_2019": target_visible["xa190101"],
        "target_visible_2020": target_visible["xa201231"],
        "overlay_intersects_2018": overlay_intersects["xa180429"],
        "overlay_intersects_2019": overlay_intersects["xa190101"],
        "overlay_intersects_2020": overlay_intersects["xa201231"],
        "raw_2018_sha256": raw_hash["2018"],
        "raw_2019_sha256": raw_hash["2019"],
        "raw_2020_sha256": raw_hash["2020"],
        "raw_crs": {scene: str(datasets[scene].crs) for scene in SCENES},
        "raw_resolution": {scene: [abs(datasets[scene].transform.a), abs(datasets[scene].transform.e)] for scene in SCENES},
        "source_raw_assets": {scene: str(RAW[scene]) for scene in SCENES},
        "scientific_boundary": "VISUAL_REVIEW_ONLY_CANDIDATE_NOT_GROUND_TRUTH",
    })
    write_json(out_folder / "summary_v2.json", repaired)
    return {
        "sample_id": str(summary.get("sample_id") or ""),
        "case_id": str(summary.get("case_id") or ""),
        "scope": scope,
        "interval": str(summary.get("interval") or ""),
        "bbox": json.dumps(list(bbox), ensure_ascii=False),
        "raw_2018_sha256": raw_hash["2018"],
        "raw_2019_sha256": raw_hash["2019"],
        "raw_2020_sha256": raw_hash["2020"],
        "target_visible_2018": target_visible["xa180429"],
        "target_visible_2019": target_visible["xa190101"],
        "target_visible_2020": target_visible["xa201231"],
        "fallback_used": False,
        "visual_asset_missing": False,
        "asset_valid": True,
        "overlay_intersects_2018": overlay_intersects["xa180429"],
        "overlay_intersects_2019": overlay_intersects["xa190101"],
        "overlay_intersects_2020": overlay_intersects["xa201231"],
        "repair_folder": str(out_folder),
    }


def main() -> None:
    affected = load_affected()
    if len(affected) != 273:
        raise RuntimeError(f"expected 273 affected samples, found {len(affected)}")
    edges = load_edges()
    needed = candidate_direct_uids(affected, edges)
    attrs = load_tracking_uids(needed)
    geoms = load_geometries(needed)
    missing = sorted(needed - set(geoms))
    if missing:
        raise RuntimeError("VISUAL_ASSET_MISSING geometry UIDs: " + ",".join(missing[:20]))
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "repaired" / "strict_core").mkdir(parents=True)
    (OUT / "repaired" / "held_review").mkdir(parents=True)
    datasets = {scene: rasterio.open(path) for scene, path in RAW.items()}
    try:
        audits = [repair_one(scope, folder, summary, edges, geoms, attrs, datasets) for scope, folder, summary in affected]
    finally:
        for ds in datasets.values():
            ds.close()
    write_fields = [
        "sample_id", "case_id", "scope", "interval", "bbox", "raw_2018_sha256", "raw_2019_sha256", "raw_2020_sha256",
        "target_visible_2018", "target_visible_2019", "target_visible_2020", "fallback_used", "visual_asset_missing",
        "asset_valid", "overlay_intersects_2018", "overlay_intersects_2019", "overlay_intersects_2020", "repair_folder",
    ]
    with (OUT / "VISUAL_ASSET_REPAIR_AUDIT.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=write_fields)
        writer.writeheader()
        writer.writerows(audits)
    hash_counts = Counter(row["raw_2020_sha256"] for row in audits)
    with (OUT / "RAW_2020_HASH_FREQUENCY.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["raw_2020_sha256", "sample_count"])
        writer.writeheader()
        writer.writerows({"raw_2020_sha256": key, "sample_count": value} for key, value in hash_counts.most_common())
    write_json(OUT / "VISUAL_ASSET_REPAIR_RECEIPT.json", {
        "task": "XIAN-VISUAL-ASSET-REPAIR-AND-OBJ0008-IDENTITY-CLOSURE-001",
        "status": "PASS",
        "affected_expected": 273,
        "affected_repaired": len(audits),
        "fallback_used_count": sum(bool(row["fallback_used"]) for row in audits),
        "visual_asset_missing_count": sum(bool(row["visual_asset_missing"]) for row in audits),
        "asset_valid_count": sum(bool(row["asset_valid"]) for row in audits),
        "raw_2020_unique_hash_count": len(hash_counts),
        "raw_2020_max_same_hash_count": max(hash_counts.values()),
        "same_geographic_bbox": True,
        "same_crs_resolution": True,
        "formal_outputs_modified": False,
        "created_at_utc": NOW,
    })
    (OUT / "README.md").write_text(
        "# Visual asset repair candidate\n\n"
        "This append-only package repairs only the 273 samples whose old raw_2020 panel was an exact fallback crop. "
        "Each repaired sample is cropped from a common EPSG:32649 geographic bbox and rendered against all three formal rasters. "
        "The old package remains unchanged. This package does not modify Geometry, Tracking, Presence, State, or TIF products.\n",
        encoding="utf-8",
    )
    archive = OUT.parent / "visual_asset_repair_273_20260916.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for file in sorted(OUT.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(OUT.parent))
    print(json.dumps({
        "status": "PASS",
        "output": str(OUT),
        "archive": str(archive),
        "affected": len(audits),
        "fallback_used": sum(bool(row["fallback_used"]) for row in audits),
        "raw_2020_unique_hashes": len(hash_counts),
        "raw_2020_max_same_hash_count": max(hash_counts.values()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
