from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.ops import transform as shp_transform


SCENES = ("xa180429", "xa190101", "xa201231")
LABELS = {"xa180429": "2018", "xa190101": "2019", "xa201231": "2020"}
RAW_NAMES = {s: f"{s}_zoom_20.tif" for s in SCENES}


def txt(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    audit_path = Path(args.audit)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    formal = root / "output/西安_R4_辅助影像接入前_三期三态候选版_20260914"
    v2 = root / "output/16_candidate_identitygraph_v2_20260916"
    gpkg = formal / "04_identity_graph/tracking_buildings.gpkg"
    raw_paths = {s: root / "input/base" / RAW_NAMES[s] for s in SCENES}

    audit = pd.read_csv(audit_path, low_memory=False)
    audit = audit[audit["promotion_eligible_zero_candidate"].astype(str).str.lower().eq("true")].copy()
    audit["area_m2"] = pd.to_numeric(audit["area_m2"], errors="coerce").fillna(0.0)
    chosen = []
    for interval in sorted(audit["interval"].unique()):
        part = audit[audit["interval"].eq(interval)].sort_values("area_m2", ascending=False).head(100).copy()
        part["sample_id"] = [f"OBSZERO_{interval}_{i:03d}" for i in range(1, len(part) + 1)]
        part["category"] = "FRESH_PIXEL_CLEAR_ZERO_CANDIDATE"
        chosen.append(part)
    samples = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()
    samples.to_csv(out / "OBSERVABILITY_ZERO_CANDIDATE_SAMPLE_INDEX.csv", index=False, encoding="utf-8-sig")
    if samples.empty:
        raise RuntimeError("no candidate rows available")

    tr = pd.read_csv(v2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv", usecols=["physical_track_id", "object_uid"], low_memory=False)
    tr["physical_track_id"] = tr["physical_track_id"].astype(str)
    tr["object_uid"] = tr["object_uid"].astype(str)
    by_pid = tr.groupby("physical_track_id")["object_uid"].apply(list).to_dict()
    needed = sorted({u for pid in samples["physical_track_id"].astype(str) for u in by_pid.get(pid, [])})
    geom = {}
    for start in range(0, len(needed), 400):
        chunk = needed[start : start + 400]
        vals = ",".join("'" + u.replace("'", "''") + "'" for u in chunk)
        g = pyogrio.read_dataframe(str(gpkg), layer="tracking_buildings", where=f"object_uid IN ({vals})", columns=["object_uid", "scene_key", "area_m2", "building_id"])
        for row in g.itertuples():
            geom[str(row.object_uid)] = row

    to_geo = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True).transform
    from_geo = {s: Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True).transform for s in SCENES}
    font = ImageFont.load_default()
    manifest = []

    def read_raw(scene: str, rows: list, fallback_xy: tuple[float, float]) -> tuple[Image.Image, rasterio.Affine]:
        shapes = [r.geometry for r in rows if getattr(r, "geometry", None) is not None and not r.geometry.is_empty]
        if shapes:
            minx = min(g.bounds[0] for g in shapes); miny = min(g.bounds[1] for g in shapes)
            maxx = max(g.bounds[2] for g in shapes); maxy = max(g.bounds[3] for g in shapes)
            bounds = transform_bounds("EPSG:32649", "EPSG:4326", minx, miny, maxx, maxy)
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            with rasterio.open(raw_paths[scene]) as src:
                px = abs(src.transform.a); py = abs(src.transform.e)
                span = max((bounds[2] - bounds[0]) / px, (bounds[3] - bounds[1]) / py)
                size = int(max(320, min(768, span + 160)))
                col = int((cx - src.transform.c) / src.transform.a); row = int((src.transform.f - cy) / abs(src.transform.e))
                win = Window(col - size // 2, row - size // 2, size, size)
                arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0, out_shape=(3, size, size))
                return Image.fromarray(np.moveaxis(arr, 0, 2).astype(np.uint8), "RGB"), src.window_transform(win)
        x, y = fallback_xy
        lon, lat = to_geo(x, y)
        with rasterio.open(raw_paths[scene]) as src:
            size = 512
            col = int((lon - src.transform.c) / src.transform.a); row = int((src.transform.f - lat) / abs(src.transform.e))
            win = Window(col - size // 2, row - size // 2, size, size)
            arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0, out_shape=(3, size, size))
            return Image.fromarray(np.moveaxis(arr, 0, 2).astype(np.uint8), "RGB"), src.window_transform(win)

    def project_xy(x: float, y: float, transform):
        lon, lat = to_geo(x, y)
        return ((lon - transform.c) / transform.a, (transform.f - lat) / abs(transform.e))

    def draw_polygon(img, row, transform, color):
        def f(x, y, z=None):
            px, py = project_xy(x, y, transform)
            return px, py
        g = shp_transform(f, row.geometry)
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        dr = ImageDraw.Draw(img)
        for poly in polys:
            xy = list(poly.exterior.coords)
            dr.line(xy, fill=color, width=3, joint="curve")
            if xy:
                dr.text((xy[0][0] + 3, xy[0][1] + 3), txt(row.object_uid), fill=color, font=font)

    for sample in samples.to_dict("records"):
        pid = txt(sample["physical_track_id"])
        rows_by_scene = {s: [geom[u] for u in by_pid.get(pid, []) if u in geom and txt(geom[u].scene_key) == s] for s in SCENES}
        # Query every canonical B0 object in a bounded 300 m window as well as the
        # track's own objects. A missing track object must not be mistaken for an
        # empty scene when another canonical object is present nearby.
        all_rows_by_scene = {}
        for scene in SCENES:
            try:
                all_g = pyogrio.read_dataframe(
                    str(gpkg),
                    layer="tracking_buildings",
                    bbox=(float(sample["x"]) - 150.0, float(sample["y"]) - 150.0, float(sample["x"]) + 150.0, float(sample["y"]) + 150.0),
                    columns=["object_uid", "scene_key", "area_m2", "building_id"],
                )
                all_rows_by_scene[scene] = [r for r in all_g.itertuples() if txt(r.scene_key) == scene]
            except Exception:
                all_rows_by_scene[scene] = []
        folder = out / "samples" / txt(sample["sample_id"])
        folder.mkdir(parents=True, exist_ok=True)
        raws = {}; overs = {}
        fallback = (float(sample["x"]), float(sample["y"]))
        missing_scenes = []
        all_object_counts = {}
        for scene in SCENES:
            all_rows = all_rows_by_scene[scene]
            all_object_counts[scene] = len(all_rows)
            raw, transform = read_raw(scene, all_rows or rows_by_scene[scene], fallback)
            over = raw.copy()
            track_uids = set(by_pid.get(pid, []))
            for row in all_rows:
                active = txt(row.object_uid) in track_uids
                active_color = (255, 60, 60) if scene == "xa180429" else ((60, 255, 60) if scene == "xa190101" else (80, 160, 255))
                draw_polygon(over, row, transform, active_color if active else (255, 220, 40))
            if not rows_by_scene[scene]:
                missing_scenes.append(scene)
                px, py = project_xy(fallback[0], fallback[1], transform)
                dr = ImageDraw.Draw(over)
                dr.line((px - 14, py, px + 14, py), fill=(255, 215, 0), width=4)
                dr.line((px, py - 14, px, py + 14), fill=(255, 215, 0), width=4)
                label = "TRACK_NO_MATCH / OTHER_OBJECTS_PRESENT" if all_rows else "NO_B0_OBJECT_IN_BBOX"
                dr.text((px + 6, py + 6), label, fill=(255, 215, 0), font=font)
            dr = ImageDraw.Draw(over)
            dr.rectangle((0, 0, over.width, 34), fill=(0, 0, 0))
            dr.text((8, 8), f"{LABELS[scene]} | {pid} | ZERO CANDIDATE | {sample['presence_sequence']}", fill="white", font=font)
            raw.save(folder / f"raw_{LABELS[scene]}.png", compress_level=6)
            over.save(folder / f"overlay_{LABELS[scene]}.png", compress_level=6)
            raws[scene] = raw; overs[scene] = over
        tile = 320; margin = 12; header = 44
        canvas = Image.new("RGB", (tile * 3 + margin * 4, header + tile * 2 + margin * 3), "white")
        dr = ImageDraw.Draw(canvas)
        dr.text((margin, 12), f"{sample['sample_id']} | {sample['interval']} | {pid} | area={float(sample['area_m2']):.1f} m2", fill="black", font=font)
        for j, scene in enumerate(SCENES):
            x = margin + j * (tile + margin)
            canvas.paste(raws[scene].resize((tile, tile)), (x, header + margin))
            canvas.paste(overs[scene].resize((tile, tile)), (x, header + margin * 2 + tile))
            dr.text((x + 5, header + margin + 5), f"raw {LABELS[scene]}", fill="yellow", font=font)
            dr.text((x + 5, header + margin * 2 + tile + 5), f"overlay {LABELS[scene]}", fill="yellow", font=font)
        canvas.save(folder / "contact_sheet.png", compress_level=6)
        summary = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in sample.items()}
        summary.update({"physical_track_id": pid, "missing_scenes": missing_scenes, "all_canonical_object_counts": all_object_counts, "visual_status": "PENDING_TARGETED_REVIEW", "formal_outputs_modified": False})
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        manifest.append({"sample_id": txt(sample["sample_id"]), "interval": txt(sample["interval"]), "physical_track_id": pid, "area_m2": float(sample["area_m2"]), "folder": str(folder.relative_to(out)), "missing_scenes": ";".join(missing_scenes), "asset_status": "GENERATED"})

    pd.DataFrame(manifest).to_csv(out / "PHASE2_VISUAL_ASSET_MANIFEST.csv", index=False, encoding="utf-8-sig")
    receipt = {"sample_count": len(manifest), "geometry_requested": len(needed), "geometry_found": len(geom), "geometry_missing": len(needed) - len(geom), "formal_outputs_modified": False, "review_status": "PENDING_TARGETED_REVIEW"}
    (out / "PHASE2_VISUAL_ASSET_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text("# Phase 2 fresh-observability visual QA\n\nThese are isolated zero-presence candidates selected only after fresh main-image pixel quality and date/identity audit. A yellow cross marks the missing canonical object position; it is not a polygon replacement. No formal result is modified.\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
