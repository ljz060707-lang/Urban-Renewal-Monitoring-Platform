"""Render tracking-case contact sheets from the real imagery and real geometry.

Layout per case (the layout requested for the delivery):

    Top:    2018 Raw | 2019 Raw | 2020 Raw
    Bottom: 2018 Overlay | 2019 Overlay | 2020 Overlay
    Right:  physical_track_id, presence sequence, state, area, reason

CRS contract
------------
GPKG geometry  : EPSG:32649 (metres)
Scene imagery  : EPSG:4326 (degrees)
Therefore: crop framing and area are computed in metres, the geometry drawn into
a crop is converted to EPSG:4326, and the crop window is derived through the
scene transform.  Getting this order wrong silently draws nothing.

Also the visual-grading harness: at a small panel size it renders many cases per
sheet so they can be graded before any recommendation is made.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "work" / "python-packages"))
spec = importlib.util.spec_from_file_location("xian_raster_core", TOOLS / "xian_rasterize_current_best.py")
core = importlib.util.module_from_spec(spec)
sys.modules["xian_raster_core"] = core
assert spec.loader is not None
spec.loader.exec_module(core)

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from shapely.ops import transform as shp_tf  # noqa: E402

import xian_figure_tools as ft  # noqa: E402

CASE_ROOT = core.DELIVERY / "03_tracking_cases"
SCENE_OF = {"2018": "xa180429", "2019": "xa190101", "2020": "xa201231"}
PX_M = 0.247

_TO_WGS = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
_TO_METRIC = Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)


def load_geometries(year: str) -> dict:
    """object_id -> (geom_32649, geom_4326, area_m2, centroid_x, centroid_y)."""
    gpkg = core.geom_path(core.SCENES[year][2])
    df = pyogrio.read_dataframe(gpkg, columns=["object_id", "area_m2", "geometry"], use_arrow=False)
    out = {}
    for oid, area, geom in zip(df["object_id"], df["area_m2"], df.geometry):
        if geom is None or geom.is_empty:
            continue
        try:
            key = int(oid)
        except Exception:
            continue
        wgs = shp_tf(_TO_WGS.transform, geom)
        c = wgs.centroid
        out[key] = (geom, wgs, float(area or 0.0), float(c.x), float(c.y))
    return out


def load_track_objects() -> dict:
    path = core.LATEST / "FINAL_CANDIDATE_TRACKING_OBJECTS.csv"
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            physical = row.get("physical_track_id", "")
            scene = row.get("scene_key", "")
            if not physical or not scene:
                continue
            try:
                oid = int(row.get("object_uid", "").split("::")[-1])
            except Exception:
                continue
            out[physical][scene].append(oid)
    return out


def metric_centroid_of_case(geoms_by_year, track_objects, physical):
    """Area-weighted metric centroid of the case track, plus the best epoch."""
    best_year, best_area, sx, sy, n = None, 0.0, 0.0, 0.0, 0
    for year, scene in SCENE_OF.items():
        for oid in track_objects.get(physical, {}).get(scene, []):
            rec = geoms_by_year[year].get(oid)
            if not rec:
                continue
            geom_metric, _wgs, area, _lon, _lat = rec
            if area <= 0:
                continue
            c = geom_metric.centroid
            sx += c.x
            sy += c.y
            n += 1
            if area > best_area:
                best_area, best_year = area, year
    if n == 0:
        return None
    return sx / n, sy / n, best_year, best_area


def window_for_centroid(reader, scene, cx, cy, size):
    lon, lat = _TO_WGS.transform(cx, cy)
    src = reader.src(scene)
    col, row = (~src.transform) * (lon, lat)
    half = size // 2
    col0 = max(0, min(int(round(col)) - half, src.width - size))
    row0 = max(0, min(int(round(row)) - half, src.height - size))
    return Window(col0, row0, size, size)


def render_case_sheet(case, geoms_by_year, track_objects, reader, out_path, panel=560, info_w=520):
    physical = case["physical_track_id"]
    cen = metric_centroid_of_case(geoms_by_year, track_objects, physical)
    if cen is None:
        return False
    cx, cy, _best_year, _best_area = cen

    raw_imgs, ov_imgs = {}, {}
    for year, scene in SCENE_OF.items():
        win = window_for_centroid(reader, scene, cx, cy, panel)
        arr = reader.src(scene).read(window=win)
        rgb = ft.to_rgb(arr)
        raw = Image.fromarray(rgb)
        ft.panel_label(raw, ft.SCENE_LABEL[scene])
        ft.add_scale_bar(raw, PX_M, 50)
        raw_imgs[year] = raw

        base = Image.fromarray(rgb).convert("RGBA")
        d = ImageDraw.Draw(base)
        src_transform = reader.src(scene).transform
        case_oids = set(track_objects.get(physical, {}).get(scene, []))
        # neighbours in white so any neighbour bleed is visible, case in green
        for oid in case_oids:
            rec = geoms_by_year[year].get(oid)
            if rec:
                ft.draw_geometry_outline(d, rec[1], win, src_transform, colour=(255, 255, 255), width=2)
        for oid in case_oids:
            rec = geoms_by_year[year].get(oid)
            if rec:
                ft.draw_geometry_outline(d, rec[1], win, src_transform, colour=(0, 255, 120), width=3)
        ov = base.convert("RGB")
        ft.panel_label(ov, ft.SCENE_LABEL[scene])
        ft.add_scale_bar(ov, PX_M, 50)
        ov_imgs[year] = ov

    gap = 8
    grid_w = panel * 3 + gap * 2
    grid_h = panel * 2 + gap
    sheet = Image.new("RGB", (grid_w + gap + info_w, grid_h + 64), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text(
        (12, 8),
        f"Case {case.get('case_id','')}  |  {str(case.get('case_category','')).upper()}",
        font=ft.font(22, True),
        fill=(0, 0, 0),
    )
    d.text(
        (12, 38),
        "Top: raw imagery    Bottom: candidate geometry overlay    Green = case track, white = other tracks",
        font=ft.font(14),
        fill=(60, 60, 60),
    )
    for i, year in enumerate(("2018", "2019", "2020")):
        sheet.paste(raw_imgs[year], (i * (panel + gap), 64))
        sheet.paste(ov_imgs[year], (i * (panel + gap), 64 + panel + gap))

    info = Image.new("RGB", (info_w, grid_h), (250, 250, 250))
    di = ImageDraw.Draw(info)
    y = 8
    blocks = [
        ("physical_track_id", physical, True),
        ("presence 2018-2019-2020", case.get("full_presence", "-"), False),
        ("state 2018-2019", (case.get("state_2018_2019") or "-"), False),
        ("state 2019-2020", (case.get("state_2019_2020") or "-"), False),
        ("max area (m2)", f"{float(case.get('area_max_m2') or 0):,.0f}", False),
        ("pixel size", f"{PX_M:.3f} m  ({1/PX_M:.1f} px/m)", False),
    ]
    for label, value, bold in blocks:
        di.text((10, y), label, font=ft.font(15, True), fill=(0, 0, 0))
        y += 19
        di.text((10, y), str(value), font=ft.font(17), fill=(0, 90, 160))
        y += 24
    di.text((10, y), "reason", font=ft.font(15, True), fill=(0, 0, 0))
    y += 19
    reason = (case.get("reason_2018_2019", "") or "")[:120]
    for chunk in [reason[i : i + 34] for i in range(0, len(reason), 34)][:4]:
        di.text((10, y), chunk, font=ft.font(14), fill=(40, 40, 40))
        y += 17
    for extra in ("gap_return", "trigger_interval", "unknown_intervals"):
        val = case.get(extra)
        if val:
            di.text((10, y), f"{extra}: {val}", font=ft.font(14), fill=(40, 40, 40))
            y += 17
    di.text((10, grid_h - 30), "CANDIDATE_NOT_GROUND_TRUTH", font=ft.font(14, True), fill=(170, 30, 30))
    sheet.paste(info, (grid_w + gap, 64))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(CASE_ROOT / "TRACKING_CASE_CANDIDATE_POOL_RANKED.csv"))
    ap.add_argument("--mode", choices=("grade", "final"), default="grade")
    ap.add_argument("--categories", default="")
    ap.add_argument("--limit-per-category", type=int, default=40)
    ap.add_argument("--panel", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.pool).open(encoding="utf-8-sig", newline="")))
    if args.categories:
        keep = {c.strip() for c in args.categories.split(",")}
        rows = [r for r in rows if r["case_category"] in keep]
    picked: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if len(picked[r["case_category"]]) < args.limit_per_category:
            picked[r["case_category"]].append(r)
    if not picked:
        print("no cases selected")
        return

    geoms_by_year = {y: load_geometries(y) for y in ("2018", "2019", "2020")}
    print({y: len(geoms_by_year[y]) for y in geoms_by_year}, flush=True)
    track_objects = load_track_objects()
    reader = ft.SceneReader()
    panel = args.panel or (300 if args.mode == "grade" else 560)
    out_root = Path(args.out) if args.out else (CASE_ROOT / "_grading" if args.mode == "grade" else CASE_ROOT)
    made = []
    try:
        for cat, items in picked.items():
            for r in items:
                case = dict(r)
                rank = int(r["rank_quality"])
                case["case_id"] = f"{cat}_{rank:02d}"
                p = out_root / f"GRADE_{cat}_{rank:02d}.png" if args.mode == "grade" else out_root / cat / f"CASE_{cat}_{rank:02d}_contact_sheet.png"
                if render_case_sheet(case, geoms_by_year, track_objects, reader, p, panel=panel):
                    made.append({"case_id": case["case_id"], "case_category": cat, "physical_track_id": case["physical_track_id"], "path": str(p)})
            print(f"{cat}: rendered {sum(1 for m in made if m['case_category']==cat)}", flush=True)
    finally:
        reader.close()
    print("sheets rendered:", len(made))
    if args.mode == "grade":
        with (CASE_ROOT / "_GRADING_SHEET_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "case_category", "physical_track_id", "path"])
            w.writeheader()
            w.writerows(made)


if __name__ == "__main__":
    main()
