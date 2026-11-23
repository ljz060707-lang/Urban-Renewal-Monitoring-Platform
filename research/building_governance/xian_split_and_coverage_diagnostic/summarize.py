"""Summarize the cut replay and make native-image review sheets for zero coverage."""
import argparse
from collections import Counter
import csv
from io import BytesIO
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pyogrio
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.windows import Window
import shapely
from shapely.ops import transform
from pyproj import Transformer

from replay import raster, read


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def detail(record, rgb):
    image = cv2.cvtColor(np.moveaxis(rgb, 0, -1), cv2.COLOR_RGB2GRAY).astype("float32")
    laplace = cv2.Laplacian(image, cv2.CV_32F)
    return dict(texture=float(laplace.std()), window=record["window"], epoch=record["epoch"],
                date=record["date"], site=record["site"], reference=len(record["details"]["AFTER_FACADE"]),
                zero=sum(x["covered_pixels"] == 0 for x in record["details"]["AFTER_FACADE"]))


def render_case(draw, image, offset, dataset, geometry, governed, title):
    x, y = offset
    h = 224
    col, row = dataset.index(geometry.centroid.x, geometry.centroid.y)[::-1]
    left, top = max(0, min(dataset.width-h, col-h//2)), max(0, min(dataset.height-h, row-h//2))
    patch = Window(left, top, h, h)
    rgb = np.moveaxis(dataset.read([1, 2, 3], window=patch), 0, -1).astype("uint8")
    mask = geometry_mask([geometry], out_shape=(h,h), transform=dataset.window_transform(patch), invert=True)
    roof = governed[top:top+h,left:left+h] > 0
    rgb[roof ^ cv2.erode(roof.astype("uint8"), np.ones((3,3),"uint8")).astype(bool)] = (35,232,140)
    rgb[mask ^ cv2.erode(mask.astype("uint8"), np.ones((3,3),"uint8")).astype(bool)] = (247,59,72)
    image.paste(Image.fromarray(rgb), (x,y+19))
    draw.text((x+4,y+3),title,fill="white")


def native_inputs(folder, record):
    key = record["window"].removeprefix("TUNE_")
    return folder / key


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "r7", "r16", "r17", "r18", "r23"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    r16 = load_json(args.r16/"xian_map_reference/BASELINE_OBJECTS.json")
    oracle = load_json(args.output/"WINDOW_ORACLE.json")
    with (args.output/"ZERO_COVERAGE.csv").open() as f:
        gaps = list(csv.DictReader(f))
    if len(oracle) != 72 or len(gaps) != 459:
        raise ValueError((len(oracle), len(gaps)))
    windows = []
    base = args.r7/"xian24"
    texture_file = args.output/"WINDOW_TEXTURE.csv"
    if texture_file.exists():
        with texture_file.open() as f:
            windows = [dict(r, texture=float(r["texture"]), reference=int(r["reference"]), zero=int(r["zero"]))
                       for r in csv.DictReader(f)]
    else:
        for record in r16:
            p = native_inputs(base, record)
            with rasterio.open(p/"RGB_NATIVE.tif") as ds:
                windows.append(detail(record, ds.read([1,2,3])))
        with texture_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(windows[0]))
            writer.writeheader()
            writer.writerows(windows)
    data = {"epochs": {}, "no_fit_on_development_windows": True}
    for ep in sorted({r["epoch"] for r in oracle}):
        ss = [r for r in oracle if r["epoch"] == ep]
        gs = [r for r in gaps if r["epoch"] == ep]
        ws = [r for r in windows if r["epoch"] == ep]
        r17, r18 = Counter(), Counter()
        for w in ws:
            key = w["window"]
            p17 = args.r17/"incoming/output/development_discovery/XIAN"/key/"PREDICTION.json"
            p18 = args.r18/"incoming/output/matched_budget_masks/XIAN"
            if p17.exists():
                r17.update({"windows":1, "added":load_json(p17)["added"]})
            for mode in ("native","control"):
                rec = p18/mode/key/"REC.json"
                if rec.exists():
                    r18.update({mode+"_windows":1, mode+"_added":load_json(rec)["added"]})
        dates = {}
        for date in sorted({r["date"] for r in ws}):
            subset = [r for r in ws if r["date"] == date]
            dates[date] = dict(windows=len(subset), n=sum(r["reference"] for r in subset),
                               zero=sum(r["zero"] for r in subset), texture_median=float(np.median([r["texture"] for r in subset])))
        parts = {name:sum(r[name] for r in ss) for name in
                 ("baseline","selected","gained","lost","added_cuts","reference_fragmenting_cuts","valid_cut_options","fragmenting_cut_options")}
        distance = [float(r["shifted_candidate_distance_m"]) for r in gs if r["shifted_candidate_distance_m"]]
        low = sorted(ws, key=lambda r:r["texture"])[:6]
        high = sorted(ws, key=lambda r:r["texture"])[-6:]
        data["epochs"][ep] = dict(**parts, sites_gaining=sum(r["gained"]>0 for r in ss),
                                  by_site_stratum={k:sum(r["gained"] for r in ss if r["site"].startswith(k))
                                                   for k in ("RANDOM","DENSE_REFERENCE","ELONGATED_REFERENCE","LARGE_REFERENCE")},
                                  dates=dates, no_coverage=len(gs), edge=sum(r["window_edge"]=="True" for r in gs),
                                  short_lt_5m=sum(float(r["short_side_m"])<5 for r in gs),
                                  short_5_to_10m=sum(5<=float(r["short_side_m"])<10 for r in gs),
                                  any_shifted_overlap=sum(float(r["shifted_best_iou"])>0 for r in gs),
                                  shifted_iou50=sum(float(r["shifted_best_iou"])>=.5 for r in gs),
                                  nearest_lte_5m=sum(x<=5 for x in distance), nearest_missing=len(gs)-len(distance),
                                  low_texture=(sum(r["zero"] for r in low),sum(r["reference"] for r in low)),
                                  high_texture=(sum(r["zero"] for r in high),sum(r["reference"] for r in high)),
                                  r17_candidate_counts=dict(r17), r18_candidate_counts=dict(r18))
    (args.output/"SUMMARY.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    third = [r for r in gaps if r["epoch"] == "xa201231"]
    groups = Counter(r["site"] for r in third)
    chosen = []
    for site, _ in groups.most_common(8):
        subset = sorted((r for r in third if r["site"] == site and r["window_edge"]=="False"),key=lambda r:float(r["short_side_m"]))
        if subset:
            chosen.append(subset[len(subset)//2])
    img = Image.new("RGB", (3*230, len(chosen)*248+22), (30,33,37))
    draw = ImageDraw.Draw(img)
    draw.text((4,3), "Same reference FID across dates | red: 2019 map | green: governed candidate (native)", fill="white")
    with tarfile.open(args.r23/"INPUTS.tar") as archive:
        by_site = {(r["site"],r["epoch"]):r for r in r16}
        for i, miss in enumerate(chosen):
            for j, ep in enumerate(("xa180429","xa190101","xa201231")):
                record = by_site[miss["site"],ep]
                root = native_inputs(base, record)
                gpkg = root/"PROVIDED_MAP_GT.gpkg"
                metadata,_,array,_ = pyogrio.raw.read(gpkg, columns=["original_gt_fid"],
                    where=f"original_gt_fid = {int(miss['fid'])} AND fusion_role = 'DESIGNATED_GT2019_ANCHOR'")
                if not len(array):
                    continue
                geom = shapely.from_wkb(array[0])
                member=f"xian20261111_r8/output/xian72_governance/{record['window']}/ROOF_GOVERNED_OBSERVATION_ID.tif"
                governed,_,_=raster(read(archive,member))
                with rasterio.open(root/"RGB_NATIVE.tif") as ds:
                    geom = transform(Transformer.from_crs(metadata['crs'], ds.crs, always_xy=True).transform, geom)
                    render_case(draw,img,(j*230,i*248+22),ds,geom,governed,
                                f"{record['date']}  {miss['site']}  FID {miss['fid']}")
    img.save(args.output/"THREE_DATE_REVIEW.png")
    sites = [site for site, _ in groups.most_common(3)]
    tiles = Image.new("RGB", (3*384, len(sites)*410), (30,33,37))
    caption = ImageDraw.Draw(tiles)
    for i, site in enumerate(sites):
        for j, ep in enumerate(("xa180429","xa190101","xa201231")):
            record = by_site[site,ep]
            root = native_inputs(base, record)
            metadata,_,array,_ = pyogrio.raw.read(
                root/"PROVIDED_MAP_GT.gpkg", where="fusion_role = 'DESIGNATED_GT2019_ANCHOR'"
            )
            with rasterio.open(root/"RGB_NATIVE.tif") as ds:
                rgb = np.moveaxis(ds.read([1,2,3]),0,-1).copy()
                project = Transformer.from_crs(metadata["crs"],ds.crs,always_xy=True).transform
                shapes = [transform(project, shapely.make_valid(g)) for g in shapely.from_wkb(array)]
                mask = rasterize(shapes, out_shape=ds.shape, transform=ds.transform, dtype="uint8")
                boundary = mask != cv2.erode(mask,np.ones((3,3),"uint8"))
                rgb[boundary] = (248,45,68)
            tiles.paste(Image.fromarray(rgb).resize((384,384), Image.Resampling.LANCZOS),(j*384,i*410+23))
            caption.text((j*384+6,i*410+5),f"{record['date']}  {site}   red: 2019 reference",fill="white")
    tiles.save(args.output/"TOP_THREE_WINDOWS.png")


if __name__ == "__main__":
    main()
