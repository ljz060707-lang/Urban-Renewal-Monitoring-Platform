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
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.ops import transform as shp_transform


SCENES = ("xa180429", "xa190101", "xa201231")
LABELS = {"xa180429": "2018", "xa190101": "2019", "xa201231": "2020"}


def txt(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--output", required=True)
    args = ap.parse_args(); root = Path(args.root); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    v2 = root / "output/16_candidate_identitygraph_v2_20260916"; formal = root / "output/西安_R4_辅助影像接入前_三期三态候选版_20260914"; gpkg = formal / "04_identity_graph/tracking_buildings.gpkg"
    queue = pd.read_csv(root / "output/19_rfinal_phase2_observability_diagnostic_20260916/phase2_identity_recovery_queue.csv", low_memory=False)
    queue = queue[queue["identity_queue_status"].astype(str).str.contains("SAFE_EDGE")].copy(); queue["area_m2"] = pd.to_numeric(queue["area_m2"], errors="coerce").fillna(0.0)
    pieces = []
    for interval in sorted(queue["interval"].unique()):
        p = queue[queue["interval"].eq(interval)].sort_values("area_m2", ascending=False).head(50).copy(); p["sample_id"] = [f"EDGE_{interval}_{i:03d}" for i in range(1, len(p)+1)]; pieces.append(p)
    samples = pd.concat(pieces, ignore_index=True); samples.to_csv(out / "IDENTITY_EDGE_VISUAL_SAMPLE_INDEX.csv", index=False, encoding="utf-8-sig")
    tr = pd.read_csv(v2 / "FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv", usecols=["physical_track_id", "object_uid"], low_memory=False); tr["physical_track_id"] = tr["physical_track_id"].astype(str); tr["object_uid"] = tr["object_uid"].astype(str)
    by_pid = tr.groupby("physical_track_id")["object_uid"].apply(list).to_dict(); needed = sorted({u for pid in samples["physical_track_id"].astype(str) for u in by_pid.get(pid, [])} | set(samples["best_candidate_object_id"].astype(str)))
    geom = {}
    for start in range(0, len(needed), 400):
        vals = ",".join("'" + u.replace("'", "''") + "'" for u in needed[start:start+400]); g = pyogrio.read_dataframe(str(gpkg), layer="tracking_buildings", where=f"object_uid IN ({vals})", columns=["object_uid", "scene_key", "area_m2", "building_id"])
        for r in g.itertuples(): geom[str(r.object_uid)] = r
    to_geo = Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True).transform; font = ImageFont.load_default(); raw_paths = {s: root / "input/base" / f"{s}_zoom_20.tif" for s in SCENES}; manifest=[]

    def project(x, y, tf):
        lon, lat = to_geo(x, y); return ((lon-tf.c)/tf.a, (tf.f-lat)/abs(tf.e))
    def read_raw(scene, rows, fallback):
        shapes=[r.geometry for r in rows if getattr(r,"geometry",None) is not None and not r.geometry.is_empty]
        if shapes:
            minx=min(g.bounds[0] for g in shapes); miny=min(g.bounds[1] for g in shapes); maxx=max(g.bounds[2] for g in shapes); maxy=max(g.bounds[3] for g in shapes); b=transform_bounds("EPSG:32649","EPSG:4326",minx,miny,maxx,maxy); cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
        else:
            lon,lat=to_geo(*fallback); cx,cy=lon,lat
        with rasterio.open(raw_paths[scene]) as src:
            size=640; col=int((cx-src.transform.c)/src.transform.a); row=int((src.transform.f-cy)/abs(src.transform.e)); win=Window(col-size//2,row-size//2,size,size); arr=src.read([1,2,3],window=win,boundless=True,fill_value=0,out_shape=(3,size,size)); return Image.fromarray(np.moveaxis(arr,0,2).astype(np.uint8),"RGB"),src.window_transform(win)
    def draw(img, row, tf, color, width, label=""):
        def f(x,y,z=None): return project(x,y,tf)
        g=shp_transform(f,row.geometry); polys=list(g.geoms) if g.geom_type=="MultiPolygon" else [g]; dr=ImageDraw.Draw(img)
        for p in polys:
            xy=list(p.exterior.coords); dr.line(xy,fill=color,width=width,joint="curve")
            if label and xy: dr.text((xy[0][0]+3,xy[0][1]+3),label,fill=color,font=font)
    for s in samples.to_dict("records"):
        pid=txt(s["physical_track_id"]); edge_obj=txt(s["best_candidate_object_id"]); active_uids=set(by_pid.get(pid,[])); fallback=(float(s["x"]),float(s["y"])); raws={}; overs={}; candidate_scenes=[]
        for scene in SCENES:
            all_g=pyogrio.read_dataframe(str(gpkg),layer="tracking_buildings",bbox=(fallback[0]-150,fallback[1]-150,fallback[0]+150,fallback[1]+150),columns=["object_uid","scene_key","area_m2","building_id"]); all_rows=[r for r in all_g.itertuples() if txt(r.scene_key)==scene]
            raw,tf=read_raw(scene,all_rows,fallback); over=raw.copy()
            for r in all_rows:
                uid=txt(r.object_uid); active=uid in active_uids; cand=uid==edge_obj; color=(0,255,255) if cand else ((255,60,60) if scene=="xa180429" and active else ((60,255,60) if scene=="xa190101" and active else ((80,160,255) if active else (255,220,40))))
                draw(over,r,tf,color,5 if cand else (3 if active else 1),"EDGE_CANDIDATE" if cand else (uid if active else ""))
            if any(txt(r.object_uid)==edge_obj for r in all_rows): candidate_scenes.append(scene)
            dr=ImageDraw.Draw(over); dr.rectangle((0,0,over.width,34),fill=(0,0,0)); dr.text((8,8),f"{LABELS[scene]} | {pid} | {s['identity_queue_status']} | {s['best_candidate_relation']}",fill="white",font=font); raws[scene]=raw; overs[scene]=over
        folder=out/"samples"/txt(s["sample_id"]); folder.mkdir(parents=True,exist_ok=True)
        for scene in SCENES: raws[scene].save(folder/f"raw_{LABELS[scene]}.png",compress_level=6); overs[scene].save(folder/f"overlay_{LABELS[scene]}.png",compress_level=6)
        tile=320; margin=12; header=48; canvas=Image.new("RGB",(tile*3+margin*4,header+tile*2+margin*3),"white"); dr=ImageDraw.Draw(canvas); dr.text((margin,12),f"{s['sample_id']} | {s['interval']} | {pid} | area={float(s['area_m2']):.1f} | score={float(s['best_score']):.3f}",fill="black",font=font)
        for j,scene in enumerate(SCENES):
            x=margin+j*(tile+margin); canvas.paste(raws[scene].resize((tile,tile)),(x,header+margin)); canvas.paste(overs[scene].resize((tile,tile)),(x,header+margin*2+tile)); dr.text((x+5,header+margin+5),f"raw {LABELS[scene]}",fill="yellow",font=font); dr.text((x+5,header+margin*2+tile+5),f"overlay {LABELS[scene]}",fill="yellow",font=font)
        canvas.save(folder/"contact_sheet.png",compress_level=6); summary={k:(v.item() if isinstance(v,np.generic) else v) for k,v in s.items()}; summary.update({"edge_candidate_object_id":edge_obj,"candidate_scenes_in_bbox":";".join(candidate_scenes),"formal_outputs_modified":False,"visual_status":"PENDING_TARGETED_REVIEW"}); (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8"); manifest.append({"sample_id":txt(s["sample_id"]),"interval":txt(s["interval"]),"physical_track_id":pid,"area_m2":float(s["area_m2"]),"status":txt(s["identity_queue_status"]),"candidate_object_id":edge_obj,"candidate_scenes":";".join(candidate_scenes),"folder":str(folder.relative_to(out))})
    pd.DataFrame(manifest).to_csv(out/"IDENTITY_EDGE_VISUAL_MANIFEST.csv",index=False,encoding="utf-8-sig"); receipt={"sample_count":len(manifest),"geometry_requested":len(needed),"geometry_found":len(geom),"formal_outputs_modified":False,"review_status":"PENDING_TARGETED_REVIEW"}; (out/"IDENTITY_EDGE_VISUAL_RECEIPT.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(receipt,ensure_ascii=False))


if __name__ == "__main__": main()
