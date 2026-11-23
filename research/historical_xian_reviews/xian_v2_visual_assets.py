"""Generate real raw/overlay/contact-sheet assets for the isolated V2 candidate."""
from __future__ import annotations
import argparse, ast, csv, json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import Transformer

SCENES = ("xa180429","xa190101","xa201231")
LABELS = {"xa180429":"2018","xa190101":"2019","xa201231":"2020"}
RAW = {"xa180429":"xa180429_zoom_20.tif","xa190101":"xa190101_zoom_20.tif","xa201231":"xa201231_zoom_20.tif"}

def txt(v):
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except Exception: pass
    return str(v)

def parse_list(v):
    try:
        x=ast.literal_eval(txt(v))
        return [str(z) for z in x] if isinstance(x,list) else []
    except (SyntaxError,ValueError):
        return []

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows({f:r.get(f,"") for f in fields} for r in rows)

def choose_held(df, n=300, seed=20260916):
    d=df[(df.source_epoch.eq("xa190101"))&(df.target_epoch.eq("xa201231"))].copy()
    if d.empty: return d
    d["score_bin"]=pd.qcut(d.score.rank(method="first"),5,labels=False,duplicates="drop")
    d["area_bin"]=pd.qcut(d.area_ratio.rank(method="first"),5,labels=False,duplicates="drop")
    d["iou_bin"]=pd.qcut(d.buffered_iou.rank(method="first"),5,labels=False,duplicates="drop")
    d["neigh_bin"]=pd.qcut(d.neighborhood.rank(method="first"),5,labels=False,duplicates="drop")
    d["disp_bin"]=pd.qcut(d.displacement.rank(method="first"),5,labels=False,duplicates="drop")
    rng=np.random.default_rng(seed)
    selected=[]; used=set()
    plans=[("score_bin",100),("area_bin",50),("iou_bin",50),("neigh_bin",50),("disp_bin",50)]
    for col,want in plans:
        groups=list(d.groupby(col,dropna=False,sort=True))
        candidates=[]
        for _,g in groups:
            ix=g.index.to_numpy().copy(); rng.shuffle(ix); candidates.extend(ix.tolist())
        for ix in candidates[:want]:
            if ix not in used:
                selected.append(ix); used.add(ix)
    if len(selected)<n:
        remain=d.loc[~d.index.isin(used)].index.to_numpy().copy(); rng.shuffle(remain); selected.extend(remain[:n-len(selected)].tolist())
    return d.loc[selected[:n]].copy().reset_index(drop=True)

class Builder:
    def __init__(self,args):
        self.args=args; self.root=Path(args.root); self.v2=Path(args.v2); self.out=Path(args.output)
        self.formal=self.root/"output"/"西安_R4_辅助影像接入前_三期三态候选版_20260914"
        self.gpkg=self.formal/"04_identity_graph"/"tracking_buildings.gpkg"
        self.raw={s:self.root/"input"/"base"/RAW[s] for s in SCENES}
        self.to_geo=Transformer.from_crs("EPSG:32649","EPSG:4326",always_xy=True).transform
        self.from_geo={s:Transformer.from_crs("EPSG:4326","EPSG:32649",always_xy=True).transform for s in SCENES}
        self.font=ImageFont.load_default()
        self.out.mkdir(parents=True)
    def read_geoms(self,uids):
        uids=sorted(set(uids)); rows=[]
        for start in range(0,len(uids),400):
            chunk=uids[start:start+400]
            vals=",".join("'" + u.replace("'","''") + "'" for u in chunk)
            where="object_uid IN ("+vals+")"
            try:
                g=pyogrio.read_dataframe(self.gpkg,layer="tracking_buildings",where=where,columns=["object_uid","scene_key","area_m2","building_id"])
                if not g.empty: rows.append(g)
            except Exception as e:
                print("geometry query failed",start,e,flush=True)
        if not rows:
            return {}
        g=pd.concat(rows,ignore_index=True)
        return {str(r.object_uid):r for r in g.itertuples()}
    def load_raw(self,scene,geom_list):
        geoms=[g.geometry for g in geom_list if g is not None and g.geometry is not None and not g.geometry.is_empty]
        if geoms:
            bounds=max([g.bounds for g in geoms],key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
            minx=min(g.bounds[0] for g in geoms); miny=min(g.bounds[1] for g in geoms); maxx=max(g.bounds[2] for g in geoms); maxy=max(g.bounds[3] for g in geoms)
            b432=transform_bounds("EPSG:32649","EPSG:4326",minx,miny,maxx,maxy)
        else:
            b432=(108.9,34.25,108.9001,34.2501)
        with rasterio.open(self.raw[scene]) as src:
            px=abs(src.transform.a); py=abs(src.transform.e)
            cx=(b432[0]+b432[2])/2; cy=(b432[1]+b432[3])/2
            span=max((b432[2]-b432[0])/px,(b432[3]-b432[1])/py)
            size=int(max(256,min(768,max(256,span+80))))
            col=int((cx-src.transform.c)/src.transform.a); row=int((src.transform.f-cy)/abs(src.transform.e))
            win=Window(col-size//2,row-size//2,size,size)
            arr=src.read([1,2,3],window=win,boundless=True,fill_value=0,out_shape=(3,size,size))
            arr=np.moveaxis(arr,0,2).astype(np.uint8)
            transform=src.window_transform(win)
        return Image.fromarray(arr,"RGB"),transform
    def pix(self,x,y,transform):
        return ((x-transform.c)/transform.a,(transform.f-y)/abs(transform.e))
    def draw_geom(self,img,geom,transform,color,width=3,label=""):
        def convert(x,y,z=None): return self.pix(*self.to_geo(x,y),transform)
        g=shp_transform(convert,geom)
        dr=ImageDraw.Draw(img)
        polys=list(g.geoms) if g.geom_type=="MultiPolygon" else [g]
        for poly in polys:
            xy=[self.pix(*self.to_geo(x,y),transform) for x,y in poly.exterior.coords]
            dr.line(xy+[xy[0]],fill=color,width=width,joint="curve")
            if label and xy:
                dr.text((xy[0][0]+3,xy[0][1]+3),label,fill=color,font=self.font)
    def make_sample(self,sample,geoms_by_uid,folder,kind):
        folder.mkdir(parents=True,exist_ok=True)
        all_geoms=[r.geometry for r in geoms_by_uid.values() if r is not None]
        raws={}; overs={}
        state=txt(sample.get("state",sample.get("final_state","")))
        pid=txt(sample.get("physical_track_id"))
        seq=txt(sample.get("presence_sequence"))
        for scene in SCENES:
            gs=[r for r in geoms_by_uid.values() if r is not None and txt(r.scene_key)==scene]
            raw,transform=self.load_raw(scene,[r for r in gs])
            over=raw.copy()
            for r in gs:
                col=(255,80,80) if scene=="xa180429" else ((80,255,80) if scene=="xa190101" else (80,160,255))
                self.draw_geom(over,r.geometry,transform,col,3,txt(r.object_uid))
            dr=ImageDraw.Draw(over); dr.rectangle((0,0,over.width,28),fill=(0,0,0)); dr.text((6,6),f"{LABELS[scene]} | {pid} | {state} | {seq}",fill="white",font=self.font)
            raws[scene]=raw; overs[scene]=over
            raw.save(folder/f"raw_{LABELS[scene]}.png",compress_level=1)
            over.save(folder/f"overlay_{LABELS[scene]}.png",compress_level=1)
        tile=320; margin=12; header=42; canvas=Image.new("RGB",(tile*3+margin*4,header+tile*2+margin*3),"white")
        dr=ImageDraw.Draw(canvas); title=f"{txt(sample.get('sample_id'))} | {txt(sample.get('category',kind))} | {pid} | {state} | {seq}"
        dr.text((margin,12),title,fill="black",font=self.font)
        for j,scene in enumerate(SCENES):
            x=margin+j*(tile+margin)
            canvas.paste(raws[scene].resize((tile,tile)),(x,header+margin))
            canvas.paste(overs[scene].resize((tile,tile)),(x,header+margin*2+tile))
            dr.text((x+5,header+margin+5),f"raw {LABELS[scene]}",fill="yellow",font=self.font)
            dr.text((x+5,header+margin*2+tile+5),f"overlay {LABELS[scene]}",fill="yellow",font=self.font)
        canvas.save(folder/"contact_sheet.png",compress_level=1)
        summary={k:(v if not isinstance(v,np.generic) else v.item()) for k,v in sample.items()}
        summary.update({"sample_id":txt(sample.get("sample_id")),"category":txt(sample.get("category",kind)),"physical_track_id":pid,"presence_sequence":seq,"final_state":state,"raw_asset_status":"GENERATED","visual_review_status":"PENDING_CODEX_REVIEW"})
        (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    def run(self):
        tr=pd.read_csv(self.v2/"FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv",encoding="utf-8-sig",low_memory=False)
        tr.object_uid=tr.object_uid.astype(str); tr.physical_track_id=tr.physical_track_id.astype(str)
        by_pid=tr.groupby("physical_track_id")["object_uid"].apply(list).to_dict()
        state=pd.read_csv(self.v2/"STATE_V2_FROM_SCRATCH.csv",encoding="utf-8-sig",low_memory=False)
        idx=pd.read_csv(self.v2/"VISUAL_QA_SAMPLE_INDEX_V2.csv",encoding="utf-8-sig",low_memory=False)
        held=pd.read_csv(self.v2/"HELD_FULL_RECOVERY_DECISION.csv",encoding="utf-8-sig",low_memory=False)
        held300=choose_held(held,300)
        held300["sample_id"]=[f"HELD_2019_2020_{i:03d}" for i in range(1,len(held300)+1)]
        held300["category"]="HELD_2019_2020"
        held300.to_csv(self.out/"HELD_2019_2020_SAMPLE_INDEX.csv",index=False,encoding="utf-8-sig")
        samples=[]
        for r in idx.to_dict("records"):
            samples.append(r)
        for r in held300.to_dict("records"):
            samples.append(r)
        all_uids=set()
        for r in samples:
            pid=txt(r.get("physical_track_id"))
            if pid: all_uids.update(by_pid.get(pid,[]))
            for c in ("source_object_id","target_object_id"):
                if txt(r.get(c)): all_uids.add(txt(r[c]))
        geom=self.read_geoms(all_uids)
        missing=sorted(all_uids-set(geom))
        (self.out/"GEOMETRY_QUERY_RECEIPT.json").write_text(json.dumps({"requested_uids":len(all_uids),"found_uids":len(geom),"missing_uids":len(missing),"missing_examples":missing[:20]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        manifest=[]; count=0
        for r in samples:
            count+=1
            pid=txt(r.get("physical_track_id"))
            uids=set(by_pid.get(pid,[]))
            for c in ("source_object_id","target_object_id"):
                if txt(r.get(c)): uids.add(txt(r[c]))
            g={u:geom[u] for u in uids if u in geom}
            folder=self.out/("state_samples" if count<=len(idx) else "held_2019_2020")/txt(r.get("sample_id"))
            self.make_sample(r,g,folder,"state" if count<=len(idx) else "held")
            manifest.append({"sample_id":txt(r.get("sample_id")),"category":txt(r.get("category")),"folder":str(folder.relative_to(self.out)),"asset_status":"GENERATED","geometry_count":len(g)})
            if count%25==0: print("generated",count,flush=True)
        write_csv(self.out/"VISUAL_ASSET_MANIFEST.csv",manifest,list(manifest[0]))
        (self.out/"README.md").write_text("# V2 visual QA assets\n\nState samples: 100 each for STABLE, ADDITION, DEMOLITION, UNKNOWN. Held calibration: 300 independent 2019->2020 samples. Raw imagery and polygon/track overlays are generated from the frozen raster and canonical geometry; labels remain pending Codex review.\n",encoding="utf-8")
        print(json.dumps({"state_samples":len(idx),"held_2019_2020":len(held300),"geometry_found":len(geom),"geometry_missing":len(missing)},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--v2",required=True); ap.add_argument("--output",required=True)
    Builder(ap.parse_args()).run()

if __name__=="__main__": main()
