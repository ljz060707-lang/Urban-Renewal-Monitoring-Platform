"""Bounded, image-only candidate alignment; never rewrites source or GT."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import box, mapping
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from xian_r5.alignment.base_tie import (estimate_component_gradient_translation,
                                       image_shift_to_projected_correction)
from xian_r5.source_partition import SourceRegion, source_partition_transport

def sha(p):
    h=hashlib.sha256();p=Path(p);before=p.stat()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    after=p.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError("Input changed while hashing: "+str(p))
    return h.hexdigest()

def write_new(p,s):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:f.write(s)

def save_json(p,x):write_new(p,json.dumps(x,ensure_ascii=False,indent=2)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    allowed=Path("F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment").resolve()
    if out.parent!=allowed or out.exists():raise ValueError("Fresh dedicated run directory required")
    source=Path("F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif")
    reference=Path("F:/xian/10m底图/Xian_2018_TrueColor.tif")
    teacher=Path("D:/weixin chat/xwechat_files/wxid_w9xgzlr6hnvp22_40c8/msg/file/2026-10/2019_BUILDING_BOUNDARY_ge100m2.tif")
    date_ledger=Path("F:/xian/亚米市区底图/date_audit/tables/tile_actual_date.csv")
    old_audit=Path("F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/20_gt10m_label_transfer/GT10M_SUPPORT_20261011_01/PROVENANCE_RECOVERY_AUDIT.json")
    history=json.loads(old_audit.read_text(encoding="utf-8"))
    legacy_date=str(Path("F:/xian/xian/date_audit/tables/tile_actual_date.csv"))
    if sha(date_ledger)!=history["source_hashes"][legacy_date]:raise ValueError("Date evidence changed")
    d=next(x for x in history["GE_training_ROI_source_date_audit"] if x["scene"]=="xa190101")
    if d["date_histogram"]!={"2018-11-01":1433} or d["missing_date_rows"] or d["conflicting_date_rows"]:
        raise ValueError("Exact pilot source date not confirmed")
    if sha(teacher)!=history["source_hashes"][str(teacher)]:raise ValueError("Teacher source changed")
    w=Window(1516,1765,256,256)
    with rasterio.open(reference) as r:
        if str(r.crs)!="EPSG:32649":raise ValueError("Metric reference required")
        affine=r.window_transform(w);crs=r.crs
        bounds=rasterio.windows.bounds(w,r.transform)
        if list(bounds)!=[308460,3792190,311020,3794750]:raise ValueError("Frozen ROI drift")
        ref=np.moveaxis(r.read([1,2,3],window=w),0,-1)
        rv=np.all(r.read_masks([1,2,3],window=w)>0,axis=0)
    st=source.stat()
    with rasterio.open(source) as s:
        grid={"crs":str(s.crs),"width":s.width,"height":s.height,
              "affine":list(s.transform)[:6],"bounds":list(s.bounds)}
        if list(s.bounds)!=history["supplied_submeter_candidate"]["bounds"]:
            raise ValueError("Teacher/source grid mismatch")
    config={"run":"TRAINROI_20261012_01","source":str(source),"teacher":str(teacher),
        "source_grid":grid,"source_size":st.st_size,"source_mtime_ns":st.st_mtime_ns,
        "source_actual_date":"2018-11-01","reference":str(reference),
        "reference_label_year":2018,"reference_actual_time":"UNKNOWN",
        "reference_choice":"SOURCE_ACTUAL_CALENDAR_YEAR_NOT_MAXIMUM_CORRELATION",
        "reference_sha256":sha(reference),"teacher_sha256":sha(teacher),
        "date_ledger":str(date_ledger),"date_ledger_sha256":sha(date_ledger),
        "roi_bounds":list(bounds),"roi_window":[1516,1765,256,256],
        "roi_affine":list(affine)[:6],"subdivision":[4,4],
        "search_radius_px":3,"search_radius_m":30,"minimum_edge_samples":100,
        "known_roof_exclusion":"teacher positive max-aggregation then 30m dilation",
        "source_regions_are_camera_acquisitions":False,
        "ground_feature_semantics":"ROOF_EXCLUSION_ONLY; STABLE_GROUND_NOT_VERIFIED",
        "student_predictions_used_in_alignment":False,"old_B04_shift_used":False,
        "actual_same_date_training_eligibility":"NOT_ESTABLISHED",
        "source_content_fingerprint":"bounded RGB array SHA256 will be recorded; no full-mosaic scan"}
    out.mkdir(parents=True);save_json(out/"FROZEN_RUN_CONFIG.json",config)
    print("FROZEN_CONFIG_WRITTEN",flush=True)
    moving=np.zeros((3,256,256),dtype=np.float32)
    with rasterio.open(source) as s:
        for i in range(3):
            reproject(rasterio.band(s,i+1),moving[i],src_transform=s.transform,src_crs=s.crs,
                dst_transform=affine,dst_crs=crs,resampling=Resampling.average,
                dst_nodata=0,warp_mem_limit=64,num_threads=1)
    moving=np.moveaxis(moving,0,-1);mv=np.any(moving>0,axis=2)
    positive=np.zeros((256,256),dtype=np.uint8)
    with rasterio.open(teacher) as s:
        reproject(rasterio.band(s,1),positive,src_transform=s.transform,src_crs=s.crs,
            dst_transform=affine,dst_crs=crs,resampling=Resampling.max,
            dst_nodata=0,warp_mem_limit=64,num_threads=1)
    exclusion=cv2.dilate((positive>0).astype(np.uint8),np.ones((7,7),dtype=np.uint8))>0
    rv &= ~exclusion;mv &= ~exclusion
    cells=[];regions=[]
    for row in range(4):
        for col in range(4):
            c0,r0=col*64,row*64;c1,r1=c0+64,r0+64
            component=np.zeros((256,256),dtype=bool);component[r0:r1,c0:c1]=True
            result=estimate_component_gradient_translation(ref,moving,rv,mv,component,radius=3,min_samples=100)
            best=result["best"];vector=None
            if best:
                vector=list(image_shift_to_projected_correction(best["dx_px"],best["dy_px"],affine))
            a=affine*(c0,r0);b=affine*(c1,r1)
            support=box(min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1]))
            cell={"region_id":f"roi_r{row}_c{col}","actual_source_date":"2018-11-01",
                  "support_geometry":mapping(support),"candidate_correction_m":vector,
                  "correlation_diagnostic":result,"positional_uncertainty_m":None,
                  "relative_registration_uncertainty_m":None,
                  "status":"UNREVIEWED_IMAGE_FEATURE_CANDIDATE" if best else "UNRESOLVED_NO_NUMERIC_PEAK",
                  "precision_eligible_ties":0,"formal_acceptance":False,
                  "stable_ground_evidence":"NOT_VERIFIED","roof_parallax_residual":"UNKNOWN"}
            cells.append(cell)
            digest=hashlib.sha256(json.dumps(cell,sort_keys=True).encode()).hexdigest()
            regions.append(SourceRegion(cell["region_id"],"2018-11-01",support,
                                        tuple(vector) if vector is not None else None,
                                        digest if vector is not None else None))
            print("CELL_DONE",row,col,flush=True)
    # Check the transport interface without transporting any actual roof.
    empty=source_partition_transport({},regions,box(*bounds))
    after=source.stat()
    if (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError("Source mosaic changed")
    if sha(reference)!=config["reference_sha256"] or sha(teacher)!=config["teacher_sha256"]:
        raise RuntimeError("Input changed during candidate production")
    result={"run":config["run"],"cells":cells,"numeric_candidate_count":sum(x["candidate_correction_m"] is not None for x in cells),
            "ge_resampled_roi_sha256":hashlib.sha256(moving.tobytes()).hexdigest(),
            "reference_roi_sha256":hashlib.sha256(ref.tobytes()).hexdigest(),
            "known_roof_excluded_pixels":int(exclusion.sum()),
            "unverified_non_roof_comparison_pixels":int((rv&mv).sum()),
            "transport_contract_preflight":{k:v for k,v in empty.items() if k!="candidate_geometries"},
            "accepted_cells":0,"applied_actual_roof_objects":0,"new_corrected_building_tifs":0,
            "absolute_position_accuracy":"UNKNOWN","training_gt_eligible":False}
    save_json(out/"SOURCE_CELL_CANDIDATES.json",result)
    save_json(out/"SOURCE_CELL_SUPPORTS.geojson",{"type":"FeatureCollection","name":"EPSG32649_CANDIDATE_ONLY",
        "crs":{"type":"name","properties":{"name":"EPSG:32649"}},"features":[{"type":"Feature","geometry":x["support_geometry"],
        "properties":{k:x[k] for k in ("region_id","actual_source_date","candidate_correction_m","status","formal_acceptance")}} for x in cells]})
    print(json.dumps({"completed":True,"output":str(out),"numeric_candidates":result["numeric_candidate_count"]}),flush=True)

if __name__=="__main__":main()

