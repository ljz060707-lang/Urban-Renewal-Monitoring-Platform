"""Audit only the five frozen candidates. No search or actual roof transport."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from affine import Affine
from shapely.geometry import shape
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from run_xian_source_partition_alignment import sha, save_json
from xian_r5.alignment.base_tie import _erode_mask, image_shift_to_projected_correction
from xian_r5.alignment.frozen_candidate_audit import audit_frozen_translation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    old=Path("F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment/TRAINROI_20261012_01")
    out=args.output.resolve()
    if out.parent!=old.parent or out.exists():raise ValueError("Fresh sibling run required")
    receipt=json.loads((old/"IMPLEMENTATION_RECEIPT.json").read_text(encoding="utf-8"))
    protected={str(old/n):h for n,h in receipt["output_sha256"].items()}
    protected[str(old/"IMPLEMENTATION_RECEIPT.json")]="609b190b3bdd8e4d52a3937047423993daec14b19707514081fdc883644ed4c0"
    protected.update(receipt["code_sha256"])
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError("Frozen input/code drift: "+p)
    cfg=json.loads((old/"FROZEN_RUN_CONFIG.json").read_text(encoding="utf-8"))
    candidates=json.loads((old/"SOURCE_CELL_CANDIDATES.json").read_text(encoding="utf-8"))
    cells=[r for r in candidates["cells"] if r["candidate_correction_m"] is not None]
    allowlist={"roi_r0_c0","roi_r1_c0","roi_r1_c2","roi_r2_c0","roi_r3_c0"}
    if len(cells)!=5 or {r["region_id"] for r in cells}!=allowlist:raise ValueError("Five-cell scope drift")
    for p,k in [(cfg["reference"],"reference_sha256"),(cfg["teacher"],"teacher_sha256"),
                (cfg["date_ledger"],"date_ledger_sha256")]:
        if sha(p)!=cfg[k]:raise ValueError("Raw input drift: "+p)
    source=Path(cfg["source"]);st=source.stat()
    if (st.st_size,st.st_mtime_ns)!=(cfg["source_size"],cfg["source_mtime_ns"]):
        raise ValueError("Moving mosaic metadata drift")
    plan={"run":"TRAINROI_20261012_02","parent":"TRAINROI_20261012_01",
        "scope":sorted(allowlist),"pixel_window":cfg["roi_window"],
        "frozen_candidate_vectors":{r["region_id"]:r["candidate_correction_m"] for r in cells},
        "protected_expected_sha256":protected,"minimum_samples":100,
        "component_erosion_px":1,"feature_kernel_radius_px":4,
        "feature_kernel_basis":"7x7 Gaussian sigma1 radius3 followed by Sobel3x3 radius1",
        "comparison":"ZERO_VS_FROZEN_OFFSET_SAME_REFERENCE_PIXEL_COHORT",
        "independent_validation":False,
        "development_exposure":"Same input images used in candidate estimation; no new held-out evidence",
        "all_numeric_statistics":"INTERNAL_DIAGNOSTIC_ONLY_NOT_METRIC_ACCURACY",
        "candidate_reestimation":False,"formal_fit_holdout":False,"real_roof_transport":False}
    out.mkdir(parents=True);save_json(out/"FROZEN_VALIDATION_PLAN.json",plan)
    affine=Affine(*cfg["roi_affine"]);w=Window(*cfg["roi_window"])
    with rasterio.open(cfg["reference"]) as s:
        if s.window_transform(w)!=affine or str(s.crs)!="EPSG:32649":raise ValueError("Grid drift")
        ref=np.moveaxis(s.read([1,2,3],window=w),0,-1)
        rv=np.all(s.read_masks([1,2,3],window=w)>0,axis=0)
    moving=np.zeros((3,256,256),dtype=np.float32)
    with rasterio.open(source) as s:
        if (list(s.transform)[:6]!=cfg["source_grid"]["affine"]
                or s.width!=cfg["source_grid"]["width"] or s.height!=cfg["source_grid"]["height"]):
            raise ValueError("Source grid drift")
        for i in range(3):
            reproject(rasterio.band(s,i+1),moving[i],src_transform=s.transform,src_crs=s.crs,
                dst_transform=affine,dst_crs="EPSG:32649",resampling=Resampling.average,
                dst_nodata=0,warp_mem_limit=64,num_threads=1)
    moving=np.moveaxis(moving,0,-1)
    if hashlib.sha256(ref.tobytes()).hexdigest()!=candidates["reference_roi_sha256"]:
        raise ValueError("Reference content replay mismatch")
    if hashlib.sha256(moving.tobytes()).hexdigest()!=candidates["ge_resampled_roi_sha256"]:
        raise ValueError("GE content replay mismatch")
    positive=np.zeros((256,256),dtype=np.uint8)
    with rasterio.open(cfg["teacher"]) as s:
        reproject(rasterio.band(s,1),positive,src_transform=s.transform,src_crs=s.crs,
            dst_transform=affine,dst_crs="EPSG:32649",resampling=Resampling.max,
            dst_nodata=0,warp_mem_limit=64,num_threads=1)
    excluded=cv2.dilate((positive>0).astype(np.uint8),np.ones((7,7),dtype=np.uint8))>0
    if int(excluded.sum())!=candidates["known_roof_excluded_pixels"]:raise ValueError("Exclusion replay mismatch")
    rv &= ~excluded;mv=np.any(moving>0,axis=2)&~excluded

    def features(rgb):
        gray=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY)
        gray=cv2.GaussianBlur(gray,(7,7),1.)
        return cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),
                             cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))

    records=[]
    for cell in cells:
        row,col=(int(x[1:]) for x in cell["region_id"].split("_")[1:])
        component=np.zeros((256,256),bool);component[row*64:(row+1)*64,col*64:(col+1)*64]=True
        component=_erode_mask(component,1)
        x0,y0,x1,y1=cell["correlation_diagnostic"]["base_pixel_window"]
        best=cell["correlation_diagnostic"]["best"]
        observed=(best["dx_px"],best["dy_px"])
        vector=list(image_shift_to_projected_correction(*observed,affine))
        if vector!=cell["candidate_correction_m"]:raise ValueError("Vector/affine sign mismatch")
        result=audit_frozen_translation(features(ref[y0:y1,x0:x1]),
            features(moving[y0:y1,x0:x1]),rv[y0:y1,x0:x1],mv[y0:y1,x0:x1],
            component[y0:y1,x0:x1],observed,minimum_samples=100,feature_radius=4)
        records.append({"region_id":cell["region_id"],"frozen_candidate_correction_m":vector,
            "frozen_cell_evidence_sha256":hashlib.sha256(json.dumps(cell,sort_keys=True).encode()).hexdigest(),
            "internal_numeric_replay":result,"formal_fit_ties":0,"formal_holdout_ties":0,
            "ground_semantic_evidence":"NOT_BOUND","localization_bounds":"NOT_PROVIDED",
            "independent_validation_status":"NOT_ESTABLISHED",
            "disposition":"UNRESOLVED_NO_INDEPENDENT_PRECISION_EVIDENCE",
            "reference_actual_time":"UNKNOWN","relative_registration_uncertainty_m":None})
    for p,h in protected.items():
        if sha(p)!=h:raise RuntimeError("Frozen artifacts changed during audit")
    for p,k in [(cfg["reference"],"reference_sha256"),(cfg["teacher"],"teacher_sha256"),
                (cfg["date_ledger"],"date_ledger_sha256")]:
        if sha(p)!=cfg[k]:raise RuntimeError("Raw input changed during audit")
    after=source.stat()
    if (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError("Source metadata changed")
    result={"run":plan["run"],"scope":sorted(allowlist),"cells":records,
        "candidate_vectors_unchanged":True,"paired_cohort_numeric_cells":sum(
            x["internal_numeric_replay"]["paired_point_support"]["status"]=="NUMERIC_INTERNAL_REPLAY_ONLY" for x in records),
        "full_kernel_numeric_cells":sum(
            x["internal_numeric_replay"]["full_kernel_support"]["status"]=="NUMERIC_INTERNAL_REPLAY_ONLY" for x in records),
        "independent_validation_completed_cells":0,"formal_fit_holdout_completed":0,
        "accepted_cells":0,"rejected_cells":0,"unresolved_cells":5,
        "new_corrected_building_tifs":0,"actual_roof_objects_transported":0,
        "old_unknown_cells_untouched":11,"absolute_position_accuracy":"UNKNOWN",
        "training_gt_eligible":False,"source_roof_exclusion_pixels_unchanged":int(excluded.sum()),
        "source_images_replayed_byte_exact_in_bounded_arrays":True}
    save_json(out/"FROZEN_VECTOR_AUDIT.json",result)
    print(json.dumps({"run":str(out),"paired_numeric":result["paired_cohort_numeric_cells"],
                     "full_kernel_numeric":result["full_kernel_numeric_cells"],
                     "cells":[{"id":r["region_id"],"replay":r["internal_numeric_replay"]} for r in records]},ensure_ascii=True))

if __name__=="__main__":main()

