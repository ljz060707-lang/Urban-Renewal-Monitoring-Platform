"""User-authorized display-only roof candidate preview. Never training labels."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling, ColorInterp
from PIL import Image,ImageDraw,ImageFont
from affine import Affine
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.label_transfer import Grid
from xian_r5.source_partition import SourceRegion
from xian_r5.alignment.roof_overlay_preview import render_roof_overlay_preview,ROOF_PREVIEW_QUALITY

def font(size):
    for p in ("C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf"):
        if Path(p).is_file():return ImageFont.truetype(p,size)
    raise RuntimeError("Readable Chinese font required")

def display_panel(reference,layer,quality,second=None):
    rgb=reference.copy()
    if second is None:
        show=layer[3]>0;rgb[show]=np.moveaxis(layer[:3],0,-1)[show]
    else:
        a=layer[3]>0;b=second[3]>0
        rgb[a]=np.array([0,220,245],np.uint8)
        rgb[b]=np.array([255,0,160],np.uint8)
        rgb[a&b]=np.array([255,255,255],np.uint8)
    unknown=quality==2;seam=quality==3
    rgb[unknown]=(rgb[unknown].astype(float)*.55+65).astype(np.uint8)
    y,x=np.indices(quality.shape);hatch=unknown & ((x+y)%9==0)
    rgb[hatch]=np.array([145,145,145],np.uint8)
    rgb[seam]=np.array([245,155,25],np.uint8)
    return rgb

def write_tif(path,array,grid,kind,tags,alpha=False):
    if path.exists():raise FileExistsError(path)
    with rasterio.open(path,"w",driver="GTiff",count=array.shape[0],dtype="uint8",
            width=grid.width,height=grid.height,crs=grid.crs,transform=grid.affine,
            tiled=True,blockxsize=256,blockysize=256,compress="deflate") as s:
        s.write(array)
        if alpha:s.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
        s.update_tags(PRODUCT_KIND=kind,TRAINING_USE_PROHIBITED="YES",
            FORMAL_CORRECTION_ACCEPTED="NO",POSITIONAL_UNCERTAINTY_M="UNKNOWN",
            ABSOLUTE_POSITION_ACCURACY="UNKNOWN",GROUND_FOOTPRINT="NOT_ESTABLISHED",
            **tags)
    with rasterio.open(path) as s:
        if s.transform!=grid.affine or not np.array_equal(s.read(),array):
            raise RuntimeError("Preview raster roundtrip mismatch")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    parent=Path("F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment")
    old=parent/"TRAINROI_20261012_01";audit=parent/"TRAINROI_20261012_02";out=args.output.resolve()
    if out.parent!=parent.resolve() or out.exists():raise ValueError("Fresh dedicated preview directory required")
    r1=json.loads((old/"IMPLEMENTATION_RECEIPT.json").read_text(encoding="utf-8"))
    r2=json.loads((audit/"VALIDATION_AUDIT_RECEIPT.json").read_text(encoding="utf-8"))
    protected={str(old/n):h for n,h in r1["output_sha256"].items()}
    protected.update({str(audit/n):h for n,h in r2["output_sha256"].items()})
    protected[str(old/"IMPLEMENTATION_RECEIPT.json")]="609b190b3bdd8e4d52a3937047423993daec14b19707514081fdc883644ed4c0"
    protected[str(audit/"VALIDATION_AUDIT_RECEIPT.json")]="b1f9ab3f5dc8dccbde6846398ae5a22344d4b6cc43b3539e2c344b96fd73df29"
    protected.update(r1["code_sha256"]);protected.update(r2["new_code_sha256"])
    for path,h in protected.items():
        if sha(path)!=h:raise ValueError("Frozen input/code drift: "+path)
    cfg=json.loads((old/"FROZEN_RUN_CONFIG.json").read_text(encoding="utf-8"))
    field=json.loads((old/"SOURCE_CELL_CANDIDATES.json").read_text(encoding="utf-8"))
    if len(field["cells"])!=16 or field["numeric_candidate_count"]!=5:
        raise ValueError("Frozen preview scope drift")
    for path,key in ((cfg["reference"],"reference_sha256"),(cfg["teacher"],"teacher_sha256")):
        if sha(path)!=cfg[key]:raise ValueError("Raw input drift")
    window=Window(*cfg["roi_window"])
    with rasterio.open(cfg["reference"]) as s:
        grid=Grid(s.window_transform(window),256,256,str(s.crs))
        reference=np.moveaxis(s.read([1,2,3],window=window),0,-1)
    if list(grid.affine)[:6]!=cfg["roi_affine"]:raise ValueError("Frozen grid drift")
    if hashlib.sha256(reference.tobytes()).hexdigest()!=field["reference_roi_sha256"]:
        raise ValueError("Reference content drift")
    plan={"run":"TRAINROI_20261012_03","user_authorization":"User: 始终允许; limited to candidate overlay previews",
        "authorization_does_not_waive_formal_acceptance":True,"scope":"16 spatial cells, 5 numeric and 11 transparent unknown",
        "source_teacher":cfg["teacher"],"source_teacher_sha256":cfg["teacher_sha256"],
        "reference":cfg["reference"],"reference_sha256":cfg["reference_sha256"],
        "source_actual_date":cfg["source_actual_date"],"reference_actual_time":"UNKNOWN",
        "grid_affine":list(grid.affine)[:6],"crs":grid.crs,"shape":[256,256],
        "foreground_semantics":"ANY_POSITIVE_TEACHER_PIXEL_ENVELOPE_FOR_DISPLAY_ONLY",
        "fraction_or_0_43_label_generation":False,"threshold_0_43_changed":False,
        "source_support_clipped_before_shift":True,"seam_quarantine_px":1,
        "unknown_displacements_filled":False,"camera_acquisition_inferred":False,
        "protected_expected_sha256":protected,"training_use_allowed":False}
    out.mkdir(parents=True);save_json(out/"FROZEN_PREVIEW_PLAN.json",plan)
    foreground=np.zeros((256,256),np.uint8)
    with rasterio.open(cfg["teacher"]) as s:
        source_tags=s.tags()
        reproject(rasterio.band(s,1),foreground,src_transform=s.transform,src_crs=s.crs,
            dst_transform=grid.affine,dst_crs=grid.crs,resampling=Resampling.max,
            dst_nodata=0,warp_mem_limit=64,num_threads=1)
    regions=[]
    for cell in field["cells"]:
        if cell["formal_acceptance"]:raise ValueError("Preview cannot reinterpret accepted rows")
        vector=cell["candidate_correction_m"]
        digest=hashlib.sha256(json.dumps(cell,sort_keys=True).encode()).hexdigest()
        regions.append(SourceRegion(cell["region_id"],cell["actual_source_date"],shape(cell["support_geometry"]),
            None if vector is None else tuple(vector),None if vector is None else digest))
    result=render_roof_overlay_preview(foreground>0,grid,regions)
    tags={"SOURCE_TEACHER_SHA256":cfg["teacher_sha256"],"SOURCE_FIELD_SHA256":sha(old/"SOURCE_CELL_CANDIDATES.json"),
          "SOURCE_ACTUAL_DATE":cfg["source_actual_date"],"REFERENCE_SHA256":cfg["reference_sha256"],
          "REFERENCE_LABEL_YEAR":"2018","REFERENCE_ACTUAL_TIME":"UNKNOWN",
          "SCOPE":"TRAINROI_256x256_5_NUMERIC_11_UNRESOLVED",
          "FOREGROUND_SEMANTICS":"ANY_POSITIVE_ENVELOPE_NOT_AREA_FRACTION_NOT_TAU043_LABEL",
          "ZERO_SEMANTICS":result["zero_semantics"],"NO_BUILDING_INSTANCE_RECONSTRUCTION":"YES"}
    write_tif(out/"ORIGINAL_ROOF_OUTLINE_PREVIEW_ONLY.tif",result["original_rgba"],grid,
              "UNSHIFTED_ROOF_OUTLINE_VISUALIZATION",tags,alpha=True)
    write_tif(out/"CANDIDATE_ROOF_OUTLINE_PREVIEW_ONLY.tif",result["candidate_rgba"],grid,
              "UNVERIFIED_PIECEWISE_ROOF_OUTLINE_VISUALIZATION",tags,alpha=True)
    write_tif(out/"ROOF_OVERLAY_QUALITY_PREVIEW_ONLY.tif",result["quality"][None],grid,
              "PREVIEW_QUALITY_NOT_POSITION_ACCURACY",
              {**tags,"QUALITY_CODES":json.dumps(ROOF_PREVIEW_QUALITY,sort_keys=True)})
    left=display_panel(reference,result["original_rgba"],result["quality"])
    right=display_panel(reference,result["original_rgba"],result["quality"],result["candidate_rgba"])
    canvas=Image.new("RGB",(1600,980),"#fafafa");draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,1600,54),fill="#70202a")
    draw.text((25,9),"候选叠加预览 · 未验证 · 禁止训练",font=font(28),fill="white")
    draw.text((24,68),"原始亚米轮廓（青）",font=font(22),fill="#183e48")
    draw.text((820,68),"原始（青）与候选（粉）叠加；重叠为白",font=font(22),fill="#5a233a")
    for x,image in ((24,left),(820,right)):
        canvas.paste(Image.fromarray(image).resize((768,768),Image.Resampling.NEAREST),(x,110))
    for i in range(4):
        for j in range(4):
            cell=field["cells"][i*4+j];vec=cell["candidate_correction_m"]
            label=f"r{i}c{j}: 未解决" if vec is None else f"r{i}c{j}: 候选东移{vec[0]:g}m"
            for x in (24,820):
                px,py=x+j*192+4,110+i*192+4
                draw.rectangle((px,py,px+182,py+24),fill="#ffffff")
                draw.text((px+2,py+2),label,font=font(15),fill="#333333")
    draw.text((24,890),"灰斜线：无位移候选；橙色：接缝/边界风险（隐藏轮廓）；零向量也未验证。",font=font(20),fill="#333333")
    draw.text((24,922),"亚米实际日期 2018-11-01；10m参考名义2018，真实时间未知。未生成纠偏GT或地面足迹。",font=font(19),fill="#333333")
    canvas.save(out/"CANDIDATE_OVERLAY_FULL_ROI.png")
    sheet=Image.new("RGB",(710,5*302+72),"#fafafa");draw=ImageDraw.Draw(sheet)
    draw.rectangle((0,0,710,62),fill="#70202a")
    draw.text((16,12),"5个候选单元 · 仅视觉对照 · 不用于训练",font=font(23),fill="white")
    numeric=[r for r in field["cells"] if r["candidate_correction_m"] is not None]
    for n,cell in enumerate(numeric):
        rr,cc=(int(s[1:]) for s in cell["region_id"].split("_")[1:])
        y=72+n*302
        draw.text((14,y),f'{cell["region_id"]}  冻结候选 {cell["candidate_correction_m"]} m（未验证）',font=font(17),fill="#333")
        for x,img in ((16,left),(370,right)):
            crop=img[rr*64:(rr+1)*64,cc*64:(cc+1)*64]
            sheet.paste(Image.fromarray(crop).resize((256,256),Image.Resampling.NEAREST),(x,y+28))
    sheet.save(out/"FIVE_CELL_VISUAL_COMPARISON.png")
    counts={str(int(k)):int(v) for k,v in zip(*np.unique(result["quality"],return_counts=True))}
    checks={"numeric_cells_rendered":sum(r["candidate_rendered"] for r in result["regions"])==5,
        "unknown_11_cells_retained":sum(not r["candidate_rendered"] for r in result["regions"])==11,
        "unknown_alpha_zero":not result["candidate_rgba"][3,result["quality"]==2].any(),
        "seam_alpha_zero":not result["candidate_rgba"][3,result["quality"]==3].any(),
        "no_training_permission":not result["training_use_allowed"],
        "fixed_full_pixel_denominator":sum(counts.values())==65536}
    if not all(checks.values()):raise RuntimeError(checks)
    for path,h in protected.items():
        if sha(path)!=h:raise RuntimeError("Frozen input changed during preview")
    if sha(cfg["reference"])!=cfg["reference_sha256"] or sha(cfg["teacher"])!=cfg["teacher_sha256"]:
        raise RuntimeError("Raw input changed during preview")
    save_json(out/"PREVIEW_SOURCE_LINEAGE_AND_CHECKS.json",{
        "run":plan["run"],"source_teacher_tags":source_tags,"regions":result["regions"],
        "quality_counts":counts,"quality_codes":ROOF_PREVIEW_QUALITY,"checks":checks,
        "source_teacher_sha256":cfg["teacher_sha256"],"reference_sha256":cfg["reference_sha256"],
        "foreground_display_pixels":int((foreground>0).sum()),
        "candidate_foreground_display_pixels":int(result["candidate_foreground"].sum()),
        "candidate_preview_rendered_cells":5,"candidate_preview_shifted_cells":2,"candidate_preview_zero_cells":3,
        "production_roof_instances_transported":0,"training_gt_eligible":False,
        "formal_correction_accepted":False,"absolute_position_accuracy":"UNKNOWN",
        "image_layout_inspection_is_not_ground_feature_review":True})
    print(json.dumps({"output":str(out),"quality_counts":counts,"checks":checks,"protected_entries":len(protected)},ensure_ascii=True))

if __name__=="__main__":main()

