from __future__ import annotations
import csv,hashlib,json,random
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
DATES=('2018-05-06','2020-07-23','2022-03-20','2023-07-16')
TRANSITIONS=('20180506_to_20230716','20200723_to_20230716','20220320_to_20230716')
SURFACE=('vacant_land_candidate','engineering_facility_candidate','construction_waste_candidate','renewal_activity_candidate','fusu_native_semantic')
def _hash(paths):
    h=hashlib.sha256()
    for p in sorted(Path(x) for x in paths if Path(x).is_file()): h.update(str(p).encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
def build_inventory(project,review_dir,seed=20260801):
    project=Path(project); review_dir=Path(review_dir); geom=project/'outputs/inference/beijing_tongzhou_building_geometry'; overlay_root=project/'outputs/integration/urban_renewal_overlay_with_geometry'; overlays=sorted(overlay_root.glob('*/MASTER_OVERLAY_MANIFEST.json')); overlay=overlays[-1].parent if overlays else None; review_dir.mkdir(parents=True,exist_ok=True); prev=review_dir/'previews'; prev.mkdir(exist_ok=True); records=[]
    for date in DATES:
        overview=geom/date/'overview.png'
        if not overview.exists(): continue
        img=Image.open(overview).convert('RGB'); w,h=img.size
        src={k:str(geom/date/f) for k,f in {'overview':'overview.png','probability':'building_probability.tif','mask':'building_mask.tif','boundary':'building_boundary.tif','gpkg':'building_objects.gpkg'}.items()}
        for i in range(16):
            col,row=i%4,i//4; box=(int(col*w/4),int(row*h/4),int((col+1)*w/4),int((row+1)*h/4)); sid=f'geometry_{date}_{i:02d}'; out=prev/(sid+'.jpg'); img.crop(box).save(out,quality=92)
            records.append({'sample_id':sid,'review_group':'geometry','scene_date':date,'transition':'','window_bounds':list(box),'source_paths':src,'preview_paths':{'tile':'previews/'+out.name},'mandatory':True,'sampling_reason':'fixed_seed_geometry_window'})
    if overlay:
        for transition in TRANSITIONS:
            source=overlay/'per_transition'/transition/'overview.png'
            if not source.exists(): source=overlay/'overlay_overview.png'
            if not source.exists(): continue
            img=Image.open(source).convert('RGB'); w,h=img.size
            for i in range(16):
                col,row=i%4,i//4; box=(int(col*w/4),int(row*h/4),int((col+1)*w/4),int((row+1)*h/4)); sid=f'transition_{transition}_{i:02d}'; out=prev/(sid+'.jpg'); img.crop(box).save(out,quality=92); records.append({'sample_id':sid,'review_group':'transition','scene_date':'','transition':transition,'window_bounds':list(box),'source_paths':{'overview':str(source),'directory':str(source.parent)},'preview_paths':{'tile':'previews/'+out.name},'mandatory':True,'sampling_reason':'fixed_seed_transition_window'})
    if overlay:
        source=overlay/'overlay_overview.png'
        if source.exists():
            img=Image.open(source).convert('RGB'); img.thumbnail((900,600))
            for surface in SURFACE:
                out=prev/('surface_'+surface+'.jpg'); img.save(out,quality=92); records.append({'sample_id':'surface_'+surface,'review_group':'surface_state','scene_date':'','transition':'','window_bounds':[],'source_paths':{'overview':str(source)},'preview_paths':{'tile':'previews/'+out.name},'mandatory':True,'sampling_reason':'fixed_surface_state_review'})
    inv_hash=_hash([p for r in records for p in r['source_paths'].values()]); [r.update(inventory_hash=inv_hash) for r in records]
    payload={'seed':seed,'records':records,'inventory_hash':inv_hash,'created_at':datetime.now(timezone.utc).isoformat()}; (review_dir/'REVIEW_INVENTORY.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (review_dir/'REVIEW_INVENTORY.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['sample_id','review_group','scene_date','transition','window_bounds','mandatory','sampling_reason','inventory_hash']; wri=csv.DictWriter(f,fieldnames=fields); wri.writeheader(); [wri.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields}) for r in records]
    return {'status':'success','sample_count':len(records),'mandatory_count':len(records),'inventory_hash':inv_hash,'review_dir':str(review_dir),'seed':seed}
