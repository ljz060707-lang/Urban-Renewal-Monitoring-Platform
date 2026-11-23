"""XIAN-R5 application V1: serial local inference and explicit candidate products.

Use the existing tools/run_xian_r5_integration.py --application entry point.
No prediction action has a GT path argument or training/cloud implementation.
"""
from __future__ import annotations
import os
os.environ['GDAL_PAM_ENABLED']='NO'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from collections import defaultdict
from datetime import datetime,date
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import yaml
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import shapes,rasterize
from shapely.geometry import shape,mapping,box
from shapely.ops import unary_union
from shapely.affinity import translate
from .application_contract import *
from .application_geometry import *
from .application_temporal import r4_tracks,candidate_presence,pair_events,formal_reconstruction

CODE=Path(__file__).resolve().parents[2]
REPOSITORY_APP = CODE / 'applications' / 'xian_r5'
HOME = REPOSITORY_APP if REPOSITORY_APP.is_dir() else CODE.parent

def config(path):
    with Path(path).open(encoding='utf-8-sig') as f:c=yaml.safe_load(f)
    if c.get('schema_version') not in (SCHEMA,'XIAN_R5_APPLICATION_V1_DRAFT'):raise ValueError('Unsupported application schema')
    safe_id(c['run_name'])
    if c.get('product_level') not in ('research_candidate','qualified_supervision'):raise ValueError('Unknown product level')
    ex=c.get('execution',{})
    if any(ex.get(x,False) for x in ('training_enabled','cloud_enabled','download_models_automatically','overwrite')):raise ValueError('Training/cloud/download/overwrite not supported by this inference application')
    if ex.get('gpu_workers',1)!=1 or ex.get('module_workers_serial',True) is not True:raise ValueError('One serial GPU worker required')
    if c.get('evaluation',{}).get('enabled',False) or c.get('evaluation',{}).get('reference_manifest'):raise ValueError('Evaluation must run in a separate process after sealing; prediction cannot mount GT')
    if c.get('m2_position_area',{}).get('prediction_uses_gt',False):raise ValueError('GT cannot enter prediction')
    if c.get('m3_temporal',{}).get('active_recheck',{}).get('use_gt_prompts',False):raise ValueError('GT prompts forbidden')
    if c.get('inputs',{}).get('prior_observations') or c.get('inputs',{}).get('prior_tracks'):
        raise ValueError('Prior-track import requires source-bound adapter; use native new observation IDs in V1')
    epochs=c['inputs']['epochs']
    if not epochs:raise ValueError('At least one epoch required')
    seen=set()
    for e in epochs:
        eid=safe_id(e['epoch_id'])
        if eid in seen:raise ValueError('Duplicate epoch_id')
        seen.add(eid);interval(e)
        parents=e['parents']
        if not parents:raise ValueError('Epoch needs a parent')
        parent_ids=set()
        for p in parents:
            safe_id(p['parent_id'])
            if p['parent_id'] in parent_ids:raise ValueError('Duplicate parent_id')
            parent_ids.add(p['parent_id'])
            if not (p.get('hr_image') or p.get('hr_building_tif')):raise ValueError('RGB or explicit building TIF required')
            if p.get('hr_building_tif') and not p.get('building_tif_schema'):raise ValueError('Existing TIF needs explicit semantic schema')
            p.setdefault('actual_date',e.get('actual_date'))
            if p.get('actual_date'):date.fromisoformat(p['actual_date'])
        if e.get('lr_building_tif') and e.get('lr_building_role')!='STUDENT_PREDICTION_DIAGNOSTIC_ONLY':raise ValueError('Untrained LR prior cannot enter A')
    m=c.get('m1_recognition',{}).get('aspect_screen',{})
    if m.get('mode','screen_only') not in ('screen_only','exclude_from_candidate_roof'):raise ValueError('Verified semantic exclusion needs an admitted external semantic-evidence adapter')
    if c.get('m3_temporal',{}).get('canonical_geometry',{}).get('union_across_dates_as_canonical',False):raise ValueError('Canonical cannot be cross-date union')
    if c.get('m3_temporal',{}).get('context_mode','retrospective') not in ('retrospective','causal'):raise ValueError('Unknown temporal context')
    if c.get('m3_temporal',{}).get('context_mode')=='causal':raise ValueError('V1 R4 full-sequence runner is retrospective; causal profile not implemented')
    return c

def collect_inputs(c):
    pins={}
    def pin(p):
        if not p:return
        p=Path(p).resolve()
        if not p.is_file():raise FileNotFoundError(p)
        pins[str(p)]=sha(p)
    for e in c['inputs']['epochs']:
        for p in e['parents']:
            for k in ('hr_image','hr_building_tif','source_valid_mask','source_registry_ref','date_source_map'):pin(p.get(k))
        for k in ('lr_image','lr_valid_mask','lr_building_tif','date_source_map','camera_geometry_ref','dsm_or_height_ref','roof_plane_ref'):pin(e.get(k))
    pin(c['grid']['target_lr_grid_template']);pin(c['inputs'].get('source_registry'))
    pin(c.get('m1_recognition',{}).get('aspect_screen',{}).get('rule_manifest'))
    pin(c.get('m3_temporal',{}).get('class_dictionary_path'))
    pin(c['model_routing']['routing_manifest'])
    return pins

def validate(path):
    c=config(path);pins=collect_inputs(c);routes=read(c['model_routing']['routing_manifest']);check_model_route(routes)
    if c.get('execution',{}).get('code_snapshot') and Path(c['execution']['code_snapshot']).resolve()!=CODE.resolve():raise ValueError('Manifest points at another execution snapshot')
    if (HOME/'CODE_FREEZE.json').exists():
        for name,h in read(HOME/'CODE_FREEZE.json')['files'].items():
            if sha(CODE/name)!=h:raise ValueError('Execution snapshot changed after freeze: '+name)
    for p,h in routes['runtime_bindings'].items():
        if sha(p)!=h:raise ValueError('Frozen runtime changed: '+p)
    for e in c['inputs']['epochs']:
        for p in e['parents']:
            source=p.get('hr_image') or p['hr_building_tif']
            with rasterio.open(source) as ds:
                if ds.crs is None or abs(ds.transform.determinant)<1e-15:raise ValueError('Missing/invalid spatial frame')
                if p.get('hr_image') and (ds.count<3 or ds.dtypes[0]!='uint8'):raise ValueError('HR image must be georeferenced uint8 RGB')
            if p.get('hr_building_tif'):
                s=p['building_tif_schema']
                if s.get('kind') not in ('binary','class','instance','probability') or 'nodata' not in s:raise ValueError('Explicit TIF kind and nodata required')
                if s['kind']=='class' and not s.get('foreground_values'):raise ValueError('Class TIF foreground values missing')
                if s['kind']=='probability' and (s.get('comparator')!='>' or s.get('threshold') is None):raise ValueError('Probability threshold/comparator missing')
        if not e.get('lr_image') and c['model_routing'].get('alignment',{}).get('enabled',True):pass
    return c,pins,routes

def plan(path):
    c,pins,routes=validate(path);epochs=c['inputs']['epochs'];pixels=0
    for e in epochs:
        for p in e['parents']:
            with rasterio.open(p.get('hr_image') or p['hr_building_tif']) as ds:pixels+=ds.width*ds.height
    maxpixels=c.get('execution',{}).get('max_input_pixels',20000000)
    if pixels>maxpixels:raise ValueError('Explicit scale limit exceeded; set max_input_pixels for an authorized bounded scope')
    return dict(mode='XIAN_R5_APPLICATION_V1',epochs=len(epochs),parents=sum(len(e['parents']) for e in epochs),input_pixels=pixels,
        modules=['B0_NATIVE_ROOF','E3_FACADE_DIAGNOSTIC','V2_M_CORR_STEP3634','R4_RULE_TRACKING','R5_FORMAL_RECONSTRUCTION','EXACT_FRACTION_EXPORT'],
        gpu_workers=1,module_workers_serial=True,max_wallclock_seconds=c['execution'].get('max_wallclock_seconds') or 1800,
        max_active_requests=c.get('m3_temporal',{}).get('active_recheck',{}).get('max_requests_per_run',200),
        run_scope='ONLY_MANIFEST_FILES',GT_mounted=False,optimizer_updates=0,cloud=False,
        gaps=['FACADE_CLEANING_NOT_VALIDATED','POSITION_CANDIDATE_NOT_CERTIFIED','PHYSICAL_AREA_NULL_WITHOUT_VIEW_GEOMETRY','R4_TRACKING_CANDIDATE_NOT_TRAINED_IDENTITY'],
        dates=[dict(epoch_id=e['epoch_id'],actual_date=e.get('actual_date'),window=e.get('actual_time_window')) for e in epochs],
        input_bindings=pins)

def invoke(action,jobs,routes,work,deadline):
    if not jobs:return
    work=Path(work);work.mkdir(parents=True,exist_ok=True);request=work/(action+'_request.json')
    save(request,dict(action=action,jobs=jobs,routing=routes,receipt=str(work/(action+'_receipt.json'))))
    env=os.environ.copy();env['PYTHONPATH']=str(CODE/'src');env['GDAL_PAM_ENABLED']='NO';env['PYTHONDONTWRITEBYTECODE']='1'
    cmd=[routes['python'],'-B','-X','utf8','-m','xian_r5.application_models',str(request)]
    with (work/(action+'.stdout.log')).open('w',encoding='utf-8') as out,(work/(action+'.stderr.log')).open('w',encoding='utf-8') as err:
        p=subprocess.run(cmd,stdout=out,stderr=err,env=env,cwd=CODE,timeout=max(1,deadline-time.monotonic()))
    if p.returncode:raise RuntimeError(action+' worker failed: '+str(work/(action+'.stderr.log')))

def prepare(c,out):
    packets=[]
    for e in c['inputs']['epochs']:
        root=Path(out)/e['epoch_id'];root.mkdir()
        image_parents=[p for p in e['parents'] if p.get('hr_image')]
        for p in image_parents:
            if p.get('hr_building_tif') and len(image_parents)>1:raise ValueError('Cached multi-parent masks must be pre-fused in their own native schema; no silent first-mask reuse')
        rr=fuse_parent_inputs(image_parents,root) if image_parents else []
        for p in e['parents']:
            if p.get('hr_image'):continue
            with rasterio.open(p['hr_building_tif']) as ds:
                v=(ds.dataset_mask()>0).astype('uint8');d=root/('mask_'+p['parent_id']);d.mkdir();np.save(d/'valid.npy',v)
                rr.append(dict(packet_id='mask_'+p['parent_id'],image=None,valid=str(d/'valid.npy'),shape=list(v.shape),crs=str(ds.crs),affine=list(ds.transform)[:6],parents=[dict(parent_id=p['parent_id'],payload_sha256=sha(p['hr_building_tif']),path=p['hr_building_tif'],native_window=[0,0,ds.width,ds.height],actual_date=p.get('actual_date'),source_acquisition_id=p.get('source_acquisition_id'))],actual_date=p.get('actual_date'),source_acquisition_id=p.get('source_acquisition_id'),source_conflict_pixels=0,hr_building_tif=p['hr_building_tif'],building_tif_schema=p['building_tif_schema']))
        for r in rr:
            r['epoch_id']=e['epoch_id'];r['packet_id']=e['epoch_id']+'_'+r['packet_id'];r['lr_image']=e.get('lr_image');r['lr_valid_mask']=e.get('lr_valid_mask')
            r['date_source_map']=e.get('date_source_map');r['date_codes']=e.get('date_codes',{})
            packets.append(r)
    save(Path(out)/'PACKETS.json',packets)

def decode_mask(packet,out):
    from scipy.ndimage import label
    p=packet['hr_building_tif'];s=packet['building_tif_schema']
    with rasterio.open(p) as ds:
        if ds.shape!=tuple(packet['shape']) or list(ds.transform)[:6]!=packet['affine'] or str(ds.crs)!=packet['crs']:raise ValueError('Existing mask is on another native frame')
        a=ds.read(1);valid=ds.dataset_mask()>0
    if s['nodata'] is not None:valid &= a!=s['nodata']
    if s['kind']=='probability':
        if not np.isfinite(a[valid]).all() or np.any((a[valid]<0)|(a[valid]>1)):raise ValueError('Probability outside [0,1]')
        pvalues=np.where(valid,a,0).astype('float32');mask=(a>s['threshold'])&valid;ids,n=label(mask,np.ones((3,3)))
    elif s['kind']=='instance':ids=np.where(valid,a,0).astype('int32');mask=ids>0;pvalues=None
    else:
        fg=s.get('foreground_values',[1]);mask=np.isin(a,fg)&valid;ids,n=label(mask,np.ones((3,3)));pvalues=None
    rows=[]
    payload=sha(p)
    for gj,i in shapes(ids.astype('int32'),mask=mask,connectivity=8):
        m=ids==int(i);rows.append(dict(candidate_id='INPUT_'+digest([payload,int(i),gj])[:24],geometry_pixel_edge_xy=gj,pixel_area=int(m.sum()),mean_probability_uncalibrated=float(pvalues[m].mean()) if pvalues is not None else None,
            source_edge_partial_risk=bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any()),model_sha256=s.get('model_sha256'),source_actual_date=packet.get('actual_date')))
    np.save(Path(out)/'mask.npy',mask.astype('uint8'));np.save(Path(out)/'valid_override.npy',valid.astype('uint8'))
    if pvalues is not None:np.save(Path(out)/'probability.npy',pvalues)
    save(Path(out)/'instances.json',dict(ROOF_INSTANCE_CANDIDATES=rows,model_called=False,input_semantics=s))

def recognize(c,packets,routes,out,deadline):
    jobs=[]
    for p in packets:
        d=Path(out)/p['packet_id'];d.mkdir()
        if p.get('hr_building_tif'):decode_mask(p,d)
        else:jobs.append(dict(image=p['image'],source_id=p['packet_id'],actual_date=p.get('actual_date'),output=str(d)))
    invoke('roof',jobs,routes,out,deadline)
    if c['model_routing'].get('facade',{}).get('enabled',True):
        fj=[dict(image=p['image'],output=str(Path(out)/p['packet_id'])) for p in packets if p.get('image')]
        try:invoke('facade',fj,routes,out,deadline)
        except Exception as e:
            if c['model_routing'].get('facade',{}).get('missing_model_action')!='mark_unavailable_continue_roof':raise
            save(Path(out)/'FACADE_UNAVAILABLE.json',dict(status='UNAVAILABLE_CONTINUE_ROOF',error=str(e)))

def align(c,packets,m1,routes,out,deadline):
    jobs=[]
    for p in packets:
        d=Path(out)/p['packet_id'];d.mkdir();instances=m1/p['packet_id']/'instances.json'
        if p.get('image') and p.get('lr_image') and c['model_routing'].get('alignment',{}).get('enabled',True):
            jobs.append(dict(image=p['image'],valid=p['valid'],affine=p['affine'],crs=p['crs'],lr_image=p['lr_image'],lr_valid_mask=p.get('lr_valid_mask'),instances=str(instances),output=str(d)))
        else:save(d/'alignment.json',dict(records=[],status='UNKNOWN_MISSING_RGB_OR_LR_OR_DISABLED'))
    invoke('alignment',jobs,routes,out,deadline)

def observations(c,packets,m1,m2):
    with rasterio.open(c['grid']['target_lr_grid_template']) as ds:metric=str(ds.crs)
    aspect_config=c.get('m1_recognition',{}).get('aspect_screen',{});rule=read(aspect_config['rule_manifest']) if aspect_config.get('rule_manifest') else None
    if aspect_config.get('enabled',True) and rule is None:raise ValueError('Aspect threshold must bind an existing versioned rule manifest')
    rows=[]
    for p in packets:
        aff=Affine(*p['affine']);valid=np.load(p['valid']);adir=m2/p['packet_id'];mr=m1/p['packet_id']
        ar={r['candidate_id']:r for r in read(adir/'alignment.json')['records']}
        raw=read(mr/'instances.json')['ROOF_INSTANCE_CANDIDATES']
        if (mr/'valid_override.npy').exists():valid=valid*np.load(mr/'valid_override.npy')
        for r in raw:
            ng=shape(r['geometry_pixel_edge_xy']);g=convert(map_geometry(ng,aff),p['crs'],metric)
            repaired=False
            if not g.is_valid:
                before=g.area;g=g.buffer(0);repaired=True
                if not g.is_valid or g.is_empty or abs(g.area-before)>max(1e-6,abs(before)*1e-8):raise ValueError('Invalid component geometry needs explicit resolution')
            mask=rasterize([(ng,1)],out_shape=valid.shape,transform=Affine.identity(),dtype='uint8')>0
            support=float(valid[mask].mean()) if mask.any() else 0.
            source_ids=[];payloads=[]
            for parent in p['parents']:
                x,y,w,h=parent['native_window']
                if ng.intersection(box(x,y,x+w,y+h)).area>0:source_ids.append(parent['parent_id']);payloads.append(parent['payload_sha256'])
            actual=p.get('actual_date')
            if p.get('date_source_map'):
                with rasterio.open(p['date_source_map']) as ds:
                    date_mask=rasterize([(convert(map_geometry(ng,aff),p['crs'],str(ds.crs)),1)],out_shape=ds.shape,transform=ds.transform,dtype='uint8')>0
                    vals=np.unique(ds.read(1)[date_mask & (ds.dataset_mask()>0)])
                    dates={p['date_codes'].get(str(int(v))) for v in vals};actual=next(iter(dates)) if len(dates)==1 and None not in dates else None
            a=ar.get(r['candidate_id'],{});delta=a.get('shift_model_xy');shift=None
            if delta is not None and a.get('complete_instance_context'):
                cc=ng.centroid;g0=convert(map_geometry(cc,aff),p['crs'],metric);g1=convert(map_geometry(translate(cc,xoff=2*delta[0],yoff=2*delta[1]),aff),p['crs'],metric)
                shift=[g1.x-g0.x,g1.y-g0.y]
            managed=translate(g,*shift) if shift else g
            if abs(g.area-managed.area)>max(1e-6,g.area*1e-8):raise ValueError('Translation did not conserve area')
            epoch=next(e for e in c['inputs']['epochs'] if e['epoch_id']==p['epoch_id'])
            phys_evidence=read(epoch['camera_geometry_ref']) if epoch.get('camera_geometry_ref') else None
            if phys_evidence and phys_evidence.get('source_frame')!=metric:
                phys,status=None,'UNRESOLVED_PHYSICAL_SOURCE_FRAME_MISMATCH'
            else:phys,status=physical(managed,phys_evidence)
            if phys is not None:managed=phys
            asp=aspect(g,r['source_edge_partial_risk'],rule['threshold']) if rule else dict(ratio=None,linear_risk=None,status='DISABLED')
            excluded=bool(aspect_config.get('mode')=='exclude_from_candidate_roof' and asp['linear_risk'])
            uid='OBS_'+digest([p['source_acquisition_id'],sorted(payloads),actual,r.get('model_sha256'),r['geometry_pixel_edge_xy'],p['affine']])[:28]
            rows.append(dict(observation_id=uid,original_candidate_id=r['candidate_id'],epoch_id=p['epoch_id'],actual_date=actual,
                packet_id=p['packet_id'],rgb_pixel_sha256=p.get('rgb_pixel_sha256'),source_parent_ids=source_ids,source_payload_sha256=digest(sorted(payloads)),source_hashes=payloads,source_acquisition_id=p.get('source_acquisition_id'),
                model_checkpoint_sha256=r.get('model_sha256'),mean_probability_uncalibrated=r.get('mean_probability_uncalibrated'),source_valid_fraction=support,source_conflict=p.get('source_conflict_pixels',0)>0 and support<1,
                source_edge_partial_risk=r['source_edge_partial_risk'],geometry_validity_repair=repaired,native_geometry=r['geometry_pixel_edge_xy'],raw_geometry=mapping(g),geometry=mapping(managed),metric_crs=metric,
                shift_east_north_m=shift,position_candidate_status='V2_COMPLETE_INSTANCE_CANDIDATE' if shift else 'RETAIN_RAW_POSITION_UNKNOWN',position_sigma_m=None,qualified_position=None,physical_id=None,
                area_raw_map_m2=g.area,area_after_translation_m2=g.area,area_on_reference_plane_m2=managed.area,area_physical_rectified_m2=phys.area if phys is not None else None,
                area_roof_surface_m2=None,area_ground_footprint_m2=None,physical_area_status=status,aspect=asp,excluded=excluded,formal_weight=0,
                transformation_chain=dict(source_frame=p['crs'],target_frame=metric,native_affine=p['affine'],resize_model_to_native=2.,V2_total_translation=shift,physical_transform=phys_evidence if phys is not None else None,compose_once=True)))
    # Same payload/date aliases are observations of the same input, never extra buildings.
    unique={}
    for r in rows:
        if r['observation_id'] not in unique:unique[r['observation_id']]=r
        else:unique[r['observation_id']]['source_parent_ids']=sorted(set(unique[r['observation_id']]['source_parent_ids']+r['source_parent_ids']))
    return list(unique.values())

def active_rechecks(c,obs,tracks,conflicts,packets,routes,out,deadline):
    budget=c.get('m3_temporal',{}).get('active_recheck',{});limit=budget.get('max_requests_per_run',200)
    enabled=budget.get('enabled',True);ledger=[];jobs=[];meta=[];byid={r['observation_id']:r for r in obs}
    cap=budget.get('max_per_track_epoch',1)
    if cap not in (0,1):raise ValueError('V1 supports max_per_track_epoch 0 or 1')
    for track in tracks:
        rr=[byid[x] for x in track['observation_ids']]
        for e in c['inputs']['epochs']:
            here=[r for r in rr if r['epoch_id']==e['epoch_id']]
            trigger='MISSING_EPOCH' if not here else ('SPLIT_MERGE_OR_IDENTITY_CONFLICT' if any(r['observation_id'] in conflicts for r in here) else ('INCOMPLETE_BOUNDARY' if any(r['source_edge_partial_risk'] for r in here) else None))
            if trigger is None:continue
            rec=dict(track_id_candidate=track['track_id_candidate'],epoch_id=e['epoch_id'],trigger=trigger,prompt_source_observation_id=track['canonical_observation_id'],uses_future_evidence=any(r['epoch_id']!=e['epoch_id'] for r in rr),GT_prompt=False,reliable_zero_candidate=False)
            pp=[p for p in packets if p['epoch_id']==e['epoch_id'] and p.get('image')]
            if not enabled or cap==0:rec['status']='DISABLED'
            elif len(jobs)>=limit:rec['status']='BUDGET_EXHAUSTED_UNKNOWN'
            elif not pp or not e.get('actual_date'):rec['status']='UNKNOWN_MISSING_RGB_OR_DATE'
            else:
                geometry=shape(track['geometry']);choices=[]
                for p in pp:
                    g=map_geometry(convert(geometry,rr[0]['metric_crs'],p['crs']),~Affine(*p['affine']));bounds=g.bounds
                    if bounds[2]>0 and bounds[3]>0 and bounds[0]<p['shape'][1] and bounds[1]<p['shape'][0]:choices.append((p,g))
                if not choices:rec['status']='UNKNOWN_OUTSIDE_SOURCE'
                else:
                    p,g=choices[0];h,w=p['shape'];size=512
                    cx,cy=g.centroid.coords[0];x=max(0,min(max(0,w-size),math.floor(cx-size/2)));y=max(0,min(max(0,h-size),math.floor(cy-size/2)))
                    local=translate(g,xoff=-x,yoff=-y);valid=np.load(p['valid'])[y:y+size,x:x+size]
                    supportmask=rasterize([(local,1)],out_shape=valid.shape,transform=Affine.identity(),dtype='uint8')>0
                    full=(g.bounds[0]>=x and g.bounds[1]>=y and g.bounds[2]<=x+size and g.bounds[3]<=y+size)
                    sf=float(valid[supportmask].mean()) if supportmask.any() else 0
                    if not full or sf<.75:rec.update(status='UNKNOWN_INSUFFICIENT_SEARCH_SUPPORT',valid_fraction=sf)
                    else:
                        d=Path(out)/('request_%04d'%len(jobs));d.mkdir();rgb=np.asarray(Image.open(p['image']).convert('RGB'))[y:y+size,x:x+size];Image.fromarray(rgb).save(d/'search_RGB.png')
                        jobs.append(dict(image=str(d/'search_RGB.png'),output=str(d),actual_date=e['actual_date'],source_id=p['packet_id']+':active'))
                        rec.update(status='MODEL_REQUEST_EXECUTED',request_id=d.name,valid_fraction=sf,packet_id=p['packet_id'],native_window=[x,y,rgb.shape[1],rgb.shape[0]])
                        meta.append((rec,p,local,supportmask,valid,d))
            ledger.append(rec)
    invoke('roof',jobs,routes,Path(out)/'worker',deadline)
    results={};proposals=[];recovered=[]
    for rec,p,local,supportmask,valid,d in meta:
        prob=np.load(d/'probability.npy');found=read(d/'instances.json')['ROOF_INSTANCE_CANDIDATES']
        maxprob=float(prob[supportmask].max()) if supportmask.any() else None
        current=[r for r in obs if r['epoch_id']==rec['epoch_id']]
        target=shape(next(t for t in tracks if t['track_id_candidate']==rec['track_id_candidate'])['geometry'])
        competitor=any(shape(r['geometry']).centroid.distance(target.centroid)<=min(80.,max(20.,1.5*math.sqrt(max(target.area,shape(r['geometry']).area)))) for r in current)
        candidate_intersect=any(shape(r['geometry_pixel_edge_xy']).intersection(local).area>0 for r in found)
        rec.update(max_probability=maxprob,recheck_candidate_count=len(found),identity_competitor=competitor,candidate_intersection=candidate_intersect,
            reliable_zero_candidate=bool(maxprob is not None and maxprob<.5 and rec['valid_fraction']>=.75 and not competitor and not candidate_intersect),scientific_level='MODEL_RESEARCH_CANDIDATE_NOT_REVIEWED_ABSENCE')
        results[(rec['track_id_candidate'],rec['epoch_id'])]=rec
        for r in found:proposals.append(dict(request_id=rec['request_id'],track_id_candidate=rec['track_id_candidate'],epoch_id=rec['epoch_id'],candidate=r,interpretation='ACTIVE_SOURCE_CROP_OBSERVATION_VERSION_NOT_AUTOMATIC_IDENTITY_PROMOTION'))
        # Register an actual recovered source observation only when the source
        # crop has a unique overlapping complete candidate and the original
        # epoch had no corresponding observation. Identity is recomputed later.
        matches=[r for r in found if not r['source_edge_partial_risk'] and shape(r['geometry_pixel_edge_xy']).intersection(local).area/max(1e-12,shape(r['geometry_pixel_edge_xy']).union(local).area)>=.35]
        if rec['trigger']=='MISSING_EPOCH' and len(matches)==1 and not competitor:
            r=matches[0];x,y,w,h=rec['native_window'];ng=translate(shape(r['geometry_pixel_edge_xy']),xoff=x,yoff=y)
            metric_crs=obs[0]['metric_crs'];g=convert(map_geometry(ng,Affine(*p['affine'])),p['crs'],metric_crs)
            if g.is_valid and not g.is_empty:
                seed=dict(next(x for x in obs if x['observation_id']==rec['prompt_source_observation_id']))
                uid='OBS_'+digest([p['packet_id'],p.get('actual_date'),r['model_sha256'],mapping(ng),'ACTIVE_VERSION_1'])[:28]
                seed.update(observation_id=uid,original_candidate_id=r['candidate_id'],epoch_id=p['epoch_id'],actual_date=p.get('actual_date'),packet_id=p['packet_id'],rgb_pixel_sha256=p.get('rgb_pixel_sha256'),
                    source_parent_ids=[v['parent_id'] for v in p['parents']],source_payload_sha256=sha(d/'search_RGB.png'),source_hashes=[v['payload_sha256'] for v in p['parents']],source_acquisition_id=p.get('source_acquisition_id'),
                    model_checkpoint_sha256=r['model_sha256'],mean_probability_uncalibrated=r['mean_probability_uncalibrated'],source_valid_fraction=rec['valid_fraction'],source_edge_partial_risk=False,source_conflict=False,
                    native_geometry=mapping(ng),raw_geometry=mapping(g),geometry=mapping(g),shift_east_north_m=None,position_candidate_status='ACTIVE_RECHECK_RETAIN_RAW_POSITION_UNKNOWN',
                    area_raw_map_m2=g.area,area_after_translation_m2=g.area,area_on_reference_plane_m2=g.area,area_physical_rectified_m2=None,physical_area_status='UNRESOLVED_NO_VIEW_GEOMETRY',
                    aspect=aspect(g,False,read(c['m1_recognition']['aspect_screen']['rule_manifest'])['threshold']),excluded=False,
                    active_recheck_lineage=dict(request_id=rec['request_id'],prompt_observation_id=rec['prompt_source_observation_id'],original_track_candidate=rec['track_id_candidate'],new_interpretation=True),
                    transformation_chain=dict(source_frame=p['crs'],target_frame=metric_crs,native_affine=p['affine'],V2_total_translation=None,physical_transform=None,compose_once=True))
                recovered.append(seed);rec['recovered_observation_id']=uid
    lines(Path(out)/'ACTIVE_SEARCH_LEDGER.jsonl',ledger);lines(Path(out)/'ACTIVE_RECHECK_PROPOSALS.jsonl',proposals)
    lines(Path(out)/'ACTIVE_RECOVERED_OBSERVATIONS.jsonl',recovered)
    return results,ledger,recovered

def support_geometry(packet):
    v=np.load(packet['valid']);gg=[shape(g) for g,val in shapes(v,mask=v>0,transform=Affine(*packet['affine'])) if val>0]
    return unary_union(gg)

def product_exports(c,packets,obs,tracks,presence,active,edges,conflicts,out):
    out=Path(out);metric=obs[0]['metric_crs'] if obs else None
    if metric is None:
        with rasterio.open(c['grid']['target_lr_grid_template']) as ds:metric=str(ds.crs)
    support_by_epoch={e['epoch_id']:unary_union([convert(support_geometry(p),p['crs'],metric) for p in packets if p['epoch_id']==e['epoch_id']]) for e in c['inputs']['epochs']}
    allsupport=unary_union(list(support_by_epoch.values()));grid=metric_grid(c['grid']['target_lr_grid_template'],allsupport,metric)
    fine=grid.nested(c['grid'].get('map_fine_grid',{}).get('subdivision_of_lr_axes',10));reports=out/'reports';reports.mkdir()
    temporal=out/'temporal';temporal.mkdir();lines(temporal/'OBSERVATIONS.jsonl',obs);lines(temporal/'TRACKS_CANDIDATE.jsonl',tracks);lines(temporal/'CANONICAL_GEOMETRY.jsonl',tracks)
    lines(temporal/'PRESENCE_CANDIDATE.jsonl',presence);lines(temporal/'ACTIVE_SEARCH_LEDGER.jsonl',active);lines(temporal/'TEMPORAL_CONFLICTS.jsonl',[dict(observation_id=x,reason='R4_COMPETITOR_OR_SPLIT_MERGE') for x in sorted(conflicts)])
    lines(temporal/'R4_EDGES.jsonl',edges);formal_reconstruction(obs,temporal)
    singles=[];intervals=[];links=[]
    for epoch in c['inputs']['epochs']:
        eid=epoch['epoch_id'];d=out/'single_date'/eid;d.mkdir(parents=True);rr=[r for r in obs if r['epoch_id']==eid];kept=[r for r in rr if not r['excluded']]
        raw=[shape(r['raw_geometry']) for r in rr];managed=[shape(r['geometry']) for r in kept];pid=digest([eid,rr])
        vector(d/'ROOF_OBSERVED.gpkg',[dict(**{k:v for k,v in r.items() if k!='geometry'},geometry=r['raw_geometry']) for r in rr],metric)
        vector(d/'ROOF_MANAGED_CANDIDATE.gpkg',kept,metric);vector(d/'ROOF_REGISTERED_QUALIFIED.gpkg',[],metric);vector(d/'EXCLUDED_CANDIDATES.gpkg',[r for r in rr if r['excluded']],metric)
        for p in [p for p in packets if p['epoch_id']==eid]:
            dd=d/'native'/p['packet_id'];dd.mkdir(parents=True)
            pg=Grid(Affine(*p['affine']),p['shape'][1],p['shape'][0],p['crs']);pr=[r for r in rr if r['packet_id']==p['packet_id']]
            write_tif(dd/'ROOF_OBSERVED_NATIVE.tif',burn([map_geometry(shape(r['native_geometry']),pg.affine) for r in pr],pg),pg,['observed_roof_candidate'],pid)
        write_tif(d/'ROOF_MANAGED_NATIVE_OR_MAP.tif',burn(managed,fine),fine,['managed_roof_candidate'],pid)
        write_tif(d/'ROOF_INSTANCE_ID.tif',burn(managed,fine,range(1,len(managed)+1),'uint32'),fine,['run_local_instance_index_see_manifest'],pid)
        frac=exact(grid,{'ROOF':unary_union(managed)},support_by_epoch[eid]);sf=frac['valid'].astype('float32')
        write_tif(d/'FRACTION_10M.tif',frac['fractions'].astype('float32'),grid,['roof_candidate_fraction_full_cell_denominator'],pid)
        for name,data in [('SOURCE_SUPPORT_10M',sf),('POSITION_UNKNOWN_10M',sf),('QUALITY_10M',(sf>0).astype('uint8')),('SUPERVISION_WEIGHT_10M',np.zeros_like(sf))]:write_tif(d/(name+'.tif'),data,grid,[name],pid)
        # Missing facade is Unknown=255; B0 never has facade subtracted.
        risk=np.full((fine.height,fine.width),255,np.uint8);facade_geoms=[];facade_support=[]
        for packet in [p for p in packets if p['epoch_id']==eid]:
            f=Path(packet.get('m1_directory','MISSING'))/'facade_probability.npy'
            if f.exists():
                mask=(np.load(f)>.5).astype('uint8');pa=Affine(*packet['affine'])
                facade_geoms.extend(convert(shape(g),packet['crs'],metric) for g,v in shapes(mask,mask=mask>0,transform=pa) if v>0)
                facade_support.append(convert(support_geometry(packet),packet['crs'],metric))
        if facade_support:
            covered=burn(facade_support,fine)>0;risk[covered]=0;risk[(burn(facade_geoms,fine)>0)&covered]=1
        write_tif(d/'FACADE_OR_NONROOF_RISK.tif',risk,fine,['unknown_255_see_native_facade_probability'],pid,255)
        table(d/'BUILDING_AREA.csv',rr,['observation_id','actual_date','area_raw_map_m2','area_after_translation_m2','area_on_reference_plane_m2','area_physical_rectified_m2','area_roof_surface_m2','area_ground_footprint_m2','position_candidate_status','physical_area_status','source_parent_ids','excluded'])
        table(d/'ASPECT_SCREEN.csv',[dict(observation_id=r['observation_id'],**r['aspect'],excluded=r['excluded']) for r in rr],['observation_id','ratio','long_axis_m','short_axis_m','linear_risk','status','excluded'])
        lines(d/'POSITION_AREA_LINEAGE.jsonl',rr)
        save(d/'ACTUAL_DATE_SOURCE_MAP.json',dict(actual_date=epoch.get('actual_date'),actual_time_window=epoch.get('actual_time_window'),source_date_raster=epoch.get('date_source_map'),source_date_codes=epoch.get('date_codes'),parents=[p['parents'] for p in packets if p['epoch_id']==eid]))
        union=unary_union(managed);summary=dict(epoch_id=eid,actual_date=epoch.get('actual_date'),instances=len(rr),retained_instances=len(kept),object_sum_m2=sum(g.area for g in managed),union_m2=union.area,within_source_support_union_m2=union.intersection(support_by_epoch[eid]).area,exact_fraction_integral_error_m2=frac['area_conservation_error_m2'],physical_area_rectified_instances=sum(r['area_physical_rectified_m2'] is not None for r in rr),qualified_instances=0)
        singles.append(summary);save(d/'PRODUCT_MANIFEST.json',dict(product_id=pid,instance_id_index={str(i):r['observation_id'] for i,r in enumerate(kept,1)},summary=summary,managed_policy='ONLY_COMPLETE_V2_NUMERIC_ELSE_RETAIN_RAW_UNKNOWN',formal_weight=0,files=[x.name for x in d.iterdir()]))
        # Render source pixels and actual candidate vectors, never invented imagery.
        packet=next((p for p in packets if p['epoch_id']==eid and p.get('image')),None)
        if packet:
            from PIL import ImageDraw
            im=Image.open(packet['image']).convert('RGB');draw=ImageDraw.Draw(im);pa=Affine(*packet['affine'])
            for r in rr:
                for field,color in [('raw_geometry',(50,255,80)),('geometry',(0,210,255))]:
                    gg=map_geometry(convert(shape(r[field]),metric,packet['crs']),~pa)
                    polygons=list(gg.geoms) if gg.geom_type=='MultiPolygon' else [gg]
                    for polygon in polygons:
                        if polygon.geom_type=='Polygon':draw.line(list(polygon.exterior.coords),fill=color,width=2)
                if r['shift_east_north_m']:
                    aa=map_geometry(convert(shape(r['raw_geometry']).centroid,metric,packet['crs']),~pa);bb=map_geometry(convert(shape(r['geometry']).centroid,metric,packet['crs']),~pa)
                    draw.line([(aa.x,aa.y),(bb.x,bb.y)],fill=(255,230,50),width=2)
            im.thumbnail((1100,800));im.save(d/'SOURCE_AND_CANDIDATE_PREVIEW.png')
        links.append((eid,'../single_date/'+eid+'/PRODUCT_MANIFEST.json'))
    epochs=comparison_epochs(c['inputs']['epochs']) if c.get('m3_temporal',{}).get('enabled',True) else [];classdict=read(c['m3_temporal']['class_dictionary_path'])
    for a,b in zip(epochs,epochs[1:]):
        pair=a['epoch_id']+'__'+b['epoch_id'];d=out/'pairs'/pair;d.mkdir(parents=True)
        events,regions=pair_events(obs,tracks,presence,a,b);pid=digest([pair,events]);support=support_by_epoch[a['epoch_id']].intersection(support_by_epoch[b['epoch_id']])
        chans={k:regions[k] for k in ('STABLE','ADDITION','DEMOLITION')};fraction=exact(grid,chans,support)
        native=np.stack([burn([regions[k].intersection(support)],fine) for k in chans]);count=native.sum(0);classes=np.full(count.shape,classdict['UNRESOLVED'],np.uint8)
        for i,k in enumerate(chans):classes[(native[i]>0)&(count==1)]=classdict[k]
        classes[count>1]=classdict['CONFLICT']
        write_tif(d/'CHANGE_THREE_STATE.tif',classes,fine,['candidate_state_not_presence_not_formal_supervision'],pid)
        rgb=np.full((*classes.shape,3),60,np.uint8)
        for label,color in [('STABLE',(55,185,110)),('ADDITION',(35,145,250)),('DEMOLITION',(245,90,70)),('CONFLICT',(235,70,230))]:rgb[classes==classdict[label]]=color
        Image.fromarray(rgb).save(d/'THREE_STATE_PREVIEW.png')
        write_tif(d/'CHANGE_EVENT_CHANNELS_NATIVE_OR_MAP.tif',native,fine,list(chans),pid)
        write_tif(d/'CHANGE_FRACTION_10M.tif',fraction['fractions'].astype('float32'),grid,list(chans),pid)
        conflict=unary_union([regions[x].intersection(regions[y]) for x,y in [('STABLE','ADDITION'),('STABLE','DEMOLITION'),('ADDITION','DEMOLITION')]])
        extra=exact(grid,{'CONFLICT':conflict},support)['fractions'][0]
        unknown_geom=support.difference(unary_union(list(chans.values())));unknown=exact(grid,{'UNKNOWN':unknown_geom},support)['fractions'][0]+(1-fraction['valid'])
        for n,arr in [('VALIDITY',fraction['valid']),('UNKNOWN',unknown),('CONFLICT',extra),('WEIGHT',np.zeros_like(extra))]:write_tif(d/('CHANGE_'+n+'_10M.tif'),arr.astype('float32'),grid,[n],pid)
        vector(d/'CHANGE_OBJECTS.gpkg',events,metric)
        areas=[dict(pair_id=pair,state=k,object_sum_m2=sum(e['area_m2'] for e in events if e['state_candidate']==k),class_union_m2=regions[k].area,within_pair_support_union_m2=regions[k].intersection(support).area,level='RESEARCH_CANDIDATE',physical_area_rectified=False) for k in regions]
        table(d/'CHANGE_AREAS.csv',areas,['pair_id','state','object_sum_m2','class_union_m2','within_pair_support_union_m2','level','physical_area_rectified']);lines(d/'EVENTS.jsonl',events)
        summary=dict(pair_id=pair,actual_dates=[a.get('actual_date'),b.get('actual_date')],temporal_comparable=valid_pair(a,b),events=len(events),candidate_state_counts={k:sum(e['state_candidate']==k for e in events) for k in regions},area_by_class=areas,
            exact_fraction_integral_error_m2=fraction['area_conservation_error_m2'],inventory_balance='NOT_IDENTIFIED_INCOMPLETE_IDENTITY_OR_ABSENCE',formal_weight=0,qualified_events=0,product_id=pid,class_dictionary=classdict,multi_label_event_sum_can_exceed_one=True)
        save(d/'PAIR_SUMMARY.json',summary);save(d/'ACTUAL_PAIR_DATE_MAP.json',dict(before=a.get('actual_date'),after=b.get('actual_date'),before_local_map=a.get('date_source_map'),after_local_map=b.get('date_source_map'),event_interval_only=True))
        intervals.extend(areas);links.append((pair,'../pairs/'+pair+'/PAIR_SUMMARY.json'))
    table(reports/'SINGLE_DATE_AREA_SUMMARY.csv',singles,['epoch_id','actual_date','instances','retained_instances','object_sum_m2','union_m2','within_source_support_union_m2','physical_area_rectified_instances','qualified_instances','exact_fraction_integral_error_m2'])
    table(reports/'INTERVAL_CHANGE_AREA_SUMMARY.csv',intervals,['pair_id','state','object_sum_m2','class_union_m2','within_pair_support_union_m2','level'])
    return dict(single_date=singles,intervals=intervals,grid=dict(crs=grid.crs,affine=list(grid.affine)[:6],width=grid.width,height=grid.height),links=links)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application',choices=['validate','plan','run','resume','export','report'],required=True)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--export-to',type=Path)
    args=parser.parse_args(argv)
    if args.application in ('report','export'):
        if args.output is None:parser.error('--output sealed run required')
        r=read(args.output/'FINAL_RECEIPT.json')
        for p,h in r['output_sha256'].items():
            if sha(args.output/p)!=h:raise ValueError('Sealed export changed: '+p)
        if args.application=='report':print(json.dumps(r,ensure_ascii=False,indent=2));return
        if args.export_to is None or args.export_to.exists():parser.error('--export-to NEW_DIRECTORY required')
        if args.output.resolve() in args.export_to.resolve().parents:raise ValueError('Export cannot be inside source')
        shutil.copytree(args.output,args.export_to);print('EXPORTED_WITHOUT_MODEL_CALLS',args.export_to);return
    if args.manifest is None:parser.error('--manifest required')
    if args.application=='validate':
        c,pins,routes=validate(args.manifest);print(json.dumps(dict(valid=True,inputs=len(pins),epochs=len(c['inputs']['epochs'])),ensure_ascii=False));return
    if args.application=='plan':print(json.dumps(plan(args.manifest),ensure_ascii=False,indent=2));return
    run_application(args.manifest,args.output,args.application=='resume')

def run_application(manifest,output=None,resume=False):
    c,pins,routes=validate(manifest);p=plan(manifest);output=Path(output) if output else Path(c['output_root'])/c['run_name']
    if any(output.resolve()==Path(x).resolve() or output.resolve() in Path(x).resolve().parents for x in pins):raise ValueError('Output contains or overlaps input')
    codefiles={str(x.relative_to(CODE)):sha(x) for x in CODE.rglob('*.py')}
    identity=digest([c,pins,routes,codefiles]);deadline=time.monotonic()+p['max_wallclock_seconds']
    if output.exists() and not resume:raise FileExistsError('Use resume or a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'FINAL_RECEIPT.json').exists():
        old=read(output/'FINAL_RECEIPT.json')
        if old['run_identity']!=identity:raise ValueError('Resume input/code/routing changed; use a new run')
        for n,h in old['output_sha256'].items():
            if sha(output/n)!=h:raise ValueError('Sealed output corrupted: '+n)
        print(json.dumps(dict(APPLICATION_REPLAY_PASS=True,reused_sealed_run=True,model_forwards=0,output=str(output)),ensure_ascii=False));return
    lock=output/'RUN_LOCK.json'
    if lock.exists():
        # Stale locks need explicit resume and a dead owning process.
        previous=read(lock);alive=True
        if os.name=='nt':
            import ctypes
            handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,previous['pid']);alive=bool(handle)
            if handle:ctypes.windll.kernel32.CloseHandle(handle)
        if not resume or alive:raise RuntimeError('Run is locked by process '+str(previous['pid']))
        lock.rename(output/('STALE_LOCK_'+str(previous['pid'])+'_'+str(time.time_ns())+'.json'))
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),run_identity=identity),f)
    stages=Stages(output,identity,deadline)
    try:
        if (output/'RUN_IDENTITY.json').exists() and read(output/'RUN_IDENTITY.json')['run_identity']!=identity:raise ValueError('Changed resume dependency requires new run')
        save(output/'RUN_IDENTITY.json',dict(run_identity=identity,code_sha256=codefiles,input_sha256=pins))
        save(output/'INPUT_MANIFEST.json',c);save(output/'MODEL_ROUTING_MANIFEST.json',routes);save(output/'PLAN.json',p)
        shutil.copyfile(manifest,output/'RUN_CONFIG.yaml')
        print('PREPARE_SOURCE',flush=True)
        src=stages.output(stages.run('01_sources',pins,lambda d:prepare(c,d)));packets=read(src/'PACKETS.json')
        print('M1_RECOGNITION',flush=True)
        m1=stages.output(stages.run('02_recognition',packets,lambda d:recognize(c,packets,routes,d,deadline)))
        for packet in packets:packet['m1_directory']=str(m1/packet['packet_id'])
        print('M2_ALIGNMENT',flush=True)
        m2=stages.output(stages.run('03_alignment',sha(m1.parent/'COMMIT.json'),lambda d:align(c,packets,m1,routes,d,deadline)))
        obs=observations(c,packets,m1,m2);epochs=sorted(c['inputs']['epochs'],key=lambda e:((interval(e) or (date.max,date.max))[0],e['epoch_id']));c['inputs']['epochs']=epochs
        def temporal_stage(d):
            if not c.get('m3_temporal',{}).get('enabled',True):
                save(d/'STATE.json',dict(observations=obs,tracks=[],edges=[],conflicts=[],presence=[],active=[]));return
            tracks,edges,ids,conflicts=r4_tracks(obs,epochs,d/'INITIAL_R4',CODE)
            rechecks,ledger,recovered=active_rechecks(c,obs,tracks,conflicts,packets,routes,d,deadline)
            final_obs=obs+recovered
            if recovered:
                oldtracks=tracks;tracks,edges,ids,conflicts=r4_tracks(final_obs,epochs,d/'RECHECK_R4',CODE)
                remapped={};versions=[]
                for t in oldtracks:
                    updated=[q for q in tracks if set(t['observation_ids'])&set(q['observation_ids'])]
                    for q in updated:
                        versions.append(dict(before=t['track_id_candidate'],after=q['track_id_candidate'],basis='R4_AFTER_ACTIVE_SOURCE_OBSERVATION'))
                        for (key,eid),v in rechecks.items():
                            if key==t['track_id_candidate']:remapped[(q['track_id_candidate'],eid)]=v
                rechecks=remapped;lines(d/'TRACK_VERSION_LINEAGE.jsonl',versions)
            presence=candidate_presence(final_obs,tracks,epochs,rechecks,conflicts)
            save(d/'STATE.json',dict(observations=final_obs,tracks=tracks,edges=edges,conflicts=sorted(conflicts),presence=presence,active=ledger))
        print('M3_R4_ACTIVE_TRACKING',flush=True)
        tmp=stages.output(stages.run('04_temporal',digest(obs),temporal_stage));state=read(tmp/'STATE.json')
        obs=state['observations']
        def exports(d):
            result=product_exports(c,packets,obs,state['tracks'],state['presence'],state['active'],state['edges'],state['conflicts'],d)
            save(d/'EXPORT_SUMMARY.json',result)
        print('PRODUCT_EXPORT',flush=True)
        products=stages.output(stages.run('05_products',sha(tmp.parent/'COMMIT.json'),exports));summary=read(products/'EXPORT_SUMMARY.json')
        # Publish committed products only. Existing partial publication can be
        # checked/replaced from the same sealed stage without any model calls.
        for name in ('single_date','pairs','temporal','reports'):
            source=products/name
            if not source.exists():continue
            shutil.copytree(source,output/name,dirs_exist_ok=True)
        source_dir=output/'SOURCE_REGISTRY';source_dir.mkdir(exist_ok=True);save(source_dir/'SOURCE_PACKETS.json',packets)
        save(output/'IMPLEMENTATION_MAP.json',read(HOME/'IMPLEMENTATION_MAP.json'))
        # Include actual facade diagnostics with native grid metadata.
        for packet in packets:
            f=m1/packet['packet_id']/'facade_probability.npy'
            if f.exists():
                pg=Grid(Affine(*packet['affine']),packet['shape'][1],packet['shape'][0],packet['crs'])
                dest=output/'single_date'/packet['epoch_id']/'native'/packet['packet_id'];dest.mkdir(parents=True,exist_ok=True)
                write_tif(dest/'E3_FACADE_PROBABILITY_DIAGNOSTIC.tif',np.load(f),pg,['uncalibrated_projected_facade_aux_probability'],identity)
        receipts=[read(directory/path) for directory in (output/'stages').iterdir() if (directory/'COMMIT.json').exists()
                  for path in read(directory/'COMMIT.json')['files'] if path.endswith('_receipt.json')]
        forwards={k:sum(r.get('actual_forwards',0) for r in receipts if r.get('action')==k) for k in ('roof','facade','alignment')}
        receipt=dict(APP_EXECUTABLE='YES',MODEL_ROUTING_FROZEN=True,SINGLE_DATE_PIPELINE_EXECUTABLE=True,INSTANCE_INFORMATION_AVAILABLE=True,
            FACADE_GOVERNANCE_LEVEL='E3_DIAGNOSTIC_ONLY' if forwards['facade'] else 'UNAVAILABLE_OR_DISABLED',ASPECT_FILTER_MODE_AND_EFFECT=c['m1_recognition']['aspect_screen']['mode'],
            ALIGNMENT_CANDIDATE_AVAILABLE=any(r['shift_east_north_m'] is not None for r in obs),QUALIFIED_ALIGNMENT_AVAILABLE=False,
            PHYSICAL_AREA_RECOVERY_AVAILABLE=dict(operator='VERIFIED_PLANAR_AFFINE_IMPLEMENTED',real_instances=sum(r['area_physical_rectified_m2'] is not None for r in obs)),
            ACTIVE_TRACKING_EXECUTED=any(r['status']=='MODEL_REQUEST_EXECUTED' for r in state['active']),MULTITEMPORAL_IDENTITY_LEVEL='R4_CANDIDATE_ONLY_R5_FORMAL_UNRESOLVED',
            PRESENCE_LAYER_EXECUTED=True,ADJACENT_PAIR_STATE_TIFS_EXPORTED=max(0,len(comparison_epochs(epochs))-1),THREE_CLASS_AREA_TABLES_EXPORTED=max(0,len(comparison_epochs(epochs))-1),
            REAL_MULTIPARENT_TEST='EXECUTED' if any(len(e['parents'])>1 for e in epochs) else 'NOT_RUN_SINGLE_PARENT',
            REAL_MULTITEMPORAL_TEST='EXECUTED_SOURCE_DATE_LEDGER_BOUND_NOT_INDEPENDENT_CHANGE_VALIDATION' if len({e.get('actual_date') for e in epochs if e.get('actual_date')})>=2 and c.get('evidence_role')!='SYNTHETIC_FIXTURE' else 'REAL_MULTITEMPORAL_NOT_VERIFIED',
            APPLICATION_REPLAY_PASS='SEE_ACCEPTANCE_TESTS',INDEPENDENT_SCIENTIFIC_VALIDATION='NOT_ESTABLISHED',GT10M_ADMISSION='NO',PRODUCTION_ADMISSION='NO',
            model_forward_counts=forwards,optimizer_updates=0,GT_reads=0,cloud_operations=0,run_identity=identity,stage_reused=stages.reused,stage_executed=stages.executed,
            observations=len(obs),candidate_tracks=len(state['tracks']),source_epochs=len(epochs),source_parents=sum(len(e['parents']) for e in epochs),summary=summary,
            known_limitations=['No trained TBIA weights in selected runtime','R4 rule tracking has no new independent identity accuracy','Active crop proposals remain explicit new interpretations; no automatic physical identity assignment','Physical camera/roof-plane correction needs frame-bound evidence','Qualified weight remains zero without admitted evidence'])
        roles=HOME/'MODEL_AND_DATA_ROLES.md';shutil.copyfile(roles,output/'reports/MODEL_AND_DATA_ROLES.md')
        import html
        index='<html><meta charset="utf-8"><title>XIAN-R5 application results</title><body style="font:16px system-ui;max-width:1100px;margin:40px auto"><h1>XIAN-R5 研究候选结果</h1><p>真实影像本地推理；候选身份和位移未经独立认证。正式训练权重为0。</p>'
        index+='<pre>'+html.escape(json.dumps({k:v for k,v in receipt.items() if k not in ('summary','run_identity')},ensure_ascii=False,indent=2))+'</pre><ul>'
        index+=''.join('<li><a href="'+html.escape(url)+'">'+html.escape(label)+'</a></li>' for label,url in summary['links'])+'</ul></body></html>'
        previews=''.join('<h2>'+html.escape(e['epoch_id'])+'</h2><img style="max-width:100%" src="../single_date/'+e['epoch_id']+'/SOURCE_AND_CANDIDATE_PREVIEW.png">' for e in epochs if (output/'single_date'/e['epoch_id']/'SOURCE_AND_CANDIDATE_PREVIEW.png').exists())
        previews+=''.join('<h2>'+d.name+'</h2><p>Green: Stable; Blue: Addition; Red: Demolition; Grey: Unknown; Magenta: Conflict.</p><img style="max-width:100%" src="../pairs/'+d.name+'/THREE_STATE_PREVIEW.png">' for d in sorted((output/'pairs').glob('*')) if d.is_dir())
        index=index.replace('</body>',previews+'</body>')
        (output/'reports/index.html').write_text(index,encoding='utf-8')
        (output/'reports/APPLICATION_ACCEPTANCE.md').write_text('# Application execution\n\n'+json.dumps({k:v for k,v in receipt.items() if k!='summary'},ensure_ascii=False,indent=2)+'\n\nSee delivery tests and explicit limitations; execution is not scientific admission.\n',encoding='utf-8')
        # Verify every dependency again before final publication.
        for path,h in pins.items():
            if sha(path)!=h:raise ValueError('Input changed during execution: '+path)
        for path,h in routes['runtime_bindings'].items():
            if sha(path)!=h:raise ValueError('Runtime changed during execution: '+path)
        # Stable fingerprint excludes GPKG byte timestamps; rows/geometries and raster bytes are bound.
        fingerprint=dict(observations=digest(obs),tracks=digest(state['tracks']),presence=digest(state['presence']),summary=digest(summary),
            rasters={str(f.relative_to(output)):sha(f) for f in output.glob('single_date/**/*.tif')},pair_rasters={str(f.relative_to(output)):sha(f) for f in output.glob('pairs/**/*.tif')})
        save(output/'CONTENT_FINGERPRINT.json',fingerprint)
        receipt['output_sha256']={str(f.relative_to(output)):sha(f) for name in ('single_date','pairs','temporal','reports','SOURCE_REGISTRY') for f in (output/name).rglob('*') if f.is_file()}
        receipt['output_sha256']['CONTENT_FINGERPRINT.json']=sha(output/'CONTENT_FINGERPRINT.json')
        stages.guard();save(output/'FINAL_RECEIPT.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('output_sha256','summary')},ensure_ascii=False),flush=True)
    except Exception as error:
        save(output/('FAILED_ATTEMPT_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),successful_receipt_published=False));raise
    finally:
        if lock.exists():lock.unlink()

if __name__=='__main__':main()
