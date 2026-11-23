"""Read-only metadata adapters. No raster pixel reads, no weight loads."""
from pathlib import Path
from datetime import date
import json,hashlib
from dataclasses import asdict
from collections import defaultdict
import numpy as np
import rasterio
from shapely.geometry import shape,mapping,box
from shapely.affinity import affine_transform
from .contracts import Observation,digest,sha,read

def validate_geometry(g):
    if not g.is_valid:
        before=g.area;g=g.buffer(0)
        if not g.is_valid or g.is_empty or abs(g.area-before)>max(1e-6,abs(before)*1e-8):raise ValueError('Geometry requires explicit nontrivial repair')
    if g.is_empty or g.area<=0:raise ValueError('Empty supplied observation; do not silently remove from denominator')
    return g

def frozen_xian(old_root,expanded=False):
    base=Path('F:/xian/XIAN_R5_APPLICATION/APPLICATION_V1_20261027_01/runs/REAL_THREE_EPOCH_MULTIPARENT')
    paths=[base/'temporal/OBSERVATIONS.jsonl']
    if expanded:paths.append(old_root/'runs/ACTIVE_EXPANDED/RECOVERED_OBSERVATIONS_VALIDATED.jsonl')
    rows=[json.loads(s) for p in paths for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
    observations=[]
    for r in rows:
        observations.append(Observation(r['observation_id'],r['epoch_id'],r['actual_date'],
            mapping(validate_geometry(shape(r['raw_geometry']))),mapping(validate_geometry(shape(r['native_geometry']))),
            str(paths[-1]) if r.get('active_recheck_lineage') else str(paths[0]),proposal_kind='FROZEN_B0_MODEL_CANDIDATE',
            alpha_support=r.get('source_valid_fraction'),source_roof_observed=True,actual_date=r['actual_date'],date_precision='ACTUAL_ACQUISITION_DAY'))
    return observations,{str(p):sha(p) for p in paths}

def load_source_metadata(shard_record,frame_indices,scope):
    """Provided-detection benchmark input, with GT sidecar returned separately.

    Proposal UID is based on geometry and source order, never on canonical Id.
    Generator receives no sidecar. Id is joined only after graph hash creation.
    """
    if shard_record['role'] not in ['FIT','TUNE']:raise ValueError('New graph preparation is restricted to FIT/TUNE')
    mp=Path(shard_record['path']).with_suffix('.json');meta=read(mp)
    if meta['role']!=shard_record['role']:raise ValueError('Inherited group mismatch')
    observations=[];gold={};truth={};pins={str(mp):sha(mp)};epochs=[];counts=[]
    for index in frame_indices:
        fr=meta['frames'][index];lp=Path(fr['labels']);ip=Path(fr['image'])
        if 'labels_match_pix' not in lp.parts or meta['aoi'] not in lp.parts:raise ValueError('Unapproved source-label path')
        source_root=(Path('E:/Data Set/SN7_buildings_train/train')/meta['aoi']).resolve()
        if not lp.resolve().is_relative_to(source_root/'labels_match_pix') or not ip.resolve().is_relative_to(source_root/'images_masked'):
            raise ValueError('Metadata attempted to escape the approved original TRAIN source')
        if sha(lp)!=fr['label_sha256']:raise ValueError('Source label metadata changed')
        with rasterio.open(ip) as ds:
            affine=ds.transform;crs=ds.crs
            if not crs.is_projected or crs.linear_units not in ['metre','meter']:raise ValueError('Projected metre frame required')
        # Metadata open only. There is deliberately no ds.read() call here.
        epoch=lp.name.split('_mosaic_')[0].replace('global_monthly_','')
        year,month=map(int,epoch.split('_'));key=f'{year:04d}-{month:02d}-01'
        epochs.append(dict(epoch=epoch,time_key=key,actual_date=None,date_precision='PUBLIC_MONTHLY_COMPOSITE_LABEL_NOT_EXACT_ACQUISITION'))
        features=read(lp)['features'];ids=[];n=0
        for fi,f in enumerate(features):
            if not f.get('geometry'):continue
            g=validate_geometry(shape(f['geometry']));physical_label=(f.get('properties') or {}).get('Id')
            if physical_label is None:raise ValueError('Missing canonical Id label; do not manufacture it')
            identity=meta['aoi']+':'+str(int(physical_label));ids.append(identity)
            if not g.intersects(box(*scope)):continue
            uid='P_'+digest([meta['aoi'],epoch,fi,mapping(g)])[:26]
            metric=affine_transform(g,[affine.a,affine.b,affine.d,affine.e,affine.c,affine.f])
            o=Observation(uid,epoch,key,mapping(metric),mapping(g),str(lp),proposal_kind='PROVIDED_OFFICIAL_FOOTPRINT_GEOMETRY_NOT_DETECTOR_OUTPUT',
                image_path=str(ip),image_sha256=fr['image_sha256'],alpha_support=None,source_roof_observed=None,actual_date=None,date_precision='PUBLIC_MONTHLY_COMPOSITE_LABEL_NOT_EXACT_ACQUISITION')
            observations.append(o);gold[uid]=identity;n+=1
        if len(ids)!=len(set(ids)):raise ValueError('Multipart canonical Id requires explicit provided-instance adapter')
        truth[epoch]=ids;counts.append(dict(epoch=epoch,full_label_records=len(ids),provided_in_extent=n))
        pins[str(lp)]=fr['label_sha256']
    return observations,gold,truth,dict(aoi=meta['aoi'],role=meta['role'],source_metadata=str(mp),frame_indices=frame_indices,scope_native_pixels=scope,
        epochs=epochs,counts=counts,pins=pins,raster_metadata_opens=len(frame_indices),raster_pixel_reads=0,
        source_ground_truth_vector_metadata_reads=len(frame_indices),provided_detection_assumption=True,
        candidate_generation_uses_identity_labels=False)

def predetermined_source_cases(old_root):
    receipt=read(old_root/'evidence/SN7_DATASET_RECEIPT.json')
    tune=[r for r in receipt['shard_records'] if r['role']=='TUNE']
    fit=[r for r in receipt['shard_records'] if r['role']=='FIT'][:2]
    cases=[]
    for r in fit+tune:
        meta=read(Path(r['path']).with_suffix('.json'));n=len(meta['frames'])
        if n<5:raise ValueError('This preparation needs five distinct monthly label epochs')
        h=hashlib.sha256(('TASKALIGN_V1_FIXED_SCOPE:'+r['aoi']).encode()).hexdigest();x=(int(h[:4],16)%5+1)*128;y=(int(h[4:8],16)%5+1)*128
        extent=[x-32,y-32,x+128+32,y+128+32]
        windows=[('CONTIGUOUS',list(range(5))),('FULL_SPAN',np.rint(np.linspace(0,n-1,5)).astype(int).tolist())]
        for name,indices in windows:cases.append(dict(case_id=r['role']+'_'+r['aoi']+'_'+name,shard=r,indices=indices,extent=extent))
    return cases
