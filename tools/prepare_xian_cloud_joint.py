"""One bounded increment over existing registries; no protected test evaluation.

Stages immutable source-group splits, source-frame task data, and an allowlisted
private training payload. Never traverses the user's drive or SSH directory.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import fiona
import numpy as np
from affine import Affine
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape, mapping

REPO=Path(__file__).resolve().parents[1]
X=Path('F:/xian')
ROOT=X/'XIAN_R5_MODEL_UPGRADE_20260923'
STAGE=ROOT/'21_source_partition_alignment'
B0=REPO/'outputs/urban-renewal-phase1-starter/outputs/v2_readiness/rsbuilding/whu/work_dirs/swin_t_whu_pilot5000_20260728_110750/iter_2000.pth'
B0_SHA='f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda:s.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def stable(s):return hashlib.sha256(('XIAN_CLOUD_V1_20261020|'+s).encode()).hexdigest()

def prepare(run,package):
    run=Path(run).resolve();package=Path(package).resolve()
    dest=run/'payload'
    dest.mkdir(exist_ok=False)
    for sub in ('code','data','config','weights','prompts'):(dest/sub).mkdir()
    inputs={};files={};records=[]
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path)
        if expected and h!=expected:raise ValueError('Frozen source hash mismatch: '+str(path))
        inputs[str(path)]=h;return h
    def add(path,relative,expected=None):
        h=pin(path,expected);relative=Path(relative).as_posix()
        if relative.startswith('/') or '..' in Path(relative).parts or 'private' in Path(relative).parts:
            raise ValueError('Forbidden payload path')
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(target)!=h:raise ValueError('Payload member collision')
        else:shutil.copyfile(path,target)
        files[relative]={'sha256':h,'bytes':target.stat().st_size}
        return relative
    pin(package)
    with zipfile.ZipFile(package) as z:
        hashes=json.loads(z.read('PACKAGE_SHA256.json'))
        for name,meta in hashes.items():
            b=z.read(name)
            if len(b)!=meta['bytes'] or hashlib.sha256(b).hexdigest()!=meta['sha256']:raise ValueError('Package hash mismatch')
            target=dest/'prompts'/name
            with target.open('xb') as stream:stream.write(b)
            files['prompts/'+name]={'sha256':sha(target),'bytes':len(b)}
    budget_path=run/'USER_AUTHORIZATION_AND_BUDGET.json';pin(budget_path)
    save(dest/'config/budget.json',read(budget_path))
    print('FREEZE_SOURCE_GROUP_SPLITS',flush=True)
    regpath=X/'XIAN_R5_S42_LOCAL_RGB_RESTORE_20261019_02/S_FACADE_TRAINING_MANIFEST.json'
    oldpath=X/'XIAN_R5_S44_FACADE_TRAINING_20261019/TRAINING_SELECTION_AND_SPLIT.json'
    pin(regpath);pin(oldpath)
    registry=read(regpath);old=read(oldpath)
    def parent(r):return '/'.join(map(str,r['parent_scene']))
    oldparents={parent(r) for r in old['TRAIN']}
    groups=sorted({parent(r) for r in registry['rows']})
    # Newly held-out parent IDs were not in the old trained482 set. Metadata
    # selection is frozen before any new score. Spatial overlap remains UNKNOWN.
    eligible=sorted(set(groups)-oldparents,key=stable)
    if len(eligible)<60:raise ValueError('Insufficient never-trained parent groups')
    group_role={g:'TRAIN' for g in groups}
    for role,subset in [('TUNE',eligible[:20]),('CALIBRATION',eligible[20:40]),('FINAL_TEST',eligible[40:60])]:
        for g in subset:group_role[g]=role
    for idx,r in enumerate(registry['rows']):
        role=group_role[parent(r)];uid='BANDON_'+Path(r['official_RGB_member']).with_suffix('').as_posix().replace('/','_')
        base=f'data/bandon/{role}/{uid}'
        im=add(r['source_path'],base+'.jpg',r['actual_SHA256'])
        label=add(r['label_path'],base+'.png',r['label_sha256'])
        records.append(dict(id=uid,dataset='BANDON',split=role,official_split='train',parent_scene=parent(r),
             city=r['parent_scene'][0],time_group=r['time_group'],actual_date=None,image=im,label=label,
             image_sha256=files[im]['sha256'],label_sha256=files[label]['sha256'],native_size=r['native_size'],
             label_kind='BANDON_ROOF_FACADE_0_1_2_255',facade_available=True,
             facade_semantics='GEOMETRIC_AUXILIARY_NOT_EXACT_VISIBLE_SURFACE_GT',
             prior_trained_parent=parent(r) in oldparents,spatial_independence='UNKNOWN',GSD_m=None))
    valzip=Path(old['validation_archive']);pin(valzip,old['validation_archive_sha256'])
    # Existing206 remain exposed diagnostics; never become new blind FINAL_TEST.
    with zipfile.ZipFile(valzip) as z:
        for i,r in enumerate(old['VALIDATION']):
            uid='BANDON_DEV_'+Path(r['RGB_member']).with_suffix('').as_posix().replace('/','_')
            paths=[]
            for key,hkey,suffix in [('RGB_member','RGB_sha256','.jpg'),('label_member','label_sha256','.png')]:
                b=z.read(r[key]);h=hashlib.sha256(b).hexdigest()
                if h!=r[hkey]:raise ValueError('Existing validation archive mismatch')
                relative=f'data/bandon/EXPOSED_DEV/{uid}{suffix}';p=dest/relative;p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as stream:stream.write(b)
                files[relative]=dict(sha256=h,bytes=len(b));paths.append(relative)
            records.append(dict(id=uid,dataset='BANDON',split='EXPOSED_DEV',official_split='val',parent_scene=None,
                  image=paths[0],label=paths[1],image_sha256=r['RGB_sha256'],label_sha256=r['label_sha256'],
                  native_size=r['shape'][:2],label_kind='BANDON_ROOF_FACADE_0_1_2_255',facade_available=True,
                  spatial_independence='UNKNOWN',GSD_m=None))
    bonpath=X/'XIAN_R5_S2_SUPERVISED_ROOF_TRAINING_20261019/TRAINING_ADMISSION_MANIFEST_V2.json';pin(bonpath)
    for r in read(bonpath)['records']:
        if r['split'] not in ('TRAIN','VALIDATION'):continue
        uid='BONAI_'+Path(r['file_name']).stem;role='TRAIN' if r['split']=='TRAIN' else 'EXPOSED_DEV'
        im=add(r['image_path'],f'data/bonai/{role}/{uid}.png',r['image_sha256'])
        label=add(r['label_path'],f'data/bonai/{role}/{uid}_label.png',r['label_sha256'])
        records.append(dict(id=uid,dataset='BONAI',split=role,official_split='trainval',parent_scene=r['source_group'],
             city=r['city'],actual_date=None,nominal_year=r['official_image'].get('date_captured'),image=im,label=label,
             image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],native_size=r['image_shape_hwc'][:2],
             label_kind='BONAI_ROOF_0_1_255',facade_available=False,GSD_m=None,
             original_annotation_ids=[a['annotation_id'] for a in r.get('instances',[])],
             semantics='SOURCE_FRAME_ROOF_NOT_FOOTPRINT',prior_exposure='OLD_S2_TRAIN_OR_DEV'))
    save(dest/'config/segmentation_data.json',{'records':records,'group_role':group_role,
         'split_selection':'SHA_GROUP_20TUNE_20CALIBRATION_20FINAL_FROM_PARENTS_OUTSIDE_OLD_TRAIN482',
         'source_independence':'PARENT_DISJOINT_ONLY_ACTUAL_SPATIAL_OVERLAP_UNKNOWN',
         'old_protected_tests_included':False,'exposed206_role':'EXPOSED_DEV_DIAGNOSTIC_ONLY'})
    print('PREPARE_FIXED_SOURCE_FRAME_A_TRAINING_DATA',flush=True)
    packetpath=STAGE/'SINGLE2019_FULLROI_ROOF_PROPOSALS_20261012_20/FULLROI_NATIVE_SOURCE_PACKETS.json'
    pin(packetpath);packets=read(packetpath)['records']
    graph=ROOT/'23_building_object_identity/XIAN_R5_IDENTITY_MINIMAL_CLOSURE_20261015/IDENTITY_GRAPH_MINIMAL.gpkg';pin(graph)
    with fiona.open(graph,layer='roof_instance_reference') as src:observed=[shape(f.geometry) for f in src]
    with fiona.open(graph,layer='manual_roof_reference') as src:manual=[(str(f.id),shape(f.geometry)) for f in src]
    wp=STAGE/'TRAINROI_20261012_01/FROZEN_RUN_CONFIG.json';pin(wp);window=read(wp)
    reference=Path(window['reference']);pin(reference,window['reference_sha256'])
    arois=[];ordered=sorted(packets,key=lambda p:stable(p['region_id']))
    region_role={p['region_id']:'TRAIN' for p in packets}
    for role,selected in [('TUNE',ordered[:2]),('CALIBRATION',ordered[2:4]),('FINAL_TEST',ordered[4:6])]:
        for p in selected:region_role[p['region_id']]=role
    with rasterio.open(reference) as lr:
        for p in packets:
            region=p['region_id'];support=shape(p['exact_support']);left,bottom,right,top=support.bounds
            width=height=256;aff=Affine((right-left)/width,0,left,0,-(top-bottom)/height,top)
            source_aff=Affine(*p['source_raster_affine'])*Affine.translation(*p['source_window'][:2])
            pin(p['image_path'],p['image_sha256']);rgb=np.asarray(Image.open(p['image_path']).convert('RGB'))
            hr=np.zeros((3,height,width),np.uint8)
            for k in range(3):reproject(rgb[:,:,k],hr[k],src_transform=source_aff,src_crs=p['source_raster_crs'],
                                        dst_transform=aff,dst_crs='EPSG:32649',resampling=Resampling.average)
            low=np.zeros((3,64,64),np.uint8);laff=aff*Affine.scale(4)
            for k in range(3):reproject(rasterio.band(lr,k+1),low[k],src_transform=lr.transform,src_crs=lr.crs,
                                        dst_transform=laff,dst_crs='EPSG:32649',resampling=Resampling.nearest)
            # Labels are separate files and are never part of inference records.
            parts=[(g,1) for g in observed if g.intersects(support)]
            refs=[(fid,g) for fid,g in manual if g.intersects(support)]
            source_mask=rasterize(parts,out_shape=(height,width),transform=aff,fill=0,dtype='uint8')
            target=rasterize([(g,1) for _,g in refs],out_shape=(height,width),transform=aff,fill=0,dtype='uint8')
            role=region_role[region];base=dest/f'data/xian/{role}';base.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(base/f'{region}_input.npz',hr_rgb=hr,lr_rgb=low,roof_mask=source_mask,
                                source_support=np.ones((height,width),np.uint8),affine=np.array(list(aff)[:6]))
            np.savez_compressed(base/f'{region}_supervision.npz',reference_roof=target,
                                reference_FIDs=np.array([fid for fid,_ in refs]))
            arois.append(dict(id=region,split=role,input=f'data/xian/{role}/{region}_input.npz',
                 training_supervision=f'data/xian/{role}/{region}_supervision.npz',
                 source_actual_date=p['source_observed_date'],reference_GT_year=2019,
                 time_compatibility='NOT_CONFIRMED_REAL_DOMAIN_WEAK_SUPERVISION_ONLY',
                 support=mapping(support),feature_pixel_size_m=list(aff)[:6][0],
                 source_pixel_frame=p['source_raster_crs'],target_crs='EPSG:32649',
                 reference_GT_semantics='REFERENCE_BUILDING_PLANIMETRY_MANUAL_ROOF',
                 ground_footprint=False,source_region_is_camera_unit=False,
                 coarse_roof_raster_not_final_area_integral=True,
                 label_completeness='REFERENCE_OBJECTS_PRESENT_BACKGROUND_COMPLETENESS_NOT_INDEPENDENTLY_VERIFIED'))
    save(dest/'config/alignment_data.json',{'records':arois,'role':'SCENE_LEVEL_REAL_DOMAIN_WEAK_SUPERVISION',
         'supervision_labels_excluded_from_prediction_inputs':True,'cross_date_physical_IDs_invented':False,
         'old267_or_53_links_promoted':False,'background_is_not_certified_absence':True,
         'loss_scope':'POSITIVE_REFERENCE_AND_OBSERVED_SUPPORT_NOT_SAME_DATE_CHANGE_GT'})
    # Source snapshots only: no .git, caches, checkpoints, arbitrary directories.
    for folder in ('opencd','configs'):
        root=Path('E:/Data Set/RSBuilding/repo')/folder
        for p in root.rglob('*.py'):
            if '__pycache__' not in p.parts:add(p,'code/RSBuilding/'+p.relative_to(root.parent).as_posix())
    add(B0,'weights/B0.pth',B0_SHA)
    add(B0.parent/'Swin_T_whu_windows_pilot5000.py','config/B0_model.py')
    save(dest/'config/input_source_hashes.json',inputs)
    save(run/'LOCAL_DATA_PREPARATION.json',dict(status='PREPARED_NOT_TRAINED',created_at=datetime.now().astimezone().isoformat(),
         package_sha256=sha(package),source_registry_reused=True,no_second_registry=True,
         segmentation_counts=dict(Counter(r['dataset']+'/'+r['split'] for r in records)),
         xian_regions=dict(Counter(r['split'] for r in arois)),
         all_1776_official_training_pairs_retained_across_experiment_roles=True,
         old206_frozen=True,old_protected_test_data_read=False,training='NOT_RUN',
         cloud_data_upload='NOT_RUN',code_pending='NEW_S_A_ADAPTERS_AND_TRAINER',
         new_final_test_is_not_certified_geographically_independent=True))
    save(run/'PRIVATE_RESEARCH_USAGE_DECISION.json',dict(DATASET_LICENSE_STATUS='UNRESOLVED',
         basis='AUTHOR_PUBLIC_RESEARCH_RELEASE_AND_USER_REQUESTED_NONPUBLIC_REPRODUCIBLE_RESEARCH',
         BANDON='https://github.com/fitzpchao/BANDON',BONAI='https://huggingface.co/datasets/jwwangchn/BONAI',
         code_MIT_not_used_as_dataset_license=True,commercial_rights_claim=False,public_redistribution=False,
         cloud_scope='SINGLE_USER_CONTROLLED_PRIVATE_INSTANCE_MINIMUM_NECESSARY_PROCESSING_NO_PUBLIC_ENDPOINT',
         not_a_statement_of_unrestricted_cloud_commercial_or_redistribution_rights=True,
         explicit_provider_restrictions_if_found='ISOLATE_AFFECTED_ASSET_DO_NOT_BYPASS'))
    # No archive yet: main agent adds tested code and freezes protocol first.
    print(json.dumps(read(run/'LOCAL_DATA_PREPARATION.json'),ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--package',type=Path,required=True)
    a=p.parse_args();prepare(a.run,a.package)
if __name__=='__main__':main()
