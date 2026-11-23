"""Actual-image contamination replay, roof-fragment provenance and partial bundle."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
import numpy as np
import rasterio,fiona,cv2
from rasterio.windows import Window
from shapely.geometry import shape,box
from shapely.affinity import translate
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.alignment.supported_ground_features import internal_spatial_check,directional_features

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if root.parent!=parent.resolve() or (root/'ACTUAL_MASK_INVARIANCE_AND_LINEAGE_QA.json').exists():
        raise ValueError('Append-only dedicated repair root required')
    plan=json.loads((root/'FROZEN_SUPPORT_REPAIR_PLAN.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'SUPPORTED_FEATURE_REPLAY.json').read_text(encoding='utf-8'))
    with rasterio.open(root/'2019_BOUNDED_GE_RGB_CACHE.tif') as s:moving=np.moveaxis(s.read(),0,-1)
    base=parent/'SINGLE2019_20261012_06'
    with rasterio.open(base/'2019_ROOF_FRACTION_RAW.tif') as s:raw=s.read(1)
    with rasterio.open('F:/xian/10m底图/Xian_2019_TrueColor.tif') as s:
        window=Window(1516,1765,256,256);reference=np.moveaxis(s.read([1,2,3],window=window),0,-1)
        rv=np.all(s.read_masks([1,2,3],window=window)>0,axis=0)
    exclusion=cv2.dilate((raw>0).astype(np.uint8),np.ones((7,7),np.uint8))>0
    rv&=~exclusion;mv=np.any(moving>0,axis=2)&~exclusion
    altered=moving.copy();altered[~mv]=1e5
    original_features,original_valid=directional_features(moving,mv)
    altered_features,altered_valid=directional_features(altered,mv)
    if not np.array_equal(original_features,altered_features) or not np.array_equal(original_valid,altered_valid):
        raise RuntimeError('Actual excluded pixels still contaminate features')
    component=np.zeros((256,256),bool);component[64:128,:64]=True
    before=internal_spatial_check(reference,moving,rv,mv,component)
    after=internal_spatial_check(reference,altered,rv,mv,component)
    if before!=after:raise RuntimeError('Actual-image supported tie changed under excluded-pixel perturbation')
    active=[c for c in summary['cells'] if c['recommended_experiment_correction_m'] is not None]
    support=box(308460,3792190,311020,3794750)
    cache=parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
    records=[];moved_geometries=[]
    with fiona.open(cache) as src:
        for f in src:
            g=shape(f.geometry);key=f.properties['component_key']
            for cell in active:
                region=shape(cell['support_geometry']);fragment=g.intersection(region)
                if fragment.area<=1e-6:continue
                dx,dy=cell['recommended_experiment_correction_m']
                valid=translate(region,xoff=dx,yoff=dy).intersection(region).intersection(support)
                moved=translate(fragment,xoff=dx,yoff=dy).intersection(valid)
                if moved.area<=1e-6:continue
                moved_geometries.append(moved)
                records.append({'component_key':key,'source_cache_sha256':sha(cache) if not records else records[0]['source_cache_sha256'],
                    'source_region':cell['region_id'],'source_observed_date':'2018-11-01','experiment_year':2019,
                    'source_component_wkb_sha256':hashlib.sha256(g.wkb).hexdigest(),
                    'source_fragment_wkb_sha256':hashlib.sha256(fragment.wkb).hexdigest(),
                    'candidate_fragment_wkb_sha256':hashlib.sha256(moved.wkb).hexdigest(),
                    'correction_m':[dx,dy],'source_fragment_area_m2':fragment.area,'candidate_fragment_area_m2':moved.area,
                    'complete_building_transport':False,'physical_id':None})
    with rasterio.open(root/'2019_REPAIRED_CANDIDATE_FRACTION.tif') as s:frac=s.read(1).astype(np.float64)
    expected_area=unary_union(moved_geometries).area
    raster_area=frac[frac!=-9999].sum()*100
    error=abs(expected_area-raster_area)
    if error>.05:raise RuntimeError('Lineage fragment union does not explain delivered fraction')
    save_json(root/'REPAIRED_COMPONENT_SOURCE_LINEAGE.json',{'records':records,'expected_union_area_m2':expected_area,
        'delivered_fraction_area_m2':raster_area,'float32_serialization_area_difference_m2':error,
        'role':'INTERNALLY_SUPPORTED_EXPERIMENT_ONLY_NOT_GROUND_FOOTPRINT'})
    qa={'excluded_actual_pixels_modified':int((~mv).sum()),'supported_feature_values_changed':0,
        'actual_fit_check_replay_unchanged':True,'matching_candidate':before['fit']['best'],
        'check':before['check'],'source_lineage_records':len(records),'fraction_lineage_area_error_m2':error,
        'formal_ground_precision_certified':False,'formal_fit_holdout_reviews':0}
    save_json(root/'ACTUAL_MASK_INVARIANCE_AND_LINEAGE_QA.json',qa)
    manifest={'branch':'SUPPORTED_SINGLE_CELL_CORRECTED_EXPERIMENT','experiment_year':2019,
        'source_observed_date':'2018-11-01','label':'2019_REPAIRED_EXPERIMENTAL_PSEUDOLABEL.tif',
        'weight':'2019_REPAIRED_EXPERIMENTAL_WEIGHT.tif','quality':'2019_REPAIRED_QUALITY.tif',
        'fraction':'2019_REPAIRED_CANDIDATE_FRACTION.tif','inputs':'2019_S1S2_INPUT_WINDOW.tif',
        'input_source_original':str(base/'2019_S1S2_INPUT_WINDOW.tif'),
        'experimental_samples':3091,'source_region':'roi_r1_c0','correction_m':[-10,0],
        'other_15_cells':'UNKNOWN_IGNORE_NO_EXTRAPOLATION','model_preprocessing':'RAW_INPUT_NORMALIZATION_NOT_RUN',
        'model_training_runs':0,'measured_gt_pixels':0,'formal_alignment_accepted':False}
    files=[root/manifest[k] for k in ('label','weight','quality','fraction')]
    files.extend([root/'SUPPORT_REPAIR_REPORT.md',root/'REPAIRED_COMPONENT_SOURCE_LINEAGE.json'])
    package=root/'SINGLE2019_PARTIAL_CORRECTED_EXPERIMENT.zip'
    checksums={f.name:sha(f) for f in files};input_file=base/'2019_S1S2_INPUT_WINDOW.tif'
    checksums[input_file.name]=sha(input_file)
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,f.name)
        z.write(input_file,input_file.name)
        z.writestr('DATASET_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('FILES_SHA256.json',json.dumps(checksums,indent=2))
    with zipfile.ZipFile(package) as z:
        if z.testzip() is not None:raise RuntimeError('Package CRC mismatch')
        for name,digest in checksums.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise RuntimeError('Package content mismatch')
    save_json(root/'PARTIAL_CORRECTED_PACKAGE_RECEIPT.json',{'path':str(package),'sha256':sha(package),
        'members_sha256':checksums,'manifest':manifest,'crc_verified':True,'whole_system_governance_completed':False})
    for path,digest in plan['protected_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Protected drift: '+path)
    print(json.dumps(qa,ensure_ascii=True))

if __name__=='__main__':main()
