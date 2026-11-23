"""One frozen support-closed feature repair replay; no threshold/radius sweep."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import rasterio,fiona,cv2
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling
from shapely.geometry import shape
from affine import Affine
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.alignment.supported_ground_features import directional_features,internal_spatial_check
from xian_r5.alignment.base_tie import image_shift_to_projected_correction
from xian_r5.source_partition import SourceRegion
from xian_r5.fraction_shadow import build_fraction_shadow,QUALITY_BANDS,NODATA
from xian_r5.single_year_experiment import experimental_pseudolabel

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);out=parser.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    base=parent/'SINGLE2019_20261012_06';old=parent/'TRAINROI_20261012_01'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh bounded repair run required')
    receipt=json.loads((base/'SINGLE2019_IMPLEMENTATION_RECEIPT.json').read_text(encoding='utf-8'))
    prior_plan=json.loads((base/'FROZEN_2019_EXPERIMENT_PLAN.json').read_text(encoding='utf-8'))
    protected=dict(prior_plan['protected_sha256']);protected.update(receipt['new_code_sha256'])
    protected.update({str(base/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(base/'SINGLE2019_IMPLEMENTATION_RECEIPT.json')]='e902011f2cc8e430272b1904c73ca07a48503acc04046c200a34713bfe2059cf'
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError('Protected drift: '+p)
    cfg=json.loads((old/'FROZEN_RUN_CONFIG.json').read_text(encoding='utf-8'))
    field=json.loads((base/'2019_SOURCE_ALIGNMENT_CANDIDATES.json').read_text(encoding='utf-8'))
    grid=Grid(Affine(*cfg['roi_affine']),256,256,'EPSG:32649')
    reference=Path(prior_plan['reference']);window=Window(*cfg['roi_window'])
    with rasterio.open(reference) as s:
        if s.window_transform(window)!=grid.affine:raise ValueError('Reference grid changed')
        ref=np.moveaxis(s.read([1,2,3],window=window),0,-1)
        rv=np.all(s.read_masks([1,2,3],window=window)>0,axis=0)
    with rasterio.open(base/'2019_ROOF_FRACTION_RAW.tif') as s:raw=s.read(1)
    exclusion=cv2.dilate((raw>0).astype(np.uint8),np.ones((7,7),np.uint8))>0
    cfg_plan={'run':out.name,'scope':'SAME_256x256_NOMINAL_2019_WINDOW','protected_sha256':protected,
        'feature_operator':'SUPPORT_CLOSED_AXIS_CENTRAL_DIFFERENCE_3_PIXELS_NO_GAUSSIAN',
        'roof_exclusion_unchanged':'RAW roof fraction >0 then 7x7/30m dilation',
        'search_radius_px':3,'minimum_distinct_fit_pixels':100,'minimum_distinct_common_check_pixels':100,
        'internal_spatial_blocks_px':16,'minimum_internal_check_gain':.01,
        'check_kernel_cannot_cross_fit_block':True,'formal_fit_holdout_review':False,
        'ground_mask_semantics':'ROOF_EXCLUSION_PROXY_NOT_VERIFIED_STABLE_GROUND',
        'no_mask_buffer_shrink_no_minimum_lowering':True,'model_visual_tie_selection':False,
        'source_observed_date':cfg['source_actual_date'],'experiment_year':2019,
        'formal_state':receipt['formal_state']}
    out.mkdir(parents=True);save_json(out/'FROZEN_SUPPORT_REPAIR_PLAN.json',cfg_plan)
    moving=np.zeros((3,256,256),np.float32);source=Path(cfg['source']);before=source.stat()
    with rasterio.open(source) as s:
        for b in range(3):reproject(rasterio.band(s,b+1),moving[b],src_transform=s.transform,src_crs=s.crs,
            dst_transform=grid.affine,dst_crs=grid.crs,resampling=Resampling.average,dst_nodata=0,
            num_threads=1,warp_mem_limit=64)
    moving=np.moveaxis(moving,0,-1)
    if hashlib.sha256(moving.tobytes()).hexdigest()!=field['moving_roi_sha256']:raise ValueError('Raw ROI pixels changed')
    rv&=~exclusion;mv=np.any(moving>0,axis=2)&~exclusion
    rf,rm=directional_features(ref,rv);mf,mm=directional_features(moving,mv)
    old_full=cv2.erode((rv&mv).astype(np.uint8),np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    features=np.concatenate([rf,mf]);valid=np.concatenate([rm,mm]).astype(np.uint8)
    write(out,'2019_SUPPORTED_FEATURE_VALUES.tif',features,grid,['REF_HORIZONTAL','REF_VERTICAL','GE_HORIZONTAL','GE_VERTICAL'],
        extra={'INVALID_FEATURE_VALUE_ZERO_IS_NOT_VALIDITY':'YES'})
    write(out,'2019_SUPPORTED_FEATURE_VALIDITY.tif',valid,grid,['REF_HORIZONTAL_VALID','REF_VERTICAL_VALID','GE_HORIZONTAL_VALID','GE_VERTICAL_VALID'])
    write(out,'2019_BOUNDED_GE_RGB_CACHE.tif',np.moveaxis(moving,-1,0),grid,['GE_RED','GE_GREEN','GE_BLUE'],
        role='BOUNDED_SOURCE_PIXEL_CACHE',extra={'MOVING_ARRAY_SHA256':field['moving_roi_sha256']})
    cells=[];regions=[]
    for cell in field['cells']:
        rr,cc=(int(x[1:]) for x in cell['region_id'].split('_')[1:]);component=np.zeros((256,256),bool)
        component[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
        diagnosis=internal_spatial_check(ref,moving,rv,mv,component,radius=3,min_samples=100,block_size=16,minimum_gain=.01)
        best=diagnosis['fit']['best'];vector=None if best is None else list(image_shift_to_projected_correction(best['dx_px'],best['dy_px'],grid.affine))
        applied=vector if diagnosis['engineering_experiment_supported'] else None
        record={'region_id':cell['region_id'],'support_geometry':cell['support_geometry'],
            'diagnostic_correction_m':vector,'recommended_experiment_correction_m':applied,
            'source_observed_date':cfg['source_actual_date'],'experiment_year':2019,
            'reference_axis_valid_pixels':int((rm.any(axis=0)&component).sum()),
            'moving_axis_valid_pixels':int((mm.any(axis=0)&component).sum()),'diagnosis':diagnosis,'formal_acceptance':False}
        digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        cells.append(record);regions.append(SourceRegion(cell['region_id'],cfg['source_actual_date'],shape(cell['support_geometry']),
            None if applied is None else tuple(applied),None if applied is None else digest))
    eligible=sum(r.correction_m is not None for r in regions)
    feature_stats={'raw_unverified_nonroof_pixels':int((rv&mv).sum()),
        'old_complete_11x11_kernel_pixels':int(old_full.sum()),
        'ref_supported_axis_pixel_count':int(rm.any(axis=0).sum()),'ge_supported_axis_pixel_count':int(mm.any(axis=0).sum()),
        'ref_horizontal_stencils':int(rm[0].sum()),'ref_vertical_stencils':int(rm[1].sum()),
        'feature_mask_invariance_regression':'LEGACY_RED_CORR_1_TO_0_78498_NEW_GREEN_UNCHANGED'}
    summary={'run':out.name,'cells':cells,'feature_stats':feature_stats,
        'numeric_fit_candidate_cells':sum(c['diagnostic_correction_m'] is not None for c in cells),
        'internally_supported_nonzero_experiment_cells':eligible,'formal_accepted_cells':0,
        'formal_state':receipt['formal_state'],'experimental_repaired_label_samples':0,
        'raw_baseline_still_available':str(base/'2019_EXPERIMENT_DATA_MANIFEST.json')}
    if eligible:
        cache=parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
        with fiona.open(cache) as s:objects={f.properties['component_key']:shape(f.geometry) for f in s}
        result=build_fraction_shadow(objects,regions,grid)
        label=experimental_pseudolabel(result['candidate_fraction'].astype(np.float32).astype(np.float64),
            result['scope_fraction'][1].astype(np.float32).astype(np.float64),result['quality'],np.ones((256,256),bool),
            allow_unverified_geometry=True,allow_assumed_background=True)
        write(out,'2019_REPAIRED_CANDIDATE_FRACTION.tif',result['candidate_fraction'],grid,['CANDIDATE_FRACTION'],nodata=NODATA)
        write(out,'2019_REPAIRED_EXPERIMENTAL_PSEUDOLABEL.tif',label['label'],grid,['EXPERIMENTAL_LABEL'],nodata=255)
        write(out,'2019_REPAIRED_EXPERIMENTAL_WEIGHT.tif',label['experiment_weight'],grid,['EXPERIMENT_SELECTION'])
        write(out,'2019_REPAIRED_QUALITY.tif',result['quality'],grid,QUALITY_BANDS)
        summary['experimental_repaired_label_samples']=label['counts']['experimental_samples']
        summary['area_conservation_error_m2']=result['area_conservation_error_m2']
    save_json(out/'SUPPORTED_FEATURE_REPLAY.json',summary)
    for p,h in protected.items():
        if sha(p)!=h:raise RuntimeError('Protected source changed: '+p)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Raw mosaic changed')
    print(json.dumps({'feature_stats':feature_stats,'numeric_fit_cells':summary['numeric_fit_candidate_cells'],
        'internally_supported_cells':eligible,'repaired_label_samples':summary['experimental_repaired_label_samples'],
        'protected_hashes_verified':len(protected)},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
