"""Bounded single-2019 engineering run under explicit nominal-year policy."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
import numpy as np
import rasterio,fiona,cv2
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from affine import Affine
from shapely.geometry import shape,mapping
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.label_transfer import Grid,export_raster
from xian_r5.source_partition import SourceRegion
from xian_r5.fraction_shadow import build_fraction_shadow,compare_fraction_probes,QUALITY_BANDS,NODATA
from xian_r5.alignment.base_tie import estimate_component_gradient_translation,image_shift_to_projected_correction
from xian_r5.single_year_experiment import nominal_year_binding,experimental_pseudolabel,experiment_batch

def write(out,name,array,grid,bands,*,nodata=None,role='DIAGNOSTIC',extra=None):
    array=np.asarray(array)
    if array.ndim==2:array=array[None]
    export_raster(out/name,array,grid,bands,root=out,nodata=nodata)
    with rasterio.open(out/name,'r+') as s:
        tags=dict(branch='SINGLE2019_EXPERIMENT',scope='USER_AUTHORIZED_NOMINAL_YEAR_EXPERIMENT',
            EXPERIMENT_YEAR='2019',YEAR_POLICY='NOMINAL_YEAR_USER_AUTHORIZED',
            SOURCE_OBSERVED_DATE='2018-11-01',MEASURED_GT='NO',FORMAL_ALIGNMENT_ACCEPTED='NO',
            ABSOLUTE_POSITION_ACCURACY='UNKNOWN',ROLE=role,
            TRAINING_USE_MODE='EXPERIMENT_ONLY_NOT_SCIENTIFIC_GT',
            BACKGROUND_SEMANTICS='MODEL_BELOW_THRESHOLD_NOT_VERIFIED_ABSENCE')
        tags.update(extra or {});s.update_tags(**tags)
    with rasterio.open(out/name) as s:
        expected=array.astype(np.float32) if array.dtype.kind=='f' else array
        if s.transform!=grid.affine or not np.array_equal(s.read(),expected):raise RuntimeError('Raster roundtrip drift')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    r5=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923');parent=r5/'21_source_partition_alignment'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh dedicated experiment directory required')
    latest=parent/'GT10M_VALIDATION_PREFLIGHT_20261012_05'
    previous=json.loads((latest/'PREFLIGHT_IMPLEMENTATION_RECEIPT.json').read_text(encoding='utf-8'))
    prev_plan=json.loads((latest/'FROZEN_REFERENCE_PREFLIGHT_PLAN.json').read_text(encoding='utf-8'))
    protected=dict(prev_plan['protected_sha256']);protected.update(previous['new_code_sha256'])
    protected.update({str(latest/n):h for n,h in previous['output_sha256'].items()})
    protected[str(latest/'PREFLIGHT_IMPLEMENTATION_RECEIPT.json')]='4d9bb7a3761a5c2456e50a020df6bdd50ab7edee9ea19e307e2024a3e8429640'
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError('Frozen source drift: '+p)
    old=parent/'TRAINROI_20261012_01';cache=parent/'GT10M_FRACTION_SHADOW_20261012_04'
    cfg=json.loads((old/'FROZEN_RUN_CONFIG.json').read_text(encoding='utf-8'))
    old_field=json.loads((old/'SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))
    reference=Path('F:/xian/10m底图/Xian_2019_TrueColor.tif')
    ref_hash='63fe103fa6bc783202e8ca2597ffd7e4b1f249e3789e4114e55b0d302185a8b6'
    if sha(reference)!=ref_hash:raise ValueError('2019 reference drift')
    protected[str(reference)]=ref_hash
    binding_path=r5/'20_gt10m_label_transfer/B04_20261011_05/TRAINING_BUNDLE_BINDING_AUDIT.json'
    binding=json.loads(binding_path.read_text(encoding='utf-8'))
    grid=Grid(Affine(*cfg['roi_affine']),256,256,'EPSG:32649');window=Window(*cfg['roi_window'])
    with rasterio.open(reference) as s:
        if str(s.crs)!=grid.crs or s.window_transform(window)!=grid.affine:raise ValueError('2019 reference grid drift')
        ref=np.moveaxis(s.read([1,2,3],window=window),0,-1)
        reference_valid=np.all(s.read_masks([1,2,3],window=window)>0,axis=0)
    mosaic=Path(cfg['source']);before=mosaic.stat()
    plan={'run':out.name,'scope':'TRAINROI_256x256_SINGLE_NOMINAL_2019',
        'user_authorization':'Use files named 2019 as 2019 images; single-period special run to tune the system',
        'year_binding':nominal_year_binding(2019,cfg['source_actual_date']),
        'date_discrepancy_not_a_blocker_for_experiment':True,'date_evidence_overwritten':False,
        'reference':str(reference),'reference_sha256':ref_hash,'source':str(mosaic),'teacher':cfg['teacher'],
        'source_size':before.st_size,'source_mtime_ns':before.st_mtime_ns,
        'grid':{'affine':list(grid.affine)[:6],'shape':[256,256],'crs':grid.crs},
        'component_cache':str(cache/'SOURCE_MASK_COMPONENTS_SHADOW.gpkg'),
        'source_positive_exclusion':'Prior RAW fraction >0 then 30m dilation; display-independent positive-area exclusion',
        'alignment_method':'Existing gradient NCC; fixed radius3/min100; no matcher/warp parameter exploration',
        'old_2018_vectors_used':False,'subpixel_candidate_applied':False,
        'student_prediction_or_manual_reference_used_in_alignment':False,
        'tau':.43,'tau_fit':False,'experimental_geometry_risk_opt_in':True,
        'experimental_model_background_assumption_opt_in':True,
        'quality_risk_exclusions':['unknown vector','partial support','boundary','component seam','overlap','input invalid'],
        'input_preprocessing':'Raw delivered six-band windows; no model normalization/forward/optimization performed',
        'original_RSBuilding_predictions_reused':True,'RBF_SAM_inference':'NOT_RUN_OFFICIAL_CHECKPOINT_UNAVAILABLE',
        'multi_period_identity_change':'NOT_APPLICABLE_SINGLE_PERIOD',
        'formal_state':previous['formal_state'],'protected_sha256':protected}
    out.mkdir(parents=True);save_json(out/'FROZEN_2019_EXPERIMENT_PLAN.json',plan)
    moving=np.zeros((3,256,256),np.float32)
    with rasterio.open(mosaic) as s:
        for band in range(3):
            reproject(rasterio.band(s,band+1),moving[band],src_transform=s.transform,src_crs=s.crs,
                dst_transform=grid.affine,dst_crs=grid.crs,resampling=Resampling.average,
                dst_nodata=0,warp_mem_limit=64,num_threads=1)
    moving=np.moveaxis(moving,0,-1)
    moving_hash=hashlib.sha256(moving.tobytes()).hexdigest()
    if moving_hash!=old_field['ge_resampled_roi_sha256']:raise ValueError('Bounded source pixels drift')
    with rasterio.open(cache/'ROOF_FRACTION_RAW.tif') as s:raw_cached=s.read(1)
    exclusion=cv2.dilate((raw_cached>0).astype(np.uint8),np.ones((7,7),np.uint8))>0
    rv=reference_valid&~exclusion;mv=np.any(moving>0,axis=2)&~exclusion
    complete_kernel=cv2.erode((rv&mv).astype(np.uint8),np.ones((11,11),np.uint8),
        borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    cells=[];regions=[]
    for cell in old_field['cells']:
        rr,cc=(int(s[1:]) for s in cell['region_id'].split('_')[1:])
        component=np.zeros((256,256),bool);component[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
        diagnosis=estimate_component_gradient_translation(ref,moving,rv,mv,component,radius=3,min_samples=100)
        best=diagnosis['best'];vector=None if best is None else list(image_shift_to_projected_correction(best['dx_px'],best['dy_px'],grid.affine))
        record={'region_id':cell['region_id'],'support_geometry':cell['support_geometry'],
            'actual_source_date':cfg['source_actual_date'],'experiment_year':2019,
            'candidate_correction_m':vector,'correlation_diagnostic':diagnosis,
            'full_feature_kernel_supported_pixels':int((complete_kernel&component).sum()),
            'formal_acceptance':False,'relative_registration_uncertainty_m':None,
            'status':'UNREVIEWED_2019_REFERENCE_IMAGE_CANDIDATE' if best else 'UNRESOLVED_NO_PEAK'}
        digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        cells.append(record);regions.append(SourceRegion(record['region_id'],cfg['source_actual_date'],shape(record['support_geometry']),
            None if vector is None else tuple(vector),None if vector is None else digest))
    save_json(out/'2019_SOURCE_ALIGNMENT_CANDIDATES.json',{'cells':cells,'numeric_cells':sum(r.correction_m is not None for r in regions),
        'reference_sha256':ref_hash,'moving_roi_sha256':moving_hash,'reference_roi_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
        'full_feature_kernel_supported_pixels':int(complete_kernel.sum()),'ground_registration_verified':False})
    print('2019-only candidate estimation completed; building fractions.',flush=True)
    objects={}
    with fiona.open(cache/'SOURCE_MASK_COMPONENTS_SHADOW.gpkg') as s:
        if s.crs.to_epsg()!=32649:raise ValueError('Component cache frame drift')
        for f in s:objects[f.properties['component_key']]=shape(f.geometry)
    result=build_fraction_shadow(objects,regions,grid)
    if not np.allclose(result['raw_fraction'],raw_cached,atol=1e-6,rtol=0):raise ValueError('Cached raw fraction drift')
    inputs=[];valid=np.ones((256,256),bool);input_receipts=[]
    with zipfile.ZipFile('F:/xian/10m结果.zip') as z:
        for suffix in ('/sen1_test2/Xian_2019_12.tif','/sen2_test2/xianS2/Xian_2019.tif'):
            names=[n for n in z.namelist() if n.endswith(suffix)]
            if len(names)!=1:raise ValueError('Input member ambiguous')
            rec=next(r for r in binding['rasters'] if r['member'].endswith(suffix));data=z.read(names[0])
            if hashlib.sha256(data).hexdigest()!=rec['sha256']:raise ValueError('Training input drift')
            with MemoryFile(data) as memory,memory.open() as s:
                if s.window_transform(window)!=grid.affine or str(s.crs)!=grid.crs:raise ValueError('Training window grid drift')
                a=s.read(window=window);valid&=np.isfinite(a).all(axis=0)&np.all(s.read_masks(window=window)>0,axis=0)
                inputs.append(a)
            input_receipts.append({'member':names[0],'sha256':rec['sha256'],'bands':a.shape[0],'year_policy':2019})
    inputs=np.concatenate(inputs).astype(np.float32);inputs[:,~valid]=0
    # Threshold the delivered precision for reproducibility, not the hidden float64 computation.
    candidate=result['candidate_fraction'].astype(np.float32).astype(np.float64)
    scope=result['scope_fraction'][1].astype(np.float32).astype(np.float64)
    pseudo=experimental_pseudolabel(candidate,scope,result['quality'],valid,
        allow_unverified_geometry=True,allow_assumed_background=True)
    batch=experiment_batch(inputs,pseudo['label'],pseudo['experiment_weight'])
    common={'SOURCE_TEACHER_SHA256':cfg['teacher_sha256'],'REFERENCE_SHA256':ref_hash,
        'EXPERIMENT_SAMPLES_ARE_QUALIFIED_GT':'NO','TAU':'0.43','TAU_COMPARISON':'STRICT_GT',
        'THRESHOLD_NUMERICS':'float64 interpretation of delivered float32 fraction'}
    for filename,key in [('2019_ROOF_FRACTION_RAW.tif','raw_fraction'),('2019_ROOF_FRACTION_CANDIDATE.tif','candidate_fraction'),
                         ('2019_FRACTION_DIFFERENCE.tif','difference')]:
        write(out,filename,result[key],grid,[key.upper()],nodata=NODATA,extra=common)
    write(out,'2019_QUALITY.tif',result['quality'],grid,QUALITY_BANDS,extra=common)
    write(out,'2019_SCOPE_FRACTION.tif',result['scope_fraction'],grid,['RAW_SCOPE','TRANSPORTED_SCOPE'],extra=common)
    write(out,'2019_EXPERIMENTAL_PSEUDOLABEL_TAU043.tif',pseudo['label'],grid,['EXPERIMENTAL_BUILDING_BELOW_THRESHOLD_IGNORE'],
        nodata=255,role='EXPERIMENTAL_PSEUDOLABEL_NOT_GT',extra=common)
    write(out,'2019_EXPERIMENTAL_SAMPLE_WEIGHT.tif',pseudo['experiment_weight'],grid,['EXPERIMENT_SAMPLE_SELECTION_NOT_CONFIDENCE'],
        role='EXPERIMENTAL_SELECTION_WEIGHT',extra=common)
    write(out,'2019_S1S2_INPUT_WINDOW.tif',inputs,grid,['S1_VV','S1_VH','S2_B2','S2_B3','S2_B4','S2_B8'],
        role='RAW_INPUT_WINDOW_NOT_MODEL_NORMALIZED',extra={'SOURCE_OBSERVED_DATE':'UNKNOWN'})
    save_json(out/'2019_COMPONENT_SOURCE_LINEAGE.json',{'records':result['component_ledger'],
        'component_source':str(cache/'SOURCE_MASK_COMPONENTS_SHADOW.gpkg'),'experiment_year':2019,'actual_source_date':cfg['source_actual_date']})
    save_json(out/'2019_DATALOADER_SMOKE.json',{'input_receipts':input_receipts,'input_shape':list(batch['inputs'].shape),
        'target_shape':list(batch['target'].shape),'target_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(batch['target'],return_counts=True))},
        'ignore_index':-100,'invalid_input_pixels':int((~valid).sum()),'selected_samples':pseudo['counts']['experimental_samples'],
        'optimization_steps':0,'model_weights_loaded':False,'preprocessing_normalization':'NOT_RUN','loss_or_forward_pass':'NOT_RUN'})
    summary={'run':out.name,'experimental_counts':pseudo['counts'],'numeric_candidate_cells':sum(r.correction_m is not None for r in regions),
        'vectors':{c['region_id']:c['candidate_correction_m'] for c in cells},
        'quality_summary_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(result['quality'][0],return_counts=True))},
        'fraction_diagnostics':compare_fraction_probes(result)['statistics'],'area_conservation_error_m2':result['area_conservation_error_m2'],
        'roof_components_reused':len(objects),'full_feature_kernel_supported_pixels':int(complete_kernel.sum()),
        'source_actual_date_retained':cfg['source_actual_date'],'experiment_year':2019,'date_policy':'USER_AUTHORIZED_NOMINAL_YEAR',
        'supervision_product':'EXPERIMENTAL_PSEUDOLABEL_NOT_MEASURED_GT','formal_state':previous['formal_state'],
        'training_runs':0,'model_forward_runs':0,'citywide_runs':0,'RBF_SAM':'NOT_RUN','multi_period_states':'NOT_APPLICABLE'}
    save_json(out/'2019_RUN_SUMMARY.json',summary)
    after=mosaic.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Raw mosaic changed')
    for p,h in protected.items():
        if sha(p)!=h:raise RuntimeError('Protected input changed: '+p)
    print(json.dumps({'completed':True,'run':out.name,'numeric_cells':summary['numeric_candidate_cells'],
        'experiment_counts':pseudo['counts'],'area_conservation':result['area_conservation_error_m2'],
        'protected_hashes_verified':len(protected),'formal_accepts':0},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
