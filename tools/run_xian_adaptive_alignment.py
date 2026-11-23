"""Bounded R5-A audit / GT-free baseline / fail-closed delivery; CPU only."""
import argparse,csv,hashlib,importlib.util,io,json,math,sys,time,unittest,urllib.request,shutil,traceback
from pathlib import Path
import numpy as np,rasterio,fiona
from rasterio.windows import Window
from rasterio.features import rasterize
from affine import Affine
from pyproj import Transformer
from shapely.geometry import box,shape,mapping,Polygon
from shapely.ops import transform,unary_union
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json,write_new
from xian_r5.label_transfer import Grid,export_raster
from xian_r5.source_partition import SourceRegion,source_partition_transport
from xian_r5.coherent_support import coalesce_equal_field_regions
from xian_r5.fraction_shadow import _areas_and_counts,NODATA
from xian_r5.alignment.mosaic_grid import MosaicTileGrid
from xian_r5.alignment.adaptive_local import fit_local_alignment,predict_local_alignment,DEFAULT_CONFIG
P=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923');S=P/'21_source_partition_alignment'
OUT=S/'XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_20261013'
GE=Path('F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif')
S2=Path('F:/xian/10m底图/Xian_2019_TrueColor.tif')
DATE=Path('F:/xian/亚米市区底图/date_audit/tables/tile_actual_date.csv')
GT=Path('F:/xian/西安建筑轮廓数据/西安.shp')
COMP=S/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
GRID=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
ROI=box(308460,3792190,311020,3794750)

def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def exact_complete_fraction(observed):
    cells=np.array([GRID.cell(c,r) for r in range(256) for c in range(256)],dtype=object)
    clipped=[g.intersection(ROI) for g in observed.values() if g.intersection(ROI).area>1e-6]
    areas,counts,conflict=_areas_and_counts(clipped,cells)
    if not conflict.any():
        error=abs(areas.sum()-sum(g.area for g in clipped))
        if error>1e-3:raise ValueError('Complete-roof fraction area conservation failed')
    return (areas/100).reshape(256,256),int(conflict.sum())
def registry():
    return [SourceRegion(c['region_id'],c['actual_source_date'],shape(c['support_geometry']))
        for c in read_json(S/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json')['cells']]
def objects():
    with fiona.open(COMP,layer='native_positive_components') as src:
        return {f.properties['component_key']:shape(f.geometry) for f in src}
def check_protected():
    plan=read_json(OUT/'FROZEN_RUN_PLAN.json')
    for p,h in plan['protected_sha256'].items():
        if sha(p)!=h:raise ValueError('Protected input drift: '+p)
    for p,s in plan['large_file_stat'].items():
        a=Path(p).stat()
        if [a.st_size,a.st_mtime_ns]!=s:raise ValueError('Large source changed: '+p)
    return len(plan['protected_sha256'])

def freeze():
    if OUT.exists():raise ValueError('Fresh append-only R5-A directory required')
    last=read_json(S/'SINGLE2019_CASE_FIGURES_20261012_28/FROZEN_CASE_FIGURE_PLAN.json')
    protected={**last['protected_sha256'],**last['new_code_sha256']}
    required=[P/'19_ground_registration_evidence/B04_20261007_01/13_reports/B04_2019_RESIDUAL_HETEROGENEITY_REPORT.json',
        P/'19_ground_registration_evidence/B04_20261007_02/13_reports/IMPLEMENTATION_REPORT.md',
        P/'20_gt10m_label_transfer/GT10M_SUPPORT_20261011_01/PROVENANCE_RECOVERY_AUDIT.json',
        P/'20_gt10m_label_transfer/B04_20261011_02/FINAL_DELIVERY_RECEIPT.json',
        S/'TRAINROI_20261012_03/PREVIEW_DELIVERY_RECEIPT.json',
        S/'SINGLE2019_ROOF_EVIDENCE_20261012_08/ROOF_CORRESPONDENCE_HYPOTHESES.json',
        S/'SINGLE2019_ROOF_EVIDENCE_20261012_08/REFERENCE_DEVELOPMENT_EXPOSURE.json',
        S/'SINGLE2019_ROOF_PAIR_CHECK_20261012_22/FROZEN_PAIR_TRANSLATION_PLAN.json',
        S/'SINGLE2019_ROOF_PAIR_CHECK_20261012_22/FULLROI_PAIR_TRANSLATION_DIAGNOSTIC.json',
        S/'SINGLE2019_RISK_QUARANTINED_WINDOW_20261012_25/2019_RECOMMENDED_PSEUDO_LABEL.tif',
        S/'SINGLE2019_SATELLITE_OVERLAY_CASES_20261012_29/SATELLITE_OVERLAY_VERIFICATION_RECEIPT.json',
        S/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21/FROZEN_MANUAL_REFERENCE_BINDING.json',
        DATE,S2,COMP,Path('C:/Users/ljz52/.codex/attachments/5968216d-3bc3-4cd5-9701-fabd03f1fa8f/已粘贴的文本.txt')]
    required.extend(GT.with_suffix(ext) for ext in ['.shp','.shx','.dbf','.prj','.sbn','.sbx'])
    for p in required:protected[str(p)]=sha(p)
    OUT.mkdir()
    save_json(OUT/'FROZEN_RUN_PLAN.json',{'run':OUT.name,'scope':'EXISTING_256x256_ONLY_NO_CITYWIDE',
        'protected_sha256':protected,'large_file_stat':{str(GE):[GE.stat().st_size,GE.stat().st_mtime_ns]},
        'config':DEFAULT_CONFIG,'evaluation_denominator':65536,'evaluation_scope':'ALL_ORIGINAL_TARGET_PIXELS',
        'unresolved_baseline_evaluation':'UNCHANGED_RAW_GEOMETRY_ONLY_FOR_COMPARISON_NOT_ADMISSION',
        'no_GT_in_prediction':True,'maximum_depth':1,'final_holdout_available':False,
        'target_MAE':.10,'source_actual_date':'2018-11-01','nominal_experiment_year':2019})
    n=check_protected();regs=registry()
    save_json(OUT/'REGION_DEFINITION_AND_FROZEN_SPLIT.json',{'regions':[{'region_id':r.region_id,
        'support_geometry':mapping(r.support),'actual_date':r.actual_date,'kind':'EMPIRICAL_ALIGNMENT_REGION',
        'source_component_id':None,'same_viewing_angle':None,'role':'DEVELOPMENT_EXPOSED'} for r in regs],
        'parent_size_m':640,'minimum_child_size_m':320,'maximum_depth':1,
        'split_trigger':'PARENT_INTERNAL_CHECK_FAILS; FOUR_CHILDREN_PASS_AND_DIFFERENT_VECTORS',
        'known_provider_seams':'UNKNOWN_NOT_INFERRED_FROM_JPG_OR_DATE',
        'calibration_regions':[],'validation_regions':[],'final_holdout_regions':[],
        'why_no_independent_split':'All 4821 ROI reference roofs already development-exposed; camera components unverified.',
        'independent_test_status':'INSUFFICIENT_VALIDATION_REFERENCE','split_not_chosen_by_test_MAE':True})
    # Reconstruct original JPEG support with the existing correct GEID row convention.
    base=Path('F:/xian/亚米市区底图')
    mosaic=MosaicTileGrid.open(base/'xa190101.geid',GE,base/'xa190101/20')
    ll=Transformer.from_crs(32649,4326,always_xy=True);utm=Transformer.from_crs(4326,32649,always_xy=True)
    llroi=transform(ll.transform,ROI.segmentize(20));inv=~Affine(*mosaic.mosaic_transform)
    corners=[inv*(x,y) for x,y in [(llroi.bounds[0],llroi.bounds[1]),(llroi.bounds[2],llroi.bounds[3])]]
    c0,c1=[math.floor(v/256) for v in [min(p[0] for p in corners),max(p[0] for p in corners)]]
    r0,r1=[math.floor(v/256) for v in [min(p[1] for p in corners),max(p[1] for p in corners)]]
    dates={}
    with DATE.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['task_name']=='xa190101' and c0<=int(row['tile_x'])-mosaic.min_tile_x<=c1 and r0<=mosaic.max_tile_y-int(row['tile_y'])<=r1:
                key=(int(row['tile_x']),int(row['tile_y']))
                if key in dates and dates[key]!=row:raise ValueError('Conflicting source dates')
                dates[key]=row
    tiles=[];aff=Affine(*mosaic.mosaic_transform)
    for cc in range(c0,c1+1):
        for rr in range(r0,r1+1):
            x,y=mosaic.min_tile_x+cc,mosaic.max_tile_y-rr
            w=mosaic.window_for_tile(x,y);px,py,pw,ph=w.pixel_window
            llpoly=Polygon([aff*(a,b) for a,b in [(px,py),(px+pw,py),(px+pw,py+ph),(px,py+ph)]]).segmentize(.00005)
            support=transform(utm.transform,llpoly);clipped=support.intersection(ROI)
            if clipped.area<=1e-6:continue
            d=dates.get((x,y))
            if not d or d['date_conflict']!='0' or d['actual_capture_date']!='2018-11-01':raise ValueError('Actual acquisition mismatch')
            jpg=mosaic.source_path_for_tile(x,y)
            tiles.append({'download_tile_id':f'{x}:{y}:20','tile_x':x,'tile_y':y,'zoom':20,
                'source_JPG':str(jpg),'source_JPG_sha256':sha(jpg),'mosaic_window':w.pixel_window,
                'actual_date':d['actual_capture_date'],'date_evidence_source_log':d['source_log'],
                'date_evidence_line':d['line_number'],'support_geometry':mapping(clipped),
                'source_component_id':None,'source_component_kind':'UNKNOWN_CAMERA_COMPONENT',
                'same_viewing_angle':None,'support_role':'DOWNLOAD_PIXEL_SUPPORT_NOT_PHOTOGRAPHIC_REGION'})
    if len(tiles)!=1433:raise ValueError('Exact original support tile count changed')
    area=sum(shape(t['support_geometry']).area for t in tiles)
    if abs(area-ROI.area)>1e-3:raise ValueError('Source support union area mismatch')
    save_json(OUT/'SOURCE_COMPONENT_LEDGER.json',{'tiles':tiles,'tile_count':len(tiles),
        'true_camera_source_components_recovered':0,'empirical_alignment_regions':16,
        'compute_halo_policy':'HALO_ONLY_CONTEXT_NO_NEW_SOURCE_LABEL_SUPPORT',
        'same_date_implies_same_viewing_angle':False,'support_intersection_area_m2':area,
        'mosaic_grid':mosaic.summary(),'GT_input_to_inference':False})
    histories={}
    for p in required[:5]:
        histories[str(p)]=read_json(p) if p.suffix=='.json' else {'report_sha256':sha(p),'terminal_state':'G06_GEOMETRY_SOURCE_ADJUDICATION_INCONCLUSIVE'}
    dependencies={k:importlib.util.find_spec(k) is not None for k in ['arosics','geoarray','osgeo','scipy','cv2','numpy']}
    save_json(OUT/'DATA_AND_PROVENANCE_AUDIT.json',{'historical_baselines':histories,
        'historical_displacement_vectors_not_authorized_or_reused':True,'protected_inputs_verified':n,
        'source_components_recovered':{'JPGs':1433,'camera_components':0,'empirical_regions':16},
        'all_reference_roofs_development_exposed':4821,'manual_GT_role':'MANUAL_ROOF_OUTLINE_GT',
        'GT_role_user_confirmed':True,'GT_absolute_accuracy':'UNKNOWN','GT_precise_date':None,
        'GE_actual_date':'2018-11-01','experiment_nominal_year':2019,'time_compatibility_not_assumed':True,
        'target_raster':str(S2),'target_reference_role':'ANNUAL_RGB_REFERENCE_NOT_MODEL_PREDICTION',
        'runtime_dependencies':dependencies,'AROSICS_import_probe':'NOT_RUN_MISSING_AROSICS_GEOARRAY_OSGEO_SCIPY',
        'root_AGENTS_file':'ABSENT_USE_USER_SUPPLIED_INSTRUCTIONS','codegraph_index':False,
        'git_fixed_point_review':'UNAVAILABLE_EMPTY_GIT_METADATA','cloud_resources_started':False})
    print(json.dumps({'stage':'FROZEN','tiles':1433,'protected_inputs':n}),flush=True)

def resources():
    if (OUT/'REFERENCE_RESOURCE_AUDIT.json').exists():raise ValueError('Append-only resource audit already exists')
    rows=[]
    urls=[('AROSICS','https://raw.githubusercontent.com/GFZ/arosics/main/pyproject.toml'),
        ('AROSICS','https://raw.githubusercontent.com/GFZ/arosics/main/arosics/CoReg_local.py'),
        ('AROSICS','https://raw.githubusercontent.com/GFZ/arosics/main/arosics/Tie_Point_Grid.py'),
        ('DragOSM','https://raw.githubusercontent.com/likaiucas/DragOSM/main/README.md'),
        ('DragOSM','https://raw.githubusercontent.com/likaiucas/DragOSM/main/test_offset.py'),
        ('DragOSM','https://raw.githubusercontent.com/likaiucas/DragOSM/main/configs/DragOSM/dragosm_vit_b_osmrand_intensity_snd_512_omni_ISRA_200.py'),
        ('BONAI','https://raw.githubusercontent.com/jwwangchn/BONAI/master/README.md'),
        ('OBM','https://raw.githubusercontent.com/likaiucas/OBM/main/README.md'),
        ('RBF-SAM','https://raw.githubusercontent.com/Xuaner624/RBF-SAM/main/README.md'),
        ('RBF-SAM','https://raw.githubusercontent.com/Xuaner624/RBF-SAM/main/requirements.txt'),
        ('tdd','https://raw.githubusercontent.com/mattpocock/skills/main/skills/engineering/tdd/SKILL.md'),
        ('ponytail','https://raw.githubusercontent.com/DietrichGebert/ponytail/2961b52fc1fe6a5a87c7a959a79512e5c2f3ffe5/skills/ponytail/SKILL.md')]
    for name,url in urls:
        try:
            with urllib.request.urlopen(url,timeout=12) as r:body=r.read();status=r.status
            row={'name':name,'url':url,'HTTP_status':status,'response_sha256':hashlib.sha256(body).hexdigest(),'response_bytes':len(body)}
        except Exception as e:row={'name':name,'url':url,'HTTP_status':getattr(e,'code',None),'error':str(e)}
        rows.append(row);print(name,row.get('HTTP_status'),flush=True)
    save_json(OUT/'REFERENCE_RESOURCE_AUDIT.json',{'resources':rows,
        'GitHub_API_commit_enumeration':'NOT_RUN_IN_THIS_SCRIPT_NO_COMMIT_ID_INVENTED',
        'AROSICS':{'paper':'https://doi.org/10.3390/rs9070676','available_modules':['COREG','COREG_LOCAL','Tie_Point_Grid'],
            'purpose':'Global/local image co-registration; user masks and reliability/SSIM/RANSAC tie filters.',
            'local_code_uses':'GCP-backed image transformation, not independent roof-plane truth',
            'runtime':'NOT_RUN_DEPENDENCIES_UNAVAILABLE_NO_SHARED_ENVIRONMENT_MODIFIED'},
        'DragOSM':{'paper':'https://arxiv.org/abs/2509.17951','available_now':'README checked; executable availability follows actual HTTP records above',
            'use':'METHOD_REFERENCE_ONLY_NOT_REUSED_AS_WORKING_TRAINER'},
        'BONAI':{'paper':'https://doi.org/10.1109/TPAMI.2022.3162583','use':'LOFT/FOA offset modelling; MMDetection/bstool dependency; no inferred GE registration'},
        'OBM':{'paper':'https://doi.org/10.1109/TGRS.2024.3487652','use':'SAM/ROAM roof-to-footprint offset, not GE-to-reference correction',
            'runtime':'NOT_RUN; legacy torch/CUDA/MMDetection setup not installed'},
        'RBF-SAM':{'paper':'https://doi.org/10.1109/TGRS.2026.3709903','use':'Robust roof/footprint prompt offsets; not absolute positioning',
            'runtime':'NOT_RUN_OFFICIAL_CHECKPOINT_UNAVAILABLE; segmentation outside this task'},
        'installed_plugins_or_changed_dependencies':False})

def baseline():
    owned=['TRAINING_DATA_MANIFEST.json','A1_ALGORITHM_CHECKPOINT.json','GT_FREE_PREDICTION.json',
        'BASELINE_COMPARISON.json','LOCAL_ALIGNMENT_FIELD.gpkg','CORRECTED_ROOF_OBJECTS.gpkg','EXAMPLE_PLOT_DATA.npz']
    if any((OUT/n).exists() for n in owned):raise ValueError('Append-only baseline output already exists; use fresh child run')
    check_protected();policy=read_json(OUT/'FROZEN_RUN_PLAN.json');regs=registry();obs=objects()
    hypotheses=read_json(S/'SINGLE2019_ROOF_EVIDENCE_20261012_08/ROOF_CORRESPONDENCE_HYPOTHESES.json')['hypotheses']
    if len(hypotheses)!=267:raise ValueError('Original pairing denominator changed')
    # Existing inventory exposed in prior development; no final holdout is read.
    metric=Transformer.from_crs(4326,32649,always_xy=True)
    ll=Transformer.from_crs(32649,4326,always_xy=True)
    query=transform(ll.transform,ROI.segmentize(20)).bounds;references={}
    with fiona.open(GT) as s:
        for f in s.filter(bbox=query):
            g=transform(metric.transform,shape(f.geometry))
            if g.intersects(ROI):references[str(f.id)]=g
    if len(references)!=4821:raise ValueError('GT development exposure inventory changed')
    pairs=[]
    native_affine=None
    with rasterio.open(GE) as s:native_affine=s.transform
    for h in hypotheses:
        key=h['component_key'];fid=h['source_fid'];g=obs[key]
        if fid not in references:raise ValueError('Reference pairing lost its geometry')
        gb=transform(ll.transform,g).bounds;inv=~native_affine
        aa=inv*(gb[0],gb[3]);bb=inv*(gb[2],gb[1]);c0,r0=math.floor(aa[0]),math.floor(aa[1])
        pairs.append({'pair_id':key+'__'+fid,'source_component_id':None,'empirical_alignment_region':h['region_id'],
            'actual_date':'2018-11-01','roi_id':'CURRENT_256x256','roof_observation_id':key,'roof_gt_id':fid,
            'image_patch':{'raster_path':str(GE),'native_pixel_window':[c0,r0,math.ceil(bb[0])-c0,math.ceil(bb[1])-r0],
                'role':'SOURCE_WINDOW_REFERENCE_NOT_EXPORTED_FOR_INELIGIBLE_TRAINING'},
            'candidate_geometry':mapping(g),'target_geometry':mapping(references[fid]),
            'label_dx_m':None,'label_dy_m':None,'diagnostic_shape_vector_m':h['roof_to_reference_candidate_m'],
            'label_semantics':'ROOF_OBSERVATION_TO_PLANIMETRIC_ROOF_REFERENCE',
            'identity_status':'GEOMETRIC_CORRESPONDENCE_HYPOTHESIS','time_compatible':None,
            'reference_quality':None,'reference_role':'MANUAL_ROOF_OUTLINE_GT',
            'source_support':mapping(next(r.support for r in regs if r.region_id==h['region_id'])),
            'training_split':'DEVELOPMENT_EXPOSED_NOT_SUPERVISION_ELIGIBLE'})
    fit=fit_local_alignment(pairs,references,regs,policy['config'])
    save_json(OUT/'TRAINING_DATA_MANIFEST.json',{'pairs':pairs,'pair_count':267,'eligible_pairs':fit['supervision_eligible_pairs'],
        'supervised_displacement_labels_created':0,'all_difficult_hypotheses_retained':True,
        'synthetic_perturbations_used':False,'fit_result':fit})
    save_json(OUT/'A1_ALGORITHM_CHECKPOINT.json',fit['checkpoint'])
    def read_roi(p):
        with rasterio.open(p) as s:
            if Grid(s.transform,s.width,s.height,str(s.crs))!=GRID:raise ValueError('Grid drift')
            return s.read()
    source=np.moveaxis(read_roi(S/'SINGLE2019_SUPPORT_REPAIR_20261012_07/2019_BOUNDED_GE_RGB_CACHE.tif'),0,-1)
    with rasterio.open(S2) as s:
        w=Window(1516,1765,256,256);target=np.moveaxis(s.read([1,2,3],window=w),0,-1)
        valid=np.all(s.read_masks([1,2,3],window=w)>0,axis=0)
        if s.window_transform(w)!=GRID.affine:raise ValueError('Reference grid mismatch')
    start=time.perf_counter()
    predicted=predict_local_alignment(source,obs,regs,{'grid':GRID,'reference_rgb':target,'reference_valid':valid},fit['checkpoint'])
    save_json(OUT/'GT_FREE_PREDICTION.json',predicted)
    # Actual second inference run with ALL Python-level filesystem opens denied.
    from unittest.mock import patch
    with patch('builtins.open',side_effect=AssertionError('GT-free inference attempted a file read')):
        replay=predict_local_alignment(source,obs,regs,{'grid':GRID,'reference_rgb':target,'reference_valid':valid},fit['checkpoint'])
    if digest(predicted)!=digest(replay):raise ValueError('GT-free replay changed')
    save_json(OUT/'GT_FREE_INFERENCE_REPLAY_RECEIPT.json',{'status':'PASS_CALL_LEVEL_NO_GT_ARGUMENT_OR_FILE_READ',
        'filesystem_guard':'builtins.open denied; in-memory pure function, not an OS sandbox',
        'prediction_sha256':sha(OUT/'GT_FREE_PREDICTION.json'),
        'checkpoint_sha256':sha(OUT/'A1_ALGORITHM_CHECKPOINT.json'),
        'inference_code_sha256':sha(ROOT/'src/xian_r5/alignment/adaptive_local.py'),
        'source_RGB_cache_sha256':sha(S/'SINGLE2019_SUPPORT_REPAIR_20261012_07/2019_BOUNDED_GE_RGB_CACHE.tif'),
        'roof_candidates_sha256':sha(COMP),'reference_RGB_sha256':sha(S2),
        'arrays_equal_replayed_prediction':True,'GT_file_dependencies_in_inference_function':False,
        'training_evaluation_code_separately_reads_exposed_GT':True})
    run_seconds=time.perf_counter()-start
    print('GT-free predictions completed',len(predicted['regions']),flush=True)
    reference=read_roi(S/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21/2019_MANUAL_REFERENCE_ROOF_FRACTION_10m.tif')[0]
    raw=read_roi(S/'SINGLE2019_20261012_06/2019_ROOF_FRACTION_RAW.tif')[0]
    reference_path=S/'SINGLE2019_MANUAL_GT_REFERENCE_20261012_21/2019_MANUAL_REFERENCE_ROOF_FRACTION_10m.tif'
    with rasterio.open(reference_path) as src:
        if not np.all(src.read_masks(1)>0):raise ValueError('Reference has unavailable cells; frozen denominator cannot shrink')
    if any(not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in [raw,reference]):raise ValueError('Original full fraction denominator unavailable')
    reconstructed,zero_conflicts=exact_complete_fraction(obs)
    max_difference=float(np.abs(reconstructed-raw).max())
    if max_difference>1e-6:raise ValueError('M0 geometry/fraction equivalence failed')
    save_json(OUT/'M0_INPUT_EQUIVALENCE_AND_VALIDITY.json',{'zero_transform_replay_max_fraction_difference':max_difference,
        'denominator':65536,'reference_native_masks_all_valid':True,'all_reference_fractions_finite_in_unit_interval':True,
        'GT_background_completeness_verified':False,'status':'PASS_DEVELOPMENT_NUMERIC_VALIDITY_NOT_GT_COMPLETENESS'})
    checkpoint_hash=digest(fit['checkpoint']);results={};geometry_sets={};transport_audits={};all_regions=predicted['regions']
    for name in ['M0','M1','M2']:
        if name=='M0':fractions=raw.astype(float);geometry_sets[name]={k:g.intersection(ROI) for k,g in obs.items()};transport_audits[name]={}
        else:
            selected=([r for r in all_regions if r['parent_region_id'] is None and r['selected_vector_m'] is not None]
                if name=='M1' else predicted['diagnostic_selected_regions'])
            rows=[SourceRegion(r['region_id'],r['actual_date'],shape(r['support_geometry']),tuple(r['selected_vector_m']),digest(r)) for r in selected]
            if rows:rows,_=coalesce_equal_field_regions(rows)
            moved=source_partition_transport(obs,rows,ROI) if rows else {'candidate_geometries':{},'objects':{}}
            # Unknown geometry is kept unshifted ONLY in the fixed diagnostic evaluator.
            evaluated={k:moved['candidate_geometries'].get(k,g.intersection(ROI)) for k,g in obs.items()}
            evaluated={k:g for k,g in evaluated.items() if not g.is_empty}
            fractions,overlap_pixels=exact_complete_fraction(evaluated)
            geometry_sets[name]=evaluated;transport_audits[name]=moved['objects']
        export_raster(OUT/(name+'_ANNOTATION_FRACTION_DIAGNOSTIC.tif'),fractions[None],GRID,
            ['FULL_PIXEL_ROOF_FRACTION_DEVELOPMENT_ONLY'],root=OUT,nodata=NODATA)
        error=np.abs(fractions-reference);related=reference>0;edge=(reference>0)&(reference<1)
        original_n=65536
        if not np.isfinite(fractions).all() or np.any((fractions<0)|(fractions>1+1e-6)) or error.size!=original_n:
            raise ValueError('Diagnostic denominator or fraction validity violated')
        per_regions=[]
        for r in regs:
            mask=rasterize([(mapping(r.support),1)],out_shape=raw.shape,transform=GRID.affine).astype(bool)
            per_regions.append({'region_id':r.region_id,'denominator':int(mask.sum()),'MAE':float(error[mask].sum()/mask.sum()),
                'building_related_n':int((mask&related).sum()),'edge_n':int((mask&edge).sum())})
        pair_metrics=[]
        for h in hypotheses:
            g=geometry_sets[name][h['component_key']];ref=references[h['source_fid']]
            pair_metrics.append({'roof_observation_id':h['component_key'],'roof_gt_id':h['source_fid'],
                'identity_proven':False,'centroid_error_m':g.centroid.distance(ref.centroid),
                'boundary_Hausdorff_error_m':g.boundary.hausdorff_distance(ref.boundary),
                'IoU':g.intersection(ref).area/g.union(ref).area})
        conflict_keys=[k for k,row in transport_audits[name].items() if row['status']=='UNRESOLVED_SOURCE_SEAM']
        results[name]={'all_pixel_numerator_absolute_fraction_error':float(error.sum()),'all_pixel_denominator':65536,
            '10M_ROOF_FRACTION_MAE':float(error.sum()/65536),'building_related_pixels':int(related.sum()),
            'building_related_MAE':float(error[related].mean()),'edge_pixel_count':int(edge.sum()),
            '10M_EDGE_PIXEL_MAE':float(error[edge].mean()),'per_empirical_region':per_regions,'pair_metrics':pair_metrics,
            'pair_denominator':267,'independent_test':False,'identity_correspondences_certified':False,
            'background_completeness_verified':False,'unknowns_dropped_from_MAE':0,
            'diagnostic_complete_objects_moved':sum(row['status'].startswith('CANDIDATE_') for row in transport_audits[name].values()),
            'cross_support_unresolved_objects':len(conflict_keys),'correction_coverage_rate_qualified':0.}
        print(name,results[name]['10M_ROOF_FRACTION_MAE'],results[name]['diagnostic_complete_objects_moved'],flush=True)
    save_json(OUT/'BASELINE_COMPARISON.json',{'baselines':results,
        'scope':'INTERNAL_DEVELOPMENT_ANNOTATION_FRACTION_NOT_INDEPENDENT_ACCURACY',
        'A0':'M0 and prior four-cell candidate audited but not promoted/reused as new region estimates',
        'A1':'GT-free supported non-roof image-grid matching reused from existing R5',
        'AROSICS':'NOT_RUN_DEPENDENCY_GATE','A2':'NOT_RUN_NO_QUALIFIED_PAIRS','A3':'NOT_RUN_NO_A2_HOLDOUT_EVIDENCE',
        'baseline_selected_by_test_metric':False,'prediction_two_runs_seconds':run_seconds})
    save_json(OUT/'OBJECT_TRANSPORT_AND_SEAM_AUDIT.json',transport_audits)
    # Qualification is distinct: no validated roof-plane transfer has been established.
    quality=np.ones((4,256,256),np.uint16);quality[0]=2;quality[1:]=0
    export_raster(OUT/'ALIGNMENT_QUALITY.tif',quality,GRID,
        ['UNRESOLVED_CODE_2','ROOF_PLANE_TRANSFER_VALIDATED','VERIFIED_GROUND_REFERENCE','TRAINING_ELIGIBLE'],root=OUT)
    export_raster(OUT/'GT10M_QUALITY.tif',quality,GRID,
        ['UNKNOWN_CODE_2','GEOMETRY_ELIGIBLE','TIME_COMPATIBLE','TRAINING_WEIGHT'],root=OUT)
    export_raster(OUT/'BUILDING_FRACTION_10M.tif',np.full((1,256,256),NODATA,np.float32),GRID,
        ['QUALIFIED_ROOF_FRACTION_UNKNOWN'],root=OUT,nodata=NODATA)
    region_map=np.zeros((256,256),np.uint16)
    for i,r in enumerate(regs,1):region_map[rasterize([(mapping(r.support),1)],out_shape=raw.shape,transform=GRID.affine).astype(bool)]=i
    export_raster(OUT/'SOURCE_COMPONENT_MAP.tif',region_map[None],GRID,
        ['EMPIRICAL_REGION_ID_NOT_CAMERA_COMPONENT_ID'],root=OUT)
    high=GRID.nested(10)
    export_raster(OUT/'CORRECTED_ROOF_CANDIDATE.tif',np.full((1,high.height,high.width),255,np.uint8),high,
        ['UNRESOLVED_NO_VALIDATED_ROOF_PLANE_TRANSFER'],root=OUT,nodata=255)
    with fiona.open(OUT/'LOCAL_ALIGNMENT_FIELD.gpkg','w',driver='GPKG',crs='EPSG:32649',layer='diagnostic_regions',
        schema={'geometry':'Unknown','properties':{'region_id':'str','parent_id':'str','dx_m':'float','dy_m':'float','qualified':'int','status':'str'}}) as dst:
        for r in all_regions:
            v=r['selected_vector_m'];dst.write({'geometry':r['support_geometry'],'properties':{'region_id':r['region_id'],
                'parent_id':r['parent_region_id'],'dx_m':v[0] if v else None,'dy_m':v[1] if v else None,'qualified':0,'status':'UNRESOLVED_ROOF_PLANE'}})
    schema={'geometry':'Unknown','properties':{'observation_id':'str','status':'str','physical_id':'str'}}
    with fiona.open(OUT/'CORRECTED_ROOF_OBJECTS.gpkg','w',driver='GPKG',crs='EPSG:32649',layer='qualified_corrected_roofs',schema=schema):pass
    with fiona.open(OUT/'CORRECTED_ROOF_OBJECTS.gpkg','w',driver='GPKG',crs='EPSG:32649',layer='unresolved_original_roofs',schema=schema) as dst:
        for k,g in obs.items():dst.write({'geometry':mapping(g),'properties':{'observation_id':k,'status':'UNRESOLVED_NOT_CORRECTED','physical_id':None}})
    np.savez_compressed(OUT/'EXAMPLE_PLOT_DATA.npz',ge_rgb=source,sentinel_rgb=target,reference_fraction=reference,
        raw_fraction=raw,empirical_regions=region_map)
    save_json(OUT/'LOCAL_ALIGNMENT_EVALUATION.json',{
        'TRAINING_DATA_AUDITED':True,'SOURCE_COMPONENTS_RECOVERED':{'JPGs':1433,'CAMERA_COMPONENTS':0},
        'ADAPTIVE_LOCAL_REGIONS_DEFINED':{'initial':16,'examined':len(all_regions),'diagnostic_selected':len(predicted['diagnostic_selected_regions'])},'LOCAL_ALIGNMENT_BASELINE_COMPLETED':True,
        'LOCAL_ALIGNMENT_MODEL_TRAINED':False,'GROUND_REFERENCE_AVAILABLE':False,'ROOF_GT_USED_FOR_TRAINING':False,
        'GT_FREE_INFERENCE_DEMONSTRATED':True,'GT_FREE_ROOF_PLANE_CORRECTION_VALIDATED':False,
        'INDEPENDENT_TEST_MAE':None,'INDEPENDENT_TEST_STATUS':'INSUFFICIENT_VALIDATION_REFERENCE',
        'INDEPENDENT_TEST_PIXEL_DENOMINATOR':0,'EDGE_PIXEL_MAE':None,
        'INTERNAL_DEVELOPMENT_EDGE_MAE':{k:v['10M_EDGE_PIXEL_MAE'] for k,v in results.items()},
        'CORRECTION_COVERAGE_RATE':0.,'HIGH_RISK_OR_UNRESOLVED_PIXEL_RATE':1.,'SEAM_CONFLICT_RATE':None,
        'SEAM_CONFLICT_RATE_REASON':'True photographic source components unknown; computational crossing counts separately reported',
        'UNRESOLVED_INITIAL_REGIONS':16,'UNRESOLVED_REGIONS':16,
        'UNIFIED_CORRECTED_TIF_GENERATED':False,'UNKNOWN_QUALIFICATION_CONTAINER_TIFS_GENERATED':True,
        '10M_FRACTION_OUTPUT_GENERATED':True,'FRACTION_OUTPUT_ROLE':'DIAGNOSTIC_M0_M1_M2_PLUS_UNKNOWN_QUALIFIED_CONTAINER',
        'READY_FOR_GT10M_TRAINING':False,'ABSOLUTE_POSITION_ACCURACY':'UNKNOWN',
        'CLOUD_GPU_REQUIRED':'NOT_DETERMINED_DATA_AND_VALIDATION_GATE_FIRST',
        'new_formal_reviews':0,'new_formal_accepts':0,'new_ground_footprints':0,
        'GT_files_read_inside_prediction':0,'historical_fixed_vectors_reused':False})
    save_json(OUT/'GPU_READINESS_AND_RESOURCE_REQUEST.json',{'CLOUD_GPU_REQUIRED':'NOT_DETERMINED',
        'qualified_training_pairs':0,'GT_validation_regions':0,'model_architecture_selected':False,
        'GPU_model':None,'peak_VRAM_GB':None,'batch_size':None,'training_time':None,'cost':None,
        'why_no_fabricated_estimates':'No qualified data or justified A2 architecture; no evidence a GPU is needed.',
        'paid_instances_started':0,'large_GPU_training_started':False,'user_confirmation_required_before_cloud':True,
        'upload_manifest':[],'resume_checkpoint':'Not applicable to untrained deterministic A1 config; A2 must specify recoverable checkpoints before authorization'})
    print(json.dumps({'stage':'BASELINES_COMPLETE','qualified_regions':0,'supervision_pairs':0}),flush=True)

def close():
    if (OUT/'RUN_RECEIPT.json').exists():raise ValueError('Append-only close receipt already exists')
    replay=read_json(OUT/'GT_FREE_INFERENCE_REPLAY_RECEIPT.json')
    if replay['prediction_sha256']!=sha(OUT/'GT_FREE_PREDICTION.json') or replay['inference_code_sha256']!=sha(ROOT/'src/xian_r5/alignment/adaptive_local.py'):
        raise ValueError('GT-free replay evidence drift')
    print('Close: replay receipt bound',flush=True)
    n=check_protected();print('Close: protected inputs verified',n,flush=True)
    stream=io.StringIO();sys.path.insert(0,str(ROOT/'tests'))
    names=sorted(p.stem for p in (ROOT/'tests').glob('test_xian_*.py') if p.stem!='test_xian_additive_post_inference')
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    print('Close: related suite loaded',suite.countTestCases(),flush=True)
    tr=unittest.TextTestRunner(stream=stream).run(suite)
    write_new(OUT/'REGRESSION_TEST_LOG.txt',stream.getvalue())
    if not tr.wasSuccessful():raise ValueError('Regression gate failed; see test log')
    for p in OUT.glob('*.tif'):
        with rasterio.open(p) as s:
            expected=GRID.nested(10) if p.name=='CORRECTED_ROOF_CANDIDATE.tif' else GRID
            if Grid(s.transform,s.width,s.height,str(s.crs))!=expected:raise ValueError('Output grid drift')
    with fiona.open(OUT/'CORRECTED_ROOF_OBJECTS.gpkg',layer='qualified_corrected_roofs') as s:
        if len(s)!=0:raise ValueError('Unqualified correction promoted')
    save_json(OUT/'RUN_RECEIPT.json',{'status':'ENGINEERING_BASELINE_DELIVERED_NOT_INDEPENDENTLY_VALIDATED',
        'tests_passed':tr.testsRun,'protected_inputs_checked':n,'protected_inputs_changed':0,
        'GT_free_inference_filesystem_denial_replay':replay['status'],'source_JPG_hashes_recorded':1433,
        'formal_state_unchanged':True,'original_recommended_training_pixels_unchanged':9866,
        'code_sha256':{str(p):sha(p) for p in [Path(__file__),ROOT/'src/xian_r5/alignment/adaptive_local.py',ROOT/'tests/test_xian_adaptive_local_alignment.py']},
        'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'tests_passed':tr.testsRun,'protected_inputs_unchanged':n}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','resources','baseline','close'],required=True)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.output:
        requested=args.output.resolve()
        if requested.parent!=OUT.resolve():raise ValueError('Only fresh bounded child outputs allowed')
        if args.stage=='baseline':
            if requested.exists():raise ValueError('Fresh child directory required')
            requested.mkdir();shutil.copy2(OUT/'FROZEN_RUN_PLAN.json',requested/'FROZEN_RUN_PLAN.json')
        elif not requested.exists():raise ValueError('Child baseline missing')
        OUT=requested
    try:globals()[args.stage]()
    except Exception as error:
        if OUT.exists():
            failure=OUT/('FAILURE_'+args.stage+'_'+str(time.time_ns())+'.json')
            save_json(failure,{'stage':args.stage,'error':str(error),'traceback':traceback.format_exc(),'formal_promotions':0})
        raise
