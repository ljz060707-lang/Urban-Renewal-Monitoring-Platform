"""CPU-only append-only numerical closure. Truth is scorer-only, never estimator input."""
import argparse
import hashlib
import json
import os
import sys
import time
import io
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
OLD=P/'XIAN_R5_ALIGNMENT_FEATURE_RECOVERY_20261013'
OUT=P/'A_LOCAL_ALIGNMENT_NUMERICAL_AND_CHECK_CLOSURE_20261013'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name,value):
    def scalar(value):
        if hasattr(value,'tolist'):return value.tolist()
        raise TypeError('Non-serializable evidence type: '+type(value).__name__)
    # Serialize BEFORE creating a destination, preserving even failed attempts.
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(payload)


def freeze():
    if OUT.exists():raise ValueError('Fresh append-only run directory required')
    prior=read(OLD/'FROZEN_RUN_PLAN.json')
    protected=dict(prior['protected_sha256'])
    for path,expected in protected.items():
        if sha(path)!=expected:raise ValueError('Old protected input drift: '+path)
    for directory in [OLD]:
        for folder,dirs,files in os.walk(directory):
            dirs[:]=[d for d in dirs if d not in ['isolated_arosics','plot_cache']]
            for name in files:
                path=Path(folder)/name;protected[str(path)]=sha(path)
    for folder in ['src/xian_r5','tools']:
        for path in (ROOT/folder).glob('**/*.py'):
            if path.name!='xian_local_numerical_closure.py':protected[str(path)]=sha(path)
    for path in (ROOT/'tests').glob('test_xian_*.py'):protected[str(path)]=sha(path)
    env=OLD/'isolated_arosics/env/Lib/site-packages'
    for package in ['arosics','geoarray','py_tools_ds']:
        for path in (env/package).rglob('*.py'):protected[str(path)]=sha(path)
    OUT.mkdir()
    save('FROZEN_RUN_PLAN.json',dict(task='A_LOCAL_ALIGNMENT_NUMERICAL_AND_CHECK_CLOSURE',
        protected_sha256=protected,large_file_stat=prior['large_file_stat'],
        baseline_authority=read(OLD/'AUTHORITATIVE_DELIVERY.json'),
        pixel_denominator=65536,region_denominator=16,actual_GE_date='2018-11-01',
        formal_numerical_threshold_m=2.0,module_B_mutation_allowed=False,
        existing_environment=str(OLD/'isolated_arosics/env'),
        no_model_training=True,no_old_vectors_reused=True,absolute_position_accuracy='UNKNOWN'))
    print('Frozen',len(protected),'input hashes',flush=True)


def estimate_arosics(reference,moving,ref_gt,mov_gt,*,ws=128,resamp='cubic',max_iter=5,binary_ws=True):
    import numpy as np
    from osgeo import osr
    from geoarray import GeoArray
    from arosics import COREG
    srs=osr.SpatialReference();srs.ImportFromEPSG(32649)
    a=GeoArray(reference,geotransform=ref_gt,projection=srs.ExportToWkt(),nodata=0)
    b=GeoArray(moving,geotransform=mov_gt,projection=srs.ExportToWkt(),nodata=0)
    cr=COREG(a,b,ws=(ws,ws),max_shift=3,max_iter=max_iter,CPUs=1,q=True,
             resamp_alg_calc=resamp,binary_ws=binary_ws)
    trace=[];original_cps=cr._calc_shifted_cross_power_spectrum
    def capture(*args,**kwargs):
        spectrum=original_cps(*args,**kwargs)
        if spectrum is not None:
            peak=cr._get_peakpos(spectrum);x,y=cr._get_shifts_from_peakpos(peak,spectrum.shape)
            trace.append(dict(fft_shape=list(spectrum.shape),integer_correction_px=[int(x),int(y)],
                              spectrum_peak=float(spectrum.max())))
        return spectrum
    cr._calc_shifted_cross_power_spectrum=capture
    cr.calculate_spatial_shifts()
    def finite(v):return None if v is None or not np.isfinite(v) else float(v)
    return dict(success=bool(cr.success),correction_m=[finite(cr.x_shift_map),finite(cr.y_shift_map)],
        output_shift_px=[finite(cr.x_shift_px),finite(cr.y_shift_px)],
        reliability=finite(cr.shift_reliability),ssim_orig=finite(cr.ssim_orig),ssim_after=finite(cr.ssim_deshifted),
        matching_grid=cr.grid2use,window_bounds=list(cr.matchBox.mapPoly.bounds),
        effective_window_px=list(cr.matchBox.imDimsYX),fft_shape=list(cr.fft_win_size_YX),
        phase_trace=trace,errors=[str(e) for e in cr.tracked_errors])


def replay_old():
    import numpy as np
    from scipy.ndimage import gaussian_filter,shift
    if (OUT/'OLD_FIXTURE_REPLAY.json').exists():raise ValueError('Old replay already recorded')
    rng=np.random.default_rng(130)
    reference=(50+gaussian_filter(rng.normal(0,30,(256,256)),1.2)).astype('float32')
    moving=shift(reference,(-.75,1.25),order=3,mode='constant',cval=0)
    gt=(308460,10,0,3794750,0,-10)
    with (OUT/'OLD_FIXTURE_RECONSTRUCTED.npz').open('xb') as stream:
        np.savez_compressed(stream,reference=reference,moving=moving,reference_gt=np.array(gt),moving_gt=np.array(gt))
    save('OLD_FIXTURE_GENERATOR_TRUTH.json',dict(content_shift_col_row=[1.25,-.75],
        reference_gt=list(gt),moving_gt=list(gt),expected_correction_m=[-12.5,-7.5],
        seed=130,interpolation='scipy.ndimage.shift(order=3,prefilter=True)',boundary='constant zero',
        recipe_reconstructed_not_original_saved_input=True,original_input_arrays_not_saved=True,
        estimator_does_not_receive_truth=True))
    records=[];old=read(OLD/'AROSICS_SYNTHETIC_TEST.json')['rows']
    for k,(a,b,truth) in enumerate([(reference,moving,[-12.5,-7.5]),(moving,reference,[12.5,7.5])]):
        result=estimate_arosics(a,b,gt,gt)
        result.update(direction='FORWARD' if k==0 else 'REVERSE',expected_correction_m=truth,
            estimation_error_m=float(np.linalg.norm(np.array(result['correction_m'])-truth)),
            old_output_replay_difference_m=float(np.linalg.norm(np.array(result['correction_m'])-
                [old[k]['dx_map_m'],old[k]['dy_map_m']])))
        records.append(result)
    save('OLD_FIXTURE_REPLAY.json',dict(rows=records,
        numerical_gate='PASS' if all(r['estimation_error_m']<=2 for r in records) else 'FAIL',
        old_failure_not_reclassified=True,code_sha256=sha(__file__),fixture_sha256=sha(OUT/'OLD_FIXTURE_RECONSTRUCTED.npz')))
    print(json.dumps(records),flush=True)
    if any(r['estimation_error_m']>2 for r in records):
        raise AssertionError('Old fixture red feedback loop: IMAGE_REGISTRATION_ESTIMATION_ERROR exceeds 2 m')


def probes():
    import numpy as np
    from scipy.ndimage import shift
    from scipy.fft import fftn,ifftn
    from scipy.ndimage import fourier_shift
    fixture=np.load(OUT/'OLD_FIXTURE_RECONSTRUCTED.npz')
    ref=fixture['reference'];mov=fixture['moving'];gt=tuple(fixture['reference_gt'])
    truth=np.array([-12.5,-7.5]);rows=[]
    variants=[('OLD_REPLAY',ref,mov,dict()),
        ('WINDOW_64',ref,mov,dict(ws=64)),('WINDOW_256',ref,mov,dict(ws=256)),
        ('NO_BINARY_WINDOW_SHRINK',ref,mov,dict(binary_ws=False)),
        ('CALC_BILINEAR',ref,mov,dict(resamp='bilinear')),
        ('SOURCE_SHIFT_LINEAR',ref,shift(ref,(-.75,1.25),order=1,mode='constant',cval=0),dict()),
        ('SOURCE_SHIFT_WRAP',ref,shift(ref,(-.75,1.25),order=3,mode='wrap'),dict()),
        ('SOURCE_SHIFT_FOURIER_PERIODIC',ref,ifftn(fourier_shift(fftn(ref),(-.75,1.25))).real.astype('float32'),dict()),
        ('OLD_MEAN_SUBTRACTED',ref-ref.mean(),mov-mov.mean(),dict())]
    for name,a,b,kwargs in variants:
        try:
            result=estimate_arosics(a,b,gt,gt,**kwargs)
            error=float(np.linalg.norm(np.array(result['correction_m'])-truth))
            result.update(variant=name,estimation_error_m=error,numerical_gate=error<=2)
        except Exception as e:result=dict(variant=name,error=str(e),numerical_gate=False)
        rows.append(result);print(name,result,flush=True)
    save('SINGLE_FACTOR_PROBES.json',dict(rows=rows,truth_scorer_only=True,
        changing_one_factor_at_a_time=True,formal_threshold_m=2,
        no_library_code_changes=True,old_failed_fixture_not_reclassified=True))


def controls():
    import numpy as np
    from scipy.ndimage import gaussian_filter,shift,map_coordinates
    from scipy.optimize import minimize
    from affine import Affine
    sys.path.insert(0,str(ROOT/'src'))
    from xian_r5.alignment.supported_ground_features import estimate_supported_translation
    from xian_r5.alignment.base_tie import image_shift_to_projected_correction
    spec=[('ZERO',(0.,0.)),('INTEGER',(1.,-1.)),('SUBPIXEL_X',(.25,0.)),
          ('SUBPIXEL_Y',(0.,-.75)),('BOTH',(1.25,-.75))]
    protocol=dict(case_shifts_col_row=[dict(case=k,shift=list(s)) for k,s in spec],
         cases_use_same_reference=True,seed=713,reference_gsd_m=10,window_px=128,
         official_arosics_parameters=dict(max_shift=3,CPUs=1,binary_ws=True,resamp_alg_calc='cubic'),
         original_fixture_remains_mandatory_fail_case=True,
         numerical_gate_definition='ALL_MANDATORY_AROSICS_ERRORS<=2m_AND_CONVENTIONS_PASS',
         NCC_baseline='EXISTING_INTEGER_SEARCH_NOT_A_SUBPIXEL_ESTIMATOR',
         v2_fixture_difference='BROADBAND_TEXTURE_CONTROL; OLD_GENERATOR_NOT_ASSUMED_WRONG',
         cross_gsd='AREA_DOWNSAMPLE_2m_REFERENCE_TO_10m; MOVING_NATIVE_2m',
         known_truth_separate_from_estimator=True)
    save('FROZEN_SYNTHETIC_PROTOCOL.json',protocol)
    rng=np.random.default_rng(713);base=(100+rng.normal(0,20,(256,256))).astype('float32')
    gt=(308460,10,0,3794750,0,-10);records=[]
    for name,(dx,dy) in spec:
        moving=shift(base,(dy,dx),order=3,mode='constant',cval=0)
        with (OUT/(name+'_SAME_GSD_INPUT.npz')).open('xb') as stream:
            np.savez_compressed(stream,reference=base,moving=moving,reference_gt=np.array(gt),moving_gt=np.array(gt))
        truth=np.array([-dx*10,dy*10])
        for direction,a,b,expected in [('FORWARD',base,moving,truth),('REVERSE',moving,base,-truth)]:
            result=estimate_arosics(a,b,gt,gt)
            result.update(case=name,GSD='SAME_10M',direction=direction,expected_correction_m=expected.tolist(),
                IMAGE_REGISTRATION_ESTIMATION_ERROR=float(np.linalg.norm(np.array(result['correction_m'])-expected)))
            records.append(result)
            rgb1=np.repeat(a[:,:,None],3,2);rgb2=np.repeat(b[:,:,None],3,2);mask=np.ones((256,256),bool)
            ncc=estimate_supported_translation(rgb1,rgb2,mask,mask,mask,radius=3,min_samples=100)
            best=ncc['best'];vector=image_shift_to_projected_correction(best['dx_px'],best['dy_px'],tuple(Affine.from_gdal(*gt))) if best else None
            records.append(dict(algorithm='NCC_EXISTING_INTEGER',case=name,GSD='SAME_10M',direction=direction,
                 correction_m=vector,expected_correction_m=expected.tolist(),
                 IMAGE_REGISTRATION_ESTIMATION_ERROR=float(np.linalg.norm(np.array(vector)-expected)) if vector is not None else None,
                 result=ncc,precision_claim=False))
    # Independent generator check: fit a two-parameter cubic sampling model,
    # starting at zero; it never receives or reads generator truth.
    z=np.load(OUT/'OLD_FIXTURE_RECONSTRUCTED.npz');rr,cc=np.mgrid[64:192,64:192]
    def objective(offset):
        predicted=map_coordinates(z['reference'],[rr-offset[1],cc-offset[0]],order=3,mode='constant',cval=0)
        return float(np.mean((predicted-z['moving'][64:192,64:192])**2,dtype='float64'))
    independent=minimize(objective,[0.,0.],method='Powell',bounds=[(-3,3),(-3,3)],options={'xtol':1e-6,'ftol':1e-10})
    vector=[-independent.x[0]*10,independent.x[1]*10]
    save('GENERATOR_INDEPENDENT_RECOVERY_CHECK.json',dict(algorithm='CUBIC_TWO_PARAMETER_LEAST_SQUARES_CONTROL_NOT_AROSICS',
        truth_not_input=True,initial_parameters=[0,0],observed_content_shift_px=independent.x.tolist(),
        recovered_correction_m=vector,mean_squared_sampling_error=objective(independent.x),
        estimation_error_m=float(np.linalg.norm(np.array(vector)-[-12.5,-7.5])),
        scipy_shift_map_coordinates_consistency='DIRECT_NUMERIC_CHECK',not_a_real_scene_approval=True))
    high=(100+gaussian_filter(rng.normal(0,20,(1280,1280)),2.5)).astype('float32')
    low=high.reshape(256,5,256,5).mean((1,3));highmov=shift(high,(-3.75,6.25),order=3,mode='constant',cval=0)
    highgt=(308460,2,0,3794750,0,-2)
    with (OUT/'CROSS_GSD_INPUT.npz').open('xb') as stream:
        np.savez_compressed(stream,reference=low,moving=highmov,reference_gt=np.array(gt),moving_gt=np.array(highgt))
    for direction,a,b,ag,bg,expected in [('FORWARD',low,highmov,gt,highgt,[-12.5,-7.5]),
         ('REVERSE',highmov,low,highgt,gt,[12.5,7.5])]:
        result=estimate_arosics(a,b,ag,bg)
        result.update(algorithm='AROSICS_COREG',case='BOTH',GSD='CROSS_2M_10M',direction=direction,
            expected_correction_m=expected,IMAGE_REGISTRATION_ESTIMATION_ERROR=float(np.linalg.norm(np.array(result['correction_m'])-expected)))
        records.append(result)
    commonmoving=highmov.reshape(256,5,256,5).mean((1,3))
    mask=np.ones((256,256),bool)
    for direction,a,b,expected in [('FORWARD',low,commonmoving,[-12.5,-7.5]),('REVERSE',commonmoving,low,[12.5,7.5])]:
        ncc=estimate_supported_translation(np.repeat(a[:,:,None],3,2),np.repeat(b[:,:,None],3,2),mask,mask,mask,radius=3,min_samples=100)
        best=ncc['best'];vector=image_shift_to_projected_correction(best['dx_px'],best['dy_px'],tuple(Affine.from_gdal(*gt))) if best else None
        records.append(dict(algorithm='NCC_EXISTING_INTEGER_AREA_COMMON_GRID',case='BOTH',GSD='CROSS_2M_10M',direction=direction,
            correction_m=vector,expected_correction_m=expected,
            IMAGE_REGISTRATION_ESTIMATION_ERROR=float(np.linalg.norm(np.array(vector)-expected)) if vector else None,
            result=ncc,interpolation_does_not_create_information=True))
    for row in records:
        row.setdefault('algorithm','AROSICS_COREG')
        row['numerical_gate_2m']=row['IMAGE_REGISTRATION_ESTIMATION_ERROR'] is not None and row['IMAGE_REGISTRATION_ESTIMATION_ERROR']<=2
    save('AROSICS_NCC_COMPARISON.json',dict(rows=records,all_original_342_tests_separate=True,
        truth_not_estimator_input=True,original_failed_case='OLD_FIXTURE_REPLAY.json',
        official_AROSICS_numerical_gate='FAIL',local_warp_not_run=True,
        local_synthetic_validation='PRIOR_16_POINT_RUN_PRESERVED_NO_NUMERIC_PRECISION_PASS',
        stop='SYNTHETIC_REGISTRATION_NUMERICAL_BLOCKER'))
    print('Matrix',[(r['algorithm'],r['case'],r['GSD'],r['direction'],round(r['IMAGE_REGISTRATION_ESTIMATION_ERROR'],4)) for r in records],flush=True)


def texture_probe():
    import numpy as np
    from scipy.ndimage import gaussian_filter,shift
    gt=(308460,10,0,3794750,0,-10)
    noise=np.random.default_rng(130).normal(0,30,(256,256));rows=[]
    for sigma in [0,.5,.8,1.2,2.0]:
        ref=(50+(gaussian_filter(noise,sigma) if sigma else noise)).astype('float32')
        mov=shift(ref,(-.75,1.25),order=3,mode='constant',cval=0)
        result=estimate_arosics(ref,mov,gt,gt)
        error=float(np.linalg.norm(np.array(result['correction_m'])-[-12.5,-7.5]))
        image=ref[65:193,63:191];spectrum=np.abs(np.fft.fftshift(np.fft.fft2(image-image.mean())))**2
        yy,xx=np.indices(spectrum.shape);rad=np.hypot(xx-64,yy-64)
        result.update(smoothing_sigma_px=sigma,error_m=error,
             high_frequency_energy_fraction=float(spectrum[rad>32].sum()/spectrum.sum()),
             reference_std=float(ref.std()), numerical_gate=error<=2)
        rows.append(result);print('sigma',sigma,'error',error,flush=True)
    save('TEXTURE_SINGLE_FACTOR_PROBE.json',dict(rows=rows,
         seed_noise_mean_variance_shift_interpolator_bounds_parameters_fixed=True,
         only_smoothing_sigma_changed=True,threshold_m=2,
         signal_identifiability_not_equal_metric_estimator_accuracy=True))


def region_screen():
    import numpy as np
    from osgeo import gdal
    from PIL import Image,ImageDraw
    ge_ds=gdal.Open(str(OLD/'GE_NATIVE_CONTEXT_DISPLAY_1M.tif'))
    ge=ge_ds.ReadAsArray().transpose(1,2,0);ge_ds=None
    ref=np.load(OLD/'MASK_DIAGNOSTIC_DATA.npz')['ref']
    # Every region is displayed in the original grid, with no shift or score.
    contact=Image.new('RGB',(4*640,4*680),(255,255,255));draw=ImageDraw.Draw(contact)
    for r in range(4):
        for c in range(4):
            block=ge[r*640:(r+1)*640,c*640:(c+1)*640]
            contact.paste(Image.fromarray(block),(c*640,r*680))
            draw.text((c*640+5,r*680+645),f'roi_r{r}_c{c} unshifted original context',fill='black')
    with (OUT/'UNSHIFTED_ALL16_REGION_SCREEN.png').open('xb') as stream:contact.save(stream,format='PNG')
    print('16 unshifted context regions; no matching residuals in selection',flush=True)


def same_region_plan():
    from affine import Affine
    from shapely.geometry import box,mapping
    from shapely.geometry import shape
    sys.path.insert(0,str(ROOT/'src'))
    from xian_r5.alignment.numerical_closure import validate_same_region_design
    affine=Affine(10,0,308460,0,-10,3794750)
    registry=read(P/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json')['cells']
    selected=next(r for r in registry if r['region_id']=='roi_r2_c3')
    support=shape(selected['support_geometry']);rows=[]
    # Distinct roundabout and northern road junction, separated by ~350m.
    # Roundabout pavilion/roof is NOT a tie: anchor is the western carriageway.
    for fid,fold,group,c,r in [('SR_FIT_ROUNDABOUT','FIT','WHOLE_ROUNDABOUT',202,173),
                            ('SR_CHECK_NORTH_JUNCTION','CHECK','WHOLE_NORTH_ROAD_JUNCTION',206,138)]:
        rows.append(dict(feature_id=fid,fold=fold,independence_group=group,region_id='roi_r2_c3',
            anchor_col_row=[c,r],anchor_xy=list(affine*(c+.5,r+.5)),
            evidence_stage='PATCH_ANCHOR_CANDIDATE',precision_eligible=False,
            GE_coordinate=None,reference_coordinate=None,actual_date='2018-11-01',reference_exact_date=None,
            localization_basis=None,relative_localization_uncertainty_m=None,matching_residual_m=None,
            feature_type='ROAD_CONNECTION_STRUCTURE',window_radius_px=10,
            group_semantics='ENTIRE_JUNCTION_NOT_REPEATED_EDGES',
            ground_stability_verified=False,roof_facade_tree_shadow_vehicle_purity_verified=False,
            localization_status='REFERENCE_LOCALIZATION_UNRESOLVED',
            unshifted_selection_evidence=str(OUT/'UNSHIFTED_ALL16_REGION_SCREEN.png'),
            source_image_path=str(OLD/'GE_NATIVE_CONTEXT_DISPLAY_1M.tif'),source_image_sha256=sha(OLD/'GE_NATIVE_CONTEXT_DISPLAY_1M.tif'),
            reference_image_path='F:/xian/10m底图/Xian_2019_TrueColor.tif',
            reference_image_sha256=sha('F:/xian/10m底图/Xian_2019_TrueColor.tif'),
            native_pixel_size=dict(reference_m=10,source_original_degrees=2.68220901489e-6,source_display_m=1)))
    audit=validate_same_region_design(rows,'roi_r2_c3',support)
    save('FIT_CHECK_REGION_DESIGN.json',dict(rows=rows,design_audit=audit,
        support_geometry=mapping(support),support_unchanged=True,region_denominator=16,pixel_denominator=65536,
        selection_frozen_before_current_real_numeric_results=True,
        old_context_and_prior_numeric_results_already_development_exposed=True,
        no_independent_external_holdout_claim=True,
        other_regions=[dict(region_id=r['region_id'],status='NOT_SELECTED_FOR_REAL_TRIAL_SYNTHETIC_GATE_BLOCKED')
                      for r in registry if r['region_id']!='roi_r2_c3'],
        status='SAME_REGION_PATCH_DESIGN_1_FIT_1_CHECK_INSUFFICIENT_PRECISION_TIES_3_PLUS_3',
        formal_test_not_started=True,why_not_more_ties='DO_NOT_INVENT_DISTINCT_PRECISION_STRUCTURES_OR_REUSE_ROAD_EDGES'))
    print('Same-region candidate design 1+1; formal precision ties 0, quantities unchanged 3+3',flush=True)


def readonly_real():
    import csv
    import numpy as np
    from affine import Affine
    from shapely.geometry import shape
    sys.path.insert(0,str(ROOT/'src'))
    from xian_r5.alignment.supported_ground_features import estimate_supported_translation
    from xian_r5.alignment.base_tie import image_shift_to_projected_correction
    plan=read(OUT/'FIT_CHECK_REGION_DESIGN.json');z=np.load(OLD/'MASK_DIAGNOSTIC_DATA.npz')
    yy,xx=np.indices((256,256));reports=[];ledger=[]
    for row in plan['rows']:
        c,r=row['anchor_col_row'];patch=(abs(xx-c)<=10)&(abs(yy-r)<=10)
        # Clip window to the SAME frozen region, no shifted support or boundaries.
        region=(xx>=192)&(xx<256)&(yy>=128)&(yy<192);patch &= region
        corridor=patch&((abs(xx-c)<=2)|(abs(yy-r)<=2))
        masks=dict(A0=patch&z['remaining'],A1=corridor&~z['touched'])
        for policy,mask in masks.items():
            numeric=estimate_supported_translation(z['ref'],z['ge'],mask&z['ref_valid'],mask&z['source_valid'],patch,
                    radius=3,min_samples=100)
            best=numeric['best'];vec=None
            if best:vec=list(image_shift_to_projected_correction(best['dx_px'],best['dy_px'],tuple(Affine(10,0,308460,0,-10,3794750))))
            reports.append(dict(feature_id=row['feature_id'],fold=row['fold'],mask_policy=policy,
                 candidate_pixel_count=int(mask.sum()),candidate_structural_groups=1,
                 numeric_matches=int(best is not None),false_matches=None,
                 false_match_truth_status='NOT_AVAILABLE_NOT_ZERO',
                 correspondence_stage='NUMERIC_MATCH_CANDIDATE' if best else 'PATCH_ANCHOR_CANDIDATE',
                 numeric_candidate_vector_m=vec,numeric_result=numeric,
                 precision_eligible_ties=0,check_coverage=0,relative_uncertainty_m=None,
                 effective_displacement_coverage=0,formal_FIT_CHECK_completed=False,
                 reason='SYNTHETIC_GATE_FAIL; GROUND_PURITY_UNVERIFIED; REFERENCE_LOCALIZATION_UNRESOLVED'))
        # A2 not synthesized from A1: no semantic ground certification exists.
        reports.append(dict(feature_id=row['feature_id'],fold=row['fold'],mask_policy='A2',status='NOT_AVAILABLE',
                candidate_pixel_count=None,candidate_structural_groups=0,numeric_matches=0,false_matches=None,
                precision_eligible_ties=0,check_coverage=0,relative_uncertainty_m=None,effective_displacement_coverage=0,
                reason='NO_VERIFIED_STABLE_GROUND_SEMANTIC_MASK'))
        ledger.append(row)
    save('READ_ONLY_REAL_DIAGNOSTIC.json',dict(rows=reports,region_id='roi_r2_c3',scope='READ_ONLY_NUMERIC_NOT_FORMAL_FIT_CHECK',
         numerical_gate='FAIL',M0='UNCHANGED',M1='NOT_ADMITTED',M2='NOT_RUN',
         source_roofs_read=False,manual_GT_read=False,source_boundaries_modified=False,
         accepted_regions=0,applied_shadow_regions=0,denominator_regions=16))
    fields=['feature_id','fold','independence_group','region_id','anchor_col_row','anchor_xy','evidence_stage',
         'GE_coordinate','reference_coordinate','actual_date','reference_exact_date','localization_basis',
         'relative_localization_uncertainty_m','matching_residual_m','feature_type','precision_eligible',
         'localization_status','native_pixel_size','source_image_path','source_image_sha256',
         'reference_image_path','reference_image_sha256','ground_stability_verified']
    with (OUT/'GROUND_FEATURE_LEDGER.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for row in ledger:writer.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(list,dict)) else row[k] for k in fields})
    print('Read-only same-region diagnostics; no formal vectors applied',flush=True)


def conventions():
    import numpy as np
    from affine import Affine
    from osgeo import gdal,osr
    sys.path.insert(0,str(ROOT/'src'))
    from xian_r5.alignment.numerical_closure import content_correction
    fixture=np.load(OUT/'OLD_FIXTURE_RECONSTRUCTED.npz');gt=tuple(fixture['reference_gt']);affine=Affine.from_gdal(*gt)
    rows=[]
    for anchor in [(0,0),(.5,.5),(65.5,63.5)]:
        x0,y0=affine@anchor;x1,y1=affine@(anchor[0]+1.25,anchor[1]-.75)
        expected=content_correction(1.25,-.75,affine)
        rows.append(dict(anchor_col_row=list(anchor),world_forward_content_displacement=[x1-x0,y1-y0],
            expected_inverse_m=list(expected),TRANSFORM_CONVENTION_ERROR=float(np.linalg.norm(np.array(expected)+[x1-x0,y1-y0]))))
    stored=[]
    for name in ['forward','reverse']:
        path=OLD/('SYNTHETIC_'+name+'_DESHIFTED.tif');ds=gdal.Open(str(path));proj=osr.SpatialReference(wkt=ds.GetProjection())
        stored.append(dict(path=str(path),sha256=sha(path),epsg=proj.GetAuthorityCode(None),geotransform=list(ds.GetGeoTransform()),
                           width=ds.RasterXSize,height=ds.RasterYSize,nodata=ds.GetRasterBand(1).GetNoDataValue()));ds=None
    base=np.load(OUT/'ZERO_SAME_GSD_INPUT.npz')['reference'];metagt=(gt[0]+12.5,10,0,gt[3]+7.5,0,-10)
    metadata_only=estimate_arosics(base,base,gt,metagt)
    metadata_only.update(expected_correction_m=[-12.5,-7.5],
        image_registration_estimation_error_m=float(np.linalg.norm(np.array(metadata_only['correction_m'])-[-12.5,-7.5])),
        content_shift_px=[0,0],geotransform_shift_m=[12.5,7.5])
    save('CONVENTION_FIXED_02/TRANSFORM_CONVENTION_TESTS.json',dict(rows=rows,old_output_header_checks=stored,
        metadata_only_test=metadata_only,TRANSFORM_CONVENTION_ERROR=max(r['TRANSFORM_CONVENTION_ERROR'] for r in rows),
        column='RIGHT_EAST',row='DOWN_SOUTH',image_content_shift_col_row=[1.25,-.75],
        reference_gt=list(gt),old_moving_gt=fixture['moving_gt'].tolist(),
        old_content_not_geotransform_shift=True,old_metadata_shift_m=[0,0],
        old_expected_inverse_m=[-12.5,-7.5],corner_vs_center_translation_identical=True,
        no_half_pixel_offset_added=True,no_direction_sign_change=True,old_transform_not_double_counted=True,
        old_generator='scipy shift dy=-.75 dx=1.25 order3 constant0 prefilter true',
        stored_input_arrays='RECONSTRUCTED_RECIPE_REPLAY_EXACT_NOT_ORIGINAL_ARCHIVE',
        CRS='EPSG:32649',relative_only=True,absolute_position_accuracy='UNKNOWN'))
    print('Convention error0; metadata-only diagnostic',metadata_only['correction_m'],flush=True)


def audit_results(final=False):
    import numpy as np
    old=read(OUT/'OLD_FIXTURE_REPLAY.json');probes=read(OUT/'SINGLE_FACTOR_PROBES.json')
    texture=read(OUT/'TEXTURE_SINGLE_FACTOR_PROBE.json');control=read(OUT/'GENERATOR_INDEPENDENT_RECOVERY_CHECK.json')
    conventions=read(OUT/'CONVENTION_FIXED_02/TRANSFORM_CONVENTION_TESTS.json');comparison=read(OUT/'AROSICS_NCC_COMPARISON.json')
    sys.path.insert(0,str(ROOT/'src'))
    from xian_r5.alignment.numerical_closure import numerical_gate
    errors=[r['estimation_error_m'] for r in old['rows']]+[
         r['IMAGE_REGISTRATION_ESTIMATION_ERROR'] for r in comparison['rows'] if r['algorithm']=='AROSICS_COREG']
    gate='PASS' if numerical_gate(errors,convention_error_m=conventions['TRANSFORM_CONVENTION_ERROR']) else 'FAIL'
    prefix='CALIBRATION_FIXED_02/' if final else ''
    save(prefix+'SYNTHETIC_CALIBRATION_AUDIT.json',dict(SYNTHETIC_NUMERICAL_GATE=gate,threshold_m=2,
        deterministic_gate_recomputed=True,mandatory_metric_errors_m=errors,
        gate_inputs_sha256={name:sha(OUT/name) for name in ['OLD_FIXTURE_REPLAY.json','AROSICS_NCC_COMPARISON.json','CONVENTION_FIXED_02/TRANSFORM_CONVENTION_TESTS.json']},
        stop_status='SYNTHETIC_REGISTRATION_NUMERICAL_BLOCKER',
        old_replay=old,old_result_replay_exact=True,
        TRANSFORM_CONVENTION_ERROR=conventions['TRANSFORM_CONVENTION_ERROR'],
        IMAGE_REGISTRATION_ESTIMATION_ERROR=[r['estimation_error_m'] for r in old['rows']],
        generator_independent_check=control,
        hypothesis_outcomes=[
          dict(hypothesis='WRONG_SIGNS_OR_HALF_PIXEL_OR_DOUBLE_METADATA',status='NOT_SUPPORTED',evidence='TRANSFORM_CONVENTION_TESTS.json'),
          dict(hypothesis='GENERATOR_INCONSISTENT',status='NOT_SUPPORTED_ON_EXACT_CUBIC_MODEL',evidence='GENERATOR_INDEPENDENT_RECOVERY_CHECK.json'),
          dict(hypothesis='BINARY_FFT_SHRINK_ONLY_CAUSE',status='NOT_SUFFICIENT; DISABLING_STILL_FAILS',evidence='SINGLE_FACTOR_PROBES.json'),
          dict(hypothesis='CALC_RESAMPLING_OR_SOURCE_LINEAR_ONLY_CAUSE',status='NOT_SUFFICIENT',evidence='SINGLE_FACTOR_PROBES.json'),
          dict(hypothesis='LOWPASS_TEXTURE_PHASE_ESTIMATION_SENSITIVITY',status='SUPPORTED_BY_SINGLE_FACTOR_SMOOTHING_PROBE',evidence='TEXTURE_SINGLE_FACTOR_PROBE.json')],
        supported_mechanism='Initial Gaussian1.2 phase peak y=0 instead of +1; integer-validation crop128->126 then power-of-two FFT64; fractional side-lobe estimator biased. Smoothing control and window probes demonstrate sensitivity.',
        mechanism_limit='Finite-window/weak-high-frequency phase estimate sensitivity is established; exclusive attribution of all error to boundary leakage not proven.',
        probes=probes,texture_probe=texture,
        wrap_and_mean_subtraction_confound='Changed zero/nodata footprints also move default window1pixel; do not call those pure boundary ablations.',
        version2_fixture='BROADBAND_CONTROL_NOT_REPLACEMENT_OF_OLD_MANDATORY_CASE',
        v2_same_GSD_AROSICS_max_error_m=max(r['IMAGE_REGISTRATION_ESTIMATION_ERROR'] for r in comparison['rows'] if r['algorithm']=='AROSICS_COREG' and r['GSD']=='SAME_10M'),
        cross_GSD_AROSICS_errors_m=[r['IMAGE_REGISTRATION_ESTIMATION_ERROR'] for r in comparison['rows'] if r['algorithm']=='AROSICS_COREG' and r['GSD']=='CROSS_2M_10M'],
        lowpass_stress_failures_not_dropped=True,library_code_or_existing_direction_modified=False,
        NCC='EXISTING_INTEGER_GRID_BASELINE; SUBPIXEL_LIMIT_QUANTIFIED_NOT_AROSICS',
        COREG_LOCAL='CURRENT_NOT_RUN_GLOBAL_PRECISION_GATE_FAILED; PRIOR_SYNTHETIC_16_POINT_TEST_RETAINED',
        actual_scene_precision_inference_from_synthetic='FORBIDDEN',
        corrected_positions_never_created=True))
    save(prefix+'SHADOW_CORRECTION_REPORT.json',dict(status='SYNTHETIC_REGISTRATION_NUMERICAL_BLOCKER',
        SYNTHETIC_NUMERICAL_GATE=gate,SAME_REGION_FIT_CHECK_DESIGN='ONE_FIT_AND_ONE_CHECK_PATCH_GROUP_SAME_FROZEN_REGION_ONLY',
        required_FIT=3,required_CHECK=3,PRECISION_ELIGIBLE_TIES=0,
        ACCEPTED_LOCAL_REGIONS=0,APPLIED_SHADOW_LOCAL_REGIONS=0,
        REFERENCE_SPATIAL_INFORMATION_STATUS='REFERENCE_LOCALIZATION_UNRESOLVED',
        INDEPENDENT_TEST_MAE=None,CORRECTION_COVERAGE_RATE=0,SEAM_CONFLICT_RATE=None,
        seam_reason='NO_ACCEPTED_FIELD_APPLIED; PHOTOGRAPHIC_COMPONENTS_UNKNOWN',
        ABSOLUTE_POSITION_ACCURACY='UNKNOWN',READY_FOR_GT10M_TRAINING=False,READY_FOR_CITYWIDE=False,
        roofs_transported=0,roof_objects_modified=False,source_support_modified=False,
        shadow_corrected_geometries_created=False,fraction_improvement_measured=False,
        formal_B04_counts_unchanged='0_reviews/0_accepted/0_rejected/9_unresolved',
        real_diagnostics_readonly=True,independent_3plus3_quantity_not_lowered=True,
        synthetic_truth_not_read_by_real_matching=True,module_B_changed=False))
    print('Audit stop status SYNTHETIC_REGISTRATION_NUMERICAL_BLOCKER',flush=True)


def identifiability():
    import numpy as np
    from scipy.ndimage import map_coordinates
    z=np.load(OUT/'OLD_FIXTURE_RECONSTRUCTED.npz');rr,cc=np.mgrid[64:192,64:192]
    offsets=np.arange(-3,3.001,.25);cost=np.zeros((len(offsets),len(offsets)))
    # Exhaustive, bounded sampling-model diagnostic. Truth not read during scan.
    for iy,dy in enumerate(offsets):
        for ix,dx in enumerate(offsets):
            sample=map_coordinates(z['reference'],[rr-dy,cc-dx],order=3,mode='constant',cval=0)
            cost[iy,ix]=np.mean((sample-z['moving'][64:192,64:192])**2,dtype='float64')
    best=np.unravel_index(cost.argmin(),cost.shape);iy,ix=best
    rank=np.sort(cost.ravel());outside=np.ones_like(cost,dtype=bool)
    outside[max(0,iy-1):iy+2,max(0,ix-1):ix+2]=False
    with (OUT/'OLD_FIXTURE_IDENTIFIABILITY_COST.npz').open('xb') as f:
        np.savez_compressed(f,cost=cost,offsets=offsets)
    save('OLD_FIXTURE_IDENTIFIABILITY.json',dict(search_range_px=[-3,3],step_px=.25,
        best_observed_content_shift_px=[float(offsets[ix]),float(offsets[iy])],
        best_cost=float(rank[0]),second_grid_candidate_cost=float(rank[1]),
        smallest_cost_outside_nearest_neighbours=float(cost[outside].min()),
        ties_at_minimum_tolerance_1e_9=int((cost<=rank[0]+1e-9).sum()),
        truth_not_read_for_scan=True,uniqueness_scope='BOUNDED_QUARTER_PIXEL_CUBIC_SAMPLING_MODEL_ONLY',
        not_a_real_reference_localization_bound=True,
        evidence='Old texture has identifiable displacement under sampling model; failing AR phase estimate is not proof missing signal.'))
    print('Identifiability minimum',offsets[ix],offsets[iy],'second cost',rank[1],flush=True)


def close():
    if (OUT/'FINAL_RECEIPT.json').exists() or (OUT/'REGRESSION_TEST_LOG.txt').exists():
        raise ValueError('Append-only close already attempted; preserve old evidence')
    authority=read(OUT/'AUTHORITATIVE_DELIVERY.json')
    required=['SYNTHETIC_CALIBRATION_AUDIT.json','TRANSFORM_CONVENTION_TESTS.json',
        'FIT_CHECK_REGION_DESIGN.json','GROUND_FEATURE_LEDGER.csv','AROSICS_NCC_COMPARISON.json',
        'LOCAL_ALIGNMENT_EVIDENCE.png','AROSICS_SYNTHETIC_CALIBRATION.png','SHADOW_CORRECTION_REPORT.json','FINAL_REPORT.md']
    for name in required:
        if not (OUT/authority['artifacts'].get(name,name)).is_file():raise ValueError('Missing required '+name)
    code_files=[Path(__file__),ROOT/'tools/plot_xian_numerical_closure.py',
                ROOT/'src/xian_r5/alignment/numerical_closure.py',ROOT/'tests/test_xian_alignment_closure.py']
    code_start={str(p):sha(p) for p in code_files}
    frozen=read(OUT/'FROZEN_RUN_PLAN.json')
    for path,expected in frozen['protected_sha256'].items():
        if sha(path)!=expected:raise ValueError('Protected input changed: '+path)
    for path,expected in frozen['large_file_stat'].items():
        stat=Path(path).stat()
        if [stat.st_size,stat.st_mtime_ns]!=expected:raise ValueError('Large source stat changed')
    audit=read(OUT/authority['artifacts']['SYNTHETIC_CALIBRATION_AUDIT.json'])
    sys.path.insert(0,str(ROOT/'src'))
    from xian_r5.alignment.numerical_closure import numerical_gate
    recomputed='PASS' if numerical_gate(audit['mandatory_metric_errors_m'],convention_error_m=audit['TRANSFORM_CONVENTION_ERROR']) else 'FAIL'
    if recomputed!=audit['SYNTHETIC_NUMERICAL_GATE']:raise ValueError('Numerical gate discrepancy')
    if recomputed!='FAIL':raise ValueError('This evidence close requires failed gate and no real application')
    shadow=read(OUT/authority['artifacts']['SHADOW_CORRECTION_REPORT.json'])
    if shadow['APPLIED_SHADOW_LOCAL_REGIONS']!=0 or shadow['ACCEPTED_LOCAL_REGIONS']!=0:
        raise ValueError('Unqualified real correction present')
    sys.path.insert(0,str(ROOT/'tests'))
    names=sorted(p.stem for p in (ROOT/'tests').glob('test_xian_*.py') if p.stem!='test_xian_additive_post_inference')
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    if suite.countTestCases()!=352:raise ValueError('Expected342preserved+10new tests')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log).run(suite)
    with (OUT/'REGRESSION_TEST_LOG.txt').open('x',encoding='utf-8') as f:f.write(log.getvalue())
    if not result.wasSuccessful():raise ValueError('Regression failure')
    if any(sha(p)!=h for p,h in code_start.items()):raise ValueError('Code changed during regression')
    outputs={}
    for folder,dirs,files in os.walk(OUT):
        dirs[:]=[d for d in dirs if d!='plot_cache']
        for name in files:
            path=Path(folder)/name;outputs[str(path.relative_to(OUT))]=sha(path)
    save('FINAL_RECEIPT.json',dict(task='A_LOCAL_ALIGNMENT_NUMERICAL_AND_CHECK_CLOSURE',
        status='SYNTHETIC_REGISTRATION_NUMERICAL_BLOCKER',
        SYNTHETIC_NUMERICAL_GATE=recomputed, TRANSFORM_CONVENTION_ERROR=audit['TRANSFORM_CONVENTION_ERROR'],
        actual_tests_passed=result.testsRun,preserved_tests=342,new_tests=10,
        protected_input_hashes_checked=len(frozen['protected_sha256']),protected_inputs_changed=0,
        unused_large_mosaic_stat_unchanged=True,large_mosaic_full_hash_claim=False,
        actual_matching_image_input_hashes_checked=True,
        PRECISION_ELIGIBLE_TIES=0,ACCEPTED_LOCAL_REGIONS=0,APPLIED_SHADOW_LOCAL_REGIONS=0,
        INDEPENDENT_TEST_MAE=None,CORRECTION_COVERAGE_RATE=0,SEAM_CONFLICT_RATE=None,
        ABSOLUTE_POSITION_ACCURACY='UNKNOWN',READY_FOR_GT10M_TRAINING=False,READY_FOR_CITYWIDE=False,
        module_B_changed=False,old_formal_counts_unchanged=True,old_failed_record_not_promoted=True,
        source_code_sha256=code_start,output_sha256=outputs,authoritative_artifacts=authority['artifacts'],
        scope='FIXED_CURRENT_WINDOW_CPU_ONLY_NO_REAL_DISPLACEMENT_APPLICATION',
        completion_boundary='DIAGNOSTIC_DELIVERY_COMPLETE; NUMERICAL_PASS_AND_PRECISION_CHECK_NOT_ACHIEVED'))
    print('Closed',result.testsRun,'tests; unchanged input hashes',len(frozen['protected_sha256']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','replay_old','probes','controls','texture_probe','region_screen','same_region_plan','readonly_real','conventions','audit_results','conventions_fixed02','audit_results_fixed02','identifiability','audit_results_final','close'])
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        save('EXECUTION_'+args.stage+'.json',dict(stage=args.stage,source_code_sha256=sha(__file__),
             baseline_source_code_sha256=sha(ROOT/'tools/run_xian_arosics_trial.py'),
             truth_not_estimator_input=True))
        name={'conventions_fixed02':'conventions','audit_results_fixed02':'audit_results'}.get(args.stage,args.stage)
        if name=='audit_results_final':audit_results(final=True)
        else:globals()[name]()
