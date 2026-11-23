"""Fixed CPU preprocessing comparison; no roofs, GT, real ties or displacement application."""
import argparse
import hashlib
import importlib.util
import json
import sys
import io
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xian_r5.alignment.subpixel_phase import refine_zero_peak

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('closure',ROOT/'tools/xian_local_numerical_closure.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
BASE=prior.OUT
OUT=BASE.parent/'A_LOCAL_ALIGNMENT_NUMERICAL_ROBUSTNESS_20261013'


def sha(p):return prior.sha(p)
def read(p):return prior.read(p)


def save(name,data):
    payload=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist())
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(payload)


def freeze():
    if OUT.exists():raise ValueError('Fresh append-only run required')
    old=read(BASE/'FROZEN_RUN_PLAN.json');protected=dict(old['protected_sha256'])
    receipt=read(BASE/'FINAL_RECEIPT.json')
    protected.update(receipt['source_code_sha256'])
    protected.update({str(BASE/name):h for name,h in receipt['output_sha256'].items()})
    protected[str(BASE/'FINAL_RECEIPT.json')]=sha(BASE/'FINAL_RECEIPT.json')
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError('Prior evidence drift: '+p)
    OUT.mkdir()
    save('FROZEN_ROBUSTNESS_PLAN.json',dict(protected_sha256=protected,
        large_file_stat=old['large_file_stat'],old_failed_case_mandatory=True,
        numerical_threshold_m=2,methods=['RAW','LAPLACIAN_FIXED','DOG_FIXED'],
        parameter_freeze=dict(laplacian='scipy laplace reflect 1px common10m',DOG_sigmas_px=[.5,1.5]),
        preprocessing_is_not_raw_official_AROSICS=True,method_not_selected_per_case=True,
        full_same_GSD_case_files=['ZERO','INTEGER','SUBPIXEL_X','SUBPIXEL_Y','BOTH'],
        both_directions=True,cross_GSD_common_grid='aligned integer block area average only',
        development_texture_sigmas_px=[0,.5,.8,1.2,2.0],
        heldout_synthetic_seeds=[331,907,1879],
        heldout_synthetic_shifts_col_row=[[.37,-1.18],[2.125,-.625],[-1.4,.8]],
        heldout_synthetic_not_external_real_validation=True,
        stage_source_sha256=sha(__file__),real_application_forbidden=True,
        absolute_position_accuracy='UNKNOWN',module_B_mutation_allowed=False))
    print('Freeze',len(protected),'protected hashes',flush=True)


def common_grid(ref,mov,ref_gt,mov_gt):
    # Only explicitly aligned, integer-ratio test arrays. Unknown alignments
    # cannot be rescued by reshaping or shifting their geotransforms.
    ref_gt=tuple(ref_gt);mov_gt=tuple(mov_gt)
    if ref_gt[0]!=mov_gt[0] or ref_gt[3]!=mov_gt[3] or any(gt[i]!=0 for gt in [ref_gt,mov_gt] for i in [2,4]):
        raise ValueError('Explicit aligned north-up origins required')
    gsd=max(ref_gt[1],mov_gt[1]);out=[]
    for image,gt in [(ref,ref_gt),(mov,mov_gt)]:
        factor=gsd/gt[1]
        if gt[5]!=-gt[1] or int(factor)!=factor or image.shape[0]%int(factor) or image.shape[1]%int(factor):
            raise ValueError('Integer aligned GSD required')
        f=int(factor)
        out.append(image.reshape(image.shape[0]//f,f,image.shape[1]//f,f).mean((1,3)).astype('float32'))
    if out[0].shape!=out[1].shape:raise ValueError('Shared physical support required')
    return out[0],out[1],(ref_gt[0],gsd,0,ref_gt[3],0,-gsd)


def preprocess(image,method):
    image=np.asarray(image,dtype='float32')
    if method=='RAW':return image.copy()
    from scipy.ndimage import laplace,gaussian_filter
    if method=='LAPLACIAN_FIXED':return laplace(image,mode='reflect')
    if method=='DOG_FIXED':return gaussian_filter(image,.5,mode='reflect')-gaussian_filter(image,1.5,mode='reflect')
    raise ValueError('Frozen method only')


def estimate(reference,moving,ref_gt,mov_gt,method):
    a,b,gt=common_grid(reference,moving,ref_gt,mov_gt)
    # No shift, scores or generator truth enters this estimator interface.
    if method=='RAW':return prior.estimate_arosics(a,b,gt,gt)
    from osgeo import osr
    from geoarray import GeoArray
    from arosics import COREG
    from scipy.fft import fft2,ifft2,fftshift
    srs=osr.SpatialReference();srs.ImportFromEPSG(32649);projection=srs.ExportToWkt()
    # Preserve input availability independently of valid zero-valued derivatives.
    arrays=[]
    for raw in [a,b]:
        processed=raw.copy() if method in ['APODIZED_CPS_V1','APODIZED_DFT_V2'] else preprocess(raw,method)
        processed[raw==0]=-9999
        arrays.append(GeoArray(processed,geotransform=gt,projection=projection,nodata=-9999))
    cr=COREG(*arrays,nodata=(-9999,-9999),ws=(128,128),max_shift=3,CPUs=1,q=True)
    if method in ['APODIZED_CPS_V1','APODIZED_DFT_V2']:
        phase_state={}
        def apodized(im0=None,im1=None,precision=np.complex64):
            if im0 is None:im0=cr.matchWin[:] if cr.matchWin.imID=='ref' else cr.otherWin[:]
            if im1 is None:im1=cr.otherWin[:] if cr.otherWin.imID=='shift' else cr.matchWin[:]
            h,w=cr._shrink_winsize_to_binarySize(im0.shape)
            h=w=min(h,w)
            y,x=(im0.shape[0]-h)//2,(im0.shape[1]-w)//2
            left=im0[y:y+h,x:x+w].astype('float64');right=im1[y:y+h,x:x+w].astype('float64')
            taper=np.outer(np.hanning(h),np.hanning(w))
            f0=fft2((left-left.mean())*taper);f1=fft2((right-right.mean())*taper)
            cross=f0*f1.conjugate();denom=np.abs(cross)
            # Fixed relative floor prevents zero-energy frequencies determining phase.
            threshold=denom.max()*1e-8
            phase=np.divide(cross,denom,out=np.zeros_like(cross),where=denom>threshold)
            phase_state['phase']=phase
            cr.fft_win_size_YX=(h,w)
            return np.abs(fftshift(ifft2(phase)))
        cr._calc_shifted_cross_power_spectrum=apodized
        if method=='APODIZED_DFT_V2':
            cr._calc_subpixel_shifts=lambda scps:refine_zero_peak(phase_state['phase'])
            # Numerical calibration only: upstream SSIM resampling can fail at
            # exact integer shifts. Do not present that auxiliary metric as run.
            cr._validate_ssim_improvement=lambda *args,**kwargs:(None,None)
    cr.calculate_spatial_shifts()
    return dict(success=bool(cr.success),correction_m=[float(cr.x_shift_map),float(cr.y_shift_map)],
        output_shift_px=[float(cr.x_shift_px),float(cr.y_shift_px)],
        reliability=None if cr.shift_reliability is None else float(cr.shift_reliability),fft_shape=list(cr.fft_win_size_YX),
        algorithm='AROSICS_COREG_CUSTOM_'+method if method in ['APODIZED_CPS_V1','APODIZED_DFT_V2'] else 'AROSICS_COREG_WITH_FIXED_PREPROCESSING',
        nodata=-9999,valid_zero_features_preserved=True,official_library_modified=False,
        SSIM_status='NOT_RUN_NUMERICAL_ADAPTER_ONLY' if method=='APODIZED_DFT_V2' else 'UPSTREAM')


def red():
    z=np.load(BASE/'OLD_FIXTURE_RECONSTRUCTED.npz')
    result=estimate(z['reference'],z['moving'],z['reference_gt'],z['moving_gt'],'RAW')
    error=float(np.linalg.norm(np.array(result['correction_m'])-[-12.5,-7.5]))
    save('OLD_RAW_RED_REPLAY.json',dict(result=result,error_m=error,threshold_m=2,gate='FAIL' if error>2 else 'PASS'))
    print('Old raw error',error,flush=True)
    if error>2:raise AssertionError('Reproduced old numerical failure >2m')


def development(output_name='DEVELOPMENT_COMPARISON.json',methods=None):
    from scipy.ndimage import shift,gaussian_filter
    plan=read(OUT/'FROZEN_ROBUSTNESS_PLAN.json');cases=[]
    old=np.load(BASE/'OLD_FIXTURE_RECONSTRUCTED.npz');gt=tuple(old['reference_gt'])
    cases.append(('OLD_GAUSSIAN_1P2',old['reference'],old['moving'],gt,gt,[-12.5,-7.5]))
    for name,(dx,dy) in zip(plan['full_same_GSD_case_files'],[(0,0),(1,-1),(.25,0),(0,-.75),(1.25,-.75)]):
        z=np.load(BASE/(name+'_SAME_GSD_INPUT.npz'))
        cases.append((name,z['reference'],z['moving'],tuple(z['reference_gt']),tuple(z['moving_gt']),[-dx*10,dy*10]))
    z=np.load(BASE/'CROSS_GSD_INPUT.npz')
    cases.append(('CROSS_GSD',z['reference'],z['moving'],tuple(z['reference_gt']),tuple(z['moving_gt']),[-12.5,-7.5]))
    noise=np.random.default_rng(130).normal(0,30,(256,256))
    for sigma in plan['development_texture_sigmas_px']:
        ref=(50+(gaussian_filter(noise,sigma) if sigma else noise)).astype('float32')
        mov=shift(ref,(-.75,1.25),order=3,mode='constant',cval=0)
        cases.append(('SMOOTH_'+str(sigma),ref,mov,gt,gt,[-12.5,-7.5]))
    results=[]
    methods=plan['methods'] if methods is None else methods
    for method in methods:
        for name,a,b,ag,bg,truth in cases:
            for direction,ref,mov,rg,mg,expected in [('FORWARD',a,b,ag,bg,np.array(truth)),('REVERSE',b,a,bg,ag,-np.array(truth))]:
                try:
                    result=estimate(ref,mov,rg,mg,method)
                    error=float(np.linalg.norm(np.array(result['correction_m'])-expected))
                except Exception as e:
                    result=dict(success=False,error=type(e).__name__+': '+str(e));error=None
                row=dict(method=method,case=name,direction=direction,result=result,expected_correction_m=expected.tolist(),error_m=error)
                results.append(row)
        errors=[r['error_m'] for r in results if r['method']==method]
        finite=[e for e in errors if e is not None]
        print(method,'max error',max(finite) if finite else None,'passed',sum(e is not None and e<=2 for e in errors),'/',len(errors),flush=True)
    save(output_name,dict(rows=results,threshold_m=2,
         input_case_count=len(cases),raw_failure_preserved=True,
         methods=[dict(method=m,all_cases_pass=all(r['error_m'] is not None and r['error_m']<=2 for r in results if r['method']==m),
                       error_count=sum(r['error_m'] is None for r in results if r['method']==m),
                       max_error_m=max([r['error_m'] for r in results if r['method']==m and r['error_m'] is not None],default=None)) for m in methods],
         hidden_truth_used_only_by_scorer=True,real_application=False))


def amendment():
    save('FROZEN_APODIZATION_AMENDMENT.json',dict(method='APODIZED_CPS_V1',
        preprocessing='FFT window only subtract own mean then separable Hann',
        frequency_floor_relative_cross_amplitude=1e-8,nodata_sentinel=-9999,
        no_truth_or_known_displacement_input=True,official_library_modified=False,
        method_semantics='CUSTOM_AROSICS_CPS_ADAPTER_NOT_OFFICIAL_UNMODIFIED_COREG',
        original_methods_and_failures_preserved=True,parameters_not_reselected_per_case=True,
        heldout_not_yet_evaluated=True,real_application_forbidden=True,
        source_sha256=sha(__file__)))


def revised_development():
    read(OUT/'FROZEN_APODIZATION_AMENDMENT.json')
    development('DEVELOPMENT_COMPARISON_REVISED.json',['RAW','LAPLACIAN_FIXED','DOG_FIXED','APODIZED_CPS_V1'])


def dft_amendment():
    save('FROZEN_DFT_AMENDMENT.json',dict(method='APODIZED_DFT_V2',
        phase_preparation='SAME_FROZEN_HANN_AND_RELATIVE_FLOOR_1e-8_AS_V1',
        subpixel_estimator='LOCAL_INVERSE_DFT_41x41_GRID_MINUS0P5_TO_PLUS0P5_FIXED',
        parameter_delta='CHANGE_SUBPIXEL_PEAK_LOCALIZATION_ONLY; NO_CASE_SPECIFIC_CHOICE',
        exact_zero_output_preserved=True,threshold_m=2,
        no_truth_input=True,raw_official_AROSICS_unchanged=True,
        unseen_holdout_not_read=True,real_application_forbidden=True,
        code_sha256=sha(__file__),refinement_code_sha256=sha(ROOT/'src/xian_r5/alignment/subpixel_phase.py')))


def dft_development():
    read(OUT/'FROZEN_DFT_AMENDMENT.json')
    development('DEVELOPMENT_DFT_V2.json',['APODIZED_DFT_V2'])


def adapter_final_amendment():
    save('FROZEN_DFT_FINAL_ENGINEERING_AMENDMENT.json',dict(
        method='APODIZED_DFT_V2',numerical_estimator_unchanged=True,threshold_m=2,
        numerical_only=True,SSIM_status='NOT_RUN',
        reason='Do not let an upstream auxiliary integer-shift SSIM dimension error alter computed numerical shifts.',
        zero_reliability=None,zero_shift_is_not_None=True,
        no_real_application=True,official_library_modified=False,
        method_parameters_not_tuned_on_heldout=True,heldout_not_yet_opened=True,
        code_sha256=sha(__file__),phase_code_sha256=sha(ROOT/'src/xian_r5/alignment/subpixel_phase.py')))


def adapter_final_development():
    read(OUT/'FROZEN_DFT_FINAL_ENGINEERING_AMENDMENT.json')
    development('DEVELOPMENT_DFT_FINAL.json',['APODIZED_DFT_V2'])


def holdout():
    from scipy.ndimage import shift,gaussian_filter
    plan=read(OUT/'FROZEN_ROBUSTNESS_PLAN.json');dev=read(OUT/'DEVELOPMENT_DFT_FINAL.json')
    if not dev['methods'][0]['all_cases_pass']:raise ValueError('Development gate failed; no heldout tuning allowed')
    rows=[];gt=(308460,10,0,3794750,0,-10)
    for seed,(dx,dy) in zip(plan['heldout_synthetic_seeds'],plan['heldout_synthetic_shifts_col_row']):
        noise=np.random.default_rng(seed).normal(0,30,(256,256))
        for sigma in [0,1.2,2.0]:
            ref=(50+(gaussian_filter(noise,sigma) if sigma else noise)).astype('float32')
            mov=shift(ref,(dy,dx),order=3,mode='constant',cval=0)
            with (OUT/f'HELDOUT_{seed}_{sigma}_INPUT.npz').open('xb') as f:
                np.savez_compressed(f,reference=ref,moving=mov,gt=np.array(gt))
            expected=np.array([-dx*10,dy*10])
            for direction,a,b,truth in [('FORWARD',ref,mov,expected),('REVERSE',mov,ref,-expected)]:
                try:
                    result=estimate(a,b,gt,gt,'APODIZED_DFT_V2')
                    error=float(np.linalg.norm(np.array(result['correction_m'])-truth))
                except Exception as e:result=dict(error=str(e));error=None
                rows.append(dict(seed=seed,sigma=sigma,direction=direction,error_m=error,result=result,
                    truth_scorer_only=True,not_external_real_validation=True))
    passed=all(r['error_m'] is not None and r['error_m']<=2 for r in rows)
    save('HELDOUT_SYNTHETIC_VALIDATION.json',dict(rows=rows,method='APODIZED_DFT_V2',
        gate='PASS' if passed else 'FAIL',threshold_m=2,
        parameter_reselection_after_holdout_forbidden=True,
        max_error_m=max([r['error_m'] for r in rows if r['error_m'] is not None],default=None),
        real_precision_or_reference_accuracy_proven=False,real_application=False))
    print('Heldout',sum(r['error_m'] is not None and r['error_m']<=2 for r in rows),'/',len(rows),
          'max',max(r['error_m'] for r in rows if r['error_m'] is not None),flush=True)


def close():
    if (OUT/'FINAL_RECEIPT.json').exists() or (OUT/'REGRESSION_TEST_LOG.txt').exists():raise ValueError('Fresh closure required')
    plan=read(OUT/'FROZEN_ROBUSTNESS_PLAN.json')
    for p,h in plan['protected_sha256'].items():
        if sha(p)!=h:raise ValueError('Protected input changed: '+p)
    for p,state in plan['large_file_stat'].items():
        s=Path(p).stat()
        if [s.st_size,s.st_mtime_ns]!=state:raise ValueError('Large source changed')
    dev=read(OUT/'DEVELOPMENT_DFT_FINAL.json');hold=read(OUT/'HELDOUT_SYNTHETIC_VALIDATION.json')
    errors=[r['error_m'] for r in dev['rows']+hold['rows']]
    passed=bool(errors and all(e is not None and np.isfinite(e) and 0<=e<=2 for e in errors))
    if not passed:raise ValueError('Numerical adapter not validated')
    if not (OUT/'FINAL_REPORT.md').is_file():raise ValueError('Report missing')
    sys.path.insert(0,str(ROOT/'tests'));log=io.StringIO()
    names=sorted(p.stem for p in (ROOT/'tests').glob('test_xian_*.py') if p.stem!='test_xian_additive_post_inference')
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    if suite.countTestCases()!=356:raise ValueError('352preserved+4new expected')
    result=unittest.TextTestRunner(stream=log).run(suite)
    with (OUT/'REGRESSION_TEST_LOG.txt').open('x',encoding='utf-8') as f:f.write(log.getvalue())
    if not result.wasSuccessful():raise ValueError('Regression failure')
    code=[Path(__file__),ROOT/'src/xian_r5/alignment/subpixel_phase.py',ROOT/'tests/test_xian_subpixel_phase.py']
    save('FINAL_RECEIPT.json',dict(status='NUMERICAL_ADAPTER_VALIDATED_PENDING_REAL_PRECISION_EVIDENCE',
        raw_official_AROSICS_gate='FAIL',adapter_method='AROSICS_COREG_CUSTOM_APODIZED_DFT_V2',
        adapter_numerical_gate='PASS',threshold_m=2,development_checks=24,heldout_synthetic_checks=18,
        development_max_error_m=max(r['error_m'] for r in dev['rows']),
        heldout_synthetic_max_error_m=max(r['error_m'] for r in hold['rows']),
        heldout_not_external_real_validation=True,SSIM_not_run=True,
        actual_tests_passed=result.testsRun,preserved_tests=352,new_tests=4,
        protected_input_hashes_checked=len(plan['protected_sha256']),protected_input_hashes_changed=0,
        original_large_source_stat_unchanged=True,PRECISION_ELIGIBLE_REAL_TIES=0,
        ACCEPTED_LOCAL_REGIONS=0,APPLIED_SHADOW_LOCAL_REGIONS=0,CORRECTION_COVERAGE_RATE=0,
        INDEPENDENT_TEST_MAE=None,ABSOLUTE_POSITION_ACCURACY='UNKNOWN',
        READY_FOR_GT10M_TRAINING=False,READY_FOR_CITYWIDE=False,module_B_changed=False,
        old_formal_counts_unchanged=True,source_code_sha256={str(p):sha(p) for p in code},
        output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print('Closed',result.testsRun,'tests; unchanged',len(plan['protected_sha256']),'hashes; adapter numerical PASS, real0',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','red','development','development_fixed02','amendment','revised_development','dft_amendment','dft_development','adapter_final_amendment','adapter_final_development','holdout','close','close_baseline_runtime']);args=p.parse_args()
    if args.stage=='freeze':freeze()
    else:
        save('EXECUTION_'+args.stage+'.json',dict(source_sha256=sha(__file__),inherited_estimator_sha256=sha(ROOT/'tools/xian_local_numerical_closure.py')))
        if args.stage=='development_fixed02':development('DEVELOPMENT_COMPARISON_FIXED_02.json')
        elif args.stage=='close_baseline_runtime':close()
        else:globals()[args.stage]()
