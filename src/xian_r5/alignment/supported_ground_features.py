"""Support-closed directional features and internal spatial-check candidates.

An axis derivative uses only three available pixels on that axis. No Gaussian
mixing with excluded roofs, no false full-2D-kernel certification, no external
ground accuracy or reviewed tie creation. User-provided masks retain their own
unverified semantic status.
"""
import numpy as np
from .base_tie import _pearson_correlation


def directional_features(rgb,mask):
    if np.ma.isMaskedArray(rgb) or np.ma.isMaskedArray(mask):raise ValueError('Explicit masks required')
    rgb=np.asarray(rgb);mask=np.asarray(mask)
    if (rgb.ndim!=3 or rgb.shape[2]!=3 or mask.shape!=rgb.shape[:2] or mask.dtype.kind!='b'
            or not np.isfinite(rgb).all()):raise ValueError('Finite RGB and boolean support required')
    gray=(rgb.astype(np.float64)*np.array([.299,.587,.114])).sum(axis=2)
    features=np.zeros((2,*mask.shape),np.float64);valid=np.zeros_like(features,dtype=bool)
    valid[0,:,1:-1]=mask[:,:-2]&mask[:,1:-1]&mask[:,2:]
    valid[1,1:-1,:]=mask[:-2,:]&mask[1:-1,:]&mask[2:,:]
    features[0,:,1:-1]=np.abs(gray[:,2:]-gray[:,:-2])*.5
    features[1,1:-1,:]=np.abs(gray[2:,:]-gray[:-2,:])*.5
    features[~valid]=0
    return features,valid


def _paired(reference,moving,reference_valid,moving_valid,dx,dy,*,also_zero=False):
    _,h,w=reference.shape
    y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
    r=reference[:,y0:y1,x0:x1];m=moving[:,y0+dy:y1+dy,x0+dx:x1+dx]
    common=reference_valid[:,y0:y1,x0:x1]&moving_valid[:,y0+dy:y1+dy,x0+dx:x1+dx]
    if also_zero:common&=moving_valid[:,y0:y1,x0:x1]
    n=int(common.any(axis=0).sum())
    return r,m,common,n,(y0,y1,x0,x1)


def estimate_supported_translation(reference_rgb,moving_rgb,reference_mask,moving_mask,component,
                                   *,radius=3,min_samples=100):
    arrays=[np.asarray(x) for x in (reference_mask,moving_mask,component)]
    if any(np.ma.isMaskedArray(x) for x in (reference_mask,moving_mask,component)):
        raise ValueError('Masked unknowns not allowed')
    if any(x.dtype.kind!='b' or x.shape!=reference_rgb.shape[:2] for x in arrays):
        raise ValueError('Boolean masks on shared grid required')
    if moving_rgb.shape!=reference_rgb.shape:raise ValueError('Shared image grid required')
    if type(radius) is not int or radius<1 or type(min_samples) is not int or min_samples<3:
        raise ValueError('Positive radius/minimum distinct pixels required')
    rv,mv,support=arrays
    ref,refvalid=directional_features(reference_rgb,rv&support)
    mov,movvalid=directional_features(moving_rgb,mv&support)
    candidates=[]
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            r,m,common,n,_=_paired(ref,mov,refvalid,movvalid,dx,dy)
            if n<min_samples:continue
            score=_pearson_correlation(r[common],m[common])
            if score is not None:candidates.append({'corr':score,'dx_px':dx,'dy_px':dy,
                'n':n,'scalar_axis_observations':int(common.sum()),
                'horizontal_observations':int(common[0].sum()),'vertical_observations':int(common[1].sum())})
    candidates.sort(key=lambda c:(-c['corr'],abs(c['dx_px'])+abs(c['dy_px']),c['dy_px'],c['dx_px']))
    return {'best':candidates[0] if candidates else None,'runner_up':candidates[1] if len(candidates)>1 else None,
        'search_radius_px':radius,'minimum_distinct_pixels':min_samples,'candidate_count':len(candidates),
        'reference_supported_feature_pixels':int(refvalid.any(axis=0).sum()),
        'moving_supported_feature_pixels':int(movvalid.any(axis=0).sum()),
        'feature_policy':'THREE_AVAILABLE_PIXELS_PER_AXIS_NO_EXCLUDED_PIXEL_MIXING',
        'shift_convention':'MOVING_FEATURE_OFFSET_ALIGN_APPLY_NEGATIVE',
        'formal_acceptance':False,'ground_feature_semantics':'PROVIDED_MASK_NOT_VERIFIED_STABLE_GROUND'}


def internal_spatial_check(reference_rgb,moving_rgb,reference_mask,moving_mask,component,
                           *,radius=3,min_samples=100,block_size=16,minimum_gain=.01):
    """FIT/check are pixel blocks, not formal signed ties or external validation.

    Select using fit blocks only. Compare zero and selected candidate on the
    identical check cohort. Check samples are never used to reselect a vector.
    Kernel validity includes the fold mask so derivatives cannot cross folds.
    """
    if type(block_size) is not int or block_size<4 or not np.isfinite(minimum_gain) or minimum_gain<0:
        raise ValueError('Invalid frozen internal-check policy')
    yy,xx=np.indices(component.shape);fit=((xx//block_size+yy//block_size)%2)==0
    check=~fit
    result=estimate_supported_translation(reference_rgb,moving_rgb,reference_mask&fit,
        moving_mask&fit,component,radius=radius,min_samples=min_samples)
    best=result['best'];reasons=[]
    check_result={'n':0,'zero_corr':None,'candidate_corr':None,'gain':None,
                  'horizontal_observations':0,'vertical_observations':0}
    if best is None:reasons.append('INSUFFICIENT_FIT_FEATURES_OR_NO_SCORE')
    else:
        dx,dy=best['dx_px'],best['dy_px']
        if abs(dx)>=radius or abs(dy)>=radius:reasons.append('FIT_PEAK_AT_SEARCH_LIMIT')
        if not best['horizontal_observations'] or not best['vertical_observations']:
            reasons.append('FIT_SINGLE_AXIS_APERTURE_RISK')
        ref,rv=directional_features(reference_rgb,reference_mask&component&check)
        mov,mv=directional_features(moving_rgb,moving_mask&component&check)
        r,m,common,n,bounds=_paired(ref,mov,rv,mv,dx,dy,also_zero=True)
        y0,y1,x0,x1=bounds;check_result['n']=n
        check_result['horizontal_observations']=int(common[0].sum())
        check_result['vertical_observations']=int(common[1].sum())
        if n<min_samples:reasons.append('INSUFFICIENT_COMMON_CHECK_FEATURES')
        else:
            zero=_pearson_correlation(r[common],mov[:,y0:y1,x0:x1][common])
            candidate=_pearson_correlation(r[common],m[common])
            gain=None if zero is None or candidate is None else candidate-zero
            check_result.update(zero_corr=zero,candidate_corr=candidate,gain=gain)
            if not check_result['horizontal_observations'] or not check_result['vertical_observations']:
                reasons.append('CHECK_SINGLE_AXIS_APERTURE_RISK')
            if gain is None:reasons.append('UNINFORMATIVE_CHECK_FEATURES')
            elif dx==0 and dy==0:reasons.append('ZERO_VECTOR_SELF_COMPARISON_NOT_CORRECTION_EVIDENCE')
            elif gain<minimum_gain:reasons.append('CHECK_DID_NOT_IMPROVE_FROZEN_BASELINE')
    return {'fit':result,'check':check_result,'engineering_experiment_supported':not reasons,
        'reasons':reasons,'block_size':block_size,'minimum_check_gain':minimum_gain,
        'independent_external_validation':False,'formal_fit_holdout_review':False,
        'relative_registration_uncertainty_m':None,'absolute_position_accuracy':'UNKNOWN'}
