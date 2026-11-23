"""Fixed-cohort, frozen-vector numerical replay; not independent validation."""
import numpy as np
from .base_tie import _erode_mask, _pearson_correlation


def audit_frozen_translation(reference_features, moving_features,
                             reference_valid, moving_valid, component_support,
                             observed_offset_px, *, minimum_samples=100,
                             feature_radius=4):
    """Compare zero and one frozen offset on exactly the same reference pixels.

    No search, fitting, tie selection, spatial uncertainty, acceptance or image
    transport. The second support policy requires the complete feature kernel
    to stay within supplied masks. Neither masks nor pixels prove stable-ground
    semantics or independent localization evidence.
    """
    inputs=(reference_features,moving_features,reference_valid,moving_valid,component_support)
    if any(np.ma.isMaskedArray(x) for x in inputs):
        raise ValueError('Masked unknowns must remain explicit')
    ref=np.asarray(reference_features);mov=np.asarray(moving_features)
    masks=[np.asarray(x) for x in inputs[2:]]
    if (ref.ndim!=2 or not all(ref.shape) or mov.shape!=ref.shape
            or ref.dtype.kind not in 'fiu' or mov.dtype.kind not in 'fiu'
            or not np.isfinite(ref).all() or not np.isfinite(mov).all()):
        raise ValueError('Finite numeric feature grids required')
    if any(x.shape!=ref.shape or x.dtype.kind!='b' for x in masks):
        raise ValueError('Explicit boolean masks on the feature grid required')
    if (not isinstance(observed_offset_px,tuple) or len(observed_offset_px)!=2
            or any(isinstance(v,bool) or not isinstance(v,int) for v in observed_offset_px)):
        raise ValueError('One frozen integer observed-feature offset required')
    if (isinstance(minimum_samples,bool) or not isinstance(minimum_samples,int) or minimum_samples<3
            or isinstance(feature_radius,bool) or not isinstance(feature_radius,int) or feature_radius<0):
        raise ValueError('Invalid fixed support policy')
    dx,dy=observed_offset_px;h,w=ref.shape
    y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)

    def shifted(values):
        out=np.zeros_like(values)
        if y0<y1 and x0<x1:
            out[y0:y1,x0:x1]=values[y0+dy:y1+dy,x0+dx:x1+dx]
        return out

    shifted_moving=shifted(mov)

    def evaluate(radius):
        rv,mv,component=(_erode_mask(x,radius) for x in masks)
        # Both zero and candidate see identical reference pixels. No wrapping.
        common=rv & component & mv & shifted(mv) & shifted(component)
        n=int(common.sum())
        record={'pixel_count':n,'minimum_samples':minimum_samples,
                'zero_score':None,'candidate_score':None,'score_difference':None,
                'status':'INSUFFICIENT_COMMON_SUPPORT'}
        if n>=minimum_samples:
            zero=_pearson_correlation(ref[common],mov[common])
            candidate=_pearson_correlation(ref[common],shifted_moving[common])
            record.update(zero_score=zero,candidate_score=candidate,
                score_difference=None if zero is None or candidate is None else candidate-zero,
                status='NUMERIC_INTERNAL_REPLAY_ONLY' if zero is not None and candidate is not None
                else 'CONSTANT_OR_UNINFORMATIVE_FEATURES')
        return record

    return {'observed_offset_px':[dx,dy],
            'paired_point_support':evaluate(0),
            'full_kernel_support':evaluate(feature_radius),
            'feature_kernel_radius_px':feature_radius,
            'zero_vector_self_comparison':dx==0 and dy==0,
            'independent_validation':False,'formal_acceptance':False,
            'relative_registration_uncertainty_m':None,
            'absolute_position_accuracy':'UNKNOWN',
            'candidate_reestimated':False}
