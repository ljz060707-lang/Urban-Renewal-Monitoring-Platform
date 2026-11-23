"""Source-anchored equal-population search against reference image context.

Offsets are SOURCE_TO_REFERENCE pixel corrections, not observed offsets.
All candidates see identical source feature observations; no annotation-shaped
target support decides which offset qualifies. Both spatial folds remain
strictly stencil-closed. This is experimental roof-image alignment only.
"""
import numpy as np
from .supported_ground_features import directional_features
from .base_tie import _pearson_correlation


def _sample_target(values,dx,dy):
    out=np.zeros_like(values);_,h,w=values.shape
    x0,x1=max(0,-dx),min(w,w-dx);y0,y1=max(0,-dy),min(h,h-dy)
    if x0<x1 and y0<y1:out[:,y0:y1,x0:x1]=values[:,y0+dy:y1+dy,x0+dx:x1+dx]
    return out


def source_template_check(source_rgb,target_rgb,source_roof_mask,target_valid,component,
                          *,radius=3,minimum_samples=100,fold_block_size=16,
                          minimum_gain=.01,minimum_peak_gap=.02,minimum_positive_blocks=3,
                          minimum_informative_block_samples=20):
    if (type(radius) is not int or radius<1 or type(minimum_samples) is not int or minimum_samples<3
            or type(fold_block_size) is not int or fold_block_size<4):
        raise ValueError('Explicit fixed search and sample policy required')
    if any(np.ma.isMaskedArray(a) for a in (source_roof_mask,target_valid,component)):
        raise ValueError('Explicit boolean masks required')
    masks=[np.asarray(a) for a in (source_roof_mask,target_valid,component)]
    if any(a.dtype.kind!='b' or a.shape!=source_rgb.shape[:2] for a in masks) or target_rgb.shape!=source_rgb.shape:
        raise ValueError('Shared image grid and boolean scope required')
    source_roof_mask,target_valid,component=masks
    yy,xx=np.indices(component.shape);fit=(yy//fold_block_size+xx//fold_block_size)%2==0
    offsets=[(dx,dy) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
    def population(fold):
        src,sv=directional_features(source_rgb,source_roof_mask&component&fold)
        target,tv=directional_features(target_rgb,target_valid&component&fold)
        shared=sv.copy()
        for dx,dy in offsets:shared&=_sample_target(tv,dx,dy)
        return src,target,shared
    src,target,fit_common=population(fit)
    fit_n=int(fit_common.any(axis=0).sum());surfaces=[]
    if fit_n>=minimum_samples:
        for dx,dy in offsets:
            sampled=_sample_target(target,dx,dy)
            score=_pearson_correlation(src[fit_common],sampled[fit_common])
            if score is not None:surfaces.append({'correction_dx_px':dx,'correction_dy_px':dy,'corr':score,'n':fit_n})
    surfaces.sort(key=lambda x:(-x['corr'],abs(x['correction_dx_px'])+abs(x['correction_dy_px']),x['correction_dy_px'],x['correction_dx_px']))
    best=surfaces[0] if surfaces else None;reasons=[];blocks=[]
    check={'n':0,'zero_corr':None,'candidate_corr':None,'gain':None}
    positive=negative=0;gap=None
    if best is None:reasons.append('INSUFFICIENT_FIXED_FIT_COHORT_OR_NO_SCORE')
    else:
        dx,dy=best['correction_dx_px'],best['correction_dy_px']
        gap=best['corr']-surfaces[1]['corr'] if len(surfaces)>1 else None
        if abs(dx)>=radius or abs(dy)>=radius:reasons.append('FIT_PEAK_AT_SEARCH_LIMIT')
        if gap is None or gap<minimum_peak_gap:reasons.append('FIT_PEAK_NOT_DISTINCT')
        if not fit_common[0].any() or not fit_common[1].any():reasons.append('FIT_SINGLE_AXIS_APERTURE_RISK')
        cs,ct,cm=population(~fit);n=int(cm.any(axis=0).sum());check['n']=n
        sampled=_sample_target(ct,dx,dy)
        if n<minimum_samples:reasons.append('INSUFFICIENT_FIXED_CHECK_COHORT')
        else:
            zero=_pearson_correlation(cs[cm],ct[cm]);candidate=_pearson_correlation(cs[cm],sampled[cm])
            gain=None if zero is None or candidate is None else candidate-zero
            check.update(zero_corr=zero,candidate_corr=candidate,gain=gain)
            if not cm[0].any() or not cm[1].any():reasons.append('CHECK_SINGLE_AXIS_APERTURE_RISK')
            if dx==0 and dy==0:reasons.append('ZERO_SELF_COMPARISON_NOT_NONZERO_CORRECTION_EVIDENCE')
            elif gain is None or gain<minimum_gain:reasons.append('CHECK_DID_NOT_IMPROVE_BASELINE')
            for rr,cc in sorted({(int(a),int(b)) for a,b in zip((yy//fold_block_size)[cm.any(axis=0)],(xx//fold_block_size)[cm.any(axis=0)])}):
                mask=cm&((yy//fold_block_size==rr)&(xx//fold_block_size==cc))[None]
                count=int(mask.any(axis=0).sum())
                if count<minimum_informative_block_samples:
                    blocks.append({'row':rr,'col':cc,'n':count,'gain':None,'status':'INSUFFICIENT_BLOCK_SUPPORT'});continue
                z=_pearson_correlation(cs[mask],ct[mask]);c=_pearson_correlation(cs[mask],sampled[mask])
                g=None if z is None or c is None else c-z
                blocks.append({'row':rr,'col':cc,'n':count,'gain':g,'status':'INTERNAL_BLOCK_ONLY'})
                if g is not None:
                    positive+=g>=minimum_gain;negative+=g<-minimum_gain
            if positive<minimum_positive_blocks or negative>0:reasons.append('SPATIAL_CHECK_BLOCKS_NOT_CONSISTENT')
    return {'best':best,'fit_fixed_population_n':fit_n,'fit_scalar_axis_observations':int(fit_common.sum()),
        'candidate_scores':surfaces,'candidate_count':len(surfaces),'fit_peak_gap':gap,'check':check,
        'check_blocks':blocks,'positive_blocks':int(positive),'materially_negative_blocks':int(negative),
        'engineering_experiment_supported':not reasons,'reasons':reasons,
        'cohort_identical_for_every_search_offset':True,'target_annotation_mask_used':False,
        'offset_convention':'SOURCE_TO_REFERENCE_CORRECTION_PIXEL','formal_alignment_accepted':False,
        'relative_registration_uncertainty_m':None,'absolute_position_accuracy':'UNKNOWN'}
