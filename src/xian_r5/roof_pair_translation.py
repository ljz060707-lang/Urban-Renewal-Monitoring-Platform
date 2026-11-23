"""Internal translation consistency of frozen roof-shape hypotheses.

Not a formal FIT/HOLDOUT review: roof centroids are not stable-ground ties,
shape hypotheses do not prove physical identity, and residuals are not metric
localization bounds. All hypotheses stay in the cohort, including failures.
"""
import math
import numpy as np


def roof_pair_translation(records, support_bounds, *, minimum_per_fold=6,
                          minimum_spatial_groups=3, maximum_residual_m=2.):
    xmin, ymin, xmax, ymax = support_bounds
    if not all(math.isfinite(v) for v in support_bounds) or xmin>=xmax or ymin>=ymax:
        raise ValueError('Finite nonempty support required')
    if minimum_per_fold < 3 or not 1 <= minimum_spatial_groups <= 9 or maximum_residual_m <= 0:
        raise ValueError('Invalid fixed diagnostic policy')
    if len({r['component_key'] for r in records}) != len(records) or len({r['source_fid'] for r in records}) != len(records):
        raise ValueError('Repeated source/reference identity is not independent evidence')
    rows=[]
    for r in sorted(records,key=lambda x:x['component_key']):
        src=np.asarray(r['source_centroid_xy'],float);ref=np.asarray(r['reference_centroid_xy'],float)
        if src.shape!=(2,) or ref.shape!=(2,) or not np.isfinite(np.r_[src,ref]).all():
            raise ValueError('Finite source and reference centroids required')
        if not (xmin<=src[0]<=xmax and ymin<=src[1]<=ymax and xmin<=ref[0]<=xmax and ymin<=ref[1]<=ymax):
            raise ValueError('Centroid outside exact support')
        vector=ref-src
        if not np.allclose(vector,r['frozen_pair_vector_m'],rtol=0,atol=1e-7):
            raise ValueError('Frozen pair vector recomputation mismatch')
        # Spatial checker fixed by SOURCE location, never residual magnitude.
        u=(src[0]-xmin)/(xmax-xmin);v=(ymax-src[1])/(ymax-ymin)
        col=min(3,int(u*4));row=min(3,int(v*4))
        group=f'{min(2,int(v*3))}:{min(2,int(u*3))}'
        rows.append({'component_key':r['component_key'],'source_fid':r['source_fid'],
            'fold':'CALIBRATION' if (row+col)%2==0 else 'INTERNAL_CHECK',
            'spatial_group':group,'pair_vector_m':vector.tolist()})
    fit=[r for r in rows if r['fold']=='CALIBRATION'];check=[r for r in rows if r['fold']=='INTERNAL_CHECK']
    reasons=[]
    for name,fold in [('CALIBRATION',fit),('INTERNAL_CHECK',check)]:
        if len(fold)<minimum_per_fold:reasons.append(name+'_COUNT_INSUFFICIENT')
        if len({r['spatial_group'] for r in fold})<minimum_spatial_groups:
            reasons.append(name+'_SPATIAL_COVERAGE_INSUFFICIENT')
    vector=None;fitmax=checkmax=loomax=perturbmax=None
    if not reasons:
        a=np.array([r['pair_vector_m'] for r in fit])
        # Median estimates a candidate ONLY. Max residuals judge consistency.
        vector=np.median(a,axis=0)
        for r in rows:r['candidate_residual_m']=float(np.linalg.norm(np.asarray(r['pair_vector_m'])-vector))
        fitmax=max(r['candidate_residual_m'] for r in fit)
        checkmax=max(r['candidate_residual_m'] for r in check)
        loo=np.array([np.median(np.delete(a,i,axis=0),axis=0) for i in range(len(a))])
        loomax=float(np.linalg.norm(a-loo,axis=1).max())
        perturbmax=float(np.linalg.norm(loo-vector,axis=1).max())
        if fitmax>maximum_residual_m:reasons.append('CALIBRATION_VECTORS_NOT_UNIFORM')
        if checkmax>maximum_residual_m:reasons.append('INTERNAL_CHECK_VECTORS_NOT_UNIFORM')
        if loomax>maximum_residual_m:reasons.append('LEAVE_ONE_SOURCE_OUT_INCONSISTENT')
    return {'records':rows,'calibration_n':len(fit),'internal_check_n':len(check),
        'diagnostic_candidate_vector_m':None if vector is None else vector.tolist(),
        'calibration_max_centroid_residual_m':fitmax,'internal_check_max_centroid_residual_m':checkmax,
        'leave_one_source_out_max_residual_m':loomax,'leave_one_source_out_max_vector_perturbation_m':perturbmax,
        'uniform_translation_diagnostically_consistent':not reasons,'reasons':reasons,
        'all_frozen_pairs_retained':True,'residual_based_outlier_selection':False,
        'external_unseen_validation':False,'source_roof_purity_certified':False,
        'physical_correspondence_certified':False,'formal_alignment_accepted':False,
        'relative_registration_uncertainty_m':None,'absolute_position_accuracy':'UNKNOWN',
        'training_admitted':False,'candidate_applied':False}
