"""Fixed source-position reference corroboration, not offset-shaped target masks."""
import numpy as np


def reference_core_cohort(raw_fraction, manual_fraction, manual_overlap,
                          manual_truncated, source_valid):
    items=(raw_fraction,manual_fraction,manual_overlap,manual_truncated,source_valid)
    if any(np.ma.isMaskedArray(a) for a in items):raise ValueError('Explicit masks required')
    raw,manual,overlap,truncated,valid=[np.asarray(a) for a in items]
    if raw.ndim!=2 or any(a.shape!=raw.shape for a in (manual,overlap,truncated,valid)):
        raise ValueError('Exact shared source/reference grid required')
    if any(a.dtype.kind!='b' for a in (overlap,truncated,valid)):
        raise ValueError('Explicit boolean risk and validity masks required')
    if any(not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in (raw,manual)):
        raise ValueError('Finite known source and annotation fractions required')
    prior=(raw>=.8)&valid
    mask=prior&(manual>=.8)&~overlap&~truncated
    return {'mask':mask,'source_prior_pixels':int(prior.sum()),
        'reference_corroborated_pixels':int(mask.sum()),'excluded_prior_pixels':int((prior&~mask).sum()),
        'source_fraction_minimum':.8,'reference_fraction_minimum':.8,
        'manual_position_used_to_select_source_population':True,
        'manual_reference_shifted_to_choose_cohort':False,
        'target_feature_support_uses_annotation':False,
        'cohort_may_not_change_with_search_offset':True,
        'native_source_roof_purity_certified':False,'external_unseen_validation':False,
        'absolute_position_accuracy':'UNKNOWN'}
