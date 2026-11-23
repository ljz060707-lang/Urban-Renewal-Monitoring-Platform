"""Conservative manual-reference corroboration of an existing pseudo-label.

Only removes samples. It never substitutes reference shapes, changes a vector,
adds unsupported supervision, or calls agreeing below-threshold areas measured
background. A positive-only companion makes the negative assumption optional.
"""
import numpy as np


def reference_label_filter(label, weight, candidate_fraction, reference_fraction,
                           reference_overlap, reference_truncated):
    arrays=[np.asarray(a) for a in (label,weight,candidate_fraction,reference_fraction,
                                   reference_overlap,reference_truncated)]
    if any(np.ma.isMaskedArray(a) for a in (label,weight,candidate_fraction,reference_fraction,
                                          reference_overlap,reference_truncated)):
        raise ValueError('Explicit masks required')
    label,weight,fraction,reference,overlap,truncated=arrays
    if label.ndim!=2 or any(a.shape!=label.shape for a in arrays):raise ValueError('Shared exact grid required')
    if (label.dtype.kind not in 'ui' or overlap.dtype.kind!='b' or truncated.dtype.kind!='b'
            or not np.isfinite(weight).all() or not np.isin(weight,[0,1]).all()
            or not np.isin(label,[0,1,255]).all() or not np.array_equal(weight>0,label!=255)
            or not np.isfinite(reference).all() or np.any((reference<0)|(reference>1))):
        raise ValueError('Invalid explicit labels/weights/reference')
    baseline=weight>0
    if (not np.isfinite(fraction[baseline]).all() or
            np.any((fraction[baseline]<0)|(fraction[baseline]>1)) or
            not np.array_equal(label[baseline],(fraction[baseline].astype(np.float64)>.43).astype(np.uint8))):
        raise ValueError('Inherited strict >0.43 label/fraction mismatch')
    agrees=label==(reference.astype(np.float64)>.43).astype(np.uint8)
    disagreement=baseline&~agrees
    overlap_excluded=baseline&overlap;truncated_excluded=baseline&truncated
    selected=baseline&agrees&~overlap&~truncated
    output=np.full(label.shape,255,np.uint8);output[selected]=label[selected]
    positive=selected&(output==1);background=selected&(output==0)
    positive_only=np.full(label.shape,255,np.uint8);positive_only[positive]=1
    flags=(disagreement.astype(np.uint8)+2*overlap_excluded.astype(np.uint8)+
           4*truncated_excluded.astype(np.uint8))
    return {'label':output,'weight':selected.astype(np.float32),
        'positive_only_label':positive_only,'positive_only_weight':positive.astype(np.float32),
        'exclusion_flags':flags,'quality':np.stack([baseline,agrees&baseline,overlap,truncated,
            selected,positive,background,np.zeros_like(selected),np.zeros_like(selected),
            np.zeros_like(selected)]).astype(np.uint8),
        'quality_bands':['INHERITED_EXPERIMENT_SELECTED','MANUAL_THRESHOLD_CLASS_AGREES',
            'MANUAL_POLYGON_OVERLAP','MANUAL_ROOF_TRUNCATED_AT_ROI','EXPERIMENT_SELECTED',
            'REFERENCE_CORROBORATED_POSITIVE','ASSUMED_BELOW_THRESHOLD_BACKGROUND',
            'DISPLACEMENT_INDEPENDENTLY_VERIFIED','EXACT_TIME_COMPATIBILITY_VERIFIED','FORMAL_GT_ADMITTED'],
        'counts':{'full_window_denominator':label.size,'inherited_experiment_selected':int(baseline.sum()),
            'reference_disagreement_exclusions':int(disagreement.sum()),
            'reference_overlap_exclusions':int(overlap_excluded.sum()),
            'reference_truncation_exclusions':int(truncated_excluded.sum()),
            'union_reference_exclusions':int((baseline&~selected).sum()),
            'recommended_experiment_selected':int(selected.sum()),
            'reference_corroborated_positive':int(positive.sum()),
            'assumed_background':int(background.sum()),'ignore':int((~selected).sum()),
            'new_samples_outside_inherited_selection':0,'formal_gt':0},
        'threshold':.43,'comparison':'>','new_displacement_application':False,
        'background_semantics':'EXPERIMENTAL_ASSUMPTION_NOT_VERIFIED_ABSENCE',
        'fraction_semantics':'UNCHANGED_SOURCE_DERIVED_CANDIDATE_NOT_REFERENCE_SUBSTITUTION',
        'external_unseen_validation':False,'absolute_position_accuracy':'UNKNOWN'}
