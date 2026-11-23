"""User-authorized nominal-year experiment, separate from measured GT gates."""
from datetime import date
import numpy as np
from .fraction_shadow import NODATA


def nominal_year_binding(label_year, observed_date, *, experiment_year=2019):
    """Use dataset year for experimental grouping; never alter observed dates."""
    if type(label_year) is not int or type(experiment_year) is not int or label_year!=experiment_year:
        raise ValueError('Dataset label year differs from the single-year scope')
    if observed_date is not None:date.fromisoformat(observed_date)
    return {'experiment_year':experiment_year,'dataset_label_year':label_year,
        'observed_date':observed_date,'year_policy':'USER_AUTHORIZED_NOMINAL_YEAR',
        'same_actual_epoch_verified':False,'permits_temporal_change_inference':False}


def experimental_pseudolabel(fraction,scope_fraction,quality,input_valid,*,
                            allow_unverified_geometry=False,allow_assumed_background=False):
    """Strict tau=.43 samples in supported, non-seam, non-boundary core only.

    This explicitly allows unverified alignment and model-derived background
    assumptions for experimentation, not scientific GT or accuracy promotion.
    Weight 1 means selected experimental sample, never measured confidence.
    """
    if allow_unverified_geometry is not True or allow_assumed_background is not True:
        raise ValueError('Explicit experimental risk policy required')
    if any(np.ma.isMaskedArray(x) for x in (fraction,scope_fraction,quality,input_valid)):
        raise ValueError('Unknown masks must be explicit')
    fraction=np.asarray(fraction);scope_fraction=np.asarray(scope_fraction)
    quality=np.asarray(quality);input_valid=np.asarray(input_valid)
    if (fraction.ndim!=2 or scope_fraction.shape!=fraction.shape or input_valid.shape!=fraction.shape
            or quality.shape!=(10,*fraction.shape) or quality.dtype.kind not in 'iu'
            or input_valid.dtype.kind!='b'):
        raise ValueError('Experiment arrays must share the exact grid')
    if (not np.isfinite(fraction).all() or not np.isfinite(scope_fraction).all()
            or not np.isin(quality[0],[0,1,2,3,4]).all()
            or ((scope_fraction<0)|(scope_fraction>1+1e-6)).any()):
        raise ValueError('Invalid fractions/quality codes')
    available=fraction!=NODATA
    if ((fraction[available]<0)|(fraction[available]>1+1e-6)).any():
        raise ValueError('Fraction outside [0,1]')
    selected=(available&(scope_fraction>=1-1e-6)&input_valid&
              (quality[0]==1)&~quality[5].astype(bool)&~quality[6].astype(bool)&~quality[7].astype(bool))
    label=np.full(fraction.shape,255,np.uint8)
    label[selected]=(fraction[selected]>.43).astype(np.uint8)
    weight=selected.astype(np.float32)
    return {'label':label,'experiment_weight':weight,'selected':selected,
        'counts':{'denominator':fraction.size,'experimental_samples':int(selected.sum()),
            'pseudo_building':int((label==1).sum()),'pseudo_below_threshold':int((label==0).sum()),
            'ignore':int((label==255).sum()),'scientifically_qualified_gt':0},
        'threshold':.43,'threshold_rule':'STRICT_GT',
        'background_semantics':'MODEL_DERIVED_BELOW_THRESHOLD_NOT_VERIFIED_ABSENCE',
        'training_use_mode':'EXPERIMENT_ONLY_NOT_MEASURED_GT',
        'independent_accuracy_verified':False}


def experiment_batch(inputs,label,weight):
    """Read-only loss/dataloader smoke interface; never performs optimization."""
    if any(np.ma.isMaskedArray(a) for a in (inputs,label,weight)):
        raise ValueError('Explicit arrays required')
    inputs=np.asarray(inputs);label=np.asarray(label);weight=np.asarray(weight)
    if inputs.shape!=(6,*label.shape) or label.ndim!=2 or weight.shape!=label.shape:
        raise ValueError('Expected 6-band input aligned to labels/weights')
    if (not np.isfinite(inputs).all() or not np.isfinite(weight).all()
            or not np.isin(label,[0,1,255]).all() or not np.isin(weight,[0.,1.]).all()
            or not np.array_equal(weight>0,label!=255)):
        raise ValueError('Invalid model input or sample/ignore relationship')
    target=label.astype(np.int64);target[target==255]=-100
    return {'inputs':inputs.copy(),'target':target,'weight':weight.copy(),
        'loss_ignore_index':-100,'optimization_steps':0,'model_weights_loaded':False}


def alignment_experiment_readiness(cells):
    """Keep unsupported kernels/search-limit peaks out of recommended samples.

    Even passing these numerical preconditions does not certify alignment.
    """
    output=[]
    for cell in cells:
        vector=cell['candidate_correction_m'];reasons=[]
        if vector is None:
            reasons.append('NO_NUMERIC_VECTOR')
        else:
            diagnosis=cell['correlation_diagnostic'];best=diagnosis['best']
            radius=diagnosis['search_radius_px']
            support=cell['full_feature_kernel_supported_pixels']
            if type(support) is not int or support<0:raise ValueError('Kernel support count invalid')
            if support==0:reasons.append('NO_COMPLETE_FEATURE_KERNEL_SUPPORT')
            if best is None:raise ValueError('Vector lacks diagnostic peak')
            if abs(best['dx_px'])>=radius or abs(best['dy_px'])>=radius:
                reasons.append('PEAK_AT_SEARCH_LIMIT')
        output.append({'region_id':cell['region_id'],'candidate_experiment_supported':not reasons,
                       'reasons':reasons,'formal_alignment_accepted':False})
    return output
