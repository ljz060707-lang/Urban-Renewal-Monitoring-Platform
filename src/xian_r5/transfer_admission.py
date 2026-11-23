"""Whole-window disposition for experimental roof transfer, not formal GT.

Every target pixel receives a disposition. Unknown alignment never becomes
background; a threshold-stable comparison never upgrades spatial precision.
"""
import numpy as np
from .fraction_shadow import NODATA

DISPOSITIONS = {
    0: 'EXPERIMENT_SELECTED_NOT_GT', 1: 'ALIGNMENT_UNRESOLVED',
    2: 'TRANSPORTED_SUPPORT_INCOMPLETE', 3: 'BOUNDARY_RISK',
    4: 'SOURCE_SEAM_RISK', 5: 'OVERLAP_CONFLICT', 6: 'INPUT_INVALID',
}


def transfer_admission(raw, candidate, quality, labels, weights, input_valid):
    arrays = (raw, candidate, quality, labels, weights, input_valid)
    if any(np.ma.isMaskedArray(a) for a in arrays): raise ValueError('Explicit masks required')
    raw, candidate, quality, labels, weights, input_valid = map(np.asarray, arrays)
    size = raw.shape
    if (raw.ndim != 2 or any(a.shape != size for a in (candidate, labels, weights, input_valid))
            or quality.shape != (10, *size) or input_valid.dtype.kind != 'b'
            or quality.dtype.kind not in 'iu'):
        raise ValueError('All arrays must share a two-dimensional grid')
    if (any(not np.isfinite(a).all() for a in (raw, candidate, weights))
            or not np.isin(labels, [0, 1, 255]).all() or not np.isin(weights, [0, 1]).all()
            or not np.array_equal(weights > 0, labels != 255)):
        raise ValueError('Invalid label/weight/ignore binding')
    if any(((a[a != NODATA] < 0) | (a[a != NODATA] > 1 + 1e-6)).any() for a in (raw, candidate)):
        raise ValueError('Fractions must be whole-pixel proportions')
    # This branch is explicitly unverified. An unexpected GT flag is a breach,
    # not an opportunity to count formal training eligibility.
    if quality[3].any() or quality[4].any() or quality[9].any():
        raise ValueError('Experimental-only branch must not contain verified/date/GT flags')
    selected = labels != 255
    available = candidate != NODATA
    if np.any(selected & ((quality[0] != 1) | ~available | ~input_valid
                           | quality[5].astype(bool) | quality[6].astype(bool) | quality[7].astype(bool))):
        raise ValueError('Selected samples bypass an experimental quality gate')
    if not np.array_equal(labels[selected], (candidate[selected] > .43).astype(np.uint8)):
        raise ValueError('Frozen strict threshold disagreement')
    disposition = np.full(size, 1, np.uint8)
    numeric = quality[2].astype(bool)
    disposition[numeric & ~selected] = 2
    disposition[numeric & quality[5].astype(bool)] = 3
    disposition[numeric & quality[6].astype(bool)] = 4
    disposition[numeric & quality[7].astype(bool)] = 5
    disposition[~input_valid] = 6
    disposition[selected] = 0
    counts = {name: int((disposition == code).sum()) for code, name in DISPOSITIONS.items()}
    if sum(counts.values()) != raw.size: raise RuntimeError('Incomplete target disposition')
    difference = np.full(size, NODATA, np.float32)
    comparable = available & (raw != NODATA)
    difference[comparable] = candidate[comparable] - raw[comparable]
    comparison = selected & comparable
    raw_class = raw > .43; candidate_class = candidate > .43
    flips = comparison & (raw_class != candidate_class)
    summary = {'denominator': raw.size, 'disposition_counts_exclusive': counts,
        'experimental_selected': int(selected.sum()), 'measured_gt_eligible': 0,
        'comparison_scope': 'SAME_SELECTED_PIXELS_RAW_VS_POINT_RGB_CANDIDATE_NOT_OLD_ORIGINAL_LABEL',
        'label_unchanged_comparison_pixels': int((comparison & ~flips).sum()),
        'threshold_flips': int(flips.sum()),
        'raw_background_to_candidate_building': int((flips & candidate_class).sum()),
        'raw_building_to_candidate_background': int((flips & raw_class).sum()),
        'mean_absolute_fraction_difference_diagnostic': float(np.abs(difference[comparison]).mean()) if comparison.any() else None,
        'maximum_absolute_fraction_difference_diagnostic': float(np.abs(difference[comparison]).max()) if comparison.any() else None,
        'label_stability_is_validated_error_envelope': False,
        'time_policy': 'USER_AUTHORIZED_NOMINAL_2019_NOT_VERIFIED_SAME_EPOCH',
        'precise_error_envelope': None, 'external_spatial_validation': 'NOT_ESTABLISHED',
        'absolute_position_accuracy': 'UNKNOWN', 'physical_ground_footprints': 0}
    return {'disposition': disposition, 'difference': difference, 'summary': summary}
