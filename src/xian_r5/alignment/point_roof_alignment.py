"""Point-RGB fixed-source replay; experimental image-grid evidence only.

Each offset sees exactly the same source pixels. Target support is raster
validity, cell, and fold, never an annotation or model output. Per-band
centering avoids mistaking a shared average RGB colour for spatial evidence.
"""
import numpy as np
from .source_template_alignment import _sample_target
from .base_tie import _pearson_correlation


def _band_scores(source, target, population):
    return [_pearson_correlation(source[k][population], target[k][population]) for k in range(3)]


def point_roof_check(source_rgb, target_rgb, source_roof_mask, target_valid, component,
                     *, radius=3, minimum_samples=100, block_size=16,
                     minimum_gain=.01, minimum_peak_gap=.02,
                     minimum_block_samples=20, minimum_positive_blocks=3):
    """Return diagnostics and a fail-closed internal experimental recommendation."""
    if (type(radius) is not int or radius < 1 or type(minimum_samples) is not int or minimum_samples < 3
            or type(block_size) is not int or block_size < 4):
        raise ValueError('Fixed integer search/sample policy required')
    if np.ndim(source_rgb) != 3 or source_rgb.shape[-1] != 3 or target_rgb.shape != source_rgb.shape:
        raise ValueError('Shared three-band RGB grid required')
    masks = [np.asarray(a) for a in (source_roof_mask, target_valid, component)]
    if any(np.ma.isMaskedArray(a) for a in (source_roof_mask, target_valid, component)) or any(
            a.dtype.kind != 'b' or a.shape != source_rgb.shape[:2] for a in masks):
        raise ValueError('Explicit boolean masks on the shared grid required')
    source_roof_mask, target_valid, component = masks
    source = np.moveaxis(np.asarray(source_rgb, dtype=np.float64), -1, 0)
    target = np.moveaxis(np.asarray(target_rgb, dtype=np.float64), -1, 0)
    source_finite = np.isfinite(source).all(axis=0)
    target_finite = np.isfinite(target).all(axis=0)
    yy, xx = np.indices(component.shape)
    fit = (yy // block_size + xx // block_size) % 2 == 0
    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

    def population(fold):
        common = source_roof_mask & source_finite & component & fold
        target_scope = target_valid & target_finite & component & fold
        for dx, dy in offsets:
            common &= _sample_target(target_scope[None], dx, dy)[0]
        return common

    fm = population(fit); n = int(fm.sum()); scores = []
    if n >= minimum_samples:
        for dx, dy in offsets:
            bands = _band_scores(source, _sample_target(target, dx, dy), fm)
            # All three bands must carry spatial variation; no cross-band mean.
            if all(c is not None for c in bands):
                scores.append({'correction_dx_px': dx, 'correction_dy_px': dy,
                    'corr': float(sum(bands) / 3), 'band_corr': bands, 'n': n})
    scores.sort(key=lambda r: (-r['corr'], abs(r['correction_dx_px']) + abs(r['correction_dy_px']),
                               r['correction_dy_px'], r['correction_dx_px']))
    best = scores[0] if scores else None
    reasons = []; blocks = []; gap = None; positive = negative = 0
    check = {'n': 0, 'zero_corr': None, 'candidate_corr': None, 'gain': None,
             'band_gain': None, 'positive_bands': 0, 'materially_negative_bands': 0}
    if best is None:
        reasons.append('INSUFFICIENT_FIXED_FIT_COHORT_OR_NO_SPATIAL_RGB_VARIATION')
    else:
        dx, dy = best['correction_dx_px'], best['correction_dy_px']
        gap = best['corr'] - scores[1]['corr'] if len(scores) > 1 else None
        if abs(dx) >= radius or abs(dy) >= radius: reasons.append('FIT_PEAK_AT_SEARCH_LIMIT')
        if gap is None or gap < minimum_peak_gap: reasons.append('FIT_PEAK_NOT_DISTINCT')
        cm = population(~fit); count = int(cm.sum()); check['n'] = count
        sampled = _sample_target(target, dx, dy)
        if count < minimum_samples: reasons.append('INSUFFICIENT_FIXED_CHECK_COHORT')
        else:
            zero = _band_scores(source, target, cm); candidate = _band_scores(source, sampled, cm)
            if any(v is None for v in zero + candidate):
                reasons.append('CHECK_RGB_SPATIAL_VARIATION_UNAVAILABLE')
            else:
                bg = [c - z for c, z in zip(candidate, zero)]
                gain = float(sum(bg) / 3)
                check.update(zero_corr=float(sum(zero) / 3), candidate_corr=float(sum(candidate) / 3),
                    gain=gain, band_gain=bg, positive_bands=sum(v >= minimum_gain for v in bg),
                    materially_negative_bands=sum(v < -minimum_gain for v in bg))
                if dx == 0 and dy == 0: reasons.append('ZERO_SELF_COMPARISON_NOT_NONZERO_CORRECTION_EVIDENCE')
                elif gain < minimum_gain: reasons.append('CHECK_DID_NOT_IMPROVE_BASELINE')
                if check['positive_bands'] < 2 or check['materially_negative_bands']:
                    reasons.append('CHECK_SPECTRAL_BANDS_NOT_CONSISTENT')
            block_ids = sorted({(int(r), int(c)) for r, c in zip((yy // block_size)[cm], (xx // block_size)[cm])})
            for rr, cc in block_ids:
                bm = cm & (yy // block_size == rr) & (xx // block_size == cc)
                bn = int(bm.sum()); gain = None
                if bn >= minimum_block_samples:
                    z = _band_scores(source, target, bm); c = _band_scores(source, sampled, bm)
                    if all(v is not None for v in z + c): gain = float(sum(a - b for a, b in zip(c, z)) / 3)
                blocks.append({'row': rr, 'col': cc, 'n': bn, 'gain': gain,
                    'status': 'INTERNAL_BLOCK_ONLY' if gain is not None else 'INSUFFICIENT_BLOCK_INFORMATION'})
                if gain is not None:
                    positive += gain >= minimum_gain; negative += gain < -minimum_gain
            if positive < minimum_positive_blocks or negative:
                reasons.append('SPATIAL_CHECK_BLOCKS_NOT_CONSISTENT')
    return {'best': best, 'candidate_scores': scores, 'candidate_count': len(scores),
        'fit_fixed_population_n': n, 'fit_peak_gap': gap, 'check': check, 'check_blocks': blocks,
        'positive_blocks': int(positive), 'materially_negative_blocks': int(negative),
        'engineering_experiment_supported': not reasons, 'reasons': reasons,
        'cohort_identical_for_every_search_offset': True, 'target_annotation_mask_used': False,
        'feature_operator': 'POINT_RGB_INDEPENDENT_PER_BAND_CENTERED_CORRELATION',
        'offset_convention': 'SOURCE_TO_REFERENCE_CORRECTION_PIXEL',
        'formal_alignment_accepted': False, 'relative_registration_uncertainty_m': None,
        'absolute_position_accuracy': 'UNKNOWN'}


def reconcile_roof_descriptors(point_result, derivative_result):
    """Never select a descriptor by whichever delivers more pseudo-labels."""
    reasons = list(point_result['reasons']); p = point_result['best']; d = derivative_result['best']
    derivative_computable = d is not None and derivative_result['check']['n'] >= 100
    if p is not None and derivative_computable:
        if (p['correction_dx_px'], p['correction_dy_px']) != (d['correction_dx_px'], d['correction_dy_px']):
            reasons.append('ROOF_DESCRIPTOR_DIRECTION_CONFLICT')
    return {'experimental_recommendation': not reasons,
        'reasons': reasons, 'derivative_computable': derivative_computable,
        'independent_physical_validation': False, 'formal_acceptance': False}
