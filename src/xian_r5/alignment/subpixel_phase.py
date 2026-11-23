"""Fixed-grid local inverse DFT refinement after the integer peak is removed.

This is a numerical adapter, not an uncertainty estimator or tie certification.
"""
import numpy as np


def refine_zero_peak(phase):
    if np.ma.isMaskedArray(phase):raise ValueError('Unknown phase support forbidden')
    phase=np.asarray(phase)
    if phase.ndim!=2 or min(phase.shape)<8 or not np.isfinite(phase).all() or not np.any(np.abs(phase)>0):
        raise ValueError('Finite nonempty 2D phase spectrum required')
    h,w=phase.shape
    offsets=np.linspace(-.5,.5,41)
    rows=np.exp(2j*np.pi*np.outer(offsets,np.fft.fftfreq(h)))
    cols=np.exp(2j*np.pi*np.outer(offsets,np.fft.fftfreq(w)))
    # Avoid native complex BLAS dispatch in the preserved Windows runtimes.
    intermediate=np.einsum('yh,hk->yk',rows,phase,optimize=False)
    magnitude=np.abs(np.einsum('yk,xk->yx',intermediate,cols,optimize=False))/(h*w)
    best=np.unravel_index(magnitude.argmax(),magnitude.shape)
    peak=magnitude[best]
    if np.ptp(magnitude)<=max(peak,1)*1e-10:
        raise ValueError('No unique subpixel peak in phase support')
    iy,ix=best
    # A direction with no peak contrast is an aperture ambiguity, not a tie.
    if np.ptp(magnitude[:,ix])<=max(peak,1)*1e-10 or np.ptp(magnitude[iy,:])<=max(peak,1)*1e-10:
        raise ValueError('Single-axis phase support is not a 2D position')
    return float(offsets[ix]),float(offsets[iy])
