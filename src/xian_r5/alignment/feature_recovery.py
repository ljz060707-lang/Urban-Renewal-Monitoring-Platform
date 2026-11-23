"""Mask attribution and explicit ground-evidence gates; no GT or saved shifts."""
import cv2
import numpy as np


def exclusion_stages(center_roof, touched_roof, *, radius_px=3):
    """Return nested masks; derivative halo is one additional target pixel.

    These are computational exclusions, not localization uncertainty bounds.
    Input masks and source roof geometry are never mutated.
    """
    center = np.asarray(center_roof)
    touched = np.asarray(touched_roof)
    if (center.ndim != 2 or center.shape != touched.shape or
            center.dtype.kind != 'b' or touched.dtype.kind != 'b' or
            np.any(center & ~touched)):
        raise ValueError('Nested boolean roof masks required')
    if type(radius_px) is not int or radius_px < 1:
        raise ValueError('Positive integer search radius required')
    search = cv2.dilate(touched.astype('uint8'),
                       np.ones((2 * radius_px + 1,) * 2, 'uint8')).astype(bool)
    derivative = cv2.dilate(search.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
    return dict(center=center.copy(), all_touched=touched.copy(),
                search_halo=search, derivative_halo=derivative)
