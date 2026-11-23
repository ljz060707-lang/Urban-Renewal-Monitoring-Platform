"""User-confirmed manual roof reference, without inventing metric accuracy.

Annotation union is measurable geometry. Unannotated pixels are not proven
background; accepting the reference role is not a registration acceptance.
"""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from .fraction_shadow import _areas_and_counts


def manual_reference_binding():
    return {
        'reference_role': '2019_MANUAL_UNBIASED_ROOF_POSITION_REFERENCE',
        'assertion_source': 'DIRECT_USER_CONFIRMATION',
        'assertion_text': '他就是无偏的底图人工标注的',
        'nominal_year': 2019,
        'actual_acquisition_date': None,
        'geometry_semantics': 'ROOF_OUTLINE_NOT_GROUND_FOOTPRINT',
        'user_confirmed_unbiased_basemap': True,
        'metric_localization_bound_m': None,
        'absolute_position_accuracy': 'UNKNOWN',
        'Sentinel_grid_binding': 'REPROJECTED_COORDINATES_NOT_INDEPENDENT_SENSOR_VALIDATION',
        'use': ['ROOF_POSITION_REFERENCE', 'FRACTION_COMPARISON', 'CALIBRATION_REFERENCE'],
        'development_exposed': True,
        'external_unseen_test_set': False,
        'complete_negative_annotation_support': 'UNKNOWN',
        'automatic_alignment_acceptance': False,
        'automatic_training_admission': False,
        'source_geometry_may_snap_to_reference': False,
    }


def manual_roof_fractions(objects, grid):
    """Union intersection / FULL pixel area; no annotation-size filtering."""
    if grid.crs != 'EPSG:32649':
        raise ValueError('Metric reference grid required')
    if any(not isinstance(k, str) or not isinstance(g, (Polygon, MultiPolygon))
           or g.is_empty or not g.is_valid for k, g in objects.items()):
        raise ValueError('Invalid reference; no silent geometry repair')
    footprint = Polygon([grid.pixel_to_map(c, r) for c, r in
                         ((0, 0), (grid.width, 0),
                          (grid.width, grid.height), (0, grid.height))])
    clipped = {k: g.intersection(footprint) for k, g in objects.items()
               if g.intersection(footprint).area > 0}
    cells = np.array([grid.cell(c, r) for r in range(grid.height)
                      for c in range(grid.width)], dtype=object)
    areas, counts, conflict = _areas_and_counts(clipped.values(), cells)
    partial = [clipped[k] for k in clipped if not footprint.covers(objects[k])]
    partial_areas, _, _ = _areas_and_counts(partial, cells)
    denominator = abs(grid.affine.determinant)
    fraction = areas.reshape(grid.height, grid.width) / denominator
    if not np.isfinite(fraction).all() or np.any(fraction < -1e-9) or np.any(fraction > 1+1e-9):
        raise ValueError('Reference union fraction outside [0,1]')
    expected = unary_union(list(clipped.values())).area
    error = abs(float(fraction.sum()) * denominator - expected)
    if error > max(1e-6, expected * 1e-10):
        raise ValueError('Reference area conservation failed')
    return {
        'fraction': np.clip(fraction, 0, 1),
        'object_count': counts.reshape(grid.height, grid.width),
        'overlap_conflict': conflict.reshape(grid.height, grid.width),
        'roi_truncated_object': (partial_areas > 1e-6).reshape(grid.height, grid.width),
        'known_background': np.zeros((grid.height, grid.width), dtype=np.uint8),
        'training_admission': np.zeros((grid.height, grid.width), dtype=np.uint8),
        'union_area_m2': expected,
        'area_conservation_error_m2': error,
        'roi_reference_object_count': len(clipped),
    }
