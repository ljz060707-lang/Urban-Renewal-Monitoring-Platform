"""Instance alignment admission seam; missing learned weights never imply zero.

Only the no-checkpoint path is implemented. Existing image-context diagnostic
vectors are deliberately not accepted as TARGET_B predictions or checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

TARGET_A = 'IMAGE_TO_MAP_REGISTRATION'
TARGET_B = 'OBSERVED_ROOF_TO_PLANIMETRIC_ROOF'
TARGET_C = 'ROOF_TO_GROUND_FOOTPRINT'


def predict_instance_alignment(ge_rgb, roof_instances, source_registry, target_grid, checkpoint):
    """Return per-observation unresolved records without reading any files/GT.

    This is a fail-closed integration path, NOT an untrained numerical model.
    Non-null checkpoints fail explicitly until a qualified predictor exists.
    No application permission, region vector or cached offset is accepted here.
    Geometry coordinates remain EPSG:32649; native image provenance is separate.
    """
    if checkpoint is not None:
        raise ValueError('QUALIFIED_LEARNED_INSTANCE_CHECKPOINT_NOT_AVAILABLE')
    rgb = np.asarray(ge_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Finite native RGB array required')
    if target_grid.crs != 'EPSG:32649':
        raise ValueError('Explicit metric target grid required')
    instances = list(roof_instances)
    ids = [r['observation_id'] for r in instances]
    if len(ids) != len(set(ids)) or any(not isinstance(k, str) or not k for k in ids):
        raise ValueError('Original unique observation IDs required')
    allowed = {'source_component_id','actual_date','native_image_frame','source_support_complete',
               'source_geometry_hash','source_payload_hashes','source_tile_ids','photographic_seam_status'}
    if set(source_registry) != set(ids):
        raise ValueError('Every observation needs its own source registry record')
    output = []
    for r in instances:
        key = r['observation_id']; source = source_registry[key]
        if set(source) - allowed:
            raise ValueError('GT, identity matches, cached vectors and unrecognized source fields forbidden')
        g = r['roof_polygon_raw']
        if g.is_empty or not g.is_valid or g.geom_type not in {'Polygon','MultiPolygon'}:
            raise ValueError('Keep malformed instances in source audit; never silently repair')
        gh = hashlib.sha256(g.wkb).hexdigest()
        if gh != source['source_geometry_hash']:
            raise ValueError('Source geometry hash mismatch')
        evidence = hashlib.sha256(json.dumps(dict(source=source,geometry_sha256=gh,
            target_semantics=TARGET_B,checkpoint=None),sort_keys=True).encode()).hexdigest()
        output.append(dict(observation_id=key,source_component_id=source.get('source_component_id'),
            actual_date=source.get('actual_date'),native_image_frame=source.get('native_image_frame'),
            roof_polygon_raw=g,roof_instance_quality=r['roof_instance_quality'],
            region_dx_m=None,region_dy_m=None,instance_residual_dx_m=None,instance_residual_dy_m=None,
            total_dx_m=None,total_dy_m=None,target_semantics=TARGET_B,prediction_uncertainty=None,
            validation_status='NOT_VALIDATED',application_status='UNRESOLVED',
            prediction_status='NOT_RUN_NO_QUALIFIED_CHECKPOINT',evidence_hash=evidence,
            source_geometry_hash=gh,physical_id=None,registered_geometry=None))
    return output
