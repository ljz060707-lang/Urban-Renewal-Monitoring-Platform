"""Bounded source-only SAM prompt correction, not GT-guided geometry editing."""
from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon,Point


def refine_source_prompt(base,refinement,surfaces):
    if base['reference_or_target_positions_used'] or refinement.get('target_or_reference_position_used') is not False:
        raise ValueError('Only source-image refinement permitted')
    if not isinstance(refinement.get('source_visual_basis'),str) or len(refinement['source_visual_basis'])<30:
        raise ValueError('Source visual defect and correction basis required')
    records={r['observation_id']:r for r in surfaces}
    h,w=base['image_shape'];extra_positive=[]
    for row in refinement.get('additional_positive',[]):
        evidence=records[row['source_surface_observation_id']]
        if evidence['source_parent_component']!=base['source_parent_component'] or evidence['target_reference_position_used']:
            raise ValueError('Positive source evidence provenance mismatch')
        xy=np.asarray(row['pixel_index_xy'],dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Explicit source pixel index required')
        if not Polygon(evidence['native_vertices_xy']).covers(Point(xy+.5)):
            raise ValueError('Added foreground must lie within its frozen source roof surface')
        extra_positive.append(xy.tolist())
    extra_negative=refinement.get('additional_negative_pixel_index_xy',[])
    if extra_negative:
        negative=np.asarray(extra_negative,dtype=float)
        if negative.ndim!=2 or negative.shape[1]!=2 or not np.isfinite(negative).all():raise ValueError('Finite source negatives required')
        positives=[Polygon(records[k]['native_vertices_xy']) for k in base['surface_observation_ids']]
        if any(any(p.covers(Point(xy+.5)) for p in positives) for xy in negative):
            raise ValueError('Negative prompt contradicts frozen roof-surface evidence')
        extra_negative=negative.tolist()
    if not extra_positive and not extra_negative:raise ValueError('Concrete source evidence correction required')
    result=deepcopy(base)
    result['group_id']=refinement['refined_group_id']
    if not result['group_id'].startswith(base['group_id']+'_SOURCE_REFINE_'):
        raise ValueError('Refinement must remain traceable to frozen base group')
    fg=[xy for xy,label in zip(base['point_coords_pixel_index_xy'],base['point_labels']) if label==1]
    bg=[xy for xy,label in zip(base['point_coords_pixel_index_xy'],base['point_labels']) if label==0]
    result['point_coords_pixel_index_xy']=fg+extra_positive+bg+extra_negative
    result['point_labels']=[1]*(len(fg)+len(extra_positive))+[0]*(len(bg)+len(extra_negative))
    xy=np.asarray(result['point_coords_pixel_index_xy'])
    if (xy[:,0]<0).any() or (xy[:,0]>=w).any() or (xy[:,1]<0).any() or (xy[:,1]>=h).any():
        raise ValueError('Refinement leaves original source image')
    result['refines_source_group_id']=base['group_id'];result['source_refinement_basis']=refinement['source_visual_basis']
    result['old_source_positions_or_masks_changed']=False;result['reference_point']=None
    result['localization_bound_m']=None;result['formal_review']=False
    return result
