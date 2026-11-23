"""Source-only segmentation prompts, explicitly never registration ties."""
import numpy as np
from shapely.geometry import Polygon


def source_segmentation_prompts(surface_records,packets,groups):
    surfaces={r['observation_id']:r for r in surface_records};output=[];used=set()
    for group in groups:
        gid=group['group_id'];key=group['source_parent_component'];packet=packets[key]
        if not isinstance(gid,str) or not gid.startswith('SAM_ROOF_CANDIDATE_') or gid in used:
            raise ValueError('Unique candidate group ID required')
        used.add(gid);h,w=packet['local_image_shape'];positive=[]
        if not group['surface_observation_ids']:raise ValueError('Source roof surface evidence required')
        for oid in group['surface_observation_ids']:
            record=surfaces[oid]
            if (record['source_parent_component']!=key or record['target_reference_position_used']
                    or record['complete_instance'] or record['source_packet_image_sha256']!=packet['image_sha256']):
                raise ValueError('Source evidence role/packet mismatch')
            pixel=Polygon(record['native_vertices_xy'])
            point=pixel.representative_point()
            # Source annotations use continuous pixel edges. SAM point prompts
            # use pixel-index coordinates; its prompt encoder adds +0.5.
            positive.append([point.x-.5,point.y-.5])
        negative=np.asarray(group['negative_points_pixel_index'],dtype=float)
        if negative.ndim!=2 or negative.shape[1]!=2 or len(negative)<1 or not np.isfinite(negative).all():
            raise ValueError('Explicit finite source-only negative prompts required')
        xy=np.array(positive+negative.tolist())
        if (xy[:,0]<0).any() or (xy[:,0]>=w).any() or (xy[:,1]<0).any() or (xy[:,1]>=h).any():
            raise ValueError('Prompt lies outside original native packet')
        if not isinstance(group.get('negative_feature_basis'),str) or len(group['negative_feature_basis'])<30:
            raise ValueError('Source visual basis required; not a confidence label')
        output.append({'group_id':gid,'source_parent_component':key,'image_path':packet['image_path'],
            'image_sha256':packet['image_sha256'],'native_rgb_pixel_sha256':packet['native_rgb_pixel_sha256'],
            'source_window':packet['source_window'],'image_shape':[h,w],
            'point_coords_pixel_index_xy':xy.tolist(),'point_labels':[1]*len(positive)+[0]*len(negative),
            'surface_observation_ids':group['surface_observation_ids'],
            'negative_feature_basis':group['negative_feature_basis'],
            'coordinate_conversion':'SOURCE_EDGE_INTERIOR_POINT_MINUS_0_5_TO_SAM_PIXEL_INDEX',
            'role':'SOURCE_SEGMENTATION_PROMPTS_NOT_REGISTRATION_TIES','reference_point':None,
            'reference_or_target_positions_used':False,'localization_bound_m':None,
            'full_instance_verified':False,'physical_id':None,'formal_review':False})
    return output
