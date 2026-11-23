"""Model-digitized source roof surfaces with exact pixel/coordinate provenance.

Observation IDs name candidates, not physical buildings. A visible roof plane
cannot stand in for a complete roof instance, occupancy background or ground
footprint. Reference/target positions are deliberately absent from this seam.
"""
import hashlib,math,re
import numpy as np
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Polygon,mapping
from shapely.ops import transform


def source_roof_observation(packet, annotation, parent_geometry, source_scope):
    required={'source_parent_component','image_path','image_sha256','native_rgb_pixel_sha256',
        'source_window','source_raster_affine','source_raster_crs','local_image_shape',
        'source_observed_date','experiment_year','native_resampling'}
    if not required<=packet.keys():raise ValueError('Complete frozen native source packet required')
    if packet['native_resampling']!='NONE' or packet['source_raster_crs']!='EPSG:4326':
        raise ValueError('Original native source image and declared frame required')
    if any(re.fullmatch(r'[0-9a-f]{64}',packet[k]) is None for k in ('image_sha256','native_rgb_pixel_sha256')):
        raise ValueError('Native source image/hash binding required')
    if annotation.get('coordinate_convention')!='LOCAL_PIXEL_EDGE_XY' or annotation.get('method')!='VISION_MODEL_RAW_SOURCE_DIGITIZATION':
        raise ValueError('Explicit model-only raw-source digitization required')
    if annotation.get('complete_instance') is not False:
        raise ValueError('This adapter accepts visible roof surfaces, not complete-instance claims')
    if annotation.get('uses_target_or_reference_position') is not False:
        raise ValueError('Target/reference-position-guided observations prohibited')
    oid=annotation.get('observation_id')
    if not isinstance(oid,str) or not oid.startswith('MODEL_ROOF_SURFACE_'):
        raise ValueError('Candidate observation ID required, never a physical ID')
    xy=np.asarray(annotation.get('vertices'),dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():
        raise ValueError('Finite native pixel vertices required')
    height,width=packet['local_image_shape'];col,row,nw,nh=packet['source_window']
    if [nh,nw]!=[height,width] or (xy[:,0]<0).any() or (xy[:,0]>width).any() or (xy[:,1]<0).any() or (xy[:,1]>height).any():
        raise ValueError('Observation must lie in unresized source packet')
    pixel=Polygon(xy)
    if not pixel.is_valid or pixel.is_empty or pixel.area<=0:raise ValueError('Valid positive-area source polygon required; no repair')
    affine=Affine(*packet['source_raster_affine']);to_metric=Transformer.from_crs(4326,32649,always_xy=True)
    def metric(x,y,z=None):
        x=np.asarray(x)+col;y=np.asarray(y)+row
        lon=affine.a*x+affine.b*y+affine.c;lat=affine.d*x+affine.e*y+affine.f
        return to_metric.transform(lon,lat)
    geometry=transform(metric,pixel)
    if not geometry.is_valid or not math.isfinite(geometry.area):raise ValueError('Metric coordinate conversion invalid')
    if not source_scope.covers(geometry):raise ValueError('Model observation leaves frozen source support')
    overlap=geometry.intersection(parent_geometry).area/geometry.area
    # Do not clip or snap a source observation to a prediction. Partial parent
    # disagreement remains diagnostic, rather than secretly changing positions.
    if overlap<.90:raise ValueError('Source parent correspondence too weak for this observation bundle')
    output={'observation_id':oid,'source_parent_component':packet['source_parent_component'],
        'source_packet_image_path':packet['image_path'],'source_packet_image_sha256':packet['image_sha256'],
        'native_rgb_pixel_sha256':packet['native_rgb_pixel_sha256'],'source_window':packet['source_window'],
        'native_vertices_xy':xy.tolist(),'coordinate_convention':'LOCAL_PIXEL_EDGE_XY',
        'source_observed_date':packet['source_observed_date'],'experiment_year':packet['experiment_year'],
        'source_geometry':mapping(geometry),'source_geometry_wkb_sha256':hashlib.sha256(geometry.wkb).hexdigest(),
        'area_m2':geometry.area,'native_area_px2':pixel.area,'source_parent_overlap_fraction':overlap,
        'method':annotation['method'],'feature_basis':annotation.get('feature_basis'),
        'role':'VISIBLE_ROOF_SURFACE_CANDIDATE_NOT_COMPLETE_INSTANCE',
        'roof_purity_independently_verified':False,'complete_instance':False,'localization_bound_m':None,
        'candidate_displacement_m':None,'target_reference_position_used':False,
        'may_generate_fraction_labels':False,'may_generate_background':False,
        'formal_review':False,'physical_building_id':None,'ground_footprint':False,
        'absolute_position_accuracy':'UNKNOWN'}
    if not isinstance(output['feature_basis'],str) or len(output['feature_basis'])<30:
        raise ValueError('Image feature basis required, not a confidence label')
    return output,geometry
