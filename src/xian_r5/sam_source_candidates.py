"""Exact source-frame polygonization of unselected official-SAM masks."""
import hashlib
import numpy as np
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import shape,MultiPolygon,Polygon,mapping
from shapely.ops import transform,unary_union


def sam_source_candidate(mask,packet,prompt,mask_index,model_score,checkpoint_sha256):
    a=np.asarray(mask)
    if np.ma.isMaskedArray(mask) or a.ndim!=2 or a.shape!=tuple(packet['local_image_shape']) or a.dtype.kind not in 'bu' or not np.isin(a,[0,1]).all():
        raise ValueError('Original native binary mask required')
    if packet['native_resampling']!='NONE' or packet['source_raster_crs']!='EPSG:4326':raise ValueError('Native source frame required')
    if (prompt['source_parent_component']!=packet['source_parent_component'] or prompt['image_sha256']!=packet['image_sha256']
            or prompt['reference_or_target_positions_used']):raise ValueError('Frozen source prompt/packet mismatch')
    if type(mask_index) is not int or mask_index not in (0,1,2) or not np.isfinite(model_score):raise ValueError('Explicit model output index/score required')
    parts=[shape(g) for g,v in shapes(a.astype(np.uint8),mask=a.astype(bool),connectivity=4,transform=Affine.identity()) if v==1]
    native=unary_union(parts)
    if native.is_empty:return {'candidate_id':prompt['group_id']+'_MASK_'+str(mask_index),'status':'EMPTY_MODEL_MASK_NOT_LABEL'},None
    if not native.is_valid:raise ValueError('Invalid native candidate; no automatic repair')
    # Retain pixel mask edges and holes. Densification changes neither native
    # geometry nor area; it only bounds projection curvature serialization.
    dense=shapely.segmentize(native,32.)
    affine=Affine(*packet['source_raster_affine']);col,row,w,h=packet['source_window']
    if [h,w]!=packet['local_image_shape']:raise ValueError('Native packet dimensions differ from raster window')
    t=Transformer.from_crs(4326,32649,always_xy=True)
    def metric(x,y,z=None):
        x=np.asarray(x)+col;y=np.asarray(y)+row
        return t.transform(affine.a*x+affine.b*y+affine.c,affine.d*x+affine.e*y+affine.f)
    g=transform(metric,dense)
    if not g.is_valid or not np.isfinite(g.area):raise ValueError('Metric candidate invalid; no snapping/repair')
    if isinstance(g,Polygon):g=MultiPolygon([g])
    pixels=np.asarray(prompt['point_coords_pixel_index_xy'])
    xy=np.floor(pixels+.5).astype(int)
    if (xy[:,0]<0).any() or (xy[:,0]>=w).any() or (xy[:,1]<0).any() or (xy[:,1]>=h).any():raise ValueError('Point outside native image')
    sampled=a[xy[:,1],xy[:,0]].astype(bool);labels=np.asarray(prompt['point_labels'])
    positive_missing=int((~sampled&(labels==1)).sum());negative_included=int((sampled&(labels==0)).sum())
    record={'candidate_id':prompt['group_id']+'_MASK_'+str(mask_index),'group_id':prompt['group_id'],
        'mask_index':mask_index,'source_parent_component':packet['source_parent_component'],
        'native_mask_sha256':hashlib.sha256(a.astype(np.uint8).tobytes()).hexdigest(),
        'native_mask_positive_pixels':int(a.sum()),'native_connected_polygon_parts':len(parts),
        'native_union_area_px2':native.area,'source_geometry':mapping(g),
        'source_geometry_wkb_sha256':hashlib.sha256(g.wkb).hexdigest(),'area_m2':g.area,
        'source_image_sha256':packet['image_sha256'],'source_window':packet['source_window'],
        'source_observed_date':packet['source_observed_date'],'experiment_year':packet['experiment_year'],
        'model':'OFFICIAL_SAM_VIT_B_NOT_RBF_SAM','checkpoint_sha256':checkpoint_sha256,
        'predicted_mask_quality_diagnostic_only':float(model_score),
        'positive_prompt_pixels_missing':positive_missing,'negative_prompt_pixels_included':negative_included,
        'prompt_consistency_is_roof_semantic_verification':False,'selected':False,
        'role':'UNSELECTED_SOURCE_MASK_CANDIDATE_NOT_GT','physical_id':None,'complete_roof_verified':False,
        'ground_footprint':False,'candidate_displacement_m':None,'localization_bound_m':None,
        'absolute_position_accuracy':'UNKNOWN'}
    if native.area!=int(a.sum()):raise RuntimeError('Native mask pixel area conservation failed')
    return record,g
