"""GT-free image-grid baseline; unverified candidates never become roof GT.

Empirical computational supports are not camera acquisitions. Supervision
qualification and independent roof-plane validation remain separate gates.
"""
import numpy as np
import cv2
from shapely.geometry import box, mapping, shape
from rasterio.features import rasterize
from .supported_ground_features import internal_spatial_check
from .base_tie import image_shift_to_projected_correction
from ..source_partition import SourceRegion

DEFAULT_CONFIG = dict(radius_px=3, minimum_samples=100, fold_block_px=16,
                      minimum_gain=.01, maximum_depth=1)

def fit_local_alignment(training_data, reference_roofs, source_registry, config):
    """Audit supervision and emit an untrained A1 algorithm configuration.

    shortcut: no A2 learner is fitted, add a learner only after qualified
    calibration/validation/holdout data exist; never manufacture displacement
    labels from shape hypotheses or year-only compatibility.
    """
    training_data=list(training_data);regions={r.region_id:r for r in source_registry}
    if len({r['pair_id'] for r in training_data})!=len(training_data):raise ValueError('Duplicate training pair ID')
    splits={}
    for index,row in enumerate(training_data):
        fold=row.get('training_split')
        if fold not in {'TRAIN','CALIBRATION','VALIDATION','FINAL_HOLDOUT'}:continue
        patch=row.get('image_patch',{})
        for previous in training_data[:index]:
            previous_fold=previous.get('training_split');old=previous.get('image_patch',{})
            if previous_fold not in {'TRAIN','CALIBRATION','VALIDATION','FINAL_HOLDOUT'} or previous_fold==fold:continue
            if patch.get('raster_path') and patch.get('raster_path')==old.get('raster_path'):
                a=patch.get('native_pixel_window');b=old.get('native_pixel_window')
                if a is None or b is None:raise ValueError('Split patch support missing')
                if box(a[0],a[1],a[0]+a[2],a[1]+a[3]).intersection(box(b[0],b[1],b[0]+b[2],b[1]+b[3])).area>0:
                    raise ValueError('Training/validation overlapping image patches')
        group=row.get('acquisition_geographic_group')
        for key in [('observation',row.get('roof_observation_id')),('reference',row.get('roof_gt_id')),('source_group',group)]:
            if key[1] is None:continue
            if key in splits and splits[key]!=fold:raise ValueError('Calibration/test object or acquisition leakage')
            splits[key]=fold
    eligible=[]; rejected=[]
    for row in training_data:
        reasons=[]
        region=regions.get(row.get('empirical_alignment_region'))
        if region is None:reasons.append('SOURCE_REGION_NOT_BOUND')
        elif row.get('actual_date')!=region.actual_date:reasons.append('ACTUAL_SOURCE_DATE_MISMATCH')
        geometry=row.get('candidate_geometry')
        if region is not None and (geometry is None or not region.support.covers(shape(geometry))):
            reasons.append('SOURCE_SUPPORT_NOT_PROVEN')
        if not row.get('acquisition_geographic_group'):reasons.append('ACQUISITION_GROUP_NOT_AUDITED')
        values=[row.get('label_dx_m'),row.get('label_dy_m')]
        if any(type(v) not in (int,float) or not np.isfinite(v) for v in values):
            reasons.append('FINITE_METRIC_DISPLACEMENT_REQUIRED')
        if row.get('identity_status')!='PROVEN_SAME_ROOF':reasons.append('IDENTITY_NOT_PROVEN')
        if row.get('time_compatible') is not True:reasons.append('TIME_OR_CONTINUITY_UNCONFIRMED')
        if row.get('reference_quality') is not True:reasons.append('REFERENCE_QUALITY_UNCONFIRMED')
        if row.get('label_semantics')!='ROOF_OBSERVATION_TO_PLANIMETRIC_ROOF_REFERENCE':
            reasons.append('LABEL_SEMANTICS_UNQUALIFIED')
        if row.get('roof_gt_id') not in reference_roofs:reasons.append('REFERENCE_NOT_BOUND')
        if row.get('training_split') not in {'TRAIN','CALIBRATION'}:reasons.append('NOT_TRAINING_SPLIT')
        if reasons:rejected.append({'pair_id':row['pair_id'],'reasons':reasons})
        else:eligible.append(row['pair_id'])
    policy={**DEFAULT_CONFIG,**config}
    if set(policy)!=set(DEFAULT_CONFIG):raise ValueError('Unknown alignment configuration')
    return {'supervision_eligible_pairs':len(eligible),'eligible_pair_ids':eligible,
        'rejected_pairs':rejected,'model_trained':False,
        'status':'INSUFFICIENT_QUALIFIED_TRAINING_PAIRS' if not eligible else 'A2_TRAINER_NOT_IMPLEMENTED',
        'checkpoint':{'algorithm':'SUPPORTED_IMAGE_GRID_NCC_A1','config':policy,
            'contains_gt':False,'contains_learned_weights':False,'contains_region_vectors':False}}

def predict_local_alignment(ge_rgb, roof_candidates, source_registry, target_grid, checkpoint):
    """No paths, GT, saved ROI vectors or model predictions enter inference.

    Only one bounded split is tried when a parent fails internal checking.
    Child candidates require all four checks and different vectors; no test
    metric selects the partition. Image evidence alone does not qualify a
    roof-to-reference transfer, so qualified_regions remains empty in A1.
    """
    if (set(checkpoint)!={'algorithm','config','contains_gt','contains_learned_weights','contains_region_vectors'}
            or checkpoint['algorithm']!='SUPPORTED_IMAGE_GRID_NCC_A1'
            or any(checkpoint[k] is not False for k in ['contains_gt','contains_learned_weights','contains_region_vectors'])):
        raise ValueError('GT-free untrained A1 checkpoint required')
    if set(target_grid)!={'grid','reference_rgb','reference_valid'}:
        raise ValueError('Only explicit target grid, reference RGB and valid mask permitted')
    cfg=checkpoint['config']
    if set(cfg)!=set(DEFAULT_CONFIG) or cfg['maximum_depth']!=1:
        raise ValueError('Frozen one-level subdivision required')
    grid=target_grid['grid'];ref=np.asarray(target_grid['reference_rgb']);ge=np.asarray(ge_rgb)
    valid=np.asarray(target_grid['reference_valid'])
    if (grid.crs!='EPSG:32649' or ge.shape!=(grid.height,grid.width,3) or ref.shape!=ge.shape
            or valid.shape!=ge.shape[:2] or valid.dtype.kind!='b'
            or not np.isfinite(ge).all() or not np.isfinite(ref).all()):
        raise ValueError('Finite shared-grid metric RGB and explicit validity required')
    registry=list(source_registry)
    if not registry or any(not isinstance(r,SourceRegion) for r in registry):raise ValueError('Source registry required')
    if len({r.actual_date for r in registry})!=1:raise ValueError('Run actual acquisition dates separately')
    if len({r.region_id for r in registry})!=len(registry):raise ValueError('Duplicate empirical region')
    for i,left in enumerate(registry):
        for right in registry[i+1:]:
            if left.support.intersection(right.support).area>1e-6:raise ValueError('Source supports overlap')
    for g in roof_candidates.values():
        if g.is_empty or not g.is_valid:raise ValueError('Unreliable roof geometry must be audited, not repaired')
    # Exclude possible roofs in BOTH frames across the entire fixed search
    # extent. This is a geometric exclusion, not verified stable ground.
    radius=cfg['radius_px']
    if type(radius) is not int or radius<1:raise ValueError('Positive search radius required')
    roof_mask=rasterize([(mapping(g),1) for g in roof_candidates.values()],
        out_shape=ge.shape[:2],transform=grid.affine,all_touched=True).astype('uint8') if roof_candidates else np.zeros(ge.shape[:2],np.uint8)
    # Conservative computational exclusion; never a metric error bound or
    # alteration of roof polygons. Avoid a costly full-vector buffer union.
    excluded=cv2.dilate(roof_mask,np.ones((2*radius+3,2*radius+3),np.uint8)).astype(bool)
    source_valid=np.any(ge!=0,axis=2)&~excluded
    reference_valid=valid&~excluded
    records=[];selected=[]
    def examine(rid,support,actual_date,parent=None):
        mask=rasterize([(mapping(support),1)],out_shape=ge.shape[:2],transform=grid.affine).astype(bool)
        checked=internal_spatial_check(ref,ge,reference_valid,source_valid,mask,
            radius=radius,min_samples=cfg['minimum_samples'],block_size=cfg['fold_block_px'],minimum_gain=cfg['minimum_gain'])
        best=checked['fit']['best'];vector=None
        if best is not None:
            vector=list(image_shift_to_projected_correction(best['dx_px'],best['dy_px'],tuple(grid.affine)))
        row={'region_id':rid,'parent_region_id':parent,'actual_date':actual_date,
            'support_geometry':mapping(support),'region_kind':'EMPIRICAL_ALIGNMENT_REGION',
            'same_viewing_angle':None,'source_component_verified':False,'image_check':checked,
            'diagnostic_vector_m':vector,'selected_vector_m':None,'qualified_vector_m':None,
            'ground_reference_semantics_verified':False,'roof_reference_validation':'NOT_AVAILABLE',
            'status':'LOCAL_CORRECTION_UNRESOLVED','evidence_semantics':'IMAGE_GRID_NCC_NOT_ROOF_PLANE_OR_GROUND'}
        records.append(row);return row
    for region in registry:
        parent=examine(region.region_id,region.support,region.actual_date)
        if parent['image_check']['engineering_experiment_supported']:
            parent['selected_vector_m']=parent['diagnostic_vector_m'];selected.append(parent);continue
        x0,y0,x1,y1=region.support.bounds
        if min(x1-x0,y1-y0)<grid.affine.a*cfg['fold_block_px']*4:continue
        xm=(x0+x1)/2;ym=(y0+y1)/2
        child_supports=[region.support.intersection(b) for b in
            [box(x0,y0,xm,ym),box(xm,y0,x1,ym),box(x0,ym,xm,y1),box(xm,ym,x1,y1)]]
        crossing=[k for k,g in roof_candidates.items() if
            sum(g.intersection(b).area>1e-6 for b in child_supports)>1]
        parent['split_crossing_roof_observation_ids']=crossing
        if crossing:
            parent['subdivision_status']='DECLINED_WHOLE_ROOF_CROSSING';continue
        children=[examine(region.region_id+f'_child{n}',b,region.actual_date,region.region_id)
            for n,b in enumerate(child_supports) if b.area>1e-6]
        vectors={tuple(c['diagnostic_vector_m']) for c in children if c['diagnostic_vector_m'] is not None}
        if len(children)==4 and len(vectors)>1 and all(c['image_check']['engineering_experiment_supported'] for c in children):
            for c in children:c['selected_vector_m']=c['diagnostic_vector_m'];selected.append(c)
    return {'regions':records,'diagnostic_selected_regions':selected,'qualified_regions':[],
        'position_semantics':'RELATIVE_IMAGE_GRID_ONLY','roof_plane_correction_validated':False,
        'absolute_position_accuracy':'UNKNOWN','gt_read_count':0,'excluded_roof_pixels':int(excluded.sum()),
        'ground_feature_semantics':'NON_ROOF_GEOMETRIC_EXCLUSION_NOT_VERIFIED_STABLE_GROUND',
        'contains_accepted_displacement':False,'input_region_vectors_reused':False,
        'roof_exclusion_policy':'ALL_TOUCHED_10M_CELLS_PLUS_RADIUS_AND_DERIVATIVE_HALO',
        'config':dict(cfg),'algorithm_trained':False}
