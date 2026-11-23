"""Explicit source-image mask retention, separate from formal roof admission."""
import hashlib
from shapely.geometry import shape

ROLES={'VISIBLE_EXTENT_CANDIDATE','EXTENT_RISK','PARTIAL_EXTENT','SURFACE_ONLY','UNRESOLVED'}


def retain_source_masks(candidates,choices):
    by_id={r['candidate_id']:r for r in candidates}
    if len(by_id)!=len(candidates):raise ValueError('Duplicate source candidate ID')
    groups={r['group_id'] for r in candidates}
    if {r['group_id'] for r in choices}!=groups or len(choices)!=len(groups):
        raise ValueError('Every frozen source group needs exactly one disposition')
    result=[];retained={}
    for choice in choices:
        gid=choice['group_id'];role=choice['source_role'];index=choice['mask_index']
        if role not in ROLES or not isinstance(choice.get('source_visual_basis'),str) or len(choice['source_visual_basis'])<30:
            raise ValueError('Explicit source-only semantic role and evidence basis required')
        if choice.get('reference_or_target_position_used') is not False:
            raise ValueError('Target-guided mask choice prohibited')
        if role=='UNRESOLVED':
            if index is not None:raise ValueError('Unresolved cannot retain a mask')
            result.append({**choice,'retained_candidate_id':None,'shape_diagnostic_eligible':False,
                'formal_acceptance':False,'physical_id':None});continue
        if type(index) is not int or index not in (0,1,2):raise ValueError('Explicit source mask index required')
        cid=gid+'_MASK_'+str(index);candidate=by_id[cid]
        if candidate['positive_prompt_pixels_missing'] or candidate['negative_prompt_pixels_included']:
            raise ValueError('Retained source mask violates its frozen prompts')
        if candidate['physical_id'] is not None or candidate['complete_roof_verified'] or candidate['ground_footprint']:
            raise ValueError('Unselected model candidate role unexpectedly promoted')
        geom=shape(candidate['source_geometry'])
        if not geom.is_valid or hashlib.sha256(geom.wkb).hexdigest()!=candidate['source_geometry_wkb_sha256']:
            raise ValueError('Source candidate geometry changed')
        retained[cid]=geom
        result.append({**choice,'retained_candidate_id':cid,'source_parent_component':candidate['source_parent_component'],
            'source_geometry_wkb_sha256':candidate['source_geometry_wkb_sha256'],
            'source_image_sha256':candidate['source_image_sha256'],'source_window':candidate['source_window'],
            'source_observed_date':candidate['source_observed_date'],'experiment_year':candidate['experiment_year'],
            'shape_diagnostic_eligible':role=='VISIBLE_EXTENT_CANDIDATE',
            'full_physical_roof_verified':False,'roof_purity_independently_verified':False,
            'formal_acceptance':False,'localization_bound_m':None,'physical_id':None,
            'may_generate_training_label':False,'model_score_used_to_select':False})
    return {'records':result,'geometries':retained,
        'role_counts':{role:sum(r['source_role']==role for r in result) for role in sorted(ROLES)},
        'retained_masks':len(retained),'formal_clean_roof_admissions':0}
