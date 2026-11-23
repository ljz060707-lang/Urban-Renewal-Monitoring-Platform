"""Evidence gates for observation-to-existing-object provenance only."""

REQUIRED_SOURCE_GATES=(
    'original_id_and_geometry_verified', 'canonical_record_geometry_verified',
    'canonical_export_complete', 'declared_rasterization_source_matches',
    'same_source_grid', 'date_and_source_hash_verified',
    'tracking_record_consistent', 'native_pixels_exact',
    'all_relevant_source_objects_checked', 'unique_native_object',
)


def adjudicate(evidence):
    """Never infer a physical identity from a successful single-date link."""
    missing=[name for name in REQUIRED_SOURCE_GATES if evidence.get(name) is not True]
    conflicts=list(evidence.get('source_conflicts',[]))
    if conflicts:
        status='SOURCE_CONFLICT'
    elif evidence.get('multiple_contributors') or evidence.get('native_exact_object_count',0)>1:
        status='COMPETING_LINKS'
    elif not missing and type(evidence.get('native_exact_object_count')) is int and evidence['native_exact_object_count']==1:
        status='TRACEABLE_SOURCE_LINK'
    elif evidence.get('geometric_plausible'):
        status='PLAUSIBLE_GEOMETRIC_LINK'
    else:
        status='INSUFFICIENT_EVIDENCE'
    return {'adjudication':status,'link_level':'TRACEABLE_SINGLE_DATE_LINK' if status=='TRACEABLE_SOURCE_LINK' else 'CANDIDATE_ONLY',
            'failed_gates':missing,'source_conflicts':conflicts,'physical_id':None,
            'cross_date_physical_identity_confirmed':False}


def validate_links(rows, observations, objects):
    for row in rows:
        if row['observation_id'] not in observations:raise ValueError('Unknown original observation')
        if row.get('building_object_id') and row['building_object_id'] not in objects:
            raise ValueError('Unknown existing object')
        if row.get('physical_id') is not None:raise ValueError('Single-date links cannot carry physical IDs')
        if row.get('link_level')=='TRACEABLE_SINGLE_DATE_LINK':
            if adjudicate(row['evidence'])['link_level']!='TRACEABLE_SINGLE_DATE_LINK':
                raise ValueError('Missing source proof')
        if row.get('GT_USED_FOR_GEOMETRY'):raise ValueError('GT replacement forbidden')
    return True
