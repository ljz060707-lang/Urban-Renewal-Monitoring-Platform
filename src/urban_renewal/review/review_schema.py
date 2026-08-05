from __future__ import annotations
VERDICTS=('PASS','MINOR_ISSUE','MAJOR_ISSUE','BLOCKER')
SEVERITY_FIELDS=('alignment_severity','false_positive_severity','false_negative_severity','fragmentation_severity','merging_severity','temporal_plausibility','semantic_plausibility','confidence')
ISSUE_TAGS=('spatial_offset','road_confusion','vacant_confusion','water_confusion','shadow_confusion','building_miss','small_building_miss','building_merge','building_fragmentation','vector_jagged','vector_holes','temporal_pseudchange','seasonal_difference','brightness_difference','before_miss_false_new','after_miss_false_demolition','surface_state_error','other')
def normalize_decision(payload,reviewer='\u674e\u91d1\u6cd3'):
    verdict=str(payload.get('verdict','')).upper()
    if verdict not in VERDICTS: raise ValueError('verdict must be PASS, MINOR_ISSUE, MAJOR_ISSUE or BLOCKER')
    reviewer=str(payload.get('reviewer') or reviewer).strip()
    if not reviewer: raise ValueError('reviewer is required')
    out={'sample_id':str(payload['sample_id']),'verdict':verdict,'severity_fields':{},'issue_tags':sorted(set(str(x) for x in payload.get('issue_tags',[]))),'notes':str(payload.get('notes','')),'reviewer':reviewer,'inventory_hash':str(payload.get('inventory_hash','')),'source_hashes':payload.get('source_hashes',{})}
    for key in SEVERITY_FIELDS:
        value=int(payload.get(key,payload.get('severity_fields',{}).get(key,0)))
        if not 0<=value<=3: raise ValueError(key+' must be between 0 and 3')
        out['severity_fields'][key]=value
    if any(x not in ISSUE_TAGS for x in out['issue_tags']): raise ValueError('unknown issue tag')
    return out

