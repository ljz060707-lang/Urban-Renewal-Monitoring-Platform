"""Reference readiness census, not matching, split selection or GT admission."""
from collections import Counter

EVIDENCE_GATES=("TIME_OR_CONTINUITY", "TARGET_IMAGE_FRAME_BINDING", "USE_HISTORY_CLEARANCE",
                "IDENTITY_CORRESPONDENCE", "BACKGROUND_COMPLETENESS")


def reference_preflight(records, evidence):
    """Require explicit true/false/unknown evidence; geometry alone grants nothing.

    The caller verifies evidence artifacts before setting a gate to True.
    Positive-only reference readiness is separate from background completeness.
    Even readiness does not fit a transform, create FIT/HOLDOUT or license GT.
    """
    records=list(records)
    if set(evidence)!=set(EVIDENCE_GATES):raise ValueError("Explicit evidence gates required")
    if any(value is not None and type(value) is not bool for value in evidence.values()):
        raise ValueError("Evidence must be True/False/None, not confidence or numeric proxies")
    ids=[r['source_fid'] for r in records]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError("Unique source FIDs required")
    census=Counter();output=[]
    for row in records:
        for key in ('geometry_valid','fully_inside_roi','crosses_source_cell','known_history_exposed'):
            if type(row[key]) is not bool:raise ValueError("Explicit structural booleans required")
        reasons=[]
        if not row['geometry_valid']:reasons.append('INVALID_GEOMETRY')
        if not row['fully_inside_roi']:reasons.append('PARTIAL_ROI_SUPPORT')
        if row['crosses_source_cell']:reasons.append('SOURCE_CELL_SEAM')
        if row['known_history_exposed']:reasons.append('KNOWN_HISTORY_EXPOSURE')
        structural=not reasons
        if structural:census['STRUCTURAL_CANDIDATES']+=1
        # Known exposure cannot be cleared merely by a global true flag.
        positive_reasons=reasons+[f'{k}_'+('MISSING' if evidence[k] is None else 'FAILED')
            for k in EVIDENCE_GATES[:-1] if evidence[k] is not True]
        background=evidence['BACKGROUND_COMPLETENESS'] is True
        if not positive_reasons:census['POSITIVE_REFERENCE_READY']+=1
        if not positive_reasons and background:census['COMPLETE_REFERENCE_READY']+=1
        for reason in positive_reasons:census[reason]+=1
        if not background:census['BACKGROUND_COMPLETENESS_'+('MISSING' if evidence['BACKGROUND_COMPLETENESS'] is None else 'FAILED')]+=1
        output.append({**row,'structural_candidate':structural,
            'positive_reference_ready':not positive_reasons,
            'complete_background_reference_ready':not positive_reasons and background,
            'positive_reference_blockers':positive_reasons,
            'reference_role':'UNASSIGNED_NO_FIT_HOLDOUT_SELECTION'})
    return {'records':output,'counts':{'REFERENCE_INVENTORY':len(records),**dict(census)},
        'evidence':evidence,'calibration_objects_selected':0,'validation_objects_selected':0,
        'external_independence_certified':False,'training_admission_performed':False,
        'candidate_vectors_fitted':0,'validated_error_envelope':None}
