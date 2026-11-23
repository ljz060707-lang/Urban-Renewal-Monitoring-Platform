"""Research presence wiring. Correspondence dustbin never implies absence."""
from .contracts import digest,read,sha
from pathlib import Path

def load_research_registry(path):
    registry=read(path)
    for key,p in registry.get('profiles',{}).items():
        if p.get('scope')=='SYNTHETIC_ONLY':continue
        if p.get('status')!='RESEARCH_ADMITTED':continue
        artifact=p.get('calibration_artifact_path')
        if not artifact or not Path(artifact).is_file() or sha(artifact)!=p.get('calibration_artifact_sha256'):
            raise ValueError('Real research absence profile has no verified calibration artifact: '+key)
        c=read(artifact)
        checks=[c.get('schema')=='VISIBILITY_AND_ABSENCE_CALIBRATION_V1',c.get('scope')==p.get('scope'),c.get('model_sha256')==p.get('model_sha256'),
            c.get('truth_kind') in ['INDEPENDENT_SOURCE_FRAME_GT','OFFICIAL_ID_WITH_SEPARATE_VALIDATED_VISIBILITY'],
            c.get('candidate_consensus_used_as_truth') is False,c.get('present_support',0)>0,c.get('absent_support',0)>0,
            bool(c.get('split_contract_sha256')),bool(c.get('supervision_sha256')),c.get('visibility_semantics')=='CURRENT_RGB_VISIBILITY',
            c.get('false_absence_error_upper',1)<=p.get('maximum_false_absence_error',0)]
        if not all(checks):raise ValueError('Calibration evidence does not qualify visual absence: '+key)
    return registry

def build_presence(observations,tracks,epochs,absence_evidence=(),registry=None,allow_synthetic=False):
    registry=registry or {'profiles':{}};profiles=registry.get('profiles',{})
    byid={r.uid:r for r in observations};evidence={(r['track_id'],r['epoch']):r for r in absence_evidence};rows=[]
    for track in tracks:
        members=[byid[u] for u in track['observation_ids']]
        for epoch in epochs:
            here=[r for r in members if r.epoch==epoch['epoch']]
            if len(here)>1:raise ValueError('Same epoch repeated in identity path')
            row=dict(track_id=track['track_id'],epoch=epoch['epoch'],actual_date=epoch.get('actual_date'),time_key=epoch['time_key'],
                source_roof_observed=here[0].source_roof_observed if here else None,observation_present=bool(here),observability='UNASSESSED',identity='RESEARCH_PATH_SUPPORT',
                presence_candidate=1 if here else 'U',presence_qualified='U',formal_weight=0,
                absolute_position_qualified=False,reason='SOURCE_MODEL_OR_PROVIDED_OBSERVATION' if here else 'ABSENCE_CAPABILITY_GATE_CLOSED',
                absence_evidence_admitted=False)
            if here:
                row['observability']='ALPHA_SUPPORT_ONLY_NOT_VISIBILITY' if here[0].alpha_support is not None else 'NOT_INDEPENDENTLY_ASSESSED'
            else:
                e=evidence.get((track['track_id'],epoch['epoch']));profile=profiles.get(e.get('profile_id')) if e else None
                if e and profile:
                    synthetic=e.get('origin')=='SYNTHETIC_FUNCTION_TEST'
                    checks=dict(profile_admitted=profile.get('status')=='RESEARCH_ADMITTED',
                        correct_scope=e.get('scope')==profile.get('scope'),correct_model=e.get('model_sha256')==profile.get('model_sha256'),
                        current_date=bool(epoch.get('actual_date')) and e.get('actual_date')==epoch['actual_date'],current_RGB_hash_bound=bool(e.get('current_RGB_sha256')),
                        current_RGB_independent_observability=e.get('observability')=='CURRENT_RGB_SUFFICIENT_VALIDATED',
                        not_alpha_only=e.get('quality_basis')!='ALPHA_ONLY',search_completed=e.get('search_complete') is True,
                        coverage_sufficient=float(e.get('coverage_fraction',0))>=float(profile.get('minimum_coverage',1)),
                        qualified_match_absent=e.get('correspondence_status')=='NO_QUALIFIED_CORRESPONDENCE',
                        supervised_calibration_bound=e.get('calibration_artifact_sha256')==profile.get('calibration_artifact_sha256') and bool(profile.get('calibration_artifact_sha256')),
                        calibrated_absence=e.get('absence_decision')=='CALIBRATED_RESEARCH_ABSENCE',
                        source_support_evidence=bool(e.get('support_evidence_sha256')),
                        synthetic_scope_allowed=(not synthetic and profile.get('scope')!='SYNTHETIC_ONLY') or (synthetic and allow_synthetic and profile.get('scope')=='SYNTHETIC_ONLY'))
                    row['absence_gate_checks']=checks
                    if all(checks.values()):row.update(presence_candidate=0,observability='SUFFICIENT_FOR_RESEARCH_ABSENCE',
                        reason='ADMITTED_RESEARCH_ABSENCE_EVIDENCE',absence_evidence_admitted=True,absence_evidence=e)
                    else:row['reason']='ABSENCE_EVIDENCE_REJECTED_KEEP_UNKNOWN'
            rows.append(row)
    return rows

def interval_events(tracks,presence,epochs):
    ordered=sorted(epochs,key=lambda r:r['time_key']);events=[];p={(r['track_id'],r['epoch']):r for r in presence}
    for a,b in zip(ordered,ordered[1:]):
        if a['time_key']>=b['time_key']:raise ValueError('Adjacent actual acquisitions must be distinct and ordered')
        for track in tracks:
            pa,pb=p[(track['track_id'],a['epoch'])],p[(track['track_id'],b['epoch'])];x,y=pa['presence_candidate'],pb['presence_candidate']
            kind={(1,1):'STABLE',(0,1):'ADDITION',(1,0):'DEMOLITION'}.get((x,y),'UNRESOLVED')
            events.append(dict(event_id='TE_'+digest([track['track_id'],a['epoch'],b['epoch'],kind])[:24],track_id=track['track_id'],
                before=a.get('actual_date') or a['time_key'],after=b.get('actual_date') or b['time_key'],time_precision=[a.get('date_precision'),b.get('date_precision')],state_candidate=kind,state_qualified='IGNORE',formal_weight=0,
                event_estimable=kind!='UNRESOLVED',reason='RESEARCH_IDENTITY_AND_PRESENCE' if kind!='UNRESOLVED' else 'PRESENCE_OR_ABSENCE_CAPABILITY_UNRESOLVED',
                geometry_XOR_used=False))
    return events

def capability_summary(events,registry):
    absence_capable=any(p.get('status')=='RESEARCH_ADMITTED' and p.get('scope')!='SYNTHETIC_ONLY' for p in registry.get('profiles',{}).values())
    return dict(candidate_counts={k:sum(r['state_candidate']==k for r in events) for k in ['STABLE','ADDITION','DEMOLITION','UNRESOLVED']},
        physical_change_measurement=None,absence_provider_admitted=absence_capable,
        addition_demolition_status='RESEARCH_EVIDENCE_CONDITIONAL' if absence_capable else 'NOT_IDENTIFIABLE_CAPABILITY_GATE_UNAVAILABLE',
        zero_candidate_count_is_not_verified_zero_change=True,production_ready=False)
