"""R4 candidate tracking, bounded rechecks and existing R5 formal reconstruction."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import importlib.util
import sys
import math
import numpy as np
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from .application_contract import save,lines,digest,valid_pair
from .application_geometry import EMPTY,medoid

def r4_tracks(observations,epochs,output,code_root):
    import geopandas as gpd
    import pandas as pd
    from affine import Affine
    from types import SimpleNamespace
    path=Path(code_root)/'work/run_xian_r4_tracking_candidate.py'
    spec=importlib.util.spec_from_file_location('application_frozen_r4',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    # New manifest groups replace the old hard-coded five blocks and three labels
    # in this private process only; source files are never rewritten.
    module.r3.BLOCKS=['APP'];module.r3.DATES=[e['epoch_id'] for e in epochs]
    class Adapter(module.R4TrackingRunner):
        def __init__(self):
            self.r3_by_scene={};self.scenes={};self.output=Path(output)
        def out_path(self,*parts):
            p=self.output.joinpath(*parts);p.parent.mkdir(parents=True,exist_ok=True);return p
        def _alignment_shift(self,*args):
            # No common-scene estimate is composed with V2 total displacements.
            return 0.,0.,float('inf')
    runner=Adapter()
    for e in epochs:
        rr=[r for r in observations if r['epoch_id']==e['epoch_id'] and not r.get('excluded',False)]
        data=[dict(r3_object_id=r['observation_id'],source_object_id=r['observation_id'],instance_status='APPLICATION_MODEL_CANDIDATE') for r in rr]
        runner.r3_by_scene[('APP',e['epoch_id'])]=gpd.GeoDataFrame(pd.DataFrame(data,columns=['r3_object_id','source_object_id','instance_status']),geometry=[shape(r['geometry']) for r in rr],crs=observations[0]['metric_crs'] if observations else 'EPSG:32649')
        runner.scenes[('APP',e['epoch_id'])]=SimpleNamespace(transform=Affine(1,0,0,0,-1,0))
    runner.run_tracking()
    raw=runner.track_objects.to_dict('records');edges=runner.track_edges.to_dict('records')
    for edge in edges:
        for k,v in list(edge.items()):
            if isinstance(v,float) and not math.isfinite(v):edge[k]=None
        edge['common_scene_alignment_status']='UNKNOWN_NO_ESTIMATE_NOT_MEASURED_ZERO'
    by_id={r['observation_id']:r for r in observations};groups=defaultdict(list)
    for row in raw:groups[row['track_id']].append(by_id[row['r3_object_id']])
    tracks=[];mapping_ids={}
    for rows in groups.values():
        uid='CT_'+digest(sorted(r['observation_id'] for r in rows))[:24]
        for r in rows:mapping_ids[r['observation_id']]=uid
        representative=medoid(rows)
        tracks.append(dict(track_id_candidate=uid,observation_ids=[r['observation_id'] for r in rows],physical_id=None,
            canonical_observation_id=representative['observation_id'],geometry=representative['geometry'],canonical_policy='BEST_OBSERVATION_MEDOID',
            canonical_level='CANDIDATE_ONLY',geometry_version=digest(representative['geometry']),position_sigma_m=None))
    conflict_ids=set()
    for edge in edges:
        edge['track_id_candidate']=mapping_ids.get(str(edge['source_object_id']))
        # Retain all R4 edges. Low-margin/repeated matches cannot establish a
        # stable identity just because the historical runner propagated an ID.
        if edge.get('identity_conflict') or (edge['mutual_best'] and min(edge['score_gap_source'],edge['score_gap_target'])<.05):
            conflict_ids.update([str(edge['source_object_id']),str(edge['target_object_id'])])
        a=by_id.get(str(edge['source_object_id']));b=by_id.get(str(edge['target_object_id']))
        if a and b:
            ga,gb=shape(a['geometry']),shape(b['geometry'])
            if edge['area_ratio']<.6 or (edge['iou']<.15 and edge['score_gap_source']<.1):conflict_ids.update([a['observation_id'],b['observation_id']])
    # Explicit overlap graph prevents a segmentation split/merge being an event.
    for ea,eb in zip(epochs,epochs[1:]):
        aa=[r for r in observations if r['epoch_id']==ea['epoch_id']];bb=[r for r in observations if r['epoch_id']==eb['epoch_id']]
        overlap=[]
        for a in aa:
            ga=shape(a['geometry'])
            for b in bb:
                gb=shape(b['geometry']);inter=ga.intersection(gb).area
                if inter>0 and inter/min(ga.area,gb.area)>.2:overlap.append((a['observation_id'],b['observation_id']))
        for a,b in overlap:
            if sum(x==a for x,y in overlap)>1 or sum(y==b for x,y in overlap)>1:conflict_ids.update([a,b])
    return tracks,edges,mapping_ids,conflict_ids

def candidate_presence(observations,tracks,epochs,recheck_results,conflicts):
    by_id={r['observation_id']:r for r in observations};presence=[]
    for track in tracks:
        rr=[by_id[x] for x in track['observation_ids']]
        for epoch in epochs:
            eid=epoch['epoch_id'];here=[r for r in rr if r['epoch_id']==eid]
            value='U';reason='NO_OBSERVATION_IS_NOT_ABSENCE'
            if here:
                r=here[0]
                if len(here)>1:reason='MULTIPLE_SAME_EPOCH_IDENTITY_CONFLICT'
                elif r['observation_id'] in conflicts:reason='IDENTITY_SPLIT_MERGE_OR_COMPETING_NEIGHBOR'
                elif r.get('source_edge_partial_risk') or r.get('source_conflict') or r.get('source_valid_fraction',0)<.75:reason='PARTIAL_OR_SOURCE_SUPPORT_UNKNOWN'
                elif not r.get('actual_date'):reason='DATE_UNKNOWN_OR_MIXED'
                else:value=1;reason='DATED_VISIBLE_ROOF_MODEL_CANDIDATE'
            else:
                evidence=recheck_results.get((track['track_id_candidate'],eid))
                if evidence and evidence.get('reliable_zero_candidate'):
                    value=0;reason='ACTIVE_B0_RECHECK_FULL_SUPPORT_MODEL_ABSENCE_CANDIDATE'
            presence.append(dict(track_id_candidate=track['track_id_candidate'],epoch_id=eid,actual_date=epoch.get('actual_date'),presence_candidate=value,
                presence_qualified='U',physical_id=None,reason=reason,confidence_calibrated=None))
    return presence

def pair_events(observations,tracks,presence,a,b):
    from .label_transfer import coordinate_states
    by_id={r['observation_id']:r for r in observations};pp={(r['track_id_candidate'],r['epoch_id']):r['presence_candidate'] for r in presence}
    events=[];regions={k:[] for k in ('STABLE','ADDITION','DEMOLITION','UNRESOLVED')}
    for t in tracks:
        tid=t['track_id_candidate'];rr=[by_id[x] for x in t['observation_ids']]
        before=[r for r in rr if r['epoch_id']==a['epoch_id']];after=[r for r in rr if r['epoch_id']==b['epoch_id']]
        x,y=pp.get((tid,a['epoch_id']),'U'),pp.get((tid,b['epoch_id']),'U')
        temporal_ok=valid_pair(a,b)
        if before and after and ({r['source_payload_sha256'] for r in before}&{r['source_payload_sha256'] for r in after}):temporal_ok=False
        if before and after and before[0].get('rgb_pixel_sha256') and before[0].get('rgb_pixel_sha256')==after[0].get('rgb_pixel_sha256'):temporal_ok=False
        if before and after and before[0].get('source_acquisition_id') and before[0]['source_acquisition_id']==after[0].get('source_acquisition_id'):temporal_ok=False
        if before and after and (not before[0].get('actual_date') or not after[0].get('actual_date') or before[0]['actual_date']>=after[0]['actual_date']):temporal_ok=False
        # Excluded instances never generate a false absence on another date.
        excluded=any(r.get('excluded') for r in rr)
        if temporal_ok and not excluded:
            bg={tid:shape(before[0]['geometry'])} if before else {};ag={tid:shape(after[0]['geometry'])} if after else {}
            regs=coordinate_states(bg,ag,{tid:(x,y)},dates=[a['actual_date'] or a['actual_time_window'][1],b['actual_date'] or b['actual_time_window'][0]],frames=['METRIC_REFERENCE_PLANE']*2)
            state=next((k for k in ('STABLE','ADDITION','DEMOLITION') if not regs[k].is_empty),'UNRESOLVED')
        else:state='UNRESOLVED'
        if state=='STABLE':
            # Genuine large geometry changes cannot be swallowed by one canonical.
            ga,gb=shape(before[0]['geometry']),shape(after[0]['geometry'])
            if min(ga.area,gb.area)/max(ga.area,gb.area)<.6:state='UNRESOLVED'
        if state=='STABLE':g=shape(t['geometry'])
        elif state=='ADDITION':g=shape(after[0]['geometry'])
        elif state=='DEMOLITION':g=shape(before[0]['geometry'])
        else:g=unary_union([shape(r['geometry']) for r in before+after])
        if g.is_empty:continue
        regions[state].append(g)
        events.append(dict(event_id='EV_'+digest([tid,a['epoch_id'],b['epoch_id'],state])[:24],track_id_candidate=tid,state_candidate=state,state_qualified='IGNORE',
            geometry=mapping(g),area_m2=g.area,stable_at_t0_m2=sum(shape(r['geometry']).area for r in before) if state=='STABLE' else None,
            stable_at_t1_m2=sum(shape(r['geometry']).area for r in after) if state=='STABLE' else None,physical_id=None,formal_weight=0,
            presence=[x,y],date_order_valid=temporal_ok,reason='CANDIDATE_ONLY' if state!='UNRESOLVED' else 'IDENTITY_PRESENCE_DATE_OR_SUPPORT_UNRESOLVED'))
    return events,{k:unary_union(v) for k,v in regions.items()}

def formal_reconstruction(observations,output):
    from .reconstruction.core import Observation,reconstruct
    rr=[Observation(uid=r['observation_id'],epoch_label=r['epoch_id'],actual_date=r.get('actual_date'),tile_id=None,image_path=None,
                    roof=shape(r['geometry']),source_ref=r['source_payload_sha256'],date_status='DATE_COHERENT_FULL_ROOF_COVERAGE' if r.get('actual_date') else 'DATE_UNKNOWN') for r in observations]
    state=reconstruct(rr)
    lines(Path(output)/'PRESENCE_QUALIFIED.jsonl',state.presence)
    lines(Path(output)/'PHYSICAL_IDENTITY_QUALIFIED.jsonl',[x for x in state.identity_graph if x.get('physical_id') is not None])
    lines(Path(output)/'R5_TBIA_DIAGNOSTIC_EDGES.jsonl',state.candidate_edges)
    lines(Path(output)/'R5_RESOLVER_EVENTS.jsonl',state.resolver_events)
    save(Path(output)/'FORMAL_STATE_RECEIPT.json',dict(implementation='existing reconstruct',physical_ids=sum(o.physical_id is not None for o in state.observations),
         primary_tracking='R4_FROZEN_RULE_REFERENCE',R5_TBIA='HANDCRAFTED_DIAGNOSTIC_NOT_TRAINED_NOT_SUPERIORITY_CLAIM',GT_reads=0))
