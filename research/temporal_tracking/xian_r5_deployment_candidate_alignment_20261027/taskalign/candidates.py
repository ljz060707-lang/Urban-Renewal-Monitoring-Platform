"""The same label-blind ragged candidate generator for training and deployment."""
from dataclasses import asdict
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape
from .contracts import digest

def build_candidate_graph(observations,policy,epoch_catalog=None):
    """No GT argument. No target injection. Every observation initializes a query.

    The frozen radius is a research search range, not measured registration RMS.
    Geometry is provided by the caller; identities are not part of Observation.
    """
    obs=sorted(observations,key=lambda r:(r.time_key,r.epoch,r.uid));byid={r.uid:r for r in obs}
    if len(byid)!=len(obs):raise ValueError('Duplicate observation UIDs')
    groups=defaultdict(list)
    for r in obs:groups[r.epoch].append(r)
    dates={e:groups[e][0].time_key for e in groups}
    for item in epoch_catalog or []:
        if item['epoch'] in dates and dates[item['epoch']]!=item['time_key']:raise ValueError('Epoch catalog disagrees with observations')
        dates[item['epoch']]=item['time_key'];groups.setdefault(item['epoch'],[])
    epochs=sorted(dates,key=lambda e:(dates[e],e))
    if any(any(r.time_key!=dates[e] for r in rr) for e,rr in groups.items()):raise ValueError('Mixed dates within epoch')
    if len(set(dates.values()))!=len(dates):raise ValueError('Collapse duplicate actual acquisitions before candidate construction')
    geometry={r.uid:shape(r.metric_geometry) for r in obs}
    trees={e:cKDTree(np.array([[geometry[r.uid].centroid.x,geometry[r.uid].centroid.y] for r in rr])) for e,rr in groups.items() if rr}
    queries=[]
    for q in obs:
        g=geometry[q.uid]
        for epoch in epochs:
            if epoch==q.epoch:continue
            local=trees[epoch].query_ball_point([g.centroid.x,g.centroid.y],policy.radius_projected_m) if epoch in trees else []
            ranked=sorted((groups[epoch][int(i)] for i in local),key=lambda r:(geometry[r.uid].centroid.distance(g.centroid),r.uid))
            keep=ranked[:policy.max_candidates]
            queries.append(dict(query_id='Q_'+digest([q.uid,epoch,policy.sha256])[:24],anchor_uid=q.uid,target_epoch=epoch,
                candidate_uids=[r.uid for r in keep],candidate_distances_m=[float(geometry[r.uid].centroid.distance(g.centroid)) for r in keep],
                candidate_count=len(keep),within_radius_before_cap=len(ranked),cap_truncated=len(ranked)>len(keep),
                dustbin_index=len(keep),dustbin_semantics=policy.dustbin_semantics,candidate_source='GEOMETRY_ONLY_BEFORE_LABEL_LOOKUP'))
    payload=dict(schema='TASKALIGNED_CANDIDATE_GRAPH_V1',policy=asdict(policy),policy_sha256=policy.sha256,epochs=epochs,dates=dates,
        observations=[r.to_dict() for r in obs],queries=queries,epoch_catalog=epoch_catalog,all_observations_initialized=True,GT_used_by_generator=False)
    payload['candidate_graph_sha256']=digest(payload)
    return payload

def label_after_generation(graph,identity_by_observation,truth_by_epoch):
    """Join GT only AFTER candidate graph is materialized/hash-bound.

    truth_by_epoch includes label records outside the proposal/search extent, so
    a missed correspondence stays in the denominator. NO_LABEL_RECORD is not
    an independently observable physical absence.
    """
    result=[];byepoch=defaultdict(set)
    for r in graph['observations']:byepoch[r['epoch']].add(r['uid'])
    for q in graph['queries']:
        gold=identity_by_observation.get(q['anchor_uid']);matches=[]
        if gold is None:status='UNLABELLED_QUERY';exists=None
        else:
            exists=gold in set(truth_by_epoch.get(q['target_epoch'],[]))
            matches=[i for i,uid in enumerate(q['candidate_uids']) if identity_by_observation.get(uid)==gold]
            bank_hit=any(identity_by_observation.get(uid)==gold for uid in byepoch[q['target_epoch']])
            if matches:status='MATCH_AVAILABLE'
            elif exists and bank_hit:status='CANDIDATE_RETRIEVAL_MISS'
            elif exists:status='EXISTS_OUTSIDE_PROVIDED_OBSERVATION_SCOPE'
            else:status='NO_LABEL_RECORD_NOT_PHYSICAL_ABSENCE'
        result.append(dict(query_id=q['query_id'],positive_indices=matches,dustbin_index=q['dustbin_index'],
            target_is_dustbin=bool(gold is not None and not matches),label_record_exists=exists,supervision_status=status,
            keep_in_candidate_recall_denominator=exists is True,loss_enabled=gold is not None,physical_absence_label=False,
            candidate_graph_sha256=graph['candidate_graph_sha256']))
    return result

def recall_report(labels):
    eligible=[r for r in labels if r['label_record_exists'] is True]
    hits=sum(bool(r['positive_indices']) for r in eligible)
    return dict(queries=len(labels),correspondence_label_exists=len(eligible),candidate_hits=hits,
        candidate_recall=hits/len(eligible) if eligible else None,
        retrieval_miss=sum(r['supervision_status']=='CANDIDATE_RETRIEVAL_MISS' for r in labels),
        observation_scope_miss=sum(r['supervision_status']=='EXISTS_OUTSIDE_PROVIDED_OBSERVATION_SCOPE' for r in labels),
        label_record_missing=sum(r['supervision_status']=='NO_LABEL_RECORD_NOT_PHYSICAL_ABSENCE' for r in labels),physical_absence_claim=False)
