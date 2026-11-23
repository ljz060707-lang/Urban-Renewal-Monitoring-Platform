"""Full provided-observation association scoring with no query-only privilege."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

def association_metrics(observations,tracks,gold_by_uid):
    predicted={u:t['track_id'] for t in tracks for u in t['observation_ids']}
    if len(predicted)!=sum(len(t['observation_ids']) for t in tracks):raise ValueError('Observation assigned twice')
    all_uids={o.uid for o in observations}
    if set(predicted)!=all_uids:raise ValueError('Every observation, including unmatched, must retain one track')
    truth_uids={u for u in all_uids if gold_by_uid.get(u) is not None}
    gt=Counter(gold_by_uid[u] for u in truth_uids);pr=Counter(predicted[u] for u in truth_uids);both=Counter((gold_by_uid[u],predicted[u]) for u in truth_uids)
    gg=sorted(gt);pp=sorted(pr);gi={g:i for i,g in enumerate(gg)};pi={p:i for i,p in enumerate(pp)}
    matrix=np.zeros((len(gg),len(pp)),int)
    for (g,p),n in both.items():matrix[gi[g],pi[p]]=n
    i,j=linear_sum_assignment(-matrix);tp=int(matrix[i,j].sum());n=len(truth_uids)
    ass=sum(v*v/(gt[g]+pr[p]-v) for (g,p),v in both.items())
    switches=0;fragments=0
    for g in gg:
        trace=[predicted[o.uid] for o in sorted(observations,key=lambda o:(o.time_key,o.uid)) if gold_by_uid.get(o.uid)==g]
        switches+=sum(a!=b for a,b in zip(trace,trace[1:]));fragments+=max(len(set(trace))-1,0)
    return dict(observation_nodes=len(all_uids),scored_nodes=n,unlabelled_nodes=len(all_uids)-n,IDTP=tp,IDFP=n-tp,IDFN=n-tp,
        IDF1=tp/n if n else None,AssA=ass/n if n else None,ID_switches=switches,identity_fragments=fragments,
        HOTA=None,SCOT=None,detector_recall=None,all_nodes_initialized=True,
        scope='FULL_PROVIDED_OBSERVATION_GRAPH_WITHIN_FROZEN_SPATIAL_WINDOW; oracle geometry, not end-to-end detector accuracy')

def labelled_pair_statistics(graph,labels):
    # Positive existence in the complete source label table is independent of
    # candidate retrieval. A retrieval miss is never removed from the denominator.
    positives=sum(r['label_record_exists'] is True for r in labels)
    in_set=sum(bool(r['positive_indices']) for r in labels)
    return dict(candidate_recall_denominator=positives,candidate_hits=in_set,candidate_recall=in_set/positives if positives else None,
        initialized_observations=len(graph['observations']),queries=len(graph['queries']),
        empty_candidate_sets=sum(r['candidate_count']==0 for r in graph['queries']),
        all_query_coverage=1. if graph['observations'] else None)
