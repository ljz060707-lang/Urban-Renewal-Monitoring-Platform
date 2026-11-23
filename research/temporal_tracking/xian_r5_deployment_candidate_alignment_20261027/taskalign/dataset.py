"""Ragged graph-query tensor adapter; identical feature construction at deploy."""
from datetime import date
import math
import numpy as np
from shapely.geometry import shape

def pair_features(anchor,candidate):
    a=shape(anchor['native_geometry']);b=shape(candidate['native_geometry']);s=max(math.sqrt(a.area),4.)
    gap=(date.fromisoformat(candidate['time_key'])-date.fromisoformat(anchor['time_key'])).days/365.25
    alpha=[r.get('alpha_support') for r in [anchor,candidate]]
    support=min(alpha) if all(v is not None for v in alpha) else 0.
    return np.asarray([(b.centroid.x-a.centroid.x)/s,(b.centroid.y-a.centroid.y)/s,
        math.log(max(b.area,1)/max(a.area,1)),a.intersection(b).area/max(a.union(b).area,1),
        (b.length-a.length)/max(a.length,1),gap,support,1.],np.float32).clip(-10,10)

def query_samples(graph,labels,embeddings):
    byid={r['uid']:r for r in graph['observations']};truth={r['query_id']:r for r in labels};group={}
    for row in graph['queries']:group.setdefault(row['anchor_uid'],[]).append(row)
    for uid,queries in sorted(group.items()):
        queries.sort(key=lambda q:(graph['dates'][q['target_epoch']],q['target_epoch']))
        target=[]
        for q in queries:
            if any(c not in embeddings for c in q['candidate_uids']) or uid not in embeddings:raise ValueError('Every actual observation needs an embedding')
            r=truth[q['query_id']];gap=(date.fromisoformat(graph['dates'][q['target_epoch']])-date.fromisoformat(byid[uid]['time_key'])).days/365.25
            target.append(dict(query=q,features=np.asarray([pair_features(byid[uid],byid[c]) for c in q['candidate_uids']],np.float32).reshape(-1,8),
                candidates=np.asarray([embeddings[c] for c in q['candidate_uids']],np.float32).reshape(-1,len(embeddings[uid])),
                time_features=[gap,0.],label=r))
        yield dict(anchor_uid=uid,anchor=np.asarray(embeddings[uid],np.float32),targets=target)

def collate(samples):
    import torch
    b=len(samples);t=max(len(s['targets']) for s in samples);k=max([len(z['query']['candidate_uids']) for s in samples for z in s['targets']]+[0]);d=len(samples[0]['anchor'])
    q=np.stack([s['anchor'] for s in samples]);c=np.zeros((b,t,k,d),np.float32);g=np.zeros((b,t,k,8),np.float32);mask=np.zeros((b,t,k),bool)
    positives=np.zeros((b,t,k),bool);null=np.zeros((b,t),bool);enabled=np.zeros((b,t),bool);times=np.zeros((b,t,2),np.float32)
    for i,s in enumerate(samples):
        for j,z in enumerate(s['targets']):
            n=len(z['query']['candidate_uids']);c[i,j,:n]=z['candidates'];g[i,j,:n]=z['features'];mask[i,j,:n]=True
            positives[i,j,z['label']['positive_indices']]=True;null[i,j]=z['label']['target_is_dustbin'];enabled[i,j]=z['label']['loss_enabled'];times[i,j]=z['time_features']
    return {key:torch.from_numpy(v) for key,v in dict(q=q,c=c,geometry=g,candidate_mask=mask,positive_mask=positives,dustbin_target=null,enabled=enabled,time_features=times).items()}
