"""Decode real candidates versus a separate correspondence dustbin."""
import math
def directed_advantages(samples,logits):
    scores={}
    for i,s in enumerate(samples):
        for j,t in enumerate(s['targets']):
            db=float(logits[i,j,-1])
            for k,uid in enumerate(t['query']['candidate_uids']):
                scores[(s['anchor_uid'],uid)]=float(logits[i,j,k])-db
    return scores

def supports_from_advantages(graph,directional):
    time={r['uid']:r['time_key'] for r in graph['observations']};pairs={frozenset([a,b]) for a,b in directional};edges=[]
    for pair in sorted(pairs,key=lambda p:sorted(p)):
        a,b=sorted(pair,key=lambda u:(time[u],u));forward=directional.get((a,b));reverse=directional.get((b,a))
        available=[v for v in [forward,reverse] if v is not None]
        if not available:continue
        utility=min(available)
        if utility>0:edges.append(dict(source=a,target=b,utility=utility,evidence_kind='BIDIRECTIONAL_SET_OVER_DUSTBIN' if len(available)==2 else 'ONE_DIRECTION_SET_SUPPORT_REVERSE_UNAVAILABLE',
            missing_reverse_is_different=False,calibrated_identity_probability=None))
    return edges
