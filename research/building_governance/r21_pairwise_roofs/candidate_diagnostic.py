"""Diagnostic maximum map matches from existing image-derived split proposals; no ref-created cuts."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
import sys,json
from prepare import R,rows,digest,verify,save
from guard import model_for_split,raw_input
from evaluate_pairs import scores,render
sys.path.insert(0,'/root/xian20261111_r20_route')
from diagnose import ref_masks
O=R/'candidate_diagnostic';O.mkdir(exist_ok=True)
def solve(edges):
    if not edges:return 0
    # Each reference and each candidate may match once. A parent and any of its
    # children cannot be used simultaneously. Integrality enforced per edge.
    buckets=defaultdict(list);modes=defaultdict(lambda:defaultdict(list))
    for i,(fid,group,cand,pid) in enumerate(edges):
        buckets[('ref',fid)].append(i);buckets[('candidate',group,cand)].append(i)
        modes[pid][(group,cand)].append(i)
    cons=list(buckets.values())
    for p,groups in modes.items():
        old=groups.get(('raw',p),[])
        if old:
            for (kind,q),values in groups.items():
                if kind=='split':cons.append(old+values)
    a=[];b=[]
    for k,entries in enumerate(cons):
        a.extend([k]*len(entries));b.extend(entries)
    mat=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(cons),len(edges))).tocsc()
    res=milp(-np.ones(len(edges)),integrality=np.ones(len(edges)),bounds=Bounds(0,1),constraints=LinearConstraint(mat,-np.inf,1))
    assert res.success
    return int(round(-res.fun))
def analyze(args):
    r,rec=args;raw=raw_input(r);verify(rec['path'],rec['sha'])
    with np.load(rec['path']) as z:labels=z['labels']
    with np.load(r['query_cache_path']) as z:support=z['support']>0
    masks=ref_masks(r,support);c=labels!=raw
    parents=set(np.unique(raw[c]));child={int(k):int(p) for p,k in np.unique(np.stack((raw[c],labels[c]),1),axis=0)}
    arrs={'raw':raw,'split':labels};areas={m:np.bincount(v.ravel()) for m,v in arrs.items()}
    edges=[];focused=set()
    for fid,g in masks.items():
        ga=g.sum()
        for m,lab in arrs.items():
            q,cc=np.unique(lab[g],return_counts=True);ix=q>0;q=q[ix];cc=cc[ix]
            if m=='raw' and cc.sum()>=.5*ga and not (cc/np.maximum(1,ga+areas[m][q]-cc)>=.5).any():focused.add(fid)
            for k,inter in zip(q,cc):
                if inter/(ga+areas[m][k]-inter)>=.5:edges.append((fid,m,int(k),int(child.get(k,k))))
    both=[e for e in edges if e[1]=='raw' or e[3] in parents]
    return dict(id=r['id'],epoch=r['epoch'],supports=len(masks),focus_n=len(focused),raw=solve([e for e in edges if e[1]=='raw']),all_split=solve([e for e in edges if e[1]=='split']),best=solve(both),
      focus_recoverable=solve([e for e in both if e[0] in focused]),proposed_parent_count=len(parents))
def main():
    model=model_for_split();target=[r for r in rows().values() if r['role']=='TUNE'];manifest={}
    for r in target:
        raw=raw_input(r);b,n,d=scores(model,r);lab=render(raw,b,n,d,.55)
        p=O/(r['id']+'.npz');np.savez_compressed(p,labels=lab);manifest[r['id']]={'sha':digest(p),'path':str(p)}
    save(O/'PROPOSALS_BEFORE_GT.json',manifest)
    with ProcessPoolExecutor(6) as pool:ws=list(pool.map(analyze,[(r,manifest[r['id']]) for r in target]))
    sums={ep:{k:sum(w[k] for w in ws if w['epoch']==ep) for k in ('supports','focus_n','raw','all_split','best','focus_recoverable','proposed_parent_count')} for ep in sorted({r['epoch'] for r in ws})}
    assert [v['raw'] for v in sums.values()]==[166,141,63]
    assert [v['focus_n'] for v in sums.values()]==[644,628,421]
    save(O/'RESULTS.json',dict(summary=sums,windows=ws,scope='exact selective upper bound for this single image-proposed .55 partition per parent only, oracle uses map labels strictly for retrospective diagnostic'))
if __name__=='__main__':main()

