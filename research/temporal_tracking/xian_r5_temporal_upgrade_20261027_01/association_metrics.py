"""Association-only IDF1/AssA with explicitly perfect, provided proposals.
AssA follows Luiten et al., https://doi.org/10.1007/s11263-020-01375-2 .
No full HOTA is claimed because detection and localization are provided by GT.
"""
import json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
def identity_scores(records):
    # records: (frame, gold identity, predicted track), one record per GT proposal.
    gt=Counter(g for _,g,p in records);pr=Counter(p for _,g,p in records);both=Counter((g,p) for _,g,p in records)
    gs=sorted(gt);ps=sorted(pr);gi={g:i for i,g in enumerate(gs)};pi={p:i for i,p in enumerate(ps)}
    matrix=np.zeros((len(gs),len(ps)),int)
    for (g,p),v in both.items():matrix[gi[g],pi[p]]=v
    ii,jj=linear_sum_assignment(-matrix);idtp=int(matrix[ii,jj].sum());n=len(records)
    ass_sum=sum(count*count/(gt[g]+pr[p]-count) for (g,p),count in both.items())
    switches=0;fragments=0
    for g in gs:
        trace=[p for f,gg,p in sorted(records,key=lambda x:x[0]) if gg==g]
        switches+=sum(a!=b for a,b in zip(trace,trace[1:]));fragments+=max(len(set(trace))-1,0)
    return dict(nodes=n,IDTP=idtp,IDFP=n-idtp,IDFN=n-idtp,IDF1=idtp/max(n,1),AssA=ass_sum/max(n,1),ass_sum=ass_sum,
        ID_switches=switches,identity_fragments=fragments,detector_fragmentation='not evaluated; detections supplied',HOTA=None)
def evaluate(probs,valid,rows,threshold):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['aoi'],tuple(r['months']))].append(i)
    results=[]
    for key,indices in groups.items():
        ids=sorted({int(cid) for i in indices for j,candidates in enumerate(rows[i]['candidate_ids']) if valid[i,j] for cid in candidates if cid>=0})
        queries=[i for i in indices if valid[i,rows[i].get('reference_local_index',rows[i]['reference_frame']-rows[i]['window'])]]
        assigned={};nodes=set()
        for j in range(5):
            targets=sorted({int(cid) for i in indices if valid[i,j] for cid in rows[i]['candidate_ids'][j] if cid>=0});ti={x:k for k,x in enumerate(targets)}
            nodes.update((j,cid) for cid in targets)
            if not targets or not queries:continue
            costs=np.full((len(queries),len(targets)+len(queries)),-1e-6,np.float64);costs[:,len(targets):]=0
            for qi,i in enumerate(queries):
                if not valid[i,j]:continue
                for k,cid in enumerate(rows[i]['candidate_ids'][j]):
                    if cid<0:continue
                    p=probs[i,j,k]
                    if p.argmax()!=0 or p[0]<threshold:continue
                    costs[qi,ti[cid]]=math.log(max(float(p[0]),1e-9)/max(float(p[1]),float(p[2]),1e-9))
            aa,bb=linear_sum_assignment(-costs)
            for qi,col in zip(aa,bb):
                if col<len(targets) and costs[qi,col]>0:assigned[(j,targets[col])]='QUERY_'+str(queries[qi])
        records=[(j,str(cid),assigned.get((j,cid),'UNMATCHED_'+str(j)+'_'+str(cid))) for j,cid in sorted(nodes)]
        results.append(dict(aoi=key[0],months=list(key[1]),queries=len(queries),**identity_scores(records)))
    sums={k:sum(r[k] for r in results) for k in ['nodes','IDTP','IDFP','IDFN','ID_switches','identity_fragments','ass_sum']}
    sums.update(IDF1=2*sums['IDTP']/max(2*sums['IDTP']+sums['IDFP']+sums['IDFN'],1),AssA=sums['ass_sum']/max(sums['nodes'],1),windows=len(results),per_window=results,
        HOTA=None,SCOT=None,scope='provided-GT-query/proposal association benchmark; overlap counted per window; Hungarian one-to-one per date with null exits; not a detector or target benchmark',
        AssA_definition='mean over provided matched nodes of n_gt_pred/(n_gt+n_pred-n_gt_pred)',source='https://doi.org/10.1007/s11263-020-01375-2')
    return sums
def main():
    receipt=json.loads((ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json').read_text());rows=[]
    for r in receipt['shard_records']:
        if r['role']=='CALIBRATION':rows.extend(json.loads(Path(r['path']).with_suffix('.json').read_text())['rows'])
    result={}
    for name in ['B2_BASELINE','20261027','20261028','20261029']:
        if name=='B2_BASELINE':p=ROOT/'evidence/B2_FULLSPAN_SOURCE_PREDICTIONS.npz';threshold=0.
        else:
            p=ROOT/'runs/T2_FULL_SPAN'/('seed_'+name)/'calibration_predictions.npz';cal=json.loads((p.parent/'CALIBRATION.json').read_text());point=cal['same_edge_operating_point'];threshold=point['threshold'] if point else 1.01
        with np.load(p) as d:result[name]=evaluate(d['probs'],d['v'],rows,threshold)
    (ROOT/'evidence/ORACLE_PROPOSAL_ASSOCIATION_METRICS.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
