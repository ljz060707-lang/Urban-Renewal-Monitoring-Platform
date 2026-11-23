"""Freeze cut thresholds on native INNER, then read the Jinan and Xi'an evaluations."""
import sys,json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np,cv2,torch
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from scipy import ndimage
from prepare import R,OUT,N,digest,verify,save,rows
from train import ARMS,Affinity
P15=Path('/root/xian20261111_r15')
sys.path.insert(0,str(P15/'code'))
from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate
sys.path.insert(0,'/root/xian20261111_r16_roi_roof/code')
from evaluate_xian_map_windows import evaluate,summarize
CUTS=(.40,.55,.70,.82,.92)
def cluster(nodes,dist,cut):
    mask=nodes>0;ids=np.unique(nodes[mask])
    if len(ids)<2:return None
    sub=dist[np.ix_(ids,ids)].astype('float64');sub=(sub+sub.T)*.5
    np.fill_diagonal(sub,0)
    group=fcluster(linkage(squareform(sub,checks=False),method='average'),cut,criterion='distance')
    lookup=np.zeros(65,int);lookup[ids]=group;lab=lookup[nodes]
    areas=np.bincount(lab[mask]);major=np.flatnonzero(areas>=max(24,.10*mask.sum()))
    major=major[major>0]
    if len(major)<2:return None
    valid=np.isin(lab,major);sl=ndimage.distance_transform_edt(~valid,return_distances=False,return_indices=True)
    return lab[tuple(sl)].astype(np.uint8)
def render(raw,boxes,nodes,dist,cut):
    result=raw.copy();last=int(raw.max())
    for (pid,y0,y1,x0,x1),n,d in zip(boxes,nodes,dist):
        sub=cluster(n,d,cut)
        if sub is None:continue
        view=result[y0:y1,x0:x1];m=view==pid
        near=cv2.resize(sub,(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
        ids,count=np.unique(near[m],return_counts=True)
        if len(ids)<2 or count.min()<max(40,.07*count.sum()):continue
        for j in ids[ids!=ids[np.argmax(count)]]:
            last+=1;view[m&(near==j)]=last
    assert np.array_equal(result>0,raw>0)
    return result
@torch.inference_mode()
def scores(model,r):
    with np.load(OUT/(r['id']+'.npz')) as d:
        x=d['imgs'];n=d['nodes'];b=d['boxes']
    result=[]
    for i in range(0,len(x),32):
        z=b[i:i+32];sz=np.stack((z[:,2]-z[:,1],z[:,4]-z[:,3]),1)
        tensors=[torch.as_tensor(v,device='cuda') for v in (x[i:i+32],n[i:i+32],sz)]
        result.append((-model(*tensors)).sigmoid().cpu().numpy())
    return b,n,np.concatenate(result) if result else []
def trace(frame,raw,pred):
    # Count original merged parents whose children separate the implicated buildings,
    # and formerly intact GT roofs split into >=2 substantial pieces.
    valid=frame['valid']&~frame['ignore'];gt=frame['prepared']['gt']
    byparent={};old_frag=set();new_frag=set()
    for i,g in enumerate(gt):
        for lab,collection in ((raw,old_frag),(pred,new_frag)):
            v,c=np.unique(lab[g&valid],return_counts=True);frac=c/max(1,g.sum())
            if ((v>0)&(frac>=.1)).sum()>=2:collection.add(i)
        v,c=np.unique(raw[g&valid],return_counts=True)
        for q in v[(v>0)&(c>=.2*g.sum())]:
            ids,h=np.unique(pred[g&valid&(raw==q)],return_counts=True)
            byparent.setdefault(int(q),[]).append(set(ids[(ids>0)&(h>=.2*g.sum())]))
    merged=[x for x in byparent.values() if len(x)>=2]
    detached=sum(all(s for s in m) and sum(map(len,m))==len(set.union(*m)) for m in merged)
    return dict(raw_multibuilding_parents=len(merged),separated_all_supported_buildings=int(detached),previously_unfragmented_became_fragmented=len(new_frag-old_frag),previously_fragmented_repaired=len(old_frag-new_frag))
def source(frames,registry,models,role,cuts):
    evaluator=RoofEvaluator(role)
    sel={a:{str(c):[] for c in cuts[a]} for a in ARMS}
    detail={a:Counter() for a in ARMS}
    for fr in evaluator.frames:
        row=registry[fr['id']]
        for arm,model in models.items():
            b,n,d=scores(model,row)
            for c in cuts[arm]:
                labels=render(fr['old'],b,n,d,c)
                sel[arm][str(c)].append(metrics(fr,labels))
                if len(cuts[arm])==1:
                    detail[arm].update(trace(fr,fr['old'],labels))
                    folder=R/'predictions'/arm/role;folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/(fr['id']+'.npz'),SPLIT=labels)
    summary={a:{k:aggregate(v) for k,v in result.items()} for a,result in sel.items()}
    return dict(raw=evaluator.baseline,arms=summary,trace={a:dict(x) for a,x in detail.items()})
def main():
    cv2.setNumThreads(1)
    frames=json.loads((R/'IMAGE_ONLY_FROZEN.json').read_text());registry=rows();models={}
    for rec in frames:verify(OUT/(rec['id']+'.npz'),rec['sha'])
    for a in ARMS:
        model=Affinity().cuda().eval();p=R/a/'weights.pt';verify(p,json.loads((R/a/'TRAINED.json').read_text())['weight_sha'])
        model.load_state_dict(torch.load(p,weights_only=True)['state_dict']);models[a]=model
    fit=source(frames,registry,models,'INNER',{a:CUTS for a in ARMS})
    save(R/'INNER_RESULTS.json',fit)
    def rank(m):
        baseline=fit['raw']
        return (m['fragmented_GT']<=baseline['fragmented_GT'] and m['complete_roof_recall90']>=baseline['complete_roof_recall90'],m['F1_50'],-m['fragmented_GT'])
    cuts={a:max(fit['arms'][a],key=lambda c:rank(fit['arms'][a][c])) for a in ARMS}
    primary=max(ARMS,key=lambda a:rank(fit['arms'][a][cuts[a]]))
    save(R/'FROZEN_SELECTION.json',dict(cuts=cuts,primary=primary,rule='source INNER only; predeclared controls',source_fit_native_independence='development only'))
    print('frozen',primary,cuts,flush=True)
    tune=source(frames,registry,models,'TUNE',{a:[float(cuts[a])] for a in ARMS});save(R/'JINAN.json',tune)
    print('Jinan',tune['trace'],flush=True)
    target=[r for r in registry.values() if r['role']=='TUNE'];assert len(target)==72
    manifests={}
    for arm,model in models.items():
        manifest={}
        for r in target:
            verify(r['query_cache_path'],r['query_cache_sha256'])
            with np.load(r['query_cache_path']) as f:raw=f['instances'].astype('int32')
            b,n,d=scores(model,r);lab=render(raw,b,n,d,float(cuts[arm]))
            folder=R/'predictions'/arm/'XIAN';folder.mkdir(parents=True,exist_ok=True)
            p=folder/(r['id']+'.npz');np.savez_compressed(p,SPLIT=lab)
            manifest[r['id']]=dict(path=str(p),variant='SPLIT',sha256=digest(p))
        manifests[arm]=manifest
    save(R/'TARGET_PREDICTIONS_BEFORE_GT.json',manifests)
    result={}
    for arm in ARMS:
        with ProcessPoolExecutor(6) as pool:ww=list(pool.map(evaluate,[(r,manifests[arm]) for r in target]))
        save(R/arm/'XIAN_DETAILS.json',ww)
        result[arm]={}
        for ep in sorted({w['epoch'] for w in ww}):
            subset=[w for w in ww if w['epoch']==ep]
            rr=summarize(subset,'RAW');gg=summarize(subset,'ROI_SELECTED')
            pairs=[(a,b) for w in subset for a,b in zip(w['details']['RAW'],w['details']['ROI_SELECTED'])]
            assert all(a['fid']==b['fid'] and a['covered_pixels']==b['covered_pixels'] for a,b in pairs)
            result[arm][ep]=dict(raw=rr,split=gg,gained=sum(not a['IoU50_match'] and b['IoU50_match'] for a,b in pairs),lost=sum(a['IoU50_match'] and not b['IoU50_match'] for a,b in pairs))
    save(R/'XIAN.json',result)
    print('complete',flush=True)
if __name__=='__main__':main()

