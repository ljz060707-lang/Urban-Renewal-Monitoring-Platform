"""One-to-one matching of independently proposed interior point and native box to annotated roofs."""
import numpy as np
from pycocotools import mask as cm
from scipy.optimize import linear_sum_assignment

def ious(a,b):
    lo=np.maximum(a[:,None,:2],b[None,:,:2])
    hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    return inter/np.maximum(1.e-8,(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter)
def matching(scores):
    if not min(scores.shape):return np.empty(0,int),np.empty(0,int)
    cardinality=(scores>=.50).astype(float)
    qp,gp=linear_sum_assignment(-(cardinality+scores/(1+min(scores.shape))))
    valid=scores[qp,gp]>=.50
    return qp[valid],gp[valid]
def get_pixels(q):
    xy=np.clip(np.floor(q[:,:2]).astype(int),0,1023)
    return xy[:,1],xy[:,0]
def assess(record,predictions,thresholds=(.15,.30,.50)):
    with np.load(record['targets']) as z:
        raw=z['raw_boxes'];gt=z['gt_boxes'];zero=z['no_raw_pixel_overlap']
        known=np.unpackbits(z['known_native']).reshape(1024,1024).astype(bool)
        valid=np.unpackbits(z['valid_native']).reshape(1024,1024).astype(bool)
        counts=z['gt_counts']
    # Do not let GT suppress a prediction. Duplicates use only raw/predicted geometry.
    independent=predictions
    duplicate=(ious(independent[:,3:],raw[:,2:]).max(axis=1)>=.60 if len(raw) else np.zeros(len(independent),bool))
    candidates=np.concatenate((raw,independent[:,1:]),axis=0)
    scores=ious(candidates[:,2:],gt[:,2:])
    yy,xx=get_pixels(candidates)
    for j,rle in enumerate(counts):
        g=cm.decode(dict(size=[1024,1024],counts=bytes(rle))).astype(bool)
        scores[:,j]*=g[yy,xx]
    _,raw_match=matching(scores[:len(raw)])
    available=np.ones(len(gt),bool);available[raw_match]=False
    raw_known=known[yy[:len(raw)],xx[:len(raw)]]
    independent_known=known[yy[len(raw):],xx[len(raw):]]
    summary={}
    for threshold in thresholds:
        kept=(independent[:,0]>=threshold)&~duplicate
        available_ids=np.flatnonzero(available)
        _,new=matching(scores[len(raw):][kept][:,available])
        extra_ids=available_ids[new]
        _,solo=matching(scores[len(raw):][independent[:,0]>=threshold])
        summary[str(threshold)]=dict(GT=len(gt),completely_uncovered_GT=int(zero.sum()),
            raw_proposals=len(raw),raw_known_centers=int(raw_known.sum()),raw_matched=len(raw_match),
            raw_match_among_zero=int(zero[raw_match].sum()),new_proposals=int(kept.sum()),
            new_known_centers=int((kept&independent_known).sum()),new_unknown_centers=int((kept&~independent_known).sum()),
            new_outside_valid=int((kept&~valid[yy[len(raw):],xx[len(raw):]]).sum()),
            new_matches=len(extra_ids),new_matches_zero_overlap=int(zero[extra_ids].sum()),
            new_matches_partial_overlap=int((~zero[extra_ids]).sum()),independent_only_matches=len(solo),
            independent_only_proposals=int((independent[:,0]>=threshold).sum()))
    return summary
def aggregate(frames):
    summary={}
    for threshold in frames[0]:
        s={k:sum(r[threshold][k] for r in frames) for k in frames[0][threshold]}
        raw=s['raw_matched'];new=s['new_matches'];gt=s['GT']
        assert raw+new<=gt and raw<=s['raw_known_centers'] and new<=s['new_known_centers']
        s.update(raw_recall=raw/max(1,gt),combined_recall=(raw+new)/max(1,gt),
            extra_precision_on_known_centers=new/max(1,s['new_known_centers']),
            extra_yield_all_centers=new/max(1,s['new_proposals']),
            raw_precision_on_known_centers=raw/max(1,s['raw_known_centers']),
            combined_precision_on_known_centers=(raw+new)/max(1,s['raw_known_centers']+s['new_known_centers']),
            F1_all_proposal_budget=2*(raw+new)/max(1,gt+s['raw_proposals']+s['new_proposals']),
            raw_F1_all_proposal_budget=2*raw/max(1,gt+s['raw_proposals']),
            recovered_fraction_previously_no_pixel=s['new_matches_zero_overlap']/max(1,s['completely_uncovered_GT']))
        summary[threshold]=s
    return summary

