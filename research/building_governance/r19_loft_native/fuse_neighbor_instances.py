"""Split individual raw parents only when multiple disjoint LOFT masks support the split."""
from pathlib import Path
import json,os,sys,time,traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from pycocotools import mask as cm
from loft_inference import O,sha,save
from measure_loft import score_target,P15
F=O/'retained_roof_split'
F.mkdir(exist_ok=True)
PRIMARY={f'S{threshold:.1f}_C{coverage:.2f}':(threshold,coverage) for threshold in [.5,.7] for coverage in [.35,.60]}
def split(raw,det,cut,coverage):
    labels=raw.copy();next_id=int(raw.max());slices=ndimage.find_objects(raw);group=defaultdict(list);bindings=[]
    for score,mask in det:
        if score<cut:continue
        area=int(mask.sum())
        if area<64:continue
        q,count=np.unique(raw[mask],return_counts=True);known=q>0
        if not known.any():continue
        j=count[known].argmax();qid=q[known][j]
        if count[known][j]<max(32,.5*area):continue
        sl=slices[qid-1];portion=mask[sl]&(raw[sl]==qid)
        if any((portion&prev).sum()>=.35*min(portion.sum(),prev.sum()) for prev in group[qid]):continue
        group[qid].append(portion)
    for qid,parts in group.items():
        if len(parts)<2:continue
        sl=slices[qid-1];raw_region=raw[sl]==qid;seeds=np.zeros(raw_region.shape,np.int32)
        for part in parts:
            fresh=part&(seeds==0)
            if fresh.sum()>=32:seeds[fresh]=seeds.max()+1
        if seeds.max()<2 or (seeds>0).sum()<coverage*raw_region.sum():continue
        dist=ndimage.distance_transform_edt(np.pad(raw_region,1))[1:-1,1:-1]
        result=watershed(-dist,seeds,mask=raw_region,connectivity=np.ones((3,3)))
        areas=np.bincount(result[result>0]);keep=int(areas.argmax());children=[]
        for idx in np.unique(result):
            if idx==0 or idx==keep:continue
            next_id+=1;labels[sl][result==idx]=next_id;children.append(next_id)
        bindings.append(dict(parent=int(qid),children=children))
    assert np.array_equal(labels>0,raw>0)
    assert set(np.unique(raw)).issubset(set(np.unique(labels)))
    return labels,bindings
def stage(role,frames,configs):
    result={k:[] for k in configs}
    def one(r):
        folder=O/'frozen_detections'/role/r['id'];d=folder/'DETECTIONS.json';assert sha(d)==json.loads((folder/'FRAME.json').read_text())['sha256']
        det=json.loads(d.read_text())['hits']
        assert sha(r['query_cache_path'])==r['query_cache_sha256']
        with np.load(r['query_cache_path']) as z:raw=z['instances'].astype('int32');support=z['support'].astype(bool)
        det=[(h['score'],cm.decode(h['mask']).astype(bool)&support) for h in det if h['score']>=min(c for c,_ in configs.values())]
        out={}
        for name,(cut,coverage) in configs.items():
            dest=F/role/name/r['id'];dest.mkdir(parents=True,exist_ok=True)
            p=dest/'SPLIT.npz';receipt=dest/'REC.json'
            if receipt.exists():
                rec=json.loads(receipt.read_text());assert sha(p)==rec['sha256']
            else:
                labels,links=split(raw,det,cut,coverage)
                np.savez_compressed(p,SPLIT=labels)
                rec=dict(frame_id=r['id'],path=str(p),sha256=sha(p),variant='SPLIT',split_parents=len(links),
                    new_child_ids=int(labels.max()-raw.max()),parents=links,raw_union_identical=True)
                save(receipt,rec)
            out[name]=rec
        return out
    with ThreadPoolExecutor(3) as pool:
        for i,out in enumerate(pool.map(one,frames)):
            for name,v in out.items():result[name].append(v)
            save(F/'STATUS.json',dict(role=role,frames=i+1,total=len(frames),pid=os.getpid(),time=time.time()))
    save(F/role/'PREDICTIONS_FROZEN.json',result)
    return result
def main():
    while not (O/'RECEIPT.json').exists():time.sleep(5)
    sys.path.insert(0,str(P15/'code'))
    from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
    rows=[json.loads(l) for l in Path('/root/xian20261111_r8/QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l];lookup={r['id']:r for r in rows}
    src=json.loads((P15/'data/native_roof_decoder/ALL_IMAGE_ONLY_INPUTS_FROZEN.json').read_text())['records']
    by={'INNER':[lookup[v['frame_id']] for v in src if v['role']=='INNER'],'TUNE':[r for r in rows if r['role']=='SOURCE_TUNE'],'XIAN':[r for r in rows if r['role']=='TUNE']}
    assert [len(x) for x in by.values()]==[32,12,72]
    save(F/'PROTOCOL.json',dict(variants=PRIMARY,selection='INNER first, source TUNE and XiAn only after selection; joint gate then instance F1 and native boundaries',
        method='assign each independent roof to a single parent at >=50% overlap; reject duplicate seeds; require >=2 distinct seeds and minimum collective parent coverage; watershed inside old parent; preserve every raw pixel',
        source_published_weights='possible overlap with development source scenes',no_manual_GT_in_prediction=True,production_changed=False))
    configs=PRIMARY.copy();summary={}
    for role,frames in by.items():
        frozen=stage(role,frames,configs)
        if role=='XIAN':
            key=next(iter(configs));s,objects=score_target(frames,frozen[key]);save(F/'TARGET_REFERENCE_OBJECTS.json',objects);summary[role]=s
        else:
            truth=RoofEvaluator(role);frame_gt={f['id']:f for f in truth.frames}
            def one(v):
                with np.load(v['path']) as z:return metrics(frame_gt[v['frame_id']],z['SPLIT'])
            results={}
            with ThreadPoolExecutor(4) as pool:
                for k,obs in frozen.items():
                    m=aggregate(list(pool.map(one,obs)))
                    assert m['pixel_recall']==truth.baseline['pixel_recall'] and m['pixel_IoU']==truth.baseline['pixel_IoU']
                    results[k]=dict(metrics=m,pass_native=passes(m,truth.baseline),
                        split_parents=sum(r['split_parents'] for r in obs),new_child_ids=sum(r['new_child_ids'] for r in obs))
            summary[role]=dict(baseline=truth.baseline,arms=results)
            if role=='INNER':
                selected=max(results,key=lambda k:(results[k]['pass_native'],results[k]['metrics']['F1_50'],results[k]['metrics']['boundary_F1_native2px']))
                configs={selected:configs[selected]};save(F/'SELECTION_FROZEN.json',dict(selected=selected,config=configs,summary=summary[role]))
        save(F/(role+'_SUMMARY.json'),summary[role])
    save(F/'RECEIPT.json',dict(status='COMPLETE_LEARNED_MASK_SPLIT_WITH_PIXEL_EXACT_ROOF_RETENTION',summary=summary,selection=configs,new_optimizer_updates=0,
        fullcity=False,physical_facade_accuracy='NOT_ESTABLISHED',protected_reads=0))
if __name__=='__main__':
    try:main()
    except Exception:save(F/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));raise

