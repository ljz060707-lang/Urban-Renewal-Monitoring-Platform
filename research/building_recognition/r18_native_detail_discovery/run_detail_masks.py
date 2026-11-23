"""Freeze both discovery arms on identical source and Xi'an windows, then compare actual new native masks."""
import fcntl,json,math,sys,time,traceback,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import rasterio
from scipy.optimize import linear_sum_assignment
import torch
from detail_inputs import O,P,R,save,sha
from train_detail import LOCK,DetailNet,ProposalNet,budget_decode,take_budget
from infer_development_roofs import add_masks,raw_boxes,PRE,ASSETS
from evaluate_proposals import ious,matching
RUN=O/'matched_budget_masks';RUN.mkdir(exist_ok=True)
def missing_diagnosis(f,labels,proposals):
    valid=f['valid']&~f['ignore']&f['support']
    missing=[g&valid for g in f['GT'] if (g&valid).any() and not (f['old'][g&valid]>0).any()]
    anchor_count=box_count=0
    ids,count=np.unique(labels[valid],return_counts=True)
    new_ids=ids[ids>f['old'].max()];sizes={k:n for k,n in zip(ids,count)}
    positions={v:i for i,v in enumerate(new_ids)}
    table=np.zeros((len(missing),len(new_ids)))
    if len(proposals):xx,yy=np.clip(np.floor(proposals[:,1:3]).astype(int),0,1023).T
    else:xx=yy=np.zeros(0,int)
    for i,g in enumerate(missing):
        hit=g[yy,xx]
        anchor_count+=int(hit.any())
        ys,xs=np.where(g)
        box=np.array([[xs.min(),ys.min(),xs.max()+1,ys.max()+1]])
        box_count+=int((ious(proposals[:,3:],box)[hit]>=.50).any())
        qids,n=np.unique(labels[g],return_counts=True)
        for k,inter in zip(qids,n):
            if k in positions:table[i,positions[k]]=inter/(g.sum()+sizes[k]-inter)
    rr,_=matching(table)
    added=(labels>0)&(f['old']==0)
    return dict(missing_GT=len(missing),any_new_anchor_inside=anchor_count,anchor_and_box_IoU50=box_count,
                one_to_one_new_mask_IoU50=len(rr),new_native_known_roof_pixels=int((added&valid&(f['label']==1)).sum()),
                new_known_nonroof_pixels=int((added&valid&(f['label']==0)).sum()),new_unknown_pixels=int((added&~valid).sum()))
def score_target(frames,records):
    sys.path.insert(0,'/root/xian20261111_r16_roi_roof/code')
    from evaluate_xian_map_windows import evaluate,summarize
    by={r['frame_id']:r for r in records}
    with ProcessPoolExecutor(6) as pool:items=list(pool.map(evaluate,[(r,by) for r in frames]))
    out={}
    for epoch in sorted({r['epoch'] for r in items}):
        subset=[r for r in items if r['epoch']==epoch]
        pairs=[(a,b) for r in subset for a,b in zip(r['details']['RAW'],r['details']['ROI_SELECTED'])]
        assert all(a['fid']==b['fid'] for a,b in pairs)
        out[epoch]=dict(raw=summarize(subset,'RAW'),combined=summarize(subset,'ROI_SELECTED'),
            changed_from_zero_to_any=sum(a['covered_pixels']==0 and b['covered_pixels']>0 for a,b in pairs),
            changed_from_zero_to_object_IoU50=sum(a['covered_pixels']==0 and b['IoU50_match'] for a,b in pairs),
            new_matches=sum(not a['IoU50_match'] and b['IoU50_match'] for a,b in pairs),lost_matches=sum(a['IoU50_match'] and not b['IoU50_match'] for a,b in pairs))
    return out,items
@torch.inference_mode()
def main():
    save(RUN/'STATUS.json',dict(event='WAIT_NATIVE_AND_CONTROL_SELECTIONS',pid=os.getpid(),time=time.time()))
    while not (O/'TRAINING_RECEIPT.json').exists():time.sleep(10)
    freeze=json.loads((O/'SELECTION_FROZEN.json').read_text());choices=freeze['choices']
    assert freeze['primary_budget']==.25
    save(RUN/'PROTOCOL.json',dict(scope='paired research application, source TUNE12 and target TUNE72',
        same_prompt_to_mask='frozen generic native SAM, predicted anchor+box, preserve original roofs and support',budget=.25,
        no_truth_in_network_or_mask=True,weights={k:v['weight_sha256'] for k,v in choices.items()},protected_reads=0))
    records=[json.loads(line) for line in Path('/root/xian20261111_r8/QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if line]
    groups={'SOURCE':[r for r in records if r['role']=='SOURCE_TUNE'],'XIAN':[r for r in records if r['role']=='TUNE']}
    assert [len(v) for v in groups.values()]==[12,72]
    sys.path.insert(0,str(ASSETS/'source'))
    from segment_anything import sam_model_registry
    all_records={}
    with LOCK.open('a') as gpu:
        fcntl.flock(gpu,fcntl.LOCK_EX)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(ASSETS/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True));sam.cuda().eval()
        models={}
        for typ,rec in choices.items():
            model=(DetailNet() if typ=='native' else ProposalNet()).cuda().to(memory_format=torch.channels_last)
            assert sha(rec['weight'])==rec['weight_sha256']
            model.load_state_dict(torch.load(rec['weight'],map_location='cuda')['model']);models[typ]=model.eval()
        for role,frames in groups.items():
            all_records[role]={k:[] for k in models}
            for index,row in enumerate(frames):
                assert sha(row['image_path'])==row['image_sha256'] and sha(row['query_cache_path'])==row['query_cache_sha256']
                with np.load(row['query_cache_path']) as z:
                    old=z['instances'].astype('int32');support=z['support'].astype(bool)
                boxes=raw_boxes(old);n=len(boxes)
                if role=='XIAN':
                    with rasterio.open(row['image_path']) as ds:rgb=torch.from_numpy(ds.read([1,2,3])[None]).cuda()
                    features=sam.image_encoder(sam.preprocess(rgb))
                else:
                    folder=PRE/'output/source_sam_pilot/frames'/row['id'];ep=folder/'IMAGE_EMBEDDING_FP32.npz'
                    assert sha(ep)==json.loads((folder/'PREDICTIONS_FROZEN_BEFORE_GT.json').read_text())['embedding_sha256']
                    with np.load(ep) as z:features=torch.from_numpy(z['image_embedding']).cuda()
                for typ,model in models.items():
                    folder=RUN/role/typ/row['id'];folder.mkdir(parents=True,exist_ok=True)
                    path=folder/'NATIVE.npz';receipt=folder/'REC.json'
                    if receipt.exists():
                        rec=json.loads(receipt.read_text());assert sha(path)==rec['sha256'] and rec['weight']==choices[typ]['weight_sha256']
                    else:
                        if role=='XIAN':
                            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(features,rgb) if typ=='native' else model(features)
                            padded=np.pad(boxes,((0,0),(2,0)));ranked=budget_decode(out,[support],[old>0],[padded])[0]
                        else:
                            stored=O/'source_tune_predictions'/typ/(row['id']+'.npz')
                            binding=json.loads((stored.parent/'FROZEN.json').read_text());assert sha(stored)==next(v['sha256'] for v in binding['rows'] if v['id']==row['id'])
                            with np.load(stored) as z:
                                assert int(z['raw_count'])==n;ranked=z['proposals']
                        selected=take_budget(ranked,n,.25)
                        lab,counts=add_masks(sam,features,selected,old,support,0.)
                        np.savez_compressed(path,ADDITIONS=lab,proposals=selected)
                        with np.load(path) as z:assert np.array_equal(lab,z['ADDITIONS'])
                        rec=dict(frame_id=row['id'],path=str(path),variant='ADDITIONS',weight=choices[typ]['weight_sha256'],
                          sha256=sha(path),proposed=len(selected),new_pixels=int(((lab>0)&(old==0)).sum()),**counts)
                        save(receipt,rec)
                    all_records[role][typ].append(rec)
                save(RUN/'STATUS.json',dict(event='PAIR_NATIVE_ROOF_MASK_INFERENCE',group=role,windows=index+1,total=len(frames),time=time.time(),pid=os.getpid()))
            save(RUN/role/'PAIRED_PREDICTIONS_FROZEN.json',all_records[role])
    sys.path.insert(0,str(PRE/'code'))
    from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
    source=RoofEvaluator('TUNE');by={r['id']:r for r in source.frames}
    report={}
    for typ in choices:
        scored=[];diag=[]
        for rec in all_records['SOURCE'][typ]:
            assert sha(rec['path'])==rec['sha256']
            f=by[rec['frame_id']]
            with np.load(rec['path']) as z:
                scored.append(metrics(f,z['ADDITIONS']));diag.append(missing_diagnosis(f,z['ADDITIONS'],z['proposals']))
        m=aggregate(scored);target,targetrows=score_target(groups['XIAN'],all_records['XIAN'][typ])
        save(RUN/(typ+'_MAP_REFERENCE_OBJECTS.json'),targetrows)
        report[typ]=dict(source_metrics=m,source_native_gate=passes(m,source.baseline),
            source_complete_miss={k:sum(x[k] for x in diag) for k in diag[0]},Xian_development=target)
    save(RUN/'RECEIPT.json',dict(status='COMPLETE_TWO_MATCHED_NATIVE_MASK_ARMS',windows_per_arm=84,
        raw_source=source.baseline,results=report,new_optimizer_steps=0,protected_reads=0,
        source_TUNE_reused=True,independent_Xian_roof_accuracy='NOT_ESTABLISHED',production_changed=False))
if __name__=='__main__':
    try:main()
    except Exception:
        save(RUN/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise

