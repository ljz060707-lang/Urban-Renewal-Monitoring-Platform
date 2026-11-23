import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import fcntl
import json
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,Subset
from detail_inputs import D,O,P,R,save,sha
from detail_model import DetailNet,NativeDataset,weighted_loss,budget_decode,take_budget
from proposal_net import ProposalNet,loss as reference_loss
from evaluate_proposals import assess,aggregate
from prepare_centers import anchor

SEED=20261127
BATCH=16
STEPS=2000
BUDGETS=(.10,.25,.50)
ARMS=('NATIVE_RGB','NATIVE_RGB_MISS_WEIGHT4')
LOCK=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock')
def status(event,**kw):
    save(O/'STATUS.json',dict(event=event,time=time.time(),pid=os.getpid(),**kw));print(event,kw,flush=True)
def selection_key(m):
    return (m['new_matches_zero_overlap'],m['new_matches'],m['extra_yield_all_centers'])
@torch.no_grad()
def predictions(model,rows,detail):
    model.eval();all_pred=[]
    for start in range(0,len(rows),4):
        batch=rows[start:start+4]
        features=torch.from_numpy(np.stack([np.load(r['features']) for r in batch])).cuda()
        if detail:
            rgb=torch.from_numpy(np.stack([np.load(r['rgb']) for r in batch])).permute(0,3,1,2).cuda()
        support=[];raw=[];boxes=[]
        for row in batch:
            with np.load(row['raw']) as z:
                raw.append(np.unpackbits(z['raw']).reshape(1024,1024).astype(bool))
                support.append(np.unpackbits(z['support']).reshape(1024,1024).astype(bool))
            with np.load(row.get('model_boxes',row.get('targets'))) as z:boxes.append(z['raw_boxes'])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(features,rgb) if detail else model(features)
        values=budget_decode(out,support,raw,boxes)
        all_pred.extend([(p,len(b)) for p,b in zip(values,boxes)])
    return all_pred
def measured(rows,preds):
    def one(t):
        r,(p,n)=t
        return {str(f):assess(r,take_budget(p,n,f),thresholds=(0.,))['0.0'] for f in BUDGETS}
    with ThreadPoolExecutor(6) as pool:return aggregate(list(pool.map(one,zip(rows,preds))))
def checkpoint(path,model,opt,step,schedule,dataset):
    assert all(int(st['step'].item())==step for st in opt.state.values() if 'step' in st)
    tmp=path.with_suffix('.pending')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),actual_optimizer_updates=step,
        schedule_sha256=schedule,dataset_sha256=dataset),tmp);tmp.replace(path)
def tune_inputs():
    r8=Path('/root/xian20261111_r8');pre=Path('/root/xian20261111_r15/output/source_sam_pilot/frames')
    records=[r for line in (r8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))['role']=='SOURCE_TUNE'];assert len(records)==12
    rows=[]
    for row in records:
        dest=O/'source_tune_inputs'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        assert sha(row['query_cache_path'])==row['query_cache_sha256'] and sha(row['image_path'])==row['image_sha256']
        binding=json.loads((pre/row['id']/'PREDICTIONS_FROZEN_BEFORE_GT.json').read_text())
        ep=pre/row['id']/'IMAGE_EMBEDDING_FP32.npz';assert sha(ep)==binding['embedding_sha256']
        with np.load(ep) as z:np.save(dest/'features.npy',z['image_embedding'][0])
        np.save(dest/'rgb.npy',np.asarray(Image.open(row['image_path']).convert('RGB')))
        with np.load(row['query_cache_path']) as z:
            labels=z['instances'];support=z['support'].astype(bool)
        boxes=np.asarray([anchor(labels==q) for q in np.unique(labels) if q],np.float32).reshape(-1,6)
        np.savez_compressed(dest/'input.npz',raw=np.packbits(labels>0),support=np.packbits(support),raw_boxes=boxes)
        rows.append(dict(id=row['id'],features=str(dest/'features.npy'),rgb=str(dest/'rgb.npy'),raw=str(dest/'input.npz'),model_boxes=str(dest/'input.npz')))
    return rows
def main():
    torch.set_num_threads(4)
    status('WAIT_PACKED_NATIVE_RGB')
    while not (D/'READY.json').exists():
        if (D/'FAILURE.json').exists():raise RuntimeError('RGB preparation failed')
        time.sleep(5)
    rows=json.loads((D/'READY.json').read_text())['rows']
    fit=[r for r in rows if r['role']=='FIT'];inner=[r for r in rows if r['role']=='INNER'];assert (len(fit),len(inner))==(972,32)
    plan_path=P/'output/SCHEDULE.npy';plan=np.load(plan_path);assert plan.shape==(STEPS,BATCH)
    sh=sha(plan_path);dh=sha(D/'READY.json')
    def verified(r):
        return all(sha(r[k])==r[h] for k,h in [('rgb','rgb_sha'),('raw','raw_sha'),('features','features_sha256'),('targets','targets_sha256')])
    with ThreadPoolExecutor(6) as pool:assert all(pool.map(verified,rows))
    save(O/'PROTOCOL.json',dict(data_sha256=dh,schedule_sha256=sh,same_as_R17=True,arms=ARMS,steps_per_arm=STEPS,
        controls='four immutable R17 SAM-feature-only checkpoints re-evaluated with identical center-eligibility and proposal budget',
        native_RGB='unshuffle stride 4 then learnt convolutions and stride 2, zero-initialized residual fused with full-frame SAM detail',
        boost='x4 weights on centers and box regression of real GT instances entirely missed by the fixed parent; no GT or raw mask as network input',
        eligibility='predicted center supported and outside raw predicted roofs BEFORE peak ranking; remove duplicate raw boxes; NMS 0.60',
        budget_fraction=BUDGETS,primary_budget=.25,per_frame='max(2,ceil(fraction*native parent instance count)); all proposals cost in denominator',
        selection='on INNER, prioritize number of known whole-roof misses recovered at same main budget, then all extra correct boxes, then per-proposal yield; predictions frozen before reused TUNE join',
        no_source_TUNE_gradient=True,production_changed=False,protected_reads=0))
    torch.manual_seed(SEED);bmodel=ProposalNet().eval();torch.manual_seed(SEED);dmodel=DetailNet().eval()
    data=NativeDataset(fit)[0];b={k:v[None] for k,v in data.items() if k!='index'}
    with torch.no_grad():
        p=bmodel(b['feature']);q=dmodel(b['feature'],b['rgb'])
        assert all(torch.equal(x,y) for x,y in zip(p,q));original,_=reference_loss(p,b);new,_=weighted_loss(q,b,False);assert torch.allclose(original,new,rtol=1e-6,atol=1e-7)
    save(O/'FUNCTIONAL.json',dict(checked_frame=fit[0]['id'],same_parent_logits_all_three_heads_bitexact=True,unweighted_loss_matches_original=True,frames=1004,actual_updates=0))
    del bmodel,dmodel,b
    while not (P/'output/development_discovery/RECEIPT.json').exists():time.sleep(10)
    with LOCK.open('a') as gpu:
        fcntl.flock(gpu,fcntl.LOCK_EX)
        summaries=[]
        base=ProposalNet().cuda().to(memory_format=torch.channels_last)
        for step in [250,500,1000,2000]:
            path=P/'output/CHECKPOINTS'/f'iter_{step}.pt';meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['weight_sha256']
            report=O/'baseline'/(path.stem+'.json')
            if report.exists():rec=json.loads(report.read_text())
            else:
                state=torch.load(path,map_location='cuda');base.load_state_dict(state['model'])
                mt=measured(inner,predictions(base,inner,False));rec=dict(arm='R17_EMBEDDING_ONLY',step=step,weight=str(path),weight_sha256=sha(path),metrics=mt)
                save(report,rec)
            summaries.append(rec)
        del base
        for arm in ARMS:
            folder=O/arm;folder.mkdir(exist_ok=True)
            torch.manual_seed(SEED)
            model=DetailNet().cuda().to(memory_format=torch.channels_last)
            opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
            start=0
            for step in [250,500,1000,2000]:
                p=folder/f'iter_{step}.json'
                if p.exists():
                    rec=json.loads(p.read_text());assert sha(rec['weight'])==rec['weight_sha256']
                    summaries.append(rec);start=step
            if start:
                state=torch.load(folder/f'iter_{start}.pt',map_location='cuda');assert state['schedule_sha256']==sh and state['dataset_sha256']==dh
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            history=json.loads((folder/'HISTORY.json').read_text()) if (folder/'HISTORY.json').exists() else []
            history=[v for v in history if v['step']<=start]
            if start<STEPS:
                loader=DataLoader(Subset(NativeDataset(fit),plan[start:].ravel().tolist()),batch_size=BATCH,num_workers=4,persistent_workers=True,pin_memory=True,prefetch_factor=2)
                for step,cpu in enumerate(loader,start+1):
                    idx=cpu.pop('index');assert np.array_equal(idx.numpy(),plan[step-1])
                    model.train();lr=.0006*min(1.,step/100)*(.05+.95*(1+math.cos(math.pi*max(0,step-100)/(STEPS-100)))/2)
                    for g in opt.param_groups:g['lr']=lr
                    b={k:v.cuda(non_blocking=True) for k,v in cpu.items()};opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['feature'],b['rgb'])
                    objective,parts=weighted_loss(out,b,arm==ARMS[1]);assert torch.isfinite(objective);objective.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);opt.step()
                    if step%25==0:
                        history.append(dict(step=step,loss=float(objective.detach()),gradient_norm=float(norm),lr=lr,**{k:float(v.detach()) for k,v in parts.items()}));save(folder/'HISTORY.json',history)
                        status('TRAIN_NATIVE_DETAIL_DISCOVERY',arm=arm,arm_updates=step,total_actual_updates=step+STEPS*ARMS.index(arm))
                    if step in [250,500,1000,2000]:
                        mt=measured(inner,predictions(model,inner,True));path=folder/f'iter_{step}.pt';checkpoint(path,model,opt,step,sh,dh)
                        rec=dict(arm=arm,step=step,weight=str(path),weight_sha256=sha(path),metrics=mt);save(path.with_suffix('.json'),rec);summaries.append(rec)
                del loader
            del model,opt
        # Freeze one winning native-detail model and one strongest control, both using INNER only.
        choices={typ:max([v for v in summaries if (v['arm']=='R17_EMBEDDING_ONLY')==(typ=='control')],key=lambda r:selection_key(r['metrics']['0.25'])) for typ in ['control','native']}
        save(O/'SELECTION_FROZEN.json',dict(choices=choices,actual_new_AdamW_updates=2*STEPS,primary_budget=.25))
        tune=tune_inputs();predset={}
        for typ,rec in choices.items():
            model=(DetailNet() if typ=='native' else ProposalNet()).cuda().to(memory_format=torch.channels_last)
            state=torch.load(rec['weight'],map_location='cuda');model.load_state_dict(state['model'])
            preds=predictions(model,tune,typ=='native');predset[typ]=preds
            output=O/'source_tune_predictions'/typ;output.mkdir(parents=True,exist_ok=True);bind=[]
            for row,(arr,n) in zip(tune,preds):
                p=output/(row['id']+'.npz');np.savez_compressed(p,proposals=arr,raw_count=n);bind.append(dict(id=row['id'],sha256=sha(p)))
            save(output/'FROZEN.json',dict(rows=bind,weight=rec['weight_sha256']))
        for row in tune:
            row['targets']=str(P/'data/TUNE'/row['id']/'targets.npz')
            receipt=json.loads(Path(row['targets']).with_name('FRAME.json').read_text());assert sha(row['targets'])==receipt['targets_sha256']
        results={typ:measured(tune,p) for typ,p in predset.items()}
        save(O/'TRAINING_RECEIPT.json',dict(status='COMPLETE_TWO_NATIVE_ROOF_DISCOVERY_TRAINING_ARMS',actual_new_AdamW_updates=4000,
             choices=choices,source_TUNE=results,reference='reused Jinan development only, interior-point and box metric',
             Xian_independent_accuracy='NOT_ESTABLISHED',production_changed=False))
        status('COMPLETE',total_actual_updates=4000)
if __name__=='__main__':
    try:main()
    except Exception:
        save(O/'FAILURE_RECEIPT.json',dict(time=time.time(),traceback=traceback.format_exc()));raise

