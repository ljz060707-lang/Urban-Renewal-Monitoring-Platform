import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import fcntl
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from prepare_centers import R,D,T,build,sha,save
from proposal_net import FrameDataset,ProposalNet,decode,loss
from evaluate_proposals import assess,aggregate

SEED=20261127
BATCH=16
STEPS=2000
THRESHOLDS=(.15,.30,.50)
R16=Path('/root/xian20261111_r16_roi_roof')
GPU_LOCK=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock')
def status(event,**fields):
    save(T/'STATUS.json',dict(event=event,time=time.time(),pid=os.getpid(),**fields))
    print(event,fields,flush=True)
def qualify(s):
    return s['new_matches_zero_overlap']>0 and s['combined_precision_on_known_centers']>=s['raw_precision_on_known_centers'] and s['F1_all_proposal_budget']>s['raw_F1_all_proposal_budget']
def rank(s):
    return (qualify(s),s['F1_all_proposal_budget'],s['new_matches_zero_overlap'],s['combined_precision_on_known_centers'])
def domain_guard_test():
    heat=torch.zeros(1,2,2)
    heat[0,0,0]=1
    valid=torch.tensor([[[1,1],[0,0]]],dtype=torch.bool)
    b=dict(heat=heat,center_valid=valid,center_owner=heat.bool(),box_reg=torch.zeros(1,6,2,2),
           known=valid.float(),roof=heat)
    out=[torch.zeros_like(heat,requires_grad=True),torch.zeros_like(b['box_reg'],requires_grad=True),torch.zeros_like(heat,requires_grad=True)]
    total,_=loss(out,b);total.backward()
    assert out[0].grad[0,0].abs().min()>0 and out[2].grad[0,0].abs().min()>0
    assert not out[0].grad[~valid].any() and not out[2].grad[~valid].any()
    assert not out[1].grad[:,:,1,:].any() and not out[1].grad[:,:,0,1].any()
    return dict(ignored_heat_and_roof_gradient=0,off_center_regression_gradient=0)
@torch.no_grad()
def predict(model,features):
    model.eval();start=time.time();result=[]
    for i in range(0,len(features),8):
        inputs=np.stack([np.load(x) for x in features[i:i+8]])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(torch.from_numpy(inputs).cuda())
        result.extend(decode(out))
    return result,time.time()-start
def measured(frames,proposed,thresholds):
    with ThreadPoolExecutor(6) as pool:
        rows=list(pool.map(lambda xy:assess(*xy,thresholds=thresholds),zip(frames,proposed)))
    return aggregate(rows)
def store_checkpoint(path,model,opt,step,selection,plan_hash,data_hash):
    assert all(int(s['step'].item())==step for s in opt.state.values() if 'step' in s)
    tmp=path.with_suffix('.pending')
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),actual_AdamW_steps=step,selection=selection,
                    schedule_sha256=plan_hash,dataset_sha256=data_hash),tmp);tmp.replace(path)
def source_tune(model,selection):
    r8=Path('/root/xian20261111_r8')
    pre=Path('/root/xian20261111_r15/output/source_sam_pilot/frames')
    source=[r for line in (r8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))['role']=='SOURCE_TUNE']
    assert len(source)==12 and (T/'SELECTION_FROZEN.json').exists()
    dest=T/'source_tune'
    dest.mkdir(exist_ok=True)
    files=[];features=[]
    for row in source:
        directory=pre/row['id']
        binding=json.loads((directory/'PREDICTIONS_FROZEN_BEFORE_GT.json').read_text())
        fp=directory/'IMAGE_EMBEDDING_FP32.npz'
        assert sha(fp)==binding['embedding_sha256']
        with np.load(fp) as z:v=z['image_embedding'][0]
        assert v.shape==(256,64,64)
        path=dest/(row['id']+'_feature.npy');np.save(path,v);features.append(path)
        files.append(dict(frame_id=row['id'],role='TUNE',image_path=row['image_path'],image_sha256=row['image_sha256'],
                  embedding_path=str(fp),embedding_sha256=binding['embedding_sha256'],GT_used_for_generation=False))
    proposals,sec=predict(model,features)
    written=[]
    for row,det in zip(source,proposals):
        path=dest/(row['id']+'_proposals.npy');np.save(path,det)
        written.append(dict(frame=row['id'],prediction_sha256=sha(path),count=len(det)))
    save(dest/'PREDICTIONS_FROZEN_BEFORE_GT.json',dict(frames=written,threshold=selection['threshold'],prediction_seconds=sec))
    native=Path('/root/xian20261111_r10/instance_data')
    lookup=json.loads((native/'NATIVE_LABEL_LOOKUP.json').read_text())
    coco=json.loads((native/'TUNE.json').read_text());anns=defaultdict(list)
    for a in coco['annotations']:anns[a['image_id']].append(a)
    images={x['file_name']:x for x in coco['images']}
    role_rows=[]
    for row,f in zip(source,files):
        im=images[row['image_path']];lp=lookup[row['image_path']]
        role=dict(label_path=lp,label_sha256=sha(lp),query_cache_path=row['query_cache_path'],query_cache_sha256=row['query_cache_sha256'])
        role_rows.append(build((f,role,anns[im['id']])))
    return measured(role_rows,proposals,(float(selection['threshold']),))
def main():
    torch.set_num_threads(4)
    status('WAIT_DATA_VERIFIED')
    while not (D/'READY.json').exists():
        if (D/'FAILURE.json').exists():raise RuntimeError('data preparation failed')
        time.sleep(10)
    manifest=json.loads((D/'READY.json').read_text())
    rows=manifest['rows']
    fit=[r for r in rows if r['role']=='FIT'];inner=[r for r in rows if r['role']=='INNER']
    assert (len(fit),len(inner))==(972,32)
    data_hash=sha(D/'READY.json')
    def verify(r):
        return sha(r['features'])==r['features_sha256'] and sha(r['targets'])==r['targets_sha256']
    with ThreadPoolExecutor(6) as pool:assert all(pool.map(verify,rows))
    torch.manual_seed(SEED)
    preflight=ProposalNet().eval()
    sample=FrameDataset(fit)[0]
    assert sample['center_owner'].sum()>0
    b={k:v[None] for k,v in sample.items() if k!='index'}
    o=preflight(b['feature']);value,_=loss(o,b);value.backward()
    assert np.isfinite(float(value.detach())) and sum(p.grad is not None and p.grad.abs().sum()>0 for p in preflight.parameters())>10
    decoded=decode(o)[0];assert decoded.shape[1]==7
    proposal=np.zeros((0,7),np.float32)
    empty=aggregate([assess(inner[0],proposal)])
    # A perfectly provided interior point and box verifies actual mask matching on a native example.
    with np.load(inner[0]['targets']) as z:
        objects=z['gt_boxes']
        ideal=np.concatenate([np.ones((len(objects),1)),objects],axis=1)
    oracle=assess(inner[0],ideal)
    assert all(rec['independent_only_matches']==len(objects) for rec in oracle.values())
    save(T/'FUNCTIONAL.json',dict(status='PASS',actual_FIT_sample=fit[0]['id'],actual_INNER_frame=inner[0]['id'],
        oracle_matching_all_actual_known_GT=len(objects),verified_native_frame_hashes=len(rows),actual_optimizer_updates=0,**domain_guard_test()))
    schedule=T/'SCHEDULE.npy'
    if not schedule.exists():
        rng=np.random.default_rng(SEED);plan=np.concatenate([rng.permutation(len(fit)) for _ in range(math.ceil(STEPS*BATCH/len(fit)))])[:STEPS*BATCH].reshape(STEPS,BATCH);np.save(schedule,plan)
    plan=np.load(schedule);assert plan.shape==(STEPS,BATCH)
    plan_hash=sha(schedule)
    save(T/'PROTOCOL.json',dict(task='B0_INDEPENDENT_ROOF_CENTER_BOX_PROPOSAL',input='RGB-derived frozen full-frame SAM embedding only',
        heads=['roof interior point','center subpixel offset and four log distances to box','known roof fraction'],stride=8,steps=STEPS,effective_batch=BATCH,seed=SEED,
        optimizer='AdamW',lr=.0006,weight_decay=.01,warmup=100,cosine_floor=.05,clip_norm=5,precision='BF16_FORWARD_FP32_LOSS_AND_MASTER_WEIGHTS',
        center_thresholds=THRESHOLDS,NMS=.60,raw_duplicate_box_IoU=.60,topk_native_centers=256,source_validation='fixed 32 INNER at 0/250/500/1000/2000; frozen single choice reused-source TUNE12',
        objective='additional one-to-one interior anchor inside true roof and predicted box IoU>=0.5; report no-raw-pixel subset and unlabelled proposals separately',
        select='prioritize proposal yield and recovered no-pixel GT, nondecreasing known-center precision and improvement in full-proposal-budget F1; no production promotion from box metrics',
        dataset=data_hash,schedule=plan_hash,protected_reads=0))
    del b,o,preflight,value
    status('WAIT_PREVIOUS_ROOF_AND_QUALIFIED_TARGET_GPU',actual_optimizer_updates=0)
    while not (R16/'output/xian_roi_selected/PILOT_RECEIPT.json').exists():
        time.sleep(15)
    with GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        status('INDEPENDENT_HEAD_GPU_INITIALIZING',actual_optimizer_updates=0)
        torch.manual_seed(SEED);assert torch.cuda.is_bf16_supported()
        model=ProposalNet().cuda().to(memory_format=torch.channels_last)
        opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
        path=T/'CHECKPOINTS';path.mkdir(exist_ok=True)
        ck=sorted([p for p in path.glob('iter_*.pt') if p.with_suffix('.json').exists()],key=lambda p:int(p.stem.split('_')[1]))
        start=0;best=None
        if ck:
            state=torch.load(ck[-1],map_location='cuda');assert state['schedule_sha256']==plan_hash and state['dataset_sha256']==data_hash
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['actual_AdamW_steps']
            best=json.loads((T/'BEST_INNER.json').read_text())
        if not (T/'ZERO_CONTROL.json').exists():
            det,seconds=predict(model,[r['features'] for r in inner]);res=measured(inner,det,THRESHOLDS)
            save(T/'ZERO_CONTROL.json',dict(actual_optimizer_updates=0,metrics=res,prediction_seconds=seconds))
        history=json.loads((T/'HISTORY.json').read_text()) if (T/'HISTORY.json').exists() else []
        history=[v for v in history if v['step']<=start]
        if start<STEPS:
            loader=DataLoader(Subset(FrameDataset(fit),plan[start:].ravel().tolist()),batch_size=BATCH,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
            for step,cpu in enumerate(loader,start+1):
                assert np.array_equal(cpu.pop('index').numpy(),plan[step-1])
                model.train();lr=.0006*min(1.,step/100)*(.05+.95*(1+math.cos(math.pi*max(0,step-100)/(STEPS-100)))/2)
                for group in opt.param_groups:group['lr']=lr
                tick=time.time()
                b={k:v.cuda(non_blocking=True) for k,v in cpu.items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=model(b['feature'].contiguous(memory_format=torch.channels_last))
                objective,parts=loss(outputs,b);assert torch.isfinite(objective)
                objective.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad);opt.step()
                if step%25==0:
                    v=dict(step=step,objective=float(objective.detach()),grad_norm=float(grad),lr=lr,**{k:float(x.detach()) for k,x in parts.items()},batch=BATCH,
                           step_GPU_submitted_to_sync_seconds=time.time()-tick)
                    history.append(v);save(T/'HISTORY.json',history);status('TRAIN_INDEPENDENT_HEAD',actual_optimizer_updates=step,loss=v['objective'])
                if step in [250,500,1000,2000]:
                    predictions,sec=predict(model,[r['features'] for r in inner]);metrics=measured(inner,predictions,THRESHOLDS)
                    threshold=max(metrics,key=lambda k:rank(metrics[k]));selected=dict(step=step,threshold=threshold,metrics=metrics[threshold])
                    dest=path/f'iter_{step}.pt';store_checkpoint(dest,model,opt,step,selected,plan_hash,data_hash)
                    save(dest.with_suffix('.json'),dict(metrics=metrics,selection=selected,weight_sha256=sha(dest),prediction_seconds=sec))
                    if best is None or rank(selected['metrics'])>rank(best['metrics']):
                        best=selected
                    save(T/'BEST_INNER.json',best)
                    status('NATIVE_PROPOSAL_INNER_MEASURED',actual_optimizer_updates=step,threshold=threshold,new_matches=selected['metrics']['new_matches'],recovered_fully_missed=selected['metrics']['new_matches_zero_overlap'])
            del loader
        weight=path/f"iter_{best['step']}.pt";save(T/'SELECTION_FROZEN.json',dict(selection=best,weight_sha256=sha(weight),weight=str(weight),actual_AdamW_updates=STEPS))
        state=torch.load(weight,map_location='cuda');model.load_state_dict(state['model'])
        tune=source_tune(model,best)
        result=tune[best['threshold']]
        save(T/'TRAINING_RECEIPT.json',dict(status='COMPLETE_INDEPENDENT_NATIVE_ROOF_PROPOSAL_EXPERIMENT',
            actual_AdamW_updates=STEPS,parameters=sum(p.numel() for p in model.parameters()),selection=best,TUNE=tune,
            qualified_source_proposal_research=qualify(best['metrics']) and qualify(result),target_roof_accuracy='NOT_ESTABLISHED',
            no_GT_Xian_examples_trained=0,native_instance_masks_from_new_proposals='NOT_YET_EVALUATED',production_changed=False))
        status('COMPLETE',actual_optimizer_updates=STEPS)
if __name__=='__main__':
    try:main()
    except Exception:
        save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise

