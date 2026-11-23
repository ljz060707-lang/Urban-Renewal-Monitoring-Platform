"""Evaluate image-only independent building discovery on source masks and Xi'an TUNE; preserve all raw queries."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import fcntl
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from scipy import ndimage
from prepare_centers import R,T,sha,save
from proposal_net import ProposalNet,decode
from evaluate_proposals import ious

DIR=T/'development_discovery'
DIR.mkdir(parents=True,exist_ok=True)
PRE=Path('/root/xian20261111_r15')
R8=Path('/root/xian20261111_r8')
ASSETS=PRE/'external/sam'
LOCK=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock')
def status(event,**kw):
    save(DIR/'STATUS.json',dict(time=time.time(),pid=os.getpid(),event=event,**kw))
def raw_boxes(old):
    output=[]
    for ysxs in ndimage.find_objects(old):
        if ysxs is not None:
            y,x=ysxs;output.append((x.start,y.start,x.stop,y.stop))
    return np.asarray(output,np.float32).reshape(-1,4)
def add_masks(sam,image_features,predictions,old,support,threshold):
    # No threshold or score is selected from Xi'an or native reference geometry.
    proposals=predictions[predictions[:,0]>=threshold]
    boxes=raw_boxes(old)
    if len(boxes):
        proposals=proposals[~(ious(proposals[:,3:],boxes)>=.60).any(1)]
    labels=old.copy()
    new_id=int(old.max())+1
    counts=dict(post_box_filter=len(proposals),center_occupied_or_unsupported=0,no_anchor_or_small=0,added=0)
    for start in range(0,len(proposals),24):
        batch=proposals[start:start+24]
        points=batch[:,1:3]
        # SAM adds .5 to pixel indices. R17 encodes points in pixel-edge coordinates.
        pt=torch.from_numpy(points.copy()).to('cuda')-.5
        # Model predicts half-open edges, SAM takes inclusive maximal pixel index.
        b=batch[:,3:].copy();b[:,2:]-=1;b=np.clip(b,0,1023)
        sparse,dense=sam.prompt_encoder(points=(pt[:,None,:],torch.ones((len(b),1),dtype=torch.long,device='cuda')),
            boxes=torch.from_numpy(b).cuda(),masks=None)
        low,quality=sam.mask_decoder(image_embeddings=image_features,image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=True)
        selected=low[torch.arange(len(b),device='cuda'),quality.argmax(1)]
        masks=(torch.nn.functional.interpolate(selected[:,None],size=(1024,1024),mode='bilinear',align_corners=False)[:,0]>0).cpu().numpy()
        for point,mask in zip(points,masks):
            x,y=np.clip(np.floor(point).astype(int),0,1023)
            if not support[y,x] or labels[y,x]!=0:
                counts['center_occupied_or_unsupported']+=1
                continue
            available=mask & support & (labels==0)
            components,_=ndimage.label(available,structure=np.ones((3,3),bool))
            k=components[y,x]
            if k==0:
                counts['no_anchor_or_small']+=1
                continue
            region=components==k
            if region.sum()<32:
                counts['no_anchor_or_small']+=1
                continue
            labels[region]=new_id;new_id+=1;counts['added']+=1
    assert np.array_equal(labels[old>0],old[old>0]) and not labels[~support].any()
    return labels,counts
def evaluate_xian(rows,predictions):
    sys.path.insert(0,'/root/xian20261111_r16_roi_roof/code')
    from evaluate_xian_map_windows import evaluate,summarize
    by={r['frame_id']:r for r in predictions}
    assert len(by)==len(rows)==72
    with ProcessPoolExecutor(6) as pool:
        values=list(pool.map(evaluate,[(w,by) for w in rows]))
    summary={}
    for ep in sorted({r['epoch'] for r in values}):
        ww=[r for r in values if r['epoch']==ep]
        raw=summarize(ww,'RAW');candidate=summarize(ww,'ROI_SELECTED')
        pairs=[(a,b) for w in ww for a,b in zip(w['details']['RAW'],w['details']['ROI_SELECTED'])]
        assert all(a['fid']==b['fid'] for a,b in pairs)
        summary[ep]=dict(raw=raw,with_independent_additions=candidate,
             formerly_zero_now_any_cover=sum(a['covered_pixels']==0 and b['covered_pixels']>0 for a,b in pairs),
             formerly_zero_now_matched_IoU50=sum(a['covered_pixels']==0 and b['IoU50_match'] for a,b in pairs),
             gained_matches=sum(not a['IoU50_match'] and b['IoU50_match'] for a,b in pairs),
             lost_matches=sum(a['IoU50_match'] and not b['IoU50_match'] for a,b in pairs))
    save(DIR/'XIAN72_REFERENCE_OBJECTS.json',values)
    return summary
@torch.inference_mode()
def main():
    status('WAIT_VERIFIED_INDEPENDENT_MODEL')
    while not (T/'TRAINING_RECEIPT.json').exists():
        if (T/'FAILURE_RECEIPT.json').exists():
            status('WAIT_TRAINER_REPAIR')
        time.sleep(20)
    freeze=json.loads((T/'SELECTION_FROZEN.json').read_text())
    assert sha(freeze['weight'])==freeze['weight_sha256']
    threshold=float(freeze['selection']['threshold'])
    save(DIR/'PROTOCOL.json',dict(weight=freeze['weight_sha256'],threshold=threshold,
        single_preselected_head=True,raw_query_boxes_used_only_to_remove_geometric_duplicates=.60,
        mask='generic frozen SAM ViT-B full original image with predicted interior point and box; select highest model mask quality; keep anchor-containing available component >=32 pixels',
        scope='diagnostic source TUNE and 72 XiAn windows even if poor; official roof or city promotion requires independent quality verification',
        outside_support='ignored',retention='never erase any raw mask',reference_read='after each full role has saved predictions',production_changed=False))
    sys.path.insert(0,str(ASSETS/'source'));from segment_anything import sam_model_registry
    source=[json.loads(l) for l in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l]
    groups={k:[r for r in source if r['role']==v] for k,v in [('SOURCE','SOURCE_TUNE'),('XIAN','TUNE')]}
    assert len(groups['SOURCE'])==12 and len(groups['XIAN'])==72
    bindings={}
    with LOCK.open('a') as gpu:
        fcntl.flock(gpu,fcntl.LOCK_EX)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(ASSETS/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True));sam.cuda().eval()
        model=ProposalNet().cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(freeze['weight'],map_location='cuda')['model']);model.eval()
        for role,frames in groups.items():
            records=[]
            for row in frames:
                dest=DIR/role/row['id'];dest.mkdir(parents=True,exist_ok=True)
                receipt=dest/'PREDICTION.json';path=dest/'INDEPENDENT_ADDITIONS.npz'
                if receipt.exists():
                    prev=json.loads(receipt.read_text()); assert prev['sha256']==sha(path) and prev['weights']==freeze['weight_sha256']
                    records.append(prev);continue
                assert sha(row['query_cache_path'])==row['query_cache_sha256'] and sha(row['image_path'])==row['image_sha256']
                with np.load(row['query_cache_path']) as z:
                    old=z['instances'].astype('int32');support=z['support'].astype(bool)
                if role=='SOURCE':
                    f=PRE/'output/source_sam_pilot/frames'/row['id'];emb=f/'IMAGE_EMBEDDING_FP32.npz'
                    assert sha(emb)==json.loads((f/'PREDICTIONS_FROZEN_BEFORE_GT.json').read_text())['embedding_sha256']
                    with np.load(emb) as z:features=torch.from_numpy(z['image_embedding']).cuda()
                    pred=np.load(T/'source_tune'/(row['id']+'_proposals.npy'))
                else:
                    with rasterio.open(row['image_path']) as src:
                        rgb=src.read([1,2,3])
                    assert rgb.shape==(3,1024,1024)
                    features=sam.image_encoder(sam.preprocess(torch.from_numpy(rgb[None]).cuda()))
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(features.contiguous(memory_format=torch.channels_last))
                    pred=decode(out)[0]
                labels,counts=add_masks(sam,features,pred,old,support,threshold)
                np.savez_compressed(path,INDEPENDENT_ADDITIONS=labels,proposals=pred)
                with np.load(path) as z:assert np.array_equal(z['INDEPENDENT_ADDITIONS'],labels)
                rec=dict(frame_id=row['id'],role=role,path=str(path),sha256=sha(path),variant='INDEPENDENT_ADDITIONS',
                         weights=freeze['weight_sha256'],raw_foreground=int((old>0).sum()),new_foreground=int((labels>0).sum()-(old>0).sum()),**counts)
                save(receipt,rec);records.append(rec);status('INFER_RESEARCH_WINDOWS',role=role,frames=len(records),total=len(frames))
            save(DIR/role/'PREDICTIONS_FROZEN.json',dict(rows=records,weight=freeze['weight_sha256'],GT_prompts=0))
            bindings[role]=records
    sys.path.insert(0,str(PRE/'code'))
    from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
    source_eval=RoofEvaluator('TUNE');ref={w['id']:w for w in source_eval.frames}
    items=[]
    for row in bindings['SOURCE']:
        with np.load(row['path']) as z:items.append(metrics(ref[row['frame_id']],z['INDEPENDENT_ADDITIONS']))
    score=aggregate(items)
    xian=evaluate_xian(groups['XIAN'],bindings['XIAN'])
    summary=dict(status='COMPLETE_SOURCE_AND_TARGET_DEVELOPMENT_DISCOVERY',source_raw=source_eval.baseline,
        source_with_new_masks=score,source_native_gate=passes(score,source_eval.baseline),Xian_map_comparison=xian,source_windows=12,target_windows=72,
        actual_new_optimization_steps=0,Xian_complete_native_roof_or_facade_accuracy='NOT_ESTABLISHED',production_changed=False)
    save(DIR/'RECEIPT.json',summary);status('COMPLETE')
if __name__=='__main__':
    try:main()
    except Exception:
        save(DIR/'FAILURE_RECEIPT.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise

