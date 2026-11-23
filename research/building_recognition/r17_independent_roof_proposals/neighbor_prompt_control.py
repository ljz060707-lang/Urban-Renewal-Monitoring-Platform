"""Matched CPU replay on 32 whole source images: independent effects of a neighboring negative point and padding."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from pathlib import Path
import time
import json
import traceback
from collections import defaultdict
import numpy as np
from scipy import ndimage
import cv2
import torch
import torch.nn.functional as F
from pycocotools import mask as cm
from prepare_centers import T,S,sha,save
PRE=Path('/root/xian20261111_r15')
A=PRE/'external/sam'
O=T/'neighbor_prompt_control';O.mkdir(exist_ok=True)
sys.path.insert(0,str(A/'source'));sys.path.insert(0,str(PRE/'code'))
from segment_anything import sam_model_registry
from sam_native_pilot import assemble
ARMS=['PAD16_ONE_POINT','PAD16_NEIGHBOR_NEGATIVE','TIGHT_ONE_POINT','PAD32_ONE_POINT']
def make_inputs(old,queries):
    items=[]
    for q in queries:
        identity=q['query_id'];yt,xt=np.nonzero(old==identity);x0,x1=int(xt.min()),int(xt.max());y0,y1=int(yt.min()),int(yt.max())
        def padded(p):
            return [max(0,x0-p),max(0,y0-p),min(1023,x1+p),min(1023,y1+p)]
        box=padded(16);assert box==q['box_native_inclusive']
        x,y,xx,yy=box
        sub=old[y:yy+1,x:xx+1];neighbors=(sub!=0)&(sub!=identity)
        # Choose a point inside a different predicted instance, not across an unknown gap or from GT.
        neg=None
        if neighbors.any():
            dist=ndimage.distance_transform_edt(sub!=identity)
            erode=(sub==ndimage.minimum_filter(sub,size=3))&(sub==ndimage.maximum_filter(sub,size=3))&neighbors
            if erode.any():
                eligible=erode & (dist<=24)
                if eligible.any():
                    iy,ix=np.unravel_index(np.where(eligible,dist,np.inf).argmin(),dist.shape)
                    neg=[int(x+ix),int(y+iy)]
                    assert old[neg[1],neg[0]] not in (0,identity)
        items.append(dict(identity=identity,pos=q['point_native_pixel_index'],boxes=[box,box,padded(0),padded(32)],
                prior=-8+16*cv2.resize((old==identity).astype('float32'),(256,256),interpolation=cv2.INTER_AREA),neg=neg))
    return items
@torch.inference_mode()
def run_frame(sam,rec):
    assert sha(rec['cache_path'])==rec['cache_sha256'] and sha(rec['prompts_path'])==rec['prompts_sha256'] and sha(rec['embedding_path'])==rec['embedding_sha256']
    queries=json.loads(Path(rec['prompts_path']).read_text())['queries']
    with np.load(rec['cache_path']) as z:old=z['instances'].astype('int32');support=z['support'].astype(bool)
    with np.load(rec['embedding_path']) as z:emb=torch.from_numpy(z['image_embedding'])
    items=make_inputs(old,queries);candidates=defaultdict(dict)
    for a,name in enumerate(ARMS):
        active=[r for r in items if a!=1 or r['neg'] is not None]
        for start in range(0,len(active),8):
            batch=active[start:start+8]
            pts=torch.tensor([[r['pos'],r['neg']] if a==1 else [r['pos']] for r in batch],dtype=torch.float32)
            labels=torch.ones(pts.shape[:2],dtype=torch.int64)
            if a==1:labels[:,1]=0
            boxes=torch.tensor([r['boxes'][a] for r in batch],dtype=torch.float32)
            prior=torch.from_numpy(np.stack([r['prior'] for r in batch])[:,None])
            sparse,dense=sam.prompt_encoder(points=(pts,labels),boxes=boxes,masks=prior)
            low,quality=sam.mask_decoder(image_embeddings=emb,image_pe=sam.prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
            masks=(F.interpolate(low,size=(1024,1024),mode='bilinear',align_corners=False)[:,0]>0).numpy()
            for row,mask,q in zip(batch,masks,quality):
                e=cm.encode(np.asfortranarray((mask&support).astype('uint8')))
                e['counts']=e['counts'].decode()
                candidates[name][row['identity']]=dict(mode='PRIOR',predicted_mask_quality=float(q[0]),mask=e)
    maps={}
    for name in ARMS:
        results=[]
        for r in items:
            identity=r['identity'];selected=candidates[name].get(identity,candidates[ARMS[0]][identity])
            results.append(dict(query_id=identity,candidates=[selected]))
        maps[name]=assemble(old,support,results,'PRIOR',True)
    target=O/rec['frame_id'];target.mkdir(exist_ok=True);p=target/'NATIVE_MAPS.npz'
    np.savez_compressed(p,**maps)
    receipt=dict(frame=rec['frame_id'],queries=len(items),negative_available=sum(r['neg'] is not None for r in items),path=str(p),sha256=sha(p))
    save(target/'FRAME.json',receipt);return receipt
def main():
    cv2.setNumThreads(0);torch.set_num_threads(6)
    rows=[r for r in json.loads((S/'ALL_IMAGE_ONLY_INPUTS_FROZEN.json').read_text())['records'] if r['role']=='INNER']
    assert len(rows)==32
    save(O/'PROTOCOL.json',dict(arms=ARMS,inputs='identical 32 INNER frames/frozen RGB-only SAM embeddings and original predicted raw instances',
        model='generic ViT-B single mask, predicted prior, common guarded ownership; one factor changed at a time',
        negative='nearest 3x3 solid interior in different predicted instance within padded box and <=24 native pixels of current mask; if absent, replay identical reference mask',
        precision='CPU FP32 no AMP 6 CPU threads; no GPU and no new optimizer steps',protected_reads=0,GT_in_prompt=False))
    sam=sam_model_registry['vit_b']()
    sam.load_state_dict(torch.load(A/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True))
    sam.eval();records=[]
    for row in rows:
        p=O/row['frame_id']/'FRAME.json'
        if p.exists():
            rec=json.loads(p.read_text());assert sha(rec['path'])==rec['sha256']
        else:rec=run_frame(sam,row)
        records.append(rec)
        save(O/'STATUS.json',dict(frames=len(records),total=32,time=time.time(),pid=os.getpid()))
    save(O/'ALL_PREDICTIONS_FROZEN_BEFORE_GT.json',dict(rows=records))
    from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
    truth=RoofEvaluator('INNER');by={v['id']:v for v in truth.frames};scored=defaultdict(list)
    for row in records:
        with np.load(row['path']) as z:
            for arm in ARMS:scored[arm].append(metrics(by[row['frame']],z[arm]))
    results={k:aggregate(v) for k,v in scored.items()}
    save(O/'RECEIPT.json',dict(status='COMPLETE_NATIVE_NEIGHBOR_AND_BOX_CONTROL',raw=truth.baseline,metrics=results,
        native_qualified={k:passes(v,truth.baseline) for k,v in results.items()},frames=32,total_queries=sum(v['queries'] for v in records),
        queries_with_neighbor=sum(v['negative_available'] for v in records),source_inner_not_city_accuracy=True,optimizer_updates=0))
if __name__=='__main__':
    try:main()
    except Exception:
        save(O/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise

