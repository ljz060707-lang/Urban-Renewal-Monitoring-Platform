"""Test the trained independent semantic heads as direct roof masks, bypassing regressed boxes and SAM prompts."""
import os,sys,json,time,fcntl,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
import torch.nn.functional as F
import rasterio
from PIL import Image
from scipy import ndimage
from detail_inputs import P,R,O,D,save,sha
from train_detail import LOCK,DetailNet,ProposalNet
from run_detail_masks import missing_diagnosis,score_target
from infer_development_roofs import PRE,ASSETS

OUT=O/'semantic_discovery';OUT.mkdir(exist_ok=True)
CUTS=(.35,.50,.65)
def components(prob,old,support,cut):
    fg=(prob>=cut)&support
    cc,n=ndimage.label(fg,structure=np.ones((3,3),bool))
    boxes=ndimage.find_objects(cc)
    count=np.bincount(cc.ravel(),minlength=n+1)
    previous=np.bincount(cc[old>0],minlength=n+1)
    mean=np.bincount(cc.ravel(),weights=prob.ravel(),minlength=n+1)/np.maximum(count,1)
    kept=np.flatnonzero((count>=32)&(previous<=.15*count)&(np.arange(n+1)>0))
    # Use the actual per-window addition budget; order by mean evidence inside the predicted component.
    k=max(2,int(np.ceil(np.count_nonzero(np.unique(old))*.25)))
    labels=old.copy();idx=int(old.max())
    for region in kept[np.argsort(-mean[kept],kind='stable')][:k]:
        sl=boxes[region-1];available=(cc[sl]==region)&(old[sl]==0)
        sub,num=ndimage.label(available,structure=np.ones((3,3),bool))
        area=np.bincount(sub.ravel(),minlength=num+1);area[0]=0
        if area.max()<32:continue
        idx+=1;labels[sl][sub==area.argmax()]=idx
    assert np.array_equal(old[old>0],labels[old>0]) and not labels[~support].any()
    return labels
def source_measure(role,bindings):
    sys.path.insert(0,str(PRE/'code'))
    from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
    evaluator=RoofEvaluator(role);by={r['id']:r for r in evaluator.frames}
    def one(pair):
        frame,row=pair;f=by[frame]
        with np.load(row['path']) as z:
            scored=metrics(f,z['ADDITIONS'])
            missed=missing_diagnosis(f,z['ADDITIONS'],np.zeros((0,7),np.float32))
        return scored,missed
    measured={}
    with ThreadPoolExecutor(4) as pool:
        for arm,rows in bindings.items():
            results=list(pool.map(one,[(r['frame_id'],r) for r in rows]))
            m=aggregate([x[0] for x in results]);diag={k:sum(x[1][k] for x in results) for k in results[0][1]}
            measured[arm]=dict(metrics=m,whole_roof=diag,source_native_gate=passes(m,evaluator.baseline))
    return evaluator.baseline,measured
def status(stage,**kw):save(OUT/'STATUS.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
@torch.inference_mode()
def main():
    status('WAIT_FOR_PAIRED_ROOF_MASK_EVALUATION')
    while not (O/'matched_budget_masks/RECEIPT.json').exists():time.sleep(10)
    selection=json.loads((O/'SELECTION_FROZEN.json').read_text())['choices'];cv2.setNumThreads(0)
    save(OUT/'PROTOCOL.json',dict(arms=list(selection),cuts=CUTS,proposal_budget=.25,min_component_px=32,max_raw_overlap=.15,
        method='use trained semantic logits; threshold native bilinear probabilities, components with at least 85 percent uncovered, retain largest connected available part, rank by mean foreground probability',
        selection='source INNER only; common qualification, recovery of wholly missed native roofs, F1, pixel IoU. Single choice on reused source TUNE and target TUNE.',
        roof_geometry_from='trained image-only roof fraction head without predicted box, manual reference or SAM mask prompts',protected_reads=0))
    old=json.loads((D/'READY.json').read_text())['rows']
    inner=[r for r in old if r['role']=='INNER']
    records=[json.loads(l) for l in Path('/root/xian20261111_r8/QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l]
    ref={r['id']:r for r in records}
    groups={'INNER':[ref[r['id']] for r in inner],'TUNE':[r for r in records if r['role']=='SOURCE_TUNE'],'XIAN':[r for r in records if r['role']=='TUNE']}
    inner_by={r['id']:r for r in inner}
    assert [len(v) for v in groups.values()]==[32,12,72]
    with LOCK.open('a') as gpu:
        fcntl.flock(gpu,fcntl.LOCK_EX);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        models={}
        for typ,rec in selection.items():
            model=(DetailNet() if typ=='native' else ProposalNet()).cuda().to(memory_format=torch.channels_last)
            assert sha(rec['weight'])==rec['weight_sha256'];model.load_state_dict(torch.load(rec['weight'],map_location='cuda')['model']);models[typ]=model.eval()
        configs={f'{typ}_{cut:.2f}':(typ,cut) for typ in models for cut in CUTS}
        results={}
        for role,frames in groups.items():
            binds={name:[] for name in configs}
            if role=='XIAN':
                sys.path.insert(0,str(ASSETS/'source'));from segment_anything import sam_model_registry
                sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(ASSETS/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True));sam.cuda().eval()
            for index,r in enumerate(frames):
                assert sha(r['query_cache_path'])==r['query_cache_sha256']
                with np.load(r['query_cache_path']) as z:old_id=z['instances'].astype('int32');sup=z['support'].astype(bool)
                if role=='XIAN':
                    assert sha(r['image_path'])==r['image_sha256']
                    with rasterio.open(r['image_path']) as ds:rgb=torch.from_numpy(ds.read([1,2,3])[None]).cuda()
                    emb=sam.image_encoder(sam.preprocess(rgb))
                else:
                    row=inner_by[r['id']] if role=='INNER' else dict(rgb=str(O/'source_tune_inputs'/r['id']/'rgb.npy'),features=str(O/'source_tune_inputs'/r['id']/'features.npy'))
                    rgb=torch.from_numpy(np.load(row['rgb'])).permute(2,0,1)[None].cuda()
                    emb=torch.from_numpy(np.load(row['features']))[None].cuda()
                probabilities={}
                for name,(typ,cut) in configs.items():
                    folder=OUT/role/name/r['id'];folder.mkdir(parents=True,exist_ok=True);p=folder/'MAP.npz';receipt=folder/'REC.json'
                    if receipt.exists():
                        rec=json.loads(receipt.read_text());assert sha(p)==rec['sha256']
                    else:
                        if typ not in probabilities:
                            with torch.autocast('cuda',dtype=torch.bfloat16):logits=models[typ](emb,rgb)[2] if typ=='native' else models[typ](emb)[2]
                            probabilities[typ]=F.interpolate(logits[:,None].float(),(1024,1024),mode='bilinear',align_corners=False)[0,0].sigmoid().cpu().numpy()
                        lab=components(probabilities[typ],old_id,sup,cut)
                        np.savez_compressed(p,ADDITIONS=lab);rec=dict(frame_id=r['id'],path=str(p),sha256=sha(p),variant='ADDITIONS',
                            new_instances=int(lab.max()-old_id.max()),new_pixels=int(((lab>0)&(old_id==0)).sum()))
                        save(receipt,rec)
                    binds[name].append(rec)
                status('EVALUATE_SEMANTIC_HEAD',role=role,frames=index+1,total=len(frames))
            save(OUT/role/'PREDICTIONS_FROZEN.json',binds)
            if role in ['INNER','TUNE']:
                baseline,summary=source_measure(role,binds)
                results[role]=dict(baseline=baseline,arms=summary)
                if role=='INNER':
                    name=max(summary,key=lambda k:(summary[k]['source_native_gate'],summary[k]['whole_roof']['one_to_one_new_mask_IoU50'],summary[k]['metrics']['F1_50'],summary[k]['metrics']['pixel_IoU']))
                    configs={name:configs[name]};save(OUT/'INNER_SELECTION_BEFORE_TUNE.json',dict(chosen=name,roles=results))
            else:
                name=next(iter(configs));summary,objs=score_target(frames,binds[name]);save(OUT/'TARGET_REFERENCE_OBJECTS.json',objs);results['XIAN']=summary
            save(OUT/(role+'_EVALUATION.json'),results[role])
    save(OUT/'RECEIPT.json',dict(status='COMPLETE_FROZEN_SEMANTIC_HEAD_CANDIDATE_DISCOVERY',results=results,selection=list(configs),actual_optimizer_updates=0,
        production_changed=False,protected_reads=0,Xian_quality_scope='partial 2019 map agreement, actual-date visibility remains separate'));status('COMPLETE')
if __name__=='__main__':
    try:main()
    except Exception:
        save(OUT/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise

