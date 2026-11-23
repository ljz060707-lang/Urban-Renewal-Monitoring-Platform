import fcntl,json,os,sys,time,traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from pycocotools import mask as cm
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch
from loft_inference import O,R,sha,save,model_components,predict
P=Path('/root/xian20261109');P15=Path('/root/xian20261111_r15')
sys.path.insert(0,'/root/xian20261111_r18_detail/code')
from run_detail_masks import score_target
from evaluate_proposals import matching
CUTS=(.3,.5,.7)
def status(stage,**kw):save(O/'STATUS.json',dict(time=time.time(),stage=stage,pid=os.getpid(),**kw))
def assemble(raw,support,detections,cut,mode,position=False):
    output=raw.copy() if mode=='AUGMENT' else np.zeros_like(raw)
    next_id=int(output.max())
    binding={}
    for det in detections:
        if det['score']<cut:continue
        g=cm.decode(det['mask']).astype(bool)&support
        n=int(g.sum())
        if n<32:continue
        if mode=='AUGMENT' and ((g&(raw>0)).sum()>.15*n):continue
        if position:
            # Author BONAI vector is roof minus footprint. Translate roof towards map position.
            dx,dy=np.rint(-np.array(det['official_offset_native_xy'])).astype(int)
            if abs(dx)>=1024 or abs(dy)>=1024:continue
            if dx or dy:
                moved=np.zeros_like(g)
                ys=slice(max(0,-dy),min(1024,1024-dy));xs=slice(max(0,-dx),min(1024,1024-dx))
                moved[max(0,dy):min(1024,1024+dy),max(0,dx):min(1024,1024+dx)]=g[ys,xs]
                g=moved&support
        take=g&(output==0)
        if take.sum()<32:continue
        next_id+=1;output[take]=next_id
        binding[next_id]=det['official_offset_native_xy']
    assert not (output[~support]>0).any()
    if mode=='AUGMENT':assert np.array_equal(output[raw>0],raw[raw>0])
    return output,binding
def frame_predict(model,head,r,role):
    folder=O/'frozen_detections'/role/r['id'];folder.mkdir(parents=True,exist_ok=True)
    file=folder/'DETECTIONS.json';rec=folder/'FRAME.json'
    if rec.exists():
        q=json.loads(rec.read_text());assert sha(file)==q['sha256'];return q
    assert sha(r['image_path'])==r['image_sha256']
    if role=='XIAN':
        with rasterio.open(r['image_path']) as src:rgb=src.read([1,2,3]).transpose(1,2,0)
    else:rgb=np.asarray(Image.open(r['image_path']).convert('RGB'))
    scores,boxes,masks,offsets=predict(model,head,rgb)
    hits=[]
    for j in np.argsort(-scores,kind='stable'):
        if scores[j]<min(CUTS):continue
        m=cm.encode(np.asfortranarray(masks[j].astype('uint8')));m['counts']=m['counts'].decode()
        hits.append(dict(score=float(scores[j]),box_native_xyxy=boxes[j].tolist(),official_offset_native_xy=offsets[j].tolist(),mask=m))
    save(file,dict(id=r['id'],image_sha256=r['image_sha256'],all_scores_ge_005=len(scores),floor_for_saved_masks=min(CUTS),hits=hits))
    q=dict(frame_id=r['id'],path=str(file),sha256=sha(file),all_predictions=len(scores),evaluated=len(hits));save(rec,q);return q
def native_source(role,frames,configs):
    from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
    evaluator=RoofEvaluator(role);by={f['id']:f for f in evaluator.frames}
    def one(r):
        f=by[r['id']];d=json.loads((O/'frozen_detections'/role/r['id']/'DETECTIONS.json').read_text())['hits'];out={}
        for key,(mode,cut) in configs.items():
            lab,binding=assemble(f['old'],f['support'],d,cut,mode)
            out[key]=metrics(f,lab)
            if role=='TUNE':
                dest=O/'TUNE'/key/r['id'];dest.mkdir(parents=True,exist_ok=True);p=dest/'NATIVE.npz';np.savez_compressed(p,ROOF=lab)
                save(dest/'REC.json',dict(sha256=sha(p),offsets_by_output_id=binding))
        return out
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,frames))
    results={k:aggregate([r[k] for r in rows]) for k in configs}
    return evaluator.baseline,{k:dict(metrics=v,qualified=passes(v,evaluator.baseline)) for k,v in results.items()}
def check_source_offsets(frames,standalone):
    raw=json.loads((P/'input/BONAI_OFFSET_METADATA.json').read_text());meta=defaultdict(list)
    for r in raw['records']:
        meta[r['image_sha256']].append(r)
    rows=[]
    for f in frames:
        gt=meta[f['image_sha256']]
        if not gt:continue
        folder=O/'TUNE'/standalone/f['id']
        labels=np.load(folder/'NATIVE.npz')['ROOF']
        bind=json.loads((folder/'REC.json').read_text())['offsets_by_output_id']
        ids=sorted(map(int,bind));index={k:i for i,k in enumerate(ids)}
        sizes=np.bincount(labels.ravel())
        score=np.zeros((len(ids),len(gt)))
        for j,row in enumerate(gt):
            mask=cm.decode(cm.merge(cm.frPyObjects(row['roof_polygons'],1024,1024))).astype(bool)
            keys,n=np.unique(labels[mask],return_counts=True)
            for k,inter in zip(keys,n):
                if k in index:score[index[k],j]=inter/(sizes[k]+mask.sum()-inter)
        qq,gg=matching(score)
        for i,j in zip(qq,gg):
            target=np.asarray(gt[j]['target_roof_to_footprint_dxdy_px'])
            correction=-np.array(bind[str(ids[i])]);assert np.allclose(target,-np.asarray(gt[j]['offset_official_roof_minus_footprint_px']))
            rows.append(dict(frame=f['id'],annotation=gt[j]['annotation_id'],query=ids[i],IoU=float(score[i,j]),zero=float(np.linalg.norm(target)),model=float(np.linalg.norm(target-correction))))
    def summary(key):
        v=[r[key] for r in rows]
        return dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.quantile(v,.95))) if v else None
    save(O/'SOURCE_OFFSET_NATIVE_PIXEL.json',dict(rows=rows,n=len(rows),zero=summary('zero'),model=summary('model'),sign='negative of official roof-minus-footprint. Native pixel x-right y-down. Supplied labels are attached after predictions.'))
    return dict(n=len(rows),zero=summary('zero'),model=summary('model'))
def main():
    torch.set_num_threads(4)
    sys.path.insert(0,str(P15/'code'))
    allrows=[json.loads(l) for l in Path('/root/xian20261111_r8/QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l]
    by={r['id']:r for r in allrows}
    src=json.loads((P15/'data/native_roof_decoder/ALL_IMAGE_ONLY_INPUTS_FROZEN.json').read_text())['records']
    groups={'INNER':[by[r['frame_id']] for r in src if r['role']=='INNER'],'TUNE':[r for r in allrows if r['role']=='SOURCE_TUNE'],'XIAN':[r for r in allrows if r['role']=='TUNE']}
    assert [len(v) for v in groups.values()]==[32,12,72]
    save(O/'PROTOCOL.json',dict(roles=['SOURCE_INNER32','SOURCE_TUNE12','XIAN_TUNE72'],raw_model='author LOFT roof and offset independent of our B0 queries',
        mask_assembly='descending score and available supported pixels >=32; augment accepts independent masks with <=15 percent raw overlap',
        threshold_choices=CUTS,select_each_mode='source inner joint native criterion, then instance F1, boundary F1. No selection on XiAn',
        target_map_position='separate standalone roof translated by negative predicted author offset rounded to native pixel; no GT input. Map footprint semantics differ',
        source_validation_exposed_to_author_training_possible=True,protected_reads=0,new_optimizer_updates=0))
    with Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);model,head=model_components()
        for role,frames in groups.items():
            out=[]
            for i,f in enumerate(frames):
                out.append(frame_predict(model,head,f,role));status('NATIVE_AUTHOR_PREDICTION',role=role,frames=i+1,total=len(frames))
            save(O/role/'IMAGE_ONLY_PREDICTIONS_FROZEN.json',dict(frames=out))
            if role=='INNER':
                cs={f'{m}_{c:.2f}':(m,c) for m in ['STANDALONE','AUGMENT'] for c in CUTS}
                base,res=native_source(role,frames,cs)
                choices={}
                for mode in ['STANDALONE','AUGMENT']:
                    cand=[k for k in cs if cs[k][0]==mode]
                    winner=max(cand,key=lambda k:(res[k]['qualified'],res[k]['metrics']['F1_50'],res[k]['metrics']['boundary_F1_native2px']))
                    choices[winner]=cs[winner]
                save(O/'INNER_SELECTION_FROZEN.json',dict(choices=choices,baseline=base,results=res))
            elif role=='TUNE':
                base,res=native_source(role,frames,choices)
                sk=next(k for k in choices if choices[k][0]=='STANDALONE');offset=check_source_offsets(frames,sk)
                save(O/'SOURCE_TUNE.json',dict(raw=base,results=res,offset_matched_native=offset))
        del model,head;torch.cuda.empty_cache()
    configs={k:(m,c,False) for k,(m,c) in choices.items()}
    configs['LEARNED_REFERENCE_POSITION']=(choices[sk][0],choices[sk][1],True)
    target_results={}
    for key,(mode,cut,position) in configs.items():
        output=[]
        for f in groups['XIAN']:
            assert sha(f['query_cache_path'])==f['query_cache_sha256']
            with np.load(f['query_cache_path']) as z:old=z['instances'].astype('int32');support=z['support'].astype(bool)
            hits=json.loads((O/'frozen_detections/XIAN'/f['id']/'DETECTIONS.json').read_text())['hits']
            lab,ids=assemble(old,support,hits,cut,mode,position)
            dest=O/'XIAN'/key/f['id'];dest.mkdir(parents=True,exist_ok=True);p=dest/'NATIVE.npz';np.savez_compressed(p,ROOF=lab)
            rec=dict(frame_id=f['id'],path=str(p),variant='ROOF',sha256=sha(p),instances=int(np.unique(lab[lab>0]).size),offset_output_ids=ids)
            save(dest/'REC.json',rec);output.append(rec)
        save(O/'XIAN'/key/'PREDICTIONS_FROZEN.json',dict(rows=output))
        summary,values=score_target(groups['XIAN'],output);save(O/'XIAN'/(key+'_REFERENCE_OBSERVATIONS.json'),values);target_results[key]=summary
    save(O/'RECEIPT.json',dict(status='COMPLETE_PUBLISHED_LOFT_PORT_SOURCE_AND_XIAN_DEVELOPMENT',Xian_map_metrics=target_results,
        source_TUNE=json.loads((O/'SOURCE_TUNE.json').read_text()),thresholds_inner_only=choices,protected_reads=0,new_AdamW_updates=0,production_changed=False,
        author_environment_bitwise_parity_and_Xian_native_roof_accuracy='NOT_ESTABLISHED'));status('COMPLETE')
if __name__=='__main__':
    try:main()
    except Exception:
        save(O/'FAILURE.json',dict(traceback=traceback.format_exc(),time=time.time()));raise

