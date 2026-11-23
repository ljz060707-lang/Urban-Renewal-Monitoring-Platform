"""Xi'an TUNE inference after frozen source selection, with one frame of ROI features at a time."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from prepare_roi_embeddings import A, R, crop_bounds, save
from roi_training_data import sha
from roi_roof_evaluation import NativeROIEvaluator
from sam_native_pilot import point_box, assemble

OUTPUT = R / 'output' / 'xian_roi_selected'
LOCK = Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock')
TRAIN = R / 'output' / 'training'
OUTPUT.mkdir(parents=True, exist_ok=True)

def prepare():
    cache_records = Path('/root/xian20261111_r8/QUERY_CACHE_RECORDS.jsonl')
    records = [r for line in cache_records.read_text().splitlines() if (r:=json.loads(line))['role']=='TUNE']
    assert len(records)==72 and len({r['site'] for r in records})==24
    frames=[]
    for r in records:
        assert sha(r['query_cache_path'])==r['query_cache_sha256'] and sha(r['image_path'])==r['image_sha256']
        with np.load(r['query_cache_path']) as z:
            old=z['instances']
            assert old.shape==(1024,1024) and not (old[~z['support'].astype(bool)]>0).any()
            sizes=np.bincount(old.ravel().astype('int64'))
            queries=[]
            for identity in np.flatnonzero(sizes>=32):
                if identity==0:
                    continue
                point,box=point_box(old==identity)
                q=dict(query_id=int(identity),point_native_pixel_index=point,box_native_inclusive=box,slot=len(queries))
                q['crop_native_xy_size']=crop_bounds(q)
                queries.append(q)
        frames.append(dict(frame_id=r['id'],cache=r['query_cache_path'],cache_sha=r['query_cache_sha256'],
            rgb=r['image_path'],rgb_sha=r['image_sha256'],queries=queries))
    manifest=dict(frames=frames,n_windows=len(frames),n_model_ROIs=sum(len(f['queries']) for f in frames),
        proposals_from='frozen predicted masks only; <32px preserved by assembly',
        no_reference_geometry_or_presence_used=True)
    save(OUTPUT/'TARGET_MODEL_PROMPTS.json',manifest)
    return frames

def launch():
    import torch
    import rasterio
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0,str(A/'source'))
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    frames=prepare()
    save(OUTPUT/'STATUS.json',dict(event='WAIT_FOR_FROZEN_SOURCE_VALIDATION',pid=os.getpid(),windows=len(frames)))
    while not (TRAIN/'TRAINING_RECEIPT.json').exists():
        if (TRAIN/'FAILURE_RECEIPT.json').exists():
            save(OUTPUT/'STATUS.json',dict(event='WAIT_SOURCE_TRAINING_REPAIR',pid=os.getpid()))
        time.sleep(30)
    result=json.loads((TRAIN/'TRAINING_RECEIPT.json').read_text())
    if not result['qualified_source_research_candidate']:
        save(OUTPUT/'PILOT_RECEIPT.json',dict(status='NOT_RUN_SOURCE_NATIVE_RECALL_QUALITY_CRITERIA',train_sha256=sha(TRAIN/'TRAINING_RECEIPT.json'),target_inference_frames=0))
        return
    chosen=result['chosen']
    variant=chosen['selection']['variant']
    assert chosen['checkpoint_sha256']==sha(chosen['checkpoint'])
    assert variant in {'ROI_PRIOR_GUARDED','ROI_PRIOR_UNGUARDED'}
    outputs=[]
    with LOCK.open('a') as lock, ThreadPoolExecutor(4) as workers, torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        sam=sam_model_registry['vit_b']()
        sam.load_state_dict(torch.load(A/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True))
        weights=torch.load(chosen['checkpoint'],map_location='cpu')
        sam.mask_decoder.load_state_dict(weights['decoder'])
        sam.cuda().eval()
        transform=ResizeLongestSide(1024)
        evaluator=NativeROIEvaluator.__new__(NativeROIEvaluator)
        for frame in frames:
            identity=frame['frame_id']
            directory=OUTPUT/'frames'/identity
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/'MODEL_INSTANCES.npz'
            receipt=directory/'PREDICTION.json'
            if receipt.exists():
                prev=json.loads(receipt.read_text())
                assert prev['weight_sha256']==chosen['checkpoint_sha256'] and sha(path)==prev['sha256'] and prev['variant']==variant
                outputs.append(prev)
                continue
            assert sha(frame['rgb'])==frame['rgb_sha'] and sha(frame['cache'])==frame['cache_sha']
            with np.load(frame['cache']) as z, rasterio.open(frame['rgb']) as ds:
                old=z['instances'].astype('int32')
                support=z['support'].astype(bool)
                rgb=ds.read([1,2,3]).transpose(1,2,0)
            assert rgb.shape==(1024,1024,3) and rgb.dtype==np.uint8
            def pixels(q):
                x,y,s=q['crop_native_xy_size']
                im=transform.apply_image(rgb[y:y+s,x:x+s])
                return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1)))
            queries=frame['queries']
            candidates=[]
            for start in range(0,len(queries),4):
                batch=queries[start:start+4]
                evaluator.byframe={identity:[dict(q,slot=j) for j,q in enumerate(batch)]}
                arr=torch.stack(list(workers.map(pixels,batch))).cuda()
                evaluator.cache=sam.image_encoder(sam.preprocess(arr)).cpu().numpy()
                candidates.extend(evaluator.predict_frame(sam,dict(id=identity,old=old,support=support)))
            lab=assemble(old,support,candidates,'ROI_PRIOR',variant=='ROI_PRIOR_GUARDED')
            np.savez_compressed(path,**{variant:lab})
            with np.load(path) as file:
                assert np.array_equal(lab,file[variant])
            rec=dict(frame_id=identity,variant=variant,path=str(path),sha256=sha(path),weight_sha256=chosen['checkpoint_sha256'])
            save(receipt,rec)
            outputs.append(rec)
            save(OUTPUT/'STATUS.json',dict(event='TARGET_NATIVE_REFINEMENT',windows=len(outputs),total=72))
    manifest=OUTPUT/'PREDICTIONS_FROZEN_BEFORE_MAP_JOIN.json'
    save(manifest,dict(rows=outputs,GT_in_prediction_inputs=False,selected_source_weight_sha256=chosen['checkpoint_sha256']))
    subprocess.run([sys.executable,str(R/'code/evaluate_xian_map_windows.py'),'--candidate-manifest',str(manifest)],check=True)
    save(OUTPUT/'PILOT_RECEIPT.json',dict(status='COMPLETE_TARGET_TUNE_72_REFERENCE_COMPARISON',windows=72,
        map_summary_sha256=sha(R/'output/xian_map_reference/SELECTED_SUMMARY.json'),protected_reads=0,production_changed=False))
if __name__=='__main__':
    try:
        launch()
    except Exception:
        save(OUTPUT/'FAILURE_RECEIPT.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise

