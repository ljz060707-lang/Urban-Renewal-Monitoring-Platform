# -*- coding: utf-8 -*-
"""Model-only native ROI definition, separate supervised sampling, exact FP32 cache."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,hashlib,sys,fcntl,shutil,traceback,gc
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
from PIL import Image
R=Path('/root/xian20261111_r16_roi_roof');O=R/'output';D=R/'data';PRE=Path('/root/xian20261111_r15');A=PRE/'external/sam';O.mkdir(parents=True,exist_ok=True);D.mkdir(exist_ok=True);torch.set_num_threads(4)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_text(json.dumps(data,indent=2));t.replace(p)
def status(event,**kw):save(O/'STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def crop_bounds(q):
 x0,y0,x1,y1=q['box_native_inclusive'];s=min(1024,max(96,int(np.ceil(2*max(x1-x0+1,y1-y0+1)))));cx=(x0+x1)/2;cy=(y0+y1)/2;left=int(np.clip(np.floor(cx-(s-1)/2),0,1024-s));top=int(np.clip(np.floor(cy-(s-1)/2),0,1024-s));return [left,top,s]
@lru_cache(maxsize=16)
def image(path,digest):
 assert sha(path)==digest;return np.asarray(Image.open(path).convert('RGB'))
def main():
 save(O/'PROTOCOL_FROZEN.json',dict(task='TRAIN_TRUE_NATIVE_ROOF_ON_MODEL_PROPOSED_HIGH_DETAIL_ROIS',decision_evidence_sha256=sha(PRE/'output/roi_detail_pilot/EVALUATION_RECEIPT.json'),decision='SOURCE_INNER8_ROI_PRIOR_IMPROVED_VS_WHOLE_IMAGE_PRIOR;NOT_PRODUCTION_PROMOTION',model='SAME_OFFICIAL_GENERIC_SAM_VITB_ENCODER_FROZEN_FP32',weights_sha256=sha(A/'sam_vit_b_01ec64.pth'),native_ROI='SAME_MODEL_ONLY_2X_BOX_SQUARE_MIN96_MAX1024_CLAMP_INSIDE_IMAGE',GT_in_crop_or_prompt=False,feature_resize='OFFICIAL_LONG_SIDE1024_FEATURE_RESAMPLING_NOT_NEW_SENSOR_DETAIL',FIT='BONAI_BJ684_HARBIN288;UP_TO16_DISTINCT_KNOWN_GT_QUERY_TARGETS_PER_FRAME',INNER='ALL_MODEL_QUERIES_ON32_CHENGDU_FRAMES;NOT_BLIND_END_TO_END_BECAUSE_QUERY_GENERATOR_PREVIOUSLY_EXPOSED',source_TUNE='NOT_READ_AT_PREPARATION;REUSED12_ONLY_AFTER_SELECTION_FREEZE',supervised_sampling='AFTER_ALL_IMAGE_ONLY_CROP_DEFINITIONS_FREEZE;NO_GT_QUERIES_EXCLUDED_ONLY_FROM_LOSS',sampling_cap_is_training_balancing_not_candidate_recall_filter=True,unknown_pixel_and_query_direct_gradient='MUST_PASS_BEFORE_TRAINING',cache='FP32_PER_ROI_IMAGE_EMBEDDINGS_VOLATILE_DERIVABLE_DEV_SHM_NOT_IMAGE_OR_LABEL_ARCHIVE',protected_reads=0,optimizer_updates_at_preparation=0,production_changed=False))
 source=PRE/'data/native_roof_decoder/ALL_IMAGE_ONLY_INPUTS_FROZEN.json';records=json.loads(source.read_text())['records'];assert len(records)==1004;byframe={r['frame_id']:r for r in records};all_crops={};bindings=[]
 for rec in records:
  assert rec['role'] in ['FIT','INNER'];assert sha(rec['prompts_path'])==rec['prompts_sha256'];prompts=json.loads(Path(rec['prompts_path']).read_text())['queries'];crops=[]
  for q in prompts:
   x,y,s=crop_bounds(q);p=np.array(q['point_native_pixel_index']);assert ((p-[x,y])>=0).all() and ((p-[x,y])<s).all();crops.append(dict(query_id=q['query_id'],crop_native_xy_size=[x,y,s],point_native_pixel_index=q['point_native_pixel_index'],box_native_inclusive=q['box_native_inclusive'],GT_in_generation=False))
  all_crops[rec['frame_id']]={q['query_id']:q for q in crops};p=D/'frames'/rec['frame_id']/'ALL_MODEL_CROPS_FROZEN_BEFORE_NEW_GT_JOIN.json';save(p,dict(source=rec,model_only_crops=crops));bindings.append(dict(frame_id=rec['frame_id'],role=rec['role'],path=str(p),sha256=sha(p),all_model_queries=len(crops)))
 save(D/'ALL_MODEL_CROPS_FROZEN_BEFORE_NEW_GT_JOIN.json',dict(records=bindings,source_metadata_sha256=sha(source),no_new_GT_sample_join_performed_yet=True,protected_reads=0))
 # Only now link prior separate native-roof supervision for TRAIN sampling.
 native_path=PRE/'data/native_roof_decoder/DATASET_READY.json';native=json.loads(native_path.read_text());assert native['protected_reads']==0;eligible=defaultdict(lambda:defaultdict(list))
 for row in native['samples']:
  if row['role']=='FIT':eligible[row['frame_id']][row['annotation_id']].append(row)
 slots=[];supervision={};selected=[]
 for rec in records:
  frame=rec['frame_id'];queries=all_crops[frame]
  if rec['role']=='FIT':
   groups=eligible[frame];ann_ids=sorted(groups,key=lambda a:hashlib.sha256(f'{frame}:{a}:20261121'.encode()).hexdigest())[:16];chosen=[min(groups[a],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()) for a in ann_ids];chosen_by_q={r['query_id']:r for r in chosen};qids=sorted(chosen_by_q)
  else:chosen_by_q={};qids=sorted(queries)
  for qid in qids:
   slot=len(slots);slots.append(dict(slot=slot,frame_id=frame,role=rec['role'],image_path=rec['image_path'],image_sha256=rec['image_sha256'],cache_path=rec['cache_path'],cache_sha256=rec['cache_sha256'],**queries[qid]))
   if qid in chosen_by_q:supervision[str(slot)]=chosen_by_q[qid];selected.append(slot)
 cache_root=Path('/dev/shm/xian_r16_roi_roof');cache_root.mkdir(exist_ok=True);cache_path=cache_root/'IMAGE_EMBEDDINGS_FP32.npy';required=len(slots)*256*64*64*4;assert required<85*1024**3
 if not cache_path.exists():assert shutil.disk_usage(cache_root).free>required+2*1024**3;array=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.float32,shape=(len(slots),256,64,64))
 else:array=np.load(cache_path,mmap_mode='r+');assert array.shape==(len(slots),256,64,64) and array.dtype==np.float32
 save(D/'SLOT_MANIFEST_FROZEN.json',dict(rows=slots,cache_path=str(cache_path),cache_bytes=required,crop_definition_receipt_sha256=sha(D/'ALL_MODEL_CROPS_FROZEN_BEFORE_NEW_GT_JOIN.json'),model_weights_sha256=sha(A/'sam_vit_b_01ec64.pth'),cache_precision='FP32_UNQUANTIZED'))
 save(D/'SUPERVISION_SEPARATE.json',dict(selected_FIT_slots=selected,labels=supervision,source_dataset_path=str(native_path),source_dataset_sha256=sha(native_path),COCO_source=native['COCO_source'],COCO_source_sha256=native['COCO_source_sha256'],all_eligible_FIT_samples=sum(r['role']=='FIT' for r in native['samples']),sample_cap_per_frame=16,original_no_GT_queries_preserved_in_all_model_crops=True,GT_in_crop_or_prompt=False,protected_reads=0));status('PREPARE_FROZEN_FP32_NATIVE_ROI_CACHE',slots=len(slots),FIT_samples=len(selected),INNER_queries=len(slots)-len(selected),cache_GiB=required/1024**3)
 sys.path.insert(0,str(A/'source'));from segment_anything import sam_model_registry;from segment_anything.utils.transforms import ResizeLongestSide
 transform=ResizeLongestSide(1024)
 def prepare(row):
  x,y,s=row['crop_native_xy_size'];im=image(row['image_path'],row['image_sha256']);crop=im[y:y+s,x:x+s];out=transform.apply_image(crop);assert out.shape==(1024,1024,3);return torch.from_numpy(np.ascontiguousarray(out.transpose(2,0,1)))
 lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');status('WAIT_EXCLUSIVE_GPU_NATIVE_ROI_ENCODING',slots=len(slots));fcntl.flock(lock,fcntl.LOCK_EX);frame_receipts=[]
 try:
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(A/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True),strict=True);sam.cuda().eval()
  for p in sam.parameters():p.requires_grad_(False)
  pilot=json.loads((PRE/'output/roi_detail_pilot/FP32_ROI_ENCODER_THROUGHPUT.json').read_text());batch=pilot['selection']['batch'];assert batch in [1,2,4];save(O/'THROUGHPUT_CONFIGURATION.json',dict(parent_benchmark_sha256=sha(PRE/'output/roi_detail_pilot/FP32_ROI_ENCODER_THROUGHPUT.json'),batch=batch,CPU_RGB_resize_threads=8,TF32=False,precision='FP32',optimizer_updates=0))
  # Real batched-vs-single native ROIs, same official preprocess, no optimizer.
  with torch.no_grad():
   pixels=torch.stack([prepare(r) for r in slots[:4]]).cuda();batched=sam.image_encoder(sam.preprocess(pixels));single=torch.cat([sam.image_encoder(sam.preprocess(pixels[i:i+1])) for i in range(4)]);delta=float((batched-single).abs().max());assert torch.allclose(batched,single,atol=2e-4,rtol=1e-4);save(O/'ENCODER_BATCH_FUNCTIONAL_RECEIPT.json',dict(status='PASS_REAL_ROI_BATCH_SINGLE_FP32_ALLCLOSE_NOT_CLAIMED_BITEXACT',ROIs=4,max_abs_error=delta,atol=2e-4,rtol=1e-4,optimizer_updates=0));del pixels,batched,single
  grouped=defaultdict(list)
  for row in slots:grouped[row['frame_id']].append(row)
  with ThreadPoolExecutor(max_workers=8) as pool,torch.no_grad():
   completed=0
   for k,(frame,rows) in enumerate(grouped.items()):
    recpath=D/'frames'/frame/'ROI_FEATURE_RECEIPT.json';ii=np.array([r['slot'] for r in rows],np.int64)
    if recpath.exists():
     receipt=json.loads(recpath.read_text());assert hashlib.sha256(array[ii].tobytes()).hexdigest()==receipt['embedding_values_sha256'];frame_receipts.append(receipt);completed+=len(rows);continue
    inputs=iter(pool.map(prepare,rows));t0=time.time()
    for begin in range(0,len(rows),batch):
     subset=rows[begin:begin+batch];pixels=torch.stack([next(inputs) for _ in subset]).cuda(non_blocking=True);emb=sam.image_encoder(sam.preprocess(pixels));assert torch.isfinite(emb).all();values=emb.cpu().numpy();jj=[r['slot'] for r in subset];array[jj]=values;assert np.array_equal(array[jj],values)
    receipt=dict(frame_id=frame,role=rows[0]['role'],slots=ii.tolist(),embedding_values_sha256=hashlib.sha256(array[ii].tobytes()).hexdigest(),seconds=time.time()-t0,image_sha256=rows[0]['image_sha256'],crop_manifest_sha256=sha(D/'frames'/frame/'ALL_MODEL_CROPS_FROZEN_BEFORE_NEW_GT_JOIN.json'),precision='FP32',GT_in_feature=False);save(recpath,receipt);frame_receipts.append(receipt);completed+=len(rows)
    if k%10==0 or k+1==len(grouped):status('NATIVE_ROI_FRAME_CACHE_COMPLETE',frames=k+1,total_frames=len(grouped),cached_ROIs=completed,total_ROIs=len(slots),actual_optimizer_updates=0)
  array.flush();save(D/'ALL_ROI_FEATURES_FROZEN.json',dict(records=frame_receipts,slot_manifest_sha256=sha(D/'SLOT_MANIFEST_FROZEN.json'),cache_path=str(cache_path),cache_shape=list(array.shape),precision='FP32',GT_pixels_in_features=False,optimizer_updates=0));save(D/'DATASET_READY.json',dict(status='COMPLETE_MODEL_PROPOSED_NATIVE_ROI_FEATURE_DATA',slots=len(slots),FIT_slots=selected,INNER_slots=[r['slot'] for r in slots if r['role']=='INNER'],slot_manifest_path=str(D/'SLOT_MANIFEST_FROZEN.json'),slot_manifest_sha256=sha(D/'SLOT_MANIFEST_FROZEN.json'),supervision_path=str(D/'SUPERVISION_SEPARATE.json'),supervision_sha256=sha(D/'SUPERVISION_SEPARATE.json'),feature_freeze_sha256=sha(D/'ALL_ROI_FEATURES_FROZEN.json'),cache_path=str(cache_path),protected_reads=0));status('NATIVE_ROI_FEATURE_PREPARATION_COMPLETE',cached_ROIs=len(slots),FIT_samples=len(selected),optimizer_updates=0)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:main()
 except Exception as err:save(O/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(err),traceback=traceback.format_exc()));raise
