"""Image-only predicted prompts and exact frozen SAM FP32 image embeddings.

1004 source TRAIN images; Chengdu32 is held for decoder selection. No native
GT polygons enter this process. A subsequent process joins supervision by SHA.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,hashlib,sys,traceback,gc,fcntl
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
from PIL import Image
from scipy import ndimage
from sam_native_pilot import point_box
R=Path('/root/xian20261111_r15');D=R/'external/sam';O=R/'data/native_roof_decoder';O.mkdir(parents=True,exist_ok=True);P=Path('/root/xian20261111_r8');N=Path('/root/xian20261111_r7')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def status(event,**kw):save(O/'STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def load_image_prompts(f):
 assert f['role']=='SOURCE_TRAIN';assert sha(f['image_path'])==f['image_sha256'] and sha(f['query_cache_path'])==f['query_cache_sha256'];dest=O/'frames'/f['id'];dest.mkdir(parents=True,exist_ok=True);rgb=np.asarray(Image.open(f['image_path']).convert('RGB'));assert rgb.shape==(1024,1024,3);z=np.load(f['query_cache_path']);old=z['instances'];queries=[]
 for ident,sl in enumerate(ndimage.find_objects(old),1):
  if sl is None:continue
  if (old[sl]==ident).sum()<32:continue
  mask=old==ident;point,box=point_box(mask);queries.append(dict(query_id=ident,point_native_pixel_index=point,box_native_inclusive=box,bounds_native=[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop],raw_area_native_pixels=int(mask.sum()),border=sl[0].start==0 or sl[1].start==0 or sl[0].stop==1024 or sl[1].stop==1024,GT_in_prompt=False))
 save(dest/'IMAGE_ONLY_PROMPTS.json',dict(frame_id=f['id'],role=f['decoder_role'],queries=queries,model_parent_cache_sha256=f['query_cache_sha256'],GT_labels_or_geometry_input=False));return f,rgb
def write_embedding(f,array):
 d=O/'frames'/f['id'];p=d/'SAM_IMAGE_EMBEDDING_FP32.npz';tmp=p.with_suffix('.pending')
 assert array.dtype==np.float32 and array.shape==(1,256,64,64) and np.isfinite(array).all()
 with tmp.open('wb') as stream:np.savez_compressed(stream,image_embedding=array)
 tmp.replace(p);record=dict(frame_id=f['id'],role=f['decoder_role'],image_path=f['image_path'],image_sha256=f['image_sha256'],cache_path=f['query_cache_path'],cache_sha256=f['query_cache_sha256'],prompts_path=str(d/'IMAGE_ONLY_PROMPTS.json'),prompts_sha256=sha(d/'IMAGE_ONLY_PROMPTS.json'),embedding_path=str(p),embedding_sha256=sha(p),native_size=[1024,1024],encoder_dtype='FP32_NO_TF32',GT_used_for_generation=False);save(d/'INPUTS_FROZEN_BEFORE_GT_JOIN.json',record);return record
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;records=[json.loads(l) for l in (P/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l];frames=[r for r in records if r['role']=='SOURCE_TRAIN'];assert len(frames)==1004;joint=json.loads((N/'JOINT_NATIVE_SOURCE_TRAIN_READY.json').read_text());assert joint['protected_reads']==0;metadata={r['image']:r for r in joint['records'] if str(r['city']).upper()!='WHU'};assert set(metadata)=={r['image_path'] for r in frames};inner=[r for r in frames if str(metadata[r['image_path']]['city']).lower()=='chengdu'];assert len(inner)==32
 for f in frames:f['decoder_role']='INNER' if f in inner else 'FIT'
 role_rows=[dict(id=f['id'],decoder_role=f['decoder_role'],image_path=f['image_path'],image_sha256=f['image_sha256'],query_cache_path=f['query_cache_path'],query_cache_sha256=f['query_cache_sha256'],label_path=metadata[f['image_path']]['label'],label_sha256=metadata[f['image_path']]['label_sha256'],city=metadata[f['image_path']]['city']) for f in frames];save(O/'ROLES_FROZEN_BEFORE_PIXEL_READ.json',dict(task='NATIVE_SOURCE_ROOF_DECODER_DATA',FIT_images=972,INNER_images=32,TUNE='12_ORIGINAL_SOURCE_FRAMES_SEPARATE_REUSED_DEVELOPMENT',rows=role_rows,protected_reads=0,WHU_region_only_excluded=True,XIAN_map_as_native_roof_GT=False,GT_polygons_or_pixels_read_for_prompt_generation=False));save(O/'PROTOCOL_FROZEN.json',dict(task='SOURCE_IMAGE_ONLY_SAM_ROOF_TRAINING_INPUTS',encoder='OFFICIAL_SAM_VITB_FROZEN_FP32_NO_TF32',image_input='ORIGINAL_NATIVE1024_RGB_SAME_PREDICTOR_PREPROCESS_AS_ZERO_UPDATE_BASELINE',prompts='FROZEN_MODEL_QUERY_MASK_INTERIOR_MAX_AND_INCLUSIVE_BBOX_PLUS16',mask_prior='RECONSTRUCT_FROM_MODEL_QUERY_ONLY',GT_join='SEPARATE_PROCESS_ONLY_AFTER_PER_FRAME_INPUT_RECEIPT_SHA',no_GT_query='NOT_SUPERVISED;RETAINED_IN_DATA_AND_INFERENCE',role_filter_before_pixels=True,protected_reads=0,optimizer_updates=0));sys.path.insert(0,str(D/'source'));from segment_anything import sam_model_registry,SamPredictor
 asset=json.loads((D/'LOCAL_SOURCE_RECEIPT.json').read_text());weights=D/'sam_vit_b_01ec64.pth';assert sha(weights)==asset['weights']['sha256'];lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');status('WAIT_EXCLUSIVE_GPU_SOURCE_ENCODER');fcntl.flock(lock,fcntl.LOCK_EX);receipts=[]
 try:
  model=sam_model_registry['vit_b']();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.cuda().eval()
  for p in model.parameters():p.requires_grad_(False)
  predictor=SamPredictor(model);pending=[];todo=[]
  for f in frames:
   p=O/'frames'/f['id']/'INPUTS_FROZEN_BEFORE_GT_JOIN.json'
   if p.exists():
    r=json.loads(p.read_text());assert sha(r['prompts_path'])==r['prompts_sha256'] and sha(r['embedding_path'])==r['embedding_sha256'];receipts.append(r)
   else:todo.append(f)
  with ThreadPoolExecutor(max_workers=4) as readers,ThreadPoolExecutor(max_workers=4) as writers:
   for i,(f,rgb) in enumerate(readers.map(load_image_prompts,todo)):
    with torch.no_grad():predictor.set_image(rgb);embedding=predictor.get_image_embedding().detach().cpu().numpy().copy()
    pending.append(writers.submit(write_embedding,f,embedding))
    if len(pending)>=8:receipts.append(pending.pop(0).result())
    if i%10==0:status('ENCODE_NATIVE_SOURCE_FP32',encoded=i+1,total_remaining=len(todo),already_cached=len(frames)-len(todo))
   receipts.extend(p.result() for p in pending)
  del predictor,model;gc.collect();torch.cuda.empty_cache();assert len(receipts)==1004;save(O/'ALL_IMAGE_ONLY_INPUTS_FROZEN.json',dict(status='COMPLETE_SAM_NATIVE_SOURCE_TRAIN_INPUTS',records=receipts,images=len(receipts),GT_used_in_generation=False,protected_reads=0,optimizer_updates=0));status('NATIVE_SOURCE_ENCODER_INPUTS_COMPLETE',images=1004)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:main()
 except Exception as e:save(O/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
