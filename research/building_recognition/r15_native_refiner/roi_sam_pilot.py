# -*- coding: utf-8 -*-
"""Source-INNER only: model-proposed ROI detail vs original whole-image SAM."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,sys,hashlib,fcntl,traceback,gc
import numpy as np,torch,cv2
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mu
from roof_decoder_model import forward_roof
from sam_native_pilot import assemble
R=Path('/root/xian20261111_r15');D=R/'data/native_roof_decoder';O=R/'output/roi_detail_pilot';A=R/'external/sam';O.mkdir(exist_ok=True);torch.set_num_threads(4)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_text(json.dumps(value,indent=2));t.replace(p)
def status(event,**kw):save('STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def crop_bounds(q):
 x0,y0,x1,y1=q['box_native_inclusive'];s=min(1024,max(96,int(np.ceil(2*max(x1-x0+1,y1-y0+1)))));cx=(x0+x1)/2;cy=(y0+y1)/2;left=int(np.clip(np.floor(cx-(s-1)/2),0,1024-s));top=int(np.clip(np.floor(cy-(s-1)/2),0,1024-s));return left,top,s
def encode_rle(m):
 r=mu.encode(np.asfortranarray(m.astype('uint8')));r['counts']=r['counts'].decode('ascii');return r
@torch.no_grad()
def full_candidates(sam,embedding,old,prompts,support):
 result=[]
 for start in range(0,len(prompts),16):
  qs=prompts[start:start+16];n=len(qs);e=torch.from_numpy(embedding).cuda().expand(n,-1,-1,-1);p=torch.tensor([q['point_native_pixel_index'] for q in qs],device='cuda',dtype=torch.float32);box=torch.tensor([q['box_native_inclusive'] for q in qs],device='cuda',dtype=torch.float32);pr=np.stack([-8+16*cv2.resize((old==q['query_id']).astype('float32'),(256,256),interpolation=cv2.INTER_AREA) for q in qs]);pr=torch.from_numpy(pr[:,None]).cuda();local=[dict(query_id=q['query_id'],candidates=[]) for q in qs]
  for prior in [False,True]:
   low,quality=forward_roof(sam,e,p,box,pr,torch.full((n,),prior,device='cuda',dtype=torch.bool));m=(F.interpolate(low.float(),(1024,1024),mode='bilinear',align_corners=False)>0)[:,0].cpu().numpy()
   for j,mask in enumerate(m):local[j]['candidates'].append(dict(mode='FULL_PRIOR' if prior else 'FULL_POINT_BOX',predicted_mask_quality=float(quality[j]),mask=encode_rle(mask&support)))
  result.extend(local)
 return result
def main():
 cv2.setNumThreads(0);records=json.loads((D/'ALL_IMAGE_ONLY_INPUTS_FROZEN.json').read_text())['records'];inner=sorted([r for r in records if r['role']=='INNER'],key=lambda r:hashlib.sha256(r['frame_id'].encode()).hexdigest())[:8];fit=sorted([r for r in records if r['role']=='FIT'],key=lambda r:r['frame_id'])[0]
 save('ROLES_FROZEN_BEFORE_RGB_READ.json',dict(INNER=inner,throughput_FIT=fit,source_TUNE_reads=0,protected_reads=0));save('PROTOCOL_FROZEN.json',dict(task='MODEL_QUERY_ROI_NATIVE_DETAIL_SAM_VS_WHOLE_IMAGE_MATCHED_GENERIC_PARENT',cohort='8_HASH_SELECTED_SOURCE_INNER_FRAMES_NO_SOURCE_TUNE',crop='MODEL_BOX_CENTER_SQUARE_2X_MAX_SIZE_MIN96_MAX1024_SHIFT_INSIDE_NATIVE_IMAGE',resize='OFFICIAL_SAM_LONGEST_SIDE1024_FEATURE_RESAMPLING_ONLY_NO_NEW_SENSOR_DETAIL',modes=['FULL_POINT_BOX','FULL_PRIOR','ROI_POINT_BOX','ROI_PRIOR'],same_parent_weight_sha256=sha(A/'sam_vit_b_01ec64.pth'),native_mask_output='BILINEAR_LOGITS_BACK_TO_ORIGINAL_ROI_NATIVE_PIXELS',no_GT_in_prompt_or_crop=True,no_GT_queries_retained=True,loss_updates=0,GT_reads='AFTER_ALL_CANDIDATE_AND_MAP_SHA_FREEZE',selection='FIXED_MATCHED_MODES_NOT_TUNE_THRESHOLD_SEARCH',production_changed=False))
 # Functional coordinate checks use actual model queries, no label pixels.
 for r in inner:
  assert sha(r['prompts_path'])==r['prompts_sha256'];assert sha(r['cache_path'])==r['cache_sha256'];old=np.load(r['cache_path'])['instances'];qs=json.loads(Path(r['prompts_path']).read_text())['queries']
  for q in qs:
   x,y,s=crop_bounds(q);p=np.array(q['point_native_pixel_index']);assert ((p-[x,y])>=0).all() and ((p-[x,y])<s).all();raw=old==q['query_id'];assert int(raw[y:y+s,x:x+s].sum())==int(raw.sum());scale=1024/s;assert np.max(np.abs(((p-[x,y])*scale)/scale+[x,y]-p))<1e-6
 save('MODEL_CROP_COORDINATE_FUNCTIONAL_RECEIPT.json',dict(status='PASS_ALL_RAW_QUERY_PIXELS_RETAINED_IN_CROP_AND_FLOAT_COORDINATES_ROUNDTRIP',frames=8,GT_pixel_reads=0,optimizer_updates=0))
 lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');status('WAIT_EXCLUSIVE_GPU_AFTER_DECODER_TRAIN');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(A/'source'));from segment_anything import sam_model_registry;from segment_anything.utils.transforms import ResizeLongestSide
  sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(A/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True),strict=True);sam.cuda().eval();transform=ResizeLongestSide(1024)
  assert sha(fit['image_path'])==fit['image_sha256'];im=np.asarray(Image.open(fit['image_path']).convert('RGB'));qs=json.loads(Path(fit['prompts_path']).read_text())['queries'];x,y,s=crop_bounds(qs[0]);crop=im[y:y+s,x:x+s];tt=torch.as_tensor(transform.apply_image(crop),device='cuda').permute(2,0,1).contiguous();inp=sam.preprocess(tt)[None];bench=[]
  for b in [1,2,4]:
   try:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.time()
    with torch.no_grad():out=sam.image_encoder(inp.expand(b,-1,-1,-1));torch.cuda.synchronize()
    bench.append(dict(batch=b,seconds=time.time()-start,images_per_second=b/(time.time()-start),peak_GiB=torch.cuda.max_memory_allocated()/1024**3));del out
   except torch.cuda.OutOfMemoryError as err:
    bench.append(dict(batch=b,error=repr(err)));gc.collect();torch.cuda.empty_cache()
  choice=max([b for b in bench if 'error' not in b and b['peak_GiB']<16],key=lambda b:b['images_per_second']);batch=choice['batch'];save('FP32_ROI_ENCODER_THROUGHPUT.json',dict(results=bench,selection=choice,optimizer_updates=0));all_frozen=[]
  for i,r in enumerate(inner):
   folder=O/'frames'/r['frame_id'];folder.mkdir(parents=True,exist_ok=True);frozen=folder/'PREDICTIONS_FROZEN_BEFORE_GT.json'
   if frozen.exists():
    row=json.loads(frozen.read_text());assert sha(folder/'CANDIDATES.json')==row['candidate_sha256'] and sha(folder/'INSTANCE_MAPS.npz')==row['maps_sha256'];all_frozen.append(row);continue
   assert sha(r['image_path'])==r['image_sha256'];image=np.asarray(Image.open(r['image_path']).convert('RGB'));cache=np.load(r['cache_path']);old=cache['instances'];support=cache['support'].astype(bool);prompts=json.loads(Path(r['prompts_path']).read_text())['queries'];assert sha(r['embedding_path'])==r['embedding_sha256'];embedding=np.load(r['embedding_path'])['image_embedding'];candidates=full_candidates(sam,embedding,old,prompts,support)
   for begin in range(0,len(prompts),batch):
    qq=prompts[begin:begin+batch];bounds=[crop_bounds(q) for q in qq];images=[];points=[];boxes=[];priors=[]
    for q,(x,y,s) in zip(qq,bounds):
     im=image[y:y+s,x:x+s];images.append(sam.preprocess(torch.as_tensor(transform.apply_image(im),device='cuda').permute(2,0,1).contiguous()));points.append(transform.apply_coords(np.array(q['point_native_pixel_index'],np.float32)[None]-[x,y],(s,s))[0]);box=np.array(q['box_native_inclusive'],np.float32)-[x,y,x,y];boxes.append(transform.apply_boxes(box[None],(s,s))[0]);priors.append(-8+16*cv2.resize((old[y:y+s,x:x+s]==q['query_id']).astype('float32'),(256,256),interpolation=cv2.INTER_AREA))
    with torch.no_grad():
     e=sam.image_encoder(torch.stack(images));pt=torch.tensor(np.array(points),device='cuda',dtype=torch.float32);bx=torch.tensor(np.array(boxes),device='cuda',dtype=torch.float32);pr=torch.from_numpy(np.stack(priors)[:,None]).cuda()
     for prior in [False,True]:
      low,quality=forward_roof(sam,e,pt,bx,pr,torch.full((len(qq),),prior,device='cuda',dtype=torch.bool))
      for j,(x,y,s) in enumerate(bounds):
       native=(F.interpolate(low[j:j+1].float(),(s,s),mode='bilinear',align_corners=False)>0)[0,0].cpu().numpy();mask=np.zeros((1024,1024),bool);mask[y:y+s,x:x+s]=native;mask&=support;candidates[begin+j]['candidates'].append(dict(mode='ROI_PRIOR' if prior else 'ROI_POINT_BOX',predicted_mask_quality=float(quality[j]),mask=encode_rle(mask),crop_native_xy_size=[x,y,s],GT_in_crop=False))
    del e,images
   maps={}
   for mode in ['FULL_POINT_BOX','FULL_PRIOR','ROI_POINT_BOX','ROI_PRIOR']:
    for guarded in [False,True]:maps[mode+('_GUARDED' if guarded else '_UNGUARDED')]=assemble(old,support,candidates,mode,guarded)
   cp=folder/'CANDIDATES.json';cp.write_text(json.dumps(dict(frame=r['frame_id'],queries=candidates,GT_in_prediction=False)));mp=folder/'INSTANCE_MAPS.npz';np.savez_compressed(mp,**maps);row=dict(frame_id=r['frame_id'],candidate_sha256=sha(cp),maps_sha256=sha(mp),query_count=len(prompts),input_image_sha256=r['image_sha256'],image_only_prompt_sha256=r['prompts_sha256']);frozen.write_text(json.dumps(row,indent=2));all_frozen.append(row);status('ROI_SOURCE_INNER_FRAME_FROZEN',completed=i+1,total=8,queries=len(prompts))
  save('ALL_PREDICTIONS_FROZEN_BEFORE_GT_JOIN.json',dict(frames=all_frozen,no_GT_read_by_pipeline_so_far=True,optimizer_updates=0))
  del sam;gc.collect();torch.cuda.empty_cache()
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
 evaluator=RoofEvaluator('INNER');frames=[f for f in evaluator.frames if f['id'] in {r['frame_id'] for r in inner}];assert len(frames)==8;raw=aggregate([metrics(f,f['old']) for f in frames]);results={}
 for f in frames:
  z=np.load(O/'frames'/f['id']/'INSTANCE_MAPS.npz')
  for mode in z.files:results.setdefault(mode,[]).append(metrics(f,z[mode]))
 report={k:aggregate(v) for k,v in results.items()};qualified=[k for k,v in report.items() if passes(v,raw)];save('EVALUATION_RECEIPT.json',dict(status='COMPLETE_SOURCE_INNER_NATIVE_ROI_DETAIL_EXPERIMENT',frames=8,raw_baseline=raw,metrics=report,qualified_inner_variants=qualified,source_TUNE_reads=0,protected_reads=0,project_optimizer_updates=0,source_development_not_XIAN_accuracy=True,production_changed=False));status('ROI_DETAIL_SOURCE_INNER_COMPLETE',qualified=qualified,metrics={n:dict(F1_50=m['F1_50'],pixel_IoU=m['pixel_IoU'],pixel_recall=m['pixel_recall']) for n,m in report.items()})
if __name__=='__main__':
 try:main()
 except Exception as err:save('FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(err),traceback=traceback.format_exc()));raise
