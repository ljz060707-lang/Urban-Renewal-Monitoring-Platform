"""Official generic SAM with actual model prompts, then identical native source metrics.

No GT boxes/points/masks are supplied to SAM. GT only after all predictions freeze.
This is the zero-update source baseline, not an independent XiAn roof evaluation.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,sys,hashlib,traceback,ast,gc,fcntl
import numpy as np,torch,cv2
from PIL import Image
from scipy import ndimage
from pycocotools import mask as mu
R=Path('/root/xian20261111_r15');D=R/'external/sam';O=R/'output/source_sam_pilot';O.mkdir(parents=True,exist_ok=True);P=Path('/root/xian20261111_r8');S=Path('/root/xian20261111_r10')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def status(event,**kw):save(O/'STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def encode(mask):
 r=mu.encode(np.asfortranarray(mask.astype('uint8')));r['counts']=r['counts'].decode('ascii');return r
def point_box(mask):
 yy,xx=np.where(mask);x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1;dt=ndimage.distance_transform_edt(np.pad(mask[y0:y1,x0:x1],1))[1:-1,1:-1];y,x=np.unravel_index(dt.argmax(),dt.shape);point=[float(x+x0),float(y+y0)];box=[max(0,x0-16),max(0,y0-16),min(1023,x1-1+16),min(1023,y1-1+16)];return point,box
def assemble(old,support,queries,mode,guarded):
 proposals=[]
 for q in queries:
  identity=q['query_id'];raw=old==identity;candidates=[c for c in q['candidates'] if c['mode']==mode]
  if not candidates:mask=raw;quality=-1e6
  else:
   chosen=max(candidates,key=lambda c:c['predicted_mask_quality']);mask=mu.decode(chosen['mask']).astype(bool)&support;quality=chosen['predicted_mask_quality'];mask&=(old==0)|(old==identity)
   if mask.sum()<32 or guarded and ((mask&raw).sum()<.75*raw.sum() or mask.sum()>1.5*raw.sum()):mask=raw
  proposals.append((quality,identity,mask,raw))
 out=np.zeros_like(old)
 for quality,identity,mask,raw in sorted(proposals,key=lambda r:(-r[0],r[1])):
  take=mask&(out==0)
  if take.sum()<32:take=raw
  out[take]=identity
 # Small original components and any lost ID retain raw, never silently disappear.
 for identity in np.unique(old):
  if identity and not (out==identity).any():out[old==identity]=identity
 assert set(np.unique(out))==set(np.unique(old));assert not (out[~support]>0).any();return out

def main():
 torch.set_num_threads(4);cv2.setNumThreads(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert json.loads((D/'CLOUD_CPU_ADMISSION.json').read_text())['status'].startswith('PASS');asset=json.loads((D/'LOCAL_SOURCE_RECEIPT.json').read_text());weights=D/'sam_vit_b_01ec64.pth';assert sha(weights)==asset['weights']['sha256'];records=[json.loads(l) for l in (P/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l];frames=[r for r in records if r['role']=='SOURCE_TUNE'];assert len(frames)==12
 save(O/'PROTOCOL_FROZEN_BEFORE_GT.json',dict(task='TRUE_NATIVE_ROOF_QUERY_SAM_BASELINE',source_roles=['SOURCE_TUNE_12_ADMITTED_NONPROTECTED_NATIVE_BONAI_FRAMES'],frame_ids=[r['id'] for r in frames],image='ORIGINAL_NATIVE1024_RGB_SHA_MATCH',model='OFFICIAL_GENERIC_SAM_VITB',weights_sha256=asset['weights']['sha256'],code_commit=asset['commit'],precision='FP32_NO_TF32_NO_AUTOCAST',prompt_source='SHA_FROZEN_R7_NATIVE_ROOF_QUERY_MASKS_ONLY',GT_boxes_points_masks_in_prompt=False,point='MODEL_MASK_INTERIOR_DISTANCE_MAX_INTEGER_PIXEL_INDEX_SAM_ENCODER_ADDS_0.5',box='PREDICTED_MASK_BOUNDING_BOX_NATIVE_INCLUSIVE_MAX_PLUS16PX_CLIPPED0_1023',mask_prompt='MODEL_QUERY_256_AREA_RESIZE_LOGIT_MINUS8_PLUS16_FRACTION',modes=['BOX','POINT_BOX','POINT_BOX_PRIOR'],masks_per_query='1_BOX_SINGLE_PLUS4_POINT_BOX_PLUS4_POINT_BOX_PRIOR',choose='ARGMAX_SAM_PREDICTED_QUALITY_NEVER_GT;NOT_CALIBRATED_ROOF_CONFIDENCE',guarded_comparison='RETAIN_GE75PCT_RAW_PIXELS_AND_AREA_LE150PCT_ELSE_RAW_FALLBACK',all_original_ids_retained=True,other_raw_query_pixels='NEVER_OVERWRITTEN',source_support='NO_ADDITIONS_OUTSIDE_FROZEN_SUPPORT',GT_read='AFTER_ALL_FRAME_PREDICTIONS_SHA_FROZEN',batch_prompts=16,optimizer_updates=0,protected_reads=0,production_changed=False,TUNE='REUSED_SOURCE_DEVELOPMENT_NOT_XIAN_NOT_BLIND'))
 sys.path.insert(0,str(D/'source'));from segment_anything import sam_model_registry,SamPredictor
 lock=(S/'output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');status('WAIT_SHARED_EXCLUSIVE_GPU');fcntl.flock(lock,fcntl.LOCK_EX);receipts=[]
 try:
  model=sam_model_registry['vit_b']();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.cuda().eval()
  for p in model.parameters():p.requires_grad_(False)
  predictor=SamPredictor(model)
  for f in frames:
   dest=O/'frames'/f['id'];dest.mkdir(parents=True,exist_ok=True);frozen=dest/'PREDICTIONS_FROZEN_BEFORE_GT.json'
   if frozen.exists():
    r=json.loads(frozen.read_text());assert sha(dest/'CANDIDATES.json')==r['candidate_sha256'];assert sha(dest/'MODEL_COMPARISON_INSTANCES.npz')==r['instance_maps_sha256'];assert sha(dest/'IMAGE_EMBEDDING_FP32.npz')==r['embedding_sha256'];receipts.append(r);continue
   assert f['role']=='SOURCE_TUNE';assert sha(f['image_path'])==f['image_sha256'] and sha(f['query_cache_path'])==f['query_cache_sha256'];rgb=np.asarray(Image.open(f['image_path']).convert('RGB'));assert rgb.shape==(1024,1024,3);z=np.load(f['query_cache_path']);old=z['instances'].astype('int32');support=z['support'].astype(bool);assert not (old[~support]>0).any();torch.cuda.reset_peak_memory_stats();start=time.time()
   with torch.no_grad():predictor.set_image(rgb)
   embedding=predictor.get_image_embedding().detach().cpu().numpy();np.savez_compressed(dest/'IMAGE_EMBEDDING_FP32.npz',image_embedding=embedding);ids=[int(i) for i in np.unique(old) if i and (old==i).sum()>=32];queries=[]
   for begin in range(0,len(ids),16):
    batch=ids[begin:begin+16];points=[];boxes=[];priors=[];qq=[]
    for identity in batch:
     mask=old==identity;p,b=point_box(mask);points.append(p);boxes.append(b);priors.append(-8+16*cv2.resize(mask.astype('float32'),(256,256),interpolation=cv2.INTER_AREA));qq.append(dict(query_id=identity,raw_area_native_pixels=int(mask.sum()),point_native_pixel_index=p,box_native_inclusive=b,candidates=[]))
    points=torch.tensor(points,dtype=torch.float32,device='cuda');boxes=torch.tensor(boxes,dtype=torch.float32,device='cuda');pts=predictor.transform.apply_coords_torch(points,(1024,1024))[:,None];bb=predictor.transform.apply_boxes_torch(boxes,(1024,1024));labels=torch.ones((len(batch),1),dtype=torch.int64,device='cuda');prior=torch.from_numpy(np.stack(priors)[:,None]).cuda()
    for mode in ['BOX','POINT_BOX','POINT_BOX_PRIOR']:
     for multi in ([False] if mode=='BOX' else [False,True]):
      with torch.no_grad():masks,quality,_=predictor.predict_torch(point_coords=None if mode=='BOX' else pts,point_labels=None if mode=='BOX' else labels,boxes=bb,mask_input=prior if mode=='POINT_BOX_PRIOR' else None,multimask_output=multi)
      masks=masks.cpu().numpy();quality=quality.cpu().numpy()
      for qi in range(len(batch)):
       for k in range(masks.shape[1]):qq[qi]['candidates'].append(dict(mode=mode,multimask=multi,mask_token_output=k,predicted_mask_quality=float(quality[qi,k]),mask=encode(masks[qi,k]&support)))
    queries.extend(qq)
   save(dest/'CANDIDATES.json',dict(queries=queries,GT_input=False));maps={'RAW':old}
   for mode in ['BOX','POINT_BOX','POINT_BOX_PRIOR']:
    for guarded in [False,True]:maps[mode+('_GUARDED' if guarded else '_UNGUARDED')]=assemble(old,support,queries,mode,guarded)
   np.savez_compressed(dest/'MODEL_COMPARISON_INSTANCES.npz',**maps);r=dict(id=f['id'],role=f['role'],RGB_sha256=f['image_sha256'],parent_cache_sha256=f['query_cache_sha256'],candidate_sha256=sha(dest/'CANDIDATES.json'),instance_maps_sha256=sha(dest/'MODEL_COMPARISON_INSTANCES.npz'),embedding_sha256=sha(dest/'IMAGE_EMBEDDING_FP32.npz'),original_queries=len(ids),SAM_candidate_masks=sum(len(q['candidates']) for q in queries),seconds=time.time()-start,GPU_peak_GiB=torch.cuda.max_memory_allocated()/1024**3,GT_pixels_or_labels_read_for_generation=False);save(frozen,r);receipts.append(r);status('SOURCE_SAM_FRAME_FROZEN',**r)
  del predictor,model;gc.collect();torch.cuda.empty_cache();save(O/'ALL_PREDICTIONS_FROZEN_BEFORE_GT_JOIN.json',dict(frames=receipts,optimizer_updates=0,GT_input=False,protected_reads=0))
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 # Native source GT first opened here, after all model predictions are frozen.
 sys.path.insert(0,'/root/xian20261111_r9/code');from ignore_aware_roof_metrics import instance_metrics
 from native_roof_quality import prepare,measure
 globals().update(instance_metrics=instance_metrics,prepare=prepare,measure=measure)
 tree=ast.parse((S/'code/serial_readonly_evaluator.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','aggregate']],type_ignores=[]),'frozen_original_native_metric', 'exec'),globals());coco=json.loads((S/'instance_data/TUNE.json').read_text());lookup=json.loads((S/'instance_data/NATIVE_LABEL_LOOKUP.json').read_text());by={im['id']:[] for im in coco['images']}
 for a in coco['annotations']:by[a['image_id']].append(a)
 meta={f['image_path']:f for f in frames};results={};query_ceiling=[]
 for im in coco['images']:
  f=meta[im['file_name']];dest=O/'frames'/f['id'];labels=np.asarray(Image.open(lookup[im['file_name']]));valid=labels!=255;ignore=np.zeros(labels.shape,bool);GT=[]
  for a in by[im['id']]:
   mask=mu.decode(mu.merge(mu.frPyObjects(a['segmentation'],*labels.shape))).astype(bool)
   if a.get('iscrowd'):ignore|=mask
   else:GT.append(mask)
  frame=dict(GT=GT,label=labels,valid=valid,ignore=ignore,prepared=prepare(GT,valid&~ignore,2));maps=np.load(dest/'MODEL_COMPARISON_INSTANCES.npz')
  for name in maps.files:results.setdefault(name,[]).append(metrics(frame,maps[name]))
  candidates=json.loads((dest/'CANDIDATES.json').read_text())['queries'];old=maps['RAW']
  for q in candidates:
   raw=old==q['query_id'];ious=np.array([int((raw&g).sum())/max(1,int((raw|g).sum())) for g in GT]);order=np.argsort(-ious)
   if not len(order) or ious[order[0]]<.4 or len(order)>1 and ious[order[0]]-ious[order[1]]<.1:continue
   gt=GT[order[0]];vv=[]
   for c in q['candidates']:
    a=mu.decode(c['mask']).astype(bool);v=int((a&gt).sum())/max(1,int((a|gt).sum()));vv.append(dict(mode=c['mode'],multimask=c['multimask'],mask_token_output=c['mask_token_output'],IoU=v,predicted_quality=c['predicted_mask_quality']))
   query_ceiling.append(dict(frame=f['id'],query_id=q['query_id'],source_GT_index=int(order[0]),raw_IoU=float(ious[order[0]]),oracle_best_candidate=max(vv,key=lambda r:r['IoU']),candidates=vv,oracle_is_not_inference_or_product=True))
 report={k:aggregate(v) for k,v in results.items()};baseline=json.loads((S/'output/common_evaluation/RESULTS.json').read_text())['baseline']
 for k in ['F1_50','pixel_IoU','pixel_recall','complete_roof_recall90','boundary_F1_native2px']:assert abs(report['RAW'][k]-baseline[k])<1e-12
 save(O/'QUERY_GT_CEILING_DIAGNOSTIC.json',dict(rows=query_ceiling,GT_used_only_for_diagnostic_not_mask_selection=True,no_oracle_product_export=True));qualified=[]
 for name,m in report.items():
  if name=='RAW':continue
  if m['F1_50']>baseline['F1_50'] and m['pixel_IoU']>=baseline['pixel_IoU'] and m['pixel_recall']>=baseline['pixel_recall'] and m['complete_roof_recall90']>=baseline['complete_roof_recall90'] and m['boundary_F1_native2px']>baseline['boundary_F1_native2px'] and m['merged_predictions']<=baseline['merged_predictions'] and m['fragmented_GT']<=baseline['fragmented_GT']:qualified.append(name)
 save(O/'EVALUATION_RECEIPT.json',dict(status='COMPLETE_TRUE_NATIVE_ROOF_SAM_ZERO_UPDATE_BASELINE',metrics=report,qualified_source_research_candidates=qualified,source_scope='12_ADMITTED_REUSED_NATIVE_SOURCE_DEVELOPMENT_NOT_XIAN',GT_query_oracle_ceiling_rows=len(query_ceiling),model_optimizer_updates=0,model_GT_input=False,protected_reads=0,production_changed=False));status('SOURCE_SAM_PILOT_COMPLETE',qualified=qualified,metrics=report)
if __name__=='__main__':
 try:main()
 except Exception as e:save(O/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
