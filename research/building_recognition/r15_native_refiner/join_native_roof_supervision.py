"""Join native roof instances only to already frozen model-generated prompts."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,hashlib,traceback,collections
import numpy as np
from PIL import Image
from pycocotools import mask as mu
R=Path('/root/xian20261111_r15');O=R/'data/native_roof_decoder';C=Path('/root/xian20261111_r10/instance_data/EXPANDED1004.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def main():
 while not (O/'ROLES_FROZEN_BEFORE_PIXEL_READ.json').exists():time.sleep(5)
 role_receipt=json.loads((O/'ROLES_FROZEN_BEFORE_PIXEL_READ.json').read_text());roles=role_receipt['rows'];assert len(roles)==1004 and all(r['decoder_role'] in ['FIT','INNER'] for r in roles);coco=json.loads(C.read_text());images={r['file_name']:r for r in coco['images']};assert set(images)=={r['image_path'] for r in roles};assert len({a['id'] for a in coco['annotations']})==len(coco['annotations']);by=collections.defaultdict(list)
 for a in coco['annotations']:
  if not a.get('iscrowd'):by[a['image_id']].append(a)
 save(O/'SUPERVISION_PROTOCOL_FROZEN.json',dict(task='PREDICTED_PROMPT_TO_KNOWN_NATIVE_ROOF_SUPERVISION',annotation_source=str(C),annotation_sha256=sha(C),source_ROOF_not_footprint_or_XIAN_map=True,match='UNIQUE_GT_ROOF_CONTAINING_PREDICTED_CORE_POINT;ELSE_NATIVE_RAW_IOU_GE0.4_MARGIN0.1',points_boxes_masks='MODEL_ONLY_FROZEN_BEFORE_GT_JOIN;NO_GT_PROMPT_INSERTION',missing_GT_query='EXCLUDED_FROM_ALL_SUPERVISED_LOSSES_RETAINED_IN_PROMPT_MANIFEST',multiple_queries_same_GT='ALLOWED_NATIVE_INSTANCE_COMPLETION_NOT_ONE_GT_INSERTION_PER_QUERY',pixel_loss_semantics='THIS_QUERY_ID_VS_NOT_THIS_QUERY_ID_NOT_CERTIFIED_GLOBAL_BACKGROUND',unknown_pixel_mask='IGNORE255_AND_KNOWN_ROOF_PIXELS_WITHOUT_ELIGIBLE_GT;KEEP_KNOWN_REGULAR_GT_POSITIVE_PIXELS_EVEN_IF_CONSERVATIVE64PX_IGNORE_TILE_OVERLAPS',TUNE_training_reads=0,protected_reads=0));done={};samples=[]
 while len(done)<len(roles):
  progress=False
  for f in roles:
   if f['id'] in done:continue
   folder=O/'frames'/f['id'];p=folder/'INPUTS_FROZEN_BEFORE_GT_JOIN.json'
   if not p.exists():continue
   out=folder/'SUPERVISION_SEPARATE.json'
   if out.exists():
    data=json.loads(out.read_text());done[f['id']]=data['audit'];samples.extend(data['samples']);continue
   receipt=json.loads(p.read_text());assert sha(receipt['prompts_path'])==receipt['prompts_sha256'] and sha(receipt['embedding_path'])==receipt['embedding_sha256'];assert sha(f['label_path'])==f['label_sha256'] and sha(f['query_cache_path'])==f['query_cache_sha256'];label=np.asarray(Image.open(f['label_path']));old=np.load(f['query_cache_path'])['instances'];prompts=json.loads(Path(receipt['prompts_path']).read_text())['queries'];im=images[f['image_path']];anns=by[im['id']];truth=[];union=np.zeros((1024,1024),bool)
   for a in anns:
    mask=mu.decode(mu.merge(mu.frPyObjects(a['segmentation'],1024,1024))).astype(bool);union|=mask;truth.append((a,mask))
   unknown_roof=(label==1)&~union;valid=(label!=255)&~unknown_roof;assert not (valid&unknown_roof).any();assert not ((union&(label!=255))&~valid).any();vp=folder/'QUERY_ID_LOSS_VALID.npz';np.savez_compressed(vp,valid=valid);new=[];covered=set();noGT=ambiguous=border=0
   for q in prompts:
    if q['border']:border+=1;continue
    raw=old==q['query_id'];x,y=map(int,q['point_native_pixel_index']);hits=[i for i,(_,g) in enumerate(truth) if g[y,x]];target=None;method=None
    if len(hits)==1:target=hits[0];method='UNIQUE_NATIVE_GT_AT_PREDICTED_POINT'
    elif len(hits)>1:ambiguous+=1;continue
    else:
     values=[]
     for i,(a,g) in enumerate(truth):
      bx,by0,bw,bh=a['bbox'];x0,y0,x1,y1=q['bounds_native']
      if bx>=x1 or by0>=y1 or bx+bw<=x0 or by0+bh<=y0:continue
      inter=int((raw&g).sum())
      if inter:values.append((inter/max(1,int(raw.sum())+int(g.sum())-inter),i))
     values.sort(reverse=True)
     if values and values[0][0]>=.4 and (len(values)==1 or values[0][0]-values[1][0]>=.1):target=values[0][1];method='NATIVE_RAW_GT_IOU_GE0.4_MARGIN0.1'
    if target is None:noGT+=1;continue
    a,g=truth[target];covered.add(a['id']);new.append(dict(id=f['id']+'_q'+str(q['query_id']),frame_id=f['id'],role=f['decoder_role'],image_id=im['id'],annotation_id=a['id'],embedding_path=receipt['embedding_path'],embedding_sha256=receipt['embedding_sha256'],cache_path=f['query_cache_path'],cache_sha256=f['query_cache_sha256'],valid_path=str(vp),valid_sha256=sha(vp),query_id=q['query_id'],point_native_pixel_index=q['point_native_pixel_index'],box_native_inclusive=q['box_native_inclusive'],bounds_native=q['bounds_native'],target_GT_native_pixels=int(g.sum()),match_method=method,loss_enabled=True,GT_in_model_prompt=False))
   audit=dict(frame_id=f['id'],role=f['decoder_role'],model_queries=len(prompts),matched_supervised_queries=len(new),no_GT_query_excluded=noGT,ambiguous_GT_point_excluded=ambiguous,border_query_excluded=border,GT_roof_instances=len(truth),GT_instances_with_supervised_query=len(covered),GT_without_query=len(truth)-len(covered),unknown_roof_pixels_ignored=int(unknown_roof.sum()),native_GT_positive_pixels_retained=int((union&valid).sum()),GT_prompt_insertion=False,protected_reads=0);data=dict(samples=new,audit=audit,parent_inputs_receipt_sha256=sha(p));save(out,data);done[f['id']]=audit;samples.extend(new);progress=True
   if len(done)%20==0:save(O/'SUPERVISION_STATUS.json',dict(time=time.time(),frames=len(done),total=len(roles),supervised_queries=len(samples)));print('JOIN_NATIVE_ROOF_GT',len(done),len(samples),flush=True)
  if not progress:time.sleep(5)
 byrole=collections.Counter(r['role'] for r in samples);save(O/'DATASET_READY.json',dict(status='READY_TRUE_NATIVE_ROOF_DECODER_SUPERVISION',samples=samples,frame_audits=list(done.values()),samples_by_role=dict(byrole),COCO_source=str(C),COCO_source_sha256=sha(C),all_model_prompts_retained_in_separate_manifest=True,no_GT_supervised_queries=0,protected_reads=0,model_optimizer_updates=0));print('NATIVE_ROOF_DECODER_DATA_READY',dict(byrole),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(O/'SUPERVISION_FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
