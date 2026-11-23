"""Native full-frame metrics; model sees only its own proposed ROIs and masks."""
from pathlib import Path
import json,sys,time,hashlib
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch,cv2
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mu
from roi_training_data import R,D,sha
from prepare_roi_embeddings import crop_bounds,save
sys.path.insert(0,'/root/xian20261111_r15/code')
from roof_decoder_model import forward_roof
from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate,passes
from sam_native_pilot import assemble
PRE=Path('/root/xian20261111_r15');SOURCE=Path('/root/xian20261111_r8');A=PRE/'external/sam'
@lru_cache(maxsize=16)
def RGB(path,digest):assert sha(path)==digest;return np.asarray(Image.open(path).convert('RGB'))
@torch.no_grad()
def prepare_TUNE(sam):
 """Called only after all selected weights/loss/guard branches have been frozen."""
 O=R/'output/training/TUNE_INPUTS';O.mkdir(exist_ok=True);assert (R/'output/training/ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json').exists();rows=[];source=[json.loads(line) for line in (SOURCE/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if line];records=[row for row in source if row['role']=='SOURCE_TUNE'];assert len(records)==12
 for rec in records:
  prompts_path=PRE/'output/source_sam_pilot/frames'/rec['id']/'CANDIDATES.json';prompts=json.loads(prompts_path.read_text())['queries']
  for q in prompts:rows.append(dict(slot=len(rows),frame_id=rec['id'],role='SOURCE_TUNE',image_path=rec['image_path'],image_sha256=rec['image_sha256'],cache_path=rec['query_cache_path'],cache_sha256=rec['query_cache_sha256'],query_id=q['query_id'],point_native_pixel_index=q['point_native_pixel_index'],box_native_inclusive=q['box_native_inclusive'],crop_native_xy_size=crop_bounds(q),model_prompt_source_sha256=sha(prompts_path)))
 cache=Path('/dev/shm/xian_r16_roi_roof/TUNE_ROI_EMBEDDINGS_FP32.npy');save(O/'MODEL_CROPS_FROZEN_BEFORE_GT.json',dict(rows=rows,selection_sha256=sha(R/'output/training/ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json'),GT_in_inputs=False,protected_reads=0));manifest=O/'FEATURES_FROZEN_BEFORE_GT.json'
 if manifest.exists():
  receipt=json.loads(manifest.read_text());array=np.load(cache,mmap_mode='r');assert array.shape==(len(rows),256,64,64)
  for rec in receipt['frames']:assert hashlib.sha256(array[rec['slots']].tobytes()).hexdigest()==rec['embedding_values_sha256']
  return rows,array,records
 array=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(len(rows),256,64,64));sys.path.insert(0,str(A/'source'));from segment_anything.utils.transforms import ResizeLongestSide
 transform=ResizeLongestSide(1024)
 def prepare(row):
  x,y,s=row['crop_native_xy_size'];im=RGB(row['image_path'],row['image_sha256']);return torch.from_numpy(np.ascontiguousarray(transform.apply_image(im[y:y+s,x:x+s]).transpose(2,0,1)))
 grouped=defaultdict(list)
 for row in rows:grouped[row['frame_id']].append(row)
 receipts=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for frame,rr in grouped.items():
   pixels=iter(pool.map(prepare,rr))
   for start in range(0,len(rr),4):
    batch=rr[start:start+4];e=sam.image_encoder(sam.preprocess(torch.stack([next(pixels) for _ in batch]).cuda()));values=e.cpu().numpy();indices=[r['slot'] for r in batch];array[indices]=values;assert np.array_equal(array[indices],values)
   indices=[r['slot'] for r in rr];receipts.append(dict(frame_id=frame,slots=indices,embedding_values_sha256=hashlib.sha256(array[indices].tobytes()).hexdigest()));print('R16_TUNE_ROI_FEATURES',frame,len(rr),flush=True)
 array.flush();save(manifest,dict(frames=receipts,cache=str(cache),shape=list(array.shape),model_sha256=sha(A/'sam_vit_b_01ec64.pth'),GT_pixels_or_shapes_in_features=False,optimizer_updates=0));return rows,array,records
class NativeROIEvaluator:
 def __init__(self,role,prepared=None):
  self.role=role;self.truth=None;self.baseline=None
  if role=='INNER':
   manifest=json.loads((D/'SLOT_MANIFEST_FROZEN.json').read_text());self.rows=[r for r in manifest['rows'] if r['role']=='INNER'];self.cache=np.load(manifest['cache_path'],mmap_mode='r');self.truth=RoofEvaluator('INNER');self.baseline=self.truth.baseline;self.frames=[{k:f[k] for k in ['id','old','support']} for f in self.truth.frames]
  else:
   assert role=='TUNE' and prepared is not None;self.rows,self.cache,records=prepared;self.frames=[]
   for rec in records:
    assert sha(rec['query_cache_path'])==rec['query_cache_sha256'];z=np.load(rec['query_cache_path']);self.frames.append(dict(id=rec['id'],old=z['instances'].astype('int32'),support=z['support'].astype(bool)))
  self.byframe=defaultdict(list)
  for row in self.rows:self.byframe[row['frame_id']].append(row)
 @torch.no_grad()
 def predict_frame(self,sam,f,batch=16):
  rows=self.byframe[f['id']];old=f['old'];candidates=[]
  for start in range(0,len(rows),batch):
   rr=rows[start:start+batch];emb=torch.from_numpy(self.cache[[r['slot'] for r in rr]].copy()).cuda();points=[];boxes=[];priors=[]
   for r in rr:
    x,y,s=r['crop_native_xy_size'];points.append((np.array(r['point_native_pixel_index'])-[x,y])*(1024/s));boxes.append((np.array(r['box_native_inclusive'])-[x,y,x,y])*(1024/s));priors.append(-8+16*cv2.resize((old[y:y+s,x:x+s]==r['query_id']).astype('float32'),(256,256),interpolation=cv2.INTER_AREA))
   pt=torch.tensor(np.array(points),device='cuda',dtype=torch.float32);bx=torch.tensor(np.array(boxes),device='cuda',dtype=torch.float32);pr=torch.from_numpy(np.stack(priors)[:,None]).cuda();low,quality=forward_roof(sam,emb,pt,bx,pr,torch.ones(len(rr),device='cuda',dtype=torch.bool))
   for j,r in enumerate(rr):
    x,y,s=r['crop_native_xy_size'];native=(F.interpolate(low[j:j+1].float(),(s,s),mode='bilinear',align_corners=False)>0)[0,0].cpu().numpy();mask=np.zeros((1024,1024),bool);mask[y:y+s,x:x+s]=native;mask&=f['support'];encoded=mu.encode(np.asfortranarray(mask.astype('uint8')));encoded['counts']=encoded['counts'].decode('ascii');candidates.append(dict(query_id=r['query_id'],candidates=[dict(mode='ROI_PRIOR',predicted_mask_quality=float(quality[j]),mask=encoded)]))
  return candidates
 @torch.no_grad()
 def evaluate(self,sam,only=None,save_directory=None):
  sam.eval();pending=[];frozen=[];variants=['ROI_PRIOR_UNGUARDED','ROI_PRIOR_GUARDED'];variants=[n for n in variants if only is None or n==only];truth={f['id']:f for f in self.truth.frames} if self.truth else None
  with ThreadPoolExecutor(max_workers=8) as pool:
   for f in self.frames:
    candidates=self.predict_frame(sam,f);outputs={}
    for name in variants:
     labels=assemble(f['old'],f['support'],candidates,'ROI_PRIOR',name.endswith('_GUARDED') and not name.endswith('_UNGUARDED'));outputs[name]=labels
     if truth is not None:pending.append((name,pool.submit(metrics,truth[f['id']],labels)))
    if save_directory is not None:
     directory=Path(save_directory);directory.mkdir(parents=True,exist_ok=True);p=directory/(f['id']+'.npz');np.savez_compressed(p,**outputs);frozen.append(dict(frame_id=f['id'],path=str(p),sha256=sha(p)))
   if truth is None:
    # Source-TUNE truth is opened only after every selected output is frozen.
    assert save_directory is not None;save(Path(save_directory)/'PREDICTIONS_FROZEN_BEFORE_GT.json',dict(rows=frozen,GT_in_prediction_inputs=False,selection_already_frozen=True));self.truth=RoofEvaluator('TUNE');self.baseline=self.truth.baseline;truth={f['id']:f for f in self.truth.frames}
    for rec in frozen:
     assert sha(rec['path'])==rec['sha256'];z=np.load(rec['path'])
     for name in variants:pending.append((name,pool.submit(metrics,truth[rec['frame_id']],z[name])))
   results=defaultdict(list)
   for name,future in pending:results[name].append(future.result())
  return {name:aggregate(values) for name,values in results.items()}
