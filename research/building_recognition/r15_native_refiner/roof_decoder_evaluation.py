"""Whole native source frames; inference sees only model-generated prompts."""
from pathlib import Path
import json,sys,ast
import numpy as np,torch,cv2
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from pycocotools import mask as mu
from roof_decoder_model import forward_roof
from sam_native_pilot import assemble
from native_roof_training_data import sha
R=Path('/root/xian20261111_r15');DATA=R/'data/native_roof_decoder';S=Path('/root/xian20261111_r10');P=Path('/root/xian20261111_r8');sys.path.insert(0,'/root/xian20261111_r9/code')
from ignore_aware_roof_metrics import instance_metrics
from native_roof_quality import prepare,measure
tree=ast.parse((S/'code/serial_readonly_evaluator.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','aggregate']],type_ignores=[]),'frozen_native_metric','exec'),globals())

class RoofEvaluator:
 def __init__(self,role):
  assert role in ['INNER','TUNE'];self.role=role;all_meta=[json.loads(l) for l in (P/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l];meta={r['image_path']:r for r in all_meta};label_lookup=json.loads((S/'instance_data/NATIVE_LABEL_LOOKUP.json').read_text());rows=[]
  if role=='INNER':
   inputs=json.loads((DATA/'ALL_IMAGE_ONLY_INPUTS_FROZEN.json').read_text())['records'];rr={r['image_path']:r for r in inputs if r['role']=='INNER'};roles={r['image_path']:r for r in json.loads((DATA/'ROLES_FROZEN_BEFORE_PIXEL_READ.json').read_text())['rows']};coco=json.loads((S/'instance_data/EXPANDED1004.json').read_text());images=[im for im in coco['images'] if im['file_name'] in rr];assert len(images)==32
  else:
   coco=json.loads((S/'instance_data/TUNE.json').read_text());images=coco['images'];assert len(images)==12;rr=None
  by={im['id']:[] for im in images}
  for a in coco['annotations']:
   if a['image_id'] in by:by[a['image_id']].append(a)
  for im in images:
   m=meta[im['file_name']];assert m['role']==('SOURCE_TRAIN' if role=='INNER' else 'SOURCE_TUNE');assert sha(m['query_cache_path'])==m['query_cache_sha256'];z=np.load(m['query_cache_path']);old=z['instances'].astype('int32');support=z['support'].astype(bool)
   if role=='INNER':
    rec=rr[im['file_name']];prompts=json.loads(Path(rec['prompts_path']).read_text())['queries'];embedding_path=rec['embedding_path'];embedding_sha=rec['embedding_sha256'];labelpath=roles[im['file_name']]['label_path']
   else:
    folder=R/'output/source_sam_pilot/frames'/m['id'];frozen=json.loads((folder/'PREDICTIONS_FROZEN_BEFORE_GT.json').read_text());assert sha(folder/'CANDIDATES.json')==frozen['candidate_sha256'];prompts=json.loads((folder/'CANDIDATES.json').read_text())['queries'];embedding_path=str(folder/'IMAGE_EMBEDDING_FP32.npz');embedding_sha=frozen['embedding_sha256'];labelpath=label_lookup[im['file_name']]
   assert sha(embedding_path)==embedding_sha;embedding=np.load(embedding_path)['image_embedding'][0];label=np.asarray(Image.open(labelpath));valid=label!=255;ignore=np.zeros((1024,1024),bool);GT=[]
   for a in by[im['id']]:
    mask=mu.decode(mu.merge(mu.frPyObjects(a['segmentation'],1024,1024))).astype(bool)
    if a.get('iscrowd'):ignore|=mask
    else:GT.append(mask)
   rows.append(dict(id=m['id'],old=old,support=support,probability=z['probability'],embedding=embedding,prompts=prompts,label=label,valid=valid,ignore=ignore,GT=GT,prepared=prepare(GT,valid&~ignore,2)))
  self.frames=rows;self.baseline=aggregate([metrics(f,f['old']) for f in rows])
  if role=='TUNE':
   expected=json.loads((S/'output/common_evaluation/RESULTS.json').read_text())['baseline']
   for k in ['F1_50','pixel_IoU','pixel_recall','complete_roof_recall90','boundary_F1_native2px']:assert abs(self.baseline[k]-expected[k])<1e-12
 @torch.no_grad()
 def evaluate(self,sam,batch_size=16,save_directory=None,only=None):
  sam.eval();results={}
  for f in self.frames:
   old=f['old'];candidates=[]
   for begin in range(0,len(f['prompts']),batch_size):
    qq=f['prompts'][begin:begin+batch_size];n=len(qq);e=torch.from_numpy(f['embedding'][None]).cuda().expand(n,-1,-1,-1);points=torch.tensor([q['point_native_pixel_index'] for q in qq],dtype=torch.float32,device='cuda');boxes=torch.tensor([q['box_native_inclusive'] for q in qq],dtype=torch.float32,device='cuda');prior=np.stack([-8+16*cv2.resize((old==q['query_id']).astype('float32'),(256,256),interpolation=cv2.INTER_AREA) for q in qq]);prior=torch.from_numpy(prior[:,None]).cuda();local=[dict(query_id=q['query_id'],candidates=[]) for q in qq]
    for use_prior in [False,True]:
     name='POINT_BOX_PRIOR' if use_prior else 'POINT_BOX';lo,quality=forward_roof(sam,e,points,boxes,prior,torch.full((n,),use_prior,dtype=torch.bool,device='cuda'));mask=(F.interpolate(lo.float(),size=(1024,1024),mode='bilinear',align_corners=False)>0)[:,0].cpu().numpy();quality=quality.cpu().numpy()
     for j,m in enumerate(mask):
      r=mu.encode(np.asfortranarray((m&f['support']).astype('uint8')));r['counts']=r['counts'].decode('ascii');local[j]['candidates'].append(dict(mode=name,predicted_mask_quality=float(quality[j]),mask=r))
    candidates.extend(local)
   outputs={}
   for mode in ['POINT_BOX','POINT_BOX_PRIOR']:
    for guarded in [False,True]:
     name=mode+('_GUARDED' if guarded else '_UNGUARDED')
     if only is not None and name!=only:continue
     labels=assemble(old,f['support'],candidates,mode,guarded);results.setdefault(name,[]).append(metrics(f,labels));outputs[name]=labels
   if save_directory is not None:
    folder=Path(save_directory);folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/(f['id']+'.npz'),**outputs)
  return {k:aggregate(v) for k,v in results.items()}

def passes(m,b):
 return m['F1_50']>b['F1_50'] and m['pixel_IoU']>=b['pixel_IoU'] and m['pixel_recall']>=b['pixel_recall'] and m['complete_roof_recall90']>=b['complete_roof_recall90'] and m['boundary_F1_native2px']>b['boundary_F1_native2px'] and m['merged_predictions']<=b['merged_predictions'] and m['fragmented_GT']<=b['fragmented_GT']
