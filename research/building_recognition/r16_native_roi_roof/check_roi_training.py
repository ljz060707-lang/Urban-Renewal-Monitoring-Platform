from pathlib import Path
import sys,json,time,traceback
import torch,numpy as np
from concurrent.futures import ThreadPoolExecutor
from roi_training_data import R,D,ROIRoofDataset,collate
from roi_roof_loss import native_logits,loss
sys.path.insert(0,'/root/xian20261111_r15/code')
from roof_decoder_model import forward_roof
from roof_decoder_evaluation import RoofEvaluator,metrics
from roi_training_data import sha
O=R/'output';torch.set_num_threads(4)
def save(n,v):(O/n).write_text(json.dumps(v,indent=2))
try:
 ds=ROIRoofDataset();selected=[];frames=set()
 for i,slot in enumerate(ds.fit_slots):
  row=ds.slots[slot];p=D/'frames'/row['frame_id']/'ROI_FEATURE_RECEIPT.json'
  if p.exists() and row['frame_id'] not in frames:selected.append(i);frames.add(row['frame_id'])
  if len(selected)==2:break
 assert len(selected)==2,'WAIT_TWO_DISTINCT_CACHED_FIT_FRAMES';b=collate([ds[i] for i in selected]);sizes=b['size'].tolist();assert len(set(b['slot'].tolist()))==2
 logits=torch.randn_like(b['target'],dtype=torch.float32,requires_grad=True);quality=torch.zeros(2,requires_grad=True);v=b['valid'].clone();v[:,:,0:3,0:3]=False
 for recall in [False,True]:
  value,_=loss(logits,quality,b['target'],v,b['raw'],b['enabled'],recall);gz,gq=torch.autograd.grad(value,[logits,quality]);assert (gz[~v]==0).all();value,_=loss(logits,quality,b['target'],v,b['raw'],torch.zeros_like(b['enabled']),recall);gz,gq=torch.autograd.grad(value,[logits,quality]);assert (gz==0).all() and (gq==0).all()
 low=torch.randn((2,1,256,256),requires_grad=True);native=native_logits(low,sizes,b['target'].shape[-1]);assert native.shape==b['target'].shape;value,_=loss(native,quality,b['target'],v,b['raw'],b['enabled'],True);value.backward();assert torch.isfinite(low.grad).all()
 # Pure CPU metric parallelism must exactly replay existing frozen real maps.
 evaluator=RoofEvaluator('INNER');cases=[];pilot=Path('/root/xian20261111_r15/output/roi_detail_pilot/frames')
 for frame in evaluator.frames:
  p=pilot/frame['id']/'INSTANCE_MAPS.npz'
  if p.exists():cases.append((frame,np.load(p)['ROI_PRIOR_GUARDED']))
  if len(cases)==2:break
 expected=[metrics(*row) for row in cases]
 with ThreadPoolExecutor(max_workers=2) as pool:actual=list(pool.map(lambda row:metrics(*row),cases))
 assert expected==actual;save('TRAINING_FUNCTIONAL_RECEIPT.json',dict(status='PASS_REAL_NATIVE_VARIABLE_ROI_LOSS_UNKNOWN_PIXEL_DISABLED_QUERY_GRADIENTS_AND_CPU_METRICS_THREADS_BITEXACT',FIT_slots=b['slot'].tolist(),native_crop_sizes=sizes,GT_crop_fraction=b['GT_crop_fraction'].tolist(),two_actual_distinct_images=True,metric_replay_cases=2,optimizer_updates=0,protected_reads=0));print('R16_NATIVE_ROI_TRAINING_FUNCTIONAL_PASS',sizes)
except Exception as err:save('TRAINING_FUNCTIONAL_FAILURE.json',dict(time=time.time(),error=repr(err),traceback=traceback.format_exc()));raise
