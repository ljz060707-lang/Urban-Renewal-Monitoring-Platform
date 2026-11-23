from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,sys,time,hashlib,fcntl
import numpy as np,torch
from roof_decoder_model import forward_roof,roof_loss
R=Path('/root/xian20261111_r15');D=R/'external/sam';O=R/'output/roof_decoder';O.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(D/'source'))
from segment_anything import sam_model_registry
lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(D/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True),strict=True);sam.cuda().eval()
 for p in sam.parameters():p.requires_grad_(False)
 source=R/'output/source_sam_pilot';frozen=json.loads((source/'ALL_PREDICTIONS_FROZEN_BEFORE_GT_JOIN.json').read_text())['frames'][:2];emb=[];points=[];boxes=[]
 for f in frozen:
  path=source/'frames'/f['id'];e=path/'IMAGE_EMBEDDING_FP32.npz';assert hashlib.sha256(e.read_bytes()).hexdigest()==f['embedding_sha256'];emb.append(np.load(e)['image_embedding'][0]);q=json.loads((path/'CANDIDATES.json').read_text())['queries'][0];points.append(q['point_native_pixel_index']);boxes.append(q['box_native_inclusive'])
 e=torch.from_numpy(np.stack(emb)).cuda();p=torch.tensor(points,device='cuda');b=torch.tensor(boxes,dtype=torch.float32,device='cuda');prior=torch.zeros((2,1,256,256),device='cuda');flag=torch.zeros(2,dtype=torch.bool,device='cuda')
 with torch.no_grad():
  actual,aq=forward_roof(sam,e,p,b,prior,flag);expected=[];qualities=[]
  for i in range(2):
   s,d=sam.prompt_encoder(points=(p[i:i+1,None],torch.ones((1,1),dtype=torch.int64,device='cuda')),boxes=b[i:i+1],masks=None);a,q=sam.mask_decoder(image_embeddings=e[i:i+1],image_pe=sam.prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=s,dense_prompt_embeddings=d,multimask_output=False);expected.append(a);qualities.append(q[:,0])
  expected=torch.cat(expected);qualities=torch.cat(qualities);error=float((actual-expected).abs().max());qerror=float((aq-qualities).abs().max());assert torch.allclose(actual,expected,atol=2e-4,rtol=1e-4) and torch.allclose(aq,qualities,atol=2e-4,rtol=1e-4)
 for par in sam.mask_decoder.parameters():par.requires_grad_(True)
 out,_=forward_roof(sam,e,p,b,prior,flag);out.mean().backward();assert any(par.grad is not None and torch.isfinite(par.grad).all() and par.grad.abs().sum()>0 for par in sam.mask_decoder.parameters());assert all(par.grad is None for par in sam.image_encoder.parameters())
 # Direct native loss check: unknown pixels and disabled query have zero gradients.
 logits=torch.randn((2,1,32,32),device='cuda',requires_grad=True);quality=torch.randn(2,device='cuda',requires_grad=True);target=torch.zeros_like(logits,dtype=torch.bool);target[:,:,12:20,12:20]=True;valid=torch.ones_like(target);valid[:,:,4:8,4:8]=False;roi=torch.ones_like(valid);loss,_=roof_loss(logits,quality,target,valid,roi,torch.tensor([True,False],device='cuda'));loss.backward();assert torch.isfinite(loss) and (logits.grad[~valid]==0).all() and (logits.grad[1]==0).all() and quality.grad[1]==0
 proof=dict(status='PASS_DISTINCT_IMAGE_BATCH_DECODER_AND_UNKNOWN_LOSS_GRADIENT',time=time.time(),images=2,maximum_lowres_logit_error=error,maximum_quality_error=qerror,numerical_equivalence='ATOL2E-4_RTOL1E-4_NOT_BITEXACT_CLAIM',mask_decoder_gradient_finite=True,image_encoder_gradient_none=True,unknown_native_pixel_gradient_zero=True,no_GT_query_gradient_zero=True,optimizer_updates=0,real_data_backward_functional_only_not_training=True,protected_reads=0);(O/'BATCH_AND_LOSS_FUNCTIONAL_RECEIPT.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
