"""Actual source-native roof mask decoder training; encoder and prompts frozen."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,sys,traceback,gc,fcntl,copy
import numpy as np,torch
import torch.nn.functional as F
from native_roof_training_data import NativeRoofDataset,D,sha
from roof_decoder_model import forward_roof,roof_loss
from roof_decoder_evaluation import RoofEvaluator,passes
R=Path('/root/xian20261111_r15');A=R/'external/sam';O=R/'output/roof_decoder';O.mkdir(exist_ok=True);torch.set_num_threads(4)
def save(name,v):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def status(event,**kw):save('STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def configure(sam):
 for p in sam.parameters():p.requires_grad_(False)
 for p in sam.mask_decoder.parameters():p.requires_grad_(True)
 for head in sam.mask_decoder.output_hypernetworks_mlps[1:]:
  for p in head.parameters():p.requires_grad_(False)
 hi=list(sam.mask_decoder.mask_tokens.parameters())+list(sam.mask_decoder.output_hypernetworks_mlps[0].parameters())+list(sam.mask_decoder.iou_prediction_head.parameters());ids={id(p) for p in hi};lo=[p for p in sam.mask_decoder.parameters() if p.requires_grad and id(p) not in ids];return [dict(params=lo,lr=2e-5,base_lr=2e-5),dict(params=hi,lr=5e-5,base_lr=5e-5)]
def batch_gpu(b,start,end):return {k:v[start:end].cuda(non_blocking=True) for k,v in b.items() if k!='sample_index'}
def loss_for(sam,b,mode):
 low,q=forward_roof(sam,b['embedding'],b['point'],b['box'],b['prior'],mode);native=F.interpolate(low.float(),size=(1024,1024),mode='bilinear',align_corners=False);return roof_loss(native,q,b['target'],b['valid'],b['roi'])
def checkpoint(name,sam,opt,step,best,config,actual_updates):
 states=sorted(set(int(v['step']) for v in opt.state.values() if 'step' in v));assert states==[actual_updates];p=O/name;tmp=p.with_suffix('.tmp');torch.save(dict(decoder={k:v.detach().cpu() for k,v in sam.mask_decoder.state_dict().items()},optimizer=opt.state_dict(),actual_optimizer_updates=actual_updates,loop_step=step,actual_parameter_step_values=states,selection=best,configuration=config,source_dataset_sha256=sha(D/'DATASET_READY.json'),source_model_sha256=sha(A/'sam_vit_b_01ec64.pth')),tmp);tmp.replace(p)
def main():
 assert json.loads((O/'BATCH_AND_LOSS_FUNCTIONAL_RECEIPT.json').read_text())['status'].startswith('PASS');save('PROTOCOL_FROZEN.json',dict(task='QUERY_CONDITIONED_TRUE_NATIVE_ROOF_SAM_DECODER_FINETUNE',native_size=1024,encoder='OFFICIAL_SAM_VITB_FROZEN_FP32_EXACT_CACHED_IMAGE_EMBEDDINGS',trainable='MASK_DECODER_TRANSFORMER_UPSCALING_TOKEN0_HYPERNETWORK_AND_ROOF_IOU_HEAD;OTHER_MASK_HYPERNETWORKS_FROZEN',prompt_encoder_frozen=True,source_roles='FIT972_BJ_HARBIN;INNER32_CHENGDU;REUSED_SOURCE_TUNE12_NOT_BLIND',GT_native_roof_not_XIAN_map=True,no_GT_query_supervision=False,unknown_roof_pixel_loss='ZERO_GRADIENT_TESTED',negative_semantics='NOT_THIS_QUERY_ID_NOT_CERTIFIED_SCENE_BACKGROUND',loss='2POS_BCE+0.75_TVERSKY_FN0.65+0.3_LOG_AREA_HUBER+0.2_NATIVE_BOUNDARY_DICE+0.1_QUALITY_MSE',effective_batch_queries=32,optimizer='FP32_ADAMW_NO_AMP_SKIP',base_lr=2e-5,roof_token_hypernet_and_quality_lr=5e-5,weight_decay=.001,actual_steps=2000,validation_steps=[0,100,250,500,1000,1500,2000],selection='INNER_FULL_NATIVE_FRAME_COMMON_GATES;SOURCE_F1_ALONE_INSUFFICIENT',resume='IMMUTABLE_FIXED_SAMPLE_AND_PRIOR_MODE_SCHEDULE_BY_OPTIMIZER_STEP',TUNE='SELECTED_INNER_CHECKPOINT_AND_VARIANT_FROZEN_BEFORE_ONE_NEW_MODEL_DEVELOPMENT_EVALUATION',protected_reads=0,production_changed=False))
 while not (D/'DATASET_READY.json').exists():status('WAIT_FROZEN_NATIVE_SOURCE_DATA');time.sleep(10)
 data=json.loads((D/'DATASET_READY.json').read_text());assert data['protected_reads']==0;dataset=NativeRoofDataset(data,'FIT');schedule=O/'FIXED_TRAINING_SCHEDULE.npz'
 if not schedule.exists():
  rng=np.random.default_rng(20261116);np.savez_compressed(schedule,indices=rng.integers(0,len(dataset),(2000,32),dtype=np.int64),use_prior=rng.integers(0,2,(2000,32),dtype=np.uint8))
 plan=np.load(schedule);indices=plan['indices'];mode_flags=plan['use_prior'];sys.path.insert(0,str(A/'source'));from segment_anything import sam_model_registry
 lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');status('WAIT_EXCLUSIVE_GPU_DECODER_TRAINING');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261116);sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(A/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True),strict=True);sam.cuda().eval();groups=configure(sam);initial={k:v.detach().cpu().clone() for k,v in sam.mask_decoder.state_dict().items()};save('ACTUAL_PARAMETER_CONFIGURATION.json',dict(trainable_parameters=sum(p.numel() for p in sam.parameters() if p.requires_grad),frozen_parameters=sum(p.numel() for p in sam.parameters() if not p.requires_grad),names=[dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in sam.named_parameters()],effective_batch=32,precision='FP32_NO_TF32',schedule_sha256=sha(schedule),dataset_sha256=sha(D/'DATASET_READY.json')))
  benchmark=O/'NATIVE_DECODER_THROUGHPUT.json'
  if not benchmark.exists():
   bb=torch.utils.data.default_collate([dataset[int(i)] for i in indices[0]]);results=[]
   for micro in [4,8,16,32]:
    sam.mask_decoder.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();b=batch_gpu(bb,0,micro);flag=torch.from_numpy(mode_flags[0,:micro].astype(bool)).cuda();torch.cuda.synchronize();start=time.time();loss,_=loss_for(sam,b,flag);assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize();sec=time.time()-start;results.append(dict(microbatch=micro,seconds=sec,queries_per_second=micro/sec,peak_GiB=torch.cuda.max_memory_allocated()/1024**3));del b,loss;sam.mask_decoder.zero_grad(set_to_none=True)
   choice=max([r for r in results if r['peak_GiB']<16],key=lambda r:r['queries_per_second']);save(benchmark.name,dict(results=results,selection=choice,optimizer_updates=0,initial_decoder_reloaded_after=True));del bb
  micro=json.loads(benchmark.read_text())['selection']['microbatch'];sam.mask_decoder.load_state_dict(initial);sam.mask_decoder.zero_grad(set_to_none=True);opt=torch.optim.AdamW(groups,weight_decay=.001);resume=O/'last.pt';start_step=0;qualified_best=None;research_best=None;history=[]
  if resume.exists():
   state=torch.load(resume,map_location='cpu');sam.mask_decoder.load_state_dict(state['decoder']);opt.load_state_dict(state['optimizer']);start_step=state['actual_optimizer_updates'];
   for st in opt.state.values():
    for k,v in st.items():
     if isinstance(v,torch.Tensor) and k!='step':st[k]=v.cuda()
  if (O/'HISTORY.json').exists():history=json.loads((O/'HISTORY.json').read_text())
  for name in ['best_source','research_best']:
   p=O/(name+'.pt')
   if p.exists():
    s=torch.load(p,map_location='cpu');r=s['selection']
    if name=='best_source':qualified_best=r
    else:research_best=r
  evaluator=RoofEvaluator('INNER');save('INNER_RAW_BASELINE.json',evaluator.baseline)
  if not (O/'INITIAL_SINGLE_TOKEN0_CONTROL.json').exists():save('INITIAL_SINGLE_TOKEN0_CONTROL.json',dict(metrics=evaluator.evaluate(sam),actual_optimizer_updates=start_step,matched_wrapper_same_as_training=True,TUNE_not_read=True))
  subset=torch.utils.data.Subset(dataset,indices[start_step:].ravel().tolist());loader=torch.utils.data.DataLoader(subset,batch_size=32,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2);iterator=iter(loader);actual=start_step
  for step in range(start_step+1,2001):
   batch=next(iterator);assert np.array_equal(batch['sample_index'].numpy(),indices[step-1]);opt.zero_grad(set_to_none=True);sam.mask_decoder.train();logs=[];total=0.;factor=min(1.,step/100)*(.2 if step>=1400 else 1.)*(.2 if step>=1800 else 1.)
   for group in opt.param_groups:group['lr']=group['base_lr']*factor
   for begin in range(0,32,micro):
    end=min(32,begin+micro);b=batch_gpu(batch,begin,end);flag=torch.from_numpy(mode_flags[step-1,begin:end].astype(bool)).cuda();loss,parts=loss_for(sam,b,flag);assert torch.isfinite(loss);(loss*((end-begin)/32)).backward();total+=float(loss.detach())*((end-begin)/32);logs.append(parts);del b,loss
   norm=torch.nn.utils.clip_grad_norm_([p for p in sam.mask_decoder.parameters() if p.requires_grad],5.);assert torch.isfinite(norm);opt.step();actual+=1
   if step%25==0:
    row=dict(actual_optimizer_updates=actual,loss=total,gradient_norm=float(norm),microbatch=micro,effective_batch=32,lr=opt.param_groups[0]['lr'],**{k:float(np.mean([r[k] for r in logs])) for k in ['BCE','Tversky_loss','log_area_loss','boundary_loss','quality_MSE']});history.append(row);save('HISTORY.json',history);status('TRAIN_TRUE_NATIVE_ROOF_DECODER',**row)
   if step in [100,250,500,1000,1500,2000]:
    report=evaluator.evaluate(sam);eligible=[name for name,m in report.items() if passes(m,evaluator.baseline)];choice=max(report,key=lambda name:report[name]['F1_50']);selection=dict(step=step,variant=choice,metrics=report[choice],qualified_inner=False)
    if research_best is None or selection['metrics']['F1_50']>research_best['metrics']['F1_50']:research_best=selection;checkpoint('research_best.pt',sam,opt,step,selection,dict(microbatch=micro),actual)
    if eligible:
     winner=max(eligible,key=lambda name:report[name]['F1_50']);r=dict(step=step,variant=winner,metrics=report[winner],qualified_inner=True)
     if qualified_best is None or r['metrics']['F1_50']>qualified_best['metrics']['F1_50']:qualified_best=r;checkpoint('best_source.pt',sam,opt,step,r,dict(microbatch=micro),actual)
    checkpoint(f'iter_{step}.pt',sam,opt,step,selection,dict(microbatch=micro),actual);save(f'iter_{step}_INNER.json',dict(metrics=report,qualified_variants=eligible,selection=selection));checkpoint('last.pt',sam,opt,step,selection,dict(microbatch=micro),actual);status('NATIVE_ROOF_DECODER_INNER',actual_optimizer_updates=actual,qualified_variants=eligible,metrics=report)
  chosen='best_source.pt' if qualified_best else 'research_best.pt';state=torch.load(O/chosen,map_location='cpu');selection=state['selection'];save('SELECTION_FROZEN_BEFORE_TUNE.json',dict(checkpoint=chosen,checkpoint_sha256=sha(O/chosen),selection=selection,actual_training_updates=actual,TUNE_new_model_evaluation_performed=False));sam.mask_decoder.load_state_dict(state['decoder']);tune=RoofEvaluator('TUNE');report=tune.evaluate(sam,save_directory=O/'TUNE_SELECTED_INSTANCE_MAPS',only=selection['variant']);m=report[selection['variant']];qualified=bool(qualified_best) and passes(m,tune.baseline);save('TRAINING_RECEIPT.json',dict(status='COMPLETE_REAL_NATIVE_ROOF_SAM_DECODER_TRAINING',actual_AdamW_updates=actual,parent_model_unchanged=True,chosen_checkpoint=chosen,chosen_sha256=sha(O/chosen),selection=selection,TUNE_metrics=report,TUNE_raw_baseline=tune.baseline,qualified_source_research_candidate=qualified,source_TUNE_reused_not_XIAN_accuracy=True,GT_map_not_native_roof_training=True,no_GT_query_training=False,unknown_pixel_loss_zero_tested=True,protected_reads=0,production_changed=False));status('NATIVE_ROOF_DECODER_COMPLETE',qualified=qualified,actual_optimizer_updates=actual,metrics=report)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:main()
 except Exception as e:save('FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
