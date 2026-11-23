# -*- coding: utf-8 -*-
"""Two fixed native-ROI decoder loss arms; real AdamW steps, immutable selection."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,sys,gc,fcntl,traceback,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
from roi_training_data import R,D,ROIRoofDataset,collate,sha
from roi_roof_loss import native_logits,loss
from roi_roof_evaluation import NativeROIEvaluator,prepare_TUNE,passes
sys.path.insert(0,'/root/xian20261111_r15/code')
from roof_decoder_model import forward_roof
T=R/'output/training';T.mkdir(exist_ok=True);A=Path('/root/xian20261111_r15/external/sam');ARMS=[('STANDARD_NATIVE_ROI',False),('RECALL_PROTECTED_NATIVE_ROI',True)];torch.set_num_threads(4)
def save(name,value):
 p=T/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def status(event,**kw):save('STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def configure(sam):
 for p in sam.parameters():p.requires_grad_(False)
 for p in sam.mask_decoder.parameters():p.requires_grad_(True)
 for head in sam.mask_decoder.output_hypernetworks_mlps[1:]:
  for p in head.parameters():p.requires_grad_(False)
 high=list(sam.mask_decoder.mask_tokens.parameters())+list(sam.mask_decoder.output_hypernetworks_mlps[0].parameters())+list(sam.mask_decoder.iou_prediction_head.parameters());ids={id(p) for p in high};low=[p for p in sam.mask_decoder.parameters() if p.requires_grad and id(p) not in ids];return [dict(params=low,lr=2e-5,base_lr=2e-5),dict(params=high,lr=5e-5,base_lr=5e-5)]
def gpu_batch(b,start,end):return {k:v[start:end].cuda(non_blocking=True) for k,v in b.items() if k not in ['size','index','slot','GT_crop_fraction']}
def loss_for(sam,b,sizes,recall):
 lo,q=forward_roof(sam,b['embedding'],b['point'],b['box'],b['prior'],torch.ones(len(sizes),device='cuda',dtype=torch.bool));native=native_logits(lo,sizes,b['target'].shape[-1]);return loss(native,q,b['target'],b['valid'],b['raw'],b['enabled'],recall)
def checkpoint(path,sam,opt,step,selection,arm):
 steps=sorted(set(int(s['step']) for s in opt.state.values() if 'step' in s));assert steps==[step];tmp=path.with_suffix('.pending');torch.save(dict(decoder={k:v.detach().cpu() for k,v in sam.mask_decoder.state_dict().items()},optimizer=opt.state_dict(),actual_optimizer_updates=step,actual_AdamW_parameter_steps=steps,selection=selection,arm=arm,feature_freeze_sha256=sha(D/'ALL_ROI_FEATURES_FROZEN.json'),training_schedule_sha256=sha(T/'FIXED_TRAINING_SCHEDULE.npz'),parent_model_sha256=sha(A/'sam_vit_b_01ec64.pth')),tmp);tmp.replace(path)
def selection_key(s):return bool(s['qualified_inner']),s['metrics']['F1_50'],s['metrics']['boundary_F1_native2px']
def main():
 if (T/'TRAINING_RECEIPT.json').exists():status('ALREADY_COMPLETE_NO_RETRAIN');return
 save('PROTOCOL_FROZEN.json',dict(task='MATCHED_TRUE_NATIVE_ROI_ROOF_DECODER_STANDARD_VS_RECALL_PROTECTED',arms=[a for a,_ in ARMS],actual_steps_per_arm=2000,encoder='SAME_GENERIC_SAM_VITB_FROZEN_FP32_NO_TF32',image_or_map_GT_in_model_inputs=False,trainable='MASK_DECODER_TRANSFORMER_UPSCALING_MASKTOKEN0_HYPERNET_AND_QUALITY;OTHER_MASK_HYPERNETS_FROZEN',geometry='MODEL_ONLY_CROP_BEFORE_GT_JOIN;NO_GT_EXPANDED_CROP',native_loss='LOW256_LOGITS_BILINEAR_BACK_TO_ACTUAL_CROP_SIZE_THEN_PAD_VALID0;NO_GT_RESIZE',standard='2POS_BCE+.75_TVERSKY_FN.65+.3_LOG_AREA_HUBER+.2_NATIVE_EDGE_DICE+.1_QUALITY_MSE',recall='4POS_BCE;TVERSKY_FN.8;ADDITIONAL.35_KNOWN_RAW_AND_GT_POSITIVE_RETENTION_BCE',no_GT_or_unknown_pixel_loss='ZERO_DIRECT_GRADIENT_FUNCTIONALLY_TESTED',partial_GT_outside_input='NOT_USED_TO_EXPAND_INPUT;FULL_GT_DENOMINATOR_RETAINED_IN_NATIVE_FRAME_EVALUATION',optimizer='FP32_ADAMW_NO_AMP_SKIP',lr=2e-5,roof_token_hypernet_quality_lr=5e-5,weight_decay=.001,clip_grad_norm=5.,effective_batch_queries=32,identical_sample_schedule=True,prior_mode='PREDICTED_QUERY_MASK_ALWAYS_PRESENT_MATCHES_INFERENCE',selection='FULL_NATIVE32_INNER_COMMON_F1_IOU_RECALL_COMPLETE_BOUNDARY_MERGE_FRAGMENT_GATES',validation_steps=[0,250,500,1000,2000],source_TUNE='ONE_CHOSEN_CHECKPOINT_AND_GUARD_AFTER_ALL_ARMS_FROZEN;REUSED_SOURCE_DEVELOPMENT_NOT_BLIND',protected_reads=0,production_changed=False))
 while not (D/'DATASET_READY.json').exists():
  if (R/'output/FAILURE_RECEIPT.json').exists():raise RuntimeError('ROI_FEATURE_PREPARATION_FAILURE_RECEIPT_REQUIRES_REPAIR')
  status('WAIT_COMPLETE_FROZEN_NATIVE_ROI_FEATURES',actual_optimizer_updates=0);time.sleep(30)
 assert json.loads((R/'output/TRAINING_FUNCTIONAL_RECEIPT.json').read_text())['status'].startswith('PASS');ready=json.loads((D/'DATASET_READY.json').read_text());assert ready['protected_reads']==0;dataset=ROIRoofDataset();feature=json.loads((D/'ALL_ROI_FEATURES_FROZEN.json').read_text())
 if not (T/'FEATURE_CACHE_READBACK.json').exists():
  def verify(row):
   assert hashlib.sha256(dataset.cache[row['slots']].tobytes()).hexdigest()==row['embedding_values_sha256'];return row['frame_id']
  with ThreadPoolExecutor(max_workers=8) as pool:verified=list(pool.map(verify,feature['records']))
  save('FEATURE_CACHE_READBACK.json',dict(status='PASS_ALL_FROZEN_FP32_FRAME_EMBEDDING_SHA',frames=len(verified),feature_freeze_sha256=sha(D/'ALL_ROI_FEATURES_FROZEN.json'),cache_GiB=dataset.cache.nbytes/1024**3))
 schedule=T/'FIXED_TRAINING_SCHEDULE.npz'
 if not schedule.exists():rng=np.random.default_rng(20261122);np.savez_compressed(schedule,indices=rng.integers(0,len(dataset),(2000,32),dtype=np.int64))
 plan=np.load(schedule)['indices'];sys.path.insert(0,str(A/'source'));from segment_anything import sam_model_registry
 lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');status('WAIT_EXCLUSIVE_GPU_NATIVE_ROI_TRAINING');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261122);sam=sam_model_registry['vit_b']();sam.load_state_dict(torch.load(A/'sam_vit_b_01ec64.pth',map_location='cpu',weights_only=True),strict=True);sam.cuda().eval();groups=configure(sam);initial={k:v.detach().cpu().clone() for k,v in sam.mask_decoder.state_dict().items()};save('ACTUAL_PARAMETER_CONFIGURATION.json',dict(trainable_parameters=sum(p.numel() for p in sam.parameters() if p.requires_grad),frozen_parameters=sum(p.numel() for p in sam.parameters() if not p.requires_grad),parameters=[dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in sam.named_parameters()],precision='FP32_NO_TF32',schedule_sha256=sha(schedule),feature_freeze_sha256=sha(D/'ALL_ROI_FEATURES_FROZEN.json')))
  benchmark=T/'THROUGHPUT_AND_GPU_BACKWARD.json'
  if not benchmark.exists():
   # Largest real crop sizes establish headroom, not synthetic enlarged labels.
   ii=sorted(range(len(dataset)),key=lambda i:dataset.slots[dataset.fit_slots[i]]['crop_native_xy_size'][2],reverse=True)[:32];batch=collate([dataset[i] for i in ii]);records=[]
   for micro in [4,8,16,32]:
    peaks=[];speeds=[];failed=False
    for arm,recall in ARMS:
     try:
      sam.mask_decoder.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();b=gpu_batch(batch,0,micro);sizes=batch['size'][:micro].tolist();torch.cuda.synchronize();start=time.time();value,_=loss_for(sam,b,sizes,recall);assert torch.isfinite(value);value.backward();assert all(torch.isfinite(p.grad).all() for p in sam.mask_decoder.parameters() if p.grad is not None);assert all(p.grad is None for p in sam.image_encoder.parameters());torch.cuda.synchronize();seconds=time.time()-start;peaks.append(torch.cuda.max_memory_allocated()/1024**3);speeds.append(micro/seconds);del value,b
     except torch.cuda.OutOfMemoryError as err:failed=True;records.append(dict(microbatch=micro,arm=arm,error=repr(err)));gc.collect();torch.cuda.empty_cache();break
    sam.mask_decoder.zero_grad(set_to_none=True)
    if not failed:records.append(dict(microbatch=micro,worst_peak_GiB=max(peaks),mean_queries_per_second=float(np.mean(speeds))))
   choice=max([r for r in records if 'mean_queries_per_second' in r and r['worst_peak_GiB']<16],key=lambda r:r['mean_queries_per_second']);save(benchmark.name,dict(status='PASS_REAL_VARIABLE_NATIVE_ROI_BOTH_LOSS_BACKWARD_ENCODER_NO_GRAD',records=records,selection=choice,actual_optimizer_updates=0,initial_decoder_reloaded_after=True));del batch
  micro=json.loads(benchmark.read_text())['selection']['microbatch'];sam.mask_decoder.load_state_dict(initial);sam.mask_decoder.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();evaluator=NativeROIEvaluator('INNER');save('INNER_RAW_BASELINE.json',evaluator.baseline)
  if not (T/'INITIAL_GENERIC_ROI_CONTROL.json').exists():
   control_dir=T/'INITIAL_GENERIC_ROI_INSTANCE_MAPS';control=evaluator.evaluate(sam,save_directory=control_dir);replay=[]
   for p in Path('/root/xian20261111_r15/output/roi_detail_pilot/frames').glob('*/INSTANCE_MAPS.npz'):
    current=np.load(control_dir/(p.parent.name+'.npz'));previous=np.load(p)
    for name in ['ROI_PRIOR_UNGUARDED','ROI_PRIOR_GUARDED']:
     a=current[name];b=previous[name];different=int((a!=b).sum());foreground=int(((a>0)|(b>0)).sum());assert different<=max(32,int(.001*foreground)),(p.parent.name,name,different,foreground);replay.append(dict(frame=p.parent.name,variant=name,different_native_pixels=different,union_foreground_pixels=foreground))
   save('GENERIC_ROI_PARENT_INTERFACE_REPLAY.json',dict(status='PASS_REAL_PRIOR_PILOT_VS_NEW_CACHE_NATIVE_MAP_REPLAY_WITH_DECLARED_BATCH_FLOAT_TOLERANCE',records=replay,allowed_difference='MAX32_OR0.1PCT_FOREGROUND_PIXELS;NOT_CLAIMED_BITEXACT',actual_optimizer_updates=0));save('INITIAL_GENERIC_ROI_CONTROL.json',dict(metrics=control,actual_optimizer_updates=0,same_interface_and_native_crops_as_training=True))
  history=json.loads((T/'HISTORY.json').read_text()) if (T/'HISTORY.json').exists() else [];choices={}
  for arm,recall in ARMS:
   directory=T/arm;directory.mkdir(exist_ok=True)
   if (directory/'TRAINING_RECEIPT.json').exists():choices[arm]=json.loads((directory/'TRAINING_RECEIPT.json').read_text());continue
   sam.mask_decoder.load_state_dict(initial);opt=torch.optim.AdamW(groups,weight_decay=.001);start_step=0;best=None;checkpoints=sorted(directory.glob('iter_*.pt'),key=lambda p:int(p.stem.split('_')[1]));complete=[p for p in checkpoints if p.with_name(p.stem+'_INNER.json').exists()]
   if complete:
    state=torch.load(complete[-1],map_location='cpu');assert state['training_schedule_sha256']==sha(schedule) and state['feature_freeze_sha256']==sha(D/'ALL_ROI_FEATURES_FROZEN.json');sam.mask_decoder.load_state_dict(state['decoder']);opt.load_state_dict(state['optimizer']);start_step=state['actual_optimizer_updates']
    for st in opt.state.values():
     for name,value in st.items():
      if isinstance(value,torch.Tensor) and name!='step':st[name]=value.cuda()
    if (directory/'best.pt').exists():best=torch.load(directory/'best.pt',map_location='cpu')['selection']
    history=[r for r in history if r['arm']!=arm or r['actual_optimizer_updates']<=start_step]
   subset=torch.utils.data.Subset(dataset,plan[start_step:].ravel().tolist());loader=torch.utils.data.DataLoader(subset,batch_size=32,shuffle=False,collate_fn=collate,num_workers=4,persistent_workers=True,pin_memory=True,prefetch_factor=2);iterator=iter(loader);actual=start_step
   for step in range(start_step+1,2001):
    begin_time=time.time();batch=next(iterator);wait=time.time()-begin_time;assert np.array_equal(batch['index'].numpy(),plan[step-1]);opt.zero_grad(set_to_none=True);sam.mask_decoder.train();parts=[];value_total=0.;factor=min(1.,step/100)*(.2 if step>=1400 else 1.)*(.2 if step>=1800 else 1.)
    for group in opt.param_groups:group['lr']=group['base_lr']*factor
    for begin in range(0,32,micro):
     end=min(32,begin+micro);b=gpu_batch(batch,begin,end);value,components=loss_for(sam,b,batch['size'][begin:end].tolist(),recall);assert torch.isfinite(value);(value*((end-begin)/32)).backward();parts.append(components);value_total+=float(value.detach())*((end-begin)/32);del value,b
    norm=torch.nn.utils.clip_grad_norm_([p for p in sam.mask_decoder.parameters() if p.requires_grad],5.);assert torch.isfinite(norm);opt.step();actual+=1
    if step%25==0:
     row=dict(arm=arm,actual_optimizer_updates=actual,loss=value_total,gradient_norm=float(norm),microbatch=micro,effective_batch=32,lr=opt.param_groups[0]['lr'],data_wait_seconds=wait,step_seconds=time.time()-begin_time,GT_crop_fraction_mean=float(batch['GT_crop_fraction'].mean()),GT_crop_below90_count=int((batch['GT_crop_fraction']<.9).sum()),**{name:float(np.mean([p[name] for p in parts])) for name in ['BCE','Tversky_loss','log_area_loss','boundary_loss','quality_MSE','known_raw_TP_keep_loss']});history.append(row);save('HISTORY.json',history);status('TRAIN_NATIVE_ROI_ROOF_DECODER',**row)
    if step in [250,500,1000,2000]:
     report=evaluator.evaluate(sam);eligible=[name for name,m in report.items() if passes(m,evaluator.baseline)];variant=max(eligible or report,key=lambda name:(report[name]['F1_50'],report[name]['boundary_F1_native2px']));selection=dict(arm=arm,step=step,variant=variant,metrics=report[variant],qualified_inner=variant in eligible)
     if best is None or selection_key(selection)>selection_key(best):best=selection;checkpoint(directory/'best.pt',sam,opt,actual,best,arm)
     checkpoint(directory/f'iter_{step}.pt',sam,opt,actual,selection,arm);save(f'{arm}/iter_{step}_INNER.json',dict(metrics=report,qualified_variants=eligible,selection=selection));status('NATIVE_ROI_INNER_VALIDATION',arm=arm,actual_optimizer_updates=actual,qualified_variants=eligible,metrics=report)
   del iterator,loader;gc.collect();choices[arm]=dict(arm=arm,actual_optimizer_updates=actual,selection=best,checkpoint=str(directory/'best.pt'),checkpoint_sha256=sha(directory/'best.pt'));save(f'{arm}/TRAINING_RECEIPT.json',choices[arm])
  chosen=max(choices.values(),key=lambda r:selection_key(r['selection']));save('ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json',dict(arms=choices,chosen=chosen,actual_total_AdamW_updates=sum(r['actual_optimizer_updates'] for r in choices.values()),TUNE_new_model_predictions_performed=False));state=torch.load(chosen['checkpoint'],map_location='cpu');assert sha(chosen['checkpoint'])==chosen['checkpoint_sha256'];sam.mask_decoder.load_state_dict(state['decoder']);sam.eval();status('PREPARE_SELECTED_SOURCE_TUNE_ROIS',actual_optimizer_updates=4000,chosen_arm=chosen['arm']);prepared=prepare_TUNE(sam);tune=NativeROIEvaluator('TUNE',prepared);report=tune.evaluate(sam,only=chosen['selection']['variant'],save_directory=T/'TUNE_SELECTED_INSTANCE_MAPS');m=report[chosen['selection']['variant']];qualified=bool(chosen['selection']['qualified_inner']) and passes(m,tune.baseline);save('TRAINING_RECEIPT.json',dict(status='COMPLETE_TWO_TRUE_NATIVE_ROI_ROOF_LOSS_ARMS',actual_AdamW_updates=sum(r['actual_optimizer_updates'] for r in choices.values()),arms=choices,chosen=chosen,TUNE_metrics=report,TUNE_raw_baseline=tune.baseline,qualified_source_research_candidate=qualified,source_TUNE_reused_not_XIAN_independent_accuracy=True,GT_map_not_native_roof_labels=True,no_GT_query_supervision=False,unknown_pixel_direct_gradient_zero_tested=True,protected_reads=0,production_changed=False));status('NATIVE_ROI_ROOF_TRAINING_COMPLETE',actual_optimizer_updates=4000,qualified=qualified,metrics=report)
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:main()
 except Exception as err:save('FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(err),traceback=traceback.format_exc()));raise
