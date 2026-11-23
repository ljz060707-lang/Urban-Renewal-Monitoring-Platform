"""Frozen FP32 DINOv2 on actual native query ROIs, then TRAIN-inner calibration.

No identity labels/GT geometry in feature extraction. All TUNE observations are
retained. This is a generic pretrained representation control, not target training.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,traceback,gc,fcntl,itertools,multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np,torch
from native_temporal_contract import O,Dataset,save,sha
from dino_model import load_model,extract,D as MODEL_DIR
T=O/'dino_identity';T.mkdir(exist_ok=True);torch.set_num_threads(4)
THRESHOLDS=[-2.,-1.,0.,.5,1.,1.5,2.,3.,4.];SHAPE_WEIGHTS=[0.,1.,2.];APPEARANCE_WEIGHTS=[0.,.5,1.,2.,4.]
DS=None;INNER=None;BANK=None;SHAPE=None
def status(event,**kw):save(T/'STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def key(r):return r['IDF1'],-r['wrong_reference_direct_edges'],-r['identity_fragments'],r['AssA']
def scores(w,mode,sw,aw):
 out=DS.baseline_scores(w,False);features=BANK[mode]
 for q in w['queries']:
  i=q['query'];cc=q['candidates'];it=[p for p in ['xa180429','xa190101','xa201231'] if p!=DS.nodes[i]['period']].index(q['target_period']);cos=features[cc]@features[i] if cc and aw else np.zeros(len(cc))
  for k,j in enumerate(cc):out[(i,j)]+=float(sw*SHAPE[i,it,k]+aw*cos[k])
 return out
def calibrate_task(task):
 mode,sw,aw=task;s=[scores(w,mode,sw,aw) for w in INNER];return [dict(mode=mode,shape_weight=sw,appearance_weight=aw,threshold=t,metrics=DS.evaluate(INNER,s,t)[0]) for t in THRESHOLDS]

def main():
 global DS,INNER,BANK,SHAPE
 assert json.loads((MODEL_DIR/'CPU_ADMISSION.json').read_text())['status'].startswith('PASS')
 save(T/'PROTOCOL_FROZEN.json',dict(task='FROZEN_GENERIC_DINOV2_VITB14_NATIVE_QUERY_ASSOCIATION',weights_sha256=json.loads((MODEL_DIR/'SOURCE_RECEIPT.json').read_text())['weights']['sha256'],source_commit=json.loads((MODEL_DIR/'SOURCE_RECEIPT.json').read_text())['commit'],paper='https://arxiv.org/abs/2304.07193',feature_modes=['CLS_CONTEXT','PREDICTED_ROOF_TOKEN_MEAN'],native_ROI=128,encoder_input=224,resize='FEATURE_INPUT_ONLY_NO_NEW_SENSOR_DETAIL',RGB='UNATTENUATED_ACTUAL_NATIVE_ROI',mask='MODEL_PREDICTED_QUERY_MASK_ONLY',precision='FP32_NO_TF32_NO_AUTOCAST',candidate_graph='SAME_FROZEN_ALL_MODEL_OBSERVATION_GRAPH',feature_SHA_before_reference_labels=True,model_optimization_updates=0,shape_weights=SHAPE_WEIGHTS,appearance_weights=APPEARANCE_WEIGHTS,thresholds=THRESHOLDS,calibration='FROZEN_R14_TRAIN_INNER_COMPLETE_GRAPH_PARTIAL_REFERENCE_IDF1',TUNE='REUSED_DEVELOPMENT_ONCE_AFTER_ALL_FEATURE_MODE_AND_WEIGHTS_THRESHOLDS_FROZEN',protected_reads=0,production_changed=False))
 manifest=json.loads((O/'rgb_identity/CROP_MANIFEST.json').read_text());roles=json.loads((O/'rgb_identity/CROP_ROLES_FROZEN_BEFORE_RGB_READ.json').read_text());selected=np.array(roles['selected_node_indices'],np.int64);lookup=np.full(int(selected.max())+1,-1,np.int64);lookup[selected]=np.arange(len(selected));crops=np.empty((len(selected),4,128,128),np.uint8)
 for r in manifest['records']:
  path=O/'rgb_identity'/r['path'];assert sha(path)==r['sha256'];z=np.load(path);crops[lookup[z['indices']]]=z['crops']
 chunks=T/'feature_chunks';chunks.mkdir(exist_ok=True);chunk_receipts=[];torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');status('WAIT_SHARED_EXCLUSIVE_GPU_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  model=load_model().cuda().eval();benchmark=[]
  for n in [8,16,32,64]:
   torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();x=torch.from_numpy(crops[:n]).cuda();extract(model,x);torch.cuda.synchronize();start=time.time()
   for _ in range(2):extract(model,x)
   torch.cuda.synchronize();seconds=(time.time()-start)/2;benchmark.append(dict(batch=n,seconds=seconds,images_per_second=n/seconds,peak_GiB=torch.cuda.max_memory_allocated()/1024**3));del x
  choice=max([r for r in benchmark if r['peak_GiB']<14],key=lambda r:r['images_per_second']);batch=choice['batch'];save(T/'FP32_THROUGHPUT_BENCHMARK.json',dict(results=benchmark,selection=choice,optimizer_updates=0));status('EXTRACT_FROZEN_DINO_FP32',batch=batch,total_ROIs=len(selected))
  for start in range(0,len(selected),2048):
   indices=selected[start:start+2048];p=chunks/f'chunk_{start:06d}.npz';receipt=p.with_suffix('.json')
   if receipt.exists():
    row=json.loads(receipt.read_text());assert sha(p)==row['sha256'];chunk_receipts.append(row);continue
   cls=[];roof=[]
   for b in range(0,len(indices),batch):
    ii=indices[b:b+batch];x=torch.from_numpy(crops[lookup[ii]]).cuda();a,c=extract(model,x);cls.append(a.cpu().numpy());roof.append(c.cpu().numpy())
   cls=np.concatenate(cls);roof=np.concatenate(roof);assert np.isfinite(cls).all() and np.isfinite(roof).all();tmp=p.with_suffix('.pending')
   with tmp.open('wb') as f:np.savez_compressed(f,indices=indices,CLS_CONTEXT=cls,PREDICTED_ROOF_TOKEN_MEAN=roof)
   tmp.replace(p);row=dict(path=str(p.relative_to(T)),sha256=sha(p),indices=len(indices),first_input_position=start,GT_input=False,precision='FP32');save(receipt,row);chunk_receipts.append(row);status('DINO_FP32_CHUNK_COMPLETE',rows=min(start+2048,len(selected)),total=len(selected))
  del model,x,crops;gc.collect();torch.cuda.empty_cache();save(T/'FEATURES_FROZEN_BEFORE_GT_LABEL_JOIN.json',dict(chunks=chunk_receipts,rows=len(selected),embedding_dimension=768,modes=['CLS_CONTEXT','PREDICTED_ROOF_TOKEN_MEAN'],crop_manifest_sha256=sha(O/'rgb_identity/CROP_MANIFEST.json'),model_sha256=sha(MODEL_DIR/'dinov2_vitb14_pretrain.pth'),GT_shape_or_identity_input=False,new_optimizer_updates=0))
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 # Only after feature freeze, connect separate reference identity labels for calibration.
 DS=Dataset();INNER=[w for w in DS.windows if w['role']=='TRAIN' and w['site'] in set(roles['inner_sites'])];BANK={m:np.zeros((DS.n,768),np.float32) for m in ['CLS_CONTEXT','PREDICTED_ROOF_TOKEN_MEAN']};SHAPE=np.load(O/'pair_shape/CENTERED_SHAPE_FEATURES.npz')['shape_IoU'];proof=json.loads((O/'pair_shape/PREPARATION_RECEIPT.json').read_text());assert sha(O/'pair_shape/CENTERED_SHAPE_FEATURES.npz')==proof['features_sha256']
 for r in chunk_receipts:
  z=np.load(T/r['path'])
  for mode in BANK:BANK[mode][z['indices']]=z[mode]
 tasks=list(itertools.product(BANK,SHAPE_WEIGHTS,APPEARANCE_WEIGHTS));expected=calibrate_task(tasks[0]);grid=[];status('CALIBRATE_DINO_ON_TRAIN_INNER_CPU12',combinations=len(tasks))
 with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('fork')) as pool:
  for i,rows in enumerate(pool.map(calibrate_task,tasks)):
   if i==0:assert rows==expected
   grid.extend(rows)
   if i%5==0:status('DINO_INNER_GRID',completed=i+1,total=len(tasks))
 selection=max(grid,key=lambda r:key(r['metrics'])+(-r['appearance_weight'],-r['shape_weight']));baseline=max([r for r in grid if r['appearance_weight']==0],key=lambda r:key(r['metrics'])+(-r['shape_weight'],));save(T/'PARALLEL_METRIC_EQUIVALENCE.json',dict(status='PASS_NUMPY_SCIPY_SINGLE_PROCESS_AND_FORK_POOL_BITEXACT',single_and_pool_rows=9,protected_reads=0));save(T/'TRAIN_INNER_SELECTION_FROZEN.json',dict(selection=selection,soft_shape_baseline=baseline,grid=grid,TUNE_evaluation_performed=False));tune=[w for w in DS.windows if w['role']=='TUNE'];results={}
 for name,config in [('FROZEN_DINO',selection),('SOFT_SHAPE_CONTROL',baseline)]:
  s=[scores(w,config['mode'],config['shape_weight'],config['appearance_weight']) for w in tune];results[name],tracks=DS.evaluate(tune,s,config['threshold'],True);save(T/(name+'_TUNE_TRACKS.json'),tracks)
 original=json.loads((O/'training/TRAINING_RECEIPT.json').read_text())['TUNE_results']['GEOMETRY_ONLY'];a=results['FROZEN_DINO'];qualified=all(a['IDF1']>=b['IDF1']+.005 and a['wrong_reference_direct_edges']<=b['wrong_reference_direct_edges'] and a['identity_fragments']<=b['identity_fragments'] for b in [original,results['SOFT_SHAPE_CONTROL']]);save(T/'EVALUATION_RECEIPT.json',dict(status='COMPLETE_FROZEN_GENERIC_DINOV2_NATIVE_ASSOCIATION',selection=selection,soft_shape_control_selection=baseline,TUNE_results=results,original_geometry_network=original,qualified_partial_label_research_candidate=qualified,all_TUNE_nodes=2523,scored_reference_labels=399,optimizer_updates=0,GT_shape_or_identity_group_input=False,protected_reads=0,formal_physical_ID_accuracy='NOT_ESTABLISHED',production_changed=False));status('DINO_NATIVE_ASSOCIATION_COMPLETE',qualified=qualified,results=results)
if __name__=='__main__':
 try:main()
 except Exception as e:save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
