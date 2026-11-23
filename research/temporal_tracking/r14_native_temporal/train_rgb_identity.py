"""Real cross-period weak-reference contrastive fine-tuning of roof appearance.

Uses admitted GT-known FIT RGB crops only for supervised loss. Evaluation uses
all observations in the frozen inner/TUNE graphs, including no-GT candidates.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,sys,hashlib,traceback,collections,fcntl,subprocess,gc,csv
import numpy as np,torch
from scipy.spatial import cKDTree
from native_temporal_contract import O,Dataset,save,sha
from rgb_identity_model import RoofAppearanceEncoder
sys.path.insert(0,str(Path(__file__).parent/'official_supcontrast'))
from losses import SupConLoss
D=O/'rgb_identity';T=D/'training';T.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(2)
LAMBDAS=[0.,.25,.5,1.,2.,4.];THRESHOLDS=[-2.,-1.,0.,.5,1.,1.5,2.,3.,4.]
def status(event,**kw):save(T/'STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def metric_key(r):return r['IDF1'],-r['wrong_reference_direct_edges'],-r['identity_fragments'],r['AssA']
def load_crops():
 manifest=json.loads((D/'CROP_MANIFEST.json').read_text());roles=json.loads((D/'CROP_ROLES_FROZEN_BEFORE_RGB_READ.json').read_text());selected=np.array(roles['selected_node_indices']);lookup=np.full(max(selected)+1,-1,np.int64);lookup[selected]=np.arange(len(selected));crops=np.empty((len(selected),4,128,128),np.uint8);read=0
 for r in manifest['records']:
  path=D/r['path'];assert sha(path)==r['sha256'];z=np.load(path);crops[lookup[z['indices']]]=z['crops'];read+=len(z['indices'])
 assert read==len(selected);return crops,lookup,roles

@torch.no_grad()
def embeddings(model,crops,lookup,indices,pretrained=False):
 out=np.zeros((len(lookup),512 if pretrained else 64),np.float32);model.eval()
 for begin in range(0,len(indices),256):
  ids=indices[begin:begin+256];assert (lookup[ids]>=0).all();x=torch.from_numpy(crops[lookup[ids]]).cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x,pretrained=pretrained)
  out[ids]=z.cpu().numpy()
 return out

def score_bank(ds,windows,z,weight):
 output=[]
 for w in windows:
  s=ds.baseline_scores(w,False)
  if weight:
   for q in w['queries']:
    i=q['query'];cc=q['candidates'];cos=z[cc]@z[i] if cc else []
    for j,v in zip(cc,cos):s[(i,j)]+=weight*float(v)
  output.append(s)
 return output

def calibrate(ds,windows,z):
 grid=[]
 for weight in LAMBDAS:
  scores=score_bank(ds,windows,z,weight)
  for threshold in THRESHOLDS:grid.append(dict(weight=weight,threshold=threshold,metrics=ds.evaluate(windows,scores,threshold)[0]))
 choice=max(grid,key=lambda r:metric_key(r['metrics'])+(-r['weight'],));return choice,grid

def main():
 assert json.loads((D/'CPU_MODEL_LOSS_SMOKE.json').read_text())['status'].startswith('PASS')
 method=Path(__file__).parent/'official_supcontrast';source=json.loads((method/'SOURCE_RECEIPT.json').read_text())
 for r in source['files']:assert sha(method/r['path'])==r['sha256']
 save(T/'PROTOCOL_FROZEN.json',dict(task='CROSS_PERIOD_NATIVE_PREDICTED_ROOF_APPEARANCE_CONTRASTIVE_TRAINING',paper='https://arxiv.org/abs/2004.11362',official_loss_commit=source['commit'],encoder='TORCHVISION_RESNET18_IMAGENET1K_V1',trainable='LAYER4_PLUS512_128_64_PROJECTION;BN_RUNNING_STATS_FROZEN',initial_weight_sha256=sha('/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'),input='128PX_PREDICTED_QUERY_CONTEXT_RGB;OUTSIDE_PREDICTED_MASK_RGB_MULTIPLIED0.25',GT_mask_or_coordinates_in_encoder=False,GT_supervision='KNOWN_FIT_GEOMETRICALLY_ADMITTED_MAP_REFERENCE_IDENTITIES_ONLY;DISTINCT_PERIOD_REAL_RGB_VIEWS',unknown_candidates='NO_SUPERVISED_NEGATIVES;ALL_AVAILABLE_IN_EVALUATION',loss='OFFICIAL_SUPCON_TEMPERATURE0.1_BASE_TEMPERATURE0.1',actual_AdamW_updates=2000,encoder_lr=.00001,projection_lr=.0003,weight_decay=.001,augmentation='INDEPENDENT_BRIGHTNESS0.8_1.2_CHANNEL0.9_1.1_NO_PRODUCT_GEOMETRY_WARP',batch_identities='FIT_ONLY_ACTUAL_GRADIENT_THROUGHPUT_BENCHMARK32_64_128;2_IMAGES_EACH;RELOAD_INITIAL_STATE_AFTER',negative_sampling='HALF_UNIFORM_IDENTITIES_HALF_DIFFERENT_KNOWN_REFERENCE_WITHIN64M_IF_AVAILABLE',selection='TRAIN_INNER_COMPLETE_CANDIDATE_GRAPH_PARTIAL_REFERENCE_IDF1_WRONG_EDGES_FRAGMENTS',cosine_weights=LAMBDAS,utility_thresholds=THRESHOLDS,TUNE='REUSED_DEVELOPMENT_EVALUATED_ONLY_AFTER_ENCODER_AND_WEIGHT_THRESHOLD_FROZEN',GPU_concurrency='EXCLUSIVE_EXISTING_R10_GPU_INFERENCE_LOCK_AFTER_NATIVE56_TRAIN',physical_identity_or_independent_roof_accuracy='NOT_ESTABLISHED',protected_reads=0,production_changed=False))
 crops,lookup,roles=load_crops();ds=Dataset();fit,inner,_,split=ds.split();assert {w['site'] for w in inner}==set(roles['inner_sites']);fitids=roles['fit_supervised_indices'];groups=collections.defaultdict(lambda:collections.defaultdict(list))
 for i in fitids:
  assert ds.known[i] and ds.nodes[i]['role']=='TRAIN' and ds.nodes[i]['site'] in set(roles['fit_sites']);groups[ds.gold[i]][ds.nodes[i]['period']].append(i)
 groups={g:p for g,p in groups.items() if len(p)>=2};refs=sorted(groups);assert len(refs)>=128;centers=np.array([np.mean([ds.position[i] for rr in groups[g].values() for i in rr],axis=0) for g in refs]);tree=cKDTree(centers);neighbors=tree.query_ball_tree(tree,64);ref_index={g:i for i,g in enumerate(refs)};fit_set=set(fitids);assert all(ds.gold[i] not in {ds.gold[j] for w in inner for j in w['nodes'] if ds.known[j]} for i in fitids)
 def sample(n,rng):
  chosen=list(rng.choice(len(refs),n//2,replace=False));pool=list(set(j for i in chosen for j in neighbors[i])-set(chosen));rng.shuffle(pool);chosen+=pool[:n-len(chosen)]
  if len(chosen)<n:chosen+=list(rng.choice(sorted(set(range(len(refs)))-set(chosen)),n-len(chosen),replace=False))
  pairs=[]
  for j in chosen:
   g=refs[j];periods=rng.choice(sorted(groups[g]),2,replace=False);pair=[int(rng.choice(groups[g][p])) for p in periods];assert ds.nodes[pair[0]]['actual_date_known'] and ds.nodes[pair[1]]['actual_date_known'] and ds.nodes[pair[0]]['actual_date']!=ds.nodes[pair[1]]['actual_date'];pairs.append(pair)
  return np.array(pairs,np.int64)
 save(T/'FIT_PAIR_ROLE_RECEIPT.json',dict(reference_identities=len(refs),eligible_fit_nodes=len(fitids),inner_nodes=sum(len(w['nodes']) for w in inner),TUNE_nodes=sum(len(w['nodes']) for w in ds.windows if w['role']=='TUNE'),crop_rows=len(crops),all_pair_members_GT_known=True,unknown_supervised_negatives=0,protected_reads=0));status('WAIT_EXCLUSIVE_GPU_NATIVE_INSTANCE_HAS_PRIORITY')
 lock=Path('/root/xian20261111_r10/output/serial_readonly_evaluation/GPU_INFERENCE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  while True:
   free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if free>=12288:break
   status('WAIT_GPU_HEADROOM_AFTER_EXCLUSIVE_LOCK',free_MiB=free);time.sleep(10)
  criterion=SupConLoss(.1,base_temperature=.1);benchmark=[];torch.manual_seed(20261115);model=RoofAppearanceEncoder().cuda().train();rng=np.random.default_rng(0)
  for n in [32,64,128]:
   pairs=sample(n,rng);x=torch.from_numpy(crops[lookup[pairs.ravel()]]).cuda();model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.time()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x)
   loss=criterion(z.reshape(n,2,64),torch.arange(n,device='cuda'));assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize();seconds=time.time()-start;peak=torch.cuda.max_memory_allocated()/1024**3;benchmark.append(dict(identities=n,images=2*n,seconds=seconds,images_per_second=2*n/seconds,peak_GiB=peak));del x,z,loss;model.zero_grad(set_to_none=True)
  choice=max([r for r in benchmark if r['peak_GiB']<12],key=lambda r:r['images_per_second']);n=choice['identities'];save(T/'THROUGHPUT_BENCHMARK.json',dict(results=benchmark,selection=choice,optimizer_updates=0,initial_state_reloaded_after_benchmark=True));del model;gc.collect();torch.cuda.empty_cache();torch.manual_seed(20261115);rng=np.random.default_rng(20261115);model=RoofAppearanceEncoder().cuda();inner_ids=[i for w in inner for i in w['nodes']];z=embeddings(model,crops,lookup,inner_ids,True);baseline,grid=calibrate(ds,inner,z);save(T/'PRETRAINED512_TRAIN_INNER_SELECTION.json',dict(selection=baseline,grid=grid));del z
  opt=torch.optim.AdamW([dict(params=model.rgb.layer4.parameters(),lr=1e-5),dict(params=model.project.parameters(),lr=3e-4)],weight_decay=.001);history=[];best=None;best_step=0;start=time.time()
  for step in range(1,2001):
   model.train();pairs=sample(n,rng);x=torch.from_numpy(crops[lookup[pairs.ravel()]]).cuda();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x,augment=True)
   loss=criterion(z.reshape(n,2,64),torch.arange(n,device='cuda'));assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);opt.step();del x,z
   if step%25==0:
    row=dict(actual_optimizer_updates=step,loss=float(loss.detach()),elapsed_seconds=time.time()-start);history.append(row);save(T/'HISTORY.json',history);status('TRAIN_NATIVE_RGB_CONTRASTIVE',batch_identities=n,**row)
   if step%250==0:
    actual=sorted(set(int(v['step']) for v in opt.state.values() if 'step' in v));assert actual==[step];z=embeddings(model,crops,lookup,inner_ids);selection,grid=calibrate(ds,inner,z);del z;state=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),actual_optimizer_updates=step,parameter_step_values=actual,selection=selection,batch_identities=n,crop_manifest_sha256=sha(D/'CROP_MANIFEST.json'));torch.save(state,T/f'iter_{step}.pt');save(T/f'iter_{step}_INNER.json',dict(selection=selection,grid=grid))
    if best is None or metric_key(selection['metrics'])>metric_key(best['metrics']):best=selection;best_step=step;torch.save(dict(model=state['model'],actual_optimizer_updates=step,selection=selection),T/'best.pt')
    status('RGB_CONTRASTIVE_INNER',actual_optimizer_updates=step,selection=selection)
  save(T/'ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json',dict(best_step=best_step,trained_selection=best,pretrained_selection=baseline,best_checkpoint_sha256=sha(T/'best.pt'),TUNE_evaluation_performed=False));tune=[w for w in ds.windows if w['role']=='TUNE'];tune_ids=[i for w in tune for i in w['nodes']];results={}
  # Actual ImageNet baseline reloaded, not the fine-tuned encoder's 512-D features.
  torch.manual_seed(20261115);initial=RoofAppearanceEncoder().cuda();z=embeddings(initial,crops,lookup,tune_ids,True);scores=score_bank(ds,tune,z,baseline['weight']);results['IMAGENET512']=ds.evaluate(tune,scores,baseline['threshold'],True)[0];del initial,z;gc.collect();torch.cuda.empty_cache();state=torch.load(T/'best.pt',map_location='cpu');model.load_state_dict(state['model']);z=embeddings(model,crops,lookup,tune_ids);scores=score_bank(ds,tune,z,best['weight']);results['CONTRASTIVE64'],tracks=ds.evaluate(tune,scores,best['threshold'],True);save(T/'CONTRASTIVE64_TUNE_TRACKS.json',tracks);np.savez_compressed(T/'CONTRASTIVE64_TUNE_FEATURES.npz',indices=np.array(tune_ids),features=z[tune_ids]);parent=json.loads((O/'training/TRAINING_RECEIPT.json').read_text());geometry=parent['TUNE_results']['GEOMETRY'];a=results['CONTRASTIVE64'];qualified=all(a['IDF1']>=b['IDF1']+.005 and a['wrong_reference_direct_edges']<=b['wrong_reference_direct_edges'] and a['identity_fragments']<=b['identity_fragments'] for b in [geometry,results['IMAGENET512']]);save(T/'TRAINING_RECEIPT.json',dict(status='COMPLETE_NATIVE_RGB_APPEARANCE_CONTRASTIVE',actual_AdamW_updates=2000,best_step=best_step,trained_selection=best,pretrained_selection=baseline,TUNE_results=results,geometry_baseline=geometry,qualified_partial_label_research_candidate=qualified,all_TUNE_nodes=len(tune_ids),partial_label_scored_nodes=399,GT_coordinates_or_identity_groups_in_inference=False,GT_missing_supervised_negative=0,protected_reads=0,physical_identity_accuracy='NOT_ESTABLISHED',production_changed=False));status('RGB_CONTRASTIVE_COMPLETE',qualified=qualified,results=results)
  with (T/'REAL_OPTIMIZER_CURVES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
 finally:
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:main()
 except Exception as e:save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
