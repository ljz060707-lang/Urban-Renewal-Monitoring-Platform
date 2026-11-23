"""Two actual CPU AdamW arms; A10 remains dedicated to native instance training.

Frozen 64-D native roof features, geometry and actual dates. Source GT consists
only of map-reference association metadata; no missing labels become absence.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,copy,traceback,hashlib,csv
import numpy as np,torch
from native_temporal_contract import Dataset,VariableSetMatcher,partial_selection_loss,save,O,sha
T=O/'training';T.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(2)
THRESHOLDS=[-2.,-1.,0.,.5,1.,1.5,2.,3.,4.]
def status(event,**kw):save(T/'STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def key(r):return (r['IDF1'],-r['wrong_reference_direct_edges'],-r['identity_fragments'],r['AssA'])
def select(ds,windows,scores):
 rows=[]
 for threshold in THRESHOLDS:rows.append(dict(threshold=threshold,metrics=ds.evaluate(windows,scores,threshold)[0]))
 return max(rows,key=lambda r:key(r['metrics'])),rows
def actual_steps(opt):return sorted(set(int(v['step']) for v in opt.state.values() if 'step' in v))
def checkpoint(p,model,opt,step,feature_enabled,ds,selection):
 steps=actual_steps(opt);assert max(steps)==step
 torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),actual_optimizer_updates=step,parameter_step_values=steps,feature_enabled=feature_enabled,normalization_mean=ds.mean,normalization_std=ds.std,inner_selection=selection,architecture='VariableSetMatcher64_H96_BiGRU_PLUS_PARTIAL_LABEL_LOSS',candidate_graph_sha256=sha(O/'CANDIDATE_GRAPH.json'),partial_GT_labels_not_model_input=True,physical_identity_accuracy='NOT_ESTABLISHED'),p)
def main():
 assert json.loads((O/'FUNCTIONAL_CONTRACT_TEST.json').read_text())['status'].startswith('PASS')
 save(T/'PROTOCOL_FROZEN.json',dict(task='ACTUAL_NATIVE_PREDICTED_ROOF_ASSOCIATION_WITH_WEAK_MAP_REFERENCE_LABELS',arms=['GEOMETRY_ONLY','GEOMETRY_AND_FROZEN_ROOF64'],architecture='PRIOR_TASKALIGNED_VARIABLE_SET_MATCHER64_H96_BIDIRECTIONAL_GRU; NO_OLD_CLASSIFIER_WEIGHT_LOAD',max_candidates=32,targets_per_anchor=2,optimizer='CPU_FLOAT32_AdamW',actual_updates_per_arm=2000,batch_anchors=64,lr=.0003,weight_decay=.001,gradient_clip=5.,candidate_dropout=.2,dropout='UNIFORM_LABEL_BLIND_PER_CANDIDATE_TRAIN_AUGMENTATION_LINEAGE_NOT_PHYSICAL_CHANGE',loss='SET_SELECTION_MASK_UNKNOWN_CANDIDATE_LOGITS_AND_NO_GT_QUERY_ROWS; DUSTBIN_ONLY_IF_LABELLED_COUNTERPART_OBSERVED_ELSEWHERE_OR_DROPPED',unknown_context='VISIBLE_FROZEN_IMAGE_CONTEXT_NO_FALSE_NEGATIVE_TARGET',partial_labels='GEOMETRICALLY_ADMITTED_MAP_REFERENCE_NOT_VERIFIED_PHYSICAL_IDENTITY',fit_inner='TRAIN_2KM_SPATIAL_BLOCK_HASH_PLUS_600M_CENTRE_PURGE_PLUS_REFERENCE_SUPERVISION_PURGE',normalization='KNOWN_FIT_NODES_ONLY',selection='COMPLETE_ACTUAL_CANDIDATE_GRAPH_PARTIAL_LABEL_IDF1_THEN_WRONG_REFERENCE_EDGES_AND_FRAGMENTS_ON_TRAIN_INNER',TUNE='REUSED_DEVELOPMENT_ONCE_AFTER_ALL_PARAMETERS_THRESHOLDS_FROZEN',thresholds=THRESHOLDS,solver='EXACT_MAX_WEIGHT_FORWARD_PATH_COVER_BY_BIPARTITE_ASSIGNMENT_WITH_NULL_EXITS;NO_EXPLICIT_CANNOT_LINKS_IN_INPUT',GPU_usage='RESERVED_FOR_SIMULTANEOUS_NATIVE_R10_CNN',presence='UNSUPERVISED_NO_PHYSICAL_ABSENCE_INFERENCE',protected_reads=0,production_changed=False,legacy_model_source_sha256=sha(Path(__file__).parent/'legacy_set_matcher.py')))
 ds=Dataset();fit,inner,fitids,split=ds.split();assert inner;save(T/'SPATIAL_ROLE_FREEZE.json',dict(**split,fit_sites=[w['site'] for w in fit],inner_sites=[w['site'] for w in inner],TUNE_reference_overlap=0,actual_graph_sha256=sha(O/'CANDIDATE_GRAPH.json')));status('DATA_READY',**split)
 baselines={};inner_base={}
 for name,app in [('GEOMETRY',False),('GEOMETRY_COSINE',True)]:
  scores=[ds.baseline_scores(w,app) for w in inner];selection,grid=select(ds,inner,scores);baselines[name]=dict(selection=selection,grid=grid);inner_base[name]=selection['metrics']
 save(T/'TRAIN_INNER_BASELINES_FROZEN.json',baselines);history=[];arm_choices={}
 for arm,feature_enabled in [('GEOMETRY_ONLY',False),('GEOMETRY_AND_FROZEN_ROOF64',True)]:
  D=T/arm;D.mkdir(exist_ok=True);torch.manual_seed(20261114);rng=np.random.default_rng(20261114);model=VariableSetMatcher().cpu();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001);best=None;best_step=0;elapsed=time.time()
  for step in range(1,2001):
   model.train();ids=rng.choice(fitids,size=64,replace=True);q,c,g,mask,dates,positive,known,enabled=ds.pack(ids,feature_enabled,True,rng);opt.zero_grad(set_to_none=True);logits=model(q,c,g,mask,dates);loss=partial_selection_loss(logits,positive,known,enabled);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
   if step%25==0:
    r=dict(arm=arm,actual_optimizer_updates=step,loss=float(loss.detach()),elapsed_seconds=time.time()-elapsed);history.append(r);save(T/'HISTORY.json',history);status('TRAIN_NATIVE_TEMPORAL_CPU',**r)
   if step%250==0:
    scores=[ds.model_scores(w,model,feature_enabled) for w in inner];selection,grid=select(ds,inner,scores);save(D/f'iter_{step}_INNER.json',dict(selection=selection,grid=grid));checkpoint(D/f'iter_{step}.pt',model,opt,step,feature_enabled,ds,selection)
    if best is None or key(selection['metrics'])>key(best['metrics']):best=selection;best_step=step;torch.save(dict(model=model.state_dict(),actual_optimizer_updates=step,feature_enabled=feature_enabled,normalization_mean=ds.mean,normalization_std=ds.std,inner_selection=selection),D/'best.pt')
    status('NATIVE_TEMPORAL_INNER',arm=arm,actual_optimizer_updates=step,threshold=selection['threshold'],metrics=selection['metrics'])
  arm_choices[arm]=dict(best_step=best_step,selection=best,feature_enabled=feature_enabled,best_sha256=sha(D/'best.pt'),actual_optimizer_updates=2000,final_actual_parameter_step_values=actual_steps(opt));save(D/'TRAINING_RECEIPT.json',dict(status='COMPLETE_REAL_NATIVE_TEMPORAL_ARM',**arm_choices[arm],protected_reads=0))
 # Freeze both arms, checkpoints and each threshold before development TUNE evaluation.
 save(T/'ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json',dict(baselines=baselines,arms=arm_choices,TUNE_evaluation_performed=False));tune=[w for w in ds.windows if w['role']=='TUNE'];results={}
 for name,app in [('GEOMETRY',False),('GEOMETRY_COSINE',True)]:
  scores=[ds.baseline_scores(w,app) for w in tune];report,tracks=ds.evaluate(tune,scores,baselines[name]['selection']['threshold'],True);results[name]=report;save(T/(name+'_TUNE_TRACKS.json'),tracks)
 for arm,choice in arm_choices.items():
  state=torch.load(T/arm/'best.pt',map_location='cpu');model=VariableSetMatcher().cpu();model.load_state_dict(state['model']);scores=[ds.model_scores(w,model,choice['feature_enabled']) for w in tune];report,tracks=ds.evaluate(tune,scores,choice['selection']['threshold'],True);results[arm]=report;save(T/(arm+'_TUNE_TRACKS.json'),tracks);save(T/(arm+'_TUNE_DIRECTION_SCORES.json'),[dict(role=w['role'],site=w['site'],scores=[[i,j,v] for (i,j),v in s.items()]) for w,s in zip(tune,scores)])
 # Label-only oracle is explicitly separate; never used as a model or an inference input.
 oracle=[]
 for w in tune:
  oracle.append({(q['query'],j):1. if ds.known[q['query']] and ds.known[j] and ds.gold[q['query']]==ds.gold[j] else -1. for q in w['queries'] for j in q['candidates']})
 results['PERFECT_KNOWN_REFERENCE_ORACLE']=ds.evaluate(tune,oracle,0)[0];qualified=[]
 for arm in arm_choices:
  a=results[arm]
  if all(a['IDF1']>=results[b]['IDF1']+.005 and a['wrong_reference_direct_edges']<=results[b]['wrong_reference_direct_edges'] and a['identity_fragments']<=results[b]['identity_fragments'] for b in ['GEOMETRY','GEOMETRY_COSINE']):qualified.append(arm)
 save(T/'TRAINING_RECEIPT.json',dict(status='COMPLETE_NATIVE_TEMPORAL_ACTUAL_MODEL_GRAPH_EXPERIMENT',arms=arm_choices,total_actual_AdamW_updates=4000,TUNE_results=results,qualified_partial_label_research_arms=qualified,automatic_physical_identity_accuracy='NOT_ESTABLISHED',requires_native_visual_ID_and_position_and_city_inference_gates=True,full_scene_recall_or_HOTA='NOT_ESTABLISHED_PARTIAL_LABEL_COVERAGE',source_GT_pixels_read=0,protected_reads=0,production_changed=False));status('NATIVE_TEMPORAL_EXPERIMENT_COMPLETE',qualified=qualified,results=results)
 with (T/'REAL_OPTIMIZER_CURVES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
if __name__=='__main__':
 try:main()
 except Exception as e:save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
