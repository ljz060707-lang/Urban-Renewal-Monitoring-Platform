"""Frozen DINO scores -> GT-free track hypotheses -> TRAIN-inner global calibration."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,traceback,collections,itertools,multiprocessing
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from native_temporal_contract import O,Dataset,score_tracks,save,sha
from temporal_hypergraph import hypotheses,optimize
import evaluate_dino_identity as appearance
T=O/'hypergraph';T.mkdir(exist_ok=True);WEIGHTS=[0.,.25,.5,1.];PAYLOADS=None;NODES=None;LABELS=None
def status(event,**kw):save(T/'STATUS.json',dict(time=time.time(),event=event,**kw));print(event,kw,flush=True)
def reference_relations(tracks,labels):
 same=wrong=0
 for track in tracks:
  known=[labels[str(i)]['reference_id'] for i in track if str(i) in labels]
  for a,b in itertools.combinations(known,2):same+=a==b;wrong+=a!=b
 return dict(known_reference_within_track_pairs=same+wrong,wrong_reference_within_track_pairs=wrong)
def summarize(rows):
 s=collections.Counter()
 for r in rows:s.update(r['metrics']);s.update(reference_relations(r['tracks'],LABELS));s['solver_fallbacks']+=not r['solver']['optimal']
 n=s['scored_nodes'];return dict(s,IDF1=s['IDTP']/n if n else None,AssA=s['AssA_sum']/n if n else None,label_coverage=n/max(1,s['nodes']),map_reference_partial_label_development=True,physical_identity_accuracy='NOT_ESTABLISHED',all_nodes_retained=True)
def task(arg):
 key,weight=arg;p=PAYLOADS[key];tracks,solver=optimize(p,NODES,weight,5.);metrics=score_tracks(p['indices'],tracks,LABELS);return dict(key=key,cycle_weight=weight,tracks=tracks,solver=solver,metrics=metrics)
def baseline_rows(keys):return [dict(key=k,tracks=PAYLOADS[k]['baseline_tracks'],solver=dict(optimal=True,status='FROZEN_EXACT_DINO_PATH_BASELINE'),metrics=score_tracks(PAYLOADS[k]['indices'],PAYLOADS[k]['baseline_tracks'],LABELS)) for k in keys]
def rank(m):return m['IDF1'],-m['wrong_reference_within_track_pairs'],-m['identity_fragments'],m['AssA']
def main():
 global PAYLOADS,NODES,LABELS
 assert json.loads((T/'FUNCTIONAL_TEST.json').read_text())['status'].startswith('PASS');source=O/'dino_identity';receipt=json.loads((source/'EVALUATION_RECEIPT.json').read_text());c={k:receipt['selection'][k] for k in ['mode','shape_weight','appearance_weight','threshold']};NODES=json.loads((O/'OBSERVATIONS_ALL.json').read_text());windows=json.loads((O/'CANDIDATE_GRAPH.json').read_text());roles=json.loads((O/'rgb_identity/CROP_ROLES_FROZEN_BEFORE_RGB_READ.json').read_text());selected_windows=[w for w in windows if w['role']=='TUNE' or w['role']=='TRAIN' and w['site'] in set(roles['inner_sites'])];save(T/'PROTOCOL_FROZEN.json',dict(task='THREE_PERIOD_DINO_POSITIVE_SUPPORT_TREE_PLUS_SOFT_CYCLE_HYPERGRAPH',DINO_score_config=c,DINO_receipt_sha256=sha(source/'EVALUATION_RECEIPT.json'),pair_threshold='UNCHANGED_FROZEN_DINO_THRESHOLD_NO_LOWERING',pair_hypothesis='BIDIRECTIONAL_MIN_MARGIN_GT0',triple='AT_LEAST_TWO_SUCH_POSITIVE_SUPPORTS_WITH_DISTINCT_PERIODS;MISSING_THIRD_EVIDENCE_ZERO_UNKNOWN',score='MAX_POSITIVE_SPANNING_TREE_PLUS_CYCLE_WEIGHT_TIMES_REMAINING_SIGNED_MARGIN',cycle_weights=WEIGHTS,beam_per_node_per_other_period=3,all_old_baseline_tracks_remain_hypotheses=True,solver='BINARY_SET_PACKING_ONE_HYPOTHESIS_PER_OBSERVATION_PLUS_EXPLICIT_SINGLETONS',per_window_solver_seconds=5.,solver_failure='FALLBACK_ORIGINAL_EXACT_DINO_TRACKS_NEVER_EXPORT_UNPROVED_PARTIAL_ASSIGNMENT',calibration='TRAIN_INNER_ONLY;HYPOTHESES_SHA_FROZEN_BEFORE_IDENTITY_LABEL_FILE_READ',TUNE='REUSED_DEVELOPMENT_SELECTED_WEIGHT_ONCE_NOT_BLIND',changed_comparison='ALL_KNOWN_WITHIN_TRACK_REFERENCE_DISCORDANT_PAIRS_COMMON_TO_ALL_CONTROLS;NOT_MIXED_WITH_OLD_DIRECT_EDGE_COUNTS',all_model_nodes_initialized=True,protected_reads=0,optimizer_updates=0,production_changed=False))
 model=SimpleNamespace(nodes=NODES,position=np.array([n['center_utm'] for n in NODES]),area=np.array([n['area_feature_m2'] for n in NODES]),raw_unit=None);model.baseline_scores=lambda w,app=False:Dataset.baseline_scores(model,w,False);appearance.DS=model;appearance.BANK={c['mode']:np.zeros((len(NODES),768),np.float32)};appearance.SHAPE=np.load(O/'pair_shape/CENTERED_SHAPE_FEATURES.npz')['shape_IoU'];feature_receipt=json.loads((source/'FEATURES_FROZEN_BEFORE_GT_LABEL_JOIN.json').read_text())
 for r in feature_receipt['chunks']:
  p=source/r['path'];assert sha(p)==r['sha256'];z=np.load(p);appearance.BANK[c['mode']][z['indices']]=z[c['mode']]
 PAYLOADS={};bindings=[]
 for i,w in enumerate(selected_windows):
  key=w['role']+'_'+w['site'];scores=appearance.scores(w,c['mode'],c['shape_weight'],c['appearance_weight']);payload=hypotheses(w['nodes'],NODES,scores,c['threshold'],3);payload.update(role=w['role'],site=w['site']);path=T/'hypotheses'/(key+'.json');save(path,payload);PAYLOADS[key]=payload;bindings.append(dict(key=key,path=str(path.relative_to(T)),sha256=sha(path),nodes=len(w['nodes']),hypotheses=len(payload['hypotheses'])));
  if i%25==0:status('GT_FREE_HYPOTHESES',windows=i+1,total=len(selected_windows))
 save(T/'HYPOTHESES_FROZEN_BEFORE_GT_JOIN.json',dict(records=bindings,GT_identity_labels_read=False,GT_geometry_or_coordinates_in_generation=False));del appearance.BANK,appearance.SHAPE
 # Only after inference hypotheses freeze, open GT identity labels for scoring.
 LABELS=json.loads((O/'SUPERVISION_SEPARATE.json').read_text())['labels'];inner_keys=[k for k,p in PAYLOADS.items() if p['role']=='TRAIN'];tune_keys=[k for k,p in PAYLOADS.items() if p['role']=='TUNE'];inner_base=summarize(baseline_rows(inner_keys));tune_base=summarize(baseline_rows(tune_keys));assert tune_base['IDTP']==receipt['TUNE_results']['FROZEN_DINO']['IDTP'] and tune_base['identity_fragments']==receipt['TUNE_results']['FROZEN_DINO']['identity_fragments'];save(T/'BASELINE_REPLAY_EQUIVALENCE.json',dict(status='PASS_RECONSTRUCTED_FROZEN_DINO_TRACK_METRICS',inner=inner_base,TUNE=tune_base,known_reference_conflict_comparison='COMMON_WITHIN_TRACK_PAIRS'))
 tasks=[(k,a) for a in WEIGHTS for k in inner_keys];rows=[];status('TRAIN_INNER_GLOBAL_CALIBRATION_CPU12',jobs=len(tasks))
 with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('fork')) as pool:
  for i,r in enumerate(pool.map(task,tasks)):
   rows.append(r)
   if i%25==0:status('HYPERGRAPH_INNER_SOLVED',jobs=i+1,total=len(tasks))
 reports=[]
 for a in WEIGHTS:
  rr=[r for r in rows if r['cycle_weight']==a];m=summarize(rr);eligible=m['solver_fallbacks']==0 and m['IDF1']>=inner_base['IDF1'] and m['wrong_reference_within_track_pairs']<=inner_base['wrong_reference_within_track_pairs'] and m['identity_fragments']<=inner_base['identity_fragments'];reports.append(dict(cycle_weight=a,metrics=m,qualified_inner=eligible))
 eligible=[r for r in reports if r['qualified_inner']];choice=max(eligible or reports,key=lambda r:rank(r['metrics']));save(T/'TRAIN_INNER_WEIGHT_FROZEN.json',dict(selection=choice,selection_qualified=bool(eligible),grid=reports,TUNE_new_solver_evaluation_performed=False));save(T/'TRAIN_INNER_PREDICTIONS.json',rows);status('HYPERGRAPH_WEIGHT_FROZEN',selection=choice)
 with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('fork')) as pool:tune_rows=list(pool.map(task,[(k,choice['cycle_weight']) for k in tune_keys]))
 metrics=summarize(tune_rows);controls={'FROZEN_DINO':tune_base}
 for name,path in [('GEOMETRY_NETWORK',O/'training/GEOMETRY_ONLY_TUNE_TRACKS.json'),('SOFT_SHAPE_CONTROL',source/'SOFT_SHAPE_CONTROL_TUNE_TRACKS.json')]:
  raw=json.loads(path.read_text());rr=[dict(key=w['role']+'_'+w['site'],tracks=w['tracks'],solver=dict(optimal=True),metrics=score_tracks([i for t in w['tracks'] for i in t],w['tracks'],LABELS)) for w in raw['windows']];controls[name]=summarize(rr)
 qualified=bool(eligible) and metrics['solver_fallbacks']==0 and all(metrics['IDF1']>=b['IDF1']+.005 and metrics['wrong_reference_within_track_pairs']<=b['wrong_reference_within_track_pairs'] and metrics['identity_fragments']<=b['identity_fragments'] for b in controls.values());save(T/'TUNE_TRACKS.json',dict(windows=[dict(role='TUNE',site=PAYLOADS[r['key']]['site'],tracks=r['tracks'],solver=r['solver']) for r in tune_rows],research_support='TWO_POSITIVE_FROZEN_MODEL_LINKS_PLUS_SOFT_CYCLE;MISSING_DIRECT_EDGE_RESEARCH_INFERENCE_NOT_INDEPENDENT_ID_CONFIRMATION',all_observations_retained=True));save(T/'EVALUATION_RECEIPT.json',dict(status='COMPLETE_NATIVE_DINO_SOFT_CYCLE_GLOBAL_ASSOCIATION',selection=choice,TUNE_metrics=metrics,controls=controls,qualified_partial_label_research_candidate=qualified,new_optimizer_updates=0,protected_reads=0,GT_shapes_or_groups_in_inference=False,independent_physical_ID_or_full_scene_accuracy='NOT_ESTABLISHED',production_changed=False));status('HYPERGRAPH_EXPERIMENT_COMPLETE',qualified=qualified,metrics=metrics)
if __name__=='__main__':
 try:main()
 except Exception as e:save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
