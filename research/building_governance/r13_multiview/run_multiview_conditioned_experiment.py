"""All fit/tuning is TRAIN-inner; TUNE once, conditional reference correspondence."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import json,time,hashlib,collections,copy,csv,traceback
import numpy as np
from pyproj import Transformer
from affine import Affine
from multiview_reference_model import prepare,fit,predict,summary,inner_holdout
R=Path('/root/xian20261111_r13');O=R/'output';O.mkdir(parents=True,exist_ok=True);input_path=Path('/root/xian20261111_r8/output/rgb_alignment/DATASET_FROZEN.json')
def save(name,r):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
save('PROTOCOL_FROZEN.json',dict(task='REFERENCE_ASSOCIATED_CONDITIONAL_MULTIVIEW_POSITION',input=str(input_path),input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),source_roles=['TRAIN','TUNE'],identity_assistance='PROVIDED_GEOMETRICALLY_ADMITTED_MAP_REFERENCE_GROUP_NOT_VERIFIED_AUTOMATIC_TRACKING',GT_reference_coordinates='TRAIN_FIT_AND_EVALUATION_ONLY_NEVER_PREDICTION_INPUT',source_raster_reads=0,protected_reads=0,actual_dates='SOURCE_LEDGER_DAYS_NO_STRICT_YEAR_REQUIREMENT',duplicate_same_day='MEDIAN_POSITION_NOT_EXTRA_VIEW',reference_semantics='MAP_REFERENCE_NOT_CONFIRMED_NATIVE_ROOF_OR_PHYSICAL_FOOTPRINT',method='ROBUST_GLOBAL_OR_AFFINE_DATE_FIELD_PLUS_RANK1_RESIDUAL_FACTOR_NOT_TOMASI_KANADE_SFM',primary_source='https://publications.ri.cmu.edu/shape-and-motion-from-image-streams-under-orthography-a-factorization-method',factor='NOT_PHYSICAL_HEIGHT',inner_selection='TRAIN_REFERENCE_GROUPS_SPLIT_BY_HASHED1KM_OBSERVED_POSITION_BLOCK',TUNE='REUSED_DEVELOPMENT_EVALUATED_ONCE_AFTER_SELECTION_NOT_BLIND',hyperparameters=dict(spatial=[False,True],lambda_h=[1.,10.,100.],modes=['corrected','consensus'],ALS_iterations=40,Huber_delta_m=3.,max_translation_m=25.),production_changed=False))
try:
 raw=json.loads(input_path.read_text());assert raw['protected_reads']==0;tr={};rows=[]
 for r in raw['rows']:
  if r['role'] not in ['TRAIN','TUNE']:continue
  if r['native_crs'] not in tr:tr[r['native_crs']]=Transformer.from_crs(r['native_crs'],32649,always_xy=True)
  t=tr[r['native_crs']];a=Affine(*r['native_affine']);row={k:r[k] for k in ['id','role','reference_id','reference_source','patch_id','actual_date','date_dominance']};row['p']=list(t.transform(*(a*tuple(r['query_center_native']))));row['g']=list(t.transform(*(a*tuple(r['target_center_native']))));rows.append(row)
 train=[r for r in rows if r['role']=='TRAIN'];tune=[r for r in rows if r['role']=='TUNE'];assert len(train)==9504 and len(tune)==399;assert not ({r['reference_id'] for r in train}&{r['reference_id'] for r in tune});train_tracks=prepare(train);hold=[t for t in train_tracks if inner_holdout(t)];inner=[t for t in train_tracks if not inner_holdout(t)];inner_rows=[r for t in hold for r in t['rows']];save('DATA_ROLE_RECEIPT.json',dict(TRAIN_rows=len(train),TRAIN_tracks=len(train_tracks),TRAIN_inner_fit_tracks=len(inner),TRAIN_inner_dev_tracks=len(hold),TUNE_rows=len(tune),TUNE_tracks=len(prepare(tune)),GT_reference_coordinate_rows_read=len(rows),raw_image_reads=0,protected_reads=0,reference_overlap=0))
 candidates=[]
 for spatial in [False,True]:
  m0=fit(inner,spatial,False);baseline={}
  for mode in ['corrected','consensus']:
   pred,_=predict(hold,m0,mode);baseline[mode]=summary(inner_rows,pred)
  for lam in [1.,10.,100.]:
   model=fit(inner,spatial,True,lam)
   for mode in ['corrected','consensus']:
    pred,_=predict(hold,model,mode);scores=summary(inner_rows,pred);ok=True
    for group in ['all','manual','public']:
     s=scores[group]
     if not s['n']:continue
     for b in baseline.values():
      if s['mean_error_m']>b[group]['mean_error_m']+.01 or s['P95_error_m']>b[group]['P95_error_m']+.01:ok=False
    candidates.append(dict(spatial=spatial,lambda_h=lam,mode=mode,inner_scores=scores,inner_baselines=baseline,qualified_inner=ok));print('INNER_MULTIVIEW',spatial,lam,mode,scores['all'],ok,flush=True)
 eligible=[c for c in candidates if c['qualified_inner']];choice=min(eligible or candidates,key=lambda c:c['inner_scores']['all']['mean_error_m']);save('TRAIN_INNER_SELECTION.json',dict(candidates=candidates,selection=choice,selection_qualified=bool(eligible),TUNE_not_read_for_selection=True))
 # Freeze choice and refit TRAIN before first TUNE model prediction.
 model=fit(train_tracks,choice['spatial'],True,choice['lambda_h']);baseline_model=fit(train_tracks,choice['spatial'],False);save('MODEL_PARAMETERS.json',dict(model=model,selection=choice,TRAIN_only_fit=True,GT_group_assistance_required=True));save('BASELINE_PARAMETERS.json',baseline_model)
 tune_tracks=prepare(tune);outputs={};reports={}
 methods=[('RAW',None,None),('DATE_FIELD',baseline_model,'corrected'),('TEMPORAL_DATE_FIELD_MEAN',baseline_model,'consensus'),('TEMPORAL_RAW_MEAN',baseline_model,'raw_mean'),('SELECTED_FACTOR',model,choice['mode'])]
 for name,m,mode in methods:
  pred={r['id']:np.asarray(r['p']) for r in tune} if m is None else predict(tune_tracks,m,mode)[0];outputs[name]=pred;reports[name]=summary(tune,pred)
 poisoned=copy.deepcopy(tune)
 for r in poisoned:r['g']=[1e8,-1e8]
 poisoned_pred=predict(prepare(poisoned),model,choice['mode'])[0];assert all(np.array_equal(poisoned_pred[k],outputs['SELECTED_FACTOR'][k]) for k in poisoned_pred)
 # Inference invariance is a functional test, not model accuracy evidence.
 save('PREDICTION_GT_COORDINATE_POISON_TEST.json',dict(status='PASS_PREDICTIONS_INVARIANT_TO_POISONED_TUNE_REFERENCE_COORDINATES',rows=len(tune),GT_group_membership_still_provided=True,physical_height_measured=False))
 chosen=reports['SELECTED_FACTOR'];qualified=bool(eligible)
 for group in ['all','manual','public']:
  if not chosen[group]['n']:continue
  for name in ['RAW','DATE_FIELD','TEMPORAL_DATE_FIELD_MEAN','TEMPORAL_RAW_MEAN']:
   b=reports[name][group]
   if chosen[group]['mean_error_m']>b['mean_error_m']+.01 or chosen[group]['P95_error_m']>b['P95_error_m']+.01:qualified=False
 query=[]
 for r in tune:
  query.append(dict(**r,predictions={k:v[r['id']].tolist() for k,v in outputs.items()}))
 save('TUNE_QUERY_PREDICTIONS.json',dict(rows=query,conditional_GT_reference_groups=True,TUNE_coordinates_not_used_for_prediction=True));save('TRAINING_RECEIPT.json',dict(status='COMPLETE_CONDITIONAL_MULTIVIEW_MAP_REFERENCE_EXPERIMENT',TRAIN_reference_rows=len(train),TUNE_queries=len(tune),inner_selection=choice,inner_qualified=bool(eligible),TUNE_reports=reports,qualified_centroid_research_candidate=qualified,GT_correspondence_assistance=True,automatic_tracking_accuracy='NOT_ESTABLISHED',map_reference_not_independent_roof_accuracy=True,requires_IoU_area_and_automatic_identity_gate=True,deep_optimizer_updates=0,coordinate_descent_iterations=40,source_pixel_reads=0,protected_reads=0,production_changed=False));print('MULTIVIEW_CONDITIONAL_COMPLETE',qualified,reports,flush=True)
 with (O/'FIT_CURVES.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(model['history'][0]));w.writeheader();w.writerows(model['history'])
except Exception as e:save('FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
