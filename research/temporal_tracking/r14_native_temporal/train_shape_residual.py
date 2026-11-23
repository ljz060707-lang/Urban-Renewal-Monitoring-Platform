"""Matched bounded temporal shape residual, plus a stronger soft-shape rule control."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,traceback
import numpy as np,torch
from native_temporal_contract import O,Dataset,partial_selection_loss,save,sha
from train_bounded_residual import BoundedResidual
T=O/'pair_shape/training';T.mkdir(exist_ok=True);torch.set_num_threads(4)
PARENT=O/'training/GEOMETRY_ONLY/best.pt';THRESHOLDS=[-2.,-1.,0.,.5,1.,1.5,2.,3.,4.];WEIGHTS=[0.,.5,1.,2.,4.]
class ShapeDataset(Dataset):
 def __init__(self):
  proof=json.loads((O/'pair_shape/PREPARATION_RECEIPT.json').read_text());p=O/'pair_shape/CENTERED_SHAPE_FEATURES.npz';assert sha(p)==proof['features_sha256'];self.shape_feature=np.load(p)['shape_IoU'];super().__init__();assert self.shape_feature.shape==self.mask.shape
 def pack(self,ids,feature_enabled,training=False,rng=None):
  q,c,g,m,t,pos,known,enabled=super().pack(ids,False,training,rng)
  if feature_enabled:g[...,4]=torch.from_numpy(self.shape_feature[np.asarray(ids)])
  return q,c,g,m,t,pos,known,enabled
 def shape_scores(self,w,weight):
  s=self.baseline_scores(w,False)
  for q in w['queries']:
   i=q['query'];it=[p for p in ['xa180429','xa190101','xa201231'] if p!=self.nodes[i]['period']].index(q['target_period'])
   for k,j in enumerate(q['candidates']):s[(i,j)]+=weight*float(self.shape_feature[i,it,k])
  return s
def key(r):return r['IDF1'],-r['wrong_reference_direct_edges'],-r['identity_fragments'],r['AssA']
def select(ds,windows,scores):
 grid=[dict(threshold=x,metrics=ds.evaluate(windows,scores,x)[0]) for x in THRESHOLDS];return max(grid,key=lambda r:key(r['metrics'])),grid
def main():
 save(T/'PROTOCOL_FROZEN.json',dict(task='LABEL_BLIND_CENTERED_SHAPE_TEMPORAL_RANKING',arms=['GEOMETRY_RESIDUAL_CONTROL','GEOMETRY_PLUS_CENTERED_SHAPE_RESIDUAL'],feature='RAW_PREDICTED_MASK_CENTERED_IOU_REPLACES_PRIOR_COSINE_SLOT;NO_GT_SHAPES',pair_feature_sha256=sha(O/'pair_shape/CENTERED_SHAPE_FEATURES.npz'),parent_sha256=sha(PARENT),new_actual_updates_per_arm=2000,base='FROZEN_GEOMETRY_NET_EXACT_START_WITH_ZERO_BOUNDED_RESIDUAL',residual_bound=1.,optimizer='CPU_FLOAT32_ADAMW',lr=.0001,weight_decay=.001,batch_anchors=64,partial_label_loss='UNKNOWN_CANDIDATES_AND_QUERY_ROWS_NO_SUPERVISION',source='SAME_R14_FROZEN_FIT_INNER_AND_REUSED_TUNE',strong_control='GEOMETRY_PLUS_SOFT_PREDICTED_SHAPE',control_weights=WEIGHTS,thresholds=THRESHOLDS,all_observations_initialized=True,missing_edge='UNKNOWN_NOT_CANNOT_LINK',protected_reads=0,production_changed=False));ds=ShapeDataset();fit,inner,fitids,split=ds.split();parent=torch.load(PARENT,map_location='cpu');q,c,g,m,t,*_=ds.pack(fitids[:16],True);model=BoundedResidual(parent);actual=model(q,c,g,m,t);bg=g.clone();bg[...,4]=0;expected=model.base(q,c,bg,m,t);assert torch.equal(actual,expected);save(T/'INITIAL_EQUIVALENCE.json',dict(status='PASS_REAL_CANDIDATE_GEOMETRY_PRIOR_BITEXACT_WITH_SHAPE_BRANCH_ZERO_INIT',anchors=16,new_optimizer_updates=0));grid=[]
 for weight in WEIGHTS:
  scores=[ds.shape_scores(w,weight) for w in inner];selection,rows=select(ds,inner,scores)
  for r in rows:r['weight']=weight
  grid.extend(rows)
 rule=max(grid,key=lambda r:key(r['metrics'])+(-r['weight'],));save(T/'SOFT_SHAPE_CONTROL_FROZEN.json',dict(selection=rule,grid=grid));choices={};history=[]
 for arm,feature in [('GEOMETRY_RESIDUAL_CONTROL',False),('GEOMETRY_PLUS_CENTERED_SHAPE_RESIDUAL',True)]:
  D=T/arm;D.mkdir(exist_ok=True);torch.manual_seed(20261116);rng=np.random.default_rng(20261116);model=BoundedResidual(parent);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.001);best=None;best_step=0
  for step in range(1,2001):
   ids=rng.choice(fitids,64,replace=True);q,c,g,m,t,pos,known,en=ds.pack(ids,feature,True,rng);model.train();opt.zero_grad(set_to_none=True);v=model(q,c,g,m,t);reg=torch.cat([known,torch.ones_like(en[...,None])],-1)&en[...,None];loss=partial_selection_loss(v,pos,known,en)+.05*model.last_residual[reg].square().mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);assert torch.isfinite(norm);opt.step()
   if step%25==0:history.append(dict(arm=arm,actual_new_optimizer_updates=step,loss=float(loss.detach())));save(T/'HISTORY.json',history)
   if step%250==0:
    scores=[ds.model_scores(w,model,feature) for w in inner];selection,grid=select(ds,inner,scores);assert sorted(set(int(s['step']) for s in opt.state.values() if 'step' in s))==[step];state=dict(model=model.state_dict(),optimizer=opt.state_dict(),new_actual_optimizer_updates=step,feature=feature,selection=selection,parent_sha256=sha(PARENT));torch.save(state,D/f'iter_{step}.pt');save(D/f'iter_{step}_INNER.json',dict(selection=selection,grid=grid))
    if best is None or key(selection['metrics'])>key(best['metrics']):best=selection;best_step=step;torch.save(dict(model=model.state_dict(),new_actual_optimizer_updates=step,feature=feature,selection=selection),D/'best.pt')
    save(T/'STATUS.json',dict(time=time.time(),arm=arm,actual_new_optimizer_updates=step,selection=selection));print('SHAPE_RESIDUAL_INNER',arm,step,selection,flush=True)
  choices[arm]=dict(selection=best,best_step=best_step,feature=feature,new_actual_optimizer_updates=2000,best_sha256=sha(D/'best.pt'));save(D/'TRAINING_RECEIPT.json',dict(status='COMPLETE_NATIVE_SHAPE_RESIDUAL_ARM',**choices[arm]))
 save(T/'ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json',dict(rule=rule,models=choices));tune=[w for w in ds.windows if w['role']=='TUNE'];scores=[ds.shape_scores(w,rule['weight']) for w in tune];metrics,tracks=ds.evaluate(tune,scores,rule['threshold'],True);results={'SOFT_SHAPE_RULE':metrics};save(T/'SOFT_SHAPE_RULE_TUNE_TRACKS.json',tracks)
 for arm,choice in choices.items():
  state=torch.load(T/arm/'best.pt',map_location='cpu');model=BoundedResidual(parent);model.load_state_dict(state['model']);scores=[ds.model_scores(w,model,choice['feature']) for w in tune];results[arm],tracks=ds.evaluate(tune,scores,choice['selection']['threshold'],True);save(T/(arm+'_TUNE_TRACKS.json'),tracks)
 baseline=json.loads((O/'training/TRAINING_RECEIPT.json').read_text())['TUNE_results']['GEOMETRY_ONLY'];a=results['GEOMETRY_PLUS_CENTERED_SHAPE_RESIDUAL'];qualified=all(a['IDF1']>=b['IDF1']+.005 and a['wrong_reference_direct_edges']<=b['wrong_reference_direct_edges'] and a['identity_fragments']<=b['identity_fragments'] for b in [baseline,results['SOFT_SHAPE_RULE'],results['GEOMETRY_RESIDUAL_CONTROL']]);save(T/'TRAINING_RECEIPT.json',dict(status='COMPLETE_GT_FREE_SHAPE_TEMPORAL_MODEL_EXPERIMENT',new_actual_AdamW_updates=4000,choices=choices,rule=rule,TUNE_results=results,original_geometry_network_baseline=baseline,qualified_partial_label_research_model=qualified,GT_shapes_or_identity_groups_in_inference=False,full_scene_or_physical_ID_accuracy='NOT_ESTABLISHED',protected_reads=0,production_changed=False));print('SHAPE_RESIDUAL_COMPLETE',qualified,results,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
