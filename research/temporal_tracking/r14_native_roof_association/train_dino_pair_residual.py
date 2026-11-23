# -*- coding: utf-8 -*-
"""Learn when frozen DINO appearance helps, preserving the trained geometry prior."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import json,time,traceback
import numpy as np,torch
from torch import nn
from native_temporal_contract import O,Dataset,VariableSetMatcher,partial_selection_loss,save,sha
from train_bounded_residual import select,key
T=O/'dino_pair_residual';T.mkdir(exist_ok=True);PARENT=O/'training/GEOMETRY_ONLY/best.pt';torch.set_num_threads(4)
class DinoPairDataset(Dataset):
 def __init__(self):
  super().__init__();receipt=json.loads((O/'dino_identity/FEATURES_FROZEN_BEFORE_GT_LABEL_JOIN.json').read_text());bank=np.zeros((self.n,768),np.float32);present=np.zeros(self.n,bool)
  for row in receipt['chunks']:
   path=O/'dino_identity'/row['path'];assert sha(path)==row['sha256'];z=np.load(path);bank[z['indices']]=z['PREDICTED_ROOF_TOKEN_MEAN'];present[z['indices']]=True
  path=O/'pair_shape/CENTERED_SHAPE_FEATURES.npz';assert sha(path)==json.loads((O/'pair_shape/PREPARATION_RECEIPT.json').read_text())['features_sha256'];shape=np.load(path)['shape_IoU'];self.pair_feature=np.zeros((*self.mask.shape,2),np.float32)
  for start in range(0,self.n,256):
   end=min(self.n,start+256);self.pair_feature[start:end,...,0]=np.einsum('ik,itjk->itj',bank[start:end],bank[self.candidate[start:end]],optimize=False)
  self.pair_feature[...,1]=shape;self.present=present
  for w in self.windows:
   if w['role']=='TUNE':assert present[w['nodes']].all()
  save(T/'FEATURE_BINDING.json',dict(dino_feature_receipt_sha256=sha(O/'dino_identity/FEATURES_FROZEN_BEFORE_GT_LABEL_JOIN.json'),shape_features_sha256=sha(path),graph_sha256=sha(O/'CANDIDATE_GRAPH.json'),features_GT_dependency=False,feature_values_sha256=__import__('hashlib').sha256(self.pair_feature.tobytes()).hexdigest(),unknown_FIT_candidates_missing_features='NO_DIRECT_SUPERVISION;ALL_INNER_AND_TUNE_OBSERVATIONS_HAVE_FEATURES',protected_reads=0))
 def pack(self,ids,feature_enabled,training=False,rng=None):
  q,c,g,m,t,pos,known,en=super().pack(ids,False,training,rng);extra=self.pair_feature[np.asarray(ids)].copy()
  if not feature_enabled:extra.fill(0)
  if training:
   supervised=known.numpy()&en.numpy()[...,None];available=self.present[np.asarray(ids),None,None]&self.present[self.candidate[np.asarray(ids)]];assert available[supervised].all()
  return q,c,torch.cat([g,torch.from_numpy(extra)],-1),m,t,pos,known,en
class DinoPairResidual(nn.Module):
 def __init__(self,parent):
  super().__init__();self.base=VariableSetMatcher();self.base.load_state_dict(parent['model'])
  for p in self.base.parameters():p.requires_grad_(False)
  self.pair=nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1));nn.init.zeros_(self.pair[-1].weight);nn.init.zeros_(self.pair[-1].bias)
 def forward(self,q,c,g,m,t):
  with torch.no_grad():prior=self.base(torch.zeros_like(q),torch.zeros_like(c),g[...,:8],m,t)
  self.residual=2*self.pair(g)[...,0].tanh();return prior+torch.cat([self.residual,torch.zeros_like(self.residual[...,:1])],-1)
def main():
 save(T/'PROTOCOL_FROZEN.json',dict(task='FROZEN_DINO_AND_NATIVE_SHAPE_LEARNED_BOUNDED_PAIR_RESIDUAL',arms=['MATCHED_GEOMETRY_PAIR_RESIDUAL','DINO_SHAPE_PAIR_RESIDUAL'],new_updates_per_arm=2000,parent_path=str(PARENT),parent_sha256=sha(PARENT),base_geometry_frozen=True,dustbin_frozen=True,architecture='PAIR_MLP10_64_32_1;ZERO_FINAL_LAYER;2TANH_BOUND',features='8_SAME_GEOMETRY_FIELDS_WITH_OLD_COSINE_ZERO_PLUS_FROZEN_DINO_COS_PLUS_CENTERED_MODEL_MASK_IOU',optimizer='CPU_FP32_ADAMW',lr=.0002,weight_decay=.001,effective_anchor_batch=64,regularizer='0.01_RESIDUAL_SQUARED_ONLY_KNOWN_ENABLED_PAIRS',candidate_dropout='SAME_FROZEN_20PCT_SET_UNMATCHED_NOT_PHYSICAL_ABSENCE',split='SAME_2KM_BLOCK_600M_AND_REFERENCE_PURGE',unknown_loss='NO_GT_QUERY_AND_UNKNOWN_CANDIDATES_ZERO_DIRECT_GRADIENT',selection='TRAIN_INNER_ONLY_ALL_MODELS_THRESHOLDS_FROZEN_BEFORE_REUSED_TUNE',project_DINO_encoder_updates=0,protected_reads=0,production_changed=False));ds=DinoPairDataset();fit,inner,fitids,split=ds.split();save(T/'DATA_SPLIT_RECEIPT.json',split);parent=torch.load(PARENT,map_location='cpu');torch.manual_seed(20261120);model=DinoPairResidual(parent);q,c,g,m,t,pos,known,en=ds.pack(fitids[:16],True);v=model(q,c,g,m,t);expected=model.base(q,c,g[...,:8],m,t);assert torch.equal(v,expected);grad=torch.autograd.grad(partial_selection_loss(v,pos,known,en),v)[0];assert (grad[...,:-1][~(known&en[...,None])]==0).all();z=partial_selection_loss(v,pos,known,torch.zeros_like(en));gz=torch.autograd.grad(z,v)[0];assert (gz==0).all();save(T/'FUNCTIONAL_RECEIPT.json',dict(status='PASS_EXACT_GEOMETRY_WARM_START_AND_ACTUAL_UNKNOWN_LOGIT_DISABLED_QUERY_ZERO_DIRECT_GRADIENT',new_optimizer_updates=0));history=json.loads((T/'HISTORY.json').read_text()) if (T/'HISTORY.json').exists() else [];choices={}
 for arm,feature in [('MATCHED_GEOMETRY_PAIR_RESIDUAL',False),('DINO_SHAPE_PAIR_RESIDUAL',True)]:
  directory=T/arm;directory.mkdir(exist_ok=True)
  if (directory/'TRAINING_RECEIPT.json').exists():choices[arm]=json.loads((directory/'TRAINING_RECEIPT.json').read_text());continue
  torch.manual_seed(20261120);rng=np.random.default_rng(20261120);model=DinoPairResidual(parent);opt=torch.optim.AdamW(model.pair.parameters(),lr=.0002,weight_decay=.001);best=None;start=0;best_step=0;checkpoints=sorted(directory.glob('iter_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
  if checkpoints:
   state=torch.load(checkpoints[-1],map_location='cpu');model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng_state'];start=state['new_actual_optimizer_updates'];best=state['best_selection'];best_step=state['best_step'];history=[r for r in history if r['arm']!=arm or r['actual_new_optimizer_updates']<=start]
  for step in range(start+1,2001):
   ids=rng.choice(fitids,64,replace=True);q,c,g,m,t,pos,known,en=ds.pack(ids,feature,True,rng);opt.zero_grad(set_to_none=True);v=model(q,c,g,m,t);reg=known&en[...,None];loss=partial_selection_loss(v,pos,known,en)+.01*model.residual[reg].square().mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.pair.parameters(),5.);assert torch.isfinite(norm);opt.step()
   if step%25==0:history.append(dict(arm=arm,actual_new_optimizer_updates=step,loss=float(loss.detach()),gradient_norm=float(norm)));save(T/'HISTORY.json',history)
   if step%250==0:
    scores=[ds.model_scores(w,model,feature) for w in inner];selection,grid=select(ds,inner,scores);assert sorted(set(int(s['step']) for s in opt.state.values() if 'step' in s))==[step]
    if best is None or key(selection['metrics'])>key(best['metrics']):best=selection;best_step=step;torch.save(dict(model=model.state_dict(),new_actual_optimizer_updates=step,feature=feature,selection=best),directory/'best.pt')
    state=dict(model=model.state_dict(),optimizer=opt.state_dict(),rng_state=rng.bit_generator.state,new_actual_optimizer_updates=step,feature=feature,selection=selection,best_selection=best,best_step=best_step,parent_sha256=sha(PARENT));p=directory/f'iter_{step}.pt';tmp=p.with_suffix('.pending');torch.save(state,tmp);tmp.replace(p);save(directory/f'iter_{step}_INNER.json',dict(selection=selection,grid=grid));save(T/'STATUS.json',dict(time=time.time(),arm=arm,actual_new_optimizer_updates=step,selection=selection));print('DINO_PAIR_RESIDUAL_INNER',arm,step,selection,flush=True)
  choices[arm]=dict(best_step=best_step,selection=best,feature=feature,new_actual_optimizer_updates=2000,best_sha256=sha(directory/'best.pt'));save(directory/'TRAINING_RECEIPT.json',choices[arm])
 save(T/'ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json',choices);tune=[w for w in ds.windows if w['role']=='TUNE'];results={}
 for arm,choice in choices.items():
  state=torch.load(T/arm/'best.pt',map_location='cpu');model=DinoPairResidual(parent);model.load_state_dict(state['model']);scores=[ds.model_scores(w,model,choice['feature']) for w in tune];results[arm],tracks=ds.evaluate(tune,scores,choice['selection']['threshold'],True);save(T/(arm+'_TUNE_TRACKS.json'),tracks)
 controls={'ORIGINAL_GEOMETRY':json.loads((O/'training/TRAINING_RECEIPT.json').read_text())['TUNE_results']['GEOMETRY_ONLY'],'FROZEN_DINO':json.loads((O/'dino_identity/EVALUATION_RECEIPT.json').read_text())['TUNE_results']['FROZEN_DINO'],'MATCHED_CONTROL':results['MATCHED_GEOMETRY_PAIR_RESIDUAL']};a=results['DINO_SHAPE_PAIR_RESIDUAL'];qualified=all(a['IDF1']>=b['IDF1']+.005 and a['wrong_reference_direct_edges']<=b['wrong_reference_direct_edges'] and a['identity_fragments']<=b['identity_fragments'] for b in controls.values());save(T/'TRAINING_RECEIPT.json',dict(status='COMPLETE_LEARNED_DINO_SHAPE_BOUNDED_RESIDUAL',new_actual_AdamW_updates=4000,choices=choices,TUNE_results=results,controls=controls,qualified_partial_label_research_candidate=qualified,source_reference_geometrically_admitted_not_verified_physical_identity=True,independent_XIAN_roof_or_full_scene_accuracy='NOT_ESTABLISHED',protected_reads=0,production_changed=False));print('DINO_PAIR_RESIDUAL_COMPLETE',qualified,results,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as err:save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(err),traceback=traceback.format_exc()));raise
