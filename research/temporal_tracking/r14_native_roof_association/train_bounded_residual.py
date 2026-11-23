"""Do frozen semantic features fail, or does naive concatenation destabilize geometry?

Matched controls preserve the same trained geometric matcher exactly at start.
Only a zero-initialized, bounded residual is fitted, with an actual loss on
GT-known queries/candidates. This is a separate experiment, not a new blind test.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
import torch,numpy as np,json,time,copy,traceback
from torch import nn
from native_temporal_contract import O,Dataset,VariableSetMatcher,partial_selection_loss,save,sha
torch.set_num_threads(4);T=O/'bounded_residual';T.mkdir(exist_ok=True);GRID=[-2.,-1.,0.,.5,1.,1.5,2.,3.,4.]
PARENT=O/'training/GEOMETRY_ONLY/best.pt'
class BoundedResidual(nn.Module):
 def __init__(self,parent):
  super().__init__();self.base=VariableSetMatcher();self.base.load_state_dict(parent['model'])
  for p in self.base.parameters():p.requires_grad_(False)
  self.pair=nn.Sequential(nn.Linear(264,64),nn.SiLU(),nn.Linear(64,1));self.null=nn.Sequential(nn.Linear(130,64),nn.SiLU(),nn.Linear(64,1));nn.init.zeros_(self.pair[-1].weight);nn.init.zeros_(self.pair[-1].bias);nn.init.zeros_(self.null[-1].weight);nn.init.zeros_(self.null[-1].bias);self.last_residual=None
 def forward(self,q,c,g,mask,dates):
  bg=g.clone();bg[...,4]=0
  with torch.no_grad():prior=self.base(torch.zeros_like(q),torch.zeros_like(c),bg,mask,dates)
  qq=q[:,None,None].expand_as(c);z=torch.cat([qq,c,abs(qq-c),qq*c,g],-1);pair=self.pair(z)[...,0].tanh();pooled=(c*mask[...,None]).sum(2)/mask.sum(2)[...,None].clamp_min(1);dust=self.null(torch.cat([q[:,None].expand(q.shape[0],2,64),pooled,dates],-1))[...,0].tanh();self.last_residual=torch.cat([pair,dust[...,None]],-1);return prior+self.last_residual
def key(r):return r['IDF1'],-r['wrong_reference_direct_edges'],-r['identity_fragments'],r['AssA']
def select(ds,inner,scores):
 grid=[dict(threshold=v,metrics=ds.evaluate(inner,scores,v)[0]) for v in GRID];return max(grid,key=lambda x:key(x['metrics'])),grid
def main():
 parent=torch.load(PARENT,map_location='cpu');save(T/'PROTOCOL_FROZEN.json',dict(task='EXACT_GEOMETRY_WARM_START_BOUNDED_FEATURE_RESIDUAL',parent_path=str(PARENT),parent_sha256=sha(PARENT),parent_actual_optimizer_updates=parent['actual_optimizer_updates'],new_updates_per_arm=2000,arms=['GEOMETRY_RESIDUAL','GEOMETRY_PLUS_SEMANTIC_RESIDUAL'],residual='PAIR_MLP264_64_1_AND_DUSTBIN_MLP130_64_1_ZERO_INITIAL_FINAL_LINEAR_PLUS_TANH_BOUND1',base='FROZEN_GEOMETRY_MATCHER_NO_FEATURE_INPUT',regularizer='0.05_RESIDUAL_SQUARED_ON_KNOWN_ENABLED_LOGITS_ONLY',optimizer='CPU_FLOAT32_ADAMW',lr=.0001,weight_decay=.001,batch_anchors=64,partial_GT_loss='SAME_FROZEN_R14_CONTRACT_UNKNOWN_LOGITS_AND_QUERIES_NO_SUPERVISION',fit_inner='SAME_FROZEN_R14_600M_AND_REFERENCE_PURGE',selection='TRAIN_INNER_FULL_GRAPH_PARTIAL_REFERENCE_IDF1_THEN_WRONG_EDGES_AND_FRAGMENTS',TUNE='REUSED_DEVELOPMENT_NOT_BLIND_NO_RETROSPECTIVE_OLD_BEST_RESELECTION',protected_reads=0,production_changed=False));ds=Dataset();fit,inner,fitids,_=ds.split();q,c,g,m,t,*_=ds.pack(fitids[:16],True);torch.manual_seed(20261116);check=BoundedResidual(parent).eval();out=check(q,c,g,m,t);bg=g.clone();bg[...,4]=0;expected=check.base(torch.zeros_like(q),torch.zeros_like(c),bg,m,t);assert torch.equal(out,expected);save(T/'WARM_START_EQUIVALENCE.json',dict(status='PASS_INITIAL_REAL_CANDIDATE_LOGITS_BITEXACT_TO_FROZEN_GEOMETRY',anchors=16,optimizer_updates=0,GT_query_geometry_inserted=False));history=[];choices={}
 for arm,feature in [('GEOMETRY_RESIDUAL',False),('GEOMETRY_PLUS_SEMANTIC_RESIDUAL',True)]:
  D=T/arm;D.mkdir(exist_ok=True);torch.manual_seed(20261116);rng=np.random.default_rng(20261116);model=BoundedResidual(parent);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.001);best=None;best_step=0
  for step in range(1,2001):
   model.train();ids=rng.choice(fitids,64,replace=True);q,c,g,m,t,pos,known,enabled=ds.pack(ids,feature,True,rng);opt.zero_grad(set_to_none=True);logits=model(q,c,g,m,t);loss=partial_selection_loss(logits,pos,known,enabled);regmask=torch.cat([known,torch.ones_like(enabled[...,None])],-1)&enabled[...,None];loss=loss+.05*model.last_residual[regmask].square().mean();assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);assert torch.isfinite(norm);opt.step()
   if step%25==0:history.append(dict(arm=arm,actual_new_optimizer_updates=step,loss=float(loss.detach())));save(T/'HISTORY.json',history)
   if step%250==0:
    scores=[ds.model_scores(w,model,feature) for w in inner];choice,grid=select(ds,inner,scores);steps=sorted(set(int(v['step']) for v in opt.state.values() if 'step' in v));assert steps==[step];state=dict(model=model.state_dict(),optimizer=opt.state_dict(),new_actual_optimizer_updates=step,parent_actual_optimizer_updates=parent['actual_optimizer_updates'],selection=choice,feature=feature);torch.save(state,D/f'iter_{step}.pt');save(D/f'iter_{step}_INNER.json',dict(selection=choice,grid=grid))
    if best is None or key(choice['metrics'])>key(best['metrics']):best=choice;best_step=step;torch.save(dict(model=model.state_dict(),new_actual_optimizer_updates=step,selection=choice,feature=feature),D/'best.pt')
    print('BOUNDED_RESIDUAL_INNER',arm,step,choice,flush=True);save(T/'STATUS.json',dict(time=time.time(),arm=arm,actual_new_optimizer_updates=step,selection=choice))
  choices[arm]=dict(best_step=best_step,selection=best,feature=feature,new_actual_optimizer_updates=2000,best_sha256=sha(D/'best.pt'));save(D/'TRAINING_RECEIPT.json',dict(status='COMPLETE_BOUNDED_RESIDUAL_ARM',**choices[arm]))
 save(T/'ALL_SELECTIONS_FROZEN_BEFORE_TUNE.json',choices);tune=[w for w in ds.windows if w['role']=='TUNE'];results={}
 for arm,choice in choices.items():
  state=torch.load(T/arm/'best.pt',map_location='cpu');model=BoundedResidual(parent);model.load_state_dict(state['model']);scores=[ds.model_scores(w,model,choice['feature']) for w in tune];results[arm],tracks=ds.evaluate(tune,scores,choice['selection']['threshold'],True);save(T/(arm+'_TUNE_TRACKS.json'),tracks)
 base=json.loads((O/'training/TRAINING_RECEIPT.json').read_text())['TUNE_results'];a=results['GEOMETRY_PLUS_SEMANTIC_RESIDUAL'];qualified=all(a['IDF1']>=b['IDF1']+.005 and a['wrong_reference_direct_edges']<=b['wrong_reference_direct_edges'] and a['identity_fragments']<=b['identity_fragments'] for b in [base['GEOMETRY'],base['GEOMETRY_ONLY'],results['GEOMETRY_RESIDUAL']]);save(T/'TRAINING_RECEIPT.json',dict(status='COMPLETE_FROZEN_GEOMETRY_BOUNDED_RESIDUAL',new_actual_AdamW_updates=4000,parent_actual_optimizer_updates=parent['actual_optimizer_updates'],choices=choices,TUNE_results=results,qualified_partial_label_research_candidate=qualified,independent_physical_ID_accuracy='NOT_ESTABLISHED',protected_reads=0,production_changed=False));print('BOUNDED_RESIDUAL_COMPLETE',qualified,results,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(T/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
