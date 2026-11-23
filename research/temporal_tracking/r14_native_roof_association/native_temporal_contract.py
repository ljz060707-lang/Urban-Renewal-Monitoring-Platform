"""Actual native model observations, partial-label loss and exact forward path cover."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,hashlib,datetime,collections
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from legacy_set_matcher import VariableSetMatcher

PERIODS=['xa180429','xa190101','xa201231']
R=Path('/root/xian20261111_r14_native_temporal');O=R/'output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)

def partial_selection_loss(logits,positive,known_candidate,enabled):
 """Unknown candidates have zero supervised-logit gradient; rows without GT disabled."""
 assert logits.shape[-1]==positive.shape[-1]+1
 if not enabled.any():return logits[...,-1].sum()*0
 masked=torch.cat([logits[...,:-1].masked_fill(~known_candidate,-float('inf')),logits[...,-1:]],-1)
 dustbin=~positive.any(-1);allowed=torch.cat([positive,dustbin[...,None]],-1)
 logp=masked[enabled].log_softmax(-1)
 return -logp.masked_fill(~allowed[enabled],-float('inf')).logsumexp(-1).mean()

def path_cover(indices,nodes,edge_utilities):
 """Exact maximum-weight forward path cover by bipartite assignment, null columns.

 No qualified cannot-links are supplied here. Periods have strictly nonoverlapping
 registered-date bounds. Unknown direct long edges do not forbid a two-edge path.
 """
 n=len(indices);where={i:j for j,i in enumerate(indices)}
 if n==0:return [],[]
 utility=np.full((n,2*n),-1e6,np.float64);utility[:,n:]=0
 for (i,j),value in edge_utilities.items():
  if value<=0:continue
  assert PERIODS.index(nodes[i]['period'])<PERIODS.index(nodes[j]['period'])
  utility[where[i],where[j]]=value
 rr,cc=linear_sum_assignment(-utility);successor={};predecessor={};accepted=[]
 for a,b in zip(rr,cc):
  if b<n and utility[a,b]>0:
   i,j=indices[a],indices[b];successor[i]=j;predecessor[j]=i;accepted.append((i,j,float(utility[a,b])))
 tracks=[]
 for i in indices:
  if i in predecessor:continue
  track=[i]
  while track[-1] in successor:track.append(successor[track[-1]])
  assert len({nodes[j]['period'] for j in track})==len(track);tracks.append(track)
 assert sorted(j for t in tracks for j in t)==sorted(indices)
 return tracks,accepted

def score_tracks(indices,tracks,labels):
 pred={i:k for k,t in enumerate(tracks) for i in t};assert set(pred)==set(indices);truth={i:labels[str(i)]['reference_id'] for i in indices if str(i) in labels};gt=collections.Counter(truth.values());pr=collections.Counter(pred[i] for i in truth);both=collections.Counter((truth[i],pred[i]) for i in truth);gg=list(gt);pp=list(pr);a=np.zeros((len(gg),len(pp)),np.int64);gm={g:i for i,g in enumerate(gg)};pm={p:i for i,p in enumerate(pp)}
 for (g,p),n in both.items():a[gm[g],pm[p]]=n
 x,y=linear_sum_assignment(-a);tp=int(a[x,y].sum());ass=float(sum(n*n/(gt[g]+pr[p]-n) for (g,p),n in both.items()));fragments=sum(max(0,len({pred[i] for i,v in truth.items() if v==g})-1) for g in gt)
 return dict(nodes=len(indices),scored_nodes=len(truth),IDTP=tp,AssA_sum=ass,identity_fragments=fragments,tracks=len(tracks))

class Dataset:
 def __init__(self):
  freeze=json.loads((O/'GRAPH_FROZEN_BEFORE_GT.json').read_text())
  for f,k in [('OBSERVATIONS_ALL.json','observation_sha256'),('FEATURES_ALL.npz','feature_sha256'),('CANDIDATE_GRAPH.json','candidate_graph_sha256')]:assert sha(O/f)==freeze[k]
  self.nodes=json.loads((O/'OBSERVATIONS_ALL.json').read_text());self.windows=json.loads((O/'CANDIDATE_GRAPH.json').read_text());self.features=np.load(O/'FEATURES_ALL.npz')['feature64'];self.labels=json.loads((O/'SUPERVISION_SEPARATE.json').read_text())['labels'];self.n=len(self.nodes);self.window_by_key={(w['role'],w['site']):w for w in self.windows};self.query={};self.position=np.array([n['center_utm'] for n in self.nodes]);self.area=np.array([n['area_feature_m2'] for n in self.nodes]);self.aspect=np.array([n['aspect'] for n in self.nodes]);self.prob=np.array([n['p_roof'] for n in self.nodes]);self.support=np.array([n['source_support'] for n in self.nodes]);self.days=np.array([datetime.date.fromisoformat(n['actual_date']).toordinal() if n['actual_date_known'] else 0 for n in self.nodes]);self.known=np.array([str(i) in self.labels for i in range(self.n)]);self.gold=np.array([self.labels.get(str(i),{}).get('reference_id','') for i in range(self.n)]);self.raw_unit=self.features/np.maximum(1e-8,np.linalg.norm(self.features,axis=1,keepdims=True))
  self.candidate=np.zeros((self.n,2,32),np.int64);self.mask=np.zeros((self.n,2,32),bool);self.enabled=np.zeros((self.n,2),bool);self.positive=np.zeros((self.n,2,32),bool);self.candidate_known=np.zeros((self.n,2,32),bool);self.geometry=np.zeros((self.n,2,32,8),np.float32);self.time=np.zeros((self.n,2,2),np.float32)
  for w in self.windows:
   byperiod=collections.defaultdict(set)
   for i in w['nodes']:
    if self.gold[i]:byperiod[self.nodes[i]['period']].add(self.gold[i])
   for q in w['queries']:
    i=q['query'];periods=[p for p in PERIODS if p!=self.nodes[i]['period']];t=periods.index(q['target_period']);c=q['candidates'];k=len(c);self.query[(i,t)]=q;self.candidate[i,t,:k]=c;self.mask[i,t,:k]=True;self.candidate_known[i,t,:k]=self.known[c];self.positive[i,t,:k]=(self.gold[c]==self.gold[i])&self.known[i]&self.known[c];self.enabled[i,t]=bool(self.known[i] and self.gold[i] in byperiod[q['target_period']])
    if not k:continue
    dx=(self.position[c]-self.position[i])/64;dt=np.array([(self.days[j]-self.days[i])/1000 if self.days[j] and self.days[i] else 0 for j in c]);cos=self.raw_unit[c]@self.raw_unit[i];self.geometry[i,t,:k]=np.column_stack([dx,np.log(self.area[c]/self.area[i])/3,np.log(self.aspect[c]/self.aspect[i])/3,cos,self.prob[c]-self.prob[i],np.minimum(self.support[c],self.support[i]),dt]).astype(np.float32);known_dt=np.array([bool(self.days[j] and self.days[i]) for j in c]);self.time[i,t]=[float(np.median(dt[known_dt])) if known_dt.any() else 0,float(known_dt.mean())]
  self.geometry=np.clip(self.geometry,-5,5);self.mean=None;self.std=None
 def split(self):
  train=[w for w in self.windows if w['role']=='TRAIN'];inner=[];fit=[]
  for w in train:
   p=np.median(self.position[w['nodes']],axis=0) if w['nodes'] else np.zeros(2);key=f'{int(p[0]//2000)}_{int(p[1]//2000)}';w['split_center']=p.tolist();(inner if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%5==0 else fit).append(w)
  tree=cKDTree([w['split_center'] for w in inner]);fit=[w for w in fit if tree.query(w['split_center'])[0]>=600];inner_refs={self.gold[i] for w in inner for i in w['nodes'] if self.gold[i]};self.fit_eligible=np.array([self.known[i] and self.gold[i] not in inner_refs for i in range(self.n)]);fitids=[i for w in fit for i in w['nodes'] if self.fit_eligible[i] and self.enabled[i].any()];normal=[i for w in fit for i in w['nodes'] if self.fit_eligible[i]];assert fitids and normal
  self.mean=self.features[normal].mean(0);self.std=np.maximum(.01,self.features[normal].std(0));self.normalized=np.clip((self.features-self.mean)/self.std,-5,5);return fit,inner,fitids,dict(fit_windows=len(fit),inner_windows=len(inner),fit_supervised_anchors=len(fitids),normalization_known_fit_nodes=len(normal),fit_inner_reference_supervision_overlap=0,center_purge_m=600,inner_reference_ids=len(inner_refs))
 def pack(self,ids,feature_enabled,training=False,rng=None):
  ids=np.asarray(ids);c=self.candidate[ids];mask=self.mask[ids].copy();positive=self.positive[ids].copy();enabled=self.enabled[ids].copy();known=self.candidate_known[ids].copy()
  if training:
   mask&=rng.random(mask.shape)>=.2;positive&=mask;known&=mask;known&=self.fit_eligible[c];enabled&=self.fit_eligible[ids,None]
  q=self.normalized[ids].copy();cf=self.normalized[c].copy();geometry=self.geometry[ids].copy()
  if not feature_enabled:q.fill(0);cf.fill(0);geometry[...,4]=0
  return [torch.from_numpy(np.ascontiguousarray(v)) for v in [q,cf,geometry,mask,self.time[ids],positive,known,enabled]]
 def baseline_scores(self,w,appearance=False):
  out={}
  for q in w['queries']:
   i=q['query'];c=q['candidates'];d=np.linalg.norm(self.position[c]-self.position[i],axis=1) if c else np.zeros(0);ratio=np.abs(np.log(self.area[c]/self.area[i]));v=2-d/12-.5*ratio
   if appearance and c:v+=.5*(self.raw_unit[c]@self.raw_unit[i])
   for j,x in zip(c,v):out[(i,j)]=float(x)
  return out
 @torch.no_grad()
 def model_scores(self,w,model,feature_enabled):
  model.eval();out={}
  for start in range(0,len(w['nodes']),64):
   ids=w['nodes'][start:start+64];q,c,g,m,t,*_=self.pack(ids,feature_enabled);v=model(q,c,g,m,t).numpy();v=v[...,:-1]-v[...,-1:]
   for b,i in enumerate(ids):
    for it in range(2):
     for k,j in enumerate(self.query[(i,it)]['candidates']):out[(i,j)]=float(v[b,it,k])
  return out
 def evaluate(self,windows,scores,threshold,export=False):
  sums=collections.Counter();records=[];accepted=[]
  for w,s in zip(windows,scores):
   utilities={}
   for (i,j),value in s.items():
    if PERIODS.index(self.nodes[i]['period'])>=PERIODS.index(self.nodes[j]['period']) or (j,i) not in s:continue
    utility=min(value,s[(j,i)])-threshold
    if utility>0:utilities[(i,j)]=utility
   tracks,edges=path_cover(w['nodes'],self.nodes,utilities);metric=score_tracks(w['nodes'],tracks,self.labels);sums.update(metric);both=[(i,j,v) for i,j,v in edges if self.known[i] and self.known[j]];sums['labelled_direct_edges']+=len(both);sums['wrong_reference_direct_edges']+=sum(self.gold[i]!=self.gold[j] for i,j,v in both)
   if export:records.append(dict(role=w['role'],site=w['site'],tracks=tracks,metrics=metric));accepted.extend([dict(source=i,target=j,utility=v,qualification='RESEARCH_BIDIRECTIONAL_MODEL_SUPPORT_NOT_VERIFIED_IDENTITY') for i,j,v in edges])
  n=sums['scored_nodes'];report=dict(sums,IDF1=sums['IDTP']/n if n else None,AssA=sums['AssA_sum']/n if n else None,label_coverage=n/max(1,sums['nodes']),physical_identity_accuracy='NOT_ESTABLISHED',map_reference_partial_label_development=True,all_model_observations_retained=True)
  return report,dict(windows=records,accepted_supports=accepted,threshold=threshold,unknown_long_edge='NOT_CANNOT_LINK',formal_presence_or_absence_inference=False)
