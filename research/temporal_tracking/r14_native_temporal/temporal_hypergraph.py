"""GT-free three-period identity hypotheses with soft cycle evidence.

Every admitted pair still exceeds the frozen DINO utility threshold. No threshold
is lowered. A triple needs two such supports; missing third evidence is Unknown.
"""
import itertools,warnings,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from native_temporal_contract import PERIODS,path_cover

def hypotheses(indices,nodes,direction_scores,threshold,beam=3):
 order=lambda i:(PERIODS.index(nodes[i]['period']),i)
 def canonical(i,j):return (i,j) if order(i)<order(j) else (j,i)
 margins={}
 for (i,j),v in direction_scores.items():
  if order(i)>=order(j) or (j,i) not in direction_scores:continue
  assert nodes[i]['period']!=nodes[j]['period'];margins[(i,j)]=min(float(v),float(direction_scores[(j,i)]))-threshold
 positive={k:v for k,v in margins.items() if v>0};baseline,_=path_cover(indices,nodes,positive);neighbors={i:{} for i in indices}
 for (i,j),v in positive.items():
  neighbors[i].setdefault(nodes[j]['period'],[]).append((v,j));neighbors[j].setdefault(nodes[i]['period'],[]).append((v,i))
 triples={tuple(sorted(t,key=order)) for t in baseline if len(t)==3}
 for i,pools in neighbors.items():
  periods=[p for p in PERIODS if p!=nodes[i]['period']]
  a=sorted(pools.get(periods[0],[]),key=lambda x:(-x[0],x[1]))[:beam];b=sorted(pools.get(periods[1],[]),key=lambda x:(-x[0],x[1]))[:beam]
  for (_,j),(_,k) in itertools.product(a,b):triples.add(tuple(sorted([i,j,k],key=order)))
 out=[]
 for pair,v in sorted(positive.items()):out.append(dict(nodes=list(pair),tree_utility=v,third_utility=0.,missing_third_evidence=False,positive_supports=[list(pair)]))
 for triple in sorted(triples):
  pairs=list(itertools.combinations(triple,2));known=[(margins[canonical(*p)],list(p)) for p in pairs if canonical(*p) in margins];pos=sorted([(v,p) for v,p in known if v>0],key=lambda x:(-x[0],x[1]));assert len({nodes[i]['period'] for i in triple})==3
  if len(pos)<2:continue
  used={tuple(p) for v,p in pos[:2]};other=[p for p in pairs if tuple(p) not in used];assert len(other)==1;third=margins.get(canonical(*other[0]),0.);out.append(dict(nodes=list(triple),tree_utility=sum(v for v,p in pos[:2]),third_utility=third,missing_third_evidence=canonical(*other[0]) not in margins,positive_supports=[p for v,p in pos[:2]]))
 return dict(indices=indices,hypotheses=out,baseline_tracks=baseline,beam=beam,frozen_pair_threshold=threshold,GT_input=False)

def optimize(payload,nodes,cycle_weight,time_limit=5.):
 indices=payload['indices'];hyp=payload['hypotheses'];weights=np.array([h['tree_utility']+cycle_weight*h['third_utility'] for h in hyp],np.float64);keep=np.flatnonzero(weights>0);where={i:j for j,i in enumerate(indices)};rows=[];cols=[]
 for col,k in enumerate(keep):
  for i in hyp[k]['nodes']:rows.append(where[i]);cols.append(col)
 if not len(keep):return [[i] for i in indices],dict(status='ALL_SINGLETON_NO_POSITIVE_HYPOTHESIS',optimal=True,selected_hypotheses=[],runtime_seconds=0.)
 matrix=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(indices),len(keep))).tocsc();start=time.time()
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',message='Unrecognized options detected')
  result=milp(-weights[keep],integrality=np.ones(len(keep),np.uint8),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,-np.inf,np.ones(len(indices))),options={'time_limit':time_limit,'mip_rel_gap':0.,'threads':1})
 elapsed=time.time()-start
 if not result.success:return payload['baseline_tracks'],dict(status='FALLBACK_FROZEN_DINO_ON_UNPROVEN_OPTIMUM',optimal=False,message=result.message,runtime_seconds=elapsed,selected_hypotheses=[])
 selected=[int(keep[j]) for j,v in enumerate(result.x) if v>.5];tracks=[hyp[k]['nodes'] for k in selected];used=[i for t in tracks for i in t];assert len(set(used))==len(used);used=set(used);tracks.extend([[i] for i in indices if i not in used]);assert sorted(i for t in tracks for i in t)==sorted(indices);assert all(len({nodes[i]['period'] for i in t})==len(t) for t in tracks)
 return tracks,dict(status='OPTIMAL',optimal=True,runtime_seconds=elapsed,objective=float(-result.fun),selected_hypotheses=selected,research_unknown_long_relations=sum(hyp[k]['missing_third_evidence'] for k in selected),explicit_verified_cannot_links=0)
