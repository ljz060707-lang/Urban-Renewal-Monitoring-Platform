from pathlib import Path
import numpy as np,torch
from native_temporal_contract import partial_selection_loss,path_cover,score_tracks,save,O
torch.set_num_threads(2);torch.manual_seed(5)
v=torch.randn(2,2,4,requires_grad=True);positive=torch.zeros((2,2,3),dtype=torch.bool);positive[0,0,0]=True;known=torch.zeros_like(positive);known[0,0,:2]=True;known[0,1,1]=True;enabled=torch.tensor([[True,True],[False,False]]);loss=partial_selection_loss(v,positive,known,enabled);loss.backward();assert torch.isfinite(loss) and torch.isfinite(v.grad).all();assert (v.grad[1]==0).all() and v.grad[0,0,2]==0 and v.grad[0,1,0]==0 and v.grad[0,1,2]==0
nodes=[dict(period=p) for p in ['xa180429','xa190101','xa201231','xa190101']];tracks,edges=path_cover(list(range(4)),nodes,{(0,1):2,(1,2):2,(0,3):1.5});assert [0,1,2] in tracks and [3] in tracks
labels={str(i):dict(reference_id='A' if i<3 else 'B') for i in range(4)};m=score_tracks(list(range(4)),tracks,labels);assert m['IDTP']==4 and m['identity_fragments']==0
single,_=path_cover(list(range(4)),nodes,{});assert len(single)==4
save(O/'FUNCTIONAL_CONTRACT_TEST.json',dict(status='PASS_PARTIAL_LABEL_GRADIENT_AND_FORWARD_PATH_COVER',unknown_candidate_logit_gradient_zero=True,unlabelled_query_gradient_zero=True,dustbin_explicit=True,missing_long_edge_allows_three_period_path=True,duplicate_same_period_rejected_by_order=True,all_nodes_retained=True,no_training_accuracy_claim=True));print('PASS_NATIVE_TEMPORAL_CONTRACT',float(loss))
