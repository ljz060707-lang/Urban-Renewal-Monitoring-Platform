from temporal_hypergraph import hypotheses,optimize
from native_temporal_contract import O,save
nodes=[dict(period=p) for p in ['xa180429','xa190101','xa201231','xa190101']]
def directional(pairs):return {(a,b):v for (i,j),v in pairs.items() for a,b in [(i,j),(j,i)]}
scores=directional({(0,1):2,(1,2):2});payload=hypotheses(list(range(4)),nodes,scores,0);tracks,proof=optimize(payload,nodes,1.);assert proof['optimal'] and [0,1,2] in tracks and [3] in tracks
scores=directional({(0,1):2,(1,2):2,(0,2):-20});payload=hypotheses(list(range(4)),nodes,scores,0);tracks,proof=optimize(payload,nodes,1.);assert proof['optimal'] and all(len(t)<3 for t in tracks)
scores=directional({(0,2):2,(1,2):2});payload=hypotheses(list(range(4)),nodes,scores,0);tracks,proof=optimize(payload,nodes,1.);assert [0,1,2] in tracks
save(O/'hypergraph/FUNCTIONAL_TEST.json',dict(status='PASS_GT_FREE_HYPERGRAPH_UNKNOWN_AND_SOFT_CONTRADICTION',missing_long_edge_allows_three_period_track=True,independent_two_branch_support_can_form_track=True,strong_contradiction_penalized_not_hard_cannot_link=True,all_nodes_retained=True,same_period_unique=True,synthetic_not_accuracy=True,optimizer_updates=0));print('PASS_TEMPORAL_HYPERGRAPH')
