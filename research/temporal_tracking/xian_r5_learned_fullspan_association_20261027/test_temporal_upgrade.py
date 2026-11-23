import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,torch
torch.set_num_threads(2)
from shapely.geometry import box,mapping
from temporal_model import TemporalIdentityNet
from prepare_sn7 import crop,geometry
from train_temporal import calibrate
from upgrade_application import graph_partition
from temporal_inference import score
from association_metrics import identity_scores
from affine import Affine
from xian_r5.application_geometry import map_geometry

def observation(i,e):
    g=mapping(box(i,0,i+2,2));return dict(observation_id=str(i),epoch_id=e,geometry=g,source_edge_partial_risk=False,source_valid_fraction=1,area_on_reference_plane_m2=4,actual_date='2020-01-01')
def edge(a,b,p=.99):return dict(source=str(a),target=str(b),eligible=True,same=p)
def test_model_all_branches_receive_gradients():
    m=TemporalIdentityNet();o=m(torch.rand(2,4,32,32),torch.rand(2,5,3,4,32,32),torch.rand(2,5,3,8),torch.rand(2,5,4,32,32),torch.rand(2,5,2))
    assert o['association'].shape==(2,5,3,3)
    sum(x.sum() for x in o.values()).backward()
    for branch in ['encoder','geometry','temporal','presence_temporal']:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.startswith(branch))
def test_context_adapter_matches_training_when_three_candidates():
    torch.manual_seed(7);m=TemporalIdentityNet().eval();args=[torch.rand(2,4,32,32),torch.rand(2,5,3,4,32,32),torch.rand(2,5,3,8),torch.rand(2,5,4,32,32),torch.rand(2,5,2)]
    a=m(*args);b=score(m,*args)
    for key in a:assert torch.allclose(a[key],b[key],atol=1e-6)
def test_far_candidates_are_scored_without_changing_near_context():
    torch.manual_seed(7);m=TemporalIdentityNet().eval();q=torch.rand(1,4,32,32);c=torch.rand(1,3,3,4,32,32);g=torch.rand(1,3,3,8);s=torch.rand(1,3,4,32,32);t=torch.rand(1,3,2)
    a=score(m,q,c,g,s,t)['association']
    far=torch.rand(1,3,5,4,32,32);fg=torch.rand(1,3,5,8);fg[...,:2]=8
    b=score(m,q,torch.cat([c,far],2),torch.cat([g,fg],2),s,t)['association']
    assert b.shape[2]==8 and torch.allclose(a,b[:,:,:3],atol=1e-6)
def test_singleton_null_exit():
    tr,ac,_=graph_partition([observation(0,'t0'),observation(1,'t1')],[])
    assert len(tr)==2 and not ac
def test_frozen_strict_argmax_below_half_is_not_discarded():
    e=dict(source='0',target='1',same=.4,different=.3,null=.3,eligible=True)
    tr,ac,_=graph_partition([observation(0,'t0'),observation(1,'t1')],[e])
    assert len(tr)==1 and len(ac)==1
def test_candidate_slots_are_not_persistent_identities():
    torch.manual_seed(2);m=TemporalIdentityNet().eval();q=torch.rand(1,4,32,32);c=torch.rand(1,5,3,4,32,32);g=torch.rand(1,5,3,8);s=torch.rand(1,5,4,32,32);t=torch.rand(1,5,2)
    p=m(q,c,g,s,t)['association'];cp=c.clone();gp=g.clone();orders=[]
    for j in range(5):
        order=torch.randperm(3);orders.append(order);cp[:,j]=c[:,j,order];gp[:,j]=g[:,j,order]
    pp=m(q,cp,gp,s,t)['association']
    for j,o in enumerate(orders):assert torch.allclose(pp[:,j],p[:,j,o],atol=1e-6)
def test_same_epoch_exclusion_even_strong_competitors():
    tr,ac,_=graph_partition([observation(0,'t0'),observation(1,'t0'),observation(2,'t1')],[edge(0,2),edge(1,2,.98)])
    assert len(ac)==1
    assert all(len(t['observation_ids'])<=2 for t in tr)
def test_inconsistent_triangle_rejects_a_chain():
    tr,ac,_=graph_partition([observation(0,'t0'),observation(1,'t1'),observation(2,'t2')],[edge(0,1),edge(1,2,.98)])
    assert len(ac)==1 and len(tr)==2
def test_consistent_triangle_and_skip_return():
    oo=[observation(0,'t0'),observation(1,'t1'),observation(2,'t2')]
    tr,ac,_=graph_partition(oo,[edge(0,1),edge(1,2),edge(0,2)])
    assert len(tr)==1 and len(ac)==3
    tr,ac,_=graph_partition(oo,[edge(0,2)])
    assert len(tr)==2
def test_no_class_specific_preprocessing_and_native_mask():
    im=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3);p=box(5,5,20,20)
    assert np.array_equal(crop(im,p),crop(im,p))
    assert set(np.unique(crop(im,p)[3]))<={0,255}
    assert np.isfinite(geometry(p,box(8,8,23,23),.7,1)).all()
def test_calibration_cannot_pass_with_few_or_wrong_same_predictions():
    a=dict(probs=np.tile([[[[.99,.005,.005]]]],(4,5,3,1)),y=np.ones((4,5,3),np.int64))
    c=calibrate(a);assert c['same_edge_operating_point'] is None
def test_dummy_targets_are_not_different_id():
    # Data contract: -100 padding never contributes supervised class 1.
    y=torch.tensor([-100,0,1,2]);p=torch.randn(4,3,requires_grad=True)
    torch.nn.functional.cross_entropy(p,y,ignore_index=-100).backward()
    assert p.grad[0].abs().sum()==0
def test_oracle_association_metric_perfect_fragment_and_merge():
    m=identity_scores([(0,'A','p'),(1,'A','p')]);assert m['IDF1']==1 and m['AssA']==1
    m=identity_scores([(0,'A','p'),(1,'A','q')]);assert m['IDF1']==.5 and m['AssA']==.5 and m['ID_switches']==1
    m=identity_scores([(0,'A','p'),(1,'A','p'),(0,'B','p'),(1,'B','p')]);assert m['IDF1']==.5 and m['AssA']==.5
def test_native_pixel_y_direction_is_not_map_northing():
    a=box(10,10,20,20);b=box(10,12,20,22)
    pixel=geometry(a,b,.5,1);aff=Affine(10,0,300000,0,-10,3780000)
    metric=geometry(map_geometry(a,aff),map_geometry(b,aff),.5,1)
    assert np.isclose(pixel[1],.2) and np.isclose(metric[1],-.2)
    # Public training and final inference explicitly bind the pixel convention.
    from upgrade_application import ROOT
    import json
    route=json.loads((ROOT/'runs/LEARNED_FULLSPAN_NATIVE_FIXED55/ROUTE.json').read_text())
    assert route['model']['geometry_feature_contract']=='PIXEL_X_RIGHT_Y_DOWN_SAME_AS_TRAINING'
