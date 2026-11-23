"""Explicit synthetic integration fixtures, never real change ground truth."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box,mapping,LineString
from shapely.affinity import translate
from xian_r5.application_contract import *
from xian_r5.application_geometry import *
from xian_r5.application_temporal import *
from xian_r5 import application as app

def epoch(name,day):return dict(epoch_id=name,actual_date=day)

def obs(uid,e,g,**extra):
    return dict(observation_id=uid,epoch_id=e,actual_date={'t0':'2020-01-01','t1':'2020-03-01','t2':'2020-05-01'}[e],geometry=mapping(g),raw_geometry=mapping(g),metric_crs='EPSG:32649',source_payload_sha256=uid,
        source_edge_partial_risk=False,source_conflict=False,source_valid_fraction=1,position_sigma_m=None,excluded=False,**extra)

def track(uid,rows):
    r=medoid(rows)
    return dict(track_id_candidate=uid,observation_ids=[x['observation_id'] for x in rows],geometry=r['geometry'],canonical_observation_id=r['observation_id'])

@pytest.mark.parametrize('shift',[(0,0),(17,-4),(-100,900)])
def test_translation_area(shift):
    g=box(1,2,18,22);assert translate(g,*shift).area==g.area

def test_physical_geometry_operator_and_null():
    g=box(0,0,10,20);assert physical(g,None)==(None,'UNRESOLVED_NO_VIEW_GEOMETRY')
    evidence=dict(method='VERIFIED_PLANAR_AFFINE',evidence_ref='SYNTHETIC_KNOWN_PLANE',source_frame='VIEW_PERPENDICULAR_METRES',target_frame='HORIZONTAL_METRES',already_rectified=False,affine6=[2,0,0,1,0,0])
    out,status=physical(g,evidence);assert out.area==400 and out.bounds!=(g.bounds)
    evidence['already_rectified']=True;assert physical(g,evidence)[0] is None

@pytest.mark.parametrize('geometry,truncated,risk,status',[(box(0,0,100,1),False,True,'METRIC_MRR_DIAGNOSTIC_UNCALIBRATED'),(box(0,0,100,20),False,False,'METRIC_MRR_DIAGNOSTIC_UNCALIBRATED'),(box(0,0,100,1),True,None,'UNKNOWN_TRUNCATED'),(LineString([(0,0),(1,1)]),False,None,'UNKNOWN_DEGENERATE')])
def test_aspect_rules_are_not_semantic_truth(geometry,truncated,risk,status):
    out=aspect(geometry,truncated,9.647);assert out['linear_risk']==risk and out['status']==status

def test_adjacent_actual_observations_no_annual_interval():
    ep=[epoch('t2','2020-05-01'),epoch('t0','2020-01-01'),epoch('copy','2020-01-01'),epoch('t1','2020-03-01'),epoch('unknown',None)]
    seq=comparison_epochs(ep);assert [x['actual_date'] for x in seq]==['2020-01-01','2020-03-01','2020-05-01'];assert valid_pair(seq[0],seq[1])

def test_overlapping_windows_unknown():
    a=dict(actual_time_window=['2020-01-01','2020-03-01']);b=dict(actual_time_window=['2020-02-01','2020-04-01'])
    assert not valid_pair(a,b)

@pytest.mark.parametrize('bad',['../escape','x/y','..','a b','x:y'])
def test_identifier_path_containment(bad):
    with pytest.raises(ValueError):safe_id(bad)

def test_candidate_presence_missing_not_zero_and_gap_not_filled():
    rr=[obs('a','t0',box(0,0,10,10)),obs('b','t2',box(0,0,10,10))];t=track('T',rr)
    epochs=[epoch('t0','2020-01-01'),epoch('t1','2020-03-01'),epoch('t2','2020-05-01')]
    pp=candidate_presence(rr,[t],epochs,{},set());assert [x['presence_candidate'] for x in pp]==[1,'U',1]
    events,_=pair_events(rr,[t],pp,epochs[0],epochs[1]);assert events[0]['state_candidate']=='UNRESOLVED'
    assert all(p['presence_qualified']=='U' and p['physical_id'] is None for p in pp)

@pytest.mark.parametrize('values,expected', [((1,1),'STABLE'),((0,1),'ADDITION'),((1,0),'DEMOLITION'),(('U',1),'UNRESOLVED'),((0,0),'UNRESOLVED')])
def test_synthetic_positive_three_states(values,expected):
    a=epoch('t0','2020-01-01');b=epoch('t1','2020-03-01');g=box(0,0,10,10)
    rr=[obs('a','t0',g),obs('b','t1',translate(g,.1,.1))];t=track('T',rr)
    pp=[dict(track_id_candidate='T',epoch_id='t0',presence_candidate=values[0]),dict(track_id_candidate='T',epoch_id='t1',presence_candidate=values[1])]
    ev,regions=pair_events(rr,[t],pp,a,b);assert ev[0]['state_candidate']==expected
    if expected=='STABLE':assert abs(regions['STABLE'].area-100)<1e-9 and regions['ADDITION'].is_empty and regions['DEMOLITION'].is_empty

def test_same_payload_different_date_is_not_stability():
    rr=[obs('a','t0',box(0,0,10,10)),obs('b','t1',box(0,0,10,10))];rr[1]['source_payload_sha256']='a';t=track('T',rr)
    pp=[dict(track_id_candidate='T',epoch_id=x,presence_candidate=1) for x in ['t0','t1']]
    ev,_=pair_events(rr,[t],pp,epoch('t0','2020-01-01'),epoch('t1','2020-03-01'));assert ev[0]['state_candidate']=='UNRESOLVED'

def test_same_site_replacement_keeps_both_events_and_multilabel():
    g=box(0,0,10,10);rr=[obs('old','t0',g),obs('new','t1',g)];tt=[track('OLD',[rr[0]]),track('NEW',[rr[1]])]
    pp=[dict(track_id_candidate=t,epoch_id=e,presence_candidate=p) for t,e,p in [('OLD','t0',1),('OLD','t1',0),('NEW','t0',0),('NEW','t1',1)]]
    ev,reg=pair_events(rr,tt,pp,epoch('t0','2020-01-01'),epoch('t1','2020-03-01'))
    assert {e['state_candidate'] for e in ev}=={'ADDITION','DEMOLITION'}
    grid=Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649');f=exact(grid,{k:reg[k] for k in ['STABLE','ADDITION','DEMOLITION']},g)
    assert np.allclose(f['fractions'][:,0,0],[0,1,1]) and f['fractions'].sum()==2

def test_partial_and_split_merge_stay_unknown():
    r=obs('a','t0',box(0,0,10,10));t=track('T',[r]);e=[epoch('t0','2020-01-01')]
    assert candidate_presence([r],[t],e,{}, {'a'})[0]['presence_candidate']=='U'
    r['source_edge_partial_risk']=True;assert candidate_presence([r],[t],e,{},set())[0]['presence_candidate']=='U'

def test_reliable_zero_requires_explicit_active_result():
    r=obs('a','t0',box(0,0,10,10));t=track('T',[r]);ee=[epoch('t0','2020-01-01'),epoch('t1','2020-03-01')]
    pp=candidate_presence([r],[t],ee,{('T','t1'):dict(reliable_zero_candidate=True)},set());assert pp[1]['presence_candidate']==0 and pp[1]['presence_qualified']=='U'

@pytest.mark.parametrize('crs',['EPSG:32649','EPSG:32650','EPSG:3857'])
def test_rotated_metric_grid_full_cell_denominator(crs):
    grid=Grid(Affine(8,6,100,6,-8,200),1,1,crs);cell=grid.cell(0,0);f=exact(grid,{'X':cell},cell)
    assert abs(f['fractions'][0,0,0]-1)<1e-12 and f['area_conservation_error_m2']<1e-8
    fine=grid.nested(10);assert fine.affine*(10,10)==grid.affine*(1,1)

def test_unknown_unassigned_not_background():
    grid=Grid(Affine(10,0,0,0,-10,10),1,1,'EPSG:32649');r=exact(grid,{'ROOF':EMPTY},box(0,0,5,10))
    assert r['fractions'].sum()==0 and r['valid'][0,0]==.5 and r['unknown'][0,0]==.5 and r['unassigned'][0,0]==.5

def test_checkpoint_semantics_mismatch_rejected_before_loading(monkeypatch):
    routes=dict(roof=dict(checkpoint_sha256=V2,state_container_key='state_dict',target_semantics='VISIBLE_ROOF_IN_SOURCE_IMAGE_FRAME'))
    with pytest.raises(ValueError,match='semantic mismatch'):check_model_route(routes)

def test_stages_resume_no_repeat_and_dependency_change(tmp_path):
    calls=[];s=Stages(tmp_path,'fixed')
    def fn(d):calls.append(1);save(d/'value.json',{'value':7})
    d=s.run('model',{'input':1},fn);s.run('model',{'input':1},fn);assert len(calls)==1 and s.reused==['model']
    with pytest.raises(ValueError,match='Changed dependency'):s.run('model',{'input':2},fn)
    save(s.output(d)/'value.json',{'value':9})
    with pytest.raises(ValueError,match='Corrupted'):s.run('model',{'input':1},fn)

def test_failed_stage_does_not_commit(tmp_path):
    def fn(d):save(d/'actual_partial.json',{'partial':True});raise RuntimeError('interruption fixture')
    with pytest.raises(RuntimeError):Stages(tmp_path,'fixed').run('stage',{},fn)
    assert not (tmp_path/'stages/stage/COMMIT.json').exists() and list(tmp_path.rglob('actual_partial.json'))

def test_real_r4_rule_runtime_repeated_neighborhood_and_split(tmp_path):
    rr=[obs('a','t0',box(300000,3800000,300010,3800010)),obs('b','t0',box(300020,3800000,300030,3800010)),
        obs('c','t1',box(300000,3800000,300030,3800010))]
    tt,edges,ids,conflicts=r4_tracks(rr,[epoch('t0','2020-01-01'),epoch('t1','2020-03-01')],tmp_path,app.CODE)
    assert conflicts=={'a','b','c'} and all(t['physical_id'] is None for t in tt)

def test_real_r4_stable_rule_calls_topology_not_nearest_only(tmp_path):
    g=box(300000,3800000,300010,3800010)
    rr=[obs('a','t0',g),obs('b','t1',translate(g,.1,.1))]
    tt,edges,ids,conflicts=r4_tracks(rr,[epoch('t0','2020-01-01'),epoch('t1','2020-03-01')],tmp_path,app.CODE)
    assert len(tt)==1 and edges[0]['mutual_best'] and edges[0]['iou']>.9 and edges[0]['local_neighborhood_consistency']==1

def test_gt_and_training_flags_are_rejected(tmp_path):
    c=dict(schema_version=SCHEMA,run_name='TEST',product_level='research_candidate',execution={'training_enabled':True})
    import yaml
    f=tmp_path/'bad.yaml';f.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='Training/cloud'):app.config(f)
    c['execution']={};c['evaluation']={'reference_manifest':'never_open_this_GT'};f.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='GT'):app.config(f)

def test_existing_class_tif_255_can_be_foreground(tmp_path):
    import rasterio
    f=tmp_path/'mask.tif';g=Grid(Affine(1,0,300000,0,-1,3800000),8,8,'EPSG:32649');a=np.zeros((8,8),np.uint8);a[2:5,2:5]=255
    write_tif(f,a,g,['declared_255_foreground'],'fixture')
    p=dict(hr_building_tif=str(f),building_tif_schema={'kind':'class','foreground_values':[255],'nodata':None},shape=[8,8],affine=list(g.affine)[:6],crs=g.crs,actual_date='2020-01-01')
    app.decode_mask(p,tmp_path);r=read(tmp_path/'instances.json');assert len(r['ROOF_INSTANCE_CANDIDATES'])==1 and np.load(tmp_path/'mask.npy').sum()==9

def test_end_to_end_mask_only_and_resume(tmp_path,monkeypatch):
    # Entire fixture is explicitly synthetic and uses no model or GT input.
    import yaml
    grid=Grid(Affine(1,0,300000,0,-1,3800020),20,20,'EPSG:32649');epochs=[]
    for i in range(2):
        a=np.zeros((20,20),np.uint8);a[4:12,4:12]=1;f=tmp_path/('e%d.tif'%i);write_tif(f,a,grid,['synthetic_roof'],'SYNTHETIC')
        epochs.append(dict(epoch_id='t%d'%i,actual_date='2020-0%d-01'%(i+1),parents=[dict(parent_id='p%d'%i,hr_image=None,hr_building_tif=str(f),building_tif_schema=dict(kind='binary',foreground_values=[1],nodata=None),actual_date='2020-0%d-01'%(i+1),source_acquisition_id='SYNTHETIC%d'%i)]))
    template=tmp_path/'grid.tif';write_tif(template,np.zeros((2,2),np.uint8),Grid(Affine(10,0,300000,0,-10,3800020),2,2,grid.crs),['template'],'SYNTHETIC')
    rules=tmp_path/'rule.json';save(rules,dict(threshold=9.647));dictionary=app.HOME/'CLASS_DICTIONARY_V1.json'
    c=dict(schema_version=SCHEMA,run_name='SYNTHETIC_INTEGRATION',output_root=str(tmp_path),product_level='research_candidate',evidence_role='SYNTHETIC_FIXTURE',execution=dict(max_wallclock_seconds=120),model_routing=dict(routing_manifest='not_read_by_fixture',facade={'enabled':False},alignment={'enabled':False}),inputs={'epochs':epochs},grid=dict(target_lr_grid_template=str(template),map_fine_grid={'subdivision_of_lr_axes':10}),m1_recognition=dict(aspect_screen=dict(enabled=True,rule_manifest=str(rules),mode='screen_only')),m3_temporal=dict(class_dictionary_path=str(dictionary),active_recheck=dict(enabled=False)))
    route={};pins={str(rules):sha(rules)}
    monkeypatch.setattr(app,'validate',lambda p:(copy.deepcopy(c),pins,dict(runtime_bindings={})))
    f=tmp_path/'run.yaml';f.write_text(yaml.safe_dump(c));out=tmp_path/'RUN'
    app.run_application(f,out);r=read(out/'FINAL_RECEIPT.json');assert r['GT_reads']==0 and r['GT10M_ADMISSION']=='NO'
    assert (out/'pairs/t0__t1/CHANGE_THREE_STATE.tif').exists() and (out/'single_date/t0/ROOF_MANAGED_NATIVE_OR_MAP.tif').exists()
    initial=sha(out/'FINAL_RECEIPT.json');app.run_application(f,out,True);assert sha(out/'FINAL_RECEIPT.json')==initial

def test_parent_fusion_real_distinct_payloads_and_source_conflict(tmp_path):
    g=Grid(Affine(1,0,300000,0,-1,3800020),20,20,'EPSG:32649');parents=[]
    for i in range(2):
        f=tmp_path/('parent%d.tif'%i);data=np.full((3,20,20),80,np.uint8)
        if i:data[:,8,8]=180
        write_tif(f,data,g,['R','G','B'],'SYNTHETIC')
        parents.append(dict(parent_id='p%d'%i,hr_image=str(f),actual_date='2020-01-01',source_acquisition_id='A'))
    out=tmp_path/'fused';out.mkdir();packets=fuse_parent_inputs(parents,out)
    assert len(packets)==1 and len(packets[0]['parents'])==2 and packets[0]['source_conflict_pixels']==1
    assert np.load(packets[0]['valid'])[8,8]==0

def test_active_budget_calls_runtime_and_registers_version(tmp_path,monkeypatch):
    # Injected source-model fixture tests the scheduler; real forward acceptance
    # is separately recorded by the three-epoch demonstration.
    from PIL import Image
    r=obs('a','t0',box(100,100,120,120));t=track('T',[r]);image=tmp_path/'rgb.png';Image.fromarray(np.full((512,512,3),80,np.uint8)).save(image)
    valid=tmp_path/'valid.npy';np.save(valid,np.ones((512,512),np.uint8));rule=tmp_path/'rule.json';save(rule,{'threshold':9.647})
    p=dict(packet_id='p1',image=str(image),valid=str(valid),shape=[512,512],affine=list(Affine.identity())[:6],crs='EPSG:32649',epoch_id='t1',actual_date='2020-03-01',source_acquisition_id='SYNTHETIC',parents=[dict(parent_id='p1',payload_sha256='hash')])
    c=dict(inputs=dict(epochs=[epoch('t0','2020-01-01'),epoch('t1','2020-03-01'),epoch('t2','2020-05-01')]),m3_temporal=dict(active_recheck=dict(enabled=True,max_requests_per_run=1,max_per_track_epoch=1)),m1_recognition=dict(aspect_screen=dict(rule_manifest=str(rule),mode='screen_only')))
    called=[]
    def fake_worker(action,jobs,*args):
        called.extend(jobs)
        for j in jobs:
            d=Path(j['output']);q=np.zeros((512,512),np.float32);q[100:120,100:120]=.9;np.save(d/'probability.npy',q)
            save(d/'instances.json',dict(ROOF_INSTANCE_CANDIDATES=[dict(candidate_id='SYNTHETIC_NEW',geometry_pixel_edge_xy=mapping(box(100,100,120,120)),source_edge_partial_risk=False,model_sha256='SYNTHETIC_MODEL',mean_probability_uncalibrated=.9)]))
    monkeypatch.setattr(app,'invoke',fake_worker)
    results,ledger,recovered=app.active_rechecks(c,[r],[t],set(),[p],{},tmp_path,time.monotonic()+20)
    assert len(called)==1 and len(recovered)==1 and recovered[0]['observation_id']!='a' and recovered[0].get('physical_id') is None
    assert any(x['status']=='BUDGET_EXHAUSTED_UNKNOWN' for x in ledger)
    assert recovered[0]['position_candidate_status']=='ACTIVE_RECHECK_RETAIN_RAW_POSITION_UNKNOWN'

def test_unknown_date_cannot_become_pair_and_date_copy_collapses():
    assert len(comparison_epochs([epoch('a','2020-01-01'),epoch('b','2020-01-01')]))==1
    assert comparison_epochs([epoch('a',None)])==[]
