"""Reuse the V1 app; source-bound rechecks and learned all-sequence association.
No target GT path is accepted. Outputs are research candidates only.
"""
import os,sys,json,math,copy,time,argparse,itertools,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'APPLICATION_CODE/src'))
LEGACY=ROOT/'LEGACY_CODE'
sys.path.insert(0,str(LEGACY))
import numpy as np,torch
from PIL import Image
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape,mapping
from shapely.affinity import translate
from xian_r5 import application as app
from xian_r5.application_contract import read,save,lines,digest,sha
from xian_r5.application_geometry import map_geometry,convert,medoid,aspect
from prepare_sn7 import crop,geometry
from temporal_model import TemporalIdentityNet
from temporal_inference import score as score_candidate_set
BASE=Path('F:/xian/XIAN_R5_APPLICATION/APPLICATION_V1_20261027_01/runs/REAL_THREE_EPOCH_MULTIPARENT')
def jsonl(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf8').splitlines() if l.strip()]
def inputs():
    c=app.config(BASE/'RUN_CONFIG.yaml');packets=read(BASE/'SOURCE_REGISTRY/SOURCE_PACKETS.json');obs=jsonl(BASE/'temporal/OBSERVATIONS.jsonl');tracks=jsonl(BASE/'temporal/TRACKS_CANDIDATE.jsonl')
    return c,packets,obs,tracks
def active():
    c,packets,obs,tracks=inputs();out=ROOT/'runs/ACTIVE_EXPANDED';out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    pp={p['epoch_id']:p for p in packets};routes=read(ROOT/'MODEL_ROUTING_MANIFEST.json')
    byid={r['observation_id']:r for r in obs};old=read(ROOT/'evidence/T0_UNKNOWN_ATTRIBUTION.json')['old_unknown_nodes']
    unresolved={(r['track_id_candidate'],r['epoch_id']) for r in old}
    ledger=[];jobs={};supports={};epochs=c['inputs']['epochs']
    old_stage=BASE/'stages/04_temporal/attempt_001'
    caches={(r['packet_id'],tuple(r['native_window'])):old_stage/r['request_id'] for r in jsonl(old_stage/'ACTIVE_SEARCH_LEDGER.jsonl') if r.get('status')=='MODEL_REQUEST_EXECUTED'}
    # Entire declared set, two distinct context sizes for every old unresolved node.
    for track in tracks:
        anchor=byid[track['canonical_observation_id']]
        for e in epochs:
            key=(track['track_id_candidate'],e['epoch_id'])
            if key not in unresolved:continue
            p=pp[e['epoch_id']];g=map_geometry(convert(shape(anchor['raw_geometry']),anchor['metric_crs'],p['crs']),~Affine(*p['affine']))
            h,w=p['shape'];cx,cy=g.centroid.coords[0]
            for size in [512,768]:
                ww,hh=min(size,w),min(size,h);x=max(0,min(w-ww,math.floor(cx-ww/2)));y=max(0,min(h-hh,math.floor(cy-hh/2)))
                uid='CTX_'+digest([p['rgb_pixel_sha256'],[x,y,ww,hh],routes['roof']['checkpoint_sha256']])[:24]
                rec=dict(track_id_candidate=key[0],epoch_id=key[1],context_size=size,request_id=uid,window=[x,y,ww,hh],
                    current_source_packet=p['packet_id'],prompt_observation_id=anchor['observation_id'],current_epoch_RGB_required=True,
                    independent_evidence=False,evidence_scope='different current RGB context, same B0; not independent detector',absence_admission=False)
                local=translate(g,xoff=-x,yoff=-y);v=np.load(p['valid'])[y:y+hh,x:x+ww]
                mask=rasterize([(local,1)],out_shape=(hh,ww),transform=Affine.identity(),dtype='uint8')>0
                vf=float(v[mask].mean()) if mask.any() else 0.
                rec['source_fraction']=vf
                if not mask.any() or vf<.75 or not g.within(shape({'type':'Polygon','coordinates':[[[x,y],[x+ww,y],[x+ww,y+hh],[x,y+hh],[x,y]]]})):
                    rec['status']='UNSUPPORTED_WINDOW_RETAIN_U'
                else:
                    rec['status']='DEDUPLICATED_CONTEXT_REFERENCE' if uid in jobs else 'SCHEDULED'
                    if uid not in jobs:
                        d=out/uid;d.mkdir(exist_ok=True);im=np.asarray(Image.open(p['image']).convert('RGB'))[y:y+hh,x:x+ww];Image.fromarray(im).save(d/'search_RGB.png')
                        jobs[uid]=dict(image=str(d/'search_RGB.png'),output=str(d),actual_date=e['actual_date'],source_id=p['packet_id']+':'+uid)
                        supports[uid]=(p,[x,y,ww,hh])
                        cache=caches.get((p['packet_id'],(x,y,ww,hh)))
                        if cache is not None:
                            assert sha(cache/'search_RGB.png')==sha(d/'search_RGB.png')
                            for filename in ['instances.json','probability.npy']:shutil.copy2(cache/filename,d/filename)
                            save(d/'CACHE_PROVENANCE.json',dict(source=str(cache),instances_sha256=sha(cache/'instances.json'),probability_sha256=sha(cache/'probability.npy'),independent=False))
                            rec['status']='REUSED_VERIFIED_V1_CONTEXT'
                ledger.append(rec)
    save(out/'SEARCH_PLAN.json',dict(old_nodes=len(unresolved),contexts=[512,768],rows=ledger,unique_requests=list(jobs),
        coverage='all 120 old U nodes, supported current-epoch windows only; no tiny global cap',GT_reads=0))
    todo=[j for j in jobs.values() if not (Path(j['output'])/'instances.json').exists()]
    if todo:
        worker=out/('worker_'+digest([j['output'] for j in todo])[:12]);app.invoke('roof',todo,routes,worker,time.monotonic()+36000)
    recovered=[];versions=[];current={e['epoch_id']:[shape(r['raw_geometry']) for r in obs if r['epoch_id']==e['epoch_id']] for e in epochs}
    for uid,(p,window) in supports.items():
        x,y,w,h=window;found=read(out/uid/'instances.json')['ROOF_INSTANCE_CANDIDATES']
        for r in found:
            ng=translate(shape(r['geometry_pixel_edge_xy']),xoff=x,yoff=y);g=convert(map_geometry(ng,Affine(*p['affine'])),p['crs'],obs[0]['metric_crs'])
            duplicates=[i for i,gg in enumerate(current[p['epoch_id']]) if g.intersection(gg).area/max(g.union(gg).area,1e-9)>=.5]
            versions.append(dict(request_id=uid,epoch_id=p['epoch_id'],native_geometry=mapping(ng),geometry=mapping(g),source_candidate=r,
                overlaps_existing=duplicates,classification='CONTEXT_VERSION' if duplicates else 'NEW_SOURCE_CANDIDATE',independent=False))
            if duplicates or r['source_edge_partial_risk'] or g.is_empty:continue
            mask=rasterize([(ng,1)],out_shape=tuple(p['shape']),transform=Affine.identity(),dtype='uint8')>0
            valid=np.load(p['valid']);vf=float(valid[mask].mean()) if mask.any() else 0.
            if vf<.98:continue
            seed=copy.deepcopy(obs[0]);oid='OBS_'+digest([p['packet_id'],mapping(ng),'EXPANDED_CONTEXT_V2'])[:28]
            seed.update(observation_id=oid,original_candidate_id=r['candidate_id'],epoch_id=p['epoch_id'],actual_date=p['actual_date'],packet_id=p['packet_id'],
                native_geometry=mapping(ng),raw_geometry=mapping(g),geometry=mapping(g),source_valid_fraction=vf,source_edge_partial_risk=False,source_conflict=False,
                source_parent_ids=[a['parent_id'] for a in p['parents']],source_hashes=[a['payload_sha256'] for a in p['parents']],
                source_payload_sha256=sha(out/uid/'search_RGB.png'),rgb_pixel_sha256=p['rgb_pixel_sha256'],source_acquisition_id=p.get('source_acquisition_id'),
                mean_probability_uncalibrated=r['mean_probability_uncalibrated'],model_checkpoint_sha256=r['model_sha256'],shift_east_north_m=None,
                position_candidate_status='ACTIVE_SOURCE_NATIVE_UNKNOWN_ABSOLUTE_POSITION',area_raw_map_m2=g.area,area_after_translation_m2=g.area,area_on_reference_plane_m2=g.area,
                area_physical_rectified_m2=None,physical_area_status='UNRESOLVED_NO_VIEW_GEOMETRY',excluded=False,
                aspect=aspect(g,False,read(c['m1_recognition']['aspect_screen']['rule_manifest'])['threshold']),
                active_recheck_lineage=dict(request_id=uid,actual_current_RGB=True,independent_detector=False))
            seed['transformation_chain']=dict(source_frame=p['crs'],target_frame=seed['metric_crs'],native_affine=p['affine'],V2_total_translation=None,physical_transform=None,compose_once=True)
            recovered.append(seed);current[p['epoch_id']].append(g)
    for r in ledger:
        if r['status']=='SCHEDULED':r['status']='EXECUTED'
    lines(out/'ACTIVE_SEARCH_LEDGER.jsonl',ledger);lines(out/'SOURCE_CONTEXT_VERSIONS.jsonl',versions);lines(out/'RECOVERED_OBSERVATIONS.jsonl',recovered)
    save(out/'COMPLETE.json',dict(status='COMPLETE_PREDECLARED_SEARCH_SET',planned_nodes=len(unresolved),ledger_rows=len(ledger),unique_contexts=len(jobs),
        dedup_references=sum(r['status']=='DEDUPLICATED_CONTEXT_REFERENCE' for r in ledger),unsupported=sum(r['status']=='UNSUPPORTED_WINDOW_RETAIN_U' for r in ledger),
        recovered_observations=len(recovered),same_model_context_versions_not_independent=True,absence_promotions=0))

def graph_partition(obs,edges):
    """Binary all-pairs equivalence optimization, explicit null and split exits.
    Transitivity with unavailable/same-epoch edges fixed to zero prevents two
    instances of one epoch entering the same track. No adjacent-only chaining.
    """
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    ids=[r['observation_id'] for r in obs];lookup={v:i for i,v in enumerate(ids)}
    pairs={tuple(sorted((lookup[e['source']],lookup[e['target']]))):e for e in edges if e['eligible']}
    keys=sorted(pairs);vi={p:i for i,p in enumerate(keys)};n=len(keys)
    if not n: labels=[[r] for r in obs];return make_tracks(labels),[],[]
    # Preserve each classifier's frozen SAME-vs-strongest-alternative rule.
    # A 0.40/0.30/0.30 strict-argmax SAME is not a negative-weight edge.
    weights=np.array([math.log(max(pairs[p]['same'],1e-6)/max(pairs[p].get('different',(1-pairs[p]['same'])/2),pairs[p].get('null',(1-pairs[p]['same'])/2),1e-6)) for p in keys])
    incident={i:[] for i in range(len(ids))}
    for p in keys:
        a,b=p;incident[a].append(b);incident[b].append(a)
    rr=[];cc=[];vv=[];ubs=[];constraints=set()
    for i,neighbors in incident.items():
        for j,k in itertools.combinations(neighbors,2):
            ij=vi[tuple(sorted((i,j)))];ik=vi[tuple(sorted((i,k)))];jk=vi.get(tuple(sorted((j,k))))
            row=tuple(sorted([(ij,1),(ik,1)]+([] if jk is None else [(jk,-1)])))
            if row in constraints:continue
            constraints.add(row);ri=len(ubs);ubs.append(1)
            for c,v in row:rr.append(ri);cc.append(c);vv.append(v)
    cons=LinearConstraint(coo_matrix((vv,(rr,cc)),shape=(len(ubs),n)).tocsc(),-np.inf,np.array(ubs)) if ubs else None
    result=milp(-weights,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=cons,options={'time_limit':120})
    if not result.success:raise RuntimeError('all-sequence optimization did not prove optimum: '+result.message)
    parent=list(range(len(ids)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    accepted=[]
    for k,x in enumerate(result.x):
        if x>.5:
            a,b=keys[k];parent[find(a)]=find(b);accepted.append(pairs[keys[k]])
    groups={}
    for i,r in enumerate(obs):groups.setdefault(find(i),[]).append(r)
    for rows in groups.values():assert len({r['epoch_id'] for r in rows})==len(rows)
    return make_tracks(list(groups.values())),accepted,[dict(status='OPTIMAL',objective=float(-result.fun),constraints=len(ubs),edges=n)]
def make_tracks(groups):
    tracks=[]
    for rows in groups:
        rep=medoid(rows);tracks.append(dict(track_id_candidate='LT_'+digest(sorted(r['observation_id'] for r in rows))[:24],observation_ids=[r['observation_id'] for r in rows],physical_id=None,
            canonical_observation_id=rep['observation_id'],geometry=rep['geometry'],canonical_policy='BEST_OBSERVATION_MEDOID',canonical_level='LEARNED_RESEARCH_CANDIDATE',geometry_version=digest(rep['geometry']),position_sigma_m=None))
    return tracks
def b2_edges(obs):
    from urban_monitor.building_tracker.checkpoint import load_checkpoint
    from urban_monitor.building_tracker.features import AssociationFeatureBuilder,standardize_features
    from urban_monitor.building_tracker.candidate_gate import GateProfile,build_candidate_edges
    from urban_monitor.building_tracker.contracts import BuildingInstance
    path=ROOT/'models/RECOVERED/B2.pth'
    model,mean,std,meta=load_checkpoint(path);profile=GateProfile('G1_balanced',24.,8.,.02,3.,16);builder=AssociationFeatureBuilder(profile=profile)
    instances=[BuildingInstance('XIAN','BOUNDED_APP',r['actual_date'],r['observation_id'],None,shape(r['geometry']),r['metric_crs'],
        confidence=min(1,max(0,r.get('mean_probability_uncalibrated',1))),validity_fraction=r.get('source_valid_fraction',1),source_kind='frozen_prediction') for r in obs]
    all_edges=[]
    for a,b in itertools.combinations(sorted({r.epoch_id if hasattr(r,'epoch_id') else r.date for r in instances}),2):
        ee=build_candidate_edges([r for r in instances if r.date==a],[r for r in instances if r.date==b],profile=profile)
        if not ee:continue
        x,_,_=standardize_features([builder.build_edge(e) for e in ee],mean,std)
        with torch.no_grad():probs=model(torch.from_numpy(x)).softmax(-1).numpy()
        for e,p in zip(ee,probs):all_edges.append(dict(source=e.left.source_instance_id,target=e.right.source_instance_id,same=float(p[0]),different=float(p[1]),null=float(p[2]),eligible=int(p.argmax())==0,route='FROZEN_B2_22_FEATURE_STRICT_ARGMAX'))
    return all_edges,dict(checkpoint=str(path),sha256=meta['checkpoint_sha256'],registration_RMS='protocol constant0; unknown measurement',target_accuracy=None)

def learned_edges(obs,packets,seed,family='compact',geometry_frame='raw'):
    run=(ROOT/'runs/T2_FULL_SPAN' if family=='fullspan' else ROOT/'runs')/('seed_'+str(seed));cp=run/'best.pt';model=TemporalIdentityNet();state=torch.load(cp,map_location='cpu');model.load_state_dict(state['model'],strict=True);model.eval()
    cal=read(run/'CALIBRATION.json');point=cal['same_edge_operating_point'];threshold=point['threshold'] if point else 1.01
    pp={p['epoch_id']:p for p in packets};epochs=sorted(pp,key=lambda e:pp[e]['actual_date']);rgb={e:np.asarray(Image.open(pp[e]['image']).convert('RGB')) for e in epochs}
    reference=packets[0]
    if not all(p['affine']==reference['affine'] and p['crs']==reference['crs'] for p in packets):
        raise ValueError('Fixed-fixture research comparison requires an explicit common native raster frame')
    # Training used pixel X right / Y down, not metric easting / northing.
    # Area/perimeter normalization is dimensionless but does not fix axis signs
    # or degree-grid anisotropy. Preserve the exact native feature convention.
    feature_geometries={r['observation_id']:(shape(r['native_geometry']) if geometry_frame=='raw' else map_geometry(convert(shape(r['geometry']),r['metric_crs'],reference['crs']),~Affine(*reference['affine']))) for r in obs}
    byepoch={e:[r for r in obs if r['epoch_id']==e] for e in epochs};k=max(len(v) for v in byepoch.values());scores={};predictions=[]
    from datetime import date
    for anchor in obs:
        ap=pp[anchor['epoch_id']];poly=shape(anchor['native_geometry']);q=crop(rgb[anchor['epoch_id']],poly);cs=[];gs=[];ss=[];times=[]
        for e in epochs:
            p=pp[e];g=map_geometry(convert(shape(anchor['raw_geometry']),anchor['metric_crs'],p['crs']),~Affine(*p['affine']))
            gap=(date.fromisoformat(p['actual_date'])-date.fromisoformat(anchor['actual_date'])).days/365.25
            cc=[];gg=[]
            for r in byepoch[e]:
                cc.append(crop(rgb[e],shape(r['native_geometry'])));gg.append(geometry(feature_geometries[anchor['observation_id']],feature_geometries[r['observation_id']],gap,min(anchor['source_valid_fraction'],r['source_valid_fraction'])))
            while len(cc)<k:cc.append(np.zeros_like(q));gg.append(np.zeros(8,np.float32))
            cs.append(cc);gs.append(gg);ss.append(crop(rgb[e],g,side=max(32.,max(g.bounds[2]-g.bounds[0],g.bounds[3]-g.bounds[1])*2.5)));times.append([gap,float(e==anchor['epoch_id'])])
        def tensor(x,scale=False):return torch.from_numpy(np.asarray(x)).float().unsqueeze(0)/(255. if scale else 1.)
        with torch.no_grad():o=score_candidate_set(model,tensor(q,True),tensor(cs,True),tensor(gs),tensor(ss,True),tensor(times))
        p=o['association'].softmax(-1)[0].numpy()
        for j,e in enumerate(epochs):
            predictions.append(dict(anchor_observation_id=anchor['observation_id'],epoch_id=e,labelled_presence_probability=float(o['presence'].sigmoid()[0,j]),alpha_support_probability=float(o['observable'].sigmoid()[0,j]),
                scope='SN7_label_record_head_OFF_DOMAIN_UNCALIBRATED; does not establish visual absence',absence_admitted=False))
            if e==anchor['epoch_id']:
                self_index=next(jj for jj,rr in enumerate(byepoch[e]) if rr['observation_id']==anchor['observation_id'])
                predictions[-1]['same_image_self_control']=p[j,self_index].tolist()
            for j2,r in enumerate(byepoch[e]):
                if r['epoch_id']!=anchor['epoch_id']:scores[(anchor['observation_id'],r['observation_id'])]=p[j,j2]
    edges=[]
    for a,b in itertools.combinations(obs,2):
        if a['epoch_id']==b['epoch_id']:continue
        p=scores[(a['observation_id'],b['observation_id'])];q=scores[(b['observation_id'],a['observation_id'])]
        same=float(min(p[0],q[0])); eligible=bool(p.argmax()==0 and q.argmax()==0 and same>=threshold)
        edges.append(dict(source=a['observation_id'],target=b['observation_id'],same=same,different=float(max(p[1],q[1])),null=float(max(p[2],q[2])),
            forward=p.tolist(),reverse=q.tolist(),eligible=eligible,route='TRAINED_RGB_GEOMETRY_TEMPORAL_GRU',threshold=threshold,
            calibration_scope='SN7_SOURCE_INTERNAL_ONLY; target absolute calibration unknown',independent_directional_evidence=False))
    return edges,predictions,dict(checkpoint=str(cp),sha256=sha(cp),parameters=state['parameters'],source_same_threshold=threshold,
        source_calibration_point=point,target_accuracy=None,model_training=True,association_feature_geometry=geometry_frame,
        geometry_feature_contract='PIXEL_X_RIGHT_Y_DOWN_SAME_AS_TRAINING',
        geometry_qualification='SOURCE_NATIVE_PIXEL' if geometry_frame=='raw' else 'V2_POSITION_MODEL_CANDIDATE_CONVERTED_TO_COMMON_NATIVE_PIXEL_NOT_ABSOLUTE_CERTIFICATION',
        temporal_context='all actual-date epochs, retrospective; nearest3 pooled context as training; all candidate pairs scored')
def route(variant,seed=20261027,expanded=False):
    tag=variant+('_SEED'+str(seed) if seed!=20261027 and not variant.startswith('B2_BASELINE') else '')
    c,packets,obs,oldtracks=inputs();out=ROOT/'runs'/(tag+('_EXPANDED_VALIDATED' if expanded else '_FIXED55'));out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():
        if (out/'RUN_PRODUCTS_SEAL.json').exists():
            for name,expected in read(out/'RUN_PRODUCTS_SEAL.json')['files'].items():
                if sha(out/name)!=expected:raise ValueError('Completed comparison artifact changed: '+name)
        model_meta=read(out/'ROUTE.json')['model']
        if sha(model_meta['checkpoint'])!=model_meta['sha256']:raise ValueError('Completed route checkpoint changed')
        return
    if expanded:obs=obs+jsonl(ROOT/'runs/ACTIVE_EXPANDED/RECOVERED_OBSERVATIONS_VALIDATED.jsonl')
    if variant.startswith('B2_BASELINE'):edges,meta=b2_edges(obs);predictions=[]
    else:edges,predictions,meta=learned_edges(obs,packets,seed,'fullspan' if variant.startswith('LEARNED_FULLSPAN') else 'compact','managed' if variant.endswith('_MANAGED') else 'raw')
    tracks,accepted,solver=graph_partition(obs,edges);mapping_ids={oid:t['track_id_candidate'] for t in tracks for oid in t['observation_ids']};byid={r['observation_id']:r for r in obs}
    # Topology conflicts stay independent from image observation.
    conflicts=set()
    for a,b in itertools.combinations(obs,2):
        if a['epoch_id']==b['epoch_id']:continue
        ga,gb=shape(a['raw_geometry']),shape(b['raw_geometry']);inter=ga.intersection(gb).area
        if inter/max(min(ga.area,gb.area),1e-9)>.2:
            other=[r for r in obs if r['epoch_id']==b['epoch_id'] and ga.intersection(shape(r['raw_geometry'])).area/max(min(ga.area,shape(r['raw_geometry']).area),1e-9)>.2]
            if len(other)>1:conflicts.update([a['observation_id'],b['observation_id']]+[r['observation_id'] for r in other])
    presence=[]
    for t in tracks:
        for e in c['inputs']['epochs']:
            here=[byid[i] for i in t['observation_ids'] if byid[i]['epoch_id']==e['epoch_id']]
            seen=bool(here);r=here[0] if seen else None
            full=bool(seen and not r['source_edge_partial_risk'] and r['source_valid_fraction']>=.98)
            presence.append(dict(track_id_candidate=t['track_id_candidate'],epoch_id=e['epoch_id'],actual_date=e['actual_date'],presence_candidate=1 if seen else 'U',
                source_roof_observed=seen,source_observable='ALPHA_SUPPORTED_ONLY' if full else 'PARTIAL_OR_NOT_OBSERVED',
                identity_resolved_candidate=len(t['observation_ids'])>1 and not any(i in conflicts for i in t['observation_ids']),
                relative_match_candidate=len(t['observation_ids'])>1,absolute_position_qualified=False,presence_qualified='U',physical_id=None,
                event_eligible=full and not any(i in conflicts for i in t['observation_ids']),
                reason='CURRENT_RGB_B0_OBSERVATION_INDEPENDENT_OF_PHYSICAL_ID' if seen else 'NO_MATCH_NOT_ABSENCE',confidence_calibrated=None))
    lines(out/'LEARNED_EDGES.jsonl',edges);lines(out/'ACCEPTED_GRAPH_EDGES.jsonl',accepted);lines(out/'LABEL_RECORD_PRESENCE_HEAD.jsonl',predictions)
    save(out/'ROUTE.json',dict(model=meta,solver=solver,variant=variant,source_observations=len(obs),target_GT_reads=0,certified_events=0))
    pointer=out/'PRODUCTS_POINTER.json'
    if pointer.exists():summary=read(pointer)['summary']
    else:
        attempt=1;products=out/'PRODUCTS'
        while products.exists():attempt+=1;products=out/('PRODUCTS_ATTEMPT_%03d'%attempt)
        products.mkdir();summary=app.product_exports(c,packets,obs,tracks,presence,[],edges,conflicts,products)
        save(pointer,dict(path=str(products),summary=summary,completed=True))
    # Fixed old-node lineage, preserving every old denominator and geometry.
    newpp={(r['track_id_candidate'],r['epoch_id']):r for r in presence};oldby={t['track_id_candidate']:t for t in oldtracks};lineage=[]
    for node in read(ROOT/'evidence/T0_UNKNOWN_ATTRIBUTION.json')['old_unknown_nodes']:
        t=oldby[node['track_id_candidate']];newids=sorted({mapping_ids[i] for i in t['observation_ids']});states=[newpp[(i,node['epoch_id'])] for i in newids]
        # If the old track splits, no arbitrary descendant is chosen to lower U.
        comparable=len(newids)==1;value=states[0]['presence_candidate'] if comparable else 'U'
        lineage.append(dict(**node,new_tracks=newids,comparable_single_descendant=comparable,new_presence_candidate=value,
            source_roof_observed_any=any(r['source_roof_observed'] for r in states),old_geometry=t['geometry'],old_geometry_area_m2=shape(t['geometry']).area,
            formal_presence='U',new_reason='OLD_TRACK_SPLIT_RETAIN_U' if not comparable else states[0]['reason']))
    lines(out/'OLD_120_UNKNOWN_LINEAGE.jsonl',lineage)
    save(out/'COMPLETE.json',dict(status='RESEARCH_COMPARISON_EXPORTED',variant=variant,observations=len(obs),tracks=len(tracks),accepted_edges=len(accepted),
        presence={str(v):sum(r['presence_candidate']==v for r in presence) for v in [1,0,'U']},fixed_old_U_nodes=120,
        old_U_now_observed=sum(r['new_presence_candidate']==1 for r in lineage),old_U_remaining=sum(r['new_presence_candidate']=='U' for r in lineage),
        old_U_to_absent=sum(r['new_presence_candidate']==0 for r in lineage),target_accuracy=None,formal_events=0,summary=summary))
def main():
    torch.set_num_threads(4);p=argparse.ArgumentParser();p.add_argument('action',choices=['active','b2','learned']);p.add_argument('--expanded',action='store_true');p.add_argument('--seed',type=int,default=20261027);p.add_argument('--family',choices=['compact','fullspan'],default='fullspan');p.add_argument('--geometry',choices=['raw','managed'],default='raw');a=p.parse_args()
    if a.action=='active':active()
    else:route('B2_BASELINE_ARGMAX' if a.action=='b2' else (('LEARNED_FULLSPAN_NATIVE' if a.family=='fullspan' else 'LEARNED_CARDINALITY_NATIVE')+('_MANAGED' if a.geometry=='managed' else '')),a.seed,a.expanded)
if __name__=='__main__':main()
