"""Frozen model caches -> label-blind native observation graph -> separate weak labels.

Map-reference labels are geometrically admitted associations, not verified physical IDs.
No RGB pixels or map geometries are read by this preparation; GT label metadata is
opened only after observations, numeric features and candidate graph are SHA-frozen.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,time,collections,traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import sparse,ndimage
from scipy.spatial import cKDTree
from affine import Affine
from pyproj import Transformer
import rasterio

R=Path('/root/xian20261111_r14_native_temporal');P=Path('/root/xian20261111_r8');O=R/'output';O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def worker(r):
 d=O/'frames'/r['id'];d.mkdir(parents=True,exist_ok=True);receipt=d/'RECEIPT.json'
 if receipt.exists():
  q=json.loads(receipt.read_text());assert sha(d/'OBSERVATIONS.json')==q['observations_sha256'] and sha(d/'FEATURES.npz')==q['features_sha256'];return q
 assert r['role'] in ['TRAIN','TUNE'];assert sha(r['query_cache_path'])==r['query_cache_sha256']
 z=np.load(r['query_cache_path']);lab=z['instances'].astype(np.int32);support=z['support'];prob=z['probability'];features=z['features'].astype(np.float32);h,w=lab.shape;assert (h,w)==(1024,1024) and features.shape==(64,256,256)
 n=int(lab.max())+1;counts=np.bincount(lab.ravel(),minlength=n);y,x=np.indices(lab.shape);cx=np.bincount(lab.ravel(),weights=x.ravel()+.5,minlength=n)/np.maximum(1,counts);cy=np.bincount(lab.ravel(),weights=y.ravel()+.5,minlength=n)/np.maximum(1,counts)
 # Sparse quarter-cell counts give exactly area-weighted 64-D cache features.
 cells=((y//4)*256+x//4).ravel();weights=sparse.coo_matrix((np.ones(lab.size,np.float32),(lab.ravel(),cells)),shape=(n,65536)).tocsr();emb=weights.dot(features.reshape(64,-1).T)/np.maximum(1,counts[:,None]);del weights,cells,features
 ps=np.bincount(lab.ravel(),weights=prob.ravel(),minlength=n)/np.maximum(1,counts);ss=np.bincount(lab.ravel(),weights=support.ravel(),minlength=n)/np.maximum(1,counts)
 aff=Affine(*r['native_affine']);proj=Transformer.from_crs(r['native_crs'],32649,always_xy=True);world=np.column_stack(proj.transform(aff.a*cx+aff.b*cy+aff.c,aff.d*cx+aff.e*cy+aff.f));
 codes=json.loads((P/'ACTUAL_DATE_CODE_INDEX.json').read_text())[r['epoch']]
 with rasterio.open(r['date_path']) as ds:date=ds.read(1)
 observations=[];vectors=[]
 for ident,sl in enumerate(ndimage.find_objects(lab),1):
  if sl is None or counts[ident]<32:continue
  q=lab[sl]==ident;v,c=np.unique(date[sl][q],return_counts=True);i=c.argmax();day=codes.get(str(int(v[i])),'UNKNOWN');dominance=float(c[i]/c.sum());border=sl[0].start==0 or sl[1].start==0 or sl[0].stop==h or sl[1].stop==w
  # Local projected Jacobian is an input shape feature, never delivered area.
  a=np.array(proj.transform(*(aff*(cx[ident]+1,cy[ident]))))-world[ident];b=np.array(proj.transform(*(aff*(cx[ident],cy[ident]+1))))-world[ident];pixel_area=float(abs(np.linalg.det(np.stack([a,b]))));width=sl[1].stop-sl[1].start;height=sl[0].stop-sl[0].start
  observations.append(dict(uid=r['id']+'_q'+str(ident),frame=r['id'],role=r['role'],site=r['site'],period=r['epoch'],mask_id=ident,center_utm=world[ident].tolist(),area_native_pixels=int(counts[ident]),area_feature_m2=float(counts[ident]*pixel_area),area_feature_semantics='LOCAL_LINEARIZED_PROJECTED_AREA_NOT_PRODUCT_AREA',aspect=float(width/height),bounds_native=[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop],border=border,p_roof=float(ps[ident]),source_support=float(ss[ident]),actual_date=day,date_dominance=dominance,actual_date_known=day!='UNKNOWN' and dominance>=.95,GT_in_input=False));vectors.append(emb[ident])
 save(d/'OBSERVATIONS.json',observations);np.savez_compressed(d/'FEATURES.npz',feature64=np.asarray(vectors,dtype=np.float32).reshape(-1,64));q=dict(id=r['id'],role=r['role'],site=r['site'],period=r['epoch'],query_cache_sha256=r['query_cache_sha256'],observations=len(observations),observations_sha256=sha(d/'OBSERVATIONS.json'),features_sha256=sha(d/'FEATURES.npz'),GT_read=False);save(receipt,q);return q

def main():
 records=[json.loads(l) for l in (P/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l];records=[r for r in records if r['role'] in ['TRAIN','TUNE']];assert len(records)==1962
 save(O/'PROTOCOL_FROZEN.json',dict(task='NATIVE_PREDICTED_OBSERVATIONS_MAP_REFERENCE_ASSOCIATION_RESEARCH',roles=['TRAIN','TUNE'],frames=len(records),query_generator='FROZEN_R7_NATIVE_IMAGE_ONLY_R8_CACHE',min_native_pixels=32,source_RGB_reads=0,GT_read_only_after_graph_freeze=True,GT_missing='NO_SUPERVISED_LOSS_RETAIN_IN_GRAPH',weak_GT_semantics='GEOMETRICALLY_ADMITTED_MAP_REFERENCE_NOT_VERIFIED_PHYSICAL_IDENTITY',candidate_radius_m=64,candidate_maximum=32,all_cross_periods=True,all_observations_initialize=True,actual_date='PER_OBSERVATION_DOMINANT_REGISTERED_DAY_GE0.95_UNKNOWN_RETAINED_NO_PHYSICAL_CHANGE_CLAIMS',nominal_period='ORDER_ONLY_NOT_ASSUMED_YEAR',candidate_ground_truth_insertion=False,evaluation='REUSED_SPATIALLY_PURGED_DEVELOPMENT_PARTIAL_LABEL_COVERAGE_NOT_SCENE_ACCURACY',protected_reads=0,production_changed=False))
 save(O/'IMAGE_ONLY_ROLE_INPUTS.json',[{k:r[k] for k in ['id','role','epoch','site','query_cache_path','query_cache_sha256','native_affine','native_crs','date_path','support_path']} for r in records])
 with ProcessPoolExecutor(max_workers=4) as ex:
  receipts=[]
  for i,q in enumerate(ex.map(worker,records)):
   receipts.append(q)
   if i%25==0:save(O/'STATUS.json',dict(time=time.time(),stage='POOL_NATIVE_OBSERVATION_FEATURES_CPU_ONLY',frames=i+1,total=len(records)));print('POOLED',i+1,len(records),flush=True)
 nodes=[];vectors=[];groups=collections.defaultdict(list)
 for r in records:groups[(r['role'],r['site'])]
 for q in receipts:
  d=O/'frames'/q['id'];rr=json.loads((d/'OBSERVATIONS.json').read_text());ff=np.load(d/'FEATURES.npz')['feature64'];assert len(rr)==len(ff)
  for r,f in zip(rr,ff):r['index']=len(nodes);groups[(r['role'],r['site'])].append(r['index']);nodes.append(r);vectors.append(f)
 feature_path=O/'FEATURES_ALL.npz';np.savez_compressed(feature_path,feature64=np.array(vectors,np.float32));save(O/'OBSERVATIONS_ALL.json',nodes);windows=[]
 for (role,site),indices in sorted(groups.items()):
  byperiod=collections.defaultdict(list)
  for i in indices:byperiod[nodes[i]['period']].append(i)
  queries=[]
  for period in ['xa180429','xa190101','xa201231']:
   targets=byperiod[period];tree=cKDTree([nodes[i]['center_utm'] for i in targets]) if targets else None
   for i in indices:
    if nodes[i]['period']==period:continue
    if tree is None:candidates=[]
    else:
     jj=tree.query_ball_point(nodes[i]['center_utm'],64);candidates=sorted((targets[j] for j in jj),key=lambda j:(np.linalg.norm(np.array(nodes[i]['center_utm'])-nodes[j]['center_utm']),nodes[j]['uid']))[:32]
    queries.append(dict(query=i,target_period=period,candidates=candidates))
  windows.append(dict(role=role,site=site,nodes=indices,frames=[r['id'] for r in records if r['role']==role and r['site']==site],queries=queries))
 save(O/'CANDIDATE_GRAPH.json',windows);freeze=dict(observation_sha256=sha(O/'OBSERVATIONS_ALL.json'),feature_sha256=sha(feature_path),candidate_graph_sha256=sha(O/'CANDIDATE_GRAPH.json'),frame_receipts=receipts,observations=len(nodes),windows=len(windows),directed_queries=sum(len(w['queries']) for w in windows),GT_labels_read_for_generation=False);save(O/'GRAPH_FROZEN_BEFORE_GT.json',freeze)
 # First identity-supervision file read occurs below this point.
 source=P/'output/rgb_alignment/DATASET_FROZEN.json';data=json.loads(source.read_text());lookup={r['id']:r for r in data['rows'] if r['role'] in ['TRAIN','TUNE']};labels={};byframe=collections.defaultdict(dict)
 cache_sha={q['id']:q['query_cache_sha256'] for q in receipts}
 for n in nodes:
  r=lookup.get(n['uid'])
  if r is not None:
   assert r['role']==n['role'];assert r['cache_sha256']==cache_sha[n['frame']];labels[str(n['index'])]=dict(reference_id=r['reference_id'],reference_source=r['reference_source'],supervision='WEAK_GEOMETRIC_MAP_REFERENCE_PAIR',weight=1.);assert r['reference_id'] not in byframe[n['frame']];byframe[n['frame']][r['reference_id']]=n['index']
 train_refs={r['reference_id'] for i,r in labels.items() if nodes[int(i)]['role']=='TRAIN'};tune_refs={r['reference_id'] for i,r in labels.items() if nodes[int(i)]['role']=='TUNE'};assert not train_refs&tune_refs
 counts=collections.defaultdict(collections.Counter);frame_index=collections.defaultdict(list)
 for r in records:frame_index[(r['role'],r['site'],r['epoch'])].append(r['id'])
 for w in windows:
  for q in w['queries']:
   lab=labels.get(str(q['query']));counts[w['role']]['all_queries']+=1
   if lab is None:continue
   counts[w['role']]['labelled_queries']+=1;targetframes=frame_index[(w['role'],w['site'],q['target_period'])];positives=[byframe[f][lab['reference_id']] for f in targetframes if lab['reference_id'] in byframe[f]]
   if not positives:counts[w['role']]['counterpart_unverified_no_absence_loss']+=1;continue
   counts[w['role']]['labelled_counterpart_observed']+=1
   if any(p in q['candidates'] for p in positives):counts[w['role']]['candidate_hit']+=1
   else:counts[w['role']]['candidate_recall_failure_kept']+=1
 save(O/'SUPERVISION_SEPARATE.json',dict(labels=labels,input_sha256=sha(source),GT_contract='NO_GT_QUERY_NO_LOSS; UNLABELLED_CANDIDATES_MASKED_FROM_SUPERVISED_LOGITS; NO_LABELLED_COUNTERPART_IS_UNVERIFIED_NOT_PHYSICAL_ABSENCE',reference_overlap=0));save(O/'PREPARATION_RECEIPT.json',dict(status='COMPLETE_NATIVE_LABEL_BLIND_GRAPH_AND_SEPARATE_WEAK_SUPERVISION',frames=len(records),nodes=len(nodes),windows=len(windows),features_shape=[len(nodes),64],labels=len(labels),coverage={k:dict(v) for k,v in counts.items()},reference_overlap=0,protected_reads=0,GT_pixels_read=0,RGB_pixels_read=0,GPU_forwards=0,optimizer_updates=0,production_changed=False));print('NATIVE_GRAPH_COMPLETE',len(nodes),len(labels),dict(counts),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(O/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
