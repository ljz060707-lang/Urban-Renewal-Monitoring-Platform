"""Prediction-to-prediction centered mask IoU; no GT parameter or label read."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,collections,traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from native_temporal_contract import O,save,sha
D=O/'pair_shape';D.mkdir(exist_ok=True)
NODES=None;FRAMES=None
def centered_iou(a,b):
 dx,dy=np.rint(a['center']-b['center']).astype(int);ax,ay=a['x'],a['y'];bx,by=b['x']+dx,b['y']+dy;x0=max(ax,bx);y0=max(ay,by);x1=min(ax+a['mask'].shape[1],bx+b['mask'].shape[1]);y1=min(ay+a['mask'].shape[0],by+b['mask'].shape[0]);inter=int((a['mask'][y0-ay:y1-ay,x0-ax:x1-ax]&b['mask'][y0-by:y1-by,x0-bx:x1-bx]).sum()) if x1>x0 and y1>y0 else 0
 return inter/max(1,a['area']+b['area']-inter)
def process(w):
 out=D/'windows'/(w['role']+'_'+w['site']+'.npz');out.parent.mkdir(exist_ok=True)
 if out.exists():return dict(path=str(out.relative_to(D)),sha256=sha(out),site=w['site'],role=w['role'])
 maps={};masks={};basis=None
 for frame in w['frames']:
  f=FRAMES[frame];assert f['role'] in ['TRAIN','TUNE'];assert sha(f['query_cache_path'])==f['query_cache_sha256'];a=f['native_affine'];b=(f['native_crs'],a[0],a[1],a[3],a[4])
  if basis is None:basis=b
  assert b[0]==basis[0] and np.allclose(b[1:],basis[1:],rtol=1e-6,atol=1e-14),'Native grids require explicit common-metric shape adapter'
  maps[frame]=np.load(f['query_cache_path'])['instances']
 for i in w['nodes']:
  n=NODES[i];x0,y0,x1,y1=n['bounds_native'];mask=maps[n['frame']][y0:y1,x0:x1]==n['mask_id'];yy,xx=np.where(mask);assert len(yy)==n['area_native_pixels'];masks[i]=dict(mask=mask,x=x0,y=y0,area=len(yy),center=np.array([xx.mean()+x0+.5,yy.mean()+y0+.5]))
 memo={};rows=[]
 for q in w['queries']:
  i=q['query'];periods=[p for p in ['xa180429','xa190101','xa201231'] if p!=NODES[i]['period']];t=periods.index(q['target_period'])
  for k,j in enumerate(q['candidates']):
   key=tuple(sorted([i,j]))
   if key not in memo:
    ab=centered_iou(masks[i],masks[j]);ba=centered_iou(masks[j],masks[i]);assert abs(ab-ba)<1e-8;memo[key]=ab
   rows.append((i,t,k,memo[key]))
 np.savez_compressed(out,rows=np.asarray(rows,np.float64).reshape(-1,4));return dict(path=str(out.relative_to(D)),sha256=sha(out),site=w['site'],role=w['role'],unique_pairs=len(memo),directed_scores=len(rows))
def main():
 global NODES,FRAMES
 NODES=json.loads((O/'OBSERVATIONS_ALL.json').read_text());FRAMES={r['id']:r for r in json.loads((O/'IMAGE_ONLY_ROLE_INPUTS.json').read_text())};windows=json.loads((O/'CANDIDATE_GRAPH.json').read_text());save(D/'PROTOCOL_FROZEN.json',dict(task='NATIVE_PREDICTED_MASK_TO_PREDICTED_MASK_CENTERED_SHAPE_IOU',graph_sha256=sha(O/'CANDIDATE_GRAPH.json'),candidate_graph_changed=False,GT_ids_or_geometry_read=False,source_RGB_reads=0,mask_source='SHA_FROZEN_R8_IMAGE_ONLY_QUERIES',centering='NATIVE_MASK_PIXEL_CENTROIDS_ROUNDED_TRANSLATION;SYMMETRY_CHECKED',grid='MUST_MATCH_CRS_NATIVE_PIXEL_BASIS_WITHIN1PPM_OR_FAIL_NO_SILENT_RESIZE',missing_edge='UNKNOWN_NOT_CANNOT_LINK',physical_identity_evidence='SHAPE_COMPATIBILITY_ONLY_NOT_INDEPENDENT_SAME_ID_CERTIFICATION',protected_reads=0,production_changed=False));records=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(process,windows)):
   records.append(r)
   if i%25==0:save(D/'STATUS.json',dict(time=time.time(),stage='PAIR_SHAPE_CPU',windows=i+1,total=len(windows)));print('PAIR_SHAPE',i+1,len(windows),flush=True)
 feature=np.zeros((len(NODES),2,32),np.float32);count=0
 for r in records:
  z=np.load(D/r['path'])['rows'];idx=z[:,:3].astype(np.int64);feature[idx[:,0],idx[:,1],idx[:,2]]=z[:,3];count+=len(z)
 np.savez_compressed(D/'CENTERED_SHAPE_FEATURES.npz',shape_IoU=feature);save(D/'PREPARATION_RECEIPT.json',dict(status='COMPLETE_GT_FREE_CENTERED_MODEL_MASK_PAIR_FEATURES',windows=len(windows),directed_pair_scores=count,features_shape=list(feature.shape),minimum=float(feature.min()),maximum=float(feature.max()),features_sha256=sha(D/'CENTERED_SHAPE_FEATURES.npz'),records=records,GT_input=False,protected_reads=0,optimizer_updates=0));print('PAIR_SHAPE_COMPLETE',count,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(D/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
