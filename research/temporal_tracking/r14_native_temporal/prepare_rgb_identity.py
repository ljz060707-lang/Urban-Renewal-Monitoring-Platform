"""Native predicted-roof RGB/mask crops; fit known labels, full inner/TUNE graphs.

No GT shape crops or inserted proposals. Scope is identity appearance, not a
resampled geometric product. Original affine/CRS and query IDs remain intact.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,hashlib,time,traceback,collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np,cv2,rasterio
from native_temporal_contract import O,sha,save
R=O.parent;D=O/'rgb_identity';D.mkdir(exist_ok=True)
def process(task):
 frame,rows=task;dest=D/'crops'/(frame['id']+'.npz');receipt=dest.with_suffix('.json');dest.parent.mkdir(exist_ok=True)
 if receipt.exists():
  v=json.loads(receipt.read_text());assert sha(dest)==v['sha256'];return v
 assert frame['role'] in ['TRAIN','TUNE'];assert sha(frame['query_cache_path'])==frame['query_cache_sha256']
 source_records=[json.loads(l) for l in (Path('/root/xian20261111_r8')/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l];source=next(r for r in source_records if r['id']==frame['id']);assert sha(source['image_path'])==source['image_sha256']
 with rasterio.open(source['image_path']) as ds:rgb=ds.read([1,2,3]).transpose(1,2,0)
 labels=np.load(frame['query_cache_path'])['instances'];assert labels.shape==rgb.shape[:2]==(1024,1024);crops=[];crop_meta=[];cv2.setNumThreads(0)
 for node in rows:
  x0,y0,x1,y1=node['bounds_native'];side=max(48,int(np.ceil(max(x1-x0,y1-y0)*1.4)));left=np.floor((x0+x1)/2-side/2);top=np.floor((y0+y1)/2-side/2);v=np.arange(128,dtype=np.float32)+.5;xx,yy=np.meshgrid(left+v*side/128-.5,top+v*side/128-.5);color=cv2.remap(rgb,xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);mask=cv2.remap((labels==node['mask_id']).astype('uint8'),xx.astype('float32'),yy.astype('float32'),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0);crops.append(np.concatenate([color.transpose(2,0,1),mask[None]*255],axis=0));crop_meta.append(dict(index=node['index'],native_pixel_edge_roi=[int(left),int(top),side],mask_id=node['mask_id'],crop_kind='PREDICTED_QUERY_BBOX_1.4_CONTEXT_NO_GT_GEOMETRY',output_size=128))
 np.savez_compressed(dest,crops=np.stack(crops),indices=np.array([r['index'] for r in rows],np.int64));value=dict(frame=frame['id'],role=frame['role'],path=str(dest.relative_to(D)),sha256=sha(dest),queries=len(rows),source_RGB_sha256=source['image_sha256'],source_cache_sha256=frame['query_cache_sha256'],native_affine=frame['native_affine'],native_crs=frame['native_crs'],GT_geometry_input=False,rows=crop_meta);save(receipt,value);return value
def main():
 nodes=json.loads((O/'OBSERVATIONS_ALL.json').read_text());labels=json.loads((O/'SUPERVISION_SEPARATE.json').read_text())['labels'];split=json.loads((O/'training/SPATIAL_ROLE_FREEZE.json').read_text());fit=set(split['fit_sites']);inner=set(split['inner_sites']);inner_refs={labels[str(n['index'])]['reference_id'] for n in nodes if n['role']=='TRAIN' and n['site'] in inner and str(n['index']) in labels};selected=[];fitids=[]
 for n in nodes:
  known_fit=n['role']=='TRAIN' and n['site'] in fit and str(n['index']) in labels and labels[str(n['index'])]['reference_id'] not in inner_refs
  evaluation=n['role']=='TUNE' or n['role']=='TRAIN' and n['site'] in inner
  if known_fit or evaluation:selected.append(n['index'])
  if known_fit:fitids.append(n['index'])
 save(D/'PROTOCOL_FROZEN.json',dict(task='NATIVE_RGB_PREDICTED_QUERY_INSTANCE_CONTRASTIVE_IDENTITY',basis='https://arxiv.org/abs/2004.11362',official_loss='https://github.com/HobbitLong/SupContrast/blob/master/losses.py',initial_encoder='TORCHVISION_IMAGENET1K_RESNET18;NO_UNQUALIFIED_BONAI_CHECKPOINT',fit_selection='ONLY_KNOWN_FIT_NODES_FROM_PREVIOUS_FROZEN_INNER_SPLIT',inner_and_TUNE='EVERY_MODEL_OBSERVATION_NO_GT_FILTERING',mask_input='PREDICTED_INSTANCE_ONLY_NOT_GT',RGB_crop=128,native_context_multiplier=1.4,min_native_side=48,resize_semantics='APPEARANCE_FEATURE_ROI_ONLY_NOT_NEW_SENSOR_DETAIL_OR_PRODUCT_GEOMETRY',GT_missing_training='EXCLUDED_NO_FALSE_NEGATIVES',GT_reference_semantics='WEAK_GEOMETRIC_MAP_REFERENCE_ID_ASSOCIATIONS',pair='DISTINCT_REGISTERED_PERIODS_SAME_REFERENCE;NOT_SAME_ACQUISITION_DUPLICATES',unknown='NO_PHYSICAL_ABSENCE_LEARNING',role_filter_before_pixels=True,protected_reads=0,production_changed=False))
 save(D/'CROP_ROLES_FROZEN_BEFORE_RGB_READ.json',dict(selected_node_indices=selected,fit_supervised_indices=fitids,inner_sites=split['inner_sites'],fit_sites=split['fit_sites'],graph_sha256=sha(O/'CANDIDATE_GRAPH.json'),source_supervision_sha256=sha(O/'SUPERVISION_SEPARATE.json')));group=collections.defaultdict(list)
 for i in selected:group[nodes[i]['frame']].append(nodes[i])
 frames=json.loads((O/'IMAGE_ONLY_ROLE_INPUTS.json').read_text());tasks=[(f,group[f['id']]) for f in frames if f['id'] in group];records=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(process,tasks)):
   records.append(r)
   if i%25==0:save(D/'STATUS.json',dict(time=time.time(),stage='CPU_NATIVE_RGB_CROPS',frames=i+1,total=len(tasks)));print('RGB_IDENTITY_CROPS',i+1,len(tasks),flush=True)
 save(D/'CROP_MANIFEST.json',dict(records=records,total_crops=len(selected),fit_known_crops=len(fitids),inner_and_TUNE_full_graph_crops=len(selected)-len(fitids),original_graph_nodes=len(nodes),protected_reads=0,GT_geometry_or_pixels_read=0,optimizer_updates=0));print('RGB_IDENTITY_CROPS_COMPLETE',len(selected),len(fitids),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(D/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
