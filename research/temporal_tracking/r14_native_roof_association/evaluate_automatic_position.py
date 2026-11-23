"""Frozen automatic predicted tracks -> frozen TRAIN-fit date/view position model.

Unlike R13's conditional grouping experiment, grouping here is produced by the
image/geometry association model, with no reference IDs supplied to inference.
Reference shape/coordinates are opened only after all position predictions freeze.
"""
from pathlib import Path
import sys,json,hashlib,time,collections,traceback
import numpy as np
from native_temporal_contract import O,save,sha
sys.path.insert(0,'/root/xian20261111_r13/code')
if not Path('/root/xian20261111_r13/code/multiview_reference_model.py').exists():sys.path.insert(0,'/root/xian20261111_r13')
from multiview_reference_model import infer
P=O/'automatic_position';P.mkdir(exist_ok=True);Q=Path('/root/xian20261111_r13/output');S=Path('/root/xian20261111_r8/output/rgb_alignment/DATASET_FROZEN.json')

def predict(nodes,windows,model,mode):
 out={n['uid']:np.array(n['center_utm']) for n in nodes if n['role']=='TUNE'};info=[]
 for w in windows:
  for k,track in enumerate(w['tracks']):
   rr=[nodes[i] for i in track];dd=collections.defaultdict(list)
   for n in rr:
    if n['actual_date_known']:dd[n['actual_date']].append(n['center_utm'])
   points={d:np.median(v,axis=0) for d,v in dd.items()};result,diag=infer(points,model,mode)
   for n in rr:
    if n['actual_date_known'] and n['actual_date'] in result:
     proposed=result[n['actual_date']];out[n['uid']]=proposed if mode in ['consensus','raw_mean'] else np.array(n['center_utm'])+proposed-points[n['actual_date']]
   info.append(dict(site=w['site'],automatic_track_number=k,nodes=track,**diag))
 return {k:v.tolist() for k,v in out.items()},info

def main():
 nodes=json.loads((O/'OBSERVATIONS_ALL.json').read_text());baseline=json.loads((Q/'BASELINE_PARAMETERS.json').read_text());modelrecord=json.loads((Q/'MODEL_PARAMETERS.json').read_text());factor=modelrecord['model'];mode=modelrecord['selection']['mode'];methods={};raw={n['uid']:n['center_utm'] for n in nodes if n['role']=='TUNE'};methods['RAW']=raw;bindings=[]
 arms=['GEOMETRY','GEOMETRY_COSINE','GEOMETRY_ONLY','GEOMETRY_AND_FROZEN_ROOF64']
 save(P/'PROTOCOL_FROZEN.json',dict(task='ACTUAL_AUTOMATIC_CANDIDATE_ASSOCIATION_THEN_TRAIN_FIT_POSITION',association_arms=arms,position_modes=['DATE_FIELD','TEMPORAL_DATE_FIELD_MEAN','TEMPORAL_RAW_MEAN','SELECTED_FACTOR'],reference_ids_or_coordinates_in_inference=False,association='ALL_MODEL_OBSERVATIONS_INITIALIZED;FROZEN_TRAIN_INNER_THRESHOLDS',reference_supervision='R13_TRAIN_ONLY_FIELD_FACTOR_PARAMETERS_ALREADY_FROZEN',TUNE='REUSED_DEVELOPMENT_NOT_BLIND',GT_reference_geometry='OPEN_ONLY_AFTER_PREDICTIONS_SHA_FROZEN',physical_roof_or_height_accuracy=False,production_changed=False,protected_reads=0))
 for arm in arms:
  path=O/'training'/(arm+'_TUNE_TRACKS.json');tracks=json.loads(path.read_text());windows=[dict(site=w['site'],tracks=w['tracks']) for w in tracks['windows']];assert sorted(i for w in windows for t in w['tracks'] for i in t)==sorted(n['index'] for n in nodes if n['role']=='TUNE');bindings.append(dict(arm=arm,source_track_sha256=sha(path)))
  for name,parameter,m in [('DATE_FIELD',baseline,'corrected'),('TEMPORAL_DATE_FIELD_MEAN',baseline,'consensus'),('TEMPORAL_RAW_MEAN',baseline,'raw_mean'),('SELECTED_FACTOR',factor,mode)]:
   out,diag=predict(nodes,windows,parameter,m);key=arm+'__'+name;methods[key]=out;save(P/(key+'_TRACK_DIAGNOSTICS.json'),diag)
 save(P/'ALL_POSITION_PREDICTIONS.json',dict(methods=methods,GT_reference_group_input=False,GT_coordinates_input=False,all_TUNE_observations=len(raw)));save(P/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json',dict(prediction_sha256=sha(P/'ALL_POSITION_PREDICTIONS.json'),candidate_graph_sha256=sha(O/'CANDIDATE_GRAPH.json'),bindings=bindings,baseline_parameter_sha256=sha(Q/'BASELINE_PARAMETERS.json'),factor_parameter_sha256=sha(Q/'MODEL_PARAMETERS.json'),GT_input=False))
 # First TUNE reference-coordinate/geometry read in this program, after prediction freeze.
 import rasterio
 from rasterio.features import shapes
 from affine import Affine
 from pyproj import Transformer
 from shapely.geometry import shape
 from shapely.ops import unary_union,transform
 from shapely.affinity import translate
 from shapely import make_valid
 rows=[r for r in json.loads(S.read_text())['rows'] if r['role']=='TUNE'];node_by_uid={n['uid']:n for n in nodes};caches={};trans={};records=[];maximum_area_error=0.
 for i,r in enumerate(rows):
  n=node_by_uid[r['id']];path=Path(r['cache_path'])
  if path not in caches:
   assert sha(path)==r['cache_sha256'];caches[path]=np.load(path)['instances']
  mask=(caches[path]==r['query_id']).astype('uint8');parts=[shape(g) for g,v in shapes(mask,mask=mask>0,connectivity=8)];q=unary_union([make_valid(g) if not g.is_valid else g for g in parts]);assert abs(q.area-int(mask.sum()))<1e-5;a=Affine(*r['native_affine'])
  if r['native_crs'] not in trans:trans[r['native_crs']]=Transformer.from_crs(r['native_crs'],32649,always_xy=True)
  tr=trans[r['native_crs']]
  def project(x,y,z=None):
   x=np.asarray(x);y=np.asarray(y);return tr.transform(a.a*x+a.b*y+a.c,a.d*x+a.e*y+a.f)
  q=transform(project,q);target=shape(r['target_geometry_native']);target=make_valid(target) if not target.is_valid else target;target=transform(project,target);truth=np.array(tr.transform(*(a*tuple(r['target_center_native']))));scores={}
  for name,output in methods.items():
   p=np.array(output[r['id']]);delta=p-n['center_utm'];shifted=translate(q,xoff=float(delta[0]),yoff=float(delta[1]));area_error=abs(shifted.area-q.area);assert area_error<max(1e-4,q.area*1e-8);maximum_area_error=max(maximum_area_error,area_error);intersection=shifted.intersection(target).area;union=shifted.area+target.area-intersection;scores[name]=dict(reference_IoU=intersection/max(1e-12,union),reference_center_error_m=float(np.linalg.norm(p-truth)),translation_m=float(np.linalg.norm(delta)),area_error_m2=float(area_error))
  records.append(dict(uid=r['id'],reference_id=r['reference_id'],reference_source=r['reference_source'],scores=scores))
  if (i+1)%50==0:print('AUTOMATIC_FULL_POLYGON_POSITION',i+1,len(rows),flush=True)
 summary={}
 for method in methods:
  summary[method]={}
  for group,rr in [('all',records),('manual',[r for r in records if r['reference_source']=='DESIGNATED_GT2019_ANCHOR']),('public',[r for r in records if r['reference_source']!='DESIGNATED_GT2019_ANCHOR'])]:
   vv=[r['scores'][method] for r in rr];summary[method][group]=dict(n=len(vv),mean_reference_IoU=float(np.mean([v['reference_IoU'] for v in vv])),mean_reference_error_m=float(np.mean([v['reference_center_error_m'] for v in vv])),P95_reference_error_m=float(np.quantile([v['reference_center_error_m'] for v in vv],.95)),max_translation_m=max(v['translation_m'] for v in vv))
 save(P/'FULL_POLYGON_REFERENCE_EVALUATION.json',dict(status='COMPLETE_AUTOMATIC_ASSOCIATION_POSITION_REFERENCE_DEVELOPMENT',summary=summary,records=records,scored_TUNE_nodes=len(rows),all_TUNE_nodes=len(raw),reference_group_input=False,reference_coordinate_input=False,GT_associated_TRAIN_parameters=True,protected_reads=0,source_RGB_reads=0,full_polygons_no_viewport_clipping=True,maximum_translation_area_error_m2=maximum_area_error,formal_independent_roof_accuracy='NOT_ESTABLISHED',whole_city_output=False,production_changed=False));print('AUTOMATIC_POSITION_COMPLETE',summary,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(P/'FAILURE_RECEIPT.json',dict(time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
