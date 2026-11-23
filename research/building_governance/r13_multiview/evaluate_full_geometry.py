"""Translate complete predicted polygons; no crop loss and no area inflation."""
from pathlib import Path
import json,hashlib,time,collections
import numpy as np,rasterio
from rasterio.features import shapes
from affine import Affine
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union,transform
from shapely.affinity import translate
from shapely import make_valid
R=Path('/root/xian20261111_r13');O=R/'output';P=O/'TUNE_QUERY_PREDICTIONS.json';D=Path('/root/xian20261111_r8/output/rgb_alignment/DATASET_FROZEN.json');meta={r['id']:r for r in json.loads(D.read_text())['rows'] if r['role']=='TUNE'};predictions=json.loads(P.read_text())['rows'];assert len(predictions)==399 and set(meta)=={r['id'] for r in predictions};caches={};trans={};records=[];max_area_error=0.
for i,row in enumerate(predictions):
 m=meta[row['id']];path=Path(m['cache_path'])
 if path not in caches:
  assert hashlib.sha256(path.read_bytes()).hexdigest()==m['cache_sha256'];z=np.load(path);caches[path]=z['instances']
 labels=caches[path];mask=(labels==m['query_id']).astype('uint8');parts=[shape(g) for g,v in shapes(mask,mask=mask>0,connectivity=8)];q=unary_union([make_valid(g) if not g.is_valid else g for g in parts]);assert abs(q.area-int(mask.sum()))<1e-5;a=Affine(*m['native_affine'])
 if m['native_crs'] not in trans:trans[m['native_crs']]=Transformer.from_crs(m['native_crs'],32649,always_xy=True)
 t=trans[m['native_crs']]
 def project(x,y,z=None):
  x=np.asarray(x);y=np.asarray(y);return t.transform(a.a*x+a.b*y+a.c,a.d*x+a.e*y+a.f)
 q=transform(project,q);target=shape(m['target_geometry_native']);target=make_valid(target) if not target.is_valid else target;target=transform(project,target);p=np.array(row['p']);scores={}
 for name,v in row['predictions'].items():
  delta=np.asarray(v)-p;shifted=translate(q,xoff=float(delta[0]),yoff=float(delta[1]));area_error=abs(shifted.area-q.area);max_area_error=max(max_area_error,area_error);assert area_error<=max(1e-4,q.area*1e-8);inter=shifted.intersection(target).area;union=shifted.area+target.area-inter;scores[name]=dict(reference_IoU=inter/max(1e-12,union),error_m=float(np.linalg.norm(np.asarray(v)-row['g'])),projected_area_m2=float(shifted.area),area_error_m2=area_error)
 records.append(dict(id=row['id'],reference_id=row['reference_id'],reference_source=row['reference_source'],scores=scores))
 if (i+1)%50==0:print('CONDITIONAL_COMPLETE_POLYGON_EVALUATION',i+1,len(predictions),flush=True)
methods=list(records[0]['scores']);summary={}
for method in methods:
 summary[method]={}
 for group,rr in [('all',records),('manual',[r for r in records if r['reference_source']=='DESIGNATED_GT2019_ANCHOR']),('public',[r for r in records if r['reference_source']!='DESIGNATED_GT2019_ANCHOR'])]:
  s=[r['scores'][method] for r in rr];summary[method][group]=dict(n=len(s),mean_reference_IoU=float(np.mean([x['reference_IoU'] for x in s])),median_reference_IoU=float(np.median([x['reference_IoU'] for x in s])),mean_error_m=float(np.mean([x['error_m'] for x in s])),P95_error_m=float(np.quantile([x['error_m'] for x in s],.95)))
refs=sorted({r['reference_id'] for r in records});by={k:[r for r in records if r['reference_id']==k] for k in refs};rng=np.random.default_rng(20261111);boot=[]
for _ in range(1000):
 selected=[r for k in rng.choice(refs,size=len(refs),replace=True) for r in by[k]];boot.append(np.mean([r['scores']['SELECTED_FACTOR']['reference_IoU']-r['scores']['RAW']['reference_IoU'] for r in selected]))
rec=dict(status='COMPLETE_CONDITIONAL_FULL_POLYGON_REFERENCE_EVALUATION',records=records,summary=summary,query_count=399,reference_tracks=len(refs),reference_cluster_bootstrap_mean_IoU_gain_95CI=np.quantile(boot,[.025,.975]).tolist(),max_translation_area_error_m2=max_area_error,full_polygons_no_viewport_clipping=True,physical_area_or_physical_height_established=False,correspondence='PROVIDED_GEOMETRICALLY_ADMITTED_MAP_REFERENCE_GROUPS',not_automatic_tracking_or_native_roof_accuracy=True,prediction_json_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),map_reference_rows=399,model_query_mask_caches_read=len(caches),source_RGB_reads=0,protected_reads=0,production_changed=False);(O/'FULL_POLYGON_EVALUATION.json').write_text(json.dumps(rec,indent=2));print(json.dumps(dict(summary=summary,CI=rec['reference_cluster_bootstrap_mean_IoU_gain_95CI'],max_area_error=max_area_error),indent=2))
