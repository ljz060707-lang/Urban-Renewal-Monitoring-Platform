import os,json,math,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd,rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
R=Path(__file__).parent;O=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01';O.mkdir(exist_ok=True)
raw=gpd.read_parquet(R/'payload/raw.parquet');result=pd.read_csv(R/'incoming/products/OBJECT_AREA.csv');assert np.array_equal(raw.instance_id,result.instance_id)
cent=raw.geometry.centroid;xy=np.column_stack([cent.x,cent.y]);anchor=result.position_status=='GT_CONSTRAINED_IMAGE_SUPPORTED';tree=cKDTree(xy[anchor]);dist,idx=tree.query(xy,k=24);within=(dist<2000).sum(1)
gt=gpd.read_file(R.parent/'XIAN_R5_SINGLE_DATE_20261028_01/reference_2019/ROOF_GT2019_REFERENCE.gpkg',engine='pyogrio');gc=gt.geometry.centroid;gtd=cKDTree(np.column_stack([gc.x,gc.y])).query(xy,k=1)[0]
with rasterio.open(r'F:\xian\亚米市区底图\xa190101_combined\xa190101_zoom_20.tif') as src:inv=~src.transform;sw,sh=src.width,src.height;srccrs=src.crs
tr=Transformer.from_crs(raw.crs,srccrs,always_xy=True);lo,la=tr.transform(xy[:,0],xy[:,1]);cx=inv.a*lo+inv.c;cy=inv.e*la+inv.f
ox=np.clip(np.floor(cx-256),0,sw-512);oy=np.clip(np.floor(cy-256),0,sh-512)
parts=np.array([list(map(int,x[1:].split('_'))) for x in raw.block_id]);rc=np.maximum(0,parts[:,1]*16384-768);rr=np.maximum(0,parts[:,0]*16384-768)
l=raw.native_xmin.to_numpy()+rc;t=raw.native_ymin.to_numpy()+rr;r=raw.native_xmax.to_numpy()+rc;b=raw.native_ymax.to_numpy()+rr
partial=raw.source_edge_partial_risk.to_numpy();mosaic_edge=(l<=2)|(t<=2)|(r>=sw-2)|(b>=sh-2);too_large=((r-l)>512)|((b-t)>512);fits=(l>=ox)&(t>=oy)&(r<=ox+512)&(b<=oy+512);complete=~partial&fits
issue=np.full(len(raw),'POSITION_ALREADY_PROCESSED',dtype=object);unknown=result.position_status=='MODEL_ONLY_UNRESOLVED'
issue[unknown & ~complete & ~partial & ~too_large]='CENTROID_CROP_PLACEMENT_EXCLUDES_COMPLETE_OBJECT'
issue[unknown & too_large & ~partial]='OBJECT_LARGER_THAN_512_CONTEXT'
issue[unknown & partial & ~mosaic_edge]='BLOCK_CONTEXT_PARTIAL_NOT_MOSAIC_EDGE'
issue[unknown & partial & mosaic_edge]='SOURCE_MOSAIC_EDGE_PARTIAL'
issue[unknown & complete & (dist[:,0]>=3000)]='OUTSIDE_VALIDATED_REFERENCE_SUPPORT_3KM'
issue[unknown & complete & (dist[:,0]<3000)]='UNEXPECTED_ROUTING_GAP_NEEDS_CHECK'
table=pd.DataFrame(dict(instance_id=raw.instance_id,observation_id=raw.observation_id,issue=issue,area_raw_m2=raw.geometry.area,nearest_trusted_anchor_m=dist[:,0],trusted_anchors_within2km=within,nearest_any_reference_m=gtd,native_width=r-l,native_height=b-t,source_partial=partial,mosaic_edge=mosaic_edge,complete_512=complete,field_eligible_without_neural_context=(within>=4)&~partial,centroid_x=xy[:,0],centroid_y=xy[:,1],mean_probability=raw.mean_probability))
table.loc[unknown].to_csv(O/'UNRESOLVED_OBJECT_DIAGNOSIS.csv',index=False)
summary=table.loc[unknown].groupby('issue').agg(objects=('instance_id','size'),object_area_sum_m2=('area_raw_m2','sum'),field_only_repair_candidates=('field_eligible_without_neural_context','sum'),near_existing_GT_within30m=('nearest_any_reference_m',lambda x:int((x<30).sum())))
summary.to_csv(O/'UNRESOLVED_CAUSE_SUMMARY.csv');payload=dict(objects_total=len(raw),unresolved=int(unknown.sum()),categories=summary.reset_index().to_dict('records'),field_routing_repair_candidates=int((unknown&table.field_eligible_without_neural_context).sum()),source_partial_flag_note='near block read-window boundary does not by itself prove missing source imagery; mosaic-edge tested separately',reference_gap_note='No trusted anchor nearby is not equivalent to no reference polygon or no real building',created_local=time.strftime('%Y-%m-%d %H:%M:%S'))
(O/'DIAGNOSIS_RECEIPT.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(payload,ensure_ascii=False,indent=2),flush=True)
