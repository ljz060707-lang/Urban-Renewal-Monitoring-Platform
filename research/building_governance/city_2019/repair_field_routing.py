import os,json,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd,shapely,rasterio
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.windows import Window
R=Path(__file__).parent;B=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01';O=B/'REPAIR_V2_1495';O.mkdir(exist_ok=True);raw=gpd.read_parquet(R/'payload/raw.parquet');d=gpd.read_file(R/'incoming/products/ROOF_GOVERNED.gpkg',engine='pyogrio').sort_values('instance_id').reset_index(drop=True);audit=pd.read_csv(B/'UNRESOLVED_OBJECT_DIAGNOSIS.csv');c=raw.geometry.centroid;xy=np.column_stack([c.x,c.y]);anchor=d.position_status=='GT_CONSTRAINED_IMAGE_SUPPORTED';tree=cKDTree(xy[anchor]);target=d.loc[anchor,['dx_m','dy_m']].to_numpy();policy=json.loads((R/'incoming/reference_field_parameters.json').read_text());assert policy['admitted']
eligible=audit.field_eligible_without_neural_context&audit.issue.isin(['CENTROID_CROP_PLACEMENT_EXCLUDES_COMPLETE_OBJECT','UNEXPECTED_ROUTING_GAP_NEEDS_CHECK']);rows=audit.loc[eligible].copy();buffer=pd.read_csv(B/'EDGE_BUFFER_FIELD_CANDIDATES.csv');buffer['issue']='EDGE_BUFFER_RISK_NOT_ACTUAL_TRUNCATION';rows=pd.concat([rows,buffer],ignore_index=True).drop_duplicates('instance_id');assert len(rows)==1495;indices=rows.instance_id.to_numpy()-1;distance,idx=tree.query(xy[indices],k=24);valid=distance<2000;w=valid/(distance+50)**2;shift=(target[idx]*w[:,:,None]).sum(1)/w.sum(1)[:,None];assert (valid.sum(1)>=4).all()
d['repair_reason']='';d['repair_position_certification']='NOT_INDEPENDENTLY_CERTIFIED';d['previous_position_status']=d.position_status;d['previous_observation_id']=d.observation_id
for j,i in enumerate(indices):
 g=d.geometry.iloc[i];new=translate(g,xoff=float(shift[j,0]),yoff=float(shift[j,1]));assert abs(new.area-g.area)<max(.001,g.area*1e-7)
 d.at[i,'geometry']=new;d.at[i,'dx_m']=float(shift[j,0]);d.at[i,'dy_m']=float(shift[j,1]);d.at[i,'position_status']='REFERENCE_FIELD_ROUTING_REPAIRED';d.at[i,'method']=str(d.method.iloc[i])+';FIELD_ROUTING_INDEPENDENT_OF_NEURAL_CROP';d.at[i,'repair_reason']=rows.iloc[j].issue
rows['dx_m']=shift[:,0];rows['dy_m']=shift[:,1];rows['neighbors_within2km']=valid.sum(1);rows.to_csv(O/'FIELD_ROUTING_REPAIRED_OBJECTS.csv',index=False)
# Audit strong duplicate fragments across block reads. Only record candidates;
# do not infer physical building identity from overlap or merge them silently.
rawtree=shapely.STRtree(raw.geometry.values);partial=np.where((raw.source_edge_partial_risk)&(d.previous_position_status=='MODEL_ONLY_UNRESOLVED'))[0];pairs=rawtree.query(raw.geometry.values[partial],predicate='intersects');ii=partial[pairs[0]];jj=pairs[1];keep=(ii!=jj)&(raw.block_id.to_numpy()[ii]!=raw.block_id.to_numpy()[jj]);ii,jj=ii[keep],jj[keep]
area=raw.geometry.area.to_numpy();inter=shapely.area(shapely.intersection(raw.geometry.values[ii],raw.geometry.values[jj]));ratio=inter/np.maximum(np.minimum(area[ii],area[jj]),1e-9);selected=ratio>=.7;seams=pd.DataFrame(dict(instance_id=ii[selected]+1,overlapping_instance_id=jj[selected]+1,intersection_over_smaller=ratio[selected],intersection_m2=inter[selected]));seams=seams.assign(pair=[tuple(sorted(x)) for x in zip(seams.instance_id,seams.overlapping_instance_id)]).drop_duplicates('pair').drop(columns='pair');seams.to_csv(O/'BLOCK_FRAGMENT_OVERLAP_REVIEW.csv',index=False)
d['area_repaired_m2']=d.geometry.area;d.to_file(O/'ROOF_POSITION_REPAIRED.gpkg',driver='GPKG',engine='pyogrio');d.drop(columns='geometry').to_csv(O/'OBJECT_AREA_REPAIRED.csv',index=False)
geoms=d.geometry.values;gtree=shapely.STRtree(geoms);code=np.array([1 if s=='GT_CONSTRAINED_IMAGE_SUPPORTED' else 2 if s in ('REFERENCE_FIELD_SUPPORTED','REFERENCE_FIELD_ROUTING_REPAIRED') else 3 if s=='LEARNED_SHIFT_CANDIDATE' else 4 for s in d.position_status],dtype='uint8')
with rasterio.open(R/'incoming/products/ROOF_GOVERNED_1M.tif') as src:
 profile=src.profile.copy()
 with rasterio.open(O/'ROOF_POSITION_REPAIRED_ALL_CANDIDATES_1M.tif','w',**profile) as dst,rasterio.open(O/'POSITION_STATUS_REPAIRED_1M.tif','w',**profile) as qs:
  for y in range(0,src.height,512):
   h=min(512,src.height-y);a=src.transform*rasterio.Affine.translation(0,y);bounds=box(a.c,a.f+a.e*h,a.c+a.a*src.width,a.f);ids=gtree.query(bounds,predicate='intersects');ids=ids[~shapely.is_empty(geoms[ids])];win=Window(0,y,src.width,h)
   arr=rasterize([(geoms[i],1) for i in ids],out_shape=(h,src.width),transform=a,dtype='uint8') if len(ids) else np.zeros((h,src.width),'uint8');q=rasterize([(geoms[i],int(code[i])) for i in ids],out_shape=(h,src.width),transform=a,dtype='uint8') if len(ids) else np.zeros((h,src.width),'uint8');missing=src.read(1,window=win)==255;arr[missing]=255;q[missing]=255;dst.write(arr,1,window=win);qs.write(q,1,window=win)
   print('write rows',y+h,'/',src.height,flush=True)
with rasterio.open(O/'ROOF_POSITION_REPAIRED_ALL_CANDIDATES_1M.tif') as check:assert check.shape==(30860,38850) and str(check.crs)=='EPSG:32649'
receipt=dict(status='PARTIAL_REPAIR_WITH_EXPLICIT_REMAINING_BLOCKERS',objects_total=len(d),additional_position_processed=len(rows),position_processed_total=int((code!=4).sum()),position_unresolved_remaining=int((code==4).sum()),strong_block_overlap_pairs=len(seams),all_candidates_retained=True,large_object_field_not_admitted=True,contour_and_facade_errors_not_globally_fixed=True,object_area_sum_m2=float(d.geometry.area.sum()),area_units='m2; object sum includes overlap, no new union-area claim',created_local=time.strftime('%Y-%m-%d %H:%M:%S'))
(O/'REPAIR_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False),flush=True)
