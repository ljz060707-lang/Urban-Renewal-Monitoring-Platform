import os,json,math,time,hashlib
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,geopandas as gpd,rasterio,shapely
from rasterio.features import rasterize
from rasterio.windows import Window,from_bounds
from shapely.geometry import box
R=Path(__file__).parent;P=R/'incoming/products';O=R/'exports/POSITION_GOVERNED_ONLY_20261029';O.mkdir(parents=True,exist_ok=True)
statuses=['GT_CONSTRAINED_IMAGE_SUPPORTED','REFERENCE_FIELD_SUPPORTED','LEARNED_SHIFT_CANDIDATE']
d=gpd.read_file(P/'ROOF_GOVERNED.gpkg',where="position_status IN ('GT_CONSTRAINED_IMAGE_SUPPORTED','REFERENCE_FIELD_SUPPORTED','LEARNED_SHIFT_CANDIDATE')",engine='pyogrio')
assert len(d)==158247 and set(d.position_status).issubset(set(statuses));d=d[~d.geometry.is_empty & d.geometry.notna()].copy();geoms=d.geometry.values;bounds=d.total_bounds
gt=gpd.read_file(R.parent/'XIAN_R5_SINGLE_DATE_20261028_01/reference_2019/ROOF_GT2019_REFERENCE.gpkg',columns=['geometry'],engine='pyogrio');gtbounds=gt.total_bounds
with rasterio.open(P/'ROOF_GOVERNED_1M.tif') as src:
 def window(bb):
  w=from_bounds(*bb,src.transform);l=max(0,math.floor(w.col_off));t=max(0,math.floor(w.row_off));r=min(src.width,math.ceil(w.col_off+w.width));b=min(src.height,math.ceil(w.row_off+w.height));return Window(l,t,r-l,b-t)
 win=window(bounds);profile=src.profile.copy();profile.update(width=int(win.width),height=int(win.height),transform=src.window_transform(win),dtype='uint8',nodata=255,compress='DEFLATE',BIGTIFF='YES')
 tree=shapely.STRtree(geoms);positive_pixels=0
 with rasterio.open(O/'POSITION_GOVERNED_MAX_EXTENT.tif','w',**profile) as dst:
  for row in range(0,dst.height,512):
   h=min(512,dst.height-row);a=dst.transform*rasterio.Affine.translation(0,row);bandbox=box(a.c,a.f+a.e*h,a.c+a.a*dst.width,a.f);idx=tree.query(bandbox,predicate='intersects')
   arr=rasterize([(geoms[i],1) for i in idx],out_shape=(h,dst.width),transform=a,dtype='uint8') if len(idx) else np.zeros((h,dst.width),'uint8')
   source=src.read(1,window=Window(win.col_off,win.row_off+row,win.width,h));arr[source==255]=255;positive_pixels+=int((arr==1).sum());dst.write(arr,1,window=Window(0,row,dst.width,h))
   print('raster rows',row+h,'/',dst.height,flush=True)
 gtwin=window(gtbounds);gp=profile.copy();gp.update(width=int(gtwin.width),height=int(gtwin.height),transform=src.window_transform(gtwin))
 with rasterio.open(O/'POSITION_GOVERNED_MAX_EXTENT.tif') as primary,rasterio.open(O/'POSITION_GOVERNED_GT_BOUNDS.tif','w',**gp) as dst:
  for row in range(0,dst.height,512):
   h=min(512,dst.height-row);offset_col=gtwin.col_off-win.col_off;offset_row=gtwin.row_off-win.row_off+row;arr=primary.read(1,window=Window(offset_col,offset_row,gtwin.width,h),boundless=True,fill_value=0);source=src.read(1,window=Window(gtwin.col_off,gtwin.row_off+row,gtwin.width,h));arr[source==255]=255;dst.write(arr,1,window=Window(0,row,dst.width,h))
info={}
for name in ('POSITION_GOVERNED_MAX_EXTENT.tif','POSITION_GOVERNED_GT_BOUNDS.tif'):
 path=O/name
 with rasterio.open(path) as ds:info[name]=dict(width=ds.width,height=ds.height,bounds=list(ds.bounds),bbox_area_km2=ds.width*ds.height/1e6,crs=str(ds.crs),pixel_size_m=list(ds.res),nodata=ds.nodata)
 hh=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):hh.update(b)
 info[name].update(bytes=path.stat().st_size,sha256=hh.hexdigest())
result=dict(status='PASS',objects=len(d),selected_statuses=statuses,unresolved_objects_excluded=50890,vector_bounds=bounds.tolist(),gt_vector_bounds=gtbounds.tolist(),gt_bbox_area_km2=float((gtbounds[2]-gtbounds[0])*(gtbounds[3]-gtbounds[1])/1e6),selected_object_area_sum_m2=float(d.geometry.area.sum()),selected_union_1m_pixel_area_m2=positive_pixels,zero_semantics='outside selected governed roof candidates, not certified background',selection='all governed objects, not largest connected roof; re-rasterized selected vectors to preserve selected/unresolved overlap',outputs=info,created_local=time.strftime('%Y-%m-%d %H:%M:%S'))
(O/'CROP_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
