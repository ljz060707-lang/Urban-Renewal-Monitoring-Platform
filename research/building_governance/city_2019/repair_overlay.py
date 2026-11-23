import os,json,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd,shapely,rasterio
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.enums import ColorInterp,Resampling
R=Path(__file__).parent;O=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01/REPAIR_V2_1495';ids=pd.read_csv(O/'FIELD_ROUTING_REPAIRED_OBJECTS.csv').instance_id.tolist();where='instance_id IN ('+','.join(map(str,ids))+')';d=gpd.read_file(O/'ROOF_POSITION_REPAIRED.gpkg',where=where,engine='pyogrio');assert len(d)==1495;geoms=d.geometry.values;tree=shapely.STRtree(geoms);count=0
with rasterio.open(O/'ROOF_POSITION_REPAIRED_ALL_CANDIDATES_1M.tif') as src,rasterio.open(O/'POSITION_STATUS_REPAIRED_1M.tif') as qs:
 profile=src.profile.copy();profile.update(count=4,nodata=None,photometric='RGB',interleave='pixel',BIGTIFF='YES',ALPHA='NON-PREMULTIPLIED',predictor=2)
 with rasterio.open(O/'ROOF_POSITION_REPAIRED_ALPHA50.tif','w',**profile) as dst:
  dst.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
  for y in range(0,src.height,512):
   h=min(512,src.height-y);win=Window(0,y,src.width,h);a=src.read(1,window=win);q=qs.read(1,window=win);yes=a==1;unknown=yes&(q==4);known=yes&~unknown;aff=src.transform*rasterio.Affine.translation(0,y);bb=box(aff.c,aff.f+aff.e*h,aff.c+aff.a*src.width,aff.f);ix=tree.query(bb,predicate='intersects');repair=rasterize([(geoms[i],1) for i in ix],out_shape=(h,src.width),transform=aff,dtype='uint8')>0 if len(ix) else np.zeros((h,src.width),'bool');repair&=yes
   rgba=np.zeros((4,h,src.width),'uint8');rgba[1,known]=220;rgba[2,known]=255;rgba[0,unknown]=255;rgba[1,unknown]=165;rgba[0,repair]=0;rgba[1,repair]=255;rgba[2,repair]=128;rgba[3,yes]=128;dst.write(rgba,window=win);count+=int(yes.sum())
  dst.build_overviews([2,4,8,16,32],Resampling.nearest)
with rasterio.open(O/'ROOF_POSITION_REPAIRED_ALL_CANDIDATES_1M.tif') as src,rasterio.open(O/'ROOF_POSITION_REPAIRED_ALPHA50.tif') as out:
 assert src.transform==out.transform and src.crs==out.crs and src.shape==out.shape
 for y in range(0,src.height,512):
  win=Window(0,y,src.width,min(512,src.height-y));assert np.array_equal(src.read(1,window=win)==1,out.read(4,window=win)>0)
(O/'OVERLAY_RECEIPT.json').write_text(json.dumps(dict(status='PASS',roof_pixels=count,raster_union_area_m2_at_1m_resolution=count,area_semantics='pixel count, not exact vector union area',colors={'cyan':'prior position processing','green':'1495 additionally position-processed objects','orange':'position unresolved retained candidates'},roof_alpha=128,background_alpha=0),ensure_ascii=False,indent=2),encoding='utf-8');print('overlay PASS',count,flush=True)
