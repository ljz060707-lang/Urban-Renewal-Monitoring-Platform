import os,json,hashlib,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,rasterio
from rasterio.windows import Window
from rasterio.enums import ColorInterp,Resampling
R=Path(__file__).parent/'exports/POSITION_GOVERNED_ONLY_20261029'
source=R/'POSITION_GOVERNED_MAX_EXTENT.tif';target=R/'POSITION_GOVERNED_MAX_EXTENT_CYAN_ALPHA50.tif'
count=0
with rasterio.open(source) as src:
 profile=src.profile.copy();profile.update(count=4,nodata=None,photometric='RGB',interleave='pixel',BIGTIFF='YES',ALPHA='NON-PREMULTIPLIED',predictor=2)
 with rasterio.open(target,'w',**profile) as dst:
  dst.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
  dst.update_tags(SOURCE_RASTER=source.name,VISUALIZATION_ONLY='YES',ROOF_RGB='0,220,255',ROOF_ALPHA='128/255',BACKGROUND_ALPHA='0',GEOMETRY='Same pixels and reference grid as source; no resampling')
  for y in range(0,src.height,512):
   h=min(512,src.height-y);win=Window(0,y,src.width,h);mask=src.read(1,window=win)==1;rgba=np.zeros((4,h,src.width),'uint8');rgba[1,mask]=220;rgba[2,mask]=255;rgba[3,mask]=128;dst.write(rgba,window=win);count+=int(mask.sum())
  dst.build_overviews([2,4,8,16,32],Resampling.nearest);dst.update_tags(ns='rio_overview',resampling='nearest')
verified=0
with rasterio.open(source) as src,rasterio.open(target) as dst:
 assert dst.count==4 and src.transform==dst.transform and src.crs==dst.crs and src.shape==dst.shape
 assert dst.colorinterp[-1]==ColorInterp.alpha
 for y in range(0,src.height,512):
  win=Window(0,y,src.width,min(512,src.height-y));expected=src.read(1,window=win)==1;a=dst.read(4,window=win);assert np.array_equal(a,expected.astype('uint8')*128);verified+=int((a==128).sum())
assert count==verified
h=hashlib.sha256()
with target.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
(R/'TRANSPARENT_OVERLAY_RECEIPT.json').write_text(json.dumps(dict(status='PASS',output=target.name,roof_color='cyan RGB(0,220,255)',roof_alpha=128,opacity=128/255,background_alpha=0,roof_pixels=verified,same_grid_and_geometry=True,sha256=h.hexdigest(),created_local=time.strftime('%Y-%m-%d %H:%M:%S')),ensure_ascii=False,indent=2),encoding='utf-8')
print(str(target),target.stat().st_size,'bytes; roof pixels',verified,flush=True)
