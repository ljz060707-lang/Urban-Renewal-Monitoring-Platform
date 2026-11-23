import os,json,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from rasterio.windows import Window
from rasterio.enums import ColorInterp,Resampling
R=Path(__file__).parent;P=R/'incoming/products';O=R/'exports/RECALL_REVIEW_20261029';O.mkdir(exist_ok=True)
d=pd.read_csv(P/'OBJECT_AREA.csv');d['area_retained_ratio']=d.area_governed_m2/d.area_raw_m2;d[d.area_retained_ratio<.8].to_csv(O/'AREA_REDUCTION_REVIEW.csv',index=False)
counts={'position_governed_pixels':0,'unresolved_candidate_pixels':0}
target=O/'ALL_CANDIDATES_STATUS_ALPHA50.tif'
with rasterio.open(P/'ROOF_GOVERNED_1M.tif') as roof,rasterio.open(P/'POSITION_STATUS_1M.tif') as pos:
 profile=roof.profile.copy();profile.update(count=4,nodata=None,photometric='RGB',interleave='pixel',BIGTIFF='YES',ALPHA='NON-PREMULTIPLIED',predictor=2)
 with rasterio.open(target,'w',**profile) as dst:
  dst.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha);dst.update_tags(VISUALIZATION_ONLY='YES',CYAN='position processing applied; not per-object accuracy certification',ORANGE='position unresolved; retained candidate',ALPHA='128/255 on all candidates, zero elsewhere')
  for y in range(0,roof.height,512):
   h=min(512,roof.height-y);w=Window(0,y,roof.width,h);a=roof.read(1,window=w);s=pos.read(1,window=w);yes=a==1;unknown=yes&(s==4);governed=yes&~unknown;rgba=np.zeros((4,h,roof.width),'uint8');rgba[1,governed]=220;rgba[2,governed]=255;rgba[0,unknown]=255;rgba[1,unknown]=165;rgba[3,yes]=128;dst.write(rgba,window=w);counts['position_governed_pixels']+=int(governed.sum());counts['unresolved_candidate_pixels']+=int(unknown.sum())
  dst.build_overviews([2,4,8,16,32],Resampling.nearest)
with rasterio.open(P/'ROOF_GOVERNED_1M.tif') as src,rasterio.open(target) as dst:
 assert src.shape==dst.shape and src.transform==dst.transform and src.crs==dst.crs
 for y in range(0,src.height,512):
  win=Window(0,y,src.width,min(512,src.height-y));assert np.array_equal(src.read(1,window=win)==1,dst.read(4,window=win)>0)
receipt=dict(status='PASS',geometry='same as full governed candidate product; visualization restores previously excluded unresolved candidates but does not repair contour errors',objects=209137,position_governed_objects=158247,unresolved_retained_objects=50890,area_reduction_review_objects=int((d.area_retained_ratio<.8).sum()),legend=dict(cyan='position governance applied',orange='position unresolved, candidate retained'),alpha=128,all_roof_pixels_preserved=True,pixel_counts=counts,created_local=time.strftime('%Y-%m-%d %H:%M:%S'))
(O/'RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False),flush=True)
