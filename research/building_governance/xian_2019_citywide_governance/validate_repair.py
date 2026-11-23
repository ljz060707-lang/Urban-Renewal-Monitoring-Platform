import os,json,hashlib
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd,rasterio,shapely
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.ops import transform as tg
from rasterio.windows import Window
from PIL import Image,ImageDraw
R=Path(__file__).parent;O=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01';raw=gpd.read_parquet(R/'payload/raw.parquet');d=pd.read_csv(R/'incoming/products/OBJECT_AREA.csv');issue=pd.read_csv(O/'UNRESOLVED_OBJECT_DIAGNOSIS.csv');c=raw.geometry.centroid;xy=np.column_stack([c.x,c.y]);anchor=d.position_status.to_numpy()=='GT_CONSTRAINED_IMAGE_SUPPORTED';bucket=np.array([int(hashlib.sha256(f'{int(x//2000)},{int(y//2000)}'.encode()).hexdigest()[:8],16)%10 for x,y in xy]);train=anchor&(bucket<6);large=((raw.native_xmax-raw.native_xmin)>512)|((raw.native_ymax-raw.native_ymin)>512);query=anchor&large&(bucket>=6);assert query.sum()>10
tree=cKDTree(xy[train]);distance,index=tree.query(xy[query],k=24);valid=distance<2000;weights=valid/(distance+50)**2;targets=d[['dx_m','dy_m']].to_numpy();pred=(targets[train][index]*weights[:,:,None]).sum(1)/np.maximum(weights.sum(1),1e-12)[:,None];enough=valid.sum(1)>=4;pred[~enough]=0;before=np.linalg.norm(targets[query],axis=1);after=np.linalg.norm(pred-targets[query],axis=1)
report=dict(role='spatial diagnostic using existing reference associations, not blind independent building identity truth',query_large_objects=int(query.sum()),query_with_field_support=int(enough.sum()),mean_before_m=float(before.mean()),mean_after_m=float(after.mean()),median_before_m=float(np.median(before)),median_after_m=float(np.median(after)),p95_before_m=float(np.quantile(before,.95)),p95_after_m=float(np.quantile(after,.95)),rule='same fixed 24 neighbors,2km radius,minimum4; spatial 2km TRAIN blocks only for anchor fitting',admitted=bool(after.mean()<before.mean()*.98 and np.median(after)<np.median(before) and np.quantile(after,.95)<=np.quantile(before,.95)*1.1))
(O/'LARGE_OBJECT_FIELD_VALIDATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
cases=[]
for key,n in [('OBJECT_LARGER_THAN_512_CONTEXT',4),('CENTROID_CROP_PLACEMENT_EXCLUDES_COMPLETE_OBJECT',2),('BLOCK_CONTEXT_PARTIAL_NOT_MOSAIC_EDGE',2),('OUTSIDE_VALIDATED_REFERENCE_SUPPORT_3KM',2)]:
 cases.extend(issue[issue.issue==key].nlargest(n,'area_raw_m2').to_dict('records'))
canvas=Image.new('RGB',(1024,570*((len(cases)+1)//2)),'white');draw=ImageDraw.Draw(canvas)
with rasterio.open(r'F:\xian\亚米市区底图\xa190101_combined\xa190101_zoom_20.tif') as src:
 tr=Transformer.from_crs(raw.crs,src.crs,always_xy=True);inv=~src.transform
 def native(x,y,z=None):
  lo,la=tr.transform(x,y);return inv.a*np.asarray(lo)+inv.c,inv.e*np.asarray(la)+inv.f
 for k,z in enumerate(cases):
  g=tg(native,raw.geometry.iloc[int(z['instance_id'])-1]);l,t,r,b=g.bounds;side=max(1024,max(r-l,b-t)*1.2);cx=(l+r)/2;cy=(t+b)/2;x0=max(0,min(src.width-side,cx-side/2));y0=max(0,min(src.height-side,cy-side/2));rgb=src.read(window=Window(x0,y0,side,side),out_shape=(3,512,512),resampling=rasterio.enums.Resampling.bilinear).transpose(1,2,0);im=Image.fromarray(rgb);q=ImageDraw.Draw(im)
  for poly in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
   if poly.geom_type=='Polygon':q.line([((x-x0)/side*512,(y-y0)/side*512) for x,y in poly.simplify(1).exterior.coords],fill='cyan',width=2)
  x=(k%2)*512;y=(k//2)*570;canvas.paste(im,(x,y));draw.text((x+4,y+516),f"CASE{k} ID{int(z['instance_id'])} area={z['area_raw_m2']:.0f}m2",fill='black');draw.text((x+4,y+535),z['issue'],fill='black');z['case_index']=k
canvas.save(O/'ISSUE_CASES.jpg');(O/'ISSUE_CASES.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
