import os,ast,json,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import torch,numpy as np,geopandas as gpd,rasterio,shapely
import torch.nn.functional as F
from rasterio.features import shapes
from rasterio.windows import Window,from_bounds
from shapely.geometry import shape
from shapely.ops import transform as tg
from shapely.affinity import translate
from pyproj import Transformer
from PIL import Image,ImageDraw
R=Path(__file__).parent;O=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01/PILOT_SOFT_CONTOURS';O.mkdir(parents=True,exist_ok=True)
source=ast.parse((R/'incoming/final_training_source/ml.py').read_text(encoding='utf-8'));scope={'torch':torch,'nn':torch.nn,'F':F,'__name__':'local_frozen_net'};exec(compile(ast.Module(body=[n for n in source.body if isinstance(n,ast.ClassDef) and n.name in ('Conv','Net')],type_ignores=[]),'frozen_model','exec'),scope)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model=scope['Net'](3,3);model.load_state_dict(torch.load(R/'incoming/facade_best.pt',map_location='cpu')['model'],strict=True);model.cuda().eval()
raw=gpd.read_file(R.parent/'XIAN_R5_2019_GOVERNANCE_20261028_01/PRIORITY_RESIDENTIAL_ROI/ROOF_RAW.gpkg',engine='pyogrio');ids=list(raw.observation_id);where='observation_id IN ('+','.join("'"+x+"'" for x in ids)+')';current=gpd.read_file(R/'incoming/products/ROOF_GOVERNED.gpkg',where=where,engine='pyogrio').set_index('observation_id').loc[ids].reset_index();assert len(current)==67
to_geo=Transformer.from_crs(raw.crs,'EPSG:4326',always_xy=True);to_metric=Transformer.from_crs('EPSG:4326',raw.crs,always_xy=True);proposals=[];stats=[]
with rasterio.open(r'F:\xian\亚米市区底图\xa190101_combined\xa190101_zoom_20.tif') as src,rasterio.open(R.parent/'XIAN_R5_2019_GOVERNANCE_20261028_01/FACADE_NATIVE_RISK.tif') as risk,torch.inference_mode():
 inv=~src.transform
 def native(x,y,z=None):
  lo,la=to_geo.transform(x,y);return inv.a*np.asarray(lo)+inv.c,inv.e*np.asarray(la)+inv.f
 for i,row in raw.iterrows():
  g=row.geometry;n=tg(native,g);cx,cy=n.centroid.coords[0];ox=int(np.clip(np.floor(cx-256),0,src.width-512));oy=int(np.clip(np.floor(cy-256),0,src.height-512));win=Window(ox,oy,512,512);rgb=src.read(window=win);z,_=model(torch.from_numpy(rgb.copy()).float()[None].cuda()/255);p=z.softmax(1)[0].cpu().numpy();old=risk.read(1,window=win)==1
  def metric_polygons(mask):
   return shapely.union_all([tg(to_metric.transform,shape(s)) for s,v in shapes(mask.astype('uint8'),mask=mask,transform=src.window_transform(win)) if v==1])
  roof=g.intersection(metric_polygons((p[1]>.9)&~old));cut=g.intersection(metric_polygons((p[2]>.941)&(p[1]<.2)&old));r=current.iloc[i];dx=float(r.dx_m);dy=float(r.dy_m);add=translate(roof,xoff=dx,yoff=dy);subtract=translate(cut,xoff=dx,yoff=dy);base=r.geometry
  # Reference positive geometry is protected, never treated as facade-negative GT.
  if r.position_status=='GT_CONSTRAINED_IMAGE_SUPPORTED':subtract=subtract.difference(base)
  proposal=base.union(add).difference(subtract)
  if proposal.is_empty:proposal=base
  proposals.append(proposal);stats.append(dict(observation_id=row.observation_id,base_area_m2=base.area,pilot_area_m2=proposal.area,roof_added_m2=proposal.difference(base).area,facade_removed_m2=base.difference(proposal).area))
 out=current.copy();out.geometry=proposals;out['pilot_role']='UNVALIDATED_SOFT_CONTOUR_RECALL_PRESERVATION';out.to_file(O/'SOFT_CONTOUR_REVIEW.gpkg',driver='GPKG',engine='pyogrio')
 import pandas as pd
 pd.DataFrame(stats).to_csv(O/'PILOT_AREA_COMPARISON.csv',index=False)
 bounds=(319180,3788800,319420,3789020);lo,la=to_geo.transform([bounds[0],bounds[2]],[bounds[1],bounds[3]]);win=from_bounds(min(lo),min(la),max(lo),max(la),src.transform);photo=src.read(window=win,out_shape=(3,770,840),resampling=rasterio.enums.Resampling.bilinear).transpose(1,2,0);canvas=Image.new('RGB',(1680,800),'white')
 def pix(x,y):
  lo1,la1=to_geo.transform(x,y);return (lo1-min(lo))/(max(lo)-min(lo))*840,(max(la)-la1)/(max(la)-min(la))*770
 for side,gs,color,label in ((0,current.geometry,'cyan','CURRENT'),(1,proposals,'lime','PILOT: preserve independent high-confidence roof evidence')):
  im=Image.fromarray(photo);draw=ImageDraw.Draw(im)
  for g in gs:
   for poly in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
    if poly.geom_type=='Polygon':draw.line([pix(x,y) for x,y in poly.exterior.coords],fill=color,width=2)
  canvas.paste(im,(side*840,30));ImageDraw.Draw(canvas).text((side*840+5,8),label,fill='black')
 canvas.save(O/'SOFT_CONTOUR_COMPARISON.png')
summary=dict(status='PILOT_FOR_VISUAL_REVIEW_NOT_GLOBAL_ADMISSION',objects=len(stats),objects_with_roof_additions=sum(s['roof_added_m2']>0 for s in stats),roof_added_m2=sum(s['roof_added_m2'] for s in stats),facade_removed_m2=sum(s['facade_removed_m2'] for s in stats),thresholds=dict(new_roof=.9,new_facade=.941,legacy_facade_gate=True),limitations='roof probability is not calibrated on independent Xian visible-roof GT; no target-domain accuracy or physical rectification claim',created_local=time.strftime('%Y-%m-%d %H:%M:%S'))
(O/'PILOT_RECEIPT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)
