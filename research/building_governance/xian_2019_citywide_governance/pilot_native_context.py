import os,sys,json,time,math
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import geopandas as gpd,numpy as np,rasterio,shapely
from rasterio.features import rasterize,shapes
from rasterio.windows import Window
from pyproj import Transformer
from shapely.ops import transform as tg
from shapely.geometry import shape
from scipy import ndimage
from PIL import Image,ImageDraw
R=Path(__file__).parent;O=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01/PILOT_NATIVE_CONTEXT';O.mkdir(parents=True,exist_ok=True);ids=[34481,37724];raw=gpd.read_parquet(R/'payload/raw.parquet',filters=[('instance_id','in',ids)]);assert len(raw)==2
sys.path.insert(0,str(R.parent/'XIAN_R5_SINGLE_DATE_20261028_01/CODE_SNAPSHOT'));from xian_single_date import b0,common
common.BATCH=4
with rasterio.open(r'F:\xian\亚米市区底图\xa190101_combined\xa190101_zoom_20.tif') as src:
 tr=Transformer.from_crs(raw.crs,src.crs,always_xy=True);back=Transformer.from_crs(src.crs,raw.crs,always_xy=True);inv=~src.transform
 def native(x,y,z=None):
  lo,la=tr.transform(x,y);return inv.a*np.asarray(lo)+inv.c,inv.e*np.asarray(la)+inv.f
 old=shapely.union_all([tg(native,g) for g in raw.geometry]);l,t,r,b=old.bounds;x0=max(0,math.floor((l-768)/192)*192);y0=max(0,math.floor((t-768)/192)*192);x1=min(src.width,math.ceil((r+768)/192)*192);y1=min(src.height,math.ceil((b+768)/192)*192);win=Window(x0,y0,x1-x0,y1-y0);rgb=src.read(window=win).transpose(1,2,0);start=time.time();prob=b0.probability(rgb,progress=lambda a,b,c:print('tiles',a,b,'seconds',round(c,1),flush=True));mask=prob>.5;labels,n=ndimage.label(mask,structure=np.ones((3,3),'uint8'));oldmask=rasterize([(old,1)],out_shape=mask.shape,transform=rasterio.Affine.translation(x0,y0),dtype='uint8')>0;selected=np.unique(labels[oldmask&mask]);selected=selected[selected>0];candidate=np.isin(labels,selected);aff=src.window_transform(win)
 geoms=[tg(back.transform,shape(s)) for s,v in shapes(candidate.astype('uint8'),mask=candidate,transform=aff) if v==1];out=gpd.GeoDataFrame({'source_component':np.arange(len(geoms))+1,'geometry_role':'FULL_NATIVE_CONTEXT_RESEARCH_ROOF_CANDIDATE'},geometry=geoms,crs=raw.crs);out.to_file(O/'NATIVE_CONTEXT_REEXTRACTED.gpkg',driver='GPKG',engine='pyogrio')
 with rasterio.open(O/'NATIVE_CONTEXT_MASK.tif','w',driver='GTiff',width=candidate.shape[1],height=candidate.shape[0],count=1,dtype='uint8',crs=src.crs,transform=aff,nodata=0,compress='DEFLATE',tiled=True) as dst:dst.write(candidate.astype('uint8'),1)
 overview=Image.fromarray(rgb).resize((900,900));draw=ImageDraw.Draw(overview)
 for g,color in ((old,'cyan'),(shapely.union_all([tg(native,g) for g in out.geometry]),'lime')):
  for poly in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
   if poly.geom_type=='Polygon':draw.line([((x-x0)/(x1-x0)*900,(y-y0)/(y1-y0)*900) for x,y in poly.exterior.coords],fill=color,width=2)
 overview.save(O/'NATIVE_CONTEXT_BEFORE_AFTER.jpg')
 partial=bool(candidate[:16].any() or candidate[-16:].any() or candidate[:,:16].any() or candidate[:,-16:].any());tiles=len(b0._positions(rgb.shape[0]))*len(b0._positions(rgb.shape[1]));report=dict(status='NATIVE_CONTEXT_PILOT_NOT_GLOBAL_ADMISSION',original_ids=ids,source_window=[x0,y0,x1-x0,y1-y0],source_pixel_size='native unchanged',threshold='>0.5',connectivity=8,scheduler_batch=4,native_tiles=tiles,new_components=len(out),candidate_touches_new_context_border=partial,original_object_area_sum_m2=float(raw.geometry.area.sum()),original_union_m2=float(shapely.union_all(raw.geometry.values).area),new_candidate_union_m2=float(shapely.union_all(out.geometry.values).area),position_status='UNRESOLVED_LARGE_OBJECT_POSITION_NOT_ADMITTED',optimizer_updates=0,GT_reads=0,seconds=time.time()-start,created_local=time.strftime('%Y-%m-%d %H:%M:%S'))
 (O/'NATIVE_CONTEXT_RECEIPT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
