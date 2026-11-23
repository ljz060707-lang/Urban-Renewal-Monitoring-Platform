import os,json,hashlib,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd,shapely,rasterio
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.windows import Window
from pyproj import Transformer
from PIL import Image,ImageDraw
R=Path(__file__).parent;B=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01';O=B/'GOOD_GT_REGIONS_RED_20261029';O.mkdir(exist_ok=True);(O/'REGION_ROOF_TIFS').mkdir(exist_ok=True)
t=pd.read_parquet(B/'GT_COMPARISON_20261029/SELECTION_OBJECTS.parquet');m=t[t.reciprocal_match&(t.position_status!='GT_CONSTRAINED_IMAGE_SUPPORTED')].copy();candidates=[]
for size in [500,1000,2000]:
 m['cx']=(m.x//size*size).astype(int);m['cy']=(m.y//size*size).astype(int)
 for (x,y),a in m.groupby(['cx','cy']):
  n=len(a);mean=a.best_iou.mean();good=(a.best_iou>=.5).mean();minimum=max(8,int(15*(size/1000)**2))
  if n>=minimum and mean>=.45 and good>=.35:candidates.append(dict(x=x,y=y,size=size,matched_excluding_gt_fit=n,mean_iou_excluding_gt_fit=mean,fraction_iou_ge50=good,geometry=box(x,y,x+size,y+size)))
c=gpd.GeoDataFrame(candidates,crs='EPSG:32649');c.to_file(O/'SELECTION_CELLS.gpkg',driver='GPKG',engine='pyogrio');u=shapely.union_all(c.geometry.values);regions=list(u.geoms) if u.geom_type=='MultiPolygon' else [u];regions=sorted(regions,key=lambda a:a.area,reverse=True)
d=gpd.read_file(B/'REPAIR_V2_1495/ROOF_POSITION_REPAIRED.gpkg',engine='pyogrio').sort_values('instance_id').reset_index(drop=True);gg=d.geometry.values;tree=shapely.STRtree(gg);tr=Transformer.from_crs(32649,4326,always_xy=True)
rows=[];alllines=[];allframes=[]
def profile(w,h,a):return dict(driver='GTiff',width=w,height=h,count=4,dtype='uint8',crs='EPSG:32649',transform=a,tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=2,BIGTIFF='IF_SAFER')
def write_red(path,lines,bounds,width,height,a):
 line_tree=shapely.STRtree(lines)
 with rasterio.open(path,'w',**profile(width,height,a)) as dst:
  dst.colorinterp=(rasterio.enums.ColorInterp.red,rasterio.enums.ColorInterp.green,rasterio.enums.ColorInterp.blue,rasterio.enums.ColorInterp.alpha)
  for y in range(0,height,256):
   h=min(256,height-y);win=Window(0,y,width,h);at=dst.window_transform(win);bb=box(*rasterio.windows.bounds(win,a));idx=line_tree.query(bb,predicate='intersects');mask=rasterize([(lines[i],1) for i in idx],out_shape=(h,width),transform=at,dtype='uint8') if len(idx) else np.zeros((h,width),'uint8');z=np.zeros((4,h,width),'uint8');z[0]=mask*255;z[3]=mask*255;dst.write(z,window=win)
  dst.update_tags(role='Post-selected GT-agreement visual review; not independent evaluation',color='red RGB255,0,0 alpha255 on lines; alpha0 elsewhere',line_width_m='2 for roof outlines; 4 for region boundary')
 print('wrote',path.name,flush=True)
for k,region in enumerate(regions,1):
 ids=tree.query(region,predicate='intersects');q=t[(t.x>=region.bounds[0])&(t.x<=region.bounds[2])&(t.y>=region.bounds[1])&(t.y<=region.bounds[3])].copy();q=q[shapely.covers(region,shapely.points(q.x.to_numpy(),q.y.to_numpy()))];match=q[q.reciprocal_match];ind=match[match.position_status!='GT_CONSTRAINED_IMAGE_SUPPORTED'];fit=match.position_status=='GT_CONSTRAINED_IMAGE_SUPPORTED';lon,lat=tr.transform(region.centroid.x,region.centroid.y)
 row=dict(region_id=f'R{k:02}',area_km2=region.area/1e6,bbox_area_km2=shapely.box(*region.bounds).area/1e6,objects_centroid_inside=len(q),objects_intersecting=len(ids),spatial_matches=len(match),matches_excluding_gt_fit=len(ind),mean_iou_all_matches=float(match.best_iou.mean()),mean_iou_excluding_gt_fit=float(ind.best_iou.mean()),fraction_ge50_excluding_gt_fit=float((ind.best_iou>=.5).mean()),direct_gt_fit_fraction_of_matches=float(fit.mean()),center_lon=lon,center_lat=lat,xmin=region.bounds[0],ymin=region.bounds[1],xmax=region.bounds[2],ymax=region.bounds[3]);rows.append(row)
 # Original roof boundaries only, clipped to region; do not invent building edges at crop border.
 lines=shapely.intersection(shapely.buffer(shapely.boundary(gg[ids]),1),region);lines=lines[~shapely.is_empty(lines)];alllines.extend(lines);frame=region.boundary.buffer(2);allframes.append(frame)
 l,b,r,top=region.bounds;a=rasterio.transform.from_origin(l,top,1,1);write_red(O/'REGION_ROOF_TIFS'/f'R{k:02}_ROOF_RED_OUTLINES.tif',lines,region.bounds,int(r-l),int(top-b),a)
gpd.GeoDataFrame(rows,geometry=regions,crs=d.crs).to_file(O/'SELECTED_REGIONS.gpkg',driver='GPKG',engine='pyogrio');pd.DataFrame(rows).to_csv(O/'SELECTED_REGIONS.csv',index=False)
with rasterio.open(B/'REPAIR_V2_1495/ROOF_POSITION_REPAIRED_ALL_CANDIDATES_1M.tif') as src:
 write_red(O/'ALL_SELECTED_ROOF_RED_OUTLINES_1M.tif',np.array(alllines,dtype=object),src.bounds,src.width,src.height,src.transform)
 write_red(O/'SELECTED_REGION_RED_FRAMES_1M.tif',np.array(allframes,dtype=object),src.bounds,src.width,src.height,src.transform)
# Real imagery previews; regional edges and labels on overview, roof lines on largest six examples.
with rasterio.open(r'F:\xian\亚米市区底图\xa190101_combined\xa190101_zoom_20.tif') as src:
 def photo(bounds,w,h):
  l,b,r,top=bounds;lo,la=tr.transform([l,r],[b,top]);win=rasterio.windows.from_bounds(min(lo),min(la),max(lo),max(la),src.transform);im=Image.fromarray(src.read(window=win,out_shape=(3,h,w),resampling=rasterio.enums.Resampling.bilinear).transpose(1,2,0))
  def px(x,y):
   lon,lat=tr.transform(x,y);return (lon-min(lo))/(max(lo)-min(lo))*w,(max(la)-lat)/(max(la)-min(la))*h
  return im,px
 bounds=(293300,3781540,332150,3812400);im,px=photo(bounds,1554,1234);dr=ImageDraw.Draw(im)
 for row,g in zip(rows,regions):
  dr.line([px(x,y) for x,y in g.exterior.coords],fill='red',width=3);dr.text(px(g.centroid.x,g.centroid.y),row['region_id'],fill='white',stroke_width=1,stroke_fill='black')
 im.save(O/'SELECTED_REGIONS_OVERVIEW.jpg')
 canvas=Image.new('RGB',(1400,1230),'white')
 for k,(row,g) in enumerate(list(zip(rows,regions))[:6]):
  l,b,r,top=g.bounds;pad=30;im,px=photo((l-pad,b-pad,r+pad,top+pad),700,370);dr=ImageDraw.Draw(im);ids=tree.query(g,predicate='intersects')
  for i in ids:
   geom=gg[i]
   for poly in ([geom] if geom.geom_type=='Polygon' else list(geom.geoms)):
    if poly.geom_type=='Polygon':dr.line([px(x,y) for x,y in poly.exterior.coords],fill='red',width=1)
  x=k%2*700;y=k//2*410;canvas.paste(im,(x,y));ImageDraw.Draw(canvas).text((x+4,y+374),f"{row['region_id']} {row['area_km2']:.2f} km2; non-fit IoU {row['mean_iou_excluding_gt_fit']:.3f}; n={row['matches_excluding_gt_fit']}",fill='black')
 canvas.save(O/'LARGEST_SIX_PREVIEW.jpg')
receipt=dict(status='COMPLETE_POST_SELECTED_VISUAL_REVIEW',selection='500m/1000m/2000m cells; non-direct-GT-fit reciprocal spatial matches >=max(8,15 per km2); mean IoU>=0.45; fraction IoU>=0.5 >=0.35; union contiguous cells',cells_by_size=c.groupby('size').size().to_dict(),regions=len(regions),selected_area_km2=u.area/1e6,total_objects_centroid_inside=sum(r['objects_centroid_inside'] for r in rows),all_candidates_inside_selected_regions_displayed=True,gt_fit_excluded_from_selection_score=True,limitations=['post-selection, not new accuracy test','relative good regions, not every building certified','unmatched candidates retained and displayed','source actual dates not guaranteed 2019','no new training or inference'],created=time.strftime('%Y-%m-%d %H:%M:%S'))
(O/'SELECTION_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
