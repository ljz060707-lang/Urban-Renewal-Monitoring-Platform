import os,json,time
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd,shapely,rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
from pyproj import Transformer
from shapely.ops import transform as tg
from PIL import Image,ImageDraw
R=Path(__file__).parent;B=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01';O=B/'GT_COMPARISON_20261029';O.mkdir(exist_ok=True)
d=gpd.read_file(B/'REPAIR_V2_1495/ROOF_POSITION_REPAIRED.gpkg',engine='pyogrio').sort_values('instance_id').reset_index(drop=True)
raw=gpd.read_parquet(R/'payload/raw.parquet');assert np.array_equal(raw.observation_id,d.observation_id)
gt=gpd.read_file(R.parent/'XIAN_R5_SINGLE_DATE_20261028_01/reference_2019/ROOF_GT2019_REFERENCE.gpkg',engine='pyogrio').to_crs(d.crs)
dg=d.geometry.values;gg=gt.geometry.values;rg=raw.geometry.values;da=shapely.area(dg);ga=shapely.area(gg);ra=shapely.area(rg);tree=shapely.STRtree(gg)
print('loaded',len(d),len(gt),flush=True)
# Explicit spatial diagnostic matching: largest IoU, reciprocal best, threshold .1.
pi,pj=tree.query(dg,predicate='intersects');ia=shapely.area(shapely.intersection(dg[pi],gg[pj]));iou=ia/(da[pi]+ga[pj]-ia+1e-12)
pairs=pd.DataFrame(dict(i=pi,j=pj,intersection=ia,iou=iou));pairs=pairs[pairs.intersection>0].sort_values('iou',ascending=False)
best=pairs.drop_duplicates('i').set_index('i');rev=pairs.drop_duplicates('j').set_index('j');valid=best.iou>=.1
matched=best[valid].copy();matched=matched[rev.loc[matched.j,'i'].to_numpy()==matched.index.to_numpy()]
tab=d.drop(columns='geometry').copy();tab['best_gt_fid']=-1;tab['best_iou']=0.;tab['matched_gt_index']=-1;tab['reciprocal_match']=False
tab.loc[best.index,'best_iou']=best.iou;tab.loc[best.index,'best_gt_fid']=gt.original_fid.to_numpy()[best.j]
ii=matched.index.to_numpy();jj=matched.j.to_numpy();tab.loc[ii,'reciprocal_match']=True;tab.loc[ii,'matched_gt_index']=jj
bi=shapely.area(shapely.intersection(rg[ii],gg[jj]));rawiou=bi/(ra[ii]+ga[jj]-bi+1e-12)
tab['raw_same_gt_iou']=np.nan;tab.loc[ii,'raw_same_gt_iou']=rawiou;tab['iou_change']=tab.best_iou-tab.raw_same_gt_iou
tab['centroid_before_m']=np.nan;tab['centroid_after_m']=np.nan
tab.loc[ii,'centroid_before_m']=shapely.distance(shapely.centroid(rg[ii]),shapely.centroid(gg[jj]));tab.loc[ii,'centroid_after_m']=shapely.distance(shapely.centroid(dg[ii]),shapely.centroid(gg[jj]))
tab['gt_area_ratio']=np.nan;tab.loc[ii,'gt_area_ratio']=da[ii]/ga[jj]
rect=shapely.minimum_rotated_rectangle(dg);coords=shapely.get_coordinates(rect).reshape(-1,5,2);sides=np.linalg.norm(coords[:,1:]-coords[:,:-1],axis=2)
tab['aspect_ratio_eval']=sides.max(1)/np.maximum(sides.min(1),1e-9);tab['compactness_eval']=4*np.pi*da/np.maximum(shapely.length(dg)**2,1e-9)
tab['size_class']=pd.cut(da,[0,200,800,3000,10000,np.inf],labels=['LT200','200_800','800_3000','3000_10000','GE10000'],include_lowest=True).astype(str)
tab['shape_class']=np.where(tab.aspect_ratio_eval>=4,'ELONGATED',np.where(tab.compactness_eval<.35,'IRREGULAR_OR_MERGED','COMPACT'))
c=shapely.centroid(dg);xx=shapely.get_x(c);yy=shapely.get_y(c);tab['cell_x_2km']=(xx//2000*2000).astype(int);tab['cell_y_2km']=(yy//2000*2000).astype(int)
tab.to_csv(O/'OBJECT_GT_COMPARISON.csv',index=False)
def stats(t):
 m=t[t.reciprocal_match];return dict(objects=len(t),matched=len(m),matching_rate=len(m)/max(1,len(t)),mean_iou=float(m.best_iou.mean()),median_iou=float(m.best_iou.median()),iou_ge50=float((m.best_iou>=.5).mean()),iou_ge75=float((m.best_iou>=.75).mean()),raw_mean_iou_same_gt=float(m.raw_same_gt_iou.mean()),mean_iou_change=float(m.iou_change.mean()),improved_fraction=float((m.iou_change>.01).mean()),worsened_fraction=float((m.iou_change<-.01).mean()),mean_centroid_before_m=float(m.centroid_before_m.mean()),mean_centroid_after_m=float(m.centroid_after_m.mean()),median_centroid_after_m=float(m.centroid_after_m.median()),median_area_ratio=float(m.gt_area_ratio.median()))
groups=[]
for col in ['position_status','size_class','shape_class']:
 for key,t in tab.groupby(col,observed=True):groups.append(dict(dimension=col,group=str(key),**stats(t)))
pd.DataFrame(groups).to_csv(O/'GROUP_METRICS.csv',index=False)
sp=[]
for (x,y),t in tab.groupby(['cell_x_2km','cell_y_2km']):
 independent=t[t.position_status!='GT_CONSTRAINED_IMAGE_SUPPORTED'];sp.append(dict(x=int(x),y=int(y),**stats(independent)))
pd.DataFrame(sp).to_csv(O/'SPATIAL_2KM_METRICS.csv',index=False)
gpd.GeoDataFrame(sp,geometry=[box(z['x'],z['y'],z['x']+2000,z['y']+2000) for z in sp],crs=d.crs).to_file(O/'SPATIAL_2KM_METRICS.gpkg',engine='pyogrio',driver='GPKG')
print('object metrics done',len(matched),flush=True)
# Exact geometric GT coverage (union overlapping predictions separately per reference).
cov=np.zeros(len(gt));cover_groups=pairs.groupby('j').i
for j,ids in cover_groups:
 ids=ids.to_numpy();cov[j]=shapely.area(shapely.intersection(gg[j],shapely.union_all(dg[ids])))
gtt=gt.drop(columns='geometry').copy();gtt['covered_area_m2']=cov;gtt['covered_fraction']=cov/ga;gtt['best_prediction_iou']=0.;gtt.loc[rev.index,'best_prediction_iou']=rev.iou;gtt.to_csv(O/'GT_COVERAGE.csv',index=False)
# Pixel overlay has aligned 1m grids; no inferred annotation-completeness mask.
counts=np.zeros(4,dtype=np.int64);gtree=tree;dtree=shapely.STRtree(dg);processed=tab.position_status.to_numpy()!='MODEL_ONLY_UNRESOLVED';pc=np.zeros(4,dtype=np.int64)
with rasterio.open(B/'REPAIR_V2_1495/ROOF_POSITION_REPAIRED_ALL_CANDIDATES_1M.tif') as src:
 p=src.profile.copy();p.update(count=1,dtype='uint8',nodata=255)
 with rasterio.open(O/'GT_OVERLAY_CLASSES_1M.tif','w',**p) as dst:
  for y in range(0,src.height,512):
   h=min(512,src.height-y);win=Window(0,y,src.width,h);a=src.window_transform(win);bb=box(*rasterio.windows.bounds(win,src.transform));ids=gtree.query(bb,predicate='intersects');g=rasterize([(gg[i],1) for i in ids],out_shape=(h,src.width),transform=a,dtype='uint8') if len(ids) else np.zeros((h,src.width),'uint8');s=src.read(1,window=win);v=s!=255;z=(s==1).astype('uint8')+2*g;counts+=np.bincount(z[v],minlength=4);z[~v]=255;dst.write(z,1,window=win)
   ids=dtree.query(bb,predicate='intersects');ids=ids[processed[ids]];pr=rasterize([(dg[i],1) for i in ids],out_shape=(h,src.width),transform=a,dtype='uint8') if len(ids) else np.zeros(g.shape,'uint8');pc+=np.bincount((pr+2*g)[v],minlength=4)
   if y%5120==0:print('raster',y,flush=True)
def px(a):
 tn,predonly,gtonly,both=map(int,a);return dict(both_m2=both,prediction_only_m2=predonly,gt_only_m2=gtonly,gt_coverage_recall=both/(both+gtonly),reference_overlap_precision_proxy=both/(both+predonly),reference_iou=both/(both+predonly+gtonly),reference_dice=2*both/(2*both+predonly+gtonly))
summary=dict(status='REFERENCE_AGREEMENT_DIAGNOSTIC_NOT_INDEPENDENT_ACCURACY',matching_rule='reciprocal largest spatial IoU >=0.1; cannot guarantee identity; unmatched retained in totals',all_objects=stats(tab),processed=stats(tab[processed]),processed_excluding_direct_gt_fit=stats(tab[processed&(tab.position_status!='GT_CONSTRAINED_IMAGE_SUPPORTED')]),direct_gt_fit=stats(tab[tab.position_status=='GT_CONSTRAINED_IMAGE_SUPPORTED']),overall_pixel=px(counts),processed_pixel_against_all_gt=px(pc),gt_objects=len(gt),gt_objects_covered_ge50=int((cov/ga>=.5).sum()),gt_objects_best_iou_ge50=int((gtt.best_prediction_iou>=.5).sum()),gt_exact_object_area_coverage=float(cov.sum()/ga.sum()),limitations=['GT absence is not background; precision proxy is not real precision','direct GT fits are circular agreement, not independent accuracy','spatial mutual IoU matching may confuse adjacent or merged buildings','GT acquisition date and roof/footprint semantics unresolved','no inference or training performed'],created=time.strftime('%Y-%m-%d %H:%M:%S'))
(O/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
# Deterministic examples across good/poor shape and size groups, excluding direct fits.
mt=tab[tab.reciprocal_match&(tab.position_status!='GT_CONSTRAINED_IMAGE_SUPPORTED')];cases=[]
for shape,quality in [('COMPACT','GOOD'),('ELONGATED','GOOD'),('IRREGULAR_OR_MERGED','GOOD'),('COMPACT','POOR'),('ELONGATED','POOR'),('IRREGULAR_OR_MERGED','POOR')]:
 t=mt[(mt.shape_class==shape)&(da[mt.index]>100)&(da[mt.index]<50000)];t=t.sort_values('best_iou',ascending=quality=='POOR');cases.extend(t.head(2).index.tolist())
canvas=Image.new('RGB',(1400,400*((len(cases)+1)//2)),'white');tr=Transformer.from_crs(d.crs,'EPSG:4326',always_xy=True)
with rasterio.open(r'F:\xian\亚米市区底图\xa190101_combined\xa190101_zoom_20.tif') as src:
 for k,i in enumerate(cases):
  j=int(tab.loc[i,'matched_gt_index']);bounds=shapely.union_all([dg[i],gg[j],rg[i]]).bounds;x0,y0,x1,y1=bounds;side=max(90,x1-x0+30,y1-y0+30);cx=(x0+x1)/2;cy=(y0+y1)/2;bounds=(cx-side/2,cy-side/2,cx+side/2,cy+side/2);lo,la=tr.transform([bounds[0],bounds[2]],[bounds[1],bounds[3]]);win=rasterio.windows.from_bounds(min(lo),min(la),max(lo),max(la),src.transform);rgb=src.read(window=win,out_shape=(3,350,700),resampling=rasterio.enums.Resampling.bilinear).transpose(1,2,0);im=Image.fromarray(rgb);dr=ImageDraw.Draw(im)
  def pixel(x,y):
   lon,lat=tr.transform(x,y);return (lon-min(lo))/(max(lo)-min(lo))*700,(max(la)-lat)/(max(la)-min(la))*350
  # all surrounding reference/current outlines, focal raw in orange
  bx=box(*bounds)
  for geoms,indices,color in [(gg,tree.query(bx,predicate='intersects'),'yellow'),(dg,dtree.query(bx,predicate='intersects'),'cyan'),(rg,[i],'orange')]:
   for n in indices:
    g=geoms[n]
    for poly in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
     if poly.geom_type=='Polygon':dr.line([pixel(x,y) for x,y in poly.exterior.coords],fill=color,width=2)
  xx1=k%2*700;yy1=k//2*400;canvas.paste(im,(xx1,yy1));ImageDraw.Draw(canvas).text((xx1+4,yy1+352),f'ID {i+1} {tab.loc[i,"shape_class"]} IoU {tab.loc[i,"raw_same_gt_iou"]:.3f}->{tab.loc[i,"best_iou"]:.3f}',fill='black');ImageDraw.Draw(canvas).text((xx1+4,yy1+370),str(tab.loc[i,'position_status']),fill='black')
canvas.save(O/'CASE_OVERLAYS.jpg');tab.loc[cases].to_csv(O/'CASE_INDEX.csv',index=False)
print('COMPLETE',flush=True)
