import os,json,time,hashlib,multiprocessing as mp
os.environ['GDAL_PAM_ENABLED']='NO';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
import geopandas as gpd,numpy as np,pandas as pd,rasterio,shapely
from rasterio.features import rasterize
from rasterio.windows import Window
from affine import Affine
from pyproj import Transformer
from shapely.geometry import box,Polygon
ROOT=Path('/root/xian2019');IN=ROOT/'input';OUT=ROOT/'output';PRODUCT=OUT/'products';PRODUCT.mkdir(exist_ok=True)
def save(name,d):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str));os.replace(t,p)
def status(stage,**kw):save('STATUS.json',dict(stage=stage,time=datetime.now().astimezone().isoformat(),**kw));print(stage,kw,flush=True)
frames=[gpd.read_parquet(p) for p in sorted((OUT/'city_chunks').glob('*.parquet'))];d=pd.concat(frames,ignore_index=True);d=gpd.GeoDataFrame(d,geometry='geometry',crs=frames[0].crs);d=d.sort_values('instance_id').reset_index(drop=True);assert len(d)==209137 and d.instance_id.nunique()==209137
with rasterio.open(IN/'lr.tif') as src:coarse=src.transform;cw,ch=src.width,src.height;crs=src.crs
fine=coarse*Affine.scale(.1,.1);fw,fh=cw*10,ch*10;aoi=box(coarse.c,coarse.f+coarse.e*ch,coarse.c+coarse.a*cw,coarse.f)
with rasterio.open(IN/'source.tif') as src:l,b,r,t=src.bounds;sourcecrs=src.crs
project=Transformer.from_crs(sourcecrs,crs,always_xy=True);xx=np.concatenate([np.linspace(l,r,128),np.full(128,r),np.linspace(r,l,128),np.full(128,l)]);yy=np.concatenate([np.full(128,b),np.linspace(b,t,128),np.full(128,t),np.linspace(t,b,128)]);x,y=project.transform(xx,yy);support=Polygon(zip(x,y)).intersection(aoi)
status('CLIP_AND_VECTOR_EXPORT',objects=len(d));full=d.geometry.values;geoms=shapely.intersection(shapely.make_valid(full),support);d['area_within_aoi_source_m2']=shapely.area(geoms);d['geometry_scope']='FULL_OBSERVED_OBJECT; TIF restricted to frozen AOI and source support';d.to_file(PRODUCT/'ROOF_GOVERNED_FULL_OBJECTS.gpkg',driver='GPKG',engine='pyogrio');d.drop(columns='geometry').to_csv(PRODUCT/'OBJECT_AREA.csv',index=False)
clipped=d.copy();clipped.geometry=geoms;clipped.to_file(PRODUCT/'ROOF_GOVERNED.gpkg',driver='GPKG',engine='pyogrio');tree=shapely.STRtree(geoms)
codes={'GT_CONSTRAINED_IMAGE_SUPPORTED':1,'REFERENCE_FIELD_SUPPORTED':2,'LEARNED_SHIFT_CANDIDATE':3};position=np.array([codes.get(s,4) for s in d.position_status],dtype='uint8')
profile=dict(driver='GTiff',width=fw,height=fh,count=1,crs=crs,transform=fine,compress='DEFLATE',tiled=True,blockxsize=256,blockysize=256,BIGTIFF='YES')
from contextlib import ExitStack
with ExitStack() as stack:
 roof=stack.enter_context(rasterio.open(PRODUCT/'ROOF_GOVERNED_1M.tif','w',dtype='uint8',nodata=255,**profile));ident=stack.enter_context(rasterio.open(PRODUCT/'ROOF_INSTANCE_ID_1M.tif','w',dtype='uint32',nodata=0,**profile));qual=stack.enter_context(rasterio.open(PRODUCT/'POSITION_STATUS_1M.tif','w',dtype='uint8',nodata=255,**profile))
 for r0 in range(0,fh,512):
  h=min(512,fh-r0);a=fine*Affine.translation(0,r0);bounds=box(a.c,a.f+a.e*h,a.c+a.a*fw,a.f);idx=tree.query(bounds,predicate='intersects');idx=idx[~shapely.is_empty(geoms[idx])];window=Window(0,r0,fw,h);sv=rasterize([(support,1)],out_shape=(h,fw),transform=a,dtype='uint8')>0
  mask=rasterize([(geoms[i],1) for i in idx],out_shape=(h,fw),transform=a,dtype='uint8') if len(idx) else np.zeros((h,fw),'uint8');mask[~sv]=255;roof.write(mask,1,window=window)
  ids=rasterize([(geoms[i],int(d.instance_id.iloc[i])) for i in idx],out_shape=(h,fw),transform=a,dtype='uint32') if len(idx) else np.zeros((h,fw),'uint32');ident.write(ids,1,window=window)
  q=rasterize([(geoms[i],int(position[i])) for i in idx],out_shape=(h,fw),transform=a,dtype='uint8') if len(idx) else np.zeros((h,fw),'uint8');q[~sv]=255;qual.write(q,1,window=window)
  status('FINE_TIF_EXPORT',rows=r0+h,total_rows=fh)
def tile(job):
 r0,c0,h,w=job;aff=coarse*Affine.translation(c0,r0);cell=box(aff.c,aff.f+aff.e*h,aff.c+aff.a*w,aff.f);idx=tree.query(cell,predicate='intersects');idx=idx[~shapely.is_empty(geoms[idx])];out=np.zeros((h,w),'float32');sv=np.zeros((h,w),'float32');uarea=0.
 rr,cc=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');xs=aff.c+cc.ravel()*aff.a;ys=aff.f+rr.ravel()*aff.e;cells=shapely.box(xs,ys+aff.e,xs+aff.a,ys)
 sv[:]=shapely.area(shapely.intersection(cells,support)).reshape(h,w)/100
 if len(idx):
  union=shapely.union_all(shapely.intersection(geoms[idx],cell));uarea=union.area;hit=shapely.intersects(cells,union);vals=np.zeros(h*w,'float64');vals[hit]=shapely.area(shapely.intersection(cells[hit],union))/100;out[:]=vals.reshape(h,w)
 out[sv<=0]=-9999
 return job,out,sv,uarea
jobs=[(r,c,min(100,ch-r),min(100,cw-c)) for r in range(0,ch,100) for c in range(0,cw,100)];fraction=np.zeros((ch,cw),'float32');sourcesupport=np.zeros((ch,cw),'float32');total_union=0.
status('EXACT_UNION_FRACTION',tiles=len(jobs),cpu_workers=12)
with mp.get_context('fork').Pool(12) as pool:
 for done,(job,frac,sv,area) in enumerate(pool.imap_unordered(tile,jobs),1):
  r,c,h,w=job;fraction[r:r+h,c:c+w]=frac;sourcesupport[r:r+h,c:c+w]=sv;total_union+=area
  if done%20==0:status('EXACT_UNION_FRACTION',tiles_done=done,tiles_total=len(jobs))
for name,arr,nodata in [('ROOF_FRACTION_10M.tif',fraction,-9999),('SOURCE_SUPPORT_10M.tif',sourcesupport,-9999)]:
 with rasterio.open(PRODUCT/name,'w',driver='GTiff',width=cw,height=ch,count=1,dtype='float32',crs=crs,transform=coarse,nodata=nodata,compress='DEFLATE',tiled=True) as dst:dst.write(arr,1)
integral=float(fraction[fraction>=0].astype('float64').sum()*100);assert abs(integral-total_union)<max(1,total_union*1e-7)
summary=dict(epoch='xa190101',nominal_year=2019,actual_dates='2018-11-01|2018-11-08',aoi_area_m2=aoi.area,source_supported_area_m2=support.area,source_supported_share=support.area/aoi.area,objects=len(d),empty_after_aoi_clip=int(shapely.is_empty(geoms).sum()),full_object_area_sum_m2=float(d.area_governed_m2.sum()),within_support_object_area_sum_m2=float(d.area_within_aoi_source_m2.sum()),within_support_union_m2=total_union,overlap_m2=float(d.area_within_aoi_source_m2.sum()-total_union),fraction_integral_m2=integral,fraction_integral_minus_exact_union_m2=integral-total_union,explicit_facade_removed_sum_m2=float(d.facade_removed_m2.sum()),GT_constrained_objects=int((position==1).sum()),reference_field_objects=int((position==2).sum()),learned_shift_objects=int((position==3).sum()),unresolved_retained_objects=int((position==4).sum()),area_units='m2',physical_area_status='UNRESOLVED_NO_VIEW_GEOMETRY')
pd.DataFrame([summary]).to_csv(PRODUCT/'AREA_SUMMARY.csv',index=False);d.groupby('position_status').agg(objects=('instance_id','size'),object_area_sum_m2=('area_within_aoi_source_m2','sum')).to_csv(PRODUCT/'AREA_BY_POSITION_STATUS.csv')
save('AREA_SUMMARY.json',summary);save('POSITION_STATUS_CODES.json',dict(no_object=0,GT_CONSTRAINED_IMAGE_SUPPORTED=1,REFERENCE_FIELD_SUPPORTED=2,LEARNED_SHIFT_CANDIDATE=3,UNRESOLVED_RETAINED=4,NO_SOURCE_SUPPORT=255))
import pyogrio
import shutil
for name in ('ACTUAL_DATE_MAP.tif','ACTUAL_DATE_MAP_CODES.json'):
 if (IN/name).exists():shutil.copy2(IN/name,PRODUCT/name)
readback={}
for f in PRODUCT.glob('*.tif'):
 with rasterio.open(f) as ds:readback[f.name]=dict(width=ds.width,height=ds.height,crs=str(ds.crs),affine=list(ds.transform),dtype=ds.dtypes[0],nodata=ds.nodata,center_sample=ds.read(1,window=Window(ds.width//2,ds.height//2,16,16)).shape)
assert pyogrio.read_info(PRODUCT/'ROOF_GOVERNED.gpkg')['features']==209137
save('PRODUCT_READBACK.json',dict(status='PASS',rasters=readback,vector_objects=209137,exact_area_check=integral-total_union))
from PIL import Image,ImageDraw
from rasterio.windows import from_bounds
roi_bounds=(319180,3788800,319420,3789020);pcent=d.geometry.centroid;chosen=d[(pcent.x>=roi_bounds[0])&(pcent.x<roi_bounds[2])&(pcent.y>=roi_bounds[1])&(pcent.y<roi_bounds[3])]
baseline=gpd.read_parquet(IN/'raw.parquet',columns=['instance_id','geometry']);baseline=baseline[baseline.instance_id.isin(chosen.instance_id)]
project_back=Transformer.from_crs(crs,sourcecrs,always_xy=True)
with rasterio.open(IN/'source.tif') as src:
 lo,la=project_back.transform([roi_bounds[0],roi_bounds[2]],[roi_bounds[1],roi_bounds[3]]);win=from_bounds(min(lo),min(la),max(lo),max(la),src.transform);rgb=src.read(window=win,out_shape=(3,770,840),resampling=rasterio.enums.Resampling.bilinear).transpose(1,2,0)
 def pix(x,y):
  xx,yy=project_back.transform(x,y);return (xx-min(lo))/(max(lo)-min(lo))*840,(max(la)-yy)/(max(la)-min(la))*770
 canvas=Image.new('RGB',(1680,800),'white')
 for side,gs,color,label in ((0,baseline.geometry,'red','BASELINE RAW'),(1,chosen.geometry,'cyan','GOVERNED: GT constrained or validated model')):
  im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
  for g in gs:
   if g.is_empty:continue
   for poly in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
    if poly.geom_type=='Polygon':draw.line([pix(x,y) for x,y in poly.exterior.coords],fill=color,width=2)
  canvas.paste(im,(side*840,30));ImageDraw.Draw(canvas).text((side*840+5,8),label,fill='black')
 canvas.save(OUT/'RESIDENTIAL_ROI_BEFORE_AFTER.png')
notes='''本轮为名义2019期xa190101，实际主要影像日期为2018-11-01及2018-11-08。\nGT有遗漏：未标注不是背景。GT约束产品使用已观察且形状兼容的互近邻参考对象，不代表独立预测精度。\n外立面标签来自BANDON几何投影辅助；不是西安逐栋可见立面真值。\n主10米分数是精确并集面积/100平方米，范围0到1；对象面积和含重叠，二者不混淆。\n1米二值：1候选屋顶/参考平面覆盖，0未被候选覆盖，255无源影像；0不是认证背景。\n物理屋顶表面积及相机校正仍未建立，不伪造相机信息。\n训练曲线为真实优化器更新，空间验证为诊断，不能称未曝光盲测试。\n'''
(OUT/'QUALITY_NOTES_CN.md').write_text(notes,encoding='utf-8')
save('FINAL_RECEIPT.json',dict(status='COMPLETE_WITH_EXPLICIT_QUALITY_LIMITATIONS',time=datetime.now().astimezone().isoformat(),summary=summary,training_artifacts=['alignment_parameters.json','alignment_curves.csv','alignment_curves.png','alignment_best.pt','alignment_validation.json','alignment_spatial_audit.json','facade_parameters.json','facade_curves.csv','facade_curves.png','facade_best.pt','facade_validation.json'],reference_constraint_not_independent_accuracy=True,physical_rectification='UNRESOLVED_NO_VIEW_GEOMETRY',readback='PASS',deadline='2026-10-29T09:30:00+08:00'))
hashes={}
for p in sorted(PRODUCT.glob('*')):
 if not p.is_file():continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 hashes[p.name]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
save('PRODUCT_SHA256.json',hashes);status('COMPLETE',products=len(hashes))
