import os,json,time,threading
os.environ['GDAL_PAM_ENABLED']='NO';os.environ['GDAL_CACHEMAX']='256'
from pathlib import Path
import geopandas as gpd,numpy as np,pandas as pd,rasterio,shapely
from rasterio.features import rasterize,shapes
from rasterio.windows import Window
from pyproj import Transformer
from shapely.ops import transform as tg
from shapely.geometry import shape
from shapely.affinity import translate
from PIL import Image
from scipy.spatial import cKDTree
from ml import *
raw=gpd.read_parquet(IN/'raw.parquet');gt=gpd.read_file(IN/'reference.gpkg',engine='pyogrio').set_index('original_fid');cent=raw.geometry.centroid
coords=np.column_stack([cent.x,cent.y]);geo=Transformer.from_crs(raw.crs,'EPSG:4326',always_xy=True);metric=Transformer.from_crs('EPSG:4326',raw.crs,always_xy=True)
with rasterio.open(IN/'source.tif') as src:aff=src.transform;sw,sh=src.width,src.height
inv=~aff;lon,lat=geo.transform(coords[:,0],coords[:,1]);native_cent=np.column_stack([inv.a*lon+inv.c,inv.e*lat+inv.f])
def native(xx,yy,z=None):
 lo,la=geo.transform(xx,yy);return inv.a*np.asarray(lo)+inv.c,inv.e*np.asarray(la)+inv.f
def native_to_metric(xx,yy,z=None):return metric.transform(aff.a*np.asarray(xx)+aff.c,aff.e*np.asarray(yy)+aff.f)
trusted={};anchor_xy=[];anchor_d=[];anchor_i=[]
for i,r in raw.iterrows():
 if r.reference_link_code!=1 or r.reference_match_fid not in gt.index or r.source_edge_partial_risk:continue
 g=r.geometry;b=gt.loc[r.reference_match_fid].geometry;gc=g.centroid;bc=b.centroid;centered=translate(b,xoff=gc.x-bc.x,yoff=gc.y-bc.y);u=g.union(centered).area;iou=g.intersection(centered).area/u if u else 0;ratio=g.area/b.area if b.area else 0
 if iou>=.60 and .60<=ratio<=1.65 and gc.distance(bc)<=25:
  trusted[i]=b;anchor_i.append(i);anchor_xy.append([gc.x,gc.y]);anchor_d.append([bc.x-gc.x,bc.y-gc.y])
anchor_xy=np.asarray(anchor_xy);anchor_d=np.asarray(anchor_d);assert len(anchor_xy)>1000
tree=cKDTree(anchor_xy)
import hashlib
bucket=np.array([int(hashlib.sha256(f'{int(x//2000)},{int(y//2000)}'.encode()).hexdigest()[:8],16)%10 for x,y in anchor_xy]);trainmask=bucket<6;testmask=(bucket>=6)&(bucket<8);train_tree=cKDTree(anchor_xy[trainmask]);field_rows=[]
for k in (4,8,12,24):
 dd,ix=train_tree.query(anchor_xy[testmask],k=k)
 for radius in (500,1000,2000):
  valid=dd<radius;weights=valid/(dd+50)**2;pred=(anchor_d[trainmask][ix]*weights[:,:,None]).sum(1)/np.maximum(weights.sum(1),1e-12)[:,None];pred[valid.sum(1)<4]=0
  field_rows.append(dict(neighbors=k,radius_m=radius,validation_mean_error_m=float(np.linalg.norm(pred-anchor_d[testmask],axis=1).mean()),zero_shift_mean_error_m=float(np.linalg.norm(anchor_d[testmask],axis=1).mean()),validation_pairs=int(testmask.sum())))
pd.DataFrame(field_rows).to_csv(OUT/'reference_field_validation.csv',index=False);field_best=min(field_rows,key=lambda x:x['validation_mean_error_m']);field_admitted=field_best['validation_mean_error_m']<field_best['zero_shift_mean_error_m']*.98
save('reference_field_parameters.json',dict(**field_best,admitted=field_admitted,validation='spatially held-out 2km blocks; TRAIN anchors only during parameter selection; all permitted anchors used for reference-constrained final map'))
val=json.loads((OUT/'alignment_validation.json').read_text())['metrics'];audit=json.loads((OUT/'alignment_spatial_audit.json').read_text())['metrics'];admit_shift=val['shift_error_m']<val['baseline_shift_error_m']*.95 and audit['shift_error_m']<audit['baseline_shift_error_m'];admit_contour=val['iou']>val['baseline_contour_iou']+.02 and audit['iou']>audit['baseline_contour_iou']
facval=json.loads((OUT/'facade_validation.json').read_text())['metrics'];facpolicy=json.loads((OUT/'facade_threshold_parameters.json').read_text());admit_facade=facpolicy['admitted'];facade_threshold=float(facpolicy['threshold'])
save('PRODUCTION_POLICY.json',dict(anchors=len(trusted),reference_contour_use='USER_AUTHORIZED_GT_CONSTRAINED: only image-supported compatible mutual-nearest pairs; not independent accuracy',unlabelled_is_background=False,neural_shift_admitted=admit_shift,neural_contour_admitted=admit_contour,facade_expert_admitted=admit_facade,facade_cut_rule='legacy E3 risk AND calibrated new facade probability threshold AND roof probability<.20; never erase whole object',facade_threshold=facade_threshold,gt_reference_capture_date='UNVERIFIED',reference_field='spatially selected k/radius; weakly supported regions retained and flagged',physical_area='UNRESOLVED_NO_VIEW_GEOMETRY'))
class City(Dataset):
 def __len__(self):return len(raw)
 def __getitem__(self,i):
  if not hasattr(self,'src'):self.src=rasterio.open(IN/'source.tif');self.risk=rasterio.open(IN/'e3_risk.tif')
  r=raw.iloc[i];g=r.geometry;cx,cy=native_cent[i];ox=int(np.clip(np.floor(cx-256),0,sw-512));oy=int(np.clip(np.floor(cy-256),0,sh-512));ng=tg(native,g);bb=ng.bounds;complete=not r.source_edge_partial_risk and bb[0]>=ox and bb[1]>=oy and bb[2]<=ox+512 and bb[3]<=oy+512
  rgb=np.zeros((512,512,3),'uint8');mask=np.zeros((256,256),'uint8');old=np.zeros((512,512),'uint8')
  if complete and i not in trusted:
   rgb=self.src.read(window=Window(ox,oy,512,512)).transpose(1,2,0);big=rasterize([(ng,1)],out_shape=(512,512),transform=rasterio.Affine.translation(ox,oy),dtype='uint8');mask=np.asarray(Image.fromarray(big).resize((256,256),Image.Resampling.NEAREST));old=(self.risk.read(1,window=Window(ox,oy,512,512))==1).astype('uint8')
  small=np.asarray(Image.fromarray(rgb).resize((256,256),Image.Resampling.BILINEAR));x=np.concatenate([small.transpose(2,0,1),mask[None]*255],0).astype('float32')/255;c=coords[i];meta=np.array([(c[0]-320000)/30000,(c[1]-3800000)/30000,np.log1p(g.area)/10,min(float(r.aspect_ratio),10)/10,g.area/20000],'float32')
  return i,torch.from_numpy(x.copy()),torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,torch.from_numpy(meta),old,np.array([ox,oy],dtype='int64'),complete
align=Net(4,1).cuda().eval();facade=Net(3,3).cuda().eval();align.load_state_dict(torch.load(OUT/'alignment_best.pt',map_location='cpu')['model']);facade.load_state_dict(torch.load(OUT/'facade_best.pt',map_location='cpu')['model'])
dl=DataLoader(City(),batch_size=16,shuffle=False,num_workers=12,pin_memory=True,persistent_workers=True,prefetch_factor=2)
records=[];geoms=[];chunk=0
def flush():
 global records,geoms,chunk
 if not records:return
 d=gpd.GeoDataFrame(records,geometry=geoms,crs=raw.crs);d.to_parquet(OUT/'city_chunks'/f'{chunk:05d}.parquet',index=False);chunk+=1;records=[];geoms=[]
(OUT/'city_chunks').mkdir(exist_ok=True)
with torch.inference_mode():
 for batch in dl:
  indices,x,rgb,meta,old,origins,complete=batch;active=torch.tensor([bool(ok) and int(i) not in trusted for i,ok in zip(indices,complete)])
  shift=np.zeros((len(indices),2),'float32');contours=np.zeros((len(indices),256,256),'bool');cuts=np.zeros((len(indices),512,512),'bool')
  if active.any():
   with torch.autocast('cuda',dtype=torch.float16):
    z,p=align(x[active].cuda(non_blocking=True),meta[active].cuda(non_blocking=True));fs,_=facade(rgb[active].cuda(non_blocking=True));prob=fs.float().softmax(1)
   ids=np.where(active.numpy())[0];shift[ids]=p.float().cpu().numpy()*10;contours[ids]=(z[:,0]>0).cpu().numpy()
   if admit_facade:cuts[ids]=((prob[:,2]>facade_threshold)&(prob[:,1]<.2)).cpu().numpy()&(old[active].numpy()>0)
  for j,ii in enumerate(indices.tolist()):
   r=raw.iloc[ii];g=r.geometry;before=g.area;status_code='MODEL_ONLY_UNRESOLVED';method='RETAINED_RAW';dx=dy=0.;removed=0.;physical='UNRESOLVED_NO_VIEW_GEOMETRY'
   if ii in trusted:
    b=trusted[ii];dx=b.centroid.x-g.centroid.x;dy=b.centroid.y-g.centroid.y;g=b;status_code='GT_CONSTRAINED_IMAGE_SUPPORTED';method='REFERENCE_CONSTRAINED_CONTOUR_AND_POSITION'
   elif bool(complete[j]):
    ox,oy=origins[j].tolist()
    if cuts[j].any():
     pieces=[tg(native_to_metric,shape(s)) for s,v in shapes(cuts[j].astype('uint8'),mask=cuts[j],transform=rasterio.Affine.translation(ox,oy)) if v==1];cut=g.intersection(shapely.union_all(pieces));candidate=g.difference(cut)
     if candidate.area>=.4*g.area:g=candidate;removed=cut.area;method='FACADE_EXPERT_SUBTRACTION'
    distance,idx=tree.query(coords[ii],k=field_best['neighbors']);keep=distance<field_best['radius_m']
    if field_admitted and keep.sum()>=4:
     weights=1/(distance[keep]+50)**2;field=np.sum(anchor_d[idx[keep]]*weights[:,None],axis=0)/weights.sum();dx,dy=field.tolist();status_code='REFERENCE_FIELD_SUPPORTED';method+=';LOCAL_REFERENCE_FIELD'
    elif admit_shift and distance[0]<3000:dx,dy=shift[j].tolist();status_code='LEARNED_SHIFT_CANDIDATE';method+=';LEARNED_DISPLACEMENT'
    if admit_contour and distance[0]<3000 and contours[j].any():
     pp=[tg(native_to_metric,shape(s)) for s,v in shapes(contours[j].astype('uint8'),mask=contours[j],transform=rasterio.Affine.translation(ox,oy)*rasterio.Affine.scale(2,2)) if v==1];proposal=shapely.union_all(pp).intersection(g.buffer(2));u=proposal.union(g).area
     if .6*g.area<=proposal.area<=1.3*g.area and proposal.intersection(g).area/max(u,1)>.6:g=proposal;method+=';LEARNED_CONTOUR'
    g=translate(g,xoff=dx,yoff=dy)
   if g.is_empty or not g.is_valid:g=shapely.make_valid(g)
   if g.is_empty:g=r.geometry;status_code='UNRESOLVED_RETAINED_RAW';method='EMPTY_RESULT_REJECTED'
   records.append(dict(instance_id=int(r.instance_id),observation_id=r.observation_id,actual_date=r.actual_date,date_status=r.date_status,position_status=status_code,method=method,reference_match_fid=int(r.reference_match_fid),area_raw_m2=before,area_governed_m2=g.area,geometry_area_delta_m2=g.area-before,facade_removed_m2=removed,facade_area_semantics='subtraction step before optional contour refinement; not total final area change',dx_m=dx,dy_m=dy,physical_area_status=physical,area_physical_rectified_m2=None,formal_weight=0,source_edge_partial=bool(r.source_edge_partial_risk)));geoms.append(g)
  done=int(indices[-1])+1
  if len(records)>=1000:flush();status('CITY_INFERENCE',objects_done=done,objects_total=len(raw))
flush();save('CITY_INFERENCE_COMPLETE.json',dict(objects=len(raw),chunks=chunk,all_objects_retained_or_governed=True));status('CITY_INFERENCE_COMPLETE',objects=len(raw))
