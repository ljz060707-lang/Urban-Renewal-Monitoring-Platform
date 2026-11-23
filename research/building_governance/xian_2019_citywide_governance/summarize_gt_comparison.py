import os,json
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import pandas as pd,numpy as np,rasterio,geopandas as gpd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
R=Path(__file__).parent;B=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01';O=B/'GT_COMPARISON_20261029'
t=pd.read_csv(O/'OBJECT_GT_COMPARISON.csv');gt=gpd.read_file(R.parent/'XIAN_R5_SINGLE_DATE_20261028_01/reference_2019/ROOF_GT2019_REFERENCE.gpkg',engine='pyogrio')
t['gt_floor_reported']=t.best_gt_fid.map(gt.set_index('original_fid').Floor)
groups=[]
for scope,tt in [('ALL_EXCLUDING_DIRECT_GT_FIT',t[t.position_status!='GT_CONSTRAINED_IMAGE_SUPPORTED']),('PROCESSED_EXCLUDING_DIRECT_GT_FIT',t[~t.position_status.isin(['GT_CONSTRAINED_IMAGE_SUPPORTED','MODEL_ONLY_UNRESOLVED'])])]:
 for dim in ['size_class','shape_class']:
  for name,sub in tt.groupby(dim):
   m=sub[sub.reciprocal_match];groups.append(dict(scope=scope,dimension=dim,group=name,objects=len(sub),matched=len(m),mean_iou=m.best_iou.mean(),raw_mean_iou=m.raw_same_gt_iou.mean(),median_iou=m.best_iou.median(),mean_iou_change=m.iou_change.mean(),iou_ge50=(m.best_iou>=.5).mean(),centroid_before_mean_m=m.centroid_before_m.mean(),centroid_after_mean_m=m.centroid_after_m.mean(),median_area_ratio=m.gt_area_ratio.median()))
pd.DataFrame(groups).to_csv(O/'MORPHOLOGY_EXCLUDING_GT_FIT.csv',index=False)
# Floor is provided by GT but has no verified semantic/date quality; descriptive only.
t.loc[t.reciprocal_match,['instance_id','best_gt_fid','gt_floor_reported','best_iou','position_status']].to_csv(O/'MATCHED_GT_FLOOR_DESCRIPTIVE.csv',index=False)
sp=pd.read_csv(O/'SPATIAL_2KM_METRICS.csv');valid=sp[sp.matched>=30].copy();valid.sort_values('mean_iou',ascending=False).to_csv(O/'SPATIAL_RANKING_MIN30.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(13,7))
for ax,col,title,vmin,vmax,cmap in [(axes[0],'mean_iou','GT agreement IoU (excludes direct GT fits)',0,1,'viridis'),(axes[1],'mean_iou_change','IoU change from RAW on same matches',-.2,.2,'RdBu')]:
 patches=[Rectangle((z.x,z.y),2000,2000) for z in valid.itertuples()];pc=PatchCollection(patches,cmap=cmap,edgecolor='gray',linewidth=.2);pc.set_array(valid[col].to_numpy());pc.set_clim(vmin,vmax);ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.ticklabel_format(style='plain');fig.colorbar(pc,ax=ax,shrink=.65)
fig.suptitle('2 km cells; >=30 reciprocal spatial matches; EPSG:32649; diagnostic, not independent accuracy');fig.tight_layout();fig.savefig(O/'SPATIAL_GT_AGREEMENT.png',dpi=160);plt.close(fig)
with rasterio.open(O/'GT_OVERLAY_CLASSES_1M.tif') as src:
 p=src.profile.copy();p.update(count=4,nodata=None,photometric='RGB');
 with rasterio.open(O/'GT_OVERLAY_ALPHA50.tif','w',**p) as dst:
  dst.colorinterp=(rasterio.enums.ColorInterp.red,rasterio.enums.ColorInterp.green,rasterio.enums.ColorInterp.blue,rasterio.enums.ColorInterp.alpha)
  for _,win in src.block_windows(1):
   z=src.read(1,window=win);a=np.zeros((4,*z.shape),'uint8')
   for code,color in [(1,(255,80,220)),(2,(255,210,0)),(3,(0,220,255))]:
    m=z==code
    for k,v in enumerate(color):a[k,m]=v
    a[3,m]=128
   dst.write(a,window=win)
  dst.update_tags(legend='1 magenta: prediction only (not certified false positive); 2 yellow: GT only; 3 cyan: overlap; alpha 128; background transparent',role='reference agreement diagnostic, not independent accuracy')
print(pd.DataFrame(groups).to_string(index=False));print('POSTPROCESS COMPLETE',flush=True)
