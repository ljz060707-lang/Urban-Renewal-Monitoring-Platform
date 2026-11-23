import os,json,hashlib,time,threading,tarfile
os.environ['GDAL_PAM_ENABLED']='NO'
os.environ['GDAL_CACHEMAX']='256'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd,numpy as np,pandas as pd,rasterio,shapely
from rasterio.features import rasterize
from rasterio.windows import Window
from pyproj import Transformer
from shapely.ops import transform as tg
from shapely.affinity import translate
from PIL import Image
IN=Path('/root/xian2019/input');OUT=Path('/root/xian2019/output')
def save(name,d):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str));os.replace(t,p)
def status(stage,**kw):save('STATUS.json',dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(stage,kw,flush=True)
with tarfile.open(IN/'facade_data.tar') as tar:
 for entry in tar.getmembers():
  assert not entry.name.startswith('/') and '..' not in Path(entry.name).parts
 tar.extractall(IN)
raw=gpd.read_parquet(IN/'raw.parquet');gt=gpd.read_file(IN/'reference.gpkg',engine='pyogrio');gt=gt.set_index('original_fid');raw['source_row']=np.arange(len(raw))
eligible=raw[(raw.reference_link_code==1)&(~raw.source_edge_partial_risk)&((raw.native_xmax-raw.native_xmin)<500)&((raw.native_ymax-raw.native_ymin)<500)&raw.reference_match_fid.isin(gt.index)].copy()
cent=eligible.geometry.centroid
def bucket(x,y):return int(hashlib.sha256(f'{int(x//2000)},{int(y//2000)}'.encode()).hexdigest()[:8],16)%10
eligible['split']=np.array([0 if (h:=bucket(x,y))<6 else 1 if h<8 else 2 for x,y in zip(cent.x,cent.y)])
selected=[]
for split,cap in [(0,20000),(1,3000),(2,3000)]:selected.append(eligible[eligible.split==split].sample(frac=1,random_state=20261028).head(cap))
pairs=pd.concat(selected).reset_index(drop=True);pairs.to_parquet(IN/'training_pairs.parquet');n=len(pairs);assert all((pairs.split==i).sum()>100 for i in range(3))
x=np.lib.format.open_memmap(IN/'align_x.npy',mode='w+',dtype='uint8',shape=(n,4,256,256));y=np.lib.format.open_memmap(IN/'align_y.npy',mode='w+',dtype='uint8',shape=(n,256,256));shifts=np.zeros((n,2),'float32');meta=np.zeros((n,5),'float32');good=np.zeros(n,'bool')
tr=Transformer.from_crs(raw.crs,'EPSG:4326',always_xy=True);local=threading.local()
def item(i):
 if not hasattr(local,'src'):local.src=rasterio.open(IN/'source.tif')
 src=local.src;inv=~src.transform;row=pairs.iloc[i];g=row.geometry;b=gt.loc[row.reference_match_fid].geometry;c=g.centroid;bc=b.centroid
 def native(xx,yy,z=None):
  lon,lat=tr.transform(xx,yy);return inv.a*np.asarray(lon)+inv.c,inv.e*np.asarray(lat)+inv.f
 ng=tg(native,g);nx,ny=native(c.x,c.y);ox=int(np.clip(np.floor(nx-256),0,src.width-512));oy=int(np.clip(np.floor(ny-256),0,src.height-512));bb=ng.bounds
 if bb[0]<ox or bb[1]<oy or bb[2]>ox+512 or bb[3]>oy+512:return
 rgb=src.read(window=Window(ox,oy,512,512)).transpose(1,2,0);mask=rasterize([(ng,1)],out_shape=(512,512),transform=rasterio.Affine.translation(ox,oy),dtype='uint8');centered=translate(b,xoff=c.x-bc.x,yoff=c.y-bc.y)
 target=rasterize([(tg(native,centered),1)],out_shape=(512,512),transform=rasterio.Affine.translation(ox,oy),dtype='uint8')
 # This is reference-plane contour supervision, never a facade label or an unlabelled-background claim.
 x[i,:3]=np.asarray(Image.fromarray(rgb).resize((256,256),Image.Resampling.BILINEAR)).transpose(2,0,1);x[i,3]=np.asarray(Image.fromarray(mask).resize((256,256),Image.Resampling.NEAREST))*255;y[i]=np.asarray(Image.fromarray(target).resize((256,256),Image.Resampling.NEAREST));shifts[i]=[(bc.x-c.x)/10,(bc.y-c.y)/10];meta[i]=[(c.x-320000)/30000,(c.y-3800000)/30000,np.log1p(g.area)/10,min(float(row.aspect_ratio),10)/10,float(g.area)/20000];good[i]=True
with ThreadPoolExecutor(max_workers=12) as pool:
 for done,_ in enumerate(pool.map(item,range(n)),1):
  if done%500==0:status('PREPARE_TRAINING',done=done,total=n)
x.flush();y.flush();np.save(IN/'align_shift.npy',shifts);np.save(IN/'align_meta.npy',meta);splits=pairs.split.to_numpy('int8');splits[~good]=-1;np.save(IN/'align_split.npy',splits)
save('DATA_CONTRACT.json',dict(source_epoch='xa190101',nominal_year=2019,actual_dates=['2018-11-01','2018-11-08'],reference_role='USER_AUTHORIZED_TRAIN_AND_REFERENCE_CONSTRAINT',reference_capture_date='UNVERIFIED',objects=len(raw),supervised_pairs=n,complete_pairs=int(good.sum()),split_counts={str(i):int(((splits==i)&good).sum()) for i in range(3)},spatial_unit_m=2000,unlabelled_objects_are_background=False,facade_label_source='BANDON TRAIN/TUNE only, 2=projected geometric facade auxiliary, not exact visible facade GT',contour_label_source='matched reference polygon centered on source object; not facade GT',legacy_protected_reads=0))
status('PREPARE_COMPLETE',pairs=int(good.sum()))
