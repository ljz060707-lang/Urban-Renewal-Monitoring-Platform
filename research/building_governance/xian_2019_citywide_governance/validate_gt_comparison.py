import os,json,hashlib
os.environ['GDAL_PAM_ENABLED']='NO'
from pathlib import Path
import numpy as np,rasterio,pandas as pd
from rasterio.windows import Window
R=Path(__file__).parent;O=R.parent/'XIAN_R5_2019_GOVERNANCE_REPAIR_20261029_01/GT_COMPARISON_20261029'
with rasterio.open(O/'GT_OVERLAY_CLASSES_1M.tif') as s,rasterio.open(O/'GT_OVERLAY_ALPHA50.tif') as a,rasterio.open(R.parent/'XIAN_R5_SINGLE_DATE_20261028_01/single_date/2018-11-01/ROOF_MANAGED.tif') as raw:
 assert a.shape==s.shape and a.transform==s.transform and a.crs==s.crs
 assert raw.shape==s.shape and raw.transform==s.transform and raw.crs==s.crs
 hist=np.zeros(4,dtype=np.int64);before=np.zeros(4,dtype=np.int64)
 for y in range(0,s.height,512):
  w=Window(0,y,s.width,min(512,s.height-y));z=s.read(1,window=w);valid=z!=255;hist+=np.bincount(z[valid],minlength=4);g=(z==2)|(z==3);r=raw.read(1,window=w);before+=np.bincount(((r==1).astype('uint8')+2*g)[valid],minlength=4)
 rng=np.random.default_rng(2019)
 for k in range(64):
  w=Window(int(rng.integers(0,s.width-128)),int(rng.integers(0,s.height-128)),128,128);z=s.read(1,window=w);q=a.read(window=w);assert np.array_equal(q[3],np.where((z>=1)&(z<=3),128,0).astype('uint8'))
  for code,col in [(1,(255,80,220)),(2,(255,210,0)),(3,(0,220,255))]:
   for i,v in enumerate(col):assert (q[i][z==code]==v).all()
summary=json.loads((O/'SUMMARY.json').read_text(encoding='utf-8'));assert int(hist[3])==summary['overall_pixel']['both_m2'];t=pd.read_csv(O/'OBJECT_GT_COMPARISON.csv');assert len(t)==209137 and t.instance_id.nunique()==209137
def px(a):
 _,p,g,b=map(int,a);return dict(both_m2=b,prediction_only_m2=p,gt_only_m2=g,gt_coverage_recall=b/(b+g),reference_iou=b/(b+p+g),reference_dice=2*b/(2*b+p+g))
summary['raw_overall_pixel_same_support']=px(before);(O/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
hashes={}
for p in O.iterdir():
 if p.is_file() and p.name!='VALIDATION_RECEIPT.json':
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  hashes[p.name]=dict(size=p.stat().st_size,sha256=h.hexdigest())
report=dict(status='PASS',objects_unique=209137,raster_class_counts=hist.tolist(),rgba_random_windows_verified=64,aligned_raw_product=True,raw_overall_pixel=px(before),governed_overall_pixel=summary['overall_pixel'],files=hashes)
(O/'VALIDATION_RECEIPT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
