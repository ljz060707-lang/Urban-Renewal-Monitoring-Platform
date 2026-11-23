import json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,rasterio
R=Path(__file__).parent;I=R/'incoming';manifest=json.loads((I/'OUTPUT_DELIVERY_SHA256.json').read_text());errors=[]
for rel,expected in manifest.items():
 p=I/rel
 if not p.is_file():errors.append([rel,'MISSING']);continue
 if p.stat().st_size!=expected['bytes']:errors.append([rel,'SIZE']);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 if h.hexdigest()!=expected['sha256']:errors.append([rel,'SHA256'])
summary=json.loads((I/'AREA_SUMMARY.json').read_text());products=I/'products'
with rasterio.open(products/'ROOF_FRACTION_10M.tif') as ds:
 a=ds.read(1);coarse=ds.transform;assert str(ds.crs)=='EPSG:32649';assert (ds.width,ds.height)==(3885,3086)
with rasterio.open(products/'SOURCE_SUPPORT_10M.tif') as ds:sv=ds.read(1);assert ds.transform==coarse
assert np.isfinite(a).all() and np.all((a==-9999)|((a>=0)&(a<=1+1e-6)))
assert np.all(a[a>=0]<=sv[a>=0]+1e-6)
integral=float(a[a>=0].astype('float64').sum()*100);assert abs(integral-summary['within_support_union_m2'])<1
for name in ('ROOF_GOVERNED_1M.tif','ROOF_INSTANCE_ID_1M.tif','POSITION_STATUS_1M.tif'):
 with rasterio.open(products/name) as ds:assert str(ds.crs)=='EPSG:32649' and (ds.width,ds.height)==(38850,30860) and ds.transform.a==1 and ds.transform.e==-1
d=pd.read_csv(products/'OBJECT_AREA.csv');assert len(d)==209137 and d.instance_id.nunique()==209137 and (d.formal_weight==0).all()
assert abs(d.area_within_aoi_source_m2.sum()-summary['within_support_object_area_sum_m2'])<.01
result=dict(status='PASS' if not errors else 'FAILED',verified_files=len(manifest)-len(errors),total_files=len(manifest),total_bytes=sum(x['bytes'] for x in manifest.values()),errors=errors,objects=len(d),fraction_integral_m2=integral,integral_minus_union_m2=integral-summary['within_support_union_m2'],checked_local=time.strftime('%Y-%m-%d %H:%M:%S'),source='cloud final output SHA256 plus local raster and area readback')
(R/'LOCAL_DELIVERY_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
