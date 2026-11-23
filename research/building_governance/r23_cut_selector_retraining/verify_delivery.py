from pathlib import Path
import json,hashlib,tarfile,csv
import numpy as np,rasterio
p=Path('F:/xian/OPTIMIZE_MODEL_20261111_R23_FINAL_01')
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(4<<20),b''):h.update(b)
 return h.hexdigest()
bundles={}
for n in ['FINAL.tgz','TRAIN_CACHE.tar','INPUTS.tar']:
 f=p/n
 assert sha(f)==(p/(n.split('.')[0]+'.sha256')).read_text().split()[0],n
 bundles[n]={'bytes':f.stat().st_size,'sha256':sha(f)}
with tarfile.open(p/'FINAL.tgz') as t:
 for m in t:
  assert (p/m.name).resolve().is_relative_to(p.resolve())
 t.extractall(p)
w=list(csv.DictReader((p/'WINDOWS.csv').open()))
for r in w:
 f=p/'position'/(r['window']+'.tif');assert sha(f)==r['sha256']
 with rasterio.open(f) as ds:
  s,a,b,c=ds.read();assert ds.count==4 and np.array_equal(b>0,c>0) and np.array_equal(a>0,c>0)
assert len(w)==72
assert sha(p/'SELECTOR.pt')==json.loads((p/'SELECTOR_FROZEN.json').read_text())['sha256']
(p/'LOCAL_READBACK.json').write_text(json.dumps({'verified_archives':bundles,'TIF_count':72,'union_changed_pixels':0,'weights_verified':True,'scope':'R21 R22 R23 training crops, all alternatives and features, compact raw instance planes, allowed source roofs, evaluation vectors and governed masks, frozen date field, scripts and environment. Original local imagery retained in existing project directories.'},indent=2))
print('VERIFIED',sum(v['bytes'] for v in bundles.values()),'bytes',len(w),'TIF')
