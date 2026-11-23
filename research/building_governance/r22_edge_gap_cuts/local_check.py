import csv,hashlib,json,tarfile
from pathlib import Path
import rasterio,numpy as np
p=Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
archive=p/'RETURN.tgz';h=sha(archive);assert h==(p/'RETURN.sha256').read_text().split()[0]
with tarfile.open(archive) as t:
    for m in t.getmembers():
        assert (p/m.name).resolve().is_relative_to(p.resolve()) and (m.isfile() or m.isdir())
    t.extractall(p)
checks={}
for dest in [p,p/'contour']:
    ref=list(csv.DictReader((dest/'REFERENCE_COMPARISON.csv').open()));wins=list(csv.DictReader((dest/'WINDOWS.csv').open()))
    mf=json.loads((dest/'PREDICTIONS_FROZEN_BEFORE_MAP.json').read_text())
    count=0
    for w in wins:
        f=dest/'position'/(w['window']+'.tif');assert sha(f)==w['sha256']
        with rasterio.open(f) as ds:
            sup,raw,prev,new=ds.read()
            assert ds.crs.to_epsg()==32649 and ds.res==(1,1) and ds.count==4 and np.array_equal(prev>0,new>0) and np.array_equal(raw>0,new>0)
        assert sha(dest/'predictions'/(w['window']+'.npz'))==mf[w['window']]['sha256']
        count+=1
    assert sha(dest/'SELECTOR.pt')==json.loads((dest/'SELECTOR_FROZEN.json').read_text())['sha256']
    assert len(ref)==2994 and count==72
    checks[dest.name]=dict(references=len(ref),windows=count,weight_sha=sha(dest/'SELECTOR.pt'),pixel_union_diff=0)
mani=json.loads((p/'CANDIDATES_FROZEN.json').read_text())
target=[r for r in mani if r['role']=='XIAN']
assert len(target)==72
for r in target:assert sha(p/'data'/(r['id']+'.npz'))==r['sha']
result=dict(archive_sha=h,verified=checks,windows_all_roles=len(mani),xian_cut_candidates=sum(r['n'] for r in target),xian_parents=sum(r['parents'] for r in target))
(p/'LOCAL_READBACK.json').write_text(json.dumps(result,indent=2));print(result)

