from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from prepare_native_graph import worker,P,O,sha,save
records=[json.loads(l) for l in (P/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if l]
r=next(r for r in records if r['role']=='TRAIN')
q=worker(r);d=O/'frames'/r['id'];obs=json.loads((d/'OBSERVATIONS.json').read_text());f=np.load(d/'FEATURES.npz')['feature64'];z=np.load(r['query_cache_path']);lab=z['instances'];features=z['features'].astype(np.float32);errors=[]
for i in sorted(set([0,len(obs)//2,len(obs)-1])):
 yy,xx=np.where(lab==obs[i]['mask_id']);direct=features[:,yy//4,xx//4].mean(1,dtype=np.float64);error=float(np.max(np.abs(direct-f[i])));assert error<2e-6;errors.append(error)
assert all(np.isfinite(f).ravel()) and all(x['source_support']>=0 and x['source_support']<=1 for x in obs)
save(O/'NATIVE_POOL_EQUIVALENCE.json',dict(status='PASS_SPARSE_FEATURE_POOL_EQUALS_DIRECT_NATIVE_PIXEL_MEAN',frame=r['id'],queries=len(obs),checked_queries=len(errors),maximum_error=max(errors),source_cache_sha256=q['query_cache_sha256'],GT_reads=0,RGB_reads=0,GPU_forwards=0,optimizer_updates=0));print('PASS_NATIVE_POOL',len(obs),max(errors),flush=True)
