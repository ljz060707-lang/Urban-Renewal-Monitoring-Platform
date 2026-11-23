"""Estimate cross-date translations using imagery only, on candidate-free texture.
Fit checkerboard geographic tiles and validate on disjoint tiles before application.
"""
import json,hashlib,sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer

HERE=Path(__file__).resolve().parent
R8=Path('/root/xian20261111_r8')
def verify(path,digest):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
def data(row):
    verify(row['image_path'],row['image_sha256'])
    verify(row['query_cache_path'],row['query_cache_sha256'])
    with rasterio.open(row['image_path']) as d:
        gray=cv2.cvtColor(d.read([1,2,3]).transpose(1,2,0),cv2.COLOR_RGB2GRAY)
        assert d.transform==Affine(*row['native_affine'])
    with np.load(row['query_cache_path']) as z:
        mask=cv2.erode(((z['instances']==0)&(z['support']>0)).astype('uint8'),np.ones((25,25),np.uint8))
    mask[:16,:]=0;mask[-16:,:]=0;mask[:,:16]=0;mask[:,-16:]=0
    k,d=cv2.SIFT_create(nfeatures=6000).detectAndCompute(gray,mask*255)
    return np.array([x.pt for x in k]),d
def ground(args):
    cv2.setNumThreads(1)
    rows=sorted(args,key=lambda r:r['epoch'])
    obs=[data(r) for r in rows]
    pairs=[]
    for i,j in [(0,1),(1,2),(0,2)]:
        a,b=rows[i],rows[j]
        assert np.allclose(a['native_affine'],b['native_affine'],rtol=0,atol=1e-10)
        votes=[list(r['source_record']['actual_date_pixel_votes']) for r in (a,b)]
        rec=dict(anchor=a['id'],moving=b['id'],dates=votes,eligible=False,dx_m=0.,dy_m=0.,site=a['site'])
        pairs.append(rec)
        if any(len(v)!=1 for v in votes) or votes[0]==votes[1]:
            rec['reason']='mixed_or_same_acquisition';continue
        pa,da=obs[i];pb,db=obs[j]
        if da is None or db is None or min(len(da),len(db))<2:
            rec['reason']='insufficient_descriptors';continue
        matcher=cv2.BFMatcher()
        def good(x,y):
            return {m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(x,y,k=2) if m.distance<.72*n.distance}
        ab=good(da,db);ba=good(db,da)
        indices=[(k,v) for k,v in ab.items() if ba.get(v)==k and np.linalg.norm(pa[k]-pb[v])<=80]
        if not indices:
            rec['reason']='no_mutual_near_matches';continue
        # Keep one independent keypoint per 16 px neighborhood.
        occupied=set();held=[];fit=[]
        for k,v in indices:
            tile=tuple((pb[v]//16).astype(int))
            if tile in occupied:continue
            occupied.add(tile)
            block=(pb[v]//128).astype(int)
            ((fit if block.sum()%2==0 else held)).append((pa[k]-pb[v],tuple(block)))
        rec.update(nfit=len(fit),nheld=len(held),held_tiles=len(set(tile for _,tile in held)))
        if min(len(fit),len(held))<8 or min(len(set(tile for _,tile in side)) for side in (fit,held))<3:
            rec['reason']='insufficient_spatially_independent_points';continue
        af=Affine(*a['native_affine'])
        proj=Transformer.from_crs(a['native_crs'],32649,always_xy=True)
        def world(x,y):return np.array(proj.transform(*(af*(x,y))))
        basis=np.column_stack([world(513,512)-world(512,512),world(512,513)-world(512,512)])
        f=np.array([v for v,_ in fit])@basis.T
        h=np.array([v for v,_ in held])@basis.T
        d=np.median(f,axis=0)
        old=np.linalg.norm(h,axis=1);new=np.linalg.norm(h-d,axis=1)
        rec.update(fit_vector_m=d.tolist(),held_median_raw=float(np.median(old)),held_median_after=float(np.median(new)),
          held_p95_raw=float(np.quantile(old,.95)),held_p95_after=float(np.quantile(new,.95)))
        ok=(np.linalg.norm(d)<=16 and np.median(old)-np.median(new)>=.25 and np.quantile(new,.95)<=np.quantile(old,.95)+.10)
        rec.update(reason='accepted' if ok else 'held_texture_gate',eligible=bool(ok))
        if ok:rec.update(dx_m=float(d[0]),dy_m=float(d[1]))
    return pairs
def main():
    rows=defaultdict(list)
    for line in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['role']=='TUNE':rows[r['site']].append(r)
    assert len(rows)==24
    with ProcessPoolExecutor(6) as pool:
        result=[r for batch in pool.map(ground,rows.values()) for r in batch]
    assert len(result)==72
    (HERE/'GROUND_IMAGE_ONLY_FROZEN.json').write_text(json.dumps(result,allow_nan=False))
if __name__=='__main__': main()

