"""Window-level geometric diagnostics. The reference-guided arms are retrospective only."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyogrio
import rasterio
import torch
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
R8 = Path('/root/xian20261111_r8')
R16 = Path('/root/xian20261111_r16_roi_roof/output/xian_map_reference')
R19 = Path('/root/xian20261111_r19_loft/output/retained_roof_split/XIAN')
RADIUS_M = 16
ARMS = ('raw','shift','loft_split','loft_split_shift','reference_seed_split','reference_seed_split_shift')

def check(path, digest):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path

def save(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False))
    tmp.replace(path)

def input_rows():
    rows=[r for line in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))['role']=='TUNE']
    assert len(rows)==72 and len({r['site'] for r in rows})==24
    return rows

def ref_masks(row,support):
    path=Path(row['image_path']).parent/'PROVIDED_MAP_GT.gpkg'
    check(path,row['source_record']['files'][path.name]['sha256'])
    g=pyogrio.read_dataframe(path)
    g=g[g.fusion_role=='DESIGNATED_GT2019_ANCHOR'].to_crs(row['native_crs'])
    out={}
    for fid,geom in zip(g.original_gt_fid,g.geometry):
        m=rasterize([(geom,1)],out_shape=support.shape,transform=Affine(*row['native_affine']),dtype='uint8').astype(bool)&support
        if m.any(): out[int(fid)]=m
    return out

def object_list(labels):
    return [(i,sl,labels[sl]==i) for i,sl in enumerate(ndimage.find_objects(labels),1) if sl is not None]

def split_from_reference(raw,refs):
    result=raw.copy()
    last=int(result.max())
    associations=defaultdict(list)
    for fid,m in refs.items():
        v,c=np.unique(raw[m],return_counts=True)
        for i,n in zip(v,c):
            if i and n>=max(8,.10*m.sum()): associations[int(i)].append(fid)
    for q,sl,part in object_list(raw):
        fids=associations[q]
        if len(fids)<2:continue
        # Each child receives pixels closest to a distinct intersecting reference.
        seed=np.zeros(part.shape,np.int32)
        for j,f in enumerate(fids,1): seed[(refs[f][sl]&part)&(seed==0)]=j
        active=np.unique(seed[seed>0])
        if len(active)<2:continue
        near=ndimage.distance_transform_edt(seed==0,return_distances=False,return_indices=True)
        child=seed[tuple(near)]
        for j in active[1:]:
            last+=1
            result[sl][part&(child==j)]=last
    assert np.array_equal(raw>0,result>0)
    return result

def overlap_surface(a,b):
    # Linear cross-correlation on GPU; round intersections back to integer pixel counts.
    h=a.shape[0]+b.shape[0]-1; w=a.shape[1]+b.shape[1]-1
    sh=(cv2.getOptimalDFTSize(h),cv2.getOptimalDFTSize(w))
    aa=torch.as_tensor(a.copy(),device='cuda',dtype=torch.float32)
    bb=torch.as_tensor(b[::-1,::-1].copy(),device='cuda',dtype=torch.float32)
    z=torch.fft.irfft2(torch.fft.rfft2(aa,s=sh)*torch.fft.rfft2(bb,s=sh),s=sh)
    return np.rint(z[:h,:w].cpu().numpy()).astype(np.int32)

def pair_scores(gsl,g,pids,dxm,dym):
    base={}; shifted={}
    ga=int(g.sum())
    for pid,psl,p in pids:
        pa=int(p.sum())
        if min(ga,pa)*2<max(ga,pa):continue
        gy,gx=gsl[0].start,gsl[1].start
        py,px=psl[0].start,psl[1].start
        if max(0,gy-psl[0].stop,py-gsl[0].stop)*dym>RADIUS_M or max(0,gx-psl[1].stop,px-gsl[1].stop)*dxm>RADIUS_M:continue
        z=overlap_surface(g,p)
        oy=py-gy+p.shape[0]-1; ox=px-gx+p.shape[1]-1
        ry=int(RADIUS_M/dym); rx=int(RADIUS_M/dxm)
        y0=max(0,oy-ry);y1=min(len(z),oy+ry+1)
        x0=max(0,ox-rx);x1=min(z.shape[1],ox+rx+1)
        if y1<=y0 or x1<=x0:continue
        dy,dx=np.ogrid[y0-oy:y1-oy,x0-ox:x1-ox]
        valid=(dy*dym)**2+(dx*dxm)**2<=RADIUS_M**2
        if not valid.any():continue
        patch=np.where(valid,z[y0:y1,x0:x1],-1)
        n=patch.max()
        if 3*n>=ga+pa:
            ys,xs=np.where(patch==n)
            ys+=y0-oy;xs+=x0-ox
            k=int(np.argmin((ys*dym)**2+(xs*dxm)**2))
            shifted[pid]=(float(n/(ga+pa-n)),int(xs[k]),int(ys[k]))
        n=z[oy,ox] if 0<=oy<z.shape[0] and 0<=ox<z.shape[1] else 0
        if 3*n>=ga+pa:base[pid]=(float(n/(ga+pa-n)),0,0)
    return base,shifted

def grid(row):
    a=Affine(*row['native_affine']);tr=Transformer.from_crs(row['native_crs'],32649,always_xy=True)
    def xy(x,y):return np.array(tr.transform(*(a*(x,y))))
    return np.linalg.norm(xy(513,512)-xy(512,512)),np.linalg.norm(xy(512,513)-xy(512,512))

def max_matches(edges):
    ids=sorted({q for e in edges for q in e})
    if not ids:return 0
    index={q:j for j,q in enumerate(ids)}
    cost=np.zeros((len(edges),len(ids)))
    for i,e in enumerate(edges):
        for q,(iou,_,_) in e.items():cost[i,index[q]]=1+iou/(1+min(cost.shape))
    ii,jj=linear_sum_assignment(-cost)
    return int((cost[ii,jj]>0).sum())

def diagnose(r,old,splitrec):
    check(r['query_cache_path'],r['query_cache_sha256']);check(splitrec['path'],splitrec['sha256'])
    with np.load(r['query_cache_path']) as f: raw=f['instances'].astype('int32');support=f['support'].astype(bool)
    raw[~support]=0
    with np.load(splitrec['path']) as f: loft=f['SPLIT'].astype('int32')
    assert np.array_equal(loft>0,raw>0)
    refs=ref_masks(r,support)
    historic={i['fid']:i for i in old['details']['RAW']}
    assert set(refs)==set(historic)
    maps=[object_list(v) for v in (raw,loft,split_from_reference(raw,refs))]
    dx,dy=grid(r)
    details=[];all_edges=defaultdict(list);cohort_edges=defaultdict(list)
    for fid,m in refs.items():
        base=historic[fid]
        focus=2*base['covered_pixels']>=base['supported_reference_pixels'] and base['best_IoU']<.5
        sl=ndimage.find_objects(m.astype(np.uint8))[0]
        matches=[]
        for objs in maps:matches.extend(pair_scores(sl,m[sl],objs,dx,dy))
        assert bool(matches[0])==(base['best_IoU']>=.5),(r['id'],fid)
        row=dict(window=r['id'],epoch=r['epoch'],site=r['site'],fid=fid,edge=base['window_edge'],focus=focus)
        for arm,e in zip(ARMS,matches):
            all_edges[arm].append(e)
            if focus:cohort_edges[arm].append(e)
            row[arm]=bool(e)
            if e:
                q=max(e,key=lambda q:e[q][0]);v=e[q]
                row[arm+'_best']=[q,*v]
        details.append(row)
    return dict(window=r['id'],epoch=r['epoch'],details=details,all_unique={a:max_matches(all_edges[a]) for a in ARMS},focus_unique={a:max_matches(cohort_edges[a]) for a in ARMS})

def main():
    # Validate the shift sign and exact intersections against brute force.
    a=np.zeros((8,7),bool);b=np.zeros((6,5),bool)
    a[3:6,2:4]=True;b[1:4,1:3]=True
    z=overlap_surface(a,b);assert z[2+b.shape[0]-1,1+b.shape[1]-1]==6
    from scipy.signal import correlate
    assert np.array_equal(z,correlate(a.astype(int),b.astype(int),method='direct'))
    base={w['window']:w for w in json.loads((R16/'BASELINE_OBJECTS.json').read_text())}
    split=json.loads((R19/'PREDICTIONS_FROZEN.json').read_text())['S0.5_C0.35']
    registry={s['frame_id']:s for s in split}
    results=[]
    for r in input_rows():
        dest=HERE/'windows'/(r['id']+'.json');dest.parent.mkdir(exist_ok=True)
        if not dest.exists():save(dest,diagnose(r,base[r['id']],registry[r['id']]))
        results.append(json.loads(dest.read_text()))
        print(len(results),r['id'],flush=True)
    summary={}
    for ep,n in zip(sorted({r['epoch'] for r in results}),(644,628,421)):
        group=[w for w in results if w['epoch']==ep]
        focus=[r for w in group for r in w['details'] if r['focus']]
        assert len(focus)==n
        summary[ep]=dict(n=n,edge_n=sum(r['edge'] for r in focus),individual={a:sum(r[a] for r in focus) for a in ARMS},
          interior={a:sum(r[a] and not r['edge'] for r in focus) for a in ARMS},
          unique_focus={a:sum(w['focus_unique'][a] for w in group) for a in ARMS},
          unique_all={a:sum(w['all_unique'][a] for w in group) for a in ARMS})
    save(HERE/'DIAGNOSIS.json',summary)

if __name__=='__main__':main()

