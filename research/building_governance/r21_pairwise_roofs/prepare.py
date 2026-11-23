"""Freeze image-only within-parent regions, then join allowed native/map identities."""
import hashlib,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import cv2,numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
import rasterio
from rasterio.features import rasterize
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Polygon
import geopandas as gpd
from pycocotools import mask as cm
from PIL import Image
R=Path(__file__).resolve().parent
R8=Path('/root/xian20261111_r8')
R17=Path('/root/xian20261111_r17_proposals')
OUT=R/'data';OUT.mkdir(exist_ok=True)
N=64
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(p,h):assert digest(p)==h,p
def save(p,data):
    p=Path(p);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,allow_nan=False));temp.replace(p)
def rows():
    return {r['id']:r for l in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(l))}
def center(r):
    a=Affine(*r['native_affine'])
    return np.array(Transformer.from_crs(r['native_crs'],32649,always_xy=True).transform(*(a*(512,512))))
def partition(group):return (group[0]*37+group[1]*13)%7==0
def select(records):
    src=json.loads((R17/'data/READY.json').read_text())['rows']
    key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()
    fit=sorted((s for s in src if s['role']=='FIT'),key=key)[:384]
    tune=[r for r in records.values() if r['role']=='TUNE']
    ref=np.array([center(r) for r in tune])
    tr=[r for r in records.values() if r['role']=='TRAIN' and r['epoch']=='xa201231' and r['sampling_stratum']=='DESIGNATED_GT2019_ANCHOR' and np.linalg.norm(center(r)-ref,axis=1).min()>=650]
    inner=sorted((r for r in tr if partition(r['spatial_group'])),key=key)[:36]
    iv=np.array([center(r) for r in inner])
    target=sorted((r for r in tr if not partition(r['spatial_group']) and np.linalg.norm(center(r)-iv,axis=1).min()>=650),key=key)[:160]
    assert inner and target and len(tune)==72
    pairs=[(r['id'],'SOURCE_FIT',r) for r in fit]
    pairs.extend((s['id'],'SOURCE_INNER',s) for s in src if s['role']=='INNER')
    pairs.extend((r['id'],'MAP_FIT',None) for r in target)
    pairs.extend((r['id'],'MAP_INNER',None) for r in inner)
    pairs.extend((r['id'],'JINAN',None) for r in records.values() if r['role']=='SOURCE_TUNE')
    pairs.extend((r['id'],'XIAN',None) for r in tune)
    return [(records[k],role,t) for k,role,t in pairs]
def region_image(im,m):
    inside=cv2.distanceTransform(np.pad(m.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]
    blur=cv2.GaussianBlur(im,(3,3),.8)
    gx=cv2.Sobel(blur,cv2.CV_32F,1,0);gy=cv2.Sobel(blur,cv2.CV_32F,0,1)
    grad=np.sqrt((gx*gx+gy*gy).mean(2))*.1
    seeds=np.zeros((N,N),np.int32);i=0
    for y in range(0,N,8):
        for x in range(0,N,8):
            sl=np.s_[y:y+8,x:x+8];p=m[sl]
            if p.sum()<6:continue
            yy,xx=np.indices(p.shape)
            score=np.minimum(inside[sl],3)-.16*((yy-3.5)**2+(xx-3.5)**2)-.05*np.minimum(grad[sl],40)
            idx=np.argmax(np.where(p,score,-1e6))
            i+=1;seeds[sl].flat[idx]=i
    if i<2:return np.zeros_like(m,dtype='uint8')
    labs=watershed(grad,seeds,mask=m,compactness=.9,connectivity=np.ones((3,3))).astype('uint8')
    return labs
def frozen_image(arg):
    r,role,_=arg;cv2.setNumThreads(1)
    p=OUT/(r['id']+'.npz')
    verify(r['image_path'],r['image_sha256']);verify(r['query_cache_path'],r['query_cache_sha256'])
    if r['role'] in ('TRAIN','TUNE'):
        with rasterio.open(r['image_path']) as d:im=d.read([1,2,3]).transpose(1,2,0)
    else:im=np.asarray(Image.open(r['image_path']).convert('RGB'))
    with np.load(r['query_cache_path']) as f:raw=f['instances'].astype('int32')
    boxes=[];images=[];segs=[]
    for q,sl in enumerate(ndimage.find_objects(raw),1):
        if not sl:continue
        y,x=sl;mask=raw[sl]==q
        if min(mask.shape)<8 or mask.sum()<192:continue
        crop=cv2.resize(im[sl],(N,N),interpolation=cv2.INTER_AREA)
        m=cv2.resize(mask.astype('uint8'),(N,N),interpolation=cv2.INTER_NEAREST)>0
        lab=region_image(crop,m)
        if lab.max()<2:continue
        boxes.append((q,y.start,y.stop,x.start,x.stop));images.append(np.dstack((crop,m.astype('uint8')*255)));segs.append(lab)
    np.savez_compressed(p,boxes=np.asarray(boxes,np.int32).reshape(-1,5),imgs=np.asarray(images,np.uint8).reshape(-1,N,N,4),nodes=np.asarray(segs,np.uint8).reshape(-1,N,N))
    return dict(id=r['id'],role=role,sha=digest(p),parents=len(boxes))
def init_map():
    global MAPS,TREE,LEGACY
    mapfile=R8/'data/GT_MAP_LIBRARY.parquet'
    MAPS=gpd.read_parquet(mapfile,columns=['geometry','original_gt_fid','fusion_role'])
    MAPS=MAPS[MAPS.fusion_role=='DESIGNATED_GT2019_ANCHOR'].reset_index(drop=True)
    TREE=MAPS.sindex
    legacy=json.loads(Path('/root/xian20261110_gt/GT_QUERY_MANIFEST.json').read_text())
    LEGACY={r['reference_id'] for r in legacy['samples'] if r['role']=='TUNE'}
def source_ids(t):
    verify(t['targets'],t['targets_sha256'])
    with np.load(t['targets']) as f:counts=f['gt_counts']
    ids=np.zeros((1024,1024),np.int32)
    for j,enc in enumerate(counts,1):
        m=cm.decode(dict(size=[1024,1024],counts=bytes(enc))).astype(bool)
        ids[m]=np.where(ids[m]==0,j,-1)
    return ids
def map_ids(r,role):
    a=Affine(*r['native_affine']);proj=Transformer.from_crs(r['native_crs'],32649,always_xy=True)
    roi=Polygon([proj.transform(*(a*p)) for p in ((0,0),(1024,0),(1024,1024),(0,1024))])
    geoms=MAPS.iloc[TREE.query(roi,predicate='intersects')]
    pairs=[]
    for v in geoms.itertuples():
        g=v.geometry;fid=int(v.original_gt_fid);grp=(int(g.centroid.x//2000),int(g.centroid.y//2000))
        if not roi.contains(g) or (grp[0]*31+grp[1])%5==0 or 'GT_'+str(fid) in LEGACY or partition(grp)!=(role=='MAP_INNER'):continue
        core=g.buffer(-2.)
        if not core.is_empty:pairs.append((core,fid+1))
    ids=np.zeros((1024,1024),np.int32)
    if pairs:
        gs=gpd.GeoSeries([g for g,_ in pairs],crs=32649).to_crs(r['native_crs'])
        from rasterio.enums import MergeAlg
        count=rasterize([(g,1) for g in gs],out_shape=ids.shape,transform=a,merge_alg=MergeAlg.add,dtype='uint16')
        ids=rasterize([(g,i) for g,(_,i) in zip(gs,pairs)],out_shape=ids.shape,transform=a,dtype='int32')
        ids[count!=1]=0
    verify(r['date_path'],r['files']['ACTUAL_DATE_MAP.tif'])
    with rasterio.open(r['date_path']) as d:dates=d.read(1)
    ids[~np.isin(dates,[1,2])]=0
    return ids
def join(arg):
    r,role,t=arg;cv2.setNumThreads(1)
    ids=source_ids(t) if t is not None else map_ids(r,role)
    p=OUT/(r['id']+'.npz')
    with np.load(p) as d:boxes=d['boxes'];nodes=d['nodes']
    targets=np.zeros((len(boxes),65),np.int32)
    for k,((q,y0,y1,x0,x1),lab) in enumerate(zip(boxes,nodes)):
        # nearest sampling in the same ROI grid used for generating nodes
        patch=cv2.resize(ids[y0:y1,x0:x1],(N,N),interpolation=cv2.INTER_NEAREST)
        for j in range(1,int(lab.max())+1):
            vals=patch[lab==j];known=vals[vals>0]
            if len(known)<8:continue
            labels,n=np.unique(known,return_counts=True)
            best=n.argmax()
            if n[best]>=.80*len(vals):targets[k,j]=labels[best]
    dest=OUT/(r['id']+'.labels.npy')
    np.save(dest,targets)
    valid=targets>0
    distinct=[len(set(a[a>0])) for a in targets]
    return dict(id=r['id'],role=role,known_nodes=int(valid.sum()),same_or_different_parents=sum(v>=1 for v in distinct),multi=sum(v>=2 for v in distinct),sha=digest(dest))
def main():
    work=select(rows())
    protocol=dict(inputs=[dict(id=r['id'],role=role) for r,role,_ in work],source='visible roof identity inside frozen predicted parents',map='only spatially separated TRAIN; non-overlapping interiors of manual GT eroded 2 m; only 2019-12-02/10 date pixels; geometry-assisted weak labels',node='64x64 normalized parent, image-gradient watershed seeds on 8x8 grid; >=80% of node jointly supported by same known identity',seed=211111)
    save(R/'PROTOCOL.json',protocol)
    with ProcessPoolExecutor(12) as pool:frames=list(pool.map(frozen_image,work,chunksize=2))
    save(R/'IMAGE_ONLY_FROZEN.json',frames)
    print('image-only',Counter(x['role'] for x in frames),flush=True)
    src=[v for v in work if v[1].startswith('SOURCE')]
    maps=[v for v in work if v[1].startswith('MAP')]
    with ProcessPoolExecutor(10) as pool:labels=list(pool.map(join,src,chunksize=2))
    with ProcessPoolExecutor(5,initializer=init_map) as pool:labels+=list(pool.map(join,maps,chunksize=2))
    save(R/'SUPERVISION.json',labels)
    print('supervised',Counter(x['role'] for x in labels),flush=True)
if __name__=='__main__':main()

