"""Image-supported map hints. Raw roof pixels are retained; external hints never create a roof."""
import sys,json
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np,cv2,torch
import pyogrio
from affine import Affine
from shapely import from_wkb,STRtree
from shapely.ops import transform
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from prepare import R,OUT,rows,verify,save,digest
from evaluate_pairs import scores,evaluate,summarize
from guard import model_for_split,raw_input
E=R/'external';E.mkdir(exist_ok=True)
METHODS=['centers','centers_and_affinity']
def unpack(x):return [from_wkb(bytes.fromhex(wkb)) for _,wkb in x]
def seeds(r,raw,record):
    a=Affine(*r['native_affine']);inv=~a
    proj=Transformer.from_crs(32649,r['native_crs'],always_xy=True)
    def pixel(x,y,z=None):
        xx,yy=proj.transform(x,y);return inv*(np.array(xx),np.array(yy))
    ga=unpack(record['a'])
    groups=defaultdict(list);slices=ndimage.find_objects(raw)
    for i,j in record['pairs']:
        g=transform(pixel,ga[i]);x0,y0,x1,y1=g.bounds
        if not (0<=x0<x1<1024 and 0<=y0<y1<1024):continue
        x0,y0=int(x0),int(y0);x1,y1=int(np.ceil(x1)),int(np.ceil(y1))
        mask=rasterize([(g,1)],out_shape=(y1-y0,x1-x0),transform=Affine.translation(x0,y0),dtype='uint8')>0
        v,n=np.unique(raw[y0:y1,x0:x1][mask],return_counts=True);valid=v>0
        if not valid.any():continue
        k=np.argmax(n[valid]);q=int(v[valid][k]);area=int(n[valid][k])
        if area<max(55,.40*mask.sum()):continue
        inside=mask&(raw[y0:y1,x0:x1]==q);ys,xs=np.where(inside)
        point=g.representative_point()
        z=np.argmin((ys+y0+.5-point.y)**2+(xs+x0+.5-point.x)**2)
        sy,sx=int(ys[z]+y0),int(xs[z]+x0)
        if np.hypot(sx+.5-point.x,sy+.5-point.y)>12:continue
        lj=slices[q-1]
        groups[q].append(dict(point=(sy,sx),covered=(ys+y0-lj[0].start,xs+x0-lj[1].start),fid=record['a'][i][0]))
    return groups
def infer(r,record,model):
    raw=raw_input(r);groups=seeds(r,raw,record);boxes,nodes,dist=scores(model,r)
    byid={int(b[0]):i for i,b in enumerate(boxes)}
    outputs={k:raw.copy() for k in METHODS};nxt={k:int(raw.max()) for k in METHODS}
    sls=ndimage.find_objects(raw);counter=Counter()
    for q,hints in groups.items():
        if len(hints)<2 or q not in byid:continue
        sl=sls[q-1];m=raw[sl]==q;ys,xs=np.where(m)
        coverage=np.zeros(m.shape,bool)
        for hint in hints:coverage[hint['covered']]=True
        if coverage.sum()<.40*m.sum():continue
        k=byid[q];n=nodes[k];ds=dist[k]
        py=np.clip(ys*64//m.shape[0],0,63);px=np.clip(xs*64//m.shape[1],0,63)
        ny=np.array([v['point'][0]-sl[0].start for v in hints]);nx=np.array([v['point'][1]-sl[1].start for v in hints])
        separation=np.hypot(ny[:,None]-ny,nx[:,None]-nx)
        np.fill_diagonal(separation,np.inf)
        if separation.min()<12:continue
        spatial=np.hypot(ys[:,None]-ny,xs[:,None]-nx)
        group=np.argmin(spatial,1);native_nodes=n[py,px]
        support=[]
        for y,x in zip(ny,nx):
            radius=min(10.,max(3.,.12*np.sqrt(m.sum()/len(hints))))
            idx=np.unique(native_nodes[((ys-y)**2+(xs-x)**2<=radius**2)&(native_nodes>0)])
            support.append(idx)
        eligible=all(len(v) for v in support)
        affinity=None
        if eligible:
            scores_between=[np.median(ds[np.ix_(v,u)]) for i,v in enumerate(support) for u in support[i+1:]]
            eligible=min(scores_between)>=.5
            if eligible:
                penalty=np.stack([ds[:,v].mean(1) for v in support],1)[native_nodes]
                norm=np.maximum(12,np.min(separation,1))
                affinity=(spatial/norm[None,:]+.8*penalty).argmin(1)
        for name,g in zip(METHODS,(group,affinity)):
            if g is None:continue
            count=np.bincount(g,minlength=len(hints))
            if min(count)<max(48,.03*m.sum()):continue
            keep=count.argmax()
            for i in range(len(count)):
                if i!=keep:
                    nxt[name]+=1
                    outputs[name][sl][ys[g==i],xs[g==i]]=nxt[name]
            counter[name]+=1
    for lab in outputs.values():assert np.array_equal(lab>0,raw>0)
    return outputs,dict(counter)
def main():
    config=json.loads((R/'EXTERNAL_MAP_PROTOCOL.json').read_text())
    verify(R/'EXTERNAL_MAP_INPUTS.json',config['sha256'])
    ext=json.loads((R/'EXTERNAL_MAP_INPUTS.json').read_text())
    target=[r for r in rows().values() if r['role']=='TUNE'];model=model_for_split()
    manifests={a:{} for a in METHODS};counts={}
    for r in target:
        output,counts[r['id']]=infer(r,ext[r['site']],model)
        for name,lab in output.items():
            folder=E/name;folder.mkdir(exist_ok=True);p=folder/(r['id']+'.npz')
            np.savez_compressed(p,SPLIT=lab);manifests[name][r['id']]=dict(path=str(p),variant='SPLIT',sha256=digest(p))
    save(E/'MAP_AND_IMAGE_PREDICTIONS_FROZEN.json',manifests)
    save(E/'SPLIT_COUNTS.json',counts)
    near={}
    for r in target:
        poly=pyogrio.read_dataframe(str(__import__('pathlib').Path(r['image_path']).parent/'PROVIDED_MAP_GT.gpkg')).to_crs(32649)
        z=ext[r['site']];external=unpack(z['a'])+unpack(z['b']);tree=STRtree(external)
        near[r['id']]=set()
        for gt in poly[poly.fusion_role=='DESIGNATED_GT2019_ANCHOR'].itertuples():
            for j in tree.query(gt.geometry,predicate='intersects'):
                g=external[j];inter=g.intersection(gt.geometry).area
                if inter/(g.area+gt.geometry.area-inter)>=.90:
                    near[r['id']].add(int(gt.original_gt_fid));break
    result={}
    for name in METHODS:
        with ProcessPoolExecutor(6) as pool:ww=list(pool.map(evaluate,[(r,manifests[name]) for r in target]))
        save(E/(name+'_DETAILS.json'),ww)
        result[name]={}
        for ep in sorted({w['epoch'] for w in ww}):
            subset=[w for w in ww if w['epoch']==ep];out={}
            for typ in ('all','excluding_near_identical'):
                pairs=[(a,b) for w in subset for a,b in zip(w['details']['RAW'],w['details']['ROI_SELECTED']) if typ=='all' or a['fid'] not in near[w['window']]]
                assert all(a['fid']==b['fid'] and a['covered_pixels']==b['covered_pixels'] for a,b in pairs)
                out[typ]=dict(n=len(pairs),raw=sum(a['IoU50_match'] for a,b in pairs),after=sum(b['IoU50_match'] for a,b in pairs),
                    gained=sum(not a['IoU50_match'] and b['IoU50_match'] for a,b in pairs),lost=sum(a['IoU50_match'] and not b['IoU50_match'] for a,b in pairs))
            result[name][ep]=out
    save(E/'RESULTS.json',result)
if __name__=='__main__':main()

