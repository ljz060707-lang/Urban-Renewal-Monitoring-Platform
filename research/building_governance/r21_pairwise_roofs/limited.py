"""Evaluate a second operating point selected on Chengdu with an explicit fragmentation allowance."""
import json,sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np,cv2,joblib,rasterio
from shapely.affinity import translate
from affine import Affine
from rasterio.features import shapes,rasterize
import shapely
from shapely.geometry import shape
from shapely.ops import transform,unary_union
from pyproj import Transformer
from prepare import R,rows,verify,digest,save
from guard import G,model_for_split,guarded,propose,raw_input
from evaluate_pairs import RoofEvaluator,metrics,aggregate,trace,evaluate,summarize
from position import load_window_geometry,SHIFT,scores
A=G/'limited_fragmentation';A.mkdir(exist_ok=True)
def displaced(args):
    r,item,delta=args
    bound,base,refs=load_window_geometry(r);raw=raw_input(r)
    verify(item['path'],item['sha256'])
    with np.load(item['path']) as f:split=f['SPLIT']
    with rasterio.open('/root/xian20261111_r8/output/xian72_governance/'+r['id']+'/ROOF_GOVERNED_OBSERVATION_ID.tif') as ds:lab=ds.read(1).astype('int32');af=ds.transform;tr=Transformer.from_crs(ds.crs,32649,always_xy=True).transform
    mod=lab.copy();use=(raw==lab)&(lab>0)&(split!=raw);mod[use]=split[use]
    moved={q:translate(g,*delta.get(r['id']+'_q'+str(q),(0,0))).intersection(bound) for q,g in base.items()}
    old=scores(moved,refs)
    changed=np.unique(raw[use]);same=np.isin(lab,changed)
    kids={}
    for g,k in shapes(mod,mask=same,transform=af):
        kids.setdefault(int(k),[]).append(transform(tr,shapely.make_valid(shape(g))))
    for q in changed:moved.pop(q)
    for k,parts in kids.items():
        parents=np.unique(lab[(mod==k)&same]);assert len(parents)==1
        moved[k]=translate(unary_union(parts),*delta.get(r['id']+'_q'+str(int(parents[0])),(0,0))).intersection(bound)
    new=scores(moved,refs)
    # The original roof union and date-field translation are retained for every split parent.
    z=R/'position'/(r['id']+'.tif')
    with rasterio.open(z) as ds:support=ds.read(1);before=ds.read(2);profile=ds.profile;aff=ds.transform
    features=[(g,k) for k,g in sorted(moved.items()) if not g.is_empty]
    output=rasterize(features,out_shape=before.shape,transform=aff,dtype='uint32') if features else np.zeros_like(before)
    output[support==0]=0;err=int(np.count_nonzero((output>0)!=(before>0)))
    path=A/'position';path.mkdir(exist_ok=True);p=path/(r['id']+'.tif')
    profile.update(count=3)
    with rasterio.open(p,'w',**profile) as ds:
        for j,(arr,name) in enumerate(((support,'source_support'),(before,'old_date_field_instance_id'),(output,'learned_split_date_field_instance_id')),1):
            ds.write(arr,j);ds.set_band_description(j,name)
    return dict(id=r['id'],site=r['site'],epoch=r['epoch'],baseline=old,split=new,changed_parents=len(changed),raster_union_delta_pixels=err,sha256=digest(p))
def main():
    cv2.setNumThreads(1);reg=rows()
    inner=json.loads((G/'INNER.json').read_text());base=inner['raw']
    candidates={c:v for c,v in inner['results'].items() if v['fragmented_GT']-base['fragmented_GT']<=base['GT']//1000 and v['complete_matched_roofs90']>=base['complete_matched_roofs90']}
    cut=float(max(candidates,key=lambda c:(candidates[c]['F1_50'],-candidates[c]['fragmented_GT'])))
    save(A/'PROTOCOL.json',dict(cut=cut,source='Chengdu previously exposed INNER',rule='allow at most one additional fragmented GT per thousand known roofs; retain complete90 count; maximize instance F1. Secondary development analysis, added after inspecting strict results; Jinan and XiAn reused.'))
    model=model_for_split();gate=joblib.load(G/'gate.joblib')
    eva=RoofEvaluator('TUNE');s=[];counter=Counter()
    for f in eva.frames:
        raw=f['old'];lab,feat=propose(reg[f['id']],model,raw);out=guarded(lab,raw,feat,gate,cut)
        s.append(metrics(f,out));counter.update(trace(f,raw,out))
    save(A/'JINAN.json',dict(raw=eva.baseline,model=aggregate(s),trace=dict(counter)));del eva
    target=[r for r in reg.values() if r['role']=='TUNE'];manifest={}
    for r in target:
        raw=raw_input(r);lab,feat=propose(r,model,raw);out=guarded(lab,raw,feat,gate,cut)
        folder=A/'XIAN';folder.mkdir(exist_ok=True);p=folder/(r['id']+'.npz');np.savez_compressed(p,SPLIT=out)
        manifest[r['id']]=dict(path=str(p),variant='SPLIT',sha256=digest(p))
    save(A/'BEFORE_GT.json',manifest)
    path=SHIFT/'ALL_POSITION_PREDICTIONS.json';verify(path,json.loads((SHIFT/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json').read_text())['prediction_sha256'])
    m=json.loads(path.read_text())['methods'];delta={k:np.array(v)-m['RAW'][k] for k,v in m['GEOMETRY__DATE_FIELD'].items()}
    with ProcessPoolExecutor(6) as pool:ws=list(pool.map(displaced,[(r,manifest[r['id']],{k:v for k,v in delta.items() if k.startswith(r['id']+'_q')}) for r in target]))
    save(A/'POSITION_DETAILS.json',ws)
    sums={}
    for ep in sorted({r['epoch'] for r in ws}):
        rr=[r for r in ws if r['epoch']==ep]
        pairs=[(a,b) for w in rr for a,b in zip(w['baseline'],w['split'])]
        sums[ep]=dict(n=len(pairs),old=sum(a['match'] for a,b in pairs),new=sum(b['match'] for a,b in pairs),gained=sum(not a['match'] and b['match'] for a,b in pairs),lost=sum(a['match'] and not b['match'] for a,b in pairs),split_parents=sum(w['changed_parents'] for w in rr))
    save(A/'POSITION_SUMMARY.json',dict(epochs=sums,changed_union_px=sum(w['raster_union_delta_pixels'] for w in ws)))
if __name__=='__main__':main()

