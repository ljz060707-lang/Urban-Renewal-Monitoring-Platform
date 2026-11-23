from cuts import *
import csv,math
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
from prepare import rows
from position import scores,load_window_geometry,SHIFT
from predict import BASE
import shapely,rasterio
from shapely.affinity import translate
from shapely.ops import transform,unary_union
from shapely.geometry import shape
from rasterio.features import shapes,rasterize
from pyproj import Transformer
def fragments(g,geoms):
    keys=list(geoms);gg=list(geoms.values());tree=shapely.STRtree(gg)
    return {fid:set(keys[j] for j in tree.query(ref,predicate='intersects') if ref.intersection(gg[j]).area>=.1*ref.area) for fid,ref in g.items()}
def work(arg):
    r,past,latest,delta=arg;key=r['id'];bound,base,refs=load_window_geometry(r)
    def move(g,q):return translate(g,*delta.get(key+'_q'+str(q),(0.,0.))).intersection(bound)
    initial={q:move(g,q) for q,g in base.items()}
    with np.load(r['query_cache_path']) as f:raw=f['instances'].astype('int32')
    with rasterio.open('/root/xian20261111_r8/output/xian72_governance/'+key+'/ROOF_GOVERNED_OBSERVATION_ID.tif') as ds:
        gov=ds.read(1).astype('int32');af=ds.transform;proj=Transformer.from_crs(ds.crs,32649,always_xy=True).transform
    def apply(item):
        verify(item['path'],item['sha256'])
        with np.load(item['path']) as f:split=f['SPLIT']
        change=(raw==gov)&(gov>0)&(raw!=split);parents=set(int(i) for i in np.unique(raw[change]))
        lab=gov.copy();lab[change]=split[change];sub=np.isin(gov,list(parents));polys=defaultdict(list);owner={q:q for q in initial}
        out=initial.copy()
        if parents:
            for geom,q in shapes(lab,mask=sub,transform=af):polys[int(q)].append(transform(proj,shapely.make_valid(shape(geom))))
        for q in parents:out.pop(q)
        for q,parts in polys.items():
            par=np.unique(gov[sub&(lab==q)]);assert len(par)==1
            owner[q]=int(par[0]);out[q]=move(unary_union(parts),owner[q])
        return out,parents,owner
    old,op,oo=apply(past);new,np_,no=apply(latest);added=np_-op
    sa=scores(initial,refs);sb=scores(old,refs);sc=scores(new,refs)
    fa=fragments(refs,initial);fb=fragments(refs,old);fc=fragments(refs,new)
    rr=[];harmed=set();unassessed=set(added);paired=set()
    for a,b,c in zip(sa,sb,sc):
        fid=a['fid'];assert fid==b['fid']==c['fid']
        # Direct attribution requires two substantial children of the same newly split original parent.
        counts=Counter(no[q] for q in fc[fid] if no[q] in added)
        direct={q for q,n in counts.items() if n>=2};harmed.update(direct)
        unassessed-=set(counts)
        gain=not b['match'] and c['match']
        if gain:paired.add(no[c['qid']])
        rr.append(dict(window=key,epoch=r['epoch'],site=r['site'],actual_date=r['source_record']['actual_date'],fid=fid,date_match=int(a['match']),prior_match=int(b['match']),new_match=int(c['match']),prior_iou=b['iou'],new_iou=c['iou'],gained=int(gain),lost=int(b['match'] and not c['match']),prior_fragments=len(fb[fid]),new_fragments=len(fc[fid]),new_fragmented=int(len(fb[fid])<2<=len(fc[fid])),split_same_ref=int(bool(direct)),broken_previously_matched=int(bool(direct) and b['match']),parents_within_reference=','.join(str(q) for q in sorted(direct))))
    with rasterio.open(BASE/'position'/(key+'.tif')) as ds:
        sup=ds.read(1);ori=ds.read(2);pre=ds.read(3);pf=ds.profile;af1=ds.transform
    order=sorted(new,key=lambda q:(no[q] if no[q] in added else q,q));items=[(new[q],q) for q in order if new[q].area>0]
    arr=rasterize(items,out_shape=pre.shape,transform=af1,dtype='uint32') if items else np.zeros_like(pre);arr[sup==0]=0
    err=int(np.sum((arr>0)!=(pre>0)));assert err==0,(key,err)
    assert not np.any((pre!=arr) & ~np.isin(pre,list(added))),key
    path=P/'position';path.mkdir(exist_ok=True);f=path/(key+'.tif');pf.update(count=4)
    with rasterio.open(f,'w',**pf) as ds:
        for i,(m,name) in enumerate(zip([sup,ori,pre,arr],['source_support','date_field','R21','R22']),1):ds.write(m,i);ds.set_band_description(i,name)
    return rr,dict(window=key,epoch=r['epoch'],actual_date=r['source_record']['actual_date'],new_parents=len(added),prior_parents=len(op),directly_fragmenting_parents=len(harmed),parents_no_substantial_reference=len(unassessed),union_pixel_changes=err,sha256=digest(f))
def table(name,rr):
    with open(P/name,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
    latest=json.loads((P/'PREDICTIONS_FROZEN_BEFORE_MAP.json').read_text());past=json.loads((BASE/'BEFORE_GT.json').read_text());reg=rows()
    path=SHIFT/'ALL_POSITION_PREDICTIONS.json';sha=json.loads((SHIFT/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json').read_text())['prediction_sha256'];verify(path,sha)
    m=json.loads(path.read_text())['methods'];delta={k:np.array(v)-m['RAW'][k] for k,v in m['GEOMETRY__DATE_FIELD'].items()}
    with ProcessPoolExecutor(9) as pool:out=list(pool.map(work,[(reg[k],past[k],v,{kk:vv for kk,vv in delta.items() if kk.startswith(k+'_q')}) for k,v in latest.items()]))
    rr=[row for batch,_ in out for row in batch];windows=[w for _,w in out]
    table('REFERENCE_COMPARISON.csv',rr);table('WINDOWS.csv',windows)
    sums={}
    for ep in sorted({row['epoch'] for row in rr}):
        sel=[row for row in rr if row['epoch']==ep];ww=[w for w in windows if w['epoch']==ep]
        sums[ep]={s:sum(row[s] for row in sel) for s in ('date_match','prior_match','new_match','gained','lost','new_fragmented','split_same_ref','broken_previously_matched')}
        sums[ep].update({s:sum(w[s] for w in ww) for s in ('new_parents','prior_parents','directly_fragmenting_parents','parents_no_substantial_reference','union_pixel_changes')})
        sums[ep].update(n=len(sel),dates=dict(Counter(str(w['actual_date']) for w in ww)))
    assert [v['prior_match'] for v in sums.values()]==[182,167,91]
    save(P/'RESULTS.json',dict(epochs=sums,date_field_sha256=sha,reference='designated manual 2019 reference reused in three actual observation periods; 24 previously exposed locations; no native roof or facade GT for each date',match='projected supported polygons IoU>=0.50 one-to-one; fragmentation: two new children of the same parent each cover >=10% of a supported reference polygon'))
if __name__=='__main__':main()
