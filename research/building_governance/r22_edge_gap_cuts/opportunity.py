"""Post-freeze, manual-reference-conditioned candidate diagnostics for previously missed FIDs."""
from assess import *
from affine import Affine
def one(args):
    r,past,baseline,delta=args;key=r['id'];bound,base,refs=load_window_geometry(r)
    keys=list(refs);gg=list(refs.values());tree=shapely.STRtree(gg)
    fp=D/(key+'.npz')
    with np.load(fp) as f:group=f['groups'];masks=f['masks']
    with np.load(OLD/'data'/(key+'.npz')) as f:boxes=f['boxes']
    with np.load(r['query_cache_path']) as f:raw=f['instances']
    with np.load(past['path']) as f:used=set(np.unique(raw[raw!=f['SPLIT']]))
    matched={int(o['fid']):o['qid'] for o in baseline if o['match']}
    with rasterio.open('/root/xian20261111_r8/output/xian72_governance/'+key+'/ROOF_GOVERNED_OBSERVATION_ID.tif') as ds:
        gov=ds.read(1);af=ds.transform;proj=Transformer.from_crs(ds.crs,32649,always_xy=True).transform
    trials=[]
    for k in np.unique(group):
        q,y0,y1,x0,x1=boxes[k]
        if q not in base or q in used:continue
        dx,dy=delta.get(key+'_q'+str(q),(0.,0.));parent=translate(base[q],dx,dy).intersection(bound)
        js=tree.query(parent,predicate='intersects')
        if not len(js):continue
        inter=np.array([parent.intersection(gg[j]).area for j in js]);areas=np.array([gg[j].area for j in js])
        missed=np.array([keys[j] not in matched for j in js])
        if not ((inter>=.5*areas)&missed).any():continue
        must=np.array([matched.get(keys[j])==q for j in js])
        win=np.s_[y0:y1,x0:x1];parentmask=raw[win]==q
        for i in np.where(group==k)[0]:
            cut=cv2.resize(np.unpackbits(masks[i]).reshape(64,64),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST).astype(bool)&parentmask
            if min(cut.sum(),(parentmask&~cut).sum())<max(32,.1*parentmask.sum()):continue
            cut&=gov[win]==q
            if not cut.any():continue
            tr=af*Affine.translation(int(x0),int(y0))
            polys=[transform(proj,shapely.make_valid(shape(g))) for g,_ in shapes(cut.astype('uint8'),mask=cut,transform=tr)]
            child=translate(unary_union(polys),dx,dy).intersection(parent)
            ca=child.area;cb=parent.area-ca;a=np.array([child.intersection(gg[j]).area for j in js]);b=np.maximum(0,inter-a)
            ma=a/(areas+ca-a).clip(1e-9)>=.5;mb=b/(areas+cb-b).clip(1e-9)>=.5
            harm=((a>=.10*areas)&(b>=.10*areas)).any()
            if harm:continue
            for idx in np.flatnonzero(missed&(ma|mb)):
                if not must.any() and inter[idx]/(parent.area+areas[idx]-inter[idx])>=.5:continue
                if must.any() and not ((ma[idx] and mb[must].any()) or (mb[idx] and ma[must].any())):continue
                trials.append(dict(window=key,epoch=r['epoch'],parent=int(q),fid=int(keys[js[idx]]),candidate=int(i)))
    return trials
def main():
    past=json.loads((BASE/'BEFORE_GT.json').read_text());bas={v['id']:v['split'] for v in json.loads((BASE/'POSITION_DETAILS.json').read_text())}
    reg=rows();fp=SHIFT/'ALL_POSITION_PREDICTIONS.json';m=json.loads(fp.read_text())['methods'];delta={k:np.array(v)-m['RAW'][k] for k,v in m['GEOMETRY__DATE_FIELD'].items()}
    with ProcessPoolExecutor(12) as pool:out=list(pool.map(one,[(reg[k],v,bas[k],{kk:vv for kk,vv in delta.items() if kk.startswith(k+'_q')}) for k,v in past.items()]))
    rr=[v for part in out for v in part]
    if rr:table('POTENTIAL_CUTS.csv',rr)
    sums={}
    for ep in sorted({reg[k]['epoch'] for k in past}):
        v=[r for r in rr if r['epoch']==ep];sums[ep]=dict(previously_missed_with_safe_local_cut=len(set((r['window'],r['fid']) for r in v)),parents=len(set((r['window'],r['parent']) for r in v)),cut_ref_pairs=len(v))
    save(P/'POTENTIAL.json',sums)
if __name__=='__main__':main()
