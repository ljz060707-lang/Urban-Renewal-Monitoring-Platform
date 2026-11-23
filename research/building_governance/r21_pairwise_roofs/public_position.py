"""Learn how far to translate an observed roof towards corroborated public map anchors."""
import json,sys
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,rasterio,shapely
from rasterio.features import shapes
from affine import Affine
from shapely.ops import transform,unary_union
from shapely.geometry import shape
from shapely.affinity import translate
from pyproj import Transformer
from scipy.optimize import least_squares
from scipy.special import expit
from prepare import R,R8,rows,digest,verify,save,partition
from position import SHIFT,load_window_geometry,scores
import measure_xian_fraction_windows as frac
D=R/'public_position';D.mkdir(exist_ok=True)
REG=rows()
def load_inputs():
    ext={}
    for prefix in ('EXTERNAL_MAP','TRAIN_EXTERNAL_MAP'):
        path=R/(prefix+'_INPUTS.json');verify(path,json.loads((R/(prefix+'_PROTOCOL.json')).read_text())['sha256'])
        ext.update(json.loads(path.read_text()))
    return ext
def geom_input(r):
    a=Affine(*r['native_affine']);tr=Transformer.from_crs(r['native_crs'],32649,always_xy=True)
    if r['role']=='TUNE':
        folder=R8/'output/xian72_governance'/r['id']
        p=folder/'ROOF_GOVERNED_OBSERVATION_ID.tif'
        rec=json.loads((folder/'WINDOW_RECEIPT.json').read_text())
        verify(p,next(v['sha256'] for v in rec['artifacts'] if Path(v['path']).name==p.name))
        with rasterio.open(p) as ds:lab=ds.read(1).astype('int32')
    else:
        verify(r['query_cache_path'],r['query_cache_sha256'])
        with np.load(r['query_cache_path']) as z:lab=z['instances'].astype('int32')
    allg=defaultdict(list)
    for g,q in shapes(lab,mask=lab>0,transform=a):
        allg[int(q)].append(transform(tr.transform,shapely.make_valid(shape(g))))
    return lab,{q:unary_union(v) for q,v in allg.items()}
def generate(arg):
    r,role,maps,field,frozen=arg;lab,geoms=geom_input(r)
    a=[shapely.from_wkb(bytes.fromhex(g)) for _,g in maps['a']];b=[shapely.from_wkb(bytes.fromhex(g)) for _,g in maps['b']]
    public=[(a[i],b[j]) for i,j in maps['pairs']]
    tree=shapely.STRtree([x for x,_ in public]);candidates=[];obs={}
    shape_center=lambda g:np.array([g.centroid.x,g.centroid.y])
    if role!='XIAN':
        verify(r['date_path'],r['files']['ACTUAL_DATE_MAP.tif'])
        with rasterio.open(r['date_path']) as ds:dates=ds.read(1)
    for q,g in geoms.items():
        center=shape_center(g);uid=r['id']+'_q'+str(q)
        if role=='XIAN':shift=np.array(frozen.get(uid,(0.,0.)))
        else:
            vals,counts=np.unique(dates[lab==q],return_counts=True)
            if len(vals)==0 or counts.max()<.95*counts.sum():continue
            date={1:'2019-12-02',2:'2019-12-10'}.get(vals[counts.argmax()])
            if not date or date not in field['dates']:continue
            features=np.r_[1,(center-field['center'])/field['scale']]
            shift=-(features@np.asarray(field['a'])[field['dates'].index(date)])
        pred=center+shift;obs[q]=(center,shift)
        for j in tree.query(g.centroid.buffer(36)):
            external,other=public[j];pc=shape_center(external);oc=shape_center(other);vec=(pc+oc)*.5-pred
            dist=np.linalg.norm(vec)
            if dist>16 or not .5<=external.area/g.area<=2:continue
            aligned=translate(g,*(pc-center));inter=aligned.intersection(external).area
            si=inter/(g.area+external.area-inter)
            if si<.48:continue
            pi=external.intersection(other).area
            consensus=pi/(external.area+other.area-pi)
            score=si-.24*dist/16
            candidates.append(dict(q=q,j=int(j),score=score,shape=si,agreement=consensus,dis=float(np.linalg.norm(pc-oc)),distance=float(dist),v=vec.tolist()))
    byq=defaultdict(list);byj=defaultdict(list)
    for x in candidates:byq[x['q']].append(x);byj[x['j']].append(x)
    for group in list(byq.values())+list(byj.values()):group.sort(key=lambda x:-x['score'])
    records={}
    for q,choices in byq.items():
        best=choices[0];mates=byj[best['j']]
        if mates[0]['q']!=q:continue
        margin=min(best['score']-choices[1]['score'] if len(choices)>1 else 1,best['score']-mates[1]['score'] if len(mates)>1 else 1)
        if margin<.06:continue
        c,s=obs[q];best.update(center=c.tolist(),baseline=s.tolist(),margin=margin)
        records[r['id']+'_q'+str(q)]=best
    return dict(id=r['id'],role=role,records=records)
def feature(r):
    return [1.,(r['shape']-.65)*3,(r['agreement']-.60)*3,(r['dis']-4)/6,(r['distance']-7)/7]
def alpha(theta,rec):return .9*expit(np.asarray([feature(r) for r in rec])@theta)
def train(records):
    src=[json.loads(l) for l in (R8/'QUERY_SAMPLES.jsonl').read_text().splitlines() if l]
    known={s['id']:s for s in src if s['role']=='TRAIN' and s['reference_source']=='DESIGNATED_GT2019_ANCHOR' and s['actual_date'] in ('2019-12-02','2019-12-10')}
    data=defaultdict(list)
    for window in records:
        role=window['role']
        if role not in ('MAP_FIT','MAP_INNER'):continue
        rr=REG[window['id']];aff=Affine(*rr['native_affine']);tr=Transformer.from_crs(rr['native_crs'],32649,always_xy=True)
        for uid,v in window['records'].items():
            if uid not in known:continue
            s=known[uid];truth=np.array(tr.transform(*(aff*tuple(s['target_center_native']))));grp=tuple(np.floor(truth/2000).astype(int))
            if partition(grp)!=(role=='MAP_INNER') or (grp[0]*31+grp[1])%5==0:continue
            data[role].append(dict(rec=v,truth=truth.tolist(),fid=s['reference_id']))
    assert not {v['fid'] for v in data['MAP_FIT']}&{v['fid'] for v in data['MAP_INNER']}
    def arrays(values):
        obs=[v['rec'] for v in values];true=np.array([v['truth'] for v in values])
        base=np.array([np.array(r['center'])+r['baseline'] for r in obs]);dr=np.array([r['v'] for r in obs])
        return obs,true,base,dr
    obs,true,base,dr=arrays(data['MAP_FIT'])
    def fun(theta):return np.r_[(base+dr*alpha(theta,obs)[:,None]-true).ravel(),3*theta]
    opt=least_squares(fun,np.zeros(5),loss='huber',f_scale=3.,max_nfev=100)
    theta=opt.x;summary={}
    for role,items in data.items():
        obs,y,b,dr=arrays(items);s=alpha(theta,obs)
        summary[role]={}
        for label,f in [('date',0),('half',.5),('learned',s[:,None])]:
            err=np.linalg.norm(b+f*dr-y,axis=1)
            summary[role][label]=dict(n=len(err),mean=float(err.mean()),median=float(np.median(err)),p95=float(np.quantile(err,.95)))
    save(D/'PARAMETERS.json',dict(theta=theta.tolist(),fit_function_evals=opt.nfev,fit_termination=opt.message,summary=summary,
        centre_target='already admitted, unique known manual TRAIN footprint; no unmatched negatives; MAP_INNER spatially disjoint from this residual fit but historical date-field fit may reuse TRAIN',
        scale='new displacement equals alpha times vector to public pair center; maximum alpha .90'))
    return theta
def evaluate(arg):
    r,parts=arg;bound,objects,refs=load_window_geometry(r);result={}
    for label,delta in parts.items():
        geo={q:translate(g,*delta.get(r['id']+'_q'+str(q),(0.,0.))).intersection(bound) for q,g in objects.items()}
        result[label]=scores(geo,refs)
    frac.DIRECTORY=D/'fraction'
    f=frac.measure((r,parts['learned']))
    # A separate exact geometric comparison with the already-validated date-only raster.
    old=Path('/root/xian20261111_r20_route/four_arm_fraction/DATE')/r['id']/'FRACTION_10M.tif'
    with rasterio.open(old) as s,rasterio.open(frac.DIRECTORY/f['relative_file']) as t:
        assert s.transform==t.transform and s.crs==t.crs
        gt=s.read(2);bl=s.read(4);new=t.read(4);valid=gt>0
        mae_base=float(np.abs(bl[valid]-gt[valid]).sum(dtype=np.float64));mae_new=float(np.abs(new[valid]-gt[valid]).sum(dtype=np.float64))
    return dict(id=r['id'],site=r['site'],epoch=r['epoch'],records=result,fraction=f,ncell=int(valid.sum()),sum_mae_base=mae_base,sum_mae_new=mae_new)
def main():
    ext=load_inputs();fsp=SHIFT/'ALL_POSITION_PREDICTIONS.json';verify(fsp,json.loads((SHIFT/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json').read_text())['prediction_sha256'])
    m=json.loads(fsp.read_text())['methods'];delta={k:(np.array(v)-m['RAW'][k]).tolist() for k,v in m['GEOMETRY__DATE_FIELD'].items()}
    field=json.loads(Path('/root/xian20261111_r13/output/BASELINE_PARAMETERS.json').read_text())
    inputs=[(REG[v['id']],v['role']) for v in json.loads((R/'PROTOCOL.json').read_text())['inputs'] if v['role'] in ('MAP_FIT','MAP_INNER','XIAN')]
    tasks=[(r,role,ext[r['site']],field,{k:v for k,v in delta.items() if k.startswith(r['id']+'_q')}) for r,role in inputs]
    with ProcessPoolExecutor(10) as pool:records=list(pool.map(generate,tasks))
    save(D/'INPUT_FEATURES_BEFORE_REF.json',records);theta=train(records)
    tasks=[];allshifts={}
    for w in records:
        if w['role']!='XIAN':continue
        r=REG[w['id']];base={k:v for k,v in delta.items() if k.startswith(r['id']+'_q')}
        parts={n:dict(base) for n in ('date','half','learned')}
        for uid,v in w['records'].items():
            for label,f in [('half',.5),('learned',alpha(theta,[v])[0])]:
                parts[label][uid]=(np.array(v['baseline'])+f*np.array(v['v'])).tolist()
        tasks.append((r,parts));allshifts[r['id']]=parts
    save(D/'CORRECTIONS_FROZEN_BEFORE_TUNE.json',allshifts)
    with ProcessPoolExecutor(8) as pool:ww=list(pool.map(evaluate,tasks))
    save(D/'DETAILS.json',ww);summary={}
    near={}
    import pyogrio
    for w in ww:
        r=REG[w['id']];mp=ext[r['site']];pol=[shapely.from_wkb(bytes.fromhex(g)) for fieldname in ('a','b') for _,g in mp[fieldname]];tree=shapely.STRtree(pol)
        gt=pyogrio.read_dataframe(Path(r['image_path']).parent/'PROVIDED_MAP_GT.gpkg').to_crs(32649);ids=set()
        for v in gt[gt.fusion_role=='DESIGNATED_GT2019_ANCHOR'].itertuples():
            g=v.geometry
            for j in tree.query(g,predicate='intersects'):
                i=g.intersection(pol[j]).area
                if i/(g.area+pol[j].area-i)>=.9:ids.add(int(v.original_gt_fid));break
        near[w['id']]=ids
    for ep,expected in zip(sorted({w['epoch'] for w in ww}),(181,166,91)):
        rr=[w for w in ww if w['epoch']==ep];summary[ep]={}
        for label in ('date','half','learned'):
            for qual in ('all','exclude_near_copies'):
                chunk=[(a,b) for w in rr for a,b in zip(w['records']['date'],w['records'][label]) if qual=='all' or a['fid'] not in near[w['id']]]
                cnt=sum(b['match'] for a,b in chunk);ds=[a['centroid_m'] for a,b in chunk if a['match'] and a['qid']==b['qid']];de=[b['centroid_m'] for a,b in chunk if a['match'] and a['qid']==b['qid']]
                if label=='date' and qual=='all':assert cnt==expected
                summary[ep][label+'_'+qual]=dict(n=len(chunk),matched=cnt,gain=sum(not a['match'] and b['match'] for a,b in chunk),loss=sum(a['match'] and not b['match'] for a,b in chunk),
                    best_iou=float(np.mean([b['iou'] for a,b in chunk])),shared_qids=len(ds),shared_before_after_mean_m=[float(np.mean(ds)),float(np.mean(de))],
                    shared_before_after_p95_m=[float(np.quantile(ds,.95)),float(np.quantile(de,.95))])
        n=sum(w['ncell'] for w in rr)
        summary[ep]['reference_positive_10m_cells']=dict(n=n,baseline=sum(w['sum_mae_base'] for w in rr)/n,learned=sum(w['sum_mae_new'] for w in rr)/n)
    save(D/'RESULTS.json',summary)
if __name__=='__main__':main()

