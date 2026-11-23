"""Test split IDs with the original date field, keeping each child's parent translation."""
import json,math,sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np,rasterio,shapely
from rasterio.features import rasterize,shapes
from shapely.geometry import shape
from shapely.ops import transform,unary_union
from shapely.affinity import translate
from scipy.optimize import linear_sum_assignment
from affine import Affine
from pyproj import Transformer
from prepare import R,R8,verify,digest,save,rows
sys.path.insert(0,'/root/xian20261111_r16_roi_roof/code')
from evaluate_xian_position_windows import load_window_geometry,SHIFT
P=R/'position';P.mkdir(exist_ok=True)
METHODS=['date_only','image_only','selective','map_centers','map_affinity']
def scores(geoms,refs):
    keys=list(geoms);gg=list(geoms.values());tree=shapely.STRtree(gg)
    mat=np.zeros((len(refs),len(gg)));cent=np.zeros_like(mat)
    for i,g in enumerate(refs.values()):
        jj=tree.query(g,predicate='intersects')
        for j in jj:
            inter=g.intersection(gg[j]).area;mat[i,j]=inter/(g.area+gg[j].area-inter)
            cent[i,j]=g.centroid.distance(gg[j].centroid)
    size=min(mat.shape);ii,jj=linear_sum_assignment(-((mat>=.5).astype(float)+mat/(size+1)))
    matches={int(i):int(j) for i,j in zip(ii,jj) if mat[i,j]>=.5}
    return [dict(fid=fid,match=i in matches,qid=keys[matches[i]] if i in matches else None,iou=float(mat[i].max()) if len(gg) else 0.,
                 centroid_m=float(cent[i,matches[i]]) if i in matches else None) for i,fid in enumerate(refs)]
def one(arg):
    r,inputs,delta=arg;name=r['id'];bounds,base,refs=load_window_geometry(r)
    with np.load(r['query_cache_path']) as f:raw=f['instances'].astype('int32')
    folder=R8/'output/xian72_governance'/name
    with rasterio.open(folder/'ROOF_GOVERNED_OBSERVATION_ID.tif') as ds:
        governed=ds.read(1).astype('int32');aff=ds.transform;proj=Transformer.from_crs(ds.crs,32649,always_xy=True).transform
    def moved(g,parent):
        return translate(g,*delta.get(name+'_q'+str(parent),(0,0))).intersection(bounds)
    bm={q:moved(g,q) for q,g in base.items()}
    union=shapely.union_all(list(bm.values()))
    x0,y0,x1,y1=bounds.bounds;aff1=Affine(1,0,math.floor(x0),0,-1,math.ceil(y1))
    shape1=(math.ceil(y1)-math.floor(y0),math.ceil(x1)-math.floor(x0))
    tiff=P/(name+'.tif');output={};errors={};counts={}
    with rasterio.open(tiff,'w',driver='GTiff',width=shape1[1],height=shape1[0],count=len(METHODS)+1,dtype='uint32',transform=aff1,crs='EPSG:32649',compress='deflate',predictor=2) as ds:
        supported=rasterize([(bounds,1)],out_shape=shape1,transform=aff1,dtype='uint32')
        ds.write(supported,1);ds.set_band_description(1,'source_support')
        for band,method in enumerate(METHODS,2):
            geoms=dict(bm);parent_of={q:q for q in bm}
            if method!='date_only':
                item=inputs[method];verify(item['path'],item['sha256'])
                with np.load(item['path']) as f:lab=f['SPLIT']
                assert np.array_equal(raw>0,lab>0)
                changed=(raw!=lab)&(raw==governed)&(governed>0)
                parents=np.unique(raw[changed]);new=governed.copy();new[changed]=lab[changed]
                if changed.any():
                    assert min(lab[changed])>max(bm),'child id collision'
                    mask=np.isin(governed,parents);parts=defaultdict(list)
                    for geom,k in shapes(new,mask=mask,transform=aff):
                        parts[int(k)].append(transform(proj,shapely.make_valid(shape(geom))))
                    for parent in parents:geoms.pop(parent);parent_of.pop(parent)
                    for k,gg in parts.items():
                        parents_for_k=np.unique(governed[(new==k)&mask]);assert len(parents_for_k)==1
                        par=int(parents_for_k[0]);geoms[k]=moved(unary_union(gg),par);parent_of[k]=par
            u=shapely.union_all(list(geoms.values()));errors[method]=float(union.symmetric_difference(u).area)
            output[method]=scores(geoms,refs);counts[method]=len(geoms)
            ordered=sorted(geoms,key=lambda k:(parent_of[k],k))
            features=[(geoms[k],k) for k in ordered if geoms[k].area>0]
            lab1=rasterize(features,out_shape=shape1,transform=aff1,dtype='uint32') if features else np.zeros(shape1,np.uint32)
            lab1[supported==0]=0
            ds.write(lab1,band);ds.set_band_description(band,method+'_date_field_instance_id')
    with rasterio.open(tiff) as ds:
        a=ds.read(2)>0;pix=[int(np.count_nonzero((ds.read(k)>0)!=a)) for k in range(2,len(METHODS)+2)]
    return dict(window=name,epoch=r['epoch'],site=r['site'],methods=output,symmetric_diff_m2=errors,geometry_counts=counts,raster_union_changed_pixels=pix,sha256=digest(tiff))
def main():
    registry=rows();target=[r for r in registry.values() if r['role']=='TUNE']
    frozen=json.loads((R/'FROZEN_SELECTION.json').read_text())
    ext=json.loads((R/'external/MAP_AND_IMAGE_PREDICTIONS_FROZEN.json').read_text())
    man={'image_only':json.loads((R/'TARGET_PREDICTIONS_BEFORE_GT.json').read_text())[frozen['primary']],
         'selective':json.loads((R/'selective_split/XIAN_FROZEN_BEFORE_GT.json').read_text()),'map_centers':ext['centers'],'map_affinity':ext['centers_and_affinity']}
    path=SHIFT/'ALL_POSITION_PREDICTIONS.json';verify(path,json.loads((SHIFT/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json').read_text())['prediction_sha256'])
    data=json.loads(path.read_text());assert not data['GT_coordinates_input'] and not data['GT_reference_group_input']
    m=data['methods'];delta={k:np.array(v)-m['RAW'][k] for k,v in m['GEOMETRY__DATE_FIELD'].items()}
    tasks=[(r,{a:v[r['id']] for a,v in man.items()},{k:v for k,v in delta.items() if k.startswith(r['id']+'_q')}) for r in target]
    with ProcessPoolExecutor(8) as pool:ws=list(pool.map(one,tasks))
    summary={}
    for ep,old_n in zip(sorted({w['epoch'] for w in ws}),(181,166,91)):
        rr=[w for w in ws if w['epoch']==ep]
        base=[r for w in rr for r in w['methods']['date_only']]
        assert len(base)==998 and sum(r['match'] for r in base)==old_n
        summary[ep]={}
        for method in METHODS:
            ss=[r for w in rr for r in w['methods'][method]]
            gained=sum(not a['match'] and b['match'] for a,b in zip(base,ss))
            lost=sum(a['match'] and not b['match'] for a,b in zip(base,ss))
            summary[ep][method]=dict(n=len(ss),matched=sum(r['match'] for r in ss),gained=gained,lost=lost,
                mean_best_iou=float(np.mean([r['iou'] for r in ss])),sites_improved=sum(sum(r['match'] for r in w['methods'][method])>sum(r['match'] for r in w['methods']['date_only']) for w in rr))
    save(P/'DETAILS.json',ws)
    save(P/'SUMMARY.json',dict(epochs=summary,max_vector_union_symmetric_diff_m2=max(max(w['symmetric_diff_m2'].values()) for w in ws),
        changed_union_pixels=sum(sum(w['raster_union_changed_pixels']) for w in ws),grid='1 m EPSG32649 integer meters, source support band plus five instance ID bands',
        scope='development only; date shift inherited per parent; candidate roof area conserved at source resolution; metric denominators include all supplied positive map observations'))
if __name__=='__main__':main()

