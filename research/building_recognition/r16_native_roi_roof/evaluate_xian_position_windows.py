"""Compare frozen automatic shifts on all designated GT observations in supported TUNE windows."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.features import shapes
from shapely import STRtree, make_valid
from shapely.affinity import translate
from shapely.geometry import shape
from shapely.ops import transform as project, unary_union
from scipy.optimize import linear_sum_assignment

from evaluate_xian_map_windows import R8, OUT, read_verified

SHIFT=Path('/root/xian20261111_r14_native_temporal/output/automatic_position')
ARMS=['RAW','GEOMETRY__DATE_FIELD','GEOMETRY__TEMPORAL_DATE_FIELD_MEAN']
def load_window_geometry(row):
    name=row['id']
    folder=R8/'output/xian72_governance'/name
    receipt=json.loads((folder/'WINDOW_RECEIPT.json').read_text())
    path=folder/'ROOF_GOVERNED_OBSERVATION_ID.tif'
    read_verified(path,next(x['sha256'] for x in receipt['artifacts'] if Path(x['path']).name==path.name))
    native=Path(row['image_path']).parent
    data=row['source_record']
    for filename in ['SOURCE_SUPPORT.tif','PROVIDED_MAP_GT.gpkg']:
        read_verified(native/filename,data['files'][filename]['sha256'])
    with rasterio.open(path) as d, rasterio.open(native/'SOURCE_SUPPORT.tif') as s:
        assert s.transform==d.transform and s.crs==d.crs
        lab=d.read(1).astype('int32')
        valid=s.read(1)>0
        tr=Transformer.from_crs(d.crs,32649,always_xy=True).transform
        def polygons(array):
            by_id=defaultdict(list)
            for geo,k in shapes(array,mask=(array>0),transform=d.transform):
                by_id[int(k)].append(project(tr,make_valid(shape(geo))))
            return {k:unary_union(v) for k,v in by_id.items()}
        boundary=polygons(valid.astype('uint8'))[1]
        obj=polygons(lab)
    gt=pyogrio.read_dataframe(native/'PROVIDED_MAP_GT.gpkg').to_crs(32649)
    gt=gt[gt.fusion_role=='DESIGNATED_GT2019_ANCHOR']
    reference={int(fid):make_valid(g).intersection(boundary) for fid,g in zip(gt.original_gt_fid,gt.geometry)}
    reference={k:g for k,g in reference.items() if g.area>1.e-6}
    return boundary, obj, reference

def window(args):
    row, coords=args
    name=row['id']
    boundary,obj,reference=load_window_geometry(row)
    keys=list(obj); fids=list(reference)
    output={}
    for method in ARMS:
        moved=[]
        for k,g in obj.items():
            q=f'{name}_q{k}'
            assert (q in coords[method]) == (q in coords['RAW'])
            dx,dy=(np.array(coords[method][q])-coords['RAW'][q]) if q in coords['RAW'] else (0.,0.)
            moved.append(translate(g,xoff=dx,yoff=dy).intersection(boundary))
        tree=STRtree(moved)
        overlap=np.zeros((len(fids),len(keys)))
        covered=[]
        for i,g in enumerate(reference.values()):
            candidates=tree.query(g,predicate='intersects')
            parts=[]
            for j in candidates:
                inter=g.intersection(moved[j])
                overlap[i,j]=inter.area/(g.area+moved[j].area-inter.area)
                parts.append(inter)
            covered.append(unary_union(parts).area if parts else 0.)
        limit=min(overlap.shape)
        weight=(overlap>=.5).astype(float)+overlap/(limit+1)
        ii,jj=linear_sum_assignment(-weight)
        assignment={i:j for i,j in zip(ii,jj) if overlap[i,j]>=.5}
        objects=[]
        for i,(fid,g) in enumerate(reference.items()):
            j=assignment.get(i)
            objects.append(dict(fid=fid,area=g.area,coverage_area=covered[i],best_IoU=overlap[i].max() if len(keys) else 0.,
                    matched_query=keys[j] if j is not None else None,distance_m=g.centroid.distance(moved[j].centroid) if j is not None else None))
        output[method]=objects
    return dict(window=name,epoch=row['epoch'],site=row['site'],retained_without_shift=sum(f'{name}_q{k}' not in coords['RAW'] for k in keys),results=output)
def main():
    freeze=json.loads((SHIFT/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json').read_text())
    p=SHIFT/'ALL_POSITION_PREDICTIONS.json';read_verified(p,freeze['prediction_sha256'])
    v=json.loads(p.read_text());assert not v['GT_reference_group_input'] and not v['GT_coordinates_input']
    windows=[r for line in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))['role']=='TUNE']
    assert len(windows)==72
    work=[(r,{m:{k:v0 for k,v0 in v['methods'][m].items() if k.startswith(r['id']+'_q')} for m in ARMS}) for r in windows]
    with ProcessPoolExecutor(6) as pool:
        ww=list(pool.map(window,work))
    result={}
    for epoch in sorted({w['epoch'] for w in ww}):
        subset=[w for w in ww if w['epoch']==epoch]
        result[epoch]={}
        for m in ARMS:
            rr=[x for w in subset for x in w['results'][m]]
            base=[x for w in subset for x in w['results']['RAW']]
            assert all(a['fid']==b['fid'] for a,b in zip(base,rr))
            shared=[(a,b) for a,b in zip(base,rr) if a['matched_query'] is not None and a['matched_query']==b['matched_query']]
            result[epoch][m]=dict(n=len(rr),matched=sum(b['matched_query'] is not None for b in rr),
               mean_best_IoU=np.mean([b['best_IoU'] for b in rr]),reference_coverage=sum(b['coverage_area'] for b in rr)/sum(b['area'] for b in rr),
               gained=sum(a['matched_query'] is None and b['matched_query'] is not None for a,b in zip(base,rr)),
               lost=sum(a['matched_query'] is not None and b['matched_query'] is None for a,b in zip(base,rr)),
               paired_identical_matches=len(shared),
               raw_paired_m=np.mean([a['distance_m'] for a,b in shared]) if shared else None,
               final_paired_m=np.mean([b['distance_m'] for a,b in shared]) if shared else None)
    for key,data in [('OBJECTS',ww),('SUMMARY',dict(epochs=result,source_prediction_sha256=freeze['prediction_sha256'],
            retained_without_shift_across_frames=sum(w['retained_without_shift'] for w in ww),input_geometry='R8 after facade, raw means no displacement',scope='partial 2019 map polygon geometric correspondence, complete supplied TUNE windows, no new selection, no physical facade or independent native roof labels',
            flags='footprints/predictions clipped to same native support; cut pixels enter neither shifted nor control; overlapping shifted objects retained separately in object matching; coverage uses intersection union',
            protected_reads=0))]:
        final=OUT/f'POSITION_{key}.json'
        temp=final.with_suffix('.tmp')
        temp.write_text(json.dumps(data,allow_nan=False))
        temp.replace(final)
if __name__=='__main__':
    main()

