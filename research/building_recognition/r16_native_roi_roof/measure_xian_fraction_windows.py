"""Exact 10 m grid intersections for tested date-field shifts on 72 development windows."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import rasterio
import shapely
from affine import Affine
from shapely.affinity import translate
from evaluate_xian_map_windows import R8, OUT, read_verified
from evaluate_xian_position_windows import SHIFT, load_window_geometry

DIRECTORY=OUT/'fraction_10m'
BANDS=['source_support','designated_2019_map_union','after_facade','after_facade_and_date_shift',
       'after_facade_intersection_reference','after_shift_intersection_reference']
ANCHOR_X, ANCHOR_Y=293300,3812400

def measure(args):
    row, delta=args
    boundary, objects, references=load_window_geometry(row)
    raw=shapely.union_all(list(objects.values())).intersection(boundary)
    shifted=shapely.union_all([translate(g,*delta.get(f"{row['id']}_q{k}",(0.,0.))) for k,g in objects.items()]).intersection(boundary)
    gt=shapely.union_all(list(references.values())).intersection(boundary)
    geometries=[boundary,gt,raw,shifted,raw.intersection(gt),shifted.intersection(gt)]
    xmin,ymin,xmax,ymax=boundary.bounds
    x0=ANCHOR_X+10*np.floor((xmin-ANCHOR_X)/10)
    y0=ANCHOR_Y-10*np.floor((ANCHOR_Y-ymax)/10)
    width=int(np.ceil((xmax-x0)/10))
    height=int(np.ceil((y0-ymin)/10))
    xx,yy=np.meshgrid(x0+10*np.arange(width),y0-10*np.arange(height))
    cells=shapely.box(xx,yy-10,xx+10,yy).ravel()
    tree=shapely.STRtree(cells)
    arr=np.zeros((len(BANDS),height,width),dtype='float64')
    for i,g in enumerate(geometries):
        # Polygon components of this union have disjoint interiors.
        parts=shapely.get_parts(g)
        indices=tree.query(parts,predicate='intersects')
        part,cell=indices
        areas=shapely.area(shapely.intersection(parts[part],cells[cell]))
        np.add.at(arr[i].ravel(),cell,areas/100.)
        assert abs(arr[i].sum()*100.-g.area)<max(.001,g.area*1.e-8)
    assert np.all(arr>=0) and np.all(arr<=1+1.e-8)
    arr=np.clip(arr,0.,1.).astype('float32')
    folder=DIRECTORY/row['id']
    folder.mkdir(parents=True,exist_ok=True)
    f=folder/'FRACTION_10M.tif'
    transform=Affine(10,0,x0,0,-10,y0)
    supported=arr[0]>0
    with rasterio.open(f,'w',driver='GTiff',height=height,width=width,count=len(BANDS),dtype='float32',
                      crs='EPSG:32649',transform=transform,nodata=-1,compress='deflate',predictor=3) as ds:
        for i,band in enumerate(BANDS):
            ds.set_band_description(i+1,band)
            ds.write(np.where(supported,arr[i],-1),i+1)
    with rasterio.open(f) as ds:
        assert ds.transform==transform and ds.descriptions==tuple(BANDS)
        check=ds.read()
        assert np.all(check[:,~supported]==-1) and np.array_equal(check[:,supported],arr[:,supported])
        readback_area=np.where(check>=0,check,0.).sum(axis=(1,2),dtype='float64')*100.
        residual=np.abs(readback_area-np.asarray([g.area for g in geometries]))
        assert residual.max()<.01
    known=arr[1]>0
    return dict(window=row['id'],date=row['source_record']['actual_date'],epoch=row['epoch'],site=row['site'],
        relative_file=str(f.relative_to(DIRECTORY)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        union_area_m2=dict(zip(BANDS,[g.area for g in geometries])),
        max_float32_integral_residual_m2=float(residual.max()),supported_cells=int(supported.sum()),
        positive_reference_cells=int(known.sum()),
        raw_sum_abs_fraction_difference_on_positive_reference_cells=float(np.abs(arr[2][known]-arr[1][known]).sum(dtype='float64')),
        after_shift_sum_abs_fraction_difference_on_positive_reference_cells=float(np.abs(arr[3][known]-arr[1][known]).sum(dtype='float64')))
def main():
    freeze=json.loads((SHIFT/'PREDICTIONS_FROZEN_BEFORE_REFERENCE_READ.json').read_text())
    source=SHIFT/'ALL_POSITION_PREDICTIONS.json'
    read_verified(source,freeze['prediction_sha256'])
    data=json.loads(source.read_text())
    assert not data['GT_reference_group_input'] and not data['GT_coordinates_input']
    raw=data['methods']['RAW']; shifted=data['methods']['GEOMETRY__DATE_FIELD']
    assert set(raw)==set(shifted)
    windows=[r for line in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))['role']=='TUNE']
    assert len(windows)==72 and len({r['site'] for r in windows})==24
    args=[(r,{k:(np.array(shifted[k])-xy).tolist() for k,xy in raw.items() if k.startswith(r['id']+'_q')}) for r in windows]
    with ProcessPoolExecutor(6) as pool:
        results=list(pool.map(measure,args))
    metrics={}
    for epoch in sorted({r['epoch'] for r in results}):
        rr=[r for r in results if r['epoch']==epoch]
        n=sum(r['positive_reference_cells'] for r in rr)
        metrics[epoch]={'reference_positive_development_cells':n,'mean_abs_fraction_difference_on_positive_reference_cells':{
            m:sum(r[k] for r in rr)/n for m,k in [('after_facade','raw_sum_abs_fraction_difference_on_positive_reference_cells'),
            ('after_facade_and_shift','after_shift_sum_abs_fraction_difference_on_positive_reference_cells')]}}
    (DIRECTORY/'MANIFEST.json').write_text(json.dumps(dict(windows=72,bands=BANDS,grid_origin=[ANCHOR_X,ANCHOR_Y],
        cell_area_m2=100,crs=32649,source_prediction_sha256=freeze['prediction_sha256'],records=results,
        summary=metrics,max_area_readback_residual_m2=max(r['max_float32_integral_residual_m2'] for r in results),
        scope='TUNE windows; 2019 partial map GT; a positive reference cell can contain omitted buildings; map disagreement is diagnostic only; no sum as a city total',
        method='projected polygon union then exact cell intersection / 100m2; edge cells use full 100m2 denominator; outside support is nodata'),allow_nan=False,indent=2))
if __name__=='__main__':
    main()

