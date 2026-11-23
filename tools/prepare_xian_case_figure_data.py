"""Read existing scientific products into a plotting-only snapshot."""
from pathlib import Path
import argparse,csv,json,shutil,sys
import numpy as np,rasterio,fiona
from affine import Affine
from pyproj import Transformer
from shapely.geometry import shape,box
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh plotting-only directory required')
    audit=parent/'SINGLE2019_FRACTION_BLOCK_AUDIT_20261012_27'
    plan=json.loads((audit/'FROZEN_FRACTION_BLOCK_AUDIT_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((audit/'FULL_WINDOW_FRACTION_AUDIT_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan['protected_sha256']);protected.update(plan['new_code_sha256'])
    protected.update({str(audit/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(audit/'FULL_WINDOW_FRACTION_AUDIT_RECEIPT.json')]=sha(audit/'FULL_WINDOW_FRACTION_AUDIT_RECEIPT.json')
    native=parent/'SINGLE2019_NEGATIVE_BLOCK_NATIVE_20261012_26'
    current=parent/'SINGLE2019_RISK_QUARANTINED_WINDOW_20261012_25'
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError('Protected evidence drift: '+p)
    code=[Path(__file__).resolve(),ROOT/'figures/gen_fig_xian_current_cases.py']
    out.mkdir();save_json(out/'FROZEN_CASE_FIGURE_PLAN.json',{
        'role':'SCIENTIFIC_DIAGNOSTIC_CASE_FIGURES_NOT_SUCCESSFUL_CORRECTION_OR_ACCURACY_CERTIFICATE',
        'selection':'TWO_PREEXISTING_NEGATIVE_BLOCKS_NOT_CHOSEN_FOR_BEST_APPEARANCE',
        'plotting_only_no_labels_geometry_or_vectors_changed':True,
        'source_actual_date':'2018-11-01','experiment_year':2019,
        'protected_sha256':protected,'new_code_sha256':{str(f):sha(f) for f in code}})
    arrays={};grid=None
    sources={'raw':parent/'SINGLE2019_20261012_06/2019_ROOF_FRACTION_RAW.tif',
        'candidate':current/'2019_ROOF_FRACTION_CANDIDATE.tif',
        'reference':current/'2019_MANUAL_REFERENCE_ROOF_FRACTION.tif',
        'label':current/'2019_RECOMMENDED_PSEUDO_LABEL.tif',
        'weight':current/'2019_RECOMMENDED_PSEUDO_WEIGHT.tif',
        'risk':current/'2019_TARGET_ALIGNMENT_RISK_QUARANTINE.tif'}
    for key,p in sources.items():
        with rasterio.open(p) as s:
            g=(s.transform,s.shape,str(s.crs))
            if grid is not None and g!=grid:raise ValueError('Figure inputs use different grids')
            grid=g;arrays[key]=s.read(1)
    if grid[1:]!=((256,256),'EPSG:32649'):raise ValueError('Current figure scope changed')
    affine=grid[0]
    cases=json.loads((native/'NATIVE_NEGATIVE_BLOCK_EVIDENCE.json').read_text(encoding='utf-8'))['records']
    ll=Transformer.from_crs(32649,4326,always_xy=True);metric=Transformer.from_crs(4326,32649,always_xy=True)
    plotted=[]
    for case in cases:
        rid=case['region_id'];tc,tr,w,h=case['target_block_pixel_window']
        sl=np.s_[tr:tr+h,tc:tc+w]
        original=native/case['native_raw_png'];shutil.copy2(original,out/original.name)
        if sha(out/original.name)!=case['native_raw_png_sha256']:raise ValueError('Native photo copy differs')
        raw=arrays['raw'][sl].astype(float);candidate=arrays['candidate'][sl].astype(float);ref=arrays['reference'][sl].astype(float)
        ma=[float(np.abs(a-ref).mean()) for a in (raw,candidate)]
        expected=[case['raw_fraction_reference_MAE_same_target_cells'],case['candidate_fraction_reference_MAE_same_target_cells']]
        if not np.allclose(ma,expected,rtol=0,atol=1e-12):raise ValueError('Case numerical annotation differs')
        rings=[];fids=[]
        window_box=box(*(affine*(tc,tr+h)),*(affine*(tc+w,tr)))
        llbox=transform(ll.transform,window_box.segmentize(10)).bounds
        with fiona.open('F:/xian/西安建筑轮廓数据/西安.shp') as s:
            for f in s.filter(bbox=llbox):
                g=transform(metric.transform,shape(f.geometry))
                if not g.intersects(window_box):continue
                fids.append(str(f.id))
                parts=[g] if g.geom_type=='Polygon' else list(g.geoms)
                for part in parts:
                    for ring in [part.exterior,*part.interiors]:
                        coords=np.array(ring.coords);inv=~affine
                        rings.append([[float(inv.a*x+inv.b*y+inv.c-tc)*10,
                                       float(inv.d*x+inv.e*y+inv.f-tr)*10] for x,y in coords])
        source_affine=Affine(*case['source_raster_affine']);inv=~source_affine
        c0,r0,_,_=case['source_native_window'];sc,sr,sw,sh=case['source_block_pixel_window'];outline=[]
        for col,row in ((sc,sr),(sc+sw,sr),(sc+sw,sr+sh),(sc,sr+sh),(sc,sr)):
            x,y=ll.transform(*(affine*(col,row)));px,py=inv*(x,y);outline.append([px-c0,py-r0])
        plotted.append({'region_id':rid,'photo':original.name,'native_source_risk_outline_px':outline,
            'source_window':case['source_native_window'],'target_window':[tc,tr,w,h],
            'manual_reference_rings_m':rings,'manual_reference_fids':fids,
            'raw_fraction_MAE':ma[0],'candidate_fraction_MAE':ma[1],
            'mean_RGB_check_gain':case['mean_check_gain_recomputed'],
            'risk_quarantined_selected_pixels':int(((arrays['risk'][sl]>0)&(arrays['weight'][sl]==0)).sum()),
            'full_target_pixels':w*h})
    np.savez_compressed(out/'PLOT_DATA_ONLY.npz',**arrays)
    counts={'reference_corroborated_positive':int((arrays['label']==1).sum()),
        'assumed_background':int((arrays['label']==0).sum()),'ignore':int((arrays['label']==255).sum())}
    if counts!={'reference_corroborated_positive':4860,'assumed_background':5006,'ignore':55670}:raise ValueError('Latest labels changed')
    save_json(out/'CASE_FIGURE_DATA.json',{'cases':plotted,'counts':counts,'crs':'EPSG:32649',
        'source_actual_date':'2018-11-01','experiment_year':2019,'reference_nominal_year':2019,
        'absolute_position_accuracy':'UNKNOWN','formal_gt_admissions':0,
        'case_selection_represents_known_failures_not_random_accuracy_sample':True})
    with (out/'CASE_METRICS.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['region_id','raw_fraction_MAE','candidate_fraction_MAE','mean_RGB_check_gain','full_target_pixels'])
        writer.writeheader()
        for r in plotted:writer.writerow({k:r[k] for k in writer.fieldnames})
    for p,h in protected.items():
        if sha(p)!=h:raise ValueError('Protected evidence changed: '+p)
    save_json(out/'CASE_FIGURE_DATA_RECEIPT.json',{'cases':len(plotted),'input_artifacts_checked':len(protected),
        'input_artifacts_changed':0,'new_code_sha256':{str(f):sha(f) for f in code},
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'source_labels_or_vectors_changed':0})
    print(json.dumps({'cases':len(plotted),'counts':counts}))


if __name__=='__main__':main()
