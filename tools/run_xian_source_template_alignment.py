"""Frozen source-population roof template replay against unmasked target context."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import rasterio,fiona
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import shape
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.alignment.source_template_alignment import source_template_check
from xian_r5.fraction_shadow import build_fraction_shadow,QUALITY_BANDS,NODATA
from xian_r5.source_partition import SourceRegion
from xian_r5.single_year_experiment import experimental_pseudolabel

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    prior=parent/'SINGLE2019_ROOF_IMAGE_CHECK_20261012_09';base=parent/'SINGLE2019_20261012_06'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh template run required')
    old_plan=json.loads((prior/'FROZEN_ROOF_IMAGE_CHECK_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'ROOF_IMAGE_CHECK_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(old_plan['protected_sha256']);protected.update(receipt['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'ROOF_IMAGE_CHECK_RECEIPT.json')]=sha(prior/'ROOF_IMAGE_CHECK_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen source drift: '+path)
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    with rasterio.open(base/'2019_ROOF_FRACTION_RAW.tif') as s:raw=s.read(1)
    cache=parent/'SINGLE2019_SUPPORT_REPAIR_20261012_07/2019_BOUNDED_GE_RGB_CACHE.tif'
    with rasterio.open(cache) as s:source=np.moveaxis(s.read(),0,-1)
    with rasterio.open('F:/xian/10m底图/Xian_2019_TrueColor.tif') as s:
        w=Window(1516,1765,256,256);target=np.moveaxis(s.read([1,2,3],window=w),0,-1)
        target_valid=np.all(s.read_masks([1,2,3],window=w)>0,axis=0)
    source_mask=(raw>=.8)&np.any(source>0,axis=2)
    cells=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    plan={'run':out.name,'scope':'SAME_256x256_NOMINAL_2019','alignment_scope':'SOURCE_ROOF_TEMPLATE_TO_REFERENCE_IMAGE_CONTEXT_NOT_GROUND',
        'source_observed_date':'2018-11-01','experiment_year':2019,'source_roof_mask_threshold':.8,
        'reference_annotation_or_prediction_mask_used':False,'reference_only_mask':'RASTER_VALIDITY + frozen cell and fold',
        'same_source_feature_population_for_all_49_offsets':True,
        'search_radius_px':3,'minimum_fit_distinct_pixels':100,'minimum_check_distinct_pixels':100,
        'fold_block_size_px':16,'minimum_global_check_gain':.01,'minimum_fit_peak_gap':.02,
        'minimum_positive_blocks':3,'minimum_informative_block_distinct_pixels':20,
        'minimum_block_samples_increased_from_scalar_math_floor_3':True,
        'no_new_radius_or_threshold_search':True,'formal_alignment_acceptance':False,
        'existing_ground_and_roof_masked_results_preserved':True,'protected_sha256':protected}
    out.mkdir(parents=True);save_json(out/'FROZEN_SOURCE_TEMPLATE_PLAN.json',plan)
    records=[];regions=[]
    for cell in cells:
        rr,cc=(int(v[1:]) for v in cell['region_id'].split('_')[1:]);support=np.zeros((256,256),bool)
        support[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
        result=source_template_check(source,target,source_mask,target_valid,support)
        best=result['best'];vector=None
        if best is not None:
            dx,dy=best['correction_dx_px'],best['correction_dy_px']
            vector=[grid.affine.a*dx+grid.affine.b*dy,grid.affine.d*dx+grid.affine.e*dy]
        recommended=vector if result['engineering_experiment_supported'] else None
        record={'region_id':cell['region_id'],'diagnostic_vector_m':vector,'recommended_experiment_vector_m':recommended,
            'source_observed_date':'2018-11-01','experiment_year':2019,'diagnosis':result,'formal_acceptance':False}
        records.append(record);digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        regions.append(SourceRegion(cell['region_id'],'2018-11-01',shape(cell['support_geometry']),
            None if recommended is None else tuple(recommended),None if recommended is None else digest))
    summary={'run':out.name,'alignment_scope':plan['alignment_scope'],'records':records,
        'source_roof_mask_pixels':int(source_mask.sum()),'numeric_fit_cells':sum(r['diagnostic_vector_m'] is not None for r in records),
        'internally_supported_template_cells':sum(r.correction_m is not None for r in regions),
        'all_candidate_sample_counts_equal':all(len({v['n'] for v in r['diagnosis']['candidate_scores']})<=1 for r in records),
        'new_corrected_experiment_samples':0,'measured_gt_pixels':0,'formal_ground_accepts':0}
    if summary['internally_supported_template_cells']:
        gcache=parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
        with fiona.open(gcache) as s:objects={f.properties['component_key']:shape(f.geometry) for f in s}
        result=build_fraction_shadow(objects,regions,grid)
        pseudo=experimental_pseudolabel(result['candidate_fraction'].astype(np.float32).astype(np.float64),
            result['scope_fraction'][1].astype(np.float32).astype(np.float64),result['quality'],np.ones((256,256),bool),
            allow_unverified_geometry=True,allow_assumed_background=True)
        write(out,'2019_TEMPLATE_ALIGNED_FRACTION.tif',result['candidate_fraction'],grid,['TEMPLATE_ALIGNED_FRACTION'],nodata=NODATA)
        write(out,'2019_TEMPLATE_EXPERIMENTAL_LABEL.tif',pseudo['label'],grid,['EXPERIMENTAL_LABEL'],nodata=255)
        write(out,'2019_TEMPLATE_EXPERIMENTAL_WEIGHT.tif',pseudo['experiment_weight'],grid,['EXPERIMENT_SELECTION'])
        write(out,'2019_TEMPLATE_ALIGNMENT_QUALITY.tif',result['quality'],grid,QUALITY_BANDS)
        save_json(out/'TEMPLATE_COMPONENT_LINEAGE.json',{'records':result['component_ledger'],'role':plan['alignment_scope']})
        summary['new_corrected_experiment_samples']=pseudo['counts']['experimental_samples']
        summary['area_conservation_error_m2']=result['area_conservation_error_m2']
    save_json(out/'SOURCE_TEMPLATE_REPLAY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Protected input changed: '+path)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
