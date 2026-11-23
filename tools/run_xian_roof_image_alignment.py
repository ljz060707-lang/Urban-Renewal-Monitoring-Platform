"""Roof-interior image-grid experiment, explicitly not a ground displacement field."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import rasterio,fiona
from rasterio.windows import Window
from shapely.geometry import shape,Polygon
from shapely.ops import transform
from pyproj import Transformer
from affine import Affine
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from run_xian_single2019_experiment import write
from xian_r5.label_transfer import Grid
from xian_r5.fraction_shadow import _areas_and_counts,build_fraction_shadow,QUALITY_BANDS,NODATA
from xian_r5.source_partition import SourceRegion
from xian_r5.alignment.supported_ground_features import internal_spatial_check
from xian_r5.alignment.image_block_replay import replay_check_blocks
from xian_r5.alignment.base_tie import image_shift_to_projected_correction
from xian_r5.single_year_experiment import experimental_pseudolabel

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    evidence=parent/'SINGLE2019_ROOF_EVIDENCE_20261012_08';repair=parent/'SINGLE2019_SUPPORT_REPAIR_20261012_07'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh separate roof-image run required')
    receipt=json.loads((evidence/'ROOF_EVIDENCE_RECEIPT.json').read_text(encoding='utf-8'))
    previous=json.loads((evidence/'FROZEN_ROOF_MATCH_PLAN.json').read_text(encoding='utf-8'))
    protected=dict(previous['protected_sha256']);protected.update(receipt['new_code_sha256'])
    protected.update({str(evidence/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(evidence/'ROOF_EVIDENCE_RECEIPT.json')]=sha(evidence/'ROOF_EVIDENCE_RECEIPT.json')
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected drift: '+path)
    grid=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
    base=parent/'SINGLE2019_20261012_06'
    with rasterio.open(base/'2019_ROOF_FRACTION_RAW.tif') as s:source_fraction=s.read(1).astype(np.float64)
    with rasterio.open(repair/'2019_BOUNDED_GE_RGB_CACHE.tif') as s:moving=np.moveaxis(s.read(),0,-1)
    with rasterio.open('F:/xian/10m底图/Xian_2019_TrueColor.tif') as s:
        reference=np.moveaxis(s.read([1,2,3],window=Window(1516,1765,256,256)),0,-1)
        refvalid=np.all(s.read_masks([1,2,3],window=Window(1516,1765,256,256))>0,axis=0)
    match_inventory=json.loads((evidence/'REFERENCE_DEVELOPMENT_EXPOSURE.json').read_text(encoding='utf-8'))
    selected_fids=set(match_inventory['reference_fids_loaded'])
    cells_raw=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    plan={'run':out.name,'scope':'SAME_256x256_NOMINAL_2019','alignment_scope':'ROOF_IMAGE_TO_REFERENCE_IMAGE_GRID_NOT_GROUND',
        'source_observed_date':'2018-11-01','experiment_year':2019,'source_roof_support_threshold':.8,
        'reference_roof_support_threshold':.8,'mask_role':'POSITIVE_ROOF_PRIOR_NOT_VERIFIED_SPECTRAL_ROOF_OR_BACKGROUND',
        'operator':'SUPPORT_CLOSED_AXIS_THREE_POINT_DIFFERENCE','search_radius_px':3,'minimum_distinct_samples':100,
        'internal_spatial_block_px':16,'minimum_global_check_gain':.01,'minimum_peak_margin':.02,
        'minimum_positive_check_blocks':3,'material_negative_block_gain':-.01,
        'manual_geometry_used_to_set_vector':False,'manual_geometry_used_as_reference_roof_mask_prior':True,
        'ground_mask_or_ground_displacement_fields_changed':False,'student_predictions_used':False,
        'no_source_shape_smoothing_or_snap':True,'formal_fit_holdout_review':False,'independent_external_validation':False,
        'protected_sha256':protected}
    out.mkdir(parents=True);save_json(out/'FROZEN_ROOF_IMAGE_CHECK_PLAN.json',plan)
    to_ll=Transformer.from_crs(32649,4326,always_xy=True);to_metric=Transformer.from_crs(4326,32649,always_xy=True)
    corners=[(308460,3792190),(311020,3792190),(311020,3794750),(308460,3794750)]
    ring=[to_ll.transform(a[0]+(b[0]-a[0])*k/64,a[1]+(b[1]-a[1])*k/64)
          for a,b in zip(corners,corners[1:]+corners[:1]) for k in range(64)]
    refs=[]
    with fiona.open('F:/xian/西安建筑轮廓数据/西安.shp') as s:
        for f in s.filter(bbox=Polygon(ring).bounds):
            if str(f.id) in selected_fids:refs.append(transform(to_metric.transform,shape(f.geometry)))
    if len(refs)!=len(selected_fids):raise ValueError('Reference coverage drift')
    cells=np.array([grid.cell(c,r) for r in range(256) for c in range(256)],dtype=object)
    area,_,conflict=_areas_and_counts(refs,cells);reference_fraction=area.reshape(256,256)/100
    refmask=(reference_fraction>=.8)&refvalid&~conflict.reshape(256,256)
    movmask=(source_fraction>=.8)&np.any(moving>0,axis=2)
    write(out,'2019_REFERENCE_ROOF_MASK_PRIOR_FRACTION.tif',reference_fraction,grid,['POSITIVE_ANNOTATED_ROOF_AREA_FRACTION'],
        extra={'ROLE':'MASK_PRIOR_NOT_TARGET_GT','ZERO_IS_KNOWN_BACKGROUND':'NO'})
    write(out,'2019_ROOF_IMAGE_MASKS.tif',np.stack([refmask,movmask]).astype(np.uint8),grid,
        ['REFERENCE_ROOF_IMAGE_PRIOR','GE_ROOF_IMAGE_PRIOR'],extra={'FEATURE_CLASS':'ROOF_NOT_STABLE_GROUND'})
    records=[];regions=[]
    for cell in cells_raw:
        rr,cc=(int(x[1:]) for x in cell['region_id'].split('_')[1:]);component=np.zeros((256,256),bool)
        component[rr*64:(rr+1)*64,cc*64:(cc+1)*64]=True
        diagnosis=internal_spatial_check(reference,moving,refmask,movmask,component,radius=3,min_samples=100)
        best=diagnosis['fit']['best'];blocks=None;reasons=list(diagnosis['reasons']);vector=None
        if best:
            runner=diagnosis['fit']['runner_up'];margin=best['corr']-runner['corr'] if runner else None
            if margin is not None and margin<.02:reasons.append('ROOF_IMAGE_FIT_PEAK_AMBIGUITY')
            blocks=replay_check_blocks(reference,moving,refmask,movmask,component,(best['dx_px'],best['dy_px']))
            if not blocks['consistent_internal_experiment']:reasons.append('CHECK_SPATIAL_BLOCKS_NOT_CONSISTENT')
            vector=list(image_shift_to_projected_correction(best['dx_px'],best['dy_px'],grid.affine))
        else:margin=None
        recommended=vector if not reasons else None
        record={'region_id':cell['region_id'],'alignment_scope':plan['alignment_scope'],
            'diagnostic_vector_m':vector,'recommended_experiment_vector_m':recommended,'fit_peak_margin':margin,
            'diagnosis':diagnosis,'check_blocks':blocks,'reasons':reasons,'formal_ground_alignment_accepted':False}
        records.append(record);digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        regions.append(SourceRegion(cell['region_id'],'2018-11-01',shape(cell['support_geometry']),
            None if recommended is None else tuple(recommended),None if recommended is None else digest))
    summary={'run':out.name,'alignment_scope':plan['alignment_scope'],'source_roof_prior_pixels':int(movmask.sum()),
        'reference_roof_prior_pixels':int(refmask.sum()),'reference_prior_overlap_conflict_pixels':int(conflict.sum()),
        'numeric_fit_cells':sum(r['diagnostic_vector_m'] is not None for r in records),
        'internally_supported_roof_image_cells':sum(r.correction_m is not None for r in regions),
        'records':records,'experimental_label_pixels':0,'formal_ground_accepts':0,'measured_gt_pixels':0}
    if summary['internally_supported_roof_image_cells']:
        cache=parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
        with fiona.open(cache) as s:objects={f.properties['component_key']:shape(f.geometry) for f in s}
        result=build_fraction_shadow(objects,regions,grid)
        pseudo=experimental_pseudolabel(result['candidate_fraction'].astype(np.float32).astype(np.float64),
            result['scope_fraction'][1].astype(np.float32).astype(np.float64),result['quality'],np.ones((256,256),bool),
            allow_unverified_geometry=True,allow_assumed_background=True)
        write(out,'2019_ROOF_IMAGE_ALIGNED_FRACTION.tif',result['candidate_fraction'],grid,['ROOF_IMAGE_ALIGNED_FRACTION'],nodata=NODATA)
        write(out,'2019_ROOF_IMAGE_EXPERIMENTAL_LABEL.tif',pseudo['label'],grid,['EXPERIMENTAL_LABEL'],nodata=255)
        write(out,'2019_ROOF_IMAGE_EXPERIMENTAL_WEIGHT.tif',pseudo['experiment_weight'],grid,['EXPERIMENT_SELECTION'])
        write(out,'2019_ROOF_IMAGE_ALIGNMENT_QUALITY.tif',result['quality'],grid,QUALITY_BANDS)
        save_json(out/'ROOF_IMAGE_COMPONENT_LINEAGE.json',{'records':result['component_ledger'],'role':'ROOF_IMAGE_CANDIDATE_NOT_GROUND_FOOTPRINT'})
        summary['experimental_label_pixels']=pseudo['counts']['experimental_samples']
        summary['area_conservation_error_m2']=result['area_conservation_error_m2']
    save_json(out/'ROOF_IMAGE_ALIGNMENT_REPLAY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Protected input changed: '+path)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
