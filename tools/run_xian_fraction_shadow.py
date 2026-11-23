"""Frozen TRAINROI fraction shadow from native teacher pixels, not preview RGB."""
from pathlib import Path
import argparse, hashlib, json, math, sys, zipfile
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from rasterio.io import MemoryFile
import fiona
import shapely
from shapely.geometry import shape, mapping, box
from shapely.ops import transform
from pyproj import Transformer
from affine import Affine
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.label_transfer import Grid,export_raster
from xian_r5.source_partition import SourceRegion
from xian_r5.fraction_shadow import (build_fraction_shadow,compare_fraction_probes,
    QUALITY_BANDS,QUALITY_CODES,NODATA)


def write_layer(out,name,values,grid,bands,*,nodata=None,tags=None):
    a=np.asarray(values)
    if a.ndim==2:a=a[None]
    export_raster(out/name,a,grid,bands,root=out,nodata=nodata)
    with rasterio.open(out/name,'r+') as s:
        s.update_tags(PRODUCT_ROLE='FRACTION_SHADOW_DIAGNOSTIC_NOT_GT',
            TRAINING_USE_PROHIBITED='YES',FORMAL_CORRECTION_ACCEPTED='NO',
            ABSOLUTE_POSITION_ACCURACY='UNKNOWN',GROUND_FOOTPRINT='NOT_ESTABLISHED',
            DENOMINATOR='FULL_100_M2_TARGET_PIXEL',SOURCE_ZERO_IS_KNOWN_BACKGROUND='NO',
            **(tags or {}))
    with rasterio.open(out/name) as s:
        expected=a.astype(np.float32) if a.dtype.kind=='f' else a
        assert np.array_equal(s.read(),expected) and s.transform==grid.affine
        assert str(s.crs)==grid.crs and list(s.descriptions)==bands
        assert s.tags()['TRAINING_USE_PROHIBITED']=='YES'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment')
    out=args.output.resolve();base=parent/'TRAINROI_20261012_01';prev=parent/'TRAINROI_20261012_03'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh isolated shadow directory required')
    receipt=json.loads((prev/'PREVIEW_DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(receipt['protected_verified_sha256'])
    protected.update(receipt['code_sha256'])
    protected.update({str(prev/name):h for name,h in receipt['output_sha256'].items()})
    protected[str(prev/'PREVIEW_DELIVERY_RECEIPT.json')]='828c3b7cc05e14a38f363ad1a0cc176d8b8e39502c86a2dcfb467a5c41ed9b15'
    cfg=json.loads((base/'FROZEN_RUN_CONFIG.json').read_text(encoding='utf-8'))
    field=json.loads((base/'SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))
    if len(field['cells'])!=16 or field['numeric_candidate_count']!=5:raise ValueError('Frozen scope drift')
    history=parent.parent/'20_gt10m_label_transfer'
    b04=history/'B04_20261011_02'
    b04_receipt=json.loads((b04/'FINAL_DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))
    b04_manifest=json.loads((b04/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    bind_path=history/'B04_20261011_05/TRAINING_BUNDLE_BINDING_AUDIT.json'
    binding=json.loads(bind_path.read_text(encoding='utf-8'))
    for path in (bind_path,b04/'FINAL_DELIVERY_RECEIPT.json',b04/'INPUT_MANIFEST.json'):
        protected[str(path)]=sha(path)
    for path,expected in protected.items():
        if sha(path)!=expected:raise ValueError('Frozen baseline drift: '+path)
    raw_hashes={cfg['teacher']:cfg['teacher_sha256'],cfg['reference']:cfg['reference_sha256'],
        cfg['date_ledger']:cfg['date_ledger_sha256'],
        'F:/xian/10m结果.zip':'cd90fde63efdf1f5cbb30f0bb53e2dba5cfa14cf6c579438ccce9c93d0e7fc10'}
    for path,expected in raw_hashes.items():
        if sha(path)!=expected:raise ValueError('Source/hash drift: '+path)
    grid=Grid(Affine(*cfg['roi_affine']),256,256,'EPSG:32649')
    roi=box(*cfg['roi_bounds'])
    old_grid=b04_manifest['grid']
    old_support=shape(b04_manifest['support_geometry'])
    scope_audit={'legacy_object_count':b04_receipt['original_objects'],
        'legacy_grid':old_grid,'legacy_source_path':b04_manifest['objects_path'],
        'legacy_training_roi_intersection_m2':old_support.intersection(roi).area,
        'legacy_objects_used_here':False}
    if scope_audit['legacy_object_count']!=57 or scope_audit['legacy_training_roi_intersection_m2']>1e-6:
        raise ValueError('Legacy 57-object scope assumptions changed')
    old_label_record=next(r for r in binding['rasters'] if r['member'].endswith('/2019_pseudo_label_R90_tau43.tif'))
    with zipfile.ZipFile('F:/xian/10m结果.zip') as archive:
        members=[name for name in archive.namelist() if name.endswith('/2019_pseudo_label_R90_tau43.tif')]
        if len(members)!=1:raise ValueError('Old binary member not unique')
        old_bytes=archive.read(members[0])
    if hashlib.sha256(old_bytes).hexdigest()!=old_label_record['sha256']:
        raise ValueError('Old binary bytes drift')
    with MemoryFile(old_bytes) as memory, memory.open() as s:
        if s.shape!=(256,256) or s.transform!=grid.affine or str(s.crs)!=grid.crs:
            raise ValueError('Old binary grid drift')
        old_binary=s.read(1);old_tags=s.tags()
        if old_tags.get('THRESHOLD')!='0.43':raise ValueError('Old threshold drift')
    plan={'run':out.name,'stage':'GT10m Fraction Builder SHADOW ONLY',
        'roi_bounds':cfg['roi_bounds'],'grid_affine':cfg['roi_affine'],'shape':[256,256],'crs':grid.crs,
        'source_teacher':cfg['teacher'],'source_hashes':raw_hashes,'source_actual_date':cfg['source_actual_date'],
        'reference_actual_time':'UNKNOWN','protected_sha256':protected,'legacy_57_object_scope_audit':scope_audit,
        'geometry_basis':'Native teacher positive raster-cell union, 4-connected components; no smoothing or snapping to reference',
        'source_crop_halo_m':20,'geographic_edge_densification_max_native_pixels':32,
        'native_polygon_projection_is_piecewise_linear':True,
        'candidate_vectors_reestimated':False,'source_regions_are_camera_acquisitions':False,
        'candidate_scope':'Frozen 5 numeric / 11 unknown cells; clip each source BEFORE translation and clip same destination cell',
        'seam_crossing_component_policy':'Fragment DIAGNOSTIC only; flag full affected raw/candidate component footprint; no production transport',
        'boundary_strip_m':10,'boundary_strip_is_measured_error_bound':False,
        'fraction_denominator_m2':100,'source_zero_semantics':'MISSING_AND_BACKGROUND_NOT_SEPARABLE',
        'threshold_probe':{'tau':.43,'comparison':'> STRICT','threshold_fitting':False,'not_adopted_as_training_label':True},
        'numerical_unchanged_tolerance':1e-6,'diagnostic_difference_cuts':[.10,.25],
        'difference_cuts_are_admission_rules':False,'area_conservation_tolerance_m2':.001,
        'old_binary_member':old_label_record,'training_weight':0,'training_eligible_pixels':0,
        'quality_bands':QUALITY_BANDS,'quality_codes':QUALITY_CODES,
        'formal_state':receipt['formal_state']}
    out.mkdir(parents=True);save_json(out/'FROZEN_FRACTION_PLAN.json',plan)
    print('Frozen native-pixel fraction plan; reading bounded source window.',flush=True)
    with rasterio.open(cfg['teacher']) as s:
        bounds=transform_bounds(grid.crs,s.crs,*roi.buffer(20).bounds,densify_pts=32)
        points=[(~s.transform)*(x,y) for x,y in ((bounds[0],bounds[1]),(bounds[2],bounds[3]))]
        c0=math.floor(min(p[0] for p in points))-2;r0=math.floor(min(p[1] for p in points))-2
        c1=math.ceil(max(p[0] for p in points))+2;r1=math.ceil(max(p[1] for p in points))+2
        if c0<0 or r0<0 or c1>s.width or r1>s.height:raise ValueError('Native source window outside evidence')
        window=Window(c0,r0,c1-c0,r1-r0);native=s.read(1,window=window,masked=False)
        if not np.isin(native,[0,1]).all():raise ValueError('Unexpected teacher mask codes')
        native_affine=s.window_transform(window);native_crs=str(s.crs);source_tags=s.tags()
        step=32*min(abs(s.transform.a),abs(s.transform.e))
    projector=Transformer.from_crs(native_crs,grid.crs,always_xy=True)
    objects={};native_hashes={};component_ordinal=0
    for coordinates,value in shapes(native,mask=native==1,connectivity=4,transform=native_affine):
        component_ordinal+=1
        g=shape(coordinates)
        if not g.is_valid:raise ValueError('Invalid native component; no silent repair')
        dense=shapely.segmentize(g,step)
        projected=transform(projector.transform,dense)
        if not projected.is_valid:raise ValueError('Invalid projected component; no silent repair')
        if projected.intersection(roi).area<=1e-6:continue
        key=f'RASTER_COMPONENT_{component_ordinal:06d}'
        objects[key]=projected;native_hashes[key]=hashlib.sha256(g.wkb).hexdigest()
    print(json.dumps({'native_window':list(window.flatten()),'shape':native.shape,
        'positive_components_intersecting_roi':len(objects),'physical_ids_created':0}),flush=True)
    schema={'geometry':'MultiPolygon','properties':{'component_key':'str:48','native_sha256':'str:64',
        'source_actual_date':'str:10','physical_id_status':'str:24'}}
    with fiona.open(out/'SOURCE_MASK_COMPONENTS_SHADOW.gpkg','w',driver='GPKG',
                    layer='native_positive_components',crs=grid.crs,schema=schema) as sink:
        for key,g in objects.items():
            geometry=shapely.MultiPolygon([g]) if g.geom_type=='Polygon' else g
            sink.write({'geometry':mapping(geometry),'properties':{'component_key':key,
                'native_sha256':native_hashes[key],'source_actual_date':cfg['source_actual_date'],
                'physical_id_status':'NOT_ESTABLISHED'}})
    regions=[]
    for cell in field['cells']:
        if cell['formal_acceptance']:raise ValueError('Frozen shadow has unexpected acceptance')
        v=cell['candidate_correction_m']
        digest=hashlib.sha256(json.dumps(cell,sort_keys=True).encode()).hexdigest()
        regions.append(SourceRegion(cell['region_id'],cell['actual_source_date'],shape(cell['support_geometry']),
            None if v is None else tuple(v),None if v is None else digest))
    print('Calculating metric polygon intersections and full-pixel fractions.',flush=True)
    result=build_fraction_shadow(objects,regions,grid)
    probes=compare_fraction_probes(result,old_binary)
    common_tags={'SOURCE_TEACHER_SHA256':cfg['teacher_sha256'],'SOURCE_ACTUAL_DATE':cfg['source_actual_date'],
        'REFERENCE_SHA256':cfg['reference_sha256'],'REFERENCE_ACTUAL_TIME':'UNKNOWN',
        'SOURCE_FIELD_SHA256':sha(base/'SOURCE_CELL_CANDIDATES.json'),
        'FRACTION_SEMANTICS':result['fraction_semantics'],'GEOMETRY_BASIS':'NATIVE_TEACHER_RASTER_PIXEL_POLYGONS',
        'CANDIDATE_MODE':result['candidate_mode']}
    for name,key,description in [('ROOF_FRACTION_RAW.tif','raw_fraction','RAW_POSITIVE_ROOF_AREA_FRACTION'),
            ('ROOF_FRACTION_CORRECTED_CANDIDATE.tif','candidate_fraction','UNVERIFIED_CANDIDATE_POSITIVE_ROOF_AREA_FRACTION'),
            ('FRACTION_DIFFERENCE.tif','difference','CANDIDATE_MINUS_RAW_FRACTION')]:
        write_layer(out,name,result[key],grid,[description],nodata=NODATA,tags=common_tags)
    write_layer(out,'QUALITY.tif',result['quality'],grid,QUALITY_BANDS,
        tags={**common_tags,'QUALITY_CODES':json.dumps(QUALITY_CODES),
              'VERIFIED_DATE_ELIGIBLE_ZERO_SEMANTICS':'0=NOT_ESTABLISHED_OR_NOT_ELIGIBLE; NOT PROOF OF INCOMPATIBILITY'})
    write_layer(out,'SOURCE_OBJECT_COUNT.tif',result['component_count'],grid,
        ['RAW_CONNECTED_COMPONENT_COUNT','CANDIDATE_PARENT_COMPONENT_COUNT'],
        tags={**common_tags,'COUNT_SEMANTICS':'Raster connected components, not buildings or physical IDs'})
    write_layer(out,'SOURCE_SCOPE_FRACTION.tif',result['scope_fraction'],grid,
        ['RAW_GEOMETRIC_SOURCE_SCOPE_FRACTION','TRANSPORTED_GEOMETRIC_SOURCE_SCOPE_FRACTION'],
        tags={**common_tags,'SCOPE_IS_OBSERVABILITY_OR_LABEL_VALIDITY':'NO'})
    write_layer(out,'THRESHOLD_043_FLIP_PROBE.tif',probes['probe_codes'],grid,['STRICT_043_RAW_CANDIDATE_PROBE'],
        nodata=255,tags={**common_tags,'CODES':'0=both0;1=raw1candidate0;2=raw0candidate1;3=both1;255=unknown_or_partial_scope',
                       'THIS_IS_TRAINING_LABEL':'NO'})
    diagnostic=np.full((256,256),255,np.uint8);full=result['full_compare']
    delta=np.abs(result['difference'])
    diagnostic[full]=3;diagnostic[full&(delta<=.25)]=2
    diagnostic[full&(delta<=.10)]=1;diagnostic[full&(delta<=1e-6)]=0
    write_layer(out,'FRACTION_STABILITY_DIAGNOSTIC.tif',diagnostic,grid,['RAW_CANDIDATE_TWO_POSITION_DIAGNOSTIC'],
        nodata=255,tags={**common_tags,'CODES':'0=numerically_unchanged;1=absdelta<=0.10;2=absdelta<=0.25;3=absdelta>0.25;255=unknown_or_partial',
            'VALIDATED_ERROR_ENVELOPE':'NOT_ESTABLISHED','STABLE_PIXELS_ARE_ELIGIBLE':'NO'})
    write_layer(out,'OLD_BINARY_COMPARISON_DIAGNOSTIC.tif',probes['old_mismatch'],grid,
        ['OLD_VS_RAW_043_PROBE_MISMATCH','OLD_VS_CANDIDATE_043_PROBE_MISMATCH'],nodata=255,
        tags={**common_tags,'OLD_BINARY_SHA256':old_label_record['sha256'],
            'CODES':'0=probe_equal;1=probe_different;255=not_comparable','DIFFERENCE_CAUSE_IDENTIFIED':'NO'})
    write_layer(out,'TRAINING_WEIGHT_ZERO.tif',np.zeros((256,256),np.float32),grid,['TRAINING_WEIGHT'],tags=common_tags)
    per_cell=[]
    for cell in field['cells']:
        rr,cc=(int(s[1:]) for s in cell['region_id'].split('_')[1:]);sl=np.s_[rr*64:(rr+1)*64,cc*64:(cc+1)*64]
        mask=np.zeros((256,256),bool);mask[sl]=True;eligible=result['full_compare']&mask
        core=result['core_compare']&mask
        per_cell.append({'region_id':cell['region_id'],'candidate_m':cell['candidate_correction_m'],
            'full_comparison_pixels':int(eligible.sum()),'core_comparison_pixels':int(core.sum()),
            'abs_difference_gt_0_10_pixels':int((eligible&(delta>.10)).sum()),
            'strict_0_43_flips':int((eligible&np.isin(probes['probe_codes'],[1,2])).sum()),
            'numerically_unchanged_pixels':int((eligible&(delta<=1e-6)).sum()),
            'formal_accepted':False})
    summary={**probes['statistics'],'per_cell':per_cell,
        'quality_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(result['quality'][0],return_counts=True))},
        'native_component_count':len(objects),'source_component_ordinal_count_in_crop':component_ordinal,
        'native_window':list(window.flatten()),'native_shape':list(native.shape),
        'native_crop_sha256':hashlib.sha256(native.tobytes()).hexdigest(),
        'native_affine':list(native_affine)[:6],'native_crs':native_crs,'source_teacher_tags':source_tags,
        'raw_recorded_roof_area_m2':float(result['raw_fraction'].sum()*100),
        'candidate_recorded_roof_area_m2':float(result['candidate_fraction'][result['candidate_fraction']!=NODATA].sum()*100),
        'candidate_unknown_pixels':int((result['candidate_fraction']==NODATA).sum()),
        'seam_risk_pixels':int(result['quality'][6].sum()),'boundary_risk_pixels':int(result['quality'][5].sum()),
        'overlap_conflict_pixels':int(result['quality'][7].sum()),
        'area_conservation_error_m2':result['area_conservation_error_m2'],
        'legacy_57_objects_not_used':True,'trained_model_or_predictions_used':False,
        'training_eligible_pixels':0,'candidate_vectors_fitted':0,
        'production_registered_roof_instances':0,'independent_geometric_validation':'NOT_ESTABLISHED',
        'temporal_compatibility':'UNKNOWN','formal_state':receipt['formal_state']}
    save_json(out/'FRACTION_DIAGNOSTICS.json',summary)
    save_json(out/'COMPONENT_SOURCE_LINEAGE.json',{'component_records':result['component_ledger'],
        'source_teacher':cfg['teacher'],'native_component_sha256':native_hashes,
        'physical_ids_created':0,'complete_building_instance_transport':False})
    for path,expected in {**protected,**raw_hashes}.items():
        if sha(path)!=expected:raise RuntimeError('Protected source changed during fraction build: '+path)
    print(json.dumps({'run':out.name,'components':len(objects),'statistics':probes['statistics'],
        'quality_counts':summary['quality_counts'],'conservation':result['area_conservation_error_m2'],
        'protected_hashes_verified':len(protected)+len(raw_hashes),'training_gt_pixels':0},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
