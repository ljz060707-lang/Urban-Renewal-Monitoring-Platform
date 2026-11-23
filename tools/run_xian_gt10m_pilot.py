"""Reproducible, append-only B04 shadow pilot. Never writes protected sources."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
ROOT = Path(r'F:\xian\XIAN_R5_MODEL_UPGRADE_20260923')
LEGACY = ROOT/'19_ground_registration_evidence'
BASE = LEGACY/'B04_20261007_02'
OUTPUT = ROOT/'20_gt10m_label_transfer'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    p=Path(path); before=p.stat(); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    after=p.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError('Input changed during hashing: '+str(p))
    return h.hexdigest()


def register_bytes(path, raw):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=raw: raise ValueError('Append-only artifact differs: '+str(path))
        return
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.partial_')
    try:
        with os.fdopen(fd,'wb') as f: f.write(raw)
        os.link(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)


def write_json(path, data):
    raw=(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
    register_bytes(path,raw)


def verify_completed(run):
    manifest_path=run/'OUTPUT_SHA256_MANIFEST.json'
    receipt=read(run/'IMPLEMENTATION_RECEIPT.json')
    if sha(manifest_path)!=receipt['manifest_sha256']:
        raise RuntimeError('Receipt/manifest binding mismatch')
    manifest=read(manifest_path)
    for p,h in manifest['outputs'].items():
        if sha(run/p)!=h: raise RuntimeError('Completed run output drift: '+p)
    for p,h in manifest['code_sha256'].items():
        if sha(p)!=h: raise RuntimeError('Completed run implementation drift: '+p)


def verify_ledger(path, manifest, tiles):
    import fiona
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import shape,Polygon
    if set(fiona.listlayers(path))!={'source_regions','download_tiles','roof_objects'}:
        raise ValueError('Incomplete ledger layers')
    with fiona.open(path,layer='source_regions') as s:
        support=shape(manifest['support_geometry'])
        expected_regions={}
        for actual in ('2018-04-29','2018-11-01','2019-12-02'):
            key='ER_'+actual.replace('-','')+'_B04_CONTEXT'
            expected_regions[key]=(actual,'EMPIRICAL_REGISTRATION_REGION','UNKNOWN','M1','null')
        for branch,t in manifest['transforms'].items():
            expected_regions[manifest['region_id']+':'+branch]=('2019-12-02',t['status'],'UNKNOWN',t['model'],json.dumps(t['vector']))
        observed={}
        for row in s:
            p=row.properties;key=p['region_id']
            observed[key]=(p['actual_date'],p['status'],p['provider'],p['model'],p['vector'])
            if not shape(row.geometry).equals(support): raise ValueError('Source/transform support geometry drift')
        if s.crs.to_epsg()!=32649 or len(s)!=len(expected_regions) or observed!=expected_regions:
            raise ValueError('Incomplete or changed source/transform records')
    expected={r['source_path']:(r['source_sha256'],r['actual_date']) for r in tiles}
    with fiona.open(path,layer='download_tiles') as s:
        observed={r.properties['path']:(r.properties['sha256'],r.properties['actual_date']) for r in s}
        if len(s)!=len(expected) or observed!=expected: raise ValueError('Incomplete tile provenance')
        if s.crs.to_epsg()!=32649: raise ValueError('Tile frame drift')
        native=Affine(*manifest['coordinate_audit']['native_affine'])
        tf=Transformer.from_crs(4326,manifest['grid']['crs'],always_xy=True)
        for row in s:
            p=row.properties;parts=Path(p['path']).stem.split('_');x=int(parts[1]);y=int(parts[2])
            if p['tile_id']!=f'{x}:{y}:20' or p['download_batch']!=Path(p['path']).parents[2].name:
                raise ValueError('Tile identity drift')
            corners=(((x-420797)*256,(312176-y)*256),((x-420796)*256,(312176-y)*256),
                     ((x-420796)*256,(312177-y)*256),((x-420797)*256,(312177-y)*256))
            geo=Polygon([tf.transform(*(native @ corner)) for corner in corners])
            if not shape(row.geometry).equals(geo): raise ValueError('Tile support geometry drift')
    expected_roofs={r['id']:shape(r['geometry']) for r in manifest['objects']}
    with fiona.open(path,layer='roof_objects') as s:
        roofs={r.properties['source_id']:shape(r.geometry) for r in s}
        if len(s)!=len(expected_roofs) or roofs.keys()!=expected_roofs.keys():
            raise ValueError('Incomplete roof objects')
        if any(not geo.equals(expected_roofs[key]) for key,geo in roofs.items()):
            raise ValueError('Frozen object geometry mismatch')


def output_inventory(run):
    return {p.name:sha(p) for p in run.iterdir() if p.is_file() and not p.name.startswith('.partial_')
            and p.name not in ['OUTPUT_SHA256_MANIFEST.json','IMPLEMENTATION_RECEIPT.json']}


def audit(run):
    """Freeze input coverage before metrics; verify every inherited frozen hash."""
    import fiona
    import rasterio
    from affine import Affine
    from PIL import Image
    from pyproj import Transformer
    from shapely.geometry import Polygon, mapping, shape
    from shapely.ops import transform
    from xian_r5.label_transfer import Grid
    expected=read(BASE/'12_qa/INPUT_HASH_AUDIT_FINAL.json')['expected_hashes']
    legacy_outputs={str(p):sha(p) for p in BASE.rglob('*') if p.is_file()}
    for p,h in expected.items():
        if sha(p)!=h: raise RuntimeError('Frozen input mismatch: '+p)
    print('verified inherited sources:',len(expected),flush=True)
    target=next(p for p in expected if 'Xian_2019_TrueColor.tif' in p)
    native=read(BASE/'02_raw_imagery/NATIVE_GRID_PROVENANCE.json')
    with rasterio.open(target) as s:
        target_info=dict(path=target,sha256=sha(target),crs=str(s.crs),
                         affine=list(s.transform)[:6],width=s.width,height=s.height,
                         bands=list(s.descriptions),dtypes=list(s.dtypes),nodata=s.nodata,
                         tags=s.tags(),actual_date=None,date_window=None,
                         date_status='ANNUAL_LABEL_ONLY',cloud_observability='UNKNOWN')
        target_grid=Grid(s.transform,s.width,s.height,str(s.crs))
    p=native['pixel_grid'];d=p['degrees_per_pixel']
    native_affine=Affine(d,0,p['west_lon'],0,-d,p['north_lat'])
    to_target=Transformer.from_crs(4326,target_info['crs'],always_xy=True)
    # Densify each footprint edge, rather than projecting a bounding box corner only.
    corners=[(0,0),(1024,0),(1024,1024),(0,1024),(0,0)]
    ring=[]
    for start,end in zip(corners,corners[1:]):
        for i in range(33):
            t=i/32; ring.append(native_affine @ (start[0]+t*(end[0]-start[0]),start[1]+t*(end[1]-start[1])))
    coverage=transform(to_target.transform,Polygon(ring))
    grid=target_grid.window(coverage.bounds)
    for t in native['tiles']:
        with Image.open(t['path']) as im:
            if im.size!=(256,256): raise ValueError('Non-native JPEG dimensions')
    gt_path=LEGACY/'B04_20261006_01/02_a13_gt_clip/B04_A13_MANUAL_ROOF_GT.gpkg'
    objects=next(p for p in expected if 'remote_full_geometry' in p and '2020-12-31' in p)
    with fiona.open(objects,layer='buildings') as s:
        if s.crs.to_epsg()!=32649: raise ValueError('Object metric frame mismatch')
        rows=[]
        for f in s.filter(bbox=coverage.buffer(20).bounds):
            if f.geometry:
                g=shape(f.geometry)
                if g.intersects(coverage.buffer(20)):
                    if not g.is_valid: raise ValueError('Invalid source object')
                    rows.append(dict(id=str(f.id),geometry=mapping(g),properties=dict(f.properties)))
    refs=[]
    with fiona.open(gt_path,layer='b04_context_manual_roof_gt') as s:
        for f in s:
            g=shape(f.geometry)
            if g.intersects(coverage): refs.append(dict(geometry=mapping(g),properties=dict(f.properties)))
    coordinates=dict(affected_artifact=str(BASE/'02_raw_imagery/NATIVE_GRID_PROVENANCE.json')+':G06_full_GE_source_bounds',
        affected_writer=str(BASE/'00_audit/inspect_lineage.py')+':117',
        readers=[str(BASE/'00_audit/close_adjudication.py')+':115 (corrected audit assertion only)'],
        unaffected=['raw JPEGs','native crop (local pixel bounds)','metric roof WKB','GT overlay','residual vectors','R4 state TIF'],
        old_offset=29184,correct_offset=28928,delta_pixels=256,
        proof='Call-site search plus exact source affine/pixel reconstruction; no scientific consumer reads incorrect full-source field.',
        shared_interface='xian_r5.alignment.mosaic_grid.MosaicTileGrid.window_for_tile',
        direction='GEID Y increases NORTH; not standard web XYZ',
        corrected_derivation='(max_tile_y - tile_y) * native_height; inclusive max from source grid',
        full_source_pixel_origin=[67328,28928],native_affine=list(native_affine)[:6])
    created=(read(run/'INPUT_MANIFEST.json')['created_at'] if (run/'INPUT_MANIFEST.json').is_file()
             else datetime.now().astimezone().isoformat())
    config=dict(schema='gt10m.shadow.v1',run_id=run.name,created_at=created,
        branch='GT10M_LABEL_TRANSFER_BRANCH',task='ROOF_PLANIMETRIC_PROJECTED_BUILDING_STATE',
        scope='B04_A13_16_NATIVE_TILE_CONTEXT; not full B04/city',scope_freeze_basis='existing 4x4 raw evidence coverage, before new fraction/stress metrics',
        target=target_info,grid=dict(affine=list(grid.affine)[:6],width=grid.width,height=grid.height,crs=grid.crs),
        support_geometry=mapping(coverage),region_id='ER_20191202_B04_NATIVE_CONTEXT',
        source_component_status='EMPIRICAL_REGISTRATION_REGION; provider acquisition grouping UNKNOWN',
        native_tiles=native['tiles'],objects_path=objects,objects=rows,manual_reference_path=str(gt_path),references=refs,
        object_denominator=len(rows),G06_id='75303',stress_amplitudes_m=[0,1,2,3,5,10],stress_directions_deg=list(range(0,360,45)),
        admission=dict(version='GT10M_POLICY_V1',min_valid_fraction=.95,max_bandwidth=.10,
                       admission_envelope_m=None,hard_semantics=None,temporal_compatibility='UNKNOWN'),
        transforms={'A':dict(model='M0',vector=[0.,0.],status='GRID_ONLY_NO_EMPIRICAL_CORRECTION'),
                    'B':dict(model='OLD_FAILED_COMMON_SHIFT',vector=[10.383335115861948,2.907954733698325],status='SENSITIVITY_BASELINE_ONLY'),
                    'C':dict(model='M1',vector=None,status='NOT_ESTIMATED_NO_INDEPENDENT_CALIBRATION_VALIDATION_ANCHORS'),
                    'D':dict(model='M0_PLUS_RISK',vector=[0.,0.],status='DIAGNOSTIC_ONLY_M1_NULL_NOT_COERCED_TO_ZERO')},
        coordinate_audit=coordinates,student_predictions=dict(status='NOT_PROVIDED',role='STUDENT_PREDICTION',construction_reads=0),
        manual_reference_semantics='2019 user-confirmed unbiased roof outline; exact capture date and measured absolute error UNKNOWN; incomplete clipped feature collection not background truth',
        split_policy='No fitting or training. Reserve whole spatial region for diagnostic evaluation; selected six not independent generalization sample.',
        frozen_expected_hashes=expected,legacy_output_hashes=legacy_outputs,
        source_priority=['only one empirical GE region; actual date verified per tile, not provider acquisition'],
        uncertainties={k:None for k in ['registration_m','roof_shape_m','reference_m','source_seam_m','temporal_days','semantic_error']})
    if not any(r['id']=='75303' for r in rows): raise RuntimeError('G06 missing from original denominator')
    run.mkdir(parents=True,exist_ok=True)
    write_json(run/'INPUT_MANIFEST.json',config)
    write_json(run/'COORDINATE_AND_BASELINE_AUDIT.json',dict(coordinates=coordinates,
               inherited_hashes_verified=len(expected),legacy_outputs_frozen=len(legacy_outputs),
               target_grid=target_info,source_grid_audit='PARTIAL; native coordinate identity tested, source scene/absolute accuracy not established'))
    print(json.dumps(dict(run=str(run),objects=len(rows),reference_roofs=len(refs),grid=config['grid']),ensure_ascii=True),flush=True)


def produce(run):
    import math
    import fiona
    import numpy as np
    from affine import Affine
    from shapely.geometry import shape,mapping,Polygon
    from shapely.ops import unary_union
    from xian_r5.label_transfer import (Grid,aggregate,deduplicate,bounded_transport,
        export_raster,load_rasters,training_mask,validate_input_role)
    if (run/'IMPLEMENTATION_RECEIPT.json').is_file():
        verify_completed(run)
        print('Completed shadow run verified; no writes on resume',flush=True)
        return
    m=read(run/'INPUT_MANIFEST.json');g=m['grid']
    grid=Grid(Affine(*g['affine']),g['width'],g['height'],g['crs'])
    support=shape(m['support_geometry'])
    validate_input_role('OBSERVED_CANDIDATE');validate_input_role('MANUAL_ROOF_REFERENCE')
    objects=deduplicate((r['id'],shape(r['geometry'])) for r in m['objects'])
    roofs=unary_union(list(objects.values()))
    refs=unary_union([shape(r['geometry']) for r in m['references']])
    empty=Polygon()
    coverage=aggregate(grid,{'coverage':support},support)['valid']
    # Positive model geometry is a diagnostic candidate; empty space is NOT known background.
    a=aggregate(grid,{'OBSERVED_ROOF_CANDIDATE':roofs},support)
    reference=aggregate(grid,{'MANUAL_2019_ROOF_REFERENCE':refs},support)
    vectors=[(0.,0.)]+[(amp*math.cos(math.radians(angle)),amp*math.sin(math.radians(angle)))
              for amp in m['stress_amplitudes_m'][1:] for angle in m['stress_directions_deg']]
    residual_path=LEGACY/'B04_20261007_01/01_frozen_pair_ledger/FROZEN_PAIR_RESIDUAL_LEDGER.csv'
    with residual_path.open(encoding='utf-8-sig') as f: residuals=list(csv.DictReader(f))
    residual_vectors=[(float(r['residual_dx_m']),float(r['residual_dy_m'])) for r in residuals]
    vectors+=residual_vectors
    summaries={}; scenario_ledger=[];outputs={}
    common=m['transforms']['B']['vector']
    for branch,base in [('A',(0.,0.)),('B',common)]:
        values=[]
        for i,v in enumerate(vectors):
            shifted,valid=bounded_transport(objects,(base[0]+v[0],base[1]+v[1]),support,support)
            out=aggregate(grid,{'ROOF_CANDIDATE':unary_union(list(shifted.values()))},valid)
            values.append(out['fractions'][0])
            scenario_ledger.append(dict(branch=branch,index=i,region_vector_m=[base[0]+v[0],base[1]+v[1]],
                object_residuals={},scenario_kind='FIXED_ENGINEERING_STRESS' if i<41 else 'OLD_SELECTED_PAIR_RESIDUAL_AS_COHERENT_REGION_SCENARIO',
                empirical_error_bound=False))
            if i==0: outputs[branch]=out
        # The full G06 stays in every scenario; one extra scenario isolates its observed residual.
        g06=next(r for r in residuals if r['label']=='G06')
        ov={'75303':(float(g06['residual_dx_m']),float(g06['residual_dy_m']))}
        changed,valid=bounded_transport(objects,base,support,support,ov)
        values.append(aggregate(grid,{'ROOF':unary_union(list(changed.values()))},valid)['fractions'][0])
        scenario_ledger.append(dict(branch=branch,index=len(values)-1,region_vector_m=list(base),
            object_residuals=ov,scenario_kind='G06_OBJECT_COHERENT_RESIDUAL_DIAGNOSTIC',empirical_error_bound=False))
        stack=np.stack(values);lo=stack.min(axis=0);hi=stack.max(axis=0);width=hi-lo
        probe=stack>.5  # diagnostic probe only, NOT adopted hard-label task semantics.
        flips=probe.any(axis=0)!=probe.all(axis=0)
        summaries[branch]=dict(scenario_count=len(values),max_bandwidth=float(width.max()),
            scenario_majority_probe_flip_pixels=int(flips.sum()),
            scenario_majority_probe_flip_rate_over_full_window=float(flips.mean()),
            fraction_area_conservation_error_m2=outputs[branch]['area_conservation_error_m2'],
            candidate_roof_area_in_support_m2=float(outputs[branch]['fractions'].sum()*100),
            difference_to_manual_reference_positive_cells_mean_abs=float(np.abs(outputs[branch]['fractions'][0]-reference['fractions'][0])[reference['fractions'][0]>0].mean()),
            comparison_semantics='DIAGNOSTIC_POSITIVE_REFERENCE_AREA_DISCREPANCY; not held-out accuracy, complete-background truth or date-certified GT')
        export_raster(run/f'{branch}_ROOF_SENSITIVITY.tif',np.stack([lo,hi,width,flips]),grid,
            ['fraction_min','fraction_max','fraction_bandwidth','diagnostic_majority_probe_flip'],root=run)
        if branch=='A': bandwidth=width;flip=flips
        print('branch',branch,'scenarios',len(values),flush=True)
    shape0=(grid.height,grid.width);zeros=np.zeros(shape0);ones=np.ones(shape0)
    training=training_mask(zeros,bandwidth,envelope=None,temporal=False)
    fractions=np.stack([zeros,zeros,zeros,zeros,zeros,ones])
    export_raster(run/'GT10M_FRACTIONS.tif',fractions,grid,
        ['STABLE','ADDITION','DEMOLITION','KNOWN_BACKGROUND','valid_fraction','unknown_fraction'],root=run)
    export_raster(run/'GT10M_HARD_LABEL.tif',np.full((1,*shape0),255,dtype=np.uint8),grid,
                  ['IGNORE_HARD_SEMANTICS_AND_DATE_UNQUALIFIED'],root=run,nodata=255)
    export_raster(run/'SINGLE_2019_ROOF_OCCUPANCY.tif',np.stack([a['fractions'][0],outputs['B']['fractions'][0],reference['fractions'][0],coverage]),grid,
        ['A_M0_observed_roof_candidate','B_old_failed_shift_diagnostic','manual_2019_roof_reference','source_pixel_coverage_NOT_LABEL_VALIDITY'],root=run)
    unknown=np.full(shape0,-9999.)
    quality=np.stack([zeros,ones,coverage,bandwidth,flip,ones,ones,ones,unknown,unknown,
                      np.where(coverage>0,2,3),training])
    quality_bands=['label_valid_fraction','label_unknown_fraction','source_pixel_coverage',
        'A_stress_fraction_bandwidth','A_diagnostic_majority_probe_flip','temporal_risk',
        'shape_uncertainty_unknown','source_acquisition_seam_uncertainty_unknown',
        'registration_bound_m_UNKNOWN','reference_bound_m_UNKNOWN','evidence_tier_2_DIAGNOSTIC_3_IGNORE','training_weight']
    export_raster(run/'GT10M_QUALITY.tif',quality,grid,quality_bands,root=run,nodata=-9999)
    export_raster(run/'GT10M_DATE_SOURCE_MAP.tif',np.stack([np.where(coverage>0,20191202,0),
        np.where(coverage>0,1,0),ones]),grid,['GE_actual_YYYYMMDD_0_outside','empirical_region_1_0_outside','target_date_unknown'],root=run)
    # A source region is not a provider scene; download tiles are separate ledger rows.
    packet=LEGACY/'B04_20261001_02/04_manual_review/A13_FIRST_PRIORITY_V1/tile_actual_date_and_hash_provenance.csv'
    with packet.open(encoding='utf-8-sig') as f: tiles=list(csv.DictReader(f))
    ledger_final=run/'SOURCE_AND_TRANSFORM_LEDGER.gpkg'
    if not ledger_final.exists():
        ledger_temp=tempfile.TemporaryDirectory(prefix='.partial_ledger_',dir=run)
        ledger_path=Path(ledger_temp.name)/'ledger.gpkg'
        with fiona.open(ledger_path,'w',driver='GPKG',layer='source_regions',crs=grid.crs,
            schema={'geometry':'Unknown','properties':{'region_id':'str','actual_date':'str','status':'str','provider':'str','model':'str','vector':'str','purpose':'str'}}) as dst:
            for actual in ('2018-04-29','2018-11-01','2019-12-02'):
                dst.write(dict(geometry=mapping(support),properties=dict(region_id='ER_'+actual.replace('-','')+'_B04_CONTEXT',
                    actual_date=actual,status='EMPIRICAL_REGISTRATION_REGION',provider='UNKNOWN',model='M1',vector='null',
                    purpose='NOT_ESTIMATED; no independent calibration/validation ground anchors')))
            for branch,t in m['transforms'].items():
                dst.write(dict(geometry=mapping(support),properties=dict(region_id=m['region_id']+':'+branch,actual_date='2019-12-02',
                    status=t['status'],provider='UNKNOWN',model=t['model'],vector=json.dumps(t['vector']),purpose='ROOF_LABEL_ONLY; shadow diagnostic')))
        with fiona.open(ledger_path,'w',driver='GPKG',layer='download_tiles',crs=grid.crs,
            schema={'geometry':'Unknown','properties':{'path':'str','sha256':'str','actual_date':'str','tile_id':'str','download_batch':'str','acquisition':'str'}}) as dst:
            for row in tiles:
                path=row['source_path'];h=row['source_sha256']
                if sha(path)!=h: raise ValueError('Tile hash mismatch')
                if row['date_conflict']!='0': raise ValueError('Conflicting source date')
                parts=Path(path).stem.split('_');x=int(parts[1]);y=int(parts[2])
                native=Affine(*m['coordinate_audit']['native_affine'])
                from pyproj import Transformer
                tf=Transformer.from_crs(4326,grid.crs,always_xy=True)
                poly=Polygon([tf.transform(*(native @ (c,r))) for c,r in
                    (((x-420797)*256,(312176-y)*256),((x-420796)*256,(312176-y)*256),
                     ((x-420796)*256,(312177-y)*256),((x-420797)*256,(312177-y)*256))])
                dst.write(dict(geometry=mapping(poly),properties=dict(path=path,sha256=h,actual_date=row['actual_date'],
                    tile_id=f'{x}:{y}:20',download_batch=Path(path).parents[2].name,acquisition='UNKNOWN; batch/date not camera scene')))
        with fiona.open(ledger_path,'w',driver='GPKG',layer='roof_objects',crs=grid.crs,
            schema={'geometry':'Unknown','properties':{'source_id':'str','actual_date':'str','source_path':'str','tier':'str','support_clip':'int','is_G06':'int'}}) as dst:
            for key,geo in objects.items():
                dst.write(dict(geometry=mapping(geo),properties=dict(source_id=key,actual_date='2019-12-02',source_path=m['objects_path'],
                    tier='DIAGNOSTIC_ONLY',support_clip=int(not support.covers(geo)),is_G06=int(key=='75303'))))
        verify_ledger(ledger_path,m,tiles)
        os.link(ledger_path,ledger_final)
        ledger_temp.cleanup()
    else:
        verify_ledger(ledger_final,m,tiles)
    loaded=load_rasters([run/'GT10M_FRACTIONS.tif',run/'GT10M_HARD_LABEL.tif',run/'GT10M_QUALITY.tif',run/'GT10M_DATE_SOURCE_MAP.tif',run/'SINGLE_2019_ROOF_OCCUPANCY.tif'])
    if np.count_nonzero(loaded['arrays'][2][-1])!=0: raise RuntimeError('Unqualified training weight nonzero')
    states=dict(LEGACY_B04_COMMON_SHIFT_GATE='FAILED_UNCHANGED',G06_CAUSAL_ATTRIBUTION='UNRESOLVED',
        SOURCE_GRID_AUDIT='PARTIAL',SOURCE_COMPONENT_LINEAGE='EMPIRICAL_ONLY',TEMPORAL_COMPATIBILITY='UNKNOWN',
        GT10M_GEOMETRIC_SENSITIVITY='MEASURED',GT10M_SILVER_SUPERVISION='NOT_DEMONSTRATED',
        GT10M_GOLD_EVALUATION_REFERENCE='NOT_ESTABLISHED',GT10M_MOSAIC_VALIDATION='SYNTHETIC_ONLY',
        GROUND_FOOTPRINT_RECOVERY='NOT_DEMONSTRATED',ABSOLUTE_POSITION_ACCURACY='UNKNOWN',CITYWIDE_PRODUCTION_RELEASE='NO')
    report=dict(schema='gt10m.evaluation.v1',run_id=run.name,states=states,
        branch=m['branch'],task=m['task'],denominators=dict(pixels=grid.width*grid.height,window_area_m2=grid.width*grid.height*100,
            source_area_m2=support.area,source_coverage_fraction=float(coverage.mean()),objects=len(objects),manual_reference_objects=len(m['references']),
            objects_intersecting_native_support=sum(g.intersects(support) for g in objects.values()),
            objects_intersecting_support_halo_only=sum(not g.intersects(support) for g in objects.values())),
        original_G06_preserved=True,trained_pixels=0,silver_pixels=0,gold_evaluation_pixels=0,
        exclusion_rate_over_original_pixel_denominator=1.,
        exclusion_reasons=['TARGET_ACTUAL_DATE_UNKNOWN','ADMISSION_ERROR_ENVELOPE_NOT_ESTABLISHED','HARD_LABEL_SEMANTICS_UNDEFINED','STATE_FRAME_AND_DATE_LINEAGE_UNQUALIFIED','SHAPE_AND_REFERENCE_ERROR_UNKNOWN'],
        formal_counts=dict(FORMAL_FIT_HOLDOUT_REVIEWS_COMPLETED=0,ACCEPTED_ALIGNMENT_CELLS=0,REJECTED_ALIGNMENT_CELLS=0,
            UNRESOLVED_ALIGNMENT_CELLS=9,REGISTERED_ROOFS=0,PHYSICAL_IDS=0,CANONICAL_GEOMETRIES=0,GROUND_FOOTPRINTS=0),
        shadow_counts=dict(roof_objects_aggregated=len(objects),M0_label_transports=1,failed_shift_sensitivity_transports=1,qualified_M1_transforms=0),
        branches={'A':summaries['A'],'B':summaries['B'],'C':dict(status='NOT_ESTIMATED',vector=None,
                    reason='Independent calibration/model-selection/validation anchors absent; no copying old g'),
                  'D':dict(status='DELIVERED_M0_FRACTIONS_RISK_IGNORE',qualification='REAL_ZERO_QUALIFIED_SUPERVISION')},
        uncertainty=dict(admission_envelope_m=None,actual_distribution='UNKNOWN',scenarios=scenario_ledger,
            finite_direction_claim='ONLY_SAMPLED_SCENARIOS; no continuous/95percent guarantee',
            shared_temporal_reference_error='UNKNOWN; no actual multi-date perturbation inferred',unmeasured=m['uncertainties']),
        geographic_validation=dict(independent_change_truth='NOT_ESTABLISHED',change_precision_recall_iou=None,
            stable_building_false_change_real=None,real_mosaic_seam_conflicts=None,
            student_comparison='NOT_RUN; absent input, forbidden construction use',
            actual_target_date_unknown_fraction=1.,provider_source_identity_unknown_fraction=1.),
        strata=dict(heights_occlusion='UNKNOWN; no invented height bins',
            area_lt100m2=sum(g.area<100 for g in objects.values()),area_ge100m2=sum(g.area>=100 for g in objects.values()),
            support_boundary_objects=sum(not support.covers(g) for g in objects.values()),all_object_supervision_retention=0.),
        dictionaries=dict(fractions=list(['STABLE','ADDITION','DEMOLITION','KNOWN_BACKGROUND','valid_fraction','unknown_fraction']),
            main_fraction_semantics='All states unqualified; class fractions=0, valid=0, unknown=1. RGB coverage is separate.',
            occupancy_semantics='Absolute candidate/reference roof area per 100m2 cell; no certified background, state, footprint or exact target-date truth.',
            quality=quality_bands,tier=dict(GOLD_REFERENCE=0,SILVER_CALIBRATED_LABEL=1,DIAGNOSTIC_ONLY=2,IGNORE=3),
            hard=dict(ignore=255,task_definition='PENDING'),date_source_map='integer date/empirical-region map; fractional source coverage in quality band3'),
        dataloader=dict(status='PASS',unmasked_read=True,rasters=len(loaded['arrays']),grid_exact_match=True),
        area_conservation=dict(A_m2=a['area_conservation_error_m2'],B_m2=outputs['B']['area_conservation_error_m2'],reference_m2=reference['area_conservation_error_m2']),
        not_run=['Physical footprint recovery','New M1/M2 fitting','Real temporally comparable change supervision','RBF-SAM inference','Citywide release','Large-model training'])
    write_json(run/'EVALUATION_AND_LIMITATIONS.json',report)
    write_json(run/'SOURCE_AND_TRANSFORM_LEDGER_METADATA.json',dict(task=m['task'],
        forward='x_target=x_source+vector_m; B roof labels only',inverse_sampling='inverse T for rasters; no RGB empirical warp performed',
        source_regions='Empirical supports, NOT known camera acquisitions',compute_chunk='one pilot window; synthetic chunk tests separate',
        support_clip='20mhalo objects are clipped to verified SOURCE support before translation. Source validity translates with labels then intersects frozen TARGET support. Boundary fragments remain diagnostic.',
        source_tile_rows=len(tiles),null_transform='UNKNOWN not zero',original_state_transport='NOT_USED; mixed nominal date/frame lineage',
        resume='Raster exports byte-preserving if equal; changed content fail closed. GPKG only created once; completion receipt verifies file hashes.',
        reference_roles='Manual outlines used only for geometry reference comparison, not observed-candidate construction or parameter selection'))
    print(json.dumps(dict(states=states,pixels=grid.width*grid.height,supervision_pixels=0),ensure_ascii=True),flush=True)


def reference_audit(run):
    """Read official versioned text only. No installers, checkpoints or models."""
    import subprocess
    import urllib.request
    records=[]
    for repo in ['mattpocock/skills','DietrichGebert/ponytail','Xuaner624/RBF-SAM',
                 'GFZ/arosics','fitzpchao/BANDON','lauraset/FFCTL','Henryjiepanli/UA-BCD','likaiucas/OBM','likaiucas/PolyFootNet']:
        r=subprocess.run(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],capture_output=True,text=True,timeout=25)
        commit=r.stdout.split()[0] if r.returncode==0 and r.stdout.split() else None
        row=dict(repository='https://github.com/'+repo,commit=commit,status='VERSION_VERIFIED' if commit else 'VERSION_NOT_OBTAINED',files=[])
        paths=['README.md']
        if repo=='mattpocock/skills': paths=['skills/engineering/'+name+'/SKILL.md' for name in ['codebase-design','tdd','code-review','diagnosing-bugs']]
        if repo=='DietrichGebert/ponytail': paths=['skills/ponytail/SKILL.md']
        if commit:
            for path in paths:
                url='https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+path
                try:
                    raw=urllib.request.urlopen(url,timeout=20).read()
                    item=dict(path=path,url=url,sha256=hashlib.sha256(raw).hexdigest(),status='READ_ONLY_FETCHED')
                    if repo=='mattpocock/skills':
                        local=Path(r'C:\Users\ljz52\.codex\skills')/path.removeprefix('skills/engineering/')
                        item.update(local_path=str(local),local_sha256=sha(local),local_matches_official=sha(local)==item['sha256'])
                    row['files'].append(item)
                except Exception as exc:
                    row['files'].append(dict(path=path,status='UNAVAILABLE',reason=str(exc)))
        records.append(row)
    rbf=PROJECT/'third_party_reference/RBF-SAM'
    read_code=['README.md','tools/test_rbf_sam_bonai.py','tools/analyse_offset_bonai.py','tools/visual_offset_bonai.py',
        'mmdet/models/detectors/rbf_sam.py','mmdet/core/bbox/coder/delta_xy_offset_coder.py',
        'configs/rbf_sam/rbf_sam_bonai.py','configs/_base_/datasets/bonai_robust.py','mmdet/datasets/pipelines/transforms.py']
    local_inputs={str(rbf/p):sha(rbf/p) for p in read_code}
    local_inputs.update({str(PROJECT/p):sha(PROJECT/p) for p in ['docs/ml_architecture_review_20260910/18_INTERFACE_CONTRACTS.md','docs/ml_architecture_review_20260910/22_ARCHITECTURE_DECISION_RECORDS.md','docs/XIAN_ADDITIVE_POST_INFERENCE.md']})
    write_json(run/'REFERENCE_AND_ENGINEERING_AUDIT.json',dict(official_sources=records,local_read_hashes=local_inputs,
        mode='PONYTAIL_FULL',installers_run=False,repo_agent_context='user-supplied CodeGraph instructions; no .codegraph/local root AGENTS/CONTEXT/GLOSSARY file found',
        code_review_baseline='FILE_BASED_NEW_ADDITIONS; empty .git metadata, no valid HEAD. No Git history review claimed.',
        RBF_SAM=dict(recorded_commit='30204da5c9edd1f8a2beec475f8969d941575951',
            checkpoint_status='NOT_RUN_NO_OFFICIAL_WEIGHTS',checkpoint_sha256=None,
            official_entry='https://pan.baidu.com/s/1vC_7cFpUh1JP9uim_QDJhA',entry_check='READ_ONLY_SUBAGENT: HTTP200 share/init HTML, NOT a weight download',
            prompt_default='GT_BUILDING_BOX; cannot enter independent no-GT-prompt evaluation',
            offset='Final decoded resized-image pixel dx/dy; footprint = roof - offset. Active1024 branch does not undo offset resize.',
            resize_padding='aspect preserving toward1024, right/bottom padding; arbitrary-size transfer unestablished',
            double_correction='FORBIDDEN_UNPROVEN_FRAME_COMPOSITION; no offsets applied here'),
        AROSICS=dict(installed='NOT_FOUND_IN_AUDITED_ENVIRONMENTS',run='NOT_RUN',reason='No independent large-ground anchor set established; no new matcher/window search'),
        references_scope='Official documentation/repository text and code only; no unacquired paper fulltext/benchmark claim; no other公众号 invented',
        primary_urls=['https://support.google.com/earth/answer/6327779/how-images-are-collected?hl=en','https://gdal.org/en/stable/tutorials/geotransforms_tut.html','https://gdal.org/en/stable/programs/gdalwarp.html',
            'https://danschef.git-pages.gfz-potsdam.de/arosics/doc/','https://arxiv.org/html/2301.10922']))


def close(run):
    import subprocess
    import numpy as np
    import fiona
    import rasterio
    from affine import Affine
    from PIL import Image,ImageDraw
    from pyproj import Transformer
    from xian_r5.alignment.mosaic_grid import MosaicTileGrid
    from xian_r5.label_transfer import Grid,load_rasters
    if (run/'IMPLEMENTATION_RECEIPT.json').is_file():
        verify_completed(run)
        print('Closed run verified; no writes on resume',flush=True);return
    if not (run/'REFERENCE_AND_ENGINEERING_AUDIT.json').exists(): reference_audit(run)
    m=read(run/'INPUT_MANIFEST.json'); e=read(run/'EVALUATION_AND_LIMITATIONS.json')
    checked={}
    for p,h in {**m['frozen_expected_hashes'],**m['legacy_output_hashes']}.items():
        value=sha(p)
        if value!=h: raise RuntimeError('Protected input drift: '+p)
        checked[p]=value
    frozen=read(BASE/'14_receipts/OUTPUT_MANIFEST.json')['outputs']
    for p,row in frozen.items():
        if sha(p)!=row['sha256']: raise RuntimeError('Authoritative baseline mismatch: '+p)
    authorities={BASE/'13_reports/R5_G06_GEOMETRY_SOURCE_ADJUDICATION_REPORT.json':'b7949b62f0c31d1be5b59d81955566f0be25ec20f11af3dddfe371690cc45f2f',
        BASE/'14_receipts/R5_G06_GEOMETRY_SOURCE_ADJUDICATION_RECEIPT.json':'34835646455506ae8ed9c141ea2732edf0209ca7b05e70d2c013e776c1c25696'}
    for p,h in authorities.items():
        if sha(p)!=h: raise RuntimeError('Wrong project authority: '+str(p))
    srcroot=Path(r'F:\xian\xian'); dates={};components=[]
    packet=LEGACY/'B04_20261001_02/04_manual_review/A13_FIRST_PRIORITY_V1/tile_actual_date_and_hash_provenance.csv'
    with packet.open(encoding='utf-8-sig') as f: tile_rows=list(csv.DictReader(f))
    for nominal,scene in [('2018-04-29','xa180429'),('2019-01-01','xa190101'),('2020-12-31','xa201231')]:
        grid=MosaicTileGrid.open(srcroot/(scene+'.geid'),srcroot/(scene+'_combined')/(scene+'_zoom_20.tif'),srcroot/scene/'20')
        date_rows=[r for r in tile_rows if scene in Path(r['source_path']).parts]
        rows=[]
        for r in date_rows:
            parts=Path(r['source_path']).stem.split('_')
            rows.append(grid.component_receipt(int(parts[1]),int(parts[2]),actual_date=r['actual_date'],
                actual_date_source_ref=str(packet),expected_sha256=r['source_sha256'],expected_path=r['source_path']))
        dates[scene]=dict(summary=grid.summary(),actual_dates=sorted(set(r['actual_date'] for r in rows)),tiles=rows,
            source_region_transform='M1_UNESTABLISHED',RGB_observability='NOT_ASSESSED_FOR_BUILDING_ABSENCE')
        components.extend(rows)
        checked[str(grid.geid_path)]=sha(grid.geid_path)
        if scene=='xa201231': current_grid=grid
    window=current_grid.window_for_tile(420797,312176)
    if window.row_off!=28928 or window.col_off!=67328: raise RuntimeError('Native 113 tile replay failed')
    with rasterio.open(current_grid.mosaic_path) as source:
        native=source.read(window=rasterio.windows.Window(window.col_off,window.row_off,1024,1024))
        native_affine=source.window_transform(rasterio.windows.Window(window.col_off,window.row_off,1024,1024))
    if not native_affine.almost_equals(Affine(*m['coordinate_audit']['native_affine']),precision=1e-10):
        raise RuntimeError('Native/master affine mismatch')
    img=np.zeros((1024,1024,3),dtype=np.uint8)
    for r in m['native_tiles']:
        with Image.open(r['path']) as im: tile=np.asarray(im.convert('RGB'))
        c=(r['tile_x']-420797)*256;row=(312176-r['tile_y'])*256
        img[row:row+256,c:c+256]=tile
    rawdiff=np.abs(native.transpose(1,2,0).astype(float)-img.astype(float))
    tf=Transformer.from_crs(4326,32649,always_xy=True);inv=Transformer.from_crs(32649,4326,always_xy=True)
    roundtrip=[]
    for px,py in [(0,0),(.5,.5),(214,674),(256,700),(1024,1024)]:
        lon,lat=native_affine @ (px,py);x,y=tf.transform(lon,lat);lon2,lat2=inv.transform(x,y)
        col,row=~native_affine @ (lon2,lat2)
        roundtrip.append(max(abs(col-px),abs(row-py)))
    if max(roundtrip)>1e-5: raise RuntimeError('Pixel/map roundtrip failed')
    delivery=Path(r'F:\xian\西安_三期建筑追踪与三态最终正式候选版_20260916')
    receipt=delivery/'09_正式TIF输出/FINAL_PRODUCT_RECEIPT.json'
    classdict=read(receipt)['products']['combined']['codes']
    states=delivery/'09_正式TIF输出/THREE_STATE_COMBINED_ge100m2.tif'
    before_state_hash=sha(states)
    with rasterio.open(states) as s:
        state_info=dict(path=str(states),sha256=before_state_hash,dtype=s.dtypes,nodata=s.nodata,
                        affine=list(s.transform)[:6],crs=str(s.crs),width=s.width,height=s.height,
                        tags=s.tags(),dictionary=classdict,role='CANDIDATE_NOT_GROUND_TRUTH',
                        fast_path='NOT_USED_STATE_ANCHOR_DATE_FRAME_NOT_DEMONSTRATED')
    alignment=delivery/'04_角度偏移与对齐/ALIGNMENT_RECEIPT.json'
    statistics=delivery/'09_正式TIF输出/STATISTICS_INTEGRATION_RECEIPT.json'
    checked.update({str(p):sha(p) for p in [states,receipt,alignment,statistics,packet]})
    semantics=dict(schema='gt10m.input_manifest.append_only_supplement.v1',parent_manifest_sha256=sha(run/'INPUT_MANIFEST.json'),
        state_product=state_info,source_date_grids=dates,
        nominal_year_not_actual_date=True,old_alignment=read(alignment)['status'],
        old_state_semantic_limit='Nominal-state product cannot identify all per-class actual-date anchors; no transport into certified change labels.',
        target_full_grid=m['target'],target_pilot_grid=m['grid'],
        single_epoch_scope='2019-12-02 raw roof candidate occupancy, spatial/sensitivity only',
        input_component_hashes=checked)
    write_json(run/'INPUT_MANIFEST_SOURCE_SEMANTICS.json',semantics)
    command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_xian_gt10m_transfer.py','-v']
    result=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True,encoding='utf8')
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    tifs=sorted(run.glob('*.tif')); loaded=load_rasters(tifs)
    with fiona.open(run/'SOURCE_AND_TRANSFORM_LEDGER.gpkg',layer='roof_objects') as s:
        records=list(s)
    checks=dict(exact_target_grid_all_exports=loaded['grid'].affine==Affine(*m['grid']['affine']),
        all_hard_ignore=bool((rasterio.open(run/'GT10M_HARD_LABEL.tif').read()==255).all()),
        zero_train_weight=bool((rasterio.open(run/'GT10M_QUALITY.tif').read(12)==0).all()),
        no_certified_change_fraction=bool((rasterio.open(run/'GT10M_FRACTIONS.tif').read([1,2,3,4,5])==0).all()),
        all_unknown_fraction=bool((rasterio.open(run/'GT10M_FRACTIONS.tif').read(6)==1).all()),
        G06_present_in_ledger=any(r.properties['is_G06']==1 for r in records),
        ledger_object_denominator=len(records)==m['object_denominator'],
        native_113_row_offset=window.row_off==28928,
        native_master_affine_exact=True,pixel_roundtrip=max(roundtrip)<1e-5,
        area_conservation=max(e['area_conservation'].values())<1e-5,
        inherited_and_baseline_hashes_unchanged=True)
    if not all(checks.values()): raise RuntimeError('Real QA failure')
    # Inspection graphic uses existing numeric raster values, not new visual adjudication.
    occ=rasterio.open(run/'SINGLE_2019_ROOF_OCCUPANCY.tif').read(); risk=rasterio.open(run/'A_ROOF_SENSITIVITY.tif').read(3)
    panel=Image.new('RGB',(4*270,360),'white');draw=ImageDraw.Draw(panel)
    for i,(label,arr) in enumerate([('A raw roof fraction',occ[0]),('B failed-shift diagnostic',occ[1]),('Manual roof reference',occ[2]),('A sampled stress bandwidth',risk)]):
        grey=np.clip(arr*255,0,255).astype('uint8');im=Image.fromarray(grey).convert('RGB').resize((270,320),Image.Resampling.NEAREST)
        panel.paste(im,(i*270,30));draw.text((i*270+5,8),label,fill='black')
    import io
    png=io.BytesIO();panel.save(png,format='PNG')
    register_bytes(run/'PILOT_NUMERIC_PREVIEW.png',png.getvalue())
    write_json(run/'ENGINEERING_VALIDATION.json',dict(synthetic_status='PASS',
        synthetic_command=command,synthetic_output=__import__('re').sub(r'Ran (\d+) tests in [\d.]+s',r'Ran \1 tests (elapsed omitted for deterministic receipt)',result.stdout+result.stderr),
        real_replay_checks=checks,raw_master_JPEG_mean_abs_difference=float(rawdiff.mean()),
        raw_master_JPEG_max_abs_difference=float(rawdiff.max()),
        native_window_pixel_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
        source_main_full_pixel_hash='NOT_COMPUTED; exact metadata and this native window only',
        pixel_roundtrip_max_pixels=max(roundtrip),protected_hashes_checked=len(checked),
        scientific_geographic_accuracy='NOT_ESTABLISHED; tests do not provide geographic error bounds'))
    for p,h in checked.items():
        if sha(p)!=h: raise RuntimeError('Input changed during final validation: '+p)
    outputs=output_inventory(run)
    code={str(PROJECT/p):sha(PROJECT/p) for p in ['src/xian_r5/label_transfer.py','tests/test_xian_gt10m_transfer.py','tools/run_xian_gt10m_pilot.py']}
    write_json(run/'OUTPUT_SHA256_MANIFEST.json',dict(run_id=run.name,outputs=outputs,code_sha256=code))
    write_json(run/'IMPLEMENTATION_RECEIPT.json',dict(run_id=run.name,created_at=datetime.now().astimezone().isoformat(),
        terminal='GT10M_SHADOW_SINGLE_EPOCH_DIAGNOSTIC_DELIVERED_ZERO_QUALIFIED_SUPERVISION',
        manifest_sha256=sha(run/'OUTPUT_SHA256_MANIFEST.json'),outputs=len(outputs),protected_hashes_checked=len(checked),
        real_rasters=len(tifs),real_pixel_denominator=864,qualified_training_pixels=0,
        formal_alignment_unchanged=True,legacy_scientific_terminal_unchanged=True,citywide_release=False))
    print(json.dumps(dict(real_checks=checks,rasters=len(tifs),files=len(outputs),protected_hashes=len(checked)),ensure_ascii=True),flush=True)


def eligibility(run):
    """Read existing products once; no new sensitivity raster or transform fit."""
    import subprocess
    import re
    import numpy as np
    import rasterio
    from affine import Affine
    from xian_r5.label_transfer import load_rasters, supervision_eligibility, fraction_supervision
    baseline=OUTPUT/'B04_20261011_02'
    if run==baseline or run.name in {'B04_20261011_01','B04_20261011_02'}:
        raise ValueError('Eligibility audit requires a new shadow run')
    if (run/'IMPLEMENTATION_RECEIPT.json').exists():
        verify_completed(run)
        receipt=read(run/'IMPLEMENTATION_RECEIPT.json')
        for path,expected in read(run/'OUTPUT_SHA256_MANIFEST.json')['protected_sha256'].items():
            if sha(path)!=expected: raise RuntimeError('Protected input drift: '+path)
        audit=read(run/'TRAINING_INPUT_AND_USAGE_AUDIT.json')
        for snapshot in audit['binding_scope_snapshots']:
            if binding_scope_snapshot(snapshot['root'])!=snapshot:
                raise RuntimeError('Binding search scope changed; new run required')
        raw=audit['underlying_source_lookup']
        if Path(raw['expected_path']).exists()!=raw['exists']:
            raise RuntimeError('Underlying source existence changed; new run required')
        print(json.dumps(dict(resumed=True,writes=0,terminal=receipt['terminal'])))
        return
    frozen=read(baseline/'FINAL_OUTPUT_SHA256_MANIFEST.json')
    protected={str(baseline/p):h for p,h in frozen['outputs'].items()}
    parent_receipt=read(baseline/'FINAL_DELIVERY_RECEIPT.json')
    protected[str(baseline/'FINAL_OUTPUT_SHA256_MANIFEST.json')]=parent_receipt['final_manifest_sha256']
    protected[str(baseline/'FINAL_DELIVERY_RECEIPT.json')]='204dc1f590209acab4151370782d6bef9b8871e098a4ea8a1796b33f174e24b7'
    # Archive the baseline implementation, not a live path that changes with R5.
    archives={str(run/'baseline_code'/Path(p).name):h for p,h in frozen['code_sha256'].items()}
    protected.update(archives)
    for path,expected in protected.items():
        if sha(path)!=expected: raise RuntimeError('Frozen baseline/archive mismatch: '+path)
    m=read(baseline/'INPUT_MANIFEST.json')
    app=PROJECT/'outputs/urban-renewal-phase1-starter'
    relative=['configs/phase1.yaml','configs/change_public_v2_full20_inference_windows.yaml',
        'src/urban_monitor/downscale.py','scripts/export_student_labels.py',
        'src/urban_monitor/change/datasets.py','src/urban_monitor/change/losses.py',
        'src/urban_monitor/change/augmentations.py']
    inspected={str(app/p):sha(app/p) for p in relative}
    target=Path(m['target']['path'])
    if sha(target)!=m['target']['sha256']: raise RuntimeError('Grid template drift')
    inspected[str(target)]=m['target']['sha256']
    if target.with_suffix('.tif.aux.xml').exists():
        inspected[str(target.with_suffix('.tif.aux.xml'))]=sha(target.with_suffix('.tif.aux.xml'))
    raw_source=target.parent/'Xian_2019.tif'
    with rasterio.open(target) as s:
        template=dict(path=str(target),sha256=sha(target),role='GRID_TEMPLATE_ONLY_IN_THIS_BRANCH',
            external_training_use='NOT_DEMONSTRATED',tags=s.tags(),bands=list(s.descriptions),
            affine=list(s.transform)[:6],gdal_geotransform=list(s.transform.to_gdal()),
            crs=str(s.crs),shape=[s.height,s.width],actual_date=None,composite_window=None)
        if list(s.transform)[:6]!=m['target']['affine']:
            raise RuntimeError('Template affine differs from shadow baseline')
    roots=[app/'configs',app/'src',app/'scripts',app/'tools',PROJECT/'src',PROJECT/'tools']
    search_command=['rg','-n','-i','sentinel|Xian_2019|TrueColor|student_export|student model|10 m',
        *map(str,roots),'-g','*.py','-g','*.yaml','-g','*.ps1']
    search=subprocess.run(search_command,capture_output=True,text=True,encoding='utf8',errors='strict')
    if search.returncode not in (0,1): raise RuntimeError(search.stderr)
    matches=sorted(search.stdout.splitlines())
    legacy_training_files=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'07_training').rglob('*') if p.is_file())
    usage_path=LEGACY/'B04_20261007_01/01_frozen_pair_ledger/FROZEN_PAIR_RESIDUAL_LEDGER.csv'
    identity_path=LEGACY/'B04_20261006_01/04_identity_matching/ROOF_GT_IDENTITY_MATCH.json'
    with usage_path.open(encoding='utf-8-sig',newline='') as f: six=list(csv.DictReader(f))
    selected={r['roof_gt_id']:r for r in six}
    screened={r['roof_gt_id']:r for r in read(identity_path)['rows']}
    if len(selected)!=6 or len(screened)!=16: raise RuntimeError('Reference usage ledger changed')
    for p in [usage_path,identity_path,Path(m['manual_reference_path'])]:
        expected=m['frozen_expected_hashes'].get(str(p))
        if expected is None or sha(p)!=expected:
            raise RuntimeError('Historical usage evidence not bound to baseline: '+str(p))
        inspected[str(p)]=expected
    scope_snapshots=[binding_scope_snapshot(p) for p in [*roots,target.parent,ROOT/'07_training']]
    references=[]
    for row in m['references']:
        prop=row['properties']; key=prop['roof_gt_id']
        references.append(dict(roof_gt_id=key,source_fid=prop['source_fid'],
            geometry_sha256=prop['source_geometry_sha256'],
            prior_union_fraction_comparison=True,prior_a13_identity_selection=key in screened,
            prior_common_shift_fit=key in selected,
            historical_label=selected.get(key,screened.get(key,{})).get('label',screened.get(key,{}).get('display_label')),
            external_holdout_eligibility='NOT_CERTIFIED_DEVELOPMENT_EXPOSED',
            use_outside_inspected_runs='UNKNOWN'))
    if len(references)!=60 or sum(r['prior_common_shift_fit'] for r in references)!=6:
        raise RuntimeError('Reference pool or six-pair membership mismatch')
    code_files=['src/xian_r5/label_transfer.py','tests/test_xian_gt10m_transfer.py','tools/run_xian_gt10m_pilot.py']
    code_hashes={str(PROJECT/p):sha(PROJECT/p) for p in code_files}
    protected.update(inspected)
    run.mkdir(parents=True,exist_ok=True)
    plan=dict(schema='gt10m.eligibility_plan.v1',run_id=run.name,
        frozen_at=datetime.now().astimezone().isoformat(),baseline=str(baseline),
        baseline_receipt_sha256=protected[str(baseline/'FINAL_DELIVERY_RECEIPT.json')],
        support_geometry=m['support_geometry'],grid=m['grid'],
        task='FULL_PIXEL_AREA_FRACTIONS_NOT_POSTERIOR_PROBABILITIES',
        channels=['STABLE','ADDITION','DEMOLITION','KNOWN_BACKGROUND'],
        auxiliary_task='SINGLE_DATE_OBSERVED_ROOF_OCCUPANCY_NOT_STABLE',
        hard_rule=None,hard_rule_does_not_block_soft_interface=True,
        loss_contract='PIXEL_WEIGHTED_AREA_FRACTION_LOSS; FULL_CELL_DENOMINATOR; NOT_BOUND_TO_A_MODEL',
        admission_policy=m['admission'],
        calibration_ids=[],validation_ids=[],external_test_ids=[],
        split_reason='NO_CERTIFIED_UNUSED_MATCHED_OBSERVATION_REFERENCE_POOL',
        permitted_transform='REGIONAL_TRANSLATION_ONLY_ON_FUTURE_FROZEN_SUPPORTED_SPLIT',
        future_metrics=['geometry_residual_m','10m_full_cell_fraction_error'],
        internal_cv_semantics='INTERNAL_ONLY_NOT_EXTERNAL_ACCURACY_OR_ERROR_BOUND',
        transform_vector=None,error_envelope=None,student_predictions_used=False,
        code_sha256=code_hashes)
    plan_path=run/'FROZEN_ELIGIBILITY_PLAN.json'
    if plan_path.exists():
        previous=read(plan_path);plan['frozen_at']=previous['frozen_at']
    write_json(plan_path,plan)
    audit=dict(schema='gt10m.training_input_usage_audit.v1',
        template=template,training_inputs=[],training_model=None,
        training_input_provenance='UNKNOWN_NOT_BOUND',
        underlying_source_lookup=dict(expected_path=str(raw_source),exists=raw_source.exists(),
            scope='EXACT_SIBLING_PATH_AND_DELIVERED_BASEMAP_TREE; NOT_GLOBAL_ABSENCE'),
        annual_files=[str(p) for p in sorted(target.parent.glob('Xian_*_TrueColor.tif'))],
        actual_generation_script='NOT_LOCATED_IN_INSPECTED_CODE_SCOPES',
        binding_scope_snapshots=scope_snapshots,
        search=dict(command=search_command,matches=matches,scope=[str(p) for p in roots],
                    r5_training_directory=str(ROOT/'07_training'),r5_training_files=legacy_training_files),
        required_missing_binding=['10m training configuration or launcher',
            'actual imagery paths and hashes per input epoch','acquisition date or composite window and source scenes',
            'bands and preprocessing','model output channels and pixel-weighted loss/dataloader binding'],
        historical_full20=dict(role='INTERFACE_COMPARISON_ONLY_NOT_XIAN_10M_MODEL',
            datasets=['DSIFN','S2Looking','SECOND'],outputs=['binary_change','new_building','demolition','semantic_t1','semantic_t2'],
            input='RGB/255',label_reader='NONZERO_TO_BINARY; SEMANTIC_TO_INT64',
            loss='BCE+DICE sample-level tasks; semantic hard CE+DICE; no supplied valid/unknown pixel weights',
            direct_fraction_TIF_compatibility='INCOMPATIBLE_WITHOUT_EXPLICIT_ADAPTER_AND_LOSS_BINDING'),
        legacy_student_export=dict(role='FUTURE_STUDENT_LABEL_EXPORT_NOT_TRAINING_BINDING',
            bands=['new_fraction','demolition_fraction','construction_fraction','any_change_fraction'],
            stable_channel=False,MMU_m2=200,does_not_define_actual_10m_training_images=True),
        inspected_sha256=inspected,reference_usage=references,
        usage_counts=dict(reference_pool=60,union_diagnostic_exposed=60,identity_screened=16,
            old_fit_development_pairs=6,certified_external_validation_pairs=0),
        broader_reference_use_history='NOT_EXHAUSTIVELY_AUDITED; remaining city objects not certified unseen')
    write_json(run/'TRAINING_INPUT_AND_USAGE_AUDIT.json',audit)
    data=load_rasters([baseline/'GT10M_FRACTIONS.tif',baseline/'SINGLE_2019_ROOF_OCCUPANCY.tif'])
    main,occupancy=data['arrays']; shape=main.shape[1:]
    no=np.zeros(shape,dtype=bool)
    # These failures are evidenced absence of qualification, not measured bad pixels.
    gate=supervision_eligibility(time_ok=no,geometry_ok=no,state_ok=no,interface_ok=no)
    batch=fraction_supervision(main[:4],valid=main[4],unknown=main[5],eligible=gate['eligible'])
    auxiliary=fraction_supervision(occupancy[:1],valid=occupancy[3],unknown=1-occupancy[3],eligible=gate['eligible'])
    report=dict(TRAINING_INPUTS_IDENTIFIED='NOT_IDENTIFIED_IN_AUDITED_SCOPE',
        TRAINING_INPUT_TEMPORAL_PROVENANCE='UNKNOWN_NOT_BOUND',
        SUPERVISION_TASK_DEFINED='SOFT_FRACTION_INTERFACE_DEFINED_MODEL_BINDING_MISSING',
        CALIBRATION_VALIDATION_SPLIT_AUDITED='PASS_USAGE_AUDIT_NO_CERTIFIED_EXTERNAL_POOL',
        REGIONAL_TRANSFORM_VALIDATION='NOT_RUN_NO_FROZEN_ELIGIBLE_SPLIT',
        ADMISSION_ERROR_ENVELOPE_STATUS='NOT_ESTABLISHED',
        SINGLE_DATE_OCCUPANCY_ELIGIBLE_PIXELS=int((auxiliary['weight']>0).sum()),
        THREE_STATE_CHANGE_ELIGIBLE_PIXELS=int((batch['weight']>0).sum()),
        counts=gate['counts'],exclusive_reason_counts=gate['exclusive_reason_counts'],
        exclusion_semantics='INCLUSIVE_CAUSES_OVERLAP; EXCLUSIVE_COMBINATIONS_PARTITION_ORIGINAL_DENOMINATOR',
        fractions_denominator='ENTIRE_100_M2_TARGET_PIXEL_NO_UNKNOWN_RENORMALIZATION',
        task_reasons=dict(
            single_date=['ACTUAL_TRAINING_IMAGE_NOT_BOUND','NO_VALIDATED_GEOMETRY_ENVELOPE',
                'MODEL_ROOF_CANDIDATE_NOT_QUALIFIED_CLEAN_OCCUPANCY_AND_KNOWN_BACKGROUND','PIXEL_WEIGHTED_MODEL_INTERFACE_NOT_BOUND'],
            three_state=['ACTUAL_TWO_TRAINING_EPOCHS_NOT_BOUND','NO_VALIDATED_GEOMETRY_ENVELOPE',
                'NO_QUALIFIED_TWO_DATE_PRESENCE_INTERVAL_STATE_LABELS','PIXEL_WEIGHTED_MODEL_INTERFACE_NOT_BOUND']),
        evidence_closed=['GRID_TEMPLATE_ROLE_SEPARATED_FROM_TRAINING_DATE_REQUIREMENT',
            'LEGACY_LOADER_SOFT_LABEL_INCOMPATIBILITY_LOCATED','REFERENCE_DEVELOPMENT_EXPOSURE_AUDITED',
            'SOFT_FRACTION_AND_REASON_ACCOUNTING_SEAMS_TESTED'],
        absolute_position_accuracy='UNKNOWN',citywide_release=False,
        baseline_formal_counts=parent_receipt['formal_counts'],legacy_failed_gate_unchanged=True,
        regional_shift_fits=0,new_rasters=0,new_G06_geometry=0)
    report.update({k:v for k,v in gate['counts'].items() if k.startswith('EXCLUDED_BY_')})
    write_json(run/'SUPERVISION_ELIGIBILITY_REPORT.json',report)
    command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_xian_gt10m_transfer.py']
    result=subprocess.run(command,cwd=PROJECT,capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    test_text=re.sub(r'Ran (\d+) tests in [\d.]+s',r'Ran \1 tests (elapsed omitted)',result.stdout+result.stderr)
    real_checks=dict(baseline_20_artifacts_verified=True,baseline_code_archived=True,
            exact_original_864_grid=shape==(32,27),soft_fraction_values_unchanged=np.array_equal(batch['target'],main[:4]),
            zero_weights_preserved=not batch['weight'].any() and not auxiliary['weight'].any(),
            exclusion_partition=sum(gate['exclusive_reason_counts'].values())==864,
            affine_gdal_explicit=Affine.from_gdal(*template['gdal_geotransform'])==Affine(*template['affine']))
    if not all(real_checks.values()): raise RuntimeError('Real eligibility QA failed closed')
    write_json(run/'ENGINEERING_VALIDATION.json',dict(synthetic_command=command,synthetic_output=test_text,
        real_checks=real_checks,
        geographic_accuracy='NOT_ESTABLISHED; engineering tests only'))
    register_bytes(run/'README.md',('''# GT10m Training-Input Binding & Supervision Eligibility

本轮沿用 B04_20261011_02 的 864 像元诊断，只读取原栅格；未重跑压力情景、修 G06 或拟合位移。

已明确：TrueColor 在本分支仅作格网模板，其标签 visualization only 不能证明任何训练调用。尚未在审计范围定位实际西安 10 m 训练配置；不能假称全机没有模型。其源标签 Xian_2019.tif 的同目录文件不存在，影像生成脚本也未在所列代码范围找到。年度名不被写成拍摄日或合成窗。

已检查旧 Full20：公开高分变化数据训练，不是已绑定的西安 Sentinel 学生模型。旧读取器会二值化分数；旧损失没有本任务的逐像元质量权重。不得直接用于新分数监督。本分支已定义全像元面积分数、valid/unknown/weight 接口；硬类别规则未定义不阻止该接口，但真实训练输入、损失适配及证据仍缺失。没有训练启动或虚构模型绑定。

参考使用审计：60 个轮廓都已参加 union 分数诊断，16 个参加 a13 选择，6 个用于历史拟合。并非断言60个都用于拟合，但不能把它们随机改称独立外部测试。未冻结任何新拟合/验证对象；区域平移及误差包络保持 null。已有内部诊断与压力情景均不作为准入误差界。

合格单期辅助/三态变化像元均为0。报告按864原分母分别列出时间、几何、状态、接口不足；原因计数重叠，独占原因组合用于分母核对。没有将来源覆盖当标签有效性，也未未知归一化。单期不足不等于必须补第二期；仅三态主任务需要合格区间状态证据。

下一项具体缺件：实际10m训练配置/入口及各期影像清单（路径、哈希、波段预处理、采集日期或合成窗口）；若尚未建学生模型，应明示，再单独批准模型绑定。空间准入另需预先冻结的匹配校准/验证对象与使用历史，不要求先解释G06。

GDAL 六元组显式采用 (c,a,b,f,d,e)，与 Affine (a,b,c,d,e,f) 不直接互传。旋转及真实格网往返测试均纳入工程验证，非地理精度证书。[GDAL geotransform](https://gdal.org/en/stable/tutorials/geotransforms_tut.html)

基线产物与历史状态只读。baseline_code 保存原实现，后续代码升级不修改基线 manifest。所有 formal 计数和 ABSOLUTE_POSITION_ACCURACY=UNKNOWN、CITYWIDE=NO 不变。
''').encode('utf8'))
    for path,expected in protected.items():
        if sha(path)!=expected: raise RuntimeError('Protected input changed: '+path)
    for snapshot in scope_snapshots:
        if binding_scope_snapshot(snapshot['root'])!=snapshot:
            raise RuntimeError('Binding scope changed during audit')
    outputs=output_inventory(run)
    write_json(run/'OUTPUT_SHA256_MANIFEST.json',dict(run_id=run.name,outputs=outputs,
        code_sha256=code_hashes,protected_sha256=protected,baseline_code_archive=archives))
    write_json(run/'IMPLEMENTATION_RECEIPT.json',dict(run_id=run.name,created_at=plan['frozen_at'],
        terminal='GT10M_INPUT_BINDING_GAP_LOCATED_SUPERVISION_INTERFACE_CLOSED_ZERO_ELIGIBLE',
        manifest_sha256=sha(run/'OUTPUT_SHA256_MANIFEST.json'),qualified_training_pixels=0,
        original_pixel_denominator=864,protected_files_checked=len(protected),
        formal_counts=parent_receipt['formal_counts'],citywide_release=False))
    print(json.dumps(dict(run_id=run.name,report=report),ensure_ascii=True))


def binding_scope_snapshot(root):
    """Freeze negative search evidence too; later new files need a new audit."""
    root=Path(root)
    return dict(root=str(root),exists=root.exists(),files={
        str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=sha(p))
        for p in sorted(root.rglob('*')) if p.is_file()})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','produce','close','eligibility']);p.add_argument('--run-id',required=True)
    args=p.parse_args()
    if not args.run_id.replace('_','').isalnum(): raise ValueError('Invalid run ID')
    run=OUTPUT/args.run_id
    if args.stage=='audit': audit(run)
    elif args.stage=='produce': produce(run)
    elif args.stage=='close': close(run)
    else: eligibility(run)


if __name__=='__main__': main()
