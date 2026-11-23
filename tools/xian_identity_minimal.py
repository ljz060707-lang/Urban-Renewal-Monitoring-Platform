"""Bounded single-date source-link audit; all output is append-only."""
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path
import fiona
from shapely.geometry import shape

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('F:/xian')
UP=BASE/'XIAN_R5_MODEL_UPGRADE_20260923'
S=UP/'21_source_partition_alignment'
PREV=UP/'23_building_object_identity/ROOF_BUILDING_IDENTITY_CLOSURE_20261013_FIXED_02'
OUT=UP/'23_building_object_identity/XIAN_R5_IDENTITY_MINIMAL_CLOSURE_20261015'
OLD=BASE/'西安_三期建筑追踪与三态最终正式候选版_20260916'
GE=BASE/'亚米市区底图/xa190101_combined/xa190101_zoom_20.tif'
CANON=OLD/'_partial_transfer_quarantine/canonical_partial/xa190101_BUILDINGS_CANONICAL.csv'
OBJECTS=OLD/'02_三期建筑几何/remote_full_geometry/2019-01-01_building_objects.gpkg'
TRACK=OLD/'05_建筑追踪/tracking_buildings.csv'
COMP=S/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
REGISTRY=S/'XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_20261013/SOURCE_COMPONENT_LEDGER.json'
DATES=BASE/'亚米市区底图/date_audit/tables/tile_actual_date.csv'
MANIFEST=BASE/'未裁剪建筑轮廓TIF_20260915/2019_BUILDINGS_GE100M2_TRUE_UNCLIPPED_FULL_EXTENT.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()
def write(name,data):
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
    with (OUT/name).open('x',encoding='utf-8') as stream:stream.write(text+'\n')
def graph_rows(layer):
    with fiona.open(PREV/'ROOF_BUILDING_IDENTITY_GRAPH.gpkg',layer=layer) as src:
        return {f['properties']['node_id']:(dict(f['properties']),shape(f['geometry'])) for f in src}
def csv_select(path,field,keys):
    selected={}; count=0
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for line,row in enumerate(csv.DictReader(stream),2):
            count+=1
            if row[field] in keys:
                if row[field] in selected:raise ValueError('Duplicate original key '+row[field])
                selected[row[field]]={'line_number':line,'row':row}
    return selected,count


def unreadable_check(path):
    result={'path':str(path),'size_bytes':path.stat().st_size,'sha256':sha(path),
        'mode':'READ_ONLY_BOUNDED_ONE_ATTEMPT','source_modified':False}
    start=time.monotonic()
    try:
        with sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True,timeout=1) as conn:
            conn.set_progress_handler(lambda:1 if time.monotonic()-start>8 else 0,1000)
            result['sqlite_integrity_check']=[r[0] for r in conn.execute('PRAGMA integrity_check(10)')]
            layers=conn.execute('SELECT table_name FROM gpkg_contents').fetchall()
            result['sqlite_layers']=[x[0] for x in layers]
    except sqlite3.Error as exc:result['sqlite_error']=str(exc)
    try:
        layers=fiona.listlayers(path); result['ogr_layers']=layers; checks=[]
        for layer in layers[:8]:
            try:
                with fiona.open(path,layer=layer) as src:
                    first=next(iter(src),None)
                    checks.append({'layer':layer,'first_feature_readable':first is not None,'count':len(src)})
            except Exception as exc:checks.append({'layer':layer,'error':str(exc)})
        result['partial_layer_checks']=checks
    except Exception as exc:result['ogr_error']=str(exc)
    result['status']='GEOPACKAGE_SOURCE_UNREADABLE' if result.get('ogr_error') else 'PARTIAL_READER_RESULT_ONLY'
    result['elapsed_seconds']=time.monotonic()-start
    return result


def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise ValueError('Preflight requires an empty append-only run directory')
    receipt=read(PREV/'VERIFIED_FINAL_RECEIPT.json')
    for name,digest in receipt['artifact_hashes'].items():
        assert sha(PREV/name)==digest,'Baseline artifact changed: '+name
    with (PREV/'ROOF_IDENTITY_MATCH_TABLE.csv').open(encoding='utf-8-sig',newline='') as stream: table=list(csv.DictReader(stream))
    plausible=sorted({r['roof_obs_id'] for r in table if r['match_status']=='PLAUSIBLE'})
    assert len(plausible)==41
    observations=graph_rows('roof_instance'); objects=graph_rows('building_object_reference')
    obj_ids={k for k in objects if k.startswith('xa190101::')}
    canonical,ncanonical=csv_select(CANON,'object_uid',obj_ids)
    tracking,ntracking=csv_select(TRACK,'object_uid',obj_ids)
    feature=read(PREV/'IDENTITY_FEATURE_EVIDENCE.json')
    grouped=defaultdict(list)
    for row in feature['existing_object_edges']:grouped[row['roof_obs_id']].append(row)
    rows=[]
    with fiona.open(OBJECTS,layer='buildings') as src:
        for oid in plausible:
            edges=grouped[oid]; best=next(x for x in edges if x['is_best_diagnostic'])
            uid=best['reference_id']; origin,g=objects[uid]
            original=src[int(origin['source_fid'])]
            if original is None:raise ValueError('Original feature not found: '+uid)
            raw=shape(original['geometry'])
            c=canonical.get(uid,{}).get('row',{}); t=tracking.get(uid,{}).get('row',{})
            rows.append({'roof_obs_id':oid,'building_object_id':uid,'object_source_fid':origin['source_fid'],
                'original_object_id':str(original['properties']['object_id']),
                'geometry_source_path':origin['source_path'],'object_geometry_matches_original':raw.equals_exact(g,0),
                'object_wkb_sha256':hashlib.sha256(raw.wkb).hexdigest(),'canonical':c,'tracking':t,
                'canonical_geometry_hash_matches':c.get('geometry_hash')==hashlib.sha256(raw.wkb).hexdigest(),
                'object_geometric_candidates':edges,'observation_date':observations[oid][0]['source_date']})
    cfg=read(S/'TRAINROI_20261012_01/FROZEN_RUN_CONFIG.json')
    import rasterio
    with rasterio.open(cfg['teacher']) as raster:
        teacher={'path':cfg['teacher'],'sha256':sha(cfg['teacher']),'tags':raster.tags(),
            'shape':[raster.height,raster.width],'affine':list(raster.transform)[:6],'crs':str(raster.crs)}
    assert teacher['sha256']==cfg['teacher_sha256']
    protected=[PREV/p for p in receipt['artifact_hashes']]+[PREV/'VERIFIED_FINAL_RECEIPT.json',CANON,OBJECTS,TRACK,
        COMP,REGISTRY,DATES,MANIFEST,Path(cfg['teacher']),GE,
        S/'TRAINROI_20261012_01/FROZEN_RUN_CONFIG.json',
        S/'GT10M_FRACTION_SHADOW_20261012_04/COMPONENT_SOURCE_LINEAGE.json']
    hashes={str(p):sha(p) for p in protected}
    write('FROZEN_INPUTS.json',{'hashes':hashes,'scope':'41_PLAUSIBLE_PLUS_BOUNDED_EVIDENCE_RECOVERABLE_AMBIGUOUS',
        'unresolved_574_untouched':True,'physical_id_policy':'ALWAYS_NULL_THIS_SINGLE_DATE_RUN',
        'source_link_gate':['source_record_object_uid_exists','exact_original_geometry_hash_matches_canonical_export',
            'teacher_declares_same_canonical_source','same_observation_frame_and_date',
            'native_raster_replay_exact_and_unique','GE_source_and_date_hash_verified'],
        'ambiguous_batch_limit':24,'no_displacement':True,'no_new_object_database':True})
    write('EXISTING_ID_CONNECTION_AUDIT.json',{'plausible_ids':plausible,'plausible_records':rows,
        'canonical_csv_rows_read_once':ncanonical,'canonical_records_in_existing_scope':canonical,
        'tracking_csv_rows_read_once':ntracking,'tracking_records_in_existing_scope':tracking,
        'teacher':teacher,'rasterization_manifest':read(MANIFEST)})
    damaged=[OLD/'02_三期建筑几何/remote_full_geometry/2018-04-29_building_objects.gpkg',
        OLD/'05_建筑追踪/tracking_buildings.gpkg',
        CANON.with_suffix('.gpkg')]
    results=[unreadable_check(p) for p in damaged]
    backups=[{'path':str(p),'size_bytes':p.stat().st_size,'role':'EXISTING_PARTIAL_NOT_PROMOTED'}
        for p in (OLD/'_partial_transfer_quarantine').glob('*.partial')]
    write('UNREADABLE_GPKG_AUDIT.json',{'checks':results,'existing_partial_backups':backups,
        'available_exports':[str(CANON),str(TRACK)],'recovery_attempts_per_source':1,'source_files_modified':False})
    print(json.dumps({'plausible':len(rows),'original_hash_matches':sum(r['canonical_geometry_hash_matches'] for r in rows),
        'canonical_dates':dict(Counter(r['canonical'].get('local_actual_date') for r in rows)),
        'tracking_dates':dict(Counter(r['tracking'].get('local_actual_date') for r in rows)),
        'source_checks':[(r['path'],r['status']) for r in results]},ensure_ascii=False),flush=True)


if __name__=='__main__':preflight()
