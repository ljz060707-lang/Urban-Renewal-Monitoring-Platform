"""Append-only, bounded relationship view of existing roof/building IDs."""
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
import fiona
from pyproj import Transformer
from shapely.geometry import box, shape, mapping
from shapely.ops import transform
from xian_r5.roof_identity import candidates, validate_table, STATUSES

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('F:/xian')
UP=BASE/'XIAN_R5_MODEL_UPGRADE_20260923'
S=UP/'21_source_partition_alignment'
OLD=BASE/'西安_三期建筑追踪与三态最终正式候选版_20260916'
OUT=UP/'23_building_object_identity'/('ROOF_BUILDING_IDENTITY_CLOSURE_20261013' if len(sys.argv)<2 else sys.argv[1])
ROI=box(308460,3792190,311020,3794750)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write(name,value):
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    with (OUT/name).open('x',encoding='utf-8') as stream: stream.write(data+'\n')


def read_features(path,layer=None,bounds=None):
    with fiona.open(path,layer=layer) as src:
        return [(f['id'],dict(f['properties']),shape(f['geometry']))
                for f in (src.filter(bbox=bounds) if bounds else src) if f['geometry']]


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    obs_path=S/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
    gt_path=BASE/'西安建筑轮廓数据/西安.shp'
    object_paths=list(sorted((OLD/'02_三期建筑几何/remote_full_geometry').glob('*.gpkg')))
    track_path=OLD/'05_建筑追踪/tracking_buildings.csv'
    edge_path=OLD/'05_建筑追踪/identity_edges.csv'
    bad_track=OLD/'05_建筑追踪/tracking_buildings.gpkg'
    lineage_paths=[UP/'18_reconstruction'/p/'building_lineage.gpkg' for p in
       ['B03_TEX04_PIPELINE_RECONSTRUCTION_FREEZE_FIXEDCASE_20260929_02',
        'B04-TQA10_posegraph05_aliked_qualitycodes_20260925_02']]
    source_path=S/'XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_20261013/SOURCE_COMPONENT_LEDGER.json'
    date_path=BASE/'亚米市区底图/date_audit/tables/tile_actual_date.csv'
    historical_path=S/'SINGLE2019_ROOF_EVIDENCE_20261012_08/ROOF_CORRESPONDENCE_HYPOTHESES.json'
    roofbank=S/'SINGLE2019_CURRENT_ROOF_BANK_20261012_19/CURRENT_SOURCE_ROOF_BANK.gpkg'
    bank_json=roofbank.with_suffix('.json')
    protected=[obs_path,*gt_path.parent.glob('西安.*'),*object_paths,track_path,edge_path,bad_track,
               *lineage_paths,source_path,date_path,historical_path,roofbank,
               bank_json,
               ROOT/'src/xian_r5/roof_identity.py',ROOT/'tools/xian_roof_identity_closure.py']
    previous=OUT.parent/'ROOF_BUILDING_IDENTITY_CLOSURE_20261013'
    if previous!=OUT and previous.exists(): protected.extend(p for p in previous.iterdir() if p.is_file())
    hashes={str(p):sha(p) for p in protected if p.is_file()}
    write('FROZEN_INPUTS.json',{'scope_bounds_epsg32649':list(ROI.bounds),'scope':'EXISTING_TRAINING_WINDOW_DEVELOPMENT_ONLY',
        'search_m':30,'original_ids_immutable':True,'geometry_transport':False,'hashes':hashes})
    print('Frozen inputs',len(hashes),flush=True)
    audit=[]; objects={}; properties={}; object_origin={}
    for path in object_paths:
        try:
            features=read_features(path,'buildings',ROI.bounds)
            for fid,props,g in features:
                if not g.intersects(ROI):continue
                scene={'2019-01-01':'xa190101','2020-12-31':'xa201231','2018-04-29':'xa180429'}[props['scene_date']]
                uid=scene+'::'+str(props['object_id'])
                if uid in objects:raise ValueError('Duplicate original object identity')
                objects[uid]=g; properties[uid]=props
                object_origin[uid]={'path':str(path),'layer':'buildings','fid':fid,'original_id':props['object_id']}
            audit.append({'path':str(path),'status':'READABLE','roi_count':len(features)})
        except fiona.errors.DriverError as exc:
            audit.append({'path':str(path),'status':'UNREADABLE_SOURCE_DATABASE','error':str(exc)})
    try:
        fiona.listlayers(bad_track)
        audit.append({'path':str(bad_track),'status':'READABLE'})
    except fiona.errors.DriverError as exc:
        audit.append({'path':str(bad_track),'status':'UNREADABLE_SOURCE_DATABASE','error':str(exc),'fallback':str(track_path)})
    observed={}; obs_props={}
    for fid,props,g in read_features(obs_path,'native_positive_components'):
        key=props['component_key']; observed[key]=g; obs_props[key]={**props,'source_fid':fid}
    to_geo=Transformer.from_crs(32649,4326,always_xy=True).transform
    to_metric=Transformer.from_crs(4326,32649,always_xy=True).transform
    geo_bounds=transform(to_geo,ROI).bounds
    refs={}; refprops={}
    for fid,props,g in read_features(gt_path,bounds=geo_bounds):
        metric=transform(to_metric,g)
        if not metric.intersects(ROI):continue
        # FID is the original source-record ID, not the potentially duplicated Id field.
        refs[fid]=metric; refprops[fid]=props
    print('Loaded roof / GT / original objects',len(observed),len(refs),len(objects),flush=True)
    historical=json.loads(historical_path.read_text(encoding='utf-8'))
    source=json.loads(source_path.read_text(encoding='utf-8'))
    source_dates=Counter(t['actual_date'] for t in source['tiles'])
    from shapely import STRtree
    supports=[shape(t['support_geometry']) for t in source['tiles']]
    support_tree=STRtree(supports)
    roof_sources=[]
    for oid,g in observed.items():
        for index in support_tree.query(g):
            index=int(index); support=supports[index]
            if g.intersects(support):
                tile=source['tiles'][index]
                roof_sources.append({'roof_obs_id':oid,'download_tile_id':tile['download_tile_id'],
                    'actual_date':tile['actual_date'],'source_JPG':tile['source_JPG'],
                    'source_sha256_from_frozen_registry':tile['source_JPG_sha256'],
                    'raw_JPG_hash_reverified_this_run':False,'support_relation':'INTERSECTS_SOURCE_SUPPORT',
                    'camera_source_component':'UNKNOWN'})
    write('ROOF_SOURCE_PROVENANCE.json',{'records':roof_sources,'tile_registry_sha256':hashes[str(source_path)],
        'covered_roof_count':len({r['roof_obs_id'] for r in roof_sources}),
        'download_tiles_not_camera_components':True})
    bank=json.loads(bank_json.read_text(encoding='utf-8'))
    bank_audit=[]
    for layer in fiona.listlayers(roofbank):
        with fiona.open(roofbank,layer=layer) as src:
            bank_audit.append({'layer':layer,'count':len(src),'schema':src.schema})
    write('ROOF_CANDIDATE_BANK_AUDIT.json',{'source_path':str(roofbank),'layers':bank_audit,
        'existing_candidate_records':bank['records'],'observed_geometry_replaced':False})
    write('SOURCE_AND_INPUT_AUDIT.json',{'datasets':audit,'source_registry_path':str(source_path),
        'source_tiles':len(source['tiles']),'actual_date_counts':dict(source_dates),
        'manual_semantics':'MANUAL_ROOF_OUTLINE_GT','manual_year':2019,'manual_precise_date':'UNKNOWN',
        'manual_duplicate_Id_count':len(refs)-len({p['Id'] for p in refprops.values()}),
        'historical_hypothesis_count':len(historical['hypotheses']),
        'historical_hypotheses_promoted':0,'roads_evidence':'NOT_BOUND',
        'object_scene_dates_are_not_reverified_tile_acquisition_dates':True})
    selected_tracks={}; history=defaultdict(list)
    with track_path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['object_uid'] in objects: selected_tracks[row['object_uid']]=row
    track_ids={r['track_id'] for r in selected_tracks.values() if r['track_id']}
    with track_path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['track_id'] in track_ids: history[row['track_id']].append(row)
    relevant_uids={r['object_uid'] for rows in history.values() for r in rows}
    historical_edges=[]
    with edge_path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['source_object_id'] in relevant_uids or row['target_object_id'] in relevant_uids:
                historical_edges.append(row)
    write('ORIGINAL_TRACK_HISTORY.json',{'role':'HISTORICAL_CANDIDATE_NOT_VERIFIED_PHYSICAL_ID',
        'track_history':dict(history),'edges':historical_edges,'new_physical_ids':0})
    lineage=[]
    for path in lineage_paths:
        for layer in fiona.listlayers(path):
            with fiona.open(path,layer=layer) as src:
                for feature in src:
                    props=dict(feature['properties'])
                    lineage.append({'source_path':str(path),'source_layer':layer,'source_fid':feature['id'],
                                    'properties':props,'in_current_scope':bool(feature['geometry'] and shape(feature['geometry']).intersects(ROI))})
    write('LINEAGE_AUDIT.json',{'records':lineage,'scope_expanded':False,'note':'Outside-ROI fixed cases are audited, not used as current-ROI identity evidence.'})
    gt_edges,gt_status=candidates(observed,refs)
    object_edges,object_status=candidates(observed,{k:g for k,g in objects.items() if k.startswith('xa190101::')})
    print('Candidate edges GT / object',len(gt_edges),len(object_edges),flush=True)
    by_gt=defaultdict(list); by_obj=defaultdict(list)
    for row in gt_edges:by_gt[row['roof_obs_id']].append(row)
    for row in object_edges:by_obj[row['roof_obs_id']].append(row)
    table=[]; cases=[]; summary_status={}
    for record in bank['records']:
        if record.get('source_role') in ('SURFACE_ONLY','EXTENT_RISK','PARTIAL_EXTENT'):
            cases.append({'case_type':'facade contamination','roof_obs_id':record.get('source_parent_component'),
                'roof_candidate_id':record.get('retained_candidate_id'),
                'disposition':'EXISTING_SOURCE_SCREEN_DIAGNOSTIC_NOT_CONFIRMED',
                'source_role':record['source_role'],'evidence_path':str(bank_json),
                'source_visual_basis':record.get('source_visual_basis'),
                'note':'Surface/extent risk does not establish a measured facade boundary.'})
    for oid in observed:
        gs=by_gt[oid]; os=by_obj[oid]
        status=('AMBIGUOUS' if 'AMBIGUOUS' in (gt_status[oid],object_status[oid]) else
                'PLAUSIBLE' if gt_status[oid]=='PLAUSIBLE' and object_status[oid]=='PLAUSIBLE' else 'UNRESOLVED')
        summary_status[oid]=status
        best_obj=next((r for r in os if r['is_best_diagnostic']),None)
        obid=best_obj['reference_id'] if best_obj else None
        pid=selected_tracks.get(obid,{}).get('track_id') or None
        # Keep every GT hypothesis; the object association remains explicitly a candidate.
        for grow in gs or [None]:
            table.append({'roof_obs_id':oid,'roof_gt_id':grow['reference_id'] if grow else None,
                'building_object_id':obid,'physical_id':pid,'match_status':status if grow is None or grow['is_best_diagnostic'] else 'AMBIGUOUS',
                'identity_candidate':True,'GT_USED_FOR_IDENTITY':grow is not None,'GT_USED_FOR_GEOMETRY':False,
                'physical_identity_status':'HISTORICAL_CANDIDATE' if pid else 'NOT_ESTABLISHED',
                'source_actual_date':obs_props[oid]['source_actual_date'],'gt_date':'2019_YEAR_ONLY',
                'temporal_continuity':'NOT_VERIFIED','geometry_status':'ORIGINAL_UNCHANGED',
                'validation_role':'development','confidence':None})
        if len(gs)>1 or len(os)>1:
            cases.append({'roof_obs_id':oid,'case_type':'unclear','disposition':'AMBIGUOUS_CANDIDATES_NOT_FORCED',
                          'gt_candidates':[r['reference_id'] for r in gs], 'object_candidates':[r['reference_id'] for r in os]})
    validate_table(table,set(observed),set(refs),set(objects),track_ids)
    write('IDENTITY_FEATURE_EVIDENCE.json',{'manual_gt_edges':gt_edges,'existing_object_edges':object_edges,
        'no_centroid_nearest_assignment':True,'no_displacement_estimated_or_applied':True})
    # Split/merge are inherited diagnostics, not newly asserted physical events.
    for row in historical_edges:
        if row['split_merge_type'] in ('MERGE','SPLIT'):
            cases.append({'case_type':row['split_merge_type'].lower(),'source_object_id':row['source_object_id'],
                          'target_object_id':row['target_object_id'],'disposition':'HISTORICAL_DIAGNOSTIC_UNVERIFIED'})
    write('DIFFICULT_CASE_LIBRARY.json',{'cases':cases,'category_status':{
        cat:('DIAGNOSTIC_CASES_AVAILABLE' if any(c['case_type']==cat for c in cases) else 'NO_VERIFIED_CASE_BOUND')
        for cat in ['merge','split','facade contamination','temporal change','unclear']},
        'missing_categories_not_fabricated':True})
    with (OUT/'ROOF_IDENTITY_MATCH_TABLE.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    gpkg=OUT/'ROOF_BUILDING_IDENTITY_GRAPH.gpkg'
    common={'node_id':'str','original_id':'str','node_role':'str','physical_id':'str','confidence':'float',
        'source_date':'str','geometry_status':'str','source_path':'str','source_layer':'str','source_fid':'str',
        'GT_USED_FOR_IDENTITY':'bool','GT_USED_FOR_GEOMETRY':'bool','validation_role':'str'}
    for layer,collection in [('roof_instance',observed),('manual_roof_reference',refs),('building_object_reference',objects)]:
        with fiona.open(gpkg,'w',driver='GPKG',layer=layer,crs='EPSG:32649',schema={'geometry':'Unknown','properties':common}) as dst:
            for key,g in collection.items():
                role='ROOF_OBSERVATION' if layer=='roof_instance' else 'MANUAL_ROOF_OUTLINE_GT' if layer=='manual_roof_reference' else 'EXISTING_OBJECT_REFERENCE'
                origin=object_origin.get(key,{}) if layer=='building_object_reference' else {}
                props={'node_id':key,'original_id':str(origin.get('original_id',refprops[key]['Id'] if layer=='manual_roof_reference' else key)),
                    'node_role':role,'physical_id':selected_tracks.get(key,{}).get('track_id') if layer=='building_object_reference' else None,
                    'confidence':None,'source_date':obs_props[key]['source_actual_date'] if layer=='roof_instance' else '2019_YEAR_ONLY' if layer=='manual_roof_reference' else properties[key]['scene_date'],
                    'geometry_status':'ORIGINAL_UNCHANGED' if layer!='manual_roof_reference' else 'REFERENCE_CRS_TRANSFORM_ONLY',
                    'source_path':origin.get('path',str(gt_path if layer=='manual_roof_reference' else obs_path)),
                    'source_layer':origin.get('layer','native_positive_components' if layer=='roof_instance' else '西安'),
                    'source_fid':origin['fid'] if layer=='building_object_reference' else key if layer=='manual_roof_reference' else obs_props[key]['source_fid'],
                    'GT_USED_FOR_IDENTITY':layer=='manual_roof_reference','GT_USED_FOR_GEOMETRY':False,'validation_role':'development'}
                dst.write({'geometry':mapping(g),'properties':props})
    with sqlite3.connect(gpkg) as conn:
        conn.execute('CREATE TABLE ROOF_IDENTITY_MATCH_TABLE (roof_gt_id TEXT,roof_obs_id TEXT,building_object_id TEXT,physical_id TEXT,match_status TEXT,identity_candidate INTEGER,GT_USED_FOR_IDENTITY INTEGER,GT_USED_FOR_GEOMETRY INTEGER)')
        conn.executemany('INSERT INTO ROOF_IDENTITY_MATCH_TABLE VALUES (?,?,?,?,?,?,?,?)',[
            (r['roof_gt_id'],r['roof_obs_id'],r['building_object_id'],r['physical_id'],r['match_status'],1,int(r['GT_USED_FOR_IDENTITY']),0) for r in table])
        conn.execute('CREATE TABLE physical_identity_reference (physical_id TEXT PRIMARY KEY,identity_status TEXT,state_history_json TEXT)')
        conn.executemany('INSERT INTO physical_identity_reference VALUES (?,?,?)',[
            (pid,'HISTORICAL_CANDIDATE_NOT_GROUND_TRUTH',json.dumps(history[pid],ensure_ascii=False)) for pid in sorted(track_ids)])
        for layer in ['ROOF_IDENTITY_MATCH_TABLE','physical_identity_reference']:
            conn.execute("INSERT INTO gpkg_contents (table_name,data_type,identifier,description) VALUES (?,'attributes',?,?)",(layer,layer,'Append-only relationship view; original IDs preserved'))
        assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    geometry_checks=[]
    for layer,collection in [('roof_instance',observed),('manual_roof_reference',refs),('building_object_reference',objects)]:
        with fiona.open(gpkg,layer=layer) as src:
            assert len(src)==len(collection)
            assert src.crs.to_epsg()==32649
            for f in src:
                assert shape(f['geometry']).equals_exact(collection[f['properties']['node_id']],1e-8)
        geometry_checks.append({'layer':layer,'count':len(collection),'original_geometry_preserved':True})
    current={p:sha(p) for p in hashes}
    assert current==hashes,'Protected input modified'
    counts=Counter(summary_status.values())
    report={'total_roof_objects':len(observed),'total_manual_roof_references':len(refs),'existing_object_nodes':len(objects),
        'status_counts':{s:counts[s] for s in STATUSES},'candidate_edge_rows':len(table),'historical_track_references':len(track_ids),
        'confirmed_physical_identities':0,'new_physical_ids':0,'geometry_checks':geometry_checks,
        'IDENTITY_GRAPH_READY':'NO','INDEPENDENT_VALIDATION_AVAILABLE':'NO','READY_FOR_GT10M_TRANSFER':'NO',
        'RELATIONSHIP_VIEW_READY':'YES','validation_status':'VALIDATION_NOT_AVAILABLE','absolute_position_accuracy':'UNKNOWN',
        'blocking_reasons':['OBSERVATION_TO_OBJECT_BINDING_NOT_VERIFIED','TEMPORAL_CONTINUITY_NOT_VERIFIED',
            'HISTORICAL_TRACKS_ARE_CANDIDATES','INDEPENDENT_VALIDATION_NOT_AVAILABLE'],
        'not_performed':['displacement','segmentation_training','10m_labels','ground_footprints']}
    write('IDENTITY_MATCH_REPORT.json',report)
    with (OUT/'FINAL_REPORT.md').open('x',encoding='utf-8') as stream:
        stream.write('# Building Object Identity Closure\n\n关系视图已生成；真实物理身份闭合尚未通过。\n\n'+
            f'当前窗口屋顶组件 {len(observed)}；人工屋顶参考 {len(refs)}；既有对象引用 {len(objects)}；历史候选轨迹 {len(track_ids)}。\n\n'+
            json.dumps(report['status_counts'],ensure_ascii=False)+'\n\n'+
            '所有对象保留原始 ID；manual Id 重复时使用原始 FID 区分，并同时保留 Id。屋顶组件不是独立建筑认证。GT 只参与身份描述，不替换 observed geometry。\n\n'+
            '历史追踪及部分几何 GPKG 无法打开，损坏记录保留，未修复源文件。使用可读的原始 building_objects 和 tracking CSV 关联已有 ID；历史候选轨迹不冒充已验证 physical ID。\n\n'+
            '道路关系、真实跨日期连续性和独立验证尚未绑定。不存在匹配不代表拆除。merge/split 仅继承历史诊断；facade、重建等证据不足时不编造。\n\n'+
            'IDENTITY_GRAPH_READY = NO\nINDEPENDENT_VALIDATION_AVAILABLE = NO\nREADY_FOR_GT10M_TRANSFER = NO\nABSOLUTE_POSITION_ACCURACY = UNKNOWN\n')
    write('FINAL_RECEIPT.json',{'status':'RELATIONSHIP_VIEW_COMPLETE_PHYSICAL_IDENTITY_UNRESOLVED',
        'protected_inputs_unchanged':len(hashes),'geometry_checks':geometry_checks,
        'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},'report':report})
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
