"""Bounded existing-object/track/source audit; no fitting or label mutation."""
from pathlib import Path
import argparse,csv,hashlib,json,sqlite3,sys
import fiona
from shapely.geometry import shape,box,mapping
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.instance_evidence import component_evidence,InstanceEvidencePolicy


def selected_csv(path,required_fields,predicate):
    selected=[];invalid=[];count=0
    with path.open('r',encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames) or not set(required_fields)<=set(reader.fieldnames):
            raise ValueError('CSV schema missing/duplicate fields: '+str(path))
        for row in reader:
            count+=1
            if None in row or any(v is None for v in row.values()):
                invalid.append({'line':reader.line_num,'error':'ROW_WIDTH_MISMATCH'});continue
            if predicate(row):selected.append(row)
    return {'rows_scanned':count,'invalid_rows':invalid,'selected_rows':selected,
        'scope':'ONLY_EXPLICIT_CURRENT_ROI_OBJECT_UIDS','csv_structural_status':'PASS' if not invalid else 'PARTIAL_INVALID_ROWS_DISCLOSED'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    parent=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923/21_source_partition_alignment');prior=parent/'SINGLE2019_COHERENT_TRANSFER_20261012_12'
    history=Path('F:/xian/西安_三期建筑追踪与三态最终正式候选版_20260916')
    objects_path=history/'02_三期建筑几何/remote_full_geometry/2019-01-01_building_objects.gpkg'
    tracking=history/'05_建筑追踪';track_gpkg=tracking/'tracking_buildings.gpkg'
    tracks_csv=tracking/'tracking_buildings.csv';edges_csv=tracking/'identity_edges.csv'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh append-only evidence audit required')
    plan0=json.loads((prior/'FROZEN_COHERENT_TRANSFER_PLAN.json').read_text(encoding='utf-8'))
    receipt=json.loads((prior/'DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(plan0['protected_sha256']);protected.update(plan0['new_code_sha256'])
    protected.update({str(prior/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(prior/'DELIVERY_RECEIPT.json')]='397807e4dc4aa60a79219c786a4b7899c126d86b2113e44f422649e41f4c6974'
    protected[str(prior/'NEXT_WORK_CHECKPOINT.json')]=sha(prior/'NEXT_WORK_CHECKPOINT.json')
    for f in (objects_path,track_gpkg,tracks_csv,edges_csv,tracking/'TRACKING_GATE.json'):protected[str(f)]=sha(f)
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected input drift: '+path)
    code=[ROOT/'src/xian_r5/instance_evidence.py',ROOT/'tests/test_xian_instance_evidence.py',Path(__file__).resolve()]
    policy=InstanceEvidencePolicy()
    plan={'run':out.name,'scope':'SAME_256x256_2019_EXPERIMENT','unresolved_cells':json.loads((prior/'NEXT_WORK_CHECKPOINT.json').read_text(encoding='utf-8'))['unresolved_cells'],
        'overlay_frame':'DIRECT_UNSHIFTED_EPSG32649_NO_NEAREST_NO_BUFFER_NO_ALIGNMENT_OPTIMIZATION',
        'geometric_diagnostic_policy':policy.__dict__,'source_or_candidate_geometry_modified':False,
        'historical_candidates_are_not_teacher_parent_or_GT':True,'legacy_track_ids_are_not_new_physical_ids':True,
        'formal_review_generated':False,'new_ties_or_localization_bounds':False,
        'input_hashes':protected,'new_code_sha256':{str(f):sha(f) for f in code}}
    out.mkdir();save_json(out/'FROZEN_INSTANCE_EVIDENCE_PLAN.json',plan)
    roi=box(308460,3792190,311020,3794750)
    cache=parent/'GT10M_FRACTION_SHADOW_20261012_04/SOURCE_MASK_COMPONENTS_SHADOW.gpkg'
    with fiona.open(cache) as s:
        if s.crs.to_epsg()!=32649:raise ValueError('Component frame drift')
        observed={str(f.properties['component_key']):shape(f.geometry) for f in s}
    historical={};properties={};fids={}
    with fiona.open(objects_path,layer='buildings') as s:
        if s.crs.to_epsg()!=32649:raise ValueError('Historical frame drift')
        schema=s.schema;global_count=len(s)
        for f in s.filter(bbox=roi.bounds):
            g=shape(f.geometry)
            if g.intersection(roi).area<=1e-6:continue
            key=str(f.properties['object_id'])
            if key in historical:raise ValueError('Duplicate historical object ID')
            historical[key]=g;properties[key]=dict(f.properties);fids[key]=str(f.id)
    if len(historical)!=2130:raise ValueError('Frozen bounded historical-object count drift')
    cells=json.loads((parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json').read_text(encoding='utf-8'))['cells']
    regions={c['region_id']:shape(c['support_geometry']) for c in cells}
    evidence=component_evidence(observed,historical,regions,policy=policy)
    print('Current source/historical geometry overlay complete; reading bounded legacy CSV evidence.',flush=True)
    uid_set={'xa190101::'+key for key in historical}
    tracks=selected_csv(tracks_csv,['scene_key','source_object_id','object_uid','track_id'],
        lambda r:r['scene_key']=='xa190101' and r['object_uid'] in uid_set)
    edges=selected_csv(edges_csv,['source_object_id','target_object_id','accepted_for_identity','from_scene','to_scene'],
        lambda r:r['source_object_id'] in uid_set or r['target_object_id'] in uid_set)
    track_by_uid={};edge_by_uid={}
    for row in tracks['selected_rows']:track_by_uid.setdefault(row['object_uid'],[]).append(row)
    for row in edges['selected_rows']:
        for uid in {row['source_object_id'],row['target_object_id']} & uid_set:edge_by_uid.setdefault(uid,[]).append(row)
    gpkg_status={'path':str(track_gpkg),'access_mode':'READ_ONLY_NO_REPAIR','sqlite_quick_check':None,'fiona_layers':None}
    try:
        with sqlite3.connect(track_gpkg.as_uri()+'?mode=ro&immutable=1',uri=True) as connection:
            gpkg_status['sqlite_quick_check']=[list(row) for row in connection.execute('PRAGMA quick_check')]
    except Exception as exc:gpkg_status['sqlite_error']=type(exc).__name__+': '+str(exc)
    try:gpkg_status['fiona_layers']=fiona.listlayers(track_gpkg)
    except Exception as exc:gpkg_status['fiona_error']=type(exc).__name__+': '+str(exc)
    gpkg_status['content_usable']=gpkg_status['sqlite_quick_check']==[['ok']] and gpkg_status['fiona_layers'] is not None
    gpkg_status['legacy_existence_gate_is_content_validation']=False
    roof= json.loads((parent/'SINGLE2019_ROOF_EVIDENCE_20261012_08/ROOF_CORRESPONDENCE_HYPOTHESES.json').read_text(encoding='utf-8'))
    prior_hypotheses={r['component_key']:r for r in roof['hypotheses']}
    bundles=[];unresolved=set(plan['unresolved_cells'])
    for row in evidence['records']:
        refs=[]
        for relation in row['relations']:
            key=relation['historical_object_id'];uid='xa190101::'+key
            refs.append({**relation,'historical_source_fid':fids[key],'historical_geometry_wkb_sha256':hashlib.sha256(historical[key].wkb).hexdigest(),
                'historical_properties':properties[key],'legacy_object_uid':uid,
                'legacy_tracking_rows':track_by_uid.get(uid,[]),'legacy_edge_indices':[]})
        hypothesis=prior_hypotheses.get(row['component_key'])
        bundles.append({**row,'touches_unresolved_cells':bool(unresolved & set(row['source_regions'])),
            'source_geometry_wkb_sha256':hashlib.sha256(observed[row['component_key']].wkb).hexdigest(),
            'source_geometry':mapping(observed[row['component_key']]),'historical_evidence':refs,
            'existing_manual_roof_shape_hypothesis':None if hypothesis is None else {k:hypothesis[k] for k in
                ('source_fid','region_id','centered_shape_iou','area_similarity','status','physical_identity_verified')},
            'raw_ge_evidence':{'path':'F:/xian/亚米市区底图/xa190101_combined/xa190101_zoom_20.tif',
                'observed_date':'2018-11-01','experiment_year':2019,'native_visual_check':'NOT_RUN_THIS_AUDIT'},
            'auxiliary_polygons_used_to_set_displacement':False,'clean_roof_reconstruction':'NOT_RUN'})
    save_json(out/'COMPONENT_EVIDENCE_BUNDLES.json',{'records':bundles,'role':'INSTANCE_DIAGNOSTIC_NOT_GROUND_OR_FORMAL_IDENTITY'})
    save_json(out/'BOUNDED_HISTORICAL_TRACKS.json',tracks);save_json(out/'BOUNDED_HISTORICAL_EDGES.json',edges)
    save_json(out/'TRACKING_GPKG_CONTENT_CHECK.json',gpkg_status)
    with fiona.open(out/'BOUNDED_HISTORICAL_OBJECTS.gpkg','w',driver='GPKG',layer='historical_candidates',crs='EPSG:32649',schema=schema) as s:
        for key,g in historical.items():s.write({'geometry':mapping(g),'properties':properties[key]})
    priority=[{'component_key':r['component_key'],'source_regions':r['source_regions'],'status':r['status'],
        'source_area_m2':r['source_area_m2'],'historical_candidate_count':r['historical_candidate_count'],
        'historical_union_covered_fraction':r['historical_union_covered_fraction'],
        'has_existing_manual_roof_shape_hypothesis':r['existing_manual_roof_shape_hypothesis'] is not None}
        for r in bundles if r['touches_unresolved_cells'] and r['status'] in {
            'POSSIBLE_MERGED_SOURCE_COMPONENT_NOT_VERIFIED','UNEXPLAINED_SOURCE_AREA_NOT_CONFIRMED_CONTAMINATION'}]
    priority.sort(key=lambda r:(r['status']!='POSSIBLE_MERGED_SOURCE_COMPONENT_NOT_VERIFIED',-r['source_area_m2'],r['component_key']))
    save_json(out/'INSTANCE_INSPECTION_QUEUE.json',{'rows':priority,
        'ordering':'MERGE_HYPOTHESES_THEN_UNEXPLAINED_SOURCE_AREA; DESCENDING_SOURCE_AREA; NO_NCC_OR_VALIDATION_SCORE',
        'queue_entries_are_not_clean_roofs_or_approved_repair':True})
    summary={'run':out.name,'source_components':len(observed),'historical_roi_objects':len(historical),
        'historical_full_layer_count':global_count,'geometry_diagnostic_counts':evidence['status_counts'],
        'unresolved_source_component_count':sum(r['touches_unresolved_cells'] for r in bundles),
        'unresolved_diagnostic_counts':{s:sum(r['touches_unresolved_cells'] and r['status']==s for r in bundles) for s in evidence['status_counts']},
        'legacy_track_uids_found':len(track_by_uid),'legacy_track_rows_selected':len(tracks['selected_rows']),
        'legacy_track_uids_missing':len(uid_set-set(track_by_uid)),'legacy_edge_rows_selected':len(edges['selected_rows']),
        'csv_invalid_rows':{'tracking':len(tracks['invalid_rows']),'edges':len(edges['invalid_rows'])},
        'tracking_gpkg_content_usable':gpkg_status['content_usable'],'inspection_queue_count':len(priority),
        'source_geometries_modified':0,'new_displacements':0,'new_clean_roofs':0,'new_training_labels':0,
        'formal_gt_eligible':0,'new_physical_ids':0,'absolute_position_accuracy':'UNKNOWN'}
    save_json(out/'INSTANCE_EVIDENCE_SUMMARY.json',summary)
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Protected input changed: '+path)
    save_json(out/'INSTANCE_EVIDENCE_RECEIPT.json',{'run':out.name,'summary':summary,
        'new_code_sha256':plan['new_code_sha256'],'protected_artifacts_checked':len(protected),'protected_artifacts_changed':0,
        'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'whole_window_governance_completed':False})
    print(json.dumps(summary))


if __name__=='__main__':main()
