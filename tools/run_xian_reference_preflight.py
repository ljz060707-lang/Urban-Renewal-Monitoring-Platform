"""Append-only current-ROI reference census; no matching, fitting or label writes."""
from pathlib import Path
import argparse,csv,json,hashlib,sys
import fiona
from pyproj import Transformer
from shapely.geometry import shape,Polygon,box
from shapely.ops import transform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_xian_source_partition_alignment import sha,save_json
from xian_r5.reference_preflight import reference_preflight,EVIDENCE_GATES

def prior_fids(value):
    result=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key in {'source_fid','external_fid','external_index','best_external_index'} and item is not None:
                result.add(str(item))
            elif key in {'matched_external_fids','external_fids_matched_all_three_dates'}:
                result.update(str(x) for x in item)
            else:result.update(prior_fids(item))
    elif isinstance(value,list):
        for item in value:result.update(prior_fids(item))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    r5=Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923');parent=r5/'21_source_partition_alignment'
    baseline=parent/'GT10M_FRACTION_SHADOW_20261012_04'
    if out.parent!=parent.resolve() or out.exists():raise ValueError('Fresh separate preflight directory required')
    receipt=json.loads((baseline/'FRACTION_SHADOW_RECEIPT.json').read_text(encoding='utf-8'))
    protected=dict(receipt['protected_verified_sha256']);protected.update(receipt['source_verified_sha256'])
    protected.update(receipt['new_code_sha256'])
    protected.update({str(baseline/n):h for n,h in receipt['output_sha256'].items()})
    protected[str(baseline/'FRACTION_SHADOW_RECEIPT.json')]='799fd4ba661bb46f0db604add4d9de222ae4acb5a10d54203e5de7138d55df9d'
    audit_path=r5/'20_gt10m_label_transfer/GT10M_SUPPORT_20261011_01/PROVENANCE_RECOVERY_AUDIT.json'
    previous=json.loads(audit_path.read_text(encoding='utf-8'))
    protected[str(audit_path)]='a3a42efa788870da241e7d657e4431c4f465050522c122b83e1b399c4b06c25d'
    source=Path(previous['manual_roof_reference']['path'])
    for path,digest in previous['source_hashes'].items():
        if Path(path).parent==source.parent:protected[path]=digest
    history_files=[r5/'20_gt10m_label_transfer/B04_20261011_04/TRAINING_INPUT_AND_USAGE_AUDIT.json',
        r5/'18_reconstruction/independent_ground_comparison_04/LOCAL_GEOMETRY_COMPARISON.json',
        r5/'18_reconstruction/external_footprint_crossdate_02/CROSS_DATE_COMPARISON.json',
        r5/'18_reconstruction/B04_full_mosaic_relative_offsets_07/FULL_MOSAIC_RELATIVE_OFFSET_DIAGNOSTIC.json',
        r5/'18_reconstruction/independent_ground_inventory_01/INVENTORY.json']
    history=[];exposures={}
    for path in history_files:
        protected[str(path)]=sha(path)
        data=json.loads(path.read_text(encoding='utf-8'))
        ids=prior_fids(data)
        if path.name=='INVENTORY.json':ids.update(str(x['fid']) for x in data['first_10_features'])
        history.append({'path':str(path),'sha256':sha(path),'explicit_reference_fids':sorted(ids),
                        'audit_covers_only_explicit_fids_not_all_features_drawn_or_loaded':True})
        for fid in ids:exposures.setdefault(fid,[]).append(str(path))
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Protected source drift: '+path)
    field_path=parent/'TRAINROI_20261012_01/SOURCE_CELL_CANDIDATES.json'
    field=json.loads(field_path.read_text(encoding='utf-8'))
    regions=[(x['region_id'],shape(x['support_geometry'])) for x in field['cells']]
    roi=box(308460,3792190,311020,3794750)
    to_ll=Transformer.from_crs(32649,4326,always_xy=True)
    to_metric=Transformer.from_crs(4326,32649,always_xy=True)
    corners=[(308460,3792190),(311020,3792190),(311020,3794750),(308460,3794750)]
    ring=[to_ll.transform(a[0]+(b[0]-a[0])*n/64,a[1]+(b[1]-a[1])*n/64)
          for a,b in zip(corners,corners[1:]+corners[:1]) for n in range(64)]
    query=Polygon(ring).bounds
    plan={'run':out.name,'stage':'REFERENCE_VALIDATION_PREFLIGHT_NOT_VALIDATION',
        'training_roi_bounds':list(roi.bounds),'crs':'EPSG:32649','reference_source':str(source),
        'protected_sha256':protected,'history_audit_sources':history,
        'history_audit_scope':'BOUNDED_EXPLICIT_FID_DISCLOSURES_ONLY; not exhaustive prior use',
        'reference_role':'USER_CONFIRMED_2019_MANUAL_ROOF_POSITION_REFERENCE',
        'reference_exact_capture_date':None,'candidate_actual_source_date':'2018-11-01',
        'reference_target_Sentinel_frame_binding':'NOT_ESTABLISHED',
        'fit_holdout_selection':'NONE; do not randomize before use history/continuity/frame clearance',
        'raw_or_candidate_fraction_values_read':False,'student_predictions_read':False,
        'candidate_geometry_or_vectors_modified':False,'matching_or_fitting':False,
        'formal_state':receipt['formal_state']}
    out.mkdir(parents=True);save_json(out/'FROZEN_REFERENCE_PREFLIGHT_PLAN.json',plan)
    rows=[]
    with fiona.open(source) as s:
        if s.crs.to_epsg()!=4326:raise ValueError('Reference frame drift')
        schema=s.schema;full_count=len(s)
        for f in s.filter(bbox=query):
            if f.geometry is None:continue
            g=transform(to_metric.transform,shape(f.geometry))
            if not g.is_valid:raise ValueError('Invalid reference; no silent repair')
            if not g.intersects(roi):continue
            fid=str(f.id)
            hits=[rid for rid,support in regions if g.intersection(support).area>1e-6]
            rows.append({'source_fid':fid,'geometry_sha256_metric':hashlib.sha256(g.wkb).hexdigest(),
                'geometry_valid':True,'fully_inside_roi':roi.covers(g),
                'crosses_source_cell':len(hits)>1,'source_cells':hits,
                'known_history_exposed':fid in exposures,'known_exposure_evidence_paths':exposures.get(fid,[]),
                'history_status':'KNOWN_EXPOSED' if fid in exposures else 'NOT_SEEN_IN_BOUNDED_DISCLOSURES_NOT_CERTIFIED_UNUSED',
                'reference_nominal_year':2019,'reference_actual_date':None,
                'clipped_roof_area_m2':g.intersection(roi).area,
                'target_image_frame_binding':'NOT_ESTABLISHED'})
    expected_fids=set(previous['manual_roof_reference']['intersecting_source_fids'])
    if set(x['source_fid'] for x in rows)!=expected_fids:raise ValueError('Reference inventory drift')
    result=reference_preflight(rows,{k:None for k in EVIDENCE_GATES})
    result.update({'run':out.name,'reference_schema':schema,'whole_file_count':full_count,
        'history_audit_scope':plan['history_audit_scope'],
        'history_artifact_intersection_counts':[{'path':x['path'],'explicit_fid_count':len(x['explicit_reference_fids']),
             'roi_explicit_fid_overlap':len(set(x['explicit_reference_fids'])&expected_fids)} for x in history],
        'formal_state':receipt['formal_state'],'training_eligible_pixels':0,
        'reference_background_completeness':'UNKNOWN','reference_truth_role_not_revoked':True,
        'fit_holdout_reviews_completed_this_run':0,'holdout_results_read':False})
    save_json(out/'REFERENCE_EVIDENCE_CENSUS.json',result)
    fields=['source_fid','geometry_sha256_metric','fully_inside_roi','crosses_source_cell',
            'known_history_exposed','history_status','structural_candidate','positive_reference_ready',
            'reference_role','source_cells','positive_reference_blockers']
    with (out/'REFERENCE_READINESS_LEDGER.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result['records']:
            writer.writerow({k:json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],list) else row[k] for k in fields})
    save_json(out/'MISSING_EVIDENCE_LOCATOR.json',{'input_paths_already_bound':{'manual_roofs':str(source),
        'teacher_mask':receipt['source_verified_sha256'], 'fraction_products':str(baseline)},
        'missing_evidence':[{'gate':'TIME_OR_CONTINUITY','needed':'Reference capture-date record OR verified stable-building continuity between 2018-11-01 and the manual 2019 observation; exact S1/S2 acquisition/composite windows',
            'current_artifact':'Manual shapefile Id/Floor only; training ZIP has nominal-year rasters, no temporal generation record',
            'status':'NOT_PROVIDED_IN_AUDITED_ARTIFACTS'},
            {'gate':'TARGET_IMAGE_FRAME_BINDING','needed':'Traceable evidence locating validation roof/ground features in the target Sentinel image frame; CRS conversion alone insufficient',
             'status':'NOT_ESTABLISHED; no independent control/orthophoto/RPC evidence supplied'},
            {'gate':'USE_HISTORY_CLEARANCE','needed':'Prior-use lineage sufficiently complete for declared internal/external validation protocol',
             'status':'BOUNDED_EXPLICIT_FID_AUDIT_ONLY'},
            {'gate':'IDENTITY_CORRESPONDENCE','needed':'Verified roof-reference correspondences with split/merge ambiguity resolved, without selecting on corrected residuals',
             'status':'NO_MATCHING_PERFORMED'},
            {'gate':'BACKGROUND_COMPLETENESS','needed':'Observed-support and missing-data/background evidence for full-pixel occupancy truth',
             'status':'REFERENCE_POSITIVE_ROOFS_ONLY; teacher 0 shares background and NoData'}],
        'no_synthetic_input_paths_or_evidence_created':True})
    for path,digest in protected.items():
        if sha(path)!=digest:raise RuntimeError('Protected source changed: '+path)
    print(json.dumps({'run':out.name,'counts':result['counts'],
        'history_overlap':sum(x['known_history_exposed'] for x in rows),
        'protected_hashes_verified':len(protected),'calibration_selected':0,'validation_selected':0,
        'training_eligible_pixels':0},ensure_ascii=True))

if __name__=='__main__':main()
