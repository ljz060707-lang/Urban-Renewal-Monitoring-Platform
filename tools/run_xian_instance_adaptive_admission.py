"""Frozen real-data A2 admission, unresolved inference and vector handoff."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import fiona
import numpy as np
from affine import Affine
from shapely.geometry import shape, mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from xian_r5.alignment.instance_adaptive import predict_instance_alignment, TARGET_A, TARGET_B, TARGET_C
from xian_r5.alignment.roof_candidate import apply_accepted_roof_grid_alignment
from xian_r5.label_transfer import Grid

UP = Path('F:/xian/XIAN_R5_MODEL_UPGRADE_20260923')
A = UP/'21_source_partition_alignment'
PREV = A/'XIAN_R5_SOURCE_COMMON_INSTANCE_RESIDUAL_20261018_01'
S = Path('F:/xian/XIAN_R5_SUPERVISED_ROOF_V1_20261018_01')
I = UP/'23_building_object_identity/XIAN_R5_IDENTITY_MINIMAL_CLOSURE_20261015'
INTEGRATION = UP/'24_system_integration/XIAN_R5_INTEGRATION_CONTRACT_V1_20261015_04'
BASE = A/'XIAN_R5_ADAPTIVE_LOCAL_ALIGNMENT_20261013'
OUT = A/'XIAN_R5_INSTANCE_ADAPTIVE_ALIGNMENT_20261019'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def vector_output(name,layer,rows,props):
    with fiona.open(OUT/name,'w',driver='GPKG',layer=layer,crs='EPSG:32649',
                    schema=dict(geometry='Unknown',properties=props)) as ds:
        for g,values in rows:
            ds.write(dict(geometry=mapping(g),properties=values))

def main():
    if OUT.exists(): raise ValueError('New append-only run required; target already exists')
    protected={}
    def protect(p,expected=None):
        h=sha(p)
        if expected is not None and h!=expected: raise ValueError('Input receipt hash mismatch: '+str(p))
        protected[str(p)]=h
    previous=read(PREV/'FINAL_RECEIPT.json')
    protect(PREV/'FINAL_RECEIPT.json',(PREV/'FINAL_RECEIPT.sha256').read_text().split()[0])
    for n,h in previous['output_sha256'].items(): protect(PREV/n,h)
    sr=read(S/'FINAL_RECEIPT.json');protect(S/'FINAL_RECEIPT.json')
    for n,info in sr['artifacts'].items(): protect(S/n,info['sha256'])
    authority=read(UP/'24_system_integration/AUTHORITATIVE_DELIVERY_20261015_04.json')
    for key in ('receipt','report','contract'): protect(authority[key]['path'],authority[key]['sha256'])
    required=[INTEGRATION/'UNIFIED_ROOF_OBSERVATIONS.jsonl',I/'FINAL_RECEIPT.json',I/'SOURCE_OBJECT_LINK_LEDGER.csv',
              BASE/'AUTHORITATIVE_DELIVERY.json',BASE/'BASELINE_FIXED_02/TRAINING_DATA_MANIFEST.json',
              BASE/'BASELINE_FIXED_02/BASELINE_COMPARISON.json',BASE/'BASELINE_FIXED_02/LOCAL_ALIGNMENT_EVALUATION.json',
              REPO/'src/xian_r5/alignment/adaptive_local.py',REPO/'src/xian_r5/alignment/roof_candidate.py',
              REPO/'src/xian_r5/source_partition.py',REPO/'src/xian_r5/label_transfer.py',
              REPO/'src/xian_r5/alignment/instance_adaptive.py',Path(__file__)]
    for p in required: protect(p)
    links=csvread(I/'SOURCE_OBJECT_LINK_LEDGER.csv')
    proven=defaultdict(list)
    for r in links:
        if r['link_level']=='TRACEABLE_SINGLE_DATE_LINK': proven[r['observation_id']].append(r['building_object_id'])
    assert len(proven)==53
    observations=[];registry={};raw={}
    with (INTEGRATION/'UNIFIED_ROOF_OBSERVATIONS.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);g=shape(r['geometry']);key=r['observation_id'];raw[key]=g
            observations.append(dict(observation_id=key,roof_polygon_raw=g,roof_instance_quality=r['geometry_status']))
            registry[key]=dict(source_component_id=None,actual_date=r['actual_date'],
                native_image_frame=dict(source_path=r['source_path'],source_layer=r['source_layer'],source_fid=r['source_fid'],
                                        native_geometry_sha256=r['native_geometry_sha256']),
                source_support_complete=r['source_support_complete'],source_geometry_hash=hashlib.sha256(g.wkb).hexdigest(),
                source_payload_hashes=r['source_hashes'],source_tile_ids=r['source_tile_ids'],photographic_seam_status='UNKNOWN')
    assert len(raw)==1662
    annex={r['observation_id']:r for r in read(PREV/'PER_INSTANCE_SOURCE_SUPPORT_ANNEX.json')['records']}
    with fiona.open(PREV/'PER_INSTANCE_OFFSET_PREDICTIONS.gpkg',layer='source_roof_candidates') as ds:
        for r in ds:
            key=r['properties']['observation_id'];g=shape(r['geometry']);raw[key]=g;a=annex[key]
            observations.append(dict(observation_id=key,roof_polygon_raw=g,roof_instance_quality='FRESH_B0_CONNECTED_CANDIDATE_NOT_CERTIFIED_SINGLE_BUILDING'))
            registry[key]=dict(source_component_id=None,actual_date='2018-11-01',native_image_frame=a['source_pixel_frame'],
                source_geometry_hash=hashlib.sha256(g.wkb).hexdigest(),source_support_complete=a['source_pixel_support_union_covers_geometry'],
                source_tile_ids=a['source_tile_ids'],source_payload_hashes=[p['sha256'] for p in a['source_payloads']],
                photographic_seam_status=a['camera_seam_status'])
    assert len(raw)==1891
    data=read(BASE/'BASELINE_FIXED_02/TRAINING_DATA_MANIFEST.json')['pairs'];assert len(data)==267
    by_obs=Counter(r['roof_observation_id'] for r in data);by_ref=Counter(r['roof_gt_id'] for r in data)
    for r in data:
        assert shape(r['candidate_geometry']).equals(raw[r['roof_observation_id']])
    OUT.mkdir()
    save('FROZEN_INPUTS.json',dict(protected_sha256=protected,scope='Existing 1662 observations + separate 229 fresh candidates; 267 old development pairs',
        semantic_target=TARGET_B,scientific_admission_rules_unchanged=True,public_data_registry=str(S/'LOCAL_DATASET_ROLE_SPLIT.json')))
    ledger=[]
    for r in data:
        key=r['roof_observation_id'];g=raw[key];ref=shape(r['target_geometry'])
        reasons=['IDENTITY_NOT_PROVEN','TEMPORAL_CONTINUITY_2018_TO_2019_UNCONFIRMED',
                 'OBSERVED_COMPLETE_ROOF_QUALITY_UNCERTIFIED','REFERENCE_PAIR_QUALITY_UNCONFIRMED',
                 'DEVELOPMENT_EXPOSED_NOT_INDEPENDENT','PHOTOGRAPHIC_SOURCE_COMPONENT_UNKNOWN']
        if not registry[key]['source_support_complete']: reasons.append('SOURCE_SUPPORT_INCOMPLETE')
        if by_obs[key]>1 or by_ref[r['roof_gt_id']]>1: reasons.append('COMPETING_PAIRS_IN_267_INVENTORY')
        if g.geom_type=='MultiPolygon' and len(g.geoms)>1: reasons.append('MULTIPART_NOT_CERTIFIED_SINGLE_BUILDING')
        ledger.append(dict(pair_id=r['pair_id'],observation_id=key,roof_gt_fid=r['roof_gt_id'],
            source_component_id=None,empirical_region=r['empirical_alignment_region'],actual_date=r['actual_date'],
            reference_year=2019,reference_actual_date=None,identity_status=r['identity_status'],
            temporal_continuity=None,reference_role='MANUAL_ROOF_OUTLINE_GT',
            observed_area_m2=g.area,reference_area_m2=ref.area,observed_polygon_parts=len(g.geoms) if g.geom_type=='MultiPolygon' else 1,
            observed_complete_quality=None,reference_pair_quality=None,
            other_pairs_same_observation=by_obs[key]-1,other_pairs_same_reference=by_ref[r['roof_gt_id']]-1,
            competition_scope='267 pairs only, NOT exhaustive neighborhood clearance',
            traceable_single_date_objects=json.dumps(proven.get(key,[])),source_link_proves_GT_identity=False,
            dx_m=None,dy_m=None,target_semantics=TARGET_B,training_eligible=False,
            independence='DEVELOPMENT_EXPOSED',failed_gates=json.dumps(reasons),
            evidence_path=str(BASE/'BASELINE_FIXED_02/TRAINING_DATA_MANIFEST.json'),
            evidence_sha256=protected[str(BASE/'BASELINE_FIXED_02/TRAINING_DATA_MANIFEST.json')] ))
    with (OUT/'INSTANCE_ALIGNMENT_PAIR_LEDGER.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    save('TARGET_SEMANTICS_AUDIT.json',dict(TARGET_A=TARGET_A,TARGET_B=TARGET_B,TARGET_C=TARGET_C,
        model_target=TARGET_B,existing_numeric_candidate_target='IMAGE_TO_REFERENCE_GRID',
        existing_numeric_candidates=228,existing_fresh_candidates=229,old_context_matches_are_TARGET_B=False,
        public_roof_to_footprint_is_TARGET_B=False,physical_decomposition='UNIDENTIFIED',
        accepted_region_displacement_count=0,region_zero_is_unknown=True,
        single_date_links_preserved=53,links_are_GT_identity=False,original_ids_preserved=True,
        inherited_context_algorithm='INSTANCE_CONTEXT_POINT_RGB_V1_NOT_LEARNED',
        inherited_E0_fraction_MAE=previous['scenario_E0_development_fraction_MAE'],
        inherited_E2_fraction_MAE=previous['scenario_E2_development_fraction_MAE'],
        inherited_metrics_role='HISTORICAL_DEVELOPMENT_DIAGNOSTIC_NOT_REMEASURED_OR_INDEPENDENT'))
    roles=read(S/'LOCAL_DATASET_ROLE_SPLIT.json')
    public=[d for d in roles['datasets'] if d['dataset'] in ('BONAI','BANDON')]
    save('PUBLIC_OFFSET_DATA_ADMISSION.json',dict(source_registry_path=str(S/'LOCAL_DATASET_ROLE_SPLIT.json'),
        source_registry_sha256=sha(S/'LOCAL_DATASET_ROLE_SPLIT.json'),datasets=public,
        admitted_offset_instances=0,scope='Consume actual frozen S registry; no repeated whole-drive search',
        native_offset_annotation_payload_verified=False,public_human_signature_required=False,
        public_task=TARGET_C,xian_task=TARGET_B,semantic_transfer_automatic=False,
        public_references=['https://github.com/jwwangchn/BONAI','https://github.com/fitzpchao/BANDON',
                           'https://github.com/likaiucas/OBM','https://github.com/Xuaner624/RBF-SAM']))
    save('INSTANCE_ALIGNMENT_TRAINING_SPLITS.json',dict(status='NOT_CREATED_NO_QUALIFIED_PAIRS',
        TRAIN=[],VALIDATION=[],FINAL_TEST=[],development_exposed_pairs=[r['pair_id'] for r in data],
        admitted_xian_pairs=0,admitted_public_offset_instances=0,independent_target_B_buildings=0,
        future_grouping=['source acquisition','geography','actual_date','physical building family','duplicate payload','overlapping native patch'],
        do_not_randomly_resplit_development_as_unseen=True))
    plan=read(PREV/'FROZEN_SPLIT_AND_TARGET_SEMANTICS.json')['plan'];t=plan['target_grid']
    grid=Grid(Affine(*t['affine']),t['width'],t['height'],t['crs'])
    rgb=np.load(PREV/'NATIVE_RGB.npy',mmap_mode='r')
    def deny(*a,**k): raise AssertionError('Inference must not read files')
    with patch('builtins.open',deny),patch('io.open',deny),patch('os.open',deny):
        predictions=predict_instance_alignment(rgb,observations,registry,grid,None)
    assert len(predictions)==1891 and all(r['total_dx_m'] is None and r['registered_geometry'] is None for r in predictions)
    props={'observation_id':'str','source_component_id':'str','actual_date':'str','native_image_frame':'str',
           'roof_instance_quality':'str','region_dx_m':'float','region_dy_m':'float',
           'instance_residual_dx_m':'float','instance_residual_dy_m':'float','total_dx_m':'float','total_dy_m':'float',
           'target_semantics':'str','prediction_uncertainty':'float','validation_status':'str','application_status':'str',
           'prediction_status':'str','evidence_hash':'str','source_geometry_hash':'str','physical_id':'str'}
    vector_output('INSTANCE_OFFSET_PREDICTIONS.gpkg','INSTANCE_ALIGNMENT_CANDIDATE',[(r['roof_polygon_raw'],{
        k:json.dumps(r[k],ensure_ascii=False) if k=='native_image_frame' else r[k] for k in props}) for r in predictions],props)
    vector_output('INSTANCE_UNCERTAINTY.gpkg','INSTANCE_UNCERTAINTY',[(r['roof_polygon_raw'],dict(
        observation_id=r['observation_id'],uncertainty_m=None,status='UNKNOWN_NOT_CALIBRATED',
        evidence_hash=r['evidence_hash'])) for r in predictions],
        dict(observation_id='str',uncertainty_m='float',status='str',evidence_hash='str'))
    # The old image-grid gate remains intact and receives NO invented reviewed fields.
    gate=Counter()
    for r in predictions:
        applied=apply_accepted_roof_grid_alignment(r['roof_polygon_raw'],r['actual_date'],[],reference_sources=[plan['reference_path']])
        assert not applied.applied and applied.geometry.equals(r['roof_polygon_raw'])
        gate[applied.status]+=1
    with fiona.open(OUT/'INSTANCE_OFFSET_PREDICTIONS.gpkg','w',driver='GPKG',layer='REGISTERED_ROOF',
                    crs='EPSG:32649',schema=dict(geometry='Unknown',properties=dict(observation_id='str'))) as ds: pass
    reread=[]
    with fiona.open(OUT/'INSTANCE_OFFSET_PREDICTIONS.gpkg',layer='INSTANCE_ALIGNMENT_CANDIDATE') as ds:
        for r in ds:
            p=dict(r['properties']);g=shape(r['geometry']);assert g.equals(raw[p['observation_id']])
            assert p['total_dx_m'] is None and p['physical_id'] is None;reread.append(p)
    assert len(reread)==1891
    save('GT_FREE_INFERENCE_TEST.json',dict(status='PASS_NO_CHECKPOINT_UNRESOLVED_INTERFACE_ONLY',
        observations=1891,original_observations=1662,fresh_candidates_separate=229,
        filesystem_open_denied_during_interface=True,GT_read_in_interface=0,
        actual_native_RGB_loaded=True,learned_model_inference_executed=False,
        numerical_predictions_generated=0,generalization_proven=False,gate_status_counts=dict(gate),
        registered_layer_rows=0,GPKG_readback_rows=1891,raw_geometries_unchanged=True,
        metric_nulls_and_physical_id_nulls_preserved=True))
    metrics=dict(roof_centroid_error_m=None,roof_boundary_error_m=None,IoU=None,instance_displacement_error_m=None,
        independent_evaluation_buildings=0,erroneous_large_displacement_rate=None,position_correction_coverage=0.0,
        neighbor_geometry_conflicts=None,fraction_10m_MAE=None)
    save('A0_A1_A2_A3_COMPARISON.json',dict(A0_NO_CORRECTION=dict(status='RAW_GEOMETRY_PRESERVED_NOT_ZERO_ERROR',**metrics),
        A1_REGION_ALIGNMENT=dict(status='NOT_AVAILABLE_NO_ACCEPTED_REGION',**metrics),
        A2_INSTANCE_ALIGNMENT=dict(status='NOT_TRAINED_NO_ADMITTED_SUPERVISION',**metrics),
        A3_REGION_PLUS_INSTANCE=dict(status='NOT_AVAILABLE_COMPONENTS_UNIDENTIFIED',**metrics),
        unresolved_observations=1891,legacy_denominator=1662,fresh_denominator=229,
        numeric_coverage_is_not_position_validation=True))
    save('CLOUD_GPU_ASSESSMENT.json',dict(CLOUD_GPU_REQUIRED=False,cloud_resources_created=0,
        training_started=False,training_steps=0,dataloader_forward_backward='NOT_RUN_NO_ADMITTED_OFFSET_LABELS',
        reason='Supervision and target semantics are current blockers, not a measured GPU resource bottleneck'))
    report='''# XIAN-R5-A2 instance admission and handoff\n\nStatus: BLOCKED_NO_QUALIFIED_TARGET_B_SUPERVISION_OR_CHECKPOINT.\n\n## Actual work\nRead and SHA-verify the newer 20261018 source-common/instance-context run, existing integration authority, BASELINE_FIXED_02, identity ledger and actual S data registry. No repeated full-drive scan. The newer run already has 228/229 numerical image-context candidates, not learned roof-to-planimetric predictions. Its E2 whole-area fraction MAE worsened; it is not promoted or rerun.\n\nAudit all 267 existing pairs into the CSV. Existing 53 traceable object-source links are joined explicitly but cannot prove 2018/2019 same-roof identity. Record actual source date, unknown reference date/continuity/complete-roof quality, repeated pair competition and multipart risks. All training dx/dy remain null. Nearby competition counts cover only the 267 inventory, not an exhaustive neighborhood identity audit. No GT-optimized translation labels or new instance identities are manufactured.\n\nConsume the real S registry at F:/xian/XIAN_R5_SUPERVISED_ROOF_V1_20261018_01. It was absent from the older A discovery's canonical locations but exists now; no second role registry is built. BONAI/BANDON native offset payloads remain unverified locally in that bounded discovery. Public roof-to-footprint TARGET_C does not substitute for XiAn TARGET_B. Source/native-image framing, stable building identities and temporal continuity are distinct missing evidence. Public annotations do not require a new human signature.\n\nImplement a small GT-free missing-checkpoint interface, not a fake A2 model. Exercise it with actual native RGB and 1662 existing + 229 separate fresh candidates while all Python file-open functions are denied. It returns 1891 explicit UNRESOLVED records, null offsets and null uncertainty. Non-null unsupported checkpoints fail. This proves only the no-checkpoint failure path, NOT numerical prediction, learned inference or generalization. No training or new model inference ran. No new numerical matcher or preview was generated.\n\nGPKGs preserve raw whole-object geometry in EPSG:32649 with original observation IDs. INSTANCE_UNCERTAINTY has null metric bounds. REGISTERED_ROOF is an empty layer, not a raster of fabricated zero shifts. Existing reviewed alignment application receives no invented reviewed fields; all 1891 application attempts leave source geometry unchanged. No shift is applied or rounded, no fractions are regenerated from unverifiable vectors. TARGET_B is not silently fed into the existing IMAGE_TO_REFERENCE_GRID acceptance adapter.\n\n## Explicit limits\nThe central goal—trained per-instance TARGET_B prediction with independent generalization—IS NOT COMPLETE. No qualified training pairs, independent target-B evaluation or trained checkpoint exists. New local scientific metrics remain null; formal applied counts remain zero. 1662/229 are observation-component denominators, not confirmed independent buildings.\n\nTo unblock supervised TARGET_B training: supply or establish dated, same-roof, complete observed-instance/reference pairs across independently grouped source regions, including untouched validation. For public pretraining alone, actual native RGB/instance/offset annotations and official splits are required; that only supports the public TARGET_C task, not XiAn acceptance. No cloud resource is requested before those data and an actual measured batch exist.\n\nMinimal skills: codebase-design and ponytail kept a narrow inference seam and reused the reviewed application gate, source/identity records and frozen dataset registry; no new model framework, database or task system.\n\nReferences: https://github.com/jwwangchn/BONAI ; https://github.com/fitzpchao/BANDON ; https://github.com/likaiucas/OBM ; https://github.com/Xuaner624/RBF-SAM .\n'''
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed,changed
    artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()}
    save('FINAL_RECEIPT.json',dict(run=OUT.name,project='XIAN-R5-GROUNDREG-FREEZE-0929',
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        status='BLOCKED_NO_QUALIFIED_TARGET_B_SUPERVISION_OR_CHECKPOINT',
        INSTANCE_OFFSET_PREDICTOR_TRAINED=False,INDEPENDENT_INSTANCE_VALIDATION_PASS=False,
        APPLIED_REAL_INSTANCE_ALIGNMENT_COUNT=0,UNRESOLVED_INSTANCE_COUNT=1891,
        count_unit='OBSERVATIONS_NOT_VERIFIED_SINGLE_BUILDINGS',legacy_unresolved_observations=1662,fresh_unresolved_candidates=229,
        admitted_target_B_training_pairs=0,pair_ledger_rows=267,legacy_source_links_preserved=53,
        REGISTERED_ROOF_READY=False,ABSOLUTE_POSITION_ACCURACY='UNKNOWN',READY_FOR_GT10M_TRAINING=False,
        SCIENTIFIC_VALIDATION_PASS=False,no_checkpoint_interface_check_pass=True,
        learned_inference_and_generalization_pass=False,protected_files_checked=len(protected),protected_files_changed=0,
        training_steps=0,new_checkpoint_created=False,cloud_resources_created=0,old_decisions_modified=False,
        artifacts=artifacts))
    (OUT/'FINAL_RECEIPT.sha256').write_text(sha(OUT/'FINAL_RECEIPT.json')+'  FINAL_RECEIPT.json\n',encoding='utf-8')
    print(json.dumps(dict(run=str(OUT),pairs=len(ledger),predictions=len(predictions),protected_files=len(protected),status='BLOCKED')))

if __name__=='__main__': main()
