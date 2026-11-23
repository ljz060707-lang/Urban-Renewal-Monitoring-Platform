"""Seal research artifacts and create a compact local handoff, no raw datasets."""
import json,hashlib,shutil,zipfile,ast
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    assert not (ROOT/'FINAL_RECEIPT.json').exists(),'Do not rewrite a sealed delivery'
    for path in list((ROOT/'source').glob('*.py'))+list((ROOT/'LEGACY_CODE').rglob('*.py')):ast.parse(path.read_text(encoding='utf8'))
    tests=(ROOT/'evidence/TESTS_NATIVE_FINAL.log').read_text(encoding='utf8');assert '54 passed' in tests
    assert read(ROOT/'evidence/INDEPENDENT_RELOAD_QA.json')['status']=='PASS'
    assert '"valid": true' in (ROOT/'evidence/NATIVE_ENTRY_COMPATIBILITY.log').read_text()
    assert all(sha(ROOT/p)==h for p,h in read(ROOT/'evidence/FINAL_NATIVE_INFERENCE_CONTRACT.json')['files'].items())
    assert read(ROOT/'evidence/CANONICAL_MULTIPART_ID_AUDIT.json')['frames_with_repeated_Id']==0
    frozen=read(ROOT/'evidence/FULLSPAN_CODE_FREEZE.json');assert all(sha(ROOT/p)==h for p,h in frozen.items())
    selected=[r['variant'] for r in read(ROOT/'reports/TARGET_COMPARISON.json')]
    for name in selected:
        run=ROOT/'runs'/name;assert read(run/'COMPLETE.json')['formal_events']==0
        files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='RUN_PRODUCTS_SEAL.json'}
        write(run/'RUN_PRODUCTS_SEAL.json',dict(files=files,level='RESEARCH_ONLY',selected_main_seed=20261027))
    code=ROOT/'APPLICATION_CODE';write(ROOT/'CODE_FREEZE.json',dict(frozen_after_research_comparisons=True,files={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')}))
    write(ROOT/'TEMPORAL_CODE_FREEZE.json',dict(files={str(p.relative_to(ROOT)):sha(p) for folder in ['source','LEGACY_CODE'] for p in (ROOT/folder).rglob('*.py')},
        training_freeze='evidence/FULLSPAN_CODE_FREEZE.json',no_promotion=True))
    routing=read(ROOT/'MODEL_ROUTING_MANIFEST.json')
    routing['temporal_upgrade']=dict(default_production_switch=False,original_application_route_retained=True,
        baseline=dict(path=str(ROOT/'models/RECOVERED/B2.pth'),sha256=sha(ROOT/'models/RECOVERED/B2.pth'),features=22,decision='strict_argmax',use='research comparator'),
        primary=[dict(seed=s,path=str(ROOT/'runs/T2_FULL_SPAN'/('seed_'+str(s))/'best.pt'),sha256=sha(ROOT/'runs/T2_FULL_SPAN'/('seed_'+str(s))/'best.pt'),use='research_not_promoted') for s in [20261027,20261028,20261029]],
        primary_architecture='TemporalIdentityNet RGB-mask CNN + geometry MLP + set-pooled bidirectional GRU',formal_weight=0,target_accuracy=None)
    write(ROOT/'MODEL_ROUTING_MANIFEST.json',routing)
    # Old baseline copies are references; never mutate the original application.
    base=Path('F:/xian/XIAN_R5_APPLICATION/APPLICATION_V1_20261027_01/runs/REAL_THREE_EPOCH_MULTIPARENT');baseline=ROOT/'BASELINE_REFERENCE';baseline.mkdir(exist_ok=True);refs=[]
    for rel in ['reports/SINGLE_DATE_AREA_SUMMARY.csv','reports/INTERVAL_CHANGE_AREA_SUMMARY.csv','temporal/OBSERVATIONS.jsonl','temporal/TRACKS_CANDIDATE.jsonl','temporal/PRESENCE_CANDIDATE.jsonl','pairs/t0__t1/EVENTS.jsonl','pairs/t1__t2/EVENTS.jsonl']:
        p=base/rel;out=baseline/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out);refs.append(dict(source=str(p),copy=str(out),sha256=sha(p)))
    write(baseline/'PROVENANCE.json',refs)
    source_recovery=Path('F:/xian/XIAN_R5_TEMPORAL_WEIGHT_RECOVERY_20261027_01/CHECKPOINT_INVENTORY_VERIFIED.json')
    recovery=read(source_recovery)
    for r in recovery['records']:assert sha(r['path'])==r['sha256']
    shutil.copy2(source_recovery,ROOT/'evidence/RECOVERED_12_WEIGHT_INVENTORY.json')
    # Explicit source request boundary, not instructions imported from a document.
    write(ROOT/'evidence/REQUEST_SCOPE.json',dict(user_authorized='Continue supplied temporal-upgrade prompt and look up prior trained temporal weights; nearest actual epochs, no annual interval requirement',
        attachment_status='User-adopted technical requirements, not system/developer instructions',prior_dialogues='evidence/PROJECT_CHAT_CONTEXT_VERIFIED.json',
        current_scope='Local research training and same-fixture candidate comparison; no paid cloud, no production promotion, no messages to other chats',
        incomplete_scientific_work=['exact HR BT correspondence contract','SpaceNet4 payload','target independent observability/absence calibration','qualified detection/tracking accuracy']))
    write(ROOT/'reports/VISUAL_QA.json',dict(status='RENDERED_AND_AGENT_INSPECTED_NOT_SEMANTIC_GT',
        images={n:sha(ROOT/'reports'/n) for n in ['SOURCE_TRAINING_AND_RISK.png','REAL_SOURCE_CANDIDATES.png']},
        source_dates_legible=True,measured_curves=True,actual_source_RGB=True,candidate_colors_distinct=True,
        finding='Off-nadir/facade ambiguity, nested same-model context fragments and small noise candidates remain visible; no independent detection recall claim',human_GT_created=False))
    write(ROOT/'TEMPORAL_IMPLEMENTATION_MAP.json',dict(entry='APPLICATION_CODE/tools/run_xian_r5_integration.py --temporal-upgrade',
        fixed_fixture_driver='source/upgrade_application.py',identity_model='source/temporal_model.py',candidate_cardinality_adapter='source/temporal_inference.py',
        full_epoch_trainer='source/train_temporal_fullspan.py',full_sequence_solver='source/upgrade_application.py:graph_partition',
        recovered_exit_adapter='source/run_recovered_exit.py',source_metrics='source/association_metrics.py',
        original_native_manifest_entry_preserved=True,fixed_fixture_only=True,production_route_changed=False))
    # Package only the reproducible code, weights, reports, selected products and
    # current-pixel recheck evidence. Data shards, range caches and failed attempts stay on disk.
    include=[]
    for folder in ['source','APPLICATION_CODE','LEGACY_CODE','models','reports','evidence','BASELINE_REFERENCE','examples']:
        include.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in selected+['T1_BT','RECOVERED_V8R5_SHADOW_EXACT','T2_FULL_SPAN']:
        include.extend(p for p in (ROOT/'runs'/name).rglob('*') if p.is_file())
    # Include all context source RGB/probabilities/geometry and the search ledgers.
    include.extend(p for p in (ROOT/'runs/ACTIVE_EXPANDED').rglob('*') if p.is_file())
    for n in ['FINAL_REPORT.md','README.md','RUN_COMPARISON.cmd','CODE_FREEZE.json','TEMPORAL_CODE_FREEZE.json','MODEL_ROUTING_MANIFEST.json','CLASS_DICTIONARY_V1.json','TEMPORAL_IMPLEMENTATION_MAP.json']:
        include.append(ROOT/n)
    include=sorted(set(include));assert all(p.exists() for p in include)
    receipt=dict(status='RESEARCH_ITERATION_EXECUTED_APPLICATION_UPGRADE_NOT_QUALIFIED',created_utc=datetime.now(timezone.utc).isoformat(),
        selected_main_architecture='TemporalIdentityNet',baseline='recovered B2',primary_seed=20261027,primary_seeds=[20261027,20261028,20261029],
        source_sequences=read(ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json')['sequences'],active=read(ROOT/'runs/ACTIVE_EXPANDED/COMPLETE.json'),
        source_independent_target_accuracy=None,production_ready=False,formal_weight_nonzero_pixels=0,stages=read(ROOT/'reports/STAGE_STATUS.json'),
        tests='54 passed',product_area_QA=read(ROOT/'evidence/EXPORTED_PRODUCT_CONSERVATION_QA.json')['status'],
        package_exclusions=['raw datasets and prepared training shards','compressed acquisition ranges and quarantined inherited FINAL_TEST auxiliaries','failed and unselected diagnostic attempts','external B0/E3/V2 weights and native original inputs'],
        selected_runs=selected,protected_original=read(ROOT/'evidence/ORIGINAL_APPLICATION_PRESERVATION.json'),
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in include})
    write(ROOT/'FINAL_RECEIPT.json',receipt);include.append(ROOT/'FINAL_RECEIPT.json')
    target=ROOT.parent/'XIAN_R5_TEMPORAL_UPGRADE_RESEARCH_20261027_01.zip'
    assert not target.exists(),'Use a new archive name instead of overwriting'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in include:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None;count=len(z.infolist())
    assert all(sha(ROOT/n)==h for n,h in receipt['output_sha256'].items())
    write(ROOT/'DELIVERY_VERIFICATION.json',dict(status='PASS',sealed_files=len(receipt['output_sha256']),archive=str(target),archive_bytes=target.stat().st_size,archive_sha256=sha(target),archive_members=count,zip_CRC='PASS',all_receipt_hashes_rechecked=True,
        final_receipt_sha256=sha(ROOT/'FINAL_RECEIPT.json'),original_application_unchanged=True,qualification='NOT_PROMOTED'))
    print(json.dumps(dict(status='PASS',sealed_files=len(receipt['output_sha256']),archive=str(target),bytes=target.stat().st_size)))
if __name__=='__main__':main()
