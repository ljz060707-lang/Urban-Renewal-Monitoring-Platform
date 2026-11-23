"""Resume publication only after timezone-data failure; no data/model rerun."""
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
from pathlib import Path

from run_xian_instance_adaptive_admission import OUT, REPO, sha, read, save


def main():
    if (OUT/'FINAL_RECEIPT.json').exists():raise ValueError('Final receipt already exists')
    frozen=read(OUT/'FROZEN_INPUTS.json')['protected_sha256']
    changed=[p for p,h in frozen.items() if sha(p)!=h]
    if changed:raise ValueError('Protected inputs changed: '+repr(changed))
    import fiona
    with fiona.open(OUT/'INSTANCE_OFFSET_PREDICTIONS.gpkg',layer='INSTANCE_ALIGNMENT_CANDIDATE') as ds:
        records=[dict(r['properties']) for r in ds]
    assert len(records)==1891
    assert all(r['total_dx_m'] is None and r['total_dy_m'] is None and r['physical_id'] is None and
               r['application_status']=='UNRESOLVED' for r in records)
    with fiona.open(OUT/'INSTANCE_OFFSET_PREDICTIONS.gpkg',layer='REGISTERED_ROOF') as ds:assert len(ds)==0
    suite=unittest.TestSuite()
    for pattern in ('test_xian_instance_adaptive.py','test_xian_source_partition.py','test_xian_adaptive_local_alignment.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern=pattern))
    stream=io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (OUT/'REGRESSION_TEST_LOG.txt').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
    save('REGRESSION_TESTS.json',dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        passed=result.wasSuccessful(),new_tests=10,existing_tests=20,
        scope='Contract and existing numeric/transport regression, NOT learned inference or scientific validation',
        test_source_sha256={str(REPO/'tests'/n):sha(REPO/'tests'/n) for n in
            ('test_xian_instance_adaptive.py','test_xian_source_partition.py','test_xian_adaptive_local_alignment.py')}))
    if not result.wasSuccessful():raise RuntimeError('Regression tests failed')
    save('PUBLICATION_RESUME.json',dict(first_attempt='Timezone formatting failed AFTER data/artifact generation and source recheck, BEFORE FINAL_RECEIPT write',
        original_error='zoneinfo.ZoneInfoNotFoundError: No time zone found with key Asia/Shanghai; tzdata absent',
        action='Resume publication only using explicit UTC+08:00 for Asia/Shanghai on client date 2026-10-19',
        original_runner_sha256=sha(REPO/'tools/run_xian_instance_adaptive_admission.py'),
        finalizer_sha256=sha(Path(__file__)),source_or_environment_modified=False,
        numerical_inference_repeated=False,prior_output_overwritten=False))
    changed=[p for p,h in frozen.items() if sha(p)!=h];assert not changed,changed
    save('FINAL_RECEIPT.json',dict(run=OUT.name,project='XIAN-R5-GROUNDREG-FREEZE-0929',
        created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),timezone='Asia/Shanghai',
        status='BLOCKED_NO_QUALIFIED_TARGET_B_SUPERVISION_OR_CHECKPOINT',
        INSTANCE_OFFSET_PREDICTOR_TRAINED=False,INDEPENDENT_INSTANCE_VALIDATION_PASS=False,
        APPLIED_REAL_INSTANCE_ALIGNMENT_COUNT=0,UNRESOLVED_INSTANCE_COUNT=1891,
        count_unit='OBSERVATIONS_NOT_VERIFIED_SINGLE_BUILDINGS',legacy_unresolved_observations=1662,fresh_unresolved_candidates=229,
        admitted_target_B_training_pairs=0,pair_ledger_rows=267,legacy_source_links_preserved=53,
        REGISTERED_ROOF_READY=False,ABSOLUTE_POSITION_ACCURACY='UNKNOWN',READY_FOR_GT10M_TRAINING=False,
        SCIENTIFIC_VALIDATION_PASS=False,no_checkpoint_interface_check_pass=True,
        learned_inference_and_generalization_pass=False,protected_files_checked=len(frozen),protected_files_changed=0,
        tests_passed=result.testsRun,training_steps=0,new_checkpoint_created=False,cloud_resources_created=0,
        old_decisions_modified=False,formal_counter_recomputation=False,
        inherited_numeric_context_offsets=228,new_numeric_target_B_offsets=0,
        publication_resume='PUBLICATION_RESUME.json',
        artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file()},
        finalizer_path=str(Path(__file__)),finalizer_sha256=sha(Path(__file__))))
    with (OUT/'FINAL_RECEIPT.sha256').open('x',encoding='utf-8') as f:f.write(sha(OUT/'FINAL_RECEIPT.json')+'  FINAL_RECEIPT.json\n')
    # Read back all newly published artifacts, including the receipt sidecar.
    final=read(OUT/'FINAL_RECEIPT.json')
    assert all(sha(OUT/n)==a['sha256'] for n,a in final['artifacts'].items())
    assert sha(OUT/'FINAL_RECEIPT.json')==(OUT/'FINAL_RECEIPT.sha256').read_text().split()[0]
    print(json.dumps(dict(status=final['status'],pairs=267,unresolved=1891,
        protected_inputs=len(frozen),tests_passed=result.testsRun,output=str(OUT))))


if __name__=='__main__':main()
