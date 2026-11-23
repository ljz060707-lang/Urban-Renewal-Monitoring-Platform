"""Freeze the inference snapshot and record the real command bindings."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r'F:\20261028西安市区结果\XIAN_R5_SINGLE_DATE_20261028_01\CODE_SNAPSHOT')

from xian_single_date import common

ROOT = common.RUN_ROOT
SNAP = ROOT / 'CODE_SNAPSHOT'
INFERENCE = ['common.py', 'b0.py', 'epoch.py', 'rasters.py']
POST = ['assemble.py', 'reference_gt.py', 'supervisor.py']

def sha(p):
    return common.sha256(p)

freeze = {
    'schema': 'XIAN_R5_SINGLE_DATE_CODE_FREEZE_V1',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'run_id': ROOT.name,
    'single_integrator': 'DSH agent (this run); no other writer touched the snapshot',
    'inference_snapshot_frozen_before_block_1': True,
    'inference_files': {n: sha(SNAP / 'xian_single_date' / n) for n in INFERENCE},
    'postprocess_files': {n: sha(SNAP / 'xian_single_date' / n) for n in POST
                          if (SNAP / 'xian_single_date' / n).exists()},
    'helpers_used_read_only': {
        'application_entry': str(common.APP_ENTRY),
        'application_source_root': str(common.APP_SRC),
        'model_routing_manifest': str(common.ROUTING),
        'b0_runtime': r'F:\xian\XIAN_R5_S2_SUPERVISED_ROOF_TRAINING_20261019\roof_runtime.py',
        'b0_checkpoint': common.read_json(common.ROUTING)['roof']['checkpoint'],
    },
    'not_modified': ['APPLICATION_V1_20261027_01 tree', 'RUN05 artefacts',
                     'old weights', 'failed receipts', 'source mosaics', 'GT shapefile'],
    'environment_relocation': {
        'what': 'the RSBuilding code tree moved D:/E: -> F: during this session',
        'evidence': 'all 448 runtime bindings re-hashed at F:\\Data Set\\RSBuilding\\repo and matched',
        'action': 'placed the relocated root on sys.path; no source file was edited',
    },
}
(ROOT / 'CODE_FREEZE.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding='utf-8')

bindings = {
    'schema': 'XIAN_R5_SINGLE_DATE_COMMAND_BINDINGS_V1',
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'run_root': str(ROOT),
    'python': str(common.PYTHON),
    'python_version_check': 'rsbuilding-win, Python 3.9.23, torch 2.0.1+cu118, CUDA available',
    'application_entry': str(common.APP_ENTRY),
    'application_cli_verified': {
        'command': 'python -B -Xutf8 <entry> --application --help',
        'result': 'choices = validate, plan, run, resume, export, report',
        'note': 'the --application dispatch precedes the legacy fiona import, so the entry runs '
                'in the frozen rsbuilding-win environment',
    },
    'production_start': {
        'file': str(ROOT / 'START_SINGLE_DATE_PRODUCTION.cmd'),
        'effective_command':
            'set PYTHONPATH=<run>\\CODE_SNAPSHOT && "%s" -B -X utf8 -m xian_single_date.supervisor'
            % common.PYTHON,
        'launched_via': 'Start-Process <START_SINGLE_DATE_PRODUCTION.cmd> -WindowStyle Hidden',
        'mutex': str(ROOT / 'state' / 'PRODUCTION.lock'),
    },
    'production_resume': {'file': str(ROOT / 'RESUME_SINGLE_DATE_PRODUCTION.cmd')},
    'production_status': {'file': str(ROOT / 'STATUS_SINGLE_DATE_PRODUCTION.cmd')},
    'hourly_check': {
        'script': str(ROOT / 'monitor' / 'CHECK_STATUS.cmd'),
        'powershell': str(ROOT / 'monitor' / 'CHECK_STATUS.ps1'),
        'task_name': 'XIAN_R5_SINGLE_DATE_20261028_01_CHECK',
        'create_command': ('schtasks /Create /SC HOURLY /MO 1 /TN '
                           '"XIAN_R5_SINGLE_DATE_20261028_01_CHECK" /TR '
                           '"%s"' % (ROOT / 'monitor' / 'CHECK_STATUS.cmd')),
        'query_command': 'schtasks /Query /TN "XIAN_R5_SINGLE_DATE_20261028_01_CHECK" /V /FO LIST',
        'delete_command': 'schtasks /Delete /TN "XIAN_R5_SINGLE_DATE_20261028_01_CHECK" /F',
        'run_once_command': 'schtasks /Run /TN "XIAN_R5_SINGLE_DATE_20261028_01_CHECK"',
    },
    'per_epoch_manifests': {
        e['actual_observation_date']: str(ROOT / 'configs' / ('epoch_%s.yaml' % e['actual_observation_date']))
        for e in common.EPOCHS},
    'reference_manifest': str(ROOT / 'configs' / '2019_reference.json'),
    'queue': str(ROOT / 'SINGLE_DATE_JOB_QUEUE.json'),
    'progress_documents': {
        'execution_state': str(ROOT / 'state' / 'EXECUTION_STATE.json'),
        'job_queue_status': str(ROOT / 'state' / 'JOB_QUEUE_STATUS.json'),
        'last_progress': str(ROOT / 'state' / 'LAST_PROGRESS.json'),
        'hourly_status': str(ROOT / 'monitor' / 'HOURLY_STATUS.jsonl'),
        'production_log': str(ROOT / 'logs' / 'supervisor.out.log'),
    },
    'model_calls': {
        'B0': 'frozen runtime + frozen checkpoint, batch scheduler %d (equivalence evidence in evidence/)'
              % common.BATCH,
        'E3_facade': 'NOT_RUN_THIS_PASS',
        'V2_alignment': 'NOT_RUN_THIS_PASS',
        'temporal_M3': 'DISABLED_FOR_SINGLE_DATE',
        'training': 'none', 'cloud': 'none', 'model_download': 'none',
    },
}
(ROOT / 'COMMAND_BINDINGS.json').write_text(json.dumps(bindings, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
print('freeze files', json.dumps(freeze['inference_files'], indent=1))
print('wrote CODE_FREEZE.json and COMMAND_BINDINGS.json')
