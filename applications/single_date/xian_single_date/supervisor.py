"""Single production supervisor: frozen queue, serial GPU, resumable stages.

One process owns the mutex.  Every stage commits its own evidence and the supervisor
records progress documents that the hourly read-only monitor inspects.
"""
from __future__ import annotations

import importlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import common

LOCK = common.STATE / 'PRODUCTION.lock'


def _cmdline(pid):
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '(Get-CimInstance Win32_Process -Filter "ProcessId=%d").CommandLine' % pid],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return ''


def acquire():
    common.STATE.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            info = common.read_json(LOCK)
            pid = int(info.get('pid', -1))
            if pid > 0 and _pid_alive(pid) and str(common.RUN_ROOT) in _cmdline(pid):
                print('another supervisor is already running: pid=%d' % pid)
                return False
        except Exception:
            pass
    common.atomic_write_json(LOCK, dict(
        pid=os.getpid(), host=socket.gethostname(), run_root=str(common.RUN_ROOT),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        python=str(common.PYTHON)))
    return True


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def release():
    if LOCK.exists():
        try:
            info = common.read_json(LOCK)
            if int(info.get('pid', -1)) == os.getpid():
                LOCK.unlink()
        except Exception:
            pass


STAGES = [
    ('REFERENCE_GT_2019', 'xian_single_date.reference_gt', 'build'),
    ('B0_EPOCH:xa180429', 'xian_single_date.epoch', 'run:xa180429'),
    ('B0_EPOCH:xa190101', 'xian_single_date.epoch', 'run:xa190101'),
    ('B0_EPOCH:xa201231', 'xian_single_date.epoch', 'run:xa201231'),
    ('ASSEMBLE', 'xian_single_date.assemble', 'build'),
    ('READBACK', 'xian_single_date.assemble', 'readback'),
]


def main(argv=None):
    if not acquire():
        return 2
    state = common.StateFile('EXECUTION_STATE.json')
    queue = common.StateFile('JOB_QUEUE_STATUS.json')
    progress = common.StateFile('LAST_PROGRESS.json')
    state.update(
        status='RUNNING', pid=os.getpid(), host=socket.gethostname(),
        run_id=common.RUN_ROOT.name, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        python=str(common.PYTHON), gpu_workers=1, module_workers_serial=True,
        training_enabled=False, cloud_enabled=False, download_models_automatically=False,
        overwrite=False, m3_temporal='DISABLED_FOR_SINGLE_DATE',
        stages=[s[0] for s in STAGES], stage_index=0)
    queue.update(
        queue=[dict(stage=s[0], status='PENDING') for s in STAGES],
        epochs=[dict(epoch_label=e['epoch_label'], epoch_id=e['epoch_id'],
                     nominal_input_date=e['nominal_input_date'],
                     actual_observation_date=e['actual_observation_date'],
                     date_source=e['date_source'], date_status=e['date_status'],
                     dominant_share_pct=e['dominant_share_pct'],
                     mixing_severity=e['mixing_severity'],
                     mosaic=e['mosaic'], status='PENDING') for e in common.EPOCHS],
        reference=dict(product='ROOF_GT2019_REFERENCE', source=str(common.GT_SHAPEFILE),
                       geometry_role=common.GT_ROLE, status='PENDING'))
    failures = {}
    try:
        for index, (name, module_name, func) in enumerate(STAGES):
            state.update(stage_index=index, stage=name, phase=name)
            progress.update(stage=name, phase='STARTING', started_utc=time.strftime(
                '%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            entry = dict(stage=name, status='RUNNING',
                         started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            _patch_queue(queue, name, entry)
            t0 = time.time()
            try:
                module = importlib.import_module(module_name)
                if ':' in func:
                    fn_name, arg = func.split(':')
                    epoch = next(e for e in common.EPOCHS if e['epoch_id'] == arg)
                    getattr(module, fn_name)(epoch, state=progress, log=common.log)
                else:
                    getattr(module, func)(state=progress, log=common.log)
                entry.update(status='COMPLETED', seconds=round(time.time() - t0, 1),
                             completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                _patch_queue(queue, name, entry)
                common.log('STAGE %s COMPLETED in %.1fs' % (name, time.time() - t0))
            except Exception as exc:
                failures[name] = '%s: %s' % (type(exc).__name__, exc)
                entry.update(status='FAILED', error=failures[name], seconds=round(time.time() - t0, 1))
                _patch_queue(queue, name, entry)
                common.log('STAGE %s FAILED %s\n%s' % (name, failures[name], traceback.format_exc()))
                # a failed independent branch never blocks the later branches that
                # do not depend on it; the failure is preserved in the receipt.
        status = 'COMPLETE' if not failures else (
            'PARTIAL_COMPLETE_WITH_BLOCKERS' if len(failures) < len(STAGES) else 'FAILED_PRESERVED')
        state.update(status=status, failures=failures,
                     finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     stage='DONE')
        progress.update(stage='DONE', phase=status, failures=failures)
    finally:
        release()
    return 0 if not failures else 1


def _patch_queue(queue, name, entry):
    try:
        cur = common.read_json(queue.path)
    except Exception:
        cur = {'queue': []}
    rows = cur.get('queue') or []
    for row in rows:
        if row['stage'] == name:
            row.update(entry)
            break
    else:
        rows.append(dict(entry))
    cur['queue'] = rows
    cur['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    common.atomic_write_json(queue.path, cur)


if __name__ == '__main__':
    raise SystemExit(main())
