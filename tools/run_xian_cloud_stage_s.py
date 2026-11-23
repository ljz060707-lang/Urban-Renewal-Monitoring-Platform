"""Frozen phase-S launch, not a scheduler. Existing instance only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(4194304),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--launch-manifest',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();launch=json.loads(a.launch_manifest.read_text())
    for name,expected in launch['input_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Frozen phase-S input changed: '+name)
    protocol=json.loads((root/'incoming/PHASED_TRAINING_PROTOCOL.json').read_text())['S']
    if datetime.now().astimezone()>=datetime.fromisoformat(protocol['hard_training_deadline']):
        raise ValueError('Training deadline already passed')
    sys.path[:0]=[str(root/'payload/code/src'),str(root/'payload/code/tools')]
    from xian_cloud_s_train import run_s
    result=run_s(root/'payload',root/'S_TRAINING_01',protocol)
    receipt=dict(training_executed=result['training_executed'],result=result,
                 frozen_input_hashes_unchanged=all(sha(root/n)==h for n,h in launch['input_sha256'].items()))
    with (root/'S_PHASE_EXIT.json').open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt),flush=True)
    if not result['training_executed'] or not receipt['frozen_input_hashes_unchanged']:raise SystemExit(1)


if __name__=='__main__':main()
