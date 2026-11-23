"""Freeze completed phase outputs for resumable copy; never include raw data."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda:s.read(4194304),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('S','FINAL'),required=True);a=p.parse_args();root=a.root.resolve()
    gate=root/('S_PHASE_EXIT.json' if a.phase=='S' else 'FOLLOWUP_EXIT.json')
    if not gate.is_file():raise ValueError('Phase still running; output not frozen')
    output=root/('EXPORT_'+a.phase+'.json')
    if output.exists():
        print(output.read_text(),flush=True);return
    directories=['S_TRAINING_01']
    if a.phase=='FINAL':directories+=['A_TRAINING_01','JOINT_TRAINING_01','RELEASE','RELEASE_GEO','RELEASE_PUBLIC_EVAL','payload/code','payload/config','logs']
    files={}
    for name in directories:
        directory=root/name
        if not directory.exists():continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.is_symlink() or path.suffix in ('.tmp','.pem','.key','.ppk','.tar'):continue
            relative=path.relative_to(root).as_posix()
            if '__pycache__' in path.parts or 'private' in path.parts:continue
            files[relative]=dict(sha256=sha(path),bytes=path.stat().st_size)
    for name in ('S_PHASE_EXIT.json','FOLLOWUP_EXIT.json','VERIFIED_DATA_PAYLOAD.json','VERIFIED_CODE_PAYLOAD.json',
                 'VERIFIED_NATIVE_PAYLOAD.json','FINAL_RECEIPT.json','FINAL_REPORT.md','FOLLOWUP_LAUNCH_MANIFEST.json'):
        path=root/name
        if path.is_file():files[name]=dict(sha256=sha(path),bytes=path.stat().st_size)
    report=dict(schema='XIAN_CLOUD_F_EXPORT_V1',phase=a.phase,files=files,
         total_bytes=sum(x['bytes'] for x in files.values()),raw_data_recopy=False,
         B0_required_sha256='f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7',
         no_credentials=True,no_instance_release_claim=True)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
