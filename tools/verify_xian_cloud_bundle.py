"""Verify a private, explicit cloud payload before exclusive extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for c in iter(lambda:s.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sha',required=True);a=p.parse_args()
    meta=json.loads(a.manifest.read_text())
    if sha(a.archive)!=a.expected_sha or meta['sha256']!=a.expected_sha:raise ValueError('Archive SHA mismatch')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.archive) as tar:
        members=tar.getmembers()
        if {m.name for m in members}!=set(meta['files']) or len(members)!=len(meta['files']):
            raise ValueError('Member inventory mismatch')
        for m in members:
            part=PurePosixPath(m.name)
            if (not m.isfile() or part.is_absolute() or '..' in part.parts or
                    part.parts[0] not in ('data','config','code','weights','prompts') or
                    'private' in part.parts or part.suffix.lower() in ('.key','.pem','.ppk')):
                raise ValueError('Unsafe/forbidden archive member')
            target=output.joinpath(*part.parts)
            if target.exists():raise ValueError('Exclusive extraction refuses overwrites: '+m.name)
        for m in members:
            target=output.joinpath(*PurePosixPath(m.name).parts);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src, target.open('xb') as dst:shutil.copyfileobj(src,dst)
            expected=meta['files'][m.name]
            if target.stat().st_size!=expected['bytes'] or sha(target)!=expected['sha256']:
                raise ValueError('Extracted member checksum mismatch')
    receipt=dict(status='PASS',archive_sha256=a.expected_sha,verified_files=len(members),
                 no_credentials=True,exclusive_extraction=True)
    path=output.parent/('VERIFIED_'+meta['part'].upper()+'_PAYLOAD.json')
    with path.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
