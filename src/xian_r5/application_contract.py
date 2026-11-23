"""Application contracts and read-only, content-addressed stage commits."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import date

SCHEMA = 'XIAN_R5_APPLICATION_V1'
B0 = 'f71c0c0807e33ef422511e9b24766dbe89e7215d0837ea21d0ffa2d8d185cfa7'
V2 = '00d668cb8838aed8537db451c54bed887ccfeb578d1f9c07031b21b5823d65e5'
E3 = 'edd60b9fa93b8f8229c64e3349e72cb49803dc600ac3a6263b7b69af368e100c'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read(path):
    with Path(path).open(encoding='utf-8-sig') as f: return json.load(f)

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    tmp = path.with_name(path.name+'.pending')
    with tmp.open('w', encoding='utf-8') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    json.loads(tmp.read_text(encoding='utf-8'))
    os.replace(tmp, path)

def lines(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n' for x in rows), encoding='utf-8')

def table(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', value) or value in ('.','..'):
        raise ValueError('Expected safe ASCII identifier: '+str(value))
    return value

def interval(epoch):
    if epoch.get('actual_date'):
        d = date.fromisoformat(epoch['actual_date']); return (d,d)
    if epoch.get('actual_time_window'):
        a,b=map(date.fromisoformat,epoch['actual_time_window'])
        if a>b: raise ValueError('Reversed observation window')
        return a,b
    return None

def valid_pair(a,b):
    aa,bb=interval(a),interval(b)
    return bool(aa and bb and aa[1]<bb[0])

def comparison_epochs(epochs):
    """Nearest distinct observation windows; no annual interval requirement."""
    result=[];seen=set()
    for epoch in sorted(epochs,key=lambda e:((interval(e) or (date.max,date.max))[0],e['epoch_id'])):
        window=interval(epoch)
        if window is None:continue
        if window in seen:continue
        seen.add(window);result.append(epoch)
    return result

def check_model_route(routes):
    for role, expected, key, semantics in [
        ('roof',B0,'state_dict','VISIBLE_ROOF_IN_SOURCE_IMAGE_FRAME'),
        ('alignment',V2,'models.M_CORR','IMAGE_TO_REFERENCE_GRID'),
        ('facade',E3,'S','PROJECTED_FACADE_AUXILIARY')]:
        r=routes[role]
        if r['checkpoint_sha256']!=expected or r['state_container_key']!=key or r['target_semantics']!=semantics:
            raise ValueError('Model role/container/semantic mismatch: '+role)
        if sha(r['checkpoint'])!=expected: raise ValueError('Checkpoint changed: '+role)
    if routes['roof']['threshold_comparator']!='>0.5' or routes['facade']['activation']!='independent_sigmoid':
        raise ValueError('Segmentation output contract changed')
    if routes['alignment']['normalization']!='RGB_FLOAT32_DIV255': raise ValueError('Alignment normalization mismatch')

class Stages:
    def __init__(self, root, identity, deadline=None):
        self.root=Path(root); self.identity=identity; self.deadline=deadline; self.reused=[]; self.executed=[]
    def guard(self):
        if self.deadline and time.monotonic()>self.deadline: raise TimeoutError('Application wallclock limit')
    def run(self, name, deps, fn):
        self.guard(); key=digest([self.identity, deps]); directory=self.root/'stages'/name
        commit=directory/'COMMIT.json'
        if commit.exists():
            r=read(commit)
            if r['dependency_sha256']!=key: raise ValueError('Changed dependency requires new run directory: '+name)
            for p,h in r['files'].items():
                if sha(directory/p)!=h: raise ValueError('Corrupted committed stage: '+name+'/'+p)
            self.reused.append(name); return directory
        directory.mkdir(parents=True,exist_ok=True)
        # Failed attempts remain for inspection. Work in a fresh attempt.
        attempt=directory/('attempt_%03d'%(1+len(list(directory.glob('attempt_*')))))
        attempt.mkdir(); fn(attempt); self.guard()
        files={str(p.relative_to(directory)):sha(p) for p in attempt.rglob('*') if p.is_file()}
        save(commit,dict(dependency_sha256=key,files=files,attempt=attempt.name))
        self.executed.append(name); return directory
    @staticmethod
    def output(directory): return directory/read(directory/'COMMIT.json')['attempt']
