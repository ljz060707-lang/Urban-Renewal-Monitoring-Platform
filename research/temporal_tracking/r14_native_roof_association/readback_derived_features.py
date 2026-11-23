from pathlib import Path
import json,hashlib,tarfile,time
R=Path(__file__).parent;p=R/'DERIVED_FEATURES_FROZEN.tar.returning';assert hashlib.sha256(p.read_bytes()).hexdigest()=='c82c4822debbc4c3c665a487aaaab9586ba141dd3507fa2f5c67f5bfa8ae917d';dest=R/'incoming'
with tarfile.open(p) as tf:
 manifest=json.loads(tf.extractfile('DERIVED_FEATURES_SHA256.json').read());expected={r['path']:r for r in manifest}
 for m in tf.getmembers():
  target=(dest/m.name).resolve();assert m.isfile() and target.is_relative_to(dest.resolve());b=tf.extractfile(m).read()
  if m.name in expected:assert hashlib.sha256(b).hexdigest()==expected[m.name]['sha256']
  target.parent.mkdir(parents=True,exist_ok=True);t=target.with_name(target.name+'.returning');t.write_bytes(b);t.replace(target)
p.replace(R/'DERIVED_FEATURES_FROZEN.tar');(R/'DERIVED_FEATURE_LOCAL_READBACK.json').write_text(json.dumps(dict(status='PASS_ARCHIVE_AND_ALL_14_MEMBER_SHA',time=time.time(),files=manifest),indent=2));print('PASS_DERIVED_LOCAL_READBACK',len(manifest))
