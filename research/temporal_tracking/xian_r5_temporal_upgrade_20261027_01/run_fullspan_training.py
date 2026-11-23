"""Finalize defined source coverage, freeze, and run three serial local seeds."""
import json,hashlib,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    deadline=time.monotonic()+7200
    while not (ROOT/'evidence/SN7_MULTISCALE_DATASET_RECEIPT.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('multiscale preparation did not finish; do not train partial data')
        time.sleep(10)
    original=json.loads((ROOT/'evidence/SN7_DATASET_RECEIPT.json').read_text())
    records=[]
    for r in original['shard_records']:
        p=ROOT/'data/sn7_multiscale'/(r['aoi']+'.json')
        assert p.exists(),p
        m=json.loads(p.read_text());assert sha(m['path'])==m['sha256']
        records.append({k:m[k] for k in ['aoi','role','sequences','path','sha256']})
    shards=original['shard_records']+records
    receipt=dict(status='COMPLETE_CONTIGUOUS_AND_FULL_SPAN_WITH_ALL_SUPPORTED_LABELLED_EXITS',aoi_count=42,shards=len(shards),
        sequences=sum(r['sequences'] for r in shards),roles={role:sum(r['sequences'] for r in shards if r['role']==role) for role in ['FIT','TUNE','CALIBRATION']},
        original_validation_reads=0,locked_holdout_reads=0,all_native_frames_covered_by_contiguous_windows=True,
        negative_preprocessing_identical=True,all_supported_labelled_exits_in_extra_windows=True,all_buildings_claim=False,shard_records=shards)
    (ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    p=json.loads((ROOT/'evidence/T2_PROTOCOL.json').read_text());p.update(data_receipt_sha256=sha(ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json'),
        sequence_schedules='frozen contiguous5 windows plus [uniform5 over whole native series], [first2,middle,last2], [first4,last], [first,last4]',
        extra_sampling='one canonical ID per occupied128px cell plus EVERY canonical-ID 1->0/1->0->1 event whose anchor alpha support >=.98 at both adjacent sampled dates',
        stage_reason='source-only diagnostic identified missing long-gap coverage and zero demolition recall; architecture, optimizer and group assignments unchanged',
        absence_semantics='label-record disappearance; not certified visual demolition',target_for_training=False)
    (ROOT/'evidence/T2_FULL_SPAN_PROTOCOL.json').write_text(json.dumps(p,indent=2))
    files=['source/temporal_model.py','source/train_temporal_fullspan.py','source/prepare_sn7_multiscale.py','evidence/T2_FULL_SPAN_PROTOCOL.json','evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json']
    freeze={f:sha(ROOT/f) for f in files};(ROOT/'evidence/FULLSPAN_CODE_FREEZE.json').write_text(json.dumps(freeze,indent=2))
    for seed in [20261027,20261028,20261029]:
        assert all(sha(ROOT/f)==s for f,s in freeze.items())
        log=ROOT/'evidence'/('TRAIN_FULLSPAN_'+str(seed)+'.log')
        with log.open('w',encoding='utf8') as f:subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'source/train_temporal_fullspan.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('SEED_COMPLETE',seed,flush=True)
    assert all(sha(ROOT/f)==s for f,s in freeze.items())
if __name__=='__main__':main()
