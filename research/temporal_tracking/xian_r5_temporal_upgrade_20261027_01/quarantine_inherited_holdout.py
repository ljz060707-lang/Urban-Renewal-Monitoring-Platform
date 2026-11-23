"""Quarantine inherited FINAL_TEST auxiliary bytes acquired with official TRAIN.
No protected RGB is decoded. Preserve acquisition receipt and record the move.
"""
import json,hashlib
from pathlib import Path
from prepare_bt import ROOT,OLD,PAY
def main():
    rows={r['id']:r for r in json.loads((PAY/'config/segmentation_data.json').read_text())['records'] if r['dataset']=='BANDON'}
    paths=[];pairs=[]
    for i,line in enumerate((OLD/'official_sources/lists/list_BANDON_train.txt').read_text().splitlines()):
        parts=line.split();ids=['BANDON_train_'+p.lstrip('/').rsplit('.',1)[0].replace('/','_') for p in parts[:2]]
        if not any(i in rows and rows[i]['split']=='FINAL_TEST' for i in ids):continue
        pairs.append(i)
        for j in [2,7]:paths.append(Path('train')/parts[j].lstrip('/'))
    records=[];source_root=(ROOT/'data/bandon_bt').resolve();dest_root=(ROOT/'data/QUARANTINE_INHERITED_FINAL_TEST').resolve()
    assert source_root.is_relative_to(ROOT.resolve()) and dest_root.is_relative_to(ROOT.resolve())
    for rel in sorted(set(paths)):
        src=(source_root/rel).resolve();dst=(dest_root/rel).resolve()
        assert src.is_relative_to(source_root) and dst.is_relative_to(dest_root)
        if src.exists():
            assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
        assert dst.exists()
        records.append(dict(original=str(src),quarantined=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
    (ROOT/'evidence/INHERITED_FINAL_TEST_QUARANTINE.json').write_text(json.dumps(dict(status='QUARANTINED_NOT_TRAINED_OR_EVALUATED',pairs=len(pairs),files=len(records),
        acquisition_scope_issue='Initial acquisition selected every official TRAIN flow/change member before inherited parent-role filtering, including54 inherited FINAL_TEST pairs. Raw auxiliary bytes were decompressed and CRC-checked. No protected RGB was decoded or used in training/evaluation.',
        correction='Role-filtered learning and source evaluation; excluded auxiliary files moved to explicit quarantine; original compressed-range evidence retained; no new formal FINAL_TEST claim.',
        original_assets_modified=False,protected_RGB_decode=0,protected_loss_or_metric_use=0,records=records),indent=2))
if __name__=='__main__':main()
