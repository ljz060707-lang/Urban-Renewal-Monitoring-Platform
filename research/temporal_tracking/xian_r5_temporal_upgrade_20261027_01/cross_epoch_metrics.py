"""Exclude same-image reference controls from cross-temporal pair metrics."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    data=json.loads((ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json').read_text());rows=[]
    for r in data['shard_records']:
        if r['role']=='CALIBRATION':rows.extend(json.loads(Path(r['path']).with_suffix('.json').read_text())['rows'])
    result={}
    for name in ['B2_BASELINE','20261027','20261028','20261029']:
        p=ROOT/'evidence/B2_FULLSPAN_SOURCE_PREDICTIONS.npz' if name=='B2_BASELINE' else ROOT/'runs/T2_FULL_SPAN'/('seed_'+name)/'calibration_predictions.npz'
        with np.load(p) as d:y=d['y'].copy();probs=d['probs']
        controls=0
        for i,r in enumerate(rows):
            j=r.get('reference_local_index',r['reference_frame']-r['window']);controls+=int((y[i,j]>=0).sum());y[i,j]=-100
        keep=y>=0;truth=y[keep];pred=probs.argmax(-1)[keep];cm=np.zeros((3,3),int);np.add.at(cm,(truth,pred),1)
        precision=np.diag(cm)/np.maximum(cm.sum(0),1);recall=np.diag(cm)/np.maximum(cm.sum(1),1);f1=2*precision*recall/np.maximum(precision+recall,1e-8)
        result[name]=dict(macro_f1=float(f1.mean()),precision=precision.tolist(),recall=recall.tolist(),confusion=cm.tolist(),cross_epoch_pairs=int(keep.sum()),same_epoch_controls_excluded=controls,
            scope='CALIBRATION source internal group, GT proposals, same-reference epoch excluded; not target or end-to-end tracking accuracy')
    (ROOT/'evidence/CROSS_EPOCH_ONLY_SOURCE_METRICS.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
