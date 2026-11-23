"""Independent CPU reload against saved CUDA calibration outputs."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from temporal_model import TemporalIdentityNet
ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(2);receipt=json.loads((ROOT/'evidence/SN7_FULL_SPAN_DATASET_RECEIPT.json').read_text());first=next(r for r in receipt['shard_records'] if r['role']=='CALIBRATION')
    with np.load(first['path']) as d:
        data={k:torch.from_numpy(d[k][:8]).float()/(255. if k in ['q','c','s'] else 1.) for k in ['q','c','g','s','times']}
    results=[]
    for seed in [20261027,20261028,20261029]:
        p=ROOT/'runs/T2_FULL_SPAN'/('seed_'+str(seed));checkpoint=p/'best.pt';c=torch.load(checkpoint,map_location='cpu');m=TemporalIdentityNet();m.load_state_dict(c['model'],strict=True);m.eval()
        with torch.no_grad():o=m(data['q'],data['c'],data['g'],data['s'],data['times'])
        with np.load(p/'calibration_predictions.npz') as d:
            error=float(np.max(abs(o['association'].softmax(-1).numpy()-d['probs'][:8])));pe=float(np.max(abs(o['presence'].sigmoid().numpy()-d['pre'][:8])))
        assert error<.0001 and pe<.0001
        results.append(dict(seed=seed,strict_load=True,checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            CPU_CUDA_association_max_abs=error,CPU_CUDA_presence_max_abs=pe,samples=8))
    counts={}
    for n,p in m.named_parameters():counts[n.split('.')[0]]=counts.get(n.split('.')[0],0)+p.numel()
    (ROOT/'evidence/INDEPENDENT_RELOAD_QA.json').write_text(json.dumps(dict(status='PASS',records=results,parameters_by_branch=counts,total_parameters=sum(counts.values()),
        primary_T2_path_parameters=sum(v for k,v in counts.items() if k!='bt_flow'),auxiliary_BT_head_T2_unused=counts['bt_flow'],
        scope='T1 trains vector auxiliary path; T2 loads encoder only and trains identity/temporal/presence paths; BT head is not called by deployment tracking'),indent=2))
if __name__=='__main__':main()
