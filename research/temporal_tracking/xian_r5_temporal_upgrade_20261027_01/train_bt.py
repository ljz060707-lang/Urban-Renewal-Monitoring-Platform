import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,hashlib,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from temporal_model import TemporalIdentityNet
ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(4);torch.manual_seed(20261027);torch.cuda.manual_seed_all(20261027);torch.use_deterministic_algorithms(True)
    run=ROOT/'runs/T1_BT';run.mkdir(exist_ok=True)
    if (run/'COMPLETE.json').exists():return
    ds={}
    for role in ['FIT','TUNE']:
        with np.load(ROOT/'data'/('bt_'+role+'.npz')) as a:
            ds[role]=torch.utils.data.TensorDataset(torch.from_numpy(a['a']),torch.from_numpy(a['b']),torch.from_numpy(a['flow']))
    model=TemporalIdentityNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);hist=[];best=float('inf');stale=0
    for epoch in range(20):
        start=time.monotonic();model.train();total=0.;seen=0
        for a,b,y in torch.utils.data.DataLoader(ds['FIT'],batch_size=128,shuffle=True,pin_memory=True):
            a=a.cuda().float()/255.;b=b.cuda().float()/255.;y=y.cuda()
            pred=model.flow_forward(a,b);loss=F.smooth_l1_loss(pred,y)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss.detach())*len(y);seen+=len(y)
        model.eval();epe=[];zero=[]
        with torch.no_grad():
            for a,b,y in torch.utils.data.DataLoader(ds['TUNE'],batch_size=128,pin_memory=True):
                y=y.cuda();p=model.flow_forward(a.cuda().float()/255.,b.cuda().float()/255.)
                epe.extend((torch.linalg.vector_norm(p-y,dim=1)*128).cpu().tolist());zero.extend((torch.linalg.vector_norm(y,dim=1)*128).cpu().tolist())
        score=float(np.mean(epe));row=dict(epoch=epoch,seen=seen,loss=total/seen,tune_native_vector_EPE=score,zero_vector_EPE=float(np.mean(zero)),seconds=time.monotonic()-start);hist.append(row)
        if score<best-.01:
            best=score;stale=0;torch.save(dict(model=model.state_dict(),epoch=epoch,history=hist,kind='BANDON_BT_AUX_ONLY'),run/'best.pt')
        else:stale+=1
        (run/'history.json').write_text(json.dumps(hist,indent=2));print(json.dumps(row),flush=True)
        if epoch>=3 and stale>=4:break
    (run/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',epochs=len(hist),checkpoint=str(run/'best.pt'),sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),best_tune_EPE=best,
        zero_vector_EPE=hist[0]['zero_vector_EPE'],benefit_verified=best<hist[0]['zero_vector_EPE'],identity_accuracy_claim=False),indent=2))
if __name__=='__main__':main()
