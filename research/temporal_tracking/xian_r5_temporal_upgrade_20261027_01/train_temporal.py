"""Full-epoch training; immutable source groups; serial independent seeds."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,hashlib,random,time,argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from temporal_model import TemporalIdentityNet
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
class Dataset(torch.utils.data.Dataset):
    def __init__(self,role):
        receipt=json.loads((ROOT/'evidence/SN7_DATASET_RECEIPT.json').read_text())
        self.meta=[]; self.arr={k:[] for k in ['q','c','g','s','y','p','v','times']}
        for r in receipt['shard_records']:
            if r['role']!=role:continue
            assert sha(r['path'])==r['sha256']
            with np.load(r['path']) as d:
                for k in self.arr:self.arr[k].append(d[k])
            self.meta.extend(json.loads(Path(r['path']).with_suffix('.json').read_text())['rows'])
        self.arr={k:np.concatenate(v) for k,v in self.arr.items()}
        assert len(self.meta)==len(self.arr['q'])
    def __len__(self):return len(self.meta)
    def __getitem__(self,i):return {k:torch.from_numpy(v[i]) for k,v in self.arr.items()}
def device_batch(batch):
    return {k:v.cuda(non_blocking=True).float()/255. if k in ['q','c','s'] else v.cuda(non_blocking=True) for k,v in batch.items()}
def metrics(probs,truth,pre,pt,valid):
    y=truth.reshape(-1); p=probs.reshape(-1,3); keep=y>=0;y=y[keep];p=p[keep]
    pred=p.argmax(1); cm=np.zeros((3,3),int)
    np.add.at(cm,(y,pred),1); precision=np.diag(cm)/np.maximum(cm.sum(0),1);recall=np.diag(cm)/np.maximum(cm.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-8)
    known=valid.astype(bool); pp=pre>=.5
    pr_cm=np.zeros((2,2),int);np.add.at(pr_cm,(pt[known].astype(int),pp[known].astype(int)),1)
    # Oracle proposals are supplied from labels. Never call these end-to-end tracking metrics.
    choice=probs[:,:,:,0].argmax(2); take=np.take_along_axis(truth,choice[:,:,None],2).squeeze(2)
    accept=np.take_along_axis(probs[:,:,:,0],choice[:,:,None],2).squeeze(2)>=.5
    present=(truth==0).any(2)&known;correct=(take==0)&accept&known
    fp=int(((take!=0)&accept&known).sum());fn=int((present&~correct).sum());tp=int(correct.sum())
    return dict(confusion=cm.tolist(),precision=precision.tolist(),recall=recall.tolist(),macro_f1=float(f1.mean()),accuracy=float((pred==y).mean()),
        labelled_presence_confusion=pr_cm.tolist(),labelled_presence_accuracy=float((pp[known]==pt[known]).mean()),
        query_conditioned_oracle_proposal_IDF1=float(2*tp/max(2*tp+fp+fn,1)),IDTP=tp,IDFP=fp,IDFN=fn,
        end_to_end_IDF1=None,HOTA=None,SCOT=None,roof_detection_recall=None,
        metric_limit='GT proposal sampling, internal source groups, not target accuracy; label-record presence is not direct visibility')
@torch.no_grad()
def evaluate(model,loader):
    model.eval(); lists={k:[] for k in ['probs','y','pre','p','v']}
    for batch in loader:
        b=device_batch(batch);o=model(b['q'],b['c'],b['g'],b['s'],b['times'])
        for k,v in dict(probs=o['association'].softmax(-1),y=b['y'],pre=o['presence'].sigmoid(),p=b['p'],v=b['v']).items():lists[k].append(v.cpu().numpy())
    a={k:np.concatenate(v) for k,v in lists.items()};return metrics(a['probs'],a['y'],a['pre'],a['p'],a['v']),a
def wilson(errors,n):
    if not n:return 1.
    p=errors/n;z=1.96
    return float((p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))
def calibrate(a):
    p=a['probs'].reshape(-1,3);y=a['y'].reshape(-1);mask=y>=0;p=p[mask];y=y[mask]
    prediction=p.argmax(-1);confidence=p.max(-1);curve=[]
    for threshold in np.linspace(.34,.999,80):
        take=confidence>=threshold;n=int(take.sum());err=int((prediction[take]!=y[take]).sum())
        curve.append(dict(threshold=float(threshold),accepted=n,coverage=float(n/len(y)),error_rate=err/n if n else None,error_upper95=wilson(err,n)))
    good=[r for r in curve if r['accepted']>=30 and r['error_upper95']<=.05]
    point=max(good,key=lambda x:x['coverage']) if good else None
    # SAME edges require their own error calibration; aggregate easy DIFF is insufficient.
    edges=[]
    for threshold in np.linspace(.34,.999,80):
        take=(prediction==0)&(p[:,0]>=threshold);n=int(take.sum());err=int((y[take]!=0).sum())
        edges.append(dict(threshold=float(threshold),accepted=n,coverage=float(n/len(y)),false_same_rate=err/n if n else None,error_upper95=wilson(err,n)))
    good=[r for r in edges if r['accepted']>=30 and r['error_upper95']<=.05]
    edge=max(good,key=lambda x:x['coverage']) if good else None
    return dict(curve=curve,same_edge_curve=edges,operating_point=point,same_edge_operating_point=edge,scope='SOURCE_INTERNAL_ONLY',
        target_calibration='NOT_AVAILABLE',production_admission=False,rule='Wilson upper95 <= .05 and n>=30; no passing point means no accepted edge')
def train(seed,fit,tune,cal):
    run=ROOT/'runs'/('seed_'+str(seed)); run.mkdir(exist_ok=True)
    if (run/'TRAINING_COMPLETE.json').exists():return json.loads((run/'TRAINING_COMPLETE.json').read_text())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=TemporalIdentityNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    parent=None
    bt=ROOT/'runs/T1_BT/COMPLETE.json'
    if bt.exists() and json.loads(bt.read_text())['benefit_verified']:
        parent=json.loads(bt.read_text());warm=torch.load(parent['checkpoint'],map_location='cpu')
        model.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in warm['model'].items() if k.startswith('encoder.')},strict=True)
    fl=torch.utils.data.DataLoader(fit,batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
    tl=torch.utils.data.DataLoader(tune,batch_size=32,num_workers=0,pin_memory=True)
    cl=torch.utils.data.DataLoader(cal,batch_size=32,num_workers=0,pin_memory=True)
    hist=[];best=-1.;start=0;stale=0
    cp=run/'last.pt'
    if cp.exists():
        c=torch.load(cp,map_location='cpu'); model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);start=c['epoch']+1
        hist=c['history'];best=c['best'];stale=c['stale'];torch.set_rng_state(c['torch_rng']);np.random.set_state(c['numpy_rng']);random.setstate(c['random_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
    # Class counts include distinct source examples. No positive/negative preprocessing difference.
    y=fit.arr['y']; counts=np.bincount(y[y>=0],minlength=3);w=torch.tensor(1/np.sqrt(np.maximum(counts,1)),device='cuda',dtype=torch.float32);w=w/w.mean()
    targets=fit.arr['p'][fit.arr['v']>0];pw=torch.tensor(float((1-targets).sum()/max(targets.sum(),1)),device='cuda').clamp(.1,10)
    for epoch in range(start,30):
        t0=time.monotonic();model.train();loss_sum=0.;seen=0
        for batch in fl:
            b=device_batch(batch)
            # Same brightness distribution for every query/candidate/search image and class.
            factor=torch.empty((len(b['q']),1,1,1),device='cuda').uniform_(.85,1.15)
            for k in ['q','c','s']:
                f=factor
                while f.ndim<b[k].ndim:f=f.unsqueeze(1)
                rgb=(b[k][...,:3,:,:]*f).clamp(0,1);b[k]=torch.cat([rgb,b[k][...,3:,:,:]],dim=-3)
            o=model(b['q'],b['c'],b['g'],b['s'],b['times'])
            ce=F.cross_entropy(o['association'].reshape(-1,3),b['y'].reshape(-1),weight=w,ignore_index=-100)
            pl=F.binary_cross_entropy_with_logits(o['presence'],b['p'],pos_weight=pw,reduction='none')
            pl=(pl*b['v']).sum()/b['v'].sum().clamp(min=1)
            vl=F.binary_cross_entropy_with_logits(o['observable'],b['v'])
            loss=ce+.4*pl+.2*vl
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            loss_sum+=float(loss.detach())*len(b['q']);seen+=len(b['q'])
        m,_=evaluate(model,tl);score=m['macro_f1'];improved=score>best+.001
        if improved:best=score;stale=0
        else:stale+=1
        row=dict(epoch=epoch,seed=seed,seen=seen,loss=loss_sum/seen,seconds=time.monotonic()-t0,tune=m,best=best,stale=stale)
        hist.append(row); print(json.dumps(row),flush=True)
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,history=hist,best=best,stale=stale,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),random_rng=random.getstate(),
            architecture='TemporalIdentityNet',parameters=sum(p.numel() for p in model.parameters()),protocol_sha256=sha(ROOT/'evidence/TRAINING_PROTOCOL.json'),
            dataset_receipt_sha256=sha(ROOT/'evidence/SN7_DATASET_RECEIPT.json'),seed=seed,classes=['SAME_ID','DIFFERENT_ID','NO_LABELLED_COUNTERPART'])
        state['pretraining_parent']=parent
        torch.save(state,cp)
        if improved:torch.save(state,run/'best.pt')
        write(run/'history.json',hist)
        if epoch>=7 and stale>=5:break
    chosen=torch.load(run/'best.pt',map_location='cpu');model.load_state_dict(chosen['model']);tm,_=evaluate(model,tl);cm,a=evaluate(model,cl)
    np.savez(run/'calibration_predictions.npz',**a);write(run/'CALIBRATION.json',calibrate(a))
    result=dict(status='TRAINING_COMPLETE_RESEARCH_ONLY',seed=seed,epochs=len(hist),best_epoch=chosen['epoch'],stop='validation_plateau' if stale>=5 else 'predeclared_max_epochs',
        full_fit_epochs=True,fit_sequences=len(fit),tune_sequences=len(tune),calibration_sequences=len(cal),parameters=chosen['parameters'],
        checkpoint=str(run/'best.pt'),checkpoint_sha256=sha(run/'best.pt'),tune=tm,calibration=cm,target_independent_accuracy=None)
    write(run/'TRAINING_COMPLETE.json',result);return result
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261027);a=p.parse_args()
    fit,tune,cal=[Dataset(r) for r in ['FIT','TUNE','CALIBRATION']]
    train(a.seed,fit,tune,cal)
if __name__=='__main__':main()
