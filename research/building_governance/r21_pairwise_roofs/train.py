"""Train within-parent same/different-identity affinity; inference has no label input."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prepare import R,OUT,N,verify,digest,save
SEED=211111
STEPS=750
ARMS=('native','native_and_map')
def conv(c,d):
    return nn.Sequential(nn.Conv2d(c,d,3,padding=1),nn.GroupNorm(4,d),nn.SiLU(),nn.Conv2d(d,d,3,padding=1),nn.GroupNorm(4,d),nn.SiLU())
class Affinity(nn.Module):
    def __init__(self):
        super().__init__()
        self.low=conv(4,16);self.mid=conv(16,24);self.high=conv(24,40)
        self.project=nn.Sequential(nn.Linear(16+24+40+3+2+3,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32))
        self.score=nn.Sequential(nn.Linear(66,48),nn.SiLU(),nn.Linear(48,1))
        ax=(torch.arange(N)+.5)/N
        self.register_buffer('xy',torch.stack(torch.meshgrid(ax,ax,indexing='ij'))[None])
    def forward(self,im,labels,sizes):
        x=im.permute(0,3,1,2).float()/255
        a=self.low(x);b=self.mid(F.avg_pool2d(a,2));c=self.high(F.avg_pool2d(b,2))
        features=torch.cat((a,F.interpolate(b,size=(N,N),mode='bilinear',align_corners=False),F.interpolate(c,size=(N,N),mode='bilinear',align_corners=False),x[:,:3],self.xy.expand(len(x),-1,-1,-1)),1)
        # Average within visible subregions, including their relative coordinates.
        ii=labels.long().flatten(1)[:,None,:]
        count=x.new_zeros(len(x),1,65).scatter_add_(2,ii,torch.ones_like(ii,dtype=x.dtype))
        pooled=x.new_zeros(len(x),features.shape[1],65).scatter_add_(2,ii.expand(-1,features.shape[1],-1),features.flatten(2))/count.clamp_min(1)
        nodes=pooled.transpose(1,2)
        size=torch.log1p(sizes.float())/8
        z=self.project(torch.cat((nodes,size[:,None,:].expand(-1,65,-1),count.transpose(1,2)/(N*N)),2))
        diff=(z[:,:,None]-z[:,None,:]).abs()
        prod=z[:,:,None]*z[:,None,:]
        coords=(nodes[:,:,None,-2:]-nodes[:,None,:,-2:]).abs()
        return self.score(torch.cat((diff,prod,coords),-1)).squeeze(-1)
def load(role,frames,labs):
    data=[[],[],[],[]]
    for rec in frames:
        if rec['role']!=role:continue
        f=OUT/(rec['id']+'.npz');t=OUT/(rec['id']+'.labels.npy')
        verify(f,rec['sha']);verify(t,labs[rec['id']]['sha'])
        targets=np.load(t)
        with np.load(f) as z:
            valid=(targets>0).sum(1)>=2
            boxes=z['boxes'][valid]
            arrays=(z['imgs'][valid],z['nodes'][valid],np.stack((boxes[:,2]-boxes[:,1],boxes[:,4]-boxes[:,3]),1),targets[valid])
            for col,v in zip(data,arrays):col.append(v)
    merged=[np.concatenate(col) for col in data]
    if role=='MAP_FIT' or role=='MAP_INNER':
        ids=set(np.unique(merged[3]).tolist())-{0}
        save(R/(role+'_IDS.json'),sorted(ids))
    types=np.array([len(set(t[t>0]))>=2 for t in merged[3]])
    return [torch.from_numpy(v).cuda() for v in merged],np.where(types)[0],np.where(~types)[0]
def get_batch(pool,number,rng):
    data,multi,single=pool
    n=number*2//3
    idx=np.concatenate([rng.choice(multi,n),rng.choice(single,number-n)])
    return [a[idx] for a in data]
def main():
    frames=json.loads((R/'IMAGE_ONLY_FROZEN.json').read_text());labs={r['id']:r for r in json.loads((R/'SUPERVISION.json').read_text())}
    params=dict(seed=SEED,steps=STEPS,optimizer='AdamW',lr=.0008,weight_decay=.01,batch=32,cuts=[.40,.55,.70,.82,.92],
        selection='source INNER: maximize F1 among operating points with no extra fragmented GT and no loss of complete90 if available; then lower fragment count; all failing variants research only',
        map_inner='spatially disjoint partial-map pair validation only; source and mixed arms evaluated without target-TUNE selection',candidate='average-linkage over all in-parent pairs then absorb small clusters, preserve raw union')
    save(R/'TRAIN_PROTOCOL.json',params)
    pools={k:load(k,frames,labs) for k in ('SOURCE_FIT','MAP_FIT','SOURCE_INNER','MAP_INNER')}
    assert not set(json.loads((R/'MAP_FIT_IDS.json').read_text())).intersection(json.loads((R/'MAP_INNER_IDS.json').read_text()))
    print({k:(len(v[1]),len(v[2])) for k,v in pools.items()},flush=True)
    for arm in ARMS:
        dest=R/arm;dest.mkdir(exist_ok=True)
        if (dest/'TRAINED.json').exists():continue
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED);secondary=np.random.default_rng(SEED+1)
        model=Affinity().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0008)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,.00008)
        with (dest/'CURVES.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['step','loss','same_accuracy','different_accuracy','lr'])
            for step in range(1,STEPS+1):
                base=get_batch(pools['SOURCE_FIT'],20,rng)
                extra=get_batch(pools['SOURCE_FIT' if arm=='native' else 'MAP_FIT'],12,secondary)
                imgs,nodes,sizes,t=[torch.cat((a,b)) for a,b in zip(base,extra)]
                if step%2:imgs=imgs.flip((2,));nodes=nodes.flip((2,))
                if step%3:imgs=imgs.flip((1,));nodes=nodes.flip((1,))
                logits=model(imgs,nodes,sizes)
                valid=((t[:,:,None]>0)&(t[:,None,:]>0)).triu(1)
                same=(t[:,:,None]==t[:,None,:])&valid
                different=(~same)&valid
                assert same.any() and different.any()
                loss=(F.softplus(-logits[same]).mean()+F.softplus(logits[different]).mean())/2
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                w.writerow([step,float(loss),float((logits[same]>0).float().mean()),float((logits[different]<0).float().mean()),opt.param_groups[0]['lr']]);schedule.step()
                if step%150==0:print(arm,step,float(loss),flush=True);f.flush()
        path=dest/'weights.pt';torch.save(dict(state_dict=model.state_dict(),steps=STEPS),path)
        model.eval();scores={}
        with torch.inference_mode():
            for name in ['SOURCE_INNER','MAP_INNER']:
                counts=np.zeros(4,np.int64)
                d=pools[name][0]
                for start in range(0,len(d[0]),16):
                    x,n,z,t=[c[start:start+16] for c in d]
                    v=((t[:,:,None]>0)&(t[:,None,:]>0)).triu(1);s=(t[:,:,None]==t[:,None,:])&v;neg=v&~s
                    logits=model(x,n,z);counts+=[int((logits[s]>0).sum()),int(s.sum()),int((logits[neg]<0).sum()),int(neg.sum())]
                scores[name]=dict(same_accuracy=counts[0]/counts[1],different_accuracy=counts[2]/counts[3],n_same=int(counts[1]),n_different=int(counts[3]))
        save(dest/'TRAINED.json',dict(steps=STEPS,weight_sha=digest(path),pair_validation=scores,parameters=sum(p.numel() for p in model.parameters())))
if __name__=='__main__':main()

