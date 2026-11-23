import json,csv,random
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
from torch import nn
from cuts import P,OLD,D,verify,save,digest
SEED=221111
def label(rec):
    key=rec['id'];fp=OLD/'data'/(key+'.labels.npy');verify(fp,rec['sha'])
    gt=np.load(fp)
    with np.load(OLD/'data'/(key+'.npz')) as f:nodes=f['nodes']
    with np.load(D/(key+'.npz')) as f:groups=f['groups']; feat=f['feat'];masks=f['masks']
    ret=[]
    for k in np.unique(groups):
        ii=np.where(groups==k)[0]; target=gt[k][nodes[k]]
        ids,n=np.unique(target[target>0],return_counts=True)
        ids=ids[n>=48]
        if not len(ids):continue
        side=np.unpackbits(masks[ii],axis=1).reshape(-1,64,64).astype(bool)
        inside=np.stack([side[:,target==v].mean(1) for v in ids],1)
        # Each known identity gets a vote. Mixed or unlabelled regions supply no background labels.
        broken=((inside>=.18)&(inside<=.82)).any(1)
        good=(inside.max(1)>=.82)&(inside.min(1)<=.18)&~broken
        verdict=np.where(broken,0,np.where(good,1,-1))
        valid=verdict>=0
        if valid.any():ret.append((feat[ii][valid],verdict[valid].astype('float32'),ii[valid],key,int(k)))
    return ret
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(19,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.12),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,1))
    def forward(self,x):return self.net(x).squeeze(-1)
def arrays(groups):
    n=len(groups);x=np.zeros((n,12,19),'float32');y=np.full((n,12),-1,'float32')
    for i,(a,b,*_) in enumerate(groups):
        x[i,:len(a)]=a;y[i,:len(b)]=b
    return torch.tensor(x,device='cuda'),torch.tensor(y,device='cuda')
def point(z,y,cut):
    # INNER ranks all supervised alternatives within a parent. Ambiguous cuts carry no labels.
    z=z.masked_fill(y<0,-1e4);v,k=z.max(1);selected=v>=cut;t=y.gather(1,k[:,None])[:,0]
    return (int((selected&(t==1)).sum()),int((selected&(t==0)).sum()))
def main():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    frozen=json.loads((P/'CANDIDATES_FROZEN.json').read_text());labels=json.loads((OLD/'SUPERVISION.json').read_text())
    hashes={r['id']:r['sha'] for r in frozen}
    for d in labels:verify(D/(d['id']+'.npz'),hashes[d['id']])
    data={}
    with ProcessPoolExecutor(10) as pool:
        for d,gs in zip(labels,pool.map(label,labels)):data.setdefault(d['role'],[]).extend(gs)
    src=data['SOURCE_FIT']+data['MAP_FIT'];x,y=arrays(src)
    inn=[arrays(data[k]) for k in ('SOURCE_INNER','MAP_INNER')]
    known=x[y>=0];mu=known.mean(0);sd=known.std(0).clamp(min=.015)
    x=(x-mu)/sd;inn=[((a-mu)/sd,b) for a,b in inn]
    net=Net().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.002)
    pos=torch.where((y==1).any(1))[0]; print('groups',len(src),'positive',len(pos),flush=True)
    curves=[];best=None
    params=dict(seed=SEED,epochs=30,updates_per_epoch=45,batch_parents=192,lr=.001,weight_decay=.002,optimizer='AdamW',loss='masked binary cross entropy plus 0.35 listwise positive mass vs alternatives and explicit no-cut logit zero',inner_selection='maximize good minus six times fragmented in combined source and map INNER; grid logits 0..6 step 0.25 plus no-cut; ties fewer fragments then higher threshold',roles={k:len(v) for k,v in data.items()},feature_count=19)
    save(P/'PARAMETERS.json',params)
    for ep in range(1,31):
        net.train();ll=[]
        for it in range(45):
            ix=torch.cat([torch.randint(len(x),(128,),device='cuda'),pos[torch.randint(len(pos),(64,),device='cuda')]])
            yy=y[ix];z=net(x[ix]);valid=yy>=0;positive=yy==1
            ce=nn.functional.binary_cross_entropy_with_logits(z[valid],yy[valid])
            zero=z.new_zeros((len(z),1));allz=torch.cat([zero,z.masked_fill(~valid,-1e4)],1)
            target=torch.cat([~positive.any(1,keepdim=True),positive],1)
            rank=(allz.logsumexp(1)-allz.masked_fill(~target,-1e4).logsumexp(1)).mean()
            loss=ce+.35*rank;opt.zero_grad();loss.backward();opt.step();ll.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            zz=[(net(a),b) for a,b in inn];stats=[]
            for c in list(np.arange(0,6.01,.25))+[100.]:
                hits=[point(z,b,c) for z,b in zz];ok,bad=np.sum(hits,axis=0)
                stats.append((int(ok-6*bad),-int(bad),float(c),int(ok),hits))
            best_epoch=max(stats);u,neg,cut,ok,hits=best_epoch
            if best is None or (u,neg)>best:
                best=(u,neg);torch.save(dict(model=net.state_dict(),mu=mu.cpu(),sd=sd.cpu(),cut=cut,epoch=ep),P/'SELECTOR.pt')
        curves.append(dict(epoch=ep,loss=np.mean(ll),utility=u,cut=cut,good=ok,fragmented=-neg,source_good=hits[0][0],source_fragmented=hits[0][1],map_good=hits[1][0],map_fragmented=hits[1][1]))
    with open(P/'CURVES.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=curves[0]);w.writeheader();w.writerows(curves)
    model=torch.load(P/'SELECTOR.pt',weights_only=True);save(P/'SELECTOR_FROZEN.json',dict(epoch=model['epoch'],cut=model['cut'],sha256=digest(P/'SELECTOR.pt'),candidate_manifest=digest(P/'CANDIDATES_FROZEN.json'),inner=curves[model['epoch']-1]))
if __name__=='__main__':main()
