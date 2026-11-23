from cuts import *
import csv,torch
from torch import nn
from prepare import rows,R17
from guard import model_for_split
from evaluate_pairs import scores
from pycocotools import mask as cm
from concurrent.futures import ProcessPoolExecutor
B=P/'contour';B.mkdir(exist_ok=True)
E=B/'features';E.mkdir(exist_ok=True)
def image():
    model=model_for_split()
    for rec in json.loads((P/'CANDIDATES_FROZEN.json').read_text()):
        if rec['role'] not in ('SOURCE_FIT','SOURCE_INNER','XIAN'):continue
        k=rec['id'];boxes,nodes,d=scores(model,rec)
        with np.load(D/(k+'.npz')) as f:groups=f['groups'];feat=f['feat'];masks=f['masks']
        extra=np.zeros((len(feat),8),np.float32)
        for i in np.unique(groups):
            inds=np.where(groups==i)[0];node=nodes[i].ravel().astype('int32')
            mask=np.unpackbits(masks[inds],axis=1);nv=np.bincount(node,minlength=65);nv[0]=0
            a=np.stack([np.bincount(node[line>0],minlength=65) for line in mask]).astype('float32');a[:,0]=0
            b=nv-a;a/=a.sum(1,keepdims=True).clip(1);b=b/b.sum(1,keepdims=True).clip(1)
            v=d[i]
            def weigh(x,y,v):return ((x@v)*y).sum(1)
            ab=weigh(a,b,v);aa=weigh(a,a,v);bb=weigh(b,b,v)
            extra[inds]=np.stack([ab,aa,bb,ab-(aa+bb)*.5,weigh(a,b,v>.75),weigh(a,b,v<.25),weigh(a,a,v>.75),weigh(b,b,v>.75)],1)
        np.save(E/(k+'.npy'),np.column_stack([feat,extra]).astype('float32'))
def supervision(arg):
    r,t=arg;key=r['id'];verify(t['targets'],t['targets_sha256'])
    with np.load(t['targets']) as f:enc=f['gt_counts']
    with np.load(r['query_cache_path']) as f:raw=f['instances'].astype('int32')
    with np.load(OLD/'data'/(key+'.npz')) as f:boxes=f['boxes']
    with np.load(D/(key+'.npz')) as f:groups=f['groups'];masks=f['masks']
    lookup={q:(i,y0,y1,x0,x1) for i,(q,y0,y1,x0,x1) in enumerate(boxes)}
    info={}
    for entry in enc:
        g=cm.decode(dict(size=[1024,1024],counts=bytes(entry)))
        yy,xx=np.nonzero(g);area=len(xx)
        if area<32:continue
        par=raw[yy,xx];ids,counts=np.unique(par,return_counts=True)
        for q,n in zip(ids,counts):
            if q not in lookup or n<max(8,.05*area):continue
            i,y0,y1,x0,x1=lookup[q];take=par==q;index=((yy[take]-y0)*64//(y1-y0))*64+((xx[take]-x0)*64//(x1-x0))
            info.setdefault(i,[]).append((area,np.bincount(index,minlength=4096)))
    y=np.full((len(groups),2),-1,np.int8)
    for i,gt in info.items():
        ii=np.where(groups==i)[0]
        if not len(ii):continue
        q,y0,y1,x0,x1=boxes[i];yy,xx=np.where(raw[y0:y1,x0:x1]==q);pa=len(xx)
        grid=yy*64//(y1-y0)*64+xx*64//(x1-x0)
        side=np.unpackbits(masks[ii],axis=1).astype(np.float32)
        ca=side@np.bincount(grid,minlength=4096).astype(np.float32)
        ga=np.array([v[0] for v in gt]);ov=np.stack([v[1] for v in gt]).astype(np.float32);s=ov.sum(1)
        a=side@ov.T;b=s-a
        old=int((s/(ga+pa-s)>=.5).any());ma=(a/(ga+ca[:,None]-a)>=.5);mb=(b/(ga+(pa-ca)[:,None]-b)>=.5)
        h=(ma|mb).any(1).astype(int)+(ma.any(1)&mb.any(1)&((ma|mb).sum(1)>1)).astype(int)
        harm=((a>=.1*ga)&(b>=.1*ga)).any(1)|(h<old)
        y[ii,0]=h-old;y[ii,1]=harm
        # Both substantial children of a known roof or one-to-one gains are supervised;
        # neutral unlabeled space contributes no negative loss.
    np.save(E/(key+'.y.npy'),y)
def labels():
    reg=rows();tar={r['id']:r for r in json.loads((R17/'data/READY.json').read_text())['rows']}
    records=json.loads((P/'CANDIDATES_FROZEN.json').read_text())
    with ProcessPoolExecutor(12) as pool:list(pool.map(supervision,[(reg[r['id']],tar[r['id']]) for r in records if r['role'] in ('SOURCE_FIT','SOURCE_INNER')]))
class Contour(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(27,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.10),nn.Linear(96,48),nn.SiLU(),nn.Linear(48,1))
    def forward(self,x):return self.net(x).squeeze(-1)
def dataset(role):
    xx=[];yy=[];group=[];offset=0
    for r in json.loads((P/'CANDIDATES_FROZEN.json').read_text()):
        if r['role']!=role:continue
        k=r['id'];x=np.load(E/(k+'.npy'));y=np.load(E/(k+'.y.npy'))
        with np.load(D/(k+'.npz')) as f:g=f['groups']
        xx.append(x);yy.append(y);group.append(g+offset);offset+=int(g.max()+1) if len(g) else 0
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(group)
def train():
    torch.manual_seed(221112)
    x,y,g=dataset('SOURCE_FIT');a,b,h=dataset('SOURCE_INNER')
    safe=(y[:,0]>0)&(y[:,1]==0);harm=y[:,1]==1;known=safe|harm
    mu=torch.tensor(x[known].mean(0));sd=torch.tensor(x[known].std(0)).clamp_min(.015)
    norm=lambda v:((torch.tensor(v)-mu)/sd).cuda()
    x=norm(x);a=norm(a);pos=torch.tensor(np.flatnonzero(safe),device='cuda');neg=torch.tensor(np.flatnonzero(harm),device='cuda')
    model=Contour().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.002)
    logs=[];best=None;first=np.r_[0,1+np.flatnonzero(np.diff(h))];last=np.r_[first[1:],len(h)]
    for ep in range(1,31):
        model.train();losses=[]
        for _ in range(50):
            ids=torch.cat([pos[torch.randint(len(pos),(96,),device='cuda')],neg[torch.randint(len(neg),(416,),device='cuda')]])
            score=model(x[ids]);loss=nn.functional.binary_cross_entropy_with_logits(score,torch.cat([torch.ones(96,device='cuda'),torch.zeros(416,device='cuda')]))
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        model.eval()
        with torch.no_grad():z=model(a).cpu().numpy()
        chosen=np.array([s+np.argmax(z[s:t]) for s,t in zip(first,last)]);observed=b[chosen,1]>=0
        pts=[]
        for cut in list(np.arange(-1,5.01,.25))+[100]:
            take=(z[chosen]>=cut)&observed;gt=b[chosen[take]];good=int(np.sum((gt[:,0]>0)&(gt[:,1]==0)));bad=int(np.sum(gt[:,1]>0))
            pts.append((good-4*bad,-bad,good,float(cut)))
        u,minus,good,cut=max(pts)
        if best is None or (u,minus)>best:
            best=(u,minus);torch.save(dict(model=model.state_dict(),mu=mu,sd=sd,cut=cut,epoch=ep),B/'SELECTOR.pt')
        logs.append(dict(epoch=ep,loss=np.mean(losses),inner_utility=u,cut=cut,safe_gains=good,harmful=-minus))
    with open(B/'CURVES.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=logs[0]);w.writeheader();w.writerows(logs)
    ck=torch.load(B/'SELECTOR.pt',weights_only=True)
    save(B/'SELECTOR_FROZEN.json',dict(epoch=ck['epoch'],cut=ck['cut'],sha256=digest(B/'SELECTOR.pt'),inner=logs[ck['epoch']-1]))
    save(B/'PARAMETERS.json',dict(updates=1500,epochs=30,batch=512,optimizer='AdamW',lr=.0009,decay=.002,seed=221112,safe_fit_cuts=len(pos),harm_fit_cuts=len(neg),features='19 visual geometry and 8 frozen R21 CNN regional identity affinity features',rule='rank all cuts per parent including unknown; on source inner maximize safe matching gains minus 4 times harmful parent splits; neutral and no GT ignored in training',source_inputs=384,source_inner_inputs=32))
if __name__=='__main__':globals()[sys.argv[1]]()
