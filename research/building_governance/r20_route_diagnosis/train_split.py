"""Learn a visible-roof interior score, then split only inside frozen roof parents."""
import csv,hashlib,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from pycocotools import mask as cm
from scipy import ndimage
from skimage.segmentation import watershed
from diagnose import check, save
HERE=Path(__file__).resolve().parent
OUT=HERE/'split_training'
OUT.mkdir(exist_ok=True)
R8=Path('/root/xian20261111_r8')
P15=Path('/root/xian20261111_r15')
P17=Path('/root/xian20261111_r17_proposals')
SEED=20261111
STEPS=400
CUTS=(.45,.65,.80)
def rgb(r):
    check(r['image_path'],r['image_sha256'])
    if r['role']=='TUNE':
        import rasterio
        with rasterio.open(r['image_path']) as ds:return ds.read([1,2,3]).transpose(1,2,0).copy()
    return np.asarray(Image.open(r['image_path']).convert('RGB'))
def source_sample(item):
    cv2.setNumThreads(1)
    r,t=item
    im=rgb(r);check(r['query_cache_path'],r['query_cache_sha256']);check(t['targets'],t['targets_sha256'])
    with np.load(r['query_cache_path']) as z:raw=z['instances'].astype('int32')
    with np.load(t['targets']) as z:
        counts=z['gt_counts'];known=np.unpackbits(z['known_native']).reshape(raw.shape).astype(bool)
    boundary=np.zeros(raw.shape,np.uint8);union=np.zeros(raw.shape,bool);count=np.zeros(raw.max()+1,np.int32)
    for enc in counts:
        m=cm.decode(dict(size=list(raw.shape),counts=bytes(enc))).astype(bool)
        if not m.any():continue
        sl=ndimage.find_objects(m.astype(np.uint8))[0]
        patch=m[sl]
        dist=cv2.distanceTransform(np.pad(patch.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
        energy=np.rint(255*np.clip((dist-1)/9,0,1)).astype(np.uint8)
        boundary[sl]=np.maximum(boundary[sl],energy)
        union[sl]|=patch
        ids,freq=np.unique(raw[sl][patch],return_counts=True)
        count[ids[(ids>0)&(freq>=max(16,.12*patch.sum()))]]+=1
    # Unknown raw pixels are never supervised; only known roof and known off-parent background.
    valid=(union | (known & (raw==0)))
    assert not np.any(valid & (raw>0) & ~union)
    rng=np.random.default_rng(int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16))
    picks=[]
    for ids,n in ((np.flatnonzero(count>=2),5),(np.flatnonzero(count==1),3)):
        if len(ids):
            for q in rng.choice(ids,size=n,replace=len(ids)<n):
                y,x=np.where((raw==q)&union)
                k=rng.integers(len(y));y0=int(np.clip(y[k]-96,0,raw.shape[0]-192));x0=int(np.clip(x[k]-96,0,raw.shape[1]-192))
                sl=np.s_[y0:y0+192,x0:x0+192]
                picks.append(np.dstack((im[sl],(raw[sl]>0).astype('uint8')*255,boundary[sl],valid[sl].astype('uint8')*255)))
    return picks
def block(c,d):
    return nn.Sequential(nn.Conv2d(c,d,3,padding=1),nn.GroupNorm(4,d),nn.SiLU(),nn.Conv2d(d,d,3,padding=1),nn.GroupNorm(4,d),nn.SiLU())
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=block(4,16);self.b=block(16,32);self.c=block(32,48);self.d=block(80,32);self.e=block(48,16);self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.max_pool2d(a,2));c=self.c(F.max_pool2d(b,2))
        d=self.d(torch.cat((b,F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False)),1))
        return self.out(self.e(torch.cat((a,F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False)),1)))
def split(raw,prob,threshold):
    out=raw.copy();last=int(raw.max())
    for q,sl in enumerate(ndimage.find_objects(raw),1):
        if sl is None:continue
        m=raw[sl]==q
        if m.sum()<140:continue
        score=prob[sl]
        seeds,n=ndimage.label((score>=threshold)&m)
        sizes=np.bincount(seeds.ravel());keep=np.flatnonzero(sizes[1:]>=max(32,.025*m.sum()))+1
        if len(keep)<2:continue
        seeds=np.where(np.isin(seeds,keep),seeds,0)
        parts=watershed(-score,seeds,mask=m,connectivity=np.ones((3,3)))
        sub,areas=np.unique(parts[parts>0],return_counts=True)
        if len(sub)<2 or areas.min()<48:continue
        largest=sub[np.argmax(areas)]
        for i in sub:
            if i!=largest:
                last+=1;out[sl][parts==i]=last
    assert np.array_equal(out>0,raw>0)
    return out
@torch.no_grad()
def predict(model,row):
    im=rgb(row);check(row['query_cache_path'],row['query_cache_sha256'])
    with np.load(row['query_cache_path']) as z:raw=z['instances'].astype(np.int32)
    x=np.dstack((im,(raw>0).astype('uint8')*255))
    x=torch.tensor(x,device='cuda').permute(2,0,1).float()[None]/255.
    return raw,model(x).sigmoid()[0,0].cpu().numpy()
def main():
    cv2.setNumThreads(1);torch.manual_seed(SEED);np.random.seed(SEED)
    rows={r['id']:r for line in (R8/'QUERY_CACHE_RECORDS.jsonl').read_text().splitlines() if (r:=json.loads(line))}
    source=json.loads((P17/'data/READY.json').read_text())['rows']
    fit=sorted((r for r in source if r['role']=='FIT'),key=lambda r:hashlib.sha256(r['id'].encode()).digest())[:192]
    assert len(fit)==192 and all(rows[t['id']]['role']=='SOURCE_TRAIN' for t in fit)
    save(OUT/'PROTOCOL.json',dict(seed=SEED,role='192 hashed source FIT frames, native supervised roof and known off-parent background; no XiAn GT in training',fit=[r['id'] for r in fit],steps=STEPS,
        batch=24,lr=.0008,loss='valid pixel BCE with weight 1+2*raw; native own-roof inward distance ramp 1..10 px',network='3-level convolutional encoder-decoder 16/32/48',cuts=CUTS,
        selection='INNER chosen by F1 then fewer fragments then complete90; frozen before Jinan and Xian; exact retention of union, no fullcity promotion'))
    ck=OUT/'final.pt'
    if not ck.exists():
        with ProcessPoolExecutor(8) as pool:patches=[p for bb in pool.map(source_sample,[(rows[t['id']],t) for t in fit],chunksize=2) for p in bb]
        print('Training patches',len(patches),flush=True)
        data=torch.as_tensor(np.stack(patches),device='cuda').permute(0,3,1,2)
        model=Model().cuda().to(memory_format=torch.channels_last);opt=torch.optim.AdamW(model.parameters(),lr=.0008)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00008)
        curves=[]
        with (OUT/'CURVES.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['step','loss','lr']);f.flush()
            for i in range(1,STEPS+1):
                patch=data[torch.randint(len(data),(24,),device='cuda')].float()/255.
                if i%2:patch=patch.flip(-1)
                if i%3:patch=patch.flip(-2)
                x=patch[:,:4].contiguous(memory_format=torch.channels_last); y=patch[:,4:5];valid=patch[:,5:6]*(1+2*x[:,3:4])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model(x);loss=(F.binary_cross_entropy_with_logits(out.float(),y,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                writer.writerow([i,float(loss),opt.param_groups[0]['lr']]);schedule.step()
                if i%50==0:f.flush();print('step',i,float(loss),flush=True)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),steps=STEPS),OUT/'final.tmp')
        (OUT/'final.tmp').replace(ck)
        save(OUT/'TRAINING.json',dict(updates=STEPS,patches=len(patches),weight_sha=hashlib.sha256(ck.read_bytes()).hexdigest()))
    model=Model().cuda().eval();model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model'])
    sys.path.insert(0,str(P15/'code'))
    from roof_decoder_evaluation import RoofEvaluator,metrics,aggregate
    selected=None
    for role in ('INNER','TUNE'):
        eva=RoofEvaluator(role)
        results={}
        cuts=CUTS if selected is None else [selected]
        for frame in eva.frames:
            raw,prob=predict(model,rows[frame['id']])
            for cut in cuts:results.setdefault(cut,[]).append(metrics(frame,split(raw,prob,cut)))
        summary={c:aggregate(group) for c,group in results.items()}
        if selected is None:
            selected=max(cuts,key=lambda c:(summary[c]['F1_50'],-summary[c]['fragmented_GT'],summary[c]['complete_roof_recall90']))
        save(OUT/(role+'.json'),dict(selected=selected,baseline=eva.baseline,results=summary))
        print(role,selected,summary[selected]['F1_50'],flush=True)
        del eva
    sys.path.insert(0,'/root/xian20261111_r16_roi_roof/code')
    from evaluate_xian_map_windows import evaluate,summarize
    dest=OUT/'xian';dest.mkdir(exist_ok=True)
    target=[r for r in rows.values() if r['role']=='TUNE'];assert len(target)==72
    manifest={}
    for r in target:
        raw,prob=predict(model,r);p=dest/(r['id']+'.npz')
        np.savez_compressed(p,SPLIT=split(raw,prob,selected))
        manifest[r['id']]=dict(path=str(p),variant='SPLIT',sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    save(OUT/'XIAN_IMAGE_ONLY_FROZEN.json',manifest)
    with ProcessPoolExecutor(6) as pool:ww=list(pool.map(evaluate,[(r,manifest) for r in target]))
    result={ep:{arm:summarize([w for w in ww if w['epoch']==ep],arm) for arm in ['RAW','ROI_SELECTED']} for ep in sorted({w['epoch'] for w in ww})}
    save(OUT/'XIAN.json',result)
    save(OUT/'XIAN_DETAILS.json',ww)
if __name__=='__main__':main()

