import os,json,time,hashlib,random,math
os.environ.setdefault('GDAL_PAM_ENABLED','NO')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
ROOT=Path('/root/xian2019');IN=ROOT/'input';OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
SEED=20261028;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=False
TRAIN_STOP=datetime.fromisoformat('2026-10-29T04:00:00+08:00').timestamp()
def save(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));os.replace(t,p)
def status(stage,**kw):save('STATUS.json',dict(stage=stage,time=datetime.now().astimezone().isoformat(),**kw));print(stage,kw,flush=True)
def checkpoint(name,model,**kw):
 p=OUT/name;t=p.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),**kw),t);os.replace(t,p)
class Conv(nn.Sequential):
 def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU(inplace=True),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.SiLU(inplace=True))
class Net(nn.Module):
 def __init__(self,channels=4,classes=1):
  super().__init__();self.e1=Conv(channels,24);self.e2=Conv(24,48);self.e3=Conv(48,96);self.e4=Conv(96,192);self.e5=Conv(192,256);self.d4=Conv(448,192);self.d3=Conv(288,96);self.d2=Conv(144,48);self.d1=Conv(72,24);self.seg=nn.Conv2d(24,classes,1);self.shift=nn.Sequential(nn.Linear(261,128),nn.SiLU(),nn.Linear(128,2))
 def forward(self,x,meta=None):
  a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.e3(F.avg_pool2d(b,2));d=self.e4(F.avg_pool2d(c,2));e=self.e5(F.avg_pool2d(d,2))
  z=self.d4(torch.cat((F.interpolate(e,size=d.shape[-2:],mode='bilinear',align_corners=False),d),1));z=self.d3(torch.cat((F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c),1));z=self.d2(torch.cat((F.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1));z=self.d1(torch.cat((F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
  if meta is None:meta=e.new_zeros((len(x),5))
  return self.seg(z),self.shift(torch.cat((e.mean((-2,-1)),meta),1)).tanh()*3
def curve(name,rows):
 pd.DataFrame(rows).to_csv(OUT/(name+'_curves.csv'),index=False)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 f,ax=plt.subplots(1,2,figsize=(11,4));d=pd.DataFrame(rows)
 for col in ('loss','validation_loss'):
  if col in d:ax[0].plot(d.epoch,d[col],label=col)
 for col in ('iou','shift_error_m','precision','recall'):
  if col in d:ax[1].plot(d.epoch,d[col],label=col)
 for a in ax:a.legend();a.grid(alpha=.3)
 f.tight_layout();f.savefig(OUT/(name+'_curves.png'));plt.close(f)
class AlignData(Dataset):
 def __init__(self,indices):
  self.indices=indices;self.x=np.load(IN/'align_x.npy',mmap_mode='r');self.y=np.load(IN/'align_y.npy',mmap_mode='r');self.shift=np.load(IN/'align_shift.npy');self.meta=np.load(IN/'align_meta.npy')
 def __len__(self):return len(self.indices)
 def __getitem__(self,j):
  i=self.indices[j];return torch.from_numpy(self.x[i].copy()).float()/255,torch.from_numpy(self.y[i].copy()).float()[None],torch.from_numpy(self.shift[i]),torch.from_numpy(self.meta[i])
class FacadeData(Dataset):
 def __init__(self,split):self.rows=[x for x in json.loads((IN/'facade_manifest.json').read_text()) if x['split']==split];self.cache={}
 def __len__(self):return len(self.rows)*16
 def __getitem__(self,i):
  from PIL import Image
  r=self.rows[i//16];key=i//16
  if key not in self.cache:
   if len(self.cache)>12:self.cache.clear()
   self.cache[key]=(np.array(Image.open(IN/r['image']).convert('RGB')),np.array(Image.open(IN/r['label'])))
  rgb,lab=self.cache[key];j=i%16;y=j//4*512;x=j%4*512;im=rgb[y:y+512,x:x+512];target=lab[y:y+512,x:x+512].copy()
  if r['split']=='TRAIN':
   k=random.randrange(4);im=np.rot90(im,k);target=np.rot90(target,k)
   if random.random()<.5:im=im[:,::-1];target=target[:,::-1]
  return torch.from_numpy(im.transpose(2,0,1).copy()).float()/255,torch.from_numpy(target.copy()).long()
def loader(ds,batch,shuffle=False):return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=3)
@torch.no_grad()
def evaluate(model,dl,kind,limit=None):
 model.eval();losses=[];inter=union=tp=fp=fn=0;dist=[];baseline=[];raw_inter=raw_union=0
 for j,b in enumerate(dl):
  if limit and j>=limit:break
  if kind=='alignment':
   x,y,s,meta=[v.cuda(non_blocking=True) for v in b];z,p=model(x,meta);loss=F.smooth_l1_loss(p,s)+.5*F.binary_cross_entropy_with_logits(z,y);pred=z>0;truth=y>.5;inter+=int((pred&truth).sum());union+=int((pred|truth).sum());raw_inter+=int(((x[:,3:4]>.5)&truth).sum());raw_union+=int(((x[:,3:4]>.5)|truth).sum());dist.extend(torch.linalg.vector_norm((p-s)*10,dim=1).cpu().tolist());baseline.extend(torch.linalg.vector_norm(s*10,dim=1).cpu().tolist())
  else:
   x,y=[v.cuda(non_blocking=True) for v in b];z,_=model(x);loss=F.cross_entropy(z,y,ignore_index=255);pred=z.argmax(1);valid=y!=255;tp+=int(((pred==2)&(y==2)&valid).sum());fp+=int(((pred==2)&(y!=2)&valid).sum());fn+=int(((pred!=2)&(y==2)&valid).sum());inter=tp;union=tp+fp+fn
  losses.append(float(loss))
 out=dict(validation_loss=float(np.mean(losses)),iou=inter/max(union,1))
 if kind=='alignment':out.update(shift_error_m=float(np.mean(dist)),baseline_shift_error_m=float(np.mean(baseline)),baseline_contour_iou=raw_inter/max(raw_union,1))
 else:out.update(precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
 return out
def fit(kind):
 if kind=='alignment':
  split=np.load(IN/'align_split.npy')
  if os.environ.get('ALIGNMENT_CLEAN')=='1':split[~np.load(IN/'alignment_quality.npy')]=-1
  train=AlignData(np.where(split==0)[0]);val=AlignData(np.where(split==1)[0]);batch=32;model=Net(4,1).cuda();hours=1.0 if os.environ.get('ALIGNMENT_CLEAN')=='1' else 1.6
  if os.environ.get('ALIGNMENT_CLEAN')=='1':model.load_state_dict(torch.load(OUT/'preliminary_alignment/alignment_best.pt',map_location='cpu')['model'])
 else:train=FacadeData('TRAIN');val=FacadeData('TUNE');batch=8;model=Net(3,3).cuda();hours=1.25
 # Probe memory with actual tensor shape; reduce only scheduling batch size if needed.
 while True:
  try:
   n=min(batch,len(train));sample=next(iter(DataLoader(train,batch_size=n,num_workers=0)));x=sample[0].cuda();z,s=model(x);(z.mean()+s.mean()).backward();model.zero_grad(set_to_none=True);del sample,x,z,s;torch.cuda.empty_cache();break
  except torch.cuda.OutOfMemoryError:
   model.zero_grad(set_to_none=True);torch.cuda.empty_cache();batch//=2
   if batch<1:raise
 dl=loader(train,batch,True);vd=loader(val,batch);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler();stop=min(TRAIN_STOP,time.time()+hours*3600)
 params=dict(architecture='Five-level GroupNorm UNet, channels 24/48/96/192/256; displacement MLP',initialization='random SEED20261028',train_samples=len(train),validation_samples=len(val),batch_size=batch,optimizer='AdamW',learning_rate=2e-4,weight_decay=1e-4,amp='float16',workers=8,deadline=datetime.fromtimestamp(stop).astimezone().isoformat(),selection='spatial validation shift error + contour IoU' if kind=='alignment' else 'BANDON TUNE projected-facade IoU; target-domain accuracy not established')
 params.update(seed=SEED,parameter_count=sum(p.numel() for p in model.parameters()),input_channels=4 if kind=='alignment' else 3,input_side=256 if kind=='alignment' else 512,maximum_epochs=30,gradient_clip_norm=5,learning_rate_decay_per_epoch=.95,evaluation_precision='FP32',loss='SmoothL1(shift/10m)+0.5 BCE(contour)+0.5 Dice' if kind=='alignment' else 'Semantic CE weights[0.3 background,1 roof,2 projected facade], ignore255')
 if os.environ.get('ALIGNMENT_CLEAN')=='1':params['initialization']='strict warm-start from preliminary_alignment/alignment_best.pt; new filtered spatial supervision'
 save(kind+'_parameters.json',params);rows=[];best=-float('inf');updates=0
 initial=evaluate(model,vd,kind,limit=None if os.environ.get('ALIGNMENT_CLEAN')=='1' else 12);rows.append(dict(epoch=0,updates=0,loss=None,**initial));curve(kind,rows)
 if os.environ.get('ALIGNMENT_CLEAN')=='1':
  best=initial['iou']-initial.get('shift_error_m',0)/30;checkpoint(kind+'_best.pt',model,epoch=0,updates=0,parameters=params,metrics=initial,warm_start_preserved=True)
 for epoch in range(1,31):
  if time.time()>stop:break
  model.train();losses=[];t=time.time()
  for b in dl:
   if time.time()>stop:break
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):
    if kind=='alignment':
     x,y,s,meta=[v.cuda(non_blocking=True) for v in b];z,p=model(x,meta);prob=z.sigmoid();dice=1-(2*(prob*y).sum((1,2,3))+1)/(prob.sum((1,2,3))+y.sum((1,2,3))+1);loss=F.smooth_l1_loss(p,s)+.5*F.binary_cross_entropy_with_logits(z,y)+.5*dice.mean()
    else:
     x,y=[v.cuda(non_blocking=True) for v in b];z,_=model(x);loss=F.cross_entropy(z,y,ignore_index=255,weight=torch.tensor([.3,1.,2.],device='cuda'))
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();losses.append(float(loss.detach()));updates+=1
   if updates%50==0:status('TRAIN_'+kind,epoch=epoch,updates=updates,loss=float(np.mean(losses[-50:])),batch=batch,gpu_allocated_gib=torch.cuda.memory_allocated()/2**30)
  if not losses:break
  metrics=evaluate(model,vd,kind);row=dict(epoch=epoch,updates=updates,loss=float(np.mean(losses)),seconds=time.time()-t,**metrics);rows.append(row);curve(kind,rows)
  score=metrics['iou']-metrics.get('shift_error_m',0)/30
  checkpoint(kind+'_last.pt',model,epoch=epoch,updates=updates,parameters=params,metrics=metrics)
  if score>best:best=score;checkpoint(kind+'_best.pt',model,epoch=epoch,updates=updates,parameters=params,metrics=metrics)
  status('VALIDATE_'+kind,**row)
  for pg in opt.param_groups:pg['lr']=2e-4*.95**epoch
 bestcp=torch.load(OUT/(kind+'_best.pt'),map_location='cpu');model.load_state_dict(bestcp['model']);model.cuda().eval();metrics=evaluate(model,vd,kind);save(kind+'_validation.json',dict(metrics=metrics,updates=updates,validation_role='TUNE_NOT_BLIND',parameters=params))
 if kind=='alignment':
  audit=loader(AlignData(np.where(split==2)[0]),batch);save('alignment_spatial_audit.json',dict(metrics=evaluate(model,audit,kind),role='SPATIAL_DIAGNOSTIC_AFTER_REPEATED_REFERENCE_ANALYSIS_NOT_BLIND_FINAL'))
 del model,opt;torch.cuda.empty_cache()
