"""Run the authors' WHU tiny binary change weights on actual neighboring observations."""
import hashlib,json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from diagnose import check,input_rows,save
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'mamba_lib'))
from MambaCD.classification.models import vmamba
from MambaCD.changedetection.models.MambaBCD import STMambaBCD

def prefix(a,b):
    # Prefix composition of (a,b): h_t = a_t*h_prev + b_t; no division by small a.
    k=1
    while k<a.shape[-1]:
        b=torch.cat((b[...,:k],b[...,k:]+a[...,k:]*b[...,:-k]),-1)
        a=torch.cat((a[...,:k],a[...,k:]*a[...,:-k]),-1)
        k*=2
    return a,b
class Scan:
    @staticmethod
    def fwd(u,delta,A,B,C,D,bias,soft,rows,oflex):
        # The selected Tiny architecture uses a scalar state per channel.
        assert A.shape[-1]==1 and B.shape[2]==C.shape[2]==1
        bs,cd,length=u.shape;groups=B.shape[1];ch=cd//groups
        dt=delta.float()
        if bias is not None:dt=dt+bias.float()[None,:,None]
        if soft:dt=F.softplus(dt)
        a=torch.exp(dt*A[:,0].float()[None,:,None])
        b=(dt*u.float()).reshape(bs,groups,ch,length)*B[:,:,0,None,:].float()
        b=b.reshape_as(a)
        stride=32
        pad=(-length)%stride
        ab=F.pad(a,(0,pad),value=1.).reshape(bs,cd,-1,stride)
        bb=F.pad(b,(0,pad)).reshape_as(ab)
        local_a,local_b=prefix(ab,bb)
        _,ends=prefix(local_a[...,-1],local_b[...,-1])
        prior=F.pad(ends[...,:-1],(1,0))
        out=(local_b+local_a*prior[...,None]).reshape(bs,groups,ch,-1)[...,:length]*C[:,:,0,None,:].float()
        out=out.reshape(bs,cd,length)
        if D is not None:out=out+D.float()[None,:,None]*u.float()
        return out,torch.empty(0,device=u.device)
def validate_scan():
    torch.manual_seed(4)
    u,dt=torch.randn(2,12,79,device='cuda'),torch.randn(2,12,79,device='cuda')
    A=-torch.rand(12,1,device='cuda');B=torch.randn(2,4,1,79,device='cuda');C=torch.randn_like(B)
    D=torch.randn(12,device='cuda');bias=torch.randn_like(D)
    got=Scan.fwd(u,dt,A,B,C,D,bias,True,1,True)[0]
    ds=F.softplus(dt+bias[None,:,None]);ref=torch.zeros(2,12,device='cuda');outputs=[]
    for i in range(79):
        ref=(ds[:,:,i]*A[:,0]).exp()*ref+ds[:,:,i]*u[:,:,i]*B[:,:,0,i].repeat_interleave(3,1)
        outputs.append(ref*C[:,:,0,i].repeat_interleave(3,1)+D*u[:,:,i])
    err=float((got-torch.stack(outputs,-1)).abs().max())
    assert err<2.e-5
    return err
def model():
    vmamba.selective_scan_cuda_oflex=Scan
    return STMambaBCD(None,patch_size=4,in_chans=3,num_classes=1000,depths=[2,2,4,2],dims=96,ssm_d_state=1,ssm_ratio=2.,ssm_rank_ratio=2.,ssm_dt_rank='auto',
        ssm_act_layer='silu',ssm_conv=3,ssm_conv_bias=False,ssm_drop_rate=0.,ssm_init='v0',forward_type='v3noz',mlp_ratio=4.,mlp_act_layer='gelu',mlp_drop_rate=0.,
        drop_path_rate=.2,patch_norm=True,norm_layer='ln',downsample_version='v3',patchembed_version='v2',gmlp=False,use_checkpoint=False).eval().cuda()
@torch.inference_mode()
def main():
    out=R/'change_evidence';out.mkdir(exist_ok=True)
    err=validate_scan()
    net=model()
    p=R/'MambaBCD_Tiny_WHU_verified.pth'
    if '--smoke' in sys.argv:
        logits=net(torch.zeros(1,3,256,256,device='cuda'),torch.zeros(1,3,256,256,device='cuda'))
        print('operator smoke only',err,logits.shape,flush=True);return
    digest=hashlib.md5(p.read_bytes()).hexdigest();assert digest=='08b24771a54869df4055b2d0a224e5bf'
    state=torch.load(p,map_location='cpu',weights_only=True)
    net.load_state_dict(state,strict=True)
    save(out/'LOAD.json',dict(commit='b4f12c7e95e2b514cc10b528342fa73bc6c41f91',state_entries=len(state),md5=digest,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        modified_operator='FP32 parallel scalar selective scan equivalent recurrence; not author CUDA bit-parity',scan_synthetic_max_abs=err,source='WHU binary change Tiny official Zenodo 14037770',
        native_tile=256,endpoint='two-class changed probability, no XiAn truth input, no local precision or direction claimed'))
    groups=defaultdict(list)
    for r in input_rows():groups[r['site']].append(r)
    mean=torch.tensor([123.675,116.28,103.53],device='cuda')[:,None,None]
    std=torch.tensor([58.395,57.12,57.375],device='cuda')[:,None,None]
    records=[]
    for site,rows in sorted(groups.items()):
        frames=[];masks=[];aff=[];rows.sort(key=lambda r:r['epoch'])
        for r in rows:
            check(r['image_path'],r['image_sha256'])
            check(r['support_path'],r['source_record']['files']['SOURCE_SUPPORT.tif']['sha256'])
            with rasterio.open(r['image_path']) as ds:
                aff.append((ds.transform,ds.crs));frames.append((torch.from_numpy(ds.read([1,2,3])).cuda().float()-mean)/std)
            with rasterio.open(r['support_path']) as ds:masks.append(ds.read(1)>0)
        assert aff[0]==aff[1]==aff[2]
        pairs=[(0,1),(1,2)]
        if site in ['DENSE_REFERENCE_00','RANDOM_02','RANDOM_03']:pairs+=[(2,2)]
        for i,j in pairs:
            before=list(rows[i]['source_record']['actual_date_pixel_votes']);after=list(rows[j]['source_record']['actual_date_pixel_votes'])
            if i!=j:assert max(before)<min(after)
            dst=out/(site+'_'+str(i+1)+'_'+str(j+1)+'.tif')
            h,w=masks[i].shape;assert h==w==1024
            vals=np.empty((h,w),np.float32);valid=masks[i]&masks[j]
            for y in range(0,h,256):
                xx=list(range(0,w,256))
                t=[torch.stack([frames[k][:,y:y+256,x:x+256] for x in xx]) for k in (i,j)]
                prob=net(*t).softmax(1)[:,1].cpu().numpy()
                for k,x in enumerate(xx):vals[y:y+256,x:x+256]=prob[k]
            assert np.isfinite(vals).all() and vals.min()>=0 and vals.max()<=1
            with rasterio.open(dst,'w',driver='GTiff',width=w,height=h,count=1,crs=aff[i][1],transform=aff[i][0],dtype='float32',nodata=-1,compress='deflate',predictor=3) as ds:
                ds.write(np.where(valid,vals,-1).astype('float32'),1);ds.set_band_description(1,'WHU_binary_change_probability_uncalibrated')
            with rasterio.open(dst) as ds:assert np.array_equal(ds.read(1)[valid],vals[valid])
            records.append(dict(site=site,pair=[i+1,j+1],control=i==j,dates=[before,after],file=dst.name,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                  supported_px=int(valid.sum()),changed_px=int((vals[valid]>=.5).sum()),mean=float(vals[valid].mean())))
            save(out/'PROGRESS.json',records);print(len(records),dst.name,flush=True)
    assert len(records)==51
    save(out/'RECEIPT.json',dict(records=records,sites=24,real_adjacent_pairs=48,self_controls=3))
if __name__=='__main__':main()

