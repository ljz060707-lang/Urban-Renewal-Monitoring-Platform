"""Full-image roof-center, native box and semantic heads. Inference accepts only RGB features."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from torch.utils.data import Dataset

class Block(nn.Module):
    def __init__(self, n, dilation=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(n,n,3,padding=dilation,dilation=dilation,bias=False),nn.GroupNorm(8,n),nn.SiLU(),
                               nn.Conv2d(n,n,3,padding=1,bias=False),nn.GroupNorm(8,n))
    def forward(self,x):
        return F.silu(self.net(x)+x)

class ProposalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.context=nn.Sequential(nn.Conv2d(256,96,1,bias=False),nn.GroupNorm(8,96),nn.SiLU(),Block(96),Block(96,2))
        self.detail=nn.Sequential(nn.Conv2d(96,256,3,padding=1),nn.PixelShuffle(2),nn.GroupNorm(8,64),nn.SiLU(),Block(64))
        self.objectness=nn.Conv2d(64,1,1)
        self.shape=nn.Conv2d(64,6,1)
        self.roof=nn.Conv2d(64,1,1)
        nn.init.constant_(self.objectness.bias,-3.)
        nn.init.constant_(self.shape.bias[2:],1.5)
    def forward(self,image_feature):
        f=self.detail(self.context(image_feature))
        return self.objectness(f)[:,0], self.shape(f), self.roof(f)[:,0]

class FrameDataset(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with np.load(row['targets']) as z:
            v={k:torch.from_numpy(z[k]) for k in ['heat','center_valid','center_owner','box_reg','known','roof']}
        v['feature']=torch.from_numpy(np.load(row['features']))
        v['index']=i
        return v

def loss(outputs,b):
    center,box,roof=(v.float() for v in outputs)
    p=center.sigmoid()
    owner=b['center_owner']
    positive=owner.float()
    neg=(~owner & b['center_valid']).float()*(1-b['heat']).pow(4)
    count=positive.sum().clamp_min(1)
    center_loss=(F.softplus(-center)*(1-p).square()*positive + F.softplus(center)*p.square()*neg).sum()/count
    # Positive cell offsets and log1p distances from interior point to each side.
    coords=torch.cat((box[:,:2].sigmoid(),box[:,2:]),dim=1)
    reg=F.smooth_l1_loss(coords,b['box_reg'],reduction='none',beta=.25)
    box_loss=(reg*positive[:,None]).sum()/(count*6)
    roof_loss=(F.binary_cross_entropy_with_logits(roof,b['roof'],reduction='none')*b['known']).sum()/b['known'].sum().clamp_min(1)
    values={'center':center_loss,'box':box_loss,'roof':roof_loss}
    return center_loss+2*box_loss+.35*roof_loss,values

@torch.no_grad()
def decode(outputs):
    scores,box,_=outputs
    scores=scores.float().sigmoid()
    maxima=scores>=F.max_pool2d(scores[:,None],3,1,1)[:,0]
    result=[]
    for p,b,keep in zip(scores,box.float(),maxima):
        values,idx=(p*keep).flatten().topk(256)
        good=values>=.10
        idx=idx[good];values=values[good]
        ix=idx%128;iy=idx//128
        params=b[:,iy,ix].T
        centers=(torch.stack([ix,iy],dim=1)+params[:,:2].sigmoid())*8
        extent=8*torch.expm1(params[:,2:].clamp(0,5.1))
        native=torch.cat([centers-extent[:,:2],centers+extent[:,2:]],dim=1).clamp(0,1024)
        usable=((native[:,2:]-native[:,:2])>=2).all(dim=1)
        values,centers,native=values[usable],centers[usable],native[usable]
        pick=nms(native,values,.60)
        result.append(torch.cat([values[pick,None],centers[pick],native[pick]],dim=1).cpu().numpy())
    return result

