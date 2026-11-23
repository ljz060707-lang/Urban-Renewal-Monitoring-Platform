"""Controlled native RGB addition and a predeclared extra-building budget."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from detail_inputs import P
from proposal_net import ProposalNet,FrameDataset
from evaluate_proposals import ious

class DetailNet(ProposalNet):
    def __init__(self):
        super().__init__()
        self.rgb_stem=nn.Sequential(nn.PixelUnshuffle(4),nn.Conv2d(48,32,3,padding=1,bias=False),
                    nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,64,3,stride=2,padding=1,bias=False),
                    nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,64,1))
        nn.init.zeros_(self.rgb_stem[-1].weight);nn.init.zeros_(self.rgb_stem[-1].bias)
    def forward(self,features,rgb):
        local=self.rgb_stem(rgb.float()/127.5-1.)
        f=self.detail(self.context(features))+local
        return self.objectness(f)[:,0],self.shape(f),self.roof(f)[:,0]

class NativeDataset(FrameDataset):
    def __getitem__(self,i):
        out=super().__getitem__(i);row=self.rows[i]
        out['rgb']=torch.from_numpy(np.load(row['rgb'])).permute(2,0,1)
        with np.load(row['raw']) as z:out['weight']=torch.from_numpy(z['weights'])
        return out

def weighted_loss(outputs,b,boost):
    c,g,r=(v.float() for v in outputs)
    prob=c.sigmoid();own=b['center_owner'];positive=own.float()
    if boost:positive=positive*b['weight']
    count=positive.sum().clamp_min(1)
    neg=(~own & b['center_valid']).float()*(1-b['heat']).pow(4)
    center=(F.softplus(-c)*(1-prob).square()*positive+F.softplus(c)*prob.square()*neg).sum()/count
    box=F.smooth_l1_loss(torch.cat([g[:,:2].sigmoid(),g[:,2:]],1),b['box_reg'],reduction='none',beta=.25)
    regression=(box*positive[:,None]).sum()/(6*count)
    roof=(F.binary_cross_entropy_with_logits(r,b['roof'],reduction='none')*b['known']).sum()/b['known'].sum().clamp_min(1)
    return center+2*regression+.35*roof,dict(center=center,box=regression,roof=roof)

@torch.no_grad()
def budget_decode(outputs,supplied,raw_ids,raw_boxes):
    # Apply the real mask-addition eligibility to predicted centers BEFORE ranking.
    c,g,_=outputs;c=c.float().sigmoid();g=g.float()
    proposals=[]
    iy,ix=torch.meshgrid(torch.arange(128,device=c.device),torch.arange(128,device=c.device),indexing='ij')
    for i in range(len(c)):
        xyoff=g[i,:2].sigmoid()
        xy=torch.stack([ix+xyoff[0],iy+xyoff[1]])*8
        px=xy[0].long().clamp(0,1023);py=xy[1].long().clamp(0,1023)
        eligible=torch.from_numpy(supplied[i]&~raw_ids[i]).to(c.device)[py,px]
        score=torch.where(eligible,c[i],-1.)
        peak=(score>=F.max_pool2d(score[None,None],3,1,1)[0,0])&eligible
        vals,idx=torch.topk(torch.where(peak,score,-1.).flatten(),min(1024,score.numel()))
        take=vals>=0;vals=vals[take];idx=idx[take];x=idx%128;y=idx//128
        centers=xy[:,y,x].T
        sides=8*torch.expm1(g[i,2:,y,x].T.clamp(0,5.1))
        boxes=torch.cat([centers-sides[:,:2],centers+sides[:,2:]],1).clamp(0,1024)
        extent=((boxes[:,2:]-boxes[:,:2])>=2).all(1)
        boxes,centers,vals=boxes[extent],centers[extent],vals[extent]
        if len(raw_boxes[i]):
            dup=ious(boxes.cpu().numpy(),raw_boxes[i][:,2:]).max(1)>=.60
            keep=torch.from_numpy(~dup).to(c.device)
            boxes,centers,vals=boxes[keep],centers[keep],vals[keep]
        idx=nms(boxes,vals,.60)
        proposals.append(torch.cat([vals[idx,None],centers[idx],boxes[idx]],1).cpu().numpy())
    return proposals

def take_budget(det,n,frac):
    return det[:max(2,math.ceil(n*frac))]

