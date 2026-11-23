"""One trainable RGB/mask encoder, geometry branch and bidirectional temporal GRU.

Classes: SAME_ID, DIFFERENT_ID, NO_LABELLED_COUNTERPART. The last class is not
certified visual absence. All candidate classes use exactly the same crop code.
"""
import torch
from torch import nn

class TemporalIdentityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(4,24,5,2,2),nn.GroupNorm(4,24),nn.SiLU(),
            nn.Conv2d(24,48,3,2,1),nn.GroupNorm(8,48),nn.SiLU(),
            nn.Conv2d(48,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.geometry=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(64*4+32,96),nn.SiLU(),nn.Dropout(.1))
        self.temporal=nn.GRU(96,48,batch_first=True,bidirectional=True)
        self.classifier=nn.Sequential(nn.Linear(192,64),nn.SiLU(),nn.Linear(64,3))
        self.search=nn.Sequential(nn.Linear(64*3+2,64),nn.SiLU())
        self.presence_temporal=nn.GRU(64,32,batch_first=True,bidirectional=True)
        self.presence=nn.Linear(64,1)
        self.observability=nn.Linear(64,1)
        self.bt_flow=nn.Sequential(nn.Linear(192,64),nn.SiLU(),nn.Linear(64,2))
    def flow_forward(self,a,b):
        x=self.encoder(a); y=self.encoder(b)
        return self.bt_flow(torch.cat([x,y,abs(x-y)],-1))
    def forward(self,q,c,g,s,times):
        # q B,4,H,W; c B,T,K,4,H,W; s B,T,4,H,W.
        b,t,k=c.shape[:3]; h,w=c.shape[-2:]
        qe=self.encoder(q); ce=self.encoder(c.reshape(-1,4,h,w)).reshape(b,t,k,64)
        a=qe[:,None,None].expand_as(ce)
        z=self.pair(torch.cat([a,ce,abs(a-ce),a*ce,self.geometry(g)],-1))
        # Candidate ordering has no physical meaning and can differ by date.
        # Pool the set before temporal context, preserving per-date permutation
        # equivariance. Never treat candidate slot 0 as a persistent identity.
        mask=(g[...,7:8]>0).to(z.dtype)
        seq=(z*mask).sum(2)/mask.sum(2).clamp(min=1)
        hist,_=self.temporal(seq)
        hist=hist[:,:,None].expand(-1,-1,k,-1)
        logits=self.classifier(torch.cat([z,hist],-1))
        se=self.encoder(s.reshape(-1,4,h,w)).reshape(b,t,64)
        aq=qe[:,None].expand_as(se)
        x=self.search(torch.cat([aq,se,abs(aq-se),times],-1))
        x,_=self.presence_temporal(x)
        return dict(association=logits,presence=self.presence(x).squeeze(-1),observable=self.observability(x).squeeze(-1))

