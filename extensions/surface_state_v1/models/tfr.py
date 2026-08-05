from __future__ import annotations
import torch
from torch import nn

class TemporalFeatureRefinement(nn.Module):
    """Per-pixel temporal attention; spatial locations are never globally attended."""
    def __init__(self,channels,heads=4,dropout=.1):
        super().__init__(); self.attn=nn.MultiheadAttention(channels,heads,dropout=dropout,batch_first=True); self.norm1=nn.LayerNorm(channels); self.ff=nn.Sequential(nn.Linear(channels,channels*2),nn.GELU(),nn.Linear(channels*2,channels)); self.norm2=nn.LayerNorm(channels)
    def forward(self,features,timestamps=None,temporal_valid_mask=None):
        x=features if isinstance(features,list) else [features]; output=[]; attention=[]
        for f in x:
            b,t,c,h,w=f.shape; q=f.permute(0,3,4,1,2).reshape(b*h*w,t,c); key_padding=None if temporal_valid_mask is None else ~temporal_valid_mask[:,None,None,:].expand(b,h,w,t).reshape(b*h*w,t)
            a,weights=self.attn(q,q,q,key_padding_mask=key_padding,need_weights=False); z=self.norm1(q+a); z=self.norm2(z+self.ff(z)); output.append(z.reshape(b,h,w,t,c).permute(0,3,4,1,2)); attention.append(weights)
        return {"refined_features":output,"temporal_attention":attention,"valid_timesteps":temporal_valid_mask if temporal_valid_mask is not None else torch.ones(features[0].shape[:2],device=features[0].device,dtype=torch.bool)}
