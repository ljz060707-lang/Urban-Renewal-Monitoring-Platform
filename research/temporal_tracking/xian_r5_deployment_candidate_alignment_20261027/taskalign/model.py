"""Untrained set-selection head over embeddings, variable K plus one dustbin.
No classifier checkpoint is loaded on import. Actual encoder choice is a
separate, budgeted experiment after the candidate/trajectory task is fixed.
"""
import torch
from torch import nn

class VariableSetMatcher(nn.Module):
    def __init__(self,embedding_dim=64,hidden=96):
        super().__init__();self.embedding_dim=embedding_dim
        self.pair=nn.Sequential(nn.Linear(embedding_dim*4+8,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.temporal=nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True)
        self.choose=nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,1))
        self.dustbin=nn.Sequential(nn.Linear(hidden+embedding_dim+2,hidden),nn.SiLU(),nn.Linear(hidden,1))
    def forward(self,q,c,geometry,candidate_mask,time_features):
        b,t,k,d=c.shape
        if d!=self.embedding_dim or geometry.shape!=(b,t,k,8):raise ValueError('Embedding/geometry dimensions')
        if time_features.shape!=(b,t,2):raise ValueError('Actual date-gap features required')
        qq=q[:,None,None].expand(b,t,k,d)
        z=self.pair(torch.cat([qq,c,abs(qq-c),qq*c,geometry],-1))
        m=candidate_mask[...,None].to(z.dtype)
        context=(z*m).sum(2)/m.sum(2).clamp(min=1)
        history,_=self.temporal(context)
        pair_logits=self.choose(torch.cat([z,history[:,:,None].expand(b,t,k,-1)],-1))[...,0]
        pair_logits=pair_logits.masked_fill(~candidate_mask,-float('inf'))
        dustbin=self.dustbin(torch.cat([q[:,None].expand(b,t,d),history,time_features],-1))[...,0]
        return torch.cat([pair_logits,dustbin[...,None]],-1)

def set_selection_loss(logits,positive_mask,dustbin_target,enabled):
    """Multi-positive selection or explicit dustbin; missed retrieval is retained.
    Unknown source labels disable loss, not observation/metric denominators.
    """
    if logits.shape[-1]!=positive_mask.shape[-1]+1:raise ValueError('Dustbin must be a separate final logit')
    allowed=torch.cat([positive_mask,dustbin_target[...,None]],-1)
    if ((allowed.sum(-1)==0)&enabled).any():raise ValueError('Enabled query without target or dustbin')
    if (positive_mask.any(-1)&dustbin_target).any():raise ValueError('Both correspondence and dustbin cannot be truth')
    if not enabled.any():return logits[...,-1].sum()*0
    # Mask unknown rows before logsumexp to avoid undefined backward gradients.
    logp=logits[enabled].log_softmax(-1)
    selected=logp.masked_fill(~allowed[enabled],-float('inf')).logsumexp(-1)
    return -selected.mean()
