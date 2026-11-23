"""Cardinality-preserving context adapter; all candidates still get scored.
Training pooled three candidates. A deployment image with 70 unrelated roofs
must not substitute all 70 into that temporal context distribution.
"""
import torch
def score(model,q,c,g,s,times):
    b,t,k=c.shape[:3];h,w=c.shape[-2:]
    qe=model.encoder(q);ce=model.encoder(c.reshape(-1,4,h,w)).reshape(b,t,k,64);a=qe[:,None,None].expand_as(ce)
    z=model.pair(torch.cat([a,ce,abs(a-ce),a*ce,model.geometry(g)],-1))
    distance=(g[...,:2]**2).sum(-1);distance=distance.masked_fill(g[...,7]<=0,float('inf'))
    near=distance.argsort(dim=-1,stable=True)[...,:min(k,3)]
    selected=torch.gather(z,2,near[...,None].expand(-1,-1,-1,96))
    mask=torch.gather((g[...,7]>0).float(),2,near)[...,None]
    context=(selected*mask).sum(2)/mask.sum(2).clamp(min=1)
    hist,_=model.temporal(context);hist=hist[:,:,None].expand(-1,-1,k,-1)
    logits=model.classifier(torch.cat([z,hist],-1))
    se=model.encoder(s.reshape(-1,4,h,w)).reshape(b,t,64);aq=qe[:,None].expand_as(se)
    x=model.search(torch.cat([aq,se,abs(aq-se),times],-1));x,_=model.presence_temporal(x)
    return dict(association=logits,presence=model.presence(x).squeeze(-1),observable=model.observability(x).squeeze(-1))
